// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
//
// Group 6, Part 2: differential video-frame processing.
// Based on the Part 1 hardware/core_tb.v (same signal names, *_q registers, stage timing
// and cycle counting). Changes for Part 2:
//   - inst widened to [42:0]: [36] mode, [37] sfu_shift, [38] A_xmem[11], [42:39] A_pmem[14:11]
//   - several passes per weight set, weights loaded ONCE per kij (no reset between passes):
//       default : differential, 5 passes   x0_hi, x0_lo (mode 0), dx1, dx2, dx3 (mode 1)
//       -DCONV  : conventional baseline, 8 passes  x0_hi, x0_lo, ..., x3_hi, x3_lo (mode 0)
//   - each pass streams 324 nij in chunks of 54 (L0 / OFIFO are 64 deep)
//   - raw psums stored at PMEM[(kij*NPASS + pass)*324 + nij], checked against python dumps
//   - SFU (relu_off = 1, raw partial sums):
//       diff : Y0 = sum_kij (psum_hi << 4) + psum_lo        -> store back to PMEM[Y_BASE + onij]
//              Yi = Y0 (first read of the burst = load) + sum_kij psum_dxi -> store back
//       conv : Yi = sum_kij (psum_hi_i << 4) + psum_lo_i    -> store back
//     sfp_out is compared with golden/y*.txt after every output, and the stored results
//     in PMEM are checked again at the end
//   - per-stage cycle counts printed at the end
// Data files come from part2/python/gen_files.py (DATA_DIR, default ../../out/).
`timescale 1ns/1ps

`ifndef DATA_DIR
`define DATA_DIR "../../out/"
`endif

module core_tb;

parameter bw = 4;
parameter psum_bw = 16;
parameter acc_bw = 24;
parameter len_kij = 9;
parameter len_onij = 256;    // 16 x 16 output
parameter col = 8;
parameter row = 8;
parameter len_nij = 324;     // 18 x 18 padded input
parameter Wp = 18;           // padded input width
parameter Wo = 16;           // output width
parameter CHUNK = 54;        // nij per L0 / OFIFO batch (<= 64), 324 = 6 x 54

parameter W_BASE = 12'd3072;     // XMEM: weights of the current kij (activations at pass*324)
parameter Y_BASE = 15'd24576;    // PMEM: stored Y0..Y3, one word per onij (raw psums below)

`ifdef CONV
parameter NPASS = 8;
`else
parameter NPASS = 5;
`endif

reg clk = 0;
reg reset = 1;

wire [42:0] inst_q;

reg [1:0]  inst_w_q = 0;
reg [bw*row-1:0] D_xmem_q = 0;
reg CEN_xmem = 1;
reg WEN_xmem = 1;
reg [11:0] A_xmem = 0;
reg CEN_xmem_q = 1;
reg WEN_xmem_q = 1;
reg [11:0] A_xmem_q = 0;
reg CEN_pmem = 1;
reg WEN_pmem = 1;
reg [14:0] A_pmem = 0;
reg CEN_pmem_q = 1;
reg WEN_pmem_q = 1;
reg [14:0] A_pmem_q = 0;
reg ofifo_rd_q = 0;
reg ififo_wr_q = 0;
reg ififo_rd_q = 0;
reg l0_rd_q = 0;
reg l0_wr_q = 0;
reg execute_q = 0;
reg load_q = 0;
reg acc_q = 0;
reg acc = 0;
reg sfu_wb = 0;
reg sfu_wb_q = 0;
reg relu_off = 1;            // Part 2 outputs raw partial sums
reg relu_off_q = 1;
reg mode = 0;                // [Part 2]
reg mode_q = 0;
reg sfu_shift = 0;           // [Part 2]
reg sfu_shift_q = 0;

reg [1:0]  inst_w;
reg [bw*row-1:0] D_xmem;
reg [psum_bw*col-1:0] answer;

reg ofifo_rd;
reg ififo_wr;
reg ififo_rd;
reg l0_rd;
reg l0_wr;
reg execute;
reg load;
reg [8*256:1] f_name;
reg [8*96:1] captured_data;
wire ofifo_valid;
wire [col*acc_bw-1:0] sfp_out;

integer x_file, x_scan_file ; // file_handler
integer w_file, w_scan_file ; // file_handler
integer psum_file, psum_scan_file ; // file_handler
integer out_file, out_scan_file ; // file_handler
integer t, i, j, k, kij, p, c, f;
integer error;
integer err_psum, err_out, err_store;

reg [col*acc_bw-1:0] gold [0:4*len_onij-1];
reg [col*acc_bw-1:0] gold_word;

// ---------------- cycle counting ----------------
integer cycle = 0;
always @ (posedge clk) cycle <= cycle + 1;

localparam N_STAGE = 9;
integer stage_cycles [0:N_STAGE-1];
integer stage_start;
task stage_begin; begin stage_start = cycle; end endtask
task stage_end(input integer s); begin stage_cycles[s] = stage_cycles[s] + (cycle - stage_start); end endtask
// ------------------------------------------------

// ---------------- address helpers ----------------
// nij of the psum that output onij needs for kernel position kk
function integer nij_of(input integer onij, input integer kk);
  nij_of = (onij / Wo + kk / 3) * Wp + (onij % Wo) + (kk % 3);
endfunction
// PMEM address of a raw psum
function integer paddr(input integer kk, input integer pass, input integer nij);
  paddr = (kk * NPASS + pass) * len_nij + nij;
endfunction
// sign-extend a 8 x 16-bit psum word to the 8 x 24-bit PMEM word
function [col*acc_bw-1:0] sext(input [col*psum_bw-1:0] w);
  integer n;
  begin
    for (n = 0; n < col; n = n + 1)
      sext[acc_bw*n +: acc_bw] = {{(acc_bw-psum_bw){w[psum_bw*n + psum_bw-1]}}, w[psum_bw*n +: psum_bw]};
  end
endfunction
// activation file of a pass
task pass_file(input integer pass, output [8*256:1] fname);
`ifdef CONV
  if (pass % 2 == 0) $sformat(fname, "%sstimulus/conventional/x%0d_hi.txt", `DATA_DIR, pass / 2);
  else               $sformat(fname, "%sstimulus/conventional/x%0d_lo.txt", `DATA_DIR, pass / 2);
`else
  case (pass)
    0:       $sformat(fname, "%sstimulus/x0_hi.txt", `DATA_DIR);
    1:       $sformat(fname, "%sstimulus/x0_lo.txt", `DATA_DIR);
    default: $sformat(fname, "%sstimulus/dx%0d.txt", `DATA_DIR, pass - 1);
  endcase
`endif
endtask
// mode of a pass: X0 / Xi nibbles are unsigned, deltas are signed
function pass_mode(input integer pass);
`ifdef CONV
  pass_mode = 1'b0;
`else
  pass_mode = (pass >= 2);
`endif
endfunction
// ------------------------------------------------

assign inst_q[42:39] = A_pmem_q[14:11];
assign inst_q[38]    = A_xmem_q[11];
assign inst_q[37]    = sfu_shift_q;
assign inst_q[36]    = mode_q;
assign inst_q[35]    = relu_off_q;
assign inst_q[34]    = sfu_wb_q;
assign inst_q[33]    = acc_q;
assign inst_q[32]    = CEN_pmem_q;
assign inst_q[31]    = WEN_pmem_q;
assign inst_q[30:20] = A_pmem_q[10:0];
assign inst_q[19]    = CEN_xmem_q;
assign inst_q[18]    = WEN_xmem_q;
assign inst_q[17:7]  = A_xmem_q[10:0];
assign inst_q[6]     = ofifo_rd_q;
assign inst_q[5]     = ififo_wr_q;
assign inst_q[4]     = ififo_rd_q;
assign inst_q[3]     = l0_rd_q;
assign inst_q[2]     = l0_wr_q;
assign inst_q[1]     = execute_q;
assign inst_q[0]     = load_q;


core  #(.bw(bw), .psum_bw(psum_bw), .acc_bw(acc_bw), .col(col), .row(row)) core_instance (
	.clk(clk),
	.inst(inst_q),
	.ofifo_valid(ofifo_valid),
        .D_xmem(D_xmem_q),
        .sfp_out(sfp_out),
	.reset(reset));


initial begin

  inst_w   = 0;
  D_xmem   = 0;
  CEN_xmem = 1;
  WEN_xmem = 1;
  A_xmem   = 0;
  ofifo_rd = 0;
  ififo_wr = 0;
  ififo_rd = 0;
  l0_rd    = 0;
  l0_wr    = 0;
  execute  = 0;
  load     = 0;
  error    = 0;
  err_psum = 0; err_out = 0; err_store = 0;
  for (i=0; i<N_STAGE; i=i+1) stage_cycles[i] = 0;

`ifdef VCD
  $dumpfile("core_tb_part2.vcd");
  $dumpvars(0,core_tb);
`endif

`ifdef CONV
  $display("############ Part 2: CONVENTIONAL baseline, %0d passes per weight set #############", NPASS);
`else
  $display("############ Part 2: DIFFERENTIAL, %0d passes per weight set #############", NPASS);
`endif

  // golden outputs y0..y3
  for (f=0; f<4; f=f+1) begin
    $sformat(f_name, "%sgolden/y%0d.txt", `DATA_DIR, f);
    out_file = $fopen(f_name, "r");
    if (out_file == 0) begin $display("ERROR: cannot open %0s", f_name); $finish; end
    out_scan_file = $fscanf(out_file,"%s", captured_data);
    out_scan_file = $fscanf(out_file,"%s", captured_data);
    out_scan_file = $fscanf(out_file,"%s", captured_data);
    for (i=0; i<len_onij; i=i+1) begin out_scan_file = $fscanf(out_file,"%192b", gold_word); gold[f*len_onij+i] = gold_word; end
    $fclose(out_file);
  end

  //////// Reset /////////
  #0.5 clk = 1'b0;   reset = 1;
  #0.5 clk = 1'b1;

  for (i=0; i<10 ; i=i+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end

  #0.5 clk = 1'b0;   reset = 0;
  #0.5 clk = 1'b1;

  #0.5 clk = 1'b0;
  #0.5 clk = 1'b1;
  /////////////////////////

  /////// Activation data writing to memory (every pass, XMEM[pass*324 + nij]) ///////
  stage_begin;
  for (p=0; p<NPASS; p=p+1) begin
    pass_file(p, f_name);
    x_file = $fopen(f_name, "r");
    if (x_file == 0) begin $display("ERROR: cannot open %0s", f_name); $finish; end
    // Following three lines are to remove the first three comment lines of the file
    x_scan_file = $fscanf(x_file,"%s", captured_data);
    x_scan_file = $fscanf(x_file,"%s", captured_data);
    x_scan_file = $fscanf(x_file,"%s", captured_data);

    // all control changes happen in the clk=0 half (assigning right after clk=1 would race the *_q registers)
    for (t=0; t<len_nij; t=t+1) begin
      #0.5 clk = 1'b0;  x_scan_file = $fscanf(x_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1; else A_xmem = p*len_nij;
      #0.5 clk = 1'b1;
    end
    $fclose(x_file);
  end

  #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
  #0.5 clk = 1'b1;
  stage_end(0);
  /////////////////////////////////////////////////


  for (kij=0; kij<9; kij=kij+1) begin  // kij loop

    $sformat(f_name, "%sstimulus/weight_itile0_otile0_kij%0d.txt", `DATA_DIR, kij);
    w_file = $fopen(f_name, "r");
    if (w_file == 0) begin $display("ERROR: cannot open %0s", f_name); $finish; end
    // Following three lines are to remove the first three comment lines of the file
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);
    w_scan_file = $fscanf(w_file,"%s", captured_data);

    stage_begin;
    #0.5 clk = 1'b0;   reset = 1;
    #0.5 clk = 1'b1;

    for (i=0; i<10 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;   reset = 0;
    #0.5 clk = 1'b1;

    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
    stage_end(8);


    /////// Kernel data writing to memory ///////
    stage_begin;

    A_xmem = W_BASE;

    for (t=0; t<col; t=t+1) begin
      #0.5 clk = 1'b0;  w_scan_file = $fscanf(w_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
    #0.5 clk = 1'b1;
    stage_end(0);
    $fclose(w_file);
    /////////////////////////////////////


    /////// Kernel data writing to L0 ///////
    // XMEM read data is on Q one cycle after the address, so l0_wr lags by 1.
    stage_begin;
    A_xmem = W_BASE;
    for (t=0; t<col+1; t=t+1) begin
      #0.5 clk = 1'b0;
        if (t<col) begin WEN_xmem = 1; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1; end
        else       begin WEN_xmem = 1; CEN_xmem = 1; end
        l0_wr = (t>0);
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  l0_wr = 0; CEN_xmem = 1; A_xmem = 0;
    #0.5 clk = 1'b1;
    stage_end(1);
    /////////////////////////////////////


    /////// Kernel loading to PEs (once per kij, kept for all passes) ///////
    stage_begin;
    for (t=0; t<col; t=t+1) begin
      #0.5 clk = 1'b0;  l0_rd = 1; load = 1;
      #0.5 clk = 1'b1;
    end

    ////// provide some intermission to clear up the kernel loading ///
    #0.5 clk = 1'b0;  load = 0; l0_rd = 0;
    #0.5 clk = 1'b1;

    for (i=0; i<10 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;
    end
    stage_end(2);
    /////////////////////////////////////


    for (p=0; p<NPASS; p=p+1) begin    // pass loop: NO reset, weights stay in the PEs

      for (c=0; c<len_nij/CHUNK; c=c+1) begin   // chunk loop

        /////// Activation data writing to L0 ///////
        stage_begin;
        for (t=0; t<CHUNK+1; t=t+1) begin
          #0.5 clk = 1'b0;
            if (t==0) mode = pass_mode(p);   // set while the array is idle
            if (t<CHUNK) begin WEN_xmem = 1; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1; else A_xmem = p*len_nij + c*CHUNK; end
            else         begin WEN_xmem = 1; CEN_xmem = 1; end
            l0_wr = (t>0);
          #0.5 clk = 1'b1;
        end

        #0.5 clk = 1'b0;  l0_wr = 0; CEN_xmem = 1; A_xmem = 0;
        #0.5 clk = 1'b1;
        stage_end(3);
        /////////////////////////////////////


        /////// Execution ///////
        stage_begin;
        for (t=0; t<CHUNK; t=t+1) begin
          #0.5 clk = 1'b0;  l0_rd = 1; execute = 1;
          #0.5 clk = 1'b1;
        end

        #0.5 clk = 1'b0;  l0_rd = 0; execute = 0;
        #0.5 clk = 1'b1;

        // drain: the last activation needs ~row+col cycles to reach the OFIFO
        for (i=0; i<row+col+4 ; i=i+1) begin
          #0.5 clk = 1'b0;
          #0.5 clk = 1'b1;
        end
        stage_end(4);

        if (!ofifo_valid) begin
          $display("kij=%0d pass=%0d chunk=%0d: ERROR, OFIFO is not valid after execution", kij, p, c);
          error = error + 1;
        end
        /////////////////////////////////////


        //////// OFIFO READ ////////
        // ofifo_rd is registered inside ofifo.v: entry t is on the output one cycle
        // after rd, so the PMEM write lags by 1.
        stage_begin;
        for (t=0; t<CHUNK+1; t=t+1) begin
          #0.5 clk = 1'b0;
            ofifo_rd = (t<CHUNK);
            sfu_wb   = 0;
            if (t>0) begin CEN_pmem = 0; WEN_pmem = 0; A_pmem = paddr(kij, p, c*CHUNK + t - 1); end
            else     begin CEN_pmem = 1; WEN_pmem = 1; end
          #0.5 clk = 1'b1;
        end

        #0.5 clk = 1'b0;  ofifo_rd = 0; CEN_pmem = 1; WEN_pmem = 1; A_pmem = 0;
        #0.5 clk = 1'b1;
        stage_end(5);
        /////////////////////////////////////

      end  // end of chunk loop
    end  // end of pass loop
    mode = 0;

  end  // end of kij loop

  // flush: inst is registered in this TB, so the last PMEM write lands one edge later
  for (i=0; i<2 ; i=i+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end


  ////////// Psum check (raw array outputs in PMEM vs python debug dumps) /////////
`ifdef CONV
  k = 2;      // only x0_hi / x0_lo have debug dumps
`else
  k = NPASS;
`endif
  for (kij=0; kij<len_kij; kij=kij+1)
    for (p=0; p<k; p=p+1) begin
      case (p)
        0:       $sformat(f_name, "%sdebug/psum_x0_hi_kij%0d.txt", `DATA_DIR, kij);
        1:       $sformat(f_name, "%sdebug/psum_x0_lo_kij%0d.txt", `DATA_DIR, kij);
        default: $sformat(f_name, "%sdebug/psum_dx%0d_kij%0d.txt", `DATA_DIR, p-1, kij);
      endcase
      psum_file = $fopen(f_name, "r");
      if (psum_file == 0) begin $display("ERROR: cannot open %0s", f_name); $finish; end
      psum_scan_file = $fscanf(psum_file,"%s", captured_data);
      psum_scan_file = $fscanf(psum_file,"%s", captured_data);
      psum_scan_file = $fscanf(psum_file,"%s", captured_data);
      for (i=0; i<len_nij; i=i+1) begin
        psum_scan_file = $fscanf(psum_file,"%128b", answer);
        if (core_instance.pmem_instance.memory[paddr(kij, p, i)] !== sext(answer)) begin
          if (err_psum < 5) $display("psum ERROR at kij=%0d pass=%0d nij=%0d", kij, p, i);
          err_psum = err_psum + 1;
        end
      end
      $fclose(psum_file);
    end
  if (err_psum == 0) $display("############ psum check: all %0d psum words matched #############", len_kij*k*len_nij);
  else               $display("############ psum check: %0d ERRORS #############", err_psum);
  error = error + err_psum;
  //////////////////////////////////


  ////////// SFU: Y0 (diff) or every Yi (conv), store back /////////
  // one acc burst of 18 reads: 9 high-nibble psums (sfu_shift = 1) + 9 low-nibble psums.
  // The first cycle of the burst loads, so no reset is needed between outputs.
  $display("############ Verification Start during accumulation #############");
  stage_begin;
`ifdef CONV
  for (f=0; f<4; f=f+1)
`else
  for (f=0; f<1; f=f+1)
`endif
  for (i=0; i<len_onij; i=i+1) begin
    for (j=0; j<2*len_kij+1; j=j+1) begin
      #0.5 clk = 1'b0;
        if (j<len_kij)        begin CEN_pmem = 0; WEN_pmem = 1; A_pmem = paddr(j, 2*f, nij_of(i, j)); end
        else if (j<2*len_kij) begin CEN_pmem = 0; WEN_pmem = 1; A_pmem = paddr(j-len_kij, 2*f+1, nij_of(i, j-len_kij)); end
        else                  begin CEN_pmem = 1; WEN_pmem = 1; end
        if (j>0) begin acc = 1; sfu_shift = (j-1 < len_kij); end   // acc lags the read by 1
      #0.5 clk = 1'b1;
    end

    // store the raw result back to PMEM
    #0.5 clk = 1'b0; acc = 0; sfu_shift = 0; CEN_pmem = 0; WEN_pmem = 0; A_pmem = Y_BASE + f*len_onij + i; sfu_wb = 1;
    #0.5 clk = 1'b1;

    #0.5 clk = 1'b0; CEN_pmem = 1; WEN_pmem = 1; sfu_wb = 0;
      if (sfp_out !== gold[f*len_onij + i]) begin
        if (err_out < 5) begin
          $display("y%0d onij %0d ERROR", f, i);
          $display("sfpout: %h", sfp_out);
          $display("answer: %h", gold[f*len_onij + i]);
        end
        err_out = err_out + 1;
      end
    #0.5 clk = 1'b1;
  end
  stage_end(6);
  //////////////////////////////////


`ifndef CONV
  ////////// SFU: Yi = Y0 + W x dXi, store back /////////
  // one acc burst of 10 reads: stored Y0 (loaded by the first acc cycle) + 9 dXi psums
  stage_begin;
  for (f=1; f<4; f=f+1)
  for (i=0; i<len_onij; i=i+1) begin
    for (j=0; j<len_kij+2; j=j+1) begin
      #0.5 clk = 1'b0;
        if (j==0)              begin CEN_pmem = 0; WEN_pmem = 1; A_pmem = Y_BASE + i; end
        else if (j<=len_kij)   begin CEN_pmem = 0; WEN_pmem = 1; A_pmem = paddr(j-1, f+1, nij_of(i, j-1)); end
        else                   begin CEN_pmem = 1; WEN_pmem = 1; end
        if (j>0) acc = 1;
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0; acc = 0; CEN_pmem = 0; WEN_pmem = 0; A_pmem = Y_BASE + f*len_onij + i; sfu_wb = 1;
    #0.5 clk = 1'b1;

    #0.5 clk = 1'b0; CEN_pmem = 1; WEN_pmem = 1; sfu_wb = 0;
      if (sfp_out !== gold[f*len_onij + i]) begin
        if (err_out < 5) begin
          $display("y%0d onij %0d ERROR", f, i);
          $display("sfpout: %h", sfp_out);
          $display("answer: %h", gold[f*len_onij + i]);
        end
        err_out = err_out + 1;
      end
    #0.5 clk = 1'b1;
  end
  stage_end(7);
  //////////////////////////////////
`endif

  if (err_out == 0) $display("############ sfp_out check: all %0d outputs (y0..y3) matched #############", 4*len_onij);
  else              $display("############ sfp_out check: %0d ERRORS #############", err_out);
  error = error + err_out;

  // flush the last store, then check every stored output word in PMEM
  for (i=0; i<2 ; i=i+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end
  for (i=0; i<4*len_onij; i=i+1)
    if (core_instance.pmem_instance.memory[Y_BASE + i] !== gold[i]) err_store = err_store + 1;
  if (err_store == 0) $display("############ stored outputs in PMEM: all %0d matched #############", 4*len_onij);
  else                $display("############ stored outputs in PMEM: %0d ERRORS #############", err_store);
  error = error + err_store;


  $display("");
  $display("############ Cycle count (all 9 kij, %0d passes each) ############", NPASS);
  $display("  L1 (XMEM) writes, act + weight     : %6d", stage_cycles[0]);
  $display("  kernel XMEM -> L0                  : %6d", stage_cycles[1]);
  $display("  kernel loading to PEs (+intermiss.) : %6d", stage_cycles[2]);
  $display("  activation XMEM -> L0              : %6d", stage_cycles[3]);
  $display("  execution (+array drain)           : %6d", stage_cycles[4]);
  $display("  OFIFO -> PMEM                      : %6d", stage_cycles[5]);
  $display("  per-kij reset                      : %6d", stage_cycles[8]);
`ifdef CONV
  $display("  SFU Y0..Y3 (hi<<4 + lo) + store    : %6d", stage_cycles[6]);
`else
  $display("  SFU Y0 (hi<<4 + lo) + store        : %6d", stage_cycles[6]);
  $display("  SFU Y1..Y3 (Y0 + dY) + store       : %6d", stage_cycles[7]);
`endif
  $display("  total simulated cycles             : %6d", cycle);
  $display("RESULT passes=%0d array=%0d sfu=%0d l1_write=%0d total=%0d errors=%0d", NPASS,
           stage_cycles[1]+stage_cycles[2]+stage_cycles[3]+stage_cycles[4]+stage_cycles[5]+stage_cycles[8],
           stage_cycles[6]+stage_cycles[7], stage_cycles[0], cycle, error);
  $display("");

  if (error == 0) begin
  	$display("############ No error detected ##############");
  	$display("########### Project Completed !! ############");
  end
  else
  	$display("############ %0d errors detected ##############", error);

  for (t=0; t<10; t=t+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end

  #10 $finish;

end

always @ (posedge clk) begin
   inst_w_q    <= inst_w;
   D_xmem_q    <= D_xmem;
   CEN_xmem_q  <= CEN_xmem;
   WEN_xmem_q  <= WEN_xmem;
   A_pmem_q    <= A_pmem;
   CEN_pmem_q  <= CEN_pmem;
   WEN_pmem_q  <= WEN_pmem;
   A_xmem_q    <= A_xmem;
   ofifo_rd_q  <= ofifo_rd;
   acc_q       <= acc;
   sfu_wb_q    <= sfu_wb;
   relu_off_q  <= relu_off;
   mode_q      <= mode;
   sfu_shift_q <= sfu_shift;
   ififo_wr_q  <= ififo_wr;
   ififo_rd_q  <= ififo_rd;
   l0_rd_q     <= l0_rd;
   l0_wr_q     <= l0_wr ;
   execute_q   <= execute;
   load_q      <= load;
end


endmodule
