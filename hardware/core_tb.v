// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission
//
// Group 6: template completed with all 7 project stages.
//   - inst widened to [35:0]: [34] sfu_wb (PMEM write source), [35] relu_off
//   - Stage 5 result checked against psum.txt
//   - accumulation result stored back to PMEM (raw), then a separate ReLU
//     stage stores ReLU results back, then they are read back and checked
//   - per-stage cycle counts printed at the end
// Run from the directory holding the data files (see hardware/README.md).
`timescale 1ns/1ps

module core_tb;

parameter bw = 4;
parameter psum_bw = 16;
parameter len_kij = 9;
parameter len_onij = 16;
parameter col = 8;
parameter row = 8;
parameter len_nij = 36;

parameter ACC_BASE  = 11'd512;   // PMEM: accumulated (raw) results, one word per onij
parameter RELU_BASE = 11'd768;   // PMEM: ReLU results, one word per onij

reg clk = 0;
reg reset = 1;

wire [35:0] inst_q;

reg [1:0]  inst_w_q = 0;
reg [bw*row-1:0] D_xmem_q = 0;
reg CEN_xmem = 1;
reg WEN_xmem = 1;
reg [10:0] A_xmem = 0;
reg CEN_xmem_q = 1;
reg WEN_xmem_q = 1;
reg [10:0] A_xmem_q = 0;
reg CEN_pmem = 1;
reg WEN_pmem = 1;
reg [10:0] A_pmem = 0;
reg CEN_pmem_q = 1;
reg WEN_pmem_q = 1;
reg [10:0] A_pmem_q = 0;
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
reg relu_off = 0;
reg relu_off_q = 0;

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
reg [8*30:1] stringvar;
reg [8*30:1] w_file_name;
wire ofifo_valid;
wire [col*psum_bw-1:0] sfp_out;

integer x_file, x_scan_file ; // file_handler
integer w_file, w_scan_file ; // file_handler
integer acc_file, acc_scan_file ; // file_handler
integer out_file, out_scan_file ; // file_handler
integer psum_file, psum_scan_file ; // file_handler
integer captured_data;
integer t, i, j, k, kij;
integer error;

// ---------------- cycle counting ----------------
integer cycle = 0;
always @ (posedge clk) cycle <= cycle + 1;

localparam N_STAGE = 9;
integer stage_cycles [0:N_STAGE-1];
integer stage_start;
task stage_begin; begin stage_start = cycle; end endtask
task stage_end(input integer s); begin stage_cycles[s] = stage_cycles[s] + (cycle - stage_start); end endtask
// ------------------------------------------------

assign inst_q[35] = relu_off_q;
assign inst_q[34] = sfu_wb_q;
assign inst_q[33] = acc_q;
assign inst_q[32] = CEN_pmem_q;
assign inst_q[31] = WEN_pmem_q;
assign inst_q[30:20] = A_pmem_q;
assign inst_q[19]   = CEN_xmem_q;
assign inst_q[18]   = WEN_xmem_q;
assign inst_q[17:7] = A_xmem_q;
assign inst_q[6]   = ofifo_rd_q;
assign inst_q[5]   = ififo_wr_q;
assign inst_q[4]   = ififo_rd_q;
assign inst_q[3]   = l0_rd_q;
assign inst_q[2]   = l0_wr_q;
assign inst_q[1]   = execute_q;
assign inst_q[0]   = load_q;


core  #(.bw(bw), .col(col), .row(row)) core_instance (
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
  for (i=0; i<N_STAGE; i=i+1) stage_cycles[i] = 0;

  $dumpfile("core_tb.vcd");
  $dumpvars(0,core_tb);

  x_file = $fopen("activation_tile0.txt", "r");
  // Following three lines are to remove the first three comment lines of the file
  x_scan_file = $fscanf(x_file,"%s", captured_data);
  x_scan_file = $fscanf(x_file,"%s", captured_data);
  x_scan_file = $fscanf(x_file,"%s", captured_data);

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

  /////// Activation data writing to memory ///////
  stage_begin;
  for (t=0; t<len_nij; t=t+1) begin
    #0.5 clk = 1'b0;  x_scan_file = $fscanf(x_file,"%32b", D_xmem); WEN_xmem = 0; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1;
    #0.5 clk = 1'b1;
  end

  #0.5 clk = 1'b0;  WEN_xmem = 1;  CEN_xmem = 1; A_xmem = 0;
  #0.5 clk = 1'b1;
  stage_end(0);

  $fclose(x_file);
  /////////////////////////////////////////////////


  for (kij=0; kij<9; kij=kij+1) begin  // kij loop

    case(kij)
     0: w_file_name = "weight_itile0_otile0_kij0.txt";
     1: w_file_name = "weight_itile0_otile0_kij1.txt";
     2: w_file_name = "weight_itile0_otile0_kij2.txt";
     3: w_file_name = "weight_itile0_otile0_kij3.txt";
     4: w_file_name = "weight_itile0_otile0_kij4.txt";
     5: w_file_name = "weight_itile0_otile0_kij5.txt";
     6: w_file_name = "weight_itile0_otile0_kij6.txt";
     7: w_file_name = "weight_itile0_otile0_kij7.txt";
     8: w_file_name = "weight_itile0_otile0_kij8.txt";
    endcase


    w_file = $fopen(w_file_name, "r");
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

    A_xmem = 11'b10000000000;

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
    A_xmem = 11'b10000000000;
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



    /////// Kernel loading to PEs ///////
    // L0 reads rows staggered (row r one cycle after row r-1); the array
    // delays inst by the same amount, so load and data stay aligned.
    stage_begin;
    for (t=0; t<col; t=t+1) begin
      #0.5 clk = 1'b0;  l0_rd = 1; load = 1;
      #0.5 clk = 1'b1;
    end
    /////////////////////////////////////



    ////// provide some intermission to clear up the kernel loading ///
    #0.5 clk = 1'b0;  load = 0; l0_rd = 0;
    #0.5 clk = 1'b1;


    for (i=0; i<10 ; i=i+1) begin
      #0.5 clk = 1'b0;
      #0.5 clk = 1'b1;
    end
    stage_end(2);
    /////////////////////////////////////



    /////// Activation data writing to L0 ///////
    stage_begin;
    A_xmem = 0;
    for (t=0; t<len_nij+1; t=t+1) begin
      #0.5 clk = 1'b0;
        if (t<len_nij) begin WEN_xmem = 1; CEN_xmem = 0; if (t>0) A_xmem = A_xmem + 1; end
        else           begin WEN_xmem = 1; CEN_xmem = 1; end
        l0_wr = (t>0);
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  l0_wr = 0; CEN_xmem = 1; A_xmem = 0;
    #0.5 clk = 1'b1;
    stage_end(3);
    /////////////////////////////////////



    /////// Execution ///////
    stage_begin;
    for (t=0; t<len_nij; t=t+1) begin
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
      $display("kij=%0d: ERROR, OFIFO is not valid after execution", kij);
      error = error + 1;
    end
    /////////////////////////////////////



    //////// OFIFO READ ////////
    // Ideally, OFIFO should be read while execution, but we have enough ofifo
    // depth so we can fetch out after execution.
    // ofifo_rd is registered inside ofifo.v: entry t is on the output one cycle
    // after rd, so the PMEM write (psum address kij*len_nij + nij) lags by 1.
    stage_begin;
    for (t=0; t<len_nij+1; t=t+1) begin
      #0.5 clk = 1'b0;
        ofifo_rd = (t<len_nij);
        sfu_wb   = 0;
        if (t>0) begin CEN_pmem = 0; WEN_pmem = 0; A_pmem = kij*len_nij + t - 1; end
        else     begin CEN_pmem = 1; WEN_pmem = 1; end
      #0.5 clk = 1'b1;
    end

    #0.5 clk = 1'b0;  ofifo_rd = 0; CEN_pmem = 1; WEN_pmem = 1; A_pmem = 0;
    #0.5 clk = 1'b1;
    stage_end(5);
    /////////////////////////////////////


  end  // end of kij loop

  // flush: inst is registered in this TB, so the last PMEM write lands one edge later
  for (i=0; i<2 ; i=i+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end


  ////////// Psum check (expected PMEM content after all kij) /////////
  psum_file = $fopen("psum.txt", "r");
  psum_scan_file = $fscanf(psum_file,"%s", answer);
  psum_scan_file = $fscanf(psum_file,"%s", answer);
  psum_scan_file = $fscanf(psum_file,"%s", answer);
  j = 0;
  for (i=0; i<len_kij*len_nij; i=i+1) begin
    psum_scan_file = $fscanf(psum_file,"%128b", answer);
    if (core_instance.pmem_instance.memory[i] !== answer) begin
      if (j < 5) begin
        $display("psum ERROR at kij=%0d nij=%0d", i/len_nij, i%len_nij);
        $display("pmem  : %128b", core_instance.pmem_instance.memory[i]);
        $display("answer: %128b", answer);
      end
      j = j + 1;
    end
  end
  $fclose(psum_file);
  if (j == 0) $display("############ psum check: all %0d psum words matched #############", len_kij*len_nij);
  else        begin $display("############ psum check: %0d ERRORS #############", j); error = error + j; end
  //////////////////////////////////


  ////////// Accumulation /////////
  out_file = $fopen("out.txt", "r");
  acc_file = $fopen("acc_add.txt", "r");   // missing in the original template

  // Following three lines are to remove the first three comment lines of the file
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);



  $display("############ Verification Start during accumulation #############");

  stage_begin;
  for (i=0; i<len_onij+1; i=i+1) begin

    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;

    if (i>0) begin
     out_scan_file = $fscanf(out_file,"%128b", answer); // reading from out file to answer
       if (sfp_out == answer)
         $display("%2d-th output featuremap Data matched! :D", i);
       else begin
         $display("%2d-th output featuremap Data ERROR!!", i);
         $display("sfpout: %128b", sfp_out);
         $display("answer: %128b", answer);
         error = error + 1;
       end
    end


    #0.5 clk = 1'b0; reset = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; reset = 0;
    #0.5 clk = 1'b1;

    if (i < len_onij) begin
      for (j=0; j<len_kij+1; j=j+1) begin

        #0.5 clk = 1'b0;
          if (j<len_kij) begin CEN_pmem = 0; WEN_pmem = 1; acc_scan_file = $fscanf(acc_file,"%11b", A_pmem); end
                         else  begin CEN_pmem = 1; WEN_pmem = 1; end

          if (j>0)  acc = 1;
        #0.5 clk = 1'b1;
      end

      // store the accumulated (raw, pre-ReLU) result back to PMEM
      #0.5 clk = 1'b0; acc = 0; CEN_pmem = 0; WEN_pmem = 0; A_pmem = ACC_BASE + i; sfu_wb = 1; relu_off = 1;
      #0.5 clk = 1'b1;

      #0.5 clk = 1'b0; CEN_pmem = 1; WEN_pmem = 1; sfu_wb = 0; relu_off = 0;
      #0.5 clk = 1'b1;
    end
  end
  stage_end(6);

  $fclose(acc_file);
  $fclose(out_file);
  //////////////////////////////////


  ////////// ReLU in SFU and store back /////////
  // read the raw sum, load it into the SFU (1-cycle acc burst), write ReLU result
  stage_begin;
  for (i=0; i<len_onij; i=i+1) begin
    #0.5 clk = 1'b0; CEN_pmem = 0; WEN_pmem = 1; A_pmem = ACC_BASE + i;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; CEN_pmem = 1; acc = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; acc = 0; CEN_pmem = 0; WEN_pmem = 0; A_pmem = RELU_BASE + i; sfu_wb = 1; relu_off = 0;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; CEN_pmem = 1; WEN_pmem = 1; sfu_wb = 0;
    #0.5 clk = 1'b1;
  end
  stage_end(7);
  //////////////////////////////////


  ////////// Read back the stored ReLU results and check /////////
  // load each stored word into the SFU and look at it with ReLU bypassed
  out_file = $fopen("out.txt", "r");
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  out_scan_file = $fscanf(out_file,"%s", answer);
  j = 0;
  for (i=0; i<len_onij; i=i+1) begin
    #0.5 clk = 1'b0; CEN_pmem = 0; WEN_pmem = 1; A_pmem = RELU_BASE + i;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; CEN_pmem = 1; acc = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0; acc = 0; relu_off = 1;
    #0.5 clk = 1'b1;
    #0.5 clk = 1'b0;
    out_scan_file = $fscanf(out_file,"%128b", answer);
    if (sfp_out !== answer) begin
      $display("stored ReLU output %0d ERROR: pmem %128b", i, sfp_out);
      $display("                         answer %128b", answer);
      j = j + 1;
    end
    relu_off = 0;
    #0.5 clk = 1'b1;
  end
  $fclose(out_file);
  if (j == 0) $display("############ stored ReLU outputs in PMEM: all %0d matched #############", len_onij);
  else        begin $display("############ stored ReLU outputs in PMEM: %0d ERRORS #############", j); error = error + j; end
  //////////////////////////////////


  $display("");
  $display("############ Cycle count (all 9 kij) ############");
  $display("  L1 (XMEM) writes, act + weight     : %6d", stage_cycles[0]);
  $display("  kernel XMEM -> L0                  : %6d", stage_cycles[1]);
  $display("  kernel loading to PEs (+intermiss.) : %6d", stage_cycles[2]);
  $display("  activation XMEM -> L0              : %6d", stage_cycles[3]);
  $display("  execution (+array drain)           : %6d", stage_cycles[4]);
  $display("  OFIFO -> PMEM                      : %6d", stage_cycles[5]);
  $display("  per-kij reset                      : %6d", stage_cycles[8]);
  $display("  SFU accumulation + store back      : %6d", stage_cycles[6]);
  $display("  SFU ReLU + store back              : %6d", stage_cycles[7]);
  $display("  total simulated cycles             : %6d", cycle);
  $display("");

  if (error == 0) begin
  	$display("############ No error detected ##############");
  	$display("########### Project Completed !! ############");

  end
  else
  	$display("############ %0d errors detected ##############", error);

  //////////////////////////////////

  for (t=0; t<10; t=t+1) begin
    #0.5 clk = 1'b0;
    #0.5 clk = 1'b1;
  end

  #10 $finish;

end

always @ (posedge clk) begin
   inst_w_q   <= inst_w;
   D_xmem_q   <= D_xmem;
   CEN_xmem_q <= CEN_xmem;
   WEN_xmem_q <= WEN_xmem;
   A_pmem_q   <= A_pmem;
   CEN_pmem_q <= CEN_pmem;
   WEN_pmem_q <= WEN_pmem;
   A_xmem_q   <= A_xmem;
   ofifo_rd_q <= ofifo_rd;
   acc_q      <= acc;
   sfu_wb_q   <= sfu_wb;
   relu_off_q <= relu_off;
   ififo_wr_q <= ififo_wr;
   ififo_rd_q <= ififo_rd;
   l0_rd_q    <= l0_rd;
   l0_wr_q    <= l0_wr ;
   execute_q  <= execute;
   load_q     <= load;
end


endmodule
