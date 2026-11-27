// core: corelet + activation/weight SRAM (xmem) + psum SRAM (pmem).
// Only corelet.v is mapped onto the FPGA.
//
// [Part 2] memories (behavioral, not mapped onto the FPGA):
//   xmem 32b x 4096 : X0_hi, X0_lo, dX1..dX3 (5 x 324 words; conventional baseline 8 x 324) + weights
//   pmem (8 x acc_bw)b x 32768 : raw psums of every (kij, pass) + stored Y0 / Yi
// With acc_bw = psum_bw (16) this is the Part 1 design with bigger SRAMs.
module core (clk, inst, ofifo_valid, D_xmem, sfp_out, reset);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter acc_bw = 24;   // [Part 2]
  parameter col = 8;
  parameter row = 8;

  input  clk;
  input  reset;
  input  [42:0] inst;   // [33:0] = course template map, [34] sfu_wb, [35] relu_off, [42:36] Part 2 (see corelet.v)
  input  [bw*row-1:0] D_xmem;
  output ofifo_valid;
  output [acc_bw*col-1:0] sfp_out;

  wire [bw*row-1:0] Q_xmem;
  wire [acc_bw*col-1:0] Q_pmem;
  wire [acc_bw*col-1:0] D_pmem;

  // activation & weight SRAM
  sram_param #(.dw(bw*row), .aw(12)) xmem_instance (   // [Part 2] was sram_32b_w2048
      .CLK(clk),
      .D(D_xmem),
      .Q(Q_xmem),
      .CEN(inst[19]),
      .WEN(inst[18]),
      .A({inst[38], inst[17:7]}));

  // psum SRAM: written from OFIFO (psum movement) or SFU (acc / ReLU store-back),
  // read by the SFU
  sram_param #(.dw(acc_bw*col), .aw(15)) pmem_instance (   // [Part 2] was sram_128b_w2048
      .CLK(clk),
      .D(D_pmem),
      .Q(Q_pmem),
      .CEN(inst[32]),
      .WEN(inst[31]),
      .A({inst[42:39], inst[30:20]}));

  corelet #(.bw(bw), .psum_bw(psum_bw), .acc_bw(acc_bw), .col(col), .row(row)) corelet_instance (
      .clk(clk),
      .reset(reset),
      .inst(inst),
      .l0_in(Q_xmem),
      .pmem_q(Q_pmem),
      .pmem_d(D_pmem),
      .ofifo_valid(ofifo_valid),
      .sfp_out(sfp_out));

endmodule
