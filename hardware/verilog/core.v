// core: corelet + activation/weight SRAM (xmem) + psum SRAM (pmem).
// Only corelet.v is mapped onto the FPGA.
module core (clk, inst, ofifo_valid, D_xmem, sfp_out, reset);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk;
  input  reset;
  input  [33:0] inst;
  input  [bw*row-1:0] D_xmem;
  output ofifo_valid;
  output [psum_bw*col-1:0] sfp_out;

  wire [bw*row-1:0] Q_xmem;
  wire [psum_bw*col-1:0] Q_pmem;
  wire [psum_bw*col-1:0] D_pmem;

  // activation & weight SRAM
  sram_32b_w2048 xmem_instance (
      .CLK(clk),
      .D(D_xmem),
      .Q(Q_xmem),
      .CEN(inst[19]),
      .WEN(inst[18]),
      .A(inst[17:7]));

  // psum SRAM (written from OFIFO, read by the SFU)
  sram_128b_w2048 pmem_instance (
      .CLK(clk),
      .D(D_pmem),
      .Q(Q_pmem),
      .CEN(inst[32]),
      .WEN(inst[31]),
      .A(inst[30:20]));

  corelet #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) corelet_instance (
      .clk(clk),
      .reset(reset),
      .inst(inst),
      .l0_in(Q_xmem),
      .pmem_q(Q_pmem),
      .pmem_d(D_pmem),
      .ofifo_valid(ofifo_valid),
      .sfp_out(sfp_out));

endmodule
