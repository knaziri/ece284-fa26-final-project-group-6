// corelet: everything that gets mapped on the FPGA (array, L0, OFIFO, SFU).
// The SRAMs live in core.v.
// inst map (same as core_tb.v): [33] acc, [32:20] pmem CEN/WEN/A, [19:7] xmem CEN/WEN/A,
//   [6] ofifo_rd, [5] ififo_wr (unused), [4] ififo_rd (unused), [3] l0_rd, [2] l0_wr,
//   [1] execute, [0] load
module corelet (clk, reset, inst, l0_in, pmem_q, pmem_d, ofifo_valid, sfp_out);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk;
  input  reset;
  input  [33:0] inst;
  input  [row*bw-1:0] l0_in;          // from xmem (activation or weight word)
  input  [psum_bw*col-1:0] pmem_q;    // from psum SRAM read port
  output [psum_bw*col-1:0] pmem_d;    // to psum SRAM write port (OFIFO output)
  output ofifo_valid;
  output [psum_bw*col-1:0] sfp_out;

  wire acc       = inst[33];
  wire ofifo_rd  = inst[6];
  // inst[5], inst[4]: ififo_wr / ififo_rd are not used (weights come through L0)
  wire l0_rd     = inst[3];
  wire l0_wr     = inst[2];
  wire execute   = inst[1];
  wire load      = inst[0];

  wire [row*bw-1:0] l0_out;
  wire l0_full, l0_ready;

  wire [psum_bw*col-1:0] array_out;
  wire [col-1:0] array_valid;

  wire ofifo_full, ofifo_ready;

  l0 #(.row(row), .bw(bw)) l0_instance (
      .clk(clk),
      .in(l0_in),
      .out(l0_out),
      .rd(l0_rd),
      .wr(l0_wr),
      .o_full(l0_full),
      .reset(reset),
      .o_ready(l0_ready));

  mac_array #(.bw(bw), .psum_bw(psum_bw), .col(col), .row(row)) mac_array_instance (
      .clk(clk),
      .reset(reset),
      .out_s(array_out),
      .in_w(l0_out),
      .in_n({psum_bw*col{1'b0}}),
      .inst_w({execute, load}),
      .valid(array_valid));

  ofifo #(.col(col), .bw(psum_bw)) ofifo_instance (
      .clk(clk),
      .in(array_out),
      .out(pmem_d),
      .rd(ofifo_rd),
      .wr(array_valid),
      .o_full(ofifo_full),
      .reset(reset),
      .o_ready(ofifo_ready),
      .o_valid(ofifo_valid));

  sfu #(.psum_bw(psum_bw), .col(col)) sfu_instance (
      .clk(clk),
      .reset(reset),
      .acc(acc),
      .psum_in(pmem_q),
      .sfp_out(sfp_out));

endmodule
