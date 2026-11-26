// corelet: everything that gets mapped on the FPGA (L0, MAC array, OFIFO, SFU,
// PMEM write-back mux). The SRAMs live in core.v.
//
// inst map ([33:0] identical to the course core_tb.v template):
//   [35]    relu      SFU output ReLU enable (1: max(0,x), 0: raw accumulated psum)
//   [34]    sfu_wb    PMEM write-data select (0: OFIFO output, 1: SFU output)
//   [33]    acc       SFU accumulate (first cycle of a burst loads, then adds)
//   [32]    CEN_pmem  [31] WEN_pmem  [30:20] A_pmem
//   [19]    CEN_xmem  [18] WEN_xmem  [17:7]  A_xmem
//   [6]     ofifo_rd
//   [5]     ififo_wr  (unused in WS: weights come from the west through L0)
//   [4]     ififo_rd  (unused in WS)
//   [3]     l0_rd     [2] l0_wr
//   [1]     execute   [0] load (kernel loading)
module corelet (clk, reset, inst, l0_in, pmem_q, pmem_d, ofifo_valid, sfp_out);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter col = 8;
  parameter row = 8;

  input  clk;
  input  reset;
  input  [35:0] inst;
  input  [row*bw-1:0] l0_in;          // from xmem (activation or weight word)
  input  [psum_bw*col-1:0] pmem_q;    // from psum SRAM read port
  output [psum_bw*col-1:0] pmem_d;    // to psum SRAM write port
  output ofifo_valid;
  output [psum_bw*col-1:0] sfp_out;

  wire relu      = inst[35];
  wire sfu_wb    = inst[34];
  wire acc       = inst[33];
  wire ofifo_rd  = inst[6];
  wire l0_rd     = inst[3];
  wire l0_wr     = inst[2];
  wire execute   = inst[1];
  wire load      = inst[0];

  wire [row*bw-1:0] l0_out;
  wire l0_full, l0_ready;

  wire [psum_bw*col-1:0] array_out;
  wire [col-1:0] array_valid;

  wire [psum_bw*col-1:0] ofifo_out;
  wire ofifo_full, ofifo_ready;

  // PMEM write data: OFIFO (psum movement) or SFU (accumulated / ReLU result)
  assign pmem_d = sfu_wb ? sfp_out : ofifo_out;

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
      .out(ofifo_out),
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
      .relu(relu),
      .psum_in(pmem_q),
      .sfp_out(sfp_out));

endmodule
