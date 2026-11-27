// corelet: everything that gets mapped on the FPGA (L0, MAC array, OFIFO, SFU,
// PMEM write-back mux). The SRAMs live in core.v.
//
// inst map ([33:0] identical to the course core_tb.v template). The two extra
// bits default to 0 = template behaviour, so the unmodified 34-bit template
// (zero-padded on this 36-bit port) still works:
//   [35]    relu_off  0: sfp_out = ReLU(acc) (template default), 1: raw accumulated psum
//   [34]    sfu_wb    PMEM write-data select (0: OFIFO output, 1: SFU output)
//   [33]    acc       SFU accumulate (first cycle of a burst loads, then adds)
//   [32]    CEN_pmem  [31] WEN_pmem  [30:20] A_pmem
//   [19]    CEN_xmem  [18] WEN_xmem  [17:7]  A_xmem
//   [6]     ofifo_rd
//   [5]     ififo_wr  (unused in WS: weights come from the west through L0)
//   [4]     ififo_rd  (unused in WS)
//   [3]     l0_rd     [2] l0_wr
//   [1]     execute   [0] load (kernel loading)
//
// [Part 2] extension (all-zero = Part 1 behaviour, so Part 1 testbenches still work):
//   [36]    mode       0: unsigned activation (Part 1, X0 nibbles), 1: signed activation (deltas)
//   [37]    sfu_shift  SFU adds/loads the PMEM word as psum << 4 (X0 high-nibble pass)
//   [38]    A_xmem[11]     (used in core.v)
//   [42:39] A_pmem[14:11]  (used in core.v)
//   PMEM lanes are acc_bw wide: OFIFO psums are sign-extended on the way in, so stored
//   results (Y0) and raw psums share one SRAM and both feed the SFU.
module corelet (clk, reset, inst, l0_in, pmem_q, pmem_d, ofifo_valid, sfp_out);

  parameter bw = 4;
  parameter psum_bw = 16;
  parameter acc_bw = 24;           // [Part 2] PMEM lane / SFU accumulator width
  parameter col = 8;
  parameter row = 8;

  input  clk;
  input  reset;
  input  [42:0] inst;
  input  [row*bw-1:0] l0_in;          // from xmem (activation or weight word)
  input  [acc_bw*col-1:0] pmem_q;     // from psum SRAM read port
  output [acc_bw*col-1:0] pmem_d;     // to psum SRAM write port
  output ofifo_valid;
  output [acc_bw*col-1:0] sfp_out;

  wire relu_off  = inst[35];
  wire sfu_wb    = inst[34];
  wire acc       = inst[33];
  wire ofifo_rd  = inst[6];
  wire l0_rd     = inst[3];
  wire l0_wr     = inst[2];
  wire execute   = inst[1];
  wire load      = inst[0];
  wire mode      = inst[36];   // [Part 2]
  wire sfu_shift = inst[37];   // [Part 2]

  wire [row*bw-1:0] l0_out;
  wire l0_full, l0_ready;

  wire [psum_bw*col-1:0] array_out;
  wire [col-1:0] array_valid;

  wire [psum_bw*col-1:0] ofifo_out;
  wire ofifo_full, ofifo_ready;

  // PMEM write data: OFIFO (psum movement) or SFU (accumulated / ReLU result)
  // [Part 2] OFIFO psums are sign-extended from psum_bw to acc_bw per lane
  wire [acc_bw*col-1:0] ofifo_out_ext;
  genvar g;
  generate
  for (g=0; g<col; g=g+1) begin : sext
    assign ofifo_out_ext[acc_bw*(g+1)-1:acc_bw*g] =
        {{(acc_bw-psum_bw){ofifo_out[psum_bw*(g+1)-1]}}, ofifo_out[psum_bw*(g+1)-1:psum_bw*g]};
  end
  endgenerate
  assign pmem_d = sfu_wb ? sfp_out : ofifo_out_ext;

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
      .valid(array_valid),
      .mode(mode));

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

  sfu #(.psum_bw(psum_bw), .acc_bw(acc_bw), .col(col)) sfu_instance (
      .clk(clk),
      .reset(reset),
      .acc(acc),
      .relu_off(relu_off),
      .shift(sfu_shift),
      .psum_in(pmem_q),
      .sfp_out(sfp_out));

endmodule
