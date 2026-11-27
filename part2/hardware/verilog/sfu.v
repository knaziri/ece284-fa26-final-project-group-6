// Special function unit (one lane per column): accumulation + optional ReLU.
//
//   acc  : while high, add the PMEM read data (psum_in) into the per-column
//          accumulator every cycle. The first cycle of an acc burst (acc
//          rising) LOADS psum_in instead of adding, so the accumulator never
//          needs the global reset (which would also wipe the PE weights).
//   relu_off : sfp_out = max(0, value) when 0 (default, matches the course
//          core_tb.v template, which compares sfp_out with the post-ReLU
//          out.txt), raw accumulated value when 1. Supports both flows:
//            (a) acc 9 psums -> store back (relu_off=1) -> read 1 word (acc 1
//                cycle) -> store back with relu_off=0   (separate ReLU stage)
//            (b) acc 9 psums -> store back with relu_off=0 (fused ReLU)
//
// sfp_out comes straight from the accumulator register, so it is stable and
// can be written into PMEM (see the write-back mux in corelet.v).
//
// [Part 2] (both changes are no-ops for Part 1: acc_bw = psum_bw, shift = 0)
//   acc_bw : accumulator / PMEM lane width (24 for Part 2: Y0 of an 8-bit input needs 19 bits)
//   shift  : the PMEM word is added (or loaded) as psum << 4, for the X0 high-nibble pass:
//            Y0 = 16 * (W x X0_hi) + (W x X0_lo)
//   Reconstruction Yi = Y0 + W x dXi needs no new control: the first read of the acc burst
//   is the stored Y0 word (loaded), the next 9 reads are the dXi psums (added).
module sfu (clk, reset, acc, relu_off, shift, psum_in, sfp_out);

  parameter psum_bw = 16;
  parameter acc_bw = psum_bw;   // [Part 2]
  parameter col = 8;

  input  clk;
  input  reset;
  input  acc;
  input  relu_off;
  input  shift;                 // [Part 2]
  input  [acc_bw*col-1:0] psum_in;
  output [acc_bw*col-1:0] sfp_out;

  reg acc_d;  // acc delayed by one cycle, to detect the start of a burst
  wire acc_first = acc & ~acc_d;

  always @ (posedge clk) begin
    if (reset) acc_d <= 1'b0;
    else       acc_d <= acc;
  end

  genvar i;
  generate
  for (i=0; i<col; i=i+1) begin : col_num
    wire signed [acc_bw-1:0] in_i = psum_in[acc_bw*(i+1)-1:acc_bw*i];
    wire signed [acc_bw-1:0] addend = shift ? (in_i <<< 4) : in_i;   // [Part 2]
    reg  signed [acc_bw-1:0] psum_q;

    always @ (posedge clk) begin
      if (reset)
        psum_q <= {acc_bw{1'b0}};
      else if (acc_first)
        psum_q <= addend;
      else if (acc)
        psum_q <= psum_q + addend;
    end

    assign sfp_out[acc_bw*(i+1)-1:acc_bw*i] =
        (!relu_off && psum_q[acc_bw-1]) ? {acc_bw{1'b0}} : psum_q;
  end
  endgenerate

endmodule
