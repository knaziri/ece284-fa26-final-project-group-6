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
module sfu (clk, reset, acc, relu_off, psum_in, sfp_out);

  parameter psum_bw = 16;
  parameter col = 8;

  input  clk;
  input  reset;
  input  acc;
  input  relu_off;
  input  [psum_bw*col-1:0] psum_in;
  output [psum_bw*col-1:0] sfp_out;

  reg acc_d;  // acc delayed by one cycle, to detect the start of a burst
  wire acc_first = acc & ~acc_d;

  always @ (posedge clk) begin
    if (reset) acc_d <= 1'b0;
    else       acc_d <= acc;
  end

  genvar i;
  generate
  for (i=0; i<col; i=i+1) begin : col_num
    wire signed [psum_bw-1:0] in_i = psum_in[psum_bw*(i+1)-1:psum_bw*i];
    reg  signed [psum_bw-1:0] psum_q;

    always @ (posedge clk) begin
      if (reset)
        psum_q <= {psum_bw{1'b0}};
      else if (acc_first)
        psum_q <= in_i;
      else if (acc)
        psum_q <= psum_q + in_i;
    end

    assign sfp_out[psum_bw*(i+1)-1:psum_bw*i] =
        (!relu_off && psum_q[psum_bw-1]) ? {psum_bw{1'b0}} : psum_q;
  end
  endgenerate

endmodule
