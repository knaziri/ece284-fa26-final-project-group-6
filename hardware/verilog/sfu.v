// Special function unit: per-column accumulation (acc) and ReLU on the output.
module sfu (clk, reset, acc, psum_in, sfp_out);

  parameter psum_bw = 16;
  parameter col = 8;

  input  clk;
  input  reset;
  input  acc;
  input  [psum_bw*col-1:0] psum_in;
  output [psum_bw*col-1:0] sfp_out;

  reg signed [psum_bw-1:0] psum_q [col-1:0];

  genvar i;
  for (i=0; i<col; i=i+1) begin : col_num
    wire signed [psum_bw-1:0] in_i = psum_in[psum_bw*(i+1)-1:psum_bw*i];

    always @ (posedge clk) begin
      if (reset)
        psum_q[i] <= 0;
      else if (acc)
        psum_q[i] <= psum_q[i] + in_i;
    end

    // ReLU: negative accumulated value -> 0
    assign sfp_out[psum_bw*(i+1)-1:psum_bw*i] = psum_q[i][psum_bw-1] ? {psum_bw{1'b0}} : psum_q[i];
  end

endmodule
