// Created by prof. Mingu Kang @VVIP Lab in UCSD ECE department
// Please do not spread this code without permission 
// [Part 2] added `mode`: 0 = unsigned activation (Part 1, X0 nibbles), 1 = signed activation (deltas)
module mac (out, a, b, c, mode);

parameter bw = 4;
parameter psum_bw = 16;

output signed [psum_bw-1:0] out;
input signed  [bw-1:0] a;  // activation
input signed  [bw-1:0] b;  // weight
input signed  [psum_bw-1:0] c;
input mode;


wire signed [2*bw:0] product;
wire signed [psum_bw-1:0] psum;
wire signed [bw:0]   a_pad;

assign a_pad = mode ? {a[bw-1], a} : {1'b0, a}; // [Part 2] sign- or zero-extend the activation
assign product = a_pad * b;

assign psum = product + c;
assign out = psum;

endmodule
