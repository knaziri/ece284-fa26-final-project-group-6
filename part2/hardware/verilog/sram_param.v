// [Part 2] Parameterized behavioral SRAM: same interface and timing as sram_*b_w2048.v
// (read: address registered on the clock edge, Q valid after it; write on the clock edge).
// SRAMs stay in core.v and are not mapped onto the FPGA (only corelet.v is).
module sram_param (CLK, D, Q, CEN, WEN, A);

  parameter dw = 128;      // word width
  parameter aw = 11;       // address width
  parameter num = 1 << aw;

  input  CLK;
  input  WEN;
  input  CEN;
  input  [dw-1:0] D;
  input  [aw-1:0] A;
  output [dw-1:0] Q;

  reg [dw-1:0] memory [num-1:0];
  reg [aw-1:0] add_q;
  assign Q = memory[add_q];

  always @ (posedge CLK) begin
   if (!CEN && WEN) // read
      add_q <= A;
   if (!CEN && !WEN) // write
      memory[A] <= D;
  end

endmodule
