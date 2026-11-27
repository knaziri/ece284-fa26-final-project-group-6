// Part 1 configuration for the regression run: PMEM lane / SFU width = psum_bw (16),
// which makes the Part 2 RTL behave exactly like the Part 1 design (mode = shift = 0 come
// from the zero-padded upper inst bits of the 36-bit Part 1 testbench).
module part1_cfg;
  defparam core_tb.core_instance.acc_bw = 16;
endmodule
