# Hardware

8x8 weight-stationary systolic array, 4-bit weights (signed) x 4-bit activations (unsigned).

```
core.v            testbench-facing top: corelet + XMEM (act/weight SRAM) + PMEM (psum SRAM)
└─ corelet.v      the part mapped onto the FPGA (Quartus top = corelet)
   ├─ l0.v        8 row FIFOs, west input of the array (weights AND activations)
   ├─ mac_array.v 8x8 array (mac_row -> mac_tile -> mac)
   ├─ ofifo.v     8 column FIFOs, collects psums from the array
   ├─ sfu.v       per-column accumulator + optional ReLU
   └─ PMEM write-back mux: pmem_d = sfu_wb ? sfp_out : ofifo_out
```

## Instruction word (`inst[35:0]`)

Bits `[33:0]` are identical to the course `core_tb.v` template. Bits `[34]` and `[35]` were added so SFU results can be stored back to PMEM.

| bits | field | meaning |
|---|---|---|
| 35 | `relu` | SFU output ReLU (1 = max(0,x), 0 = raw psum) |
| 34 | `sfu_wb` | PMEM write data: 0 = OFIFO output, 1 = SFU output |
| 33 | `acc` | SFU accumulate. First cycle of an `acc` burst **loads**, later cycles add |
| 32 / 31 / 30:20 | `CEN_pmem` / `WEN_pmem` / `A_pmem` | psum SRAM |
| 19 / 18 / 17:7 | `CEN_xmem` / `WEN_xmem` / `A_xmem` | act/weight SRAM |
| 6 | `ofifo_rd` | |
| 5 / 4 | `ififo_wr` / `ififo_rd` | unused in WS (tie to 0) |
| 3 / 2 | `l0_rd` / `l0_wr` | |
| 1 / 0 | `execute` / `load` | |

In `core_tb.v`, widen `inst_q` to `[35:0]` and add:

```verilog
assign inst_q[35] = relu_q;
assign inst_q[34] = sfu_wb_q;
```

## Timing rules (all verified in `tb/core_smoke_tb.v`)

- **XMEM/PMEM read:** data is on `Q` one cycle after the address. So `l0_wr` lags the XMEM read by one cycle, and `acc` lags the PMEM read by one cycle.
- **OFIFO -> PMEM:** `ofifo_rd` is registered inside `ofifo.v`, so entry *j* is on the output one cycle after `ofifo_rd`. Write PMEM (`WEN_pmem=0`, `sfu_wb=0`) one cycle behind `ofifo_rd`.
- **Kernel load:** `l0_rd=1, load=1` for `col` cycles, then idle about `row+col` cycles before loading activations or executing. The first weight word read from L0 goes to column 0. In a weight word, row *r* sits at bits `[4r+3:4r]`.
- **Execute:** `l0_rd=1, execute=1` for nij cycles, then idle about `row+col` cycles so the array drains into the OFIFO.
- **SFU accumulate + store back:** read the 9 psum addresses back-to-back with `acc=1` delayed by one cycle. Then in one cycle: `acc=0, sfu_wb=1, WEN_pmem=0, A_pmem=<dst>`. `relu=1` in that cycle gives fused ReLU.
- **Separate ReLU stage:** read 1 word, `acc=1` for 1 cycle (loads it), then store back with `relu=1, sfu_wb=1`.
- The SFU no longer needs `reset` between output pixels. Global `reset` also clears PE weights and FIFOs, so only use it between kij.

## Running the smoke test

From `hardware/`:

```bash
iverilog -g2012 -o smoke.vvp -c filelist tb/core_smoke_tb.v && vvp smoke.vvp
```

This runs all 7 project stages on random data sized like `features[27]` (nij=36, kij=9) with a built-in golden, and should print `PASS: 0 mismatches`. Add `+vcd` to dump `core_smoke_tb.vcd` for waveforms. `tb/core_smoke_tb.v` is a working reference for the stage sequencing in the real `core_tb.v`.

## Notes

- **psum width:** `psum_bw=16`. The Part 1 worst case is 15*8*8*9 = 8640, so it fits. Part 2's 8-bit Y0 needs about 20 bits. PMEM (`sram_128b_w2048`) is 8x16 bits, so widening `psum_bw` also needs a wider PMEM or a dedicated Y0 bank.
- **FPGA pins:** with `corelet` as the Quartus top there are 455 I/O pins (36 inst + 32 + 3x128 + 3). This is close to the EP4CGX150DF31 user-I/O limit, so check the fitter report. Assigning the wide buses as virtual pins avoids the problem. Widening `psum_bw` will definitely overflow the pin count.
