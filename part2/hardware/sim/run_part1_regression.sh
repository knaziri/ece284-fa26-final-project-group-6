#!/usr/bin/env bash
# Run the group's Part 1 testbench (hardware/core_tb.v + its data files) on the Part 2 RTL.
# Usage: ./run_part1_regression.sh [path to the group's hardware/ folder]
# Work files go to part1_regression/ (not committed); the log is written to part1_regression.log here.
set -e
cd "$(dirname "$0")"
HW=${1:-}
if [[ -z $HW ]]; then
  for c in ../../../hardware ../../../ece284-fa26-final-project-group-6/hardware; do
    [[ -f $c/core_tb.v ]] && HW=$c && break
  done
fi
[[ -f $HW/core_tb.v ]] || { echo "Part 1 hardware/ folder not found"; exit 1; }
rm -rf part1_regression && mkdir part1_regression
cp "$HW"/core_tb.v "$HW"/activation_tile0.txt "$HW"/weight_itile0_otile0_kij*.txt \
   "$HW"/psum.txt "$HW"/acc_add.txt "$HW"/out.txt part1_regression/
sed 's#^\.\./verilog/#../../verilog/#; /core_tb_part2.v/d' filelist > part1_regression/filelist
cd part1_regression
iverilog -g2012 -o p1.vvp -c filelist core_tb.v ../part1_cfg.v
vvp -n p1.vvp > ../part1_regression.log
grep -E "####|ERROR|total simulated" ../part1_regression.log
rm -f core_tb.vcd
