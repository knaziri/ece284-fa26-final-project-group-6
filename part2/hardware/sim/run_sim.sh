#!/usr/bin/env bash
# Run the Part 2 testbench (core_tb_part2.v).
#
# Usage:  ./run_sim.sh [diff|conv|both] [extra iverilog flags, e.g. -DVCD]
#   diff : differential method, 5 passes per weight set  -> diff.vvp, diff.log
#   conv : conventional baseline, 8 passes per weight set -> conv.vvp, conv.log
#   both : (default) both, in parallel
#
# Needs ../../out/ from part2/python/gen_files.py and Icarus Verilog (iverilog, vvp) on PATH.
# Each run takes a few minutes; the console only shows the summary lines, the full output is in *.log.
set -e
cd "$(dirname "$0")"
what=${1:-both}; shift || true

build() {   # build <name> [flags]
  local name=$1; shift
  iverilog -g2012 -o "$name.vvp" -c filelist "$@"
}

case $what in
  diff) build diff "$@";          vvp -n diff.vvp > diff.log ;;
  conv) build conv -DCONV "$@";   vvp -n conv.vvp > conv.log ;;
  both) build diff "$@"; build conv -DCONV "$@"
        vvp -n diff.vvp > diff.log & vvp -n conv.vvp > conv.log & wait ;;
  *) echo "usage: $0 [diff|conv|both] [iverilog flags]"; exit 1 ;;
esac

for name in diff conv; do
  [[ ($what == both || $what == $name) && -f $name.log ]] || continue
  echo "==== $name.log"
  grep -E "####|RESULT" "$name.log"
done
