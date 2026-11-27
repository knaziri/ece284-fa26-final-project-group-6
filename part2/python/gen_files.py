"""
Generate Part 2 stimulus and golden files from X0..X3 (+ weights) using ref_model.

Usage (from anywhere):
    python gen_files.py                              # default: Part 1 weight files of features[27]
                                                     #   (<repo>/hardware/weight_itile0_otile0_kij*.txt)
    python gen_files.py --weight-dir <dir>           # Part 1 weight files from another folder
    python gen_files.py --weight w_int.npy           # integer weights: .npy/.pt/.txt, [oc,ic,kh,kw]
    python gen_files.py --random-weights             # placeholder random int4 weights (--seed)
    python gen_files.py --padding 0 --out-bw 32      # other options, see --help

File format (same style as the course core_tb.v template):
    - 3 comment lines without spaces (core_tb skips them with three %s scans)
    - binary files: one word per line, MSB = highest index (ch7/oc7), LSB = index 0
      activation : one line per nij,  {ch7..ch0} x 4 bit   (nibbles unsigned, deltas 2's complement)
      weight     : one line per oc,   {ic7..ic0} x 4 bit   (2's complement), one file per kij
      psum/out   : one line per nij/onij, {oc7..oc0} x N bit (2's complement)
    - *_dec.txt : same data in decimal, index 0 first, for humans / debugging
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np

import ref_model as rm

HERE = Path(__file__).resolve().parent
# X0..X3: part2/data/ (inside the group repo) or ../differential_input_batch (local working folder)
DEFAULT_BATCH = next((d for d in (HERE.parent / "data", HERE.parents[1] / "differential_input_batch")
                      if (d / "X0.txt").exists()), HERE.parent / "data")
DEFAULT_OUT = HERE.parent / "out"
# Part 1 hardware/ folder (holds weight_itile0_otile0_kij*.txt): <repo>/hardware when part2/ sits in
# the group repo, or the cloned repo next to part2/ in the local working folder
PART1_HW_CANDIDATES = [HERE.parents[1] / "hardware",
                       HERE.parents[1] / "ece284-fa26-final-project-group-6" / "hardware"]


# ----------------------------------------------------------------------------- writers
def to_bin(v: int, bits: int) -> str:
    lo, hi = -(1 << (bits - 1)), (1 << bits) - 1
    if not lo <= v <= hi:
        raise ValueError(f"value {v} does not fit in {bits} bits")
    return format(v & ((1 << bits) - 1), f"0{bits}b")


def write_words(path: Path, rows: np.ndarray, bits: int, comments: list[str], signed: bool = True):
    """rows: [n_lines, n_lanes]; lane 0 goes to the LSB end of each word."""
    rows = np.asarray(rows, dtype=np.int64)
    if signed:
        lim_lo, lim_hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    else:
        lim_lo, lim_hi = 0, (1 << bits) - 1
    if rows.min() < lim_lo or rows.max() > lim_hi:
        raise ValueError(f"{path.name}: values [{rows.min()}, {rows.max()}] do not fit "
                         f"{'signed' if signed else 'unsigned'} {bits}-bit")
    assert len(comments) == 3 and all(" " not in c for c in comments)
    with open(path, "w", newline="\n") as f:
        f.write("\n".join(comments) + "\n")
        for r in rows:
            f.write("".join(to_bin(int(v), bits) for v in r[::-1]) + "\n")


def write_dec(path: Path, rows: np.ndarray, header: str):
    with open(path, "w", newline="\n") as f:
        f.write(f"# {header}\n")
        for r in np.asarray(rows, dtype=np.int64):
            f.write(" ".join(str(int(v)) for v in r) + "\n")


def act_rows(x_pad: np.ndarray) -> np.ndarray:
    """[C, Hp, Wp] -> [nij, C]"""
    return x_pad.reshape(rm.C, -1).T


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch-dir", default=str(DEFAULT_BATCH), help="folder with X0..X3.txt")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output folder (recreated)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--weight", help="integer weights file (.npy/.pt/.txt), [oc,ic,kh,kw]")
    g.add_argument("--weight-dir", help="folder with weight_itile0_otile0_kij{0..8}.txt (default: Part 1 hardware/)")
    g.add_argument("--random-weights", action="store_true", help="placeholder random int4 weights")
    ap.add_argument("--seed", type=int, default=284, help="seed for placeholder weights")
    ap.add_argument("--padding", type=int, default=1, choices=[0, 1], help="conv zero padding")
    ap.add_argument("--k", type=int, default=0, help="delta scale 2^k before 4-bit saturation (0 = none)")
    ap.add_argument("--psum-bw", type=int, default=16, help="array psum width (mac_array psum_bw)")
    ap.add_argument("--out-bw", type=int, default=24, help="SFU accumulator / Y storage width per lane")
    ap.add_argument("--no-debug", action="store_true", help="skip per-pass per-kij psum dumps")
    args = ap.parse_args()

    frames = rm.load_frames(args.batch_dir)
    if args.weight:
        w, w_src = rm.load_weights(args.weight), f"file:{args.weight}"
    elif args.weight_dir:
        w, w_src = rm.load_weights_kij_files(args.weight_dir), f"kij-files:{args.weight_dir}"
    elif args.random_weights:
        w, w_src = rm.random_weights(args.seed), f"PLACEHOLDER random int4, seed={args.seed}"
    else:
        hw = next((c for c in PART1_HW_CANDIDATES if (c / "weight_itile0_otile0_kij0.txt").exists()), None)
        if hw is None:
            raise SystemExit("Part 1 weight files not found; pass --weight-dir, --weight or --random-weights")
        w, w_src = rm.load_weights_kij_files(hw), f"Part 1 weight files: {Path(os.path.relpath(hw, HERE.parent)).as_posix()}"

    r = rm.run(frames, w, padding=args.padding, k=args.k)
    p = args.padding

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    stim, conv, gold, dbg = out / "stimulus", out / "stimulus" / "conventional", out / "golden", out / "debug"
    for d in (stim, conv, gold) + (() if args.no_debug else (dbg,)):
        d.mkdir(parents=True)

    # ---------------- stimulus: activations (padded, one line per nij)
    for name in rm.PASS_ORDER:
        signed = rm.PASS_MODE[name] == 1
        rows = act_rows(rm.pad(r["acts"][name], p))
        write_words(stim / f"{name}.txt", rows, 4,
                    [f"#{name}_mode{rm.PASS_MODE[name]}_nij{len(rows)}_pad{p}",
                     "#each_line=one_nij",
                     "#ch7[msb]...ch0[lsb]_4bit_" + ("signed" if signed else "unsigned")],
                    signed=signed)
        write_dec(stim / f"{name}_dec.txt", rows, f"{name}: one line per nij, ch0..ch7")

    for name, x in r["conv_acts"].items():
        rows = act_rows(rm.pad(x, p))
        write_words(conv / f"{name}.txt", rows, 4,
                    [f"#{name}_mode0_nij{len(rows)}_pad{p}", "#each_line=one_nij",
                     "#ch7[msb]...ch0[lsb]_4bit_unsigned"], signed=False)

    # ---------------- stimulus: weights, one file per kij (template naming)
    for kij in range(rm.KIJ):
        kh, kw = divmod(kij, rm.K)
        rows = w[:, :, kh, kw]                         # [oc, ic]
        write_words(stim / f"weight_itile0_otile0_kij{kij}.txt", rows, 4,
                    [f"#weight_kij{kij}_kh{kh}_kw{kw}", "#each_line=one_output_channel(col)",
                     "#ic7[msb]...ic0[lsb]_4bit_signed"])
    np.save(stim / "weight_int.npy", w)

    # ---------------- golden
    ob = args.out_bw
    gold_items = {
        "y0": r["y"][0],
        **{f"y{i}": r["y"][i] for i in (1, 2, 3)},
        "acc_x0_hi": r["acc"]["x0_hi"], "acc_x0_lo": r["acc"]["x0_lo"],
        **{f"dy{i}": r["dy"][i] for i in (1, 2, 3)},
    }
    for name, Y in gold_items.items():
        write_words(gold / f"{name}.txt", Y, ob,
                    [f"#{name}_onij{len(Y)}_Ho{r['Ho']}_Wo{r['Wo']}", "#each_line=one_onij",
                     f"#oc7[msb]...oc0[lsb]_{ob}bit_signed"])
        write_dec(gold / f"{name}_dec.txt", Y, f"{name}: one line per onij, oc0..oc7")
    for i in range(rm.FRAMES):
        write_dec(gold / f"ref_full_y{i}_dec.txt", r["ref"][i],
                  f"W x X{i} with full 8-bit X{i}: one line per onij, oc0..oc7")
    write_dec(gold / "nij_map_dec.txt", r["nmap"], "per onij: nij for kij0..kij8 (SFU accumulation addresses)")

    # ---------------- debug: raw array outputs, per pass per kij (what OFIFO should emit)
    if not args.no_debug:
        for name in rm.PASS_ORDER:
            for kij in range(rm.KIJ):
                P = r["psums"][name][kij]
                write_words(dbg / f"psum_{name}_kij{kij}.txt", P, args.psum_bw,
                            [f"#psum_{name}_kij{kij}_nij{len(P)}", "#each_line=one_nij",
                             f"#oc7[msb]...oc0[lsb]_{args.psum_bw}bit_signed"])

    # ---------------- summary
    def rng(a):
        return [int(np.min(a)), int(np.max(a))]

    widths = {
        "array_psum": rng(np.stack([r["psums"][n] for n in rm.PASS_ORDER])),
        "acc_x0_hi": rng(r["acc"]["x0_hi"]), "acc_x0_lo": rng(r["acc"]["x0_lo"]),
        "y0": rng(r["y"][0]), "dy": rng(np.stack([r["dy"][i] for i in (1, 2, 3)])),
        "y1_y3": rng(np.stack([r["y"][i] for i in (1, 2, 3)])),
    }
    widths = {k_: {"range": v, "signed_bits": rm.signed_bits(*v)} for k_, v in widths.items()}
    taps = rm.ROW * rm.KIJ
    worst = {   # data-independent bound with |W| <= 8
        "array_psum_one_kij": rm.signed_bits(-15 * 8 * rm.ROW, 15 * 8 * rm.ROW),
        "acc_nibble_pass": rm.signed_bits(-15 * 8 * taps, 15 * 8 * taps),
        "y0_any_8bit_input": rm.signed_bits(-255 * 8 * taps, 255 * 8 * taps),
        "dy": rm.signed_bits(-8 * 8 * taps << args.k, 8 * 8 * taps << args.k),
    }
    frames_stats = {}
    for i in (1, 2, 3):
        d, sat = r["delta_raw"][i], r["delta_sat"][i]
        frames_stats[f"frame{i}"] = {
            "delta_raw_range": rng(d), "delta_std": float(d.std()),
            "delta_zero_frac": float(np.mean(r["acts"][f"dx{i}"] == 0)),
            "saturation_frac": float(sat.mean()),
            "vs_full_8bit_ref": rm.error_stats(r["y"][i], r["ref"][i]),
        }
    summary = {
        "weight_source": w_src,
        "padding": p, "k": args.k, "psum_bw": args.psum_bw, "out_bw": ob,
        "shapes": {"input": [rm.C, rm.IN_H, rm.IN_W], "padded": [rm.C, r["Hp"], r["Wp"]],
                   "nij_per_pass": r["Hp"] * r["Wp"], "output": [rm.COL, r["Ho"], r["Wo"]],
                   "onij": r["Ho"] * r["Wo"]},
        "pass_order_per_kij": list(rm.PASS_ORDER), "pass_mode": rm.PASS_MODE,
        "passes": {"conventional": rm.FRAMES * 2, "differential": 2 + 3,
                   "reduction": 1 - 5 / 8, "ideal_speedup": 8 / 5},
        "y0_exact_vs_full_ref": bool(np.array_equal(r["y"][0], r["ref"][0])),
        "frames": frames_stats,
        "bit_widths_observed": widths,
        "bit_widths_worst_case": worst,
    }
    fits = widths["array_psum"]["signed_bits"] <= args.psum_bw and \
        max(v["signed_bits"] for k_, v in widths.items() if k_ != "array_psum") <= ob
    summary["fits_configured_widths"] = bool(fits)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    # ---------------- console report
    print(f"weights      : {w_src}")
    print(f"shape        : in {summary['shapes']['padded']} (pad {p}) -> out {summary['shapes']['output']}, "
          f"{summary['shapes']['nij_per_pass']} nij/pass, {summary['shapes']['onij']} onij")
    print(f"Y0 == W*X0   : {summary['y0_exact_vs_full_ref']}")
    for i in (1, 2, 3):
        s = frames_stats[f"frame{i}"]
        e = s["vs_full_8bit_ref"]
        print(f"frame {i}      : dX range {s['delta_raw_range']}, sat {s['saturation_frac']:.2%}, "
              f"zeros {s['delta_zero_frac']:.1%} | vs W*X{i}: max|err| {e['max_abs_err']}, "
              f"MSE {e['mse']:.3g}, SQNR {e['sqnr_db']} dB")
    print("bit widths   : " + ", ".join(f"{k_} {v['signed_bits']}b" for k_, v in widths.items()))
    print(f"worst case   : {worst}")
    print(f"fits psum_bw={args.psum_bw}, out_bw={ob}: {fits}")
    print(f"passes       : conventional 8, differential 5 (-37.5%, 1.6x ideal)")
    print(f"written to   : {out}")


if __name__ == "__main__":
    main()
