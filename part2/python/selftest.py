"""
Self-test for the Part 2 reference model and generated files.

  1. Model vs PyTorch: the integer model's W x Xi must equal torch conv2d (float64) exactly,
     so the conv convention (kij order, padding, channel order) matches the Part 1 PyTorch model.
  2. Identities: X0 = 16*hi + lo, Y0 = 16*acc_hi + acc_lo, Yi = Y0 + dYi.
  3. Round trip: decode every generated binary file back and compare with the model.

Run after gen_files.py with the same options, e.g.
    python gen_files.py && python selftest.py
    python gen_files.py --weight w.npy && python selftest.py --weight w.npy
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import ref_model as rm
from gen_files import DEFAULT_BATCH, DEFAULT_OUT, act_rows

fails = 0


def check(cond: bool, msg: str):
    global fails
    print(("  PASS  " if cond else "  FAIL  ") + msg)
    fails += 0 if cond else 1


def read_words(path: Path, bits: int, lanes: int, signed: bool = True) -> np.ndarray:
    tokens = path.read_text().split()[3:]          # same as core_tb: skip three %s tokens
    rows = []
    for t in tokens:
        assert len(t) == bits * lanes, f"{path.name}: word length {len(t)} != {bits * lanes}"
        vals = []
        for lane in range(lanes):
            v = int(t[len(t) - bits * (lane + 1): len(t) - bits * lane], 2)
            if signed and v >= 1 << (bits - 1):
                v -= 1 << bits
            vals.append(v)
        rows.append(vals)
    return np.array(rows, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-dir", default=str(DEFAULT_BATCH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    out = Path(args.out)
    summary = json.loads((out / "summary.json").read_text())
    p, k, psum_bw, out_bw = summary["padding"], summary["k"], summary["psum_bw"], summary["out_bw"]

    frames = rm.load_frames(args.batch_dir)
    w = np.load(out / "stimulus" / "weight_int.npy")
    r = rm.run(frames, w, padding=p, k=k)

    print("[1] model vs torch.nn.functional.conv2d")
    try:
        import torch
        import torch.nn.functional as F
        wt = torch.tensor(w, dtype=torch.float64)
        for i in range(rm.FRAMES):
            xt = torch.tensor(frames[i], dtype=torch.float64)[None]
            yt = F.conv2d(xt, wt, padding=p)[0].numpy()             # [oc, Ho, Wo]
            yt = np.rint(yt).astype(np.int64).reshape(rm.COL, -1).T  # [onij, oc]
            check(np.array_equal(yt, r["ref"][i]), f"W x X{i}: integer model == torch conv2d")
    except ImportError:
        print("  SKIP  torch not installed")

    print("[2] identities")
    check(np.array_equal(16 * r["acts"]["x0_hi"] + r["acts"]["x0_lo"], frames[0]), "X0 == 16*hi + lo")
    check(np.array_equal(r["y"][0], r["ref"][0]), "Y0 (two nibble passes) == W x X0 (8-bit)")
    for i in (1, 2, 3):
        check(np.array_equal(r["y"][i], r["y"][0] + r["dy"][i]), f"Y{i} == Y0 + dY{i}")
        if not r["delta_sat"][i].any() and k == 0:
            check(np.array_equal(r["y"][i], r["ref"][i]), f"Y{i} == W x X{i} (no saturation -> exact)")

    print("[3] file round trip")
    stim, gold, dbg = out / "stimulus", out / "golden", out / "debug"
    for name in rm.PASS_ORDER:
        signed = rm.PASS_MODE[name] == 1
        got = read_words(stim / f"{name}.txt", 4, rm.C, signed=signed)
        check(np.array_equal(got, act_rows(rm.pad(r["acts"][name], p))), f"stimulus/{name}.txt")
    for name, x in r["conv_acts"].items():
        got = read_words(stim / "conventional" / f"{name}.txt", 4, rm.C, signed=False)
        check(np.array_equal(got, act_rows(rm.pad(x, p))), f"stimulus/conventional/{name}.txt")
    check(np.array_equal(rm.load_weights_kij_files(stim), w), "weight_itile0_otile0_kij*.txt (via loader)")
    gold_items = {"y0": r["y"][0], **{f"y{i}": r["y"][i] for i in (1, 2, 3)},
                  "acc_x0_hi": r["acc"]["x0_hi"], "acc_x0_lo": r["acc"]["x0_lo"],
                  **{f"dy{i}": r["dy"][i] for i in (1, 2, 3)}}
    for name, Y in gold_items.items():
        check(np.array_equal(read_words(gold / f"{name}.txt", out_bw, rm.COL), Y), f"golden/{name}.txt")
        check(np.array_equal(np.loadtxt(gold / f"{name}_dec.txt", dtype=np.int64), Y), f"golden/{name}_dec.txt")
    if dbg.exists():
        ok = all(np.array_equal(read_words(dbg / f"psum_{n}_kij{kij}.txt", psum_bw, rm.COL), r["psums"][n][kij])
                 for n in rm.PASS_ORDER for kij in range(rm.KIJ))
        check(ok, "debug/psum_*_kij*.txt (45 files)")
        # SFU replay from the debug dumps alone must give the golden outputs
        P = {n: np.stack([read_words(dbg / f"psum_{n}_kij{kij}.txt", psum_bw, rm.COL) for kij in range(rm.KIJ)])
             for n in rm.PASS_ORDER}
        nmap = np.loadtxt(gold / "nij_map_dec.txt", dtype=np.int64)
        acc = {n: rm.sfu_accumulate(P[n], nmap) for n in rm.PASS_ORDER}
        y0 = (acc["x0_hi"] << 4) + acc["x0_lo"]
        check(np.array_equal(y0, gold_items["y0"]), "SFU replay from debug psums -> y0")
        for i in (1, 2, 3):
            check(np.array_equal(y0 + (acc[f"dx{i}"] << k), gold_items[f"y{i}"]), f"SFU replay -> y{i}")

    print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILED'}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
