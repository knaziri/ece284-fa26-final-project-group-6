"""
Part 2 bit-exact integer reference model (software model of Part 2 Steps 1-3).

It computes exactly what the RTL is expected to compute, pass by pass and kij by kij,
so every intermediate value can be compared against simulation.

Conventions (keep in sync with RTL / testbench):
  X    : [C=8, H=16, W=16], channel-first, row-major, unsigned 8-bit (0..255)
  W    : [oc=8, ic=8, kh=3, kw=3], signed 4-bit, kij = kh*3 + kw
  nij  : row-major index into the (zero-padded) input plane,  nij  = r*Wp + c
  onij : row-major index into the output plane,                onij = oh*Wo + ow
  Array: weight stationary, row = input channel ic, column = output channel oc.
         For one kij, streaming activation vector X[:, nij] gives
             P[nij, oc] = sum_ic W[oc, ic, kh, kw] * X[ic, nij]
  SFU  : accumulates over kij
             Y[onij, oc] = sum_kij P_kij[nij(onij, kij), oc],
             nij(onij, kij) = (oh + kh) * Wp + (ow + kw)

Pass schedule (weights loaded once per kij, never reloaded across passes):
  for kij in 0..8:
      load W[:, :, kh, kw]
      x0_hi (mode 0, unsigned) -> x0_lo (mode 0) -> dx1 (mode 1, signed) -> dx2 -> dx3

Reconstruction:
  Y0 = 16 * (W x X0_hi) + (W x X0_lo)
  Yi = Y0 + (W x dXi) << k          (k = 0 by default, i.e. plain 4-bit saturation)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

C = 8            # input channels  = array rows
ROW = 8
COL = 8          # output channels = array columns
K = 3            # kernel size
KIJ = K * K
FRAMES = 4
IN_H = IN_W = 16

W_MIN, W_MAX = -8, 7      # signed 4-bit weight
D_MIN, D_MAX = -8, 7      # signed 4-bit delta
NIB_MAX = 15              # unsigned 4-bit nibble

PASS_ORDER = ("x0_hi", "x0_lo", "dx1", "dx2", "dx3")
PASS_MODE = {"x0_hi": 0, "x0_lo": 0, "dx1": 1, "dx2": 1, "dx3": 1}  # 0: unsigned act, 1: signed act


# ----------------------------------------------------------------------------- inputs
def load_frames(batch_dir) -> list[np.ndarray]:
    """Load X0..X3 as int64 arrays of shape [C, H, W]."""
    frames = []
    for i in range(FRAMES):
        x = np.loadtxt(Path(batch_dir) / f"X{i}.txt", dtype=np.int64)
        if x.size != C * IN_H * IN_W:
            raise ValueError(f"X{i}.txt has {x.size} values, expected {C * IN_H * IN_W}")
        if x.min() < 0 or x.max() > 255:
            raise ValueError(f"X{i}.txt out of unsigned 8-bit range: [{x.min()}, {x.max()}]")
        frames.append(x.reshape(C, IN_H, IN_W))
    return frames


def _check_weights(w: np.ndarray) -> np.ndarray:
    w = np.asarray(w, dtype=np.float64)
    if w.size != COL * ROW * KIJ:
        raise ValueError(f"weight has {w.size} values, expected {COL * ROW * KIJ} ([oc,ic,kh,kw])")
    wi = np.rint(w)
    if not np.allclose(w, wi):
        raise ValueError("weights must be integers (pass the quantized integer weights, not float)")
    wi = wi.astype(np.int64).reshape(COL, ROW, K, K)
    if wi.min() < W_MIN or wi.max() > W_MAX:
        raise ValueError(f"weights out of signed 4-bit range: [{wi.min()}, {wi.max()}]")
    return wi


def random_weights(seed: int = 284) -> np.ndarray:
    """Placeholder weights until the real Part 1 squeezed-layer weights arrive."""
    rng = np.random.default_rng(seed)
    return rng.integers(W_MIN, W_MAX + 1, size=(COL, ROW, K, K), dtype=np.int64)


def load_weights(path) -> np.ndarray:
    """Load integer weights from .npy, .pt/.pth (a single tensor) or text (576 numbers, [oc,ic,kh,kw] order)."""
    p = Path(path)
    if p.suffix == ".npy":
        w = np.load(p)
    elif p.suffix in (".pt", ".pth"):
        import torch
        w = torch.load(p, map_location="cpu")
        if not hasattr(w, "numpy"):
            raise ValueError(".pt file must contain a single weight tensor; export it first")
        w = w.detach().double().numpy()
    else:
        w = np.loadtxt(p, comments="#").ravel()
    return _check_weights(w)


def load_weights_kij_files(weight_dir, pattern: str = "weight_itile0_otile0_kij{}.txt") -> np.ndarray:
    """
    Load weights from HW7/project-style per-kij files: 3 comment tokens, then COL lines
    (one per output channel), each a 32-bit binary word {ic7, ..., ic0} of signed 4-bit values.
    """
    w = np.zeros((COL, ROW, K, K), dtype=np.int64)
    for kij in range(KIJ):
        kh, kw = divmod(kij, K)
        tokens = (Path(weight_dir) / pattern.format(kij)).read_text().split()
        words = [t for t in tokens[3:] if t.strip()]
        if len(words) < COL:
            raise ValueError(f"kij{kij}: expected {COL} weight words, got {len(words)}")
        for oc in range(COL):
            word = words[oc]
            for ic in range(ROW):
                bits = word[len(word) - 4 * (ic + 1): len(word) - 4 * ic]
                v = int(bits, 2)
                w[oc, ic, kh, kw] = v - 16 if v >= 8 else v
    return _check_weights(w)


# ----------------------------------------------------------------------------- helpers
def signed_bits(lo: int, hi: int) -> int:
    """Minimum two's-complement width that holds every value in [lo, hi]."""
    n = 1
    while not (-(1 << (n - 1)) <= lo and hi <= (1 << (n - 1)) - 1):
        n += 1
    return n


def split_nibbles(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x >> 4, x & 0xF


def make_delta(xi: np.ndarray, x0: np.ndarray, k: int = 0):
    """dX = Xi - X0, optionally scaled by 2^-k (round half up), then saturated to signed 4-bit."""
    d = xi - x0
    q = np.floor(d / (1 << k) + 0.5).astype(np.int64) if k else d.copy()
    sat = (q < D_MIN) | (q > D_MAX)
    return np.clip(q, D_MIN, D_MAX), d, sat


def pad(x: np.ndarray, p: int) -> np.ndarray:
    return np.pad(x, ((0, 0), (p, p), (p, p))) if p else x


def array_psums(x_pad: np.ndarray, w: np.ndarray) -> np.ndarray:
    """What the 8x8 array emits through OFIFO: P[kij, nij, oc] for all 9 weight sets."""
    xf = x_pad.reshape(C, -1)                       # [ic, nij]
    P = np.empty((KIJ, xf.shape[1], COL), dtype=np.int64)
    for kij in range(KIJ):
        kh, kw = divmod(kij, K)
        P[kij] = xf.T @ w[:, :, kh, kw].T           # [nij, ic] @ [ic, oc]
    return P


def nij_map(Wp: int, Ho: int, Wo: int) -> np.ndarray:
    """[onij, kij] -> nij of the psum that SFU must add for that output/kernel position."""
    oh, ow = np.divmod(np.arange(Ho * Wo), Wo)
    return np.stack([(oh + kh) * Wp + (ow + kw) for kh in range(K) for kw in range(K)], axis=1)


def sfu_accumulate(P: np.ndarray, nmap: np.ndarray) -> np.ndarray:
    """Y[onij, oc] = sum_kij P[kij, nmap[onij, kij], oc]."""
    return sum(P[kij][nmap[:, kij]] for kij in range(KIJ))


# ----------------------------------------------------------------------------- main model
def run(frames: list[np.ndarray], w: np.ndarray, padding: int = 1, k: int = 0) -> dict:
    x0 = frames[0]
    Hp, Wp = IN_H + 2 * padding, IN_W + 2 * padding
    Ho, Wo = Hp - K + 1, Wp - K + 1
    nmap = nij_map(Wp, Ho, Wo)

    # stimulus (what goes into activation SRAM, per pass)
    x0_hi, x0_lo = split_nibbles(x0)
    assert np.array_equal(16 * x0_hi + x0_lo, x0)
    acts = {"x0_hi": x0_hi, "x0_lo": x0_lo}
    delta_raw, delta_sat = {}, {}
    for i in (1, 2, 3):
        q, d, sat = make_delta(frames[i], x0, k)
        acts[f"dx{i}"], delta_raw[i], delta_sat[i] = q, d, sat

    # conventional baseline stimulus: every frame as two nibble passes
    conv_acts = {}
    for i in range(FRAMES):
        conv_acts[f"x{i}_hi"], conv_acts[f"x{i}_lo"] = split_nibbles(frames[i])

    # array outputs per pass per kij, then SFU accumulation over kij
    psums = {name: array_psums(pad(acts[name], padding), w) for name in PASS_ORDER}
    acc = {name: sfu_accumulate(psums[name], nmap) for name in PASS_ORDER}

    y0 = (acc["x0_hi"] << 4) + acc["x0_lo"]
    dy = {i: acc[f"dx{i}"] << k for i in (1, 2, 3)}
    y = {0: y0, **{i: y0 + dy[i] for i in (1, 2, 3)}}

    # full-precision reference W x Xi (8-bit Xi directly, no nibble split / delta)
    ref = {i: sfu_accumulate(array_psums(pad(frames[i], padding), w), nmap) for i in range(FRAMES)}

    return dict(
        padding=padding, k=k, Hp=Hp, Wp=Wp, Ho=Ho, Wo=Wo, nmap=nmap, w=w, frames=frames,
        acts=acts, conv_acts=conv_acts, delta_raw=delta_raw, delta_sat=delta_sat,
        psums=psums, acc=acc, y=y, dy=dy, ref=ref,
    )


def error_stats(y: np.ndarray, ref: np.ndarray) -> dict:
    e = (y - ref).astype(np.float64)
    sig = float(np.sum(ref.astype(np.float64) ** 2))
    noise = float(np.sum(e ** 2))
    return dict(
        max_abs_err=int(np.max(np.abs(e))),
        mse=float(np.mean(e ** 2)),
        sqnr_db=float("inf") if noise == 0 else 10 * np.log10(sig / noise),
        exact=bool(noise == 0),
    )
