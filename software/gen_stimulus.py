"""Generate the stimulus and expected-output text files for core_tb.v (Part 1, Step 4).

Run from the software/ directory:

    python gen_stimulus.py [--data-root ./data] [--out-dir ../hardware] [--image 0]

It loads the squeezed 4-bit VGG16 checkpoint, captures the input of the squeezed 8x8 layer
(features[27]) and of the next layer (features[29]) for one CIFAR-10 test image, and writes
the files hardware/core_tb.v opens from its working directory (hardware/):

    activation_tile0.txt             36 lines, 32 bits: padded 6x6 input, one line per nij, row7..row0
    weight_itile0_otile0_kij<k>.txt  8 lines, 32 bits: one line per array column (output channel), row7..row0
    psum.txt                         324 lines, 128 bits: array output before accumulation, [kij][nij], col7..col0
    acc_add.txt                      144 lines, 11 bits: psum SRAM addresses to accumulate, 9 per output pixel
    out.txt                          16 lines, 128 bits: accumulated + ReLU output, one line per o_nij, col7..col0

Array mapping: row r = input channel r, column c = output channel c.
"""
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from models import *
from models.quant_layer import QuantConv2d, act_quantization

SQ, NEXT = 27, 29          # squeezed layer and the layer after its ReLU
W_BIT, ACT_BIT = 4, 4
PSUM_BW = 16
ARRAY = 8
HEADER_LINES = 3           # core_tb.v skips three comment tokens (no spaces allowed in them)


def to_bin(value, bits):
    """Two's-complement binary string of an integer."""
    value = int(value)
    assert -(1 << (bits - 1)) <= value < (1 << bits), f"{value} does not fit in {bits} bits"
    return format(value & ((1 << bits) - 1), f"0{bits}b")


def write_file(path, header, lines):
    assert len(header) == HEADER_LINES and all(" " not in h for h in header)
    with open(path, "w", newline="\n") as f:
        for h in header:
            f.write(h + "\n")
        for line in lines:
            f.write(line + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="result/VGG16_quant_squeeze/model_best.pth.tar")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-dir", default="../hardware")
    ap.add_argument("--image", type=int, default=0, help="index of the test image in the first batch")
    ap.add_argument("--batch-size", type=int, default=128)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TF32 would add ~1e-3 relative error to the float reference
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    model = VGG16_quant_squeeze()
    checkpoint = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    print(f"loaded {args.ckpt} (best_prec {float(checkpoint['best_prec']):.2f}%)")

    sq_layer, next_layer = model.features[SQ], model.features[NEXT]
    assert isinstance(sq_layer, QuantConv2d) and sq_layer.in_channels == ARRAY and sq_layer.out_channels == ARRAY
    assert isinstance(model.features[SQ + 1], nn.ReLU) and isinstance(next_layer, QuantConv2d)

    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    captured = {}
    h_sq = sq_layer.register_forward_pre_hook(lambda m, i: captured.__setitem__("sq", i[0].detach()))
    h_next = next_layer.register_forward_pre_hook(lambda m, i: captured.__setitem__("next", i[0].detach()))
    images, _ = next(iter(testloader))
    with torch.no_grad():
        model(images.to(device))
    h_sq.remove()
    h_next.remove()
    x_sq, x_next = captured["sq"], captured["next"]      # [B, 8, 4, 4] each

    # ---- integer weights / activations -------------------------------------------------
    w_alpha = sq_layer.weight_quant.wgt_alpha
    a_alpha = sq_layer.act_alpha
    w_scale = (w_alpha / (2 ** (W_BIT - 1) - 1)).detach()
    a_scale = (a_alpha / (2 ** ACT_BIT - 1)).detach()
    with torch.no_grad():
        weight_int = sq_layer.weight_q / w_scale
        act_int = act_quantization(ACT_BIT)(x_sq, a_alpha) / a_scale
    assert torch.allclose(weight_int, weight_int.round(), atol=1e-3)
    assert torch.allclose(act_int, act_int.round(), atol=1e-3)
    weight_int, act_int = weight_int.round(), act_int.round()
    assert weight_int.min() >= -7 and weight_int.max() <= 7
    assert act_int.min() >= 0 and act_int.max() <= 15

    # ---- map onto the 8x8 array (same math as the notebook) ----------------------------
    padding, stride = sq_layer.padding[0], sq_layer.stride[0]
    a_int = act_int[args.image]                                            # [8, 4, 4]
    w_int = weight_int.reshape(weight_int.size(0), weight_int.size(1), -1)  # [oc, ic, kij]
    ic, ni, nj = a_int.shape
    a_pad = torch.zeros(ic, ni + 2 * padding, nj + 2 * padding, device=device)
    a_pad[:, padding:padding + ni, padding:padding + nj] = a_int
    a_pad_ni_dim = a_pad.size(1)                                           # 6
    a_pad = a_pad.reshape(ic, -1)                                          # [8, 36]
    len_nij, len_kij = a_pad.size(1), w_int.size(2)
    ki_dim = int(math.sqrt(len_kij))

    psum = torch.zeros(ARRAY, len_nij, len_kij, device=device)             # [col, nij, kij]
    for kij in range(len_kij):
        psum[:, :, kij] = w_int[:, :, kij] @ a_pad

    o_ni_dim = (a_pad_ni_dim - ki_dim) // stride + 1                       # 4
    len_onij = o_ni_dim ** 2
    out = torch.zeros(ARRAY, len_onij, device=device)
    acc_addr = []                                                          # [o_nij][kij] psum SRAM address
    for o_nij in range(len_onij):
        addrs = []
        for kij in range(len_kij):
            nij = (o_nij // o_ni_dim) * stride * a_pad_ni_dim + (o_nij % o_ni_dim) * stride \
                  + (kij // ki_dim) * a_pad_ni_dim + kij % ki_dim
            out[:, o_nij] += psum[:, nij, kij]
            addrs.append(kij * len_nij + nij)                              # psum SRAM is filled kij by kij
        acc_addr.append(addrs)
    out_relu = torch.relu(out)

    # ---- check against the next layer's prehooked input --------------------------------
    psum_recovered = out_relu * a_scale * w_scale
    ref = x_next[args.image].reshape(ARRAY, -1)
    err = (psum_recovered - ref).abs().max().item()
    print(f"image {args.image}: max |psum_recovered - next-layer input| = {err:.3e} "
          f"(largest output {ref.abs().max().item():.3f})")
    assert err < 1e-3, "integer model does not match the network"

    # ---- write files -------------------------------------------------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    p = lambda name: os.path.join(args.out_dir, name)
    rows_hi_to_lo = range(ARRAY - 1, -1, -1)

    write_file(p("activation_tile0.txt"),
               ["#time0row7[msb-lsb],time0row6[msb-lsb],....,time0row0[msb-lsb]#",
                "#time1row7[msb-lsb],time1row6[msb-lsb],....,time1row0[msb-lsb]#",
                "#................#"],
               ["".join(to_bin(a_pad[r, t], ACT_BIT) for r in rows_hi_to_lo) for t in range(len_nij)])

    for kij in range(len_kij):
        write_file(p(f"weight_itile0_otile0_kij{kij}.txt"),
                   ["#col0row7[msb-lsb],col0row6[msb-lsb],....,col0row0[msb-lsb]#",
                    "#col1row7[msb-lsb],col1row6[msb-lsb],....,col1row0[msb-lsb]#",
                    "#................#"],
                   ["".join(to_bin(w_int[c, r, kij], W_BIT) for r in rows_hi_to_lo) for c in range(ARRAY)])

    write_file(p("psum.txt"),
               ["#kij0nij0col7[msb-lsb],kij0nij0col6[msb-lsb],....,kij0nij0col0[msb-lsb]#",
                "#kij0nij1col7[msb-lsb],kij0nij1col6[msb-lsb],....,kij0nij1col0[msb-lsb]#",
                "#................#"],
               ["".join(to_bin(psum[c, t, kij], PSUM_BW) for c in rows_hi_to_lo)
                for kij in range(len_kij) for t in range(len_nij)])

    write_file(p("out.txt"),
               ["#onij0col7[msb-lsb],onij0col6[msb-lsb],....,onij0col0[msb-lsb]#",
                "#onij1col7[msb-lsb],onij1col6[msb-lsb],....,onij1col0[msb-lsb]#",
                "#................#"],
               ["".join(to_bin(out_relu[c, o], PSUM_BW) for c in rows_hi_to_lo) for o in range(len_onij)])

    with open(p("acc_add.txt"), "w", newline="\n") as f:               # no header (read with %11b)
        for addrs in acc_addr:
            for a in addrs:
                f.write(to_bin(a, 11) + "\n")

    print(f"nij={len_nij} kij={len_kij} o_nij={len_onij}; psum range "
          f"{int(psum.min())}..{int(psum.max())}, out range {int(out.min())}..{int(out.max())}")
    print(f"wrote activation_tile0.txt, weight_itile0_otile0_kij0..{len_kij - 1}.txt, psum.txt, acc_add.txt, out.txt "
          f"to {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
