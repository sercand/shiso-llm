#!/usr/bin/env python3
"""Check that make_hybrid_mtp.py paired each trellis tensor with the right BF16 one.

Reconstructs trellis tensors of the draft GGUF with the reference EXL3 kernels
(mul1 codebook, the only one shiso's kernel set carries) and compares each with
the BF16 tensor the hybrid GGUF put in its place. A right pairing at 4-6 bits
gives a cosine near 1; a transposed, permuted or mis-split one gives near 0.
Needs a GPU and the reference venv (`~/work/exllamav3/.venv/bin/python`).

  python check_hybrid_mtp.py <draft.gguf> <hybrid.gguf> [tensor ...]
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_hybrid_mtp import EXL3, G_BF16, read_gguf  # noqa: E402

REF = os.path.expanduser("~/work/exllamav3")
sys.path.insert(0, REF)
from exllamav3.modules.quant.exl3 import LinearEXL3  # noqa: E402

K_BITS = {68: 4, 69: 5, 70: 6}


def payload(path, g, name):
    t = next(t for t in g["tensors"] if t["name"] == name)
    with open(path, "rb") as f:
        f.seek(g["data_start"] + t["offset"])
        return t, f.read(t["span"])


def decode(path, g, name):
    t, raw = payload(path, g, name)
    k, n = t["dims"]
    K = K_BITS[t["type"]]
    tiles = k * n * K // 8
    tr = np.frombuffer(raw[:tiles], dtype=np.int16).reshape(k // 16, n // 16, 16 * K)
    suh = np.frombuffer(raw[tiles:tiles + 2 * k], dtype=np.float16)
    svh = np.frombuffer(raw[tiles + 2 * k:tiles + 2 * k + 2 * n], dtype=np.float16)
    dev = torch.device("cuda")
    lin = LinearEXL3(None, k, n, suh=torch.from_numpy(suh.copy()).to(dev),
                     svh=torch.from_numpy(svh.copy()).to(dev),
                     trellis=torch.from_numpy(tr.copy()).to(dev),
                     mul1=torch.zeros(1, device=dev))
    return lin.get_weight_tensor().float().cpu().numpy()  # [in, out]


def bf16(path, g, name):
    t, raw = payload(path, g, name)
    assert t["type"] == G_BF16, name
    k, n = t["dims"]
    words = np.frombuffer(raw[:k * n * 2], dtype=np.uint16).astype(np.uint32) << 16
    return words.view(np.float32).reshape(n, k).T  # [in, out]


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


def main():
    draft, hybrid = sys.argv[1], sys.argv[2]
    gd, gh = read_gguf(draft), read_gguf(hybrid)
    names = sys.argv[3:] or [t["name"] for t in gh["tensors"]
                             if t["type"] == G_BF16 and
                             next(d for d in gd["tensors"] if d["name"] == t["name"])["type"] in EXL3]
    worst = 1.0
    for name in names:
        q, w = decode(draft, gd, name), bf16(hybrid, gh, name)
        c = cos(q, w)
        worst = min(worst, c)
        rel = float(np.linalg.norm(q - w) / np.linalg.norm(w))
        print(f"{name:34s} cos={c:.5f} rel_err={rel:.4f} "
              f"(transposed-shape cos={cos(q.T.reshape(q.shape), w):+.4f})")
    print(f"worst cosine {worst:.5f}")
    sys.exit(0 if worst > 0.99 else 1)


if __name__ == "__main__":
    main()
