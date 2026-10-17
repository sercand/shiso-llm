#!/usr/bin/env python3
"""Capture independent references for FLUX.3 reference-image preparation.

Imports the reference implementation's `pipeline.py` unmodified (with its text
encoder module stubbed out, since preparing a picture needs no language model)
and prints, as JSON, the canvas it computes for a table of sizes, the padded
canvas and fit box of a small seeded picture for each of the two padding modes,
and the size a multi-reference picture is cropped to. The expected values in
`crates/shiso-models/src/flux3/reference_tests.rs` were taken from this output.

    flux3_reference_prep.py --src <reference checkout>/src
"""
import argparse, json, sys, types

import numpy as np
from PIL import Image

SIZES = [(3000, 2000), (40, 4000), (1000, 750), (1024, 1024), (5000, 100), (4096, 4096), (17, 17), (100, 40),
         (1, 1), (2000, 2000), (4000, 1000), (2048, 2048), (2049, 2048)]
PADDED = [(20, 17), (4, 40), (33, 20)]
MULTI = [(2000, 1500), (1000, 1000), (1023, 17), (31, 47)]


def picture(w, h):
    """The deterministic picture the Rust tests rebuild: v(x, y, c) = (7x + 13y + 29c + xy) mod 256."""
    y, x, c = np.mgrid[0:h, 0:w, 0:3]
    return Image.fromarray(((7 * x + 13 * y + 29 * c + x * y) % 256).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    args = ap.parse_args()
    sys.path.insert(0, args.src)
    stub = types.ModuleType("flux3.text_encoder")
    stub.TextEncoder = object
    sys.modules["flux3.text_encoder"] = stub
    from flux3 import pipeline as P

    out = {"canvas": {f"{w}x{h}": P.reference_canvas(w, h) for w, h in SIZES}, "padded": {}, "multi": {}}
    for w, h in PADDED:
        prepared, box = P.prepare_reference(picture(w, h))
        flat = np.asarray(prepared).reshape(-1).astype(np.int64)
        out["padded"][f"{w}x{h}"] = {
            "size": prepared.size, "box": box,
            # position-weighted, so a mirrored or shifted pad changes it
            "weighted_sum": int(((np.arange(flat.size, dtype=np.int64) + 1) * flat).sum()),
        }
    for w, h in MULTI:
        out["multi"][f"{w}x{h}"] = list(P.prepare_multi_reference(picture(w, h)).size)
    json.dump(out, sys.stdout)


if __name__ == "__main__":
    main()
