#!/usr/bin/env python3
"""Capture independent references for the FLUX.3 autoencoder.

Builds the reference implementation's own `AutoEncoder` (its `autoencoder.py`,
imported unmodified) at a tiny width - 32 base channels instead of 128, the
same four levels, two residual blocks per level, mid attention, group norm over
32 groups - with seeded random weights, runs `encode` and `decode` in float32
on the CPU, and writes the weights, an input picture and both outputs under
test-vectors/references/flux3/vae-tiny/. The Rust autoencoder is replayed
against these by `crates/shiso-models/src/flux3/vae_tests.rs`; the weight file
also carries the checkpoint's own tensor names, which the Rust loader's names
are checked against.

It never overwrites: delete the directory to recapture.

    flux3_vae_reference.py --src <reference checkout>/src
"""
import argparse, json, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "test-vectors" / "references" / "flux3" / "vae-tiny"
CH = 32
HEIGHT, WIDTH = 64, 48  # pixels; 4 x 3 latent tokens


def f32_file(path, tensor):
    tensor.detach().float().contiguous().numpy().astype("<f4").tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the reference checkout's src directory")
    args = ap.parse_args()
    if OUT.exists():
        sys.exit(f"{OUT} exists; delete it to recapture")
    sys.path.insert(0, args.src)
    from flux3 import autoencoder as A

    A.CH = CH  # the module builds its stages from this global at construction time
    torch.manual_seed(20261018)
    ae = A.AutoEncoder().eval()
    with torch.no_grad():
        for name, p in ae.named_parameters():
            if ".norm" in name and name.endswith(".weight"):  # group-norm gains and offsets
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
            elif ".norm" in name and name.endswith(".bias"):
                p.copy_(0.1 * torch.randn_like(p))
        ae.bn.running_mean.copy_(0.3 * torch.randn_like(ae.bn.running_mean))
        ae.bn.running_var.copy_(0.5 + torch.rand_like(ae.bn.running_var))

    image = (torch.rand(3, HEIGHT, WIDTH) * 2 - 1)
    tokens, ids = ae.encode(image)

    # decode takes arbitrary latent tokens; capture the exact ones it is given
    torch.manual_seed(7)
    latent = torch.randn_like(tokens)
    decoded = ae.decode(latent, ids)

    OUT.mkdir(parents=True)
    state = {k: v.contiguous().float() for k, v in ae.state_dict().items() if v.is_floating_point()}
    save_file(state, str(OUT / "weights.safetensors"))
    f32_file(OUT / "image.f32", image)
    f32_file(OUT / "tokens.f32", tokens)
    f32_file(OUT / "latent.f32", latent)
    f32_file(OUT / "decoded.f32", decoded)
    (OUT / "meta.json").write_text(json.dumps({
        "ch": CH, "height": HEIGHT, "width": WIDTH, "tokens": list(tokens.shape),
        "state_keys": sorted(ae.state_dict().keys()),
    }, indent=1))
    assert torch.isfinite(tokens).all() and torch.isfinite(decoded).all()
    print("wrote", OUT, "tokens", tuple(tokens.shape), "decoded", tuple(decoded.shape),
          "decoded range", float(decoded.min()), float(decoded.max()))


if __name__ == "__main__":
    main()
