#!/usr/bin/env python3
"""Capture independent references for the FLUX.3 diffusion transformer.

Builds the reference implementation's own `DiT` module (its `dit.py`, imported
unmodified) at a tiny size - hidden 256, two heads, text conditioning at its
true 20480 width - with seeded random weights, runs it in float32 on the CPU,
and writes the weights and every input and output under
test-vectors/references/flux3/dit-tiny/. The Rust transformer is replayed
against these by `crates/shiso-models/src/flux3/transformer_tests.rs`.

Two reference shapes share one weight file: a full-depth model (two per-mode
blocks, three single-stream blocks) and a one-block model made from the first
block of each stack of the same weights. Each runs with a reference image
(`image_cond`) and without one (the reference then pushes a phantom token
through the empty mode; the Rust side skips the mode).

The reference carries its time tensors in bfloat16, so its sinusoidal embedding
rounds `t`, `1000 * t` and the `[cos | sin]` result to bf16. In float32 that
function would not do so, so it is replaced here by a float32 twin that applies
exactly those three roundings, and `time_embedding.json` records the reference
function's own bfloat16 output for several times, which the script checks the
twin against before using it.

It never overwrites: delete the directory to recapture.

    flux3_dit_reference.py --src <reference checkout>/src
"""
import argparse, json, math, sys
from pathlib import Path

import torch
from safetensors.torch import save_file

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "test-vectors" / "references" / "flux3" / "dit-tiny"
HIDDEN, HEADS, TEXT = 256, 2, 20480
FULL = (2, 3)  # per-mode depth, single-stream depth
SMALL = (1, 1)
TIMES = (0.0, 0.0123, 0.6209, 0.8731, 1.0)
T, T_REF = 0.8731, 0.6209
IMAGE_GRID, COND_GRID, TEXT_ROWS = (3, 4), (2, 3), 7


def f32_file(path, tensor):
    tensor.detach().float().contiguous().numpy().astype("<f4").tofile(path)


def bf16_twin(t, dim=256, max_period=10000):
    """float32 `timestep_embedding` with the reference's bfloat16 roundings."""
    t = 1000.0 * t.to(torch.bfloat16)
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32) / half)
    args = t[:, None].float() * freqs[None]
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1).to(torch.bfloat16).float()


def grid_ids(h, w, l=0.0):
    rows, cols = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    ids = torch.stack([torch.zeros_like(rows), rows, cols, torch.zeros_like(rows)], dim=-1).reshape(h * w, 4).float()
    ids[:, 1:3] *= 0.5
    ids[:, 3] = l
    return ids


def build(D, depth, single, state=None):
    D.HIDDEN, D.HEADS, D.DEPTH, D.DEPTH_SINGLE = HIDDEN, HEADS, depth, single
    model = D.DiT(flow_map=True).eval()
    if state is not None:
        model.load_state_dict(state, strict=False)
    return model


def packed(D, image, image_ids, cond, cond_ids, text, text_ids, t):
    n = {"image": image.shape[0], "image_cond": 0 if cond is None else cond.shape[0], "text": text.shape[0]}
    data = {"image": image, "image_cond": cond if cond is not None else torch.zeros(0, 128), "text": text}
    ids = {"image": image_ids, "image_cond": cond_ids if cond is not None else torch.zeros(0, 4), "text": text_ids}
    steps = {m: torch.zeros(n[m]) for m in n}
    steps["image"].fill_(t)
    return D.Packed(data=data, ids=ids, timesteps=steps, mode_indices={m: [(0, n[m])] for m in n})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the reference checkout's src directory")
    args = ap.parse_args()
    if OUT.exists():
        sys.exit(f"{OUT} exists; delete it to recapture")
    sys.path.insert(0, args.src)
    from flux3 import dit as D

    # The reference's own embedding in its own dtype, for several times.
    reference_embeddings = {
        str(t): D.timestep_embedding(torch.tensor([t]).bfloat16())[0].float().tolist() for t in TIMES
    }
    D.timestep_embedding = bf16_twin
    for t in TIMES:
        twin = bf16_twin(torch.tensor([t]))[0].tolist()
        assert twin == reference_embeddings[str(t)], f"the float32 twin disagrees with the reference at t={t}"

    torch.manual_seed(20261017)
    full = build(D, *FULL)
    with torch.no_grad():
        for name, p in full.named_parameters():
            if name.endswith(".scale"):
                p.copy_(1.0 + 0.1 * torch.randn_like(p))
    small = build(D, *SMALL, state=full.state_dict())

    image = torch.randn(IMAGE_GRID[0] * IMAGE_GRID[1], 128)
    cond = torch.randn(COND_GRID[0] * COND_GRID[1], 128)
    text = torch.randn(TEXT_ROWS, TEXT)
    image_ids, cond_ids = grid_ids(*IMAGE_GRID), grid_ids(*COND_GRID, l=-1.0)
    text_ids = torch.tensor([[0.0, 0.0, 0.0, float(i)] for i in range(TEXT_ROWS)])

    OUT.mkdir(parents=True)
    save_file({k: v.contiguous() for k, v in full.state_dict().items()}, str(OUT / "weights.safetensors"))
    for name, tensor in (("image", image), ("cond", cond), ("text", text), ("image_ids", image_ids),
                         ("cond_ids", cond_ids), ("text_ids", text_ids)):
        f32_file(OUT / f"{name}.f32", tensor)
    meta = {"hidden": HIDDEN, "heads": HEADS, "text": TEXT, "full": list(FULL), "small": list(SMALL),
            "t": T, "t_ref": T_REF, "image_grid": IMAGE_GRID, "cond_grid": COND_GRID, "text_rows": TEXT_ROWS,
            "image_rows": image.shape[0], "cond_rows": cond.shape[0], "outputs": {}}
    for tag, model, depths in (("full", full, FULL), ("small", small, SMALL)):
        # `DiT.forward` reads the per-mode depth from the module global at call
        # time, not from the model it runs, so the global is set per model.
        D.HIDDEN, D.HEADS, D.DEPTH, D.DEPTH_SINGLE = HIDDEN, HEADS, *depths
        for with_cond in (True, False):
            pack = packed(D, image, image_ids, cond if with_cond else None, cond_ids, text, text_ids, T)
            with torch.inference_mode():
                out = model(pack, timesteps_ref=torch.tensor([T_REF]))
            name = f"out_{tag}_{'cond' if with_cond else 'nocond'}"
            f32_file(OUT / f"{name}.f32", out)
            meta["outputs"][name] = list(out.shape)
            assert torch.isfinite(out).all() and out.abs().max() > 1e-3, name
    (OUT / "time_embedding.json").write_text(json.dumps(reference_embeddings))
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
