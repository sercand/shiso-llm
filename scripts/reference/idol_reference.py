#!/usr/bin/env python3
"""Capture independent references for the 3D splat family, stage by stage.

Runs the model's own SDK (its `idol/` package, imported unmodified from a
checkout of the published Space) and the two collateral encoders through
`transformers` and `diffusers`, all in float32 on the CPU, and writes every
stage's inputs and outputs under test-vectors/references/idol/. The Rust
stages are replayed against these by the `idol` device tests.

Float32 is deliberate: the shipped pipeline runs the image tower in bf16 and
the transformer trunks in fp16, and a port carrying f32 activations is graded
against f32 so a difference is the port's, not the reference's rounding.

Every random draw the pipeline makes is recorded (the VAE posterior noise, the
latent and camera noise, the octree's per-cell offsets and the leaf jitter),
so the end-to-end case can replay them instead of reproducing torch's RNG.

It never overwrites a stage's files: delete them to recapture.

    idol_reference.py --sdk <space checkout> --image <picture> [stage ...]

Stages: preprocess, encoders, denoiser, decoders, e2e (default: all).
Checkpoints are read from the Hugging Face cache (`--weights`, `--dinov3`,
`--vae` override the snapshot directories).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "test-vectors" / "references" / "idol"
HUB = Path.home() / ".cache" / "huggingface" / "hub"


def snapshot(repo):
    root = HUB / ("models--" + repo.replace("/", "--")) / "snapshots"
    snaps = sorted(root.iterdir())
    if not snaps:
        sys.exit(f"no snapshot of {repo} under {root}")
    return snaps[-1]


def f32_file(name, tensor):
    path = OUT / name
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")
    np.ascontiguousarray(tensor.detach().float().cpu().numpy()).astype("<f4").tofile(path)


def read_f32(name, shape):
    return torch.from_numpy(np.fromfile(OUT / name, dtype="<f4").reshape(shape).copy())


def write_json(name, value):
    path = OUT / name
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")
    path.write_text(json.dumps(value, indent=2) + "\n")


def pixels_of(png):
    arr = np.array(Image.open(png).convert("RGB"))
    return torch.from_numpy(arr).permute(2, 0, 1).float().div(255.0)[None]


def stage_preprocess(args, sdk):
    from idol.models import BiRefNet
    from idol.preprocess import preprocess_image
    from safetensors.torch import load_file

    net = BiRefNet()
    net.load_state_dict(load_file(str(args.weights / "birefnet.safetensors")), strict=True)
    net = net.eval().float()
    captured = {}
    original = net.remove_background

    def remove_background(image):
        rgba = original(image)
        captured["rgb"] = image
        captured["alpha"] = rgba.getchannel(3)
        return rgba

    net.remove_background = remove_background
    with torch.no_grad():
        out = preprocess_image(Image.open(args.image), background_remover=net, image_size=1024,
                               crop_margin=1.2)
    out.save(OUT / "preprocessed.png")
    if "rgb" in captured:
        captured["rgb"].save(OUT / "matte_input.png")
        captured["alpha"].save(OUT / "matte_alpha.png")


def stage_encoders(args, sdk):
    from diffusers.models import AutoencoderKLFlux2
    from transformers import AutoModel

    if not (OUT / "preprocessed.png").exists():
        # Without the matte stage, any 1024 square picture exercises the
        # encoders; the preprocess stage writes the pipeline's own one.
        if args.image is None:
            sys.exit("the encoders stage needs preprocessed.png or --image")
        Image.open(args.image).convert("RGB").resize((1024, 1024), Image.LANCZOS).save(OUT / "preprocessed.png")
    images = pixels_of(OUT / "preprocessed.png")
    vae = AutoencoderKLFlux2.from_pretrained(str(args.vae), subfolder="vae", torch_dtype=torch.float32).eval()
    with torch.no_grad():
        moments = vae.quant_conv(vae.encoder(images * 2 - 1))
    mean, logvar = moments.chunk(2, dim=1)
    logvar = logvar.clamp(-30.0, 20.0)
    generator = torch.Generator().manual_seed(0)
    noise = torch.randn(mean.shape, generator=generator)
    latents = mean + torch.exp(0.5 * logvar) * noise
    f32_file("vae_moments.f32", moments[0])
    f32_file("vae_noise.f32", noise[0])
    f32_file("vae_sample.f32", latents[0])
    batch, channels, height, width = latents.shape
    packed = latents.view(batch, channels, height // 2, 2, width // 2, 2).permute(0, 1, 3, 5, 2, 4)
    packed = packed.reshape(batch, channels * 4, height // 2, width // 2)
    bn_mean = vae.bn.running_mean.view(1, -1, 1, 1)
    bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps)
    tokens = ((packed - bn_mean) / bn_std).flatten(2).transpose(1, 2)
    tokens = torch.cat([torch.zeros(batch, 5, tokens.shape[2]), tokens], dim=1)
    f32_file("vae_tokens.f32", tokens[0])
    del vae

    dino = AutoModel.from_pretrained(str(args.dinov3), torch_dtype=torch.float32).eval()
    mean = torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
    with torch.no_grad():
        hidden = dino(pixel_values=(images - mean) / std).last_hidden_state
    f32_file("dino_hidden.f32", hidden[0])
    f32_file("dino_tokens.f32", F.layer_norm(hidden, hidden.shape[-1:])[0])


def load_idol(args, sdk):
    from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
    from safetensors.torch import load_file

    config = json.loads((args.weights / "config.json").read_text())

    def load(model, name):
        model.load_state_dict(load_file(str(args.weights / f"{name}.safetensors")), strict=True)
        model = model.eval().requires_grad_(False).float()
        model.dtype = torch.float32
        return model

    return (config,
            load(LatentSeqMMFlowModel(**config["denoiser"]["args"]), "denoiser"),
            load(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), "decoder"),
            load(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), "decoder_gs"))


def cond_fixture():
    from tensordict import TensorDict

    return TensorDict({"feature1": read_f32("dino_tokens.f32", (1, 4101, 1280)),
                       "feature2": read_f32("vae_tokens.f32", (1, 4101, 128))}, batch_size=1)


def stage_denoiser(args, sdk):
    from tensordict import TensorDict

    _, denoiser, _, _ = load_idol(args, sdk)
    cond = cond_fixture()
    generator = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 8192, 16, generator=generator)
    camera = torch.randn(1, 1, 5, generator=generator)
    timestep = 700.0
    taps = {}
    denoiser.noise_refiner[-1].register_forward_hook(lambda m, i, o: taps.__setitem__("noise", o))
    denoiser.context_refiner[-1].register_forward_hook(lambda m, i, o: taps.__setitem__("context", o))
    denoiser.blocks[0].register_forward_hook(lambda m, i, o: taps.__setitem__("block0", o))
    x = TensorDict({"latent": latent, "camera": camera}, batch_size=1)
    with torch.no_grad():
        out = denoiser(x, torch.tensor([timestep]), cond)
        negative = denoiser(x, torch.tensor([timestep]), cond.apply(torch.zeros_like))
    f32_file("den_in_latent.f32", latent[0])
    f32_file("den_in_camera.f32", camera[0, 0])
    f32_file("den_out_latent.f32", out["latent"][0])
    f32_file("den_out_camera.f32", out["camera"][0, 0])
    f32_file("den_neg_latent.f32", negative["latent"][0])
    f32_file("den_neg_camera.f32", negative["camera"][0, 0])
    f32_file("den_tap_noise.f32", taps["noise"][0])
    f32_file("den_tap_context.f32", taps["context"][0])
    f32_file("den_tap_block0.f32", taps["block0"][0])
    write_json("denoiser.json", {"timestep": timestep})


def stage_decoders(args, sdk):
    _, _, octree, gs = load_idol(args, sdk)
    generator = torch.Generator().manual_seed(2)
    latent = torch.randn(1, 8192, 16, generator=generator)
    f32_file("dec_latent.f32", latent[0])
    # Level 3's parents: every cell of the 4x4x4 grid, in the descent's order.
    coords = torch.tensor([[i, j, k] for i in range(4) for j in range(4) for k in range(4)],
                          dtype=torch.float32)
    parents = ((coords + 0.5) / 4)[None]
    with torch.no_grad():
        logits = octree(parents, torch.tensor([8]), latent)
    f32_file("oct_coords.f32", coords)
    f32_file("oct_logits.f32", logits[0])
    anchors = torch.rand(1, 512, 3, generator=generator)
    with torch.no_grad():
        features = gs(anchors, latent)
        gaussian = gs.to_representation(anchors, features)[0]
    f32_file("gs_anchors.f32", anchors[0])
    f32_file("gs_features.f32", features[0])
    f32_file("gs_xyz.f32", gaussian.xyz)
    f32_file("gs_features_dc.f32", gaussian.features_dc.reshape(-1, 3))
    f32_file("gs_scaling.f32", gaussian.scaling)
    f32_file("gs_rotation.f32", gaussian.rotation)
    f32_file("gs_opacity.f32", gaussian.opacity.reshape(-1))
    gaussian.save_ply(str(OUT / "gs.ply"))


def stage_e2e(args, sdk):
    import idol.models.decoder as decoder_module
    from tensordict import TensorDict
    from idol.sampler import sample_flow

    config, denoiser, octree, gs = load_idol(args, sdk)
    cond = cond_fixture()
    generator = torch.Generator().manual_seed(3)
    noise = TensorDict({"latent": torch.randn(1, 8192, 16, generator=generator),
                        "camera": torch.randn(1, 1, 5, generator=generator)}, batch_size=1)
    f32_file("e2e_noise_latent.f32", noise["latent"][0])
    f32_file("e2e_noise_camera.f32", noise["camera"][0, 0])
    with torch.no_grad():
        sample = sample_flow(denoiser, noise, cond=cond, neg_cond=cond.apply(torch.zeros_like),
                             steps=args.steps, rescale_t=3.0, cfg_strength=3.0, verbose=False)
    f32_file("e2e_latent.f32", sample["latent"][0])
    f32_file("e2e_camera.f32", sample["camera"][0, 0])

    # The octree's draws, one offset per cell in the descent's cell order: the
    # reference groups rows by count before drawing, so the order it draws in
    # is not the cell order, and the replay needs the cell order.
    offsets = []
    original_rand = torch.rand

    def systematic_sample(probs, counts):
        batch_shape = counts.shape
        rows_total = counts.numel()
        bins = probs.size(-1)
        probs = probs.view(rows_total, bins).to(torch.float32).clamp_min_(0)
        row_sums = probs.sum(1, keepdim=True)
        zero_mask = row_sums.eq(0)
        probs = probs / row_sums.clamp_min_(1)
        if zero_mask.any():
            probs = probs.clone()
            probs[zero_mask.expand_as(probs)] = 1.0 / bins
        counts = counts.view(rows_total).to(torch.long)
        out = torch.zeros(rows_total, bins, dtype=torch.long)
        cdf = probs.cumsum(dim=1).clamp(max=1.0 - 1e-12)
        raw = torch.zeros(rows_total)
        unique_n, inv = counts.unique(sorted=False, return_inverse=True)
        for index, n in enumerate(unique_n.tolist()):
            if n == 0:
                continue
            rows = (inv == index).nonzero(as_tuple=False).squeeze(1)
            r = rows.numel()
            draw = original_rand(r, 1, generator=generator)
            raw[rows] = draw[:, 0]
            u0 = draw / float(n)
            grid = torch.arange(n, dtype=torch.float32)[None, :] / float(n)
            us = (u0 + grid).clamp(max=1.0 - 1e-12)
            idx = torch.searchsorted(cdf.index_select(0, rows), us).clamp_max(bins - 1)
            buf = torch.zeros(r, bins, dtype=torch.float32)
            buf.scatter_add_(1, idx, torch.ones_like(idx, dtype=buf.dtype))
            out.index_copy_(0, rows, buf.to(torch.long))
        offsets.append(raw)
        return out.view(*batch_shape, bins)

    decoder_module.systematic_sample = systematic_sample
    jitter = []
    original_rand_like = torch.rand_like

    def rand_like(tensor, dtype=None):
        value = torch.rand(tensor.shape, generator=generator, dtype=dtype or tensor.dtype)
        jitter.append(value)
        return value

    torch.rand_like = rand_like
    try:
        with torch.no_grad():
            anchors = decoder_module.OctreeProbabilityFixedlenDecoder.sample(
                octree, sample["latent"], num_points=args.num_gaussians // 32, level=config["max_voxel_level"])
            gaussian = gs.to_representation(anchors, gs(anchors, sample["latent"]))[0]
    finally:
        torch.rand_like = original_rand_like
    for level, raw in enumerate(offsets, start=1):
        f32_file(f"e2e_octree_offsets_{level}.f32", raw)
    f32_file("e2e_jitter.f32", jitter[0][0])
    f32_file("e2e_anchors.f32", anchors[0])
    gaussian.save_ply(str(OUT / "e2e.ply"))
    write_json("e2e.json", {"steps": args.steps, "num_gaussians": args.num_gaussians, "cfg_strength": 3.0,
                            "rescale_t": 3.0, "levels": len(offsets)})


STAGES = {"preprocess": stage_preprocess, "encoders": stage_encoders, "denoiser": stage_denoiser,
          "decoders": stage_decoders, "e2e": stage_e2e}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdk", type=Path, default=None, help="the Space checkout holding idol/ (the idol stages)")
    ap.add_argument("--image", type=Path, help="the picture the preprocess stage reads")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--dinov3", type=Path, default=None)
    ap.add_argument("--vae", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--num-gaussians", type=int, default=32768)
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    args.weights = args.weights or snapshot("speridlabs/idol")
    args.dinov3 = args.dinov3 or snapshot("facebook/dinov3-vith16plus-pretrain-lvd1689m")
    args.vae = args.vae or snapshot("black-forest-labs/FLUX.2-dev")
    if args.sdk is not None:
        sys.path.insert(0, str(args.sdk))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    for stage in args.stages:
        if stage == "preprocess" and args.image is None:
            sys.exit("the preprocess stage needs --image")
        STAGES[stage](args, args.sdk)
        print(f"{stage}: written under {OUT}")


if __name__ == "__main__":
    main()
