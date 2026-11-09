#!/usr/bin/env python3
"""Capture independent references for the 3D splat family, stage by stage.

Runs the two collateral encoders through `transformers` and `diffusers`, and
the model's three networks through `idol_models.py` (a PyTorch transcription
of the published SDK, loaded strictly against the checkpoint), all in float32
on the CPU, and writes every stage's inputs and outputs under
test-vectors/references/idol/. The Rust stages are replayed against these by
`crates/shiso-models/src/idol/device_tests.rs`.

The `crosscheck` stage grades the transcription against the SDK's own modules
(`--sdk <checkout>`, the Space's `idol/` package imported unmodified); the
`preprocess` stage runs the SDK's matte and crop (also `--sdk`).

Float32 is deliberate: the shipped pipeline runs the image tower in bf16 and
the transformer trunks in fp16, and a port carrying f32 activations is graded
against f32 so a difference is the port's, not the reference's rounding.

Every random draw a generation makes is recorded (the VAE posterior noise, the
latent and camera noise, the octree's per-cell offsets and the leaf jitter),
so the end-to-end case can replay them instead of reproducing torch's RNG.

It never overwrites a stage's files: delete them to recapture.

    idol_reference.py --image <picture> [stage ...]
    idol_reference.py --sdk <space checkout> crosscheck

Stages: preprocess, encoders, denoiser, decoders, e2e, matte, crosscheck
(default: encoders denoiser decoders e2e matte).
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


def stage_preprocess(args):
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


def stage_encoders(args):
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


def load_idol(args):
    """The three networks, from the transcription in idol_models.py, loaded strictly."""
    from safetensors.torch import load_file
    import idol_models as m

    config = json.loads((args.weights / "config.json").read_text())

    def load(model, name):
        model.load_state_dict(load_file(str(args.weights / f"{name}.safetensors")), strict=True)
        return model.eval().requires_grad_(False).float()

    return (config,
            load(m.Denoiser(**config["denoiser"]["args"]), "denoiser"),
            load(m.OctreeDecoder(**config["decoder"]["args"]), "decoder"),
            load(m.GaussianDecoder(**config["decoder_gs"]["args"]), "decoder_gs"))


def cond_fixture():
    return (read_f32("dino_tokens.f32", (1, 4101, 1280)), read_f32("vae_tokens.f32", (1, 4101, 128)))


def stage_denoiser(args):
    _, denoiser, _, _ = load_idol(args)
    feature1, feature2 = cond_fixture()
    generator = torch.Generator().manual_seed(1)
    latent = torch.randn(1, 8192, 16, generator=generator)
    camera = torch.randn(1, 1, 5, generator=generator)
    timestep = 700.0
    taps = {}
    out_latent, out_camera = denoiser(latent, camera, torch.tensor([timestep]), feature1, feature2, taps)
    neg_latent, neg_camera = denoiser(latent, camera, torch.tensor([timestep]), torch.zeros_like(feature1),
                                      torch.zeros_like(feature2))
    f32_file("den_in_latent.f32", latent[0])
    f32_file("den_in_camera.f32", camera[0, 0])
    f32_file("den_out_latent.f32", out_latent[0])
    f32_file("den_out_camera.f32", out_camera[0, 0])
    f32_file("den_neg_latent.f32", neg_latent[0])
    f32_file("den_neg_camera.f32", neg_camera[0, 0])
    f32_file("den_tap_noise.f32", taps["noise"][0])
    f32_file("den_tap_context.f32", taps["context"][0])
    f32_file("den_tap_block0.f32", taps["block0"][0])
    write_json("denoiser.json", {"timestep": timestep})


def stage_decoders(args):
    import idol_models as m

    _, _, octree, gs = load_idol(args)
    generator = torch.Generator().manual_seed(2)
    latent = torch.randn(1, 8192, 16, generator=generator)
    f32_file("dec_latent.f32", latent[0])
    # Level 3's parents: every cell of the 4x4x4 grid, x slowest.
    coords = torch.tensor([[i, j, k] for i in range(4) for j in range(4) for k in range(4)], dtype=torch.float32)
    logits = octree(((coords + 0.5) / 4)[None], torch.tensor([8]), latent)
    f32_file("oct_coords.f32", coords)
    f32_file("oct_logits.f32", logits[0])
    anchors = torch.rand(512, 3, generator=generator)
    features = gs(anchors[None], latent)[0]
    xyz, colour, scaling, rotation, opacity = gs.activate(anchors, features)
    f32_file("gs_anchors.f32", anchors)
    f32_file("gs_features.f32", features)
    f32_file("gs_xyz.f32", xyz)
    f32_file("gs_features_dc.f32", colour)
    f32_file("gs_scaling.f32", scaling)
    f32_file("gs_rotation.f32", rotation)
    f32_file("gs_opacity.f32", opacity)
    path = OUT / "gs.ply"
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")
    path.write_bytes(m.ply_bytes(xyz, colour, scaling, rotation, opacity))


def stage_e2e(args):
    """A whole generation from the fixture conditioning, with every draw recorded."""
    import idol_models as m

    config, denoiser, octree, gs = load_idol(args)
    feature1, feature2 = cond_fixture()
    generator = torch.Generator().manual_seed(3)
    latent = torch.randn(1, 8192, 16, generator=generator)
    camera = torch.randn(1, 1, 5, generator=generator)
    f32_file("e2e_noise_latent.f32", latent[0])
    f32_file("e2e_noise_camera.f32", camera[0, 0])
    zeros1, zeros2 = torch.zeros_like(feature1), torch.zeros_like(feature2)
    steps, cfg, shift = args.steps, 3.0, 3.0
    grid = np.linspace(1, 0, steps + 1)
    grid = shift * grid / (1 + (shift - 1) * grid)
    for i in range(steps):
        t, t_prev = grid[i], grid[i + 1]
        tt = torch.tensor([1000 * t], dtype=torch.float32)
        v_latent, v_camera = denoiser(latent, camera, tt, feature1, feature2)
        if cfg > 0:
            n_latent, n_camera = denoiser(latent, camera, tt, zeros1, zeros2)
            v_latent = (1 + cfg) * v_latent - cfg * n_latent
            v_camera = (1 + cfg) * v_camera - cfg * n_camera
        latent = latent - v_latent * (t - t_prev)
        camera = camera - v_camera * (t - t_prev)
    f32_file("e2e_latent.f32", latent[0])
    f32_file("e2e_camera.f32", camera[0, 0])

    def rand(n):
        return torch.rand(n, generator=generator)

    anchors, level_draws, jitter = m.place_anchors(octree, latent, args.num_gaussians // 32,
                                                   config["max_voxel_level"], rand)
    for level, draws in enumerate(level_draws, start=1):
        f32_file(f"e2e_octree_offsets_{level}.f32", draws)
    f32_file("e2e_jitter.f32", jitter)
    f32_file("e2e_anchors.f32", anchors)
    features = gs(anchors[None], latent)[0]
    cloud = gs.activate(anchors, features)
    path = OUT / "e2e.ply"
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")
    path.write_bytes(m.ply_bytes(*cloud))
    write_json("e2e.json", {"steps": steps, "num_gaussians": args.num_gaussians, "cfg_strength": cfg,
                            "rescale_t": shift, "levels": len(level_draws), "seed_note": "torch generator 3"})


def stage_crosscheck(args):
    """The transcription against the SDK's own modules, same weights, random inputs (needs --sdk)."""
    if args.sdk is None:
        sys.exit("the crosscheck stage needs --sdk")
    from tensordict import TensorDict
    from idol.models import GaussianFixedlenDecoder, LatentSeqMMFlowModel, OctreeProbabilityFixedlenDecoder
    from safetensors.torch import load_file

    config, denoiser, octree, gs = load_idol(args)

    def sdk(model, name):
        model.load_state_dict(load_file(str(args.weights / f"{name}.safetensors")), strict=True)
        model = model.eval().float()
        model.dtype = torch.float32
        return model

    s_den = sdk(LatentSeqMMFlowModel(**config["denoiser"]["args"]), "denoiser")
    s_oct = sdk(OctreeProbabilityFixedlenDecoder(**config["decoder"]["args"]), "decoder")
    s_gs = sdk(GaussianFixedlenDecoder(**config["decoder_gs"]["args"]), "decoder_gs")
    g = torch.Generator().manual_seed(4)
    latent, camera = torch.randn(1, 8192, 16, generator=g), torch.randn(1, 1, 5, generator=g)
    f1, f2 = torch.randn(1, 4101, 1280, generator=g), torch.randn(1, 4101, 128, generator=g)
    mine = denoiser(latent, camera, torch.tensor([400.0]), f1, f2)
    theirs = s_den(TensorDict({"latent": latent, "camera": camera}, batch_size=1), torch.tensor([400.0]),
                   TensorDict({"feature1": f1, "feature2": f2}, batch_size=1))
    report = {"denoiser_latent": (mine[0] - theirs["latent"]).abs().max().item(),
              "denoiser_camera": (mine[1] - theirs["camera"]).abs().max().item()}
    x = torch.rand(1, 64, 3, generator=g)
    report["octree"] = (octree(x, torch.tensor([16]), latent) - s_oct(x, torch.tensor([16]), latent)).abs().max().item()
    pts = torch.rand(1, 256, 3, generator=g)
    f_mine, f_theirs = gs(pts, latent), s_gs(pts, latent)
    report["gaussian_features"] = (f_mine - f_theirs).abs().max().item()
    ref = s_gs.to_representation(pts, f_theirs)[0]
    act = gs.activate(pts[0], f_mine[0])
    for name, a, b in zip(["xyz", "colour", "scaling", "rotation", "opacity"], act,
                          [ref.xyz, ref.features_dc.reshape(-1, 3), ref.scaling, ref.rotation, ref.opacity.reshape(-1)]):
        report[f"gaussian_{name}"] = (a - b).abs().max().item()
    print(json.dumps(report, indent=2))


def stage_matte(args):
    """The background remover on a picture with a background, at its own size (it resizes to 1024 and back)."""
    import idol_models as m
    from safetensors.torch import load_file

    if args.image is None:
        sys.exit("the matte stage needs --image")
    net = m.BiRefNet()
    net.load_state_dict(load_file(str(args.weights / "birefnet.safetensors")), strict=True)
    net = net.eval().float()
    rgb = np.array(Image.open(args.image).convert("RGB"))
    Image.fromarray(rgb).save(OUT / "matte_input.png")
    taps = {}
    alpha = net.alpha(rgb, taps)
    path = OUT / "matte_alpha.u8"
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")
    path.write_bytes(alpha.tobytes())
    f32_file("matte_net_input.f32", taps["input"][0])
    f32_file("matte_x4.f32", taps["x4"][0])
    f32_file("matte_squeezed.f32", taps["squeezed"][0])
    f32_file("matte_p4.f32", taps["p4"][0])
    f32_file("matte_mask.f32", taps["mask"][0, 0])
    write_json("matte.json", {"height": int(rgb.shape[0]), "width": int(rgb.shape[1])})


STAGES = {"preprocess": stage_preprocess, "encoders": stage_encoders, "denoiser": stage_denoiser,
          "decoders": stage_decoders, "e2e": stage_e2e, "matte": stage_matte,
          "crosscheck": stage_crosscheck}
DEFAULT_STAGES = ["encoders", "denoiser", "decoders", "e2e", "matte"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sdk", type=Path, default=None, help="the Space checkout holding idol/ (the idol stages)")
    ap.add_argument("--image", type=Path, help="the picture the preprocess stage reads")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--dinov3", type=Path, default=None)
    ap.add_argument("--vae", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--num-gaussians", type=int, default=32768)
    ap.add_argument("stages", nargs="*", default=DEFAULT_STAGES)
    args = ap.parse_args()
    args.weights = args.weights or snapshot("speridlabs/idol")
    args.dinov3 = args.dinov3 or snapshot("facebook/dinov3-vith16plus-pretrain-lvd1689m")
    args.vae = args.vae or snapshot("black-forest-labs/FLUX.2-dev")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    if args.sdk is not None:
        sys.path.insert(0, str(args.sdk))
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(False)
    for stage in args.stages:
        if stage == "preprocess" and args.image is None:
            sys.exit("the preprocess stage needs --image")
        STAGES[stage](args)
        print(f"{stage}: written under {OUT}")


if __name__ == "__main__":
    main()
