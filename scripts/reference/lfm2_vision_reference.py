#!/usr/bin/env python3
"""Capture an independent reference for the LFM2 vision tower and projector.

Runs `transformers`' own `Lfm2VlImageProcessor` and the vision half of
`Lfm2VlForConditionalGeneration` (the SigLIP2 tower, then
`Lfm2VlMultiModalProjector`) in float32 on the CPU, over three generated
pictures chosen to cover the tower's three position-grid cases:

  square-384  384x384, no resize, a 24x24 patch grid (both axes grow)
  strip-96    96x1536, no resize, a 6x96 grid (height SHRINKS, which is
              where the antialiased position resize differs from a plain one)
  photo-480   480x640, resized by the processor to 416x576, a 26x36 grid

None of them is large enough for the processor to tile, so every case is one
run of soft tokens. Writes under test-vectors/vision/references/lfm2/:

  <case>.png             the picture, as fed to the processor
  <case>-pixels.f32      the processor's patch rows, [grid_h * grid_w][768]
                         (the padding rows the processor appends are cut)
  <case>-features.f32    the projected rows, [tokens][2048]
  manifest.json          grids, token counts, library versions, sha256s

It never overwrites: an existing manifest stops the capture, and deleting the
directory deliberately is how it is recaptured.

    lfm2_vision_reference.py --model-dir \\
        ~/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/<rev>
"""
import argparse, hashlib, json, sys
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor
from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "test-vectors" / "vision" / "references" / "lfm2"

CASES = [("square-384", 384, 384), ("strip-96", 96, 1536), ("photo-480", 480, 640)]


def picture(height: int, width: int, seed: int) -> Image.Image:
    """A deterministic picture with gradients, shapes and fine texture, so every
    patch has content and every position matters."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    base = np.stack(
        [
            128 + 100 * np.sin(x / 37.0) * np.cos(y / 23.0),
            128 + 90 * np.cos((x + y) / 51.0),
            255 * x / max(width - 1, 1),
        ],
        axis=-1,
    )
    base += rng.normal(0, 12, size=base.shape)
    img = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
    draw = ImageDraw.Draw(img)
    for _ in range(12):
        x0, y0 = int(rng.integers(0, width)), int(rng.integers(0, height))
        r = int(rng.integers(8, max(9, min(height, width) // 3)))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            draw.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=color)
        else:
            draw.rectangle([x0 - r, y0 - r // 2, x0 + r, y0 + r // 2], fill=color)
    return img


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_f32(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.detach().to(torch.float32).contiguous().numpy().astype("<f4").tobytes())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-dir", required=True, type=Path)
    args = parser.parse_args()

    if (ROOT / "manifest.json").exists():
        print(f"{ROOT / 'manifest.json'} exists; delete the directory to recapture", file=sys.stderr)
        return 1
    ROOT.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(args.model_dir)
    model = Lfm2VlForConditionalGeneration.from_pretrained(args.model_dir, dtype=torch.float32)
    model.eval()

    manifest = {
        "model_dir": str(args.model_dir),
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "float32",
        "device": "cpu",
        "cases": [],
    }
    for seed, (name, height, width) in enumerate(CASES):
        png = ROOT / f"{name}.png"
        picture(height, width, seed).save(png)
        image = Image.open(png).convert("RGB")
        inputs = processor(images=[image], return_tensors="pt")
        grid_h, grid_w = (int(v) for v in inputs["spatial_shapes"][0])
        rows = grid_h * grid_w
        assert int(inputs["pixel_attention_mask"][0].sum()) == rows, "one tile per picture"
        with torch.inference_mode():
            features = model.model.get_image_features(
                pixel_values=inputs["pixel_values"],
                spatial_shapes=inputs["spatial_shapes"],
                pixel_attention_mask=inputs["pixel_attention_mask"],
            ).pooler_output[0]
        tokens = features.shape[0]
        assert tokens * 4 == rows, (tokens, rows)
        write_f32(ROOT / f"{name}-pixels.f32", inputs["pixel_values"][0, :rows])
        write_f32(ROOT / f"{name}-features.f32", features)
        manifest["cases"].append(
            {
                "name": name,
                "height": height,
                "width": width,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "tokens": tokens,
                "png_sha256": sha256(png),
                "features_sha256": sha256(ROOT / f"{name}-features.f32"),
            }
        )
        print(f"{name}: {height}x{width} -> {grid_h}x{grid_w} patches, {tokens} tokens")
    (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
