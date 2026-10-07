#!/usr/bin/env python3
"""Capture independent references for the multimodal embedding checkpoint.

Runs `transformers`' own `EmbeddingGemma2Model` and processor, wrapped by
`sentence-transformers` exactly as the checkpoint's `modules.json` declares
(Transformer -> mean Pooling -> Normalize), in float32 on the CPU, over a small
fixed set of text, image, audio and video inputs. Writes one
`<case>.json` per case under test-vectors/references/embedding-gemma2/, with
large float arrays in little-endian `.f32` sidecars whose shapes are in the
JSON.

Every embedding is produced twice: once by `SentenceTransformer.encode` (what a
user of the checkpoint gets) and once by calling the model directly on the same
processor output and mean-pooling `last_hidden_state` here. The two must agree
to float32 rounding or the capture stops, so the recorded pooling is the
library's and not a guess about it.

The media inputs (PNG, WAV, MP4) are generated or cut deterministically and
saved next to the fixtures, so a replay feeds identical bytes. It never
overwrites a fixture: a case whose JSON is present is kept, and deleting it
deliberately is how it is recaptured.

    embedding_gemma2_reference.py --model-dir \\
        ~/.cache/huggingface/hub/models--google--embeddinggemma-2/snapshots/914f7f89142e33e77833254d9c9b90c3cef7303b
"""
import argparse, hashlib, json, sys
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import transformers

REPO = Path(__file__).resolve().parents[2]
ROOT = REPO / "test-vectors" / "references" / "embedding-gemma2"
CLIPS = REPO / "test-vectors" / "references" / "audio" / "clips"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
MRL_DIMS = (128, 256, 512)

CODE = """fn merge_sorted(a: &[i32], b: &[i32]) -> Vec<i32> {
    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if a[i] <= b[j] {
            out.push(a[i]);
            i += 1;
        } else {
            out.push(b[j]);
            j += 1;
        }
    }
    out.extend_from_slice(&a[i..]);
    out.extend_from_slice(&b[j..]);
    out
}"""

# The long case is a whole-line prefix of a committed-corpus prompt, cut at the
# first line boundary past LONG_TARGET tokens, so the Rust side can rebuild the
# exact string from the corpus file and the recorded line count.
LONG_SOURCE = "long_memory_archive.txt"
LONG_TARGET = 1500


def refuse_existing(path: Path):
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_f32(name: str, array) -> dict:
    path = ROOT / name
    refuse_existing(path)
    array = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    path.write_bytes(array.tobytes())
    return {"file": name, "shape": list(array.shape), "dtype": "f32le"}


# ---------------------------------------------------------------- media inputs


def make_image(path: Path):
    """640x480: a two-axis gradient with a few hard-edged shapes on it."""
    from PIL import Image, ImageDraw
    if path.exists():
        return
    w, h = 640, 480
    x = np.linspace(0, 255, w, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, h, dtype=np.float32)[:, None]
    rgb = np.stack([np.broadcast_to(x, (h, w)), np.broadcast_to(y, (h, w)),
                    np.broadcast_to(255 - (x + y) / 2, (h, w))], axis=-1)
    img = Image.fromarray(np.round(rgb).astype(np.uint8), "RGB")
    d = ImageDraw.Draw(img)
    d.rectangle([60, 60, 260, 220], fill=(220, 20, 20))
    d.ellipse([360, 80, 580, 300], fill=(20, 200, 40), outline=(0, 0, 0), width=6)
    d.polygon([(120, 420), (300, 260), (420, 440)], fill=(250, 230, 30))
    for i in range(0, w, 40):
        d.line([(i, 0), (i, 30)], fill=(0, 0, 0), width=3)
    img.save(path, format="PNG", optimize=False)


def cut_wav(path: Path, source: Path, seconds: float | None):
    import soundfile
    if path.exists():
        return
    data, rate = soundfile.read(source, dtype="int16")
    assert rate == 16000 and data.ndim == 1, (rate, data.shape)
    if seconds is not None:
        data = data[: int(seconds * rate)]
    soundfile.write(path, data, rate, subtype="PCM_16")


VIDEO_FPS, VIDEO_SECONDS, VIDEO_W, VIDEO_H = 8, 4, 320, 240


def make_video(path: Path):
    """4 s at 8 fps of a red square crossing a static gradient, H.264 yuv420p."""
    import av
    if path.exists():
        return
    x = np.linspace(0, 255, VIDEO_W, dtype=np.float32)[None, :]
    y = np.linspace(0, 255, VIDEO_H, dtype=np.float32)[:, None]
    bg = np.stack([np.broadcast_to(x * 0.5, (VIDEO_H, VIDEO_W)), np.broadcast_to(y * 0.5, (VIDEO_H, VIDEO_W)),
                   np.full((VIDEO_H, VIDEO_W), 160.0)], axis=-1)
    bg = np.round(bg).astype(np.uint8)
    container = av.open(str(path), mode="w", format="mp4")
    stream = container.add_stream("libx264", rate=Fraction(VIDEO_FPS, 1))
    stream.width, stream.height, stream.pix_fmt = VIDEO_W, VIDEO_H, "yuv420p"
    # One thread and a fixed preset keep the encoder's output reproducible.
    stream.options = {"crf": "18", "preset": "medium", "threads": "1", "bframes": "0"}
    n = VIDEO_FPS * VIDEO_SECONDS
    for i in range(n):
        frame = bg.copy()
        side = 64
        x0 = int(round(i * (VIDEO_W - side) / (n - 1)))
        y0 = (VIDEO_H - side) // 2 + int(round(40 * np.sin(2 * np.pi * i / n)))
        frame[y0:y0 + side, x0:x0 + side] = (230, 20, 20)
        for packet in stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def decode_video(path: Path):
    """Every frame as RGB uint8 [T, H, W, 3], with the metadata the sampler needs.

    Decoded here with PyAV rather than by the processor's loader, which in this
    `transformers` needs `torchcodec`. The decoded frames the sampler keeps are
    saved as a sidecar, so a replay can check its own decoder separately from
    the model.
    """
    import av
    from transformers.video_utils import VideoMetadata
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        frames = np.stack([f.to_ndarray(format="rgb24") for f in container.decode(stream)])
        fps = float(stream.average_rate)
    meta = VideoMetadata(total_num_frames=int(frames.shape[0]), fps=fps, width=int(frames.shape[2]),
                         height=int(frames.shape[1]), duration=frames.shape[0] / fps, video_backend="pyav")
    return frames, meta


# ---------------------------------------------------------------- capture


class Recorder:
    """Wraps the processor's chat-template call and the video sampler so the
    case JSON can say exactly what text was rendered and which frames were kept.
    """

    def __init__(self, processor):
        self.processor = processor
        self.rendered = []
        self.frames = []
        apply = processor.apply_chat_template

        def apply_chat_template(conversation, *args, **kwargs):
            plain = {k: v for k, v in kwargs.items() if k in ("chat_template", "add_generation_prompt")}
            self.rendered.append(apply(conversation, tokenize=False, **plain))
            return apply(conversation, *args, **kwargs)

        processor.apply_chat_template = apply_chat_template
        vp = processor.video_processor
        sample = vp.sample_frames

        def sample_frames(metadata, *args, **kwargs):
            idx = sample(metadata, *args, **kwargs)
            self.frames.append({"indices": [int(i) for i in idx], "source_fps": metadata.fps,
                                "source_frames": metadata.total_num_frames, "duration_s": metadata.duration,
                                "width": metadata.width, "height": metadata.height})
            return idx

        vp.sample_frames = sample_frames

    def reset(self):
        self.rendered.clear()
        self.frames.clear()


def to_device(features):
    return {k: v for k, v in features.items() if k != "modality"}


def capture(model, rec, case: dict):
    name = case["name"]
    meta_path = ROOT / f"{name}.json"
    if meta_path.exists():
        print(f"kept {meta_path.name}: delete it to recapture")
        return None
    st_input, prompt_name = case["input"], case.get("prompt_name")
    prompt = model.prompts[prompt_name] if prompt_name else None
    tok = model[0].processor.tokenizer
    cfg = model[0].model.config

    # 1. What a user gets.
    rec.reset()
    st_emb = model.encode([st_input], prompt_name=prompt_name, convert_to_numpy=True)[0]
    rendered = list(rec.rendered)
    frames = list(rec.frames)
    st_trunc = {d: model.encode([st_input], prompt_name=prompt_name, truncate_dim=d, normalize_embeddings=True,
                                convert_to_numpy=True)[0] for d in MRL_DIMS}

    # 2. The same processor output through the model directly, pooled here.
    rec.reset()
    features = model.preprocess([st_input], prompt=prompt)
    assert rec.rendered == rendered, (rec.rendered, rendered)
    inputs = to_device(features)
    hf = model[0].model
    with torch.no_grad():
        out = hf(**inputs, return_dict=True)
    hidden = out.last_hidden_state[0]  # [tokens, 768]
    mask = inputs["attention_mask"][0].to(hidden.dtype)
    assert bool(mask.all()), "a batch of one carries no padding"
    pooled = (hidden * mask[:, None]).sum(0) / mask.sum()
    normed = pooled / pooled.norm()
    pooled_np, normed_np = pooled.numpy(), normed.numpy()
    diff = float(np.abs(normed_np - st_emb).max())
    if diff > 1e-5:
        sys.exit(f"{name}: SentenceTransformer.encode and the direct forward differ by {diff}")

    ids = inputs["input_ids"][0].tolist()
    counts = {
        "image_soft_tokens": ids.count(cfg.image_token_id),
        "video_soft_tokens": ids.count(cfg.video_token_id),
        "audio_soft_tokens": ids.count(cfg.audio_token_id),
        "boi": ids.count(cfg.boi_token_id), "eoi": ids.count(cfg.eoi_token_id),
        "boa": ids.count(cfg.boa_token_id), "eoa": ids.count(cfg.eoa_token_index),
    }
    counts["text_tokens"] = len(ids) - counts["image_soft_tokens"] - counts["video_soft_tokens"] - \
        counts["audio_soft_tokens"]

    record = {
        "case": name,
        "description": case["description"],
        "st_input": case["st_input_desc"],
        "prompt_name": prompt_name,
        "prompt": prompt,
        "rendered_text": rendered[0] if len(rendered) == 1 else rendered,
        "input_ids": ids,
        "tokens": tokenize_view(tok, ids),
        "n_tokens": len(ids),
        "counts": counts,
        "embedding": normed_np.tolist(),
        "pooled_unnormalized": pooled_np.tolist(),
        "pooled_norm": float(pooled.norm()),
        "truncated": {str(d): st_trunc[d].tolist() for d in MRL_DIMS},
        "st_vs_direct_max_abs": diff,
        "last_hidden_state": write_f32(f"{name}-hidden.f32", hidden.numpy()),
    }
    for d in MRL_DIMS:
        manual = normed_np[:d] / np.linalg.norm(normed_np[:d])
        record.setdefault("truncated_check_max_abs", {})[str(d)] = float(np.abs(manual - st_trunc[d]).max())
    record.update(case.get("extra", {}))

    if "pixel_values" in inputs:
        pv, pos = inputs["pixel_values"], inputs["image_position_ids"]
        real = (pos[0] != -1).all(-1)
        n_real = int(real.sum())
        assert bool(real[:n_real].all()), "padding patches trail the real ones"
        # Positions are (x, y), x fastest: the grid is all a replay needs.
        grid_w, grid_h = int(pos[0, :n_real, 0].max()) + 1, int(pos[0, :n_real, 1].max()) + 1
        assert grid_w * grid_h == n_real and pos[0, 1, 0] == 1
        record["image_position_ids"] = {"shape": list(pos.shape), "real_patches": n_real,
                                        "patch_grid_w": grid_w, "patch_grid_h": grid_h,
                                        "order": "row-major, x fastest; padding rows are (-1, -1)"}
        record["pixel_values_shape"] = list(pv.shape)
        if case.get("tower"):
            record["pixel_values"] = write_f32(f"{name}-pixel-values.f32", pv[0, :n_real].numpy())
            with torch.no_grad():
                rows = hf.get_image_features(pv, pos, return_dict=True).pooler_output
            rows = torch.cat(rows, 0)
            assert rows.shape[0] == counts["image_soft_tokens"], (rows.shape, counts)
            record["image_features"] = write_f32(f"{name}-image-features.f32", rows.numpy())
    if "pixel_values_videos" in inputs:
        pv, pos = inputs["pixel_values_videos"], inputs["video_position_ids"]
        real = (pos != -1).all(-1)
        record["video"] = {
            "sampled_frames": frames,
            "num_frames_per_video": inputs["num_frames_per_video"].tolist(),
            "pixel_values_videos_shape": list(pv.shape),
            "real_patches_per_frame": real.sum(-1).tolist(),
            "soft_tokens_per_frame": (real.sum(-1) // cfg.vision_config.pooling_kernel_size ** 2).tolist(),
            "video_position_ids_frame0": pos[0].tolist(),
        }
        n_real = int(real[0].sum())
        assert bool((real.sum(-1) == n_real).all())
        record["video"]["pixel_values_videos"] = write_f32(f"{name}-pixel-values-videos.f32",
                                                           pv[:, :n_real].numpy())
        decoded = case["input"]["video"]["array"]
        kept = decoded[frames[0]["indices"]]
        path = ROOT / f"{name}-frames.u8"
        refuse_existing(path)
        path.write_bytes(np.ascontiguousarray(kept, dtype=np.uint8).tobytes())
        record["video"]["sampled_frames_rgb"] = {"file": path.name, "shape": list(kept.shape), "dtype": "u8",
                                                 "layout": "frame, row, column, RGB"}
    if "input_features" in inputs:
        feats, fmask = inputs["input_features"], inputs["input_features_mask"]
        record["audio"] = {"input_features_shape": list(feats.shape),
                           "mel_frames_valid": int(fmask[0].sum())}
        if case.get("tower"):
            record["audio"]["input_features"] = write_f32(f"{name}-mel.f32", feats[0].numpy())
            with torch.no_grad():
                a = hf.get_audio_features(feats, fmask, return_dict=True)
            rows = a.pooler_output[0][a.attention_mask[0]]
            record["audio"]["encoder_frames_total"] = int(a.attention_mask.shape[1])
            record["audio"]["encoder_frames_valid"] = int(a.attention_mask[0].sum())
            assert rows.shape[0] == counts["audio_soft_tokens"], (rows.shape, counts)
            record["audio"]["audio_features"] = write_f32(f"{name}-audio-features.f32", rows.numpy())
            record["audio"]["tower_output"] = write_f32(f"{name}-audio-tower.f32",
                                                        a.last_hidden_state[0][a.attention_mask[0]].numpy())

    refuse_existing(meta_path)
    meta_path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n")
    print(f"{name}: {len(ids)} tokens {counts}")
    return record


def tokenize_view(tok, ids):
    """Run-length view of the token strings: long runs of one soft token collapse."""
    toks = tok.convert_ids_to_tokens(ids)
    out, i = [], 0
    while i < len(toks):
        j = i
        while j < len(toks) and toks[j] == toks[i]:
            j += 1
        out.append(toks[i] if j - i == 1 else f"{toks[i]} x{j - i}")
        i = j
    return out


def long_text(tok) -> tuple[str, int]:
    lines = (REPO / "test-vectors" / "prompts" / LONG_SOURCE).read_text().splitlines(keepends=True)
    for n in range(1, len(lines) + 1):
        text = "".join(lines[:n]).rstrip("\n")
        if len(tok(text)["input_ids"]) >= LONG_TARGET:
            return text, n
    sys.exit(f"{LONG_SOURCE} is shorter than {LONG_TARGET} tokens")


def main():
    global ROOT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", required=True, help="the checkpoint snapshot directory")
    # sdpa: the audio tower's eager path inverts its attention mask.
    ap.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    ap.add_argument("--out", type=Path, default=ROOT, help="fixture directory (default: %(default)s)")
    args = ap.parse_args()
    ROOT = args.out

    import av, sentence_transformers, soundfile
    from PIL import Image
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    ROOT.mkdir(parents=True, exist_ok=True)
    model = SentenceTransformer(args.model_dir, device="cpu",
                                model_kwargs={"torch_dtype": torch.float32, "attn_implementation": args.attn})
    model.eval()
    rec = Recorder(model[0].processor)
    tok = model[0].processor.tokenizer

    image_path = ROOT / "image-synthetic.png"
    make_image(image_path)
    speech10 = ROOT / "audio-speech-10s.wav"
    cut_wav(speech10, CLIPS / "speech-en.wav", 10)
    speech30 = ROOT / "audio-speech-30s.wav"
    cut_wav(speech30, CLIPS / "speech-en.wav", None)
    video_path = ROOT / "video-synthetic.mp4"
    make_video(video_path)
    video_frames, video_meta = decode_video(video_path)

    def wav(path):
        data, rate = soundfile.read(path, dtype="float32")
        return {"array": data, "sampling_rate": rate}

    def image():
        with Image.open(image_path) as im:
            return im.convert("RGB")

    long, long_lines = long_text(tok)
    cases = [
        dict(name="query-northern-lights", input="What causes the northern lights?", prompt_name="SearchQuery",
             description="text query with the SearchQuery prompt",
             st_input_desc="str 'What causes the northern lights?'"),
        dict(name="doc-northern-lights", input="The northern lights are caused by charged particles from the sun.",
             prompt_name="Document", description="text document with the Document prompt",
             st_input_desc="str 'The northern lights are caused by charged particles from the sun.'"),
        dict(name="plain-multilingual", input="Merhaba dünya! Bugün hava çok güzel.",
             description="Turkish text, no prompt", st_input_desc="str 'Merhaba dünya! Bugün hava çok güzel.'"),
        dict(name="code", input=CODE, prompt_name="CodeRetrieval",
             description="a 16-line Rust function with the CodeRetrieval prompt", st_input_desc="str: the `text` field"),
        dict(name="long-1500", input=long,
             description=f"the first {long_lines} lines of test-vectors/prompts/{LONG_SOURCE} (trailing newline "
                         f"stripped), no prompt: past the sliding window on both sides of most tokens",
             st_input_desc="str: the `text` field",
             extra={"text": long, "source": f"test-vectors/prompts/{LONG_SOURCE}", "source_lines": long_lines}),
        dict(name="image-synthetic", input=image(), tower=True,
             description="image-synthetic.png alone (640x480 RGB), no prompt",
             st_input_desc="PIL.Image from image-synthetic.png, converted to RGB",
             extra={"media": {"file": image_path.name, "sha256": sha256(image_path)}}),
        dict(name="image-text-interleaved",
             input={"text": "A product photo <|image|> of a red square", "image": image()},
             description="text with one image placeholder, image-synthetic.png, no prompt",
             st_input_desc="dict {'text': 'A product photo <|image|> of a red square', 'image': PIL.Image}",
             extra={"media": {"file": image_path.name, "sha256": sha256(image_path)}}),
        dict(name="audio-speech", input=wav(speech10), tower=True,
             description="the first 10 s of references/audio/clips/speech-en.wav, alone, no prompt",
             st_input_desc="dict {'array': float32 samples of audio-speech-10s.wav, 'sampling_rate': 16000}",
             extra={"media": {"file": speech10.name, "sha256": sha256(speech10), "seconds": 10,
                              "derived_from": "references/audio/clips/speech-en.wav"}}),
        dict(name="audio-30s", input=wav(speech30),
             description="all 30 s of references/audio/clips/speech-en.wav, alone, no prompt; "
                         "probes whether the processor caps audio soft tokens at audio_seq_length",
             st_input_desc="dict {'array': float32 samples of audio-speech-30s.wav, 'sampling_rate': 16000}",
             extra={"media": {"file": speech30.name, "sha256": sha256(speech30), "seconds": 30,
                              "derived_from": "references/audio/clips/speech-en.wav"},
                    "audio_seq_length": model[0].processor.audio_seq_length}),
        dict(name="video-synthetic", input={"video": {"array": video_frames, "video_metadata": video_meta}},
             description=f"video-synthetic.mp4 alone ({VIDEO_W}x{VIDEO_H}, {VIDEO_FPS} fps, {VIDEO_SECONDS} s, "
                         "H.264 yuv420p), no prompt; decoded with PyAV to rgb24",
             st_input_desc="dict {'video': {'array': uint8 [T, H, W, 3] PyAV rgb24 frames of video-synthetic.mp4, "
                           "'video_metadata': VideoMetadata(fps, total_num_frames, duration, width, height)}}",
             extra={"media": {"file": video_path.name, "sha256": sha256(video_path), "fps": VIDEO_FPS,
                              "seconds": VIDEO_SECONDS, "width": VIDEO_W, "height": VIDEO_H}}),
    ]
    for case in cases:
        if case["name"] in ("code", "long-1500"):
            case.setdefault("extra", {})["text"] = case["input"]
        capture(model, rec, case)

    manifest_path = ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "checkpoint": "google/embeddinggemma-2", "revision": REVISION,
        "transformers": transformers.__version__, "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__, "pyav": av.__version__,
        "dtype": "float32", "device": "cpu", "attn_implementation": args.attn,
        "attn_note": ("sdpa and eager agree to 2.5e-7 on every text, image and video case, but the audio "
                      "tower's eager path inverts its mask (cosine 0.75 against sdpa on both audio cases), "
                      "so captures use sdpa"),
        "pipeline": "SentenceTransformer(modules.json): Transformer -> Pooling(mean, include_prompt) -> Normalize",
        "produced_by": "scripts/reference/embedding_gemma2_reference.py",
        "how": ("each case: SentenceTransformer.encode([input], prompt_name) gives `embedding`; "
                "encode(truncate_dim=d, normalize_embeddings=True) gives `truncated[d]`; "
                "model.preprocess(...) then EmbeddingGemma2Model(**features) gives `last_hidden_state`, "
                "whose mean over the attention mask is `pooled_unnormalized`; its L2 normalization must match "
                "`embedding` within 1e-5 or the capture stops"),
        "media": {p.name: sha256(p) for p in (image_path, speech10, speech30, video_path)},
    })
    cases_index = manifest.setdefault("cases", {})
    for case in cases:
        if (ROOT / f"{case['name']}.json").exists():
            cases_index[case["name"]] = {"file": f"{case['name']}.json", "description": case["description"]}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
