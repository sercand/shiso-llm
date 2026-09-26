#!/usr/bin/env python3
"""Capture independent audio references for shiso's audio tests.

Runs the reference feature extractors and reference models from `transformers` over a small fixed set of public clips, and writes
fixtures under test-vectors/references/audio/. It never overwrites an
existing fixture: one already present is kept, and deleting it deliberately is
how it is recaptured.

    audio_reference.py mel --asr-dir ~/models/nemotron-3.5-asr-streaming-0.6b/nvidia \
                           --gemma-dir ~/models/gemma/gemma-4-E4B-it-qat-4bit-mlx
"""
import argparse, hashlib, json, subprocess, sys, urllib.request
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoFeatureExtractor

ROOT = Path(__file__).resolve().parents[2] / "test-vectors" / "references" / "audio"
CLIPS = {
    # name: (url, seconds to keep)
    "weather-es": ("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/bcn_weather.mp3", 30),
    "speech-en": ("https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main/obama.mp3", 30),
}
# name: (source clip, samples to keep). Lengths that are NOT a multiple of the
# 128-sample padding unit, so the left-padded frontend's valid-frame mask meets
# a boundary the downloaded clips never reach.
DERIVED = {
    "speech-en-odd": ("speech-en", 176_641),
}
# name: [(source clip, start second, end second), ...], joined in order. Two
# real voices with a known turn order, for the diarizer.
SPLICED = {
    # 41 s (weather-es is about 11 s long, so its second span is short): two
    # offline steps, the second over the compressed cache.
    "two-speakers": [("speech-en", 0, 10), ("weather-es", 0, 10), ("speech-en", 10, 20),
                     ("weather-es", 10, 20), ("speech-en", 20, 30)],
    # 71 s: three offline steps, so the steady-state step (cache, FIFO, chunk
    # and lookahead all full: 684 frames) and a later compression both run.
    "long-dialogue": [("speech-en", 0, 30), ("weather-es", 0, 11), ("speech-en", 0, 30)],
}


def refuse_existing(path: Path):
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")


def fetch_clips() -> dict:
    clips = ROOT / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    manifest_path = clips / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name, (url, keep) in CLIPS.items():
        wav = clips / f"{name}.wav"
        if wav.exists():
            continue
        raw = urllib.request.urlopen(url).read()
        subprocess.run(
            ["ffmpeg", "-v", "error", "-i", "pipe:0", "-t", str(keep), "-ac", "1", "-ar", "16000",
             "-c:a", "pcm_s16le", str(wav)],
            input=raw, check=True)
        manifest[name] = {"url": url, "source_sha256": hashlib.sha256(raw).hexdigest(),
                          "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(), "seconds": keep}
    import soundfile
    for name, (source, keep) in DERIVED.items():
        wav = clips / f"{name}.wav"
        if wav.exists():
            continue
        data, rate = soundfile.read(clips / f"{source}.wav", dtype="int16")
        soundfile.write(wav, data[:keep], rate, subtype="PCM_16")
        manifest[name] = {"derived_from": source, "samples": keep,
                          "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest()}
    for name, spans in SPLICED.items():
        wav = clips / f"{name}.wav"
        if wav.exists():
            continue
        parts = []
        for source, start, end in spans:
            data, rate = soundfile.read(clips / f"{source}.wav", dtype="int16")
            parts.append(data[start * rate:end * rate])
        soundfile.write(wav, np.concatenate(parts), rate, subtype="PCM_16")
        manifest[name] = {"spliced_from": [list(s) for s in spans],
                          "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_wav(path: Path) -> np.ndarray:
    import soundfile
    data, rate = soundfile.read(path, dtype="float32")
    assert rate == 16000 and data.ndim == 1
    return data


def write_mel(clip: str, style: str, feats: np.ndarray, valid: int, extractor, source: str):
    meta = ROOT / f"mel-{clip}-{style}.json"
    blob = ROOT / f"mel-{clip}-{style}.f32"
    if meta.exists() and blob.exists():
        print(f"kept {meta.name}: delete it to recapture")
        return
    refuse_existing(meta)
    refuse_existing(blob)
    feats = np.ascontiguousarray(feats, dtype="<f4")
    blob.write_bytes(feats.tobytes())
    meta.write_text(json.dumps({
        "clip": clip, "style": style, "frames_total": int(feats.shape[0]), "frames_valid": int(valid),
        "n_mels": int(feats.shape[1]), "transformers": transformers.__version__, "torch": torch.__version__,
        "extractor": type(extractor).__name__, "source_dir": source,
    }, indent=2) + "\n")


def cmd_mel(args):
    fetch_clips()
    asr = AutoFeatureExtractor.from_pretrained(args.asr_dir)
    gemma = AutoFeatureExtractor.from_pretrained(args.gemma_dir)
    for clip in [*CLIPS, *DERIVED]:
        audio = read_wav(ROOT / "clips" / f"{clip}.wav")
        out = asr(audio, sampling_rate=16000, return_tensors="np")
        feats, mask = out["input_features"][0], out["attention_mask"][0]
        write_mel(clip, "centred", feats, int(mask.sum()), asr, args.asr_dir)
        # The Gemma extractor only takes a batch: a bare 1-D array trips its
        # own squeeze.
        out = gemma([audio], sampling_rate=16000, return_tensors="np")
        feats, mask = out["input_features"][0], out["input_features_mask"][0]
        write_mel(clip, "left-padded", feats, int(mask.sum()), gemma, args.gemma_dir)


def write_blob(path: Path, array) -> None:
    refuse_existing(path)
    path.write_bytes(np.ascontiguousarray(array, dtype="<f4").tobytes())


# (clip, language tag, prompt id) cases. `auto` is prompt 101; the others are
# the clip's own language from the checkpoint's prompt dictionary.
ASR_CASES = [
    ("speech-en", "auto", 101), ("speech-en", "en-US", 0),
    ("weather-es", "auto", 101), ("weather-es", "es-US", 3),
    ("speech-en-odd", "auto", 101),
]


def cmd_asr(args):
    fetch_clips()
    from transformers import AutoTokenizer, Nemotron3_5AsrForRNNT
    fe = AutoFeatureExtractor.from_pretrained(args.asr_dir)
    tok = AutoTokenizer.from_pretrained(args.asr_dir)
    model = Nemotron3_5AsrForRNNT.from_pretrained(args.asr_dir, dtype=torch.float32).eval()
    blank = model.config.blank_token_id
    for lookahead in (13, 3):
        for clip, lang, prompt in ASR_CASES:
            meta = ROOT / f"asr-{clip}-r{lookahead}-{lang}.json"
            if meta.exists():
                print(f"kept {meta.name}: delete it to recapture")
                continue
            audio = read_wav(ROOT / "clips" / f"{clip}.wav")
            out = fe(audio, sampling_rate=16000, return_tensors="pt")
            valid = int(out["attention_mask"][0].sum())
            x = out["input_features"][:, :valid].to(torch.float32)
            with torch.no_grad():
                feats = model.get_audio_features(
                    input_features=x, num_lookahead_tokens=lookahead, prompt_ids=torch.tensor([prompt]))
                gen = model.generate(input_features=x, num_lookahead_tokens=lookahead,
                                     prompt_ids=torch.tensor([prompt]))
            seq = gen.sequences[0].tolist()
            dur = gen.durations[0]
            steps = (dur.cumsum(-1) - dur).tolist()
            tokens = [t for t in seq[1:] if t != blank]
            frames = [steps[i] for i, t in enumerate(seq) if i > 0 and t != blank]
            enc = feats.last_hidden_state[0].numpy()
            proj = feats.pooler_output[0].numpy()
            # The encoder output does not depend on the prompt; keep it once per
            # clip and lookahead, in full, beside the auto-language case.
            if lang == "auto":
                write_blob(ROOT / f"asr-{clip}-r{lookahead}-encoder.f32", enc)
            write_blob(ROOT / f"asr-{clip}-r{lookahead}-{lang}-projected.f32", proj)
            meta.write_text(json.dumps({
                "clip": clip, "lookahead": lookahead, "language": lang, "prompt_id": prompt,
                "mel_frames": valid, "encoder_frames": int(enc.shape[0]), "hidden": int(enc.shape[1]),
                "projected": int(proj.shape[1]), "tokens": tokens, "frames": frames,
                "text": tok.decode(tokens, skip_special_tokens=False),
                "transformers": transformers.__version__, "torch": torch.__version__,
                "source_dir": args.asr_dir,
            }, indent=2, ensure_ascii=False) + "\n")
            print(f"{meta.name}: {len(tokens)} tokens: {tok.decode(tokens, skip_special_tokens=False)[:80]!r}")


def mlx_dequant(words, scales, biases, bits):
    """MLX affine: element j of a u32 word sits in its low bits first; w = q*scale + bias per group."""
    words = np.asarray(words)
    per = 32 // bits
    q = np.stack([(words >> (bits * i)) & ((1 << bits) - 1) for i in range(per)], axis=-1)
    q = q.reshape(words.shape[0], -1).astype(np.float32)
    groups = scales.shape[1]
    q = q.reshape(words.shape[0], groups, -1)
    return (q * scales[..., None] + biases[..., None]).reshape(words.shape[0], -1)


GEMMA_CLIPS = ["speech-en", "weather-es", "speech-en-odd"]


def cmd_gemma(args):
    fetch_clips()
    from safetensors import safe_open
    from transformers import Gemma4AudioConfig, Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4AudioModel, Gemma4MultimodalEmbedder
    root = Path(args.gemma_dir)
    config = json.loads((root / "config.json").read_text())
    audio_config = Gemma4AudioConfig(**config["audio_config"])
    # The masked blocked attention is only right on the sdpa path; the eager
    # path inverts its mask.
    audio_config._attn_implementation = "sdpa"
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    tensors = {}
    for shard in sorted({index[k] for k in index if "audio" in k}):
        with safe_open(str(root / shard), "pt") as f:
            for k in f.keys():
                if "audio" in k:
                    tensors[k] = f.get_tensor(k)
    state = {}
    for k, v in tensors.items():
        if not k.startswith("audio_tower."):
            continue
        name, v = k[len("audio_tower."):], v.float()
        if name.endswith("conv.weight") and v.ndim == 4:
            v = v.permute(0, 3, 1, 2)          # MLX [out, kh, kw, in] -> [out, in, kh, kw]
        if name.endswith("depthwise_conv1d.weight"):
            v = v.permute(0, 2, 1)             # MLX [out, k, in] -> [out, in, k]
        state[name] = v
    tower = Gemma4AudioModel(audio_config).eval()
    result = tower.load_state_dict(state, strict=False)
    if result.missing_keys:
        sys.exit(f"the audio tower is missing {result.missing_keys[:5]}")
    embedder = Gemma4MultimodalEmbedder(audio_config, Gemma4TextConfig(hidden_size=config["text_config"]["hidden_size"])).eval()
    stem = "embed_audio.embedding_projection"
    weight = mlx_dequant(tensors[f"{stem}.weight"].numpy(), tensors[f"{stem}.scales"].float().numpy(),
                         tensors[f"{stem}.biases"].float().numpy(), config["quantization"]["bits"])
    with torch.no_grad():
        embedder.embedding_projection.weight.copy_(torch.from_numpy(weight))
    fe = AutoFeatureExtractor.from_pretrained(args.gemma_dir)
    for clip in GEMMA_CLIPS:
        meta = ROOT / f"gemma-{clip}.json"
        if meta.exists():
            print(f"kept {meta.name}: delete it to recapture")
            continue
        audio = read_wav(ROOT / "clips" / f"{clip}.wav")
        out = fe([audio], sampling_rate=16000, return_tensors="pt")
        x, mask = out["input_features"].float(), out["input_features_mask"].bool()
        with torch.no_grad():
            enc = tower(x, mask)
            hidden, valid = enc.last_hidden_state[0], enc.attention_mask[0].bool()
            embedded = embedder(enc.last_hidden_state)[0]
        write_blob(ROOT / f"gemma-{clip}-tower.f32", hidden[valid].numpy())
        write_blob(ROOT / f"gemma-{clip}-embedded.f32", embedded[valid].numpy())
        meta.write_text(json.dumps({
            "clip": clip, "samples": int(len(audio)), "mel_frames": int(x.shape[1]), "mel_valid": int(mask.sum()),
            "tokens": int(valid.sum()), "tower_width": int(hidden.shape[1]), "embedded_width": int(embedded.shape[1]),
            "attention": "sdpa", "transformers": transformers.__version__, "torch": torch.__version__,
            "source_dir": args.gemma_dir,
        }, indent=2) + "\n")
        print(f"{meta.name}: {int(valid.sum())} audio tokens")


# Clips the diarizer fixtures cover; the odd-length one only needs its
# probabilities, to pin the trim at a mel length that is not a multiple of 8.
DIAR_CLIPS = [("speech-en", True), ("two-speakers", True), ("speech-en-odd", False),
              ("long-dialogue", False)]


def segments(probs: np.ndarray, threshold: float = 0.5):
    """[speaker, start frame, end frame) runs above `threshold`, by start."""
    out = []
    for s in range(probs.shape[1]):
        active = np.concatenate([[False], probs[:, s] > threshold, [False]])
        edges = np.flatnonzero(active[1:] != active[:-1])
        out += [[s, int(a), int(b)] for a, b in zip(edges[::2], edges[1::2])]
    return sorted(out, key=lambda r: (r[1], r[0]))


def cmd_diar(args):
    fetch_clips()
    from transformers import Nemotron3DiarizationForAudioFrameClassification
    fe = AutoFeatureExtractor.from_pretrained(args.diar_dir)
    model = Nemotron3DiarizationForAudioFrameClassification.from_pretrained(
        args.diar_dir, dtype=torch.float32).eval()
    tower = model.model.audio_tower
    for clip, full in DIAR_CLIPS:
        meta = ROOT / f"diar-{clip}.json"
        if meta.exists():
            print(f"kept {meta.name}: delete it to recapture")
            continue
        audio = read_wav(ROOT / "clips" / f"{clip}.wav")
        out = fe(audio, sampling_rate=16000, return_tensors="pt")
        valid = int(out["attention_mask"][0].sum())
        x = out["input_features"][:, :valid].to(torch.float32)
        if not (ROOT / f"mel-{clip}-centred.json").exists():
            write_mel(clip, "centred", x[0].numpy(), valid, fe, args.diar_dir)
        captured = []
        hook = tower.register_forward_hook(lambda m, i, o: captured.append(o.last_hidden_state[0]))
        with torch.no_grad():
            probs = model(input_features=x).logits[0].sigmoid().numpy()
            embeds = tower.embedder(x)[0].numpy()
        hook.remove()
        write_blob(ROOT / f"diar-{clip}-probs.f32", probs)
        if full:
            write_blob(ROOT / f"diar-{clip}-embeds.f32", embeds)
            write_blob(ROOT / f"diar-{clip}-step0.f32", captured[0].numpy())
        segs = segments(probs)
        meta.write_text(json.dumps({
            "clip": clip, "mel_frames": valid, "encoder_frames": int(embeds.shape[0]),
            "step0_frames": int(captured[0].shape[0]), "steps": len(captured), "speakers": int(probs.shape[1]),
            "speakers_active": sorted({s for s, _, _ in segs}), "segments_at_0_5": segs,
            "mode": "offline", "transformers": transformers.__version__, "torch": torch.__version__,
            "source_dir": args.diar_dir,
        }, indent=2) + "\n")
        print(f"{meta.name}: {len(captured)} steps, speakers {sorted({s for s, _, _ in segs})}, "
              f"{len(segs)} segments")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mel", help="log-mel features for every clip, both frontend styles")
    m.add_argument("--asr-dir", required=True)
    m.add_argument("--gemma-dir", required=True)
    m.set_defaults(func=cmd_mel)
    a = sub.add_parser("asr", help="speech recognizer: encoder outputs, tokens, frames and text")
    a.add_argument("--asr-dir", required=True)
    a.set_defaults(func=cmd_asr)
    g = sub.add_parser("gemma", help="Gemma 4 audio tower and embedder outputs")
    g.add_argument("--gemma-dir", required=True)
    g.set_defaults(func=cmd_gemma)
    d = sub.add_parser("diar", help="speaker diarizer: offline probabilities, embeddings, first-step encoder")
    d.add_argument("--diar-dir", required=True)
    d.set_defaults(func=cmd_diar)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
