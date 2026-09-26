#!/usr/bin/env python3
"""Capture independent Qwen3-ASR / R2T2 reference fixtures for shiso's speech
recognizer tests.

Runs the reference `qwen_asr` package and `transformers` in fp32 on CPU over
the fixed clips in test-vectors/references/audio/clips/ and writes fixtures
under test-vectors/references/audio/. It never overwrites an existing
fixture: one already present is kept, and deleting it deliberately is how it
is recaptured.

Qwen3-ASR and R2T2 share the same architecture (`Qwen3ASRForConditionalGeneration`)
and, per the model notes, an audio tower that is bit-identical between the two
checkpoints. So the mel and windowed-encoder fixtures are captured once from
the Qwen3-ASR run and copied byte-for-byte to the R2T2 names, after asserting
the two checkpoints' audio tower weights actually match. Prompt ids and
generated tokens differ per model (different text decoder, and R2T2 carries
its own tokenizer additions), so those are captured independently for each.

    qwen3_asr_reference.py --qwen3 ~/models/Qwen3-ASR-1.7B/qwen --r2t2 ~/models/R2T2/bf16
"""
import argparse
import json
import sys
from importlib.metadata import version as pkg_version
from pathlib import Path

import numpy as np
import torch
import transformers
from qwen_asr import Qwen3ASRModel
from qwen_asr.inference.utils import normalize_audios, parse_asr_output

import qwen_asr.core.transformers_backend.modeling_qwen3_asr as M

ROOT = Path(__file__).resolve().parents[2] / "test-vectors" / "references" / "audio"

CLIPS = ["speech-en", "speech-en-odd", "weather-es", "two-speakers"]

# The forced language per clip (NOTES §3: canonicalised to first-letter-upper,
# rest-lower, and must be one of the checkpoint's 30 supported languages).
FORCED_LANGUAGE = {
    "speech-en": "English",
    "speech-en-odd": "English",
    "two-speakers": "English",
    "weather-es": "Spanish",
}

AUDIO_PAD_ID = 151676  # `<|audio_pad|>` (NOTES §3)

# NOTES §8.B: the windowed-encoder, auto-language golden for speech-en. This is
# the recommended golden (matches vLLM/FA2), not the HF sdpa/eager default,
# which silently runs full-clip attention over the encoder (NOTES §2.4/§7.1).
SPEECH_EN_AUTO_IDS_NOTES = [
    11528, 6364, 151704, 1986, 2003, 11, 358, 30696, 311, 10557, 311, 6359, 847, 1590, 71543, 2621, 311,
    279, 6995, 13, 22713, 304, 279, 13815, 315, 46508, 1573, 752, 11, 432, 572, 458, 6638, 311, 1977, 9702,
    498, 13, 13139, 582, 3003, 3884, 7912, 311, 7912, 476, 18521, 7230, 518, 678, 11, 847, 20959, 448, 498,
    11, 279, 3693, 1251, 11, 304, 5382, 12026, 11, 304, 8682, 11, 518, 33224, 11, 323, 389, 8633, 25945, 11,
    518, 11623, 388, 323, 389, 28727, 6277, 700, 12664, 86147, 1493, 20959, 525, 1128, 614, 8604, 752, 10745,
    13, 151645,
]

# NOTES §2.4/§7.1: in transformers 4.57.6 the encoder layer is called as
# `encoder_layer(hidden_states, cu_seqlens)` with no attention_mask, and the
# sdpa/eager attention paths silently ignore cu_seqlens, giving full-clip
# attention instead of vLLM/FA2's 104-token windows. Patch the layer so every
# run in this script (both the standalone encoder fixture and generation)
# takes the windowed path the notes call the recommended golden.
_ORIG_LAYER_FORWARD = M.Qwen3ASRAudioEncoderLayer.forward


def _windowed_layer_forward(self, hidden_states, cu_seqlens, attention_mask=None, **kwargs):
    del attention_mask  # the default call site never passes one; always build our own
    mask = torch.full(
        (1, 1, hidden_states.shape[0], hidden_states.shape[0]),
        torch.finfo(hidden_states.dtype).min,
        dtype=hidden_states.dtype,
    )
    for i in range(1, len(cu_seqlens)):
        mask[..., cu_seqlens[i - 1] : cu_seqlens[i], cu_seqlens[i - 1] : cu_seqlens[i]] = 0
    return _ORIG_LAYER_FORWARD(self, hidden_states, cu_seqlens, attention_mask=mask, **kwargs)


M.Qwen3ASRAudioEncoderLayer.forward = _windowed_layer_forward


def versions() -> dict:
    return {
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "qwen_asr": pkg_version("qwen-asr"),
    }


def refuse_existing(path: Path):
    if path.exists():
        sys.exit(f"{path} exists; delete it to recapture")


def kept(path: Path) -> bool:
    if path.exists():
        print(f"kept {path.name}: delete it to recapture")
        return True
    return False


def write_blob(path: Path, array: np.ndarray):
    refuse_existing(path)
    path.write_bytes(np.ascontiguousarray(array, dtype="<f4").tobytes())


def write_json(path: Path, obj: dict):
    refuse_existing(path)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def load_model(model_dir: str) -> Qwen3ASRModel:
    wrapper = Qwen3ASRModel.from_pretrained(
        model_dir, dtype=torch.float32, device_map="cpu", max_new_tokens=512
    )
    wrapper.model.eval()
    return wrapper


def clone_state_dict(module: torch.nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def assert_towers_equal(a: dict, b: dict):
    """Fail loudly unless R2T2's audio tower is bit-identical to Qwen3-ASR's (NOTES §0/§8's cross-model claim)."""
    if a.keys() != b.keys():
        sys.exit(f"audio tower tensor names differ: {sorted(set(a) ^ set(b))}")
    for k in a:
        if a[k].shape != b[k].shape:
            sys.exit(f"audio tower {k} shape differs: {tuple(a[k].shape)} vs {tuple(b[k].shape)}")
        if not torch.equal(a[k], b[k]):
            diff = (a[k] - b[k]).abs().max().item()
            sys.exit(f"audio tower {k} differs: maxabs {diff}")


def build_inputs(wrapper: Qwen3ASRModel, wav: np.ndarray, force_language):
    text = wrapper._build_text_prompt(context="", force_language=force_language)
    return wrapper.processor(text=[text], audio=[wav], return_tensors="pt", padding=True)


def run_clip(wrapper: Qwen3ASRModel, model_key: str, clip: str, source_dir: str, capture_mel_encoder: bool):
    wav = normalize_audios(str(ROOT / "clips" / f"{clip}.wav"))[0]
    forced_lang = FORCED_LANGUAGE[clip]

    inputs_auto = build_inputs(wrapper, wav, None)
    inputs_forced = build_inputs(wrapper, wav, forced_lang)

    ids_auto = inputs_auto["input_ids"][0].tolist()
    ids_forced = inputs_forced["input_ids"][0].tolist()
    feature_len = int(inputs_auto["feature_attention_mask"][0].sum())

    prompt_path = ROOT / f"{model_key}-{clip}-prompt.json"
    if not kept(prompt_path):
        write_json(
            prompt_path,
            {
                "clip": clip,
                "model": model_key,
                "auto": {"ids": ids_auto, "n_ids": len(ids_auto), "n_audio_pad": ids_auto.count(AUDIO_PAD_ID)},
                "forced": {
                    "language": forced_lang,
                    "ids": ids_forced,
                    "n_ids": len(ids_forced),
                    "n_audio_pad": ids_forced.count(AUDIO_PAD_ID),
                },
                "source_dir": source_dir,
                **versions(),
            },
        )

    if capture_mel_encoder:
        mel_path = ROOT / f"{model_key}-{clip}-mel.f32"
        mel_meta_path = ROOT / f"{model_key}-{clip}-mel.json"
        enc_path = ROOT / f"{model_key}-{clip}-encoder.f32"
        if mel_path.exists() and mel_meta_path.exists() and enc_path.exists():
            print(f"kept {mel_path.name}, {mel_meta_path.name}, {enc_path.name}: delete them to recapture")
        else:
            # NOTES §6: the encoder is run per clip on input_features[:, :feature_len]
            # (`input_feature[:, :feature_len]` in the thinker's own get_audio_features).
            mel = inputs_auto["input_features"][0][:, :feature_len]
            with torch.no_grad():
                enc = wrapper.model.thinker.audio_tower(
                    mel, feature_lens=torch.tensor([feature_len])
                ).last_hidden_state.numpy()
            if not enc_path.exists():
                write_blob(enc_path, enc)
            else:
                print(f"kept {enc_path.name}: delete it to recapture")
            if not (mel_path.exists() and mel_meta_path.exists()):
                write_blob(mel_path, mel.numpy().T)  # row-major [frames][128]
                write_json(
                    mel_meta_path,
                    {
                        "clip": clip,
                        "model": model_key,
                        "frames": feature_len,
                        "n_mels": int(mel.shape[0]),
                        "n_audio_tokens": int(enc.shape[0]),
                        "source_dir": source_dir,
                        **versions(),
                    },
                )
            else:
                print(f"kept {mel_path.name}, {mel_meta_path.name}: delete them to recapture")

    tokens_path = ROOT / f"{model_key}-{clip}-tokens.json"
    if kept(tokens_path):
        return
    with torch.no_grad():
        gen_auto = wrapper.model.generate(**inputs_auto, max_new_tokens=512)
        gen_forced = wrapper.model.generate(**inputs_forced, max_new_tokens=512)
    seq_auto = gen_auto.sequences[0, len(ids_auto) :].tolist()
    seq_forced = gen_forced.sequences[0, len(ids_forced) :].tolist()
    text_auto = wrapper.processor.batch_decode(
        [seq_auto], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    text_forced = wrapper.processor.batch_decode(
        [seq_forced], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    lang_auto, parsed_auto = parse_asr_output(text_auto, user_language=None)
    lang_forced, parsed_forced = parse_asr_output(text_forced, user_language=forced_lang)
    write_json(
        tokens_path,
        {
            "clip": clip,
            "model": model_key,
            "auto": {
                "generated_ids": seq_auto,
                "decoded_text": text_auto,
                "language": lang_auto,
                "text": parsed_auto,
            },
            "forced": {
                "language_forced": forced_lang,
                "generated_ids": seq_forced,
                "decoded_text": text_forced,
                "language": lang_forced,
                "text": parsed_forced,
            },
            "source_dir": source_dir,
            **versions(),
        },
    )
    print(
        f"{tokens_path.name}: auto {len(seq_auto)} ids ({lang_auto!r}), "
        f"forced {len(seq_forced)} ids ({lang_forced!r})"
    )


def copy_mel_encoder(clip: str):
    """R2T2's audio tower is bit-identical to Qwen3-ASR's (asserted by the caller before this
    runs), so its mel/encoder fixtures are byte-identical copies of the Qwen3-ASR ones, not
    symlinks, so each model's tests find its own file names."""
    for suffix in ("mel.f32", "mel.json", "encoder.f32"):
        src = ROOT / f"qwen3asr-{clip}-{suffix}"
        dst = ROOT / f"r2t2-{clip}-{suffix}"
        if dst.exists():
            print(f"kept {dst.name}: delete it to recapture")
            continue
        dst.write_bytes(src.read_bytes())


def check_speech_en_golden():
    path = ROOT / "qwen3asr-speech-en-tokens.json"
    got = json.loads(path.read_text())["auto"]["generated_ids"]
    if got != SPEECH_EN_AUTO_IDS_NOTES:
        sys.exit(
            "NEEDS_CONTEXT: qwen3asr speech-en auto tokens do not match NOTES §8. "
            f"got {len(got)} ids, expected {len(SPEECH_EN_AUTO_IDS_NOTES)}.\n"
            f"got:      {got}\n"
            f"expected: {SPEECH_EN_AUTO_IDS_NOTES}"
        )
    print(f"speech-en auto tokens match NOTES §8 ({len(got)} ids)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen3", required=True, help="Qwen3-ASR HF bf16 directory (…/Qwen3-ASR-1.7B/qwen)")
    p.add_argument("--r2t2", required=True, help="R2T2 HF bf16 directory (…/R2T2/bf16)")
    args = p.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)

    print(f"loading qwen3asr from {args.qwen3}")
    qwen = load_model(args.qwen3)
    qwen_tower_state = clone_state_dict(qwen.model.thinker.audio_tower)
    for clip in CLIPS:
        run_clip(qwen, "qwen3asr", clip, args.qwen3, capture_mel_encoder=True)
    del qwen

    print(f"loading r2t2 from {args.r2t2}")
    r2t2 = load_model(args.r2t2)
    assert_towers_equal(qwen_tower_state, clone_state_dict(r2t2.model.thinker.audio_tower))
    print("r2t2 audio tower tensors are bit-identical to qwen3asr's")
    for clip in CLIPS:
        run_clip(r2t2, "r2t2", clip, args.r2t2, capture_mel_encoder=False)
        copy_mel_encoder(clip)
    del r2t2

    check_speech_en_golden()


if __name__ == "__main__":
    main()
