#!/usr/bin/env python3
"""Capture independent Qwen3-ASR / R2T2 reference fixtures for shiso's speech
recognizer tests.

Runs the reference `qwen_asr` package and `transformers` in fp32 on CPU over
the fixed clips in test-vectors/references/audio/clips/ and writes fixtures
under test-vectors/references/audio/. It never overwrites an existing
fixture: one already present is kept, and deleting it deliberately is how it
is recaptured.

Qwen3-ASR and R2T2 share the same architecture (`Qwen3ASRForConditionalGeneration`)
and an audio tower that is bit-identical between the two checkpoints (the
fine-tune retrained only the decoder; assert_towers_equal checks it). So the mel and windowed-encoder fixtures are captured once from
the Qwen3-ASR run and copied byte-for-byte to the R2T2 names, after asserting
the two checkpoints' audio tower weights actually match. Prompt ids and
generated tokens differ per model (different text decoder, and R2T2 carries
its own tokenizer additions), so those are captured independently for each.

    qwen3_asr_reference.py --qwen3 ~/models/Qwen3-ASR-1.7B/qwen --r2t2 ~/models/R2T2/bf16

A separate mode loads no weights: `--context-prompt-only --tokenizer <dir>`
loads only the reference tokenizer and chat template (`transformers.AutoTokenizer`
and, when the `qwen_asr` package is importable, its own `_build_text_prompt`,
which needs a processor but not a loaded model) and writes
`qwen3asr-context-prompt.json`, the ids of the system turn holding the fixed
`CONTEXT` string and the whole prompt around its one `<|audio_pad|>`. This is
what `context_tests.rs` and `prompt_tests.rs` check the encoder and the
prompt layout against without a checkpoint's weights on hand.
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

# The forced language per clip (the processor canonicalises it to
# first-letter-upper, rest-lower, and must be one of the checkpoint's 30 supported languages).
FORCED_LANGUAGE = {
    "speech-en": "English",
    "speech-en-odd": "English",
    "two-speakers": "English",
    "weather-es": "Spanish",
}

AUDIO_PAD_ID = 151676  # `<|audio_pad|>` in tokenizer_config.json

# The context string `--context-prompt-only` renders and tokenizes. Public,
# fixed, and exercises both non-ASCII text and keywords: it is exactly
# `wording(Context { prompt: Some("A farewell address."), keywords: ["Chicago",
# "İstanbul", "Çanakkale", "Ağrı Dağı"] })` under the Rust wording
# (`qwen3_asr::context::wording`), pinned by a unit test in context_tests.rs.
CONTEXT = "A farewell address.\nKeywords: Chicago, İstanbul, Çanakkale, Ağrı Dağı"

# The windowed-encoder, auto-language golden for speech-en, from an earlier
# reference run. It is the windowed one (matches vLLM/FA2), not the HF
# sdpa/eager default, which silently runs full-clip attention over the encoder
# (see the patch below).
SPEECH_EN_AUTO_IDS_GOLDEN = [
    11528, 6364, 151704, 1986, 2003, 11, 358, 30696, 311, 10557, 311, 6359, 847, 1590, 71543, 2621, 311,
    279, 6995, 13, 22713, 304, 279, 13815, 315, 46508, 1573, 752, 11, 432, 572, 458, 6638, 311, 1977, 9702,
    498, 13, 13139, 582, 3003, 3884, 7912, 311, 7912, 476, 18521, 7230, 518, 678, 11, 847, 20959, 448, 498,
    11, 279, 3693, 1251, 11, 304, 5382, 12026, 11, 304, 8682, 11, 518, 33224, 11, 323, 389, 8633, 25945, 11,
    518, 11623, 388, 323, 389, 28727, 6277, 700, 12664, 86147, 1493, 20959, 525, 1128, 614, 8604, 752, 10745,
    13, 151645,
]

# In transformers 4.57.6 the encoder layer is called as
# `encoder_layer(hidden_states, cu_seqlens)` with no attention_mask, and the
# sdpa/eager attention paths silently ignore cu_seqlens, giving full-clip
# attention instead of vLLM/FA2's 104-token windows. Patch the layer so every
# run in this script (both the standalone encoder fixture and generation)
# takes the windowed path, the golden this script checks.
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


def _rendered_with_qwen_asr(tokenizer_dir: str) -> str:
    """`CONTEXT`'s system turn and one audio item, rendered by the reference
    package's own `_build_text_prompt` over a processor built from
    `tokenizer_dir` - no model, no weights. Raises whatever loading the
    processor or building the prompt raises; the caller falls back."""
    processor = transformers.AutoProcessor.from_pretrained(tokenizer_dir, fix_mistral_regex=True)
    # `_build_text_prompt` only reads `self.processor`; passing no model
    # (backend "vllm" skips the transformers-only device/dtype lookup in
    # `__init__`) keeps this path weight-free.
    wrapper = Qwen3ASRModel(backend="vllm", model=None, processor=processor)
    return wrapper._build_text_prompt(context=CONTEXT, force_language=None)


def _rendered_with_chat_template_json(tokenizer_dir: str) -> str:
    """The same rendering read directly from the checkpoint's own
    `chat_template.json` through `jinja2`, for a directory `qwen_asr`
    cannot process (no processor registered, or its call raises)."""
    import jinja2

    doc = json.loads((Path(tokenizer_dir) / "chat_template.json").read_text())
    template = jinja2.Environment(trim_blocks=True, lstrip_blocks=True).from_string(
        doc["chat_template"]
    )
    messages = [
        {"role": "system", "content": CONTEXT},
        {"role": "user", "content": [{"type": "audio", "audio": ""}]},
    ]
    return template.render(messages=messages, add_generation_prompt=True)


def context_prompt_text(tokenizer_dir: str) -> tuple[str, str]:
    """The rendered text and which source rendered it: the reference
    package's `_build_text_prompt` when it can run with no weights, else the
    checkpoint's own `chat_template.json`."""
    try:
        return _rendered_with_qwen_asr(tokenizer_dir), "qwen_asr._build_text_prompt"
    except Exception as e:  # noqa: BLE001 - any failure falls back, by design
        print(f"qwen_asr._build_text_prompt failed ({e!r}); falling back to chat_template.json")
        return _rendered_with_chat_template_json(tokenizer_dir), "chat_template.json"


def context_ids_between(text: str, ids: list, offsets: list) -> list:
    """The ids whose span sits strictly inside the system turn: after
    `"<|im_start|>system\\n"` and before the first `"<|im_end|>"` that
    follows it, located in `text` and mapped back to `ids` through the fast
    tokenizer's offsets rather than by hardcoding token ids, so this does not
    silently agree with a wrong vocabulary."""
    prefix = "<|im_start|>system\n"
    start = text.index(prefix) + len(prefix)
    end = text.index("<|im_end|>", start)
    return [i for i, (s, e) in zip(ids, offsets) if s >= start and e <= end]


def run_context_prompt_only(tokenizer_dir: str):
    """Loads only the reference tokenizer and chat template from
    `tokenizer_dir` (no model weights) and writes
    `qwen3asr-context-prompt.json`: `CONTEXT`, the ids of its system turn, and
    the whole rendered prompt around its one `<|audio_pad|>`."""
    path = ROOT / "qwen3asr-context-prompt.json"
    if kept(path):
        return
    text, source = context_prompt_text(tokenizer_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir, fix_mistral_regex=True)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = encoded["input_ids"]
    n_audio_pad = ids.count(AUDIO_PAD_ID)
    if n_audio_pad != 1:
        sys.exit(
            f"the rendered context prompt has {n_audio_pad} <|audio_pad|> ids, expected exactly 1: {ids}"
        )
    context_ids = context_ids_between(text, ids, encoded["offset_mapping"])
    write_json(
        path,
        {
            "context": CONTEXT,
            "context_ids": context_ids,
            "ids": ids,
            "n_audio_pad": n_audio_pad,
            "template_source": source,
            **versions(),
        },
    )
    print(f"{path.name}: {len(context_ids)} context ids, {len(ids)} prompt ids, from {source}")


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
    """Fail loudly unless R2T2's audio tower is bit-identical to Qwen3-ASR's (the fine-tune's claim to share it)."""
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
            # The encoder is run per clip on input_features[:, :feature_len]
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
    if got != SPEECH_EN_AUTO_IDS_GOLDEN:
        sys.exit(
            "NEEDS_CONTEXT: qwen3asr speech-en auto tokens do not match the golden. "
            f"got {len(got)} ids, expected {len(SPEECH_EN_AUTO_IDS_GOLDEN)}.\n"
            f"got:      {got}\n"
            f"expected: {SPEECH_EN_AUTO_IDS_GOLDEN}"
        )
    print(f"speech-en auto tokens match the golden ({len(got)} ids)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qwen3", help="Qwen3-ASR HF bf16 directory (…/Qwen3-ASR-1.7B/qwen)")
    p.add_argument("--r2t2", help="R2T2 HF bf16 directory (…/R2T2/bf16)")
    p.add_argument(
        "--context-prompt-only",
        action="store_true",
        help="load only the reference tokenizer and chat template (no weights) and write "
        "qwen3asr-context-prompt.json; needs --tokenizer",
    )
    p.add_argument(
        "--tokenizer",
        help="the sidecar directory to load the tokenizer and chat template from "
        "(…/Qwen3-ASR-1.7B), for --context-prompt-only",
    )
    args = p.parse_args()

    ROOT.mkdir(parents=True, exist_ok=True)

    if args.context_prompt_only:
        if not args.tokenizer:
            p.error("--context-prompt-only needs --tokenizer")
        run_context_prompt_only(args.tokenizer)
        return

    if not args.qwen3 or not args.r2t2:
        p.error("--qwen3 and --r2t2 are required unless --context-prompt-only is given")

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
