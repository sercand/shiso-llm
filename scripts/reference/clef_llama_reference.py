#!/usr/bin/env python3
"""Capture independent references for a Clef checkpoint over a dense llama backbone.

Runs the checkpoint's own backbone (`LlamaForCausalLM` from `transformers`, the
bf16 `model.safetensors` upcast to float32 on the CPU) and the release's own
joint schema head (`JointSchemaHead` from `joint_schema_model.py` of the public
`Cloudflare/clef-flash` repository, over the checkpoint's `joint_head.safetensors`
in float32), and records:

- per prompt (a chat turn, the decision records below, and a long corpus prompt
  that crosses several prefill chunks): the ids and the top-64 next-token
  log-probabilities of the last row;
- per decision record: the spans `encode_record` lays out, the backbone's final
  normalized rows for every token (`<case>-hidden.f32`, `[tokens][hidden]`
  little-endian float32), and the head's per-option logits and probabilities.

With `--gguf`, the backbone's weights are instead the named Clef GGUF's own,
dequantized to float32 (`gguf-py`), and the fixture is `reference-gguf.json`:
the same rounding the container carries, run by an implementation that shares
no kernel with the port, so a replay against it measures the port's own error
apart from the export's. The head stays the release's float32 one.

Ids are the checkpoint tokenizer's, written into the fixture so a replay needs
no tokenizer. It never overwrites a fixture.

    clef_llama_reference.py --model-dir <snapshot of sercand/SmolLM2-135M-Instruct> \\
        --head-source <path to joint_schema_model.py> \\
        [--gguf <the Clef GGUF>] [--out test-vectors/references/clef-llama]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import transformers
from safetensors.torch import load_file

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "test-vectors" / "references" / "clef-llama"
PROMPTS = REPO / "test-vectors" / "prompts"
TOP = 64

RECORDS = {
    "support": {
        "state": {
            "message": "Hi, I was charged twice for my order #5521 last week and I want my "
            "money back. The app also crashes when I open the invoices tab.",
            "customer_tier": "gold",
        },
        "questions": {
            "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
            "team": {
                "type": "choice",
                "instructions": "Which team should handle this?",
                "criteria": {
                    "billing": "Charges, refunds, invoices",
                    "technical": "App or site faults",
                    "fraud": "Suspected unauthorised use",
                },
            },
            "urgency": {
                "type": "score",
                "instructions": "How urgent is this?",
                "criteria": ["Can wait", "Today", "Blocking the customer now"],
            },
        },
    },
    "sentiment": {
        "state": "The delivery was late again and nobody answered the phone. I am done with this shop.",
        "questions": {
            "negative": {"type": "noul", "instructions": "Is the writer unhappy?"},
            "mood": {
                "type": "choice",
                "instructions": "What is the writer's mood?",
                "criteria": {
                    "angry": None,
                    "calm": None,
                    "delighted": None,
                    "sad": None,
                    "worried": None,
                },
            },
        },
    },
}


def load_head_module(path: Path):
    spec = importlib.util.spec_from_file_location("joint_schema_model", path)
    module = importlib.util.module_from_spec(spec)
    # Its dataclasses look their module up while the class is built.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The container's block roles, by the reference module's parameter names.
GGUF_ROLES = {
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_k": "self_attn.k_proj",
    "attn_v": "self_attn.v_proj",
    "attn_output": "self_attn.o_proj",
    "post_attention_norm": "post_attention_layernorm",
    "ffn_gate": "mlp.gate_proj",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
}


def load_gguf_backbone(model, path: Path) -> None:
    """Replaces every backbone weight with the GGUF's own, dequantized."""
    import gguf

    reader = gguf.GGUFReader(path)
    state = {}
    for tensor in reader.tensors:
        name = tensor.name
        if name == "token_embd.weight":
            key = "model.embed_tokens.weight"
        elif name == "output_norm.weight":
            key = "model.norm.weight"
        elif name.startswith("blk."):
            _, layer, role, _ = name.split(".")
            key = f"model.layers.{layer}.{GGUF_ROLES[role]}.weight"
        else:
            continue
        values = gguf.quants.dequantize(tensor.data, tensor.tensor_type)
        state[key] = torch.from_numpy(values.copy()).float().reshape(model.state_dict()[key].shape)
    missing, unexpected = model.load_state_dict(state, strict=False)
    # The output projection is tied to the embedding and is not a key of its own.
    if [k for k in missing if k != "lm_head.weight"] or unexpected:
        sys.exit(f"the GGUF does not cover the backbone: {missing} {unexpected}")
    model.tie_weights()


def top_logprobs(logits: torch.Tensor) -> dict:
    logp = torch.log_softmax(logits.double(), dim=-1)
    values, ids = torch.topk(logp, TOP)
    return {"ids": ids.tolist(), "logprobs": values.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--head-source", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--gguf", type=Path)
    args = ap.parse_args()

    suffix = "-gguf" if args.gguf else ""
    manifest = args.out / f"reference{suffix}.json"
    if manifest.exists():
        sys.exit(f"refusing to overwrite {manifest}")
    args.out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(0)
    reference = load_head_module(args.head_source)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_dir)
    model = transformers.LlamaForCausalLM.from_pretrained(args.model_dir, dtype=torch.float32)
    if args.gguf:
        load_gguf_backbone(model, args.gguf)
    model.eval()
    head_config = json.loads((args.model_dir / "joint_head_config.json").read_text())
    head = reference.JointSchemaHead(**head_config)
    head.load_state_dict(load_file(args.model_dir / "joint_head.safetensors"), strict=True)
    head = head.float().eval()
    embedding = model.get_output_embeddings().weight

    cases = []
    with torch.inference_mode():
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": "What is the capital of France? Answer in one word."}],
            add_generation_prompt=True,
            tokenize=True,
        )
        chat = chat["input_ids"] if isinstance(chat, dict) or hasattr(chat, "keys") else chat
        long_text = (PROMPTS / "long_code_audit.txt").read_text()
        long_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": long_text}], add_generation_prompt=True, tokenize=True
        )
        long_ids = long_ids["input_ids"] if hasattr(long_ids, "keys") else long_ids
        for name, ids in (("chat", list(chat)), ("long_code_audit", list(long_ids))):
            input_ids = torch.tensor([ids])
            logits = model(input_ids=input_ids).logits[0, -1]
            cases.append({"name": name, "ids": ids, "top": top_logprobs(logits)})

        for name, record in RECORDS.items():
            encoded = reference.encode_record(tokenizer, record)
            ids = list(encoded.input_ids)
            input_ids = torch.tensor([ids])
            mask = torch.ones_like(input_ids)
            hidden = model.model(input_ids=input_ids, attention_mask=mask).last_hidden_state
            logits = model.lm_head(hidden)[0, -1]
            head_logits = head(hidden, input_ids, mask, [encoded], embedding)[0]
            rows = hidden[0].contiguous().numpy().astype("<f4")
            (args.out / f"{name}{suffix}-hidden.f32").write_bytes(rows.tobytes())
            cases.append(
                {
                    "name": name,
                    "ids": ids,
                    "top": top_logprobs(logits),
                    "hidden_file": f"{name}{suffix}-hidden.f32",
                    "hidden_width": int(rows.shape[1]),
                    "record": record,
                    "questions": [
                        {
                            "id": q.question_id,
                            "kind": q.question_type,
                            "span": list(q.question_span),
                            "options": [list(s) for s in q.option_spans],
                            "option_ids": list(q.option_ids),
                            "logits": logits_q.tolist(),
                            "probabilities": logits_q.float().softmax(-1).tolist(),
                        }
                        for q, logits_q in zip(encoded.questions, head_logits)
                    ],
                }
            )

    manifest.write_text(
        json.dumps(
            {
                "model_dir": str(args.model_dir),
                "head_source": str(args.head_source),
                "backbone_weights": str(args.gguf) if args.gguf else "model.safetensors",
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "dtype": "float32 (bf16 weights upcast)",
                "cases": cases,
            },
            indent=1,
        )
    )
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
