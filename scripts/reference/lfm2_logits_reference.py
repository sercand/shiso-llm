#!/usr/bin/env python3
"""Capture independent next-token references for the LFM2 decision checkpoint.

Runs the checkpoint's own forward (`LiquidAI/d1-3B`, loaded through its
`trust_remote_code` entry point, in float32 on the CPU) over a small fixed set
of prompts and records, at each prompt's last position, the top-64 log-
probabilities of the full vocabulary and the log-probabilities of every token a
decision readout scores (`yes`/`Yes`/`YES`, `no`/`No`/`NO`, `A`..`J`, ` A`..` J`,
`0`..`9`).

The prompts are of three kinds: a plain chat turn; the three decision prompts
the checkpoint's own `SystemOne.render` builds for the model card's example
(one per question type), so the numbers a decision is read from are graded
directly; and the two long corpus prompts under test-vectors/prompts/, which
cross several prefill chunks and so carry the convolution window and the
attention cache across chunk boundaries.

Ids are the checkpoint tokenizer's, written into the fixture so a replay needs
no tokenizer. It never overwrites a fixture.

    lfm2_logits_reference.py --model-dir \\
        ~/.cache/huggingface/hub/models--LiquidAI--d1-3B/snapshots/051bcc464b01b9f92942b364d9586b0ef5912432
"""
import argparse, json, sys
from pathlib import Path

import torch
import transformers

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "test-vectors" / "references" / "d1" / "lfm2-logits.json"
PROMPTS = REPO / "test-vectors" / "prompts"
TOP = 64

QUESTIONS = {
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
}
STATE = "I was charged twice this month, please refund one of them."
READOUT = ["yes", "Yes", "YES", "no", "No", "NO"] + [chr(c) for c in range(ord("A"), ord("K"))] \
    + [" " + chr(c) for c in range(ord("A"), ord("K"))] + [str(d) for d in range(10)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    args = ap.parse_args()
    if OUT.exists():
        print(f"{OUT} exists; delete it to recapture", file=sys.stderr)
        return 1
    model = transformers.AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, dtype=torch.float32).eval()
    tok = model.engine.tokenizer
    engine = model.engine
    cases = []
    chat = "<|startoftext|><|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n"
    texts = [("chat_capital", chat)]
    from importlib import import_module
    prompt = import_module(type(engine).__module__.rsplit(".", 1)[0] + ".prompt")
    for name, q in QUESTIONS.items():
        texts.append((f"decision_{name}", engine.render(STATE, prompt.as_question(q))))
    for name in ("long_code_audit", "long_memory_archive"):
        body = (PROMPTS / f"{name}.txt").read_text()
        texts.append((name, f"<|startoftext|><|im_start|>user\n{body}<|im_end|>\n<|im_start|>assistant\n"))
    readout_ids = {}
    for text in READOUT:
        enc = tok.encode(text, add_special_tokens=False)
        if len(enc) == 1:
            readout_ids[text] = enc[0]
    with torch.inference_mode():
        for name, text in texts:
            ids = tok.encode(text, add_special_tokens=False)
            row = model(input_ids=torch.tensor([ids]), logits_to_keep=1).logits[0, -1].double()
            logz = row - torch.logsumexp(row, dim=-1)
            vals, idx = torch.topk(logz, TOP)
            cases.append({
                "name": name,
                "text": text,
                "ids": ids,
                "top": [[int(i), float(v)] for i, v in zip(idx, vals)],
                "readout": {t: [i, float(logz[i])] for t, i in readout_ids.items()},
            })
            print(name, len(ids), "argmax", int(idx[0]), repr(tok.decode([int(idx[0])])), file=sys.stderr)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "model": "LiquidAI/d1-3B",
        "revision": Path(args.model_dir).name,
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "dtype": "float32",
        "cases": cases,
    }, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
