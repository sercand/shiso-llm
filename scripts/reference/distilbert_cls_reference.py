#!/usr/bin/env python3
"""ONNX Runtime goldens for the bash-command classifier.

Tokenizes each corpus command exactly like the shipped classifier does (encode
with special tokens, keep the first 128 ids, pad to 128 with id 0 / mask 0),
runs the original int8 ONNX graph on CPU and writes one row per command:
`{"label", "text", "ids", "logits", "decision"}`. `decision` applies the
calibration outside the model (temperature scaling, energy gate, then the
0.90 safe-probability rule), so the converted weights can be graded on the
decision as well as on the logits.

Usage:
    uv run --with onnxruntime --with tokenizers --with numpy \
        scripts/reference/distilbert_cls_reference.py \
        --onnx model.onnx --tokenizer tokenizer.json \
        --corpus scripts/reference/distilbert_cls_corpus.txt --out goldens.json
"""

import argparse
import json

import numpy as np

MAX_LENGTH = 128
TEMPERATURE = 1.3366203308105469
ENERGY_P95 = -2.543365240097046
SAFE_THRESHOLD = 0.90


def decide(logits):
    """Calibrated decision for one `[3]` logit vector: approve, defer or deny."""
    x = np.asarray(logits, dtype=np.float32) / np.float32(TEMPERATURE)
    top = x.max()
    sum_exp = np.exp(x - top).sum()
    energy = -(np.float32(TEMPERATURE) * (top + np.log(sum_exp)))
    if energy > ENERGY_P95:
        return "defer"
    probs = np.exp(x - top) / sum_exp
    idx = int(np.argmax(probs))
    if idx == 0 and probs[idx] > SAFE_THRESHOLD:
        return "approve"
    if idx == 2:
        return "deny"
    return "defer"


def read_corpus(path):
    """Rows `(label, text)` from a `label<TAB>text` file; `#` lines are comments.

    Text of the form `json:<JSON string>` is decoded, so empty and whitespace-only
    commands can be written without trailing whitespace in the file.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("#") or not line:
                continue
            label, _, text = line.partition("\t")
            if text.startswith("json:"):
                text = json.loads(text[5:])
            rows.append((label, text))
    return rows


def encode(tok, text):
    """First 128 real ids of the special-token encoding, as `(ids, mask)`.

    The shipped tokenizer file enables fixed-length padding, so the raw encoding
    carries pad ids whose attention mask is 0; the classifier copies ids and mask
    together for the first 128 positions, so keeping the mask-1 ids is identical.
    """
    enc = tok.encode(text, add_special_tokens=True)
    ids, mask = enc.ids[:MAX_LENGTH], enc.attention_mask[:MAX_LENGTH]
    return [i for i, m in zip(ids, mask) if m]


def main():
    import onnxruntime as ort
    from tokenizers import Tokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    tok = Tokenizer.from_file(a.tokenizer)
    sess = ort.InferenceSession(a.onnx, providers=["CPUExecutionProvider"])
    out = []
    for label, text in read_corpus(a.corpus):
        ids = encode(tok, text)
        input_ids = np.zeros((1, MAX_LENGTH), dtype=np.int64)
        mask = np.zeros((1, MAX_LENGTH), dtype=np.int64)
        input_ids[0, : len(ids)] = ids
        mask[0, : len(ids)] = 1
        (logits,) = sess.run(None, {"input_ids": input_ids, "attention_mask": mask})
        logits = [float(v) for v in logits[0]]
        out.append(
            {
                "label": label,
                "text": text,
                "ids": ids,
                "logits": logits,
                "decision": decide(logits),
            }
        )
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)
    dist = {}
    for r in out:
        dist[r["decision"]] = dist.get(r["decision"], 0) + 1
    print(f"{len(out)} rows, decisions {dist}")


if __name__ == "__main__":
    main()
