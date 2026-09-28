"""Capture an independent-reference logit fixture from mlx-lm.

Writes a `shiso-local-logit-vectors-v1` file (`# values logprob`): for each case,
the top-64 log-probabilities over the whole vocabulary at one frontier, from
mlx-lm's own forward over the checkpoint. The replay in `crates/shiso-bench`
digests the prompts, the token ids and the vocabulary exactly the way this file
does, so a mismatch in any of the three is named rather than measured.

Optionally also writes a greedy trajectory lock: 64 argmax ids after the first
case's prompt.

    python scripts/reference/mlx_logit_vectors.py MODEL_DIR OUT.vec \
        --case short_italian_fact:1024:15 --case long_code_audit:4096:3957 \
        --greedy OUT-greedy-64.json

A case is `<id>:<ctx>:<frontier>`; its prompt is `test-vectors/prompts/<id>.txt`.
The script refuses to overwrite an existing output: a reference is captured
once, and a re-capture is a decision someone makes by deleting the old file.
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import mlx.core as mx
import mlx_lm
from mlx_lm import load
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
VECTORS = ROOT / "test-vectors"
TOP = 64


def fnv(data: bytes) -> str:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def tokens_digest(ids) -> str:
    return fnv(b"".join(int(t).to_bytes(4, "little") for t in ids))


def machine() -> str:
    try:
        return subprocess.check_output(["sysctl", "-n", "hw.model"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return platform.machine()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--case", action="append", required=True)
    p.add_argument("--greedy")
    p.add_argument("--leading-token", type=int)
    p.add_argument("--f32", action="store_true",
                   help="carry activations in float32 (every non-quantized parameter cast), so the "
                        "reference computes at the width this port does and is graded at REFERENCE")
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; delete it to re-capture")
    if args.greedy and Path(args.greedy).exists():
        sys.exit(f"{args.greedy} exists; delete it to re-capture")

    model_dir = Path(args.model)
    vocab_path = model_dir / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(vocab_path))
    model, _ = load(str(model_dir))
    if args.f32:
        model.set_dtype(mx.float32)

    cases = []
    for spec in args.case:
        cid, ctx, frontier = spec.split(":")
        cases.append((cid, int(ctx), int(frontier)))

    prompts = b""
    for cid, _, _ in cases:
        prompts += cid.encode() + (VECTORS / "prompts" / f"{cid}.txt").read_bytes()

    label = f"mlx-lm {mlx_lm.__version__} mlx {mx.__version__}" + (" float32" if args.f32 else "")
    head = [
        "# shiso-local-logit-vectors-v1",
        "# captured_from reference",
        f"# reference {label}",
        "# values logprob",
        f"# activations {'float32' if args.f32 else 'bfloat16'}",
    ]
    if args.leading_token is not None:
        head.append(f"# leading_token {args.leading_token}")
    head += [
        f"# model_file {model_dir.name}",
        f"# vocabulary_digest {fnv(vocab_path.read_bytes())}",
        f"# prompts_digest {fnv(prompts)}",
        f"# label {label} {machine()} metal",
    ]
    body = []
    first_ids = None
    for cid, ctx, frontier in cases:
        text = (VECTORS / "prompts" / f"{cid}.txt").read_text()
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if args.leading_token is not None:
            ids = [args.leading_token] + ids
        if len(ids) < frontier:
            sys.exit(f"{cid}: {len(ids)} tokens, frontier {frontier}")
        ids = ids[:frontier]
        if first_ids is None:
            first_ids = (text, ids)
        head.append(f"# tokens_digest_{cid} {tokens_digest(ids)}")
        logits = model(mx.array(ids)[None])[0, -1].astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits)
        order = mx.argsort(-logprobs)[:TOP]
        mx.eval(order, logprobs)
        body.append(f"case {cid} text {ctx} {frontier} ../prompts/{cid}.txt {TOP}")
        for rank, tid in enumerate(order.tolist()):
            body.append(f"top {rank} {tid} {logprobs[tid].item():.9g}")
        print(f"{cid}: top-1 {order[0].item()} {logprobs[order[0]].item():.4f}", file=sys.stderr)
    out.write_text("\n".join(head + body) + "\n")

    if args.greedy:
        text, ids = first_ids
        full = tokenizer.encode(text, add_special_tokens=False).ids
        if args.leading_token is not None:
            full = [args.leading_token] + full
        seq = list(full)
        gen = []
        for _ in range(64):
            logits = model(mx.array(seq)[None])[0, -1]
            nxt = int(mx.argmax(logits).item())
            gen.append(nxt)
            seq.append(nxt)
        record = {"prompt": text, "prompt_tokens": full, "text": tokenizer.decode(gen, skip_special_tokens=False),
                  "tokens": gen}
        if args.leading_token is not None:
            record["leading_token"] = args.leading_token
        Path(args.greedy).write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
