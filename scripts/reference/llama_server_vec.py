#!/usr/bin/env python3
"""Capture a logprob `.vec` from a running llama-server.

The independent reference for a GGUF this engine reads: an existing fixture's
cases (prompt files, context, frontier, top count) are re-asked of a
llama-server that has the checkpoint loaded, and the answers are written as a
new `.vec` in `test-vectors/references/`.

Tokens. Each prompt is tokenized by the server (`/tokenize`, no special
tokens added), its FNV digest is computed exactly as `shiso_bench::run::
tokens_digest` computes it, and it must equal the digest the skeleton fixture
recorded for that case - so both engines are graded on the same ids, and a
tokenizer divergence is refused here rather than measured as a kernel defect.
The ids, not the text, are what is sent for completion.

The request is the one `test-vectors/README.md` describes for every llama.cpp
capture: `n_predict: 1`, `n_probs: <top>`, `temperature: 0`,
`cache_prompt: false`, `post_sampling_probs: false`. The values are the
server's log-probabilities over the whole row (`# values logprob`).

Stdlib only. Refuses to overwrite its output.

    python3 scripts/reference/llama_server_vec.py \
        --server http://127.0.0.1:8799 \
        --skeleton test-vectors/references/qwen38-27b-mlx-f32-golden.vec \
        --tokenizer <the checkpoint's tokenizer.json> \
        --model-file <gguf> --reference "<fork> <commit>" --label "<machine>" \
        --out test-vectors/references/<name>.vec
"""

import argparse
import json
import os
import sys
import urllib.request


def fnv(data: bytes) -> str:
    """`shiso_bench::replay::fnv_digest`: FNV-1a 64, hex."""
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def post(server: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(
        server + path,
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=3600) as resp:
        return json.load(resp)


def read_skeleton(path: str):
    """Headers and `case` lines of an existing fixture."""
    headers, cases = {}, []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# ") and " " in line[2:]:
                key, value = line[2:].split(" ", 1)
                headers[key] = value
            elif line.startswith("case "):
                _, cid, kind, ctx, frontier, prompt, top = line.split(" ")
                cases.append(
                    {
                        "id": cid,
                        "kind": kind,
                        "ctx": int(ctx),
                        "frontier": int(frontier),
                        "prompt": prompt,
                        "top": int(top),
                    }
                )
    return headers, cases


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if os.path.exists(args.out):
        print(f"refusing to overwrite {args.out}", file=sys.stderr)
        return 1
    headers, cases = read_skeleton(args.skeleton)
    base = os.path.dirname(os.path.abspath(args.skeleton))
    with open(args.tokenizer, "rb") as f:
        vocabulary_digest = fnv(f.read())
    prompts = b""
    lines = []
    for case in cases:
        with open(os.path.join(base, case["prompt"]), "rb") as f:
            text = f.read()
        prompts += case["id"].encode() + text
        ids = post(
            args.server,
            "/tokenize",
            {"content": text.decode("utf-8"), "add_special": False},
        )["tokens"]
        digest = fnv(b"".join(i.to_bytes(4, "little") for i in ids))
        want = headers.get(f"tokens_digest_{case['id']}")
        if digest != want:
            print(
                f"{case['id']}: the server tokenized to {digest}, the skeleton recorded {want}",
                file=sys.stderr,
            )
            return 1
        if len(ids) != case["frontier"]:
            print(f"{case['id']}: {len(ids)} tokens, frontier {case['frontier']}", file=sys.stderr)
            return 1
        answer = post(
            args.server,
            "/completion",
            {
                "prompt": ids,
                "n_predict": 1,
                "n_probs": case["top"],
                "temperature": 0,
                "cache_prompt": False,
                "post_sampling_probs": False,
            },
        )
        tops = answer["completion_probabilities"][0]["top_logprobs"]
        if len(tops) != case["top"]:
            print(f"{case['id']}: {len(tops)} entries, wanted {case['top']}", file=sys.stderr)
            return 1
        lines.append(
            f"case {case['id']} {case['kind']} {case['ctx']} {case['frontier']} "
            f"{case['prompt']} {case['top']}"
        )
        for rank, entry in enumerate(tops):
            lines.append(f"top {rank} {entry['id']} {entry['logprob']:.9g}")
        print(f"{case['id']}: top-1 {tops[0]['id']} {tops[0]['logprob']:.4f}", file=sys.stderr)
    out = [
        "# shiso-local-logit-vectors-v1",
        "# captured_from reference",
        f"# reference {args.reference}",
        "# values logprob",
        f"# model_file {os.path.basename(args.model_file)}",
        f"# vocabulary_digest {vocabulary_digest}",
        f"# prompts_digest {fnv(prompts)}",
        f"# label {args.label}",
    ]
    for case in cases:
        key = f"tokens_digest_{case['id']}"
        out.append(f"# {key} {headers[key]}")
    with open(args.out, "x") as f:
        f.write("\n".join(out + lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
