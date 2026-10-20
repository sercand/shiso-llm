#!/usr/bin/env python3
"""Capture a logprob `.vec`, a greedy lock and media answers from a running vLLM server.

The vLLM sibling of `llama_server_vec.py`: the independent reference for a
checkpoint only a vLLM build can run (the NVIDIA modelopt mixed-precision
Qwen3.8-Flash-Next export). An existing fixture's cases are re-asked of the
server and written as a new `.vec` under `test-vectors/references/`.

Tokens. Each prompt is tokenized by the server (`/tokenize`, no special tokens)
and its FNV digest computed exactly as `shiso_bench::run::tokens_digest` does.
When the skeleton recorded a digest for the case it must match; the ids, not
the text, are what is sent. The server must run with `--max-logprobs 64` or
more, and vLLM's top-k is taken over raw log-probabilities
(`logprobs_mode` default), the same values the other references record.

Besides the `.vec`, `--greedy-out` records the first case's 64 greedy ids
(the `*-greedy-64.json` shape), and `--media-out` records greedy chat answers
for a text, an image and a video question, so a port's media path can be read
against the reference's.

Stdlib only. Refuses to overwrite its outputs.
"""

import argparse
import base64
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
                    {"id": cid, "kind": kind, "ctx": int(ctx), "frontier": int(frontier),
                     "prompt": prompt, "top": int(top)}
                )
    return headers, cases


def data_url(path: str, mime: str) -> str:
    with open(path, "rb") as f:
        return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"


def chat(server: str, model: str, content, max_tokens: int) -> dict:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    answer = post(server, "/v1/chat/completions", body)
    return {"text": answer["choices"][0]["message"]["content"], "usage": answer["usage"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model", required=True, help="served model name")
    ap.add_argument("--skeleton", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--greedy-out")
    ap.add_argument("--media-out")
    ap.add_argument("--image")
    ap.add_argument("--video")
    args = ap.parse_args()
    for path in (args.out, args.greedy_out, args.media_out):
        if path and os.path.exists(path):
            print(f"refusing to overwrite {path}", file=sys.stderr)
            return 1
    headers, cases = read_skeleton(args.skeleton)
    base = os.path.dirname(os.path.abspath(args.skeleton))
    with open(args.tokenizer, "rb") as f:
        vocabulary_digest = fnv(f.read())
    prompts = b""
    lines, digests, first_ids = [], {}, None
    for case in cases:
        with open(os.path.join(base, case["prompt"]), "rb") as f:
            text = f.read()
        prompts += case["id"].encode() + text
        ids = post(args.server, "/tokenize",
                   {"model": args.model, "prompt": text.decode("utf-8"),
                    "add_special_tokens": False})["tokens"]
        digest = fnv(b"".join(i.to_bytes(4, "little") for i in ids))
        want = headers.get(f"tokens_digest_{case['id']}")
        if want is not None and digest != want:
            print(f"{case['id']}: the server tokenized to {digest}, the skeleton recorded {want}",
                  file=sys.stderr)
            return 1
        if len(ids) != case["frontier"]:
            print(f"{case['id']}: {len(ids)} tokens, frontier {case['frontier']}", file=sys.stderr)
            return 1
        digests[case["id"]] = digest
        first_ids = first_ids or ids
        answer = post(args.server, "/v1/completions",
                      {"model": args.model, "prompt": ids, "max_tokens": 1,
                       "logprobs": case["top"], "temperature": 0, "return_tokens_as_token_ids": True})
        top = answer["choices"][0]["logprobs"]["top_logprobs"][0]
        tops = sorted(((int(k.split(":", 1)[1]), v) for k, v in top.items()), key=lambda e: -e[1])
        if len(tops) < case["top"]:
            print(f"{case['id']}: {len(tops)} entries, wanted {case['top']}", file=sys.stderr)
            return 1
        tops = tops[: case["top"]]
        lines.append(f"case {case['id']} {case['kind']} {case['ctx']} {case['frontier']} "
                     f"{case['prompt']} {case['top']}")
        for rank, (tid, lp) in enumerate(tops):
            lines.append(f"top {rank} {tid} {lp:.9g}")
        print(f"{case['id']}: top-1 {tops[0][0]} {tops[0][1]:.4f}", file=sys.stderr)
    out = [
        "# shiso-local-logit-vectors-v1",
        "# captured_from reference",
        f"# reference {args.reference}",
        "# values logprob",
        f"# model_file {args.model}",
        f"# vocabulary_digest {vocabulary_digest}",
        f"# prompts_digest {fnv(prompts)}",
        f"# label {args.label}",
    ]
    for case in cases:
        out.append(f"# tokens_digest_{case['id']} {digests[case['id']]}")
    with open(args.out, "x") as f:
        f.write("\n".join(out + lines) + "\n")

    if args.greedy_out:
        answer = post(args.server, "/v1/completions",
                      {"model": args.model, "prompt": first_ids, "max_tokens": 64,
                       "temperature": 0, "logprobs": 0, "return_tokens_as_token_ids": True})
        choice = answer["choices"][0]
        tokens = [int(t.split(":", 1)[1]) for t in choice["logprobs"]["tokens"]]
        with open(os.path.join(base, cases[0]["prompt"])) as f:
            prompt_text = f.read()
        with open(args.greedy_out, "x") as f:
            json.dump({"prompt": prompt_text, "prompt_tokens": first_ids, "text": choice["text"],
                       "tokens": tokens, "reference": args.reference}, f, indent=2)

    if args.media_out:
        media = {"reference": args.reference}
        media["text"] = chat(args.server, args.model,
                             "In two sentences: what is a hyper-connection in a transformer?", 160)
        if args.image:
            media["image"] = chat(args.server, args.model, [
                {"type": "image_url", "image_url": {"url": data_url(args.image, "image/png")}},
                {"type": "text", "text": "Describe this image in detail."}], 256)
        if args.video:
            media["video"] = chat(args.server, args.model, [
                {"type": "video_url", "video_url": {"url": data_url(args.video, "video/mp4")}},
                {"type": "text", "text": "Describe what happens in this video."}], 256)
        with open(args.media_out, "x") as f:
            json.dump(media, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
