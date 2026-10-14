#!/usr/bin/env python3
"""Replays an agent's /v1/decisions session against a server, in order.

The steps file is the one an agent run writes: every step's calls, each with
the request it sent (its screenshot replaced by a `<name: N bytes>` stub) and
the image file name. The screenshot is read from `--images` under that name;
a step whose file is missing there takes the next picture of `--fallback`
instead, so a session can be replayed without its own screenshots.

Prints each request's input and cached tokens and its wall time, and writes
every answer to `--out` so two runs (one server against another, or the cache
on against off) can be compared with `--compare`.

    bench_decisions_replay.py STEPS.json --base http://127.0.0.1:8080 \\
        --model qwen3.8-flash --images RUN_DIR --out new.json
    bench_decisions_replay.py --compare base.json new.json
"""

import argparse
import base64
import json
import mimetypes
import pathlib
import sys
import time
import urllib.request


def load_requests(steps_path):
    steps = json.loads(pathlib.Path(steps_path).read_text())["steps"]
    for step in steps:
        for call in step["Calls"]:
            if call.get("Request"):
                yield step["Step"], call.get("Image"), json.loads(call["Request"])


def data_uri(path):
    kind = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{kind};base64,{base64.b64encode(path.read_bytes()).decode()}"


def picture_for(image_name, images, fallback, index):
    if images and image_name and (images / image_name).is_file():
        return images / image_name
    if not fallback:
        sys.exit(f"no picture for {image_name!r}: pass --images or --fallback")
    return fallback[index % len(fallback)]


def send(base, body):
    request = urllib.request.Request(
        f"{base}/v1/decisions",
        data=json.dumps(body).encode(),
        headers={"content-type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=900) as response:
        answer = json.load(response)
    return answer, (time.time() - started) * 1000


def summary(answer):
    parts = []
    for item in answer["answers"]:
        value = item.get("choice")
        if value is None:
            value = round(item.get("probability", 0.0), 4)
        parts.append(f"{item['name']}={value}")
    return " ".join(parts)


def replay(args):
    images = pathlib.Path(args.images) if args.images else None
    fallback = sorted(pathlib.Path(args.fallback).glob("*")) if args.fallback else []
    results = []
    for index, (step, image_name, body) in enumerate(load_requests(args.steps)):
        if index >= args.first:
            break
        picture = None
        for part in body["input"][0]["content"]:
            if part["type"] == "input_image":
                picture = picture_for(image_name, images, fallback, index)
                part["image_url"] = data_uri(picture)
        if args.model:
            body["model"] = args.model
        for again in range(2 if args.repeat else 1):
            answer, ms = send(args.base, body)
            usage = answer["usage"]
            cached = usage["input_tokens_details"]["cached_tokens"]
            print(
                f"step {step:3d}{' again' if again else '      '} "
                f"input {usage['input_tokens']:6d} cached {cached:6d} {ms:8.0f} ms  "
                f"{summary(answer)}",
                flush=True,
            )
            results.append(
                {"step": step, "again": bool(again), "ms": ms, "picture": str(picture), "answer": answer}
            )
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(results, indent=1))


def probabilities(result):
    out = []
    for item in result["answer"]["answers"]:
        if "probabilities" in item:
            out.extend(p["probability"] for p in item["probabilities"])
        else:
            out.append(item["probability"])
    return out


def compare(left_path, right_path):
    left = json.loads(pathlib.Path(left_path).read_text())
    right = json.loads(pathlib.Path(right_path).read_text())
    worst, flips, ms = 0.0, 0, [0.0, 0.0]
    for a, b in zip(left, right):
        drift = max(abs(x - y) for x, y in zip(probabilities(a), probabilities(b)))
        worst = max(worst, drift)
        choices_a = [item.get("choice") for item in a["answer"]["answers"]]
        choices_b = [item.get("choice") for item in b["answer"]["answers"]]
        flips += choices_a != choices_b
        ms[0] += a["ms"]
        ms[1] += b["ms"]
        flag = "  CHOICE DIFFERS" if choices_a != choices_b else ""
        print(f"step {a['step']:3d}{' again' if a['again'] else '      '} "
              f"max|dp| {drift:.2e}  {a['ms']:8.0f} -> {b['ms']:8.0f} ms{flag}")
    print(f"worst max|dp| {worst:.3e}; choices that differ: {flips}; "
          f"total {ms[0] / 1000:.1f} s -> {ms[1] / 1000:.1f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="?", help="the agent's steps.json")
    parser.add_argument("--base", default="http://127.0.0.1:8080", help="server base URL")
    parser.add_argument("--model", help="model name to send instead of the recorded one")
    parser.add_argument("--images", help="directory holding the run's screenshots by name")
    parser.add_argument("--fallback", help="directory of pictures for steps whose screenshot is missing")
    parser.add_argument("--first", type=int, default=10**9, help="replay only the first N requests")
    parser.add_argument("--repeat", action="store_true", help="send every request twice")
    parser.add_argument("--out", help="write the answers here")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"), help="compare two --out files")
    args = parser.parse_args()
    if args.compare:
        compare(*args.compare)
    elif args.steps:
        replay(args)
    else:
        parser.error("a steps file or --compare is needed")


main()
