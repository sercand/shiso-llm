"""Collect the model's own answers (reasoning included) to the vocabulary prompts.

usage: generate.py OUT_JSONL [--rounds N] [--max-tokens M] [--url URL]
Each prompt is asked once per round at temperature 1.0 (the generation-config
default), so rounds give different samples. Writes {"group","prompt","text"} lines.
"""
import argparse, concurrent.futures as cf, json, sys, time
import requests

sys.path.insert(0, __import__("os").path.dirname(__file__))
from prompts import EN, TR, CODE  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--rounds", type=int, default=1)
ap.add_argument("--max-tokens", type=int, default=900)
ap.add_argument("--url", default="http://127.0.0.1:8899/v1/chat/completions")
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()

jobs = [(g, p) for _ in range(a.rounds) for g, ps in (("en", EN), ("tr", TR), ("code", CODE)) for p in ps]


def ask(job):
    group, prompt = job
    body = {"model": "qwen3.8-flash", "messages": [{"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "stream": False}
    r = requests.post(a.url, json=body, timeout=900)
    r.raise_for_status()
    msg = r.json()["choices"][0]["message"]
    text = (msg.get("reasoning_content") or msg.get("reasoning") or "") + "\n" + (msg.get("content") or "")
    return {"group": group, "prompt": prompt, "text": text}


t0 = time.time()
done = 0
with open(a.out, "w") as f, cf.ThreadPoolExecutor(a.workers) as ex:
    for res in ex.map(ask, jobs):
        f.write(json.dumps(res, ensure_ascii=False) + "\n")
        f.flush()
        done += 1
        if done % 10 == 0:
            print(f"{done}/{len(jobs)} in {time.time() - t0:.0f}s", flush=True)
print(f"done {done} in {time.time() - t0:.0f}s")
