#!/usr/bin/env python3
"""Server-level latency of /v1/systemone by state length and question count.

One request at a time, as the Decision Index runner sends them. Each cell is
repeated three times and the minimum kept: the first request of a new state
pays nothing extra, so the minimum is the steady cost, not a warm-up artefact.
"""
import argparse, json, random, time
import httpx

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()


def request(client, words, questions, salt):
    rng = random.Random(salt)
    state = f"[{salt}] " + " ".join(rng.choice(WORDS) for _ in range(words))
    qs = {f"q{i}": {"type": "noul", "instructions": f"Does the text mention word {i}?"} for i in range(questions)}
    started = time.perf_counter()
    reply = client.post("/v1/systemone", json={"state": state, "questions": qs})
    reply.raise_for_status()
    return (time.perf_counter() - started) * 1000, reply.json()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    client = httpx.Client(base_url=args.url, timeout=600)
    cells = []
    for words in (20, 1000, 4000):
        for questions in (1, 8, 32, 64):
            runs = [request(client, words, questions, f"{words}-{questions}-{k}-{time.time()}") for k in range(3)]
            ms = min(r[0] for r in runs)
            cells.append({"words": words, "questions": questions, "ms": round(ms), "per_question_ms": round(ms / questions)})
            print(json.dumps(cells[-1]), flush=True)
    json.dump(cells, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
