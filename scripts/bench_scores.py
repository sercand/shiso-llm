#!/usr/bin/env python3
"""Server-level latency of /v1/systemone by state length and question count.

One request at a time, as the Decision Index runner sends them. Each cell is
repeated three times and the minimum kept: the first request of a new state
pays nothing extra, so the minimum is the steady cost, not a warm-up artefact.
"""
import argparse
import json
import random
import time

import httpx

WORDS = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike".split()


def request(client, words, questions, salt):
    rng = random.Random(salt)
    state = f"[{salt}] " + " ".join(rng.choice(WORDS) for _ in range(words))
    qs = {f"q{i}": {"type": "noul", "instructions": f"Does the text mention word {i}?"} for i in range(questions)}
    started = time.perf_counter()
    reply = client.post("/v1/systemone", json={"state": state, "questions": qs})
    reply.raise_for_status()
    elapsed = (time.perf_counter() - started) * 1000
    body = reply.json()
    answers = body.get("answers", {})
    if len(answers) != questions:
        raise SystemExit(f"{words} words x {questions} questions: got {len(answers)} answers")
    for name, answer in answers.items():
        if not answer.get("probabilities"):
            raise SystemExit(f"{words} words x {questions} questions: {name} carries no probabilities: {answer}")
    return elapsed, body


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8899")
    ap.add_argument("--out", required=True)
    ap.add_argument("--words", type=int, nargs="+", default=[20, 1000, 4000])
    ap.add_argument("--questions", type=int, nargs="+", default=[1, 8, 32, 64])
    args = ap.parse_args()
    client = httpx.Client(base_url=args.url, timeout=600)
    cells = []
    for words in args.words:
        for questions in args.questions:
            runs = [request(client, words, questions, f"{words}-{questions}-{k}-{time.time()}") for k in range(3)]
            ms = min(r[0] for r in runs)
            cells.append({"words": words, "questions": questions, "ms": round(ms), "per_question_ms": round(ms / questions)})
            print(json.dumps(cells[-1]), flush=True)
    with open(args.out, "w") as out:
        json.dump(cells, out, indent=1)


if __name__ == "__main__":
    main()
