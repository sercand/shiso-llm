#!/usr/bin/env python3
"""Re-ask the LFM2 float32 reference's prompts of a running llama-server.

The independent reference for one quantized GGUF: every case of
test-vectors/references/d1/lfm2-logits.json (written by
lfm2_logits_reference.py) is sent to a llama-server that has the container
loaded, by its ids, and the top-64 log-probabilities of the whole row and the
log-probabilities of the readout tokens are written to a new fixture of the
same shape. A quantized export is then graded against the same quantization
run by an implementation that shares no kernel with this one, which separates
the export's own error from this port's.

The request is the one test-vectors/README.md describes for every llama.cpp
capture: `n_predict: 1`, `n_probs: <top>`, `temperature: 0`,
`cache_prompt: false`, `post_sampling_probs: false`. Readout tokens outside the
returned top are recorded as absent rather than guessed.

Stdlib only. Refuses to overwrite its output.

    lfm2_llama_reference.py --server http://127.0.0.1:8799 \\
        --model-file d1-3B-Q4_K_M.gguf --reference "llama.cpp 3d65c90" \\
        --out test-vectors/references/d1/lfm2-q4_k_m-llama.json
"""
import argparse, json, sys, urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SOURCE = REPO / "test-vectors" / "references" / "d1" / "lfm2-logits.json"
TOP = 64


def ask(server: str, ids: list[int]) -> list[tuple[int, float]]:
    body = json.dumps({
        "prompt": ids, "n_predict": 1, "n_probs": TOP, "temperature": 0,
        "cache_prompt": False, "post_sampling_probs": False,
    }).encode()
    req = urllib.request.Request(f"{server}/completion", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        out = json.load(resp)
    top = out["completion_probabilities"][0]["top_logprobs"]
    return [(int(e["id"]), float(e["logprob"])) for e in top]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--model-file", required=True)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    if out.exists():
        print(f"{out} exists; delete it to recapture", file=sys.stderr)
        return 1
    source = json.loads(SOURCE.read_text())
    cases = []
    for case in source["cases"]:
        top = ask(args.server, case["ids"])
        have = dict(top)
        readout = {t: [i, have[i]] for t, (i, _) in case["readout"].items() if i in have}
        cases.append({"name": case["name"], "ids": case["ids"], "top": [list(e) for e in top],
                      "readout": readout})
        print(case["name"], "argmax", top[0][0], file=sys.stderr)
    out.write_text(json.dumps({"model_file": args.model_file, "reference": args.reference,
                               "cases": cases}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
