"""Capture an independent reference for a block drafter over its target.

Runs the target through mlx-lm and the drafter through its own reference
implementation (the drafter repository's `model_mlx.py`, passed with
`--drafter-module`), both with activations in float32 and the drafter's
projections quantized to 4-bit affine in groups of 64 - the form the engine
runs it in. Writes one JSON record per prompt:

- `prompt_tokens`: the ids the prompt encoded to;
- `first`: the target's greedy pick after the prompt, which the first block
  drafts behind;
- `drafts`: the drafter's greedy block behind `first`, and `candidates`: its
  top candidate ids at each drafted position, best first;
- `rounds`: a greedy speculative run of `--tokens` tokens: per round, how many
  drafts the target accepted - the acceptance a correct drafter reproduces.

    python scripts/reference/block_draft_reference.py TARGET_DIR DRAFTER_DIR OUT.json \
        --drafter-module PATH/model_mlx.py --prompt short_code_completion --tokens 64

The script refuses to overwrite an existing output.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
PROMPTS = ROOT / "test-vectors" / "prompts"


def import_module(path: Path):
    spec = importlib.util.spec_from_file_location("drafter_reference", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("target")
    p.add_argument("drafter")
    p.add_argument("out")
    p.add_argument("--drafter-module", required=True)
    p.add_argument("--prompt", action="append", required=True)
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--top", type=int, default=16)
    args = p.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"{out} exists; delete it to re-capture")

    ref = import_module(Path(args.drafter_module))
    target, _ = load(args.target)
    target.set_dtype(mx.float32)
    drafter = ref.load_draft(args.drafter)
    nn.quantize(drafter, group_size=64, bits=4,
                class_predicate=lambda path, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0)
    drafter.set_dtype(mx.float32)
    drafter.bind(target)
    ref._patch_model(target, drafter.config.target_layer_ids)
    tokenizer = Tokenizer.from_file(str(Path(args.target) / "tokenizer.json"))
    block = int(drafter.config.block_size)
    mask = int(drafter.config.mask_token_id)

    records = []
    for name in args.prompt:
        text = (PROMPTS / f"{name}.txt").read_text()
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        cache = make_prompt_cache(target)
        dcache = drafter.make_cache()
        logits = target(mx.array(ids)[None], cache)
        hidden = mx.concatenate(target._hidden_states, axis=-1)
        first = int(mx.argmax(logits[0, -1]).item())
        tokens = [first]
        record = {"prompt": name, "prompt_tokens": ids, "first": first, "block": block}
        rounds = []
        while len(tokens) < args.tokens:
            inp = mx.array([[tokens[-1]] + [mask] * (block - 1)])
            h = drafter.hidden_states(inp, hidden, dcache, logits_start=1)
            dlogits = drafter.compute_logits(h)
            drafts, candidates, _ = drafter.candidate_selector.select(h, dlogits, inp[:, 0], 0.0)
            drafts = drafts[0].tolist()
            if "drafts" not in record:
                record["drafts"] = drafts
                record["candidates"] = candidates[0].tolist()
                top = mx.argsort(-dlogits[0], axis=-1)[:, : args.top]
                record["logit_top"] = top.tolist()
            # The drafter's cache holds context it absorbed; drop anything past
            # the committed prefix, as the reference loop does.
            committed = len(ids) + len(tokens) - 1
            if (trim := dcache[0].offset - committed) > 0:
                ref._trim_recent_cache(dcache, trim)
            verify = mx.array([[tokens[-1]] + drafts])
            vlogits = target(verify, cache)
            vhidden = mx.concatenate(target._hidden_states, axis=-1)
            picks = mx.argmax(vlogits[0], axis=-1).tolist()
            accepted = next((i for i, d in enumerate(drafts) if d != picks[i]), len(drafts))
            rounds.append(accepted)
            tokens.extend(drafts[:accepted] + [picks[accepted]])
            trim = block - accepted - 1
            if trim > 0:
                # The target's recurrent state cannot be trimmed; recompute the
                # kept rows from a snapshot instead. Simplest faithful way here:
                # re-run the committed prefix. Slow, but this is a capture.
                cache = make_prompt_cache(target)
                logits = target(mx.array(ids + tokens[:-1])[None], cache)
                vhidden = mx.concatenate(target._hidden_states, axis=-1)[:, -(accepted + 1):]
            hidden = vhidden[:, : accepted + 1]
        record["rounds"] = rounds
        record["tokens"] = tokens[: args.tokens]
        record["tokens_per_round"] = sum(r + 1 for r in rounds) / len(rounds)
        print(f"{name}: first {first} drafts {record['drafts']} "
              f"{record['tokens_per_round']:.2f} tokens/round", file=sys.stderr)
        records.append(record)
    out.write_text(json.dumps({"target": Path(args.target).name, "drafter": Path(args.drafter).name,
                               "records": records}, indent=1) + "\n")


if __name__ == "__main__":
    main()
