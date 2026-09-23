#!/usr/bin/env python3
"""Render the qwen4exp chat fixtures with the reference chat-template implementation.

For every <case>.request.json with an "hf" block, writes <case>.txt (the rendered
prompt, byte for byte) and <case>.ids (one token id per line, no special tokens
added beyond what the template wrote). Refuses to overwrite unless --force: a fixture
that silently changes is a reference that silently changes.

    SHISO_REF_PYTHON=~/work/exllamav3/.venv/bin/python
    $SHISO_REF_PYTHON scripts/reference/render_chat_fixtures.py \
        --sidecar ~/work/qwen-3.8-flash/sidecar
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

FIXTURES = Path(__file__).resolve().parents[2] / "test-vectors/references/chat/qwen4exp"


def render(tok, hf: dict) -> tuple[str, list[int]]:
    text = tok.apply_chat_template(
        hf["messages"],
        tools=hf.get("tools") or None,
        tokenize=False,
        add_generation_prompt=True,
        **hf.get("kwargs", {}),
    )
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return text, ids


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sidecar", required=True, type=Path)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(str(args.sidecar))
    wrote = 0
    for request in sorted(FIXTURES.glob("*.request.json")):
        case = json.loads(request.read_text(encoding="utf-8"))
        if "hf" not in case:
            continue
        stem = request.name.removesuffix(".request.json")
        txt, ids_path = FIXTURES / f"{stem}.txt", FIXTURES / f"{stem}.ids"
        if not args.force and (txt.exists() or ids_path.exists()):
            print(f"refusing to overwrite {stem} (pass --force)", file=sys.stderr)
            return 1
        text, ids = render(tok, case["hf"])
        txt.write_text(text, encoding="utf-8", newline="")
        ids_path.write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
        wrote += 1
        print(f"{stem}: {len(text)} bytes, {len(ids)} tokens")
    print(f"rendered {wrote} fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
