"""Build a draft-vocabulary counts file focused on English, Turkish and code, and grade it.

usage: count.py build --data DIR --tokenizer tokenizer.json --code-root DIR [--code-root DIR ...]
                      --out counts.txt [--compare NAME=PATH ...]
Counts token ids per source with the model's tokenizer, mixes the sources (one third
per language, half of each third the model's own answers), pins the byte-fallback and
special-token blocks, writes `id count` lines (the format shiso's --draft-vocab reads)
under 1 MiB, then reports held-out coverage of the 65,536 ids shiso keeps for this file
and, with --compare, for other files (a counts file or a plain id list).
See README.md beside this script for the data it expects.
"""
import argparse, collections, glob, json, os, random, sys

import pyarrow.parquet as pq
from tokenizers import Tokenizer

VOCAB = 248320           # the model's padded vocabulary (config vocab_size)
BLOCK = 128              # shiso headslice::BLOCK
KEEP_IDS = 65536         # shiso headslice::DEFAULT_IDS
MAX_BYTES = 1_000_000    # under the hub's 1 MiB limit with room for the header
CODE_EXT = {".rs", ".py", ".ts", ".tsx", ".js", ".go", ".c", ".cc", ".cpp", ".h", ".hpp",
            ".cu", ".cuh", ".java", ".sh", ".sql", ".toml", ".yaml", ".yml"}
SKIP_DIRS = {".git", "target", "node_modules", ".venv", "venv", "__pycache__", "build", "dist",
             "third_party", "vendor", ".superpowers", "site-packages"}


def parquet_texts(path, column, max_bytes, skip_bytes=0):
    """Texts from a parquet file's rows, after skipping `skip_bytes` of text, up to `max_bytes`."""
    seen = got = 0
    pf = pq.ParquetFile(path)
    for rg in range(pf.num_row_groups):
        for text in pf.read_row_group(rg, columns=[column]).column(column).to_pylist():
            if not text:
                continue
            n = len(text.encode())
            seen += n
            if seen <= skip_bytes:
                continue
            yield text
            got += n
            if got >= max_bytes:
                return


def code_texts(roots, max_bytes, holdout):
    """Source files from the local repositories, shuffled with a fixed seed; every 10th
    file is held out (holdout=True yields only those)."""
    files = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(os.path.expanduser(root)):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for fn in filenames:
                if os.path.splitext(fn)[1] in CODE_EXT:
                    files.append(os.path.join(dirpath, fn))
    files.sort()
    random.Random(7).shuffle(files)
    got = 0
    for i, path in enumerate(files):
        if (i % 10 == 0) != holdout:
            continue
        try:
            if os.path.getsize(path) > 512_000:
                continue
            text = open(path, encoding="utf-8", errors="ignore").read()
        except OSError:
            continue
        yield text
        got += len(text.encode())
        if got >= max_bytes:
            return


def model_texts(path, group, holdout):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r["group"] == group]
    for i, r in enumerate(rows):
        if (i % 5 == 0) == holdout:      # every 5th answer is held out
            yield r["text"]


def count(tok, texts):
    c = collections.Counter()
    batch = []
    for t in texts:
        batch.append(t)
        if len(batch) == 64:
            for enc in tok.encode_batch(batch, add_special_tokens=False):
                c.update(enc.ids)
            batch = []
    for enc in tok.encode_batch(batch, add_special_tokens=False):
        c.update(enc.ids)
    return c


def normalized(c):
    total = sum(c.values()) or 1
    return {i: n / total for i, n in c.items()}


def kept_blocks(mass_by_id):
    """shiso's headslice::blocks_from_counts: the KEEP_IDS/BLOCK blocks of most mass."""
    total = VOCAB // BLOCK
    mass = [0.0] * total
    for i, m in mass_by_id.items():
        if 0 <= i < VOCAB:
            mass[i // BLOCK] += m
    order = sorted(range(total), key=lambda b: (-mass[b], b))
    return set(order[: KEEP_IDS // BLOCK])


def read_counts(path):
    out = {}
    for line in open(path):
        if line.startswith("#"):
            continue
        f = line.split()
        if f and f[0].isdigit():
            out[int(f[0])] = float(f[1]) if len(f) > 1 else 1.0
    return out


def coverage(blocks, c):
    total = sum(c.values()) or 1
    return sum(n for i, n in c.items() if i // BLOCK in blocks) / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--code-root", action="append", required=True,
                    help="a source tree to sample code from; repeatable")
    ap.add_argument("--compare", action="append", default=[], help="NAME=PATH, repeatable")
    ap.add_argument("--mb", type=int, default=60, help="MB of text per web/code source")
    a = ap.parse_args()
    tok = Tokenizer.from_file(a.tokenizer)
    mb = a.mb * 1_000_000
    d = a.data
    fw2 = glob.glob(os.path.join(d, "fw2/data/tur_Latn/train/*.parquet"))[0]
    mo = os.path.join(d, "model_outputs.jsonl")

    print("counting...", file=sys.stderr)
    train = {
        "wiki_en": count(tok, parquet_texts(os.path.join(d, "wiki_en.parquet"), "text", mb)),
        "wiki_tr": count(tok, parquet_texts(os.path.join(d, "wiki_tr.parquet"), "text", mb // 2)),
        "fw2_tr": count(tok, parquet_texts(fw2, "text", mb // 2)),
        "code": count(tok, code_texts(a.code_root, mb, holdout=False)),
        "model_en": count(tok, model_texts(mo, "en", holdout=False)),
        "model_tr": count(tok, model_texts(mo, "tr", holdout=False)),
        "model_code": count(tok, model_texts(mo, "code", holdout=False)),
    }
    held = {
        "English (wiki)": count(tok, parquet_texts(os.path.join(d, "wiki_en.parquet"), "text", 5_000_000, skip_bytes=mb + 1)),
        "Turkish (wiki)": count(tok, parquet_texts(os.path.join(d, "wiki_tr.parquet"), "text", 5_000_000, skip_bytes=mb // 2 + 1)),
        "Turkish (web)": count(tok, parquet_texts(fw2, "text", 5_000_000, skip_bytes=mb // 2 + 1)),
        "code (local repos)": count(tok, code_texts(a.code_root, 5_000_000, holdout=True)),
        "model answers, en": count(tok, model_texts(mo, "en", holdout=True)),
        "model answers, tr": count(tok, model_texts(mo, "tr", holdout=True)),
        "model answers, code": count(tok, model_texts(mo, "code", holdout=True)),
    }
    for k, v in train.items():
        print(f"  train {k:12s} {sum(v.values()):>12,} tokens {len(v):>7,} ids", file=sys.stderr)

    # One third per language, half of each third the model's own answers.
    weights = {"model_en": 1 / 6, "wiki_en": 1 / 6,
               "model_tr": 1 / 6, "wiki_tr": 1 / 12, "fw2_tr": 1 / 12,
               "model_code": 1 / 6, "code": 1 / 6}
    mix = collections.defaultdict(float)
    for src, w in weights.items():
        for i, p in normalized(train[src]).items():
            mix[i] += w * p
    # Pin the byte-fallback range and the special/added tokens: a draft that cannot
    # propose them is wrong exactly at structure and at non-ASCII bytes.
    special = [i for i in range(248044, VOCAB)]
    top = max(mix.values())
    for i in list(range(512)) + special:
        mix[i] = max(mix.get(i, 0.0), top)

    scale = 1e9
    rows = sorted(((i, m) for i, m in mix.items() if 0 <= i < VOCAB and m * scale >= 1),
                  key=lambda x: (-x[1], x[0]))
    header = ("# qwen3.8-flash draft vocabulary counts (English, Turkish, code): "
              f"weights {weights}; ids pinned: 0-511 and {special[0]}-{VOCAB - 1}; "
              f"mass per 1e9\n")
    out, size = [header], len(header.encode())
    for i, m in rows:
        line = f"{i} {int(round(m * scale))}\n"
        if size + len(line) > MAX_BYTES:
            break
        out.append(line)
        size += len(line)
    open(a.out, "w").write("".join(out))
    print(f"wrote {len(out) - 1:,} ids, {size:,} bytes -> {a.out}")

    files = {"new": read_counts(a.out)}
    for spec in a.compare:
        name, path = spec.split("=", 1)
        files[name] = read_counts(path)
    blocks = {name: kept_blocks(normalized(c)) for name, c in files.items()}
    for name in files:
        if name != "new":
            print(f"blocks shared by {name} and new: {len(blocks['new'] & blocks[name])} of {KEEP_IDS // BLOCK}")
        # How much of a plain id list survives shiso's block narrowing.
        ids = [i for i in files[name] if 0 <= i < VOCAB]
        kept = sum(1 for i in ids if i // BLOCK in blocks[name])
        print(f"  {name}: {len(ids):,} ids listed, {kept:,} inside its kept blocks")
    print(f"{'held-out set':24s}" + "".join(f"{n:>10s}" for n in blocks))
    for name, c in held.items():
        print(f"{name:24s}" + "".join(f"{100 * coverage(b, c):9.2f}%" for b in blocks.values()))


if __name__ == "__main__":
    main()
