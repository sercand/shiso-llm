# Draft-vocabulary counts for a narrowed draft head

`--draft-vocab` (and a recipe's `draft_vocab` role) narrows the MTP draft head's
vocabulary projection to the 65,536 ids (512 blocks of 128) its counts file
weights most. That projection is most of a draft pass, so the narrowing is
worth about 8% of decode on qwen3.8-flash. A guess outside the kept ids is
just a rejected draft, never wrong output. These scripts build such a file for
chosen languages and grade it before it ships.

The published file, `draft_vocab_en_tr_code.txt` in the registry (English,
Turkish and code), was built with exactly the commands below.

## Files

- `prompts.py`: English, Turkish and code prompts whose answers form the
  in-distribution part of the corpus.
- `generate.py`: asks a running server those prompts and saves the answers,
  reasoning included. The draft head predicts reasoning too.
- `count.py`: tokenizes every source with the model's tokenizer and mixes the
  sources. Each language gets a third; within each third, half is the model's
  answers and half is text. It pins ids 0-511 (byte fallback) and the special
  tokens, then writes `id count` lines under 1 MB. Finally it reports how much
  of each held-out set falls inside the blocks shiso would keep, for the new
  file and any files passed with `--compare`.
- `bench_tr.py`: 12 held-out Turkish chat and code prompts, greedy, one at a
  time, for an A/B of two files on a served model. English and code were
  measured with the qbench harness, which is local and not in the repo; any
  fixed greedy prompt set works.

## Data

```sh
D=/path/to/vocab_data && mkdir -p $D
curl -L -o $D/wiki_en.parquet \
  "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.en/train/0000.parquet"
curl -L -o $D/wiki_tr.parquet \
  "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/refs%2Fconvert%2Fparquet/20231101.tr/train/0000.parquet"
hf download HuggingFaceFW/fineweb-2 --type dataset \
  --include "data/tur_Latn/train/000_00000.parquet" --local-dir $D/fw2
# the model's own answers: a server on :8899 serving the model (about 30 min on a GB10)
python3 scripts/draft_vocab/generate.py $D/model_outputs.jsonl --rounds 2 --max-tokens 900
```

## Build and grade

```sh
uv venv /tmp/vocab_venv && VIRTUAL_ENV=/tmp/vocab_venv uv pip install tokenizers pyarrow
/tmp/vocab_venv/bin/python scripts/draft_vocab/count.py build \
  --data $D --tokenizer ~/.shiso/models/qwen3.8-flash/exl3-405/base/tokenizer.json \
  --code-root ~/work/project-a --code-root ~/work/project-b \
  --out draft_vocab_en_tr_code.txt --mb 60 \
  --compare old=/path/to/previous_counts.txt
```

Tokenizing about 280 MB takes about 30 s. The report looks like this (the
2026-10-06 build against the earlier generic counts):

```
held-out set                   new       old
English (wiki)              94.25%    96.12%
Turkish (wiki)              81.06%    72.00%
Turkish (web)               80.04%    70.58%
code (local repos)          96.67%    98.31%
model answers, en           95.92%    96.23%
model answers, tr           83.91%    73.75%
model answers, code         96.33%    96.33%
```

The rows that matter most are the model's answers: they are what drafts have to
match. A plain id list (one id per line) also parses. Shiso keeps whole blocks,
so only the part of the list that falls in its 512 densest blocks survives, and
`count.py` prints how many ids that is.

## A/B on the GPU

Serve the model by path (a registry reference refuses `--draft-vocab`) once per
file. Run a fixed English/code prompt set and `bench_tr.py` for Turkish, and
compare the server's `request finished` lines (`decode_tps`,
`spec_tokens_per_step`). Greedy text must match across the two runs: a draft
vocabulary changes speed, never output.

Measured for the published file on a GB10, spec_k 3: English/code 54.16 ->
54.09 tok/s, Turkish 44.35 -> 48.83 tok/s (+10.1%), identical output.

Measure on an idle GPU. Another process on the GPU cut decode by 30-40% in one
run, and nothing in shiso's own logs showed why.
