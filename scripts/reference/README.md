# scripts/reference

Tools that produce or grade fixtures against an **independent implementation**
of a model family (see `test-vectors/README.md` for why only those belong in
`test-vectors/references/`). They are Python, run with the reference venv
(`SHISO_REF_PYTHON`, default `~/work/exllamav3/.venv/bin/python`), are not run
by `scripts/check.sh`, and must hold the GPU lock
(`~/work/qwen-3.8-flash/gpu_lock.sh <command>`) for any run that loads a model.
What each depends on upstream is in `PROVENANCE.md`.

| Script | What it does |
|---|---|
| `qwen4exp_fp8_ref.py` | The qwen4exp FP8 reference: runs the independent implementation's own modules over the FP8 checkpoint (W8A16, layer-streamed experts, teacher-forced, no cache) and writes a replayable `.vec` plus full fp16 logit rows at chosen frontiers. `selfcheck` grades its e4m3 decode against `shiso-modelfile`'s for all 256 codes and its block dequant against NumPy; `capture` fills a skeleton `.vec`; `greedy` prints a short cached continuation to read. Refuses to start while a `shiso`/`ds4`/vLLM server runs, below 60 GiB `MemAvailable`, below 40 GiB free disk, or when a dump would exceed `--max-dump-bytes`. |
