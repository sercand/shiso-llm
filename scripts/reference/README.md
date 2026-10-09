# Reference tooling

Scripts that produce or consume *independent* evidence about this engine: renders
and logits from an independent implementation of a model family (see
`test-vectors/README.md` for why only those belong in `test-vectors/references/`),
and behavioural probes run against several servers. They are Python, run with the
reference venv (`SHISO_REF_PYTHON`, default `~/work/exllamav3/.venv/bin/python`)
unless noted, and are not run by `scripts/check.sh`. What each depends on upstream is
in `PROVENANCE.md`.

| Script | What it does |
|---|---|
| `qwen4exp_fp8_ref.py` | The qwen4exp FP8 reference: runs the independent implementation's own modules over the FP8 checkpoint (W8A16, layer-streamed experts, teacher-forced, no cache) and writes a replayable `.vec` plus full fp16 logit rows at chosen frontiers. `selfcheck` grades its e4m3 decode against `shiso-modelfile`'s for all 256 codes and its block dequant against NumPy; `capture` fills a skeleton `.vec`; `greedy` prints a short cached continuation; `probe` times the first few layers at long lengths before a full pass is attempted; `check-attention` runs the library's own indexer and attention beside the driver's query-blocked one on real hidden states and requires the selection mask and the output to be bit-identical. The full-attention layers run query-blocked (1024 queries at a time, memory-efficient sdpa only, never a `[T, T]` mask), which is what lets a pass reach 32k tokens. Refuses to start while a `shiso`/`ds4`/vLLM server runs, below 60 GiB `MemAvailable`, below 40 GiB free disk, or when a dump would exceed `--max-dump-bytes`. |
| `compare_rows.py` | Grades a candidate row set against a reference row set (the `--dump-rows` / `index.json` format): top-1, top-5/20 overlap, exact KL, max \|Δlogit\| over the reference top-20, and KL by frontier. `--self-test` checks itself on synthetic rows. |
| `render_chat_fixtures.py` | Renders `test-vectors/references/chat/qwen4exp/*.request.json` with the reference chat template, for `crates/shiso-server/tests/chat_parity_qwen4exp.rs`. |
| `repro_readback.py` | An agent loop with `read_file`/`edit_file` over an email workspace; it flags runs where the model claims text it just wrote came back different. Stdlib only. Endpoint and key come from `LLM_BASE_URL`/`LLM_API_KEY` or flags; nothing is embedded. |
| `embedding_gemma2_reference.py` | The multimodal embedding checkpoint's references: runs the checkpoint's own `sentence-transformers` pipeline (the `transformers` model and processor, mean pooling, normalize) in f32 on the CPU over five text, two image, two audio and one video input, and writes each case's `input_ids`, embedding, pooled vector, Matryoshka truncations and per-token hidden states, plus tower-level intermediates for one image and one audio case, to `test-vectors/references/embedding-gemma2/`. Generates or cuts its media inputs and saves them beside the fixtures. Cross-checks `encode` against a direct forward pooled in the script and stops if they disagree. Needs `sentence-transformers`, `av` and `soundfile` in the venv; `--model-dir` is the checkpoint snapshot. Never overwrites a case. |
| `LAYER_DUMP.md` | What shiso can and cannot dump per layer for the sparse family, and the per-position layout a reference should write. |

GB10 rules: any run that loads a model holds the GPU lock
(`~/work/qwen-3.8-flash/gpu_lock.sh <command>`); one model at a time; check `free -g`
and `df -h /` first; stop servers with `pkill -x shiso`.
