# Test vectors

Fixtures the engine is held to. This directory is **gitignored** - nothing here
is committed. It is kept on a working machine because several of these files
cannot be regenerated without hardware or an upstream build that is not always
at hand, and the smoke tests want them.

## Layout

| Directory | What lives there |
|---|---|
| `goldens/` | **Regression locks.** Captured on this machine by the implementation that replays them. They prove today's build answers what yesterday's did; they prove nothing about correctness. |
| `references/` | **Independent references.** Captured from an implementation that shares no kernel with this one - llama.cpp, `mlx-lm`, `transformers`. These are the only fixtures here that can show this port is *wrong*. |
| `vision/` | Vision-tower references and the end-to-end media regression locks. |
| `loader/` | Checkpoint documents read verbatim: `config.json`, quant configs, layouts, manifests. No model behaviour in them. |
| `prompts/` | The prompt corpus. Plain user text, model-agnostic, shared by every `.vec`. |
| `grammar-samples/` | **Parser conformance only.** Synthetic numbers, no model behind them, never replayed. |

A `.vec` names its prompt file relative to its own directory, so fixtures under
a subdirectory spell it `../prompts/<name>`. The recorded `prompts_digest` is
taken over the prompt *contents*, not the path, so moving a fixture between
directories does not invalidate it.

## The files

| File | What it is |
|---|---|
| `goldens/qwen35-local-golden.vec` | **Regression lock.** Top-64 logits at five frontiers, captured on this machine by the implementation that replays it. |
| `goldens/qwen35-local-golden.skeleton.vec` | The capture request the golden was filled from: the five cases, no recorded entries. |
| `goldens/qwen35-greedy-64.json` | The greedy token-sequence lock. Owned by the decode test; the parity lane reads it, never writes it. |
| `references/qwen35-reference-golden.vec` | **Independent reference.** Top-64 log-probabilities at the same five frontiers, captured from upstream llama.cpp commit `c6824a9` on the GB10. This family declares no `bos_token`, so there is no leading token and a replay prepends none. Replayed at `FLOOR` with top-1 asserted exactly; `REFERENCE` is printed and not asserted, and the reason and the numbers are on the case in `crates/shiso-bench/tests/parity.rs`. |
| `references/qwen35-reference-greedy-64.json` | The greedy token-sequence lock for Qwen 3.5, captured from llama.cpp. Sixty-four argmaxes after the first prompt. This port agrees with all sixty-four. |
| `references/gemma4-reference-golden.vec` | **Independent reference.** Top-64 log-probabilities at five frontiers, captured from upstream llama.cpp commit `c6824a9` on this machine's own Metal device, with the model's own leading `<bos>`. |
| `references/gemma4-greedy-64.json` | The greedy token-sequence lock for Gemma 4, also captured from llama.cpp. Sixty-four argmaxes after the first prompt, same leading token. |
| `references/gemma4-mlx-golden.vec` | **Independent reference.** Top-64 log-probabilities at five frontiers, captured from `mlx-lm` on this machine's own Metal device. Graded at the `MLX_REFERENCE` budget, not `REFERENCE`: `mlx-lm` carries its activations in `bfloat16` and this port carries them in `f32`, which costs about 0.18 nats over forty-two layers whatever the kernels do. The same container against llama.cpp meets `REFERENCE` at 0.011. |
| `references/gemma4-mlx-greedy-64.json` | The greedy token-sequence lock for Gemma 4 MLX, captured from `mlx-lm`. |
| `references/qwen38-27b-mlx-f32-golden.vec` | **Independent reference.** Top-64 log-probabilities at the five frontiers the 4B references use, from `mlx-lm` 0.31.3 / MLX 0.31.2 over `Vontra/Qwen3.8-27B-MLX-4bit` (rev `70ae7fa`) with its activations cast to `float32`, captured by `scripts/reference/mlx_logit_vectors.py --f32` on an M1 Max. Replayed by `crates/shiso-bench/tests/parity_qwen38.rs`: the three short frontiers at `REFERENCE` (measured 0.013-0.037 nats), the two long ones at `FLOOR` with `REFERENCE` printed (measured 0.092 and 0.158). |
| `references/qwen38-27b-mlx-f32-greedy-64.json` | The greedy lock of the same capture: 64 argmaxes after the first prompt. This port agrees with all 64. |
| `references/qwen38-27b-mlx-golden.vec`, `references/qwen38-27b-mlx-greedy-64.json` | **Independent reference.** The same capture at `mlx-lm`'s default `bfloat16` activations. Graded at `FLOOR` only: `mlx-lm` in `bfloat16` sits 0.59 nats from itself in `float32` on `long_code_audit`, which is this port's whole distance from it there. |
| `references/qwen38-27b-dflash2-block.json` | **Independent reference.** The DFlash2 block drafter (`z-lab/Qwen3.8-27B-DFlash2` rev `50307d4`, quantized to 4-bit groups of 64 by MLX) over mlx-lm's 27B target, both in `float32`: per prompt, the target's first pick, the drafter's first block and its greedy acceptance over 64 tokens (4.85 and 4.60 tokens a round). Captured by `scripts/reference/block_draft_reference.py`. Replayed by `crates/shiso-models/tests/qwen38_block_draft_metal.rs`, which also holds speculative decoding to serial decoding id for id. |
| `references/qwen35moe-reference-logits.json` | **Independent reference.** The last row's logit bounds, argmax and top-16, from `transformers` 5.16.1's own `qwen3_5_moe` forward over eight token ids, captured by `scripts/qwen35moe_reference.py`. It reads the checkpoint's *native* head layout where the engine permutes the weights at load and indexes them the other way, so an agreement is evidence about both. Replayed by `crates/shiso-cuda/tests/q35moe_load.rs` at a budget of 0.1 of a logit; measured worst divergence 0.0786. |
| `references/chat/qwen4exp/*` | **Independent reference.** Chat-template renders for the sparse family: each `<case>.request.json` holds a wire body (OpenAI or Anthropic) and, in its `hf` block, the conversation the official serving path would hand the template. `<case>.txt` and `<case>.ids` are what `transformers` 5.16.1's own `apply_chat_template` and tokenizer produced for it from the checkpoint's sidecar, written by `scripts/reference/render_chat_fixtures.py`, which refuses to overwrite. A case with `expected_from` shares another case's output. Replayed byte for byte by `crates/shiso-server/tests/chat_parity_qwen4exp.rs` when `SHISO_TEST_SIDECAR_QWEN4EXP` names the sidecar. |
| `references/audio/clips/*.wav`, `references/audio/clips/manifest.json` | Public speech clips (source URL and sha256 in the manifest), cut to 30 s and converted once to 16 kHz mono PCM16, plus `speech-en-odd` (the first 176641 samples of `speech-en`), whose length is not a multiple of the 128-sample padding unit so the left-padded valid-frame mask meets a boundary, and `two-speakers` (spans of `speech-en` and `weather-es` joined in the order the manifest lists), two real voices with a known turn order. Inputs to every audio reference. |
| `references/audio/mel-<clip>-{centred,left-padded}.{json,f32}` | **Independent reference.** Log-mel features from `transformers` 5.18.0.dev0's own feature extractors (the Nemotron 3.5 ASR checkpoint's and Gemma 4's), captured by `scripts/reference/audio_reference.py mel`. Replayed by `crates/shiso-audio/tests/mel_reference.rs` at the budgets recorded there; measured worst 4.6e-4 (centred) and 2.8e-5 (left-padded) log units. |
| `references/audio/asr-<clip>-r<R>-<lang>.json`, `asr-<clip>-r<R>-encoder.f32`, `asr-<clip>-r<R>-<lang>-projected.f32` | **Independent reference.** The Nemotron 3.5 ASR checkpoint run by `transformers` 5.18.0.dev0 in f32 (`get_audio_features` and `generate`) at lookahead R in {13, 3}, with the auto prompt and the clip's own language: greedy token ids, their encoder frames, the decoded text (special tokens kept), the full encoder output `[frames][1024]` and the prompt-fused projection `[frames][640]`. Captured by `scripts/reference/audio_reference.py asr`. |
| `references/audio/gemma-<clip>.json`, `gemma-<clip>-tower.f32`, `gemma-<clip>-embedded.f32` | **Independent reference.** `transformers` 5.18.0.dev0's own `Gemma4AudioModel` (sdpa attention; the eager path inverts its mask) running the MLX checkpoint's bf16 audio-tower weights, over the Gemma 4 feature extractor's mel: the tower output `[tokens][1536]` and the multimodal embedder's output `[tokens][2560]` (its 4-bit projection dequantised), valid audio tokens only. Captured by `scripts/reference/audio_reference.py gemma`. |
| `references/audio/diar-<clip>.json`, `diar-<clip>-probs.f32`, `diar-<clip>-embeds.f32`, `diar-<clip>-step0.f32` | **Independent reference.** Nemotron-3-Diarization run by `transformers` 5.18.0.dev0 in f32 in its offline mode over the ASR extractor's mel: per-10 ms speaker probabilities `[mel frames][8]`, the stacked-feature projections `[encoder frames][512]`, the encoder output of the first offline step, and the probabilities' segments at 0.5. Captured by `scripts/reference/audio_reference.py diar`. |
| `references/q4e-fp8-long.vec`, `references/q4e-fp8-long.skeleton.vec` | **Independent reference.** Top-128 raw logits at 31 frontiers over three long traces (8422, 16490 and 32512 tokens), from `transformers` 5.16.1's own `modeling_qwen4_exp` over the FP8 checkpoint (W8A16), captured by `scripts/reference/qwen4exp_fp8_ref.py capture`. Its softmax-attention layers run query-blocked so a pass fits at 32k; `check-attention` holds that path bit-identical to the library's on real hidden states, and a re-capture of `q4e-fp8-short.vec` with it is byte-identical. Frontiers at strides, on both sides of 8192/8193, 16384/16385 and 24576/24577 (the sparse family's server prefill chunk), past the indexer budget (2052), and at the answer. Full rows are kept outside the tree. |
| `prompts/qwen4exp/trace_{8k5,16k,32k}_gutenberg<N>_*.txt` | The long traces: public-domain Project Gutenberg ebooks 1661 (*The Adventures of Sherlock Holmes*), 1342 (*Pride and Prejudice*) and 35 (*The Time Machine*), with the Gutenberg header, footer and illustration captions stripped, cut at a paragraph, each ending with a question whose answer is in its first paragraphs. |
| `vision/qwen35-vision-tower.vec` | **Independent reference.** 4096 sampled output rows and full-block FNV digest from mlx-vlm 0.7.1 on the Qwen 3.5 vision tower, grading the tower's projected activations directly. |
| `vision/gemma4-vision-tower.vec` | **Independent reference.** The same for the Gemma 4 vision tower. Replaced the retired `gemma4-vision-golden.vec`, which held llama.cpp chat-endpoint logits and carried prompt-formatting dependencies this does not. |
| `vision/qwen35moe-vision-tower.json` | **Independent reference.** 256 sampled values of the vision tower's projected block, from `transformers` 5.16.1's own `Qwen3_5MoeVisionModel` run over the patch rows this engine fed its own tower, captured by `scripts/qwen35moe_vision_reference.py`. The script permutes those rows into the reference's conventions first - this tree emits a patch row channel-fastest in raster order, `transformers` channel-slowest in merge-block order - so what is compared is the tower and not two preprocessings. Replayed by `crates/shiso-cuda/tests/q35moe_vision.rs` at a budget of 1e-3; measured worst 8.4e-6, and 4.9e-5 over the whole block. |
| `vision/qwen35-vision-tower-manifest.json` | The recorded Qwen 3.5 tower manifest: the names and shapes the local checkpoint's shard actually held when it was read. Produced from the checkpoint, not from this code, so the loader cannot agree with its own mistake. Owned by `crates/shiso-models/src/qwen35/vision/testing.rs`. Previously sat in a second `crates/shiso-models/test-vectors/` directory of its own. |
| `vision/vision-qwen35-image.json` | **Regression lock.** The answer a request carrying a generated 224x224 image produced on this machine, over real HTTP. Owned by `image_e2e_golden`. |
| `vision/vision-qwen35-video.json` | **Regression lock.** The same, for a generated 8-frame GIF that folds into four temporal groups. Owned by `video_e2e_synthetic`. |
| `vision/vision-gemma4-image.json` | **Regression lock.** The same through the Gemma 4 tower. Owned by `gemma4_image_e2e_golden`. |
| `vision/vision-gemma4-mlx-image.json` | **Regression lock.** The same lock for the mlx container, which quantizes the tower's projector to four bits and so answers differently. Owned by the same case under `SHISO_TEST_GEMMA4_WEIGHTS=mlx`. |
| `loader/qwen35-config.json`, `loader/qwen35-optiq-layout.json`, `loader/qwen35-q4km-manifest.tsv` | Loader fixtures, owned elsewhere. |
| `loader/gemma4-config.json` | Loader fixture. The Gemma 4 E4B `config.json`, verbatim from the local sidecar directory; owned by `crates/shiso-models/tests/gemma4_config.rs`. |
| `loader/qwen35moe-config.json`, `loader/qwen35moe-quant-config.json` | Loader fixtures. The `config.json` and `hf_quant_config.json` of `Qwen3.6-35B-A3B-NVFP4`, verbatim from the local snapshot; owned by `crates/shiso-models/src/qwen35moe/`. Checked in rather than read from the 23.4 GB checkpoint so the config cases run on a machine that does not carry it - a skipped case and a passing one look identical in the output. |
| `loader/ple-codec-vectors.json` | Loader fixture for the PLE codec; owned by `crates/shiso-ple/tests/ple_codec.rs`. |
| `grammar-samples/*.vec` | **Parser conformance only.** Synthetic numbers, no model behind them, never replayed. |
| `prompts/*.txt` | The prompt corpus. Plain user text, model-agnostic. |

### Removed

| File | Why |
|---|---|
| `gemma4-vision-golden.vec`, `prompts/gemma4-vision-quadrants.txt` | Retired. Held llama.cpp chat-endpoint logits; superseded by `vision/gemma4-vision-tower.vec`, which grades the tower's projected output rows directly and has no prompt-formatting dependency. Nothing replayed them. |
| `q4e-reference-golden.vec`, `prompts/mid_memory_archive.txt` | The **sparse** family's reference, captured on the DGX Spark. The parity lane stopped replaying it in Phase 8 when `references/qwen35-reference-golden.vec` took its place, and no test in this tree read it afterwards. The recipe is in the root `HANDOFF.md` if it is ever needed again. |

## What a locally-captured golden proves

**A locally-captured golden locks regression; it proves nothing about
correctness.** It was captured by the same implementation that replays it, so it
says that today's build answers what the build that captured it answered — to a
thousandth of a nat, across sixty-four ranks, on five prompts. It cannot say that
either answer is right.

The only fixture that could say that is one captured from an *independent*
implementation of the same model. Two now exist, for two families:
`references/gemma4-reference-golden.vec` and, since Phase 8, `references/qwen35-reference-golden.vec`.
Both are below under **The independent reference**. The second was produced on
the DGX Spark by `scripts/capture_qwen35_golden.sh`, because llama.cpp
`c6824a9` carries `LLM_ARCH_QWEN35` and serves that container.

For the **sparse** family there is none of this kind captured here any more.
`q4e-reference-golden.vec` was captured on the DGX Spark from that family's own
reference implementation over the sparse checkpoint - the root `HANDOFF.md` has
the recipe - so it was a reference for *that* family and for no other. Replaying
it against the qwen35 checkpoint is a category error, and it was one: until
Phase 8 `crates/shiso-bench/tests/parity.rs` did exactly that, and reported
top-1 disagreeing on five of six cases and 3.7 to 7.6 nats. That file now
replays `references/qwen35-reference-golden.vec`, and the sparse fixture has
been removed from this tree; recapture it from `HANDOFF.md` if the sparse family
is picked back up.

So for the families with no reference, the harness extracts what a
single-implementation machine honestly can:

1. **Determinism** — two identical runs produce bit-identical logits. Without
   this nothing else here is a fact rather than a sample.
2. **Two routes through the same weights** — one prompt fed at ten different
   chunk widths. Different grids, different tile counts, different boundaries in
   the recurrent scan, no reference implementation required. This is the
   strongest claim available locally, and it is the one that catches a defect
   whose effect depends on the shape of a launch.

   The ten widths do **not** all reach the same row, and the shape of how they
   differ is the finding: they fall into five exactly bit-identical classes.
   `shiso_bench::chunking` derives which class a width lands in from the three
   splits between a caller's width and the arithmetic, so the test asserts
   bit-equality *within* each class and rank stability across them. A width of
   384 is in the same class as one of 128 and not as one of 320 — derived first,
   then confirmed on the device. What the classes mean for a caller is recorded
   as an open item in `PROGRESS.md`; the top-1 and top-20 of the row do not move
   between any of them.
3. **The recorded slice, not the argmax** — see below.

### The independent reference

`references/gemma4-reference-golden.vec` is the one fixture here that is not shiso's own
answer. It was captured from upstream **llama.cpp commit
`c6824a9e42ceeda5d58089fa274ddd816e59e68e`**, built for Metal, reading the same
Gemma 4 E4B container, on this same M1 Max. Two implementations that share no
kernel, no reduction order and no dequantization routine agreeing about the top
of a row is a correctness claim; a disagreement is a defect in one of them.

Three things it is careful about, all of them recorded in its own header and
re-checked on every replay:

- **The request.** `llama-server` with `n_probs: 64`, `n_predict: 1`,
  `temperature: 0`, `cache_prompt: false`, `post_sampling_probs: false`, and the
  raw prompt file as the `prompt` string. One forward pass, only the frontier
  row, no chat template — the same prompt files the qwen35 golden uses.
- **The tokens, and the leading one.** The prompt goes over as text and
  `llama-server` tokenizes it its own way, which for this family prepends the
  model's `bos_token_id` — id **2**, as `config.json`'s `text_config` declares
  it. That is the regime Gemma is run in, and a golden that skipped it would
  grade a state the model never sees. The sidecar `tokenizer.json` does *not*
  add it: its post-processor is `single: [$A]` with no special tokens, so the
  `tokenizers` crate adds nothing at either `special` setting. So the fixture
  records `# leading_token 2` and the replay prepends it, cross-checking the
  header against the config; a fixture without that header is refused rather
  than replayed without one. The fixture also records an FNV digest of each
  case's ids *including* the leading token, which the replay recomputes from its
  own vocabulary — so a tokenizer divergence is *named* instead of being
  measured as a wrong kernel.
- **The units.** `llama-server` exposes no raw logit; it softmaxes the whole row
  and returns log-probabilities. The fixture says `# values logprob` and the
  replay subtracts its own row's log-sum-exp — over the full vocabulary, never
  over the recorded sixty-four — before comparing.

Capture it with `scripts/capture_gemma4_golden.sh` (which stops the server before
returning, because Metal runs one workload at a time) and replay it with
`crates/shiso-bench/tests/parity_gemma4.rs`. The raw HTTP answers are provenance,
not fixtures, and are deliberately not committed: what the commit carries is the
converted `.vec`, the request in its header, and the llama.cpp commit.

### The two vision fixtures, and what carries their weight

`vision-qwen35-image.json` and `vision-qwen35-video.json` record one answer each,
as text off the wire, and each file repeats in its own header that a locally
captured golden locks regression and proves nothing about correctness. Nothing
about the WORDING of either answer is load-bearing: a build that wrote different
but equally sensible prose would fail them, and the fix in that case is to look
at the change and recapture deliberately, exactly as for the logit goldens.

What carries the weight in those two cases is not the recorded answer at all. It
is the pair of requests beside it: the same prompt and the same media DIMENSIONS
- so the same rendered prompt, token for token - with different PIXELS. A build
that dropped the pixels between the request and the forward pass would answer
those two identically, because its inputs would be identical, and the test would
fail on that rather than on the lock. The lock catches drift; the pair catches
the path being disconnected, which is the failure that produces a fluent answer
and no error at all.

## Why the fixtures record a slice and not a token sequence

Greedy decoding throws away almost everything a forward pass computed. Two builds
can agree on every argmax for sixty-four steps and disagree by whole nats on the
rest of the row, which is what a wrong activation or a transposed weight pack
usually looks like: the winner survives because it was winning by a mile, and the
damage sits in ranks two through sixty-four. Every case here therefore records
the top of the distribution, and the comparator grades the winner, three overlap
depths and the worst logit disagreement at once.

`crates/shiso-bench/tests/vectors.rs` measures that sensitivity rather than
asserting it: it takes the real recorded slices and damages them the way those
defects would have — a uniform shift, a non-uniform wobble, a reversed block, an
off-by-one in the ranks, a collapsed row — and pins which budget catches each.

## The grammars

Both are line-oriented, `#` introduces a comment, and both are deliberately
trivial to parse from any language. The field order is the upstream
implementation's, column for column, so a fixture written by either side is
readable by the other's tooling after substituting the first line. See
`shiso/README.md` § Provenance for what the upstream is; it is not named in the
files, the parser, or the tests.

### `# shiso-local-logit-vectors-v1` — token ids and raw logits

```
case <id> <mode> <ctx> <frontier> <prompt-file> <top-count>
top <rank> <token-id> <logit>
```

`<mode>` is `text` (tokenize the file plainly) or `rendered` (the file already
carries its control tokens). `<prompt-file>` is relative to the fixture. Ranks
must be sequential from 0, a block may not exceed 128 entries, and `ctx` must
exceed `frontier`. This is the working grammar: token ids and raw logits are
exactly what an engine that owns the checkpoint can emit.

### `# shiso-official-logprob-vectors-v1` — token bytes and log-probabilities

```
case <id> <ctx> <steps> <prompt-file>
step <index> <selected-hex> <top-count>
top <token-hex> <logprob>
end
```

Tokens are hex-encoded UTF-8 bytes, not ids, because the kind of source this
grammar exists for — a hosted endpoint returning text and log-probabilities —
has no ids to give. Nothing in this tree replays one; the reader carries the
grammar so that the shape does not have to be invented under pressure the day a
hosted slice exists.

### Header comments shiso adds

Ignored by any reader that skips `#`, so they cannot break one:
`label`, `model_file`, `model_bytes`, `backend`, `captured_from`,
`vocabulary_digest`, `prompts_digest`. The last two are rechecked on every
replay: a different vocabulary encodes the prompt differently and an edited
prompt moves the frontier, and either one moves every logit. A replay whose
inputs disagree with the capture's is refused rather than graded, because a
parity failure that blames the kernels for a changed tokenizer is worse than no
test.

## Capturing

```sh
cargo run -p shiso-bench --release --bin parity -- \
  --model "$SHISO_TEST_MODEL" --backend metal \
  --vectors test-vectors/goldens/qwen35-local-golden.skeleton.vec \
  --capture test-vectors/goldens/qwen35-local-golden.vec --preview 24 \
  --label "$(git rev-parse --short HEAD) $(sysctl -n hw.model) metal"
```

`--preview N` decodes N greedy tokens per case and prints them. Read them. A
golden nobody read locks in whatever was wrong on the day it was taken; the point
of committing one is that a person looked at the continuation and judged it
coherent. Capture refuses to overwrite an existing fixture, so a recapture is a
deliberate act with a deliberate deletion in front of it.

Run the capture twice and diff the two files before committing: they must be
byte-identical. That is the determinism floor, and it is cheaper to check here
than to discover later.

## Replaying

```sh
cargo run -p shiso-bench --release --bin parity -- \
  --model "$SHISO_TEST_MODEL" --backend metal \
  --vectors test-vectors/goldens/qwen35-local-golden.vec [--case ID]... [--strict] [--json OUT.json]
```

Two budgets, both fixed, each defined exactly once in
`crates/shiso-bench/src/compare.rs`:

| | top-1 | top-5 | top-20 | top-64 | worst \|Δlogit\| over the top 20 |
|---|---|---|---|---|---|
| floor (cross-build) | exact | ≥ 4/5 | ≥ 15/20 | ≥ 40/64 | ≤ 8.0 |
| `--strict` (same-source) | exact | 5/5 | 20/20 | 64/64 | ≤ 1e-3 |
| `REFERENCE` (independent implementation) | exact | 5/5 | 20/20 | ≥ 60/64 | ≤ 0.05 |

The floor is for a candidate built differently or running on other hardware; it
is deliberately blind to the last bits of a reordered reduction. `--strict` is
for a candidate that should be reproducing the reference on the same device, and
on this machine the golden replays at `--strict` with a worst disagreement of
exactly zero.

`REFERENCE` is for the one comparison against a different implementation. It sits
between the two: no shared kernels means a thousandth of a nat is not available,
but the same weights on the same hardware means eight nats would pass a port with
a whole operator wrong. Measured on Gemma 4 E4B against llama.cpp `c6824a9`: all
five cases agree on 64/64 ids at every depth, with the worst shared-entry gap
running from 0.003 to 0.011 nats. The ceiling is not lowered to that, because the
same five cases *without* the leading `<bos>` measured 0.041 — four times worse,
which says the gap is a function of what is being evaluated rather than a
constant of the two builds. Three budgets, each defined exactly once; a failing
one is a bug to locate.

## When a replay fails

**A mismatch is a bug to find, not a budget to widen.** Widening one converts a
real finding into a permanently weaker test, and the two defects this project has
actually shipped were both of the kind a widened budget would have hidden. The
ladder, in order:

1. The failing case and rank name the row. Dump that logits row.
2. Walk the per-op ladder against a host-side reference at 1e-5, at production
   shapes. The first op over 1e-5 names the kernel.
3. If every kernel matches its reference but the model still diverges, the
   kernels are right and the *launches* are wrong: diff grid, block and shared
   memory against the recorded kernel table and the launcher they were derived
   from.
4. If a kernel itself diverges, compare its compiled body against the source it
   was copied from (`scripts/q4e_ptx_oracle.sh`). Equal bodies mean a host bug.
   Unequal bodies mean the copy drifted, and the fix is to re-copy it — never to
   hand-patch the arithmetic.

The fix commit names the kernel or op it changed.

## What is never recorded here

No throughput number, from any lane. Parity and performance are measured under
different conditions and a number that moved because the machine was busy would
make a parity artifact unfalsifiable.
