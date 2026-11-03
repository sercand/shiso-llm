---
name: optimizing-metal-kernels
description: Use when writing, porting, tuning or debugging a Metal (Apple Silicon, MSL) kernel in shiso, or when a model's decode tok/s, prefill tok/s, image or audio step time on macOS is too slow, or when a Metal kernel gives wrong, nondeterministic or batch-dependent results.
---

# Optimizing Metal kernels and models in shiso

## Overview

Speed work here is **measure → change one thing → prove the bits → measure again**.
A faster kernel that changes an answer, or that answers differently depending on
how many rows share the pass, is a bug, not a win. Read `AGENTS.md` first; its
rules are enforced by `bash scripts/check.sh`.

`lessons.md` in this directory lists measured wins and failures from this repo,
with numbers. Read the section that matches your kernel before you start.

## The loop (do every step, in order)

1. **Get a baseline number** on the same machine, with the release build and
   nothing else running (see Benchmarking). Write it down: model, quant, prompt
   length, machine (e.g. M1 Max), and the median.
2. **Find the bound.** Decode at 1 token is almost always memory-bound. Compute
   GB/s = bytes of weights read ÷ kernel time. Peak bandwidth: base M-chip
   ~100–120 GB/s, Pro ~150–273, Max ~400 (M4 Max ~546), Ultra ~800. Good
   decode kernels reach 60–90% of peak (q4_0 matvec: 195 GB/s on an M1 Max).
   Under half of peak means the kernel is ALU- or latency-bound:
   it is doing too much per weight (decode, scale math, divides) or launching
   too few threadgroups. Prefill (many tokens) is compute-bound, so use
   simdgroup matrices.
3. **Make one change.** Pick it from the Optimization checklist below.
4. **Prove correctness.** Run the kernel's `shiso-kerneltest` case on the
   device (cases live in `crates/shiso-kerneltest/src/<set>/<file>.rs`; the
   Metal test module that runs them (all in the one `kernels` target) is named after that file, e.g.
   `qwen35/gemv.rs` → `--test kernels q35_gemv::`; `grep -l <case_fn> crates/shiso-metal/tests/`
   finds it), then model parity. Both must pass. "Greedy text unchanged" is the
   minimum you report.
5. **Measure again** with the same protocol. Keep the change only if the median
   improves beyond noise (more than 2%).
6. **Record the number** in a comment next to the choice: `measured on an M1
   Max: 115 ms against 228 ms per layer`. Put the before → after numbers in the
   commit message too.

## Optimization checklist (most common wins first)

| Symptom | Fix |
|---|---|
| Quantized matvec far below bandwidth | Unpack a whole sub-block per lane once. Fold scales **after** the dot: `d*sum(q*y) - m*sum(y)`. Do not rescale every 4 weights. |
| Matvec uses few lanes per row | Wider per-lane runs (8 consecutive weights = 16 B loads). Reduce with `simd_shuffle_down` or the repo's fixed-order sum helpers. |
| Many tokens on a matvec kernel | Matvec re-reads the whole matrix per row block. Above the decode batch, route to a GEMM (`GEMM_MIN_TOKENS`). |
| Prefill GEMM slow | 64×64 tile, 4 simdgroups (2×2), each 4×4 `simdgroup_float8x8` accumulators, k staged 16 at a time, rows padded +4 floats, next slice prefetched into registers. |
| Simdgroup-matrix kernel 6× slower than expected | Fragment loop not unrolled; add `_Pragma("clang loop unroll(full)")`. |
| Many tiny launches per layer | Fuse: residual add into the next norm, QK-norm + RoPE + KV append, gate+up+SiLU, several projections into one launch, gated residual in the GEMM epilogue. |
| A kernel uses one threadgroup and the GPU idles | Split the work (slices), then combine in a **fixed slice order**. |
| Attention slow | One pass, online softmax in registers, K/V streamed through threadgroup memory, stage as `half` with f32 accumulation. |
| Decode attention slow per key | Look for 64-bit integer divide/modulo; use 32-bit. Use fast `exp` when every argument is ≤ 0. |
| Register spills (slow, no clear reason) | Fewer rows/heads per thread; keep arrays compile-time indexed; shrink the per-thread accumulator count. |
| Many small clears or copies | Use the compute `kernel_fill_zero_u8`, not a blit. A blit closes the compute encoder. |
| Unsure which lane width is best | Don't hard-code it. Compile variants and let the startup tuner pick (`shiso-engine/src/tune.rs`, `gdn_tune.rs`). |

Weights stay in their stored width (bf16, fp8, quant) and are widened in
registers. Never dequantize a whole matrix to float in memory.

## Hard rules (a violation breaks the answer, not just the speed)

- **Batch invariance:** a row's result depends on the shape only, never on how
  many rows share the call. Split counts come from the weight shape alone.
- **Fixed reduction order:** use `dense_warp_sum`/`dense_block_sum` or the
  `q35_*` helpers. Bare `simd_sum` has unspecified order.
- **Fused = identical:** a fused kernel computes exactly what the unfused chain
  computed.
- **Limits are per pipeline:** `maxTotalThreadsPerThreadgroup` depends on
  register use (448, not 512, has happened). Probe it. Threadgroup memory is at
  most 32768 B on every M-series part.
- **Alignment:** checkpoint tensors are only element-aligned (q6_K blocks are
  210 B). Read them through `packed_float4`/`packed_ushort4`/`packed_uchar4`.
- **No buffer-length query in MSL:** pass explicit flags (e.g. `has_bias`), never
  a zero sentinel.
- **Fast-math pragma:** every `#pragma METAL fp math_mode(safe)` must be closed
  by `math_mode(fast)` in the same file. All sources become one translation
  unit, so an unclosed `safe` leaks into later files.

## Adding a kernel (files to touch)

1. Write MSL in `kernels/metal/<set>/`. Instantiate templates with
   `template [[host_name("kernel_...")]]`. Keep the CUDA entry name for a port.
2. For a new file, append it to `SOURCES` in `crates/shiso-metal/src/compile.rs`,
   after its set's `*_common.h`.
3. Add a row to the set's `KERNELS.md`: bind order, grid, block, smem.
4. Write the host launcher. Generic kernels go in `shiso-engine`; family-specific
   ones go in `shiso-models/src/<family>/kernels/`. Use a `#[repr(C)]` +
   `bytemuck::Pod` args struct with explicit padding and a
   `const _: () = assert!(size_of::<Args>() == N);`.
5. Write the case in `crates/shiso-kerneltest/src/<set>/`, compared against a
   plain Rust f32 oracle. Cover every instantiation.
6. Add a thin `#[ignore = "requires a Metal device"]` wrapper in
   `crates/shiso-metal/tests/<name>.rs` (copy `q35_gemv.rs`), and add a
   `[[test]]` entry in `crates/shiso-metal/Cargo.toml` (`autotests = false`).
7. Dispatch through `Backend::run(KernelRef::new(MODULE, "kernel_..."), ...)`.
   `grid` is the **threadgroup count**, not the thread count.

Iterate on MSL without rebuilding: `SHISO_METAL_SOURCE_DIR=kernels/metal`
(read at device creation, so it applies to tests and bench bins too).

## Profiling

Three tools, one question each. All of them time every launch on its own
(one command buffer per launch), so a profiled total is **above** the batched
time by the overlap and submission gaps it gives up. Read the per-kernel
shares, and take speed from the batched number each tool prints beside them.

```sh
# Where does a text model's decode step (or prefill) spend its GPU time?
cargo run --release -p shiso-bench --bin profile -- \
  --model <ckpt.gguf | moe-dir> --phase decode --steps 32 --json /tmp/decode.json
#   --phase prefill --prompt-tokens 512   for prefill; --config <cfg> for a non-recorded shape

# Is this kernel faster? (A/B one kernel at a realistic shape)
cargo run --release -p shiso-bench --bin kernel_bench -- --list
cargo run --release -p shiso-bench --bin kernel_bench -- q4_0 --reps 200 --json /tmp/kb.json
#   add a case: a `pub(crate) fn bench_*` beside the kernel's correctness case in
#   shiso-kerneltest, plus a row in `bench::CASES`

# Embedding / vision passes
cargo run --release -p shiso-bench --bin embedding -- --model <gguf> --profile /tmp/emb/
cargo run --release -p shiso-bench --bin vision_stages -- --model <mmproj.gguf> --family qwen35 \
  --image <png> --out /tmp/stages --profile /tmp/vision.json

# A real server request (chat, completions, decisions, images, embeddings)
shiso serve <model-id | --model ckpt> --max-num-seqs 1 --profile-dir /tmp/prof
#   one <kind>-<pid>-<id>.json per request (kind: engine | image | embed); refused
#   with --asr-model. <id> is the engine's own counter, in arrival order, not the
#   HTTP `cmpl-...` id. A report covers only its service's backend.
```

The JSON is `ProfileReport::to_json` (schema 1). Times are integer ns. Rows
are keyed by entry, specialization constants, grid and block, and sorted by
`total_ns`. For the overlapped timeline (what the serialized numbers cannot
show) use `xcrun xctrace record --template 'Metal System Trace' --launch -- <cmd>`.

## Benchmarking and correctness commands

```sh
# kernel cases on the device (pick the test target)
cargo test -p shiso-metal --features metal --test kernels q35_gemv:: -- --ignored

# quick throughput number (prefill + decode tok/s); also: gemma4, q35moe, q35moe_gguf
SHISO_TEST_MODEL=/path/model.gguf cargo run --release -p shiso-bench --bin qwen35

# prefill slope at 64/512/2048 tokens
cargo run --release -p shiso-bench --bin prefill_sweep -- <ckpt> [config]

# model parity against vectors (goldens = regression lock only)
cargo run -p shiso-bench --release --bin parity -- --model "$SHISO_TEST_MODEL" \
  --backend metal --vectors test-vectors/goldens/<file>.vec [--case ID] [--dump-rows DIR]
SHISO_TEST_MODEL=... cargo test -p shiso-bench --release --test parity -- --ignored --nocapture

# perf gate: record on the pinned machine only; check is arithmetic
cargo run --release -p shiso-bench --bin perf_gate -- record --target metal --family qwen35 --model PATH
cargo run -p shiso-bench --bin perf_gate -- check   # 1 = regression (>5% slower), 2 = re-record, NOT a regression

bash scripts/check.sh          # the gate, before you say "done"
```

Only the user knows which machine is pinned. If you are not told you are on
it, don't `record`; report your ad-hoc numbers and say the record is stale.

The protocol discards one warm-up, drops the first decode step, reports the
**median**, and refuses runs with more than 2% spread. Do the same by hand:
never quote a single run or a best-of-N.

To find a diverging layer, use `--dump-layers` with
`scripts/reference/bisect_layers.py` and `compare_rows.py`.

## Common mistakes

- Saying a device test passed when it was skipped. A skip prints a note on
  stderr and looks like a pass. Check that it ran (`--nocapture`).
- Treating `test-vectors/goldens/` as truth. They only lock yesterday's answer.
  Only `references/` is independent.
- Changing compile options globally to win speed. Read the comment near
  `compile.rs:~345` first.
- Binding args at the wrong index: some dense affine kernels take the args
  block at `[[buffer(5)]]`, last. Read `KERNELS.md`.
- Asking for CUDA-sized dynamic smem on a kernel that declares no
  `[[threadgroup(n)]]` parameter. Declare static `threadgroup float smem[N]`.
- Putting a model family name in `shiso-core`/`shiso-engine`/`shiso-kvstore`,
  or upstream project names in code or comments.
- Using `#[allow]`, `println!`, `unwrap()`, or `unsafe` without `// SAFETY:`.
