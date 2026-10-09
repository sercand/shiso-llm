---
name: optimizing-cuda-gb10-kernels
description: Use when writing, porting, tuning or debugging a CUDA kernel in shiso, or when decode tok/s, prefill tok/s, MoE, GDN/linear-attention, attention or speculative-decode speed on the NVIDIA GB10 (DGX Spark, Blackwell sm_121, unified memory) is too slow, or when a CUDA kernel gives wrong, nondeterministic or batch-dependent results.
---

# Optimizing CUDA kernels and models on GB10 in shiso

## Overview

Speed work here is **measure → change one thing → prove the bits → measure again**.
A faster kernel that changes an answer, or that answers differently depending on
how many rows share the pass, is a bug, not a win. Read `AGENTS.md` first; its
rules are enforced by `bash scripts/check.sh`.

`lessons.md` in this directory lists measured wins and failures from this repo,
with numbers. Read the section that matches your kernel before you start.

## GB10 facts to design against

| Fact | Value |
|---|---|
| SMs | 48 |
| Registers | 64K per SM; only 64 per thread at 1024 threads |
| Shared memory | 100 KB per SM; per-block opt-in max 101376 B (131072 refused); 48 KiB works without opt-in |
| Weight streaming | ~230 GB/s achievable (~273 GB/s peak LPDDR5x). Under ~55% of that is bad. |
| Memory | Unified, committed on first touch. Page cache counts as "used". |
| PTX arch | Build `compute_80`; the driver JITs it for sm_121. `sm_121a` enables FP4 MMA (`SHISO_CUDA_HAVE_MXF4`). |

A launch grid smaller than 48 blocks leaves SMs idle. One block per SM
starves narrow outputs. Decode is bandwidth-bound: **fewer bytes beat less
arithmetic.**

## The loop (do every step, in order)

1. **Get a baseline number** on the GB10, with the release build, one model
   loaded, and the GPU lock held (see `scripts/reference/README.md`). Check
   `free -g`. Write down the model, quant, prompt length and median.
2. **Find the bound.** Time the kernel:
   `nsys profile -t cuda --stats=true <cmd>` for the timeline and per-kernel
   times; `ncu --kernel-name <entry> --launch-count 1 --set full <cmd>` for one
   kernel (look at DRAM throughput, achieved occupancy, and "local load/store"
   bytes, which mean spills). Spills at compile time: add `-Xptxas -v` to the
   directory's flags locally and read `bytes spill stores`. Compute GB/s = weight bytes ÷ time and compare it to
   ~230 GB/s.
   - Far below → too few blocks/warps, register spills, uncoalesced or
     misaligned loads, or too much decode math per weight.
   - Gaps between kernels in the timeline → launch/host overhead. Use CUDA
     graphs or fuse launches.
3. **Make one change.** Pick it from the Optimization checklist below.
4. **Prove correctness.** Run the kernel's `shiso-kerneltest` case on the
   device, then model parity. Also check that speculative greedy == plain
   greedy when the kernel is on the decode or verify path.
5. **Measure again** with the same protocol. Keep the change only if the median
   improves beyond noise (more than 2%).
6. **Record the number** in a comment (`GB10: 95 → 42 µs`) and put the
   end-to-end before → after tok/s in the commit message, with
   "bit-identical" or "greedy text unchanged".

## Optimization checklist

| Symptom | Fix |
|---|---|
| GEMV well below 230 GB/s | Let a warp own several output rows (1 at ≤4 activation rows, 2 at ≤16, 4 beyond) and walk activations once. Use 4-warp blocks, not 8, to keep more blocks resident. |
| Narrow output (≤ ~1024 cols) on a one-block-per-SM kernel | Use a decode-shaped launch: warp per column group, split-K sized to ~8 warps per SM, last-block fold **in slice order**. |
| Misaligned quant blocks (e.g. 34 B q8_0) | Repack or align the code stream to 16 B for vector loads (+10–25% GB/s). |
| fp16 weights dominate the bytes | Lossless repack (e.g. 12-bit pack); fewer bytes = faster decode. |
| Chain of tiny dependent launches (router → sort → scan) | Fuse them into one block for small passes. Run independent work (the shared expert) on a forked branch. |
| Local-memory traffic / spills (`ncu`, `-Xptxas -v`) | Make every register-array index compile-time. Move per-thread arrays to `__shared__`. Fewer threads per block. |
| Loads stall the arithmetic | Issue all global loads first, then compute. Use `cp.async` double buffering into dynamic smem. |
| CUDA-core matmul inside attention or a recurrence | Rewrite as tensor-core MMA (WMMA / `mma.sync`), with the state held in accumulator registers. |
| Host encoding gaps per step | Capture into a CUDA graph (`--capture-batch`, `--capture-verify`). Kernels must read counts from device buffers, not launch args. |
| One launch per run or sequence | Put all runs in one launch (grid `(heads, runs)`). |
| Long contraction read per 16 rows | Above a width cutoff, send it to cuBLAS (Pinned tier). |
| Unsure which variant/lane width wins | Don't hard-code it. Compile variants and let the startup tuner pick (`shiso-engine/src/tune.rs`). |

Things that were tried and **did not** help are listed in `lessons.md`. Check
there before spending a day on occupancy: occupancy was often not the limiter.

## Hard rules (a violation breaks the answer, not just the speed)

- **Batch and row invariance:** a row's output never depends on how many rows
  share the pass. The path choice (cooperative vs decode-shaped, grouped vs
  not) must not depend on batch composition. Choosing a path by **weight
  shape** (fixed per model) is fine; choosing it by row count or batch size
  is not, unless both paths give identical bits. This is why cuBLAS uses the
  Pinned tier (no split-K) and grouped launches are off for the target model.
- **Fixed-order folds:** no atomic scatter-add, and no compaction by atomic
  arrival order. Use per-slot stores plus a fixed-order combine, or a
  block-wide scan with the lowest index first.
- **Sync your shared memory:** a missing `__syncwarp()` caused nondeterministic
  MoE routing on sm_121. Run `compute-sanitizer --tool racecheck`.
- **Initialize scratch:** an uninitialized tail tile read stale pool memory and
  flipped near-tie argmaxes.
- **Paged KV:** translate every row through the page table. Never assume a
  tile's pages are contiguous. Test with scattered page tables.
- **Fast math:** `--use_fast_math` raised rope error at position 32768 from
  3.4e-4 to 3.0e-3. Directories that need precision drop it in `KERNEL.toml`.
- **Graphs:** no allocation while capturing; no per-layer host sync or id
  readback; the cooperative scratch slab can't grow during capture.
- **Smem opt-in:** check both the thread limit and the launch-time smem refusal.

## Adding a kernel (files to touch)

1. Add the `.cu`/`.cuh` to `kernels/cuda/<dir>/`. Use `extern "C"` entries or
   explicit template instantiations.
2. List it in that directory's `KERNEL.toml` (`sources`, `[modules]`). Unknown
   keys are an error. Keep `arch = ["compute_80"]`: tensor-core paths write
   **zeros** below sm_80.
3. Add the entry to `KERNEL_NAMES.md`. Get mangled template names from the PTX:
   `grep -o '\.visible \.entry [^(]*' <module>.ptx`.
4. Add the engine launch constant and geometry helper (generic code in
   `shiso-engine`, family code in `shiso-models/src/<family>/`). Many arguments
   go into a `...Desc` struct.
5. Write the case in `crates/shiso-kerneltest/src/<family>/`, compared against a
   plain Rust f32 oracle. It must not depend on `shiso-cuda`.
6. Add a thin wrapper in `crates/shiso-cuda/tests/<name>.rs`:
   `#![cfg(target_os = "linux")]`, an `open()` that prints on stderr why it
   skipped, and `#[ignore = "requires a CUDA device"]` (copy `q4e_idx.rs`).
7. If the family also runs on Metal, write its twin with the same entry name.
8. Don't edit `kernels/cuda/mmq` or `kernels/cuda/exl3` without updating the
   file's row in that directory's `VENDOR.md`. Put new work in your own unit.

## Build, benchmark and correctness commands

```sh
# real kernel build (default builds embed an empty registry → first launch refuses)
SHISO_CUDA_COMPILE=1 cargo build --release -p shiso-server --features cuda --bin shiso
# FP4 MMA paths: SHISO_CUDA_ARCH=sm_121a   (plain sm_121 rejects them)

# kernel cases on the device
SHISO_CUDA_COMPILE=1 cargo test -p shiso-cuda --release --test q4e_idx -- --ignored --nocapture

# prefill slope at 64/512/2048 tokens; aggregate decode at 1/2/4/8 streams
SHISO_CUDA_COMPILE=1 cargo run --release -p shiso-bench --bin prefill_sweep -- <ckpt> [config]
SHISO_CUDA_COMPILE=1 cargo run --release -p shiso-bench --bin streams -- ...

# model parity (goldens = regression lock only; references/ = truth)
cargo run -p shiso-bench --release --bin parity -- --model M --backend cuda \
  --vectors test-vectors/<file>.vec [--case ID] [--chunk N] [--dump-rows DIR] [--dump-layers DIR]
SHISO_TEST_MODEL=... cargo test -p shiso-bench --release --test parity -- --ignored --nocapture

# perf gate: record on the pinned GB10 only; check is arithmetic
cargo run --release -p shiso-bench --bin perf_gate -- record --target cuda --model X.gguf --config cfg.json
cargo run -p shiso-bench --bin perf_gate -- check   # 1 = regression (>5%), 2 = re-record, NOT a regression

# fault hunting
CUDA_LAUNCH_BLOCKING=1 ...                                   # blame the right kernel
compute-sanitizer --tool memcheck|racecheck|initcheck ...     # each found real bugs here

bash scripts/check.sh          # the gate, before you say "done"
```

The protocol discards a warm-up, drops the first decode step, reports the
**median**, and refuses runs with more than 2% spread. Do the same by hand.
Stop servers with `pkill -x shiso` before benchmarking.

Only the user knows which machine is pinned. If you are not told you are on
it, don't `record`; report your ad-hoc numbers and say the record is stale.

To find a diverging layer, use `--dump-layers` with
`scripts/reference/bisect_layers.py` and `compare_rows.py`.

## Common mistakes

- Saying a device test passed when it was skipped, or when it ran without
  `SHISO_CUDA_COMPILE=1` (no kernels). A skip looks like a pass.
- Treating `test-vectors/goldens/` as truth. They only lock yesterday's answer.
- Optimizing for occupancy without measuring. Several occupancy changes here
  gained nothing.
- Choosing a faster path only for some batch sizes, which makes two identical
  requests diverge.
- Reading free memory from `cuMemGetInfo` alone; on unified memory use the
  ledger (`--gpu-memory-utilization`). A pool that doesn't fit is refused.
- Putting a model family name in `shiso-core`/`shiso-engine`/`shiso-kvstore`,
  or upstream project names in code or comments.
- Adding a scattered `std::env::var`. Knobs are clap flags with `SHISO_*` env
  vars in `crates/shiso-server/src/cli.rs`.
- Using `#[allow]`, `println!`, `unwrap()`, or `unsafe` without `// SAFETY:`.
