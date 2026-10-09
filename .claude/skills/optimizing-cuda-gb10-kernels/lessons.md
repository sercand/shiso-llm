# CUDA lessons measured on GB10 in this repo

Each lesson below gives the commit or file it came from and the measured
numbers. Use them to choose what to try first, then re-measure on your own
shape.

## Decode GEMV

- **Multiple rows per warp** (e1a9971)
  - A warp owns R output rows: 1 at 1–4 activation rows, 2 at up to 16, 4 beyond. It walks the activations once.
  - A 20-row verify went from 95 to 42 µs on down, 119 to 65 on up, and 53 to 26 on router.
  - Batch-4 decode went from 96–110 to 118–120 tok/s.
  - Bit-identical, because every accumulator adds in the same order.
- **4-warp blocks instead of 8** for one-row-per-warp kernels: more blocks stay resident. 28 to 25 µs (`sparse/kernels.rs:~968`).
- **Decode-shaped launch for narrow outputs** (8961c18, `exl3_dgemv.cuh`, `sparse/matmul.rs:~790`)
  - Problem: a cooperative one-block-per-SM GEMM reached only 76–166 GB/s on narrow outputs.
  - Fix: a warp owns an n16 column group, split-K targets about 384 warps, and the last block folds the partial sums in slice order.
  - Decode went from 27.0 to 30.0 tok/s.
  - The cutoff is `TRELLIS_DECODE_NARROW = 1024`.
- **256-wide tile for long contractions from K ≥ 4096** instead of 8192 (4914830): 65 to 56 µs.
- **Lock chains have a limit.** Past about 4 blocks per output column, the lock chain costs more than the extra blocks gain. 48 blocks ran 3× slower than 8 (`exl3.cu:~212`).
- **Repack fp16 weights losslessly to 12 bits** (6a4f498): decode went from 31.8 to 33.0 tok/s. Bytes matter most.
- **Align q8_0 streams to 16 B.** A q8_0 block is 34 B, so its natural alignment is only 2 B. Aligning gave 217→235, 172→218, 199→230 and 224→243 GB/s (`mmq.cu:~4285`).

## MoE

- **Fuse routing into one block for passes of 32 tokens or fewer** (96f32e2)
  - Router, counting sort and inverse sort became one block: 7 dependent launches became 1.
  - The shared expert runs on a forked graph branch.
  - Decode went from 30.0 to 30.6 tok/s.
- **Make register-array indices compile-time** (d5f2a6a)
  - Run-time indices spilled the logits to local memory, costing 14.8 µs.
  - Issue all loads before the arithmetic that needs them.
  - Decode went from 30.6 to 31.2 tok/s.
- **Watch the register budget at 1024 threads** (5d0adcd). Each thread gets only 64 registers. Moving `chosen_w[16]` and `chosen_i[16]` into `__shared__`, written by lane 0, removed the spills.
- **Split the prefill expert stage into prep, gateup and down launches**: one block per (expert, 128-column tile) over 64-row passes. 28 to 9.4 ms per layer at a 2048-row chunk (`sparse/moe.rs:~566`).
- **Tried and slower** (`exl3_moe_ws.cuh:~34`). The baseline is 9.4 ms; the gap to the 5.4 ms weight floor is issue latency, not occupancy.
  - 32-row passes with 2 blocks per SM: 12.6 ms.
  - KT = 32: 10.6 ms.
  - Splitting an expert across blocks: 10.3 ms.
  - Fusing the Hadamard (218 registers): 10.9 ms.
- **64-row MMQ tiles** (2 CTAs per SM) gave no prefill gain. Occupancy was not the limiter (`mmq.cuh:~106`).
- **Give each expert slot its own block group in MoE mgemm.** 10 groups of 4 were 23% faster than 6 groups of 8 (`exl3.cu:~258`).
- **Use a separate decode path for one-token decode.** On the weight-stationary grid it took 2 waves and reached only 55% of bandwidth (185 µs per layer). `WSD_MAX_ROWS = 32` is the cutoff.

## GDN / linear attention

- **Tensor-core chunked delta rule** (`gdn_tc.cu`)
  - WMMA prep kernel, then a scan with one block per head that holds the 128×128 state in accumulators.
  - 21.3 to 6.4 ms per layer. Relative L2 error stays within 5e-4.
- **Parallel conv** replaced a 3.9 ms-per-layer sequential walk (`recurrent.rs:~224`).
- **One launch for all verify runs** (grid `(n_head_v, runs)`). One launch per run gave 32 blocks on 48 SMs and cost 10 ms per step (`gdn.cu:~284`).
- **Roll back by replay** (113d169). Store the start state plus the per-row inputs instead of one state per drafted row: the slab shrank from depth × 113 MB to ~115 MB.
- **Smem limits** (`q35_gdn.cuh`)
  - 101376 B is the opt-in maximum.
  - A dynamically indexed local array lives in local memory whatever its size.
  - Nine lane-width variants are compiled and the startup tuner picks one.

## Attention (vision tower, 21b3e4f)

These steps were applied in order. Each time is for one 896 px image.

| Change | Time |
|---|---|
| Starting point | 5413 ms |
| `mma.sync` m16n8k16 flash attention, online softmax in base 2 | 358 ms |
| Fused neighbouring elementwise passes | 311 ms |
| `ex2.approx` | 294 ms |
| `cp.async` double-buffered K/V | 239 ms |

## Launch overhead and graphs

- **Capture the batch in a graph** (1df6be4, `--capture-batch`). With 4 sessions the device idled 6 of 58 ms. Aggregate went from 68.3 to 75.7 tok/s.
- **Capture the verify pass** (03eb962, `--capture-verify`). At K=3, code went from 56.8 to 62 tok/s and chat from 42.9 to 47.
- **Read inline instead of spawning a thread per gather** (192b8dd). Spawn plus join cost 470 µs per token against 6 µs of real work. Decode went from 31.1 to 31.8 tok/s.
- **Graph-safe kernels** read counts and lists from device buffers, because a graph replays its launch args verbatim (`kernels/cuda/sample/sample.cu`).

## Speculative decoding

- **Speculation's gain depends on the text.** It measured 0.84× to 1.35× of plain decode, so it is off by default.
- **Narrow the draft vocab head.** The full head streams 477 MB per guess.
- **Invariance costs some speed, and the repo pays it.** Grouped launches reach 29.2 tok/s against 25.4, but they are disabled for the target model to keep the output batch-invariant (`sparse/moe.rs:~525`).

## Prefill

- **Larger solo chunk:** 8192 gives 1580 tok/s against 1000 at 2048, because every pass reads all experts.
- **Small chunks are slow:** 512 rows reached 550–610 tok/s. The scratch cost of large chunks is about 2 GB.
- **Re-reading a 6.5 MB weight every 16 rows** cost 2.5 s of a 10.6 s prefill. Contractions wider than 8192 now go to BLAS.
- **`PREFILL_BLAS_PAD = 256`** won across 64, 512 and 2048 tokens.
- **`PREFILL_GRID = 64`.** Chunk ends must land on the recurrent grid, or the answers depend on where the scheduler cut.

## Memory and loading on unified memory

- **Pageable copies are slow.** Weight upload ran at 0.9 GB/s. Four pinned 64 MB slots with their own streams and 8 filler threads reached 16–35 GB/s.
- **Release file pages after import.** Otherwise 17 s of a 48 s load went to reclaiming page cache.
- **Free memory is `max(cuMemGetInfo free, MemAvailable)`.** The pool is sized from the ledger and refused at boot if it doesn't fit. Evict idle spans before refusing.

## Bugs to not repeat

- **6846cfa:** pages were assumed contiguous, so a recycled page was read from the wrong place.
- **01e6038:** atomic-order top-k compaction made the same prompt produce different text.
- **Atomic scatter-add flipped router near-ties at 26k tokens.** Replaced with per-slot stores and a fixed-order combine.
- **Missing `__syncwarp()`** caused nondeterministic routing (`mmid.cu:~96`).
- **Uninitialized tail tile** (`mmq.cu:~665`).
- **4b4dd05:** two identical requests took different paths at a row bound and diverged.
- **4b802bf:** the cooperative scratch slab was not freed by name at teardown.
- **Tearing down a second backend during another backend's capture** invalidates that capture.
- **A hard-coded lane constant** was the original bug in the GDN tuner.
- **Output gate `sigmoid` vs `silu` mix-up** turned the output into punctuation.
- **Head indexing:** GQA head index is a divide; the delta-rule key head is a modulo.
