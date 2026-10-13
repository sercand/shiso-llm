# Metal lessons measured in this repo

Every number here was measured on an M1 Max unless the entry says otherwise.
Use them to choose what to try first. Re-measure on your own case, because
shapes differ.

## Quantized matvec (decode)

- **Factor the scales out of the inner loop** (c182238, `generic/mul_mv_ext.metal`,
  `qwen35/gemv_ext_q5k_q6k.metal`).
  - Before: the kernel unpacked a float4 per turn and re-derived the scale and min every 4 weights. It reached 6 GB/s on q6_K and ~60 GB/s on q4_K, against 195 GB/s on q4_0.
  - After: a lane takes a 32-weight sub-block, unpacks the codes once, and applies `dl*sum(q*y) - ml*sum(y)`. q6_K reached ~150 GB/s and q4_K ~140 GB/s.
  - Qwen3.5-4B Q4_K_M decode went **5.6 → 38 tok/s**.
  - The grid, args and lane split did not change.
- **The affine (scale + bias per group of 64) case uses the same trick:** `scale*sum(q*x) + bias*sum(x)` (`dense/dense_affine.metal`).
- **Lane-split table** (`shiso-engine/src/launch.rs:~1103`):
  - Lanes per row are 16 when `in_dim % 256 == 0` and there are fewer than 3 tokens, 8 when `% 128`, otherwise 4.
  - 1–5 token rows go per threadgroup, and 2 simdgroups.
- **Activation reuse:** 8 token columns per threadgroup gave 16 → 40 tok/s on a 64-token prefill. 16 columns was no better, and 8 accumulators still fit in registers.
- **Too many rows per lane hurts:** 8 rows per lane cost occupancy and took 15.6 ms, against 11.1 ms with 4 (`audio/audio_common.h`).
- **8 consecutive weights per lane (16 B loads)** reached 330–410 GB/s against ~150. The 151936-wide vocab head ran at 355 against 175 GB/s (`audio/audio_gemm.metal`).
- **Matvec re-reads the matrix per row block,** so 64 prompt tokens read every weight 16×. Above `GEMM_MIN_TOKENS` (`projection.rs:~886`) use a GEMM.

## Prefill GEMM (simdgroup matrices)

- **64×64 tile layout** (6357b14, `flux2/flux2.metal:~955`, `audio/audio_gemm.metal`):
  - 128 threads in 4 simdgroups arranged 2×2, each holding 4×4 `simdgroup_float8x8` accumulators.
  - k is staged 16 deep, rows padded +4 floats against bank conflicts, 9.25 KiB of threadgroup memory.
  - The next slice is loaded into registers after the current MMAs.
  - Result: ~4.2 TFLOPS, against a few hundred GFLOP/s for a scalar 16×16 tile.
- **Variants that were slower:** 128-row tiles, 128-column tiles, double buffering, 8-deep k, extra prefetch, and 8 simdgroups at 32×16 with 20 KiB staging (27.7 vs 22.9 ms).
  - Carrying ragged-edge checks in the fast tile cost 8–14%, so ragged shapes go to a separate kernel.
- **Without a full unroll of the fragment loops,** the fragments live in memory and the kernel runs 6× slower.
- **Gate the tile on shape** (e.g. `k % 16 == 0`) and keep a scalar fallback.
- **Widen weights while staging:** bf16 via `as_type<float>(bits << 16)`. Decode FP8 in registers and apply its scale in the epilogue.
- **Quantized prompt GEMM** (4ebe5a0, `q4e/affine.metal`):
  - The general tile is 32×32 with 4 simdgroups of 16×16, dequantizing into threadgroup memory.
  - Routed MoE experts get only about a dozen columns per chunk, so their tile is 16 columns × 64 rows. A 32-wide tile wasted 2/3 of its MMAs on zeros.
- **The quantized prefill tile (`qwen35/gemm_q4k.metal`, 64 rows x 32 tokens, 4 simdgroups) is a local optimum on the M1 Max** (2026-10-13, d1-3B shapes, interleaved A/B in one process, outputs bit-identical across widths):
  - 64 tokens per threadgroup (each simdgroup 32x32, 16 accumulators) was slower or equal everywhere: `ffn_down` 10752->2048 at 64 tokens went 0.69 -> 0.94 ms. Half the threadgroups cost more occupancy than the doubled weight reuse saved.
  - 16 tokens per threadgroup was slower everywhere (0.54 -> 0.64 ms at 39 tokens, up to 60% slower at 154): every dequantized weight is reused across fewer tokens.
  - What it reaches: q8_0 5.2-5.9 TFLOP/s, q4_K ~15% less (costlier unpack), narrow outputs (2048 rows, 64 threadgroups at <= 64 tokens) 3.5-4.1.
- **Implicit-GEMM convolution:** gather im2col inside the operand reader and never materialize it. VAE decode went **97 s → 2.8 s**.
- **Reader and epilogue structs** let one tile body serve linear, convolution and fused variants.

## Attention

- **Single-pass flash attention** (`flux2.metal:~1286`):
  - 32 query rows per threadgroup, keys streamed 32 at a time, softmax state kept in registers.
  - Staging Q/K/V/P as **half** with f32 accumulation took 115 ms against 228 ms per layer.
- **Decode attention:**
  - Split K across simdgroups and finish with a merge kernel.
  - Each lane owns 4 consecutive dims, so a 128-dim row is one 512 B read.
  - Fast `exp` took 66 µs against 153 µs.
  - 8 rows × 4 heads per block spilled registers and ran 1.7× slower.
- **Window attention:** 64 columns × 16 keys took 14.7 ms. 32 keys took 27.5 ms (lost occupancy), and staging K/V in threadgroup memory took 18.6 ms (4 barriers per block).
- **A 64-bit integer divide** is a software routine of ~3000 cycles per key. Paged-KV math uses 32-bit.
- **The KV cache is f16.**

## Fusion and launch count

- **Batching:** every launch goes into one open command buffer and encoder. Only `sync` and host copies flush.
  - Waiting on an `MTLSharedEvent` instead of `waitUntilCompleted` took 145 µs against 210 µs per round trip (4ebe5a0).
- **Clears:** `fill_zero` runs as a compute kernel, because a blit closes the encoder.
- **flux2 fused GEMM:**
  - It builds the `[dense | silu(g)*u]` input while staging and applies the gated residual in the epilogue. One launch replaces swiglu, concat, linear and add.
  - Together with skipping an unused preview decode, FLUX.2 512² at 4 steps went **165 s → ~16 s**, within 1 LSB of the old output.
- **Audio decoder fused entries:** `add_rms_norm`, `qk_norm_rope_append`, `matvec3` and `matvec_silu` (`shiso-models/src/audio_ops/fused.rs`).
- **MoE routing:** route, sort and inverse sort run in one threadgroup, bit-identical to the four-launch version with 1/6 of the dependent launches.
  - The prefix-scan sort replaced a 60 µs-per-layer serial walk.
- **Group norm:** one threadgroup per group took 70 ms per norm. Slicing it and combining with Chan's formula in slice order fixed it.

## Precision

- Activations are f32.
  - bf16 rows have 2.0e-3 relative error against 7e-7 for f32. If bf16 ever looks *tighter* than f32, suspect the f32 path.
  - Q8_0 staged through half gave 1.1e-2 error against 2.3e-5 through f32.

## Tuning

- **Startup tuner** (`shiso-engine/src/tune.rs`, `shiso-models/src/qwen35/kernels/gdn_tune.rs`):
  - It compiles several lane widths, probes each pipeline's own thread limit, and times trial dispatches. Differences under the noise are treated as ties.
  - Every variant must give identical bits.
  - `SHISO_GDN_VARIANT` pins a variant.
- **Intuition is often wrong:** the threadgroup-staged `gdn_recurrent_hybrid` beat the register-resident variants at 32 tokens (782 vs 945 µs). Measure.

## Bugs to not repeat

- **950a9bf:**
  - A pipeline reported 448 max threads, not the 512 it was written for, so the verify pass was refused. Probe per pipeline.
  - Scattered (uncoalesced) stores took 417 µs per layer, 19 ms of a 145 ms step.
- **653f14f / 735d6c9:** a zero-sentinel "absent bias" dropped real biases whose first value was +0.0.
- **d47171a:** asking for CUDA-sized smem on a Metal kernel that declares none; resolving templates by `host_name`; copying imports whose start isn't 16 B aligned.
- **Over-limit smem:** a kernel that needed 33924 B (over the 32 KiB limit) "happened to run" until `run.rs` started refusing it.
