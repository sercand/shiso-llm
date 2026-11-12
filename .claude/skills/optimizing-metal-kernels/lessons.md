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

- **GGUF block step tier** (`q4e/ggml.metal`, `q4e_g_step_*`; qwen4exp `UD-IQ3_XXS`, M1 Max, 2026-11-11):
  - Byte-at-a-time loads with a scale multiplied into every weight read 5 GB/s (`IQ4_NL`) and 18 GB/s (`Q6_K`). `packed_uchar4` loads, codes formed four at a time and the block scale applied to the 32-term sum: routed down 1672 -> 377 us, the 248k-row head 2147 -> 1152 us, a plain decode step 3.1 -> 5.3 tok/s.
  - One simdgroup a row leaves a 640-wide row with 12 idle lanes. Few lanes a row and many rows a threadgroup helped the narrow rows; four rows a lane on the dense projections made a step slower (3.7 ms a layer against 2.9).
  - What did **not** move a routed launch: decode tables staged in threadgroup memory, several rows sharing one activation read, spinning on the completion event before blocking.
  - The microbench has a ~100 us floor a launch (`q4_0` 4096x4096 reads 64 GB/s in it, 195 in a model), so it ranks sub-millisecond layouts badly. Rank those on the server's `drain_seconds`.
- **A streamed model's first bottleneck is memory, not kernels:** the expert pool was sized to the driver's working set (28 GB pool + 13 GB stack on a 64 GB machine with 30 GB in other programs), the pool is wired once read, and the machine swapped: 0.6 tok/s. Held to `Backend::host_available` it is 3.2 tok/s before any kernel changed. Check `sysctl vm.swapusage` before profiling.
- **Speculation does not pay when a verify row costs a full row:** three verify rows took 0.53 s against 0.18 s for one plain row, for 2.3 tokens a round. Plain decode was faster (5.3 against 4.3 tok/s).
- **Fusing SwiGLU into the routed gate/up epilogue is bit-identical but SLOWER** (`q4e_g_step_gate_up_act_*`, M1 Max, 2026-11-11, UD-IQ3_XXS decode): 2.52 ms of device time a layer against 2.37 unfused, back to back at 100% pool hits (~6% regression, two fused runs agreed). The unit already holds gate row `o` and up row `o` in one lane, so `silu(g)*u` rides in the store branch — but `precise::exp` there runs only on lane 0 while the other 31 lanes mask off, where the separate `q4e_swiglu_kernel` runs it across all 6400 lanes. The exp stall costs more than the dropped launch + gate/up memory round trip saves. The kernel and its bit-for-bit case (`the_fused_gate_up_activation_matches_the_unfused_chain`) are kept as the regression lock; the decode plan does NOT take it. Do not retry an epilogue activation that calls `precise::exp` from a divergent lane.
- **The dense `Q6_K` step projections are parallelism-bound, not decode-bound** (M1 Max, 2026-11-11): the packed step kernel reads ~45 GB/s at 160-640 threadgroups (2560x2560 .. 10240x2560) but 118-154 GB/s on the 62080-row head (3880 threadgroups). Raising lanes a row (lpr 8→16→32) makes it WORSE (139→160→204 us at 2560x2560): more fold overhead and less per-lane ILP, not more useful parallelism. The generic `kernel_mul_mv_ext_q6_K_f32_r1_1` is only ~6-10% faster at these shapes (128 vs 139, 445 vs 473, 275 vs 306 us) — the "~150 GB/s" ext reputation does not transfer to narrow projections. `packed_lanes` (8 sub-blocks a lane) is a local optimum; do not route these to ext for the bandwidth.

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

- **The processor is a second multiplier on shared memory** (`shiso-metal/src/host_gemm.rs`, `diffusion/ops.rs` `split_columns`, M4 10-core GPU, 2026-11-12): every Metal buffer is shared storage, so the system matrix library can multiply some of a projection's columns in place while the GPU multiplies the rest.
  - The GPU's share needs no kernel change: launch the same entry with the same arguments over fewer tile columns. Its columns are bit-identical to the unsplit launch.
  - The host widens the INT8 codes a 16 MiB piece at a time (`vDSP_vflt8`), multiplies with `cblas_sgemm`, and applies scale/bias/gate as the GPU epilogue does. Order: `sync` (the host reads the input), launch, `submit`, host multiply.
  - Qwen-Image-2.1 INT8, 1024x1024, 6 steps: a step 22.4 -> 16.5 s at a 34% host share, the job 143 -> 110 s. With 16 MiB pieces (see below) shares 22/28/34/38/40/46 gave 18.5/17.5/17.0/17.5/18.7/20.4 s a step: below the optimum the GPU is the long side, above it the host.
  - Effective rates while sharing: GPU 2.9 TFLOPS (3.05 alone), host about 1.5. The split must be a function of shape and the configured share only - a share tuned from timing would make a picture depend on the machine's load.
  - The widening and epilogue passes ran on one core at first and held the host to a fifth of its time; four threads fixed that (38%: 18.2 -> 17.5 s).
  - Do not cut the host's range into cache-sized pieces: the library walks the whole left operand (67 MB at 4096 rows x 4096) once per call. A step at 34%: 19.5 s with 4 MiB pieces, 17.0 s with 16 MiB, 16.4 s with the whole range in one call. The picture was bit-identical at every piece size, so the library's sum for a column does not depend on how many columns share the call.
  - A split measured per shape showed both sides finishing together at 34% for every projection shape of the model, so one share serves all of them.
  - Pictures agree across shares to rounding (58 dB PSNR against the GPU-only picture), not bit for bit: the matrix library sums in its own order.

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
- **`precise::exp` in a softmax costs a third of a flash-attention launch** (`splat/splat.metal`, `kernel_splat_attn_f32`, 12294 rows x 16 heads of 64, f32 throughout, cool):
  - 193 ms with `precise::exp`, 130 ms with the fast `exp` (3.2 -> 4.75 TFLOP/s). The fast form costs nothing measurable: replacing it with a multiply-add gave the same 129 ms.
  - Every argument is a score less its running maximum, so at most zero; -inf (a masked key) still answers zero. The cases' 1e-5 bound against the f64 oracle held, and the denoiser's velocity against the reference did not move (worst 1.3e-4).
  - Two row blocks per simdgroup (each K fragment reused for 16 query rows) spilled registers even with 64-wide heads: 230 ms against 130.
- **Identical rows are one weighted key.** `n` equal keys with equal values weigh `n exp(s) v` in a softmax, which is one key scoring `s + ln(n)`.
  - The splat denoiser's guidance pass has a zero conditioning: 4101 rows that are, and stay, the same row. Holding it once took that pass from 12294 rows to 8194 (attention x0.44, linears x0.67). Its error against the reference did not grow (mean 1.5e-4 of rms, from 2e-4).
  - Look for this before tuning a kernel: it was worth more than any change inside one.
- **Half arithmetic does not buy throughput on an M4** (`kernel_splat_attn_f32`, 12294 rows x 16 heads of 64, M4, 2026-11-11): the f32 kernel runs 207 ms (2.99 TFLOP/s, 82% of the bench's peak). K and V staged as half with f32 accumulation: 216 ms. Every fragment `simdgroup_half8x8`: 199 ms. Four percent for a model's worth of rounding is not a trade; the 2x of the flux2 entry above was the threadgroup staging, not the multiply.
- **When the kernels are at the ALU ceiling, cut rows, not cycles** (splat denoiser, M4, 25 steps of 262144 Gaussians, 2026-11-11). Attention and the GEMMs both sat near 3 TFLOP/s, so a step's 34 TFLOP was 13.2 s whatever the kernel did. What moved it:
  - The latent's refiner reads neither conditioning, so a step's two passes share it: the second copies the first's rows. 0.3 s a step, the same bits.
  - A step between two full ones runs the conditional pass over the latent and the camera alone (8193 of 12294 query rows), attending to the conditioning rows' keys and values as the last full step left them, and adds that step's guidance direction: 6.1 s against 13.2. Every second step held, after two full ones: 362 s -> 278 s, and renders score against a 50-step request's as the exact sampler's do. Holding the second step too sent the flow elsewhere (22.6 dB from the exact 25-step render against 27.7). Holding the guidance direction alone (the conditioning rows still run) was 8.8 s a held step and softened the object more.
  - Judge such a change on renders, not on the latent: the same seed at 25 and at 50 exact steps differs by 0.78 of the latent's norm, so a latent distance cannot tell an approximation from a step count.
- **The KV cache is f16.**

- **Qwen-Image-2.1 on an M4 (10-core GPU, 24 GB) is compute-bound, not kernel-bound** (2026-11-12, 1024x1024, 6-step INT8 checkpoint, per-kernel profile of one job):
  - `kernel_dit_gemm64_i8` + `kernel_dit_gemm64_fused_i8` are 79% of the job and every launch runs at 3.05 TFLOPS (4096x4096x4096 in 45 ms, 24576x4096 in 271 ms); the fused attention is 12% and also at 3.1 TFLOPS (4096 rows x 4119 keys x 32 heads of 128 in 89 ms). That is about 70% of the part's peak. A step is 22.5 s and nothing in it is cheap to move.
  - An edit adds one prefix pass over the reference's tokens (4,117 for a 1024x1024 reference: 21.8 s, the same GEMMs) and doubles the keys of every step's attention (25.3 s a step).
  - Run-to-run thermal drift on this fanless-class load is 21.6 -> 23.0 s a step inside one job: A/B a change with cool-downs and alternating order, or at the kernel level, never from two back-to-back jobs.
- **The fused attention's half staging is not free for a language model** (same day, Qwen3-VL-8B INT8 text encoder, 1,059-token edit prompt): routing the encoder's grouped-query attention through `kernel_dit_flash_sdpa_causal_d128_f32` (keys and values repeated across the query group) took the prefill from 10.3 s to 5.1 s, but the worst token's features fell to cosine 0.9928 against the per-row f32 kernel (mean 0.99999), under the 0.9995 parity bar. Not adopted. The per-row kernel (`kernel_dit_gqa_f32`) costs 150 ms a layer there, 5.4 s over 36 layers, and grows with the square of the prompt; the fix that keeps parity is a fused entry that stages q, k and v in f32 (16 keys a block to stay under 32 KiB of threadgroup memory), not the half one.
- **A cache of keys and values can be stored as the halves the fused attention stages** (`kernel_dit_flash_sdpa_d128_kvhc`): bit-identical to attending the f32 cache, half the bytes, and the cached rows are never narrowed again. It is a memory win (2.15 GB instead of 4.3 GB per 1024x1024 reference), not a measurable speed one: the narrowing it removes was 1 ms a layer.

- **A language model's attention wants the fused tile staged in f32** (`kernel_dit_flash_gqa_d128_f32`, M4, 2026-11-12): the text encoder's causal grouped-query prefill ran a threadgroup per (row, head); at 1060 rows x 16 heads of 128 that was 150 ms a layer, 5.4 s of a one-picture edit.
  - Routing it through the half-staged fused tile was fast (prefill 10.3 -> 5.1 s) and wrong: worst-token cosine against the reference 0.9928. The residual stream is not query-key-normed; half K and V lose it.
  - The same tile with every operand in f32, 16 keys a block (32 f32 keys are 34816 B of the 32768 B of threadgroup memory), kv heads read natively instead of repeated: tower + prefill 12.8 -> 7.5 s, worst-token cosine 0.9999987 (the per-row kernel: 0.9999978).
  - Used from 768 rows up, so a text-only prompt keeps the per-row kernel's bits.

## Fusion and launch count

- **One command buffer with host-handled fetch cuts beat the per-layer drain by ~1.7x** (qwen4exp UD-IQ3_XXS, M1 Max, 2026-11-11, `--spec-k 0`, full pool): plain decode 7.2 -> **12.1 tok/s** (third request, stable across runs), greedy text bit-identical, and still identical with a 512-slot pool that misses every layer. The serial path drains a command buffer at every layer's `ExpertStream::plan` (~48 commit/drain round trips a step, ~0.29 ms each, plus GPU idle while the host re-encodes). The segmented path (`Exec::run_steps_segmented`, Metal `encode_cut`/`commit_segments`/`wait_cut`/`resume_cut`) encodes the WHOLE step into one buffer with a shared-event signal/wait cut before each layer's experts; the host places the experts at the cut (`fetch_at_cut`, non-draining shared IO) and releases the GPU. Ported from Strata's `mtl_rt.mm::run_graph_seg`. Device time fell 114 -> 76 ms/step: continuous execution keeps the GPU clock up where the per-layer gaps dropped it.
  - **`wait_cut` must BUSY-SPIN on `signaledValue()`, not block.** A blocking `waitUntilSignaledValue_timeoutMS` wake-up lands on every cut and took decode to 2.8 tok/s; the lock-free spin (Strata does the same) is what makes it fast. Release the backend lock while spinning so read-ahead workers are not blocked.
  - **Pool size confounds the measurement.** A reference server left running hogged RAM, shrank the pool to 2000 slots (66% hits) and made the segmented path look slow (2.9 tok/s). Check `slots=` and the hit rate in the stream log before judging a decode change; compare at equal pool size.
  - The n-gram rows land up front in the segmented path (they depend only on the input tokens), losing the prologue overlap the serial path uses; correctness is unchanged.

- **Batching:** every launch goes into one open command buffer and encoder. Only `sync` and host copies flush.
  - Waiting on an `MTLSharedEvent` instead of `waitUntilCompleted` took 145 µs against 210 µs per round trip (4ebe5a0).
- **Clears:** `fill_zero` runs as a compute kernel, because a blit closes the encoder.
- **flux2 fused GEMM:**
  - It builds the `[dense | silu(g)*u]` input while staging and applies the gated residual in the epilogue. One launch replaces swiglu, concat, linear and add.
  - Together with skipping an unused preview decode, FLUX.2 512² at 4 steps went **165 s → ~16 s**, within 1 LSB of the old output.
- **A rotation carried by a second projection is a seventh GEMM** (`kernel_splat_rope_norm_f32`): the splat denoiser turned q and k by per-token angles as `q cos + turn(q) sin`, with `turn(q)` from a permuted copy of the weight. One elementwise launch that turns and head-norms q and k in place replaced 2 of a block's 7 width-squared GEMMs, the sine and cosine tables, six gated adds and two RMS norms: 811 -> 551 launches a pass.
- **Audio decoder fused entries:** `add_rms_norm`, `qk_norm_rope_append`, `matvec3` and `matvec_silu` (`shiso-models/src/audio_ops/fused.rs`).
- **MoE routing:** route, sort and inverse sort run in one threadgroup, bit-identical to the four-launch version with 1/6 of the dependent launches.
  - The prefix-scan sort replaced a 60 µs-per-layer serial walk.
- **Group norm:** one threadgroup per group took 70 ms per norm. Slicing it and combining with Chan's formula in slice order fixed it.

## Precision

- Activations are f32.
  - bf16 rows have 2.0e-3 relative error against 7e-7 for f32. If bf16 ever looks *tighter* than f32, suspect the f32 path.
  - Q8_0 staged through half gave 1.1e-2 error against 2.3e-5 through f32.

## Loading

- **A checkpoint's tensors may all be off the vector boundary, and then every import is a copy** (`shiso-metal` `import_raw_bytes`, `diffusion/weights.rs` `import_bytes`, M4, 2026-11-12): a safetensors header whose length is 8 mod 16 puts every tensor 8 bytes off a 16-byte boundary, and the no-copy wrap refuses those because float kernels load 16-byte vectors. Both Qwen-Image-2.1 INT8 exports are such, so 7 GB of codes were copied at every load.
  - INT8 codes are read through `packed_char4` loads, which no address misaligns, so they are now wrapped wherever they lie. Transformer load 1.0-1.2 s -> 48 ms, text encoder 0.9 s -> 3-8 ms, server boot 12.4 -> 4.5 s.
  - The copy also held the weights twice (private buffer plus page cache), which pushed the other swapped component out of the page cache: its prefetch read from the SSD at 3.9 GB/s (1.8 s). Wrapped, both files stay cached and the prefetch runs at 30 GB/s (0.23 s).
  - Small per-job imports from a file the swap traffic evicts fault in a page at a time: the VAE encoder's 311 MB took 1.9 s to load until its byte ranges were read ahead (0.2 s).

## Tuning

- **Startup tuner** (`shiso-engine/src/tune.rs`, `shiso-models/src/qwen35/kernels/gdn_tune.rs`):
  - It compiles several lane widths, probes each pipeline's own thread limit, and times trial dispatches. Differences under the noise are treated as ties.
  - Every variant must give identical bits.
  - `SHISO_GDN_VARIANT` pins a variant.
- **Intuition is often wrong:** the threadgroup-staged `gdn_recurrent_hybrid` beat the register-resident variants at 32 tokens (782 vs 945 µs). Measure.

## Bugs to not repeat

- **Fast math refolds one body differently per instantiation** (M4, 2026-11-03):
  the two-rows-a-lane K-quant matvec matched the one-row walk bit for bit at
  `q4_K` but landed one ulp off at `q5_K`/`q6_K`, and the `q6_K` body at two
  token rows differed from itself at one. Whole-template `math_mode(safe)` fixed
  the bits and cost 30% (373 vs 286 us, four rows), because the inlined decode
  lost its contractions. What worked: only the dot/accumulate chain as tiny
  safe-mode helpers (`shs_kq_dot8`, `shs_kq_acc`), the decode left fast. The
  per-instruction flags survive inlining.
- **A GEMM fixture without the output guard writes past the matrix**: the
  broadcast-off constant table (701 = false) stores all 64 rows of a partial
  tile. The engine runs 701 = true and 8192 bytes of threadgroup memory; a case
  with a partial tile needs both, or it corrupts the next token's rows and,
  through out-of-bounds stores, the buffers of tests running beside it - which
  looks like a race in unrelated kernels.
- **Batched decode paid the vocabulary head once a row** (M4): `decode_batch`
  projected logits row by row, 5.3 ms each for a `q6_K` head. One projection
  over the batch's rows reads the matrix once.

- **950a9bf:**
  - A pipeline reported 448 max threads, not the 512 it was written for, so the verify pass was refused. Probe per pipeline.
  - Scattered (uncoalesced) stores took 417 µs per layer, 19 ms of a 145 ms step.
- **653f14f / 735d6c9:** a zero-sentinel "absent bias" dropped real biases whose first value was +0.0.
- **d47171a:** asking for CUDA-sized smem on a Metal kernel that declares none; resolving templates by `host_name`; copying imports whose start isn't 16 B aligned.
- **Over-limit smem:** a kernel that needed 33924 B (over the 32 KiB limit) "happened to run" until `run.rs` started refusing it.
