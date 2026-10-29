# dense Metal kernel contract

The dispatch contract for the engine: entry-point names, argument blocks, binding
order, launch geometry and buffer layouts. Where this file and a kernel body
disagree, the kernel is right and this file is the bug.

## Conventions

- `[[buffer(0)]]` is the argument block (`constant dense_args_* &`, or the
  generic `shs_metal_args_*` block where a dense kernel is the parametric twin of
  a generic one) for every entry here EXCEPT the four affine matmuls. Buffers
  follow in declaration order at `[[buffer(1)]]`, `[[buffer(2)]]`... and the
  runtime binds the i-th `KernelArg` to binding index i, so the buffer lists
  below are the bind order, in order.
- **The exception, and it is a silent one.** `kernel_dense_affine_qmv_4_f32`,
  `kernel_dense_affine_qmv_8_f32`, `kernel_dense_affine_qmm_4_f32` and
  `kernel_dense_affine_qmm_8_f32` carry explicit `[[buffer(N)]]` attributes in
  `dense_affine.metal` that bind `w, scales, biases, x, y` at 0..=4 and
  `dense_args_affine` LAST, at `[[buffer(5)]]`. They are the four ported from
  mlx's `quantized.h`, and they kept that file's order. A twin written from the
  convention above rather than from the body would shift every parameter by one
  and read the argument block as a weight pointer. `kernel_dense_embed_affine4_f32`
  and `kernel_dense_ple_prepare_affine4` are NOT affected - they take the block
  first, as everything else does. This paragraph exists because an earlier
  revision of this file stated the convention without the exception, and the
  first twin written against it had to be found by disagreement rather than by
  reading. Threadgroup memory
  is bound separately and never occupies an argument index.
- `grid` in a launch is the threadgroup count, dispatched as given: no division,
  no rounding. `block` is the threads-per-threadgroup the kernel reads as
  `threads_per_threadgroup`.
- Reductions only go through `dense_warp_sum`/`dense_block_sum` (`dense_common.h`),
  whose shape is copied from `metal/qwen35/q35_common.h`. Bare `simd_sum` has an
  unspecified summation order, and a golden that pins a decode step pins the
  order its partial sums were added in. Inner products fold through `dense_fma`
  for the same reason, and it is `q35_fma` under another name.
- Dimensions are arguments, never constants. Nothing in this set assumes a hidden
  width, a head count or a head dimension: "dense" is the parametric family, and
  a second checkpoint of the same shape is a different set of numbers rather than
  a second set of kernels.
- Every activation buffer below is f32.

## Geometry

`n` is the element count the flat kernels cover, `n_tok` the rows of the chunk.

| Entry point | Buffers after the args | grid | block | smem |
|---|---|---|---|---|
| `kernel_dense_geglu_flat_f32` | `gate, up, dst` | `ceil(ne0/256)` | 256 | 0 |
| `kernel_dense_add_scale_rms_norm_mul_f32_4` | `src0, src1, weight, sum_dst, norm_dst, scale` | `n_tok` | 32 | 128 |
| `kernel_dense_softcap_f32` | `x` | `ceil(n/256)` | 256 | 0 |
| `kernel_dense_scale_f32` | `x` | `ceil(n/256)` | 256 | 0 |
| `kernel_dense_q_norm_rope` | `q_out, q_in, w, pos` | `(n_head, n_tok)` | 128 | 2180 |
| `kernel_dense_kv_norm_rope_store` | `k_cache, v_cache, k_in, v_in, kw, pos, pages` | `(n_head_kv, n_tok)` | 128 | 2180 |
| `kernel_dense_q_rope` | `q_out, q_in, pos` | `(n_head, n_tok)` | 128 | 2180 |
| `kernel_dense_kv_rope_store` | `k_cache, v_cache, k_in, v_in, pos, pages` | `(n_head_kv, n_tok)` | 128 | 2180 |
| `kernel_dense_attn_decode` | `out, q, k_cache, v_cache, pos, bound, pages` | `(n_head, n_tok)` | 256 | 16448 |
| `kernel_dense_attn_prefill` | `out, q, k_cache, v_cache, pos, bound, pages` | `(n_head, ceil(n_tok/16))` | 256 | 16384 |
| `kernel_dense_attn_flash_hd64`, `kernel_dense_attn_flash_hd128` | `out, q, k_cache, v_cache, pos, bound, pages` | `(n_head, ceil(n_tok/64))` | 256 | 0 (static) |
| `kernel_dense_ple_prepare_q6_K` | `out, proj, norm_w, rows, ids` | `(n_layers, n_tok)` | 128 | 1156 |
| `kernel_dense_ple_gate_mul` | `gate, ple` | `ceil(n_tok*dim/256)` | 256 | 0 |
| `kernel_dense_affine_qmv_4_f32` | `w, scales, biases, x, y`, then the args block LAST | `out_dim` | 32 | 0 |
| `kernel_dense_affine_qmv_8_f32` | `w, scales, biases, x, y`, then the args block LAST | `out_dim` | 32 | 0 |
| `kernel_dense_affine_qmm_4_f32` | `w, scales, biases, x, y`, then the args block LAST | `(out_dim, n_tok)` | 32 | 0 |
| `kernel_dense_affine_qmm_8_f32` | `w, scales, biases, x, y`, then the args block LAST | `(out_dim, n_tok)` | 32 | 0 |
| `kernel_dense_embed_affine4_f32` | `out, words, scales, biases, ids` | `(ceil(hidden/64), n_tok)` | 64 | 0 |
| `kernel_dense_layer_norm_mul_f32` | `src, w, dst` | `rows` | 256 | 0 (static, 132 B) |
| `kernel_dense_experts_iq4nl_f32` | `blocks, map, input, sorted, bounds, active, out` | `(ceil(out_dim/16), min(columns, n_expert))` | 128 | 0 |
| `kernel_dense_experts_iq4nl_gemm_f32` | `blocks, map, input, sorted, bounds, active, out` | `(ceil(out_dim/64), min(columns, n_expert))` | 128 | 0 (static, 11.5 KiB) |
| `kernel_dense_ple_prepare_affine4` | `out, proj, norm_w, rows, ids` | `(n_layers, n_tok)` | 128 | 1156 |

The four attention entries take their threadgroup memory from the compile-time
strides in `dense_common.h`, never from `head_dim`: the norm pair asks for
`4 * (DENSE_BLOCK_SUM_SLOTS + DENSE_ATTN_HD)` = `4 * (33 + 512)` bytes, the decode
softmax for `DENSE_ATTN_WARPS * DENSE_ATTN_HD * 4 + 2 * DENSE_ATTN_WARPS * 4`
= `8 * 512 * 4 + 64`, and the prefill softmax for `DENSE_ATTN_HD * DENSE_ATTN_KT * 2`
= `512 * 16 * 2`. That is what bounds the set: `head_dim` may not exceed
`DENSE_ATTN_HD` (512). `kernel_dense_attn_prefill` additionally requires
`head_dim % 32 == 0`, because its final store is unguarded and the 32 lanes cover
a row between them; `kernel_dense_attn_decode` has no such requirement, since it
guards every access with `i < head_dim` and rounds its slot count up. All three
fit this device's 32 KiB per-threadgroup limit with room to spare, and the
engine's mirrors are
`DENSE_NORM_ROPE_SMEM_BYTES`, `DENSE_ATTN_DECODE_SMEM_BYTES` and
`DENSE_ATTN_PREFILL_SMEM_BYTES` in `crates/shiso-engine/src/launch.rs`.

The prefill entry dispatches one threadgroup per `DENSE_ATTN_PREFILL_TILE` (16)
queries and walks keys `DENSE_ATTN_KT` (16) at a time, one key per lane of a
simdgroup's low half. The engine picks it over the decode entry at `n_tok >= 16`:
below one tile it would stage a key tile for tokens a single decode threadgroup
covers in one pass.

`kernel_dense_add_scale_rms_norm_mul_f32_4` takes its scale as `constant float &`
at binding index 6, ahead of the threadgroup allocation: the argument block it
shares is the generic norm's, which every other norm entry in the library also
reads, so the one field this variant adds rides beside it instead of in it. Its
threadgroup memory is one f32 per lane (`32 * 4` bytes), the same reduction
scratch `kernel_add_rms_norm_mul_f32_4` asks for; the block is 32 threads because
the work loop advances by `threads_per_threadgroup.x` alone and never reads its
simdgroup index, so a taller threadgroup would have every extra simdgroup
recompute and rewrite the same row.

## Argument blocks

Field order is the struct order in `dense_common.h`; the engine mirrors it field
for field in `crates/shiso-engine/src/launch.rs`.

| Struct | Fields | Engine mirror |
|---|---|---|
| `dense_args_scale` | `uint32_t n; float s;` | `ScaleArgs`, 8 bytes |
| `dense_args_softcap` | `uint32_t n; float cap;` | `SoftcapArgs`, 8 bytes |
| `dense_args_q_norm_rope` | `head_dim, n_head, n_rot, freq_dims, n_tok, rope_base, eps` | `DenseQNormRopeArgs`, 28 bytes |
| `dense_args_kv_norm_rope` | `head_dim, n_head_kv, n_rot, freq_dims, n_tok, rope_base, eps, page_tokens, v_norm, page_stride` | `DenseKvNormRopeArgs`, 40 bytes |
| `dense_args_attn` | `head_dim, n_head, n_head_kv, n_tok, page_tokens, window, scale, page_stride` | `DenseAttnArgs`, 32 bytes |
| `dense_args_ple` | `n_tok, n_layers, dim, row_bytes, resident, eps, proj_scale, embed_scale, mix_scale` | `DensePleArgs`, 36 bytes |
| `dense_args_ple_gate` | `n_tok, n_layers, dim, layer` | `DensePleGateArgs`, 16 bytes |
| `shs_metal_args_glu` (`metal/generic/glu.metal`) | `ne00, nb01, ne10, nb11, ne0, nb1, i00, i10, alpha, limit` | `GluArgs`, 64 bytes |
| `shs_metal_args_norm` (`metal/generic/norm.metal`) | `ne00, ne00_t, nb1, nb2, nb3, eps, nef1..3, nbf1..3` | `NormArgs`, 144 bytes |

## Formulas

- `kernel_dense_geglu_flat_f32`: `dst[i] = dense_gelu_tanh(gate[i]) * up[i] * alpha`,
  with the same clamp pair `kernel_swiglu_flat_f32` applies when `limit` is above
  its epsilon. `dense_gelu_tanh(x)` is `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`
  — the tanh approximation, which differs from the exact GELU by about 1e-4 per
  value, so the two are separate entry points rather than one with a flag.
- `kernel_dense_add_scale_rms_norm_mul_f32_4`: `sum = (src0 + src1) * scale`, then
  `norm = sum * rsqrt(mean(sum^2) + eps) * weight`. The scale is folded in before
  the store AND before the sum of squares; that is the only difference from
  `kernel_add_rms_norm_mul_f32_4`.
- `kernel_dense_softcap_f32`: `x[i] = cap * tanh(x[i] / cap)`, in place.
- `kernel_dense_scale_f32`: `x[i] *= s`, in place.
- `kernel_dense_q_norm_rope`: per head, `inv = rsqrt(mean(x^2) + eps)`, then
  `x[i] * inv * w[i]`, then the rotation below. The weight multiplies the
  normalized value directly; this family stores the weight rather than a
  deviation from one.
- The rotation (`dense_rope_pairs`): with `half = freq_dims / 2`, every `i` below
  `n_rot / 2` rotates the pair `(v[i], v[i + half])` by
  `theta = pos * base^(-2i / freq_dims)`. The frequency width and the rotated
  width are separate arguments: at `head_dim` 512, `n_rot` 128 and `freq_dims`
  512 the pairs are `(i, i + 256)` for `i < 64`, where a kernel that read the
  pair stride off `n_rot` would pair `(i, i + 64)` — every head still normalized,
  still rotated, and wrong. Setting `freq_dims = n_rot` recovers the neox
  rotation `q35_rope_neox` performs. `pow`, `sin` and `cos` are the accurate
  builtins, as they are there. The helper requires `freq_dims >= n_rot`: below
  that the partner index `i + freq_dims / 2` lands back inside the loop's own
  range and two lanes race over one dimension. The engine validates it on the
  host, because a kernel cannot refuse.
- `kernel_dense_kv_norm_rope_store`: the key is the same weighted norm and the
  same rotation; the value is `v * rsqrt(mean(v^2) + eps)` when `v_norm` is
  non-zero and raw otherwise, with NO weight either way. Both halves are written
  as `half` at `dense_kv_row(dense_row_pages(pages, t, page_stride), pos[t], page_tokens)`, which is the
  translation `q35_kv_row` performs.
- `kernel_dense_q_rope` and `kernel_dense_kv_rope_store`: the norm pair with the
  norm taken out, for a slot that rotates its queries and keys and normalizes
  neither. The query is the rotation above and nothing else; the key is the same
  rotation and the value is stored raw, both as `half` at the row the norm
  pair's store uses. They read the norm pair's argument blocks and leave `eps`
  and `v_norm` unread: the engine refuses a value norm on a slot with no key
  norm. They stage the head at the same offset as the norm pair, so the launch
  asks for the same 2180 bytes. A unit weight on the norm pair is not this
  pair, because the division by the head's root mean square is unconditional.
- `kernel_dense_attn_decode` and `kernel_dense_attn_prefill`: a query at `pos[t]`
  reads keys `p_lo..=p_hi`, where
  `p_lo = (window != 0 && pos + 1 > window) ? pos + 1 - window : 0` and
  `p_hi = bound[t]` inclusive. The score is `dot(q, k) * scale` — the scale is
  applied to the reduced score, so the number in the argument block is the number
  the softmax sees — and `hkv = h / (n_head / n_head_kv)`. The bound is not the
  query's own position: a bidirectional prefix reaches past it, and neither end
  of the span is derivable from the other. An empty span (a bound below the
  window's low end) contributes a row of zeros rather than a row of NaN.
  The prefill entry stages the transposed key tile as `half` and masks each query
  by its own span inside a tile whose range is the union of the tile's spans.
- `kernel_dense_attn_flash_hd64` and `kernel_dense_attn_flash_hd128`: the same
  span, scale and empty-span contract as the prefill entry, on simdgroup
  matrices. A threadgroup takes 64 query rows of one head as 8x8 fragments and
  streams the union of their spans through a static threadgroup array 32 keys
  (hd64) or 16 keys (hd128) at a time, keys and values widened to f32, with an
  online softmax per row: `2 x KB x (HD + 4) x 4` bytes, under 18 KiB at both
  widths. Everything is f32 from the cache on, so a score is the prefill
  entry's dot summed in another order. The engine dispatches it for a chunk
  of 16 tokens or more at those two head widths, after asking the device for
  it (`max_block_threads`); a device without it runs the prefill entry.
- `kernel_dense_ple_prepare_q6_K`: per (token `t`, layer slot `l`), with
  `row = resident ? ids[t] : t`, `p_i = proj[t, l*dim + i] * proj_scale`,
  `p_i <- p_i * rsqrt(sum_i(p_i^2) / dim + eps) * norm_w[i]`,
  `e_i = dequant(rows + row*row_bytes + l*210)[i] * embed_scale` and
  `out[t, l, i] = (p_i + e_i) * mix_scale`. The scale is folded into the
  projection BEFORE the sum of squares, so it reaches the magnitudes the
  normalization divides by as well as the value it multiplies. `row_bytes` is one
  TOKEN's whole row — one `q6_K` block per slot — and the slot inside it is
  reached by stepping a `shs_dense_block_q6_K *`, so the 210-byte stride comes
  from the struct rather than from a literal. `dim` must be `DENSE_PLE_DIM`
  (256): one `q6_K` block exactly, which is also the staging the kernel asks for.
  The decoder is `dequantize_dense_q6_K` from
  `metal/qwen35/gemv_ext_q5k_q6k.metal`, reused rather than re-read, and the sum
  of squares goes through `dense_block_sum`. Threadgroup memory is
  `4 * (DENSE_BLOCK_SUM_SLOTS + DENSE_PLE_DIM)` = `4 * (33 + 256)` = 1156 bytes:
  the block sum's slots first, the decoded row after them. The engine's mirror is
  `DENSE_PLE_SMEM_BYTES` in `crates/shiso-engine/src/launch.rs`.
- `kernel_dense_ple_gate_mul`: `gate[t, i] = dense_gelu_tanh(gate[t, i]) * ple[t, layer, i]`,
  in place, flat over `n_tok * dim`. The slot is a field of the argument block
  rather than a grid dimension, because one layer of the loop runs one of these.

Device cases: `crates/shiso-metal/tests/dense_misc.rs`,
`crates/shiso-metal/tests/dense_attn.rs` and
`crates/shiso-metal/tests/dense_ple.rs`
(`cargo test -p shiso-metal --features metal --test kernels dense_misc:: -- --ignored`,
`--test kernels dense_attn:: -- --ignored`, `--test kernels dense_ple:: -- --ignored`).
