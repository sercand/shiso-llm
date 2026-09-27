# The `audio` kernel set: launch contract

Buffer index 0 is always the argument block, passed as `KernelArg::Bytes`. The buffers follow in the order
listed.

| Entry | Args block | Buffers (after args) | Grid × block, smem |
|---|---|---|---|
| `kernel_audio_unary_f32` | `audio_args_unary{count, op, a, b}` | out, in | `ceil(count/256)` × 256, 0 |
| `kernel_audio_axpy_f32` | `audio_args_axpy{count, alpha}` | y (in/out), x | `ceil(count/256)` × 256, 0 |
| `kernel_audio_bias_rows_f32` | `audio_args_bias_rows{rows, cols}` | y (in/out) `[rows][cols]`, b `[cols]` | `ceil(rows·cols/256)` × 256, 0 |
| `kernel_audio_argmax_rows_f32` | `audio_args_argmax{rows, cols}` | out u32 `[rows]`, in `[rows][cols]` | `rows` × 256, 0 |
| `kernel_audio_argmax_split_f32` | `audio_args_argmax_split{cols, span, parts}` | parts u32 `[parts][2]` (value bits, index), in `[cols]` | `parts` × 256, 0 |
| `kernel_audio_argmax_merge_f32` | `audio_args_argmax_split` | out u32 `[1]`, parts (from the entry above) | 1 × 256, 0 |
| `kernel_audio_glu_f32` | `audio_args_glu{rows, channels}` | out `[rows][C]`, in `[rows][2C]` | `ceil(rows·C/256)` × 256, 0 |
| `kernel_audio_silu_mul_f32` | `audio_args_silu_mul{count}` | out, gate, up (`out` may be `gate`) | `ceil(count/256)` × 256, 0 |
| `kernel_audio_add_rms_norm_f32` | `audio_args_norm{rows, dim, eps, flags}` | out (may be add, not x), x (in/out) `[rows][dim]`, add, weight `[dim]` | rows × 256, 132 |
| `kernel_audio_layer_norm_f32` | `audio_args_norm{rows, dim, eps, flags}` | out, in, weight `[dim]`, bias `[dim]` | rows × 256, 132 |
| `kernel_audio_rms_norm_f32` | `audio_args_norm{rows, dim, eps, flags}` | out, in, weight `[dim]` | rows × 256, 132 |
| `kernel_audio_gemm_{f32w,bf16w,f16w,q8_0w}` | `audio_args_gemm{m, n, k, flags, k_chunk, splits}` | y `[splits][m][n]` (the output itself with one split), x `[m][k]`, w `[n][k]` (q8_0w: `n·k/32` 34-byte blocks, `k % 32 == 0`), bias `[n]` | `(ceil(n/64), ceil(m/32), splits)` × 128, 12288 |
| `kernel_audio_gemm64_{f32w,bf16w,f16w,q8_0w}` | as `kernel_audio_gemm_*` | as `kernel_audio_gemm_*` | `(ceil(n/64), ceil(m/64), splits)` × 256, 20480 |
| `kernel_audio_gemv_{f32w,bf16w,f16w,q8_0w}` | `audio_args_gemm` (one split) | y `[m][n]`, x `[m][k]`, w `[n][k]`, bias `[n]` | `(ceil(n/16), ceil(m/8))` × 128, 0 |
| `kernel_audio_matvec_{f32w,bf16w,f16w,q8_0w}` | `audio_args_gemm` (one split; `k % 8 == 0`) | y `[m][n]`, x `[m][k]` (16-byte aligned), w `[n][k]` (16-byte aligned; q8_0w as the GEMM's), bias `[n]` | `(ceil(n/4), m)` × 64, 0 (static) |
| `kernel_audio_matvec3_{f32w,bf16w,f16w,q8_0w}` | `audio_args_matvec3{n0, n1, n2, k}` (`k % 8 == 0`) | y0 `[n0]`, y1 `[n1]`, y2 `[n2]`, x `[k]`, w0 `[n0][k]`, w1 `[n1][k]`, w2 `[n2][k]` (x and w 16-byte aligned) | `ceil(n0/4) + ceil(n1/4) + ceil(n2/4)` × 64, 0 (static) |
| `kernel_audio_matvec_silu_{f32w,bf16w,f16w,q8_0w}` | `audio_args_gemm` (`flags`, `k_chunk`, `splits` unread) | y `[m][n]`, x `[m][k]`, w_gate `[n][k]`, w_up `[n][k]` | `(ceil(n/4), m)` × 64, 0 (static) |
| `kernel_audio_gemm_affine8w`, `kernel_audio_gemm64_affine8w` | `audio_args_gemm` (`k % 64 == 0`) | as `kernel_audio_gemm_*` with w the bytes u32 `[n][k/4]`, then scales bf16 `[n][k/64]`, biases bf16 (as scales) | as `kernel_audio_gemm_*` / `kernel_audio_gemm64_*` |
| `kernel_audio_matvec_affine8w` | `audio_args_gemm` (one split; `k % 64 == 0`) | as `kernel_audio_matvec_*` with w the bytes (8-byte aligned), then scales, biases | as `kernel_audio_matvec_*` |
| `kernel_audio_matvec3_affine8w` | `audio_args_matvec3` (`k % 64 == 0`) | y0, y1, y2, x, the three weights' bytes, their three scales, their three biases | as `kernel_audio_matvec3_*` |
| `kernel_audio_matvec_silu_affine8w` | `audio_args_gemm` | y, x, gate bytes, up bytes, gate scales, up scales, gate biases, up biases | as `kernel_audio_matvec_silu_*` |
| `kernel_audio_matvec_rows_{f32w,bf16w,f16w,q8_0w,affine8w}` | as `kernel_audio_matvec_*` | as `kernel_audio_matvec_*` (affine8w as `kernel_audio_matvec_affine8w`) | `(ceil(n/4), ceil(m/8))` × 64, 0 (static); a threadgroup's up to 8 rows share each weight run, each row summed as the one-row kernel sums it |
| `kernel_audio_matvec_silu_rows_{f32w,bf16w,f16w,q8_0w,affine8w}` | as `kernel_audio_matvec_silu_*` | as `kernel_audio_matvec_silu_*` (affine8w as `kernel_audio_matvec_silu_affine8w`) | `(ceil(n/4), ceil(m/8))` × 64, 0 (static); as `kernel_audio_matvec_rows_*` |
| `kernel_audio_split_sum_f32` | `audio_args_gemm` | y `[m][n]`, parts `[splits][m][n]`, bias `[n]` | `ceil(m·n/256)` × 256, 0 |
| `kernel_audio_conv2d_s2_f32` | `audio_args_conv2d{t_in, f_in, c_in, t_out, f_out, c_out, pad_t, pad_f, depthwise, has_bias, relu, in_valid, out_valid, batch}` | out `[batch][t_out][f_out][c_out]`, in `[batch][t_in][f_in][c_in]`, w `[c_out][c_in/g][3][3]`, bias `[c_out]` | `ceil(batch·t_out·f_out·c_out/256)` × 256, 0 |
| `kernel_audio_im2col_s2_f32` | `audio_args_conv2d` (as above; `c_out`, `depthwise`, `has_bias`, `relu` unread) | out `[batch][t_out][f_out][9·c_in]` (tap `kt·3+kf` major, channel minor), in `[batch][t_in][f_in][c_in]` | `ceil(batch·t_out·f_out·9·c_in/256)` × 256, 0 |
| `kernel_audio_dwconv1d_f32` | `audio_args_dwconv1d{rows, channels, taps}` | out `[rows][C]`, in `[rows+taps−1][C]`, w `[C][taps]` | `ceil(rows·C/256)` × 256, 0 |
| `kernel_audio_unfold_f32` | `audio_args_unfold{rows, channels, taps, pad}` | out `[rows][taps·C]`, in `[rows][C]` | `ceil(rows·taps·C/256)` × 256, 0 |
| `kernel_audio_transpose_f32` | `audio_args_transpose{batch, rows, cols}` | out `[b][cols][rows]`, in `[b][rows][cols]` | `ceil(b·rows·cols/256)` × 256, 0 |
| `kernel_audio_rel_scores_f32` | `audio_args_window_attn` (as below) | scores `[n_q][H][n_d]`, q `[n_q][H·hd]`, bias_v `[H·hd]`, q_dim_scale `[hd]`, rel `[n_d][H·hd]` | `(ceil(n_d/4), heads, ceil(n_q/8))` × 32, 0 |
| `kernel_audio_window_attn_f32` | `audio_args_window_attn{n_q, n_k, heads, head_dim, q_abs0, k_abs0, k_end, chunk, left_chunks, d_min, n_d, scale, ac_scale, softcap, has_bias_uv, has_q_dim_scale}` | out, q `[n_q][H·hd]`, k, v `[n_k][H·hd]`, bias_u `[H·hd]`, q_dim_scale `[hd]`, rel_scores `[n_q][H][n_d]` (from the entry above; unread when `n_d` is 0) | `(heads, ceil(n_q/8))` × 32, 8·hd·4 |
| `kernel_audio_rope_half_f32` | `audio_args_rope{rows, heads, head_dim, pos0, theta}` | x (in place) `[rows][H·hd]` | `ceil(rows·H·hd/2/256)` × 256, 0 |
| `kernel_audio_lstm_cell_f32` (the reference form of the next) | `audio_args_lstm{hidden}` | h_out `[H]`, c (in/out) `[H]`, gates `[4H]` | `ceil(H/256)` × 256, 0 |
| `kernel_audio_lstm_cell_gated_f32` | `audio_args_lstm{hidden}` | x_out `[H]`, h (in/out) `[H]`, c (in/out) `[H]`, gates `[4H]`, state u32 `[8]` | `ceil(H/256)` × 256, 0 |
| `kernel_audio_greedy_window_f32` | `audio_args_greedy_window{rows, cols, avail}` | out `[rows][cols]`, pending `[avail][cols]`, dec `[cols]`, state u32 `[8]` | `ceil(rows·cols/256)` × 256, 0 |
| `kernel_audio_greedy_step_u32` | `audio_args_greedy_step{rows, avail, blank, max_symbols, slot}` | state u32 `[8]` (in/out), out u32 `[slots][2]`, ids u32 `[rows]` | 1 × 1, 0 |
| `kernel_audio_embed_row_f32` | `audio_args_embed_row{width, vocab}` | out `[width]`, table `[vocab][width]`, state u32 `[8]` | `ceil(width/256)` × 256, 0 |
| `kernel_audio_keep_f32` | `audio_args_keep{count}` | dst `[count]`, src `[count]`, state u32 `[8]` | `ceil(count/256)` × 256, 0 |
| `kernel_audio_rope_half_state_f32` | `audio_args_rope{rows, heads, head_dim, pos0, theta}` (`pos0` an offset on the state's position) | x (in place) `[rows][H·hd]`, decoder state u32 `[8]` | `ceil(rows·H·hd/2/256)` × 256, 0 |
| `kernel_audio_kv_append_f32` | `audio_args_kv_append{rows, kv_heads, head_dim, cap}` | k cache (in/out) `[Hkv][cap][hd]`, v cache (in/out), k `[rows][Hkv·hd]`, v, decoder state u32 `[8]` | `ceil(rows·Hkv·hd/256)` × 256, 0 |
| `kernel_audio_qk_norm_rope_append_f32` | `audio_args_qk_rope{rows, heads, kv_heads, head_dim, cap, eps, theta}` | q (in/out) `[rows][H·hd]`, k `[rows][Hkv·hd]`, v, q_norm `[hd]`, k_norm `[hd]`, k cache (in/out) `[Hkv][cap][hd]`, v cache (in/out), decoder state u32 `[8]` | `(heads + kv_heads, rows)` × 256, 0 (static) |
| `kernel_audio_causal_attn_f32` | `audio_args_cache_attn{n_q, heads, kv_heads, head_dim, pos0, cap, scale, splits}` (`splits` unread) | out `[n_q][H·hd]`, q `[n_q][H·hd]`, k cache `[Hkv][cap][hd]`, v cache | `(heads, ceil(n_q/8))` × 32, 8·hd·4 |
| `kernel_audio_decode_attn_f32` | `audio_args_cache_attn` (`n_q`, `pos0` unread) | partials `[H·splits·(hd+2)]` (accumulators `[H][splits][hd]`, then (max, sum) pairs `[H][splits][2]`), q `[H·hd]`, k cache `[Hkv][cap][hd]`, v cache, decoder state u32 `[8]` | `(Hkv, ceil(splits/4), ceil((H/Hkv)/2))` × 128, 0 |
| `kernel_audio_decode_attn_merge_f32` | `audio_args_cache_attn` (as above) | out `[H·hd]`, partials (from the entry above) | `heads` × 256, 0 (static) |
| `kernel_audio_decode_step_u32` | `audio_args_decode_step{eos0, eos1, max_tokens, advance, slot}` | decoder state u32 `[8]` (in/out), out u32 `[slots]`, ids u32 `[1]` | 1 × 1, 0 |
| `kernel_audio_embed_token_{f32w,bf16w,f16w,q8_0w}` | `audio_args_embed_token{width, vocab, group}` (`group` unread) | out `[width]`, table `[vocab][width]` (q8_0w: `vocab·width/32` 34-byte blocks), decoder state u32 `[8]` | `ceil(width/256)` × 256, 0 |
| `kernel_audio_embed_token_affine8w` | `audio_args_embed_token{width, vocab, group}` | out `[width]`, words u32 `[vocab][width/4]`, scales bf16 `[vocab][width/group]`, biases bf16 (as scales), decoder state u32 `[8]` | `ceil(width/256)` × 256, 0 |
| `kernel_audio_causal_attn_seg_f32` | `audio_args_batch_attn{seqs, heads, kv_heads, head_dim, scale, splits}` (`splits` unread) | out `[rows][H·hd]`, q `[rows][H·hd]` (stacked), k cache (shared), v cache (shared), sequence table u32 `[seqs][8]` | `(heads, ceil(max seq rows/8), seqs)` × 32, 8·hd·4 |
| `kernel_audio_qk_norm_rope_append_seg_f32` | `audio_args_qk_rope_seq{rows, seqs, heads, kv_heads, head_dim, eps, theta}` | q (in/out) `[rows][H·hd]` (stacked), k `[rows][Hkv·hd]`, v, q_norm `[hd]`, k_norm `[hd]`, k cache (shared, in/out), v cache (shared, in/out), sequence table u32 `[seqs][8]` | `(heads + kv_heads, rows)` × 256, 0 (static) |
| `kernel_audio_qk_norm_rope_append_batch_f32` | `audio_args_qk_rope_seq` (`rows` unread) | q (in/out) `[seqs][H·hd]`, k `[seqs][Hkv·hd]`, v, q_norm, k_norm, k cache (shared, in/out), v cache (shared, in/out), decoder states u32 `[seqs][8]`, sequence table u32 `[seqs][8]` | `(heads + kv_heads, seqs)` × 256, 0 (static) |
| `kernel_audio_decode_attn_batch_f32` | `audio_args_batch_attn` | partials `[seqs·H·splits·(hd+2)]` (the single entry's layout for `seqs·H` heads), q `[seqs][H·hd]`, k cache (shared), v cache (shared), decoder states u32 `[seqs][8]`, sequence table u32 `[seqs][8]` | `(Hkv, ceil(splits/4), ceil((H/Hkv)/2)·seqs)` × 128, 0; then `kernel_audio_decode_attn_merge_f32` with `heads = seqs·H` into out `[seqs][H·hd]` |
| `kernel_audio_decode_step_batch_u32` | `audio_args_decode_step_batch{eos0, eos1, max_tokens, advance, slot, seqs}` | decoder states u32 `[seqs][8]` (in/out), out u32 `[slots][seqs]`, ids u32 `[seqs]` | `ceil(seqs/32)` × 32, 0 |
| `kernel_audio_embed_token_batch_{f32w,bf16w,f16w,q8_0w}` | `audio_args_embed_token` (`group` unread) | out `[seqs][width]`, table (as `kernel_audio_embed_token_*`), decoder states u32 `[seqs][8]` | `(ceil(width/256), seqs)` × 256, 0 |
| `kernel_audio_embed_token_batch_affine8w` | `audio_args_embed_token` | out `[seqs][width]`, words, scales, biases (as `kernel_audio_embed_token_affine8w`), decoder states u32 `[seqs][8]` | `(ceil(width/256), seqs)` × 256, 0 |

Unary op codes: `0` relu, `1` silu, `2` sigmoid, `3` exact GELU (erf), `4` tanh, `5` clamp to `[a, b]`, `6` scale by
`a`. Norm flags: bit 0 has-weight (RMS only; the layer norm always reads one), bit 1 has-bias (layer norm). GEMM
flags: bit 0 has-bias. A buffer the flags leave unread must still be bound.

Window attention visibility: key `ka = k_abs0 + j` is visible to query `qa = q_abs0 + i` iff `j < n_k`,
`ka < k_end` and `0 ≤ ⌊qa/chunk⌋ − ⌊ka/chunk⌋ ≤ left_chunks`. The logit is
`scale · (ac_scale · (q'+bu)·k + (q'+bv)·rel[qa−ka−d_min])` with `q' = q ⊙ q_dim_scale`, then `softcap·tanh(·/softcap)`
when `softcap > 0`; a query with no visible key writes zeros.

Every entry accumulates in f32. Launch geometry is fixed per entry: the linear kernels hard-code a 256-thread
stride, the GEMM assumes 128 threads (four simdgroups) and 12288 bytes of threadgroup memory for a 32x64 tile, the
64-row GEMM 256 threads (eight simdgroups) and 20480 bytes for a 64x64 tile, the matrix-vector kernel 128 threads (four
simdgroups) and none, and the window attention one simdgroup of 32 lanes per (head, block of eight queries) with
head_dim at most 128. The host takes the matrix-vector kernel for at most 4 rows, the 32-row GEMM up to 32 and the
64-row GEMM above that.

The matrix-vector-by-runs kernel is the one-row product of a decoder step: each lane dots runs of eight
consecutive weights (for q8_0w a quarter block, the run's integer dot scaled once), sums its runs in k order, a
simdgroup sum closes each of the two simdgroups, and their sums are added in simdgroup order. That order is not the
matrix-vector kernel's, so a model takes one or the other for a product. Every row re-reads the weights.

An `affine8w` weight is MLX 8-bit affine with groups of 64: the bytes four to a little-endian u32 along each row and
a bf16 scale and bias per group. The tiled GEMMs stage each weight as the fused `q * scale + bias` in f32; the
matrix-vector-by-runs entries dot a run of eight as `scale * (q . x) + bias * sum(x)`, so the two orders differ and a
model takes one or the other for a product.

The decoder step's fused entries each do the work of a chain of the plain entries in one launch and form every
value exactly as that chain does, so a decoder may take either: `add_rms_norm` is the axpy at alpha 1 then the RMS
norm (each thread loads its eight-value runs before storing any); `qk_norm_rope_append` is the RMS norm over each
query and key head (256 threads a head), the rotation at the state's position (offset 0) and the append, with a row
at or past `cap` not written; `matvec3` is three one-product matvecs (threadgroups `[0, t0)` on the first weight,
then the second, then the third) and `matvec_silu` two of them and the SiLU product. The split argmax cuts a wide row
into spans of `span` columns, each threadgroup writing its span's first largest (value, index), and the merge picks
among them; the answer is the one-pass argmax's whatever the cut.

A split GEMM covers `k` in slices of `k_chunk` (a multiple of 32): block `z` multiplies `[z·k_chunk, (z+1)·k_chunk)`
into plane `z` of y without the bias, and the split-sum pass adds the planes in order and the bias once, so the result
does not depend on scheduling. With `splits == 1` the GEMM writes the output and its bias directly. The host splits
only when the output tiles alone are fewer than 64, and sends at most four rows to the matrix-vector kernel, which
gives each simdgroup four columns of up to eight rows.

The `q8_0w` GEMM reads GGUF `Q8_0` rows directly (34-byte blocks: an f16 scale, then 32 signed bytes) and
dequantises in f32; `k` must be a multiple of 32. It exists because the engine's tiled `Q8_0` path stages both
operands through half precision from nine rows up, measured at 1.1e-2 relative error against 2.3e-5 here.

The layer norm is compiled in `math_mode(safe)` and takes every sum about the row's first value: fast-math
reassociation of `(x - pivot) - dmean` back into `x - mean` rounded the mean to the row's magnitude and lost
about 6% on a row near 1e4 with a deviation of 1e-2.

Greedy search's state is eight u32 words: the frame (relative to the first pending row), the symbols emitted on it,
done (the frame reached `avail`), the last token, whether the last step emitted it, and three unused. The step kernel
writes `(token, frame)` to its slot, or `(0xffffffff, 0)` when it emitted nothing; the gated cell, the embedding row
and keep read the emit flag and the token it left.

A causal decoder's state is eight u32 words: `0` the cache rows filled (the position of the token the next pass
feeds), `1` the tokens emitted, `2` done, `3` the last token emitted (the one the next pass feeds, and the row the
embedding entries fetch), `4` whether the last step emitted, `5` why it stopped (`0` running, `1` an end id, `2` the
token cap), and two unused. The rotation, the append and decode attention read word 0 and never write it; only the
step kernel moves it, by its `advance` (0 for the step over the prefill's own argmax, 1 after a decode pass), so every
layer of a pass appends at the same row. The prefill hands over by a host write: its rows go in through the append
from a state whose word 0 is 0, and the host then writes `n`, the prefill's rows, into word 0 of the decode state
(`copy_h2d`) before it queues the step over the prefill's argmax; no kernel adds a prefill's length. Decode
attention sees rows `0..=word 0` (clamped to `cap`); the append drops a row at or past `cap`, so the host grows the
cache before it queues a batch that would reach it. The step writes the
token to `out[slot]`, or `0xffffffff` for a step that emitted nothing: an end id (which is never emitted), a done
state, or a state that already has `max_tokens`. Once done, a later step in the batch changes no word but the emit
flag, which it lowers.

Causal attention visibility: query `i` sits at `pos0 + i` and sees cache rows `0..=pos0 + i` of kv head
`h / (heads / kv_heads)`, with the logit `scale · q·k`; rows past `pos0 + n_q − 1` are never read.

Decode attention runs as two launches. The first cuts the `len` rows present into `splits` runs of
`ceil(len/splits)` rounded up to four (`len` is only known on the device, so `splits` is fixed per launch, at most
256); one simdgroup per (kv head, split, pair of the group's query heads), four to a threadgroup, scores its run and
leaves an unnormalised accumulator and its (max, sum). The merge then closes each head in split order, so the result
does not depend on scheduling. It wants `head_dim` a multiple of four and every view 16-byte aligned (lanes read four
dims at a time, so a cached row is one 512-byte read). Measured on an M1 Max (24-core GPU), 16 query heads over 8 kv
heads of 128 with 32 splits: about 20 µs over 600 rows and 67 µs over 4096 (both launches). Its inner loop uses the
fast `exp`, unlike the window attention: the precise one doubled that time, and on arguments at most zero the fast
one's `3 + 2|x|` ulp keeps every softmax weight within 3 ulp of 1. The CUDA twin uses the accurate `expf`.

A per-head RMS norm (a decoder's q and k norms over each head's 128 dims) is `kernel_audio_rms_norm_f32` with
`rows = n·H` and `dim = hd`: each row is normalised on its own against a weight of length `dim`.

A batch of causal-decoder sequences runs through the `_seg` (prefill) and `_batch` (decode) entries. Every
sequence has its own decoder state, the eight words above, and the batch keeps them back to back (`[seqs][8]`). The
sequences' caches share one key and one value buffer per layer: a sequence table of eight u32 words per sequence
gives `0` the first cache row of the sequence's `[kv_heads][cap][hd]` plane (row `r` of kv head `h` is at
`(plane + h·cap + r)·hd`), `1` its `cap`, and for a segmented prefill `2` its first stacked row, `3` its row count
and `4` the position of that first row; `5`–`7` are unused. The decode entries read only words 0 and 1, so each
sequence's plane can sit anywhere in the buffers with its own capacity, and a launch binds three buffers for the
cache whatever the batch size (Metal allows 31 bindings a kernel). Each batched entry runs the same body as the
single-sequence entry it batches, over that sequence's plane, rows and state, and forms every value exactly as that
entry does on the sequence alone; a stacked row no sequence owns is neither read nor written. A finished sequence
(state word 2 set) appends nothing and its decode attention sees no rows, so its row merges to zeros; its rows may
keep flowing through the batch's products. The batched step writes sequence `s`'s token of step `slot` to
`out[slot·seqs + s]`.
