# The `audio` kernel set: launch contract

Buffer index 0 is always the argument block, passed as `KernelArg::Bytes`. The buffers follow in the order
listed.

| Entry | Args block | Buffers (after args) | Grid × block, smem |
|---|---|---|---|
| `kernel_audio_unary_f32` | `audio_args_unary{count, op, a, b}` | out, in | `ceil(count/256)` × 256, 0 |
| `kernel_audio_axpy_f32` | `audio_args_axpy{count, alpha}` | y (in/out), x | `ceil(count/256)` × 256, 0 |
| `kernel_audio_bias_rows_f32` | `audio_args_bias_rows{rows, cols}` | y (in/out) `[rows][cols]`, b `[cols]` | `ceil(rows·cols/256)` × 256, 0 |
| `kernel_audio_argmax_rows_f32` | `audio_args_argmax{rows, cols}` | out u32 `[rows]`, in `[rows][cols]` | `rows` × 256, 0 |
| `kernel_audio_glu_f32` | `audio_args_glu{rows, channels}` | out `[rows][C]`, in `[rows][2C]` | `ceil(rows·C/256)` × 256, 0 |
| `kernel_audio_layer_norm_f32` | `audio_args_norm{rows, dim, eps, flags}` | out, in, weight `[dim]`, bias `[dim]` | rows × 256, 132 |
| `kernel_audio_rms_norm_f32` | `audio_args_norm{rows, dim, eps, flags}` | out, in, weight `[dim]` | rows × 256, 132 |
| `kernel_audio_gemm_{f32w,bf16w,f16w,q8_0w}` | `audio_args_gemm{m, n, k, flags, k_chunk, splits}` | y `[splits][m][n]` (the output itself with one split), x `[m][k]`, w `[n][k]` (q8_0w: `n·k/32` 34-byte blocks, `k % 32 == 0`), bias `[n]` | `(ceil(n/64), ceil(m/32), splits)` × 128, 12288 |
| `kernel_audio_gemm64_{f32w,bf16w,f16w,q8_0w}` | as `kernel_audio_gemm_*` | as `kernel_audio_gemm_*` | `(ceil(n/64), ceil(m/64), splits)` × 256, 20480 |
| `kernel_audio_gemv_{f32w,bf16w,f16w,q8_0w}` | `audio_args_gemm` (one split) | y `[m][n]`, x `[m][k]`, w `[n][k]`, bias `[n]` | `(ceil(n/16), ceil(m/8))` × 128, 0 |
| `kernel_audio_split_sum_f32` | `audio_args_gemm` | y `[m][n]`, parts `[splits][m][n]`, bias `[n]` | `ceil(m·n/256)` × 256, 0 |
| `kernel_audio_conv2d_s2_f32` | `audio_args_conv2d{t_in, f_in, c_in, t_out, f_out, c_out, pad_t, pad_f, depthwise, has_bias, relu, in_valid, out_valid}` | out `[t_out][f_out][c_out]`, in `[t_in][f_in][c_in]`, w `[c_out][c_in/g][3][3]`, bias `[c_out]` | `ceil(t_out·f_out·c_out/256)` × 256, 0 |
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
