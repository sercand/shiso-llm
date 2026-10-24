# qwen35 Metal kernel contract

The dispatch contract for the engine: entry-point names, argument blocks, binding
order, launch geometry and buffer layouts. Where this file and a kernel body
disagree, the kernel is right and this file is the bug.

## Conventions

- `[[buffer(0)]]` is always the argument block (`constant q35_args_* &`). Buffers
  follow in declaration order at `[[buffer(1)]]`, `[[buffer(2)]]`... and the
  runtime binds the i-th `KernelArg` to binding index i, so the buffer lists below
  are the bind order, in order.
- `grid` in a launch is the threadgroup count, dispatched as given: no division,
  no rounding. `block` is the threads-per-threadgroup the kernel reads as
  `threads_per_threadgroup`.
- Reductions only go through `q35_warp_sum`/`q35_block_sum`/explicit xor steps.
  Bare `simd_sum` has an unspecified summation order, and these kernels are ported
  against a fixed reduction shape rather than a mathematically equal one.
- qwen35 numbers used in the formulas below: hidden 2560; GDN head_dim 128,
  16 key/query heads and 32 value heads, so the conv mixes 2048 k + 2048 q +
  4096 v = 8192 channels with conv state 3 x 8192 and conv kernel 4; attention
  16 query heads, 4 kv heads, head_dim 256, `n_rot` 64.
- Measured device limits: 32768 B threadgroup memory, 1024 threads per
  threadgroup -- Apple's per-family figure (Mac GPU family 2 covers every
  Apple Silicon Mac), not just a reading off one machine. Every row below is
  inside both, and `run.rs`'s dispatch path refuses (rather than silently
  runs) a launch that is not.

## Geometry

`grid` is the threadgroup count as the CUDA launcher computes it, dispatched
directly -- no division by the block size happens at dispatch time.

| Entry point | Buffers after the args | grid | block | smem |
|---|---|---|---|---|
| `kernel_q35_gdn_conv` | `out, state, x, w` | `ceil(channels/256)` | 256 | 0 |
| `kernel_q35_gdn_l2norm` | `qk` | `(n_heads, n_tok)` | 128 | 132 |
| `kernel_q35_gdn_gates` | `decay, beta, alpha_proj, beta_proj, dt_bias, a` | `ceil(n_head*n_tok/256)` | 256 | 0 |
| `kernel_q35_gdn_out_gate` | `out, attn, z, w` | `n_tok*n_head` | 128 | 132 |
| `kernel_q35_gdn_recurrent_s1` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 128 | 1156 |
| `kernel_q35_gdn_recurrent_s2` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 256 | 1156 |
| `kernel_q35_gdn_recurrent_s4` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 512 | 1156 |
| `kernel_q35_gdn_recurrent_hybrid` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 128 | 17540 |
| `kernel_q35_gdn_chunk_c1` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 128 | 16896 |
| `kernel_q35_gdn_chunk_c2` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 256 | 16896 |
| `kernel_q35_gdn_chunk_c4` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 512 | 16896 |
| `kernel_q35_gdn_chunk_c8` | `attn_out, state, qkv, decay, beta` | `n_head_v` | 1024 | 16896 |
| `kernel_q35_gdn_chunk_hybrid` | `attn_out, state, qkv, decay, beta, dreg_spill` | `n_head_v` | 128 | 16896 |
| `kernel_q35_q_norm_rope` | `q_out, gate_out, qkv, w, pos` | `(n_head, n_tok)` | 128 | 1156 |
| `kernel_q35_k_norm_rope_kv` | `k_cache, v_cache, k, v, kw, pos, pages` | `(n_head_kv, n_tok)` | 128 | 1156 |
| `kernel_q35_q_norm_rope_mrope` | `q_out, gate_out, qkv, w, pos_t, pos_h, pos_w` | `(n_head, n_tok)` | 128 | 1156 |
| `kernel_q35_k_norm_rope_kv_mrope` | `k_cache, v_cache, k, v, kw, pos, pages, pos_t, pos_h, pos_w` | `(n_head_kv, n_tok)` | 128 | 1156 |
| `kernel_q35_attn_decode` | `out, q, k_cache, v_cache, pos, pages` | `(n_head, n_tok)` | 256 | 8256 |
| `kernel_q35_attn_prefill` | `out, q, k_cache, v_cache, pos, pages` | `(n_head, ceil(n_tok/16))` | 256 | 16384 |
| `kernel_q35_attn_gate_mul` | `x, gate` | `ceil(n_tok*n_head*256/256)` | 256 | 0 |


## Argument blocks

Field order is the struct order in `q35_common.h`; the engine mirrors it field for
field.

- `q35_args_gdn_conv {channels, conv_kernel, n_tok}`
- `q35_args_gdn_l2norm {head_dim, n_heads, stride, offset}`
- `q35_args_gdn_gates {n_head, n_tok}`
- `q35_args_gdn_out_gate {head_dim, n_head, eps}`
- `q35_args_q_norm_rope {head_dim, n_head, n_rot, n_tok, rope_base, eps}`
- `q35_args_k_norm_rope {head_dim, n_head_kv, n_rot, n_tok, rope_base, eps, page_tokens, page_stride}`
- The two `_mrope` entry points reuse those two blocks unchanged: three position
  streams are three more buffers, not more fields.
- `q35_args_attn_decode {head_dim, n_head, n_head_kv, n_tok, page_tokens, page_stride}`
- `q35_args_attn_prefill {head_dim, n_head, n_head_kv, n_tok, page_tokens, page_stride}`
- `q35_args_attn_gate {head_dim, n_head, n_tok}`
- `q35_args_gdn_recurrent {head_dim, n_head_k, n_head_v, k_offset, v_offset, stride, n_tok}`
- `q35_args_gdn_chunk` is the same seven fields as the recurrent block.
- `q35_args_gdn_conv {channels, conv_kernel, n_tok}`,
  `q35_args_gdn_l2norm {head_dim, n_heads, stride, offset}`,
  `q35_args_gdn_gates {n_head, n_tok}`,
  `q35_args_gdn_out_gate {head_dim, n_head, eps}`.

## The multimodal rotation

`kernel_q35_q_norm_rope_mrope` and `kernel_q35_k_norm_rope_kv_mrope` are the same
normalization, the same gate split and the same cache write as the pair above
them. Only the rotary angle differs, and only in where its position comes from.

- Rotary pair `i` belongs to section `i % 3`, and the sections are the temporal,
  height and width streams in that order. The pairing `(i, i + n_rot/2)` and the
  frequency `base^(-2i/n_rot)` are the one-dimensional form's, untouched.
- A text token carries the same number in all three streams, so three agreeing
  streams reproduce the one-dimensional rotation exactly. A chunk with no media
  is still dispatched to the one-dimensional entry points rather than to these:
  the equality is a property to rely on when reasoning, not a path to take.
- The key entry point takes BOTH the linear position and the three streams. The
  linear one, and only the linear one, reaches the page table: a placeholder
  token occupies one sequence slot like every other token, while its three grid
  coordinates say nothing about where its cache row is. Translating the temporal
  stream instead would fold a whole image onto a handful of cache rows, and every
  count in the request would still be right.
- The query entry point takes no linear position, because nothing in it addresses
  memory by one.

## Buffer layouts

- **GDN state**: `[n_head_v][head_dim][head_dim]` f32, **transposed**: row `j` of
  head `h` at `state[(h * D + j) * head_dim]` holds, for value column `j`, the
  vector over key space. Both GDN kernels read and write it in a rotated,
  quarter-row-per-lane order; only the address mapping matters to the host.
- **Conv state**: `[conv_kernel - 1][channels]` f32, oldest step first, indexed
  `state[step * channels + c]`.
- **`decay` and `beta`**: `[n_tok][n_head_v]` f32. `decay` is already the
  logarithm of the per-step decay and is negative (`a` carries `-exp(A_log)`), so
  both consumers exponentiate it directly.
- **qkv mix** (GDN): one row per token of `stride` floats laid out as
  `[q | k | v]`, i.e. `q` at 0, keys at `k_offset = head_dim * n_head_k`, values
  at `v_offset`, one head-width per head and `stride` between tokens. For qwen35
  that is `k_offset` 2048, `v_offset` 4096, `stride` 8192. The key head shared by
  value head `h` is `h % n_head_k`.
- **GDN output** (`attn_out`): `[n_tok][n_head_v][head_dim]` f32.
- **decay / beta**: `[n_tok][n_head_v]` f32. `decay` is already the negative
  log-decay (`a` holds `-exp(A_log)`), so both GDN paths exponentiate it directly.
- **Gated-attention q**: `[n_tok][n_head][2 * head_dim]` f32 -- query in the first
  half, the raw output gate in the second. `kernel_q35_q_norm_rope` splits that
  row into the normalized/roped query and an untouched copy of the gate; the gate
  gets no activation until `kernel_q35_attn_gate_mul`.
- **KV cache**: f16, `[physical row][n_head_kv][head_dim]`, written by
  `kernel_q35_k_norm_rope_kv` and read by both attention kernels, which take the
  causal bound from `pos[token]` rather than from a token count. A logical
  position reaches its row through `pages`, the session page table, as
  `pages[position / page_tokens] * page_tokens + position % page_tokens`, over
  the table `q35_row_pages(pages, t, page_stride)` selects for row `t`. A
  `page_stride` of zero is one table for the whole launch, which is every
  prefill chunk and every single-session step; a batched decode step packs one
  row per session and passes the row width.
  The engine owns that table, copies it from its own frontier before every
  KV-touching launch, and no other part of this contract computes a KV address:
  all three KV addresses go through `q35_kv_row` in `q35_common.h`. A kernel
  handed a null table treats a position as a row, which exists only so a
  single-page fixture can launch without a table.

## Decode versus prefill

The recurrent family (`_s1`/`_s2`/`_s4`/`_hybrid`, one lane split among them chosen
per device) walks tokens one at a time and refreshes the state after every token,
so it is also the path that fills speculative-rollback checkpoints; the chunk
family (`_c1`/`_c2`/`_c4`/`_c8`/`_hybrid`, same choice) runs 64-token runs and
produces only the final state of a run, no per-token states, so it cannot fill
rollback checkpoints. That is the only reason the two families exist; the
launcher keeps them on either side of a length threshold, and which member of a
family is dispatched is a per-device choice the tuner (`shiso-engine/src/tune.rs`,
`shiso-engine/src/gdn_tune.rs`) makes at startup, not a compile-time constant.

## Threadgroup memory

These kernels replace CUDA's shared memory with registers or device reads because
the CUDA shapes exceed the 32 KiB every Apple Silicon GPU is built to. None of
them changes a summation order:

- `kernel_q35_gdn_recurrent` (and its `_s1`/`_s2`/`_s4` lane-width siblings) keep
  the 128x128 state in registers (one quarter-row per lane, or per SPLIT lanes)
  instead of the 66 KiB of shared memory the CUDA launcher asks for, and keep the
  rotated per-quarter sweep order, which is what fixes which lane adds which term.
  `kernel_q35_gdn_recurrent_hybrid` is the same body at 128 lanes with one of the
  four quarters staged in threadgroup memory instead of kept in registers (16384 B
  of staging plus the 1156 B of scratch every recurrent variant uses, 17540 B
  total -- comfortably inside the 32768 B family limit, unlike staging two
  quarters), so a device that cannot afford the all-register row still has
  something to launch; the staged quarter is per-lane private, so no extra
  barrier guards it.
- `kernel_q35_gdn_chunk` (and its `_c1`/`_c2`/`_c4`/`_c8` lane-width siblings)
  keeps only `s_A` (64x64, plus the 128 gate entries) in threadgroup memory.
  Keys and queries come from the device, the deltas are registers, and the
  deltas of a token `t` live in the lane whose `sub == t % CSPLIT`, which the
  carry phase reaches with one shuffle per token. 16896 B instead of the CUDA
  99072 B, at every width: the split changes how many lanes share a value
  column, not what is staged.
- `kernel_q35_gdn_chunk_hybrid` is that body at 128 lanes with the deltas
  spilled to a device scratch bound at index 6 -- the only delta-rule entry
  point that is not the six-buffer one. The deltas are per value column, so at
  one lane per column a threadgroup holds 128 * 64 * 4 B = 32768 B of them:
  the whole family threadgroup limit, with `s_A`'s 16896 B already inside it,
  so unlike the recurrent hybrid's staged quarter there is no arrangement that
  fits and device memory is the only place left. The host sizes the scratch
  `n_head_v * Q35_GDN_CHUNK * Q35_GDN_D` floats, laid out `[head][token][column]`
  so the 128 lanes read one contiguous span per token.
- `kernel_q35_attn_prefill` stages the transposed keys only (16 KiB); staging the
  values and queries as well would need 50 KiB, past the device limit. Its
  warp-private key-weight bank is a register broadcast from the lane that owns the
  weight, so no barrier sits inside the query loop.

## Entry points not listed here

`kernel_mul_mm_f16_f32`, `kernel_mul_mm_q8_0_f32`, `kernel_mul_mm_q4_K_f32`,
`kernel_mul_mm_q4_0_f32` and `kernel_mul_mm_q6_K_f32`
(`gemm_q4k.metal`) are the prefill GEMM. They are copied verbatim from the generic
kernel set with instantiations deleted and nothing else, so their launch geometry
and argument block (`shs_metal_args_mul_mm`) are the generic ones, including the
function constants the generic set declares for them. The `q4_0` instantiation is
this file's own addition, of the same template over the block struct and
`dequantize_q4_0` the generic set declares. So is the `q6_K` one, over the block
struct and `dequantize_dense_q6_K` that `gemv_ext_q5k_q6k.metal` declares ahead
of this file.

`kernel_mul_mv_ext_q5_K_f32_r1_{1..5}` and `kernel_mul_mv_ext_q6_K_f32_r1_{1..5}`
are instantiations of the generic `kernel_mul_mv_ext_q4_f32_disp` template with
`epb` 256, so their argument block, threadgroup shape and function constants
(600 simdgroups, 601 lanes-per-x-split) are the generic ext-gemv ones, including
the `#600:s<nsg>,601:s<nxpsg>` specialization suffix. `r1_N` is the
rows-per-threadgroup variant: T6 picks the instantiation for its batch remainder.

`kernel_get_rows_q6_K_f32` takes `shs_metal_args_get_rows_q8_0` and mirrors the
generic q4_K gather: grid.x is the 256-element block index within the row, grid.y
is the token index, one threadgroup per block, threads striding over `QK_K`. The
unpack is q6_K's (`ql` low nibbles plus `qh` high twos, per-16-group scales, the
`- 32` bias), so a gathered row is the same vector a q6_K matvec would read.
