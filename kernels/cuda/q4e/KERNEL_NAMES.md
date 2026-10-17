# q4e entry-point names

The lookup table a loader resolves a kernel through. A module is loaded by its
registry key; a kernel inside it is fetched by the exact name recorded here.

## How to read this

Two kinds of entry live in these units, and they are fetched differently.

**Plain kernels** carry C linkage, so the name below is the name the driver
answers to, verbatim. There are 65 of them.

**Template kernels** do not. They are instantiated explicitly at the bottom of
their translation unit - 155 instantiations across 12 templates - and each one
gets a C++-mangled name that depends on the compiler's mangling of its argument
list. Those names are NOT written out below, and guessing them would be the
worst kind of plausible: a mangled name that is subtly wrong fails at lookup
time with "not found", and a mangled name that is subtly wrong *and happens to
match another instantiation* does not fail at all.

Capture them from the built module instead, on a machine that has the toolchain:

```sh
# every entry the module actually exposes, mangled exactly as the driver wants
grep -o '\.visible \.entry [^(]*' <module>.ptx | sed 's/.*entry //'
```

That output is the truth. This file records which template each instantiation
came from and what it was instantiated with, so a captured name can be matched
back to the kernel it belongs to.

## Modules

| Registry key | Unit | Plain entries | Template instantiations |
|---|---|---|---|
| `q4e_hc` | `hc.cu` | 4 | 6 |
| `q4e_moe` | `moe.cu` | 9 | 4 |
| `q4e_gdn` | `gdn.cu` | 8 | 0 |
| `q4e_gdn_tc` | `gdn_tc.cu` | 4 | 0 |
| `q4e_ple` | `ple.cu` | 5 | 0 |
| `q4e_matvec` | `matvec.cu` | 6 | 95 |
| `q4e_qsa` | `qsa.cu` | 7 | 0 |
| `q4e_qsa_sparse` | `qsa_sparse.cu` | 4 | 0 |
| `q4e_idx` | `idx.cu` | 8 | 0 |
| `q4e_misc` | `misc.cu` | 3 | 0 |
| `q4e_embed` | `embed.cu` | 4 | 0 |
| `q4e_modelopt` | `modelopt.cu` | 9 | 0 |
| `q4e_exact` | `exact.cu` | 2 | 0 |
| `q4e_kv8` | `kv8.cu` | 8 | 0 |
| `q4e_mxfp8` | `mxfp8.cu` | 9 | 0 |
| `q4e_idxring` | `idxring.cu` | 1 | 0 |
| `q4e_bf16` | `bf16.cu` | 12 | 0 |
| **total** | 17 units | **103** | **105** |

## Plain entries, by module

### `q4e_hc`

- `q4e_hc_init_kernel`
- `q4e_hc_init_add_kernel`
- `q4e_scale_silu_kernel`
- `q4e_hc_collapse_f16_dual_kernel`
- `q4e_hc_combine_scale_kernel` - the fused combine leaving the next norm's
  per-(token, stream) scale in place of the normed streams.

### `q4e_modelopt`

Not an upstream unit: this repository's own NVFP4 routed stages and FP8 and NVFP4
n-gram decode, for the `modelopt` export.

- `q4e_nvfp4_experts_gate_up_kernel`
- `q4e_nvfp4_experts_gate_up_act_kernel` - the same rows, each column's
  `silu(gate) * up` stored as `q4e_swiglu_kernel` would compute it.
- `q4e_nvfp4_experts_gate_up_act_held_kernel` - the same activated rows, a
  2560-wide row held in registers across an expert's columns (a verify).
- `q4e_nvfp4_experts_down_kernel`
- `q4e_nvfp4_experts_gemm_gate_up_kernel`
- `q4e_nvfp4_experts_gemm_down_kernel`
- `q4e_fp8_experts_gate_up_kernel`
- `q4e_fp8_experts_down_kernel`
- `q4e_ple_dequant_fp8_kernel`
- `q4e_ple_dequant_nvfp4_kernel`

### `q4e_exact`

Not an upstream unit: this repository's own variants of two ported paths,
derived from `qsa_sparse.cu` and `gdn.cu` and kept out of them so neither
ported unit is edited.

- `q4e_qsa_attention_tc_exact_kernel` - `q4e_qsa_attention_tc_kernel` with
  every softmax weight carried into the P V product as two fp16 halves.
- `q4e_gdn_step_kernel` - one decode row of a recurrent layer after its
  convolution: the two L2 norms, the gates, the recurrence and the output
  gate, bit for bit the five launches it replaces.

### `q4e_kv8`

Not an upstream unit: the key/value store and every reader of the attention
K and V planes over a cache of FP8 E4M3 bytes (static scale 1) instead of
f16, for `--kv-cache-dtype fp8`. Each is the named kernel with the cache's
element type changed: the store rounds to E4M3 (saturating at 448, never the
NaN code), and each reader decodes a byte to exactly the f16 value it stands
for and then runs the ported arithmetic unchanged.

- `q4e_qsa_store_kv_fp8_kernel` - `q4e_qsa_store_kv_kernel`.
- `q4e_qsa_store_kv_batch_fp8_kernel` - `q4e_qsa_store_kv_batch_kernel`.
- `q4e_qsa_attention_fp8_kernel` - `q4e_qsa_attention_kernel`.
- `q4e_qsa_attention_batch_fp8_kernel` - `q4e_qsa_attention_batch_kernel`.
- `q4e_qsa_attention_tiled_fp8_kernel` - `q4e_qsa_attention_tiled_kernel`.
- `q4e_qsa_attention_split_fp8_kernel` - `q4e_qsa_attention_split_kernel`.
- `q4e_qsa_attention_gather_fp8_kernel` - `q4e_qsa_attention_gather_kernel`.
- `q4e_qsa_attention_tc_exact_fp8_kernel` - `q4e_qsa_attention_tc_exact_kernel`.

### `q4e_mxfp8`

Not an upstream unit: this repository's own dense projections over an MXFP8
weight (`e4m3` codes, one `ue8m0` scale per 32), read as stored. One argument
list for all five: `out, codes, scales, x, in_dim, out_dim, rows`.

- `q4e_mxfp8_gemv_kernel` - one activation row; a row's sum is the
  brain-float rows kernel's on the exactly widened weight. Two output rows a
  warp.
- `q4e_mxfp8_gemv_narrow_kernel` - the same sums, one output row a warp, for
  outputs of at most 1024 rows.
- `q4e_mxfp8_rows_kernel` - two to eight activation rows, each summed as the
  one-row kernel sums it; the activations staged once a block, four output
  rows a warp.
- `q4e_mxfp8_rows_narrow_kernel` - the same sums, one output row a warp, for
  outputs of at most 1024 rows.
- `q4e_mxfp8_gemm_kernel` - a prompt chunk on tensor cores, 128 x 128 tiles,
  one scale group a contraction step, activations rounded to halves.

The group entries take up to four matrices that read one activation, each
writing its own output, in one grid whose blocks each own rows of one matrix:
`out0, codes0, scales0, out_dim0` for slots 0 to 3 (an unused slot has
`out_dim` 0), then `x, in_dim, rows`. A row of every matrix is summed exactly
as the single-matrix entries sum it.

- `q4e_mxfp8_gemv_group_kernel` - one activation row, two output rows a warp.
- `q4e_mxfp8_gemv_group_narrow_kernel` - one activation row, one output row a
  warp with the next chunk's codes in flight, for groups of at most 4096 rows.
- `q4e_mxfp8_rows_group_kernel` - two to eight activation rows, four output
  rows a warp.
- `q4e_mxfp8_rows_group_narrow_kernel` - two to eight activation rows, one
  output row a warp, for groups of at most 1024 rows.

### `q4e_idxring`

Not an upstream unit: the indexer's pooled key pass over a per-session ring of
raw keys, so a page carries no raw key row per position.

- `q4e_idx_ring_pool_kernel` - `q4e_idx_pool_kernel`'s arithmetic, reading a
  raw key from the pass's own f32 rows (rounded to f16 as the store rounded
  it) or, before the pass's first position, from the ring; block 0 then copies
  the frontier block's raw keys into a checkpoint's ring section when asked
  (`snap_row != 0xFFFFFFFF`) and stores the pass's last 16 rows into the ring.

### `q4e_bf16`

Not an upstream unit: this repository's own dense projections over a
brain-float weight, read as stored. The step kernels take
`out, w, x, in_dim, out_dim, splits, rows` (each output row's contraction
sliced across `splits` warps of one block and folded there); the prompt
kernels take `out, w, x, in_dim, out_dim, rows`.

- `q4e_bf16_gemv_kernel` - one activation row, one output row a warp, eight
  contiguous weights a lane a step.
- `q4e_bf16_rows_kernel` - two to eight activation rows, each summed as the
  one-row kernel sums it; two output rows a warp.
- `q4e_bf16_rows_wide_kernel` - nine to thirty-two rows, the same sums; four
  output rows a warp.
- `q4e_bf16_gemm_kernel` - a prompt chunk on tensor cores, 128 x 128 tiles,
  activations rounded to brain floats.
- `q4e_bf16_gemm_split_kernel` - the same with each activation carried as two
  brain floats, for the narrow projections.
- `q4e_bf16_gemm_pair_kernel` - the plain prompt kernel over two weights of one
  input stacked in one grid (`out_a, w_a, out_dim_a, out_b, w_b, out_dim_b, x,
  in_dim, rows`): the stream mix's contraction and injection logits.
- `q4e_bf16_gemm_collapse_kernel` - the stream mix's expansion with the
  collapse of the gated streams as its epilogue (`out, w_up, x, xn, in_dim,
  n_embd, n_hc, rows`); `q4e_bf16_gemm_kernel` then `q4e_hc_collapse_kernel`,
  bit for bit.
- `q4e_bf16_gemm_pair_norm_kernel` - the pair over the residual streams, each
  value normed as it is read (`..., res, scale, nw, n_embd, in_dim, rows`).
- `q4e_bf16_gemm_collapse_norm_kernel` - the fused collapse over the residual
  streams, normed as they are read (`out, w_up, x, res, scale, nw, in_dim,
  n_embd, n_hc, rows`).
- `q4e_bf16_gemv_pair_kernel`, `q4e_bf16_rows_pair_kernel` - the step kernels
  over two weights of one input stacked in one grid (`out_a, w_a, out_dim_a,
  out_b, w_b, out_dim_b, x, in_dim, splits, rows`).
- `q4e_bf16_mix_step_kernel` - the stream mix's activation, expansion and
  collapse for one to eight step rows (`out, w_up, x, xn, inv_hc, in_dim,
  n_embd, n_hc, rows`), four 320-wide expansion rows a warp.

### `q4e_moe`

- `q4e_moe_route_kernel`
- `q4e_moe_route_sort_small_kernel`
- `q4e_moe_route_sort_tiny_kernel` - the small kernel's results for at most
  eight tokens, at a shorter serial chain.
- `q4e_swiglu_kernel`
- `q4e_moe_combine_kernel`
- `q4e_shared_add_kernel`
- `q4e_moe_rows_per_slot_kernel`
- `q4e_moe_sort_count_kernel`
- `q4e_moe_sort_scan_kernel`
- `q4e_moe_sort_scatter_kernel`

### `q4e_gdn`

- `q4e_gdn_conv_kernel`
- `q4e_gdn_l2norm_kernel`
- `q4e_gdn_gates_kernel`
- `q4e_gdn_recurrent_kernel`
- `q4e_gdn_chunk_kernel`
- `q4e_gdn_out_gate_kernel`
- `q4e_gdn_conv_batch_kernel`
- `q4e_gdn_recurrent_batch_kernel`
- `q4e_gdn_conv_seg_kernel`
- `q4e_gdn_recurrent_seg_kernel`
- `q4e_gdn_replay_kernel`

### `q4e_gdn_tc`

- `q4e_gdn_tc_prep_kernel`
- `q4e_gdn_tc_scan_kernel`
- `q4e_gdn_conv_par_kernel`
- `q4e_gdn_conv_tail_kernel`

### `q4e_ple`

- `q4e_ple_dequant_kernel`
- `q4e_ple_dequant_exl3_kernel`
- `q4e_ple_gated_value_kernel`
- `q4e_ple_conv_kernel`
- `q4e_ple_conv_batch_kernel`

### `q4e_matvec`

- `q4e_matvec_q8_0_narrow_kernel`
- `q4e_f32_to_f16_kernel`
- `q4e_f16_to_f32_kernel`
- `q4e_exl3_slice_tiles_kernel`
- `q4e_exl3_slice_svh_kernel`
- `q4e_matvec_f32_kernel`
- `q4e_matvec_f32_combine_kernel`

### `q4e_qsa`

- `q4e_qsa_q_norm_rope_kernel`
- `q4e_qsa_store_kv_kernel`
- `q4e_qsa_attention_kernel`
- `q4e_qsa_store_kv_batch_kernel`
- `q4e_qsa_attention_batch_kernel`
- `q4e_qsa_attention_tiled_kernel`
- `q4e_qsa_gate_kernel`

### `q4e_qsa_sparse`

- `q4e_qsa_attention_split_kernel`
- `q4e_qsa_attention_combine_kernel`
- `q4e_qsa_attention_gather_kernel`
- `q4e_qsa_attention_tc_kernel`

### `q4e_idx`

- `q4e_idx_store_k_kernel`
- `q4e_idx_store_k_batch_kernel`
- `q4e_idx_pool_kernel`
- `q4e_idx_q_kernel`
- `q4e_idx_score_kernel`
- `q4e_idx_score_tc_kernel`
- `q4e_idx_topk_kernel`
- `q4e_idx_expand_kernel`

### `q4e_misc`

- `q4e_add2_kernel`
- `q4e_argmax_rows_kernel`
- `q4e_copy_row_kernel`

### `q4e_embed`

The row gather, one entry per stored element type of a vocabulary table. These
four keep the names their upstream file gives them rather than a `q4e_` prefix,
because they are not this family's: the same four serve every model in the tree
that gathers rows, and renaming a copied entry is the one edit an extraction is
never allowed to make.

- `embed_token_hc_kernel`
- `embed_tokens_hc_kernel`
- `embed_tokens_q8_0_kernel`
- `embed_tokens_bf16_kernel`

## Template instantiations, by module

Each row is one instantiation. The driver name is mangled and must be captured
from the built module as described above; the template and its arguments are what
identify the kernel a captured name belongs to.

### `q4e_hc`

| Template | Instantiated with |
|---|---|
| `q4e_hc_norm_kernel` | `float` |
| `q4e_hc_norm_kernel` | `__half` |
| `q4e_hc_collapse_kernel` | `float` |
| `q4e_hc_collapse_kernel` | `__half` |
| `q4e_hc_combine_norm_kernel` | `float` |
| `q4e_hc_combine_norm_kernel` | `__half` |

### `q4e_moe`

| Template | Instantiated with |
|---|---|
| `q4e_moe_down_grouped_kernel` | `7, 20` |
| `q4e_moe_down_grouped_kernel` | `8, 20` |
| `q4e_moe_down_kernel` | `7` |
| `q4e_moe_down_kernel` | `8` |

### `q4e_matvec`

| Template | Instantiated with |
|---|---|
| `q4e_matmul_q8_0_rows_narrow_kernel` | `2` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `3` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `4` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `5` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `6` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `7` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `8` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `9` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `10` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `11` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `12` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `13` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `14` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `15` |
| `q4e_matmul_q8_0_rows_narrow_kernel` | `16` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `2` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `3` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `4` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `5` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `6` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `7` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `8` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `9` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `10` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `11` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `12` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `13` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `14` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `15` |
| `q4e_matmul_q8_0_rows_warp_kernel` | `16` |
| `q4e_matmul_bf16_rows_kernel` | `2` |
| `q4e_matmul_bf16_rows_kernel` | `3` |
| `q4e_matmul_bf16_rows_kernel` | `4` |
| `q4e_matmul_bf16_rows_kernel` | `5` |
| `q4e_matmul_bf16_rows_kernel` | `6` |
| `q4e_matmul_bf16_rows_kernel` | `7` |
| `q4e_matmul_bf16_rows_kernel` | `8` |
| `q4e_matmul_bf16_rows_kernel` | `9` |
| `q4e_matmul_bf16_rows_kernel` | `10` |
| `q4e_matmul_bf16_rows_kernel` | `11` |
| `q4e_matmul_bf16_rows_kernel` | `12` |
| `q4e_matmul_bf16_rows_kernel` | `13` |
| `q4e_matmul_bf16_rows_kernel` | `14` |
| `q4e_matmul_bf16_rows_kernel` | `15` |
| `q4e_matmul_bf16_rows_kernel` | `16` |
| `q4e_matmul_f16_rows_kernel` | `1, float, float` |
| `q4e_matmul_f16_rows_kernel` | `2, float, float` |
| `q4e_matmul_f16_rows_kernel` | `3, float, float` |
| `q4e_matmul_f16_rows_kernel` | `4, float, float` |
| `q4e_matmul_f16_rows_kernel` | `5, float, float` |
| `q4e_matmul_f16_rows_kernel` | `6, float, float` |
| `q4e_matmul_f16_rows_kernel` | `7, float, float` |
| `q4e_matmul_f16_rows_kernel` | `8, float, float` |
| `q4e_matmul_f16_rows_kernel` | `9, float, float` |
| `q4e_matmul_f16_rows_kernel` | `10, float, float` |
| `q4e_matmul_f16_rows_kernel` | `11, float, float` |
| `q4e_matmul_f16_rows_kernel` | `12, float, float` |
| `q4e_matmul_f16_rows_kernel` | `13, float, float` |
| `q4e_matmul_f16_rows_kernel` | `14, float, float` |
| `q4e_matmul_f16_rows_kernel` | `15, float, float` |
| `q4e_matmul_f16_rows_kernel` | `16, float, float` |
| `q4e_matmul_f16_rows_kernel` | `17, float, float` |
| `q4e_matmul_f16_rows_kernel` | `18, float, float` |
| `q4e_matmul_f16_rows_kernel` | `19, float, float` |
| `q4e_matmul_f16_rows_kernel` | `20, float, float` |
| `q4e_matmul_f16_rows_kernel` | `21, float, float` |
| `q4e_matmul_f16_rows_kernel` | `22, float, float` |
| `q4e_matmul_f16_rows_kernel` | `23, float, float` |
| `q4e_matmul_f16_rows_kernel` | `24, float, float` |
| `q4e_matmul_f16_rows_kernel` | `25, float, float` |
| `q4e_matmul_f16_rows_kernel` | `26, float, float` |
| `q4e_matmul_f16_rows_kernel` | `27, float, float` |
| `q4e_matmul_f16_rows_kernel` | `28, float, float` |
| `q4e_matmul_f16_rows_kernel` | `29, float, float` |
| `q4e_matmul_f16_rows_kernel` | `30, float, float` |
| `q4e_matmul_f16_rows_kernel` | `31, float, float` |
| `q4e_matmul_f16_rows_kernel` | `32, float, float` |
| `q4e_matmul_p12_rows_kernel` | `1` |
| `q4e_matmul_p12_rows_kernel` | `2` |
| `q4e_matmul_p12_rows_kernel` | `3` |
| `q4e_matmul_p12_rows_kernel` | `4` |
| `q4e_matmul_p12_rows_kernel` | `5` |
| `q4e_matmul_p12_rows_kernel` | `6` |
| `q4e_matmul_p12_rows_kernel` | `7` |
| `q4e_matmul_p12_rows_kernel` | `8` |
| `q4e_matmul_p12_rows_kernel` | `9` |
| `q4e_matmul_p12_rows_kernel` | `10` |
| `q4e_matmul_p12_rows_kernel` | `11` |
| `q4e_matmul_p12_rows_kernel` | `12` |
| `q4e_matmul_p12_rows_kernel` | `13` |
| `q4e_matmul_p12_rows_kernel` | `14` |
| `q4e_matmul_p12_rows_kernel` | `15` |
| `q4e_matmul_p12_rows_kernel` | `16` |
| `q4e_matmul_p12_rows_kernel` | `17` |
| `q4e_matmul_p12_rows_kernel` | `18` |
| `q4e_matmul_p12_rows_kernel` | `19` |
| `q4e_matmul_p12_rows_kernel` | `20` |
| `q4e_matmul_p12_rows_kernel` | `21` |
| `q4e_matmul_p12_rows_kernel` | `22` |
| `q4e_matmul_p12_rows_kernel` | `23` |
| `q4e_matmul_p12_rows_kernel` | `24` |
| `q4e_matmul_p12_rows_kernel` | `25` |
| `q4e_matmul_p12_rows_kernel` | `26` |
| `q4e_matmul_p12_rows_kernel` | `27` |
| `q4e_matmul_p12_rows_kernel` | `28` |
| `q4e_matmul_p12_rows_kernel` | `29` |
| `q4e_matmul_p12_rows_kernel` | `30` |
| `q4e_matmul_p12_rows_kernel` | `31` |
| `q4e_matmul_p12_rows_kernel` | `32` |
| `q4e_p12_expand_kernel` | `__half` |
| `q4e_p12_expand_kernel` | `float` |
| `q4e_matmul_f16_rows_kernel` | `1, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `2, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `3, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `4, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `5, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `6, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `7, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `8, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `9, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `10, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `11, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `12, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `13, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `14, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `15, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `16, __half, float` |
| `q4e_matmul_f16_rows_kernel` | `1, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `2, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `3, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `4, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `5, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `6, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `7, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `8, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `9, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `10, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `11, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `12, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `13, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `14, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `15, float, __half` |
| `q4e_matmul_f16_rows_kernel` | `16, float, __half` |
| `q4e_matvec_combine_t_kernel` | `float` |
| `q4e_matvec_combine_t_kernel` | `__half` |

## Launch geometry

Every one of the 61 kernels, with the grid, the block and the dynamic
shared-memory size as the *expressions* the host computes them from. Numbers for
one particular shape would be a trap: the same kernel runs at decode (one row)
and at prefill (hundreds), and a row that read `grid = 4` would be right once.

These are transcribed from the host launchers. The launchers themselves were not
extracted - they are host code, and this port does not carry host C - so the
formulas live here and nowhere else in the tree.

### How to read the expressions

- `ceil(a/b)` is the launcher's integer `(a + b - 1) / b`. Where the launcher
  writes something that is *not* a ceiling, the row says so.
- A grid written as a single expression is one-dimensional (`y = z = 1`). A
  triple `(x, y, z)` is the launcher's `dim3` in that order.
- Dynamic shared memory is in bytes. `0` means the kernel takes none; its
  `__shared__` arrays, where it has them, are static and cost nothing at launch.
- `Q4E_*` are the compile-time constants defined in the kernel text itself, in
  the translation unit that holds the kernel. They are part of what was
  vendored, so a dispatcher can read them from the same source the kernel came
  from rather than hard-coding a number here.
- The remaining names are runtime quantities the calling op already has:
  `n_tok` (rows in this call; `n_row` for the batched entries), `n_embd`,
  `n_hc`, `n_head`, `n_head_k`, `n_head_v`, `n_head_kv`, `head_dim`, `in_dim`,
  `out_dim`, `n_expert`, `n_used`, `channels`, `n_rows`, `n_vocab`, and `n` for
  the flat element count of an elementwise op.

### The shared-memory opt-in

Anything past the per-block default (48 KiB) has to be opted into before the
launch, or it fails at launch time rather than at load time. The rows that need
it are marked **opt-in**; every one of them is a kernel whose formula can exceed
48 KiB at a shape this model actually runs.

Three further kernels - `q4e_idx_score_kernel`, `q4e_qsa_attention_split_kernel`
and `q4e_qsa_attention_gather_kernel` - are opted in unconditionally by their
host launcher even though their formula stays under 48 KiB at the shipped head
count. Those rows are marked **opt-in (unconditional)**: replaying the opt-in for
them is what the source does and is harmless, but a dispatcher that skips it is
not thereby broken at the shipped shape.

### `q4e_hc`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_hc_init_kernel` | `ceil(n_tok * n_hc * n_embd / 256)` | `256` | `0` |
| `q4e_hc_init_add_kernel` | `ceil(n_tok * n_hc * n_embd / 256)` | `256` | `0` |
| `q4e_scale_silu_kernel` | `ceil(n / 256)` | `256` | `0` |
| `q4e_hc_norm_kernel` ⊤ | `n_tok * n_hc` | `256` | `0` |
| `q4e_hc_collapse_kernel` ⊤ | `n_tok` | `256` | `0` |
| `q4e_hc_collapse_f16_dual_kernel` | `n_tok` | `256` | `0` |
| `q4e_hc_combine_norm_kernel` ⊤ | `(n_tok, n_tok >= 256 ? 1 : n_hc, 1)` | `256` | `0` |
| `q4e_hc_combine_scale_kernel` | `(n_tok, n_tok >= 256 ? 1 : n_hc, 1)` | `256` | `0` |

`q4e_hc_combine_norm_kernel`'s `y` is the launcher's own switch: one block per
token once there are enough tokens to fill the machine that way, one block per
(token, stream) below that. The block width is fixed at 256 and the launcher
refuses the call outright when `n_embd > 256 * Q4E_HC_FUSE_MAX`.

### `q4e_moe`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_moe_route_kernel` | `n_tok` | `256` | `n_expert * 4` |
| `q4e_moe_route_sort_small_kernel` | `1` | `1024` | `0` |
| `q4e_moe_route_sort_tiny_kernel` | `1` | `256` | `0` |
| `q4e_swiglu_kernel` | `ceil(n / 256)` | `256` | `0` |
| `q4e_moe_combine_kernel` | `(ceil(n_embd / 256), n_tok, 1)` | `256` | `0` |
| `q4e_shared_add_kernel` | `(ceil(n_embd / 256), n_tok, 1)` | `256` | `0` |
| `q4e_moe_rows_per_slot_kernel` | `ceil(n_tok * n_used * in_dim / 256)` | `256` | `0` |
| `q4e_moe_sort_count_kernel` | `ceil(n_rows / 256)` | `256` | `0` |
| `q4e_moe_sort_scan_kernel` | `1` | `32` | `0` |
| `q4e_moe_sort_scatter_kernel` | `ceil(n_rows / 256)` | `256` | `0` |
| `q4e_moe_down_grouped_kernel` ⊤ | `(ceil(out_dim / 4), min(n_rows, n_expert), 1)` | `128` | `0` |
| `q4e_moe_down_kernel` ⊤ | `(ceil(out_dim / 4), n_rows, 1)` | `128` | `0` |

`n_rows = n_tok * n_used` throughout this module. The two `down` kernels share
one launcher, which fixes `threads = 128` and derives `rows_per_block =
threads / 32 = 4` - four warps per block, one output row each; the `4` in
`ceil(out_dim / 4)` is that, not a constant of its own. The grouped kernel's `y`
spans only the experts that received a column (`min(n_rows, n_expert)` bounds
the active list), which is why it is not `n_expert`.

`q4e_moe_route_kernel`'s shared memory is the logits staged for the selection
rounds: one float per expert.

### `q4e_gdn`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_gdn_conv_kernel` | `ceil(channels / 256)` | `256` | `0` |
| `q4e_gdn_l2norm_kernel` | `(n_heads, n_tok, 1)` | `128` | `0` |
| `q4e_gdn_gates_kernel` | `ceil(n_head * n_tok / 256)` | `256` | `0` |
| `q4e_gdn_recurrent_kernel` | `n_head_v` | `head_dim * Q4E_GDN_SPLIT` | `head_dim * (head_dim + 1) * 4` — **opt-in** |
| `q4e_gdn_chunk_kernel` | `n_head_v` | `head_dim * Q4E_GDN_CSPLIT` | `(C*D + C*(D+1) + C*C + Q4E_GDN_QTILE*D + 2*C) * 4`, `C = Q4E_GDN_CHUNK`, `D = 128` — **opt-in** |
| `q4e_gdn_out_gate_kernel` | `n_tok * n_head` | `128` | `0` |
| `q4e_gdn_conv_batch_kernel` | `(ceil(channels / 256), n_row, 1)` | `256` | `0` |
| `q4e_gdn_recurrent_batch_kernel` | `(n_head_v, n_row, 1)` | `head_dim * Q4E_GDN_SPLIT` | `head_dim * (head_dim + 1) * 4` — **opt-in** |
| `q4e_gdn_conv_seg_kernel` | `(ceil(channels / 256), n_run, 1)` | `256` | `0` |
| `q4e_gdn_recurrent_seg_kernel` | `(n_head_v, n_run, 1)` | `head_dim * Q4E_GDN_SPLIT` | `head_dim * (head_dim + 1) * 4` — **opt-in** |
| `q4e_gdn_replay_kernel` | `n_head_v` | `head_dim * Q4E_GDN_SPLIT` | `head_dim * (head_dim + 1) * 4` — **opt-in** |

The two `_seg` kernels are a segmented verify's: one launch for every
sequence's run, `blockIdx.y` the run, with the run lengths packed one byte per
run into a by-value `uint64_t`. Their per-run bodies are the scalar kernels'
own device functions.

The recurrent kernels' launchers reject any `head_dim` but 128 - their state
tiles are sized for the shipped head - so the shared-memory formula is
66,048 B in practice, and the `+ 1` is a float of row padding that keeps the
state's shared reads off a single bank. The chunk kernel evaluates to 99,072 B.
Both figures match the four sizes this file already recorded.

### `q4e_gdn_tc`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_gdn_tc_prep_kernel` | `(n_head_v, ceil(n_tok / 64), 1)` | `512` | `2*C*136*2 + C*68*4 + 3*C*4`, `C = 64` — 53,504 B, **opt-in** |
| `q4e_gdn_tc_scan_kernel` | `n_head_v` | `512` | `D*136*2 + 2*C*136*2 + 2*C*4`, `D = 128` — 70,144 B, **opt-in** |
| `q4e_gdn_conv_par_kernel` | `(ceil(channels / 256), ceil(n_tok / 64), 1)` | `256` | `0` |
| `q4e_gdn_conv_tail_kernel` | `ceil(channels / 256)` | `256` | `0` |

Both are shiso's own (see the unit's header); they replace
`q4e_gdn_chunk_kernel` on the prefill path and read its inputs unchanged.

The chunk kernel is a prefill-only path: its launcher takes it only when the row
count clears a threshold and the device can actually grant the shared memory,
and falls back to `q4e_gdn_recurrent_kernel` - correct at any shape, only
slower - when it cannot.

### `q4e_ple`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_ple_dequant_kernel` | `n_tok * n_heads` | `32` | `0` |
| `q4e_ple_dequant_exl3_kernel` | `n_tok * n_heads` | `160` | `0` |
| `q4e_ple_gated_value_kernel` | `(n_tok, n_hc, 1)` | `256` | `0` |
| `q4e_ple_conv_kernel` | `ceil(channels / 256)` | `256` | `0` |
| `q4e_ple_conv_batch_kernel` | `(ceil(channels / 256), n_row, 1)` | `256` | `0` |

The two dequant kernels share a launcher that picks between them on the row
type; the exl3 one is reachable only at `head_dim == 160`, which is also its
block width.

### `q4e_matvec`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_matvec_q8_0_narrow_kernel` | `out_dim` | `blocks <= 128 ? 128 : 256` | `0` |
| `q4e_f32_to_f16_kernel` | `ceil(n / 256)` | `256` | `0` |
| `q4e_f16_to_f32_kernel` | `ceil(n / 256)` | `256` | `0` |
| `q4e_exl3_slice_tiles_kernel` | `ceil(n_pieces / 256)` | `256` | `0` |
| `q4e_exl3_slice_svh_kernel` | `ceil(n_sel / 256)` | `256` | `0` |
| `q4e_matvec_f32_kernel` | `(out_dim, n_split, 1)` | `256` | `0` |
| `q4e_matvec_f32_combine_kernel` | `ceil(out_dim / 128)` | `128` | `0` |
| `q4e_matmul_q8_0_rows_narrow_kernel` ⊤ | `out_dim` | `blocks <= 128 ? 128 : 256` | `0` |
| `q4e_matmul_q8_0_rows_warp_kernel` ⊤ | `ceil(out_dim / 8)` | `256` | `0` |
| `q4e_matmul_bf16_rows_kernel` ⊤ | `ceil(out_dim / 8)` | `256` | `0` |
| `q4e_matmul_f16_rows_kernel` ⊤ | `(ceil(out_dim / 8), n_split, 1)` | `256` | `0` |
| `q4e_matvec_combine_t_kernel` ⊤ | `ceil(NT * out_dim / 128)` | `128` | `0` |

`blocks = in_dim / 32` - the q8_0 block count of one weight row. The two narrow
kernels take a 128-wide block when a row is short enough that 256 threads would
idle.

`ceil(out_dim / 8)` in the three warp-per-row kernels is 256 threads / 32 lanes
= 8 output rows per block; it is derived from the block width, not a tunable.

`n_split` is a K-split factor the launcher computes, not an input:

- `q4e_matvec_f32_kernel`: start at 1, double while `out_dim * n_split * 2 <= 512`
  and `in_dim / (n_split * 2) >= 512`.
- `q4e_matmul_f16_rows_kernel`: start at 1, double while
  `out_dim * n_split < 1024` and `in_dim / (n_split * 2) >= 256`.

Each of the two split kernels is followed by its combine kernel *only* when
`n_split > 1`; at `n_split == 1` the split kernel writes the destination
directly and the combine launch does not happen. `q4e_matvec_combine_t_kernel`'s
`NT * out_dim` is the f16 launcher's row count times the width - the whole
partial set, not one row.

`n_pieces` and `n_sel` belong to the one-off expert-head slicing helper, not to
a per-step op: for `n_blocks` selected blocks of a `k x n` exl3 weight at `bits`
bits, `n_sel = n_blocks * 128` and `n_pieces = k * n_blocks * bits`.

### `q4e_qsa`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_qsa_q_norm_rope_kernel` | `(n_head, n_tok, 1)` | `128` | `head_dim * 4` |
| `q4e_qsa_store_kv_kernel` | `(n_head_kv, n_tok, 1)` | `128` | `head_dim * 4` |
| `q4e_qsa_store_kv_batch_kernel` | `(n_head_kv, n_row, 1)` | `128` | `head_dim * 4` |
| `q4e_qsa_attention_kernel` | `(n_head, n_tok, 1)` | `32 * Q4E_ATTN_WARPS` | `Q4E_ATTN_WARPS * head_dim * 4` |
| `q4e_qsa_attention_batch_kernel` | `(n_head, n_row, 1)` | `32 * Q4E_ATTN_WARPS` | `Q4E_ATTN_WARPS * head_dim * 4` |
| `q4e_qsa_attention_tiled_kernel` | `(n_head, ceil(n_tok / Q4E_ATTN_QT), 1)` | `32 * Q4E_ATTN_TILE_WARPS` | `2*head_dim*Q4E_ATTN_KT*2 + Q4E_ATTN_QT*head_dim*4 + Q4E_ATTN_TILE_WARPS*Q4E_ATTN_KT*4` — **opt-in** |
| `q4e_qsa_gate_kernel` | `ceil(n / 256)` | `256` | `0` |

The tiled kernel's shared memory reaches 50,176 B at `head_dim == 256`, which is
past the default: hence the opt-in. Its launcher also refuses the path unless
`n_tok >= Q4E_ATTN_QT` and `head_dim % 32 == 0`, and falls back to
`q4e_qsa_attention_kernel` if the launch is rejected.

`q4e_qsa_attention_kernel` is the always-correct path the dense launcher falls
back to; the three kernels above it in the dense launcher's preference order
(`q4e_qsa_attention_tc_kernel`, `q4e_qsa_attention_tiled_kernel`,
`q4e_qsa_attention_split_kernel`) each carry a shape precondition.

### `q4e_qsa_sparse`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_qsa_attention_split_kernel` | `(n_tok, Q4E_ATTN_SPLITS, n_head_kv)` - one block per row (D2), `G = n_head / n_head_kv <= Q4E_ATTN_QG` | `256` | `Q4E_ATTN_SPLIT_HD*Q4E_ATTN_KT*2*2 + Q4E_ATTN_QG*Q4E_ATTN_SPLIT_HD*2 + Q4E_ATTN_TILE_WARPS*Q4E_ATTN_KT*4` — **opt-in (unconditional)** |
| `q4e_qsa_attention_gather_kernel` | `(Q4E_ATTN_GSPLITS, n_head_kv, n_tok)` | `256` | same formula as the split kernel — **opt-in (unconditional)** |
| `q4e_qsa_attention_tc_kernel` | `(splits, n_head_kv, n_tok)` | `256` | `Q4E_TC_SMEM` — **opt-in** |
| `q4e_qsa_attention_combine_kernel` | `n_head * n_tok` | `Q4E_ATTN_SPLIT_HD` | `0` |

The split and gather formula evaluates to 41,984 B - under the default, which is
why their opt-in is marked unconditional rather than required. `Q4E_TC_SMEM` is
79,936 B and genuinely needs it.

`splits` for the tensor-core kernel is the launcher's own quantity and depends on
`n_tok` alone, deliberately, so that a captured graph never sees a different
geometry on replay:

- sparse entry: `splits = (n_tok * n_head_kv >= 64) ? 1 : Q4E_ATTN_GSPLITS`.
- dense entry: `splits = 1`, always, and `grid.x` is `1`.

`q4e_qsa_attention_combine_kernel` runs after the split, gather, and multi-split
tensor-core paths and has the same geometry at all three call sites; only the
split count it is handed differs (`Q4E_ATTN_SPLITS`, `Q4E_ATTN_GSPLITS`, and
`splits` respectively). It is skipped entirely when the tensor-core path ran at
`splits == 1`, because that path wrote the destination directly.

### `q4e_exact`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_qsa_attention_tc_exact_kernel` | `(splits, n_head_kv, n_tok)` | `256` | `Q4E_EX_TC_SMEM` — **opt-in** |
| `q4e_gdn_step_kernel` | `n_head_v` | `head_dim * 4` | `head_dim * (head_dim + 1) * 4` — **opt-in** |

`Q4E_EX_TC_SMEM` is `Q4E_TC_SMEM` plus the low-half weight tile, `16 *
Q4E_EX_TC_KT * 2`: 81,984 B. Its argument list, grid and `splits` rule are the
ported tensor-core kernel's (above); both tensor-core paths launch it in place
of that kernel.

`q4e_gdn_step_kernel` carries the sequential recurrence's geometry and state
tile unchanged, 66,048 B at the one head width it is planned at (128). It is
planned only for one row of one session's decode step; every other pass runs
the chain it replaces.

### `q4e_kv8`

Every entry takes the grid, block, dynamic shared memory and argument list of
the f16 kernel it is named after (tables above), with the two cache pointers
typed `uint8_t *` instead of `__half *`. The readers' shared tiles stay f16 -
the codes are decoded as they are staged - so the reservations are the same
numbers, including the opt-in ones.

### `q4e_idx`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_idx_store_k_kernel` | `n_tok` | `128` | `0` |
| `q4e_idx_store_k_batch_kernel` | `n_row` | `128` | `0` |
| `q4e_idx_pool_kernel` | `n_tok / Q4E_IDX_R + 1` | `128` | `0` |
| `q4e_idx_q_kernel` | `(n_head, n_tok, 1)` | `128` | `0` |
| `q4e_idx_score_kernel` | `(ceil(max_blocks / Q4E_IDX_BT), ceil(n_tok / Q4E_IDX_QT), 1)` | `256` | `(Q4E_IDX_QT * n_head + Q4E_IDX_BT) * Q4E_IDX_DIM * 2` — **opt-in (unconditional)** |
| `q4e_idx_score_tc_kernel` | `(ceil(max_blocks / Q4E_IDX_TC_BT), ceil(n_tok / Q4E_IDX_QT), 1)` | `256` | `Q4E_IDX_TC_SMEM` — **opt-in** |
| `q4e_idx_topk_kernel` | `n_tok` | `1024` | `0` |
| `q4e_idx_expand_kernel` | `n_tok` | `256` | `0` |

`q4e_idx_pool_kernel`'s grid is **not** a ceiling. The launcher writes
`n_tok / Q4E_IDX_R + 1` - a truncating divide, then plus one - because at most
`n_tok/r + 1` key blocks complete in one forward and the extra block covers the
partial one. Rounding it up instead is wrong at every `n_tok` that is a multiple
of `Q4E_IDX_R`: it launches the same count, but for the wrong reason, and it
diverges the moment either quantity changes.

`max_blocks` is the indexer's key-block capacity, passed in by the caller.
The two score kernels are alternatives chosen on head count: the tensor-core one
runs only at `n_head == 4` and when its 49,152 B opt-in succeeded; otherwise the
scalar one, whose shared memory depends on `n_head` and reaches 32,768 B at the
shipped 4. Their tile widths differ (`Q4E_IDX_TC_BT` vs `Q4E_IDX_BT`), so the
`x` extents are not interchangeable.

### `q4e_idxring`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_idx_ring_pool_kernel` | `n_tok / Q4E_IDX_R + 1` | `128` | `0` |

Its arguments are `pooled, ring, k, w, pos, pages, n_tok, n_rot, rope_base,
mrope_all, sec_t, sec_h, sec_w, eps, snap, snap_row`; the grid follows the
same truncating-divide rule as `q4e_idx_pool_kernel` above, and block 0 always
exists, because it is the one that writes the ring.

### `q4e_misc`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `q4e_add2_kernel` | `ceil(n / 256)` | `256` | `0` |
| `q4e_argmax_rows_kernel` | `n_rows` | `1024` | `0` |
| `q4e_copy_row_kernel` | `ceil(n / 256)` | `256` | `0` |

### `q4e_embed`

| Kernel | Grid | Block | Dynamic shared memory |
|---|---|---|---|
| `embed_token_hc_kernel` | `ceil(n_embd * n_hc / 256)` | `256` | `0` |
| `embed_tokens_hc_kernel` | `ceil(n_tok * n_hc * n_embd / 256)` | `256` | `0` |
| `embed_tokens_q8_0_kernel` | `ceil(n_tok * n_embd / 256)` | `256` | `0` |
| `embed_tokens_bf16_kernel` | `ceil(n_tok * n_embd / 256)` | `256` | `0` |

All four are flat elementwise launches over the destination, and `n_hc` is the
fan-out the half-precision pair writes: at `n_hc = 1` it gathers one row per
token and the stream broadcast is left to `q4e_hc_init_kernel`, which is what a
stack whose residual is initialised separately wants. The two that take
`n_vocab` clamp an out-of-range id (and a negative one) rather than reading past
the table; the block-quantized entry does not, so its caller owns the range.

### Templates

⊤ marks a template. Launch geometry is a property of the launcher, not of the
instantiation: every instantiation of a template listed above launches with that
template's row. The one place an argument enters a formula is
`q4e_matvec_combine_t_kernel`, whose grid uses `NT` - the row count of the f16
matmul it is combining, which is the *caller's* template argument, not its own.
