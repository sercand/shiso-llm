# dense entry-point names

The lookup table a loader resolves a kernel through. A module is loaded by its
registry key; a kernel inside it is fetched by the exact name recorded here.

There is one module, `dense`, and one translation unit in it. That is forced
rather than chosen: on this backend a module name is the key of a separately
compiled image, the driver resolves entry names inside one image only, and the
engine dispatches this whole set under `launch::DENSE` ("dense"). Every entry
below therefore lives in `dense.cu`, which includes the focused `.cuh` pieces.

Every entry carries C linkage, so each name below is the name the driver answers
to, verbatim. The names are the host's: the engine writes them as `&'static str`
literals at the call sites in `crates/shiso-engine/src/dense`, and a name it
writes that the module does not export is a model that loads and then refuses at
the first layer.

Pasted from what `bash scripts/q4e_ptx_oracle.sh --mode=ptx` printed on this
machine (nvcc 13.0.88, `compute_80`), not from intent. A name in this file that
the module does not export is worse than no file.

## Module `dense` — 19 entries

Geometry and buffer order are `crates/shiso-metal/metal/dense/KERNELS.md`'s,
because the twins share an engine that names them and hands both the same
argument block and the same grid. `n` is the element count a flat kernel covers
and `n_tok` the rows of the chunk.

```
kernel_dense_geglu_flat_f32
kernel_dense_add_scale_rms_norm_mul_f32_4
kernel_dense_softcap_f32
kernel_dense_scale_f32
kernel_dense_q_norm_rope
kernel_dense_kv_norm_rope_store
kernel_dense_attn_decode
kernel_dense_attn_prefill
kernel_dense_ple_prepare_q6_K
kernel_dense_ple_prepare_affine4
kernel_dense_ple_gate_mul
kernel_dense_affine_qmv_4_f32
kernel_dense_affine_qmv_8_f32
kernel_dense_affine_qmm_4_f32
kernel_dense_affine_qmm_8_f32
kernel_dense_embed_affine4_f32
kernel_dense_vit_qkv_norm_rope2d
kernel_dense_vit_avgpool3_scale
kernel_dense_clamp_f32
```

| Entry point | Parameters after the argument block | grid | block | smem |
|---|---|---|---|---|
| `kernel_dense_geglu_flat_f32` | `gate, up, dst` | `ceil(ne0/256)` | 256 | 0 |
| `kernel_dense_add_scale_rms_norm_mul_f32_4` | `src0, src1, weight, sum_dst, norm_dst, scale` | `n_tok` | 32 | 128 |
| `kernel_dense_softcap_f32` | `x` | `ceil(n/256)` | 256 | 0 |
| `kernel_dense_scale_f32` | `x` | `ceil(n/256)` | 256 | 0 |
| `kernel_dense_q_norm_rope` | `q_out, q_in, w, pos` | `(n_head, n_tok)` | 128 | 2180 |
| `kernel_dense_kv_norm_rope_store` | `k_cache, v_cache, k_in, v_in, kw, pos, pages` | `(n_head_kv, n_tok)` | 128 | 2180 |
| `kernel_dense_attn_decode` | `out, q, k_cache, v_cache, pos, bound, pages` | `(n_head, n_tok)` | 256 | 16448 |
| `kernel_dense_attn_prefill` | `out, q, k_cache, v_cache, pos, bound, pages` | `(n_head, ceil(n_tok/16))` | 256 | 16384 |
| `kernel_dense_ple_prepare_q6_K` | `out, proj, norm_w, rows, ids` | `(n_layers, n_tok)` | 128 | 1156 |
| `kernel_dense_ple_prepare_affine4` | `out, proj, norm_w, rows, ids` | `(n_layers, n_tok)` | 128 | 1156 |
| `kernel_dense_ple_gate_mul` | `gate, ple` | `ceil(n_tok*dim/256)` | 256 | 0 |
| `kernel_dense_affine_qmv_4_f32` | `w, scales, biases, x, y` **then** the argument block | `out_dim` | 32 | 0 |
| `kernel_dense_affine_qmv_8_f32` | `w, scales, biases, x, y` **then** the argument block | `out_dim` | 32 | 0 |
| `kernel_dense_affine_qmm_4_f32` | `w, scales, biases, x, y` **then** the argument block | `(out_dim, ceil(n_tok/8))` | 32 | 0 |
| `kernel_dense_affine_qmm_8_f32` | `w, scales, biases, x, y` **then** the argument block | `(out_dim, ceil(n_tok/8))` | 32 | 0 |
| `kernel_dense_embed_affine4_f32` | `out, words, scales, biases, ids` | `(ceil(hidden/64), n_tok)` | 64 | 0 |
| `kernel_dense_vit_qkv_norm_rope2d` | `q, k, v, q_norm_w, k_norm_w` | `(heads, rows)` | 128 | `4 * (33 + 3 * head_dim)` |
| `kernel_dense_vit_avgpool3_scale` | `out, in` | `ceil(out_len/256)` | 256 | 0 |
| `kernel_dense_clamp_f32` | `out, in` | `ceil(count/256)` | 256 | 0 |

The rotary pair's 2180 bytes are compile-time, `4 * (DENSE_BLOCK_SUM_SLOTS +
DENSE_ATTN_HD)` = `4 * (33 + 512)`: the block reduction's slots first, the staged
head after them. `head_dim` may not exceed `DENSE_ATTN_HD` (512), which is what
the figure bounds, and sizing it off `head_dim` would pass at Gemma's width and
overrun at another. `n_rot` and `freq_dims` are separate arguments: with
`half = freq_dims / 2`, every `i` below `n_rot / 2` rotates `(v[i], v[i + half])`.

`kernel_dense_add_scale_rms_norm_mul_f32_4` takes its scale as a trailing `float`
parameter rather than a field of the argument block, because the block it shares
is the generic norm's, which every other norm entry also reads. Its block is one
warp: the work loop advances by `blockDim.x` alone and never reads a warp index,
so a taller block would have every extra warp recompute and rewrite the same row.

The two attention figures are compile-time as well: `16448 = 8*512*4 + 2*8*4`
(one accumulator row per warp, then the two partial arrays) and
`16384 = 512*16*2` (the transposed key tile). A query at `pos[t]` reads keys
`p_lo..=p_hi` with `p_lo = (window != 0 && pos + 1 > window) ? pos + 1 - window
: 0` and `p_hi = bound[t]` inclusive; `bound[t]` is NOT the query's own position
and neither end is derivable from the other. An empty span contributes a row of
zeros. `kernel_dense_attn_prefill` requires `head_dim % 32 == 0` because its
final store is unguarded; `kernel_dense_attn_decode` does not, guarding every
access with `i < head_dim`.

The per-layer prepare's 1156 bytes are `4 * (DENSE_BLOCK_SUM_SLOTS +
DENSE_PLE_DIM)` = `4 * (33 + 256)`: the block sum's slots first, the decoded row
after them. `dim` must be `DENSE_PLE_DIM` (256), which is one `q6_K` block
exactly. The projection scale is folded in BEFORE the sum of squares. The
`q6_K` slot inside a row is reached by stepping a `shs_dense_block_q6_K *`, so
the 210-byte stride comes from the struct; the `affine4` twin derives its three
row bases from the slot count it is handed, so neither is pinned to the shipped
checkpoint's shape.

The four affine matmul entries are the one place the argument block is NOT the
first parameter: it rides at index 5, after the five buffers, because that is
where the Metal twins bind it and the engine packs one argument list for both
backends. `KERNELS.md`'s geometry table lists them under "buffers after the
args", which is the one thing in that table these four do not do; the ordering
above is what the host actually passes and what the cases launch.

`kernel_dense_affine_qmm_*` carry EIGHT activation columns per threadgroup -
`shiso_engine::projection::AFFINE_GEMM_TOKENS` - which is why the grid divides
`n_tok` by eight. One column a threadgroup made a 64-token prefill read the whole
weight matrix 64 times. A row is walked as `in_dim / 64` groups, exact because
`projection::affine_geometry` refuses an `in_dim` that is not a whole number of
groups before a launch is ever built.

The three vision entries are Gemma 4's tower's, and they are here rather than in
the `vision` module for the reason the Metal originals are: they are written
against argument blocks rather than one checkpoint's widths, and they reuse this
set's `dense_block_sum` and `dense_rope_pairs`. Their argument blocks are
`dense_args_vit_rope2d {rows, heads, head_dim, grid_w, eps, rope_base}`,
`dense_args_vit_pool {grid_h, grid_w, hidden, scale}` and
`dense_args_clamp {count, min, max}`, each at parameter 0.

`kernel_dense_vit_qkv_norm_rope2d`'s shared allocation is host-sized rather than
compile-time: the block reduction's 33 slots, then the three staged heads. Its
rotation is two calls per plane, the leading half of the head by the patch's
COLUMN and the trailing half by its ROW, each over a ladder built for half the
head's width -- and the value plane is normalized by its own root mean square
with NO weight and never rotated.

`kernel_dense_clamp_f32` declares neither pointer `__restrict__`, and that is
deliberate: the same buffer as both `out` and `in` is a documented call, and a
restrict qualifier over an aliased pair is undefined behaviour rather than a
hint. `kernel_vit_layer_norm` in the `vision` module is the same shape for the
same reason.
