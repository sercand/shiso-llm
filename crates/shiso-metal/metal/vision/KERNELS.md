# vision tower Metal kernel contract

The dispatch contract for the encoder: entry-point names, argument blocks,
binding order, launch geometry and buffer layouts. Where this file and a kernel
body disagree, the kernel is right and this file is the bug.

## Conventions

- `[[buffer(0)]]` is always the argument block (`constant vit_args_* &`). Buffers
  follow in declaration order at `[[buffer(1)]]`, `[[buffer(2)]]`... and the
  runtime binds the i-th `KernelArg` to binding index i, so the buffer lists
  below are the bind order, in order. Threadgroup memory is bound separately and
  never occupies an argument index.
- `grid` in a launch is the threadgroup count, dispatched as given: no division,
  no rounding. `block` is the threads-per-threadgroup the kernel reads as
  `threads_per_threadgroup`; every kernel below derives its stride from that
  value rather than from a literal, so a caller may narrow a block without
  changing what the kernel computes.
- Reductions only go through `vit_warp_sum`/`vit_warp_sum_all`/`vit_block_sum`.
  Bare `simd_sum` has an unspecified summation order, and there is no upstream
  shading-language source for this tower to compare against -- the naive f32
  oracle is the whole gate, so the kernels are written against a fixed reduction
  shape rather than a mathematically equal one.
- **Activations are f32; weights are bf16.** Every buffer named `in`, `out`,
  `x`, `add`, `q`, `k`, `v` or `qkv` below is f32, and every buffer named
  `weight` or `bias` is bf16, widened on read by shifting the 16 bits into the
  high half of an f32. Accumulation is f32 throughout. The residual stream this
  tower feeds is f32, so nothing here narrows a value that later has to be added
  to a hidden state.
- Dimensions are arguments, never constants. Nothing in this set assumes a
  width, a depth, a head count or a patch geometry; the two towers the loader
  reads differ in all four.

## Geometry

`rows` is the patch rows of one media item, `plane` the patch rows of one
temporal group (`grid_h * grid_w`), `hidden` the tower's internal width,
`merged_rows` the rows the merger emits.

| Entry point | Buffers after the args | grid | block | smem |
|---|---|---|---|---|
| `kernel_vit_gemm_bias` | `out, in, weight, bias` | `(ceil(n/16), ceil(m/16))` | `(16, 16)` | 0 |
| `kernel_vit_layer_norm` | `out, in, weight, bias` | `rows` | 256 | 132 |
| `kernel_vit_gelu_tanh` | `x` | `ceil(count/256)` | 256 | 0 |
| `kernel_vit_gelu_erf` | `x` | `ceil(count/256)` | 256 | 0 |
| `kernel_vit_add_inplace` | `x, add` | `ceil(count/256)` | 256 | 0 |
| `kernel_vit_qkv_rope` | `q, k, v, qkv` | `(heads, rows)` | 128 | 0 |
| `kernel_vit_attention` | `out, q, k, v` | `(heads, rows)` | 128 | `4 * (head_dim * (block/32 + 1) + 2 * block/32)` |
| `kernel_vit_spatial_merge` | `out, in` | `merged_rows` | 256 | 0 |
| `kernel_vit_splice_rows` | `hidden, src, map` | `rows` | 256 | 0 |
| `kernel_vit_gemm64_bias` | `out, in, weight, bias` | `(ceil(n/64), ceil(m/64))` | 128 | 0 |
| `kernel_vit_flash_attn_d{64,72,80}` | `out, q, k, v` | `(groups * ceil(plane/64), heads)` | 256 | 0 |

`kernel_vit_gemm_bias_f16` and `kernel_vit_layer_norm_f16` are the same two
entries for a tower stored in IEEE halves rather than brain floats: same
buffers, grid, block and threadgroup memory, one body each, instantiated per
width. The encoder picks one width per tower.

## The simdgroup-matrix entries

`kernel_vit_gemm64_bias` (and its `_f16` twin) and `kernel_vit_flash_attn_d*`
exist only in this Metal set. They are the counterpart of the CUDA set's
tensor-core path. The encoder resolves them per tower, and an absent or
unresolvable entry falls back to the scalar kernel above it.

- `kernel_vit_gemm64_bias` computes `kernel_vit_gemm_bias`'s product with
  `k % 16 == 0` and 16-byte-aligned input rows. The encoder sends every other
  width to the blocked kernel. Operands stay f32, the weights are widened
  exactly, and accumulation is in 8-k steps in k order. Its staging is 9472 B
  of fixed threadgroup arrays.
- `kernel_vit_flash_attn_d{64,72,80}` computes `kernel_vit_attention`'s
  output for a head that wide. It takes the same argument block, applies the
  same `scale` to every score, and attends over the same temporal-group span.
  It runs in one pass with an online softmax, everything f32, and its key and
  value staging is a fixed `2 * 32 * (HD + 4)` floats. A wider head would not
  fit two f32 blocks in 32 KiB, so none is instantiated.

`kernel_vit_gemm_bias` reports 0 threadgroup bytes because its two staging tiles
are fixed-size threadgroup arrays declared in the kernel, not a dynamic
allocation the host sizes: 2 x 16 x 16 x 4 = 2048 B, which the compiler reserves.

`kernel_vit_attention`'s threadgroup memory is the only host-sized allocation in
the set: the query row, one accumulator per simdgroup, then the per-simdgroup
softmax maximum and denominator. At the shipped tower's head width of 64 and a
128-thread block that is 4 * (64 * 5 + 8) = 1312 B.

## Argument blocks

Field order is the struct order in `vit_common.h`; the host mirrors it field for
field.

- `vit_args_gemm {m, n, k}`
- `vit_args_layer_norm {rows, dim, eps}`
- `vit_args_unary {count}`
- `vit_args_add {count, period}`
- `vit_args_qkv_rope {rows, heads, head_dim, grid_w, plane, rope_base}`
- `vit_args_attention {rows, heads, head_dim, plane}`
- `vit_args_merge {merged_rows, hidden, grid_h, grid_w, merge}`
- `vit_args_splice {rows, hidden, src_rows}`

## Buffer layouts

- **Patch rows and hidden states**: `[rows][hidden]` f32, one row per patch, in
  the order the preprocessor emitted them -- grid-major within a temporal group
  (row by row across the frame), groups back to back. That order is shared by
  three things that must agree: the rotation's grid positions, the position
  embedding the host resamples, and which patches the merge block gathers. It is
  stated once, here, and derived nowhere else.
- **Linear weights**: `[out_width][in_width]` bf16, output width slowest. That is
  how a checkpoint writes a linear layer, so no transpose pass runs anywhere in
  this tower. `kernel_vit_gemm_bias` reads it as `weight[n * k + i]`.
- **Fused projection** (`qkv`): `[rows][3 * hidden]` f32, laid out as
  `[q | k | v]` -- query at 0, keys at `hidden`, values at `2 * hidden`, one
  head-width per head inside each third. The bias is already applied: it rides
  the GEMM's epilogue, so `kernel_vit_qkv_rope` only reshapes and rotates.
- **Rotated planes** (`q`, `k`, `v`): `[rows][hidden]` f32 each, head `h` at
  offset `h * head_dim`.
- **Merged rows**: `[merged_rows][merge^2 * hidden]` f32. Output row `o` belongs
  to temporal group `o / (out_gh * out_gw)`; inside a group the rows are
  grid-major over the merged grid, so the merger's output is in the same order
  as its input at a coarser scale.
- **Position embedding** (`add` of `kernel_vit_add_inplace`): `[plane][hidden]`
  f32, already resampled onto this item's grid by the host. `period` is
  `plane * hidden`, so a clip's temporal groups each receive the same grid's
  worth of vectors.

## The splice

`kernel_vit_splice_rows` is the only entry in this set that does not belong to
the tower. It is the seam: the language model's prefill embeds every token of a
chunk, placeholders included, and this overwrites the placeholder rows with the
rows the merger produced. Both buffers are f32 `[rows][hidden]` at the LANGUAGE
model's hidden width, not the tower's.

`map` is one `int32` per token row of the chunk. A negative entry is a text
token and is left alone; a non-negative one names the encoder row that token
takes. The host builds it in the same walk that builds the multimodal position
streams, which is what keeps the two from disagreeing about where a placeholder
run started. An entry at or past `src_rows` is skipped rather than read: the row
count is checked on the host, where a mismatch can still be reported, and this
is the last line of defence against handing the model whatever followed the
tower's output.

## The rotation

The rotation is two-dimensional and applies to the whole head.

- Channels pair `(i, i + head_dim/2)`, the same pairing the language model's
  blocks use.
- The first `head_dim/4` pairs rotate by the patch's ROW in its frame's grid and
  the second `head_dim/4` by its COLUMN.
- Both axes share one frequency ladder, built for a width of `head_dim/2`:
  `inv[j] = rope_base ^ (-2j / (head_dim/2))` for `j` in `[0, head_dim/4)`.

A checkpoint that describes this as a partial rotation is describing how those
frequencies are built, not a subset of channels being left alone. Every channel
of every head rotates.

## Attention span

Attention is bidirectional -- a patch sees every patch, there is no causal mask
and no window -- and bounded by the temporal group: a query at row `n` attends
over `[(n / plane) * plane, (n / plane) * plane + plane)`. For a still image
`plane == rows` and that is the whole item. For a clip, each temporal group is
its own span and no query attends across a frame boundary.

## Two activations, deliberately

The blocks apply the tanh approximation of GELU; the merger applies the exact
error-function form. They differ by about 1e-4 per value, which is why they are
two entry points rather than one with a flag: the choice belongs to the layer,
and a branch that picked between them could be got wrong at one call site
without anything failing loudly.

## Threadgroup memory

Nothing in this set exceeds the measured 32768 B device limit, and no kernel
here replaces a shared-memory shape with registers to fit. The one kernel that
holds per-lane state in registers, `kernel_vit_attention`, does so because its
accumulator is touched once per key: staging it would be a round trip per key
position rather than a saving.
