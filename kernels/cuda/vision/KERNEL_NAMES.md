# vision entry-point names

The lookup table a loader resolves a kernel through. A module is loaded by its
registry key; a kernel inside it is fetched by the exact name recorded here.

Every entry in this directory carries C linkage, so each name below is the name
the driver answers to, verbatim. There are no templates here and therefore no
mangled names to capture.

## Two decompositions of one tower, and which one the engine is sequenced against

This directory holds **two** kernel sets, and they are not two spellings of one
thing. They are different decompositions of the same arithmetic:

- the **`kernel_vit_*` set**, which is what
  `crates/shiso-models/src/qwen35/vision/encoder.rs`,
  `crates/shiso-models/src/gemma4/vision/encoder.rs` and
  `crates/shiso-engine/src/vision/mod.rs` actually launch, under the module name
  `vision`. It is parametric: every width, head count, patch geometry, epsilon,
  rope base and softmax scale arrives in an argument block, because the two
  towers this tree loads differ in all of them.

The ruling is the second one, and the reason is not preference. **Six entry
names appear to match across the two sets and not one of the six is a matching
contract.** The evidence is per entry below. A name that matches with an
argument block that does not is the worst outcome available here, so it is
written down rather than summarised.

### `kernel_vit_layer_norm` vs `qwen3vl_layernorm_kernel` — REFUTED

| | Metal contract (`metal/vision/vit_norm_gelu.metal`) | `vit_norm.cu` |
|---|---|---|
| parameters | `args{rows,dim,eps}`, `out`, `in`, `weight`, `bias` | `out`, `in`, `weight`, `bias`, `dim` |
| argument block | index 0 | **none** |
| `rows` | an argument, guarded | implied by the grid, unguarded |
| `eps` | an argument | `SHS_QWEN3VL_VISION_EPS`, compile-time |
| shared memory | 132 B dynamic, host-sized | two static floats |

The arithmetic agrees — mean, then variance about the mean, then the affine map
over widened bf16 — but the host's first argument is a 12-byte struct and the
kernel's first parameter is `float *out`. Dispatching the engine's list at this
entry writes through an argument block reinterpreted as a pointer.

### `kernel_vit_qkv_rope` vs `qwen3vl_qkv_rope_kernel` — REFUTED

| | Metal contract | `vit_attn.cu` |
|---|---|---|
| parameters | `args{rows,heads,head_dim,grid_w,plane,rope_base}`, `q`, `k`, `v`, `qkv` | `q`, `k`, `v`, `qkv`, `bias`, `rows`, `grid_w` |
| bias | already applied, in the GEMM's epilogue | **applied here**, from a bf16 row |
| widths | arguments | `DIM` 1152, `HEAD_DIM` 72, `ROT` 36, constants |
| patch position | `local = n % plane`, `(local / grid_w, local % grid_w)` — grid-major | `qwen3vl_patch_rc` — **merge-block order**, 2x2 blocks row-major |
| temporal group | `plane` bounds it | no such concept |
| reads rotated from | the fused row `src` | `qd`/`kd`, after a `__syncthreads()` |

The frequency ladder is the same in both: `half = head_dim / 2`, the first
`half/2` pairs by the row axis and the second `half/2` by the column axis, at
`base^(-2j/half)`. That law is shared. Everything around it — the argument
block, the bias, the widths and **which patch a sequence row is** — is not.
The patch-ordering difference is the one that would fail silently: every buffer
the right size, every launch succeeding, and an image described rearranged.

### `kernel_vit_attention` vs `qwen3vl_attention_kernel` — REFUTED

| | Metal contract | `vit_attn.cu` |
|---|---|---|
| parameters | `args{rows,heads,head_dim,plane,scale}`, `out`, `q`, `k`, `v` | `out`, `q`, `k`, `v`, `rows` |
| softmax scale | **an argument** | `rsqrtf(head_dim)`, derived |
| key span | `[(n/plane)*plane, +plane)` | every row of the launch |
| widest head | `VIT_ATTN_SLOTS * 32` = 256 | `acc[3]` → 96 |
| final combine | `for d = tid; d < hd; d += nthreads` | `if (threadIdx.x < hd)` |

The derived scale is a hard blocker rather than a mismatch to paper over:
`crates/shiso-models/src/gemma4/vision/encoder.rs` passes `scale: 1.0` and the
qwen tower passes `1/sqrt(head_dim)`. That is exactly why
`crates/shiso-metal/tests/dense_vision.rs::vit_attention_takes_its_scale`
exists. A kernel that derives the number cannot serve both towers.

### `kernel_vit_gelu_tanh` vs `qwen3vl_gelu_tanh_kernel` — REFUTED

| | Metal contract | `vit_norm.cu` |
|---|---|---|
| parameters | `args{count}`, `x` | `x`, `bias`, `n`, `dim` |
| bias | already in `x` | **added here** |
| saturation | selected on the input at `+/-10` | **absent** |

The missing saturation guard is arithmetic, not plumbing. `vit_common.h`
records what its absence cost: the cubic drives `tanh`'s argument past 44 at an
input of about 10.3, a feed-forward row of this tower reaches that several dozen
times per image, and one NaN spreads over a whole row at the next normalization.

### `kernel_vit_gelu_erf` vs `qwen3vl_gelu_erf_kernel` — REFUTED

Same parameter mismatch as the tanh form: `(args{count}, x)` against
`(x, bias, n, dim)`. The error function differs as well — the Metal set carries
the Abramowitz and Stegun 7.1.26 rational form because the shading language has
none, and the CUDA extraction calls `erff`. That difference is about 1.5e-7 and
is not what refutes the row; the argument block is.

### `kernel_vit_gemm_bias` vs `qwen3vl_bias_kernel` — REFUTED, and not close

`kernel_vit_gemm_bias` is a blocked matrix multiply:
`out[m,n] = sum_k in[m,k] * weight[n,k] + bias[n]`, argument block `{m,n,k}`,
five parameters, grid `(ceil(n/16), ceil(m/16))` at a 16x16 block, two staged
tiles. Every projection in the tower goes through it.

`qwen3vl_bias_kernel` is `x[i] += bias[i % dim]` with an optional residual add.
It is a flat elementwise pass. **The two have no arithmetic in common.** The
name match is entirely spurious, and it is the single worst row in the table
that the plan's mapping proposed.

Two further corrections to that mapping fall out of the same reading:

- `qwen3vl_bias_kernel` is in fact the nearest thing this directory has to
  `kernel_vit_add_inplace`, not to `kernel_vit_gemm_bias` — its bias half is a
  periodic add with the period fixed at `dim`. It is still not that entry: the
  Metal one adds an f32 buffer at a host-chosen period and reads no bf16.
- **`kernel_vit_gemm_bias` therefore has no CUDA counterpart at all**, which
  makes three missing entries here rather than the two the plan counted, before
  the splice.

### `kernel_vit_add_inplace`, `kernel_vit_spatial_merge`, `kernel_vit_splice_rows`

No counterpart, as the plan says. Confirmed by reading the directory: no entry
here folds a merge block, and no entry here writes a destination row selected by
a signed per-row index.

### `qwen3vl_pos_embed_kernel`

Exists, and the encoder never asks for it. Confirmed and unchanged: the
resampling is done on the host, in `VisionEncoder::resample_positions`, and the
result reaches the device through `kernel_vit_add_inplace` at a period of one
grid. The asymmetry `HANDOFF.md` names is real.

### The ruling

Zero of six. Every candidate differs in its argument block; three differ in
arithmetic as well (the patch order, the derived scale, the missing saturation
guard); one is a different kernel entirely. So the `qwen3vl_*` set is left
exactly where it is — it is a fidelity-gated verbatim extraction and editing it
would be a defect — and the module the engine names gains the entries the
engine names, transcribed from the Metal contract.

That is the same decision Phase 7 took for `dense`, and for the same reason:
the engine and the two encoders are backend-neutral, so the names and the
argument blocks are theirs, not a backend's.

## The module name

`crates/shiso-models/src/qwen35/vision/encoder.rs:50`,
`crates/shiso-models/src/gemma4/vision/encoder.rs:67` and
`crates/shiso-engine/src/launch.rs:124` all name the module **`vision`**. On
this backend a module name is the key of a separately compiled image
(`crates/shiso-cuda/src/ptxmod.rs`), so an entry that is not in an image called
`vision` cannot be reached however it is spelled — which is why every vision
dispatch on this device refused before `vit.cu` existed, while the sibling
backend, whose one flat library resolves by entry name alone, never noticed.

The nine entries are therefore ONE translation unit, `vit.cu`, because
`crates/shiso-kernels/build_plan.rs` maps one source stem to one registry key
and refuses two stems claiming one name. The `.cuh` pieces beside it are split
for reading, not for linking.

## Modules

| Registry key | Unit | Entries |
|---|---|---|
| `vision` | `vit.cu` | 9 |
| `vision_norm` | `vit_norm.cu` | 4 |
| `vision_embed` | `vit_embed.cu` | 1 |
| `vision_attn` | `vit_attn.cu` | 2 |
| **total** | 4 units | **16** |

## Entries, by module

### `vision` — what the engine and both encoders dispatch

```
kernel_vit_gemm_bias
kernel_vit_layer_norm
kernel_vit_gelu_tanh
kernel_vit_gelu_erf
kernel_vit_add_inplace
kernel_vit_qkv_rope
kernel_vit_attention
kernel_vit_spatial_merge
kernel_vit_splice_rows
```

Every entry that reads a tower weight also has an `_f16` twin - the same body,
instantiated for a tower stored in IEEE halves rather than brain floats, with
the same parameters, grid and block: `kernel_vit_gemm_bias_f16` and
`kernel_vit_layer_norm_f16` here, and among the tensor-core path's entries
`kernel_vit_to_f16` (the twin of `kernel_vit_to_bf16`, writing half-precision
operand planes), `kernel_vit_add_bias_f16`, `kernel_vit_layer_norm_split_f16`,
`kernel_vit_bias_gelu_split_f16`, `kernel_vit_add_bias_residual_f16` and
`kernel_vit_qkv_rope_split_f16`. The encoder picks one width per tower.

Geometry and parameter order are `crates/shiso-metal/metal/vision/KERNELS.md`'s,
because the twins share an engine that names them and hands both the same
argument block and the same grid. The argument block is parameter 0 in all nine
— unlike the `dense` set's four affine matmuls, none of these binds it last.

| Entry point | Parameters after the argument block | grid | block | smem |
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

Argument blocks, field order as `vit_common.cuh` declares them and as the two
encoders pack them:

- `vit_args_gemm {m, n, k}`
- `vit_args_layer_norm {rows, dim, eps}`
- `vit_args_unary {count}`
- `vit_args_add {count, period}`
- `vit_args_qkv_rope {rows, heads, head_dim, grid_w, plane, rope_base}`
- `vit_args_attention {rows, heads, head_dim, plane, scale}`
- `vit_args_merge {merged_rows, hidden, grid_h, grid_w, merge}`
- `vit_args_splice {rows, hidden, src_rows}`

`vit_args_attention` carries **five** fields. `metal/vision/KERNELS.md`'s table
still lists four; the struct in `vit_common.h` has the scale and so does
`shiso_engine`'s mirror, and where that file and a kernel body disagree the body
is right. It is noted here rather than patched there, because that file is the
Metal set's contract and this is the CUDA set's names.

Three launch constraints are not visible in a signature:

- `kernel_vit_gemm_bias` must be launched at exactly a 16x16 block: the two
  staging tiles are `VIT_GEMM_TILE` squares and the thread's tile coordinates
  are its own `threadIdx`.
- the three flat kernels derive their index as `blockIdx.x * 256 + threadIdx.x`,
  as the Metal twins do, so 256 is the one block width they may take.
- `kernel_vit_attention` holds its accumulator in `VIT_ATTN_SLOTS` registers per
  lane, bounding `head_dim` at 256, and returns without writing above that
  rather than truncating silently.

### `vision` — the CUDA-only tensor-core entries (`vit_fast.cuh`)

```
kernel_vit_to_bf16
kernel_vit_add_bias_bf16
kernel_vit_layer_norm_split
kernel_vit_bias_gelu_split
kernel_vit_add_bias_residual
kernel_vit_split_kv
kernel_vit_qkv_rope_split
kernel_vit_attention_mma
```

These have no Metal twin. The Qwen encoder looks for them in the module's entry
list (`Backend::module_entries`) and runs the scalar set above when any is
missing, so a backend that does not compile them needs nothing. The
projections take `kernel_vit_to_bf16`, two brain-float GEMMs through the
device's BLAS, then `kernel_vit_add_bias_bf16`. The attention takes
`kernel_vit_split_kv` and then `kernel_vit_attention_mma`. Inside a block the
encoder fuses the elementwise passes between projections:
- `kernel_vit_layer_norm_split` writes the next projection's operand.
- `kernel_vit_bias_gelu_split` writes the second feed-forward layer's operand.
- `kernel_vit_add_bias_residual` carries each residual add's bias.
- `kernel_vit_qkv_rope_split` writes Q and the split K/V planes in one pass.

Each fused kernel repeats the unfused sequence's arithmetic, value for value.

| Entry point | Parameters after the argument block | grid | block | smem |
|---|---|---|---|---|
| `kernel_vit_to_bf16` | `out (2 * count bf16: hi, then lo), in` | `ceil(count/256)` | 256 | 0 |
| `kernel_vit_add_bias_bf16` | `x, bias (bf16)` | `ceil(count/256)` | 256 | 0 |
| `kernel_vit_layer_norm_split` | `out (2 * rows * dim bf16), in, weight, bias` | `rows` | 256 | 132 |
| `kernel_vit_bias_gelu_split` | `out (2 * count bf16), x, bias` | `ceil(count/256)` | 256 | 0 |
| `kernel_vit_add_bias_residual` | `x, bias, residual` | `ceil(count/256)` | 256 | 0 |
| `kernel_vit_split_kv` | `kp, vp, k, v` | `groups * plane_pad` | 256 | 0 |
| `kernel_vit_qkv_rope_split` | `q, kp, vp, qkv, bias` | `(heads, rows)` | 128 | 0 |
| `kernel_vit_attention_mma` | `out, q, kp, vp` | `(heads, groups * ceil(plane/128))` | 256 | 91 136 (`VIT_FA_SMEM`) |

Argument blocks:

- `kernel_vit_to_bf16` takes `vit_args_unary {count}`.
- `kernel_vit_add_bias_bf16` takes `vit_args_add {count, period}`, with `period` the bias width.
- `kernel_vit_layer_norm_split` takes `vit_args_layer_norm`.
- `kernel_vit_bias_gelu_split` and `kernel_vit_add_bias_residual` take `vit_args_add`.
- `kernel_vit_qkv_rope_split` takes `vit_args_qkv_rope_split {rows, heads, head_dim, grid_w, plane, plane_pad, rope_base}`, 28 bytes.
- `kernel_vit_split_kv` and `kernel_vit_attention_mma` take `vit_args_attention_mma {rows, heads, head_dim, plane, plane_pad, scale}`, 24 bytes.
- In both of the last two, `plane_pad` is `plane` rounded up to 64.

Two constraints are not visible in a signature:

- `kernel_vit_attention_mma` holds a head of at most `VIT_FA_HD` = 80 columns and
  must be launched at exactly 256 threads (eight warps of sixteen queries). The
  host sends a wider head to `kernel_vit_attention`.
- `kp` and `vp` are laid out as `kernel_vit_split_kv` writes them - two halves,
  head-major, each group padded to `plane_pad`, values transposed.
  `kernel_vit_qkv_rope_split` writes the same layout but only real key rows,
  so the host clears both planes once per encode.

### `vision_norm`

- `qwen3vl_layernorm_kernel`
- `qwen3vl_bias_kernel`
- `qwen3vl_gelu_tanh_kernel`
- `qwen3vl_gelu_erf_kernel`

### `vision_embed`

- `qwen3vl_pos_embed_kernel`

### `vision_attn`

- `qwen3vl_qkv_rope_kernel`
- `qwen3vl_attention_kernel`

## Launch geometry of the `qwen3vl_*` set

The geometry each kernel was written for, taken from its own indexing rather
than from a host that is not in this tree. `rows` is the patch rows of one
image, `dim` is 1152 and `hd` is 72.

| Entry | grid | block | dynamic shared |
|---|---|---|---|
| `qwen3vl_layernorm_kernel` | `rows` | a block the reduction covers | 0 |
| `qwen3vl_bias_kernel` | `ceil(n / block)` | any | 0 |
| `qwen3vl_gelu_tanh_kernel` | `ceil(n / block)` | any | 0 |
| `qwen3vl_gelu_erf_kernel` | `ceil(n / block)` | any | 0 |
| `qwen3vl_pos_embed_kernel` | `rows` | any | 0 |
| `qwen3vl_qkv_rope_kernel` | `(heads, rows)` | any | 0 |
| `qwen3vl_attention_kernel` | `(heads, rows)` | a multiple of 32 | `4 * (hd + warps * hd + 2 * warps)` |

Three of these have a launch constraint that is not visible in their signature
and is worth stating once:

- `qwen3vl_layernorm_kernel` reduces through the shared block sum, which folds at
  most 32 warps, so a block above 1024 threads silently drops the lanes past it.
- `qwen3vl_attention_kernel` keeps its accumulator in three registers per lane,
  which covers a head width up to `3 * 32 = 96`. This tower's 72 fits; a wider
  head would be truncated with nothing reporting it.
- `qwen3vl_attention_kernel`'s final combine is written by threads below `hd`,
  so a block narrower than the head width would leave part of every output row
  unwritten.

## What has run

The `vision` module's nine entries are graded by
`crates/shiso-kerneltest/src/vision/tower.rs`, lifted out of
`crates/shiso-metal/tests/vision_kernels.rs` into a crate neither backend owns,
and by `crates/shiso-kerneltest/src/dense/vision.rs` for the softmax scale.
Through `crates/shiso-cuda/tests/vision_kernels.rs` on a GB10 (`sm_121`, nvcc
13.0.88) that is 13 cases, 13 passed, 0 failed.

The seven `qwen3vl_*` entries still have not executed. Nothing dispatches them,
this tree has no oracle for them, and what is established about them is the
extraction: each unit is byte-identical to the lines of the upstream header it
names, after the one `extern "C"` delta, which
`scripts/q4e_ptx_oracle.sh --mode=fidelity` checks on every run of
`scripts/check.sh`.
