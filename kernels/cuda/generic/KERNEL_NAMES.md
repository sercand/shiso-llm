# generic entry-point names

The lookup table a loader resolves a kernel through. A module is loaded by its
registry key; a kernel inside it is fetched by the exact name recorded here.

Every entry in this unit carries C linkage, so each name below is the name
the driver answers to, verbatim. There are no mangled instantiations to capture:
the matvec variants are template *instantiations* wrapped one at a time in an
`extern "C"` entry point, precisely so that the host can write their names down
as `&'static str` literals. It has to, because it does: the engine names these
kernels itself, in `shiso_engine::launch::EXT_Q4_0_BY_NAME` and `plain_mv`, and
a name it writes that the module does not export is a model that loads and then
refuses at the first projection.

Pasted from what `bash scripts/q4e_ptx_oracle.sh --mode=ptx` printed on this
machine (nvcc 13.0.88, `compute_80`), not from intent. A name in this file that
the module does not export is worse than no file.

## Module `generic` — 21 entries

### The `q4_0` extended matvec — 15 entries

One entry per `(r1ptg, nxpsg)` pair. `r1ptg` is the token rows one threadgroup
takes and `nxpsg` the lanes that split the input; `ext_r1ptg` and `ext_nxpsg`
pick them independently, so all fifteen are reachable and all fifteen are here.

```
kernel_mul_mv_ext_q4_0_f32_r1_1_nx4
kernel_mul_mv_ext_q4_0_f32_r1_1_nx8
kernel_mul_mv_ext_q4_0_f32_r1_1_nx16
kernel_mul_mv_ext_q4_0_f32_r1_2_nx4
kernel_mul_mv_ext_q4_0_f32_r1_2_nx8
kernel_mul_mv_ext_q4_0_f32_r1_2_nx16
kernel_mul_mv_ext_q4_0_f32_r1_3_nx4
kernel_mul_mv_ext_q4_0_f32_r1_3_nx8
kernel_mul_mv_ext_q4_0_f32_r1_3_nx16
kernel_mul_mv_ext_q4_0_f32_r1_4_nx4
kernel_mul_mv_ext_q4_0_f32_r1_4_nx8
kernel_mul_mv_ext_q4_0_f32_r1_4_nx16
kernel_mul_mv_ext_q4_0_f32_r1_5_nx4
kernel_mul_mv_ext_q4_0_f32_r1_5_nx8
kernel_mul_mv_ext_q4_0_f32_r1_5_nx16
```

Geometry, the same the Metal twin is launched with: grid
`(ceil(out_dim / (32/nxpsg * 2)), ceil(n_tok / r1ptg), n_batch)`, block
`(32, 2, 1)`, no dynamic shared memory. Buffers after the argument block:
`src0` (packed weights), `src1` (activations), `dst`.

### The plain matvec — 2 entries

```
kernel_mul_mv_f16_f32_4
kernel_mul_mv_q8_0_f32
```

The first is the decode half of the per-layer model projection, which is the
only unquantized weight in either Gemma 4 container. The second is the whole of
how a `q8_0` weight is read below two tokens: `launch::ext_name_for` returns
`None` for that codec on a backend that specializes by name, so unlike every
other quantized format it has no extended instantiation to fall through from.
A qwen35 `Q4_K_M` container keeps 48 tensors there — `ssm_alpha` and
`ssm_beta`, one pair per recurrent layer — so the dense hybrid cannot decode a
token without it.

Geometry, both: grid `(ceil(out_dim / nr0), n_tok, n_batch)`, block
`(32, nsg, 1)`, 256 bytes of dynamic shared memory. Buffers after the argument
block: `src0`, `src1`, `dst`. `nsg` is the block's own second extent — on Metal
that count is a function constant, and `crates/shiso-cuda/src/launch.rs`
refuses a launch-time constant, so these twins read the host's number off the
launch instead. `nr0` is a field of the argument block for the f16 entry, which
the host sets to 2 or 4; the `q8_0` entry has one row count on both backends
(`N_R0_Q8_0` there, `PLAIN_ROWS` here) and reads neither.

The `_short` and wide spellings of the same family have no entry here. The
engine picks `_short` below an input width of 32 and the wide one for a width
that is not a multiple of four; neither width occurs in either Gemma 4
container, and a module that exported an entry with no caller would be claiming
coverage it has no oracle for.

### The gated SiLU — 1 entry

```
kernel_swiglu_flat_f32
```

The activation a qwen35 feed-forward closes on;
`kernel_dense_geglu_flat_f32` in the `dense` module is the same body at the
other curve, and `crates/shiso-engine/src/ffn.rs` picks between the two off the
checkpoint's own shape. Geometry: grid `(ceil(ne0 / 256), 1, 1)`, block
`(256, 1, 1)`, no shared memory. Buffers after the argument block: `src0` (the
gate projection), `src1` (the up projection), `dst`.

### The fused norms and the residual add — 3 entries

```
kernel_rms_norm_mul_f32_4
kernel_add_rms_norm_mul_f32_4
kernel_add2_f32
```

The first two: grid `(n_tok, 1, 1)`, block `(32, 1, 1)`, 128 bytes of dynamic
shared memory. Buffers after the argument block are `src0, src1_0, src1_1, dst`
for the plain norm — `src1_1` is the second fuse operand this instantiation
never reads, and the host binds the weight into it so the destination lands in
the slot it belongs in — and `src0, src1, weight, sum_dst, norm_dst` for the
fused close.

`kernel_add2_f32`: grid `(ceil(n / 256), 1, 1)`, block `(256, 1, 1)`, no shared
memory, buffers `a, b, out`.

## Where they are graded

`crates/shiso-cuda/tests/generic.rs` drives the cases in
`crates/shiso-kerneltest/src/generic/`, eleven in all: the two norms, the
residual add, the gated SiLU at both its clamp settings, the gather, every one
of the fifteen `q6_K` matvec pairs, the two halves of the f16 projection and
the two halves of the `q8_0` one — four of those cover entries that live in the
`qwen35` module next door. The `q4_0`
matvec and GEMM are graded by `crates/shiso-cuda/tests/dense_q4_0.rs`.
