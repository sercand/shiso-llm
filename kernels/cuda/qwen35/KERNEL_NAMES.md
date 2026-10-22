# qwen35 entry-point names

The lookup table a loader resolves a kernel through. A module is loaded by its
registry key; a kernel inside it is fetched by the exact name recorded here.

Every entry in this unit carries C linkage, so each name below is the name
the driver answers to, verbatim. There are no mangled instantiations to capture:
the matvec variants are template *instantiations* wrapped one at a time in an
`extern "C"` entry point, precisely so that the host can write their names down
as `&'static str` literals. It has to, because it does: the engine names these
kernels itself, in `shiso_engine::launch::EXT_Q6_K_BY_NAME`, `mm_name` and
`layers.rs`'s gather table, and a name it writes that the module does not export
is a model that loads and then refuses at the first projection.

Pasted from what `bash scripts/q4e_ptx_oracle.sh --mode=ptx` printed on this
machine (nvcc 13.0.88, `compute_80`), not from intent. A name in this file that
the module does not export is worse than no file.

## Module `qwen35` — 59 entries

The module name is the engine's — `launch::Q35`. Eighteen of these are the
entries a Gemma 4 pass asks for under that name and belong to no family; the
`q8_0` GEMM belongs to no family either but is reached only by this one; the
rest are the Qwen 3.5 dense hybrid's own.

### The rotary and norm pair — 4 entries

```
kernel_q35_q_norm_rope
kernel_q35_q_norm_rope_mrope
kernel_q35_k_norm_rope_kv
kernel_q35_k_norm_rope_kv_mrope
```

Twins of `kernels/metal/qwen35/rmsnorm_rope.metal`, over the laws
`kernels/cuda/q4e/qsa.cu` carries as `q4e_qsa_q_norm_rope_kernel` and
`q4e_qsa_store_kv_kernel`. Geometry: grid `(n_head, n_tok)` for the query pair
and `(n_head_kv, n_tok)` for the key pair, block `(128, 1, 1)`, 1156 bytes of
dynamic shared memory — `Q35_BLOCK_SUM_SLOTS` plus one staged head row at
head_dim 256.

Buffers after the argument block: `q_out, gate_out, qkv, w, pos` for the query
entry, with the `_mrope` one replacing `pos` by `pos_t, pos_h, pos_w`;
`k_cache, v_cache, k, v, kw, pos, pages` for the key entry, with the `_mrope`
one APPENDING `pos_t, pos_h, pos_w` and keeping `pos`. The key entry keeps the
linear position because that, and only that, reaches the page table.

The two `_mrope` entries reuse the same two argument blocks unchanged: three
position streams are three more buffers, not more fields.


### The prefill GEMMs — 5 entries

```
kernel_mul_mm_q4_0_f32
kernel_mul_mm_f16_f32
kernel_mul_mm_q4_K_f32
kernel_mul_mm_q8_0_f32
kernel_mul_mm_q6_K_f32
```

One per codec `launch::mm_name` answers for, and there is no sixth: a name that
table hands out and this module does not export is a prefill that refuses at
its first chunk of two tokens or more. The `q8_0` one is this family's two gate
projections, `ssm_alpha` and `ssm_beta`. The `q6_K` one is the down and value
projections a `Q4_K_M` export keeps at that codec, about half of each.

`launch::mm_name` hands every codec's tiled product to this module whatever
family the weight belongs to. Geometry: grid
`(ceil(n_tok / 32), ceil(out_dim / 64), n_batch)`, block `(128, 1, 1)`, 8192
bytes of dynamic shared memory. Buffers after the argument block: `src0`,
`src1`, `dst`.

### The prefill staging kernels — 5 entries

```
kernel_dequant_plane_q4_0_f32
kernel_dequant_plane_f16_f32
kernel_dequant_plane_q4_K_f32
kernel_dequant_plane_q8_0_f32
kernel_dequant_plane_q6_K_f32
```

The first half of the vendor-GEMM route: `cublas` has no quantized operand
type, so `projection::blas_projection` widens the whole weight plane to f32
and multiplies that. One per codec above and for the same reason - the two
tables are read together, and a codec with a tile and no stager would silently
never take the faster route.

The output is row-major, `out_dim` rows of `in_dim` f32, which is how the
weight is already stored; read column-major that is the transposed left
operand, so nothing is repacked. Geometry: grid `(out_dim, 1, 1)`, block
`(256, 1, 1)`, no shared memory. Buffers after the argument block: `src`,
`dst`.

### The `q6_K` vocabulary gather — 1 entry

```
kernel_get_rows_q6_K_f32
```

`crates/shiso-engine/src/layers.rs` maps `DType::Q6_K` here, and a Gemma 4 GGUF
needs it because the QAT export packs the vocabulary table at `q6_K` whatever
its linear layers are packed at. Geometry: grid
`(ceil(n_embd / 256), n_tokens, 1)`, block `(256, 1, 1)`, no shared memory.
Buffers after the argument block: `src0` (the table), `src1` (the ids), `dst`.

### The `q6_K` extended matvec — 15 entries

```
kernel_mul_mv_ext_q6_K_f32_r1_1_nx4
kernel_mul_mv_ext_q6_K_f32_r1_1_nx8
kernel_mul_mv_ext_q6_K_f32_r1_1_nx16
kernel_mul_mv_ext_q6_K_f32_r1_2_nx4
kernel_mul_mv_ext_q6_K_f32_r1_2_nx8
kernel_mul_mv_ext_q6_K_f32_r1_2_nx16
kernel_mul_mv_ext_q6_K_f32_r1_3_nx4
kernel_mul_mv_ext_q6_K_f32_r1_3_nx8
kernel_mul_mv_ext_q6_K_f32_r1_3_nx16
kernel_mul_mv_ext_q6_K_f32_r1_4_nx4
kernel_mul_mv_ext_q6_K_f32_r1_4_nx8
kernel_mul_mv_ext_q6_K_f32_r1_4_nx16
kernel_mul_mv_ext_q6_K_f32_r1_5_nx4
kernel_mul_mv_ext_q6_K_f32_r1_5_nx8
kernel_mul_mv_ext_q6_K_f32_r1_5_nx16
```

The output head: Gemma 4 ties it to the vocabulary table, and `q6_K` has no
tiled product, so every width of that product comes through this family rather
than through a GEMM at the wide end. `launch::ext_module` sends this codec to
`qwen35` and `q4_0` to `generic`, which is why the same fifteen names appear
under two modules with two codecs. Geometry and buffer order are the `q4_0`
family's above.

### Gated attention — 3 entries

```
kernel_q35_attn_decode
kernel_q35_attn_prefill
kernel_q35_attn_gate_mul
```

Twins of `kernels/metal/qwen35/gated_attn.metal` and
`gated_attn_prefill.metal`, over the laws `kernels/cuda/q4e/qsa.cu` carries as
`q4e_qsa_attention_kernel`, `q4e_qsa_attention_tiled_kernel` and
`q4e_qsa_gate_kernel`. Geometry: decode grid `(n_head, n_tok)`, block
`(256, 1, 1)`, 8256 bytes — `s_v[8][256]` plus the two eight-slot partial
arrays; prefill grid `(n_head, ceil(n_tok / 16))`, same block, 16384 bytes —
the transposed key tile at head_dim 256; the gate flat at
`(ceil(n_tok * n_head * head_dim / 256), 1, 1)`, same block, no shared memory.

Buffers after the argument block: `out, q, k_cache, v_cache, pos, pages` for
both attention entries, `x, gate` for the gate. The two attention entries share
one argument block, as they do on the Metal side.

`hkv = h / (n_head / n_head_kv)` is an integer divide. The delta rule's key
head is `h % n_head_k`, a modulo. The two live ten lines apart in the source
both were extracted from.

### The gated delta network — 13 entries

```
kernel_q35_gdn_conv
kernel_q35_gdn_l2norm
kernel_q35_gdn_gates
kernel_q35_gdn_out_gate
kernel_q35_gdn_recurrent_s1
kernel_q35_gdn_recurrent_s2
kernel_q35_gdn_recurrent_s4
kernel_q35_gdn_recurrent_hybrid
kernel_q35_gdn_chunk_c1
kernel_q35_gdn_chunk_c2
kernel_q35_gdn_chunk_c4
kernel_q35_gdn_chunk_c8
kernel_q35_gdn_chunk_hybrid
```

Twins of `kernels/metal/qwen35/gdn_conv.metal`,
`gdn_recurrent.metal`, `gdn_recurrent_hybrid.metal`, `gdn_chunk.metal` and
`gdn_chunk_hybrid.metal`, over the laws `kernels/cuda/q4e/gdn.cu` carries.

Geometry, per `KERNELS.md`: the conv flat at `ceil(channels / 256)`, block
`(256, 1, 1)`, no shared memory; `l2norm` at grid `(n_heads, n_tok)`, block
`(128, 1, 1)`, 132 bytes; the gates flat at `ceil(n_head * n_tok / 256)`, block
`(256, 1, 1)`, no shared memory; `out_gate` at grid `(n_tok * n_head, 1, 1)`,
block `(128, 1, 1)`, 132 bytes.

The delta rule is grid `(n_head_v, 1, 1)` for every variant, at the block the
variant's index arithmetic assumes — 128 for `_s1`, `_recurrent_hybrid`, `_c1`
and `_chunk_hybrid`, 256 for `_s2`/`_c2`, 512 for `_s4`/`_c4`, 1024 for `_c8` —
with 1156 bytes for the register-resident recurrent variants, 17540 for the
recurrent hybrid, and 16896 for every chunk variant. Each entry carries a
`__launch_bounds__` of exactly its lane count, so the driver reports that
number through `cuFuncGetAttribute` and `shiso_engine::tune` can pick among
them; `crates/shiso-cuda/tests/q35_gdn.rs` asserts all nine.

Buffers after the argument block: `out, state, x, w` for the conv; `qk` for the
L2 norm; `decay, beta, alpha_proj, beta_proj, dt_bias, a` for the gates;
`out, attn, z, w` for the output gate; `attn_out, state, qkv, decay, beta` for
every delta-rule entry, with `kernel_q35_gdn_chunk_hybrid` alone APPENDING
`dreg_spill` — the only delta-rule entry point that is not the six-buffer one.

`kernel_q35_gdn_out_gate` applies a SILU gate, not the sigmoid the law
`q4e_gdn_out_gate_kernel` applies. `gdn_conv.metal` explains why and the
committed goldens were produced through the silu.

### The `q5_K` extended matvec — 15 entries

```
kernel_mul_mv_ext_q5_K_f32_r1_1_nx4
kernel_mul_mv_ext_q5_K_f32_r1_1_nx8
kernel_mul_mv_ext_q5_K_f32_r1_1_nx16
kernel_mul_mv_ext_q5_K_f32_r1_2_nx4
kernel_mul_mv_ext_q5_K_f32_r1_2_nx8
kernel_mul_mv_ext_q5_K_f32_r1_2_nx16
kernel_mul_mv_ext_q5_K_f32_r1_3_nx4
kernel_mul_mv_ext_q5_K_f32_r1_3_nx8
kernel_mul_mv_ext_q5_K_f32_r1_3_nx16
kernel_mul_mv_ext_q5_K_f32_r1_4_nx4
kernel_mul_mv_ext_q5_K_f32_r1_4_nx8
kernel_mul_mv_ext_q5_K_f32_r1_4_nx16
kernel_mul_mv_ext_q5_K_f32_r1_5_nx4
kernel_mul_mv_ext_q5_K_f32_r1_5_nx8
kernel_mul_mv_ext_q5_K_f32_r1_5_nx16
```

A `Q4_K_M` qwen35 container keeps 48 tensors at `q5_K`, and `q5_K` has no tiled
product — `launch::mm_name` returns `None` for it — so every width of that
projection comes through this family rather than through a GEMM at the wide end.
`launch::ext_module` sends this codec here beside `q6_K`. Geometry and buffer
order are the `q6_K` family's above.

### The recorded recurrence — 4 entries

```
kernel_q35_gdn_recurrent_trail_s1
kernel_q35_gdn_recurrent_trail_s2
kernel_q35_gdn_recurrent_trail_s4
kernel_q35_gdn_conv_trail
```

The recurrence and the convolution with the state (or window) after every row
but the last written into a trail of `n_tok - 1` records laid out as the state
is: what a verify pass needs to wind a session back to any accepted prefix. The
recurrent three are the plain recurrence's body with a trail pointer appended
after `beta`; the convolution's appends it after `w`. Geometry is the plain
entries'. Graded by `recurrent_trail_records_are_the_per_row_walk` and
`conv_trail_records_are_the_per_row_walk` in `crates/shiso-cuda/tests/q35_gdn.rs`.

## Where the family's own entries are graded

`crates/shiso-cuda/tests/q35_norm_rope.rs` drives
`crates/shiso-kerneltest/src/qwen35/norm_rope.rs`, six cases of which five are
device ones; `crates/shiso-cuda/tests/q35_attn.rs` drives
`crates/shiso-kerneltest/src/qwen35/attn.rs`, five of which four are device
ones; `crates/shiso-cuda/tests/q35_gdn.rs` drives
`crates/shiso-kerneltest/src/qwen35/gdn.rs`, nine device cases, plus one
CUDA-only case on the block-limit probe the variant selection turns on.

`crates/shiso-kerneltest/src/qwen35/chunk_fold.rs` has no driver here and is
not meant to: `CSPLIT` is a Metal answer to a threadgroup-memory cap this
device does not have, and those five cases run as `shiso-kerneltest`'s own unit
tests on every machine.

## Where the rest are graded

`crates/shiso-cuda/tests/generic.rs` drives the cases in
`crates/shiso-kerneltest/src/generic/` that cover the `q6_K` gather, the fifteen
`q6_K` matvec pairs, the `f16` halves of the projection and the `q8_0` GEMM.
The `q4_0` GEMM is graded by `crates/shiso-cuda/tests/dense_q4_0.rs`. Both case sets predate this
directory and neither moved with the unit: they name entries and a module, not a
directory.

The `q5_K` matvec and the `q4_K` GEMM are graded by
`crates/shiso-cuda/tests/q35_gemv.rs`, which drives
`crates/shiso-kerneltest/src/qwen35/gemv.rs`. Only the `_r1_1_nx16` pair of the
fifteen is launched there: the case's fixture is one token and one quant block
per row, and the remaining fourteen are the same body at a different row
blocking and lane split. The `q4_0` family's case
(`crates/shiso-kerneltest/src/dense/q4_0.rs`) walks all fifteen of its own and
is what holds the template's blocking arithmetic.
