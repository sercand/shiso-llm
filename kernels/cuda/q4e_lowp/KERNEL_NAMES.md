# q4e_lowp entry-point names

The lookup table for the low-precision prompt kernels of the qwen4exp family.
Every entry carries C linkage, so the names below are the names the driver
answers to, verbatim. There are no template entries.

| Module | Unit | Plain entries |
|---|---|---|
| `q4e_nvfp4_a4` | `nvfp4_a4.cu` | 5 |
| `q4e_mxfp8_a8` | `mxfp8_a8.cu` | 2 |

Both modules are built for `sm_121a` (see `KERNEL.toml`) and exist only in a
kernel set compiled for a family-specific architecture; a device that cannot
load them leaves them unresolved, and the engine refuses
`--prefill-activations lowp` there.

## `q4e_nvfp4_a4`

The routed NVFP4 expert stages of a prompt pass with NVFP4 activations
(W4A4), on `mma.sync ... kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`.

- `q4e_nvfp4_a4_quant_kernel` - `f32` rows to NVFP4 (codes, `e4m3` scales
  per 16, an `f32` global per row, dynamic or static).
  `q, x, static_gs, static_idx, rows, k`.
- `q4e_nvfp4_a4_quant_bf16_kernel` - the same over brain-float rows.
- `q4e_nvfp4_a4_gate_up_kernel` - gate and up with the SwiGLU fused; writes
  brain-float `silu(g) * u` per routed column. `mid, gate_w, gate_s, gate_g,
  up_w, up_s, up_g, xq, sorted, bounds, active, map, in_dim, out_dim,
  n_expert, n_used, rows`.
- `q4e_nvfp4_a4_down_kernel` - down; one brain-float row per routed column.
  `out, down_w, down_s, down_g, midq, sorted, bounds, active, map, in_dim,
  out_dim, n_expert, n_used, rows`.
- `q4e_nvfp4_a4_combine_kernel` - each token's brain-float routed rows,
  weighted and summed in slot order into `f32`. `out, down, weights, n_embd,
  n_used`.

## `q4e_mxfp8_a8`

The dense MXFP8 projections of a prompt pass with MXFP8 activations (W8A8),
on `mma.sync ... kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32`.

- `q4e_mxfp8_a8_quant_kernel` - `f32` rows to MXFP8 (`e4m3` codes, a
  `ue8m0` scale per 32). `q, x, rows, k`.
- `q4e_mxfp8_a8_gemm_kernel` - `out[row][o] = sum_k w[o][k] x[row][k]`, 128 x
  128 tiles; `in_dim` a multiple of 512. `out, codes, scales, xq, in_dim,
  out_dim, rows`.
