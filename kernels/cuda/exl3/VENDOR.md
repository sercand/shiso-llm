# Vendored exllamav3 EXL3 kernels

The `.cuh` files here are copied from turboderp's
[exllamav3](https://github.com/turboderp-org/exllamav3), the reference
implementation of the EXL3 trellis format, with the torch host wrappers left
behind. 

## Upstream pin

| Field   | Value |
|---------|-------|
| Source  | https://github.com/turboderp-org/exllamav3 |
| Commit  | `499890c75d20d8e7c9d061f37189ae611a5c9f0b` (v1.4.6, 2026-09-02) |
| License | MIT, Copyright (c) 2025 Turboderp (`LICENSE` in this directory) |

## File inventory

| File | Origin (`exllamav3/exllamav3_ext/`) | Status |
|------|------|--------|
| `codebook.cuh` | `quant/codebook.cuh` | verbatim (mul1 decode: `x * 0x83DCD12D`, byte sum, fp16 affine) |
| `exl3_dq.cuh` | `quant/exl3_dq.cuh` | verbatim (bit-window extraction for K = 1..8) |
| `hadamard_inner.cuh` | `quant/hadamard_inner.cuh` | include path only |
| `exl3_gemm_inner.cuh` | `quant/exl3_gemm_inner.cuh` | include path only |
| `exl3_gemm_kernel.cuh` | `quant/exl3_gemm_kernel.cuh` | **modified**, see below |
| `exl3_kernel_map.cuh` | `quant/exl3_kernel_map.cuh` | **modified**: argument lists; host declarations dropped |
| `exl3_moe_common.cuh` | `quant/exl3_moe_common.cuh` | **modified**: ds4 argument list (see below) |
| `exl3_moe_kernel.cuh` | `quant/exl3_moe_kernel.cuh` | **modified**, see below |
| `exl3_moe_ws.cuh` | none | **shiso's own**: the weight-stationary routed prefill (prep, gate+up, down with a float or a half slot output, and the half-slot combine that also adds the gated shared expert), and its decode-shaped twin (`exl3_moe_wsd_*`: the same MMA sequence per column over the active experts in 16-row passes, the Hadamard epilogues as launches of their own), built on exllamav3's decoder, Hadamard helpers and PTX wrappers |
| `exl3_dgemv.cuh` | none | **shiso's own**: the decode-shaped dense projection (input transform into MMA fragment order, split-K slices, a last-block fold in slice order), built on exllamav3's decoder, Hadamard helpers and PTX wrappers |
| `exl3_reconstruct.cuh` | `quant/reconstruct.cu` | `reconstruct_had_kernel` only; C++17 `register` dropped |
| `exl3_devctx.cuh` | `quant/exl3_devctx.cuh` | lock-buffer layout constants only |
| `ptx.cuh` | `ptx.cuh` | verbatim (mma.m16n8k16, cp.async, barriers) |
| `util.cuh` | `util.cuh` | cuBLAS error helpers removed |
| `compat.cuh` | `compat.cuh` | verbatim |

## ds4 modifications to the kernels

- Activations are fp32.  ds4 keeps every qwen4exp activation in fp32, so the
  GEMM's `A` and the mgemm's `A` are `const float*` and the input Hadamard
  phase (`had_fh_r_128_inner`) converts to the fp16 `A_had` the MMA loop
  reads.  Outputs are fp32 (`c_fp32 = true` is the only instantiation).
- `exl3_mgemm_kernel` addresses the experts of a projection as one stacked
  tensor: matrix `q` is `B_base + q * B_stride`, `suh_base + q * k`,
  `svh_base + q * n`, selected per slot by `int32` ids (ds4's router output).
  The pointer-table arguments, the expert-range filtering, the weighted
  reduction into slot 0 and the per-matrix width lists are removed; ds4 does
  the routing-weight combine in its own kernel.  A second stacked tensor
  (`B_base2`, `suh_base2`, `svh_base2`, `C2`, `split`) lets one launch run the
  slots over a gate and an up projection that share their input.
- Launch geometry (`exl3.cu`) is tuned for GB10 rather than autotuned:
  the dense GEMM runs four blocks per output tile column (8..48), the mgemm
  one block group per slot.
- `exl3_moe_kernel` (the fused routed-expert prefill block) takes the same
  treatment: fp32 `hidden_state` gathered by `had_fh_r_128_inner`, the nine
  pointer tables replaced by three stacked tensors, the routing as int32
  expert bounds plus expert-sorted slots (the map `cuda/mmq`'s
  `shs_mmq_moe_map_build` already produces), one bit width per instance,
  SiLU only, staging sized by the token count (so the per-expert overflow
  fallback is gone).  Its weighted atomic scatter-add is replaced by a
  per-assignment store (`had_hf_r_128_out_inner`) that ds4's own combine
  kernel sums: the atomics' order-dependent rounding flipped router near-ties
  at 26k tokens from run to run.  Launched cooperatively so the co-residency
  its group barriers assume is enforced rather than hoped for.
- Prefill of the dense tensors follows exllamav3's `reconstruct_hgemm`: above
  144 rows `shs_exl3_reconstruct` expands the tensor to fp16 in the original
  basis and cuBLAS runs the GEMM on the raw activations .
- Instances: K = 4, 5, 6 only, codebook mul1 only (`exl3.cu`).  The
  QTIP-style GEMV and int8 GEMV fast paths are not vendored: upstream keeps
  them off on Blackwell and they exclude K = 6.

Why vendored rather than depended on: like `cuda/mmq`, ds4 is a flat C/CUDA
tree with no build dependencies, and exllamav3 is a torch extension.
