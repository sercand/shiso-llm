# diffusion entry-point names

The lookup table a loader resolves a kernel through. A module is loaded by its
registry key (`diffusion`); a kernel inside it is fetched by the exact name
recorded here.

Every entry carries C linkage and is spelled out in full in the source
(`extern "C" __global__ void <name>(`, no macro generates a name), so each
name below is the name the driver answers to, verbatim and unmangled. These are
the names `REQUIRED_KERNELS` in `crates/shiso-models/src/diffusion/ops.rs`
lists, and the names the Metal twin
(`crates/shiso-metal/metal/diffusion/diffusion.metal`) registers. The host-only
test `required_kernels_all_present_on_cuda` (same file) fails when a name in
that list is missing from this file or from the sources.

**Status: written on a machine with no CUDA toolchain; these names are read from
the source, not pasted from compiled PTX.** The first CUDA build should replace
this with the output of
`grep -o '\.visible \.entry [^(]*' <module>.ptx` (module `diffusion`).

## Module `diffusion` - 59 entries

```
kernel_dit_add_f32
kernel_dit_add_gated_f32
kernel_dit_axpy_f32
kernel_dit_channel_rms_inv_f32
kernel_dit_concat_rows_f32
kernel_dit_conv2d_f32
kernel_dit_conv2d_gemm_f32
kernel_dit_conv2d_rms_gemm_f32
kernel_dit_conv2d_up2_gemm_f32
kernel_dit_dup_up_add_f32
kernel_dit_embed_bf16_f32
kernel_dit_flash_sdpa_causal_d128_f32
kernel_dit_flash_sdpa_causal_f32
kernel_dit_flash_sdpa_d128_f32
kernel_dit_flash_sdpa_f32
kernel_dit_gelu_tanh_f32
kernel_dit_gemm64_bf16
kernel_dit_gemm64_f32
kernel_dit_gemm64_fp8
kernel_dit_gemm64_fused_bf16
kernel_dit_gemm64_fused_f32
kernel_dit_gemm64_fused_fp8
kernel_dit_gemm64_fused_i8
kernel_dit_gemm64_fused_q2k
kernel_dit_gemm64_fused_q3k
kernel_dit_gemm64_fused_q4k
kernel_dit_gemm64_fused_q5k
kernel_dit_gemm64_fused_q6k
kernel_dit_gemm64_fused_q8_0
kernel_dit_gemm64_i8
kernel_dit_gemm64_q2k
kernel_dit_gemm64_q3k
kernel_dit_gemm64_q4k
kernel_dit_gemm64_q5k
kernel_dit_gemm64_q6k
kernel_dit_gemm64_q8_0
kernel_dit_gemm_bf16
kernel_dit_gemm_f32
kernel_dit_gemm_fp8
kernel_dit_gemm_i8
kernel_dit_gqa_f32
kernel_dit_group_norm_apply_f32
kernel_dit_group_norm_stats_f32
kernel_dit_hadamard256_f32
kernel_dit_layer_norm_f32
kernel_dit_modulate_f32
kernel_dit_qk_norm_f32
kernel_dit_qk_norm_rope_f32
kernel_dit_rmsnorm_f32
kernel_dit_rope_4d
kernel_dit_rope_half
kernel_dit_sdpa_f32
kernel_dit_silu_f32
kernel_dit_softmax_rows_f32
kernel_dit_swiglu2_f32
kernel_dit_swiglu_f32
kernel_dit_tanh_f32
kernel_dit_transpose_f32
kernel_dit_upsample_nearest_f32
```

## Translation unit

One unit, `diffusion.cu`, which includes `common.cuh`, `elementwise.cu`,
`kquant.cuh`, `gemm.cu`, `attention.cu`, `rope.cu`, `conv.cu` and `norm.cu`.
The split is a file split only; a module is the key of one separately compiled
image on this backend.
