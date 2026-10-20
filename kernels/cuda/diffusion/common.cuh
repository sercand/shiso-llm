/* Shared argument blocks and helpers for the diffusion kernel pack.
 *
 * Included first by `diffusion.cu`, so no kernel redeclares a block or helper.
 *
 * LAYOUT CONTRACT. Every `dit_args_*` struct below is a by-value kernel
 * parameter whose bytes are the host's `#[repr(C)]` `Pod` struct, marshalled
 * with `bytemuck::bytes_of` in `crates/shiso-models/src/diffusion/ops.rs` (and
 * `ops/vae.rs`). All fields are 4-byte (u32 / f32), so there is no padding and
 * the size equals the Rust size; the `static_assert`s at the bottom repeat the
 * Rust `assert!(size_of::<...>() == N)` constants. A field added on one side
 * only is a wrong number under load, not a compile error on the other.
 *
 * LAUNCH CONTRACT. `grid` is a block count, exactly the host's `Extent3`
 * (`threadgroup_position_in_grid` on Metal == `blockIdx` here), and `block` is
 * threads per block. The host passes `smem_bytes: 0`, so every kernel keeps its
 * shared memory static. Kernel parameters are positional: the Metal
 * `[[buffer(i)]]` index is the position in the host's `KernelArg` list, and the
 * same list order is the C parameter order below.
 *
 * Buffers may alias (the stream is updated in place by the gated epilogues,
 * the rotations run in place), so no pointer is `__restrict__`.
 */
#pragma once

#include <stdint.h>
#include <cuda_fp16.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef uint8_t u8;
typedef int8_t i8;

/* Square tile of the scalar GEMM kernels: 16x16 threads, two 1 KiB tiles. */
#define DIT_GEMM_TILE 16u

/* The 64x64 output tile GEMM and the implicit-GEMM convolutions: 128 threads,
 * k staged 16 at a time. (The Metal twin uses simdgroup matrices here; this
 * twin uses plain FMAs, each thread owning 32 outputs, so it is correct on any
 * architecture and can be moved to tensor cores later without touching a
 * caller.) */
#define DIT_G64_TILE 64u
#define DIT_G64_K 16u
#define DIT_G64_THREADS 128u

/* Fused attention: 64 query rows per block, 256 threads (four per row), keys
 * streamed 32 at a time, heads up to 128 wide in multiples of 8. */
#define DIT_FA_QROWS 64u
#define DIT_FA_THREADS 256u
#define DIT_FA_KEYS 32u
#define DIT_FA_MAX_D 128u

/* Threads per elementwise block, as the host launches it (`BLOCK` in ops.rs).
 * Every block reduction below is sized for exactly this (two warps). */
#define DIT_BLOCK 64u

/* Partial slots per thread in the per-row attention kernels: one per 64
 * head-dim values, so head_dim is bounded at 512. */
#define DIT_ATTN_SLOTS 8u

#define DIT_INF __int_as_float(0x7f800000)

/* One brain-float, widened: the 16 bits shifted into the high half of an f32.
 * Exact for every finite value. */
static __device__ __forceinline__ float dit_bf16(const u16* p) {
    return __uint_as_float(((u32)(*p)) << 16);
}

/* One E4M3 byte as a float: 1 sign, 4 exponent (bias 7), 3 mantissa bits, no
 * infinities, the all-ones pattern NaN. */
static __device__ __forceinline__ float dit_e4m3_bits(u32 b) {
    const u32 e = (b >> 3u) & 0xFu;
    const u32 m = b & 7u;
    float v = (e == 0u) ? (float)m * 0.001953125f /* m * 2^-9 */
                        : __uint_as_float(((e + 120u) << 23) | (m << 20));
    v = (e == 15u && m == 7u) ? __uint_as_float(0x7fc00000) : v;
    return (b & 0x80u) ? -v : v;
}

/* A little-endian f16 at an arbitrary even byte address, widened. Quant blocks
 * are 34..210 bytes, so a block start is only 2-aligned at best: read bytes. */
static __device__ __forceinline__ float dit_f16(const u8* p) {
    return __half2float(__ushort_as_half((unsigned short)(p[0] | (p[1] << 8))));
}

/* SiLU in the host twin's own form. */
static __device__ __forceinline__ float dit_silu(float x) {
    return x / (1.0f + expf(-x));
}

/* Sum over one warp, left-folded in decreasing-offset order so lane 0 holds the
 * total. Fixed association: deterministic run to run. */
static __device__ __forceinline__ float dit_warp_sum(float v) {
    for (u32 off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

static __device__ __forceinline__ float dit_warp_max(float v) {
    for (u32 off = 16u; off > 0u; off >>= 1u) {
        v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, off));
    }
    return v;
}

/* Block-wide sum over the host's 64-thread block. `scratch` is three floats of
 * `__shared__`. Every thread of the block must call it (it barriers). */
static __device__ __forceinline__ float dit_block_sum(float v, float* scratch, u32 tid) {
    const u32 lane = tid & 31u;
    const u32 warp = tid >> 5u;
    v = dit_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    const u32 n_warp = DIT_BLOCK / 32u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = dit_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[n_warp] = v;
    }
    __syncthreads();
    return scratch[n_warp];
}

static __device__ __forceinline__ float dit_block_max(float v, float* scratch, u32 tid) {
    const u32 lane = tid & 31u;
    const u32 warp = tid >> 5u;
    v = dit_warp_max(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    const u32 n_warp = DIT_BLOCK / 32u;
    v = (tid < n_warp) ? scratch[tid] : -DIT_INF;
    if (warp == 0u) {
        v = dit_warp_max(v);
    }
    if (tid == 0u) {
        scratch[n_warp] = v;
    }
    __syncthreads();
    return scratch[n_warp];
}

/* --- argument blocks (field order = the host's Pods) ---------------------- */

/* `out[m,n] = in[m,k] . weight[n,k] * scale + bias[n]`. `has_bias` is the
 * presence flag: the host binds a 4-byte zero stand-in when there is no bias
 * and the kernel must not read it then. */
struct dit_args_gemm { u32 m, n, k, has_bias; };

/* `out[m,n] += gate[n] * (x . weight * scale + bias)`, x = [a (cols_a) |
 * silu(g) * u (inner)], g at column gu_offset of a gu_stride-wide row of gu. */
struct dit_args_gemm_fused { u32 m, n, cols_a, inner, gu_stride, gu_offset, has_bias; };

struct dit_args_hadamard { u32 rows, cols_a, inner, gu_stride, gu_offset; };
struct dit_args_unary { u32 count; };
struct dit_args_swiglu { u32 rows, inner, stride; };
struct dit_args_modulate { u32 count, period; };
struct dit_args_add_gated { u32 count, period; };
struct dit_args_axpy { u32 count; float dt; };
/* Also the transpose's block (rows, dim), eps unused. */
struct dit_args_norm { u32 rows, dim; float eps; };
struct dit_args_softmax { u32 rows, cols; float scale; };
struct dit_args_qk_norm { u32 tokens, heads, head_dim, row_stride; float eps; };
struct dit_args_qk_norm_rope { u32 tokens, heads, head_dim, q_stride, k_stride; float eps; };
struct dit_args_concat { u32 rows, cols_a, cols_b; };
struct dit_args_sdpa {
    u32 txt_tokens, img_tokens, cache_tokens, heads, head_dim;
    u32 txt_stride, img_stride, cache_stride, q_offset, k_offset, v_offset;
};
/* Keys at or past valid_keys are masked for every row (`tokens` masks nothing).
 * Added for the padded text prefill; the Rust `GqaArgs` carries it as the 5th
 * field. */
struct dit_args_gqa { u32 tokens, q_heads, kv_heads, head_dim, valid_keys; };
struct dit_args_rope_4d {
    u32 tokens, heads, head_dim, freqs_per_axis, row_stride, q_offset, k_offset;
};
/* Rows from pinned_from on turn by position pinned_pos instead of their own
 * index. Added for the padded text prefill; Rust `RopeHalfArgs` fields 4 and 5. */
struct dit_args_rope_half { u32 tokens, heads, head_dim, pinned_from, pinned_pos; };
struct dit_args_conv2d {
    u32 batch, in_c, in_h, in_w, out_c, out_h, out_w, k_h, k_w, stride, pad, has_bias;
};
struct dit_args_group_norm {
    u32 batch, num_groups, channels_per_group, spatial;
    float eps;
    u32 chunks;
};
struct dit_args_upsample { u32 batch, channels, h, w; };
struct dit_args_embed { u32 tokens, hidden; };
struct dit_args_channel_rms { u32 channels, spatial; float scale, eps; };
struct dit_args_conv_vae {
    u32 in_c, in_h, in_w, out_c, out_h, out_w, k, pad, has_bias, accumulate, silu, reserved;
};
struct dit_args_dup_up { u32 out_c, h, w, factor, offset, repeats; };

static_assert(sizeof(dit_args_gemm) == 16, "GemmArgs");
static_assert(sizeof(dit_args_unary) == 4, "UnaryArgs");
static_assert(sizeof(dit_args_gemm_fused) == 28, "GemmFusedArgs");
static_assert(sizeof(dit_args_hadamard) == 20, "HadamardArgs");
static_assert(sizeof(dit_args_swiglu) == 12, "SwigluArgs");
static_assert(sizeof(dit_args_modulate) == 8, "ModulateArgs");
static_assert(sizeof(dit_args_add_gated) == 8, "AddGatedArgs");
static_assert(sizeof(dit_args_axpy) == 8, "AxpyArgs");
static_assert(sizeof(dit_args_norm) == 12, "NormArgs");
static_assert(sizeof(dit_args_softmax) == 12, "SoftmaxArgs");
static_assert(sizeof(dit_args_qk_norm) == 20, "QkNormArgs");
static_assert(sizeof(dit_args_qk_norm_rope) == 24, "QkNormRopeArgs");
static_assert(sizeof(dit_args_concat) == 12, "ConcatArgs");
static_assert(sizeof(dit_args_sdpa) == 44, "SdpaArgs");
static_assert(sizeof(dit_args_gqa) == 20, "GqaArgs");
static_assert(sizeof(dit_args_rope_4d) == 28, "Rope4dArgs");
static_assert(sizeof(dit_args_rope_half) == 20, "RopeHalfArgs");
static_assert(sizeof(dit_args_conv2d) == 48, "Conv2dArgs");
static_assert(sizeof(dit_args_group_norm) == 24, "GroupNormArgs");
static_assert(sizeof(dit_args_upsample) == 16, "UpsampleArgs");
static_assert(sizeof(dit_args_embed) == 8, "EmbedArgs");
static_assert(sizeof(dit_args_channel_rms) == 16, "ChannelRmsArgs");
static_assert(sizeof(dit_args_conv_vae) == 48, "ConvVaeArgs");
static_assert(sizeof(dit_args_dup_up) == 24, "DupUpArgs");
