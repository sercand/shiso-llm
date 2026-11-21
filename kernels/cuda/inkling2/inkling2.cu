/* The `inkling2` module: CUDA twin of `kernels/metal/inkling2/inkling2.metal`.
 *
 * Every argument block, kernel name, launch geometry and arithmetic formula
 * matches the Metal twin entry for entry. Built for compute_80+ (including
 * Blackwell / GB10).
 */

#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint16_t u16;
typedef uint8_t u8;

#define INK2_BLOCK 256u
#define INK2_MV_SG 4u
#define INK2_MV_THREADS (32u * INK2_MV_SG)
#define INK2_MAX_ATTN_RES_SRC 16u
#define INK2_MAX_LOOP_RANK 128u
#define INK2_MAX_CONV_K 8u
#define INK2_MAX_EXPERTS 256u
#define INK2_MAX_TOPK 16u

// ---- helpers ---------------------------------------------------------------

__device__ __forceinline__ float ink2_bf16_to_f32(u16 bits) {
    return __uint_as_float(u32(bits) << 16u);
}

__device__ __forceinline__ float ink2_e4m3_to_f32(u32 b) {
    const u32 e = (b >> 3u) & 0xFu;
    const u32 m = b & 7u;
    float v;
    if (e == 0u) {
        v = float(m) * 0.001953125f;
    } else if (e == 15u && m == 7u) {
        v = 0.0f;
    } else {
        v = (1.0f + float(m) * 0.125f) * exp2f(float(e) - 7.0f);
    }
    return (b & 0x80u) ? -v : v;
}

__device__ __forceinline__ float ink2_ue8m0_to_f32(u32 b) {
    if (b == 255u) {
        return 0.0f;
    }
    return __uint_as_float(b << 23u);
}

__device__ __forceinline__ float ink2_e2m1_to_f32(u32 nib) {
    static constexpr float LUT[16] = {
        0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
       -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return LUT[nib & 0xFu];
}

__device__ __forceinline__ float ink2_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float ink2_softcap(float x, float cap) {
    if (cap <= 0.0f) {
        return x;
    }
    return cap * tanhf(x / cap);
}

__device__ __forceinline__ float ink2_warp_sum(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float ink2_warp_sum_all(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

// ---- argument blocks -------------------------------------------------------

struct ink2_args_norm {
    u32 rows;
    u32 hidden;
    float eps;
};
static_assert(sizeof(ink2_args_norm) == 12, "ink2_args_norm layout");

struct ink2_args_linear {
    u32 rows;
    u32 in_dim;
    u32 out_dim;
};
static_assert(sizeof(ink2_args_linear) == 12, "ink2_args_linear layout");

struct ink2_args_qkvr_gate_sink {
    u32 rows;
    u32 in_dim;
    u32 q_dim;
    u32 kv_dim;
    u32 r_dim;
    u32 n_head;
};
static_assert(sizeof(ink2_args_qkvr_gate_sink) == 24, "ink2_args_qkvr_gate_sink layout");

struct ink2_args_lm_head {
    u32 rows;
    u32 hidden;
    u32 vocab;
    float softcap;
};
static_assert(sizeof(ink2_args_lm_head) == 16, "ink2_args_lm_head layout");

struct ink2_args_attn_res {
    u32 rows;
    u32 hidden;
    u32 n_blocks;
    u32 n_queries;
    float eps;
    u32 has_addend;
};
static_assert(sizeof(ink2_args_attn_res) == 24, "ink2_args_attn_res layout");

struct ink2_args_loop_ctrl {
    u32 rows;
    u32 hidden;
    u32 rank;
    float eps;
};
static_assert(sizeof(ink2_args_loop_ctrl) == 16, "ink2_args_loop_ctrl layout");

struct ink2_args_sconv_static {
    u32 rows;
    u32 channels;
    u32 kernel_size;
    float scale;
};
static_assert(sizeof(ink2_args_sconv_static) == 16, "ink2_args_sconv_static layout");

struct ink2_args_sconv_dynamic {
    u32 rows;
    u32 kv_heads;
    u32 head_dim;
    u32 kernel_size;
    float scale;
};
static_assert(sizeof(ink2_args_sconv_dynamic) == 20, "ink2_args_sconv_dynamic layout");

struct ink2_args_qkvr_prep {
    u32 rows;
    u32 n_head;
    u32 n_head_kv;
    u32 head_dim;
    u32 d_rel;
    float qk_clip;
    float rel_clip;
    float eps;
    u32 is_full;
    float log_n_floor;
    float log_alpha;
    u32 pos0;
    u32 kv_cap;
    u32 is_swa;
};
static_assert(sizeof(ink2_args_qkvr_prep) == 56, "ink2_args_qkvr_prep layout");

struct ink2_args_rel_attn {
    u32 rows;
    u32 n_head;
    u32 n_head_kv;
    u32 head_dim;
    u32 rel_extent;
    u32 pos0;
    u32 kv_cap;
    u32 window_size;
    u32 is_swa;
    float scale;
};
static_assert(sizeof(ink2_args_rel_attn) == 40, "ink2_args_rel_attn layout");

struct ink2_args_route {
    u32 rows;
    u32 n_experts;
    u32 top_k;
    float route_scale;
};
static_assert(sizeof(ink2_args_route) == 16, "ink2_args_route layout");

struct ink2_args_moe_w13 {
    u32 rows;
    u32 in_dim;
    u32 inter_dim;
    u32 top_k;
    float gate_cap;
    float up_cap;
};
static_assert(sizeof(ink2_args_moe_w13) == 24, "ink2_args_moe_w13 layout");

struct ink2_args_moe_w2 {
    u32 rows;
    u32 inter_dim;
    u32 out_dim;
    u32 top_k;
};
static_assert(sizeof(ink2_args_moe_w2) == 16, "ink2_args_moe_w2 layout");

struct ink2_args_add3 {
    u32 count;
};
static_assert(sizeof(ink2_args_add3) == 4, "ink2_args_add3 layout");

// ---- 1. Embedding + RMSNorm (BF16) -----------------------------------------

extern "C" __global__ void kernel_inkling2_embed_rmsnorm_bf16(
        const ink2_args_norm args,
        const u32* tokens,
        const u16* embed,
        const u16* norm_w,
        float* out) {
    const u32 gid = blockIdx.x;
    if (gid >= args.rows) {
        return;
    }
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 H = args.hidden;
    const u64 tok = u64(tokens[gid]);
    const u16* src = embed + tok * H;
    float* dst = out + u64(gid) * H;

    float sum_sq = 0.0f;
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        const float v = ink2_bf16_to_f32(src[i]);
        sum_sq += v * v;
    }
    sum_sq = ink2_warp_sum(sum_sq);
    __shared__ float partials[8];
    if (lane == 0u) {
        partials[sg] = sum_sq;
    }
    __syncthreads();
    if (tid == 0u) {
        float tot = 0.0f;
        for (u32 s = 0u; s < 8u; ++s) {
            tot += partials[s];
        }
        partials[0] = rsqrtf(tot / float(H) + args.eps);
    }
    __syncthreads();
    const float inv = partials[0];
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = ink2_bf16_to_f32(src[i]) * inv * ink2_bf16_to_f32(norm_w[i]);
    }
}

// ---- 2. RMSNorm (BF16 weight) ----------------------------------------------

extern "C" __global__ void kernel_inkling2_rmsnorm_bf16(
        const ink2_args_norm args,
        const float* inp,
        const u16* w,
        float* out) {
    const u32 gid = blockIdx.x;
    if (gid >= args.rows) {
        return;
    }
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 H = args.hidden;
    const float* src = inp + u64(gid) * H;
    float* dst = out + u64(gid) * H;

    float sum_sq = 0.0f;
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        const float v = src[i];
        sum_sq += v * v;
    }
    sum_sq = ink2_warp_sum(sum_sq);
    __shared__ float partials[8];
    if (lane == 0u) {
        partials[sg] = sum_sq;
    }
    __syncthreads();
    if (tid == 0u) {
        float tot = 0.0f;
        for (u32 s = 0u; s < 8u; ++s) {
            tot += partials[s];
        }
        partials[0] = rsqrtf(tot / float(H) + args.eps);
    }
    __syncthreads();
    const float inv = partials[0];
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = src[i] * inv * ink2_bf16_to_f32(w[i]);
    }
}

// ---- 3. Linear BF16 (8-row batched prefill + 1-row decode fast path) -------

extern "C" __global__ void kernel_inkling2_linear_bf16(
        const ink2_args_linear args,
        const float* x,
        const u16* w,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 r0 = blockIdx.y * 8u;
    if (o >= args.out_dim || r0 >= args.rows) {
        return;
    }
    const u32 n_r = (args.rows - r0 < 8u) ? (args.rows - r0) : 8u;
    const u32 in_dim = args.in_dim;
    const u16* wr = w + u64(o) * in_dim;

    if (n_r == 1u) {
        const float* xr = x + u64(r0) * in_dim;
        float acc = 0.0f;
        for (u32 k = lane * 4u; k + 3u < in_dim; k += 128u) {
            const float4 xv = *reinterpret_cast<const float4*>(xr + k);
            const ushort4 wv = *reinterpret_cast<const ushort4*>(wr + k);
            acc += xv.x * ink2_bf16_to_f32(wv.x) +
                   xv.y * ink2_bf16_to_f32(wv.y) +
                   xv.z * ink2_bf16_to_f32(wv.z) +
                   xv.w * ink2_bf16_to_f32(wv.w);
        }
        const u32 tail_idx = (in_dim & ~3u) + lane;
        if (tail_idx < in_dim) {
            acc += xr[tail_idx] * ink2_bf16_to_f32(wr[tail_idx]);
        }
        acc = ink2_warp_sum(acc);
        if (lane == 0u) {
            out[u64(r0) * args.out_dim + o] = acc;
        }
        return;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 k = lane * 4u; k + 3u < in_dim; k += 128u) {
        const ushort4 wv = *reinterpret_cast<const ushort4*>(wr + k);
        const float w0 = ink2_bf16_to_f32(wv.x);
        const float w1 = ink2_bf16_to_f32(wv.y);
        const float w2 = ink2_bf16_to_f32(wv.z);
        const float w3 = ink2_bf16_to_f32(wv.w);
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const float4 xv = *reinterpret_cast<const float4*>(x + u64(r0 + rr) * in_dim + k);
            acc[rr] += xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
        }
    }
    const u32 tail_idx = (in_dim & ~3u) + lane;
    if (tail_idx < in_dim) {
        const float wt = ink2_bf16_to_f32(wr[tail_idx]);
        for (u32 rr = 0u; rr < n_r; ++rr) {
            acc[rr] += x[u64(r0 + rr) * in_dim + tail_idx] * wt;
        }
    }
    for (u32 rr = 0u; rr < n_r; ++rr) {
        const float s = ink2_warp_sum(acc[rr]);
        if (lane == 0u) {
            out[u64(r0 + rr) * args.out_dim + o] = s;
        }
    }
}

// ---- 3b. Fused Q+K+V+R+Gate+Sink BF16 mega-projection (8-row batched) ------

extern "C" __global__ void kernel_inkling2_qkvr_gate_sink_bf16(
        const ink2_args_qkvr_gate_sink args,
        const float* x_q,
        const float* x_k,
        const float* x_v,
        const u16* wq,
        const u16* wk,
        const u16* wv,
        const u16* wr,
        const u16* wgate,
        const u16* wsink,
        float* out_q,
        float* out_k,
        float* out_v,
        float* out_r,
        float* out_gate,
        float* out_sink) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 r0 = blockIdx.y * 8u;
    const u32 b1 = args.q_dim;
    const u32 b2 = b1 + args.kv_dim;
    const u32 b3 = b2 + args.kv_dim;
    const u32 b4 = b3 + args.r_dim;
    const u32 b5 = b4 + args.q_dim;
    const u32 b6 = b5 + args.n_head;
    if (o >= b6 || r0 >= args.rows) {
        return;
    }
    const u32 n_r = (args.rows - r0 < 8u) ? (args.rows - r0) : 8u;
    const u32 in_dim = args.in_dim;

    const float* x_base;
    const u16* wr_ptr;
    float* dst;
    u32 local_o;
    u32 out_stride;
    if (o < b1) {
        local_o = o;
        x_base = x_q;
        wr_ptr = wq + u64(local_o) * in_dim;
        dst = out_q;
        out_stride = args.q_dim;
    } else if (o < b2) {
        local_o = o - b1;
        x_base = x_k;
        wr_ptr = wk + u64(local_o) * in_dim;
        dst = out_k;
        out_stride = args.kv_dim;
    } else if (o < b3) {
        local_o = o - b2;
        x_base = x_v;
        wr_ptr = wv + u64(local_o) * in_dim;
        dst = out_v;
        out_stride = args.kv_dim;
    } else if (o < b4) {
        local_o = o - b3;
        x_base = x_q;
        wr_ptr = wr + u64(local_o) * in_dim;
        dst = out_r;
        out_stride = args.r_dim;
    } else if (o < b5) {
        local_o = o - b4;
        x_base = x_q;
        wr_ptr = wgate + u64(local_o) * in_dim;
        dst = out_gate;
        out_stride = args.q_dim;
    } else {
        local_o = o - b5;
        x_base = x_q;
        wr_ptr = wsink + u64(local_o) * in_dim;
        dst = out_sink;
        out_stride = args.n_head;
    }

    if (n_r == 1u) {
        const float* xr = x_base + u64(r0) * in_dim;
        float acc = 0.0f;
        for (u32 k = lane * 4u; k + 3u < in_dim; k += 128u) {
            const float4 xv = *reinterpret_cast<const float4*>(xr + k);
            const ushort4 wv_u = *reinterpret_cast<const ushort4*>(wr_ptr + k);
            acc += xv.x * ink2_bf16_to_f32(wv_u.x) +
                   xv.y * ink2_bf16_to_f32(wv_u.y) +
                   xv.z * ink2_bf16_to_f32(wv_u.z) +
                   xv.w * ink2_bf16_to_f32(wv_u.w);
        }
        const u32 tail_idx = (in_dim & ~3u) + lane;
        if (tail_idx < in_dim) {
            acc += xr[tail_idx] * ink2_bf16_to_f32(wr_ptr[tail_idx]);
        }
        acc = ink2_warp_sum(acc);
        if (lane == 0u) {
            dst[u64(r0) * out_stride + local_o] = acc;
        }
        return;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 k = lane * 4u; k + 3u < in_dim; k += 128u) {
        const ushort4 wv_u = *reinterpret_cast<const ushort4*>(wr_ptr + k);
        const float w0 = ink2_bf16_to_f32(wv_u.x);
        const float w1 = ink2_bf16_to_f32(wv_u.y);
        const float w2 = ink2_bf16_to_f32(wv_u.z);
        const float w3 = ink2_bf16_to_f32(wv_u.w);
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const float4 xv = *reinterpret_cast<const float4*>(x_base + u64(r0 + rr) * in_dim + k);
            acc[rr] += xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
        }
    }
    const u32 tail_idx = (in_dim & ~3u) + lane;
    if (tail_idx < in_dim) {
        const float wt = ink2_bf16_to_f32(wr_ptr[tail_idx]);
        for (u32 rr = 0u; rr < n_r; ++rr) {
            acc[rr] += x_base[u64(r0 + rr) * in_dim + tail_idx] * wt;
        }
    }
    for (u32 rr = 0u; rr < n_r; ++rr) {
        const float s = ink2_warp_sum(acc[rr]);
        if (lane == 0u) {
            dst[u64(r0 + rr) * out_stride + local_o] = s;
        }
    }
}

// ---- 4. LM Head BF16 + Softcap ---------------------------------------------

extern "C" __global__ void kernel_inkling2_lm_head_softcap(
        const ink2_args_lm_head args,
        const float* x,
        const u16* w,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 r = blockIdx.y;
    if (o >= args.vocab || r >= args.rows) {
        return;
    }
    const u32 H = args.hidden;
    const float* xr = x + u64(r) * H;
    const u16* wr = w + u64(o) * H;

    float acc = 0.0f;
    for (u32 k = lane * 4u; k + 3u < H; k += 128u) {
        const float4 xv = *reinterpret_cast<const float4*>(xr + k);
        const ushort4 wv = *reinterpret_cast<const ushort4*>(wr + k);
        acc += xv.x * ink2_bf16_to_f32(wv.x) +
               xv.y * ink2_bf16_to_f32(wv.y) +
               xv.z * ink2_bf16_to_f32(wv.z) +
               xv.w * ink2_bf16_to_f32(wv.w);
    }
    acc = ink2_warp_sum(acc);
    if (lane == 0u) {
        out[u64(r) * args.vocab + o] = ink2_softcap(acc, args.softcap);
    }
}

// ---- 5. Block Attention Residual -------------------------------------------

extern "C" __global__ void kernel_inkling2_attn_res(
        const ink2_args_attn_res args,
        const float* blocks,
        const float* partial,
        const float* addend,
        const float* queries,
        float* out) {
    const u32 gid = blockIdx.x;
    if (gid >= args.rows) {
        return;
    }
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 H = args.hidden;
    const u32 n_blocks = args.n_blocks;
    const u32 n_src = n_blocks + 1u;
    const u32 n_q = args.n_queries;
    const u64 row_off = u64(gid) * H;
    const u64 block_stride = u64(args.rows) * H;
    const float scale = rsqrtf(float(H));

    __shared__ float partials[8];
    __shared__ float inv_rms[INK2_MAX_ATTN_RES_SRC];
    __shared__ float weights[4][INK2_MAX_ATTN_RES_SRC];

    for (u32 s = 0u; s < n_src; ++s) {
        float sq = 0.0f;
        for (u32 i = tid; i < H; i += INK2_BLOCK) {
            float v = (s < n_blocks)
                          ? blocks[u64(s) * block_stride + row_off + i]
                          : (partial[row_off + i] + (args.has_addend ? addend[row_off + i] : 0.0f));
            sq += v * v;
        }
        sq = ink2_warp_sum(sq);
        if (lane == 0u) {
            partials[sg] = sq;
        }
        __syncthreads();
        if (tid == 0u) {
            float tot = 0.0f;
            for (u32 k = 0u; k < 8u; ++k) {
                tot += partials[k];
            }
            inv_rms[s] = rsqrtf(tot / float(H) + args.eps);
        }
        __syncthreads();

        const float inv = inv_rms[s];
        for (u32 q = 0u; q < n_q; ++q) {
            const float* qr = queries + u64(q) * H;
            float dot = 0.0f;
            for (u32 i = tid; i < H; i += INK2_BLOCK) {
                float v = (s < n_blocks)
                              ? blocks[u64(s) * block_stride + row_off + i]
                              : (partial[row_off + i] + (args.has_addend ? addend[row_off + i] : 0.0f));
                dot += qr[i] * (v * inv);
            }
            dot = ink2_warp_sum(dot);
            if (lane == 0u) {
                partials[sg] = dot;
            }
            __syncthreads();
            if (tid == 0u) {
                float tot = 0.0f;
                for (u32 k = 0u; k < 8u; ++k) {
                    tot += partials[k];
                }
                weights[q][s] = tot * scale;
            }
            __syncthreads();
        }
    }

    if (tid == 0u) {
        for (u32 q = 0u; q < n_q; ++q) {
            float mx = weights[q][0];
            for (u32 s = 1u; s < n_src; ++s) {
                mx = fmaxf(mx, weights[q][s]);
            }
            float denom = 0.0f;
            for (u32 s = 0u; s < n_src; ++s) {
                weights[q][s] = expf(weights[q][s] - mx);
                denom += weights[q][s];
            }
            const float inv_d = 1.0f / denom;
            for (u32 s = 0u; s < n_src; ++s) {
                weights[q][s] *= inv_d;
            }
        }
    }
    __syncthreads();

    for (u32 q = 0u; q < n_q; ++q) {
        float* dst = out + u64(q) * block_stride + row_off;
        for (u32 i = tid; i < H; i += INK2_BLOCK) {
            float acc = 0.0f;
            for (u32 s = 0u; s < n_src; ++s) {
                float v = (s < n_blocks)
                              ? blocks[u64(s) * block_stride + row_off + i]
                              : (partial[row_off + i] + (args.has_addend ? addend[row_off + i] : 0.0f));
                acc += weights[q][s] * (v * inv_rms[s]);
            }
            dst[i] = acc;
        }
    }
}

// ---- 6. Loop Message Controller --------------------------------------------

extern "C" __global__ void kernel_inkling2_loop_control(
        const ink2_args_loop_ctrl args,
        const float* cur,
        const float* msg,
        const u16* down_w,
        const u16* gain_w,
        const u16* forget_w,
        float* out) {
    const u32 gid = blockIdx.x;
    if (gid >= args.rows) {
        return;
    }
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 H = args.hidden;
    const u32 R = args.rank;
    const float* cr = cur + u64(gid) * H;
    const float* mr = msg + u64(gid) * H;
    float* dst = out + u64(gid) * H;

    float sq_c = 0.0f;
    float sq_m = 0.0f;
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        const float vc = cr[i];
        const float vm = mr[i];
        sq_c += vc * vc;
        sq_m += vm * vm;
    }
    sq_c = ink2_warp_sum(sq_c);
    sq_m = ink2_warp_sum(sq_m);
    __shared__ float pc[8];
    __shared__ float pm[8];
    if (lane == 0u) {
        pc[sg] = sq_c;
        pm[sg] = sq_m;
    }
    __syncthreads();
    if (tid == 0u) {
        float tc = 0.0f;
        float tm = 0.0f;
        for (u32 k = 0u; k < 8u; ++k) {
            tc += pc[k];
            tm += pm[k];
        }
        pc[0] = rsqrtf(tc / float(H) + args.eps);
        pm[0] = rsqrtf(tm / float(H) + args.eps);
    }
    __syncthreads();
    const float inv_c = pc[0];
    const float inv_m = pm[0];

    __shared__ float hvec[INK2_MAX_LOOP_RANK];
    for (u32 r = sg; r < R; r += 8u) {
        const u16* dr = down_w + u64(r) * (2u * H);
        float acc = 0.0f;
        for (u32 i = lane; i < H; i += 32u) {
            acc += ink2_bf16_to_f32(dr[i]) * (cr[i] * inv_c) +
                   ink2_bf16_to_f32(dr[H + i]) * (mr[i] * inv_m);
        }
        acc = ink2_warp_sum(acc);
        if (lane == 0u) {
            hvec[r] = acc;
        }
    }
    __syncthreads();

    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        const u16* gr = gain_w + u64(i) * R;
        const u16* fr = forget_w + u64(i) * R;
        float g = 0.0f;
        float f = 0.0f;
        for (u32 r = 0u; r < R; ++r) {
            const float hv = hvec[r];
            g += ink2_bf16_to_f32(gr[r]) * hv;
            f += ink2_bf16_to_f32(fr[r]) * hv;
        }
        dst[i] = (1.0f - tanhf(f)) * cr[i] + tanhf(g) * mr[i];
    }
}

// ---- 6b. Parallel Loop Message Controller (down + apply) -------------------

extern "C" __global__ void kernel_inkling2_loop_ctrl_down(
        const ink2_args_loop_ctrl args,
        const float* cur,
        const float* msg,
        const u16* down_w,
        float* z_out) {
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 r = blockIdx.x * INK2_MV_SG + sg;
    const u32 row = blockIdx.y;
    if (row >= args.rows) {
        return;
    }
    const u32 H = args.hidden;
    const float* cr = cur + u64(row) * H;
    const float* mr = msg + u64(row) * H;

    float sq_c = 0.0f;
    float sq_m = 0.0f;
    for (u32 i = tid * 4u; i + 3u < H; i += 512u) {
        const float4 vc = *reinterpret_cast<const float4*>(cr + i);
        const float4 vm = *reinterpret_cast<const float4*>(mr + i);
        sq_c += vc.x * vc.x + vc.y * vc.y + vc.z * vc.z + vc.w * vc.w;
        sq_m += vm.x * vm.x + vm.y * vm.y + vm.z * vm.z + vm.w * vm.w;
    }
    const u32 tail_i = (H & ~3u) + tid;
    if (tail_i < H) {
        sq_c += cr[tail_i] * cr[tail_i];
        sq_m += mr[tail_i] * mr[tail_i];
    }
    sq_c = ink2_warp_sum(sq_c);
    sq_m = ink2_warp_sum(sq_m);
    __shared__ float pc[4];
    __shared__ float pm[4];
    if (lane == 0u) {
        pc[sg] = sq_c;
        pm[sg] = sq_m;
    }
    __syncthreads();
    if (tid == 0u) {
        const float tc = pc[0] + pc[1] + pc[2] + pc[3];
        const float tm = pm[0] + pm[1] + pm[2] + pm[3];
        pc[0] = rsqrtf(tc / float(H) + args.eps);
        pm[0] = rsqrtf(tm / float(H) + args.eps);
    }
    __syncthreads();
    if (r >= args.rank) {
        return;
    }
    const float inv_c = pc[0];
    const float inv_m = pm[0];

    const u16* dr = down_w + u64(r) * (2u * H);
    float acc_c = 0.0f;
    float acc_m = 0.0f;
    for (u32 i = lane * 4u; i + 3u < H; i += 128u) {
        const float4 vc = *reinterpret_cast<const float4*>(cr + i);
        const float4 vm = *reinterpret_cast<const float4*>(mr + i);
        const ushort4 wc = *reinterpret_cast<const ushort4*>(dr + i);
        const ushort4 wm = *reinterpret_cast<const ushort4*>(dr + H + i);
        acc_c += vc.x * ink2_bf16_to_f32(wc.x) + vc.y * ink2_bf16_to_f32(wc.y) +
                 vc.z * ink2_bf16_to_f32(wc.z) + vc.w * ink2_bf16_to_f32(wc.w);
        acc_m += vm.x * ink2_bf16_to_f32(wm.x) + vm.y * ink2_bf16_to_f32(wm.y) +
                 vm.z * ink2_bf16_to_f32(wm.z) + vm.w * ink2_bf16_to_f32(wm.w);
    }
    const u32 tail_k = (H & ~3u) + lane;
    if (tail_k < H) {
        acc_c += cr[tail_k] * ink2_bf16_to_f32(dr[tail_k]);
        acc_m += mr[tail_k] * ink2_bf16_to_f32(dr[H + tail_k]);
    }
    const float tot = ink2_warp_sum(acc_c * inv_c + acc_m * inv_m);
    if (lane == 0u) {
        z_out[u64(row) * args.rank + r] = tot;
    }
}

extern "C" __global__ void kernel_inkling2_loop_ctrl_apply(
        const ink2_args_loop_ctrl args,
        const float* cur,
        const float* msg,
        const float* z_in,
        const u16* gain_w,
        const u16* forget_w,
        float* out) {
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 i = blockIdx.x * INK2_MV_SG + sg;
    const u32 row = blockIdx.y;
    if (row >= args.rows) {
        return;
    }
    const u32 R = args.rank;
    __shared__ float zs[INK2_MAX_LOOP_RANK];
    if (tid < R) {
        zs[tid] = z_in[u64(row) * R + tid];
    }
    __syncthreads();
    if (i >= args.hidden) {
        return;
    }
    const u16* gr = gain_w + u64(i) * R;
    const u16* fr = forget_w + u64(i) * R;
    float g = 0.0f;
    float f = 0.0f;
    for (u32 r = lane; r < R; r += 32u) {
        const float hv = zs[r];
        g += ink2_bf16_to_f32(gr[r]) * hv;
        f += ink2_bf16_to_f32(fr[r]) * hv;
    }
    g = ink2_warp_sum(g);
    f = ink2_warp_sum(f);
    if (lane == 0u) {
        const u64 idx = u64(row) * args.hidden + i;
        out[idx] = (1.0f - tanhf(f)) * cur[idx] + tanhf(g) * msg[idx];
    }
}

// ---- 7. Static Short Causal Convolution ------------------------------------

extern "C" __global__ void kernel_inkling2_short_conv_static(
        const ink2_args_sconv_static args,
        const float* x,
        const u16* w,
        float* state,
        float* out) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= args.channels) {
        return;
    }
    const u32 C = args.channels;
    const u32 K = args.kernel_size;
    const u32 lags = K - 1u;
    float hist[INK2_MAX_CONV_K];
    for (u32 l = 0u; l < lags; ++l) {
        hist[l] = state[u64(l) * C + c];
    }
    float wt[INK2_MAX_CONV_K];
    for (u32 k = 0u; k < K; ++k) {
        wt[k] = ink2_bf16_to_f32(w[u64(c) * K + k]);
    }
    for (u32 t = 0u; t < args.rows; ++t) {
        const float cur = x[u64(t) * C + c];
        float acc = cur * wt[lags];
        for (u32 l = 0u; l < lags; ++l) {
            acc += hist[l] * wt[l];
        }
        out[u64(t) * C + c] = (acc + cur) * args.scale;
        for (u32 l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    for (u32 l = 0u; l < lags; ++l) {
        state[u64(l) * C + c] = hist[l];
    }
}

// ---- 8. Dynamic Short Causal Convolution -----------------------------------

extern "C" __global__ void kernel_inkling2_short_conv_dynamic(
        const ink2_args_sconv_dynamic args,
        const float* x,
        const float* dyn_w,
        const u16* static_w,
        float* state,
        float* out) {
    const u32 C = args.kv_heads * args.head_dim;
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    const u32 h = c / args.head_dim;
    const u32 K = args.kernel_size;
    const u32 lags = K - 1u;
    float hist[INK2_MAX_CONV_K];
    for (u32 l = 0u; l < lags; ++l) {
        hist[l] = state[u64(l) * C + c];
    }
    float sw[INK2_MAX_CONV_K];
    for (u32 k = 0u; k < K; ++k) {
        sw[k] = ink2_bf16_to_f32(static_w[u64(k) * C + c]);
    }
    for (u32 t = 0u; t < args.rows; ++t) {
        const float cur = x[u64(t) * C + c];
        const float* dw = dyn_w + (u64(t) * args.kv_heads + h) * K;
        float acc = cur * (sw[0] + dw[0]);
        for (u32 lag = 1u; lag < K; ++lag) {
            const float prev = hist[lags - lag];
            acc += prev * (sw[lag] + dw[lag]);
        }
        out[u64(t) * C + c] = (acc + cur) * args.scale;
        for (u32 l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    for (u32 l = 0u; l < lags; ++l) {
        state[u64(l) * C + c] = hist[l];
    }
}

// ---- 9. QKVR Prep & KV Cache Store -----------------------------------------

extern "C" __global__ void kernel_inkling2_qkvr_prep_store(
        const ink2_args_qkvr_prep args,
        float* q,
        const float* k,
        const float* v,
        float* r,
        const u16* q_scale,
        const u16* r_scale,
        __half* k_cache,
        __half* v_cache) {
    const u32 row = blockIdx.y;
    if (row >= args.rows) {
        return;
    }
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 pos = args.pos0 + row;
    float tau = 1.0f;
    if (args.is_full != 0u) {
        const float ratio = float(pos + 1u) / args.log_n_floor;
        if (ratio > 1.0f) {
            tau = 1.0f + args.log_alpha * logf(ratio);
        }
    }

    __shared__ float partials[4];

    if (blockIdx.x < args.n_head) {
        const u32 h = blockIdx.x;
        const u32 D = args.head_dim;
        float* qh = q + (u64(row) * args.n_head + h) * D;
        const u16* qs = q_scale + u64(h) * D;
        float sq = 0.0f;
        for (u32 d = tid; d < D; d += 128u) {
            const float val = qh[d] * ink2_bf16_to_f32(qs[d]);
            qh[d] = val;
            sq += val * val;
        }
        sq = ink2_warp_sum(sq);
        if (lane == 0u) {
            partials[sg] = sq;
        }
        __syncthreads();
        if (tid == 0u) {
            const float norm = sqrtf(partials[0] + partials[1] + partials[2] + partials[3]);
            partials[0] = fminf(args.qk_clip / (norm + 1e-6f), 1.0f) * tau;
        }
        __syncthreads();
        const float qf = partials[0];
        for (u32 d = tid; d < D; d += 128u) {
            qh[d] *= qf;
        }

        if (sg == 0u) {
            const u32 DR = args.d_rel;
            float* rh = r + (u64(row) * args.n_head + h) * DR;
            const u16* rs = r_scale + u64(h) * DR;
            float val = (lane < DR) ? (rh[lane] * ink2_bf16_to_f32(rs[lane])) : 0.0f;
            const float rnorm = sqrtf(ink2_warp_sum_all(val * val));
            const float rf = fminf(args.rel_clip / (rnorm + 1e-6f), 1.0f) * tau;
            if (lane < DR) {
                rh[lane] = val * rf;
            }
        }
    } else {
        const u32 kvh = blockIdx.x - args.n_head;
        if (kvh >= args.n_head_kv) {
            return;
        }
        const u32 D = args.head_dim;
        const u32 slot = args.is_swa != 0u ? (pos % args.kv_cap) : pos;
        const float* kh = k + (u64(row) * args.n_head_kv + kvh) * D;
        const float* vh = v + (u64(row) * args.n_head_kv + kvh) * D;
        __half* kdst = k_cache + (u64(slot) * args.n_head_kv + kvh) * D;
        __half* vdst = v_cache + (u64(slot) * args.n_head_kv + kvh) * D;

        float ksq = 0.0f;
        float vsq = 0.0f;
        for (u32 d = tid; d < D; d += 128u) {
            const float kv = kh[d];
            const float vv = vh[d];
            ksq += kv * kv;
            vsq += vv * vv;
        }
        ksq = ink2_warp_sum(ksq);
        vsq = ink2_warp_sum(vsq);
        __shared__ float kp[4];
        __shared__ float vp[4];
        if (lane == 0u) {
            kp[sg] = ksq;
            vp[sg] = vsq;
        }
        __syncthreads();
        if (tid == 0u) {
            const float knorm = sqrtf(kp[0] + kp[1] + kp[2] + kp[3]);
            kp[0] = fminf(args.qk_clip / (knorm + 1e-6f), 1.0f);
            vp[0] = rsqrtf((vp[0] + vp[1] + vp[2] + vp[3]) / float(D) + args.eps);
        }
        __syncthreads();
        const float kf = kp[0];
        const float vf = vp[0];
        for (u32 d = tid; d < D; d += 128u) {
            kdst[d] = __float2half(kh[d] * kf);
            vdst[d] = __float2half(vh[d] * vf);
        }
    }
}

// ---- 9b. Fast Relative-Bias Projection (d_rel <= 16, zero idle lanes) ------

extern "C" __global__ void kernel_inkling2_rel_bias_bf16(
        const ink2_args_linear args,
        const float* r_in,
        const u16* w,
        float* out) {
    const u32 d = blockIdx.x * 128u + threadIdx.x;
    const u32 row = blockIdx.y;
    if (d >= args.out_dim || row >= args.rows) {
        return;
    }
    const u32 D = args.in_dim;
    const float* rr = r_in + u64(row) * D;
    const u16* wr = w + u64(d) * D;
    float acc = 0.0f;
    for (u32 k = 0u; k < D; ++k) {
        acc += rr[k] * ink2_bf16_to_f32(wr[k]);
    }
    out[u64(row) * args.out_dim + d] = acc;
}

// ---- 10. Relative-Bias + Per-Query Sink + Gated Attention (ZERO barriers) --

extern "C" __global__ void kernel_inkling2_rel_attn(
        const ink2_args_rel_attn args,
        const float* q,
        const float* rel_bias,
        const float* sinks,
        const float* gate,
        const __half* k_cache,
        const __half* v_cache,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 h = blockIdx.x;
    const u32 row = blockIdx.y * 4u + sg;
    if (h >= args.n_head || row >= args.rows) {
        return;
    }
    const u32 D = args.head_dim;
    const u32 kvh = h / (args.n_head / args.n_head_kv);
    const u32 q_pos = args.pos0 + row;
    const u32 start_pos = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                              ? (q_pos + 1u - args.window_size)
                              : 0u;

    const u64 q_base = (u64(row) * args.n_head + h) * D;
    const float* rb = rel_bias + (u64(row) * args.n_head + h) * args.rel_extent;
    const float sink_logit = sinks[u64(row) * args.n_head + h];

    if (D == 128u) {
        const u32 d0 = lane * 4u;
        const float4 qv = *reinterpret_cast<const float4*>(q + q_base + d0);
        float m_max = sink_logit;
        float l_sum = 1.0f;
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

        for (u32 k_pos = start_pos; k_pos <= q_pos; ++k_pos) {
            const u32 slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;
            const u64 kv_off = (u64(slot) * args.n_head_kv + kvh) * 128u + d0;
            const float k0 = __half2float(k_cache[kv_off + 0u]);
            const float k1 = __half2float(k_cache[kv_off + 1u]);
            const float k2 = __half2float(k_cache[kv_off + 2u]);
            const float k3 = __half2float(k_cache[kv_off + 3u]);
            float dot = ink2_warp_sum(qv.x * k0 + qv.y * k1 + qv.z * k2 + qv.w * k3) * args.scale;
            const u32 dist = q_pos - k_pos;
            const float bias = (dist < args.rel_extent) ? rb[dist] : 0.0f;
            const float score = dot + bias;

            const float new_max = fmaxf(m_max, score);
            const float exp_old = expf(m_max - new_max);
            const float exp_new = expf(score - new_max);
            l_sum = l_sum * exp_old + exp_new;
            a0 = a0 * exp_old + exp_new * __half2float(v_cache[kv_off + 0u]);
            a1 = a1 * exp_old + exp_new * __half2float(v_cache[kv_off + 1u]);
            a2 = a2 * exp_old + exp_new * __half2float(v_cache[kv_off + 2u]);
            a3 = a3 * exp_old + exp_new * __half2float(v_cache[kv_off + 3u]);
            m_max = new_max;
        }

        const float inv_l = 1.0f / l_sum;
        const float4 gv = *reinterpret_cast<const float4*>(gate + q_base + d0);
        float4 res;
        res.x = (a0 * inv_l) * ink2_sigmoid(gv.x);
        res.y = (a1 * inv_l) * ink2_sigmoid(gv.y);
        res.z = (a2 * inv_l) * ink2_sigmoid(gv.z);
        res.w = (a3 * inv_l) * ink2_sigmoid(gv.w);
        *reinterpret_cast<float4*>(out + q_base + d0) = res;
        return;
    }

    const float q0 = (lane < D) ? q[q_base + lane] : 0.0f;
    const float q1 = (lane + 32u < D) ? q[q_base + lane + 32u] : 0.0f;
    const float q2 = (lane + 64u < D) ? q[q_base + lane + 64u] : 0.0f;
    const float q3 = (lane + 96u < D) ? q[q_base + lane + 96u] : 0.0f;
    float m_max = sink_logit;
    float l_sum = 1.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (u32 k_pos = start_pos; k_pos <= q_pos; ++k_pos) {
        const u32 slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;
        const u64 kv_off = (u64(slot) * args.n_head_kv + kvh) * D;
        float dot = 0.0f;
        if (lane < D) dot += q0 * __half2float(k_cache[kv_off + lane]);
        if (lane + 32u < D) dot += q1 * __half2float(k_cache[kv_off + lane + 32u]);
        if (lane + 64u < D) dot += q2 * __half2float(k_cache[kv_off + lane + 64u]);
        if (lane + 96u < D) dot += q3 * __half2float(k_cache[kv_off + lane + 96u]);
        dot = ink2_warp_sum(dot) * args.scale;
        const u32 dist = q_pos - k_pos;
        const float bias = (dist < args.rel_extent) ? rb[dist] : 0.0f;
        const float score = dot + bias;

        const float new_max = fmaxf(m_max, score);
        const float exp_old = expf(m_max - new_max);
        const float exp_new = expf(score - new_max);
        l_sum = l_sum * exp_old + exp_new;
        if (lane < D) a0 = a0 * exp_old + exp_new * __half2float(v_cache[kv_off + lane]);
        if (lane + 32u < D) a1 = a1 * exp_old + exp_new * __half2float(v_cache[kv_off + lane + 32u]);
        if (lane + 64u < D) a2 = a2 * exp_old + exp_new * __half2float(v_cache[kv_off + lane + 64u]);
        if (lane + 96u < D) a3 = a3 * exp_old + exp_new * __half2float(v_cache[kv_off + lane + 96u]);
        m_max = new_max;
    }

    const float inv_l = 1.0f / l_sum;
    if (lane < D) out[q_base + lane] = (a0 * inv_l) * ink2_sigmoid(gate[q_base + lane]);
    if (lane + 32u < D) out[q_base + lane + 32u] = (a1 * inv_l) * ink2_sigmoid(gate[q_base + lane + 32u]);
    if (lane + 64u < D) out[q_base + lane + 64u] = (a2 * inv_l) * ink2_sigmoid(gate[q_base + lane + 64u]);
    if (lane + 96u < D) out[q_base + lane + 96u] = (a3 * inv_l) * ink2_sigmoid(gate[q_base + lane + 96u]);
}

// ---- 10b. Split-K Flash-Decoding for Long-Context Full Attention -----------

struct ink2_args_rel_attn_splitk {
    u32 n_head;
    u32 n_head_kv;
    u32 head_dim;
    u32 rel_extent;
    u32 pos;
    u32 n_splits;
    float scale;
};
static_assert(sizeof(ink2_args_rel_attn_splitk) == 28, "ink2_args_rel_attn_splitk layout");

extern "C" __global__ void kernel_inkling2_rel_attn_splitk(
        const ink2_args_rel_attn_splitk args,
        const float* q,
        const float* rel_bias,
        const float* sinks,
        const __half* k_cache,
        const __half* v_cache,
        float* split_ml,
        float* split_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 h = blockIdx.x;
    const u32 s = blockIdx.y * 4u + sg;
    if (h >= args.n_head || s >= args.n_splits) {
        return;
    }
    const u32 kvh = h / (args.n_head / args.n_head_kv);
    const u32 q_pos = args.pos;
    const u32 total_kv = q_pos + 1u;
    const u32 chunk = (total_kv + args.n_splits - 1u) / args.n_splits;
    const u32 start_pos = s * chunk;
    const u32 end_pos = (start_pos + chunk < total_kv) ? (start_pos + chunk) : total_kv;

    const u32 d0 = lane * 4u;
    const u64 q_base = u64(h) * 128u;
    const float4 qv = *reinterpret_cast<const float4*>(q + q_base + d0);
    const float* rb = rel_bias + u64(h) * args.rel_extent;

    float m_max = (s == 0u) ? sinks[h] : -1e30f;
    float l_sum = (s == 0u) ? 1.0f : 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (u32 k_pos = start_pos; k_pos < end_pos; ++k_pos) {
        const u64 kv_off = (u64(k_pos) * args.n_head_kv + kvh) * 128u + d0;
        const float k0 = __half2float(k_cache[kv_off + 0u]);
        const float k1 = __half2float(k_cache[kv_off + 1u]);
        const float k2 = __half2float(k_cache[kv_off + 2u]);
        const float k3 = __half2float(k_cache[kv_off + 3u]);
        float dot = ink2_warp_sum(qv.x * k0 + qv.y * k1 + qv.z * k2 + qv.w * k3) * args.scale;
        const u32 dist = q_pos - k_pos;
        const float bias = (dist < args.rel_extent) ? rb[dist] : 0.0f;
        const float score = dot + bias;

        const float new_max = fmaxf(m_max, score);
        const float exp_old = (l_sum > 0.0f) ? expf(m_max - new_max) : 0.0f;
        const float exp_new = expf(score - new_max);
        l_sum = l_sum * exp_old + exp_new;
        a0 = a0 * exp_old + exp_new * __half2float(v_cache[kv_off + 0u]);
        a1 = a1 * exp_old + exp_new * __half2float(v_cache[kv_off + 1u]);
        a2 = a2 * exp_old + exp_new * __half2float(v_cache[kv_off + 2u]);
        a3 = a3 * exp_old + exp_new * __half2float(v_cache[kv_off + 3u]);
        m_max = new_max;
    }

    const u64 s_idx = u64(h) * args.n_splits + s;
    if (lane == 0u) {
        split_ml[s_idx * 2u] = m_max;
        split_ml[s_idx * 2u + 1u] = l_sum;
    }
    float4 out4;
    out4.x = a0; out4.y = a1; out4.z = a2; out4.w = a3;
    *reinterpret_cast<float4*>(split_out + s_idx * 128u + d0) = out4;
}

extern "C" __global__ void kernel_inkling2_rel_attn_reduce(
        const ink2_args_rel_attn_splitk args,
        const float* split_ml,
        const float* split_out,
        const float* gate,
        float* out) {
    const u32 h = blockIdx.x;
    const u32 lane = threadIdx.x & 31u;
    if (h >= args.n_head) {
        return;
    }
    const u32 d0 = lane * 4u;
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (u32 s = 0u; s < args.n_splits; ++s) {
        const u64 s_idx = u64(h) * args.n_splits + s;
        const float ms = split_ml[s_idx * 2u];
        const float ls = split_ml[s_idx * 2u + 1u];
        if (ls <= 0.0f) {
            continue;
        }
        const float4 vs = *reinterpret_cast<const float4*>(split_out + s_idx * 128u + d0);
        const float nm = fmaxf(m_max, ms);
        const float eo = (l_sum > 0.0f) ? expf(m_max - nm) : 0.0f;
        const float en = expf(ms - nm);
        l_sum = l_sum * eo + ls * en;
        a0 = a0 * eo + vs.x * en;
        a1 = a1 * eo + vs.y * en;
        a2 = a2 * eo + vs.z * en;
        a3 = a3 * eo + vs.w * en;
        m_max = nm;
    }

    const float inv_l = 1.0f / l_sum;
    const u64 base = u64(h) * 128u + d0;
    const float4 gv = *reinterpret_cast<const float4*>(gate + base);
    float4 res;
    res.x = (a0 * inv_l) * ink2_sigmoid(gv.x);
    res.y = (a1 * inv_l) * ink2_sigmoid(gv.y);
    res.z = (a2 * inv_l) * ink2_sigmoid(gv.z);
    res.w = (a3 * inv_l) * ink2_sigmoid(gv.w);
    *reinterpret_cast<float4*>(out + base) = res;
}

// ---- 11. Top-K Sigmoid Router (parallel 128-thread block) ------------------

extern "C" __global__ void kernel_inkling2_route_topk(
        const ink2_args_route args,
        const float* logits,
        const float* bias,
        u32* out_ids,
        float* out_weights) {
    const u32 row = blockIdx.x;
    if (row >= args.rows) {
        return;
    }
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 E = args.n_experts;
    const u32 K = args.top_k;
    const float* lr = logits + u64(row) * E;

    __shared__ float s_arr[INK2_MAX_EXPERTS];
    __shared__ float b_arr[INK2_MAX_EXPERTS];
    __shared__ float sg_val[4];
    __shared__ u32 sg_idx[4];
    __shared__ u32 picked_id[INK2_MAX_TOPK];
    __shared__ float picked_w[INK2_MAX_TOPK];

    if (tid < E) {
        const float s = ink2_sigmoid(lr[tid]);
        s_arr[tid] = s;
        b_arr[tid] = s + bias[tid];
    }
    __syncthreads();

    for (u32 k = 0u; k < K; ++k) {
        float wmax = (tid < E) ? b_arr[tid] : -1e38f;
        const float cur_b = wmax;
        for (int off = 16; off > 0; off >>= 1) {
            wmax = fmaxf(wmax, __shfl_xor_sync(0xFFFFFFFFu, wmax, off));
        }
        u32 win_idx = (cur_b == wmax) ? tid : 0xFFFFFFFFu;
        for (int off = 16; off > 0; off >>= 1) {
            const u32 other = __shfl_xor_sync(0xFFFFFFFFu, win_idx, off);
            win_idx = (other < win_idx) ? other : win_idx;
        }
        if (lane == 0u) {
            sg_val[sg] = wmax;
            sg_idx[sg] = win_idx;
        }
        __syncthreads();
        if (tid == 0u) {
            float bv = sg_val[0];
            u32 bi = sg_idx[0];
            for (u32 s = 1u; s < 4u; ++s) {
                if (sg_val[s] > bv || (sg_val[s] == bv && sg_idx[s] < bi)) {
                    bv = sg_val[s];
                    bi = sg_idx[s];
                }
            }
            picked_id[k] = bi;
            picked_w[k] = s_arr[bi];
            b_arr[bi] = -1e38f;
        }
        __syncthreads();
    }

    if (tid < K) {
        float w_sum = 0.0f;
        for (u32 k = 0u; k < K; ++k) {
            w_sum += picked_w[k];
        }
        const float scale = args.route_scale / (w_sum + 1e-20f);
        out_ids[u64(row) * K + tid] = picked_id[tid];
        out_weights[u64(row) * K + tid] = picked_w[tid] * scale;
    }
}

// ---- 12. NVFP4 MoE W13 + Softcapped SwiGLU (shared E4M3 LUT + vectorized) --

__device__ __forceinline__ void ink2_fill_e4m3_lut(float* lut, u32 tid) {
    lut[tid] = ink2_e4m3_to_f32(tid);
    lut[tid + 128u] = ink2_e4m3_to_f32(tid + 128u);
    __syncthreads();
}

__device__ __forceinline__ float ink2_nvfp4_row_dot_lut(
        const u8* w_row,
        const u8* s_row,
        const float* x,
        const float* e4m3_lut,
        u32 in_dim,
        u32 lane) {
    float acc = 0.0f;
    const u32 n_groups = in_dim / 16u;
    for (u32 g = lane; g < n_groups; g += 32u) {
        const float gs = e4m3_lut[u32(s_row[g])];
        const uchar4* wp4 = reinterpret_cast<const uchar4*>(w_row + g * 8u);
        const float4* xp4 = reinterpret_cast<const float4*>(x + g * 16u);
        const uchar4 b0 = wp4[0];
        const uchar4 b1 = wp4[1];
        const float4 x0 = xp4[0];
        const float4 x1 = xp4[1];
        const float4 x2 = xp4[2];
        const float4 x3 = xp4[3];
        float part =
            ink2_e2m1_to_f32(b0.x & 0xFu) * x0.x + ink2_e2m1_to_f32(b0.x >> 4u) * x0.y +
            ink2_e2m1_to_f32(b0.y & 0xFu) * x0.z + ink2_e2m1_to_f32(b0.y >> 4u) * x0.w +
            ink2_e2m1_to_f32(b0.z & 0xFu) * x1.x + ink2_e2m1_to_f32(b0.z >> 4u) * x1.y +
            ink2_e2m1_to_f32(b0.w & 0xFu) * x1.z + ink2_e2m1_to_f32(b0.w >> 4u) * x1.w +
            ink2_e2m1_to_f32(b1.x & 0xFu) * x2.x + ink2_e2m1_to_f32(b1.x >> 4u) * x2.y +
            ink2_e2m1_to_f32(b1.y & 0xFu) * x2.z + ink2_e2m1_to_f32(b1.y >> 4u) * x2.w +
            ink2_e2m1_to_f32(b1.z & 0xFu) * x3.x + ink2_e2m1_to_f32(b1.z >> 4u) * x3.y +
            ink2_e2m1_to_f32(b1.w & 0xFu) * x3.z + ink2_e2m1_to_f32(b1.w >> 4u) * x3.w;
        acc += gs * part;
    }
    return ink2_warp_sum(acc);
}

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w13_swiglu(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const float* w13_amax,
        const float* w13_denom,
        const u32* expert_ids,
        const float* expert_weights,
        float* out) {
    __shared__ float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, threadIdx.x);

    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 j = blockIdx.x * INK2_MV_SG + sg;
    const u32 col = blockIdx.y;
    const u32 total_cols = args.rows * args.top_k;
    if (j >= args.inter_dim || col >= total_cols) {
        return;
    }
    const u32 row = col / args.top_k;
    const u32 expert = expert_ids[col];
    const float rw = expert_weights[col];
    const float outer = w13_amax[expert] / w13_denom[0];

    const u32 in_dim = args.in_dim;
    const u32 two_inter = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(in_dim / 2u);
    const u64 s_row_bytes = u64(in_dim / 16u);
    const u64 e_w_base = u64(expert) * two_inter * w_row_bytes;
    const u64 e_s_base = u64(expert) * two_inter * s_row_bytes;

    const float* xr = x + u64(row) * in_dim;
    const u32 g_row = 2u * j;
    const u32 u_row = 2u * j + 1u;

    const float g_val = ink2_nvfp4_row_dot_lut(
        w13 + e_w_base + u64(g_row) * w_row_bytes,
        w13_scale + e_s_base + u64(g_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane) * outer;
    const float u_val = ink2_nvfp4_row_dot_lut(
        w13 + e_w_base + u64(u_row) * w_row_bytes,
        w13_scale + e_s_base + u64(u_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane) * outer;

    if (lane == 0u) {
        const float act = ink2_softcap(g_val, args.gate_cap) *
                          ink2_sigmoid(g_val) *
                          ink2_softcap(u_val, args.up_cap) * rw;
        out[u64(col) * args.inter_dim + j] = act;
    }
}

// ---- 13. NVFP4 MoE W2 + Accum ----------------------------------------------

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w2_accum(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const float* w2_amax,
        const float* w2_denom,
        const u32* expert_ids,
        float* out) {
    __shared__ float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, threadIdx.x);

    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 row = blockIdx.y;
    if (o >= args.out_dim || row >= args.rows) {
        return;
    }
    const u32 inter_dim = args.inter_dim;
    const u64 w_row_bytes = u64(inter_dim / 2u);
    const u64 s_row_bytes = u64(inter_dim / 16u);
    const float denom = w2_denom[0];

    float total = 0.0f;
    for (u32 slot = 0u; slot < args.top_k; ++slot) {
        const u32 col = row * args.top_k + slot;
        const u32 expert = expert_ids[col];
        const float outer = w2_amax[expert] / denom;
        const u64 e_w_base = (u64(expert) * args.out_dim + o) * w_row_bytes;
        const u64 e_s_base = (u64(expert) * args.out_dim + o) * s_row_bytes;
        const float* ar = act + u64(col) * inter_dim;
        total += ink2_nvfp4_row_dot_lut(w2 + e_w_base, w2_scale + e_s_base, ar, e4m3_lut, inter_dim, lane) * outer;
    }
    if (lane == 0u) {
        out[u64(row) * args.out_dim + o] = total;
    }
}

// ---- 14. MXFP8 MoE W13 + Softcapped SwiGLU ---------------------------------

__device__ __forceinline__ float ink2_mxfp8_row_dot_lut(
        const u8* w_row,
        const u8* s_row,
        const float* x,
        const float* e4m3_lut,
        u32 in_dim,
        u32 lane) {
    float acc = 0.0f;
    const u32 n_groups = in_dim / 32u;
    for (u32 g = lane; g < n_groups; g += 32u) {
        const float gs = ink2_ue8m0_to_f32(u32(s_row[g]));
        const uchar4* wp4 = reinterpret_cast<const uchar4*>(w_row + g * 32u);
        const float4* xp4 = reinterpret_cast<const float4*>(x + g * 32u);
        float part = 0.0f;
        for (u32 i = 0u; i < 8u; ++i) {
            const uchar4 wv = wp4[i];
            const float4 xv = xp4[i];
            part += e4m3_lut[wv.x] * xv.x +
                    e4m3_lut[wv.y] * xv.y +
                    e4m3_lut[wv.z] * xv.z +
                    e4m3_lut[wv.w] * xv.w;
        }
        acc += gs * part;
    }
    return ink2_warp_sum(acc);
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w13_swiglu(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const u32* expert_ids,
        const float* expert_weights,
        float* out) {
    __shared__ float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, threadIdx.x);

    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 j = blockIdx.x * INK2_MV_SG + sg;
    const u32 col = blockIdx.y;
    const u32 total_cols = args.rows * args.top_k;
    if (j >= args.inter_dim || col >= total_cols) {
        return;
    }
    const u32 row = col / args.top_k;
    const u32 expert = expert_ids[col];
    const float rw = expert_weights[col];

    const u32 in_dim = args.in_dim;
    const u32 two_inter = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(in_dim);
    const u64 s_row_bytes = u64(in_dim / 32u);
    const u64 e_w_base = u64(expert) * two_inter * w_row_bytes;
    const u64 e_s_base = u64(expert) * two_inter * s_row_bytes;

    const float* xr = x + u64(row) * in_dim;
    const u32 g_row = 2u * j;
    const u32 u_row = 2u * j + 1u;

    const float g_val = ink2_mxfp8_row_dot_lut(
        w13 + e_w_base + u64(g_row) * w_row_bytes,
        w13_scale + e_s_base + u64(g_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane);
    const float u_val = ink2_mxfp8_row_dot_lut(
        w13 + e_w_base + u64(u_row) * w_row_bytes,
        w13_scale + e_s_base + u64(u_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane);

    if (lane == 0u) {
        const float act = ink2_softcap(g_val, args.gate_cap) *
                          ink2_sigmoid(g_val) *
                          ink2_softcap(u_val, args.up_cap) * rw;
        out[u64(col) * args.inter_dim + j] = act;
    }
}

// ---- 15. MXFP8 MoE W2 + Accum ----------------------------------------------

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w2_accum(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const u32* expert_ids,
        float* out) {
    __shared__ float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, threadIdx.x);

    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 row = blockIdx.y;
    if (o >= args.out_dim || row >= args.rows) {
        return;
    }
    const u32 inter_dim = args.inter_dim;
    const u64 w_row_bytes = u64(inter_dim);
    const u64 s_row_bytes = u64(inter_dim / 32u);

    float total = 0.0f;
    for (u32 slot = 0u; slot < args.top_k; ++slot) {
        const u32 col = row * args.top_k + slot;
        const u32 expert = expert_ids[col];
        const u64 e_w_base = (u64(expert) * args.out_dim + o) * w_row_bytes;
        const u64 e_s_base = (u64(expert) * args.out_dim + o) * s_row_bytes;
        const float* ar = act + u64(col) * inter_dim;
        total += ink2_mxfp8_row_dot_lut(w2 + e_w_base, w2_scale + e_s_base, ar, e4m3_lut, inter_dim, lane);
    }
    if (lane == 0u) {
        out[u64(row) * args.out_dim + o] = total;
    }
}

// ---- 15b. Token-Grouped 8-Row Batched MoE Prefill Kernels ------------------

struct ink2_args_moe_bucket {
    u32 rows;
    u32 top_k;
    u32 n_experts;
};
static_assert(sizeof(ink2_args_moe_bucket) == 12, "ink2_args_moe_bucket layout");

extern "C" __global__ void kernel_inkling2_moe_bucket(
        const ink2_args_moe_bucket args,
        const u32* expert_ids,
        u32* expert_counts,
        u32* expert_cols) {
    const u32 e = blockIdx.x;
    const u32 tid = threadIdx.x;
    if (e >= args.n_experts) {
        return;
    }
    const u32 total = args.rows * args.top_k;
    u32* dst = expert_cols + u64(e) * args.rows;
    __shared__ u32 cnt;
    if (tid == 0u) {
        cnt = 0u;
    }
    __syncthreads();
    for (u32 col = tid; col < total; col += 256u) {
        if (expert_ids[col] == e) {
            const u32 slot = atomicAdd(&cnt, 1u);
            if (slot < args.rows) {
                dst[slot] = col;
            }
        }
    }
    __syncthreads();
    if (tid == 0u) {
        expert_counts[e] = cnt;
    }
}

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w13_grouped(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const float* w13_amax,
        const float* w13_denom,
        const float* expert_weights,
        const u32* expert_counts,
        const u32* expert_cols,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 j = blockIdx.x * INK2_MV_SG + sg;
    const u32 expert = blockIdx.y;
    if (j >= args.inter_dim) return;
    const u32 cnt = expert_counts[expert];
    if (cnt == 0u) return;
    const float outer = w13_amax[expert] / w13_denom[0];
    const u32 in_dim = args.in_dim;
    const u32 two_inter = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(in_dim / 2u);
    const u64 s_row_bytes = u64(in_dim / 16u);
    const u64 e_w_base = u64(expert) * two_inter * w_row_bytes;
    const u64 e_s_base = u64(expert) * two_inter * s_row_bytes;
    const u8* gw_row = w13 + e_w_base + u64(2u * j) * w_row_bytes;
    const u8* gs_row = w13_scale + e_s_base + u64(2u * j) * s_row_bytes;
    const u8* uw_row = w13 + e_w_base + u64(2u * j + 1u) * w_row_bytes;
    const u8* us_row = w13_scale + e_s_base + u64(2u * j + 1u) * s_row_bytes;
    const u32* cols = expert_cols + u64(expert) * args.rows;
    const u32 n_groups = in_dim / 16u;

    for (u32 base = 0u; base < cnt; base += 8u) {
        const u32 n_r = (cnt - base < 8u) ? (cnt - base) : 8u;
        u32 c_ids[8], r_ids[8];
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const u32 col = cols[base + rr];
            c_ids[rr] = col;
            r_ids[rr] = col / args.top_k;
        }
        float g_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float u_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (u32 g = lane; g < n_groups; g += 32u) {
            const float gsc = ink2_e4m3_to_f32(u32(gs_row[g]));
            const float usc = ink2_e4m3_to_f32(u32(us_row[g]));
            const uchar4* gwp = reinterpret_cast<const uchar4*>(gw_row + g * 8u);
            const uchar4* uwp = reinterpret_cast<const uchar4*>(uw_row + g * 8u);
            const uchar4 gb0 = gwp[0], gb1 = gwp[1];
            const uchar4 ub0 = uwp[0], ub1 = uwp[1];
            const float gw[16] = {
                INK2_E2M1_LUT[gb0.x & 0xFu] * gsc, INK2_E2M1_LUT[gb0.x >> 4u] * gsc,
                INK2_E2M1_LUT[gb0.y & 0xFu] * gsc, INK2_E2M1_LUT[gb0.y >> 4u] * gsc,
                INK2_E2M1_LUT[gb0.z & 0xFu] * gsc, INK2_E2M1_LUT[gb0.z >> 4u] * gsc,
                INK2_E2M1_LUT[gb0.w & 0xFu] * gsc, INK2_E2M1_LUT[gb0.w >> 4u] * gsc,
                INK2_E2M1_LUT[gb1.x & 0xFu] * gsc, INK2_E2M1_LUT[gb1.x >> 4u] * gsc,
                INK2_E2M1_LUT[gb1.y & 0xFu] * gsc, INK2_E2M1_LUT[gb1.y >> 4u] * gsc,
                INK2_E2M1_LUT[gb1.z & 0xFu] * gsc, INK2_E2M1_LUT[gb1.z >> 4u] * gsc,
                INK2_E2M1_LUT[gb1.w & 0xFu] * gsc, INK2_E2M1_LUT[gb1.w >> 4u] * gsc
            };
            const float uw[16] = {
                INK2_E2M1_LUT[ub0.x & 0xFu] * usc, INK2_E2M1_LUT[ub0.x >> 4u] * usc,
                INK2_E2M1_LUT[ub0.y & 0xFu] * usc, INK2_E2M1_LUT[ub0.y >> 4u] * usc,
                INK2_E2M1_LUT[ub0.z & 0xFu] * usc, INK2_E2M1_LUT[ub0.z >> 4u] * usc,
                INK2_E2M1_LUT[ub0.w & 0xFu] * usc, INK2_E2M1_LUT[ub0.w >> 4u] * usc,
                INK2_E2M1_LUT[ub1.x & 0xFu] * usc, INK2_E2M1_LUT[ub1.x >> 4u] * usc,
                INK2_E2M1_LUT[ub1.y & 0xFu] * usc, INK2_E2M1_LUT[ub1.y >> 4u] * usc,
                INK2_E2M1_LUT[ub1.z & 0xFu] * usc, INK2_E2M1_LUT[ub1.z >> 4u] * usc,
                INK2_E2M1_LUT[ub1.w & 0xFu] * usc, INK2_E2M1_LUT[ub1.w >> 4u] * usc
            };
            for (u32 rr = 0u; rr < n_r; ++rr) {
                const float4* xp = reinterpret_cast<const float4*>(x + u64(r_ids[rr]) * in_dim + g * 16u);
                const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                g_acc[rr] += gw[0]*x0.x + gw[1]*x0.y + gw[2]*x0.z + gw[3]*x0.w +
                             gw[4]*x1.x + gw[5]*x1.y + gw[6]*x1.z + gw[7]*x1.w +
                             gw[8]*x2.x + gw[9]*x2.y + gw[10]*x2.z + gw[11]*x2.w +
                             gw[12]*x3.x + gw[13]*x3.y + gw[14]*x3.z + gw[15]*x3.w;
                u_acc[rr] += uw[0]*x0.x + uw[1]*x0.y + uw[2]*x0.z + uw[3]*x0.w +
                             uw[4]*x1.x + uw[5]*x1.y + uw[6]*x1.z + uw[7]*x1.w +
                             uw[8]*x2.x + uw[9]*x2.y + uw[10]*x2.z + uw[11]*x2.w +
                             uw[12]*x3.x + uw[13]*x3.y + uw[14]*x3.z + uw[15]*x3.w;
            }
        }
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const float gv = ink2_warp_sum(g_acc[rr]) * outer;
            const float uv = ink2_warp_sum(u_acc[rr]) * outer;
            if (lane == 0u) {
                const u32 col = c_ids[rr];
                const float rw = expert_weights[col];
                out[u64(col) * args.inter_dim + j] =
                    ink2_softcap(gv, args.gate_cap) * ink2_sigmoid(gv) * ink2_softcap(uv, args.up_cap) * rw;
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w2_grouped(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const float* w2_amax,
        const float* w2_denom,
        const u32* expert_counts,
        const u32* expert_cols,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 expert = blockIdx.y;
    if (o >= args.out_dim) return;
    const u32 cnt = expert_counts[expert];
    if (cnt == 0u) return;
    const float outer = w2_amax[expert] / w2_denom[0];
    const u32 inter_dim = args.inter_dim;
    const u64 w_row_bytes = u64(inter_dim / 2u);
    const u64 s_row_bytes = u64(inter_dim / 16u);
    const u8* w_row = w2 + (u64(expert) * args.out_dim + o) * w_row_bytes;
    const u8* s_row = w2_scale + (u64(expert) * args.out_dim + o) * s_row_bytes;
    const u32* cols = expert_cols + u64(expert) * args.rows;
    const u32 n_groups = inter_dim / 16u;

    for (u32 base = 0u; base < cnt; base += 8u) {
        const u32 n_r = (cnt - base < 8u) ? (cnt - base) : 8u;
        u32 c_ids[8];
        for (u32 rr = 0u; rr < n_r; ++rr) c_ids[rr] = cols[base + rr];
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (u32 g = lane; g < n_groups; g += 32u) {
            const float sc = ink2_e4m3_to_f32(u32(s_row[g]));
            const uchar4* wp = reinterpret_cast<const uchar4*>(w_row + g * 8u);
            const uchar4 b0 = wp[0], b1 = wp[1];
            const float wv[16] = {
                INK2_E2M1_LUT[b0.x & 0xFu] * sc, INK2_E2M1_LUT[b0.x >> 4u] * sc,
                INK2_E2M1_LUT[b0.y & 0xFu] * sc, INK2_E2M1_LUT[b0.y >> 4u] * sc,
                INK2_E2M1_LUT[b0.z & 0xFu] * sc, INK2_E2M1_LUT[b0.z >> 4u] * sc,
                INK2_E2M1_LUT[b0.w & 0xFu] * sc, INK2_E2M1_LUT[b0.w >> 4u] * sc,
                INK2_E2M1_LUT[b1.x & 0xFu] * sc, INK2_E2M1_LUT[b1.x >> 4u] * sc,
                INK2_E2M1_LUT[b1.y & 0xFu] * sc, INK2_E2M1_LUT[b1.y >> 4u] * sc,
                INK2_E2M1_LUT[b1.z & 0xFu] * sc, INK2_E2M1_LUT[b1.z >> 4u] * sc,
                INK2_E2M1_LUT[b1.w & 0xFu] * sc, INK2_E2M1_LUT[b1.w >> 4u] * sc
            };
            for (u32 rr = 0u; rr < n_r; ++rr) {
                const float4* ap = reinterpret_cast<const float4*>(act + u64(c_ids[rr]) * inter_dim + g * 16u);
                const float4 a0 = ap[0], a1 = ap[1], a2 = ap[2], a3 = ap[3];
                acc[rr] += wv[0]*a0.x + wv[1]*a0.y + wv[2]*a0.z + wv[3]*a0.w +
                           wv[4]*a1.x + wv[5]*a1.y + wv[6]*a1.z + wv[7]*a1.w +
                           wv[8]*a2.x + wv[9]*a2.y + wv[10]*a2.z + wv[11]*a2.w +
                           wv[12]*a3.x + wv[13]*a3.y + wv[14]*a3.z + wv[15]*a3.w;
            }
        }
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const float v = ink2_warp_sum(acc[rr]) * outer;
            if (lane == 0u) {
                slots_out[u64(c_ids[rr]) * args.out_dim + o] = v;
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w13_grouped(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const float* expert_weights,
        const u32* expert_counts,
        const u32* expert_cols,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 j = blockIdx.x * INK2_MV_SG + sg;
    const u32 expert = blockIdx.y;
    if (j >= args.inter_dim) return;
    const u32 cnt = expert_counts[expert];
    if (cnt == 0u) return;
    const u32 in_dim = args.in_dim;
    const u32 two_inter = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(in_dim);
    const u64 s_row_bytes = u64(in_dim / 32u);
    const u64 e_w_base = u64(expert) * two_inter * w_row_bytes;
    const u64 e_s_base = u64(expert) * two_inter * s_row_bytes;
    const u8* gw_row = w13 + e_w_base + u64(2u * j) * w_row_bytes;
    const u8* gs_row = w13_scale + e_s_base + u64(2u * j) * s_row_bytes;
    const u8* uw_row = w13 + e_w_base + u64(2u * j + 1u) * w_row_bytes;
    const u8* us_row = w13_scale + e_s_base + u64(2u * j + 1u) * s_row_bytes;
    const u32* cols = expert_cols + u64(expert) * args.rows;
    const u32 n_groups = in_dim / 32u;

    for (u32 base = 0u; base < cnt; base += 8u) {
        const u32 n_r = (cnt - base < 8u) ? (cnt - base) : 8u;
        u32 c_ids[8], r_ids[8];
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const u32 col = cols[base + rr];
            c_ids[rr] = col;
            r_ids[rr] = col / args.top_k;
        }
        float g_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float u_acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (u32 g = lane; g < n_groups; g += 32u) {
            const float gsc = ink2_ue8m0_to_f32(u32(gs_row[g]));
            const float usc = ink2_ue8m0_to_f32(u32(us_row[g]));
            const uchar4* gwp = reinterpret_cast<const uchar4*>(gw_row + g * 32u);
            const uchar4* uwp = reinterpret_cast<const uchar4*>(uw_row + g * 32u);
            for (u32 i = 0u; i < 8u; ++i) {
                const uchar4 gv4 = gwp[i], uv4 = uwp[i];
                const float g0 = ink2_e4m3_to_f32(gv4.x) * gsc;
                const float g1 = ink2_e4m3_to_f32(gv4.y) * gsc;
                const float g2 = ink2_e4m3_to_f32(gv4.z) * gsc;
                const float g3 = ink2_e4m3_to_f32(gv4.w) * gsc;
                const float u0 = ink2_e4m3_to_f32(uv4.x) * usc;
                const float u1 = ink2_e4m3_to_f32(uv4.y) * usc;
                const float u2 = ink2_e4m3_to_f32(uv4.z) * usc;
                const float u3 = ink2_e4m3_to_f32(uv4.w) * usc;
                for (u32 rr = 0u; rr < n_r; ++rr) {
                    const float4 xv = *reinterpret_cast<const float4*>(x + u64(r_ids[rr]) * in_dim + g * 32u + i * 4u);
                    g_acc[rr] += g0 * xv.x + g1 * xv.y + g2 * xv.z + g3 * xv.w;
                    u_acc[rr] += u0 * xv.x + u1 * xv.y + u2 * xv.z + u3 * xv.w;
                }
            }
        }
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const float gv = ink2_warp_sum(g_acc[rr]);
            const float uv = ink2_warp_sum(u_acc[rr]);
            if (lane == 0u) {
                const u32 col = c_ids[rr];
                const float rw = expert_weights[col];
                out[u64(col) * args.inter_dim + j] =
                    ink2_softcap(gv, args.gate_cap) * ink2_sigmoid(gv) * ink2_softcap(uv, args.up_cap) * rw;
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w2_grouped(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const u32* expert_counts,
        const u32* expert_cols,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 expert = blockIdx.y;
    if (o >= args.out_dim) return;
    const u32 cnt = expert_counts[expert];
    if (cnt == 0u) return;
    const u32 inter_dim = args.inter_dim;
    const u64 w_row_bytes = u64(inter_dim);
    const u64 s_row_bytes = u64(inter_dim / 32u);
    const u8* w_row = w2 + (u64(expert) * args.out_dim + o) * w_row_bytes;
    const u8* s_row = w2_scale + (u64(expert) * args.out_dim + o) * s_row_bytes;
    const u32* cols = expert_cols + u64(expert) * args.rows;
    const u32 n_groups = inter_dim / 32u;

    for (u32 base = 0u; base < cnt; base += 8u) {
        const u32 n_r = (cnt - base < 8u) ? (cnt - base) : 8u;
        u32 c_ids[8];
        for (u32 rr = 0u; rr < n_r; ++rr) c_ids[rr] = cols[base + rr];
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (u32 g = lane; g < n_groups; g += 32u) {
            const float sc = ink2_ue8m0_to_f32(u32(s_row[g]));
            const uchar4* wp = reinterpret_cast<const uchar4*>(w_row + g * 32u);
            for (u32 i = 0u; i < 8u; ++i) {
                const uchar4 wv = wp[i];
                const float w0 = ink2_e4m3_to_f32(wv.x) * sc;
                const float w1 = ink2_e4m3_to_f32(wv.y) * sc;
                const float w2v = ink2_e4m3_to_f32(wv.z) * sc;
                const float w3 = ink2_e4m3_to_f32(wv.w) * sc;
                for (u32 rr = 0u; rr < n_r; ++rr) {
                    const float4 av = *reinterpret_cast<const float4*>(act + u64(c_ids[rr]) * inter_dim + g * 32u + i * 4u);
                    acc[rr] += w0 * av.x + w1 * av.y + w2v * av.z + w3 * av.w;
                }
            }
        }
        for (u32 rr = 0u; rr < n_r; ++rr) {
            const float v = ink2_warp_sum(acc[rr]);
            if (lane == 0u) {
                slots_out[u64(c_ids[rr]) * args.out_dim + o] = v;
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_reduce_topk(
        const ink2_args_moe_w2 args,
        const float* slots_in,
        float* out) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 total = args.rows * args.out_dim;
    if (tid >= total) return;
    const u32 row = tid / args.out_dim;
    const u32 o = tid % args.out_dim;
    float acc = 0.0f;
    for (u32 s = 0u; s < args.top_k; ++s) {
        acc += slots_in[u64(row * args.top_k + s) * args.out_dim + o];
    }
    out[tid] = acc;
}

// ---- 16. Residual Adds -----------------------------------------------------

extern "C" __global__ void kernel_inkling2_add2_f32(
        const ink2_args_add3 args,
        const float* a,
        const float* b,
        float* out) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < args.count) {
        out[tid] = a[tid] + b[tid];
    }
}

extern "C" __global__ void kernel_inkling2_add3_f32(
        const ink2_args_add3 args,
        const float* a,
        const float* b,
        const float* c,
        float* out) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < args.count) {
        out[tid] = a[tid] + b[tid] + c[tid];
    }
}

// ---- 17. Vision Encoder Kernels --------------------------------------------

struct ink2_args_vit_norm {
    u32 rows;
    u32 hidden;
    float eps;
};
static_assert(sizeof(ink2_args_vit_norm) == 12, "ink2_args_vit_norm layout");

struct ink2_args_vit_rope {
    u32 rows;
    u32 heads;
    u32 head_dim;
    float theta;
};
static_assert(sizeof(ink2_args_vit_rope) == 16, "ink2_args_vit_rope layout");

struct ink2_args_vit_attn {
    u32 rows;
    u32 heads;
    u32 head_dim;
    float scale;
};
static_assert(sizeof(ink2_args_vit_attn) == 16, "ink2_args_vit_attn layout");

struct ink2_args_vit_ls {
    u32 rows;
    u32 hidden;
};
static_assert(sizeof(ink2_args_vit_ls) == 8, "ink2_args_vit_ls layout");

extern "C" __global__ void kernel_inkling2_vit_layernorm_bf16(
        const ink2_args_vit_norm args,
        const float* inp,
        const u16* w,
        const u16* b,
        float* out) {
    const u32 gid = blockIdx.x;
    if (gid >= args.rows) return;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 H = args.hidden;
    const float* src = inp + u64(gid) * H;
    float* dst = out + u64(gid) * H;

    float sum = 0.0f;
    for (u32 i = tid; i < H; i += INK2_BLOCK) sum += src[i];
    sum = ink2_warp_sum(sum);
    __shared__ float p[8];
    if (lane == 0u) p[sg] = sum;
    __syncthreads();
    if (tid == 0u) {
        float tot = 0.0f;
        for (u32 s = 0u; s < 8u; ++s) tot += p[s];
        p[0] = tot / float(H);
    }
    __syncthreads();
    const float mean = p[0];

    float sq = 0.0f;
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        const float d = src[i] - mean;
        sq += d * d;
    }
    sq = ink2_warp_sum(sq);
    if (lane == 0u) p[sg] = sq;
    __syncthreads();
    if (tid == 0u) {
        float tot = 0.0f;
        for (u32 s = 0u; s < 8u; ++s) tot += p[s];
        p[0] = rsqrtf(tot / float(H) + args.eps);
    }
    __syncthreads();
    const float inv = p[0];
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = (src[i] - mean) * inv * ink2_bf16_to_f32(w[i]) + ink2_bf16_to_f32(b[i]);
    }
}

extern "C" __global__ void kernel_inkling2_vit_linear_bias_bf16(
        const ink2_args_linear args,
        const float* x,
        const u16* w,
        const u16* b,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 r = blockIdx.y;
    if (o >= args.out_dim || r >= args.rows) return;
    const u32 in_dim = args.in_dim;
    const float* xr = x + u64(r) * in_dim;
    const u16* wr = w + u64(o) * in_dim;
    float acc = 0.0f;
    for (u32 k = lane * 4u; k + 3u < in_dim; k += 128u) {
        const float4 xv = *reinterpret_cast<const float4*>(xr + k);
        const ushort4 wv = *reinterpret_cast<const ushort4*>(wr + k);
        acc += xv.x * ink2_bf16_to_f32(wv.x) +
               xv.y * ink2_bf16_to_f32(wv.y) +
               xv.z * ink2_bf16_to_f32(wv.z) +
               xv.w * ink2_bf16_to_f32(wv.w);
    }
    acc = ink2_warp_sum(acc);
    if (lane == 0u) {
        out[u64(r) * args.out_dim + o] = acc + ink2_bf16_to_f32(b[o]);
    }
}

extern "C" __global__ void kernel_inkling2_vit_axial_rope(
        const ink2_args_vit_rope args,
        const u32* pos_ids,
        float* q,
        float* k) {
    const u32 h = blockIdx.x;
    const u32 row = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 half_d = args.head_dim / 2u;
    if (h >= args.heads || row >= args.rows || tid >= half_d) return;
    const u32 py = pos_ids[row * 2u];
    const u32 px = pos_ids[row * 2u + 1u];
    const u32 quarter = half_d / 2u;
    const bool is_w = tid < quarter;
    const u32 f_idx = is_w ? tid : (tid - quarter);
    const float coord = float(is_w ? px : py);
    const float inv_freq = 1.0f / powf(args.theta, float(f_idx * 4u) / float(args.head_dim));
    const float angle = coord * inv_freq;
    const float c = cosf(angle);
    const float s = sinf(angle);
    const u64 base = (u64(row) * args.heads + h) * args.head_dim + tid * 2u;
    const float q0 = q[base];
    const float q1 = q[base + 1u];
    q[base] = q0 * c - q1 * s;
    q[base + 1u] = q0 * s + q1 * c;
    const float k0 = k[base];
    const float k1 = k[base + 1u];
    k[base] = k0 * c - k1 * s;
    k[base + 1u] = k0 * s + k1 * c;
}

extern "C" __global__ void kernel_inkling2_vit_attn(
        const ink2_args_vit_attn args,
        const float* q,
        const float* k,
        const float* v,
        float* out) {
    const u32 h = blockIdx.x;
    const u32 row = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 D = args.head_dim;
    if (h >= args.heads || row >= args.rows || tid >= D) return;
    const float qv = q[(u64(row) * args.heads + h) * D + tid];
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;
    __shared__ float p[4];
    const u32 n_sg = (D + 31u) / 32u;

    for (u32 kp = 0u; kp < args.rows; ++kp) {
        const u64 off = (u64(kp) * args.heads + h) * D;
        float dot = ink2_warp_sum(qv * k[off + tid]);
        if (lane == 0u) p[sg] = dot;
        __syncthreads();
        if (tid == 0u) {
            float tot = 0.0f;
            for (u32 s = 0u; s < n_sg; ++s) tot += p[s];
            p[0] = tot * args.scale;
        }
        __syncthreads();
        const float sc = p[0];
        const float nm = fmaxf(m_max, sc);
        const float eo = expf(m_max - nm);
        const float en = expf(sc - nm);
        l_sum = l_sum * eo + en;
        acc = acc * eo + en * v[off + tid];
        m_max = nm;
    }
    out[(u64(row) * args.heads + h) * D + tid] = acc / l_sum;
}

extern "C" __global__ void kernel_inkling2_vit_gelu_erf(
        const ink2_args_add3 args,
        float* x) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < args.count) {
        const float v = x[tid];
        x[tid] = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
    }
}

extern "C" __global__ void kernel_inkling2_vit_layerscale_add(
        const ink2_args_vit_ls args,
        const float* residual,
        const float* delta,
        const u16* gamma,
        float* out) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 total = args.rows * args.hidden;
    if (tid < total) {
        const u32 col = tid % args.hidden;
        out[tid] = residual[tid] + ink2_bf16_to_f32(gamma[col]) * delta[tid];
    }
}

// ---- 18. MTP / DSpark Speculator Kernels -----------------------------------

struct ink2_args_mtp_mix {
    u32 rows;
    u32 stride_rows;
    u32 hidden;
    u32 n_taps;
    float eps;
};
static_assert(sizeof(ink2_args_mtp_mix) == 20, "ink2_args_mtp_mix layout");

struct ink2_args_mtp_qk {
    u32 rows;
    u32 n_head;
    u32 head_dim;
    u32 pos0;
    float rope_theta;
    float eps;
    u32 kv_cap;
    u32 is_swa;
    u32 write_cache;
};
static_assert(sizeof(ink2_args_mtp_qk) == 36, "ink2_args_mtp_qk layout");

struct ink2_args_mtp_attn {
    u32 block_size;
    u32 n_head;
    u32 n_head_kv;
    u32 head_dim;
    u32 ctx_len;
    u32 kv_cap;
    u32 window_size;
    u32 is_swa;
    float scale;
};
static_assert(sizeof(ink2_args_mtp_attn) == 36, "ink2_args_mtp_attn layout");

struct ink2_args_mtp_rnn {
    u32 rows;
    u32 rank;
};
static_assert(sizeof(ink2_args_mtp_rnn) == 8, "ink2_args_mtp_rnn layout");

extern "C" __global__ void kernel_inkling2_mtp_mix_rmsnorm(
        const ink2_args_mtp_mix args,
        const float* taps,
        const float* alpha,
        const u16* norm_w,
        float* out) {
    const u32 gid = blockIdx.x;
    if (gid >= args.rows) return;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 H = args.hidden;
    const u64 tap_stride = u64(args.stride_rows) * H;
    const u64 row_off = u64(gid) * H;
    float* dst = out + row_off;

    float sq = 0.0f;
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        float mixed = 0.0f;
        for (u32 t = 0u; t < args.n_taps; ++t) {
            mixed += alpha[t] * taps[u64(t) * tap_stride + row_off + i];
        }
        dst[i] = mixed;
        sq += mixed * mixed;
    }
    sq = ink2_warp_sum(sq);
    __shared__ float p[8];
    if (lane == 0u) p[sg] = sq;
    __syncthreads();
    if (tid == 0u) {
        float tot = 0.0f;
        for (u32 s = 0u; s < 8u; ++s) tot += p[s];
        p[0] = rsqrtf(tot / float(H) + args.eps);
    }
    __syncthreads();
    const float inv = p[0];
    for (u32 i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = dst[i] * inv * ink2_bf16_to_f32(norm_w[i]);
    }
}

extern "C" __global__ void kernel_inkling2_mtp_qk_norm_rope(
        const ink2_args_mtp_qk args,
        float* x,
        const u16* norm_w,
        __half* kv_cache) {
    const u32 h = blockIdx.x;
    const u32 row = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 D = args.head_dim;
    if (h >= args.n_head || row >= args.rows || tid >= D) return;
    float* vec = x + (u64(row) * args.n_head + h) * D;
    const float v0 = vec[tid];
    float sq = ink2_warp_sum(v0 * v0);
    __shared__ float p[4];
    if (lane == 0u) p[sg] = sq;
    __syncthreads();
    if (tid == 0u) {
        const u32 n_sg = (D + 31u) / 32u;
        float tot = 0.0f;
        for (u32 s = 0u; s < n_sg; ++s) tot += p[s];
        p[0] = rsqrtf(tot / float(D) + args.eps);
    }
    __syncthreads();
    __shared__ float normed[128];
    normed[tid] = v0 * p[0] * ink2_bf16_to_f32(norm_w[tid]);
    __syncthreads();

    const u32 half_d = D / 2u;
    const u32 pos = args.pos0 + row;
    float rotated;
    if (tid < half_d) {
        const float freq = 1.0f / powf(args.rope_theta, float(tid * 2u) / float(D));
        const float ang = float(pos) * freq;
        rotated = normed[tid] * cosf(ang) - normed[tid + half_d] * sinf(ang);
    } else {
        const u32 pair = tid - half_d;
        const float freq = 1.0f / powf(args.rope_theta, float(pair * 2u) / float(D));
        const float ang = float(pos) * freq;
        rotated = normed[pair] * sinf(ang) + normed[tid] * cosf(ang);
    }
    vec[tid] = rotated;
    if (args.write_cache != 0u) {
        const u32 slot = args.is_swa != 0u ? (pos % args.kv_cap) : pos;
        kv_cache[(u64(slot) * args.n_head + h) * D + tid] = __float2half(rotated);
    }
}

extern "C" __global__ void kernel_inkling2_mtp_block_attn(
        const ink2_args_mtp_attn args,
        const float* q,
        const float* k_blk,
        const float* v_blk,
        const __half* k_ctx,
        const __half* v_ctx,
        float* out) {
    const u32 h = blockIdx.x;
    const u32 row = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 D = args.head_dim;
    if (h >= args.n_head || row >= args.block_size || tid >= D) return;
    const u32 kvh = h / (args.n_head / args.n_head_kv);
    const float qv = q[(u64(row) * args.n_head + h) * D + tid];
    const u32 ctx_start = (args.is_swa != 0u && args.ctx_len > args.window_size)
                              ? (args.ctx_len - args.window_size)
                              : 0u;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;
    __shared__ float p[4];

    for (u32 kp = ctx_start; kp < args.ctx_len; ++kp) {
        const u32 slot = args.is_swa != 0u ? (kp % args.kv_cap) : kp;
        const u64 off = (u64(slot) * args.n_head_kv + kvh) * D;
        float dot = ink2_warp_sum(qv * __half2float(k_ctx[off + tid]));
        if (lane == 0u) p[sg] = dot;
        __syncthreads();
        if (tid == 0u) p[0] = (p[0] + p[1] + p[2] + p[3]) * args.scale;
        __syncthreads();
        const float sc = p[0];
        const float nm = fmaxf(m_max, sc);
        const float eo = expf(m_max - nm);
        const float en = expf(sc - nm);
        l_sum = l_sum * eo + en;
        acc = acc * eo + en * __half2float(v_ctx[off + tid]);
        m_max = nm;
    }

    for (u32 bp = 0u; bp < args.block_size; ++bp) {
        const u64 off = (u64(bp) * args.n_head_kv + kvh) * D;
        float dot = ink2_warp_sum(qv * k_blk[off + tid]);
        if (lane == 0u) p[sg] = dot;
        __syncthreads();
        if (tid == 0u) p[0] = (p[0] + p[1] + p[2] + p[3]) * args.scale;
        __syncthreads();
        const float sc = p[0];
        const float nm = fmaxf(m_max, sc);
        const float eo = expf(m_max - nm);
        const float en = expf(sc - nm);
        l_sum = l_sum * eo + en;
        acc = acc * eo + en * v_blk[off + tid];
        m_max = nm;
    }

    out[(u64(row) * args.n_head + h) * D + tid] = acc / l_sum;
}

extern "C" __global__ void kernel_inkling2_mtp_swiglu_mul(
        const ink2_args_add3 args,
        const float* gate,
        const float* up,
        float* out) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < args.count) {
        const float g = gate[tid];
        out[tid] = (g * ink2_sigmoid(g)) * up[tid];
    }
}

extern "C" __global__ void kernel_inkling2_mtp_rnn_step(
        const ink2_args_mtp_rnn args,
        const float* proj,
        float* state,
        float* latent) {
    const u32 c = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 r = blockIdx.y * blockDim.y + threadIdx.y;
    if (r >= args.rows || c >= args.rank) return;
    const u32 R = args.rank;
    const float* pr = proj + u64(r) * (3u * R);
    const float g = ink2_sigmoid(pr[c]);
    const float cand = tanhf(pr[R + c]);
    const float out_v = tanhf(pr[2u * R + c]);
    const u64 idx = u64(r) * R + c;
    state[idx] = g * state[idx] + (1.0f - g) * cand;
    latent[idx] = out_v;
}
