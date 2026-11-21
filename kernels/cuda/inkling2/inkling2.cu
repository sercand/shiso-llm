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
    u32 defer_state;
};
static_assert(sizeof(ink2_args_sconv_static) == 20, "ink2_args_sconv_static layout");

struct ink2_args_sconv_dynamic {
    u32 rows;
    u32 kv_heads;
    u32 head_dim;
    u32 kernel_size;
    float scale;
    u32 defer_state;
};
static_assert(sizeof(ink2_args_sconv_dynamic) == 24, "ink2_args_sconv_dynamic layout");

struct ink2_args_sconv_commit {
    u32 total;
    u32 kv_dim;
    u32 hidden;
    u32 lags;
    u32 hist_rows;
    u32 keep;
};
static_assert(sizeof(ink2_args_sconv_commit) == 24, "ink2_args_sconv_commit layout");

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
// Grid: (ceil(vocab / 4), ceil(rows / 8), 1), block 128. A warp owns one
// vocabulary row against up to eight hidden rows, so a pass's rows share one
// read of the head. `hidden % 4 == 0`.

__device__ __forceinline__ float ink2_bf16_dot4_acc(float acc, float4 x, ushort4 w) {
    const float part = fmaf(x.x, ink2_bf16_to_f32(w.x),
                       fmaf(x.y, ink2_bf16_to_f32(w.y),
                       fmaf(x.z, ink2_bf16_to_f32(w.z), x.w * ink2_bf16_to_f32(w.w))));
    return acc + part;
}

extern "C" __global__ void kernel_inkling2_lm_head_softcap(
        const ink2_args_lm_head args,
        const float* x,
        const u16* w,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 o = blockIdx.x * INK2_MV_SG + sg;
    const u32 r0 = blockIdx.y * 8u;
    if (o >= args.vocab || r0 >= args.rows) {
        return;
    }
    const u32 n_r = (args.rows - r0) < 8u ? (args.rows - r0) : 8u;
    const u32 H = args.hidden;
    const u16* wr = w + u64(o) * H;

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 k = lane * 4u; k + 3u < H; k += 128u) {
        const ushort4 wv = *reinterpret_cast<const ushort4*>(wr + k);
        for (u32 rr = 0u; rr < n_r; ++rr) {
            acc[rr] = ink2_bf16_dot4_acc(
                acc[rr], *reinterpret_cast<const float4*>(x + u64(r0 + rr) * H + k), wv);
        }
    }
    for (u32 rr = 0u; rr < n_r; ++rr) {
        const float sum = ink2_warp_sum(acc[rr]);
        if (lane == 0u) {
            out[u64(r0 + rr) * args.vocab + o] = ink2_softcap(sum, args.softcap);
        }
    }
}

// ---- 5. Block Attention Residual -------------------------------------------
// See the Metal twin. `attn_res_stats` reads every source once, a block per
// (source, slice of the width), and writes its sum of squares and its dot with
// every query; `attn_res_mix` adds a source's slices in slice order, takes the
// softmax and mixes.
// stats: grid (n_src * INK2_ATTN_RES_SLICES, rows, 1), block 256
//        -> stats[row][source][slice][1 + INK2_MAX_ATTN_RES_Q]
// mix:   grid (ceil(hidden / 256), rows, 1), block 256

#define INK2_MAX_ATTN_RES_Q 4u
#define INK2_ATTN_RES_SLICES 4u
#define INK2_ATTN_RES_STATS (1u + INK2_MAX_ATTN_RES_Q)

__device__ __forceinline__ float ink2_attn_res_src(
        const float* blocks, const float* partial, const float* addend,
        u32 has_addend, u32 s, u32 n_blocks, u64 block_stride, u64 idx) {
    if (s < n_blocks) {
        return blocks[u64(s) * block_stride + idx];
    }
    return partial[idx] + (has_addend != 0u ? addend[idx] : 0.0f);
}

extern "C" __global__ void kernel_inkling2_attn_res_stats(
        const ink2_args_attn_res args,
        const float* blocks,
        const float* partial,
        const float* addend,
        const float* queries,
        float* stats) {
    const u32 tid = threadIdx.x;
    const u32 lane = tid & 31u;
    const u32 sg = tid >> 5u;
    const u32 H = args.hidden;
    const u32 n_blocks = args.n_blocks < INK2_MAX_ATTN_RES_SRC - 1u ? args.n_blocks
                                                                   : INK2_MAX_ATTN_RES_SRC - 1u;
    const u32 n_src = n_blocks + 1u;
    const u32 n_q = args.n_queries < INK2_MAX_ATTN_RES_Q ? args.n_queries : INK2_MAX_ATTN_RES_Q;
    const u32 s = blockIdx.x / INK2_ATTN_RES_SLICES;
    const u32 slice = blockIdx.x % INK2_ATTN_RES_SLICES;
    const u32 row = blockIdx.y;
    if (s >= n_src || row >= args.rows) {
        return;
    }
    const u32 span = (H + INK2_ATTN_RES_SLICES - 1u) / INK2_ATTN_RES_SLICES;
    const u32 lo = slice * span < H ? slice * span : H;
    const u32 hi = lo + span < H ? lo + span : H;
    const u64 row_off = u64(row) * H;
    const u64 block_stride = u64(args.rows) * H;

    float sums[INK2_ATTN_RES_STATS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 i = lo + tid; i < hi; i += INK2_BLOCK) {
        const float v = ink2_attn_res_src(blocks, partial, addend, args.has_addend,
                                          s, n_blocks, block_stride, row_off + i);
        sums[0] += v * v;
        for (u32 q = 0u; q < n_q; ++q) {
            sums[1u + q] += queries[u64(q) * H + i] * v;
        }
    }
    __shared__ float parts[8][INK2_ATTN_RES_STATS];
    for (u32 c = 0u; c < INK2_ATTN_RES_STATS; ++c) {
        const float total = ink2_warp_sum(sums[c]);
        if (lane == 0u) {
            parts[sg][c] = total;
        }
    }
    __syncthreads();
    if (tid < INK2_ATTN_RES_STATS) {
        float tot = 0.0f;
        for (u32 k = 0u; k < 8u; ++k) {
            tot += parts[k][tid];
        }
        stats[((u64(row) * n_src + s) * INK2_ATTN_RES_SLICES + slice) * INK2_ATTN_RES_STATS + tid] = tot;
    }
}

extern "C" __global__ void kernel_inkling2_attn_res_mix(
        const ink2_args_attn_res args,
        const float* blocks,
        const float* partial,
        const float* addend,
        const float* stats,
        float* out) {
    const u32 tid = threadIdx.x;
    const u32 H = args.hidden;
    const u32 n_blocks = args.n_blocks < INK2_MAX_ATTN_RES_SRC - 1u ? args.n_blocks
                                                                   : INK2_MAX_ATTN_RES_SRC - 1u;
    const u32 n_src = n_blocks + 1u;
    const u32 n_q = args.n_queries < INK2_MAX_ATTN_RES_Q ? args.n_queries : INK2_MAX_ATTN_RES_Q;
    const u32 row = blockIdx.y;
    if (row >= args.rows) {
        return;
    }
    const u64 row_off = u64(row) * H;
    const u64 block_stride = u64(args.rows) * H;

    // weights[q][s], the source's 1 / rms folded in.
    __shared__ float weights[INK2_MAX_ATTN_RES_Q][INK2_MAX_ATTN_RES_SRC];
    if (tid < n_q) {
        const u32 q = tid;
        const float scale = rsqrtf(float(H));
        float raw[INK2_MAX_ATTN_RES_SRC];
        float inv[INK2_MAX_ATTN_RES_SRC];
        float mx = -1e30f;
        for (u32 s = 0u; s < n_src; ++s) {
            const float* st = stats + (u64(row) * n_src + s) * INK2_ATTN_RES_SLICES * INK2_ATTN_RES_STATS;
            float sq = 0.0f;
            float d = 0.0f;
            for (u32 sl = 0u; sl < INK2_ATTN_RES_SLICES; ++sl) {
                sq += st[sl * INK2_ATTN_RES_STATS];
                d += st[sl * INK2_ATTN_RES_STATS + 1u + q];
            }
            inv[s] = rsqrtf(sq / float(H) + args.eps);
            raw[s] = d * inv[s] * scale;
            mx = fmaxf(mx, raw[s]);
        }
        float denom = 0.0f;
        for (u32 s = 0u; s < n_src; ++s) {
            raw[s] = expf(raw[s] - mx);
            denom += raw[s];
        }
        const float inv_d = 1.0f / denom;
        for (u32 s = 0u; s < n_src; ++s) {
            weights[q][s] = raw[s] * inv_d * inv[s];
        }
    }
    __syncthreads();

    const u32 i = blockIdx.x * INK2_BLOCK + tid;
    if (i >= H) {
        return;
    }
    float acc[INK2_MAX_ATTN_RES_Q] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (u32 s = 0u; s < n_src; ++s) {
        const float v = ink2_attn_res_src(blocks, partial, addend, args.has_addend,
                                          s, n_blocks, block_stride, row_off + i);
        for (u32 q = 0u; q < n_q; ++q) {
            acc[q] += weights[q][s] * v;
        }
    }
    for (u32 q = 0u; q < n_q; ++q) {
        out[u64(q) * block_stride + row_off + i] = acc[q];
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
        if (args.defer_state != 0u) {
            state[u64(lags + t) * C + c] = cur;
        }
        for (u32 l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    if (args.defer_state != 0u) {
        return;
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
        if (args.defer_state != 0u) {
            state[u64(lags + t) * C + c] = cur;
        }
        for (u32 l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    if (args.defer_state != 0u) {
        return;
    }
    for (u32 l = 0u; l < lags; ++l) {
        state[u64(l) * C + c] = hist[l];
    }
}

// ---- 8b. Short-convolution state commit after a deferring pass -------------
// See the Metal twin: one buffer holds every layer's four states, per layer
// `[k | v | attn | mlp]`, each `[lags + hist_rows, C]`. A deferring conv launch
// wrote the pass's inputs at rows `lags + t`; this moves the `lags` rows that
// start at `keep` to the front, through registers so the ranges may overlap.
// Grid: (ceil(total / 256), 1, 1), block 256.

extern "C" __global__ void kernel_inkling2_sconv_commit(
        const ink2_args_sconv_commit args,
        float* state) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= args.total) {
        return;
    }
    const u32 per = 2u * (args.kv_dim + args.hidden);
    const u32 rows = args.lags + args.hist_rows;
    const u32 layer = tid / per;
    const u32 rem = tid % per;
    u32 C;
    u32 c;
    u32 before;
    if (rem < args.kv_dim) {
        C = args.kv_dim;
        c = rem;
        before = 0u;
    } else if (rem < 2u * args.kv_dim) {
        C = args.kv_dim;
        c = rem - args.kv_dim;
        before = args.kv_dim;
    } else if (rem < 2u * args.kv_dim + args.hidden) {
        C = args.hidden;
        c = rem - 2u * args.kv_dim;
        before = 2u * args.kv_dim;
    } else {
        C = args.hidden;
        c = rem - 2u * args.kv_dim - args.hidden;
        before = 2u * args.kv_dim + args.hidden;
    }
    float* seg = state + (u64(layer) * per + before) * rows;
    float hist[INK2_MAX_CONV_K];
    for (u32 l = 0u; l < args.lags; ++l) {
        hist[l] = seg[u64(args.keep + l) * C + c];
    }
    for (u32 l = 0u; l < args.lags; ++l) {
        seg[u64(l) * C + c] = hist[l];
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

// ---- 10b. Split-K attention for a short pass --------------------------------
// See the Metal twin: `rel_attn`'s contract for `head_dim == 128`, each row's
// keys `[window edge, position]` cut into runs, a warp each, and the runs
// merged afterwards in run order. A row's run count is a function of its key
// count alone, so a position is summed the same way in a decode step and in a
// verify pass.
// Stage 1: grid (n_head, ceil(max_splits / 4), rows), block 128.
// Stage 2: grid (n_head, rows, 1), block 32.

#define INK2_ATTN_MAX_SPLITS 64u

__device__ __forceinline__ u32 ink2_attn_splits(u32 keys) {
    const u32 want = ((keys + 127u) / 128u + 3u) & ~3u;
    return want < 4u ? 4u : (want > INK2_ATTN_MAX_SPLITS ? INK2_ATTN_MAX_SPLITS : want);
}

// The key walk shared by the split and the tiled entries (a lane owns four
// dims): eight keys' scores, then one maximum, then the eight weights.
// `a` is the lane's four partial outputs.
__device__ __forceinline__ void ink2_rel_attn_keys(
        const ink2_args_rel_attn& args, u32 first, u32 end, u32 q_pos, u32 kvh, u32 d0,
        float4 qv, const float* rb, const __half* k_cache, const __half* v_cache,
        float& m_max, float& l_sum, float* a) {
    for (u32 k0 = first; k0 < end; k0 += 8u) {
        const u32 n_k = (end - k0) < 8u ? (end - k0) : 8u;
        float sc[8];
        float b_max = m_max;
        for (u32 i = 0u; i < n_k; ++i) {
            const u32 k_pos = k0 + i;
            const u32 slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;
            const u64 off = (u64(slot) * args.n_head_kv + kvh) * 128u + d0;
            const float dot = ink2_warp_sum_all(
                qv.x * __half2float(k_cache[off]) + qv.y * __half2float(k_cache[off + 1u]) +
                qv.z * __half2float(k_cache[off + 2u]) + qv.w * __half2float(k_cache[off + 3u])) * args.scale;
            const u32 dist = q_pos - k_pos;
            sc[i] = dot + ((dist < args.rel_extent) ? rb[dist] : 0.0f);
            b_max = fmaxf(b_max, sc[i]);
        }
        const float exp_old = (l_sum > 0.0f) ? expf(m_max - b_max) : 0.0f;
        l_sum *= exp_old;
        for (u32 c = 0u; c < 4u; ++c) {
            a[c] *= exp_old;
        }
        for (u32 i = 0u; i < n_k; ++i) {
            const u32 k_pos = k0 + i;
            const u32 slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;
            const u64 off = (u64(slot) * args.n_head_kv + kvh) * 128u + d0;
            const float e = expf(sc[i] - b_max);
            l_sum += e;
            for (u32 c = 0u; c < 4u; ++c) {
                a[c] += e * __half2float(v_cache[off + c]);
            }
        }
        m_max = b_max;
    }
}

extern "C" __global__ void kernel_inkling2_rel_attn_splitk(
        const ink2_args_rel_attn args,
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
    const u32 row = blockIdx.z;
    if (h >= args.n_head || row >= args.rows) {
        return;
    }
    const u32 kvh = h / (args.n_head / args.n_head_kv);
    const u32 q_pos = args.pos0 + row;
    const u32 start_pos = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                              ? (q_pos + 1u - args.window_size)
                              : 0u;
    const u32 total_kv = q_pos + 1u - start_pos;
    const u32 n_splits = ink2_attn_splits(total_kv);
    if (s >= n_splits) {
        return;
    }
    const u32 chunk = (total_kv + n_splits - 1u) / n_splits;
    const u32 first = start_pos + (s * chunk < total_kv ? s * chunk : total_kv);
    const u32 end_pos = (first + chunk < q_pos + 1u) ? (first + chunk) : (q_pos + 1u);

    const u32 d0 = lane * 4u;
    const u64 head_row = u64(row) * args.n_head + h;
    const float4 qv = *reinterpret_cast<const float4*>(q + head_row * 128u + d0);
    const float* rb = rel_bias + head_row * args.rel_extent;

    float m_max = (s == 0u) ? sinks[head_row] : -1e30f;
    float l_sum = (s == 0u) ? 1.0f : 0.0f;
    float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    ink2_rel_attn_keys(args, first, end_pos, q_pos, kvh, d0, qv, rb, k_cache, v_cache, m_max, l_sum, a);

    const u64 s_idx = head_row * INK2_ATTN_MAX_SPLITS + s;
    if (lane == 0u) {
        split_ml[s_idx * 2u] = m_max;
        split_ml[s_idx * 2u + 1u] = l_sum;
    }
    float4 out4;
    out4.x = a[0]; out4.y = a[1]; out4.z = a[2]; out4.w = a[3];
    *reinterpret_cast<float4*>(split_out + s_idx * 128u + d0) = out4;
}

extern "C" __global__ void kernel_inkling2_rel_attn_reduce(
        const ink2_args_rel_attn args,
        const float* split_ml,
        const float* split_out,
        const float* gate,
        float* out) {
    const u32 h = blockIdx.x;
    const u32 row = blockIdx.y;
    const u32 lane = threadIdx.x & 31u;
    if (h >= args.n_head || row >= args.rows) {
        return;
    }
    const u32 q_pos = args.pos0 + row;
    const u32 start_pos = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                              ? (q_pos + 1u - args.window_size)
                              : 0u;
    const u32 n_splits = ink2_attn_splits(q_pos + 1u - start_pos);
    const u64 head_row = u64(row) * args.n_head + h;
    const u32 d0 = lane * 4u;
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (u32 s = 0u; s < n_splits; ++s) {
        const u64 s_idx = head_row * INK2_ATTN_MAX_SPLITS + s;
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
    const u64 base = head_row * 128u + d0;
    const float4 gv = *reinterpret_cast<const float4*>(gate + base);
    float4 res;
    res.x = (a0 * inv_l) * ink2_sigmoid(gv.x);
    res.y = (a1 * inv_l) * ink2_sigmoid(gv.y);
    res.z = (a2 * inv_l) * ink2_sigmoid(gv.z);
    res.w = (a3 * inv_l) * ink2_sigmoid(gv.w);
    *reinterpret_cast<float4*>(out + base) = res;
}

// ---- 10c. Attention for a long pass -----------------------------------------
// The Metal twin multiplies 64 query rows of one head a threadgroup on
// simdgroup matrices. This twin keeps that contract - `rel_attn`'s bindings,
// `head_dim == 128` - and walks it with the per-row key walk: a warp owns
// eight of the block's rows and takes them one after another. It has not been
// tuned on a device.
// Grid: (ceil(rows / 64), n_head, 1), block 256.

extern "C" __global__ void kernel_inkling2_rel_attn_tile(
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
    const u32 h = blockIdx.y;
    if (h >= args.n_head) {
        return;
    }
    const u32 kvh = h / (args.n_head / args.n_head_kv);
    const u32 d0 = lane * 4u;
    for (u32 r = 0u; r < 8u; ++r) {
        const u32 row = blockIdx.x * 64u + sg * 8u + r;
        if (row >= args.rows) {
            break;
        }
        const u32 q_pos = args.pos0 + row;
        const u32 start_pos = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                                  ? (q_pos + 1u - args.window_size)
                                  : 0u;
        const u64 head_row = u64(row) * args.n_head + h;
        const float4 qv = *reinterpret_cast<const float4*>(q + head_row * 128u + d0);
        const float* rb = rel_bias + head_row * args.rel_extent;
        float m_max = sinks[head_row];
        float l_sum = 1.0f;
        float a[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        ink2_rel_attn_keys(args, start_pos, q_pos + 1u, q_pos, kvh, d0, qv, rb, k_cache, v_cache,
                           m_max, l_sum, a);
        const float inv_l = 1.0f / l_sum;
        const float4 gv = *reinterpret_cast<const float4*>(gate + head_row * 128u + d0);
        float4 res;
        res.x = (a[0] * inv_l) * ink2_sigmoid(gv.x);
        res.y = (a[1] * inv_l) * ink2_sigmoid(gv.y);
        res.z = (a[2] * inv_l) * ink2_sigmoid(gv.z);
        res.w = (a[3] * inv_l) * ink2_sigmoid(gv.w);
        *reinterpret_cast<float4*>(out + head_row * 128u + d0) = res;
    }
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

// ---- 12. Routed MoE experts (NVFP4 and MXFP8) ------------------------------
//
// See the Metal twin for the layout and what it was measured against. An
// expert's W13 is `[2 * inter, in]` with gate row `2j` and up row `2j + 1`;
// its W2 is `[out, inter]`. One inner body per codec serves the one-row
// entries, the batched entries and the tiled entries, each (weight row,
// activation row) sum taken in the same order whichever entry asked for it.

__device__ __forceinline__ float ink2_swiglu_capped(float g, float u, float gate_cap, float up_cap, float rw) {
    return ((ink2_softcap(g, gate_cap) * ink2_sigmoid(g)) * ink2_softcap(u, up_cap)) * rw;
}

// 16-term sum in the Metal twin's order: four chains of four, then pairwise.
__device__ __forceinline__ float ink2_dot16(const float* w, const float* x) {
    float c[4];
    for (u32 q = 0u; q < 4u; ++q) {
        const float* wq = w + 4u * q;
        const float* xq = x + 4u * q;
        c[q] = fmaf(wq[0], xq[0], fmaf(wq[1], xq[1], fmaf(wq[2], xq[2], wq[3] * xq[3])));
    }
    return (c[0] + c[1]) + (c[2] + c[3]);
}

// WR consecutive NVFP4 weight rows (starting at `w` / `sc`, `w_stride` and
// `s_stride` bytes apart) against up to NR activation rows. `out[wr * NR + rr]`
// is the warp-reduced dot of weight row `wr` with activation row `r_ids[rr]`,
// valid on lane 0.
template <u32 WR, u32 NR>
__device__ __forceinline__ void ink2_nvfp4_block(
        const u8* w, u64 w_stride,
        const u8* sc, u64 s_stride,
        const float* x, u32 in_dim,
        const u32* r_ids, u32 n_r, u32 lane,
        float* out) {
    float acc[WR * NR];
    for (u32 i = 0u; i < WR * NR; ++i) {
        acc[i] = 0.0f;
    }
    const u32 n_groups = in_dim / 16u;
    for (u32 g = lane; g < n_groups; g += 32u) {
        float scale[WR];
        float wv[WR][16];
        for (u32 wr = 0u; wr < WR; ++wr) {
            scale[wr] = ink2_e4m3_to_f32(u32(sc[wr * s_stride + g]));
            const u8* p = w + wr * w_stride + g * 8u;
            for (u32 b = 0u; b < 8u; ++b) {
                wv[wr][2u * b] = ink2_e2m1_to_f32(u32(p[b]) & 0xFu);
                wv[wr][2u * b + 1u] = ink2_e2m1_to_f32(u32(p[b]) >> 4u);
            }
        }
        for (u32 rr = 0u; rr < NR; ++rr) {
            if (rr < n_r) {
                const float* xp = x + u64(r_ids[rr]) * in_dim + g * 16u;
                for (u32 wr = 0u; wr < WR; ++wr) {
                    acc[wr * NR + rr] = fmaf(scale[wr], ink2_dot16(wv[wr], xp), acc[wr * NR + rr]);
                }
            }
        }
    }
    for (u32 i = 0u; i < WR * NR; ++i) {
        out[i] = ink2_warp_sum(acc[i]);
    }
}

// The MXFP8 twin of `ink2_nvfp4_block`: a 32-weight group is two 16-term sums,
// each added under the group's scale.
template <u32 WR, u32 NR>
__device__ __forceinline__ void ink2_mxfp8_block(
        const u8* w, u64 w_stride,
        const u8* sc, u64 s_stride,
        const float* x, u32 in_dim,
        const u32* r_ids, u32 n_r, u32 lane,
        float* out) {
    float acc[WR * NR];
    for (u32 i = 0u; i < WR * NR; ++i) {
        acc[i] = 0.0f;
    }
    const u32 n_groups = in_dim / 32u;
    for (u32 g = lane; g < n_groups; g += 32u) {
        float scale[WR];
        for (u32 wr = 0u; wr < WR; ++wr) {
            scale[wr] = ink2_ue8m0_to_f32(u32(sc[wr * s_stride + g]));
        }
        for (u32 h = 0u; h < 2u; ++h) {
            float wv[WR][16];
            for (u32 wr = 0u; wr < WR; ++wr) {
                const u8* p = w + wr * w_stride + g * 32u + h * 16u;
                for (u32 i = 0u; i < 16u; ++i) {
                    wv[wr][i] = ink2_e4m3_to_f32(u32(p[i]));
                }
            }
            for (u32 rr = 0u; rr < NR; ++rr) {
                if (rr < n_r) {
                    const float* xp = x + u64(r_ids[rr]) * in_dim + g * 32u + h * 16u;
                    for (u32 wr = 0u; wr < WR; ++wr) {
                        acc[wr * NR + rr] = fmaf(scale[wr], ink2_dot16(wv[wr], xp), acc[wr * NR + rr]);
                    }
                }
            }
        }
    }
    for (u32 i = 0u; i < WR * NR; ++i) {
        out[i] = ink2_warp_sum(acc[i]);
    }
}

// ---- 12a. One-row entries: a routed column per block row -------------------
// A warp owns four weight rows: two (gate, up) pairs of W13, four output rows
// of W2. `inter_dim` must be even and `out_dim` a multiple of four.
// W13: grid (ceil(inter_dim / 8), rows * top_k, 1), block 128.
// W2:  grid (ceil(out_dim / 16), rows * top_k, 1), block 128; one output row
//      per routed column, which `moe_reduce_topk` then sums per token.

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
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.in_dim;
    const u32 n = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(k / 2u);
    const u64 s_row_bytes = u64(k / 16u);
    const u32 j = 2u * (blockIdx.x * INK2_MV_SG + sg);
    const u32 col = blockIdx.y;
    if (j >= args.inter_dim || col >= args.rows * args.top_k) {
        return;
    }
    const u32 expert = expert_ids[col];
    const u32 n_r = 1u;
    u32 c_ids[1] = {col};
    u32 r_ids[1] = {col / args.top_k};
    const u64 e_row = u64(expert) * n + 2u * j;
    float v[4];
    ink2_nvfp4_block<4u, 1u>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = w13_amax[expert] / w13_denom[0];
        for (u32 q = 0u; q < 2u; ++q) {
            if (j + q >= args.inter_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                out[u64(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    v[(2u * q) * 1u + rr] * outer, v[(2u * q + 1u) * 1u + rr] * outer,
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w2_cols(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const float* w2_amax,
        const float* w2_denom,
        const u32* expert_ids,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.inter_dim;
    const u32 n = args.out_dim;
    const u64 w_row_bytes = u64(k / 2u);
    const u64 s_row_bytes = u64(k / 16u);
    const u32 o = 4u * (blockIdx.x * INK2_MV_SG + sg);
    const u32 col = blockIdx.y;
    if (o >= args.out_dim || col >= args.rows * args.top_k) {
        return;
    }
    const u32 expert = expert_ids[col];
    const u32 n_r = 1u;
    u32 c_ids[1] = {col};
    u32 r_ids[1] = {col};
    const u64 e_row = u64(expert) * n + o;
    float v[4];
    ink2_nvfp4_block<4u, 1u>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = w2_amax[expert] / w2_denom[0];
        for (u32 wr = 0u; wr < 4u; ++wr) {
            if (o + wr >= args.out_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                slots_out[u64(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 1u + rr] * outer;
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w13_swiglu(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const u32* expert_ids,
        const float* expert_weights,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.in_dim;
    const u32 n = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(k);
    const u64 s_row_bytes = u64(k / 32u);
    const u32 j = 2u * (blockIdx.x * INK2_MV_SG + sg);
    const u32 col = blockIdx.y;
    if (j >= args.inter_dim || col >= args.rows * args.top_k) {
        return;
    }
    const u32 expert = expert_ids[col];
    const u32 n_r = 1u;
    u32 c_ids[1] = {col};
    u32 r_ids[1] = {col / args.top_k};
    const u64 e_row = u64(expert) * n + 2u * j;
    float v[4];
    ink2_mxfp8_block<4u, 1u>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        for (u32 q = 0u; q < 2u; ++q) {
            if (j + q >= args.inter_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                out[u64(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    v[(2u * q) * 1u + rr], v[(2u * q + 1u) * 1u + rr],
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w2_cols(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const u32* expert_ids,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.inter_dim;
    const u32 n = args.out_dim;
    const u64 w_row_bytes = u64(k);
    const u64 s_row_bytes = u64(k / 32u);
    const u32 o = 4u * (blockIdx.x * INK2_MV_SG + sg);
    const u32 col = blockIdx.y;
    if (o >= args.out_dim || col >= args.rows * args.top_k) {
        return;
    }
    const u32 expert = expert_ids[col];
    const u32 n_r = 1u;
    u32 c_ids[1] = {col};
    u32 r_ids[1] = {col};
    const u64 e_row = u64(expert) * n + o;
    float v[4];
    ink2_mxfp8_block<4u, 1u>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        for (u32 wr = 0u; wr < 4u; ++wr) {
            if (o + wr >= args.out_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                slots_out[u64(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 1u + rr];
            }
        }
    }
}

// ---- 12b. Multi-row entries: an expert's weights streamed once per batch ---
// `moe_plan` turns a pass's routing into work items, each one expert and up to
// `batch` of its columns. `cols` is `[n_experts, rows]`, an expert's columns
// ascending; `items` is `[1 + 2 * max_items]`: the item count, then
// `(expert, first | count << 24)` per item. Blocks past the count return.
// Grid: (1, 1, 1), block 256 (one thread an expert, at most 256 experts).

struct ink2_args_moe_plan {
    u32 rows;
    u32 top_k;
    u32 n_experts;
    u32 batch;
};
static_assert(sizeof(ink2_args_moe_plan) == 16, "ink2_args_moe_plan layout");

extern "C" __global__ void kernel_inkling2_moe_plan(
        const ink2_args_moe_plan args,
        const u32* expert_ids,
        u32* cols,
        u32* items) {
    __shared__ u32 first[INK2_MAX_EXPERTS];
    __shared__ u32 counts[INK2_MAX_EXPERTS];
    const u32 tid = threadIdx.x;
    const u32 E = args.n_experts < INK2_MAX_EXPERTS ? args.n_experts : INK2_MAX_EXPERTS;
    const u32 total = args.rows * args.top_k;
    u32 cnt = 0u;
    if (tid < E) {
        u32* dst = cols + u64(tid) * args.rows;
        for (u32 col = 0u; col < total; ++col) {
            if (expert_ids[col] == tid) {
                dst[cnt] = col;
                cnt += 1u;
            }
        }
        counts[tid] = cnt;
    }
    __syncthreads();
    if (tid == 0u) {
        u32 run = 0u;
        for (u32 e = 0u; e < E; ++e) {
            first[e] = run;
            run += (counts[e] + args.batch - 1u) / args.batch;
        }
        items[0] = run;
    }
    __syncthreads();
    if (tid < E) {
        const u32 n_items = (cnt + args.batch - 1u) / args.batch;
        for (u32 b = 0u; b < n_items; ++b) {
            const u32 idx = first[tid] + b;
            const u32 base = b * args.batch;
            const u32 left = cnt - base;
            items[1u + 2u * idx] = tid;
            items[2u + 2u * idx] = base | ((left < args.batch ? left : args.batch) << 24u);
        }
    }
}

// A warp owns two weight rows against up to eight columns.
// W13: grid (ceil(inter_dim / 4), max_items, 1), block 128.
// W2:  grid (ceil(out_dim / 8), max_items, 1), block 128.

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w13_grouped(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const float* w13_amax,
        const float* w13_denom,
        const float* expert_weights,
        const u32* cols,
        const u32* items,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.in_dim;
    const u32 n = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(k / 2u);
    const u64 s_row_bytes = u64(k / 16u);
    const u32 j = 1u * (blockIdx.x * INK2_MV_SG + sg);
    if (j >= args.inter_dim || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 n_r = packed >> 24u;
    u32 c_ids[8];
    u32 r_ids[8];
    for (u32 rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
        r_ids[rr] = c_ids[rr] / args.top_k;
    }
    const u64 e_row = u64(expert) * n + 2u * j;
    float v[16];
    ink2_nvfp4_block<2u, 8u>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = w13_amax[expert] / w13_denom[0];
        for (u32 q = 0u; q < 1u; ++q) {
            if (j + q >= args.inter_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                out[u64(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    v[(2u * q) * 8u + rr] * outer, v[(2u * q + 1u) * 8u + rr] * outer,
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
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
        const u32* cols,
        const u32* items,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.inter_dim;
    const u32 n = args.out_dim;
    const u64 w_row_bytes = u64(k / 2u);
    const u64 s_row_bytes = u64(k / 16u);
    const u32 o = 2u * (blockIdx.x * INK2_MV_SG + sg);
    if (o >= args.out_dim || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 n_r = packed >> 24u;
    u32 c_ids[8];
    u32 r_ids[8];
    for (u32 rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
        r_ids[rr] = c_ids[rr];
    }
    const u64 e_row = u64(expert) * n + o;
    float v[16];
    ink2_nvfp4_block<2u, 8u>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = w2_amax[expert] / w2_denom[0];
        for (u32 wr = 0u; wr < 2u; ++wr) {
            if (o + wr >= args.out_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                slots_out[u64(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 8u + rr] * outer;
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
        const u32* cols,
        const u32* items,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.in_dim;
    const u32 n = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(k);
    const u64 s_row_bytes = u64(k / 32u);
    const u32 j = 1u * (blockIdx.x * INK2_MV_SG + sg);
    if (j >= args.inter_dim || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 n_r = packed >> 24u;
    u32 c_ids[8];
    u32 r_ids[8];
    for (u32 rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
        r_ids[rr] = c_ids[rr] / args.top_k;
    }
    const u64 e_row = u64(expert) * n + 2u * j;
    float v[16];
    ink2_mxfp8_block<2u, 8u>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        for (u32 q = 0u; q < 1u; ++q) {
            if (j + q >= args.inter_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                out[u64(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    v[(2u * q) * 8u + rr], v[(2u * q + 1u) * 8u + rr],
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w2_grouped(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const u32* cols,
        const u32* items,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.inter_dim;
    const u32 n = args.out_dim;
    const u64 w_row_bytes = u64(k);
    const u64 s_row_bytes = u64(k / 32u);
    const u32 o = 2u * (blockIdx.x * INK2_MV_SG + sg);
    if (o >= args.out_dim || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 n_r = packed >> 24u;
    u32 c_ids[8];
    u32 r_ids[8];
    for (u32 rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
        r_ids[rr] = c_ids[rr];
    }
    const u64 e_row = u64(expert) * n + o;
    float v[16];
    ink2_mxfp8_block<2u, 8u>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, k, r_ids, n_r, lane, v);
    if (lane == 0u) {
        for (u32 wr = 0u; wr < 2u; ++wr) {
            if (o + wr >= args.out_dim) {
                break;
            }
            for (u32 rr = 0u; rr < n_r; ++rr) {
                slots_out[u64(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 8u + rr];
            }
        }
    }
}

// ---- 12c. Tiled entries: work items of up to 64 columns ---------------------
// The Metal twin multiplies a 64-column by 64-weight-row tile on simdgroup
// matrices. This twin keeps the tile's contract - the same plan, grid and
// bindings - and walks it with the batched body: a warp owns 16 of the tile's
// weight rows and takes them two at a time against the item's columns eight at
// a time. It has not been tuned on a device.
// W13: grid (ceil(2 * inter_dim / 64), max_items, 1), block 128.
// W2:  grid (ceil(out_dim / 64), max_items, 1), block 128.

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w13_tile(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const float* w13_amax,
        const float* w13_denom,
        const float* expert_weights,
        const u32* cols,
        const u32* items,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.in_dim;
    const u32 n = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(k / 2u);
    const u64 s_row_bytes = u64(k / 16u);
    const u32 n0 = blockIdx.x * 64u;
    if (n0 >= n || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 m = packed >> 24u;
    const float outer = w13_amax[expert] / w13_denom[0];
    for (u32 pair = 0u; pair < 8u; ++pair) {
        const u32 row0 = n0 + sg * 16u + pair * 2u;
        if (row0 >= n) {
            break;
        }
        const u32 j = row0 / 2u;
        const u64 e_row = u64(expert) * n + row0;
        for (u32 base = 0u; base < m; base += 8u) {
            const u32 n_r = (m - base) < 8u ? (m - base) : 8u;
            u32 c_ids[8];
            u32 r_ids[8];
            for (u32 rr = 0u; rr < 8u; ++rr) {
                c_ids[rr] = rr < n_r ? ec[base + rr] : 0u;
                r_ids[rr] = c_ids[rr] / args.top_k;
            }
            float v[16];
            ink2_nvfp4_block<2u, 8u>(
                w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
                x, k, r_ids, n_r, lane, v);
            if (lane == 0u) {
                for (u32 q = 0u; q < 1u; ++q) {
                    if (j + q >= args.inter_dim) {
                        break;
                    }
                    for (u32 rr = 0u; rr < n_r; ++rr) {
                        out[u64(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                            v[(2u * q) * 8u + rr] * outer, v[(2u * q + 1u) * 8u + rr] * outer,
                            args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_nvfp4_w2_tile(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const float* w2_amax,
        const float* w2_denom,
        const u32* cols,
        const u32* items,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.inter_dim;
    const u32 n = args.out_dim;
    const u64 w_row_bytes = u64(k / 2u);
    const u64 s_row_bytes = u64(k / 16u);
    const u32 n0 = blockIdx.x * 64u;
    if (n0 >= n || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 m = packed >> 24u;
    const float outer = w2_amax[expert] / w2_denom[0];
    for (u32 pair = 0u; pair < 8u; ++pair) {
        const u32 row0 = n0 + sg * 16u + pair * 2u;
        if (row0 >= n) {
            break;
        }
        const u32 o = row0;
        const u64 e_row = u64(expert) * n + row0;
        for (u32 base = 0u; base < m; base += 8u) {
            const u32 n_r = (m - base) < 8u ? (m - base) : 8u;
            u32 c_ids[8];
            u32 r_ids[8];
            for (u32 rr = 0u; rr < 8u; ++rr) {
                c_ids[rr] = rr < n_r ? ec[base + rr] : 0u;
                r_ids[rr] = c_ids[rr];
            }
            float v[16];
            ink2_nvfp4_block<2u, 8u>(
                w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
                act, k, r_ids, n_r, lane, v);
            if (lane == 0u) {
                for (u32 wr = 0u; wr < 2u; ++wr) {
                    if (o + wr >= args.out_dim) {
                        break;
                    }
                    for (u32 rr = 0u; rr < n_r; ++rr) {
                        slots_out[u64(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 8u + rr] * outer;
                    }
                }
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w13_tile(
        const ink2_args_moe_w13 args,
        const float* x,
        const u8* w13,
        const u8* w13_scale,
        const float* expert_weights,
        const u32* cols,
        const u32* items,
        float* out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.in_dim;
    const u32 n = 2u * args.inter_dim;
    const u64 w_row_bytes = u64(k);
    const u64 s_row_bytes = u64(k / 32u);
    const u32 n0 = blockIdx.x * 64u;
    if (n0 >= n || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 m = packed >> 24u;
    for (u32 pair = 0u; pair < 8u; ++pair) {
        const u32 row0 = n0 + sg * 16u + pair * 2u;
        if (row0 >= n) {
            break;
        }
        const u32 j = row0 / 2u;
        const u64 e_row = u64(expert) * n + row0;
        for (u32 base = 0u; base < m; base += 8u) {
            const u32 n_r = (m - base) < 8u ? (m - base) : 8u;
            u32 c_ids[8];
            u32 r_ids[8];
            for (u32 rr = 0u; rr < 8u; ++rr) {
                c_ids[rr] = rr < n_r ? ec[base + rr] : 0u;
                r_ids[rr] = c_ids[rr] / args.top_k;
            }
            float v[16];
            ink2_mxfp8_block<2u, 8u>(
                w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
                x, k, r_ids, n_r, lane, v);
            if (lane == 0u) {
                for (u32 q = 0u; q < 1u; ++q) {
                    if (j + q >= args.inter_dim) {
                        break;
                    }
                    for (u32 rr = 0u; rr < n_r; ++rr) {
                        out[u64(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                            v[(2u * q) * 8u + rr], v[(2u * q + 1u) * 8u + rr],
                            args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
                    }
                }
            }
        }
    }
}

extern "C" __global__ void kernel_inkling2_moe_mxfp8_w2_tile(
        const ink2_args_moe_w2 args,
        const float* act,
        const u8* w2,
        const u8* w2_scale,
        const u32* cols,
        const u32* items,
        float* slots_out) {
    const u32 lane = threadIdx.x & 31u;
    const u32 sg = threadIdx.x >> 5u;
    const u32 k = args.inter_dim;
    const u32 n = args.out_dim;
    const u64 w_row_bytes = u64(k);
    const u64 s_row_bytes = u64(k / 32u);
    const u32 n0 = blockIdx.x * 64u;
    if (n0 >= n || blockIdx.y >= items[0]) {
        return;
    }
    const u32 expert = items[1u + 2u * blockIdx.y];
    const u32 packed = items[2u + 2u * blockIdx.y];
    const u32* ec = cols + u64(expert) * args.rows + (packed & 0xFFFFFFu);
    const u32 m = packed >> 24u;
    for (u32 pair = 0u; pair < 8u; ++pair) {
        const u32 row0 = n0 + sg * 16u + pair * 2u;
        if (row0 >= n) {
            break;
        }
        const u32 o = row0;
        const u64 e_row = u64(expert) * n + row0;
        for (u32 base = 0u; base < m; base += 8u) {
            const u32 n_r = (m - base) < 8u ? (m - base) : 8u;
            u32 c_ids[8];
            u32 r_ids[8];
            for (u32 rr = 0u; rr < 8u; ++rr) {
                c_ids[rr] = rr < n_r ? ec[base + rr] : 0u;
                r_ids[rr] = c_ids[rr];
            }
            float v[16];
            ink2_mxfp8_block<2u, 8u>(
                w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
                act, k, r_ids, n_r, lane, v);
            if (lane == 0u) {
                for (u32 wr = 0u; wr < 2u; ++wr) {
                    if (o + wr >= args.out_dim) {
                        break;
                    }
                    for (u32 rr = 0u; rr < n_r; ++rr) {
                        slots_out[u64(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 8u + rr];
                    }
                }
            }
        }
    }
}

// Sums a row's `top_k` slots in slot order.
extern "C" __global__ void kernel_inkling2_moe_reduce_topk(
        const ink2_args_moe_w2 args,
        const float* slots_in,
        float* out) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    const u32 total = args.rows * args.out_dim;
    if (tid >= total) {
        return;
    }
    const u32 row = tid / args.out_dim;
    const u32 o = tid % args.out_dim;
    float acc = 0.0f;
    for (u32 s = 0u; s < args.top_k; ++s) {
        acc = acc + slots_in[u64(row * args.top_k + s) * args.out_dim + o];
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

// Stores `rows` f32 rows as halves at their positions' slots of a context
// cache (a ring of `kv_cap` slots on a sliding-window layer).
// Grid: one thread an element, block 256.
struct ink2_args_mtp_store {
    u32 rows;
    u32 width;
    u32 pos0;
    u32 kv_cap;
    u32 is_swa;
};
static_assert(sizeof(ink2_args_mtp_store) == 20, "ink2_args_mtp_store layout");

extern "C" __global__ void kernel_inkling2_mtp_store_f16(
        const ink2_args_mtp_store args,
        const float* src,
        __half* cache) {
    const u32 tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= args.rows * args.width) {
        return;
    }
    const u32 row = tid / args.width;
    const u32 c = tid - row * args.width;
    const u32 pos = args.pos0 + row;
    const u32 slot = args.is_swa != 0u ? (pos % args.kv_cap) : pos;
    cache[u64(slot) * args.width + c] = __float2half(src[tid]);
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
