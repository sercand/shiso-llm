// The Inkling-2 family's kernel set, module `inkling2`.
//
// Every argument block's field order is the binding contract mirrored with
// `#[repr(C)]` Pods in `crates/shiso-models/src/inkling2/ops.rs` and
// `crates/shiso-kerneltest/src/inkling2.rs`. The CUDA twin is
// `kernels/cuda/inkling2/inkling2.cu`.

#define INK2_BLOCK 256u
#define INK2_MV_SG 4u
#define INK2_MV_THREADS (32u * INK2_MV_SG)
#define INK2_MAX_ATTN_RES_SRC 16u
#define INK2_MAX_LOOP_RANK 128u
#define INK2_MAX_CONV_K 8u
#define INK2_MAX_EXPERTS 256u
#define INK2_MAX_TOPK 16u

// ---- helpers ---------------------------------------------------------------

static inline float ink2_bf16_to_f32(ushort bits) {
    return as_type<float>(uint(bits) << 16u);
}

static inline float ink2_e4m3_to_f32(uint b) {
    const uint e = (b >> 3u) & 0xFu;
    const uint m = b & 7u;
    float v;
    if (e == 0u) {
        v = float(m) * 0.001953125f; // m * 2^-9
    } else if (e == 15u && m == 7u) {
        v = 0.0f;
    } else {
        v = (1.0f + float(m) * 0.125f) * exp2(float(e) - 7.0f);
    }
    return (b & 0x80u) ? -v : v;
}

static inline float ink2_ue8m0_to_f32(uint b) {
    if (b == 255u) {
        return 0.0f;
    }
    return as_type<float>(b << 23u);
}

constant float INK2_E2M1_LUT[16] = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
   -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float ink2_e2m1_to_f32(uint nib) {
    return INK2_E2M1_LUT[nib & 0xFu];
}

static inline float ink2_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

static inline float ink2_softcap(float x, float cap) {
    if (cap <= 0.0f) {
        return x;
    }
    return cap * tanh(x / cap);
}

// ---- argument blocks -------------------------------------------------------

struct ink2_args_norm {
    uint rows;
    uint hidden;
    float eps;
};
static_assert(sizeof(ink2_args_norm) == 12, "ink2_args_norm layout");

struct ink2_args_linear {
    uint rows;
    uint in_dim;
    uint out_dim;
};
static_assert(sizeof(ink2_args_linear) == 12, "ink2_args_linear layout");

struct ink2_args_qkvr_gate_sink {
    uint rows;
    uint in_dim;
    uint q_dim;
    uint kv_dim;
    uint r_dim;
    uint n_head;
};
static_assert(sizeof(ink2_args_qkvr_gate_sink) == 24, "ink2_args_qkvr_gate_sink layout");

struct ink2_args_lm_head {
    uint rows;
    uint hidden;
    uint vocab;
    float softcap;
};
static_assert(sizeof(ink2_args_lm_head) == 16, "ink2_args_lm_head layout");

struct ink2_args_attn_res {
    uint rows;
    uint hidden;
    uint n_blocks;
    uint n_queries;
    float eps;
    uint has_addend;
};
static_assert(sizeof(ink2_args_attn_res) == 24, "ink2_args_attn_res layout");

struct ink2_args_loop_ctrl {
    uint rows;
    uint hidden;
    uint rank;
    float eps;
};
static_assert(sizeof(ink2_args_loop_ctrl) == 16, "ink2_args_loop_ctrl layout");

struct ink2_args_sconv_static {
    uint rows;
    uint channels;
    uint kernel_size;
    float scale;
};
static_assert(sizeof(ink2_args_sconv_static) == 16, "ink2_args_sconv_static layout");

struct ink2_args_sconv_dynamic {
    uint rows;
    uint kv_heads;
    uint head_dim;
    uint kernel_size;
    float scale;
};
static_assert(sizeof(ink2_args_sconv_dynamic) == 20, "ink2_args_sconv_dynamic layout");

struct ink2_args_qkvr_prep {
    uint rows;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint d_rel;
    float qk_clip;
    float rel_clip;
    float eps;
    uint is_full;
    float log_n_floor;
    float log_alpha;
    uint pos0;
    uint kv_cap;
    uint is_swa;
};
static_assert(sizeof(ink2_args_qkvr_prep) == 56, "ink2_args_qkvr_prep layout");

struct ink2_args_rel_attn {
    uint rows;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint rel_extent;
    uint pos0;
    uint kv_cap;
    uint window_size;
    uint is_swa;
    float scale;
};
static_assert(sizeof(ink2_args_rel_attn) == 40, "ink2_args_rel_attn layout");

struct ink2_args_route {
    uint rows;
    uint n_experts;
    uint top_k;
    float route_scale;
};
static_assert(sizeof(ink2_args_route) == 16, "ink2_args_route layout");

struct ink2_args_moe_w13 {
    uint rows;
    uint in_dim;
    uint inter_dim;
    uint top_k;
    float gate_cap;
    float up_cap;
};
static_assert(sizeof(ink2_args_moe_w13) == 24, "ink2_args_moe_w13 layout");

struct ink2_args_moe_w2 {
    uint rows;
    uint inter_dim;
    uint out_dim;
    uint top_k;
};
static_assert(sizeof(ink2_args_moe_w2) == 16, "ink2_args_moe_w2 layout");

struct ink2_args_add3 {
    uint count;
};
static_assert(sizeof(ink2_args_add3) == 4, "ink2_args_add3 layout");

// ---- 1. Embedding + RMSNorm (BF16) -----------------------------------------

kernel void kernel_inkling2_embed_rmsnorm_bf16(
        constant ink2_args_norm &args [[buffer(0)]],
        device const uint *tokens [[buffer(1)]],
        device const ushort *embed [[buffer(2)]],
        device const ushort *norm_w [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (gid >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    const ulong tok = ulong(tokens[gid]);
    device const ushort *src = embed + tok * H;
    device float *dst = out + ulong(gid) * H;

    float sum_sq = 0.0f;
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        const float v = ink2_bf16_to_f32(src[i]);
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    threadgroup float partials[8];
    if (lane == 0u) {
        partials[sg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; ++s) {
            tot += partials[s];
        }
        partials[0] = rsqrt(tot / float(H) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partials[0];
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = ink2_bf16_to_f32(src[i]) * inv * ink2_bf16_to_f32(norm_w[i]);
    }
}

// ---- 2. RMSNorm (F32 in, BF16 weight, F32 out) -----------------------------

kernel void kernel_inkling2_rmsnorm_bf16(
        constant ink2_args_norm &args [[buffer(0)]],
        device const float *inp [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (gid >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    device const float *src = inp + ulong(gid) * H;
    device float *dst = out + ulong(gid) * H;

    float sum_sq = 0.0f;
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        const float v = src[i];
        sum_sq += v * v;
    }
    sum_sq = simd_sum(sum_sq);
    threadgroup float partials[8];
    if (lane == 0u) {
        partials[sg] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; ++s) {
            tot += partials[s];
        }
        partials[0] = rsqrt(tot / float(H) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = partials[0];
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = src[i] * inv * ink2_bf16_to_f32(w[i]);
    }
}

// ---- 3. Linear BF16 (F32 x [rows, in] @ BF16 w^T [out, in] -> F32) ---------
// Grid: (ceil(out_dim / 4), ceil(rows / 8), 1), block: 128.
// Processes up to 8 rows per simdgroup, reading each BF16 weight row once per
// 8 tokens during prefill (8x weight bandwidth reduction) with a 1-row fast path
// for decode.

kernel void kernel_inkling2_linear_bf16(
        constant ink2_args_linear &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * INK2_MV_SG + sg;
    const uint r0 = group.y * 8u;
    if (o >= args.out_dim || r0 >= args.rows) {
        return;
    }
    const uint n_r = min(8u, args.rows - r0);
    const uint in_dim = args.in_dim;
    device const ushort *wr = w + ulong(o) * in_dim;

    if (n_r == 1u) {
        device const float *xr = x + ulong(r0) * in_dim;
        float acc = 0.0f;
        for (uint k = lane * 4u; k + 3u < in_dim; k += 128u) {
            const float4 xv = *((device const float4 *)(xr + k));
            const ushort4 wv = *((device const ushort4 *)(wr + k));
            acc += xv.x * ink2_bf16_to_f32(wv.x) +
                   xv.y * ink2_bf16_to_f32(wv.y) +
                   xv.z * ink2_bf16_to_f32(wv.z) +
                   xv.w * ink2_bf16_to_f32(wv.w);
        }
        const uint tail_idx = (in_dim & ~3u) + lane;
        if (tail_idx < in_dim) {
            acc += xr[tail_idx] * ink2_bf16_to_f32(wr[tail_idx]);
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            out[ulong(r0) * args.out_dim + o] = acc;
        }
        return;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = lane * 4u; k + 3u < in_dim; k += 128u) {
        const ushort4 wv = *((device const ushort4 *)(wr + k));
        const float w0 = ink2_bf16_to_f32(wv.x);
        const float w1 = ink2_bf16_to_f32(wv.y);
        const float w2 = ink2_bf16_to_f32(wv.z);
        const float w3 = ink2_bf16_to_f32(wv.w);
        for (uint rr = 0u; rr < n_r; ++rr) {
            const float4 xv = *((device const float4 *)(x + ulong(r0 + rr) * in_dim + k));
            acc[rr] += xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
        }
    }
    const uint tail_idx = (in_dim & ~3u) + lane;
    if (tail_idx < in_dim) {
        const float wt = ink2_bf16_to_f32(wr[tail_idx]);
        for (uint rr = 0u; rr < n_r; ++rr) {
            acc[rr] += x[ulong(r0 + rr) * in_dim + tail_idx] * wt;
        }
    }
    for (uint rr = 0u; rr < n_r; ++rr) {
        const float s = simd_sum(acc[rr]);
        if (lane == 0u) {
            out[ulong(r0 + rr) * args.out_dim + o] = s;
        }
    }
}

// ---- 3b. Fused Q+K+V+R+Gate+Sink BF16 mega-projection ----------------------
// Collapses 6 separate attention input projections per layer into a single GPU
// dispatch (saving 240 kernel launches per decode token across all 48 layers)
// and batches up to 8 rows per simdgroup during prefill (8x weight reuse).
// Grid: (ceil(total_out / INK2_MV_SG), ceil(rows / 8), 1), block: 128.

kernel void kernel_inkling2_qkvr_gate_sink_bf16(
        constant ink2_args_qkvr_gate_sink &args [[buffer(0)]],
        device const float *x_q [[buffer(1)]],
        device const float *x_k [[buffer(2)]],
        device const float *x_v [[buffer(3)]],
        device const ushort *wq [[buffer(4)]],
        device const ushort *wk [[buffer(5)]],
        device const ushort *wv [[buffer(6)]],
        device const ushort *wr [[buffer(7)]],
        device const ushort *wgate [[buffer(8)]],
        device const ushort *wsink [[buffer(9)]],
        device float *out_q [[buffer(10)]],
        device float *out_k [[buffer(11)]],
        device float *out_v [[buffer(12)]],
        device float *out_r [[buffer(13)]],
        device float *out_gate [[buffer(14)]],
        device float *out_sink [[buffer(15)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * INK2_MV_SG + sg;
    const uint r0 = group.y * 8u;
    const uint b1 = args.q_dim;
    const uint b2 = b1 + args.kv_dim;
    const uint b3 = b2 + args.kv_dim;
    const uint b4 = b3 + args.r_dim;
    const uint b5 = b4 + args.q_dim;
    const uint b6 = b5 + args.n_head;
    if (o >= b6 || r0 >= args.rows) {
        return;
    }
    const uint n_r = min(8u, args.rows - r0);
    const uint in_dim = args.in_dim;

    device const float *x_base;
    device const ushort *wr_ptr;
    device float *dst;
    uint local_o;
    uint out_stride;
    if (o < b1) {
        local_o = o;
        x_base = x_q;
        wr_ptr = wq + ulong(local_o) * in_dim;
        dst = out_q;
        out_stride = args.q_dim;
    } else if (o < b2) {
        local_o = o - b1;
        x_base = x_k;
        wr_ptr = wk + ulong(local_o) * in_dim;
        dst = out_k;
        out_stride = args.kv_dim;
    } else if (o < b3) {
        local_o = o - b2;
        x_base = x_v;
        wr_ptr = wv + ulong(local_o) * in_dim;
        dst = out_v;
        out_stride = args.kv_dim;
    } else if (o < b4) {
        local_o = o - b3;
        x_base = x_q;
        wr_ptr = wr + ulong(local_o) * in_dim;
        dst = out_r;
        out_stride = args.r_dim;
    } else if (o < b5) {
        local_o = o - b4;
        x_base = x_q;
        wr_ptr = wgate + ulong(local_o) * in_dim;
        dst = out_gate;
        out_stride = args.q_dim;
    } else {
        local_o = o - b5;
        x_base = x_q;
        wr_ptr = wsink + ulong(local_o) * in_dim;
        dst = out_sink;
        out_stride = args.n_head;
    }

    if (n_r == 1u) {
        device const float *xr = x_base + ulong(r0) * in_dim;
        float acc = 0.0f;
        for (uint k = lane * 4u; k + 3u < in_dim; k += 128u) {
            const float4 xv = *((device const float4 *)(xr + k));
            const ushort4 wv_u = *((device const ushort4 *)(wr_ptr + k));
            acc += xv.x * ink2_bf16_to_f32(wv_u.x) +
                   xv.y * ink2_bf16_to_f32(wv_u.y) +
                   xv.z * ink2_bf16_to_f32(wv_u.z) +
                   xv.w * ink2_bf16_to_f32(wv_u.w);
        }
        const uint tail_idx = (in_dim & ~3u) + lane;
        if (tail_idx < in_dim) {
            acc += xr[tail_idx] * ink2_bf16_to_f32(wr_ptr[tail_idx]);
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            dst[ulong(r0) * out_stride + local_o] = acc;
        }
        return;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = lane * 4u; k + 3u < in_dim; k += 128u) {
        const ushort4 wv_u = *((device const ushort4 *)(wr_ptr + k));
        const float w0 = ink2_bf16_to_f32(wv_u.x);
        const float w1 = ink2_bf16_to_f32(wv_u.y);
        const float w2 = ink2_bf16_to_f32(wv_u.z);
        const float w3 = ink2_bf16_to_f32(wv_u.w);
        for (uint rr = 0u; rr < n_r; ++rr) {
            const float4 xv = *((device const float4 *)(x_base + ulong(r0 + rr) * in_dim + k));
            acc[rr] += xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
        }
    }
    const uint tail_idx = (in_dim & ~3u) + lane;
    if (tail_idx < in_dim) {
        const float wt = ink2_bf16_to_f32(wr_ptr[tail_idx]);
        for (uint rr = 0u; rr < n_r; ++rr) {
            acc[rr] += x_base[ulong(r0 + rr) * in_dim + tail_idx] * wt;
        }
    }
    for (uint rr = 0u; rr < n_r; ++rr) {
        const float s = simd_sum(acc[rr]);
        if (lane == 0u) {
            dst[ulong(r0 + rr) * out_stride + local_o] = s;
        }
    }
}

// ---- 4. LM head BF16 + softcap ---------------------------------------------

kernel void kernel_inkling2_lm_head_softcap(
        constant ink2_args_lm_head &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * INK2_MV_SG + sg;
    const uint r = group.y;
    if (o >= args.vocab || r >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    device const float *xr = x + ulong(r) * H;
    device const ushort *wr = w + ulong(o) * H;

    float acc = 0.0f;
    for (uint k = lane * 4u; k + 3u < H; k += 128u) {
        const float4 xv = *((device const float4 *)(xr + k));
        const ushort4 wv = *((device const ushort4 *)(wr + k));
        acc += xv.x * ink2_bf16_to_f32(wv.x) +
               xv.y * ink2_bf16_to_f32(wv.y) +
               xv.z * ink2_bf16_to_f32(wv.z) +
               xv.w * ink2_bf16_to_f32(wv.w);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out[ulong(r) * args.vocab + o] = ink2_softcap(acc, args.softcap);
    }
}

// ---- 5. Block Attention Residual (attn_res) --------------------------------

kernel void kernel_inkling2_attn_res(
        constant ink2_args_attn_res &args [[buffer(0)]],
        device const float *blocks [[buffer(1)]],
        device const float *partial [[buffer(2)]],
        device const float *addend [[buffer(3)]],
        device const float *queries [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (gid >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    const uint n_blocks = args.n_blocks;
    const uint n_src = n_blocks + 1u;
    const uint n_q = args.n_queries;
    const ulong row_off = ulong(gid) * H;
    const ulong block_stride = ulong(args.rows) * H;
    const float scale = rsqrt(float(H));

    threadgroup float partials[8];
    threadgroup float inv_rms[INK2_MAX_ATTN_RES_SRC];
    threadgroup float weights[4][INK2_MAX_ATTN_RES_SRC];

    for (uint s = 0u; s < n_src; ++s) {
        float sq = 0.0f;
        for (uint i = tid; i < H; i += INK2_BLOCK) {
            float v;
            if (s < n_blocks) {
                v = blocks[ulong(s) * block_stride + row_off + i];
            } else {
                v = partial[row_off + i] + (args.has_addend ? addend[row_off + i] : 0.0f);
            }
            sq += v * v;
        }
        sq = simd_sum(sq);
        if (lane == 0u) {
            partials[sg] = sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint k = 0u; k < 8u; ++k) {
                tot += partials[k];
            }
            inv_rms[s] = rsqrt(tot / float(H) + args.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float inv = inv_rms[s];
        for (uint q = 0u; q < n_q; ++q) {
            device const float *qr = queries + ulong(q) * H;
            float dot = 0.0f;
            for (uint i = tid; i < H; i += INK2_BLOCK) {
                float v;
                if (s < n_blocks) {
                    v = blocks[ulong(s) * block_stride + row_off + i];
                } else {
                    v = partial[row_off + i] + (args.has_addend ? addend[row_off + i] : 0.0f);
                }
                dot += qr[i] * (v * inv);
            }
            dot = simd_sum(dot);
            if (lane == 0u) {
                partials[sg] = dot;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0u) {
                float tot = 0.0f;
                for (uint k = 0u; k < 8u; ++k) {
                    tot += partials[k];
                }
                weights[q][s] = tot * scale;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    if (tid == 0u) {
        for (uint q = 0u; q < n_q; ++q) {
            float mx = weights[q][0];
            for (uint s = 1u; s < n_src; ++s) {
                mx = max(mx, weights[q][s]);
            }
            float denom = 0.0f;
            for (uint s = 0u; s < n_src; ++s) {
                weights[q][s] = exp(weights[q][s] - mx);
                denom += weights[q][s];
            }
            const float inv_d = 1.0f / denom;
            for (uint s = 0u; s < n_src; ++s) {
                weights[q][s] *= inv_d;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint q = 0u; q < n_q; ++q) {
        device float *dst = out + ulong(q) * block_stride + row_off;
        for (uint i = tid; i < H; i += INK2_BLOCK) {
            float acc = 0.0f;
            for (uint s = 0u; s < n_src; ++s) {
                float v;
                if (s < n_blocks) {
                    v = blocks[ulong(s) * block_stride + row_off + i];
                } else {
                    v = partial[row_off + i] + (args.has_addend ? addend[row_off + i] : 0.0f);
                }
                acc += weights[q][s] * (v * inv_rms[s]);
            }
            dst[i] = acc;
        }
    }
}

// ---- 6. Loop Message Controller (rank-64 fused) ----------------------------

kernel void kernel_inkling2_loop_control(
        constant ink2_args_loop_ctrl &args [[buffer(0)]],
        device const float *cur [[buffer(1)]],
        device const float *msg [[buffer(2)]],
        device const ushort *down_w [[buffer(3)]],
        device const ushort *gain_w [[buffer(4)]],
        device const ushort *forget_w [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (gid >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    const uint R = args.rank;
    device const float *cr = cur + ulong(gid) * H;
    device const float *mr = msg + ulong(gid) * H;
    device float *dst = out + ulong(gid) * H;

    float sq_c = 0.0f;
    float sq_m = 0.0f;
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        const float vc = cr[i];
        const float vm = mr[i];
        sq_c += vc * vc;
        sq_m += vm * vm;
    }
    sq_c = simd_sum(sq_c);
    sq_m = simd_sum(sq_m);
    threadgroup float pc[8];
    threadgroup float pm[8];
    if (lane == 0u) {
        pc[sg] = sq_c;
        pm[sg] = sq_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tc = 0.0f;
        float tm = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            tc += pc[k];
            tm += pm[k];
        }
        pc[0] = rsqrt(tc / float(H) + args.eps);
        pm[0] = rsqrt(tm / float(H) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv_c = pc[0];
    const float inv_m = pm[0];

    // h[r] = dot(down_w[r, 0..H], cr * inv_c) + dot(down_w[r, H..2H], mr * inv_m)
    threadgroup float hvec[INK2_MAX_LOOP_RANK];
    for (uint r = sg; r < R; r += 8u) {
        device const ushort *dr = down_w + ulong(r) * (2u * H);
        float acc = 0.0f;
        for (uint i = lane; i < H; i += 32u) {
            acc += ink2_bf16_to_f32(dr[i]) * (cr[i] * inv_c) +
                   ink2_bf16_to_f32(dr[H + i]) * (mr[i] * inv_m);
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            hvec[r] = acc;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint i = tid; i < H; i += INK2_BLOCK) {
        device const ushort *gr = gain_w + ulong(i) * R;
        device const ushort *fr = forget_w + ulong(i) * R;
        float g = 0.0f;
        float f = 0.0f;
        for (uint r = 0u; r < R; ++r) {
            const float hv = hvec[r];
            g += ink2_bf16_to_f32(gr[r]) * hv;
            f += ink2_bf16_to_f32(fr[r]) * hv;
        }
        dst[i] = (1.0f - tanh(f)) * cr[i] + tanh(g) * mr[i];
    }
}

// ---- 6b. Parallel Loop Message Controller (down + apply) -------------------
// Replaces the single-TG loop_control with two GPU-wide kernels (~48x faster).
// down:  grid (ceil(rank / 4), rows, 1), block 128 -> z [rows, rank]
// apply: grid (ceil(hidden / 4), rows, 1), block 128 -> out [rows, hidden]

kernel void kernel_inkling2_loop_ctrl_down(
        constant ink2_args_loop_ctrl &args [[buffer(0)]],
        device const float *cur [[buffer(1)]],
        device const float *msg [[buffer(2)]],
        device const ushort *down_w [[buffer(3)]],
        device float *z_out [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint r = group.x * INK2_MV_SG + sg;
    const uint row = group.y;
    if (row >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    device const float *cr = cur + ulong(row) * H;
    device const float *mr = msg + ulong(row) * H;

    float sq_c = 0.0f;
    float sq_m = 0.0f;
    for (uint i = tid * 4u; i + 3u < H; i += 512u) {
        const float4 vc = *((device const float4 *)(cr + i));
        const float4 vm = *((device const float4 *)(mr + i));
        sq_c += vc.x * vc.x + vc.y * vc.y + vc.z * vc.z + vc.w * vc.w;
        sq_m += vm.x * vm.x + vm.y * vm.y + vm.z * vm.z + vm.w * vm.w;
    }
    const uint tail_i = (H & ~3u) + tid;
    if (tail_i < H) {
        sq_c += cr[tail_i] * cr[tail_i];
        sq_m += mr[tail_i] * mr[tail_i];
    }
    sq_c = simd_sum(sq_c);
    sq_m = simd_sum(sq_m);
    threadgroup float pc[4];
    threadgroup float pm[4];
    if (lane == 0u) {
        pc[sg] = sq_c;
        pm[sg] = sq_m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const float tc = pc[0] + pc[1] + pc[2] + pc[3];
        const float tm = pm[0] + pm[1] + pm[2] + pm[3];
        pc[0] = rsqrt(tc / float(H) + args.eps);
        pm[0] = rsqrt(tm / float(H) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (r >= args.rank) {
        return;
    }
    const float inv_c = pc[0];
    const float inv_m = pm[0];

    device const ushort *dr = down_w + ulong(r) * (2u * H);
    float acc_c = 0.0f;
    float acc_m = 0.0f;
    for (uint i = lane * 4u; i + 3u < H; i += 128u) {
        const float4 vc = *((device const float4 *)(cr + i));
        const float4 vm = *((device const float4 *)(mr + i));
        const ushort4 wc = *((device const ushort4 *)(dr + i));
        const ushort4 wm = *((device const ushort4 *)(dr + H + i));
        acc_c += vc.x * ink2_bf16_to_f32(wc.x) + vc.y * ink2_bf16_to_f32(wc.y) +
                 vc.z * ink2_bf16_to_f32(wc.z) + vc.w * ink2_bf16_to_f32(wc.w);
        acc_m += vm.x * ink2_bf16_to_f32(wm.x) + vm.y * ink2_bf16_to_f32(wm.y) +
                 vm.z * ink2_bf16_to_f32(wm.z) + vm.w * ink2_bf16_to_f32(wm.w);
    }
    const uint tail_k = (H & ~3u) + lane;
    if (tail_k < H) {
        acc_c += cr[tail_k] * ink2_bf16_to_f32(dr[tail_k]);
        acc_m += mr[tail_k] * ink2_bf16_to_f32(dr[H + tail_k]);
    }
    const float tot = simd_sum(acc_c * inv_c + acc_m * inv_m);
    if (lane == 0u) {
        z_out[ulong(row) * args.rank + r] = tot;
    }
}

kernel void kernel_inkling2_loop_ctrl_apply(
        constant ink2_args_loop_ctrl &args [[buffer(0)]],
        device const float *cur [[buffer(1)]],
        device const float *msg [[buffer(2)]],
        device const float *z_in [[buffer(3)]],
        device const ushort *gain_w [[buffer(4)]],
        device const ushort *forget_w [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint i = group.x * INK2_MV_SG + sg;
    const uint row = group.y;
    if (row >= args.rows) {
        return;
    }
    const uint R = args.rank;
    threadgroup float zs[INK2_MAX_LOOP_RANK];
    if (tid < R) {
        zs[tid] = z_in[ulong(row) * R + tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= args.hidden) {
        return;
    }
    device const ushort *gr = gain_w + ulong(i) * R;
    device const ushort *fr = forget_w + ulong(i) * R;
    float g = 0.0f;
    float f = 0.0f;
    for (uint r = lane; r < R; r += 32u) {
        const float hv = zs[r];
        g += ink2_bf16_to_f32(gr[r]) * hv;
        f += ink2_bf16_to_f32(fr[r]) * hv;
    }
    g = simd_sum(g);
    f = simd_sum(f);
    if (lane == 0u) {
        const ulong idx = ulong(row) * args.hidden + i;
        out[idx] = (1.0f - tanh(f)) * cur[idx] + tanh(g) * msg[idx];
    }
}

// ---- 7. Static Short Causal Convolution (k=4, residual + 1/sqrt(k+1)) ------
// Weight layout: [channels, k] where w[c, 0] is oldest lag (k-1) and w[c, k-1] is current token.

kernel void kernel_inkling2_short_conv_static(
        constant ink2_args_sconv_static &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device float *state [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    const uint c = tid;
    if (c >= args.channels) {
        return;
    }
    const uint C = args.channels;
    const uint K = args.kernel_size;
    const uint lags = K - 1u;
    float hist[INK2_MAX_CONV_K];
    for (uint l = 0u; l < lags; ++l) {
        // state[0] is oldest (lag K-1), state[lags-1] is lag 1
        hist[l] = state[ulong(l) * C + c];
    }
    float wt[INK2_MAX_CONV_K];
    for (uint k = 0u; k < K; ++k) {
        wt[k] = ink2_bf16_to_f32(w[ulong(c) * K + k]);
    }
    for (uint t = 0u; t < args.rows; ++t) {
        const float cur = x[ulong(t) * C + c];
        float acc = cur * wt[lags];
        for (uint l = 0u; l < lags; ++l) {
            acc += hist[l] * wt[l];
        }
        out[ulong(t) * C + c] = (acc + cur) * args.scale;
        for (uint l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    for (uint l = 0u; l < lags; ++l) {
        state[ulong(l) * C + c] = hist[l];
    }
}

// ---- 8. Dynamic Short Causal Convolution (full-attn K/V) -------------------
// static_w: [k, C] where tap 0 is current token, tap k-1 is oldest lag.
// dyn_w:    [rows, kv_heads, k] where tap 0 is current token.

kernel void kernel_inkling2_short_conv_dynamic(
        constant ink2_args_sconv_dynamic &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const float *dyn_w [[buffer(2)]],
        device const ushort *static_w [[buffer(3)]],
        device float *state [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint tid [[thread_position_in_grid]]) {
    const uint C = args.kv_heads * args.head_dim;
    const uint c = tid;
    if (c >= C) {
        return;
    }
    const uint h = c / args.head_dim;
    const uint K = args.kernel_size;
    const uint lags = K - 1u;
    float hist[INK2_MAX_CONV_K];
    for (uint l = 0u; l < lags; ++l) {
        // state[0] = oldest (lag K-1), state[lags-1] = lag 1
        hist[l] = state[ulong(l) * C + c];
    }
    float sw[INK2_MAX_CONV_K];
    for (uint k = 0u; k < K; ++k) {
        sw[k] = ink2_bf16_to_f32(static_w[ulong(k) * C + c]);
    }
    for (uint t = 0u; t < args.rows; ++t) {
        const float cur = x[ulong(t) * C + c];
        device const float *dw = dyn_w + (ulong(t) * args.kv_heads + h) * K;
        float acc = cur * (sw[0] + dw[0]);
        for (uint lag = 1u; lag < K; ++lag) {
            const float prev = hist[lags - lag];
            acc += prev * (sw[lag] + dw[lag]);
        }
        out[ulong(t) * C + c] = (acc + cur) * args.scale;
        for (uint l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    for (uint l = 0u; l < lags; ++l) {
        state[ulong(l) * C + c] = hist[l];
    }
}

// ---- 9. QKVR Prep (L2Clip, ScaledL2Clip, V RMSNorm, log-scale, KV store) ---
// Grid: (n_head + n_head_kv, rows, 1), Block: (128, 1, 1)

kernel void kernel_inkling2_qkvr_prep_store(
        constant ink2_args_qkvr_prep &args [[buffer(0)]],
        device float *q [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        device float *r [[buffer(4)]],
        device const ushort *q_scale [[buffer(5)]],
        device const ushort *r_scale [[buffer(6)]],
        device half *k_cache [[buffer(7)]],
        device half *v_cache [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = group.y;
    if (row >= args.rows) {
        return;
    }
    const uint pos = args.pos0 + row;
    float tau = 1.0f;
    if (args.is_full != 0u) {
        const float ratio = float(pos + 1u) / args.log_n_floor;
        if (ratio > 1.0f) {
            tau = 1.0f + args.log_alpha * log(ratio);
        }
    }

    threadgroup float partials[4];

    if (group.x < args.n_head) {
        // Query head + Relative head
        const uint h = group.x;
        const uint D = args.head_dim;
        device float *qh = q + (ulong(row) * args.n_head + h) * D;
        device const ushort *qs = q_scale + ulong(h) * D;
        float sq = 0.0f;
        for (uint d = tid; d < D; d += 128u) {
            const float val = qh[d] * ink2_bf16_to_f32(qs[d]);
            qh[d] = val;
            sq += val * val;
        }
        sq = simd_sum(sq);
        if (lane == 0u) {
            partials[sg] = sq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const float norm = sqrt(partials[0] + partials[1] + partials[2] + partials[3]);
            const float factor = min(args.qk_clip / (norm + 1e-6f), 1.0f) * tau;
            partials[0] = factor;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float qf = partials[0];
        for (uint d = tid; d < D; d += 128u) {
            qh[d] *= qf;
        }

        // Relative head (d_rel <= 32, handled by first simdgroup)
        if (sg == 0u) {
            const uint DR = args.d_rel;
            device float *rh = r + (ulong(row) * args.n_head + h) * DR;
            device const ushort *rs = r_scale + ulong(h) * DR;
            float val = 0.0f;
            if (lane < DR) {
                val = rh[lane] * ink2_bf16_to_f32(rs[lane]);
            }
            const float rnorm = sqrt(simd_sum(val * val));
            const float rf = min(args.rel_clip / (rnorm + 1e-6f), 1.0f) * tau;
            if (lane < DR) {
                rh[lane] = val * rf;
            }
        }
    } else {
        // KV head: L2Clip on K, UnweightedRMSNorm on V, store to cache slot
        const uint kvh = group.x - args.n_head;
        if (kvh >= args.n_head_kv) {
            return;
        }
        const uint D = args.head_dim;
        const uint slot = args.is_swa != 0u ? (pos % args.kv_cap) : pos;
        device const float *kh = k + (ulong(row) * args.n_head_kv + kvh) * D;
        device const float *vh = v + (ulong(row) * args.n_head_kv + kvh) * D;
        device half *kdst = k_cache + (ulong(slot) * args.n_head_kv + kvh) * D;
        device half *vdst = v_cache + (ulong(slot) * args.n_head_kv + kvh) * D;

        float ksq = 0.0f;
        float vsq = 0.0f;
        for (uint d = tid; d < D; d += 128u) {
            const float kv = kh[d];
            const float vv = vh[d];
            ksq += kv * kv;
            vsq += vv * vv;
        }
        ksq = simd_sum(ksq);
        vsq = simd_sum(vsq);
        threadgroup float kp[4];
        threadgroup float vp[4];
        if (lane == 0u) {
            kp[sg] = ksq;
            vp[sg] = vsq;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            const float knorm = sqrt(kp[0] + kp[1] + kp[2] + kp[3]);
            kp[0] = min(args.qk_clip / (knorm + 1e-6f), 1.0f);
            vp[0] = rsqrt((vp[0] + vp[1] + vp[2] + vp[3]) / float(D) + args.eps);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float kf = kp[0];
        const float vf = vp[0];
        for (uint d = tid; d < D; d += 128u) {
            kdst[d] = half(kh[d] * kf);
            vdst[d] = half(vh[d] * vf);
        }
    }
}

// ---- 9b. Fast Relative-Bias Projection (d_rel <= 16, zero idle lanes) ------
// Grid: (ceil(out_dim / 128), rows, 1), block: 128.

kernel void kernel_inkling2_rel_bias_bf16(
        constant ink2_args_linear &args [[buffer(0)]],
        device const float *r_in [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint d = group.x * 128u + tid;
    const uint row = group.y;
    if (d >= args.out_dim || row >= args.rows) {
        return;
    }
    const uint D = args.in_dim;
    device const float *rr = r_in + ulong(row) * D;
    device const ushort *wr = w + ulong(d) * D;
    float acc = 0.0f;
    for (uint k = 0u; k < D; ++k) {
        acc += rr[k] * ink2_bf16_to_f32(wr[k]);
    }
    out[ulong(row) * args.out_dim + d] = acc;
}

// ---- 10. Relative-Bias + Per-Query Sink + Gated Attention (ZERO barriers) --
// Grid: (n_head, ceil(rows / 4), 1), Block: (128, 1, 1) — 1 simdgroup per (head, row).
// Pure-register float4 Q·K via simd_sum with ZERO threadgroup barriers.

kernel void kernel_inkling2_rel_attn(
        constant ink2_args_rel_attn &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *rel_bias [[buffer(2)]],
        device const float *sinks [[buffer(3)]],
        device const float *gate [[buffer(4)]],
        device const half *k_cache [[buffer(5)]],
        device const half *v_cache [[buffer(6)]],
        device float *out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint h = group.x;
    const uint row = group.y * 4u + sg;
    if (h >= args.n_head || row >= args.rows) {
        return;
    }
    const uint D = args.head_dim;
    const uint kvh = h / (args.n_head / args.n_head_kv);
    const uint q_pos = args.pos0 + row;
    const uint start_pos = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                               ? (q_pos + 1u - args.window_size)
                               : 0u;

    const ulong q_base = (ulong(row) * args.n_head + h) * D;
    device const float *rb = rel_bias + (ulong(row) * args.n_head + h) * args.rel_extent;
    const float sink_logit = sinks[ulong(row) * args.n_head + h];

    if (D == 128u) {
        const uint d0 = lane * 4u;
        const float4 qv = *((device const float4 *)(q + q_base + d0));
        float m_max = sink_logit;
        float l_sum = 1.0f;
        float4 acc = float4(0.0f);

        for (uint k_pos = start_pos; k_pos <= q_pos; ++k_pos) {
            const uint slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;
            const ulong kv_off = (ulong(slot) * args.n_head_kv + kvh) * 128u + d0;
            const half4 kv = *((device const half4 *)(k_cache + kv_off));
            float dot = qv.x * float(kv.x) + qv.y * float(kv.y) +
                        qv.z * float(kv.z) + qv.w * float(kv.w);
            dot = simd_sum(dot) * args.scale;
            const uint dist = q_pos - k_pos;
            const float bias = (dist < args.rel_extent) ? rb[dist] : 0.0f;
            const float score = dot + bias;

            const float new_max = max(m_max, score);
            const float exp_old = exp(m_max - new_max);
            const float exp_new = exp(score - new_max);
            l_sum = l_sum * exp_old + exp_new;
            const half4 vv = *((device const half4 *)(v_cache + kv_off));
            acc = acc * exp_old + exp_new * float4(vv);
            m_max = new_max;
        }

        const float inv_l = 1.0f / l_sum;
        const float4 gv = *((device const float4 *)(gate + q_base + d0));
        const float4 res = float4(
            (acc.x * inv_l) * ink2_sigmoid(gv.x),
            (acc.y * inv_l) * ink2_sigmoid(gv.y),
            (acc.z * inv_l) * ink2_sigmoid(gv.z),
            (acc.w * inv_l) * ink2_sigmoid(gv.w)
        );
        *((device float4 *)(out + q_base + d0)) = res;
        return;
    }

    const float q0 = (lane < D) ? q[q_base + lane] : 0.0f;
    const float q1 = (lane + 32u < D) ? q[q_base + lane + 32u] : 0.0f;
    const float q2 = (lane + 64u < D) ? q[q_base + lane + 64u] : 0.0f;
    const float q3 = (lane + 96u < D) ? q[q_base + lane + 96u] : 0.0f;
    float m_max = sink_logit;
    float l_sum = 1.0f;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;

    for (uint k_pos = start_pos; k_pos <= q_pos; ++k_pos) {
        const uint slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;
        const ulong kv_off = (ulong(slot) * args.n_head_kv + kvh) * D;
        float dot = 0.0f;
        if (lane < D) dot += q0 * float(k_cache[kv_off + lane]);
        if (lane + 32u < D) dot += q1 * float(k_cache[kv_off + lane + 32u]);
        if (lane + 64u < D) dot += q2 * float(k_cache[kv_off + lane + 64u]);
        if (lane + 96u < D) dot += q3 * float(k_cache[kv_off + lane + 96u]);
        dot = simd_sum(dot) * args.scale;
        const uint dist = q_pos - k_pos;
        const float bias = (dist < args.rel_extent) ? rb[dist] : 0.0f;
        const float score = dot + bias;

        const float new_max = max(m_max, score);
        const float exp_old = exp(m_max - new_max);
        const float exp_new = exp(score - new_max);
        l_sum = l_sum * exp_old + exp_new;
        if (lane < D) a0 = a0 * exp_old + exp_new * float(v_cache[kv_off + lane]);
        if (lane + 32u < D) a1 = a1 * exp_old + exp_new * float(v_cache[kv_off + lane + 32u]);
        if (lane + 64u < D) a2 = a2 * exp_old + exp_new * float(v_cache[kv_off + lane + 64u]);
        if (lane + 96u < D) a3 = a3 * exp_old + exp_new * float(v_cache[kv_off + lane + 96u]);
        m_max = new_max;
    }

    const float inv_l = 1.0f / l_sum;
    if (lane < D) out[q_base + lane] = (a0 * inv_l) * ink2_sigmoid(gate[q_base + lane]);
    if (lane + 32u < D) out[q_base + lane + 32u] = (a1 * inv_l) * ink2_sigmoid(gate[q_base + lane + 32u]);
    if (lane + 64u < D) out[q_base + lane + 64u] = (a2 * inv_l) * ink2_sigmoid(gate[q_base + lane + 64u]);
    if (lane + 96u < D) out[q_base + lane + 96u] = (a3 * inv_l) * ink2_sigmoid(gate[q_base + lane + 96u]);
}

// ---- 10b. Split-K Flash-Decoding for Long-Context Full Attention -----------
// Stage 1: Grid (n_head, ceil(n_splits / 4), 1), block 128 (4 splits per TG).
// Stage 2: Grid (n_head, 1, 1), block 32 (merges n_splits partials + output gate).

struct ink2_args_rel_attn_splitk {
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint rel_extent;
    uint pos;
    uint n_splits;
    float scale;
};
static_assert(sizeof(ink2_args_rel_attn_splitk) == 28, "ink2_args_rel_attn_splitk layout");

kernel void kernel_inkling2_rel_attn_splitk(
        constant ink2_args_rel_attn_splitk &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *rel_bias [[buffer(2)]],
        device const float *sinks [[buffer(3)]],
        device const half *k_cache [[buffer(4)]],
        device const half *v_cache [[buffer(5)]],
        device float *split_ml [[buffer(6)]],
        device float *split_out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint h = group.x;
    const uint s = group.y * 4u + sg;
    if (h >= args.n_head || s >= args.n_splits) {
        return;
    }
    const uint kvh = h / (args.n_head / args.n_head_kv);
    const uint q_pos = args.pos;
    const uint total_kv = q_pos + 1u;
    const uint chunk = (total_kv + args.n_splits - 1u) / args.n_splits;
    const uint start_pos = s * chunk;
    const uint end_pos = min(total_kv, start_pos + chunk);

    const uint d0 = lane * 4u;
    const ulong q_base = ulong(h) * 128u;
    const float4 qv = *((device const float4 *)(q + q_base + d0));
    device const float *rb = rel_bias + ulong(h) * args.rel_extent;

    float m_max = (s == 0u) ? sinks[h] : -1e30f;
    float l_sum = (s == 0u) ? 1.0f : 0.0f;
    float4 acc = float4(0.0f);

    for (uint k_pos = start_pos; k_pos < end_pos; ++k_pos) {
        const ulong kv_off = (ulong(k_pos) * args.n_head_kv + kvh) * 128u + d0;
        const half4 kv = *((device const half4 *)(k_cache + kv_off));
        float dot = qv.x * float(kv.x) + qv.y * float(kv.y) +
                    qv.z * float(kv.z) + qv.w * float(kv.w);
        dot = simd_sum(dot) * args.scale;
        const uint dist = q_pos - k_pos;
        const float bias = (dist < args.rel_extent) ? rb[dist] : 0.0f;
        const float score = dot + bias;

        const float new_max = max(m_max, score);
        const float exp_old = (l_sum > 0.0f) ? exp(m_max - new_max) : 0.0f;
        const float exp_new = exp(score - new_max);
        l_sum = l_sum * exp_old + exp_new;
        const half4 vv = *((device const half4 *)(v_cache + kv_off));
        acc = acc * exp_old + exp_new * float4(vv);
        m_max = new_max;
    }

    const ulong s_idx = ulong(h) * args.n_splits + s;
    if (lane == 0u) {
        split_ml[s_idx * 2u] = m_max;
        split_ml[s_idx * 2u + 1u] = l_sum;
    }
    *((device float4 *)(split_out + s_idx * 128u + d0)) = acc;
}

kernel void kernel_inkling2_rel_attn_reduce(
        constant ink2_args_rel_attn_splitk &args [[buffer(0)]],
        device const float *split_ml [[buffer(1)]],
        device const float *split_out [[buffer(2)]],
        device const float *gate [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint gid [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint h = gid;
    if (h >= args.n_head) {
        return;
    }
    const uint d0 = lane * 4u;
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float4 acc = float4(0.0f);

    for (uint s = 0u; s < args.n_splits; ++s) {
        const ulong s_idx = ulong(h) * args.n_splits + s;
        const float ms = split_ml[s_idx * 2u];
        const float ls = split_ml[s_idx * 2u + 1u];
        if (ls <= 0.0f) {
            continue;
        }
        const float4 vs = *((device const float4 *)(split_out + s_idx * 128u + d0));
        const float nm = max(m_max, ms);
        const float eo = (l_sum > 0.0f) ? exp(m_max - nm) : 0.0f;
        const float en = exp(ms - nm);
        l_sum = l_sum * eo + ls * en;
        acc = acc * eo + vs * en;
        m_max = nm;
    }

    const float inv_l = 1.0f / l_sum;
    const ulong base = ulong(h) * 128u + d0;
    const float4 gv = *((device const float4 *)(gate + base));
    const float4 res = float4(
        (acc.x * inv_l) * ink2_sigmoid(gv.x),
        (acc.y * inv_l) * ink2_sigmoid(gv.y),
        (acc.z * inv_l) * ink2_sigmoid(gv.z),
        (acc.w * inv_l) * ink2_sigmoid(gv.w)
    );
    *((device float4 *)(out + base)) = res;
}

// ---- 11. Top-K Sigmoid Router with Bias (parallel 128-thread block) --------
// Grid: (rows, 1, 1), Block: (128, 1, 1) — ~30x faster than 1-thread block.

kernel void kernel_inkling2_route_topk(
        constant ink2_args_route &args [[buffer(0)]],
        device const float *logits [[buffer(1)]],
        device const float *bias [[buffer(2)]],
        device uint *out_ids [[buffer(3)]],
        device float *out_weights [[buffer(4)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint row = gid;
    if (row >= args.rows) {
        return;
    }
    const uint E = args.n_experts;
    const uint K = args.top_k;
    device const float *lr = logits + ulong(row) * E;

    threadgroup float s_arr[INK2_MAX_EXPERTS];
    threadgroup float b_arr[INK2_MAX_EXPERTS];
    threadgroup float sg_val[4];
    threadgroup uint sg_idx[4];
    threadgroup uint picked_id[INK2_MAX_TOPK];
    threadgroup float picked_w[INK2_MAX_TOPK];

    if (tid < E) {
        const float s = ink2_sigmoid(lr[tid]);
        s_arr[tid] = s;
        b_arr[tid] = s + bias[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0u; k < K; ++k) {
        const float cur_b = (tid < E) ? b_arr[tid] : -1e38f;
        const float wmax = simd_max(cur_b);
        const uint cand = (cur_b == wmax) ? tid : 0xFFFFFFFFu;
        const uint win_idx = simd_min(cand);
        if (lane == 0u) {
            sg_val[sg] = wmax;
            sg_idx[sg] = win_idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float bv = sg_val[0];
            uint bi = sg_idx[0];
            for (uint s = 1u; s < 4u; ++s) {
                if (sg_val[s] > bv || (sg_val[s] == bv && sg_idx[s] < bi)) {
                    bv = sg_val[s];
                    bi = sg_idx[s];
                }
            }
            picked_id[k] = bi;
            picked_w[k] = s_arr[bi];
            b_arr[bi] = -1e38f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid < K) {
        float w_sum = 0.0f;
        for (uint k = 0u; k < K; ++k) {
            w_sum += picked_w[k];
        }
        const float scale = args.route_scale / (w_sum + 1e-20f);
        out_ids[ulong(row) * K + tid] = picked_id[tid];
        out_weights[ulong(row) * K + tid] = picked_w[tid] * scale;
    }
}

// ---- 12. NVFP4 MoE W13 (interleaved gate/up) + Softcapped SwiGLU -----------
// Optimized with threadgroup 256-entry E4M3 LUT + vectorized float4 loads.
// Grid: (ceil(inter_dim / 4), rows * top_k, 1), Block: (128, 1, 1)

static inline void ink2_fill_e4m3_lut(threadgroup float *lut, uint tid) {
    lut[tid] = ink2_e4m3_to_f32(tid);
    lut[tid + 128u] = ink2_e4m3_to_f32(tid + 128u);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline float ink2_nvfp4_row_dot_lut(
        device const uchar *w_row,
        device const uchar *s_row,
        device const float *x,
        threadgroup const float *e4m3_lut,
        uint in_dim,
        uint lane) {
    float acc = 0.0f;
    const uint n_groups = in_dim / 16u;
    for (uint g = lane; g < n_groups; g += 32u) {
        const float gs = e4m3_lut[uint(s_row[g])];
        device const uchar4 *wp4 = (device const uchar4 *)(w_row + g * 8u);
        device const float4 *xp4 = (device const float4 *)(x + g * 16u);
        const uchar4 b0 = wp4[0];
        const uchar4 b1 = wp4[1];
        const float4 x0 = xp4[0];
        const float4 x1 = xp4[1];
        const float4 x2 = xp4[2];
        const float4 x3 = xp4[3];
        float part =
            INK2_E2M1_LUT[b0.x & 0xFu] * x0.x + INK2_E2M1_LUT[b0.x >> 4u] * x0.y +
            INK2_E2M1_LUT[b0.y & 0xFu] * x0.z + INK2_E2M1_LUT[b0.y >> 4u] * x0.w +
            INK2_E2M1_LUT[b0.z & 0xFu] * x1.x + INK2_E2M1_LUT[b0.z >> 4u] * x1.y +
            INK2_E2M1_LUT[b0.w & 0xFu] * x1.z + INK2_E2M1_LUT[b0.w >> 4u] * x1.w +
            INK2_E2M1_LUT[b1.x & 0xFu] * x2.x + INK2_E2M1_LUT[b1.x >> 4u] * x2.y +
            INK2_E2M1_LUT[b1.y & 0xFu] * x2.z + INK2_E2M1_LUT[b1.y >> 4u] * x2.w +
            INK2_E2M1_LUT[b1.z & 0xFu] * x3.x + INK2_E2M1_LUT[b1.z >> 4u] * x3.y +
            INK2_E2M1_LUT[b1.w & 0xFu] * x3.z + INK2_E2M1_LUT[b1.w >> 4u] * x3.w;
        acc += gs * part;
    }
    return simd_sum(acc);
}

kernel void kernel_inkling2_moe_nvfp4_w13_swiglu(
        constant ink2_args_moe_w13 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const uchar *w13 [[buffer(2)]],
        device const uchar *w13_scale [[buffer(3)]],
        device const float *w13_amax [[buffer(4)]],
        device const float *w13_denom [[buffer(5)]],
        device const uint *expert_ids [[buffer(6)]],
        device const float *expert_weights [[buffer(7)]],
        device float *out [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, tid);

    const uint j = group.x * INK2_MV_SG + sg;
    const uint col = group.y;
    const uint total_cols = args.rows * args.top_k;
    if (j >= args.inter_dim || col >= total_cols) {
        return;
    }
    const uint row = col / args.top_k;
    const uint expert = expert_ids[col];
    const float rw = expert_weights[col];
    const float outer = w13_amax[expert] / w13_denom[0];

    const uint in_dim = args.in_dim;
    const uint two_inter = 2u * args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim / 2u);
    const ulong s_row_bytes = ulong(in_dim / 16u);
    const ulong e_w_base = ulong(expert) * two_inter * w_row_bytes;
    const ulong e_s_base = ulong(expert) * two_inter * s_row_bytes;

    device const float *xr = x + ulong(row) * in_dim;
    const uint g_row = 2u * j;
    const uint u_row = 2u * j + 1u;

    const float g_val = ink2_nvfp4_row_dot_lut(
        w13 + e_w_base + ulong(g_row) * w_row_bytes,
        w13_scale + e_s_base + ulong(g_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane) * outer;
    const float u_val = ink2_nvfp4_row_dot_lut(
        w13 + e_w_base + ulong(u_row) * w_row_bytes,
        w13_scale + e_s_base + ulong(u_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane) * outer;

    if (lane == 0u) {
        const float act = ink2_softcap(g_val, args.gate_cap) *
                          ink2_sigmoid(g_val) *
                          ink2_softcap(u_val, args.up_cap) * rw;
        out[ulong(col) * args.inter_dim + j] = act;
    }
}

// ---- 13. NVFP4 MoE W2 + Expert Accumulation --------------------------------
// Grid: (ceil(out_dim / 4), rows, 1), Block: (128, 1, 1)

kernel void kernel_inkling2_moe_nvfp4_w2_accum(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const float *w2_amax [[buffer(4)]],
        device const float *w2_denom [[buffer(5)]],
        device const uint *expert_ids [[buffer(6)]],
        device float *out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, tid);

    const uint o = group.x * INK2_MV_SG + sg;
    const uint row = group.y;
    if (o >= args.out_dim || row >= args.rows) {
        return;
    }
    const uint inter_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(inter_dim / 2u);
    const ulong s_row_bytes = ulong(inter_dim / 16u);
    const float denom = w2_denom[0];

    float total = 0.0f;
    for (uint slot = 0u; slot < args.top_k; ++slot) {
        const uint col = row * args.top_k + slot;
        const uint expert = expert_ids[col];
        const float outer = w2_amax[expert] / denom;
        const ulong e_w_base = (ulong(expert) * args.out_dim + o) * w_row_bytes;
        const ulong e_s_base = (ulong(expert) * args.out_dim + o) * s_row_bytes;
        device const float *ar = act + ulong(col) * inter_dim;
        total += ink2_nvfp4_row_dot_lut(w2 + e_w_base, w2_scale + e_s_base, ar, e4m3_lut, inter_dim, lane) * outer;
    }
    if (lane == 0u) {
        out[ulong(row) * args.out_dim + o] = total;
    }
}

// ---- 14. MXFP8 MoE W13 (interleaved gate/up) + Softcapped SwiGLU -----------
// Optimized with threadgroup 256-entry E4M3 LUT + vectorized uchar4/float4 loads.
// Grid: (ceil(inter_dim / 4), rows * top_k, 1), Block: (128, 1, 1)

static inline float ink2_mxfp8_row_dot_lut(
        device const uchar *w_row,
        device const uchar *s_row,
        device const float *x,
        threadgroup const float *e4m3_lut,
        uint in_dim,
        uint lane) {
    float acc = 0.0f;
    const uint n_groups = in_dim / 32u;
    for (uint g = lane; g < n_groups; g += 32u) {
        const float gs = ink2_ue8m0_to_f32(uint(s_row[g]));
        device const uchar4 *wp4 = (device const uchar4 *)(w_row + g * 32u);
        device const float4 *xp4 = (device const float4 *)(x + g * 32u);
        float part = 0.0f;
        for (uint i = 0u; i < 8u; ++i) {
            const uchar4 wv = wp4[i];
            const float4 xv = xp4[i];
            part += e4m3_lut[wv.x] * xv.x +
                    e4m3_lut[wv.y] * xv.y +
                    e4m3_lut[wv.z] * xv.z +
                    e4m3_lut[wv.w] * xv.w;
        }
        acc += gs * part;
    }
    return simd_sum(acc);
}

kernel void kernel_inkling2_moe_mxfp8_w13_swiglu(
        constant ink2_args_moe_w13 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const uchar *w13 [[buffer(2)]],
        device const uchar *w13_scale [[buffer(3)]],
        device const uint *expert_ids [[buffer(4)]],
        device const float *expert_weights [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, tid);

    const uint j = group.x * INK2_MV_SG + sg;
    const uint col = group.y;
    const uint total_cols = args.rows * args.top_k;
    if (j >= args.inter_dim || col >= total_cols) {
        return;
    }
    const uint row = col / args.top_k;
    const uint expert = expert_ids[col];
    const float rw = expert_weights[col];

    const uint in_dim = args.in_dim;
    const uint two_inter = 2u * args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim);
    const ulong s_row_bytes = ulong(in_dim / 32u);
    const ulong e_w_base = ulong(expert) * two_inter * w_row_bytes;
    const ulong e_s_base = ulong(expert) * two_inter * s_row_bytes;

    device const float *xr = x + ulong(row) * in_dim;
    const uint g_row = 2u * j;
    const uint u_row = 2u * j + 1u;

    const float g_val = ink2_mxfp8_row_dot_lut(
        w13 + e_w_base + ulong(g_row) * w_row_bytes,
        w13_scale + e_s_base + ulong(g_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane);
    const float u_val = ink2_mxfp8_row_dot_lut(
        w13 + e_w_base + ulong(u_row) * w_row_bytes,
        w13_scale + e_s_base + ulong(u_row) * s_row_bytes,
        xr, e4m3_lut, in_dim, lane);

    if (lane == 0u) {
        const float act = ink2_softcap(g_val, args.gate_cap) *
                          ink2_sigmoid(g_val) *
                          ink2_softcap(u_val, args.up_cap) * rw;
        out[ulong(col) * args.inter_dim + j] = act;
    }
}

// ---- 15. MXFP8 MoE W2 + Expert Accumulation --------------------------------
// Grid: (ceil(out_dim / 4), rows, 1), Block: (128, 1, 1)

kernel void kernel_inkling2_moe_mxfp8_w2_accum(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const uint *expert_ids [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float e4m3_lut[256];
    ink2_fill_e4m3_lut(e4m3_lut, tid);

    const uint o = group.x * INK2_MV_SG + sg;
    const uint row = group.y;
    if (o >= args.out_dim || row >= args.rows) {
        return;
    }
    const uint inter_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(inter_dim);
    const ulong s_row_bytes = ulong(inter_dim / 32u);

    float total = 0.0f;
    for (uint slot = 0u; slot < args.top_k; ++slot) {
        const uint col = row * args.top_k + slot;
        const uint expert = expert_ids[col];
        const ulong e_w_base = (ulong(expert) * args.out_dim + o) * w_row_bytes;
        const ulong e_s_base = (ulong(expert) * args.out_dim + o) * s_row_bytes;
        device const float *ar = act + ulong(col) * inter_dim;
        total += ink2_mxfp8_row_dot_lut(w2 + e_w_base, w2_scale + e_s_base, ar, e4m3_lut, inter_dim, lane);
    }
    if (lane == 0u) {
        out[ulong(row) * args.out_dim + o] = total;
    }
}

// ---- 15b. Token-Grouped 8-Row Batched MoE Prefill Kernels ------------------
// Groups routed token-expert assignments by expert on GPU, then streams each
// expert's NVFP4/MXFP8 weights ONCE per 8 routed tokens (~8x prefill speedup).

struct ink2_args_moe_bucket {
    uint rows;
    uint top_k;
    uint n_experts;
};
static_assert(sizeof(ink2_args_moe_bucket) == 12, "ink2_args_moe_bucket layout");

kernel void kernel_inkling2_moe_bucket(
        constant ink2_args_moe_bucket &args [[buffer(0)]],
        device const uint *expert_ids [[buffer(1)]],
        device uint *expert_counts [[buffer(2)]],
        device uint *expert_cols [[buffer(3)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint e = gid;
    if (e >= args.n_experts) {
        return;
    }
    const uint total = args.rows * args.top_k;
    device uint *dst = expert_cols + ulong(e) * args.rows;
    threadgroup atomic_uint cnt;
    if (tid == 0u) {
        atomic_store_explicit(&cnt, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint col = tid; col < total; col += 256u) {
        if (expert_ids[col] == e) {
            const uint slot = atomic_fetch_add_explicit(&cnt, 1u, memory_order_relaxed);
            if (slot < args.rows) {
                dst[slot] = col;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        expert_counts[e] = atomic_load_explicit(&cnt, memory_order_relaxed);
    }
}

kernel void kernel_inkling2_moe_nvfp4_w13_grouped(
        constant ink2_args_moe_w13 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const uchar *w13 [[buffer(2)]],
        device const uchar *w13_scale [[buffer(3)]],
        device const float *w13_amax [[buffer(4)]],
        device const float *w13_denom [[buffer(5)]],
        device const float *expert_weights [[buffer(6)]],
        device const uint *expert_counts [[buffer(7)]],
        device const uint *expert_cols [[buffer(8)]],
        device float *out [[buffer(9)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint j = group.x * INK2_MV_SG + sg;
    const uint expert = group.y;
    if (j >= args.inter_dim) {
        return;
    }
    const uint cnt = expert_counts[expert];
    if (cnt == 0u) {
        return;
    }
    const float outer = w13_amax[expert] / w13_denom[0];
    const uint in_dim = args.in_dim;
    const uint two_inter = 2u * args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim / 2u);
    const ulong s_row_bytes = ulong(in_dim / 16u);
    const ulong e_w_base = ulong(expert) * two_inter * w_row_bytes;
    const ulong e_s_base = ulong(expert) * two_inter * s_row_bytes;
    device const uchar *gw_row = w13 + e_w_base + ulong(2u * j) * w_row_bytes;
    device const uchar *gs_row = w13_scale + e_s_base + ulong(2u * j) * s_row_bytes;
    device const uchar *uw_row = w13 + e_w_base + ulong(2u * j + 1u) * w_row_bytes;
    device const uchar *us_row = w13_scale + e_s_base + ulong(2u * j + 1u) * s_row_bytes;
    device const uint *cols = expert_cols + ulong(expert) * args.rows;
    const uint n_groups = in_dim / 16u;

    for (uint base = 0u; base < cnt; base += 8u) {
        const uint n_r = min(8u, cnt - base);
        uint c_ids[8];
        uint r_ids[8];
        for (uint rr = 0u; rr < n_r; ++rr) {
            const uint col = cols[base + rr];
            c_ids[rr] = col;
            r_ids[rr] = col / args.top_k;
        }
        float g_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float u_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

        for (uint g = lane; g < n_groups; g += 32u) {
            const float gsc = ink2_e4m3_to_f32(uint(gs_row[g]));
            const float usc = ink2_e4m3_to_f32(uint(us_row[g]));
            device const uchar4 *gwp = (device const uchar4 *)(gw_row + g * 8u);
            device const uchar4 *uwp = (device const uchar4 *)(uw_row + g * 8u);
            const uchar4 gb0 = gwp[0], gb1 = gwp[1];
            const uchar4 ub0 = uwp[0], ub1 = uwp[1];

            const float4 gw0 = float4(INK2_E2M1_LUT[gb0.x & 0xFu], INK2_E2M1_LUT[gb0.x >> 4u],
                                      INK2_E2M1_LUT[gb0.y & 0xFu], INK2_E2M1_LUT[gb0.y >> 4u]) * gsc;
            const float4 gw1 = float4(INK2_E2M1_LUT[gb0.z & 0xFu], INK2_E2M1_LUT[gb0.z >> 4u],
                                      INK2_E2M1_LUT[gb0.w & 0xFu], INK2_E2M1_LUT[gb0.w >> 4u]) * gsc;
            const float4 gw2 = float4(INK2_E2M1_LUT[gb1.x & 0xFu], INK2_E2M1_LUT[gb1.x >> 4u],
                                      INK2_E2M1_LUT[gb1.y & 0xFu], INK2_E2M1_LUT[gb1.y >> 4u]) * gsc;
            const float4 gw3 = float4(INK2_E2M1_LUT[gb1.z & 0xFu], INK2_E2M1_LUT[gb1.z >> 4u],
                                      INK2_E2M1_LUT[gb1.w & 0xFu], INK2_E2M1_LUT[gb1.w >> 4u]) * gsc;

            const float4 uw0 = float4(INK2_E2M1_LUT[ub0.x & 0xFu], INK2_E2M1_LUT[ub0.x >> 4u],
                                      INK2_E2M1_LUT[ub0.y & 0xFu], INK2_E2M1_LUT[ub0.y >> 4u]) * usc;
            const float4 uw1 = float4(INK2_E2M1_LUT[ub0.z & 0xFu], INK2_E2M1_LUT[ub0.z >> 4u],
                                      INK2_E2M1_LUT[ub0.w & 0xFu], INK2_E2M1_LUT[ub0.w >> 4u]) * usc;
            const float4 uw2 = float4(INK2_E2M1_LUT[ub1.x & 0xFu], INK2_E2M1_LUT[ub1.x >> 4u],
                                      INK2_E2M1_LUT[ub1.y & 0xFu], INK2_E2M1_LUT[ub1.y >> 4u]) * usc;
            const float4 uw3 = float4(INK2_E2M1_LUT[ub1.z & 0xFu], INK2_E2M1_LUT[ub1.z >> 4u],
                                      INK2_E2M1_LUT[ub1.w & 0xFu], INK2_E2M1_LUT[ub1.w >> 4u]) * usc;

            for (uint rr = 0u; rr < n_r; ++rr) {
                device const float4 *xp = (device const float4 *)(x + ulong(r_ids[rr]) * in_dim + g * 16u);
                const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                g_acc[rr] += dot(gw0, x0) + dot(gw1, x1) + dot(gw2, x2) + dot(gw3, x3);
                u_acc[rr] += dot(uw0, x0) + dot(uw1, x1) + dot(uw2, x2) + dot(uw3, x3);
            }
        }
        for (uint rr = 0u; rr < n_r; ++rr) {
            const float gv = simd_sum(g_acc[rr]) * outer;
            const float uv = simd_sum(u_acc[rr]) * outer;
            if (lane == 0u) {
                const uint col = c_ids[rr];
                const float rw = expert_weights[col];
                out[ulong(col) * args.inter_dim + j] =
                    ink2_softcap(gv, args.gate_cap) * ink2_sigmoid(gv) * ink2_softcap(uv, args.up_cap) * rw;
            }
        }
    }
}

kernel void kernel_inkling2_moe_nvfp4_w2_grouped(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const float *w2_amax [[buffer(4)]],
        device const float *w2_denom [[buffer(5)]],
        device const uint *expert_counts [[buffer(6)]],
        device const uint *expert_cols [[buffer(7)]],
        device float *slots_out [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * INK2_MV_SG + sg;
    const uint expert = group.y;
    if (o >= args.out_dim) {
        return;
    }
    const uint cnt = expert_counts[expert];
    if (cnt == 0u) {
        return;
    }
    const float outer = w2_amax[expert] / w2_denom[0];
    const uint inter_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(inter_dim / 2u);
    const ulong s_row_bytes = ulong(inter_dim / 16u);
    device const uchar *w_row = w2 + (ulong(expert) * args.out_dim + o) * w_row_bytes;
    device const uchar *s_row = w2_scale + (ulong(expert) * args.out_dim + o) * s_row_bytes;
    device const uint *cols = expert_cols + ulong(expert) * args.rows;
    const uint n_groups = inter_dim / 16u;

    for (uint base = 0u; base < cnt; base += 8u) {
        const uint n_r = min(8u, cnt - base);
        uint c_ids[8];
        for (uint rr = 0u; rr < n_r; ++rr) {
            c_ids[rr] = cols[base + rr];
        }
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = lane; g < n_groups; g += 32u) {
            const float sc = ink2_e4m3_to_f32(uint(s_row[g]));
            device const uchar4 *wp = (device const uchar4 *)(w_row + g * 8u);
            const uchar4 b0 = wp[0], b1 = wp[1];
            const float4 w0 = float4(INK2_E2M1_LUT[b0.x & 0xFu], INK2_E2M1_LUT[b0.x >> 4u],
                                     INK2_E2M1_LUT[b0.y & 0xFu], INK2_E2M1_LUT[b0.y >> 4u]) * sc;
            const float4 w1 = float4(INK2_E2M1_LUT[b0.z & 0xFu], INK2_E2M1_LUT[b0.z >> 4u],
                                     INK2_E2M1_LUT[b0.w & 0xFu], INK2_E2M1_LUT[b0.w >> 4u]) * sc;
            const float4 w2v = float4(INK2_E2M1_LUT[b1.x & 0xFu], INK2_E2M1_LUT[b1.x >> 4u],
                                      INK2_E2M1_LUT[b1.y & 0xFu], INK2_E2M1_LUT[b1.y >> 4u]) * sc;
            const float4 w3 = float4(INK2_E2M1_LUT[b1.z & 0xFu], INK2_E2M1_LUT[b1.z >> 4u],
                                     INK2_E2M1_LUT[b1.w & 0xFu], INK2_E2M1_LUT[b1.w >> 4u]) * sc;
            for (uint rr = 0u; rr < n_r; ++rr) {
                device const float4 *ap = (device const float4 *)(act + ulong(c_ids[rr]) * inter_dim + g * 16u);
                acc[rr] += dot(w0, ap[0]) + dot(w1, ap[1]) + dot(w2v, ap[2]) + dot(w3, ap[3]);
            }
        }
        for (uint rr = 0u; rr < n_r; ++rr) {
            const float v = simd_sum(acc[rr]) * outer;
            if (lane == 0u) {
                slots_out[ulong(c_ids[rr]) * args.out_dim + o] = v;
            }
        }
    }
}

kernel void kernel_inkling2_moe_mxfp8_w13_grouped(
        constant ink2_args_moe_w13 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const uchar *w13 [[buffer(2)]],
        device const uchar *w13_scale [[buffer(3)]],
        device const float *expert_weights [[buffer(4)]],
        device const uint *expert_counts [[buffer(5)]],
        device const uint *expert_cols [[buffer(6)]],
        device float *out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint j = group.x * INK2_MV_SG + sg;
    const uint expert = group.y;
    if (j >= args.inter_dim) {
        return;
    }
    const uint cnt = expert_counts[expert];
    if (cnt == 0u) {
        return;
    }
    const uint in_dim = args.in_dim;
    const uint two_inter = 2u * args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim);
    const ulong s_row_bytes = ulong(in_dim / 32u);
    const ulong e_w_base = ulong(expert) * two_inter * w_row_bytes;
    const ulong e_s_base = ulong(expert) * two_inter * s_row_bytes;
    device const uchar *gw_row = w13 + e_w_base + ulong(2u * j) * w_row_bytes;
    device const uchar *gs_row = w13_scale + e_s_base + ulong(2u * j) * s_row_bytes;
    device const uchar *uw_row = w13 + e_w_base + ulong(2u * j + 1u) * w_row_bytes;
    device const uchar *us_row = w13_scale + e_s_base + ulong(2u * j + 1u) * s_row_bytes;
    device const uint *cols = expert_cols + ulong(expert) * args.rows;
    const uint n_groups = in_dim / 32u;

    for (uint base = 0u; base < cnt; base += 8u) {
        const uint n_r = min(8u, cnt - base);
        uint c_ids[8];
        uint r_ids[8];
        for (uint rr = 0u; rr < n_r; ++rr) {
            const uint col = cols[base + rr];
            c_ids[rr] = col;
            r_ids[rr] = col / args.top_k;
        }
        float g_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        float u_acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = lane; g < n_groups; g += 32u) {
            const float gsc = ink2_ue8m0_to_f32(uint(gs_row[g]));
            const float usc = ink2_ue8m0_to_f32(uint(us_row[g]));
            device const uchar4 *gwp = (device const uchar4 *)(gw_row + g * 32u);
            device const uchar4 *uwp = (device const uchar4 *)(uw_row + g * 32u);
            for (uint i = 0u; i < 8u; ++i) {
                const uchar4 gv4 = gwp[i];
                const uchar4 uv4 = uwp[i];
                const float4 gw = float4(ink2_e4m3_to_f32(gv4.x), ink2_e4m3_to_f32(gv4.y),
                                         ink2_e4m3_to_f32(gv4.z), ink2_e4m3_to_f32(gv4.w)) * gsc;
                const float4 uw = float4(ink2_e4m3_to_f32(uv4.x), ink2_e4m3_to_f32(uv4.y),
                                         ink2_e4m3_to_f32(uv4.z), ink2_e4m3_to_f32(uv4.w)) * usc;
                for (uint rr = 0u; rr < n_r; ++rr) {
                    const float4 xv = *((device const float4 *)(x + ulong(r_ids[rr]) * in_dim + g * 32u + i * 4u));
                    g_acc[rr] += dot(gw, xv);
                    u_acc[rr] += dot(uw, xv);
                }
            }
        }
        for (uint rr = 0u; rr < n_r; ++rr) {
            const float gv = simd_sum(g_acc[rr]);
            const float uv = simd_sum(u_acc[rr]);
            if (lane == 0u) {
                const uint col = c_ids[rr];
                const float rw = expert_weights[col];
                out[ulong(col) * args.inter_dim + j] =
                    ink2_softcap(gv, args.gate_cap) * ink2_sigmoid(gv) * ink2_softcap(uv, args.up_cap) * rw;
            }
        }
    }
}

kernel void kernel_inkling2_moe_mxfp8_w2_grouped(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const uint *expert_counts [[buffer(4)]],
        device const uint *expert_cols [[buffer(5)]],
        device float *slots_out [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * INK2_MV_SG + sg;
    const uint expert = group.y;
    if (o >= args.out_dim) {
        return;
    }
    const uint cnt = expert_counts[expert];
    if (cnt == 0u) {
        return;
    }
    const uint inter_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(inter_dim);
    const ulong s_row_bytes = ulong(inter_dim / 32u);
    device const uchar *w_row = w2 + (ulong(expert) * args.out_dim + o) * w_row_bytes;
    device const uchar *s_row = w2_scale + (ulong(expert) * args.out_dim + o) * s_row_bytes;
    device const uint *cols = expert_cols + ulong(expert) * args.rows;
    const uint n_groups = inter_dim / 32u;

    for (uint base = 0u; base < cnt; base += 8u) {
        const uint n_r = min(8u, cnt - base);
        uint c_ids[8];
        for (uint rr = 0u; rr < n_r; ++rr) {
            c_ids[rr] = cols[base + rr];
        }
        float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (uint g = lane; g < n_groups; g += 32u) {
            const float sc = ink2_ue8m0_to_f32(uint(s_row[g]));
            device const uchar4 *wp = (device const uchar4 *)(w_row + g * 32u);
            for (uint i = 0u; i < 8u; ++i) {
                const uchar4 wv = wp[i];
                const float4 fw = float4(ink2_e4m3_to_f32(wv.x), ink2_e4m3_to_f32(wv.y),
                                         ink2_e4m3_to_f32(wv.z), ink2_e4m3_to_f32(wv.w)) * sc;
                for (uint rr = 0u; rr < n_r; ++rr) {
                    const float4 av = *((device const float4 *)(act + ulong(c_ids[rr]) * inter_dim + g * 32u + i * 4u));
                    acc[rr] += dot(fw, av);
                }
            }
        }
        for (uint rr = 0u; rr < n_r; ++rr) {
            const float v = simd_sum(acc[rr]);
            if (lane == 0u) {
                slots_out[ulong(c_ids[rr]) * args.out_dim + o] = v;
            }
        }
    }
}

kernel void kernel_inkling2_moe_reduce_topk(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *slots_in [[buffer(1)]],
        device float *out [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.rows * args.out_dim;
    if (tid >= total) {
        return;
    }
    const uint row = tid / args.out_dim;
    const uint o = tid % args.out_dim;
    float acc = 0.0f;
    for (uint s = 0u; s < args.top_k; ++s) {
        acc += slots_in[ulong(row * args.top_k + s) * args.out_dim + o];
    }
    out[tid] = acc;
}

// ---- 16. Residual Add (out = a + b, or out = a + b + c) --------------------

kernel void kernel_inkling2_add2_f32(
        constant ink2_args_add3 &args [[buffer(0)]],
        device const float *a [[buffer(1)]],
        device const float *b [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid < args.count) {
        out[tid] = a[tid] + b[tid];
    }
}

kernel void kernel_inkling2_add3_f32(
        constant ink2_args_add3 &args [[buffer(0)]],
        device const float *a [[buffer(1)]],
        device const float *b [[buffer(2)]],
        device const float *c [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid < args.count) {
        out[tid] = a[tid] + b[tid] + c[tid];
    }
}

// ---- 17. Vision Encoder Kernels --------------------------------------------

struct ink2_args_vit_norm {
    uint rows;
    uint hidden;
    float eps;
};
static_assert(sizeof(ink2_args_vit_norm) == 12, "ink2_args_vit_norm layout");

struct ink2_args_vit_rope {
    uint rows;
    uint heads;
    uint head_dim;
    float theta;
};
static_assert(sizeof(ink2_args_vit_rope) == 16, "ink2_args_vit_rope layout");

struct ink2_args_vit_attn {
    uint rows;
    uint heads;
    uint head_dim;
    float scale;
};
static_assert(sizeof(ink2_args_vit_attn) == 16, "ink2_args_vit_attn layout");

struct ink2_args_vit_ls {
    uint rows;
    uint hidden;
};
static_assert(sizeof(ink2_args_vit_ls) == 8, "ink2_args_vit_ls layout");

kernel void kernel_inkling2_vit_layernorm_bf16(
        constant ink2_args_vit_norm &args [[buffer(0)]],
        device const float *inp [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device const ushort *b [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (gid >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    device const float *src = inp + ulong(gid) * H;
    device float *dst = out + ulong(gid) * H;

    float sum = 0.0f;
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        sum += src[i];
    }
    sum = simd_sum(sum);
    threadgroup float p[8];
    if (lane == 0u) {
        p[sg] = sum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; ++s) tot += p[s];
        p[0] = tot / float(H);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mean = p[0];

    float sq = 0.0f;
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        const float d = src[i] - mean;
        sq += d * d;
    }
    sq = simd_sum(sq);
    if (lane == 0u) {
        p[sg] = sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; ++s) tot += p[s];
        p[0] = rsqrt(tot / float(H) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = p[0];
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = (src[i] - mean) * inv * ink2_bf16_to_f32(w[i]) + ink2_bf16_to_f32(b[i]);
    }
}

kernel void kernel_inkling2_vit_linear_bias_bf16(
        constant ink2_args_linear &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device const ushort *b [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * INK2_MV_SG + sg;
    const uint r = group.y;
    if (o >= args.out_dim || r >= args.rows) {
        return;
    }
    const uint in_dim = args.in_dim;
    device const float *xr = x + ulong(r) * in_dim;
    device const ushort *wr = w + ulong(o) * in_dim;
    float acc = 0.0f;
    for (uint k = lane * 4u; k + 3u < in_dim; k += 128u) {
        const float4 xv = *((device const float4 *)(xr + k));
        const ushort4 wv = *((device const ushort4 *)(wr + k));
        acc += xv.x * ink2_bf16_to_f32(wv.x) +
               xv.y * ink2_bf16_to_f32(wv.y) +
               xv.z * ink2_bf16_to_f32(wv.z) +
               xv.w * ink2_bf16_to_f32(wv.w);
    }
    acc = simd_sum(acc);
    if (lane == 0u) {
        out[ulong(r) * args.out_dim + o] = acc + ink2_bf16_to_f32(b[o]);
    }
}

kernel void kernel_inkling2_vit_axial_rope(
        constant ink2_args_vit_rope &args [[buffer(0)]],
        device const uint *pos_ids [[buffer(1)]],
        device float *q [[buffer(2)]],
        device float *k [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint h = group.x;
    const uint row = group.y;
    const uint half_d = args.head_dim / 2u;
    if (h >= args.heads || row >= args.rows || tid >= half_d) {
        return;
    }
    const uint py = pos_ids[row * 2u];
    const uint px = pos_ids[row * 2u + 1u];
    const uint quarter = half_d / 2u;
    const bool is_w = tid < quarter;
    const uint f_idx = is_w ? tid : (tid - quarter);
    const float coord = float(is_w ? px : py);
    const float inv_freq = 1.0f / pow(args.theta, float(f_idx * 4u) / float(args.head_dim));
    const float angle = coord * inv_freq;
    const float c = cos(angle);
    const float s = sin(angle);
    const ulong base = (ulong(row) * args.heads + h) * args.head_dim + tid * 2u;
    const float q0 = q[base];
    const float q1 = q[base + 1u];
    q[base] = q0 * c - q1 * s;
    q[base + 1u] = q0 * s + q1 * c;
    const float k0 = k[base];
    const float k1 = k[base + 1u];
    k[base] = k0 * c - k1 * s;
    k[base + 1u] = k0 * s + k1 * c;
}

kernel void kernel_inkling2_vit_attn(
        constant ink2_args_vit_attn &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k [[buffer(2)]],
        device const float *v [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint h = group.x;
    const uint row = group.y;
    const uint D = args.head_dim;
    if (h >= args.heads || row >= args.rows || tid >= D) {
        return;
    }
    const float qv = q[(ulong(row) * args.heads + h) * D + tid];
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;
    threadgroup float p[4];
    const uint n_sg = (D + 31u) / 32u;

    for (uint kp = 0u; kp < args.rows; ++kp) {
        const ulong off = (ulong(kp) * args.heads + h) * D;
        float dot = simd_sum(qv * k[off + tid]);
        if (lane == 0u) {
            p[sg] = dot;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float tot = 0.0f;
            for (uint s = 0u; s < n_sg; ++s) tot += p[s];
            p[0] = tot * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float score = p[0];
        const float nm = max(m_max, score);
        const float eo = exp(m_max - nm);
        const float en = exp(score - nm);
        l_sum = l_sum * eo + en;
        acc = acc * eo + en * v[off + tid];
        m_max = nm;
    }
    out[(ulong(row) * args.heads + h) * D + tid] = acc / l_sum;
}

kernel void kernel_inkling2_vit_gelu_erf(
        constant ink2_args_add3 &args [[buffer(0)]],
        device float *x [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid < args.count) {
        const float v = x[tid];
        // Abramowitz-Stegun erf approximation (max error < 1.5e-7)
        const float z = abs(v) * 0.7071067811865475f;
        const float t = 1.0f / (1.0f + 0.3275911f * z);
        const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f + t * (-1.453152027f + t * 1.061405429f))));
        const float erf_z = 1.0f - poly * exp(-z * z);
        const float erf_v = (v >= 0.0f) ? erf_z : -erf_z;
        x[tid] = 0.5f * v * (1.0f + erf_v);
    }
}

kernel void kernel_inkling2_vit_layerscale_add(
        constant ink2_args_vit_ls &args [[buffer(0)]],
        device const float *residual [[buffer(1)]],
        device const float *delta [[buffer(2)]],
        device const ushort *gamma [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint tid [[thread_position_in_grid]]) {
    const uint total = args.rows * args.hidden;
    if (tid < total) {
        const uint col = tid % args.hidden;
        out[tid] = residual[tid] + ink2_bf16_to_f32(gamma[col]) * delta[tid];
    }
}

// ---- 18. MTP / DSpark Speculator Kernels -----------------------------------

struct ink2_args_mtp_mix {
    uint rows;
    uint stride_rows;
    uint hidden;
    uint n_taps;
    float eps;
};
static_assert(sizeof(ink2_args_mtp_mix) == 20, "ink2_args_mtp_mix layout");

struct ink2_args_mtp_qk {
    uint rows;
    uint n_head;
    uint head_dim;
    uint pos0;
    float rope_theta;
    float eps;
    uint kv_cap;
    uint is_swa;
    uint write_cache;
};
static_assert(sizeof(ink2_args_mtp_qk) == 36, "ink2_args_mtp_qk layout");

struct ink2_args_mtp_attn {
    uint block_size;
    uint n_head;
    uint n_head_kv;
    uint head_dim;
    uint ctx_len;
    uint kv_cap;
    uint window_size;
    uint is_swa;
    float scale;
};
static_assert(sizeof(ink2_args_mtp_attn) == 36, "ink2_args_mtp_attn layout");

struct ink2_args_mtp_rnn {
    uint rows;
    uint rank;
};
static_assert(sizeof(ink2_args_mtp_rnn) == 8, "ink2_args_mtp_rnn layout");

kernel void kernel_inkling2_mtp_mix_rmsnorm(
        constant ink2_args_mtp_mix &args [[buffer(0)]],
        device const float *taps [[buffer(1)]],
        device const float *alpha [[buffer(2)]],
        device const ushort *norm_w [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    if (gid >= args.rows) {
        return;
    }
    const uint H = args.hidden;
    const ulong tap_stride = ulong(args.stride_rows) * H;
    const ulong row_off = ulong(gid) * H;
    device float *dst = out + row_off;

    float sq = 0.0f;
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        float mixed = 0.0f;
        for (uint t = 0u; t < args.n_taps; ++t) {
            mixed += alpha[t] * taps[ulong(t) * tap_stride + row_off + i];
        }
        dst[i] = mixed;
        sq += mixed * mixed;
    }
    sq = simd_sum(sq);
    threadgroup float p[8];
    if (lane == 0u) p[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        float tot = 0.0f;
        for (uint s = 0u; s < 8u; ++s) tot += p[s];
        p[0] = rsqrt(tot / float(H) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = p[0];
    for (uint i = tid; i < H; i += INK2_BLOCK) {
        dst[i] = dst[i] * inv * ink2_bf16_to_f32(norm_w[i]);
    }
}

kernel void kernel_inkling2_mtp_qk_norm_rope(
        constant ink2_args_mtp_qk &args [[buffer(0)]],
        device float *x [[buffer(1)]],
        device const ushort *norm_w [[buffer(2)]],
        device half *kv_cache [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint h = group.x;
    const uint row = group.y;
    const uint D = args.head_dim;
    if (h >= args.n_head || row >= args.rows || tid >= D) {
        return;
    }
    device float *vec = x + (ulong(row) * args.n_head + h) * D;
    const float v0 = vec[tid];
    float sq = simd_sum(v0 * v0);
    threadgroup float p[4];
    if (lane == 0u) p[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        const uint n_sg = (D + 31u) / 32u;
        float tot = 0.0f;
        for (uint s = 0u; s < n_sg; ++s) tot += p[s];
        p[0] = rsqrt(tot / float(D) + args.eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float normed[128];
    normed[tid] = v0 * p[0] * ink2_bf16_to_f32(norm_w[tid]);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint half_d = D / 2u;
    const uint pos = args.pos0 + row;
    float rotated;
    if (tid < half_d) {
        const float freq = 1.0f / pow(args.rope_theta, float(tid * 2u) / float(D));
        const float ang = float(pos) * freq;
        rotated = normed[tid] * cos(ang) - normed[tid + half_d] * sin(ang);
    } else {
        const uint pair = tid - half_d;
        const float freq = 1.0f / pow(args.rope_theta, float(pair * 2u) / float(D));
        const float ang = float(pos) * freq;
        rotated = normed[pair] * sin(ang) + normed[tid] * cos(ang);
    }
    vec[tid] = rotated;
    if (args.write_cache != 0u) {
        const uint slot = args.is_swa != 0u ? (pos % args.kv_cap) : pos;
        kv_cache[(ulong(slot) * args.n_head + h) * D + tid] = half(rotated);
    }
}

kernel void kernel_inkling2_mtp_block_attn(
        constant ink2_args_mtp_attn &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *k_blk [[buffer(2)]],
        device const float *v_blk [[buffer(3)]],
        device const half *k_ctx [[buffer(4)]],
        device const half *v_ctx [[buffer(5)]],
        device float *out [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint h = group.x;
    const uint row = group.y;
    const uint D = args.head_dim;
    if (h >= args.n_head || row >= args.block_size || tid >= D) {
        return;
    }
    const uint kvh = h / (args.n_head / args.n_head_kv);
    const float qv = q[(ulong(row) * args.n_head + h) * D + tid];
    const uint ctx_start = (args.is_swa != 0u && args.ctx_len > args.window_size)
                               ? (args.ctx_len - args.window_size)
                               : 0u;

    float m_max = -1e30f;
    float l_sum = 0.0f;
    float acc = 0.0f;
    threadgroup float p[4];

    // Attend over committed context KV
    for (uint kp = ctx_start; kp < args.ctx_len; ++kp) {
        const uint slot = args.is_swa != 0u ? (kp % args.kv_cap) : kp;
        const ulong off = (ulong(slot) * args.n_head_kv + kvh) * D;
        float dot = simd_sum(qv * float(k_ctx[off + tid]));
        if (lane == 0u) p[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            p[0] = (p[0] + p[1] + p[2] + p[3]) * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sc = p[0];
        const float nm = max(m_max, sc);
        const float eo = exp(m_max - nm);
        const float en = exp(sc - nm);
        l_sum = l_sum * eo + en;
        acc = acc * eo + en * float(v_ctx[off + tid]);
        m_max = nm;
    }

    // Attend non-causally over all positions in the current draft block
    for (uint bp = 0u; bp < args.block_size; ++bp) {
        const ulong off = (ulong(bp) * args.n_head_kv + kvh) * D;
        float dot = simd_sum(qv * k_blk[off + tid]);
        if (lane == 0u) p[sg] = dot;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            p[0] = (p[0] + p[1] + p[2] + p[3]) * args.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sc = p[0];
        const float nm = max(m_max, sc);
        const float eo = exp(m_max - nm);
        const float en = exp(sc - nm);
        l_sum = l_sum * eo + en;
        acc = acc * eo + en * v_blk[off + tid];
        m_max = nm;
    }

    out[(ulong(row) * args.n_head + h) * D + tid] = acc / l_sum;
}

kernel void kernel_inkling2_mtp_swiglu_mul(
        constant ink2_args_add3 &args [[buffer(0)]],
        device const float *gate [[buffer(1)]],
        device const float *up [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid < args.count) {
        const float g = gate[tid];
        out[tid] = (g * ink2_sigmoid(g)) * up[tid];
    }
}

kernel void kernel_inkling2_mtp_rnn_step(
        constant ink2_args_mtp_rnn &args [[buffer(0)]],
        device const float *proj [[buffer(1)]],
        device float *state [[buffer(2)]],
        device float *latent [[buffer(3)]],
        uint2 pos [[thread_position_in_grid]]) {
    const uint c = pos.x;
    const uint r = pos.y;
    if (r >= args.rows || c >= args.rank) {
        return;
    }
    const uint R = args.rank;
    device const float *pr = proj + ulong(r) * (3u * R);
    const float g = ink2_sigmoid(pr[c]);
    const float cand = tanh(pr[R + c]);
    const float out_v = tanh(pr[2u * R + c]);
    const ulong idx = ulong(r) * R + c;
    state[idx] = g * state[idx] + (1.0f - g) * cand;
    latent[idx] = out_v;
}
