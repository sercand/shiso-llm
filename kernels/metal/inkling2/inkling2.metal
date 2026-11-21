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
#define INK2_UNROLL _Pragma("clang loop unroll(full)")

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

// Lane reduction in a fixed order (a bare simd_sum has none), compiled safe so
// the optimizer cannot re-fold it; the default is restored right after.
#pragma METAL fp math_mode(safe)
static inline float ink2_lane_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}
#pragma METAL fp math_mode(fast)

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
    uint defer_state;
};
static_assert(sizeof(ink2_args_sconv_static) == 20, "ink2_args_sconv_static layout");

struct ink2_args_sconv_dynamic {
    uint rows;
    uint kv_heads;
    uint head_dim;
    uint kernel_size;
    float scale;
    uint defer_state;
};
static_assert(sizeof(ink2_args_sconv_dynamic) == 24, "ink2_args_sconv_dynamic layout");

struct ink2_args_sconv_commit {
    uint total;
    uint kv_dim;
    uint hidden;
    uint lags;
    uint hist_rows;
    uint keep;
};
static_assert(sizeof(ink2_args_sconv_commit) == 24, "ink2_args_sconv_commit layout");

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

#pragma METAL fp math_mode(safe)
static inline float ink2_bf16_dot4_acc(float acc, float4 x, ushort4 w) {
    const float part = fma(x.x, ink2_bf16_to_f32(w.x),
                       fma(x.y, ink2_bf16_to_f32(w.y),
                       fma(x.z, ink2_bf16_to_f32(w.z), x.w * ink2_bf16_to_f32(w.w))));
    return acc + part;
}
static inline float ink2_bf16_tail_acc(float acc, float x, ushort w) {
    return acc + x * ink2_bf16_to_f32(w);
}
static inline float ink2_softcap_safe(float x, float cap) {
    return cap > 0.0f ? cap * tanh(x / cap) : x;
}
#pragma METAL fp math_mode(fast)

// One BF16 weight row `W` against rows `r0 .. r0 + n_r` of `X` (`in_dim`
// wide), written to `DST[row * STRIDE + COL]`. Reads r0, n_r, in_dim, lane.
// The one-row branch and the eight-row branch take the same helper sums in
// the same order, so a row's value does not depend on how many share the
// weight read.
#define INK2_BF16_ROWS(X, W, DST, STRIDE, COL)                                              \
    if (n_r == 1u) {                                                                       \
        device const float *xr = (X) + ulong(r0) * in_dim;                                 \
        float acc = 0.0f;                                                                  \
        for (uint k = lane * 4u; k + 3u < in_dim; k += 128u) {                             \
            acc = ink2_bf16_dot4_acc(acc, *((device const float4 *)(xr + k)),              \
                                     *((device const ushort4 *)((W) + k)));                \
        }                                                                                  \
        const uint tail_idx = (in_dim & ~3u) + lane;                                       \
        if (tail_idx < in_dim) {                                                           \
            acc = ink2_bf16_tail_acc(acc, xr[tail_idx], (W)[tail_idx]);                    \
        }                                                                                  \
        acc = ink2_lane_sum(acc);                                                          \
        if (lane == 0u) {                                                                  \
            (DST)[ulong(r0) * (STRIDE) + (COL)] = acc;                                     \
        }                                                                                  \
        return;                                                                            \
    }                                                                                      \
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};                       \
    for (uint k = lane * 4u; k + 3u < in_dim; k += 128u) {                                 \
        const ushort4 wv = *((device const ushort4 *)((W) + k));                           \
        INK2_UNROLL for (uint rr = 0u; rr < 8u; ++rr) {                                    \
            if (rr < n_r) {                                                                \
                acc[rr] = ink2_bf16_dot4_acc(                                              \
                    acc[rr], *((device const float4 *)((X) + ulong(r0 + rr) * in_dim + k)), wv); \
            }                                                                              \
        }                                                                                  \
    }                                                                                      \
    const uint tail_idx = (in_dim & ~3u) + lane;                                           \
    if (tail_idx < in_dim) {                                                               \
        for (uint rr = 0u; rr < n_r; ++rr) {                                               \
            acc[rr] = ink2_bf16_tail_acc(acc[rr], (X)[ulong(r0 + rr) * in_dim + tail_idx], \
                                         (W)[tail_idx]);                                   \
        }                                                                                  \
    }                                                                                      \
    for (uint rr = 0u; rr < n_r; ++rr) {                                                   \
        const float sum = ink2_lane_sum(acc[rr]);                                          \
        if (lane == 0u) {                                                                  \
            (DST)[ulong(r0 + rr) * (STRIDE) + (COL)] = sum;                                \
        }                                                                                  \
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

    INK2_BF16_ROWS(x, wr, out, args.out_dim, o)
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

    INK2_BF16_ROWS(x_base, wr_ptr, dst, out_stride, local_o)
}

// ---- 4. LM head BF16 + softcap ---------------------------------------------
// Grid: (ceil(vocab / 4), ceil(rows / 8), 1), block 128.
// A simdgroup owns one vocabulary row against up to eight hidden rows, so a
// pass's rows share one read of the head (922 MB here) instead of each making
// its own: a speculative verify pass of eight rows spent 20 ms in eight reads
// on an M1 Max. The sums are the safe-math helpers, so a row's logits are the
// same bits alone or among eight - a verify row has to pick what a decode row
// picks. `hidden % 4 == 0`.

kernel void kernel_inkling2_lm_head_softcap(
        constant ink2_args_lm_head &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const ushort *w [[buffer(2)]],
        device float *out [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * INK2_MV_SG + sg;
    const uint r0 = group.y * 8u;
    if (o >= args.vocab || r0 >= args.rows) {
        return;
    }
    const uint n_r = min(8u, args.rows - r0);
    const uint H = args.hidden;
    device const ushort *wr = w + ulong(o) * H;

    if (n_r == 1u) {
        device const float *xr = x + ulong(r0) * H;
        float acc = 0.0f;
        for (uint k = lane * 4u; k + 3u < H; k += 128u) {
            acc = ink2_bf16_dot4_acc(acc, *((device const float4 *)(xr + k)),
                                     *((device const ushort4 *)(wr + k)));
        }
        acc = ink2_lane_sum(acc);
        if (lane == 0u) {
            out[ulong(r0) * args.vocab + o] = ink2_softcap_safe(acc, args.softcap);
        }
        return;
    }

    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint k = lane * 4u; k + 3u < H; k += 128u) {
        const ushort4 wv = *((device const ushort4 *)(wr + k));
        INK2_UNROLL for (uint rr = 0u; rr < 8u; ++rr) {
            if (rr < n_r) {
                acc[rr] = ink2_bf16_dot4_acc(
                    acc[rr], *((device const float4 *)(x + ulong(r0 + rr) * H + k)), wv);
            }
        }
    }
    for (uint rr = 0u; rr < n_r; ++rr) {
        const float sum = ink2_lane_sum(acc[rr]);
        if (lane == 0u) {
            out[ulong(r0 + rr) * args.vocab + o] = ink2_softcap_safe(sum, args.softcap);
        }
    }
}

// ---- 5. Block Attention Residual (attn_res) --------------------------------
// out[q] = sum_s softmax_s(dot(query[q], v_s / rms(v_s)) / sqrt(H)) * v_s / rms(v_s)
// over the completed blocks and the current partial stream.
//
// Two launches. `attn_res_stats` reads every source once, a threadgroup per
// (source, slice of the width): a thread carries the sum of squares and the
// dot with every query together, and the threadgroup writes its five sums.
// `attn_res_mix` adds a source's slices in slice order, takes the softmax and
// mixes, a threadgroup per 256 outputs.
//
// One threadgroup a row that did all of it took 47 us a decode row on an M1
// Max, and 55 us with its barrier pairs cut from two per sum to two in all:
// the cost was one GPU core walking every source three times, not the
// barriers.
// stats: grid (n_src * INK2_ATTN_RES_SLICES, rows, 1), block 256
//        -> stats[row][source][slice][1 + INK2_MAX_ATTN_RES_Q]
// mix:   grid (ceil(hidden / 256), rows, 1), block 256

#define INK2_MAX_ATTN_RES_Q 4u
#define INK2_ATTN_RES_SLICES 4u
#define INK2_ATTN_RES_STATS (1u + INK2_MAX_ATTN_RES_Q)

static inline float ink2_attn_res_src(
        device const float *blocks, device const float *partial, device const float *addend,
        uint has_addend, uint s, uint n_blocks, ulong block_stride, ulong idx) {
    if (s < n_blocks) {
        return blocks[ulong(s) * block_stride + idx];
    }
    return partial[idx] + (has_addend != 0u ? addend[idx] : 0.0f);
}

kernel void kernel_inkling2_attn_res_stats(
        constant ink2_args_attn_res &args [[buffer(0)]],
        device const float *blocks [[buffer(1)]],
        device const float *partial [[buffer(2)]],
        device const float *addend [[buffer(3)]],
        device const float *queries [[buffer(4)]],
        device float *stats [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint H = args.hidden;
    const uint n_blocks = min(args.n_blocks, INK2_MAX_ATTN_RES_SRC - 1u);
    const uint n_src = n_blocks + 1u;
    const uint n_q = min(args.n_queries, INK2_MAX_ATTN_RES_Q);
    const uint s = group.x / INK2_ATTN_RES_SLICES;
    const uint slice = group.x % INK2_ATTN_RES_SLICES;
    const uint row = group.y;
    if (s >= n_src || row >= args.rows) {
        return;
    }
    const uint span = (H + INK2_ATTN_RES_SLICES - 1u) / INK2_ATTN_RES_SLICES;
    const uint lo = min(H, slice * span);
    const uint hi = min(H, lo + span);
    const ulong row_off = ulong(row) * H;
    const ulong block_stride = ulong(args.rows) * H;

    float sq = 0.0f;
    float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
    for (uint i = lo + tid; i < hi; i += INK2_BLOCK) {
        const float v = ink2_attn_res_src(blocks, partial, addend, args.has_addend,
                                          s, n_blocks, block_stride, row_off + i);
        sq += v * v;
        d0 += queries[i] * v;
        if (n_q > 1u) d1 += queries[ulong(H) + i] * v;
        if (n_q > 2u) d2 += queries[2ul * H + i] * v;
        if (n_q > 3u) d3 += queries[3ul * H + i] * v;
    }
    threadgroup float parts[8][INK2_ATTN_RES_STATS];
    sq = ink2_lane_sum(sq);
    d0 = ink2_lane_sum(d0);
    d1 = ink2_lane_sum(d1);
    d2 = ink2_lane_sum(d2);
    d3 = ink2_lane_sum(d3);
    if (lane == 0u) {
        parts[sg][0] = sq;
        parts[sg][1] = d0;
        parts[sg][2] = d1;
        parts[sg][3] = d2;
        parts[sg][4] = d3;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < INK2_ATTN_RES_STATS) {
        float tot = 0.0f;
        for (uint k = 0u; k < 8u; ++k) {
            tot += parts[k][tid];
        }
        stats[((ulong(row) * n_src + s) * INK2_ATTN_RES_SLICES + slice) * INK2_ATTN_RES_STATS + tid] = tot;
    }
}

kernel void kernel_inkling2_attn_res_mix(
        constant ink2_args_attn_res &args [[buffer(0)]],
        device const float *blocks [[buffer(1)]],
        device const float *partial [[buffer(2)]],
        device const float *addend [[buffer(3)]],
        device const float *stats [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]]) {
    const uint H = args.hidden;
    const uint n_blocks = min(args.n_blocks, INK2_MAX_ATTN_RES_SRC - 1u);
    const uint n_src = n_blocks + 1u;
    const uint n_q = min(args.n_queries, INK2_MAX_ATTN_RES_Q);
    const uint row = group.y;
    if (row >= args.rows) {
        return;
    }
    const ulong row_off = ulong(row) * H;
    const ulong block_stride = ulong(args.rows) * H;

    // weights[q][s], the source's 1 / rms folded in.
    threadgroup float weights[INK2_MAX_ATTN_RES_Q][INK2_MAX_ATTN_RES_SRC];
    if (tid < n_q) {
        const uint q = tid;
        const float scale = rsqrt(float(H));
        float raw[INK2_MAX_ATTN_RES_SRC];
        float inv[INK2_MAX_ATTN_RES_SRC];
        float mx = -1e30f;
        for (uint s = 0u; s < n_src; ++s) {
            device const float *st =
                stats + (ulong(row) * n_src + s) * INK2_ATTN_RES_SLICES * INK2_ATTN_RES_STATS;
            float sq = 0.0f;
            float d = 0.0f;
            for (uint sl = 0u; sl < INK2_ATTN_RES_SLICES; ++sl) {
                sq += st[sl * INK2_ATTN_RES_STATS];
                d += st[sl * INK2_ATTN_RES_STATS + 1u + q];
            }
            inv[s] = rsqrt(sq / float(H) + args.eps);
            raw[s] = d * inv[s] * scale;
            mx = max(mx, raw[s]);
        }
        float denom = 0.0f;
        for (uint s = 0u; s < n_src; ++s) {
            raw[s] = exp(raw[s] - mx);
            denom += raw[s];
        }
        const float inv_d = 1.0f / denom;
        for (uint s = 0u; s < n_src; ++s) {
            weights[q][s] = raw[s] * inv_d * inv[s];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint i = group.x * INK2_BLOCK + tid;
    if (i >= H) {
        return;
    }
    float acc[INK2_MAX_ATTN_RES_Q] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (uint s = 0u; s < n_src; ++s) {
        const float v = ink2_attn_res_src(blocks, partial, addend, args.has_addend,
                                          s, n_blocks, block_stride, row_off + i);
        for (uint q = 0u; q < n_q; ++q) {
            acc[q] += weights[q][s] * v;
        }
    }
    for (uint q = 0u; q < n_q; ++q) {
        out[ulong(q) * block_stride + row_off + i] = acc[q];
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
//
// `defer_state` is the speculative verify pass: its rows are not all kept, so
// the lags the next pass starts from are not the last `k - 1` rows of this
// one. A deferring launch leaves `state[0..lags)` as it found it and writes the
// pass's own inputs behind it, at rows `lags + t`; `sconv_commit` then moves
// the `lags` rows that end at the last kept row to the front. The state needs
// `lags + rows` rows for it.

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
        if (args.defer_state != 0u) {
            state[ulong(lags + t) * C + c] = cur;
        }
        for (uint l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    if (args.defer_state != 0u) {
        return;
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
        if (args.defer_state != 0u) {
            state[ulong(lags + t) * C + c] = cur;
        }
        for (uint l = 0u; l + 1u < lags; ++l) {
            hist[l] = hist[l + 1u];
        }
        if (lags > 0u) {
            hist[lags - 1u] = cur;
        }
    }
    if (args.defer_state != 0u) {
        return;
    }
    for (uint l = 0u; l < lags; ++l) {
        state[ulong(l) * C + c] = hist[l];
    }
}

// ---- 8b. Short-convolution state commit after a deferring pass -------------
// One buffer holds every layer's four states: per layer `[k | v | attn | mlp]`,
// each `[lags + hist_rows, C]` with C = kv_dim, kv_dim, hidden, hidden. A
// thread owns one channel of one state, reads its `lags` rows starting at
// `keep` (rows kept by the pass) and writes them to the front; reading into
// registers first is what lets the two ranges overlap.
// Grid: (ceil(total / 256), 1, 1), block 256; total = layers * 2 * (kv_dim + hidden).

kernel void kernel_inkling2_sconv_commit(
        constant ink2_args_sconv_commit &args [[buffer(0)]],
        device float *state [[buffer(1)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.total) {
        return;
    }
    const uint per = 2u * (args.kv_dim + args.hidden);
    const uint rows = args.lags + args.hist_rows;
    const uint layer = tid / per;
    const uint rem = tid % per;
    uint C;
    uint c;
    uint before;
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
    device float *seg = state + (ulong(layer) * per + before) * rows;
    float hist[INK2_MAX_CONV_K];
    for (uint l = 0u; l < args.lags; ++l) {
        hist[l] = seg[ulong(args.keep + l) * C + c];
    }
    for (uint l = 0u; l < args.lags; ++l) {
        seg[ulong(l) * C + c] = hist[l];
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

// The key walk shared by `rel_attn` and `rel_attn_splitk` (head_dim 128, a
// lane owning four dims): eight keys' scores first, then one maximum, then the
// eight weights. An online softmax that rescales at every key makes each
// `exp` wait on the key before it; this one waits once per eight. With the
// per-key form a decode row over ~70 keys took 86 us on an M1 Max.
// Reads q_pos, kvh, d0, qv, rb, args; updates m_max, l_sum, acc.
#define INK2_REL_ATTN_BLOCK 8u
#define INK2_REL_ATTN_KEYS(FIRST, END)                                                     \
    for (uint k0 = (FIRST); k0 < (END); k0 += INK2_REL_ATTN_BLOCK) {                       \
        const uint n_k = min(INK2_REL_ATTN_BLOCK, (END) - k0);                             \
        float sc[INK2_REL_ATTN_BLOCK];                                                     \
        float b_max = m_max;                                                               \
        _Pragma("clang loop unroll(full)")                                                 \
        for (uint i = 0u; i < INK2_REL_ATTN_BLOCK; ++i) {                                  \
            sc[i] = -1e30f;                                                                \
            if (i < n_k) {                                                                 \
                const uint k_pos = k0 + i;                                                 \
                const uint slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;       \
                const half4 kv = *((device const half4 *)(                                 \
                    k_cache + (ulong(slot) * args.n_head_kv + kvh) * 128u + d0));          \
                float dot = qv.x * float(kv.x) + qv.y * float(kv.y) +                      \
                            qv.z * float(kv.z) + qv.w * float(kv.w);                       \
                dot = simd_sum(dot) * args.scale;                                          \
                const uint dist = q_pos - k_pos;                                           \
                sc[i] = dot + ((dist < args.rel_extent) ? rb[dist] : 0.0f);                \
                b_max = max(b_max, sc[i]);                                                 \
            }                                                                              \
        }                                                                                  \
        const float exp_old = (l_sum > 0.0f) ? exp(m_max - b_max) : 0.0f;                  \
        l_sum *= exp_old;                                                                  \
        acc *= exp_old;                                                                    \
        _Pragma("clang loop unroll(full)")                                                 \
        for (uint i = 0u; i < INK2_REL_ATTN_BLOCK; ++i) {                                  \
            if (i < n_k) {                                                                 \
                const uint k_pos = k0 + i;                                                 \
                const uint slot = args.is_swa != 0u ? (k_pos % args.kv_cap) : k_pos;       \
                const half4 vv = *((device const half4 *)(                                 \
                    v_cache + (ulong(slot) * args.n_head_kv + kvh) * 128u + d0));          \
                const float e = exp(sc[i] - b_max);                                        \
                l_sum += e;                                                                \
                acc += e * float4(vv);                                                     \
            }                                                                              \
        }                                                                                  \
        m_max = b_max;                                                                     \
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

        INK2_REL_ATTN_KEYS(start_pos, q_pos + 1u)

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

// ---- 10b. Split-K attention for a short pass --------------------------------
// `rel_attn`'s contract for `head_dim == 128`, with each row's keys
// `[window edge, position]` cut into runs, a simdgroup each, and the runs'
// online-softmax states merged afterwards in run order. One simdgroup per
// (head, row) leaves the GPU idle on a decode row: 24 simdgroups walk the
// whole window. A row's run count is a function of its key count alone
// (`ink2_attn_splits`), so a position is summed the same way as a decode row
// and as one of a verify pass's rows - which is why every short pass takes
// this entry.
// Stage 1: grid (n_head, ceil(max_splits / 4), rows), block 128 (4 runs a threadgroup).
// Stage 2: grid (n_head, rows, 1), block 32 (merges the runs, applies the output gate).
// `split_ml` is `[rows, n_head, INK2_ATTN_MAX_SPLITS, 2]`, `split_out`
// `[rows, n_head, INK2_ATTN_MAX_SPLITS, 128]`.

#define INK2_ATTN_MAX_SPLITS 64u

// About 128 keys a run, a multiple of four, between 4 and INK2_ATTN_MAX_SPLITS.
static inline uint ink2_attn_splits(uint keys) {
    const uint want = ((keys + 127u) / 128u + 3u) & ~3u;
    return clamp(want, 4u, INK2_ATTN_MAX_SPLITS);
}

kernel void kernel_inkling2_rel_attn_splitk(
        constant ink2_args_rel_attn &args [[buffer(0)]],
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
    const uint row = group.z;
    if (h >= args.n_head || row >= args.rows) {
        return;
    }
    const uint kvh = h / (args.n_head / args.n_head_kv);
    const uint q_pos = args.pos0 + row;
    const uint start_pos = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                               ? (q_pos + 1u - args.window_size)
                               : 0u;
    const uint total_kv = q_pos + 1u - start_pos;
    const uint n_splits = ink2_attn_splits(total_kv);
    if (s >= n_splits) {
        return;
    }
    const uint chunk = (total_kv + n_splits - 1u) / n_splits;
    const uint first = start_pos + min(total_kv, s * chunk);
    const uint end_pos = min(q_pos + 1u, first + chunk);

    const uint d0 = lane * 4u;
    const ulong head_row = ulong(row) * args.n_head + h;
    const float4 qv = *((device const float4 *)(q + head_row * 128u + d0));
    device const float *rb = rel_bias + head_row * args.rel_extent;

    float m_max = (s == 0u) ? sinks[head_row] : -1e30f;
    float l_sum = (s == 0u) ? 1.0f : 0.0f;
    float4 acc = float4(0.0f);

    INK2_REL_ATTN_KEYS(first, end_pos)

    const ulong s_idx = head_row * INK2_ATTN_MAX_SPLITS + s;
    if (lane == 0u) {
        split_ml[s_idx * 2u] = m_max;
        split_ml[s_idx * 2u + 1u] = l_sum;
    }
    *((device float4 *)(split_out + s_idx * 128u + d0)) = acc;
}

kernel void kernel_inkling2_rel_attn_reduce(
        constant ink2_args_rel_attn &args [[buffer(0)]],
        device const float *split_ml [[buffer(1)]],
        device const float *split_out [[buffer(2)]],
        device const float *gate [[buffer(3)]],
        device float *out [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint h = group.x;
    const uint row = group.y;
    if (h >= args.n_head || row >= args.rows) {
        return;
    }
    const uint q_pos = args.pos0 + row;
    const uint start_pos = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                               ? (q_pos + 1u - args.window_size)
                               : 0u;
    const uint n_splits = ink2_attn_splits(q_pos + 1u - start_pos);
    const ulong head_row = ulong(row) * args.n_head + h;
    const uint d0 = lane * 4u;
    float m_max = -1e30f;
    float l_sum = 0.0f;
    float4 acc = float4(0.0f);

    for (uint s = 0u; s < n_splits; ++s) {
        const ulong s_idx = head_row * INK2_ATTN_MAX_SPLITS + s;
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
    const ulong base = head_row * 128u + d0;
    const float4 gv = *((device const float4 *)(gate + base));
    const float4 res = float4(
        (acc.x * inv_l) * ink2_sigmoid(gv.x),
        (acc.y * inv_l) * ink2_sigmoid(gv.y),
        (acc.z * inv_l) * ink2_sigmoid(gv.z),
        (acc.w * inv_l) * ink2_sigmoid(gv.w)
    );
    *((device float4 *)(out + base)) = res;
}

// ---- 10c. Attention for a long pass, on simdgroup matrices ------------------
// `rel_attn`'s contract and bindings for `head_dim == 128`: 64 query rows of
// one head a threadgroup, eight a simdgroup held in registers, the keys
// streamed 16 at a time through threadgroup memory in f32 with the next block
// prefetched, the softmax folded online. The tile is the shared flash
// attention's f32 grouped-query one; what this family adds is the key ring of
// a sliding-window layer, the window's lower edge, the relative bias by
// distance, the per-query sink (a key of logit `sink` and value zero, which
// is where the running maximum and sum start) and the output gate.
//
// One simdgroup a (head, row) walking its keys with scalar dots took 27 ms a
// layer for a 1024-row pass on an M1 Max, the largest share of a prefill.
// Grid: (ceil(rows / 64), n_head, 1), block 256.

#define INK2_ATTN_TILE_ROWS 64u
#define INK2_ATTN_TILE_KEYS 16u
#define INK2_ATTN_TILE_THREADS 256u
#define INK2_ATTN_TILE_HD 128u
#define INK2_ATTN_TILE_LDK (INK2_ATTN_TILE_HD + 8u)

kernel void kernel_inkling2_rel_attn_tile(
        constant ink2_args_rel_attn &args [[buffer(0)]],
        device const float *q [[buffer(1)]],
        device const float *rel_bias [[buffer(2)]],
        device const float *sinks [[buffer(3)]],
        device const float *gate [[buffer(4)]],
        device const half *k_cache [[buffer(5)]],
        device const half *v_cache [[buffer(6)]],
        device float *out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint HD = INK2_ATTN_TILE_HD;
    const uint KB = INK2_ATTN_TILE_KEYS;
    const uint NB = KB / 8u;
    const uint NT = INK2_ATTN_TILE_THREADS;
    // Each staged row is padded by eight floats so the transposed K loads do
    // not land eight rows on one bank.
    const uint LDK = INK2_ATTN_TILE_LDK;
    threadgroup float kv[2u * INK2_ATTN_TILE_KEYS * INK2_ATTN_TILE_LDK];

    const uint h = group.y;
    const uint q0 = group.x * INK2_ATTN_TILE_ROWS;
    if (h >= args.n_head || q0 >= args.rows) {
        return;
    }
    const uint kvh = h / (args.n_head / args.n_head_kv);
    // The tile's keys: from its first row's window edge to its last row.
    const uint last_pos = args.pos0 + min(args.rows, q0 + INK2_ATTN_TILE_ROWS) - 1u;
    const uint first_pos = args.pos0 + q0;
    const uint key_lo = (args.is_swa != 0u && first_pos + 1u > args.window_size)
                            ? (first_pos + 1u - args.window_size)
                            : 0u;
    const uint key_end = last_pos + 1u;

    // The lane's place in every 8x8 fragment: columns `fn, fn + 1` of row
    // `fr`; the four lanes sharing a row differ in lane bits 0 and 3.
    const ushort fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    const ushort fr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    const uint qrow = q0 + sg * 8u + fr;
    const bool live = qrow < args.rows;
    const uint q_pos = args.pos0 + qrow;
    const uint row_lo = (args.is_swa != 0u && q_pos + 1u > args.window_size)
                            ? (q_pos + 1u - args.window_size)
                            : 0u;
    const ulong head_row = ulong(live ? qrow : 0u) * args.n_head + h;

    simdgroup_float8x8 qf[HD / 8u];
    simdgroup_float8x8 of[HD / 8u];
    device const float *qp = q + head_row * HD;
    INK2_UNROLL for (uint i = 0u; i < HD / 8u; ++i) {
        const float2 qv = live ? float2(qp[8u * i + fn], qp[8u * i + fn + 1u]) * args.scale
                               : float2(0.0f);
        qf[i].thread_elements()[0] = qv.x;
        qf[i].thread_elements()[1] = qv.y;
        of[i] = simdgroup_float8x8(0.0f);
    }
    device const float *rb = rel_bias + head_row * args.rel_extent;

    // The sink: a key of logit `sink` whose value is zero.
    float run_max = live ? sinks[head_row] : 0.0f;
    float run_sum = 1.0f;
    threadgroup float *sk = kv;
    threadgroup float *sv = kv + KB * LDK;
    // Four consecutive head values per load; a key past the end stages as
    // zero and is masked below.
    const uint hd4 = HD / 4u;
    const uint SL = (KB * hd4 + NT - 1u) / NT;
    float4 rk[SL];
    float4 rv[SL];
    auto fetch = [&](uint s0) {
        INK2_UNROLL for (uint i = 0u; i < SL; ++i) {
            const uint e = tid + i * NT;
            const uint j = e / hd4;
            const uint d = (e - j * hd4) * 4u;
            const uint s = s0 + j;
            rk[i] = float4(0.0f);
            rv[i] = float4(0.0f);
            if (e < KB * hd4 && s < key_end) {
                const uint slot = args.is_swa != 0u ? (s % args.kv_cap) : s;
                const ulong o = (ulong(slot) * args.n_head_kv + kvh) * HD + d;
                rk[i] = float4(*((device const half4 *)(k_cache + o)));
                rv[i] = float4(*((device const half4 *)(v_cache + o)));
            }
        }
    };
    fetch(key_lo);
    for (uint s0 = key_lo; s0 < key_end; s0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INK2_UNROLL for (uint i = 0u; i < SL; ++i) {
            const uint e = tid + i * NT;
            if (e < KB * hd4) {
                const uint j = e / hd4;
                const uint d = (e - j * hd4) * 4u;
                *((threadgroup float4 *)(sk + j * LDK + d)) = rk[i];
                *((threadgroup float4 *)(sv + j * LDK + d)) = rv[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s0 + KB < key_end) {
            fetch(s0 + KB);
        }

        // Scores: S (8 x KB) = Q (8 x HD) . K^T, one fragment per 8 keys.
        simdgroup_float8x8 sf[NB];
        INK2_UNROLL for (uint b = 0u; b < NB; ++b) {
            sf[b] = simdgroup_float8x8(0.0f);
        }
        INK2_UNROLL for (uint i = 0u; i < HD / 8u; ++i) {
            INK2_UNROLL for (uint b = 0u; b < NB; ++b) {
                simdgroup_float8x8 kt;
                // kt(r, c) = K[8b + c][8i + r]: the transposed load.
                simdgroup_load(kt, sk + 8u * b * LDK + 8u * i, LDK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[b], qf[i], kt, sf[b]);
            }
        }

        simdgroup_float8x8 pf[NB];
        float block_max = -INFINITY;
        INK2_UNROLL for (uint b = 0u; b < NB; ++b) {
            INK2_UNROLL for (uint c = 0u; c < 2u; ++c) {
                const uint key = s0 + 8u * b + fn + c;
                float score = -INFINITY;
                if (live && key <= q_pos && key >= row_lo) {
                    const uint dist = q_pos - key;
                    score = sf[b].thread_elements()[c] +
                            (dist < args.rel_extent ? rb[dist] : 0.0f);
                }
                sf[b].thread_elements()[c] = score;
                block_max = max(block_max, score);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float new_max = max(run_max, block_max);
        const float alpha = exp(run_max - new_max);
        float block_sum = 0.0f;
        INK2_UNROLL for (uint b = 0u; b < NB; ++b) {
            INK2_UNROLL for (uint c = 0u; c < 2u; ++c) {
                const float p = exp(sf[b].thread_elements()[c] - new_max);
                pf[b].thread_elements()[c] = p;
                block_sum += p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        run_sum = run_sum * alpha + block_sum;
        run_max = new_max;
        INK2_UNROLL for (uint i = 0u; i < HD / 8u; ++i) {
            of[i].thread_elements()[0] *= alpha;
            of[i].thread_elements()[1] *= alpha;
            INK2_UNROLL for (uint b = 0u; b < NB; ++b) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, sv + 8u * b * LDK + 8u * i, LDK);
                simdgroup_multiply_accumulate(of[i], pf[b], vf, of[i]);
            }
        }
    }

    if (!live) {
        return;
    }
    const float inv = 1.0f / run_sum;
    device const float *gp = gate + head_row * HD;
    device float *op = out + head_row * HD;
    INK2_UNROLL for (uint i = 0u; i < HD / 8u; ++i) {
        op[8u * i + fn] = (of[i].thread_elements()[0] * inv) * ink2_sigmoid(gp[8u * i + fn]);
        op[8u * i + fn + 1u] =
            (of[i].thread_elements()[1] * inv) * ink2_sigmoid(gp[8u * i + fn + 1u]);
    }
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

// ---- 12. Routed MoE experts (NVFP4 and MXFP8) ------------------------------
//
// An expert's W13 is `[2 * inter, in]` with gate row `2j` and up row `2j + 1`;
// its W2 is `[out, inter]`. NVFP4 packs two E2M1 weights a byte with one E4M3
// scale per 16 weights and an F32 `amax / denom` per expert; MXFP8 stores one
// E4M3 weight a byte with one UE8M0 (power of two) scale per 32.
//
// One inner body per codec serves the one-row decode entries and the batched
// entries, and every sum in it is one of the safe-math helpers below, so a row
// computes the same bits whether it is alone in the pass or shares an expert's
// weights with seven others. That is what lets a speculative verify pass
// (several rows, experts shared) agree with a plain decode step on a router
// near-tie.
//
// Measured on an M1 Max, one decode row, NVFP4, per layer (W13 + W2):
//   243 + 134 us  a 256-entry scale table filled into threadgroup memory
//                 behind a barrier, one weight row a simdgroup, gate and up
//                 each reading the activations, W2 walking its eight experts
//                 in one thread;
//   170 + 128 us  no table fill, gate and up sharing the activation read, W2
//                 one expert a threadgroup row;
//   147 +  90 us  four weight rows a simdgroup sharing the activation read
//                 (two rows: 170 + 104; eight: 186 + 98, lost occupancy).
// What did not help: a 256-entry byte -> weight-pair table (248 + 193 us; a
// table that size is a per-lane gather, the 16-entry one is not), decoding
// E2M1 by arithmetic (210 us W13), and the shape of the 16-term sum.
// The floor of a launch is 6 us; the pass with no weight decode at all is 97.

#pragma METAL fp math_mode(safe)

static inline float ink2_dot4(float4 w, float4 x) {
    return fma(w.x, x.x, fma(w.y, x.y, fma(w.z, x.z, w.w * x.w)));
}

static inline float ink2_dot16(float4 w0, float4 w1, float4 w2, float4 w3,
                               float4 x0, float4 x1, float4 x2, float4 x3) {
    return (ink2_dot4(w0, x0) + ink2_dot4(w1, x1)) + (ink2_dot4(w2, x2) + ink2_dot4(w3, x3));
}

// acc + scale * part
static inline float ink2_scaled_add(float acc, float scale, float part) {
    return fma(scale, part, acc);
}

static inline float ink2_mul(float a, float b) {
    return a * b;
}

static inline float ink2_div(float a, float b) {
    return a / b;
}

// softcap(g) * sigmoid(g) * softcap(u) * route weight
static inline float ink2_swiglu_capped(float g, float u, float gate_cap, float up_cap, float rw) {
    const float gc = gate_cap > 0.0f ? gate_cap * tanh(g / gate_cap) : g;
    const float uc = up_cap > 0.0f ? up_cap * tanh(u / up_cap) : u;
    const float sg = 1.0f / (1.0f + exp(-g));
    return ((gc * sg) * uc) * rw;
}

#pragma METAL fp math_mode(fast)

// One E4M3 code, by arithmetic: a 256-entry table is a per-lane gather.
// Codes 0x7F / 0xFF (the format's NaN) decode to zero, as `ink2_e4m3_to_f32`.
static inline float ink2_e4m3_fast(uint b) {
    const uint lo = b & 0x7Fu;
    const float normal = as_type<float>((lo << 20u) + 0x3C000000u);
    const float sub = float(lo) * 0.001953125f;
    const float mag = lo < 8u ? sub : (lo == 0x7Fu ? 0.0f : normal);
    return as_type<float>(as_type<uint>(mag) | ((b & 0x80u) << 24u));
}

// 16 E2M1 weights (8 bytes), unscaled.
static inline void ink2_e2m1_group(device const uchar *p,
                                   thread float4 &w0, thread float4 &w1,
                                   thread float4 &w2, thread float4 &w3) {
    device const uchar4 *wp = (device const uchar4 *)p;
    const uchar4 b0 = wp[0];
    const uchar4 b1 = wp[1];
    w0 = float4(INK2_E2M1_LUT[b0.x & 0xFu], INK2_E2M1_LUT[b0.x >> 4u],
                INK2_E2M1_LUT[b0.y & 0xFu], INK2_E2M1_LUT[b0.y >> 4u]);
    w1 = float4(INK2_E2M1_LUT[b0.z & 0xFu], INK2_E2M1_LUT[b0.z >> 4u],
                INK2_E2M1_LUT[b0.w & 0xFu], INK2_E2M1_LUT[b0.w >> 4u]);
    w2 = float4(INK2_E2M1_LUT[b1.x & 0xFu], INK2_E2M1_LUT[b1.x >> 4u],
                INK2_E2M1_LUT[b1.y & 0xFu], INK2_E2M1_LUT[b1.y >> 4u]);
    w3 = float4(INK2_E2M1_LUT[b1.z & 0xFu], INK2_E2M1_LUT[b1.z >> 4u],
                INK2_E2M1_LUT[b1.w & 0xFu], INK2_E2M1_LUT[b1.w >> 4u]);
}

// 4 E4M3 weights, unscaled.
static inline float4 ink2_e4m3_x4(uchar4 b) {
    return float4(ink2_e4m3_fast(b.x), ink2_e4m3_fast(b.y), ink2_e4m3_fast(b.z), ink2_e4m3_fast(b.w));
}

// WR consecutive NVFP4 weight rows (starting at `w` / `sc`, `w_stride` and
// `s_stride` bytes apart) against up to NR activation rows. `out[wr * NR + rr]`
// is the lane-reduced dot of weight row `wr` with activation row `r_ids[rr]`,
// valid on every lane.
template <uint WR, uint NR>
static inline void ink2_nvfp4_block(
        device const uchar *w, ulong w_stride,
        device const uchar *sc, ulong s_stride,
        device const float *x, uint in_dim,
        thread const uint *r_ids, uint n_r, uint lane,
        thread float *out) {
    float acc[WR * NR];
    for (uint i = 0u; i < WR * NR; ++i) {
        acc[i] = 0.0f;
    }
    const uint n_groups = in_dim / 16u;
    for (uint g = lane; g < n_groups; g += 32u) {
        float scale[WR];
        float4 w0[WR], w1[WR], w2[WR], w3[WR];
        _Pragma("clang loop unroll(full)")
        for (uint wr = 0u; wr < WR; ++wr) {
            scale[wr] = ink2_e4m3_fast(uint(sc[wr * s_stride + g]));
            ink2_e2m1_group(w + wr * w_stride + g * 8u, w0[wr], w1[wr], w2[wr], w3[wr]);
        }
        _Pragma("clang loop unroll(full)")
        for (uint rr = 0u; rr < NR; ++rr) {
            if (rr < n_r) {
                device const float4 *xp =
                    (device const float4 *)(x + ulong(r_ids[rr]) * in_dim + g * 16u);
                const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                _Pragma("clang loop unroll(full)")
                for (uint wr = 0u; wr < WR; ++wr) {
                    acc[wr * NR + rr] = ink2_scaled_add(
                        acc[wr * NR + rr], scale[wr],
                        ink2_dot16(w0[wr], w1[wr], w2[wr], w3[wr], x0, x1, x2, x3));
                }
            }
        }
    }
    for (uint i = 0u; i < WR * NR; ++i) {
        out[i] = ink2_lane_sum(acc[i]);
    }
}

// The MXFP8 twin of `ink2_nvfp4_block`: a 32-weight group is two 16-term sums,
// each added under the group's scale.
template <uint WR, uint NR>
static inline void ink2_mxfp8_block(
        device const uchar *w, ulong w_stride,
        device const uchar *sc, ulong s_stride,
        device const float *x, uint in_dim,
        thread const uint *r_ids, uint n_r, uint lane,
        thread float *out) {
    float acc[WR * NR];
    for (uint i = 0u; i < WR * NR; ++i) {
        acc[i] = 0.0f;
    }
    const uint n_groups = in_dim / 32u;
    for (uint g = lane; g < n_groups; g += 32u) {
        float scale[WR];
        _Pragma("clang loop unroll(full)")
        for (uint wr = 0u; wr < WR; ++wr) {
            scale[wr] = ink2_ue8m0_to_f32(uint(sc[wr * s_stride + g]));
        }
        _Pragma("clang loop unroll(full)")
        for (uint h = 0u; h < 2u; ++h) {
            float4 w0[WR], w1[WR], w2[WR], w3[WR];
            _Pragma("clang loop unroll(full)")
            for (uint wr = 0u; wr < WR; ++wr) {
                device const uchar4 *wp =
                    (device const uchar4 *)(w + wr * w_stride + g * 32u + h * 16u);
                w0[wr] = ink2_e4m3_x4(wp[0]);
                w1[wr] = ink2_e4m3_x4(wp[1]);
                w2[wr] = ink2_e4m3_x4(wp[2]);
                w3[wr] = ink2_e4m3_x4(wp[3]);
            }
            _Pragma("clang loop unroll(full)")
            for (uint rr = 0u; rr < NR; ++rr) {
                if (rr < n_r) {
                    device const float4 *xp = (device const float4 *)(
                        x + ulong(r_ids[rr]) * in_dim + g * 32u + h * 16u);
                    const float4 x0 = xp[0], x1 = xp[1], x2 = xp[2], x3 = xp[3];
                    _Pragma("clang loop unroll(full)")
                    for (uint wr = 0u; wr < WR; ++wr) {
                        acc[wr * NR + rr] = ink2_scaled_add(
                            acc[wr * NR + rr], scale[wr],
                            ink2_dot16(w0[wr], w1[wr], w2[wr], w3[wr], x0, x1, x2, x3));
                    }
                }
            }
        }
    }
    for (uint i = 0u; i < WR * NR; ++i) {
        out[i] = ink2_lane_sum(acc[i]);
    }
}

// ---- 12a. One-row entries: a routed column per threadgroup row -------------
// A simdgroup owns four weight rows: two (gate, up) pairs of W13, four output
// rows of W2. `inter_dim` must be even and `out_dim` a multiple of four; the
// host refuses other shapes.
// W13: grid (ceil(inter_dim / 8), rows * top_k, 1), block 128.
// W2:  grid (ceil(out_dim / 16), rows * top_k, 1), block 128; one output row
//      per routed column, which `moe_reduce_topk` then sums per token.

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
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint j = 2u * (group.x * INK2_MV_SG + sg);
    const uint col = group.y;
    if (j >= args.inter_dim || col >= args.rows * args.top_k) {
        return;
    }
    const uint expert = expert_ids[col];
    const uint n_r = 1u;
    uint c_ids[1] = {col};
    uint r_ids[1] = {col / args.top_k};
    const uint in_dim = args.in_dim;
    const ulong w_row_bytes = ulong(in_dim / 2u);
    const ulong s_row_bytes = ulong(in_dim / 16u);
    const ulong e_row = ulong(expert) * (2u * args.inter_dim) + 2u * j;
    float v[4];
    ink2_nvfp4_block<4, 1>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, in_dim, r_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = ink2_div(w13_amax[expert], w13_denom[0]);
        for (uint q = 0u; q < 2u; ++q) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                out[ulong(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    ink2_mul(v[(2u * q) * 1u + rr], outer),
                    ink2_mul(v[(2u * q + 1u) * 1u + rr], outer),
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
            }
        }
    }
}

kernel void kernel_inkling2_moe_nvfp4_w2_cols(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const float *w2_amax [[buffer(4)]],
        device const float *w2_denom [[buffer(5)]],
        device const uint *expert_ids [[buffer(6)]],
        device float *slots_out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = 4u * (group.x * INK2_MV_SG + sg);
    const uint col = group.y;
    if (o >= args.out_dim || col >= args.rows * args.top_k) {
        return;
    }
    const uint expert = expert_ids[col];
    const uint n_r = 1u;
    uint c_ids[1] = {col};
    const uint in_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim / 2u);
    const ulong s_row_bytes = ulong(in_dim / 16u);
    const ulong e_row = ulong(expert) * args.out_dim + o;
    float v[4];
    ink2_nvfp4_block<4, 1>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, in_dim, c_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = ink2_div(w2_amax[expert], w2_denom[0]);
        for (uint wr = 0u; wr < 4u; ++wr) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                slots_out[ulong(c_ids[rr]) * args.out_dim + o + wr] = ink2_mul(v[wr * 1u + rr], outer);
            }
        }
    }
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
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint j = 2u * (group.x * INK2_MV_SG + sg);
    const uint col = group.y;
    if (j >= args.inter_dim || col >= args.rows * args.top_k) {
        return;
    }
    const uint expert = expert_ids[col];
    const uint n_r = 1u;
    uint c_ids[1] = {col};
    uint r_ids[1] = {col / args.top_k};
    const uint in_dim = args.in_dim;
    const ulong w_row_bytes = ulong(in_dim);
    const ulong s_row_bytes = ulong(in_dim / 32u);
    const ulong e_row = ulong(expert) * (2u * args.inter_dim) + 2u * j;
    float v[4];
    ink2_mxfp8_block<4, 1>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, in_dim, r_ids, n_r, lane, v);
    if (lane == 0u) {
        for (uint q = 0u; q < 2u; ++q) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                out[ulong(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    v[(2u * q) * 1u + rr],
                    v[(2u * q + 1u) * 1u + rr],
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
            }
        }
    }
}

kernel void kernel_inkling2_moe_mxfp8_w2_cols(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const uint *expert_ids [[buffer(4)]],
        device float *slots_out [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = 4u * (group.x * INK2_MV_SG + sg);
    const uint col = group.y;
    if (o >= args.out_dim || col >= args.rows * args.top_k) {
        return;
    }
    const uint expert = expert_ids[col];
    const uint n_r = 1u;
    uint c_ids[1] = {col};
    const uint in_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim);
    const ulong s_row_bytes = ulong(in_dim / 32u);
    const ulong e_row = ulong(expert) * args.out_dim + o;
    float v[4];
    ink2_mxfp8_block<4, 1>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, in_dim, c_ids, n_r, lane, v);
    if (lane == 0u) {
        for (uint wr = 0u; wr < 4u; ++wr) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                slots_out[ulong(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 1u + rr];
            }
        }
    }
}

// ---- 12b. Multi-row entries: an expert's weights streamed once per batch ---
//
// `moe_plan` turns a pass's routing into work items, each one expert and up to
// `batch` of the columns routed to it, so a threadgroup of the batched entries
// reads a weight row once for all of them. `cols` is `[n_experts, rows]`, an
// expert's columns ascending; `items` is `[1 + 2 * max_items]`: the item count,
// then `(expert, first | count << 24)` per item. The host sizes the grid for
// `max_items = min(rows * top_k, n_experts + rows * top_k / batch)` and the
// threadgroups past the count return.
// Grid: (1, 1, 1), block 256 (one thread an expert, at most 256 experts).

struct ink2_args_moe_plan {
    uint rows;
    uint top_k;
    uint n_experts;
    uint batch;
};
static_assert(sizeof(ink2_args_moe_plan) == 16, "ink2_args_moe_plan layout");

kernel void kernel_inkling2_moe_plan(
        constant ink2_args_moe_plan &args [[buffer(0)]],
        device const uint *expert_ids [[buffer(1)]],
        device uint *cols [[buffer(2)]],
        device uint *items [[buffer(3)]],
        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint first[INK2_MAX_EXPERTS];
    threadgroup uint counts[INK2_MAX_EXPERTS];
    const uint E = min(args.n_experts, INK2_MAX_EXPERTS);
    const uint total = args.rows * args.top_k;
    uint cnt = 0u;
    if (tid < E) {
        device uint *dst = cols + ulong(tid) * args.rows;
        for (uint col = 0u; col < total; ++col) {
            if (expert_ids[col] == tid) {
                dst[cnt] = col;
                cnt += 1u;
            }
        }
        counts[tid] = cnt;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
        uint run = 0u;
        for (uint e = 0u; e < E; ++e) {
            first[e] = run;
            run += (counts[e] + args.batch - 1u) / args.batch;
        }
        items[0] = run;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < E) {
        const uint n_items = (cnt + args.batch - 1u) / args.batch;
        for (uint b = 0u; b < n_items; ++b) {
            const uint idx = first[tid] + b;
            const uint base = b * args.batch;
            items[1u + 2u * idx] = tid;
            items[2u + 2u * idx] = base | (min(args.batch, cnt - base) << 24u);
        }
    }
}

// A simdgroup owns two weight rows against up to eight columns (16 sums in
// registers). W13: grid (ceil(inter_dim / 4), max_items, 1), block 128.
// W2: grid (ceil(out_dim / 8), max_items, 1), block 128.

kernel void kernel_inkling2_moe_nvfp4_w13_grouped(
        constant ink2_args_moe_w13 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const uchar *w13 [[buffer(2)]],
        device const uchar *w13_scale [[buffer(3)]],
        device const float *w13_amax [[buffer(4)]],
        device const float *w13_denom [[buffer(5)]],
        device const float *expert_weights [[buffer(6)]],
        device const uint *cols [[buffer(7)]],
        device const uint *items [[buffer(8)]],
        device float *out [[buffer(9)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint j = 1u * (group.x * INK2_MV_SG + sg);
    if (j >= args.inter_dim || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    const uint n_r = packed >> 24u;
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    uint c_ids[8];
    uint r_ids[8];
    for (uint rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
        r_ids[rr] = c_ids[rr] / args.top_k;
    }
    const uint in_dim = args.in_dim;
    const ulong w_row_bytes = ulong(in_dim / 2u);
    const ulong s_row_bytes = ulong(in_dim / 16u);
    const ulong e_row = ulong(expert) * (2u * args.inter_dim) + 2u * j;
    float v[16];
    ink2_nvfp4_block<2, 8>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, in_dim, r_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = ink2_div(w13_amax[expert], w13_denom[0]);
        for (uint q = 0u; q < 1u; ++q) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                out[ulong(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    ink2_mul(v[(2u * q) * 8u + rr], outer),
                    ink2_mul(v[(2u * q + 1u) * 8u + rr], outer),
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
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
        device const uint *cols [[buffer(6)]],
        device const uint *items [[buffer(7)]],
        device float *slots_out [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = 2u * (group.x * INK2_MV_SG + sg);
    if (o >= args.out_dim || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    const uint n_r = packed >> 24u;
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    uint c_ids[8];
    for (uint rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
    }
    const uint in_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim / 2u);
    const ulong s_row_bytes = ulong(in_dim / 16u);
    const ulong e_row = ulong(expert) * args.out_dim + o;
    float v[16];
    ink2_nvfp4_block<2, 8>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, in_dim, c_ids, n_r, lane, v);
    if (lane == 0u) {
        const float outer = ink2_div(w2_amax[expert], w2_denom[0]);
        for (uint wr = 0u; wr < 2u; ++wr) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                slots_out[ulong(c_ids[rr]) * args.out_dim + o + wr] = ink2_mul(v[wr * 8u + rr], outer);
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
        device const uint *cols [[buffer(5)]],
        device const uint *items [[buffer(6)]],
        device float *out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint j = 1u * (group.x * INK2_MV_SG + sg);
    if (j >= args.inter_dim || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    const uint n_r = packed >> 24u;
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    uint c_ids[8];
    uint r_ids[8];
    for (uint rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
        r_ids[rr] = c_ids[rr] / args.top_k;
    }
    const uint in_dim = args.in_dim;
    const ulong w_row_bytes = ulong(in_dim);
    const ulong s_row_bytes = ulong(in_dim / 32u);
    const ulong e_row = ulong(expert) * (2u * args.inter_dim) + 2u * j;
    float v[16];
    ink2_mxfp8_block<2, 8>(
        w13 + e_row * w_row_bytes, w_row_bytes, w13_scale + e_row * s_row_bytes, s_row_bytes,
        x, in_dim, r_ids, n_r, lane, v);
    if (lane == 0u) {
        for (uint q = 0u; q < 1u; ++q) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                out[ulong(c_ids[rr]) * args.inter_dim + j + q] = ink2_swiglu_capped(
                    v[(2u * q) * 8u + rr],
                    v[(2u * q + 1u) * 8u + rr],
                    args.gate_cap, args.up_cap, expert_weights[c_ids[rr]]);
            }
        }
    }
}

kernel void kernel_inkling2_moe_mxfp8_w2_grouped(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const uint *cols [[buffer(4)]],
        device const uint *items [[buffer(5)]],
        device float *slots_out [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint o = 2u * (group.x * INK2_MV_SG + sg);
    if (o >= args.out_dim || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    const uint n_r = packed >> 24u;
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    uint c_ids[8];
    for (uint rr = 0u; rr < 8u; ++rr) {
        c_ids[rr] = rr < n_r ? ec[rr] : 0u;
    }
    const uint in_dim = args.inter_dim;
    const ulong w_row_bytes = ulong(in_dim);
    const ulong s_row_bytes = ulong(in_dim / 32u);
    const ulong e_row = ulong(expert) * args.out_dim + o;
    float v[16];
    ink2_mxfp8_block<2, 8>(
        w2 + e_row * w_row_bytes, w_row_bytes, w2_scale + e_row * s_row_bytes, s_row_bytes,
        act, in_dim, c_ids, n_r, lane, v);
    if (lane == 0u) {
        for (uint wr = 0u; wr < 2u; ++wr) {
            for (uint rr = 0u; rr < n_r; ++rr) {
                slots_out[ulong(c_ids[rr]) * args.out_dim + o + wr] = v[wr * 8u + rr];
            }
        }
    }
}

// ---- 12c. Tiled entries: an expert's columns as a simdgroup-matrix GEMM ----
//
// A long pass routes dozens of columns to an expert, and then the product is
// a GEMM: `moe_plan` with a batch of 64 makes each work item one expert and up
// to 64 of its columns, and a threadgroup multiplies a 64-column by 64-weight-
// row tile of it. The layout is the shared GEMM set's 64x64 tile (four
// simdgroups 2x2, each a 4x4 grid of 8x8 accumulators, k staged 16 deep, the
// weights transposed to `[k][row]`, rows padded by four floats); what differs
// is the operands: the activation rows are gathered through the item's
// columns, the weights are decoded to f32 with their group scale while they
// are staged, and the epilogue scatters to the routed columns - fused with
// the softcapped SwiGLU for W13, whose gate and up rows are adjacent tile
// columns. Every output is its row's multiply-accumulates over 8-k steps in k
// order, so a column's value does not depend on which columns share its tile.
// `in_dim % 16 == 0`; the host sends other shapes to the batched entries.
// W13: grid (ceil(2 * inter_dim / 64), max_items, 1), block 128.
// W2:  grid (ceil(out_dim / 64), max_items, 1), block 128.

#define INK2_TILE_M 64u
#define INK2_TILE_N 64u
#define INK2_TILE_K 16u
#define INK2_TILE_THREADS 128u
#define INK2_TILE_LDA (INK2_TILE_K + 4u)
#define INK2_TILE_LDB (INK2_TILE_N + 4u)
#define INK2_TILE_SMEM (INK2_TILE_M * INK2_TILE_LDA + INK2_TILE_K * INK2_TILE_LDB)

// Four consecutive weights of one row, scaled, for the tile's staging.
struct ink2_nvfp4_reader {
    device const uchar *w;   // the expert's first row
    device const uchar *sc;
    uint k;
    float4 load4(uint row, uint kk) const {
        device const uchar *p = w + ulong(row) * (k / 2u) + kk / 2u;
        const uint b0 = p[0];
        const uint b1 = p[1];
        const float s = ink2_e4m3_fast(uint(sc[ulong(row) * (k / 16u) + kk / 16u]));
        return float4(INK2_E2M1_LUT[b0 & 0xFu], INK2_E2M1_LUT[b0 >> 4u],
                      INK2_E2M1_LUT[b1 & 0xFu], INK2_E2M1_LUT[b1 >> 4u]) * s;
    }
};

struct ink2_mxfp8_reader {
    device const uchar *w;
    device const uchar *sc;
    uint k;
    float4 load4(uint row, uint kk) const {
        const uchar4 b = *((device const uchar4 *)(w + ulong(row) * k + kk));
        return ink2_e4m3_x4(b) * ink2_ue8m0_to_f32(uint(sc[ulong(row) * (k / 32u) + kk / 32u]));
    }
};

// One 64-column by 64-weight-row tile of a work item. `ec` is the item's `m`
// routed columns; activation row = `ec[i] / x_div`, output row = `ec[i]`.
// SWIGLU: tile columns `2j, 2j + 1` are gate and up, and `out[col][j]` gets
// the softcapped SwiGLU; otherwise `out[col][n]` gets the scaled product.
template <typename R, uint SWIGLU>
static inline void ink2_moe_tile(
        const R w, uint n, uint k,
        device const float *x, uint x_div,
        device const uint *ec, uint m,
        device float *out, uint out_stride,
        float outer, float gate_cap, float up_cap,
        device const float *expert_weights,
        threadgroup float *smem,
        uint n0, ushort tid, ushort lane, ushort sg) {
    const uint BM = INK2_TILE_M;
    const uint BK = INK2_TILE_K;
    const uint NT = INK2_TILE_THREADS;
    const uint LDA = INK2_TILE_LDA;
    const uint LDB = INK2_TILE_LDB;
    threadgroup float *sa = smem;            // [BM columns][BK k]
    threadgroup float *sb = smem + BM * LDA; // [BK k][BN weight rows]
    const uint r0 = (sg >> 1u) * 32u;
    const uint c0 = (sg & 1u) * 32u;
    // A thread's two float4 slots of each operand: slot e is row e / 4, k (e % 4) * 4.
    const uint SLOTS = BM * BK / 4u / NT;
    float4 ra[SLOTS];
    float4 rb[SLOTS];
    uint xrow[SLOTS];
    INK2_UNROLL for (uint q = 0u; q < SLOTS; ++q) {
        const uint rr = (tid + q * NT) / (BK / 4u);
        xrow[q] = rr < m ? ec[rr] / x_div : 0u;
    }
    auto fetch = [&](uint k0) {
        INK2_UNROLL for (uint q = 0u; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            ra[q] = rr < m ? *((device const float4 *)(x + ulong(xrow[q]) * k + k0 + kk))
                           : float4(0.0f);
            rb[q] = n0 + rr < n ? w.load4(n0 + rr, k0 + kk) : float4(0.0f);
        }
    };

    simdgroup_float8x8 acc[4][4];
    INK2_UNROLL for (uint i = 0u; i < 4u; ++i) {
        INK2_UNROLL for (uint j = 0u; j < 4u; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    fetch(0u);
    for (uint k0 = 0u; k0 < k; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INK2_UNROLL for (uint q = 0u; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            *((threadgroup float4 *)(sa + rr * LDA + kk)) = ra[q];
            sb[(kk + 0u) * LDB + rr] = rb[q].x;
            sb[(kk + 1u) * LDB + rr] = rb[q].y;
            sb[(kk + 2u) * LDB + rr] = rb[q].z;
            sb[(kk + 3u) * LDB + rr] = rb[q].w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        INK2_UNROLL for (uint kk = 0u; kk < BK; kk += 8u) {
            simdgroup_float8x8 a[4];
            simdgroup_float8x8 b[4];
            INK2_UNROLL for (uint i = 0u; i < 4u; ++i) {
                simdgroup_load(a[i], sa + (r0 + 8u * i) * LDA + kk, LDA);
            }
            INK2_UNROLL for (uint j = 0u; j < 4u; ++j) {
                // b[j](r, c) = W[n0 + c0 + 8j + c][k0 + kk + r].
                simdgroup_load(b[j], sb + kk * LDB + c0 + 8u * j, LDB);
            }
            INK2_UNROLL for (uint i = 0u; i < 4u; ++i) {
                INK2_UNROLL for (uint j = 0u; j < 4u; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
                }
            }
        }
        if (k0 + BK < k) {
            fetch(k0 + BK);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The staged operands are done with; each simdgroup writes its fragments
    // through its own 8x8 slot at the start of the tile.
    threadgroup float *slot = smem + sg * 64u;
    INK2_UNROLL for (uint i = 0u; i < 4u; ++i) {
        INK2_UNROLL for (uint j = 0u; j < 4u; ++j) {
            simdgroup_store(acc[i][j], slot, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                const uint gm = r0 + 8u * i + (e >> 3u);
                const uint gn = n0 + c0 + 8u * j + (e & 7u);
                if (gm < m && gn < n) {
                    const uint col = ec[gm];
                    if (SWIGLU != 0u) {
                        if ((e & 1u) == 0u) {
                            out[ulong(col) * out_stride + gn / 2u] = ink2_swiglu_capped(
                                slot[e] * outer, slot[e + 1u] * outer,
                                gate_cap, up_cap, expert_weights[col]);
                        }
                    } else {
                        out[ulong(col) * out_stride + gn] = slot[e] * outer;
                    }
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

kernel void kernel_inkling2_moe_nvfp4_w13_tile(
        constant ink2_args_moe_w13 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const uchar *w13 [[buffer(2)]],
        device const uchar *w13_scale [[buffer(3)]],
        device const float *w13_amax [[buffer(4)]],
        device const float *w13_denom [[buffer(5)]],
        device const float *expert_weights [[buffer(6)]],
        device const uint *cols [[buffer(7)]],
        device const uint *items [[buffer(8)]],
        device float *out [[buffer(9)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[INK2_TILE_SMEM];
    const uint n = 2u * args.inter_dim;
    const uint k = args.in_dim;
    const uint n0 = group.x * INK2_TILE_N;
    if (n0 >= n || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    const ulong e_row = ulong(expert) * n;
    const ink2_nvfp4_reader w = {w13 + e_row * ulong(k/ 2u), w13_scale + e_row * ulong(k / 16u), k};
    ink2_moe_tile<ink2_nvfp4_reader, 1>(
        w, n, k, x, args.top_k, ec, packed >> 24u, out, args.inter_dim,
        ink2_div(w13_amax[expert], w13_denom[0]), args.gate_cap, args.up_cap, expert_weights,
        smem, n0, tid, lane, sg);
}

kernel void kernel_inkling2_moe_nvfp4_w2_tile(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const float *w2_amax [[buffer(4)]],
        device const float *w2_denom [[buffer(5)]],
        device const uint *cols [[buffer(6)]],
        device const uint *items [[buffer(7)]],
        device float *slots_out [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[INK2_TILE_SMEM];
    const uint n = args.out_dim;
    const uint k = args.inter_dim;
    const uint n0 = group.x * INK2_TILE_N;
    if (n0 >= n || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    const ulong e_row = ulong(expert) * n;
    const ink2_nvfp4_reader w = {w2 + e_row * ulong(k/ 2u), w2_scale + e_row * ulong(k / 16u), k};
    ink2_moe_tile<ink2_nvfp4_reader, 0>(
        w, n, k, act, 1u, ec, packed >> 24u, slots_out, args.out_dim,
        ink2_div(w2_amax[expert], w2_denom[0]), 0.0f, 0.0f, act,
        smem, n0, tid, lane, sg);
}

kernel void kernel_inkling2_moe_mxfp8_w13_tile(
        constant ink2_args_moe_w13 &args [[buffer(0)]],
        device const float *x [[buffer(1)]],
        device const uchar *w13 [[buffer(2)]],
        device const uchar *w13_scale [[buffer(3)]],
        device const float *expert_weights [[buffer(4)]],
        device const uint *cols [[buffer(5)]],
        device const uint *items [[buffer(6)]],
        device float *out [[buffer(7)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[INK2_TILE_SMEM];
    const uint n = 2u * args.inter_dim;
    const uint k = args.in_dim;
    const uint n0 = group.x * INK2_TILE_N;
    if (n0 >= n || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    const ulong e_row = ulong(expert) * n;
    const ink2_mxfp8_reader w = {w13 + e_row * ulong(k), w13_scale + e_row * ulong(k / 32u), k};
    ink2_moe_tile<ink2_mxfp8_reader, 1>(
        w, n, k, x, args.top_k, ec, packed >> 24u, out, args.inter_dim,
        1.0f, args.gate_cap, args.up_cap, expert_weights,
        smem, n0, tid, lane, sg);
}

kernel void kernel_inkling2_moe_mxfp8_w2_tile(
        constant ink2_args_moe_w2 &args [[buffer(0)]],
        device const float *act [[buffer(1)]],
        device const uchar *w2 [[buffer(2)]],
        device const uchar *w2_scale [[buffer(3)]],
        device const uint *cols [[buffer(4)]],
        device const uint *items [[buffer(5)]],
        device float *slots_out [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[INK2_TILE_SMEM];
    const uint n = args.out_dim;
    const uint k = args.inter_dim;
    const uint n0 = group.x * INK2_TILE_N;
    if (n0 >= n || group.y >= items[0]) {
        return;
    }
    const uint expert = items[1u + 2u * group.y];
    const uint packed = items[2u + 2u * group.y];
    device const uint *ec = cols + ulong(expert) * args.rows + (packed & 0xFFFFFFu);
    const ulong e_row = ulong(expert) * n;
    const ink2_mxfp8_reader w = {w2 + e_row * ulong(k), w2_scale + e_row * ulong(k / 32u), k};
    ink2_moe_tile<ink2_mxfp8_reader, 0>(
        w, n, k, act, 1u, ec, packed >> 24u, slots_out, args.out_dim,
        1.0f, 0.0f, 0.0f, act,
        smem, n0, tid, lane, sg);
}

// Sums a row's `top_k` slots in slot order; compiled safe so it stays that
// order whichever entries wrote the slots.
#pragma METAL fp math_mode(safe)
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
        acc = acc + slots_in[ulong(row * args.top_k + s) * args.out_dim + o];
    }
    out[tid] = acc;
}
#pragma METAL fp math_mode(fast)

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

// Stores `rows` f32 rows as halves at their positions' slots of a context
// cache (a ring of `kv_cap` slots on a sliding-window layer): the drafter's
// values, which need no norm or rotation on the way. Grid: one thread an
// element, block 256.
struct ink2_args_mtp_store {
    uint rows;
    uint width;
    uint pos0;
    uint kv_cap;
    uint is_swa;
};
static_assert(sizeof(ink2_args_mtp_store) == 20, "ink2_args_mtp_store layout");

kernel void kernel_inkling2_mtp_store_f16(
        constant ink2_args_mtp_store &args [[buffer(0)]],
        device const float *src [[buffer(1)]],
        device half *cache [[buffer(2)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.rows * args.width) {
        return;
    }
    const uint row = tid / args.width;
    const uint c = tid - row * args.width;
    const uint pos = args.pos0 + row;
    const uint slot = args.is_swa != 0u ? (pos % args.kv_cap) : pos;
    cache[ulong(slot) * args.width + c] = half(src[tid]);
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
