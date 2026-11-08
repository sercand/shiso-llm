// GGML block-quantized weights for the qwen4exp kernel set: the dense
// projections, the routed expert stages and the vocabulary gather of a GGUF
// checkpoint whose tensors stay in their stored blocks.
//
// Every entry reads a weight row as consecutive 32-weight sub-blocks, decoded
// in registers by `q4e_g_deq32` and multiplied against single-precision
// activations; nothing widens a matrix in memory and no activation is
// quantized. A row's dot product is the same sum in the same order whatever
// row count or column count shares the launch: the lanes split the
// sub-blocks by index and `q4e_warp_sum` folds them in a fixed order, and the
// columns a lane carries are independent accumulators.
//
// This is the correctness tier: one simdgroup an output row, a few
// activation rows a pass. It reads each weight sub-block once per pass of
// `Q4E_G_COLS` activation rows and is bound by decode arithmetic, not by
// bandwidth.
//
// The 256-weight formats reuse the generic set's `deq32_*` decoders and grid
// tables, which come earlier in the same translation unit.

#include <metal_stdlib>
using namespace metal;

// GGML type ids of the formats decoded here.
constant constexpr uint Q4E_G_F32 = 0;
constant constexpr uint Q4E_G_Q8_0 = 8;
constant constexpr uint Q4E_G_Q6_K = 14;
constant constexpr uint Q4E_G_IQ3_XXS = 18;
constant constexpr uint Q4E_G_IQ4_NL = 20;
constant constexpr uint Q4E_G_IQ2_S = 22;
constant constexpr uint Q4E_G_IQ4_XS = 23;
constant constexpr uint Q4E_G_Q2_0 = 42;

/// Output rows a threadgroup covers: one simdgroup each.
#define Q4E_G_ROWS 4u
/// Activation rows (or routed columns) one pass carries in registers.
#define Q4E_G_COLS 8u

/// Bytes of one stored row of `n` weights.
template <uint T>
static inline ulong q4e_g_row_bytes(uint n) {
    switch (T) {
        case Q4E_G_F32: return ulong(n) * 4ul;
        case Q4E_G_Q8_0: return ulong(n / 32u) * 34ul;
        case Q4E_G_IQ4_NL: return ulong(n / 32u) * 18ul;
        case Q4E_G_Q2_0: return ulong(n / 64u) * 18ul;
        case Q4E_G_Q6_K: return ulong(n / 256u) * 210ul;
        case Q4E_G_IQ2_S: return ulong(n / 256u) * 82ul;
        case Q4E_G_IQ3_XXS: return ulong(n / 256u) * 98ul;
        case Q4E_G_IQ4_XS: return ulong(n / 256u) * 136ul;
        default: return 0ul;
    }
}

static inline float q4e_g_half(device const uchar *p) {
    return float(*(device const half *)p);
}

/// Sub-block `s` (weights `32 s .. 32 s + 31`) of a row starting at `row`.
///
/// Every block size here is even, so the half-precision scales the decoders
/// read are 2-byte aligned whenever the row is.
template <uint T>
static inline void q4e_g_deq32(device const uchar *row, uint s, thread float4 *qv) {
    float2 ab;
    if (T == Q4E_G_F32) {
        // Single precision is here for the stream mixes' injections, whose
        // other tier takes one row a launch.
        device const float4 *v = (device const float4 *)(row + ulong(s) * 128ul);
        for (short k = 0; k < 8; ++k) {
            qv[k] = v[k];
        }
    } else if (T == Q4E_G_Q8_0) {
        device const uchar *b = row + ulong(s) * 34ul;
        const float d = q4e_g_half(b);
        device const char *q = (device const char *)(b + 2);
        for (short j = 0; j < 32; ++j) {
            qv[j / 4][j % 4] = d * float(q[j]);
        }
    } else if (T == Q4E_G_IQ4_NL) {
        device const uchar *b = row + ulong(s) * 18ul;
        const float d = q4e_g_half(b);
        for (short j = 0; j < 16; ++j) {
            const uchar q = b[2 + j];
            qv[j / 4][j % 4] = d * float(IQ4NL[q & 0xF]);
            qv[4 + j / 4][j % 4] = d * float(IQ4NL[q >> 4]);
        }
    } else if (T == Q4E_G_Q2_0) {
        // 64 weights a block, four 2-bit codes a byte low bits first; the
        // codes 0..3 stand for -1..2 steps of the block scale.
        device const uchar *b = row + ulong(s / 2u) * 18ul;
        const float d = q4e_g_half(b);
        const uint base = (s & 1u) * 32u;
        for (short j = 0; j < 32; ++j) {
            const uint e = base + uint(j);
            const uint q = (uint(b[2 + e / 4u]) >> ((e % 4u) * 2u)) & 3u;
            qv[j / 4][j % 4] = d * (float(q) - 1.0f);
        }
    } else if (T == Q4E_G_Q6_K) {
        // A 256-weight block is two halves of 128; a 32-weight run `k` of a
        // half takes its low nibbles from `ql` (the high nibble for k >= 2),
        // its two high bits from `qh` at shift 2k, and one 8-bit scale per 16.
        device const uchar *b = row + ulong(s / 8u) * 210ul;
        const uint ss = s % 8u;
        const uint n = ss / 4u;
        const uint k = ss % 4u;
        device const uchar *ql = b + n * 64u + (k & 1u) * 32u;
        device const uchar *qh = b + 128u + n * 32u;
        device const char *sc = (device const char *)(b + 192u) + n * 8u + 2u * k;
        const float d = q4e_g_half(b + 208u);
        const uint lshift = (k >> 1u) * 4u;
        const uint hshift = 2u * k;
        for (short l = 0; l < 32; ++l) {
            const int q = int(((uint(ql[l]) >> lshift) & 0xFu) | (((uint(qh[l]) >> hshift) & 3u) << 4u)) - 32;
            qv[l / 4][l % 4] = d * float(sc[l / 16]) * float(q);
        }
    } else if (T == Q4E_G_IQ2_S) {
        deq32_iq2_s((device const shs_block_iq2_s *)(row + ulong(s / 8u) * 82ul), short(s % 8u), qv, ab);
    } else if (T == Q4E_G_IQ3_XXS) {
        deq32_iq3_xxs((device const shs_block_iq3_xxs *)(row + ulong(s / 8u) * 98ul), short(s % 8u), qv, ab);
    } else if (T == Q4E_G_IQ4_XS) {
        deq32_iq4_xs((device const shs_block_iq4_xs *)(row + ulong(s / 8u) * 136ul), short(s % 8u), qv, ab);
    }
}

static inline float q4e_g_dot32(thread const float4 *qv, device const float *x) {
    device const float4 *x4 = (device const float4 *)x;
    float4 acc = 0.0f;
    for (short k = 0; k < 8; ++k) {
        acc += qv[k] * x4[k];
    }
    return (acc[0] + acc[1]) + (acc[2] + acc[3]);
}

/// `out[c] = row . x_c` for up to `Q4E_G_COLS` activation rows, one lane's
/// share: the sub-blocks `lane, lane + 32, ...`, each decoded once.
template <uint T>
static inline void q4e_g_row_cols(device const uchar *row, uint in_dim, device const float *x,
                                  thread const ulong *xoff, uint n_cols, uint lane,
                                  thread float *acc) {
    const uint nsub = in_dim / 32u;
    for (uint u = lane; u < nsub; u += 32u) {
        float4 qv[8];
        q4e_g_deq32<T>(row, u, qv);
        for (uint c = 0; c < Q4E_G_COLS; ++c) {
            if (c < n_cols) {
                acc[c] += q4e_g_dot32(qv, x + xoff[c] + u * 32u);
            }
        }
    }
}

/// `out[r * out_dim + o] = W[o] . x[r]` over every row `r < rows`.
///
/// Grid `(ceil(out_dim / Q4E_G_ROWS), ceil(rows / Q4E_G_COLS))`, block
/// `32 * Q4E_G_ROWS`.
template <uint T>
kernel void q4e_g_matvec_kernel(
        device float *out [[buffer(0)]],
        device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]],
        constant uint &in_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        constant uint &rows [[buffer(5)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    const uint o = group.x * Q4E_G_ROWS + simd;
    if (o >= out_dim) {
        return;
    }
    const uint r0 = group.y * Q4E_G_COLS;
    const uint n_cols = min(Q4E_G_COLS, rows - r0);
    device const uchar *row = w + ulong(o) * q4e_g_row_bytes<T>(in_dim);
    ulong xoff[Q4E_G_COLS];
    for (uint c = 0; c < Q4E_G_COLS; ++c) {
        xoff[c] = ulong(r0 + min(c, n_cols - 1u)) * in_dim;
    }
    float acc[Q4E_G_COLS] = {0.0f};
    q4e_g_row_cols<T>(row, in_dim, x, xoff, n_cols, lane, acc);
    for (uint c = 0; c < Q4E_G_COLS; ++c) {
        const float v = q4e_warp_sum(acc[c]);
        if (lane == 0u && c < n_cols) {
            out[ulong(r0 + c) * out_dim + o] = v;
        }
    }
}

/// One routed matrix over an active expert's sorted columns, the shared body
/// of the gate/up and down stages. `slot` is the expert's place in the bank
/// and `stride` the bytes between slots: one expert's own bytes in a resident
/// bank, the pool's slot width (which may be wider) in a streamed one.
template <uint T>
static inline void q4e_g_expert_rows(device float *out, device const uchar *w,
                                     device const float *x, device const int *sorted, int c0,
                                     int c1, ulong slot, ulong stride, uint o, uint in_dim,
                                     uint out_dim, uint x_div, uint lane) {
    device const uchar *row = w + slot * stride + ulong(o) * q4e_g_row_bytes<T>(in_dim);
    for (int c = c0; c < c1; c += int(Q4E_G_COLS)) {
        const uint n_cols = min(Q4E_G_COLS, uint(c1 - c));
        ulong xoff[Q4E_G_COLS];
        uint cols[Q4E_G_COLS];
        for (uint k = 0; k < Q4E_G_COLS; ++k) {
            cols[k] = uint(sorted[c + int(min(k, n_cols - 1u))]);
            xoff[k] = ulong(cols[k] / x_div) * in_dim;
        }
        float acc[Q4E_G_COLS] = {0.0f};
        q4e_g_row_cols<T>(row, in_dim, x, xoff, n_cols, lane, acc);
        for (uint k = 0; k < Q4E_G_COLS; ++k) {
            const float v = q4e_warp_sum(acc[k]);
            if (lane == 0u && k < n_cols) {
                out[ulong(cols[k]) * out_dim + o] = v;
            }
        }
    }
}

/// The routed gate and up projections: `q4e_affine_experts_gate_up_kernel`'s
/// contract without the affine parameter arrays.
///
/// `out_gate, out_up, gate_w, up_w, x, sorted, bounds, active, map, in_dim,
/// out_dim, n_expert, n_used, gate_stride, up_stride`. Grid `(ceil(out_dim / Q4E_G_ROWS), groups)`,
/// block `32 * Q4E_G_ROWS`. A column `col` reads token `col / n_used`.
template <uint T>
kernel void q4e_g_experts_gate_up_kernel(
        device float *out_gate [[buffer(0)]],
        device float *out_up [[buffer(1)]],
        device const uchar *gate_w [[buffer(2)]],
        device const uchar *up_w [[buffer(3)]],
        device const float *x [[buffer(4)]],
        device const int *sorted [[buffer(5)]],
        device const int *bounds [[buffer(6)]],
        device const int *active [[buffer(7)]],
        device const uint *map [[buffer(8)]],
        constant uint &in_dim [[buffer(9)]],
        constant uint &out_dim [[buffer(10)]],
        constant uint &n_expert [[buffer(11)]],
        constant uint &n_used [[buffer(12)]],
        constant uint &gate_stride [[buffer(13)]],
        constant uint &up_stride [[buffer(14)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint o = group.x * Q4E_G_ROWS + simd;
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const ulong slot = map[expert];
    q4e_g_expert_rows<T>(out_gate, gate_w, x, sorted, c0, c1, slot, gate_stride, o, in_dim,
                         out_dim, n_used, lane);
    q4e_g_expert_rows<T>(out_up, up_w, x, sorted, c0, c1, slot, up_stride, o, in_dim, out_dim,
                         n_used, lane);
}

/// The routed down projection: `out, down_w, mid, sorted, bounds, active,
/// map, in_dim, out_dim, n_expert, n_used, stride`. A column reads its own row of
/// `mid`. Grid and block as the gate/up stage.
template <uint T>
kernel void q4e_g_experts_down_kernel(
        device float *out [[buffer(0)]],
        device const uchar *down_w [[buffer(1)]],
        device const float *mid [[buffer(2)]],
        device const int *sorted [[buffer(3)]],
        device const int *bounds [[buffer(4)]],
        device const int *active [[buffer(5)]],
        device const uint *map [[buffer(6)]],
        constant uint &in_dim [[buffer(7)]],
        constant uint &out_dim [[buffer(8)]],
        constant uint &n_expert [[buffer(9)]],
        constant uint &n_used [[buffer(10)]],
        constant uint &stride [[buffer(11)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint o = group.x * Q4E_G_ROWS + simd;
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    q4e_g_expert_rows<T>(out, down_w, mid, sorted, c0, c1, map[expert], stride, o, in_dim,
                         out_dim, 1u, lane);
}

/// Vocabulary rows: `out[t * n_embd + i] = W[tokens[t]][i]`, the row decoded
/// whole. `out, tokens, w, n_tokens, n_embd`; grid `(n_tokens)`, block 32.
template <uint T>
kernel void q4e_g_embed_kernel(
        device float *out [[buffer(0)]],
        device const int *tokens [[buffer(1)]],
        device const uchar *w [[buffer(2)]],
        constant uint &n_tokens [[buffer(3)]],
        constant uint &n_embd [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]]) {
    const uint t = group.x;
    if (t >= n_tokens) {
        return;
    }
    device const uchar *row = w + ulong(tokens[t]) * q4e_g_row_bytes<T>(n_embd);
    device float4 *dst = (device float4 *)(out + ulong(t) * n_embd);
    for (uint u = lane; u < n_embd / 32u; u += 32u) {
        float4 qv[8];
        q4e_g_deq32<T>(row, u, qv);
        for (short k = 0; k < 8; ++k) {
            dst[u * 8u + k] = qv[k];
        }
    }
}

#define Q4E_G_INSTANTIATE(T, NAME)                                                              \
    template [[host_name("q4e_g_matvec_" NAME "_kernel")]] kernel void q4e_g_matvec_kernel<T>(  \
        device float *, device const uchar *, device const float *, constant uint &,            \
        constant uint &, constant uint &, uint3, uint, uint);                                   \
    template [[host_name("q4e_g_experts_gate_up_" NAME "_kernel")]] kernel void                 \
    q4e_g_experts_gate_up_kernel<T>(device float *, device float *, device const uchar *,       \
                                    device const uchar *, device const float *,                 \
                                    device const int *, device const int *, device const int *, \
                                    device const uint *, constant uint &, constant uint &,      \
                                    constant uint &, constant uint &, constant uint &,          \
                                    constant uint &, uint3, uint, uint);                        \
    template [[host_name("q4e_g_experts_down_" NAME "_kernel")]] kernel void                    \
    q4e_g_experts_down_kernel<T>(device float *, device const uchar *, device const float *,    \
                                 device const int *, device const int *, device const int *,    \
                                 device const uint *, constant uint &, constant uint &,         \
                                 constant uint &, constant uint &, constant uint &, uint3,      \
                                 uint, uint);                                                   \
    template [[host_name("q4e_g_embed_" NAME "_kernel")]] kernel void q4e_g_embed_kernel<T>(    \
        device float *, device const int *, device const uchar *, constant uint &,              \
        constant uint &, uint3, uint);

Q4E_G_INSTANTIATE(Q4E_G_F32, "f32")
Q4E_G_INSTANTIATE(Q4E_G_Q8_0, "q8_0")
Q4E_G_INSTANTIATE(Q4E_G_Q6_K, "q6_k")
Q4E_G_INSTANTIATE(Q4E_G_IQ3_XXS, "iq3_xxs")
Q4E_G_INSTANTIATE(Q4E_G_IQ4_NL, "iq4_nl")
Q4E_G_INSTANTIATE(Q4E_G_IQ2_S, "iq2_s")
Q4E_G_INSTANTIATE(Q4E_G_IQ4_XS, "iq4_xs")
Q4E_G_INSTANTIATE(Q4E_G_Q2_0, "q2_0")
