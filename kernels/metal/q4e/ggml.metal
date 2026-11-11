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
/// share: the sub-blocks `lane, lane + lpr, ...`, each decoded once.
template <uint T>
static inline void q4e_g_row_cols(device const uchar *row, uint in_dim, device const float *x,
                                  thread const ulong *xoff, uint n_cols, uint lane, uint lpr,
                                  thread float *acc) {
    const uint nsub = in_dim / 32u;
    for (uint u = lane; u < nsub; u += lpr) {
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
    q4e_g_row_cols<T>(row, in_dim, x, xoff, n_cols, lane, 32u, acc);
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
        q4e_g_row_cols<T>(row, in_dim, x, xoff, n_cols, lane, 32u, acc);
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

// ---------------------------------------------------------------------------
// Step passes (a decode step or a verify).
//
// The entries above give a simdgroup to one output row, so a contraction of
// 640 weights leaves twelve of its lanes idle and the other twenty a single
// sub-block each: the launch is bound by how many threadgroups it starts, not
// by the weights it reads. Measured on an M1 Max over thirty routed columns,
// a row cost the same 18 ns whether it held 640 weights or 2560. These give
// `lpr` lanes to a row - `32 / lpr` rows ("units") a simdgroup - so a lane
// walks several sub-blocks and a threadgroup covers many rows. A unit's lane
// `j` takes sub-blocks `j, j + lpr, ...`; the `lpr` lanes fold by butterfly.
// `lpr` is the host's choice from the contraction width alone, so a row's sum
// does not depend on how many activation rows share the pass, and a verify's
// rows are a single-row step's.
// ---------------------------------------------------------------------------

/// Simdgroups a step threadgroup runs.
#define Q4E_GS_WARPS 4u

/// Words of threadgroup memory a format's decode tables take: the `IQ2_S`
/// grid as word pairs, or the `IQ3_XXS` grid and the sign table after it.
///
/// A lookup in a `constant` table is a load from device memory behind the
/// kernel's own; staged once a threadgroup, the tables sit where the lanes
/// reach them fastest.
template <uint T>
constexpr uint q4e_gs_tab_words() {
    return T == Q4E_G_IQ2_S ? 2048u : (T == Q4E_G_IQ3_XXS ? 384u : 1u);
}

/// Stages format `T`'s tables in `tab`, every thread a share; the caller
/// puts a barrier after it.
template <uint T>
static inline void q4e_gs_load_tab(threadgroup uint *tab, uint tid, uint threads) {
    if (T == Q4E_G_IQ2_S) {
        for (uint i = tid; i < 1024u; i += threads) {
            const ulong g = IQ2S_GRID[i];
            tab[2u * i] = uint(g);
            tab[2u * i + 1u] = uint(g >> 32ul);
        }
    } else if (T == Q4E_G_IQ3_XXS) {
        for (uint i = tid; i < 384u; i += threads) {
            tab[i] = i < 256u ? IQ3XXS_GRID[i] : uint(KSIGNS_IQ2XS[i - 256u]);
        }
    }
}

/// A grid word's four bytes as floats, each negated where its bit of `signs`
/// (from `bit` up) is set.
static inline float4 q4e_gs_signed4(uint grid, uint signs, uint bit) {
    const float4 v = float4(as_type<uchar4>(grid));
    const uint4 mask = uint4(bit, bit << 1u, bit << 2u, bit << 3u);
    return select(v, -v, (uint4(signs) & mask) != uint4(0u));
}

/// Sub-block `s` of a row as 32 codes and the scale its weights are the
/// codes times.
///
/// `q4e_g_deq32` reads a byte and multiplies a scale in per weight, which is
/// what bound the entries above: `IQ4_NL` read 5 GB/s and `Q6_K` 18 on an M1
/// Max. Here the bytes arrive four to a load through packed vectors (a block
/// is only two-byte aligned), the codes are formed four at a time, and the
/// block's scale is applied once to the 32-term sum; a scale that changes
/// within the sub-block rides on its codes.
template <uint T>
static inline float q4e_gs_codes32(device const uchar *row, uint s, thread float4 *qv,
                                   threadgroup const uint *tab) {
    if (T == Q4E_G_Q8_0) {
        device const uchar *b = row + ulong(s) * 34ul;
        device const packed_char4 *q = (device const packed_char4 *)(b + 2);
        for (short k = 0; k < 8; ++k) {
            qv[k] = float4(char4(q[k]));
        }
        return q4e_g_half(b);
    } else if (T == Q4E_G_IQ4_NL) {
        device const uchar *b = row + ulong(s) * 18ul;
        device const packed_uchar4 *q = (device const packed_uchar4 *)(b + 2);
        for (short k = 0; k < 4; ++k) {
            const uchar4 v = uchar4(q[k]);
            const uchar4 lo = v & uchar(0xF);
            const uchar4 hi = v >> uchar(4);
            qv[k] = float4(float(IQ4NL[lo.x]), float(IQ4NL[lo.y]), float(IQ4NL[lo.z]),
                           float(IQ4NL[lo.w]));
            qv[4 + k] = float4(float(IQ4NL[hi.x]), float(IQ4NL[hi.y]), float(IQ4NL[hi.z]),
                               float(IQ4NL[hi.w]));
        }
        return q4e_g_half(b);
    } else if (T == Q4E_G_IQ4_XS) {
        // `IQ4_NL`'s codes, sixteen bytes a sub-block of a 256-weight block,
        // under a 6-bit scale split across two planes.
        device const uchar *b = row + ulong(s / 8u) * 136ul;
        const uint ss = s % 8u;
        const uint scales_h = uint(*(device const ushort *)(b + 2));
        const uint ls = ((uint(b[4u + ss / 2u]) >> (4u * (ss % 2u))) & 0xFu)
                      | (((scales_h >> (2u * ss)) & 3u) << 4u);
        device const packed_uchar4 *q = (device const packed_uchar4 *)(b + 8u + 16u * ss);
        for (short k = 0; k < 4; ++k) {
            const uchar4 v = uchar4(q[k]);
            const uchar4 lo = v & uchar(0xF);
            const uchar4 hi = v >> uchar(4);
            qv[k] = float4(float(IQ4NL[lo.x]), float(IQ4NL[lo.y]), float(IQ4NL[lo.z]),
                           float(IQ4NL[lo.w]));
            qv[4 + k] = float4(float(IQ4NL[hi.x]), float(IQ4NL[hi.y]), float(IQ4NL[hi.z]),
                               float(IQ4NL[hi.w]));
        }
        return q4e_g_half(b) * (float(ls) - 32.0f);
    } else if (T == Q4E_G_Q2_0) {
        device const uchar *b = row + ulong(s / 2u) * 18ul;
        device const packed_uchar4 *q = (device const packed_uchar4 *)(b + 2u + (s & 1u) * 8u);
        for (short k = 0; k < 2; ++k) {
            const uchar4 v = uchar4(q[k]);
            for (short i = 0; i < 4; ++i) {
                const uint4 c = (uint4(uint(v[i])) >> uint4(0u, 2u, 4u, 6u)) & 3u;
                qv[4 * k + i] = float4(c) - 1.0f;
            }
        }
        return q4e_g_half(b);
    } else if (T == Q4E_G_Q6_K) {
        device const uchar *b = row + ulong(s / 8u) * 210ul;
        const uint ss = s % 8u;
        const uint n = ss / 4u;
        const uint k4 = ss % 4u;
        device const packed_uchar4 *ql =
            (device const packed_uchar4 *)(b + n * 64u + (k4 & 1u) * 32u);
        device const packed_uchar4 *qh = (device const packed_uchar4 *)(b + 128u + n * 32u);
        device const char *sc = (device const char *)(b + 192u) + n * 8u + 2u * k4;
        const float sc0 = float(sc[0]);
        const float sc1 = float(sc[1]);
        const uchar lshift = uchar((k4 >> 1u) * 4u);
        const uchar hshift = uchar(2u * k4);
        for (short k = 0; k < 8; ++k) {
            const uchar4 lo = (uchar4(ql[k]) >> lshift) & uchar(0x0F);
            const uchar4 hi = (uchar4(qh[k]) >> hshift) & uchar(3);
            qv[k] = (float4(lo | (hi << uchar(4))) - 32.0f) * (k < 4 ? sc0 : sc1);
        }
        return q4e_g_half(b + 208u);
    } else if (T == Q4E_G_IQ2_S) {
        device const uchar *b = row + ulong(s / 8u) * 82ul;
        const uint ss = s % 8u;
        const uchar4 idx = uchar4(*(device const packed_uchar4 *)(b + 2u + 4u * ss));
        const uchar4 sg = uchar4(*(device const packed_uchar4 *)(b + 34u + 4u * ss));
        const uint qh = b[66u + ss];
        const uint sc = b[74u + ss];
        const float s0 = 0.5f + float(sc & 0xFu);
        const float s1 = 0.5f + float(sc >> 4u);
        for (short l = 0; l < 4; ++l) {
            const uint gi = uint(idx[l]) | ((qh << (8u - 2u * uint(l))) & 0x300u);
            const uint signs = uint(sg[l]);
            const float sl = l < 2 ? s0 : s1;
            qv[2 * l] = q4e_gs_signed4(tab[2u * gi], signs, 1u) * sl;
            qv[2 * l + 1] = q4e_gs_signed4(tab[2u * gi + 1u], signs, 16u) * sl;
        }
        return q4e_g_half(b) * 0.25f;
    } else if (T == Q4E_G_IQ3_XXS) {
        device const uchar *b = row + ulong(s / 8u) * 98ul;
        const uint ss = s % 8u;
        const uint aux = as_type<uint>(uchar4(*(device const packed_uchar4 *)(b + 66u + 4u * ss)));
        device const packed_uchar4 *q = (device const packed_uchar4 *)(b + 2u + 8u * ss);
        const uchar4 i0 = uchar4(q[0]);
        const uchar4 i1 = uchar4(q[1]);
        for (short l = 0; l < 4; ++l) {
            const uint signs = tab[256u + ((aux >> (7u * uint(l))) & 0x7Fu)];
            const uchar4 ia = l < 2 ? i0 : i1;
            const uint g1 = tab[ia[(2 * l) & 3]];
            const uint g2 = tab[ia[(2 * l + 1) & 3]];
            qv[2 * l] = q4e_gs_signed4(g1, signs, 1u);
            qv[2 * l + 1] = q4e_gs_signed4(g2, signs, 16u);
        }
        return q4e_g_half(b) * (0.5f + float(aux >> 28u)) * 0.5f;
    } else {
        // Single precision, whose values are the codes.
        q4e_g_deq32<T>(row, s, qv);
        return 1.0f;
    }
}

/// `q4e_g_row_cols` through `q4e_gs_codes32`: one lane's share of
/// `acc[c] = row . x_c`, the sub-blocks `j, j + lpr, ...`.
template <uint T>
static inline void q4e_gs_row_cols(device const uchar *row, uint in_dim, device const float *x,
                                   thread const ulong *xoff, uint n_cols, uint j, uint lpr,
                                   threadgroup const uint *tab, thread float *acc) {
    const uint nsub = in_dim / 32u;
    for (uint u = j; u < nsub; u += lpr) {
        float4 qv[8];
        const float scale = q4e_gs_codes32<T>(row, u, qv, tab);
        for (uint c = 0; c < Q4E_G_COLS; ++c) {
            if (c < n_cols) {
                acc[c] += scale * q4e_g_dot32(qv, x + xoff[c] + u * 32u);
            }
        }
    }
}

/// Rows a routed unit computes against one read of the activations: of the
/// down matrix, or of gate and up together.
///
/// A row's product reads `4 in_dim` bytes of activations against well under
/// `in_dim` of weights, and a lane that walks one row reads them again for
/// every row. Measured on an M1 Max, thirty routed columns of gate/up took
/// the same 1.1 ms whatever the decode cost or the lanes a row; that was 393
/// MB of activations.
#define Q4E_GS_UNIT_ROWS 8u

/// One lane's share of `NR` rows against one or two activation rows: the
/// sub-blocks `j, j + ul, ...`, the activations read once a sub-block for
/// every row. `acc[2 r + t]` is row `r` against activation row `t`; a row's
/// sum is the same whatever `NR` it shares the unit with.
template <uint T, uint NR>
static inline void q4e_gs_unit(thread device const uchar *const *rowp, uint in_dim,
                               device const float *x, ulong xoff0, ulong xoff1, bool two, uint j,
                               uint ul, threadgroup const uint *tab, thread float *acc) {
    const uint nsub = in_dim / 32u;
    for (uint u = j; u < nsub; u += ul) {
        device const float4 *xa = (device const float4 *)(x + xoff0 + u * 32u);
        device const float4 *xb = (device const float4 *)(x + xoff1 + u * 32u);
        float4 a[8];
        float4 b[8];
        for (short k = 0; k < 8; ++k) {
            a[k] = xa[k];
            b[k] = xb[k];
        }
        for (uint r = 0; r < NR; ++r) {
            float4 qv[8];
            const float scale = q4e_gs_codes32<T>(rowp[r], u, qv, tab);
            float4 sa = 0.0f;
            float4 sb = 0.0f;
            for (short k = 0; k < 8; ++k) {
                sa += qv[k] * a[k];
                sb += qv[k] * b[k];
            }
            acc[2u * r] += scale * ((sa[0] + sa[1]) + (sa[2] + sa[3]));
            if (two) {
                acc[2u * r + 1u] += scale * ((sb[0] + sb[1]) + (sb[2] + sb[3]));
            }
        }
    }
}

/// Folds a unit's `lpr` lanes (a power of two up to 32), lowest offset last;
/// every lane of the unit ends with the sum.
static inline float q4e_gs_fold(float v, uint lpr) {
    for (ushort off = ushort(lpr >> 1u); off > 0; off >>= 1) {
        v += simd_shuffle_xor(v, off);
    }
    return v;
}

/// `out[r * out_dim + o] = W[o] . x[r]`: `q4e_g_matvec_kernel`'s arguments
/// and `lpr`, a power of two up to `32 * Q4E_GS_WARPS`. A unit is `R`
/// consecutive rows, walked one after another, so a row's sum is the same
/// whatever `R` is. Grid `(ceil(out_dim / (R * Q4E_GS_WARPS * 32 / lpr)),
/// ceil(rows / Q4E_G_COLS))`, block `32 * Q4E_GS_WARPS`.
template <uint T, uint R>
kernel void q4e_g_step_matvec_kernel(
        device float *out [[buffer(0)]],
        device const uchar *w [[buffer(1)]],
        device const float *x [[buffer(2)]],
        constant uint &in_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        constant uint &rows [[buffer(5)]],
        constant uint &lpr [[buffer(6)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup float rpart[R * Q4E_GS_WARPS * Q4E_G_COLS];
    threadgroup uint tab[q4e_gs_tab_words<T>()];
    q4e_gs_load_tab<T>(tab, simd * 32u + lane, 32u * Q4E_GS_WARPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint r0 = group.y * Q4E_G_COLS;
    const uint n_cols = min(Q4E_G_COLS, rows - r0);
    const ulong rb = q4e_g_row_bytes<T>(in_dim);
    ulong xoff[Q4E_G_COLS];
    for (uint c = 0; c < Q4E_G_COLS; ++c) {
        xoff[c] = ulong(r0 + min(c, n_cols - 1u)) * in_dim;
    }
    if (lpr > 32u) {
        // `lpr / 32` simdgroups a row: a narrow output over a wide
        // contraction has too few rows for a simdgroup each to keep the
        // device busy. Each simdgroup folds its lanes, then the row's
        // simdgroups fold through threadgroup memory, lowest first.
        const uint wpr = lpr / 32u;
        const uint o0 = (group.x * (Q4E_GS_WARPS / wpr) + simd / wpr) * R;
        const uint j = (simd % wpr) * 32u + lane;
        for (uint r = 0; r < R; ++r) {
            const uint o = o0 + r;
            float acc[Q4E_G_COLS] = {0.0f};
            if (o < out_dim) {
                q4e_gs_row_cols<T>(w + ulong(o) * rb, in_dim, x, xoff, n_cols, j, lpr, tab, acc);
            }
            for (uint c = 0; c < Q4E_G_COLS; ++c) {
                const float v = q4e_gs_fold(acc[c], 32u);
                if (lane == 0u) {
                    rpart[(r * Q4E_GS_WARPS + simd) * Q4E_G_COLS + c] = v;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd % wpr == 0u && lane < n_cols) {
            for (uint r = 0; r < R; ++r) {
                if (o0 + r < out_dim) {
                    float v = rpart[(r * Q4E_GS_WARPS + simd) * Q4E_G_COLS + lane];
                    for (uint k = 1; k < wpr; ++k) {
                        v += rpart[(r * Q4E_GS_WARPS + simd + k) * Q4E_G_COLS + lane];
                    }
                    out[ulong(r0 + lane) * out_dim + o0 + r] = v;
                }
            }
        }
        return;
    }
    const uint o0 = ((group.x * Q4E_GS_WARPS + simd) * (32u / lpr) + lane / lpr) * R;
    const uint j = lane % lpr;
    device const uchar *rowp[R];
    for (uint r = 0; r < R; ++r) {
        rowp[r] = w + ulong(min(o0 + r, out_dim - 1u)) * rb;
    }
    for (uint c = 0; c < n_cols; c += 2u) {
        const bool two = c + 1u < n_cols;
        float acc[2u * R] = {0.0f};
        q4e_gs_unit<T, R>(rowp, in_dim, x, xoff[c], xoff[two ? c + 1u : c], two, j, lpr, tab,
                          acc);
        for (uint r = 0; r < R; ++r) {
            const bool live = j == 0u && o0 + r < out_dim;
            const float v0 = q4e_gs_fold(acc[2u * r], lpr);
            const float v1 = q4e_gs_fold(acc[2u * r + 1u], lpr);
            if (live) {
                out[ulong(r0 + c) * out_dim + o0 + r] = v0;
                if (two) {
                    out[ulong(r0 + c + 1u) * out_dim + o0 + r] = v1;
                }
            }
        }
    }
}

/// One routed unit - row `o` of one matrix - over an active expert's sorted
/// columns.
template <uint T>
static inline void q4e_gs_expert_unit(device float *out, device const uchar *w,
                                      device const float *x, device const int *sorted, int c0,
                                      int c1, uint o, bool live, uint in_dim, uint out_dim,
                                      uint x_div, uint j, uint lpr,
                                      threadgroup const uint *tab) {
    device const uchar *row = w + ulong(live ? o : 0u) * q4e_g_row_bytes<T>(in_dim);
    for (int c = c0; c < c1; c += int(Q4E_G_COLS)) {
        const uint n_cols = min(Q4E_G_COLS, uint(c1 - c));
        ulong xoff[Q4E_G_COLS];
        uint cols[Q4E_G_COLS];
        for (uint k = 0; k < Q4E_G_COLS; ++k) {
            cols[k] = uint(sorted[c + int(min(k, n_cols - 1u))]);
            xoff[k] = ulong(cols[k] / x_div) * in_dim;
        }
        float acc[Q4E_G_COLS] = {0.0f};
        if (live) {
            q4e_gs_row_cols<T>(row, in_dim, x, xoff, n_cols, j, lpr, tab, acc);
        }
        for (uint k = 0; k < Q4E_G_COLS; ++k) {
            const float v = q4e_gs_fold(acc[k], lpr);
            if (live && j == 0u && k < n_cols) {
                out[ulong(cols[k]) * out_dim + o] = v;
            }
        }
    }
}

/// Gate and up of a step pass: `q4e_g_experts_gate_up_kernel`'s arguments
/// and `lpr` (up to 32). A simdgroup's units alternate gate and up, so a
/// threadgroup covers `Q4E_GS_WARPS * 16 / lpr` rows of both. Grid
/// `(ceil(out_dim / (Q4E_GS_WARPS * 16 / lpr)), groups)`, block
/// `32 * Q4E_GS_WARPS`.
template <uint T>
kernel void q4e_g_step_gate_up_kernel(
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
        constant uint &lpr [[buffer(15)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    threadgroup uint tab[q4e_gs_tab_words<T>()];
    q4e_gs_load_tab<T>(tab, simd * 32u + lane, 32u * Q4E_GS_WARPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong slot = map[expert];
    if (lpr * Q4E_GS_UNIT_ROWS <= 32u) {
        // A unit is four rows of gate and the same four of up, against one
        // read of the activations.
        constexpr uint R = Q4E_GS_UNIT_ROWS / 2u;
        const uint ul = lpr * Q4E_GS_UNIT_ROWS;
        const uint o0 = ((group.x * Q4E_GS_WARPS + simd) * (32u / ul) + lane / ul) * R;
        const ulong rb = q4e_g_row_bytes<T>(in_dim);
        device const uchar *rowp[Q4E_GS_UNIT_ROWS];
        for (uint r = 0; r < R; ++r) {
            const ulong at = ulong(min(o0 + r, out_dim - 1u)) * rb;
            rowp[r] = gate_w + slot * gate_stride + at;
            rowp[R + r] = up_w + slot * up_stride + at;
        }
        for (int c = c0; c < c1; c += 2) {
            const bool two = c + 1 < c1;
            const uint col0 = uint(sorted[c]);
            const uint col1 = uint(sorted[two ? c + 1 : c]);
            float acc[2u * Q4E_GS_UNIT_ROWS] = {0.0f};
            q4e_gs_unit<T, Q4E_GS_UNIT_ROWS>(rowp, in_dim, x, ulong(col0 / n_used) * in_dim,
                                             ulong(col1 / n_used) * in_dim, two, lane % ul, ul,
                                             tab, acc);
            for (uint r = 0; r < Q4E_GS_UNIT_ROWS; ++r) {
                const uint o = o0 + (r < R ? r : r - R);
                const bool live = lane % ul == 0u && o < out_dim;
                const float v0 = q4e_gs_fold(acc[2u * r], ul);
                const float v1 = q4e_gs_fold(acc[2u * r + 1u], ul);
                if (live && r < R) {
                    out_gate[ulong(col0) * out_dim + o] = v0;
                    if (two) {
                        out_gate[ulong(col1) * out_dim + o] = v1;
                    }
                } else if (live) {
                    out_up[ulong(col0) * out_dim + o] = v0;
                    if (two) {
                        out_up[ulong(col1) * out_dim + o] = v1;
                    }
                }
            }
        }
        return;
    }
    const uint unit = (group.x * Q4E_GS_WARPS + simd) * (32u / lpr) + lane / lpr;
    const uint o = unit >> 1u;
    if ((unit & 1u) != 0u) {
        q4e_gs_expert_unit<T>(out_up, up_w + slot * up_stride, x, sorted, c0, c1, o, o < out_dim,
                              in_dim, out_dim, n_used, lane % lpr, lpr, tab);
    } else {
        q4e_gs_expert_unit<T>(out_gate, gate_w + slot * gate_stride, x, sorted, c0, c1, o,
                              o < out_dim, in_dim, out_dim, n_used, lane % lpr, lpr, tab);
    }
}

/// [`q4e_g_step_gate_up_kernel`] with the SwiGLU activation folded into the
/// epilogue: one output `silu(gate) * up` instead of the two matrices, so the
/// separate activation launch and the gate/up round trip through memory are
/// gone. A unit already holds gate row `o` and up row `o` in the same lane, so
/// the activation costs no extra read.
///
/// Only the unit path is instantiated (`lpr * Q4E_GS_UNIT_ROWS <= 32`), which
/// is the decode step's; the host picks it only there. The activation is the
/// separate kernel's own arithmetic - `g / (1 + precise::exp(-g)) * u`, the
/// division kept, not a sigmoid - so a column's bits are the unfused chain's.
///
/// Buffers: out_mid, gate_w, up_w, x, sorted, bounds, active, map, in_dim,
/// out_dim, n_expert, n_used, gate_stride, up_stride, lpr.
template <uint T>
kernel void q4e_g_step_gate_up_act_kernel(
        device float *out_mid [[buffer(0)]],
        device const uchar *gate_w [[buffer(1)]],
        device const uchar *up_w [[buffer(2)]],
        device const float *x [[buffer(3)]],
        device const int *sorted [[buffer(4)]],
        device const int *bounds [[buffer(5)]],
        device const int *active [[buffer(6)]],
        device const uint *map [[buffer(7)]],
        constant uint &in_dim [[buffer(8)]],
        constant uint &out_dim [[buffer(9)]],
        constant uint &n_expert [[buffer(10)]],
        constant uint &n_used [[buffer(11)]],
        constant uint &gate_stride [[buffer(12)]],
        constant uint &up_stride [[buffer(13)]],
        constant uint &lpr [[buffer(14)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    threadgroup uint tab[q4e_gs_tab_words<T>()];
    q4e_gs_load_tab<T>(tab, simd * 32u + lane, 32u * Q4E_GS_WARPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const ulong slot = map[expert];
    constexpr uint R = Q4E_GS_UNIT_ROWS / 2u;
    const uint ul = lpr * Q4E_GS_UNIT_ROWS;
    const uint o0 = ((group.x * Q4E_GS_WARPS + simd) * (32u / ul) + lane / ul) * R;
    const ulong rb = q4e_g_row_bytes<T>(in_dim);
    device const uchar *rowp[Q4E_GS_UNIT_ROWS];
    for (uint r = 0; r < R; ++r) {
        const ulong at = ulong(min(o0 + r, out_dim - 1u)) * rb;
        rowp[r] = gate_w + slot * gate_stride + at;
        rowp[R + r] = up_w + slot * up_stride + at;
    }
    for (int c = c0; c < c1; c += 2) {
        const bool two = c + 1 < c1;
        const uint col0 = uint(sorted[c]);
        const uint col1 = uint(sorted[two ? c + 1 : c]);
        float acc[2u * Q4E_GS_UNIT_ROWS] = {0.0f};
        q4e_gs_unit<T, Q4E_GS_UNIT_ROWS>(rowp, in_dim, x, ulong(col0 / n_used) * in_dim,
                                         ulong(col1 / n_used) * in_dim, two, lane % ul, ul,
                                         tab, acc);
        for (uint r = 0; r < R; ++r) {
            const uint o = o0 + r;
            const bool live = lane % ul == 0u && o < out_dim;
            // Gate row r is acc[2r..2r+1], the same row of up acc[2(r+R)..].
            const float g0 = q4e_gs_fold(acc[2u * r], ul);
            const float g1 = q4e_gs_fold(acc[2u * r + 1u], ul);
            const float u0 = q4e_gs_fold(acc[2u * (r + R)], ul);
            const float u1 = q4e_gs_fold(acc[2u * (r + R) + 1u], ul);
            if (live) {
                out_mid[ulong(col0) * out_dim + o] = (g0 / (1.0f + precise::exp(-g0))) * u0;
                if (two) {
                    out_mid[ulong(col1) * out_dim + o] = (g1 / (1.0f + precise::exp(-g1))) * u1;
                }
            }
        }
    }
}

/// The routed down projection of a step pass: `q4e_g_experts_down_kernel`'s
/// arguments and `lpr` (up to 32). Grid `(ceil(out_dim / (Q4E_GS_WARPS * 32 /
/// lpr)), groups)`, block `32 * Q4E_GS_WARPS`.
template <uint T>
kernel void q4e_g_step_down_kernel(
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
        constant uint &lpr [[buffer(12)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint simd [[simdgroup_index_in_threadgroup]]) {
    if (int(group.y) >= active[n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    threadgroup uint tab[q4e_gs_tab_words<T>()];
    q4e_gs_load_tab<T>(tab, simd * 32u + lane, 32u * Q4E_GS_WARPS);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lpr * Q4E_GS_UNIT_ROWS <= 32u) {
        const uint ul = lpr * Q4E_GS_UNIT_ROWS;
        const uint o0 =
            ((group.x * Q4E_GS_WARPS + simd) * (32u / ul) + lane / ul) * Q4E_GS_UNIT_ROWS;
        const ulong rb = q4e_g_row_bytes<T>(in_dim);
        device const uchar *rowp[Q4E_GS_UNIT_ROWS];
        for (uint r = 0; r < Q4E_GS_UNIT_ROWS; ++r) {
            rowp[r] = down_w + ulong(map[expert]) * stride + ulong(min(o0 + r, out_dim - 1u)) * rb;
        }
        for (int c = c0; c < c1; c += 2) {
            const bool two = c + 1 < c1;
            const uint col0 = uint(sorted[c]);
            const uint col1 = uint(sorted[two ? c + 1 : c]);
            float acc[2u * Q4E_GS_UNIT_ROWS] = {0.0f};
            q4e_gs_unit<T, Q4E_GS_UNIT_ROWS>(rowp, in_dim, mid, ulong(col0) * in_dim,
                                             ulong(col1) * in_dim, two, lane % ul, ul, tab, acc);
            for (uint r = 0; r < Q4E_GS_UNIT_ROWS; ++r) {
                const bool live = lane % ul == 0u && o0 + r < out_dim;
                const float v0 = q4e_gs_fold(acc[2u * r], ul);
                const float v1 = q4e_gs_fold(acc[2u * r + 1u], ul);
                if (live) {
                    out[ulong(col0) * out_dim + o0 + r] = v0;
                    if (two) {
                        out[ulong(col1) * out_dim + o0 + r] = v1;
                    }
                }
            }
        }
        return;
    }
    const uint o = (group.x * Q4E_GS_WARPS + simd) * (32u / lpr) + lane / lpr;
    q4e_gs_expert_unit<T>(out, down_w + ulong(map[expert]) * stride, mid, sorted, c0, c1, o,
                          o < out_dim, in_dim, out_dim, 1u, lane % lpr, lpr, tab);
}

#define Q4E_GS_MATVEC(T, NAME)                                                                  \
    template [[host_name("q4e_g_step_matvec_" NAME "_kernel")]] kernel void                     \
    q4e_g_step_matvec_kernel<T, 1u>(device float *, device const uchar *, device const float *, \
                                    constant uint &, constant uint &, constant uint &,          \
                                    constant uint &, uint3, uint, uint);                        \
    template [[host_name("q4e_g_step_matvec4_" NAME "_kernel")]] kernel void                    \
    q4e_g_step_matvec_kernel<T, 4u>(device float *, device const uchar *, device const float *, \
                                    constant uint &, constant uint &, constant uint &,          \
                                    constant uint &, uint3, uint, uint);

#define Q4E_GS_INSTANTIATE(T, NAME)                                                             \
    Q4E_GS_MATVEC(T, NAME)                                                                      \
    template [[host_name("q4e_g_step_gate_up_" NAME "_kernel")]] kernel void                    \
    q4e_g_step_gate_up_kernel<T>(device float *, device float *, device const uchar *,          \
                                 device const uchar *, device const float *, device const int *,\
                                 device const int *, device const int *, device const uint *,   \
                                 constant uint &, constant uint &, constant uint &,             \
                                 constant uint &, constant uint &, constant uint &,             \
                                 constant uint &, uint3, uint, uint);                           \
    template [[host_name("q4e_g_step_gate_up_act_" NAME "_kernel")]] kernel void                \
    q4e_g_step_gate_up_act_kernel<T>(device float *, device const uchar *, device const uchar *,\
                                 device const float *, device const int *, device const int *,  \
                                 device const int *, device const uint *, constant uint &,      \
                                 constant uint &, constant uint &, constant uint &,             \
                                 constant uint &, constant uint &, constant uint &,             \
                                 uint3, uint, uint);                                            \
    template [[host_name("q4e_g_step_down_" NAME "_kernel")]] kernel void                       \
    q4e_g_step_down_kernel<T>(device float *, device const uchar *, device const float *,       \
                              device const int *, device const int *, device const int *,       \
                              device const uint *, constant uint &, constant uint &,            \
                              constant uint &, constant uint &, constant uint &,                \
                              constant uint &, uint3, uint, uint);

Q4E_GS_MATVEC(Q4E_G_F32, "f32")
Q4E_GS_INSTANTIATE(Q4E_G_Q8_0, "q8_0")
Q4E_GS_INSTANTIATE(Q4E_G_Q6_K, "q6_k")
Q4E_GS_INSTANTIATE(Q4E_G_IQ3_XXS, "iq3_xxs")
Q4E_GS_INSTANTIATE(Q4E_G_IQ4_NL, "iq4_nl")
Q4E_GS_INSTANTIATE(Q4E_G_IQ2_S, "iq2_s")
Q4E_GS_INSTANTIATE(Q4E_G_IQ4_XS, "iq4_xs")
Q4E_GS_INSTANTIATE(Q4E_G_Q2_0, "q2_0")
