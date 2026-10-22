// Row-exact MLX affine matmul: 4- and 8-bit weights in groups of 64, f32 rows.
//
// A row's bits do not depend on how many rows share the call. That is what lets
// a verify window of drafted tokens be checked against serial decoding by
// equality: row r of an R-row call equals the same row computed alone, for
// every R.
//
// The arithmetic, identical in both kernels and for every row count. Per group
// g of 64 inputs, with PER = 32 / BITS values a word and WPG = 64 / PER words a
// group:
//
//   P    = one FMA chain over (s = 0..PER-1, i = 0..WPG-1), k = PER * i + s:
//          P = fma(x[k] * 2^-BITS*s, float(word_i & (MASK << BITS*s)), P)
//   sum  = eight left-to-right sums of 8 inputs, combined as a fixed pairwise
//          tree ((b0+b1)+(b2+b3)) + ((b4+b5)+(b6+b7))
//   acc  = fma(bias, sum, fma(scale, P, acc))
//
// The nibble (or byte) enters unshifted, against the input pre-scaled by the
// matching power of two, so their product is exactly x * q. The groups are
// dealt to S chunks (chunk c takes groups c, c + S, ...), each with its own
// accumulator, and the chunks combine by a fixed pairwise tree. S is a function
// of the weight's shape only, so a row count never changes it.
//
// - `kernel_dense_exact_qmm_*` (any row count): simdgroup_multiply_accumulate
//   on 8x8 fp32 tiles computing D^T = W X^T. Lane (fm, fn) holds words fn and
//   fn + 1 of weight row n + fm, and rows fn, fn + 1 of the inputs; step s
//   multiplies value s of each word, so one MMA is eight links of P's chain and
//   one weight load serves up to eight rows. A threadgroup is S simdgroups, one
//   a chunk, combining through threadgroup memory at the end.
// - `kernel_dense_exact_qmv_*` (one row): the same chain with scalar FMAs.
//   Lane (chunk c = lane % S, slot j = lane / S) holds whole groups of its
//   outputs' words; the threadgroup stages pre-scaled inputs in chain order;
//   chunks combine by butterfly shuffles in the same tree. Serial decode goes
//   through it, so it streams weights at matvec speed instead of paying for an
//   8-row tile.
//
// The two agree only because an fp32 MMA on Apple GPUs evaluates each output as
// the forward FMA chain d = fma(a7, b7, ... fma(a0, b0, c)). The engine checks
// that per shape at load and routes one-row calls of a shape where it fails
// through the MMA kernel.
//
// Inputs need K % 64 == 0 and N % 8 == 0. The file is compiled in safe math
// mode: the chains above are fixed orders, and fast math may refold them.
#pragma METAL fp math_mode(safe)

#include <metal_stdlib>
using namespace metal;

/// Chunks a weight's groups are dealt to, and the shape of the one-row launch.
constant constexpr uint DENSE_EXACT_SPLITS = 8u;
/// Input groups the one-row kernel stages at a time.
constant constexpr uint DENSE_EXACT_XB = 32u;
/// Floats a staged group: 64 inputs, 8 block sums, 4 pad.
constant constexpr uint DENSE_EXACT_XP = 76u;

/// 2^-(bits * s), exactly.
static inline float dense_exact_pre(uint bits, uint s) {
    return as_type<float>(uint(127u - bits * s) << 23);
}

/// Sum of the eight inputs at `p`, left to right.
static inline float dense_exact_sum8(float4 a, float4 b) {
    float t = a.x;
    t = t + a.y;
    t = t + a.z;
    t = t + a.w;
    t = t + b.x;
    t = t + b.y;
    t = t + b.z;
    t = t + b.w;
    return t;
}

// ---------------------------------------------------------------------------
// One row
// ---------------------------------------------------------------------------

template <uint BITS, uint NR>
kernel void kernel_dense_exact_qmv(
        constant dense_args_exact_affine & args [[buffer(0)]],
        device const uint32_t * W [[buffer(1)]],
        device const ushort * SC [[buffer(2)]],
        device const ushort * BI [[buffer(3)]],
        device const float * X [[buffer(4)]],
        device float * OUT [[buffer(5)]],
        uint3 tg_pos [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint n_sg [[simdgroups_per_threadgroup]]) {
    constexpr uint PER = 32u / BITS;
    constexpr uint WPG = 64u / PER;
    constexpr uint V4 = WPG / 4u;
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr uint S = DENSE_EXACT_SPLITS;
    constexpr uint SLOTS = 32u / S;
    constexpr uint XB = DENSE_EXACT_XB;
    constexpr uint XP = DENSE_EXACT_XP;
    threadgroup float xs[XB * XP];

    const uint K = args.in_dim;
    const uint N = args.out_dim;
    const uint G = K / 64u;
    const uint tid = sg * 32u + lane;
    const uint nthreads = n_sg * 32u;
    const uint c = lane % S;
    const uint n0 = (tg_pos.x * n_sg + sg) * (SLOTS * NR) + lane / S;

    device const uint4 * wr[NR];
    device const ushort * sr[NR];
    device const ushort * br[NR];
    float acc[NR];
    for (uint u = 0; u < NR; u++) {
        const uint nn = min(n0 + SLOTS * u, N - 1u);
        wr[u] = (device const uint4 *) (W + (uint64_t) nn * (K / PER));
        sr[u] = SC + (uint64_t) nn * G;
        br[u] = BI + (uint64_t) nn * G;
        acc[u] = 0.0f;
    }

    uint4 nxt[NR][V4];
    for (uint u = 0; u < NR; u++) {
        for (uint v = 0; v < V4; v++) {
            nxt[u][v] = c < G ? wr[u][c * V4 + v] : uint4(0u);
        }
    }
    for (uint b0 = 0; b0 < G; b0 += XB) {
        const uint nbk = min(XB, G - b0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Stage the inputs in chain order, pre-scaled, x'[s * WPG + i] =
        // x[PER i + s] * 2^-BITS s, and the group's eight block sums: one
        // thread a block of eight contiguous inputs, which is one word's
        // values at 4 bits and two words' at 8.
        for (uint idx = tid; idx < nbk * 8u; idx += nthreads) {
            const uint gl = idx / 8u;
            const uint b = idx % 8u;
            device const float4 * p = (device const float4 *) (X + (uint64_t) (b0 + gl) * 64u + 8u * b);
            const float4 lo = p[0];
            const float4 hi = p[1];
            threadgroup float * dst = xs + gl * XP;
            const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
            _Pragma("clang loop unroll(full)")
            for (uint e = 0; e < 8u; e++) {
                const uint k = 8u * b + e;
                const uint sv = k % PER;
                dst[sv * WPG + k / PER] = v[e] * dense_exact_pre(BITS, sv);
            }
            dst[64u + b] = dense_exact_sum8(lo, hi);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint g = b0 + c; g < b0 + nbk; g += S) {
            // This group's words were fetched an iteration ago; fetch the
            // next group's before the chain, so the load is in flight while
            // the FMAs run.
            uint4 cur[NR][V4];
            _Pragma("clang loop unroll(full)")
            for (uint u = 0; u < NR; u++) {
                _Pragma("clang loop unroll(full)")
                for (uint v = 0; v < V4; v++) {
                    cur[u][v] = nxt[u][v];
                }
            }
            if (g + S < G) {
                _Pragma("clang loop unroll(full)")
                for (uint u = 0; u < NR; u++) {
                    _Pragma("clang loop unroll(full)")
                    for (uint v = 0; v < V4; v++) {
                        nxt[u][v] = wr[u][(g + S) * V4 + v];
                    }
                }
            }
            threadgroup const float * xg = xs + (g - b0) * XP;
            const float4 p0 = *(threadgroup const float4 *) (xg + 64u);
            const float4 p1 = *(threadgroup const float4 *) (xg + 68u);
            const float xsum = ((p1.w + p1.z) + (p1.y + p1.x)) + ((p0.w + p0.z) + (p0.y + p0.x));
            float P[NR];
            _Pragma("clang loop unroll(full)")
            for (uint u = 0; u < NR; u++) {
                P[u] = 0.0f;
            }
            _Pragma("clang loop unroll(full)")
            for (uint s = 0; s < PER; s++) {
                const uint mask = MASK << (BITS * s);
                _Pragma("clang loop unroll(full)")
                for (uint v = 0; v < V4; v++) {
                    const float4 xq = *(threadgroup const float4 *) (xg + s * WPG + 4u * v);
                    _Pragma("clang loop unroll(full)")
                    for (uint u = 0; u < NR; u++) {
                        P[u] = fma(xq.x, float(cur[u][v].x & mask), P[u]);
                        P[u] = fma(xq.y, float(cur[u][v].y & mask), P[u]);
                        P[u] = fma(xq.z, float(cur[u][v].z & mask), P[u]);
                        P[u] = fma(xq.w, float(cur[u][v].w & mask), P[u]);
                    }
                }
            }
            _Pragma("clang loop unroll(full)")
            for (uint u = 0; u < NR; u++) {
                acc[u] = fma(dense_bf16(sr[u] + g), P[u], acc[u]);
                acc[u] = fma(dense_bf16(br[u] + g), xsum, acc[u]);
            }
        }
    }
    for (uint u = 0; u < NR; u++) {
        float v = acc[u];
        for (uint m = 1u; m < S; m <<= 1u) {
            v = simd_shuffle_xor(v, ushort(m)) + v;
        }
        const uint n = n0 + SLOTS * u;
        if (n < N && c == 0u) {
            OUT[n] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Any number of rows
// ---------------------------------------------------------------------------

template <uint BITS, uint NT, uint RT>
kernel void kernel_dense_exact_qmm(
        constant dense_args_exact_affine & args [[buffer(0)]],
        device const uint32_t * W [[buffer(1)]],
        device const ushort * SC [[buffer(2)]],
        device const ushort * BI [[buffer(3)]],
        device const float * X [[buffer(4)]],
        device float * OUT [[buffer(5)]],
        uint3 tg_pos [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint c [[simdgroup_index_in_threadgroup]]) {
    constexpr uint PER = 32u / BITS;
    constexpr uint WPG = 64u / PER;
    constexpr uint HALVES = WPG / 8u;
    constexpr uint MASK = (1u << BITS) - 1u;
    constexpr uint S = DENSE_EXACT_SPLITS;
    threadgroup float red[S * RT * NT * 64u];

    const uint K = args.in_dim;
    const uint N = args.out_dim;
    const uint R = args.n_rows;
    const uint G = K / 64u;
    const uint qid = lane / 4u;
    const uint fm = (qid & 4u) + ((lane / 2u) % 4u);
    const uint fn = (qid & 2u) * 2u + (lane % 2u) * 2u;
    const uint nb = tg_pos.x * (8u * NT);
    const uint rb = tg_pos.y * (8u * RT);

    uint wrow[NT];
    for (uint t = 0; t < NT; t++) {
        wrow[t] = min(nb + 8u * t + fm, N - 1u);
    }
    uint xr0[RT];
    uint xr1[RT];
    for (uint rt = 0; rt < RT; rt++) {
        xr0[rt] = min(rb + 8u * rt + fn, R - 1u);
        xr1[rt] = min(rb + 8u * rt + fn + 1u, R - 1u);
    }
    float acc[RT][NT][2];
    for (uint rt = 0; rt < RT; rt++) {
        for (uint t = 0; t < NT; t++) {
            acc[rt][t][0] = 0.0f;
            acc[rt][t][1] = 0.0f;
        }
    }

    // Lane (fm, fn)'s words of each tile row, as uint2 pairs (fn is even).
    device const uint2 * wbase[NT];
    uint2 wn[NT][HALVES];
    for (uint t = 0; t < NT; t++) {
        wbase[t] = (device const uint2 *) (W + (uint64_t) wrow[t] * (K / PER) + fn);
        for (uint h = 0; h < HALVES; h++) {
            wn[t][h] = c < G ? wbase[t][c * (WPG / 2u) + 4u * h] : uint2(0u);
        }
    }
    for (uint g = c; g < G; g += S) {
        // Lane fm's inputs of each of its two rows: block fm (eight inputs,
        // which is also what the group sum reads) for 4-bit, whose value s of
        // word fm is input 8 fm + s; for 8-bit, word 8 h + fm holds inputs
        // 32 h + 4 fm + s, so the two halves' four each.
        float4 xa[RT][2];
        float4 xb[RT][2];
        float4 ha[RT][2];
        float4 hb[RT][2];
        float xs0[RT];
        float xs1[RT];
        _Pragma("clang loop unroll(full)")
        for (uint rt = 0; rt < RT; rt++) {
            device const float4 * pa = (device const float4 *) (X + (uint64_t) xr0[rt] * K + 64u * g);
            device const float4 * pb = (device const float4 *) (X + (uint64_t) xr1[rt] * K + 64u * g);
            xa[rt][0] = pa[2u * fm];
            xa[rt][1] = pa[2u * fm + 1u];
            xb[rt][0] = pb[2u * fm];
            xb[rt][1] = pb[2u * fm + 1u];
            if (BITS == 8u) {
                ha[rt][0] = pa[fm];
                ha[rt][1] = pa[8u + fm];
                hb[rt][0] = pb[fm];
                hb[rt][1] = pb[8u + fm];
            }
            float v = dense_exact_sum8(xa[rt][0], xa[rt][1]);
            float u = dense_exact_sum8(xb[rt][0], xb[rt][1]);
            v = simd_shuffle_xor(v, ushort(2)) + v;
            u = simd_shuffle_xor(u, ushort(2)) + u;
            v = simd_shuffle_xor(v, ushort(4)) + v;
            u = simd_shuffle_xor(u, ushort(4)) + u;
            v = simd_shuffle_xor(v, ushort(16)) + v;
            u = simd_shuffle_xor(u, ushort(16)) + u;
            xs0[rt] = v;
            xs1[rt] = u;
        }
        // This group's words came in an iteration ago; start the next
        // group's now so the fetch overlaps this group's multiplies.
        uint2 wv[NT][HALVES];
        _Pragma("clang loop unroll(full)")
        for (uint t = 0; t < NT; t++) {
            _Pragma("clang loop unroll(full)")
            for (uint h = 0; h < HALVES; h++) {
                wv[t][h] = wn[t][h];
            }
        }
        if (g + S < G) {
            _Pragma("clang loop unroll(full)")
            for (uint t = 0; t < NT; t++) {
                _Pragma("clang loop unroll(full)")
                for (uint h = 0; h < HALVES; h++) {
                    wn[t][h] = wbase[t][(g + S) * (WPG / 2u) + 4u * h];
                }
            }
        }
        simdgroup_matrix<float, 8, 8> P[RT][NT];
        _Pragma("clang loop unroll(full)")
        for (uint rt = 0; rt < RT; rt++) {
            _Pragma("clang loop unroll(full)")
            for (uint t = 0; t < NT; t++) {
                P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
            }
        }
        _Pragma("clang loop unroll(full)")
        for (uint s = 0; s < PER; s++) {
            const float ps = dense_exact_pre(BITS, s);
            const uint mask = MASK << (BITS * s);
            _Pragma("clang loop unroll(full)")
            for (uint h = 0; h < HALVES; h++) {
                simdgroup_matrix<float, 8, 8> bm[RT];
                _Pragma("clang loop unroll(full)")
                for (uint rt = 0; rt < RT; rt++) {
                    const float a = BITS == 4u ? (s < 4u ? xa[rt][0][s] : xa[rt][1][s - 4u]) : ha[rt][h][s];
                    const float b = BITS == 4u ? (s < 4u ? xb[rt][0][s] : xb[rt][1][s - 4u]) : hb[rt][h][s];
                    bm[rt].thread_elements()[0] = a * ps;
                    bm[rt].thread_elements()[1] = b * ps;
                }
                _Pragma("clang loop unroll(full)")
                for (uint t = 0; t < NT; t++) {
                    simdgroup_matrix<float, 8, 8> am;
                    am.thread_elements()[0] = float(wv[t][h].x & mask);
                    am.thread_elements()[1] = float(wv[t][h].y & mask);
                    _Pragma("clang loop unroll(full)")
                    for (uint rt = 0; rt < RT; rt++) {
                        simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
                    }
                }
            }
        }
        for (uint t = 0; t < NT; t++) {
            const float sc = dense_bf16(SC + (uint64_t) wrow[t] * G + g);
            const float bi = dense_bf16(BI + (uint64_t) wrow[t] * G + g);
            for (uint rt = 0; rt < RT; rt++) {
                acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
                acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
            }
        }
    }

    for (uint rt = 0; rt < RT; rt++) {
        for (uint t = 0; t < NT; t++) {
            for (uint e = 0; e < 2u; e++) {
                red[((c * RT + rt) * NT + t) * 64u + lane * 2u + e] = acc[rt][t][e];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = c * 32u + lane; idx < RT * NT * 64u; idx += S * 32u) {
        float v[S];
        for (uint k = 0; k < S; k++) {
            v[k] = red[k * (RT * NT * 64u) + idx];
        }
        for (uint w = 1u; w < S; w *= 2u) {
            for (uint k = 0; k + w < S; k += 2u * w) {
                v[k] = v[k + w] + v[k];
            }
        }
        const uint rt = idx / (NT * 64u);
        const uint t = (idx / 64u) % NT;
        const uint l = (idx % 64u) / 2u;
        const uint e = idx % 2u;
        const uint lq = l / 4u;
        const uint row = rb + 8u * rt + (lq & 2u) * 2u + (l % 2u) * 2u + e;
        const uint n = nb + 8u * t + (lq & 4u) + ((l / 2u) % 4u);
        if (row < R && n < N) {
            OUT[(uint64_t) row * N + n] = v[0];
        }
    }
}

#define DENSE_EXACT_QMV(BITS, NR) \
    template [[host_name("kernel_dense_exact_qmv_" #BITS "_nr" #NR "_f32")]] kernel void \
    kernel_dense_exact_qmv<BITS, NR>(constant dense_args_exact_affine &, device const uint32_t *, \
        device const ushort *, device const ushort *, device const float *, device float *, uint3, \
        uint, uint, uint);
DENSE_EXACT_QMV(4, 1)
DENSE_EXACT_QMV(4, 2)
DENSE_EXACT_QMV(8, 1)
DENSE_EXACT_QMV(8, 2)

#define DENSE_EXACT_QMM(BITS, NT, RT) \
    template [[host_name("kernel_dense_exact_qmm_" #BITS "_nt" #NT "_rt" #RT "_f32")]] kernel void \
    kernel_dense_exact_qmm<BITS, NT, RT>(constant dense_args_exact_affine &, device const uint32_t *, \
        device const ushort *, device const ushort *, device const float *, device float *, uint3, \
        uint, uint);
DENSE_EXACT_QMM(4, 1, 1)
DENSE_EXACT_QMM(4, 2, 1)
DENSE_EXACT_QMM(4, 4, 1)
DENSE_EXACT_QMM(4, 1, 2)
DENSE_EXACT_QMM(4, 2, 2)
DENSE_EXACT_QMM(4, 4, 2)
DENSE_EXACT_QMM(8, 1, 1)
DENSE_EXACT_QMM(8, 2, 1)
DENSE_EXACT_QMM(8, 4, 1)
DENSE_EXACT_QMM(8, 1, 2)
DENSE_EXACT_QMM(8, 2, 2)
DENSE_EXACT_QMM(8, 4, 2)

#pragma METAL fp math_mode(fast)
