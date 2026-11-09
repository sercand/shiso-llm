/* GGML block-quantized weights for the qwen4exp kernel set: the dense
 * projections, the routed expert stages and the vocabulary gather of a GGUF
 * checkpoint whose tensors stay in their stored blocks.
 *
 * Not an upstream unit: the CUDA twin of `kernels/metal/q4e/ggml.metal`, entry
 * for entry, with the same argument order (the Metal `constant uint &`
 * parameters arrive here by value). The 256-weight i-quants reuse the generic
 * family's `shiso_deq32_*` decoders and grid tables, which are the arithmetic
 * the Metal twin's `deq32_*` decoders do.
 *
 * Every entry reads a weight row as consecutive 32-weight sub-blocks, decoded
 * in registers and multiplied against single-precision activations; nothing
 * widens a matrix in memory and no activation is quantized. A row's dot
 * product is the same sum in the same order whatever row count or column
 * count shares the launch: the lanes split the sub-blocks by index and a
 * butterfly folds them in a fixed order, and the columns a lane carries are
 * independent accumulators.
 *
 * This is the correctness tier: one warp an output row, a few activation rows
 * a pass.
 */

#include "generic.cuh"

/* GGML type ids of the formats decoded here. */
#define Q4E_G_F32 0
#define Q4E_G_Q8_0 8
#define Q4E_G_Q6_K 14
#define Q4E_G_IQ3_XXS 18
#define Q4E_G_IQ4_NL 20
#define Q4E_G_IQ2_S 22
#define Q4E_G_IQ4_XS 23
#define Q4E_G_Q2_0 42

/* Output rows a block covers: one warp each. */
#define Q4E_G_ROWS 4u
/* Activation rows (or routed columns) one pass carries in registers. */
#define Q4E_G_COLS 8u

/* Bytes of one stored row of `n` weights. */
template <int T>
__device__ __forceinline__ uint64_t q4e_g_row_bytes(uint32_t n) {
    switch (T) {
        case Q4E_G_F32: return (uint64_t) n * 4ull;
        case Q4E_G_Q8_0: return (uint64_t) (n / 32u) * 34ull;
        case Q4E_G_IQ4_NL: return (uint64_t) (n / 32u) * 18ull;
        case Q4E_G_Q2_0: return (uint64_t) (n / 64u) * 18ull;
        case Q4E_G_Q6_K: return (uint64_t) (n / 256u) * 210ull;
        case Q4E_G_IQ2_S: return (uint64_t) (n / 256u) * 82ull;
        case Q4E_G_IQ3_XXS: return (uint64_t) (n / 256u) * 98ull;
        case Q4E_G_IQ4_XS: return (uint64_t) (n / 256u) * 136ull;
        default: return 0ull;
    }
}

__device__ __forceinline__ float q4e_g_half(const uint8_t *p) {
    return __half2float(*(const __half *) p);
}

/* Sub-block `s` (weights `32 s .. 32 s + 31`) of a row starting at `row`.
 *
 * Every block size here is even, so the half-precision scales the decoders
 * read are 2-byte aligned whenever the row is. */
template <int T>
__device__ __forceinline__ void q4e_g_deq32(const uint8_t *row, uint32_t s, float4 *qv) {
    float *out = (float *) qv;
    float2 ab;
    if (T == Q4E_G_F32) {
        /* Single precision is here for the stream mixes' injections, whose
         * other tier takes one row a launch. */
        const float4 *v = (const float4 *) (row + (uint64_t) s * 128ull);
#pragma unroll
        for (int k = 0; k < 8; ++k) qv[k] = v[k];
    } else if (T == Q4E_G_Q8_0) {
        const uint8_t *b = row + (uint64_t) s * 34ull;
        const float d = q4e_g_half(b);
        const int8_t *q = (const int8_t *) (b + 2);
#pragma unroll
        for (int j = 0; j < 32; ++j) out[j] = d * (float) q[j];
    } else if (T == Q4E_G_IQ4_NL) {
        const uint8_t *b = row + (uint64_t) s * 18ull;
        const float d = q4e_g_half(b);
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t q = b[2 + j];
            out[j] = d * (float) SHISO_IQ4NL[q & 0xF];
            out[16 + j] = d * (float) SHISO_IQ4NL[q >> 4];
        }
    } else if (T == Q4E_G_Q2_0) {
        /* 64 weights a block, four 2-bit codes a byte low bits first; the
         * codes 0..3 stand for -1..2 steps of the block scale. */
        const uint8_t *b = row + (uint64_t) (s / 2u) * 18ull;
        const float d = q4e_g_half(b);
        const uint8_t *q = b + 2 + (s & 1u) * 8u;
#pragma unroll
        for (int j = 0; j < 32; ++j) {
            const uint32_t c = ((uint32_t) q[j / 4] >> ((j % 4) * 2)) & 3u;
            out[j] = d * ((float) c - 1.0f);
        }
    } else if (T == Q4E_G_Q6_K) {
        /* A 256-weight block is two halves of 128; a 32-weight run `k` of a
         * half takes its low nibbles from `ql` (the high nibble for k >= 2),
         * its two high bits from `qh` at shift 2k, and one 8-bit scale per
         * 16. */
        const uint8_t *b = row + (uint64_t) (s / 8u) * 210ull;
        const uint32_t ss = s % 8u;
        const uint32_t n = ss / 4u;
        const uint32_t k = ss % 4u;
        const uint8_t *ql = b + n * 64u + (k & 1u) * 32u;
        const uint8_t *qh = b + 128u + n * 32u;
        const int8_t *sc = (const int8_t *) (b + 192u) + n * 8u + 2u * k;
        const float d = q4e_g_half(b + 208u);
        const uint32_t lshift = (k >> 1u) * 4u;
        const uint32_t hshift = 2u * k;
        const float d0 = d * (float) sc[0];
        const float d1 = d * (float) sc[1];
#pragma unroll
        for (int l = 0; l < 32; ++l) {
            const int q = (int) ((((uint32_t) ql[l] >> lshift) & 0xFu)
                                 | ((((uint32_t) qh[l] >> hshift) & 3u) << 4u)) - 32;
            out[l] = (l < 16 ? d0 : d1) * (float) q;
        }
    } else if (T == Q4E_G_IQ2_S) {
        shiso_deq32_iq2_s((const struct shiso_block_iq2_s *) (row + (uint64_t) (s / 8u) * 82ull),
                          (int) (s % 8u), qv, &ab);
    } else if (T == Q4E_G_IQ3_XXS) {
        shiso_deq32_iq3_xxs(
            (const struct shiso_block_iq3_xxs *) (row + (uint64_t) (s / 8u) * 98ull),
            (int) (s % 8u), qv, &ab);
    } else if (T == Q4E_G_IQ4_XS) {
        shiso_deq32_iq4_xs(
            (const struct shiso_block_iq4_xs *) (row + (uint64_t) (s / 8u) * 136ull),
            (int) (s % 8u), qv, &ab);
    }
}

__device__ __forceinline__ float q4e_g_dot32(const float4 *qv, const float *x) {
    const float4 *x4 = (const float4 *) x;
    float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float4 a = x4[k];
        acc.x += qv[k].x * a.x;
        acc.y += qv[k].y * a.y;
        acc.z += qv[k].z * a.z;
        acc.w += qv[k].w * a.w;
    }
    return (acc.x + acc.y) + (acc.z + acc.w);
}

/* `acc[c] += row . x_c` for up to `Q4E_G_COLS` activation rows, one lane's
 * share: the sub-blocks `lane, lane + 32, ...`, each decoded once. */
template <int T>
__device__ __forceinline__ void q4e_g_row_cols(const uint8_t *row, uint32_t in_dim,
                                               const float *x, const uint64_t *xoff,
                                               uint32_t n_cols, uint32_t lane, float *acc) {
    const uint32_t nsub = in_dim / 32u;
    for (uint32_t u = lane; u < nsub; u += 32u) {
        float4 qv[8];
        q4e_g_deq32<T>(row, u, qv);
#pragma unroll
        for (uint32_t c = 0; c < Q4E_G_COLS; ++c) {
            if (c < n_cols) acc[c] += q4e_g_dot32(qv, x + xoff[c] + u * 32u);
        }
    }
}

/* `out[r * out_dim + o] = W[o] . x[r]` over every row `r < rows`.
 * Grid `(ceil(out_dim / Q4E_G_ROWS), ceil(rows / Q4E_G_COLS))`, block
 * `32 * Q4E_G_ROWS`. */
template <int T>
__device__ __forceinline__ void q4e_g_matvec(float *out, const uint8_t *w, const float *x,
                                             uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * Q4E_G_ROWS + (threadIdx.x >> 5);
    if (o >= out_dim) return;
    const uint32_t r0 = blockIdx.y * Q4E_G_COLS;
    const uint32_t n_cols = min(Q4E_G_COLS, rows - r0);
    const uint8_t *row = w + (uint64_t) o * q4e_g_row_bytes<T>(in_dim);
    uint64_t xoff[Q4E_G_COLS];
#pragma unroll
    for (uint32_t c = 0; c < Q4E_G_COLS; ++c) {
        xoff[c] = (uint64_t) (r0 + min(c, n_cols - 1u)) * in_dim;
    }
    float acc[Q4E_G_COLS];
#pragma unroll
    for (uint32_t c = 0; c < Q4E_G_COLS; ++c) acc[c] = 0.0f;
    q4e_g_row_cols<T>(row, in_dim, x, xoff, n_cols, lane, acc);
#pragma unroll
    for (uint32_t c = 0; c < Q4E_G_COLS; ++c) {
        const float v = shiso_warp_all_sum(acc[c]);
        if (lane == 0u && c < n_cols) out[(uint64_t) (r0 + c) * out_dim + o] = v;
    }
}

/* One routed matrix over an active expert's sorted columns, the shared body
 * of the gate/up and down stages. `slot` is the expert's place in the bank
 * and `stride` the bytes between slots: one expert's own bytes in a resident
 * bank, the pool's slot width (which may be wider) in a streamed one. */
template <int T>
__device__ __forceinline__ void q4e_g_expert_rows(float *out, const uint8_t *w, const float *x,
                                                  const int *sorted, int c0, int c1,
                                                  uint64_t slot, uint64_t stride, uint32_t o,
                                                  uint32_t in_dim, uint32_t out_dim,
                                                  uint32_t x_div, uint32_t lane) {
    const uint8_t *row = w + slot * stride + (uint64_t) o * q4e_g_row_bytes<T>(in_dim);
    for (int c = c0; c < c1; c += (int) Q4E_G_COLS) {
        const uint32_t n_cols = min(Q4E_G_COLS, (uint32_t) (c1 - c));
        uint64_t xoff[Q4E_G_COLS];
        uint32_t cols[Q4E_G_COLS];
#pragma unroll
        for (uint32_t k = 0; k < Q4E_G_COLS; ++k) {
            cols[k] = (uint32_t) sorted[c + (int) min(k, n_cols - 1u)];
            xoff[k] = (uint64_t) (cols[k] / x_div) * in_dim;
        }
        float acc[Q4E_G_COLS];
#pragma unroll
        for (uint32_t k = 0; k < Q4E_G_COLS; ++k) acc[k] = 0.0f;
        q4e_g_row_cols<T>(row, in_dim, x, xoff, n_cols, lane, acc);
#pragma unroll
        for (uint32_t k = 0; k < Q4E_G_COLS; ++k) {
            const float v = shiso_warp_all_sum(acc[k]);
            if (lane == 0u && k < n_cols) out[(uint64_t) cols[k] * out_dim + o] = v;
        }
    }
}

/* The routed gate and up projections. `out_gate, out_up, gate_w, up_w, x,
 * sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used,
 * gate_stride, up_stride`. Grid `(ceil(out_dim / Q4E_G_ROWS), groups)`, block
 * `32 * Q4E_G_ROWS`. A column `col` reads token `col / n_used`. */
template <int T>
__device__ __forceinline__ void q4e_g_experts_gate_up(
        float *out_gate, float *out_up, const uint8_t *gate_w, const uint8_t *up_w,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used, uint32_t gate_stride, uint32_t up_stride) {
    if ((int) blockIdx.y >= active[n_expert]) return;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * Q4E_G_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) return;
    const uint64_t slot = map[expert];
    q4e_g_expert_rows<T>(out_gate, gate_w, x, sorted, c0, c1, slot, gate_stride, o, in_dim,
                         out_dim, n_used, lane);
    q4e_g_expert_rows<T>(out_up, up_w, x, sorted, c0, c1, slot, up_stride, o, in_dim, out_dim,
                         n_used, lane);
}

/* The routed down projection: `out, down_w, mid, sorted, bounds, active, map,
 * in_dim, out_dim, n_expert, n_used, stride`. A column reads its own row of
 * `mid`. Grid and block as the gate/up stage. */
template <int T>
__device__ __forceinline__ void q4e_g_experts_down(
        float *out, const uint8_t *down_w, const float *mid, const int *sorted,
        const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_used, uint32_t stride) {
    (void) n_used;
    if ((int) blockIdx.y >= active[n_expert]) return;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o = blockIdx.x * Q4E_G_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) return;
    q4e_g_expert_rows<T>(out, down_w, mid, sorted, c0, c1, map[expert], stride, o, in_dim,
                         out_dim, 1u, lane);
}

/* Vocabulary rows: `out[t * n_embd + i] = W[tokens[t]][i]`, the row decoded
 * whole. `out, tokens, w, n_tokens, n_embd`; grid `(n_tokens)`, block 32. */
template <int T>
__device__ __forceinline__ void q4e_g_embed(float *out, const int *tokens, const uint8_t *w,
                                            uint32_t n_tokens, uint32_t n_embd) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tokens) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint8_t *row = w + (uint64_t) tokens[t] * q4e_g_row_bytes<T>(n_embd);
    float4 *dst = (float4 *) (out + (uint64_t) t * n_embd);
    for (uint32_t u = lane; u < n_embd / 32u; u += 32u) {
        float4 qv[8];
        q4e_g_deq32<T>(row, u, qv);
#pragma unroll
        for (int k = 0; k < 8; ++k) dst[u * 8u + k] = qv[k];
    }
}

#define Q4E_G_INSTANTIATE(T, NAME)                                                              \
    extern "C" __global__ void __launch_bounds__(32 * Q4E_G_ROWS) q4e_g_matvec_##NAME##_kernel( \
            float *out, const uint8_t *w, const float *x, uint32_t in_dim, uint32_t out_dim,    \
            uint32_t rows) {                                                                    \
        q4e_g_matvec<T>(out, w, x, in_dim, out_dim, rows);                                      \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(32 * Q4E_G_ROWS)                               \
    q4e_g_experts_gate_up_##NAME##_kernel(                                                      \
            float *out_gate, float *out_up, const uint8_t *gate_w, const uint8_t *up_w,         \
            const float *x, const int *sorted, const int *bounds, const int *active,            \
            const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,          \
            uint32_t n_used, uint32_t gate_stride, uint32_t up_stride) {                        \
        q4e_g_experts_gate_up<T>(out_gate, out_up, gate_w, up_w, x, sorted, bounds, active,     \
                                 map, in_dim, out_dim, n_expert, n_used, gate_stride,           \
                                 up_stride);                                                    \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(32 * Q4E_G_ROWS)                               \
    q4e_g_experts_down_##NAME##_kernel(                                                         \
            float *out, const uint8_t *down_w, const float *mid, const int *sorted,             \
            const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,         \
            uint32_t out_dim, uint32_t n_expert, uint32_t n_used, uint32_t stride) {            \
        q4e_g_experts_down<T>(out, down_w, mid, sorted, bounds, active, map, in_dim, out_dim,   \
                              n_expert, n_used, stride);                                        \
    }                                                                                           \
    extern "C" __global__ void q4e_g_embed_##NAME##_kernel(                                     \
            float *out, const int *tokens, const uint8_t *w, uint32_t n_tokens,                 \
            uint32_t n_embd) {                                                                  \
        q4e_g_embed<T>(out, tokens, w, n_tokens, n_embd);                                       \
    }

Q4E_G_INSTANTIATE(Q4E_G_F32, f32)
Q4E_G_INSTANTIATE(Q4E_G_Q8_0, q8_0)
Q4E_G_INSTANTIATE(Q4E_G_Q6_K, q6_k)
Q4E_G_INSTANTIATE(Q4E_G_IQ3_XXS, iq3_xxs)
Q4E_G_INSTANTIATE(Q4E_G_IQ4_NL, iq4_nl)
Q4E_G_INSTANTIATE(Q4E_G_IQ2_S, iq2_s)
Q4E_G_INSTANTIATE(Q4E_G_IQ4_XS, iq4_xs)
Q4E_G_INSTANTIATE(Q4E_G_Q2_0, q2_0)

/* ---------------------------------------------------------------------------
 * Prompt passes on tensor cores.
 *
 * The same products for a prompt chunk, through `mma.sync` m16n8k16 with half
 * inputs and single-precision accumulation. Every format here stores a
 * 16-weight run as small integers - codes, codebook entries or signed grid
 * bytes, none wider than 255 - times one single-precision scale (the block
 * scale times its sub-scale), so the run goes into the tile as exact halves
 * and its scale stays out of it: each run's 16-term product is formed from
 * zero, multiplied by its row's scale and added to the running sum in run
 * order. The weights are therefore exact; the activations are rounded to
 * halves, as the other prompt tiers round them.
 *
 * A column's sum reads only its own activation row, in an order fixed by the
 * contraction width, so a prompt's rows do not depend on where it was cut or
 * what shared its chunk. These sums are not the step kernels' sums; a prompt
 * pass is the only caller.
 * ------------------------------------------------------------------------ */

/* Contraction elements one tile step covers; the output rows and columns are
 * the entry's own (`Q4E_GG_*_BM`, `Q4E_GG_*_BN`), two threads an output
 * row. */
#define Q4E_GG_BK 32u
/* Halves a staged tile row takes: 32 plus 8 of padding, so the eight rows an
 * ldmatrix phase reads fall on eight distinct bank groups. */
#define Q4E_GG_LD 40u
/* Output rows a block of each prompt entry covers. A block stages its
 * columns' activations once for all its rows, and those activations - f32
 * rows read from L2 and rounded every step - not the weights, were what a
 * 64-row tile was bound by (gate/up at 31 GB/s of weights, L2 at 54%). The
 * gate/up block covers 128 rows of each of its two matrices. */
#define Q4E_GG_DENSE_BM 128u
#define Q4E_GG_GATE_UP_BM 256u
#define Q4E_GG_DOWN_BM 128u
/* Columns a block of each covers. Each weight run is decoded once a column
 * tile, so a dense tile's columns are what its decode is spread over; a
 * routed stage seldom has many columns an expert. */
#define Q4E_GG_DENSE_BN 128u
#define Q4E_GG_EXPERT_BN 64u

/* Dynamic shared memory a `bm x bn` tile takes: two buffers of the weight
 * and activation tiles, two of the run scales. */
#define Q4E_GG_SMEM(bm, bn) (2u * ((bm) + (bn)) * Q4E_GG_LD * 2u + 2u * (bm) * 2u * 4u)

__device__ __forceinline__ uint32_t q4e_gg_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

/* Four bytes at a 2-byte-aligned address, as one word: every block here
 * keeps its payload two bytes past an even block start, so a run's bytes are
 * word aligned or two bytes off. Two aligned loads and a funnel, instead of
 * four byte loads. */
__device__ __forceinline__ uint32_t q4e_gg_ld4(const uint8_t *p) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t *w = (const uint32_t *) (a & ~(uintptr_t) 3);
    if ((a & 2u) == 0u) return w[0];
    return __byte_perm(w[0], w[1], 0x5432);
}

/* Sixteen bytes at a 2-byte-aligned address: five aligned loads at most. */
__device__ __forceinline__ void q4e_gg_ld16(const uint8_t *p, uint32_t *q) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t *w = (const uint32_t *) (a & ~(uintptr_t) 3);
    if ((a & 2u) == 0u) {
#pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = w[i];
    } else {
        uint32_t v[5];
#pragma unroll
        for (int i = 0; i < 5; ++i) v[i] = w[i];
#pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = __byte_perm(v[i], v[i + 1], 0x5432);
    }
}

/* Four signed bytes as two half pairs, exactly: biased to 0..255, each byte
 * becomes the low mantissa byte of 1024 (where a half's spacing is one), and
 * 1152 comes off. */
__device__ __forceinline__ void q4e_gg_s8_h2(uint32_t w, uint32_t *h) {
    const uint32_t u = w ^ 0x80808080u;
    const __half2 bias = __floats2half2_rn(1152.0f, 1152.0f);
    uint32_t a = __byte_perm(u, 0x64646464u, 0x5140);
    uint32_t b = __byte_perm(u, 0x64646464u, 0x7362);
    __half2 ha = __hsub2(*reinterpret_cast<__half2 *>(&a), bias);
    __half2 hb = __hsub2(*reinterpret_cast<__half2 *>(&b), bias);
    h[0] = *reinterpret_cast<uint32_t *>(&ha);
    h[1] = *reinterpret_cast<uint32_t *>(&hb);
}

/* The non-linear codebook's sixteen entries as signed bytes, four a word. */
#define Q4E_GG_NL0 0xBFAD9881u /* -127 -104  -83  -65 */
#define Q4E_GG_NL1 0xF6EADDCFu /*  -49  -35  -22  -10 */
#define Q4E_GG_NL2 0x26190D01u /*    1   13   25   38 */
#define Q4E_GG_NL3 0x71594535u /*   53   69   89  113 */

/* Codebook entries of the four nibbles in the low nibble of each byte of
 * `n`: a byte select from each half of the book, then the half bit 3 picks. */
__device__ __forceinline__ uint32_t q4e_gg_nl4(uint32_t n) {
    const uint32_t sel = (n & 0x7u) | ((n >> 4) & 0x70u) | ((n >> 8) & 0x700u) | ((n >> 12) & 0x7000u);
    const uint32_t hi8 = ((n >> 3) & 1u) | ((n >> 10) & 2u) | ((n >> 17) & 4u) | ((n >> 24) & 8u);
    const uint32_t lo = __byte_perm(Q4E_GG_NL0, Q4E_GG_NL1, sel);
    const uint32_t hi = __byte_perm(Q4E_GG_NL2, Q4E_GG_NL3, sel);
    const uint32_t pick = 0x3210u | ((hi8 & 1u) << 2) | ((hi8 & 2u) << 5) | ((hi8 & 4u) << 8)
                        | ((hi8 & 8u) << 11);
    return __byte_perm(lo, hi, pick);
}

/* Four unsigned grid bytes negated where the matching bit of `signs` (the
 * low four) is set. */
__device__ __forceinline__ uint32_t q4e_gg_signed4(uint32_t g, uint32_t signs) {
    const uint32_t m = (((signs & 0xFu) * 0x00204081u) & 0x01010101u) * 0xFFu;
    return __vsub4(g ^ m, m);
}

/* The seven stored sign bits of a quad and the eighth that makes their
 * parity even. */
__device__ __forceinline__ uint32_t q4e_gg_ksigns(uint32_t s7) {
    return s7 | ((__popc(s7) & 1u) << 7);
}

/* A run's stored bytes, fetched and not yet used: aligned words, so a
 * caller can issue the loads of a later run before it decodes this one. */
struct q4e_gg_raw {
    uint32_t w[12];
    uint32_t odd; /* bit i set: piece i starts two bytes into its first word */
};

/* Five aligned words covering the 16 bytes at `p` into `w`, and whether `p`
 * is two bytes past the first. */
__device__ __forceinline__ uint32_t q4e_gg_fetch16(const uint8_t *p, uint32_t *w) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t *q = (const uint32_t *) (a & ~(uintptr_t) 3);
    const uint32_t odd = (a & 2u) ? 1u : 0u;
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = q[i];
    w[4] = odd ? q[4] : 0u;
    return odd;
}

/* Two aligned words covering the 4 bytes at `p`. */
__device__ __forceinline__ uint32_t q4e_gg_fetch4(const uint8_t *p, uint32_t *w) {
    const uintptr_t a = (uintptr_t) p;
    const uint32_t *q = (const uint32_t *) (a & ~(uintptr_t) 3);
    const uint32_t odd = (a & 2u) ? 1u : 0u;
    w[0] = q[0];
    w[1] = odd ? q[1] : 0u;
    return odd;
}

__device__ __forceinline__ void q4e_gg_take16(const uint32_t *w, uint32_t odd, uint32_t *q) {
#pragma unroll
    for (int i = 0; i < 4; ++i) q[i] = odd ? __byte_perm(w[i], w[i + 1], 0x5432) : w[i];
}

__device__ __forceinline__ uint32_t q4e_gg_take4(const uint32_t *w, uint32_t odd) {
    return odd ? __byte_perm(w[0], w[1], 0x5432) : w[0];
}

/* The loads of run `t` of a row. */
template <int T>
__device__ __forceinline__ void q4e_gg_fetch(const uint8_t *row, uint32_t t, q4e_gg_raw &r) {
    if (T == Q4E_G_Q8_0) {
        const uint8_t *b = row + (uint64_t) (t / 2u) * 34ull;
        r.w[5] = *(const uint16_t *) b;
        r.odd = q4e_gg_fetch16(b + 2 + (t & 1u) * 16u, r.w);
    } else if (T == Q4E_G_IQ4_NL) {
        const uint8_t *b = row + (uint64_t) (t / 2u) * 18ull;
        r.w[5] = *(const uint16_t *) b;
        r.odd = q4e_gg_fetch16(b + 2, r.w);
    } else if (T == Q4E_G_IQ4_XS) {
        const uint8_t *b = row + (uint64_t) (t / 16u) * 136ull;
        const uint32_t s = (t % 16u) / 2u;
        r.w[5] = *(const uint16_t *) b;          /* d */
        r.w[6] = *(const uint16_t *) (b + 2);    /* scales_h */
        r.w[7] = b[4 + s / 2u];                  /* scales_l byte */
        r.odd = q4e_gg_fetch16(b + 8 + 16u * s, r.w);
    } else if (T == Q4E_G_Q2_0) {
        const uint8_t *b = row + (uint64_t) (t / 4u) * 18ull;
        r.w[5] = *(const uint16_t *) b;
        r.odd = q4e_gg_fetch4(b + 2 + (t % 4u) * 4u, r.w);
    } else if (T == Q4E_G_Q6_K) {
        const uint8_t *b = row + (uint64_t) (t / 16u) * 210ull;
        const uint32_t c = t % 16u;
        const uint32_t ss = c / 2u;
        const uint32_t n = ss / 4u;
        const uint32_t k = ss % 4u;
        const uint32_t hh = c & 1u;
        r.odd = q4e_gg_fetch16(b + n * 64u + (k & 1u) * 32u + hh * 16u, r.w);
        r.odd |= q4e_gg_fetch16(b + 128u + n * 32u + hh * 16u, r.w + 5) << 1;
        r.w[10] = (uint32_t) *(const int8_t *) (b + 192u + n * 8u + 2u * k + hh);
        r.w[11] = *(const uint16_t *) (b + 208u);
    } else if (T == Q4E_G_IQ2_S) {
        const uint8_t *b = row + (uint64_t) (t / 16u) * 82ull;
        const uint32_t s = (t % 16u) / 2u;
        r.odd = q4e_gg_fetch4(b + 2 + 4u * s, r.w);
        r.odd |= q4e_gg_fetch4(b + 34 + 4u * s, r.w + 2) << 1;
        r.w[4] = b[66 + s];
        r.w[5] = b[74 + s];
        r.w[6] = *(const uint16_t *) b;
    } else if (T == Q4E_G_IQ3_XXS) {
        const uint8_t *b = row + (uint64_t) (t / 16u) * 98ull;
        const uint32_t s = (t % 16u) / 2u;
        const uint32_t hh = t & 1u;
        r.odd = q4e_gg_fetch4(b + 66 + 4u * s, r.w);
        r.odd |= q4e_gg_fetch4(b + 2 + 8u * s + 4u * hh, r.w + 2) << 1;
        r.w[4] = *(const uint16_t *) b;
    }
}

__device__ __forceinline__ float q4e_gg_h16(uint32_t bits) {
    return __half2float(__ushort_as_half((unsigned short) bits));
}

/* Run `t`'s sixteen signed byte codes, four a word, and its scale, from its
 * fetched bytes. */
template <int T>
__device__ __forceinline__ float q4e_gg_decode(const q4e_gg_raw &r, uint32_t t, uint32_t *q) {
    float scale = 0.0f;
    if (T == Q4E_G_Q8_0) {
        scale = q4e_gg_h16(r.w[5]);
        q4e_gg_take16(r.w, r.odd, q);
    } else if (T == Q4E_G_IQ4_NL || T == Q4E_G_IQ4_XS) {
        if (T == Q4E_G_IQ4_NL) {
            scale = q4e_gg_h16(r.w[5]);
        } else {
            const uint32_t s = (t % 16u) / 2u;
            const uint32_t ls = ((r.w[7] >> (4u * (s % 2u))) & 0xFu) | (((r.w[6] >> (2u * s)) & 3u) << 4);
            scale = q4e_gg_h16(r.w[5]) * ((float) ls - 32.0f);
        }
        uint32_t raw[4];
        q4e_gg_take16(r.w, r.odd, raw);
        const uint32_t sh = (t & 1u) * 4u;
#pragma unroll
        for (int i = 0; i < 4; ++i) q[i] = q4e_gg_nl4((raw[i] >> sh) & 0x0F0F0F0Fu);
    } else if (T == Q4E_G_Q2_0) {
        scale = q4e_gg_h16(r.w[5]);
        const uint32_t w = q4e_gg_take4(r.w, r.odd);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t v = (w >> (8 * i)) & 0xFFu;
            const uint32_t c = (v & 3u) | ((v << 6) & 0x300u) | ((v << 12) & 0x30000u)
                             | ((v << 18) & 0x3000000u);
            q[i] = __vsub4(c, 0x01010101u);
        }
    } else if (T == Q4E_G_Q6_K) {
        const uint32_t c = t % 16u;
        const uint32_t k = (c / 2u) % 4u;
        uint32_t ql[4], qh[4];
        q4e_gg_take16(r.w, r.odd & 1u, ql);
        q4e_gg_take16(r.w + 5, r.odd >> 1, qh);
        scale = q4e_gg_h16(r.w[11]) * (float) (int) r.w[10];
        const uint32_t lshift = (k >> 1u) * 4u;
        const uint32_t hshift = 2u * k;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t v = ((ql[i] >> lshift) & 0x0F0F0F0Fu) | (((qh[i] >> hshift) & 0x03030303u) << 4);
            q[i] = __vsub4(v, 0x20202020u);
        }
    } else if (T == Q4E_G_IQ2_S) {
        const uint32_t s = (t % 16u) / 2u;
        const uint32_t hh = t & 1u;
        const uint32_t idx = q4e_gg_take4(r.w, r.odd & 1u) >> (16u * hh);
        const uint32_t sg = q4e_gg_take4(r.w + 2, r.odd >> 1) >> (16u * hh);
        const uint32_t qh = r.w[4];
        const uint32_t sc = r.w[5];
        scale = q4e_gg_h16(r.w[6]) * (0.5f + (float) (hh ? (sc >> 4) : (sc & 0xFu))) * 0.25f;
        (void) s;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t l = 2u * hh + (uint32_t) i;
            const uint32_t gi = ((idx >> (8 * i)) & 0xFFu) | ((qh << (8 - 2 * l)) & 0x300u);
            const uint64_t g = SHISO_IQ2S_GRID[gi];
            const uint32_t signs = (sg >> (8 * i)) & 0xFFu;
            q[2 * i] = q4e_gg_signed4((uint32_t) g, signs);
            q[2 * i + 1] = q4e_gg_signed4((uint32_t) (g >> 32), signs >> 4);
        }
    } else if (T == Q4E_G_IQ3_XXS) {
        const uint32_t hh = t & 1u;
        const uint32_t aux = q4e_gg_take4(r.w, r.odd & 1u);
        const uint32_t idx = q4e_gg_take4(r.w + 2, r.odd >> 1);
        scale = q4e_gg_h16(r.w[4]) * (0.5f + (float) ((aux >> 28) & 0xFu)) * 0.5f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const uint32_t l = 2u * hh + (uint32_t) i;
            const uint32_t signs = q4e_gg_ksigns((aux >> (7 * l)) & 0x7Fu);
            const uint32_t g1 = SHISO_IQ3XXS_GRID[(idx >> (16 * i)) & 0xFFu];
            const uint32_t g2 = SHISO_IQ3XXS_GRID[(idx >> (16 * i + 8)) & 0xFFu];
            q[2 * i] = q4e_gg_signed4(g1, signs);
            q[2 * i + 1] = q4e_gg_signed4(g2, signs >> 4);
        }
    }
    return scale;
}

/* Run `t` (weights `16 t .. 16 t + 15`) of a row as sixteen signed byte codes,
 * four a word, and the scale they are multiplied by. */
template <int T>
__device__ __forceinline__ float q4e_gg_codes16(const uint8_t *row, uint32_t t, uint32_t *q) {
    q4e_gg_raw r;
    q4e_gg_fetch<T>(row, t, r);
    return q4e_gg_decode<T>(r, t, q);
}

/* Run `t` of a row as eight packed half pairs of exact small integers, and
 * the scale they are multiplied by. */
template <int T>
__device__ __forceinline__ float q4e_gg_deq16(const uint8_t *row, uint32_t t, uint32_t *h) {
    uint32_t q[4];
    const float scale = q4e_gg_codes16<T>(row, t, q);
#pragma unroll
    for (int i = 0; i < 4; ++i) q4e_gg_s8_h2(q[i], h + 2 * i);
    return scale;
}

__device__ __forceinline__ void q4e_gg_mma0(float *c, const uint32_t *a, const uint32_t *b) {
    const float z = 0.0f;
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(z));
}

__device__ __forceinline__ void q4e_gg_ldsm4(uint32_t *r, const void *p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

/* Half of a tile's rows: `BM / 2` consecutive rows of one matrix from
 * `row0`, written to `out`. A dense or down tile is two halves of one
 * matrix; a gate/up tile is the same rows of both. */
struct q4e_gg_half {
    const uint8_t *w;
    float *out;
    uint32_t row0;
};

/* One `BM x BN` tile of `out[col][o] = sum_k w[o][k] x[col][k]`.
 *
 * `xrow[n]` is tile column `n`'s activation row (null past the columns) and
 * `col[n]` the row of `out` it writes. `BM / 16` warps, each 32 output rows
 * by `BN / 2` columns; a thread decodes one 16-weight run and converts
 * `32 BN / (2 BM)` activations a step, one step ahead of the products in
 * registers, into a double-buffered tile. */
template <int T, uint32_t BM, uint32_t BN>
__device__ __forceinline__ void q4e_gg_tile(uint8_t *smem, const q4e_gg_half *h,
                                            uint64_t row_bytes, uint32_t out_dim, uint32_t in_dim,
                                            const float *const *xrow, const uint32_t *col,
                                            uint32_t n_valid) {
    constexpr uint32_t THREADS = 2u * BM;
    constexpr uint32_t BPT = BN * Q4E_GG_BK / THREADS; /* activations a thread */
    constexpr uint32_t BQ = Q4E_GG_BK / BPT;           /* threads a column */
    constexpr uint32_t NT = BN / 16u;                  /* a warp's column groups of 8 */
    static_assert(BPT % 4u == 0u && BPT <= 16u, "a thread converts whole float4s of one run");
    __half(*sa)[BM][Q4E_GG_LD] = reinterpret_cast<__half(*)[BM][Q4E_GG_LD]>(smem);
    __half(*sb)[BN][Q4E_GG_LD] =
        reinterpret_cast<__half(*)[BN][Q4E_GG_LD]>(smem + 2u * BM * Q4E_GG_LD * 2u);
    float(*ss)[BM][2] =
        reinterpret_cast<float(*)[BM][2]>(smem + 2u * (BM + BN) * Q4E_GG_LD * 2u);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint32_t wm = warp >> 1;
    const uint32_t wn = warp & 1u;
    const uint32_t nk = in_dim / Q4E_GG_BK;

    const uint32_t ar = tid >> 1;
    const uint32_t ac = tid & 1u;
    const q4e_gg_half ha = ar < BM / 2u ? h[0] : h[1];
    const uint32_t a_o = ha.row0 + ar % (BM / 2u);
    const bool a_live = a_o < out_dim;
    const uint8_t *a_row = ha.w + (uint64_t) (a_live ? a_o : 0u) * row_bytes;
    const uint32_t bn = tid / BQ;
    const uint32_t bq = (tid % BQ) * BPT;

    const float *b_src = nullptr;
    uint32_t ra[8];
    uint32_t rb[BPT / 2u];
    float rs;
    /* A step's loads, issued two steps before its decode: the fetched raw
     * bytes and activations wait in registers while the step between runs
     * its products, so a step no longer waits out its own loads. */
    struct staged {
        q4e_gg_raw a;
        float4 b[BPT / 4u];
    };
    auto fetch = [&](uint32_t ks, staged &st) {
        if (a_live) q4e_gg_fetch<T>(a_row, ks * 2u + ac, st.a);
        if (b_src != nullptr) {
            const float4 *p = (const float4 *) (b_src + ks * Q4E_GG_BK + bq);
#pragma unroll
            for (uint32_t j = 0; j < BPT / 4u; ++j) st.b[j] = p[j];
        }
    };
    auto decode = [&](uint32_t ks, const staged &st) {
        if (a_live) {
            uint32_t q[4];
            rs = q4e_gg_decode<T>(st.a, ks * 2u + ac, q);
#pragma unroll
            for (int i = 0; i < 4; ++i) q4e_gg_s8_h2(q[i], ra + 2 * i);
        } else {
            rs = 0.0f;
#pragma unroll
            for (int j = 0; j < 8; ++j) ra[j] = 0u;
        }
        if (b_src != nullptr) {
#pragma unroll
            for (uint32_t j = 0; j < BPT / 4u; ++j) {
                rb[2 * j] = q4e_gg_h2(st.b[j].x, st.b[j].y);
                rb[2 * j + 1] = q4e_gg_h2(st.b[j].z, st.b[j].w);
            }
        } else {
#pragma unroll
            for (uint32_t j = 0; j < BPT / 2u; ++j) rb[j] = 0u;
        }
    };
    auto store = [&](uint32_t buf) {
        uint4 *pa = (uint4 *) &sa[buf][ar][ac * 16u];
        pa[0] = make_uint4(ra[0], ra[1], ra[2], ra[3]);
        pa[1] = make_uint4(ra[4], ra[5], ra[6], ra[7]);
        uint2 *pb = (uint2 *) &sb[buf][bn][bq];
#pragma unroll
        for (uint32_t j = 0; j < BPT / 4u; ++j) pb[j] = make_uint2(rb[2 * j], rb[2 * j + 1]);
        ss[buf][ar][ac] = rs;
    };

    float acc[2][NT][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
        for (uint32_t j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.0f;

    const bool n_on = wn * (BN / 2u) < n_valid;

    /* A previous tile is done with both buffers, and the column table the
     * caller filled is visible. */
    __syncthreads();
    b_src = xrow[bn];
    staged st0, st1;
    fetch(0u, st0);
    if (nk > 1u) fetch(1u, st1);
    decode(0u, st0);
    store(0u);
    __syncthreads();
    for (uint32_t ks = 0; ks < nk; ++ks) {
        const uint32_t buf = ks & 1u;
        /* st1 holds step ks + 1; step ks + 2 goes into st0, free since its
         * step was decoded. */
        if (ks + 2u < nk) fetch(ks + 2u, st0);
        if (n_on) {
#pragma unroll
            for (uint32_t c = 0; c < 2u; ++c) {
                uint32_t a[2][4];
#pragma unroll
                for (uint32_t mi = 0; mi < 2u; ++mi) {
                    const uint32_t r = wm * 32u + mi * 16u + (lane & 15u);
                    q4e_gg_ldsm4(a[mi], &sa[buf][r][c * 16u + (lane >> 4) * 8u]);
                }
                float sc[2][2];
#pragma unroll
                for (uint32_t mi = 0; mi < 2u; ++mi) {
                    sc[mi][0] = ss[buf][wm * 32u + mi * 16u + gid][c];
                    sc[mi][1] = ss[buf][wm * 32u + mi * 16u + gid + 8u][c];
                }
#pragma unroll
                for (uint32_t np = 0; np < NT / 2u; ++np) {
                    const uint32_t nb = wn * (BN / 2u) + np * 16u;
                    if (nb < n_valid) {
                        uint32_t b[4];
                        const uint32_t n = nb + (lane & 7u) + ((lane >> 4) << 3);
                        q4e_gg_ldsm4(b, &sb[buf][n][c * 16u + ((lane >> 3) & 1u) * 8u]);
#pragma unroll
                        for (uint32_t mi = 0; mi < 2u; ++mi) {
#pragma unroll
                            for (uint32_t hh = 0; hh < 2u; ++hh) {
                                float p[4];
                                q4e_gg_mma0(p, a[mi], b + 2u * hh);
                                float *o = acc[mi][2u * np + hh];
                                o[0] = fmaf(sc[mi][0], p[0], o[0]);
                                o[1] = fmaf(sc[mi][0], p[1], o[1]);
                                o[2] = fmaf(sc[mi][1], p[2], o[2]);
                                o[3] = fmaf(sc[mi][1], p[3], o[3]);
                            }
                        }
                    }
                }
            }
        }
        if (ks + 1u < nk) {
            decode(ks + 1u, st1);
            store(buf ^ 1u);
        }
        st1 = st0;
        __syncthreads();
    }

    /* A warp's 32 rows sit in one half. */
    const q4e_gg_half hw = wm * 32u < BM / 2u ? h[0] : h[1];
#pragma unroll
    for (uint32_t mi = 0; mi < 2u; ++mi) {
#pragma unroll
        for (uint32_t ni = 0; ni < NT; ++ni) {
#pragma unroll
            for (uint32_t e = 0; e < 4u; ++e) {
                const uint32_t o =
                    hw.row0 + (wm * 32u) % (BM / 2u) + mi * 16u + gid + (e >> 1) * 8u;
                const uint32_t n = wn * (BN / 2u) + ni * 8u + tig * 2u + (e & 1u);
                if (o < out_dim && n < n_valid) {
                    hw.out[(uint64_t) col[n] * out_dim + o] = acc[mi][ni][e];
                }
            }
        }
    }
}

/* A prompt chunk's dense projection: the `q4e_g_matvec_*` contract. `in_dim`
 * a multiple of 32, `x` sixteen-byte aligned. */
template <int T, uint32_t BM, uint32_t BN>
__device__ __forceinline__ void q4e_gg_matmul(float *out, const uint8_t *w, const float *x,
                                              uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    extern __shared__ __align__(16) uint8_t q4e_gg_dyn[];
    __shared__ const float *xrow[BN];
    __shared__ uint32_t col[BN];
    const uint32_t col0 = blockIdx.y * BN;
    if (threadIdx.x < BN) {
        const uint32_t c = col0 + threadIdx.x;
        xrow[threadIdx.x] = c < rows ? x + (uint64_t) c * in_dim : nullptr;
        col[threadIdx.x] = c;
    }
    const uint32_t row0 = blockIdx.x * BM;
    const q4e_gg_half h[2] = {{w, out, row0}, {w, out, row0 + BM / 2u}};
    q4e_gg_tile<T, BM, BN>(q4e_gg_dyn, h, q4e_g_row_bytes<T>(in_dim), out_dim, in_dim, xrow,
                           col, min(BN, rows - col0));
}

/* A routed stage of a prompt pass over an active expert's sorted columns,
 * `Q4E_GG_EXPERT_BN` at a time, its two halves as the entry sets them. */
template <int T, uint32_t BM>
__device__ __forceinline__ void q4e_gg_expert(const q4e_gg_half *h, const float *x,
                                              const int *sorted, int c0, int c1, uint32_t in_dim,
                                              uint32_t out_dim, uint32_t x_div) {
    extern __shared__ __align__(16) uint8_t q4e_gg_dyn[];
    constexpr uint32_t BN = Q4E_GG_EXPERT_BN;
    __shared__ const float *xrow[BN];
    __shared__ uint32_t col[BN];
    for (int cb = c0; cb < c1; cb += (int) BN) {
        __syncthreads(); /* the previous tile has read its column table */
        if (threadIdx.x < BN) {
            const int c = cb + (int) threadIdx.x;
            if (c < c1) {
                const uint32_t cc = (uint32_t) sorted[c];
                xrow[threadIdx.x] = x + (uint64_t) (cc / x_div) * in_dim;
                col[threadIdx.x] = cc;
            } else {
                xrow[threadIdx.x] = nullptr;
                col[threadIdx.x] = 0u;
            }
        }
        q4e_gg_tile<T, BM, BN>(q4e_gg_dyn, h, q4e_g_row_bytes<T>(in_dim), out_dim, in_dim,
                               xrow, col, min(BN, (uint32_t) (c1 - cb)));
    }
}

/* Gate and up of a prompt pass: `q4e_g_experts_gate_up_*`'s arguments. A
 * block covers the same `BM / 2` rows of both matrices, so one staged
 * activation tile serves both. */
template <int T, uint32_t BM>
__device__ __forceinline__ void q4e_gg_gate_up(
        float *out_gate, float *out_up, const uint8_t *gate_w, const uint8_t *up_w,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used, uint32_t gate_stride, uint32_t up_stride) {
    if ((int) blockIdx.y >= active[n_expert]) return;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) return;
    const uint64_t slot = map[expert];
    const uint32_t row0 = blockIdx.x * (BM / 2u);
    const q4e_gg_half h[2] = {{gate_w + slot * gate_stride, out_gate, row0},
                              {up_w + slot * up_stride, out_up, row0}};
    q4e_gg_expert<T, BM>(h, x, sorted, c0, c1, in_dim, out_dim, n_used);
}

/* The routed down projection of a prompt pass: `q4e_g_experts_down_*`'s
 * arguments, `BM` rows of the matrix a block. */
template <int T, uint32_t BM>
__device__ __forceinline__ void q4e_gg_down(
        float *out, const uint8_t *down_w, const float *mid, const int *sorted,
        const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,
        uint32_t out_dim, uint32_t n_expert, uint32_t stride) {
    if ((int) blockIdx.y >= active[n_expert]) return;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) return;
    const uint8_t *w = down_w + (uint64_t) map[expert] * stride;
    const uint32_t row0 = blockIdx.x * BM;
    const q4e_gg_half h[2] = {{w, out, row0}, {w, out, row0 + BM / 2u}};
    q4e_gg_expert<T, BM>(h, mid, sorted, c0, c1, in_dim, out_dim, 1u);
}

/* Grids: dense `(ceil(out_dim / Q4E_GG_DENSE_BM), ceil(rows /
 * Q4E_GG_DENSE_BN))`, gate/up `(ceil(out_dim / (Q4E_GG_GATE_UP_BM / 2)),
 * groups)`, down `(ceil(out_dim / Q4E_GG_DOWN_BM), groups)`; `2 * BM`
 * threads and `Q4E_GG_SMEM(BM, BN)` bytes of dynamic shared memory, each
 * entry at its own `BM` and `BN`. */
#define Q4E_GG_INSTANTIATE(T, NAME)                                                             \
    extern "C" __global__ void __launch_bounds__(2u * Q4E_GG_DENSE_BM)                          \
    q4e_g_gemm_##NAME##_kernel(float *out, const uint8_t *w, const float *x, uint32_t in_dim,   \
                               uint32_t out_dim, uint32_t rows) {                               \
        q4e_gg_matmul<T, Q4E_GG_DENSE_BM, Q4E_GG_DENSE_BN>(out, w, x, in_dim, out_dim, rows);                    \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(2u * Q4E_GG_GATE_UP_BM)                        \
    q4e_g_experts_gemm_gate_up_##NAME##_kernel(                                                 \
            float *out_gate, float *out_up, const uint8_t *gate_w, const uint8_t *up_w,         \
            const float *x, const int *sorted, const int *bounds, const int *active,            \
            const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,          \
            uint32_t n_used, uint32_t gate_stride, uint32_t up_stride) {                        \
        q4e_gg_gate_up<T, Q4E_GG_GATE_UP_BM>(out_gate, out_up, gate_w, up_w, x, sorted, bounds, \
                                             active, map, in_dim, out_dim, n_expert, n_used,    \
                                             gate_stride, up_stride);                           \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(2u * Q4E_GG_DOWN_BM)                           \
    q4e_g_experts_gemm_down_##NAME##_kernel(                                                    \
            float *out, const uint8_t *down_w, const float *mid, const int *sorted,             \
            const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,         \
            uint32_t out_dim, uint32_t n_expert, uint32_t n_used, uint32_t stride) {            \
        (void) n_used;                                                                          \
        q4e_gg_down<T, Q4E_GG_DOWN_BM>(out, down_w, mid, sorted, bounds, active, map, in_dim,   \
                                       out_dim, n_expert, stride);                              \
    }

Q4E_GG_INSTANTIATE(Q4E_G_Q8_0, q8_0)
Q4E_GG_INSTANTIATE(Q4E_G_Q6_K, q6_k)
Q4E_GG_INSTANTIATE(Q4E_G_IQ3_XXS, iq3_xxs)
Q4E_GG_INSTANTIATE(Q4E_G_IQ4_NL, iq4_nl)
Q4E_GG_INSTANTIATE(Q4E_G_IQ2_S, iq2_s)
Q4E_GG_INSTANTIATE(Q4E_G_IQ4_XS, iq4_xs)
Q4E_GG_INSTANTIATE(Q4E_G_Q2_0, q2_0)

/* ---------------------------------------------------------------------------
 * Step passes (a decode step or a verify) on CUDA.
 *
 * The step entries above give one warp to one output row, so a contraction of
 * 320 or 640 weights leaves most of a warp's lanes idle, and their 32-weight
 * decode reads a byte at a time. These give `lpr` lanes to a row - `32 / lpr`
 * rows ("units") a warp - and read 16-weight runs through the prompt
 * kernels' decoder: the run's own codes, a few aligned words a run, and its
 * stored scale. A unit's lane `j` takes runs `j, j + lpr, ...`, each run's
 * 16-term product with single-precision activations formed from zero and
 * added to the lane's sum scaled; the `lpr` lanes fold by butterfly. `lpr` is
 * the host's choice from the contraction width alone, so a row's sum does not
 * depend on how many activation rows share the pass, and a verify's rows are
 * a single-row step's.
 * ------------------------------------------------------------------------ */

/* Activation rows (or routed columns) a step pass carries in registers. */
#define Q4E_GS_COLS 8u
/* Warps a step block runs. */
#define Q4E_GS_WARPS 4u

/* A signed code byte `k` of `w`, exactly, as a float. */
__device__ __forceinline__ float q4e_gs_f(uint32_t u, uint32_t sel) {
    return __int_as_float((int) __byte_perm(u, 0x4B000000u, sel)) - 8388736.0f;
}

/* A run's sixteen codes as floats, exactly. */
__device__ __forceinline__ void q4e_gs_floats(const uint32_t *q, float *v) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t u = q[i] ^ 0x80808080u;
        v[4 * i] = q4e_gs_f(u, 0x7440u);
        v[4 * i + 1] = q4e_gs_f(u, 0x7441u);
        v[4 * i + 2] = q4e_gs_f(u, 0x7442u);
        v[4 * i + 3] = q4e_gs_f(u, 0x7443u);
    }
}

/* `sum_j code_j * x_j` over a run's sixteen codes, from zero, in index
 * order. */
__device__ __forceinline__ float q4e_gs_dot16(const float *v, const float *x) {
    const float4 *x4 = (const float4 *) x;
    float s = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float4 a = x4[i];
        s = fmaf(v[4 * i], a.x, s);
        s = fmaf(v[4 * i + 1], a.y, s);
        s = fmaf(v[4 * i + 2], a.z, s);
        s = fmaf(v[4 * i + 3], a.w, s);
    }
    return s;
}

/* Run `t` of a row as sixteen floats and their scale: the codes of a block
 * format, exactly, or a single-precision row's own values (scale one). */
template <int T>
__device__ __forceinline__ float q4e_gs_run(const uint8_t *row, uint32_t t, float *v) {
    if (T == Q4E_G_F32) {
        const float4 *p = (const float4 *) (row + (uint64_t) t * 64ull);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 f = p[i];
            v[4 * i] = f.x;
            v[4 * i + 1] = f.y;
            v[4 * i + 2] = f.z;
            v[4 * i + 3] = f.w;
        }
        return 1.0f;
    } else {
        uint32_t q[4];
        const float scale = q4e_gg_codes16<T>(row, t, q);
        q4e_gs_floats(q, v);
        return scale;
    }
}

/* One unit's share of `acc[c] = row . x_c` for up to `Q4E_GS_COLS` columns:
 * the runs `j, j + lpr, ...` of a row of `runs` runs, each decoded once for
 * every column. */
template <int T>
__device__ __forceinline__ void q4e_gs_row(const uint8_t *row, uint32_t runs, uint32_t j,
                                           uint32_t lpr, const float *x, const uint64_t *xoff,
                                           uint32_t n_cols, float *acc) {
#pragma unroll 2
    for (uint32_t t = j; t < runs; t += lpr) {
        float v[16];
        const float scale = q4e_gs_run<T>(row, t, v);
#pragma unroll
        for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) {
            if (c < n_cols) acc[c] = fmaf(scale, q4e_gs_dot16(v, x + xoff[c] + t * 16u), acc[c]);
        }
    }
}

/* Folds a unit's `lpr` lanes, lowest offset last; every lane ends with the
 * sum. */
__device__ __forceinline__ float q4e_gs_fold(float v, uint32_t lpr) {
    for (uint32_t off = lpr >> 1; off > 0u; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
    return v;
}

/* `out[r * out_dim + o] = W[o] . x[r]`: `q4e_g_matvec_*`'s arguments and
 * `lpr` (a power of two up to `32 * Q4E_GS_WARPS`). Grid `(ceil(out_dim /
 * (Q4E_GS_WARPS * 32 / lpr)), ceil(rows / Q4E_GS_COLS))`, `32 *
 * Q4E_GS_WARPS` threads. */
/* `R` consecutive rows from `o0` at once: each run's activations are read
 * once for all of them. A row's sum is the one-row sum, in the same order. */
template <int T, uint32_t R>
__device__ __forceinline__ void q4e_gs_rows(const uint8_t *w, uint64_t row_bytes, uint32_t o0,
                                            uint32_t out_dim, uint32_t runs, uint32_t j,
                                            uint32_t lpr, const float *x, const uint64_t *xoff,
                                            uint32_t n_cols, float (*acc)[Q4E_GS_COLS]) {
    if (R == 1u) {
        if (o0 < out_dim) q4e_gs_row<T>(w + (uint64_t) o0 * row_bytes, runs, j, lpr, x, xoff, n_cols, acc[0]);
        return;
    }
    const uint8_t *row[R];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) row[r] = w + (uint64_t) min(o0 + r, out_dim - 1u) * row_bytes;
    for (uint32_t t = j; t < runs; t += lpr) {
        float v[R][16];
        float scale[R];
#pragma unroll
        for (uint32_t r = 0; r < R; ++r) scale[r] = q4e_gs_run<T>(row[r], t, v[r]);
#pragma unroll
        for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) {
            if (c < n_cols) {
                const float4 *x4 = (const float4 *) (x + xoff[c] + t * 16u);
                float s[R];
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) s[r] = 0.0f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const float4 a = x4[i];
#pragma unroll
                    for (uint32_t r = 0; r < R; ++r) {
                        s[r] = fmaf(v[r][4 * i], a.x, s[r]);
                        s[r] = fmaf(v[r][4 * i + 1], a.y, s[r]);
                        s[r] = fmaf(v[r][4 * i + 2], a.z, s[r]);
                        s[r] = fmaf(v[r][4 * i + 3], a.w, s[r]);
                    }
                }
#pragma unroll
                for (uint32_t r = 0; r < R; ++r) acc[r][c] = fmaf(scale[r], s[r], acc[r][c]);
            }
        }
    }
}

template <int T, uint32_t R>
__device__ __forceinline__ void q4e_gs_matvec(float *out, const uint8_t *w, const float *x,
                                              uint32_t in_dim, uint32_t out_dim, uint32_t rows,
                                              uint32_t lpr) {
    if (lpr > 32u) {
        /* `lpr / 32` warps a row, `Q4E_GS_WARPS * 32 / lpr` rows a block: a
         * narrow output over a wide contraction (the stream mixes' 320-row
         * contraction of 10240) has too few rows for a warp each to keep the
         * memory system busy. Each warp folds its lanes, then the row's warps
         * fold through shared memory, lowest warp first. */
        const uint32_t warp = threadIdx.x >> 5;
        const uint32_t lane = threadIdx.x & 31u;
        const uint32_t wpr = lpr / 32u;
        __shared__ float rpart[R][Q4E_GS_WARPS][Q4E_GS_COLS];
        const uint32_t o0 = (blockIdx.x * (Q4E_GS_WARPS / wpr) + warp / wpr) * R;
        const uint32_t j = (warp % wpr) * 32u + lane;
        const uint32_t r0 = blockIdx.y * Q4E_GS_COLS;
        const uint32_t n_cols = min(Q4E_GS_COLS, rows - r0);
        const uint64_t rb = q4e_g_row_bytes<T>(in_dim);
        uint64_t xoff[Q4E_GS_COLS];
#pragma unroll
        for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) xoff[c] = (uint64_t) (r0 + min(c, n_cols - 1u)) * in_dim;
        float acc[R][Q4E_GS_COLS];
#pragma unroll
        for (uint32_t r = 0; r < R; ++r)
#pragma unroll
            for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) acc[r][c] = 0.0f;
        q4e_gs_rows<T, R>(w, rb, o0, out_dim, in_dim / 16u, j, lpr, x, xoff, n_cols, acc);
#pragma unroll
        for (uint32_t r = 0; r < R; ++r)
#pragma unroll
            for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) {
                const float v = q4e_gs_fold(acc[r][c], 32u);
                if (lane == 0u) rpart[r][warp][c] = v;
            }
        __syncthreads();
        if (warp % wpr == 0u && lane < n_cols) {
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                if (o0 + r < out_dim) {
                    float v = rpart[r][warp][lane];
                    for (uint32_t k = 1; k < wpr; ++k) v += rpart[r][warp + k][lane];
                    out[(uint64_t) (r0 + lane) * out_dim + o0 + r] = v;
                }
            }
        }
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t units = 32u / lpr;
    const uint32_t o0 = ((blockIdx.x * Q4E_GS_WARPS + (threadIdx.x >> 5)) * units + lane / lpr) * R;
    const uint32_t j = lane % lpr;
    const uint32_t r0 = blockIdx.y * Q4E_GS_COLS;
    const uint32_t n_cols = min(Q4E_GS_COLS, rows - r0);
    uint64_t xoff[Q4E_GS_COLS];
#pragma unroll
    for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) xoff[c] = (uint64_t) (r0 + min(c, n_cols - 1u)) * in_dim;
    float acc[R][Q4E_GS_COLS];
#pragma unroll
    for (uint32_t r = 0; r < R; ++r)
#pragma unroll
        for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) acc[r][c] = 0.0f;
    q4e_gs_rows<T, R>(w, q4e_g_row_bytes<T>(in_dim), o0, out_dim, in_dim / 16u, j, lpr, x, xoff,
                      n_cols, acc);
#pragma unroll
    for (uint32_t r = 0; r < R; ++r)
#pragma unroll
        for (uint32_t c = 0; c < Q4E_GS_COLS; ++c) {
            const float v = q4e_gs_fold(acc[r][c], lpr);
            if (o0 + r < out_dim && j == 0u && c < n_cols) out[(uint64_t) (r0 + c) * out_dim + o0 + r] = v;
        }
}

/* One routed unit over an active expert's sorted columns. */
template <int T>
__device__ __forceinline__ void q4e_gs_expert_unit(float *out, const uint8_t *w, const float *x,
                                                   const int *sorted, int c0, int c1,
                                                   uint32_t o, bool live, uint32_t in_dim,
                                                   uint32_t out_dim, uint32_t x_div, uint32_t j,
                                                   uint32_t lpr) {
    const uint8_t *row = w + (uint64_t) (live ? o : 0u) * q4e_g_row_bytes<T>(in_dim);
    for (int c = c0; c < c1; c += (int) Q4E_GS_COLS) {
        const uint32_t n_cols = min(Q4E_GS_COLS, (uint32_t) (c1 - c));
        uint64_t xoff[Q4E_GS_COLS];
        uint32_t cols[Q4E_GS_COLS];
#pragma unroll
        for (uint32_t k = 0; k < Q4E_GS_COLS; ++k) {
            cols[k] = (uint32_t) sorted[c + (int) min(k, n_cols - 1u)];
            xoff[k] = (uint64_t) (cols[k] / x_div) * in_dim;
        }
        float acc[Q4E_GS_COLS];
#pragma unroll
        for (uint32_t k = 0; k < Q4E_GS_COLS; ++k) acc[k] = 0.0f;
        if (live) q4e_gs_row<T>(row, in_dim / 16u, j, lpr, x, xoff, n_cols, acc);
#pragma unroll
        for (uint32_t k = 0; k < Q4E_GS_COLS; ++k) {
            const float v = q4e_gs_fold(acc[k], lpr);
            if (live && j == 0u && k < n_cols) out[(uint64_t) cols[k] * out_dim + o] = v;
        }
    }
}

/* Gate and up of a step pass: `q4e_g_experts_gate_up_*`'s arguments and
 * `lpr`. A warp's units alternate gate and up over `16 / lpr` rows, so a
 * block covers `Q4E_GS_WARPS * 16 / lpr` rows of both. Grid `(ceil(out_dim /
 * (Q4E_GS_WARPS * 16 / lpr)), groups)`, `32 * Q4E_GS_WARPS` threads. */
template <int T>
__device__ __forceinline__ void q4e_gs_gate_up(
        float *out_gate, float *out_up, const uint8_t *gate_w, const uint8_t *up_w,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used, uint32_t gate_stride, uint32_t up_stride, uint32_t lpr) {
    if ((int) blockIdx.y >= active[n_expert]) return;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t unit = (blockIdx.x * Q4E_GS_WARPS + (threadIdx.x >> 5)) * (32u / lpr) + lane / lpr;
    const uint32_t o = unit >> 1;
    const bool up = (unit & 1u) != 0u;
    const uint64_t slot = map[expert];
    const uint8_t *w = up ? up_w + slot * up_stride : gate_w + slot * gate_stride;
    q4e_gs_expert_unit<T>(up ? out_up : out_gate, w, x, sorted, c0, c1, o, o < out_dim, in_dim,
                          out_dim, n_used, lane % lpr, lpr);
}

/* The routed down projection of a step pass: `q4e_g_experts_down_*`'s
 * arguments and `lpr`. Grid `(ceil(out_dim / (Q4E_GS_WARPS * 32 / lpr)),
 * groups)`, `32 * Q4E_GS_WARPS` threads. */
template <int T>
__device__ __forceinline__ void q4e_gs_down(
        float *out, const uint8_t *down_w, const float *mid, const int *sorted,
        const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,
        uint32_t out_dim, uint32_t n_expert, uint32_t stride, uint32_t lpr) {
    if ((int) blockIdx.y >= active[n_expert]) return;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) return;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t o = (blockIdx.x * Q4E_GS_WARPS + (threadIdx.x >> 5)) * (32u / lpr) + lane / lpr;
    const uint8_t *w = down_w + (uint64_t) map[expert] * stride;
    q4e_gs_expert_unit<T>(out, w, mid, sorted, c0, c1, o, o < out_dim, in_dim, out_dim, 1u,
                          lane % lpr, lpr);
}

#define Q4E_GS_INSTANTIATE(T, NAME)                                                             \
    extern "C" __global__ void __launch_bounds__(32u * Q4E_GS_WARPS)                            \
    q4e_g_step_matvec_##NAME##_kernel(float *out, const uint8_t *w, const float *x,             \
                                      uint32_t in_dim, uint32_t out_dim, uint32_t rows,         \
                                      uint32_t lpr) {                                           \
        q4e_gs_matvec<T, 1u>(out, w, x, in_dim, out_dim, rows, lpr);                            \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(32u * Q4E_GS_WARPS)                            \
    q4e_g_step_matvec4_##NAME##_kernel(float *out, const uint8_t *w, const float *x,            \
                                       uint32_t in_dim, uint32_t out_dim, uint32_t rows,        \
                                       uint32_t lpr) {                                          \
        q4e_gs_matvec<T, 4u>(out, w, x, in_dim, out_dim, rows, lpr);                            \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(32u * Q4E_GS_WARPS)                            \
    q4e_g_step_gate_up_##NAME##_kernel(                                                         \
            float *out_gate, float *out_up, const uint8_t *gate_w, const uint8_t *up_w,         \
            const float *x, const int *sorted, const int *bounds, const int *active,            \
            const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,          \
            uint32_t n_used, uint32_t gate_stride, uint32_t up_stride, uint32_t lpr) {          \
        q4e_gs_gate_up<T>(out_gate, out_up, gate_w, up_w, x, sorted, bounds, active, map,       \
                          in_dim, out_dim, n_expert, n_used, gate_stride, up_stride, lpr);      \
    }                                                                                           \
    extern "C" __global__ void __launch_bounds__(32u * Q4E_GS_WARPS)                            \
    q4e_g_step_down_##NAME##_kernel(                                                            \
            float *out, const uint8_t *down_w, const float *mid, const int *sorted,             \
            const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,         \
            uint32_t out_dim, uint32_t n_expert, uint32_t n_used, uint32_t stride,              \
            uint32_t lpr) {                                                                     \
        (void) n_used;                                                                          \
        q4e_gs_down<T>(out, down_w, mid, sorted, bounds, active, map, in_dim, out_dim,          \
                       n_expert, stride, lpr);                                                  \
    }

extern "C" __global__ void __launch_bounds__(32u * Q4E_GS_WARPS)
q4e_g_step_matvec_f32_kernel(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
                             uint32_t out_dim, uint32_t rows, uint32_t lpr) {
    q4e_gs_matvec<Q4E_G_F32, 1u>(out, w, x, in_dim, out_dim, rows, lpr);
}
extern "C" __global__ void __launch_bounds__(32u * Q4E_GS_WARPS)
q4e_g_step_matvec4_f32_kernel(float *out, const uint8_t *w, const float *x, uint32_t in_dim,
                              uint32_t out_dim, uint32_t rows, uint32_t lpr) {
    q4e_gs_matvec<Q4E_G_F32, 4u>(out, w, x, in_dim, out_dim, rows, lpr);
}
Q4E_GS_INSTANTIATE(Q4E_G_Q8_0, q8_0)
Q4E_GS_INSTANTIATE(Q4E_G_Q6_K, q6_k)
Q4E_GS_INSTANTIATE(Q4E_G_IQ3_XXS, iq3_xxs)
Q4E_GS_INSTANTIATE(Q4E_G_IQ4_NL, iq4_nl)
Q4E_GS_INSTANTIATE(Q4E_G_IQ2_S, iq2_s)
Q4E_GS_INSTANTIATE(Q4E_G_IQ4_XS, iq4_xs)
Q4E_GS_INSTANTIATE(Q4E_G_Q2_0, q2_0)
