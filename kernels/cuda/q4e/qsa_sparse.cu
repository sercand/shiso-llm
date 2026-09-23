/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels);
 * D2 (shiso): q4e_qsa_attention_split_kernel gives each row its own block -
 * its G query heads - and cuts that row's own key range, where the source
 * packed Q4E_ATTN_QG (head, row) pairs a block and cut the key range of the
 * pass's last row.  A row's sums then do not depend on the rows beside it, so
 * a verify pass or a decode batch computes each row exactly as a single-row
 * step does.
 * No other line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 *
 * The leading define block (2907-2909) is the one place a span is
 * carried into two units: the source defines the key-tile geometry once,
 * inside the dense tiled-attention block, and both the dense kernel and
 * the split/gather kernels here read Q4E_ATTN_KT from it. Both copies are
 * the same three source lines and the fidelity gate checks both against
 * them, so they cannot drift apart.
 */

#include "q4e.cuh"

#define Q4E_ATTN_QT 16u          /* queries per block */
#define Q4E_ATTN_KT 32u          /* keys per tile: one per lane */
#define Q4E_ATTN_TILE_WARPS 8u

/* Split-key attention for decode and the speculative verify batch.
 *
 * The per-query kernel above gives one block per (head, query) and walks the
 * whole cache from that block, so at 26k context a 4-row verify pass ran 96
 * blocks that each streamed 53 MB serially: 230 ms of a 390 ms step, latency
 * bound at a few GB/s.  Here the key range is cut into Q4E_ATTN_SPLITS pieces
 * and a block owns one piece for one key/value head and up to Q4E_ATTN_QG of
 * the queries that share it -- every query head in the group, every row of the
 * batch -- so each K/V tile is staged once for 16 queries and the grid covers
 * the GPU.  Blocks leave (max, denominator, accumulator) partials that
 * q4e_qsa_attention_combine_kernel merges; with one split the result is
 * written directly.
 *
 * The split count is fixed rather than derived from the context length so the
 * launch geometry is stable under graph capture: the tiles per split come from
 * pos[] on the device, and splits past the end write an empty partial.
 *
 * Inner loops follow the tiled prefill kernel -- key l lives in lane l for the
 * scores, lane l owns dims 8l..8l+7 for the accumulation -- staged in 16-byte
 * f16 chunks so each shared load carries 8 dims and neither loop reduces.
 * Specialised to head_dim 256 (every qwen4exp attention layer). */
#define Q4E_ATTN_QG      16u          /* queries per block: 2 per warp */
#define Q4E_ATTN_SPLITS  48u          /* key splits per (kv head, query group) */
#define Q4E_ATTN_SPLIT_HD 256u
#define Q4E_ATTN_PART_STRIDE (Q4E_ATTN_SPLIT_HD + 8u)   /* m, l, pad, acc[256] */

extern "C" __global__ void __launch_bounds__(256, 2)
q4e_qsa_attention_split_kernel(
        float *out, float *part, const __half *k_cache, const __half *v_cache,
        const float *q, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_ATTN_SPLIT_HD;
    constexpr uint32_t KT = Q4E_ATTN_KT;
    constexpr uint32_t C8 = HD / 8u;                    /* 16-byte chunks per row */
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t hkv = blockIdx.z;
    const uint32_t split = blockIdx.y;
    const uint32_t G = n_head / n_head_kv;              /* query heads per kv head */
    const uint32_t NQ = G * n_tok;
    /* D2: block x is row x, all G of its query heads (G <= QG, which the
     * launcher checks). */
    const uint32_t q0 = blockIdx.x * G;
    if (q0 >= NQ || G > Q4E_ATTN_QG) return;
    const uint32_t n_q = G;
    const float scale = rsqrtf((float)HD);

    /* Everything staged is f16: a 16-byte shared load then carries 8 dims,
     * which halves the shared-memory wavefronts of an f32 layout, and the
     * 41 KB footprint lets two blocks share an SM so one block's cache loads
     * overlap the other's arithmetic.  Accumulation stays f32. */
    extern __shared__ __align__(16) uint4 q4e_split_smem[];
    uint4 *s_k8 = q4e_split_smem;                          /* [C8][KT]: chunk-major */
    uint4 *s_v8 = s_k8 + (size_t)C8 * KT;                  /* [KT][C8]: key-major */
    uint4 *s_q8 = s_v8 + (size_t)KT * C8;                  /* [QG][C8] */
    float *s_w  = (float *)(s_q8 + (size_t)Q4E_ATTN_QG * C8); /* [warps][KT] */

    /* This block's key range: its own row's (D2). */
    const uint32_t last = (uint32_t)pos[blockIdx.x];
    const uint32_t n_tiles = (last + KT) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t k_begin = split * tiles_per_split * KT;
    uint32_t k_end = k_begin + tiles_per_split * KT;
    if (k_end > last + 1u) k_end = last + 1u;

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[s][j] = 0.0f;
    }

    if (k_begin < k_end) {
        /* Stage the scaled queries as f16. */
        for (uint32_t idx = tid; idx < n_q * C8; idx += blockDim.x) {
            const uint32_t qi = idx / C8, c = idx % C8;
            const uint32_t t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + h) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            union { uint4 u; __half2 h2[4]; } pk;
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
            s_q8[(size_t)qi * C8 + c] = pk.u;
        }
        __syncthreads();

        for (uint32_t p0 = k_begin; p0 < k_end; p0 += KT) {
            const uint32_t n_k = (k_end - p0 < KT) ? (k_end - p0) : KT;
            /* k_begin and the stride are multiples of KT, so this tile lies in
             * one page: one page-table lookup covers all 32 keys. */
            const uint32_t krow = q4e_kv_row(pages, p0);
            /* K: lane = key, warp = chunk (4 passes cover 32 chunks); stored
             * chunk-major so a lane's score loop reads its own column. */
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_k8[(size_t)c * KT + key] = *(const uint4 *)(k_cache +
                            (((uint64_t)(krow + key) * n_head_kv + hkv) * HD + 8u * c));
                }
            }
            /* V: lane = chunk, warp = key (4 passes cover 32 keys); key-major
             * so the accumulation reads a lane's own 8 dims of each key. */
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_v8[(size_t)key * C8 + c] = *(const uint4 *)(v_cache +
                            (((uint64_t)(krow + key) * n_head_kv + hkv) * HD + 8u * c));
                } else {
                    /* The accumulation runs the whole tile with zero weights on
                     * the tail, and 0 * NaN from stale memory is NaN. */
                    s_v8[(size_t)key * C8 + c] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();

            /* Scores: key p0 + lane against both of this warp's queries in one
             * pass over the staged K, four partial sums each so the FMA chain
             * is 64 deep rather than 256. */
            const uint32_t qi0 = warp * 2u;
            const uint32_t nq_w = (qi0 >= n_q) ? 0u : ((n_q - qi0 >= 2u) ? 2u : 1u);
            if (nq_w == 0u) { __syncthreads(); continue; }
            const uint4 *qrow0 = s_q8 + (size_t)qi0 * C8;
            const uint4 *qrow1 = s_q8 + (size_t)(qi0 + (nq_w - 1u)) * C8;
            float dot[2] = { -INFINITY, -INFINITY };
            if (lane < n_k) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll 4
                for (uint32_t c = 0; c < C8; c++) {
                    union { uint4 u; __half2 h2[4]; } kk, qa, qb;
                    kk.u = s_k8[(size_t)c * KT + lane];
                    qa.u = qrow0[c];
                    qb.u = qrow1[c];
                    const float2 k0 = __half22float2(kk.h2[0]), k1 = __half22float2(kk.h2[1]);
                    const float2 k2 = __half22float2(kk.h2[2]), k3 = __half22float2(kk.h2[3]);
                    const float2 x0 = __half22float2(qa.h2[0]), x1 = __half22float2(qa.h2[1]);
                    const float2 x2 = __half22float2(qa.h2[2]), x3 = __half22float2(qa.h2[3]);
                    const float2 y0 = __half22float2(qb.h2[0]), y1 = __half22float2(qb.h2[1]);
                    const float2 y2 = __half22float2(qb.h2[2]), y3 = __half22float2(qb.h2[3]);
                    a0 = fmaf(x0.x, k0.x, a0); a1 = fmaf(x0.y, k0.y, a1);
                    a2 = fmaf(x1.x, k1.x, a2); a3 = fmaf(x1.y, k1.y, a3);
                    a0 = fmaf(x2.x, k2.x, a0); a1 = fmaf(x2.y, k2.y, a1);
                    a2 = fmaf(x3.x, k3.x, a2); a3 = fmaf(x3.y, k3.y, a3);
                    b0 = fmaf(y0.x, k0.x, b0); b1 = fmaf(y0.y, k0.y, b1);
                    b2 = fmaf(y1.x, k1.x, b2); b3 = fmaf(y1.y, k1.y, b3);
                    b0 = fmaf(y2.x, k2.x, b0); b1 = fmaf(y2.y, k2.y, b1);
                    b2 = fmaf(y3.x, k3.x, b2); b3 = fmaf(y3.y, k3.y, b3);
                }
                dot[0] = (a0 + a1) + (a2 + a3);
                dot[1] = (b0 + b1) + (b2 + b3);
            }

#pragma unroll
            for (int s = 0; s < 2; s++) {
                if ((uint32_t)s >= nq_w) break;
                const uint32_t qi = qi0 + (uint32_t)s;
                const uint32_t qpos = (uint32_t)pos[(q0 + qi) / G];
                const float d = (lane < n_k && p0 + lane <= qpos) ? dot[s] : -INFINITY;
                float tile_max = d;
                for (int off = 16; off > 0; off >>= 1) {
                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
                }
                if (tile_max == -INFINITY) continue;
                const float m_new = fmaxf(m[s], tile_max);
                const float corr = __expf(m[s] - m_new);
                const float w = __expf(d - m_new);          /* 0 where masked */
                s_w[(size_t)warp * KT + lane] = w;
                float wsum = w;
                for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
                l[s] = l[s] * corr + wsum;
                m[s] = m_new;
                __syncwarp();

                /* Lane owns dims 8*lane .. 8*lane+7: one 16-byte load per key,
                 * lanes contiguous, so no bank conflicts.  Masked keys carry a
                 * zero weight, so the loop always runs the full tile. */
#pragma unroll
                for (int j = 0; j < 8; j++) acc[s][j] *= corr;
                const float *wrow = s_w + (size_t)warp * KT;
                const uint4 *vcol = s_v8 + lane;
#pragma unroll 4
                for (uint32_t k = 0; k < KT; k++) {
                    const float wk = wrow[k];
                    union { uint4 u; __half2 h2[4]; } vv;
                    vv.u = vcol[(size_t)k * C8];
                    const float2 v0 = __half22float2(vv.h2[0]), v1 = __half22float2(vv.h2[1]);
                    const float2 v2 = __half22float2(vv.h2[2]), v3 = __half22float2(vv.h2[3]);
                    acc[s][0] = fmaf(wk, v0.x, acc[s][0]);
                    acc[s][1] = fmaf(wk, v0.y, acc[s][1]);
                    acc[s][2] = fmaf(wk, v1.x, acc[s][2]);
                    acc[s][3] = fmaf(wk, v1.y, acc[s][3]);
                    acc[s][4] = fmaf(wk, v2.x, acc[s][4]);
                    acc[s][5] = fmaf(wk, v2.y, acc[s][5]);
                    acc[s][6] = fmaf(wk, v3.x, acc[s][6]);
                    acc[s][7] = fmaf(wk, v3.y, acc[s][7]);
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }

    /* Epilogue: the final row when there is one split, a partial otherwise. */
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const uint32_t qi = warp * 2u + (uint32_t)s;
        if (qi >= n_q) break;
        const uint32_t t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
        const uint64_t qg = (uint64_t)t * n_head + h;
        if (n_splits == 1u) {
            const float inv = 1.0f / l[s];
            float4 *dst = (float4 *)(out + qg * HD + 8u * lane);
            dst[0] = make_float4(acc[s][0] * inv, acc[s][1] * inv, acc[s][2] * inv, acc[s][3] * inv);
            dst[1] = make_float4(acc[s][4] * inv, acc[s][5] * inv, acc[s][6] * inv, acc[s][7] * inv);
        } else {
            float *prow = part + (qg * n_splits + split) * Q4E_ATTN_PART_STRIDE;
            if (lane == 0u) { prow[0] = m[s]; prow[1] = l[s]; }
            float4 *dst = (float4 *)(prow + 8u + 8u * lane);
            dst[0] = make_float4(acc[s][0], acc[s][1], acc[s][2], acc[s][3]);
            dst[1] = make_float4(acc[s][4], acc[s][5], acc[s][6], acc[s][7]);
        }
    }
}

/* One block per (row, head); thread d merges dim d across the splits. */
extern "C" __global__ void q4e_qsa_attention_combine_kernel(
        float *out, const float *part, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_ATTN_SPLIT_HD;
    const uint64_t qg = blockIdx.x;
    const float *base = part + qg * n_splits * Q4E_ATTN_PART_STRIDE;
    const uint32_t d = threadIdx.x;
    float gm = -INFINITY;
    for (uint32_t sp = 0; sp < n_splits; sp++) gm = fmaxf(gm, base[(size_t)sp * Q4E_ATTN_PART_STRIDE]);
    float den = 0.0f, num = 0.0f;
    for (uint32_t sp = 0; sp < n_splits; sp++) {
        const float *prow = base + (size_t)sp * Q4E_ATTN_PART_STRIDE;
        const float ms = prow[0];
        if (ms == -INFINITY) continue;
        const float wgt = __expf(ms - gm);
        den = fmaf(prow[1], wgt, den);
        num = fmaf(prow[8u + d], wgt, num);
    }
    out[qg * HD + d] = num / den;
}

/* Sparse attention over a per-row token list: the split kernel's structure
 * with gathered K/V rows.  A block owns one row t, one kv head and all of
 * that head's query heads (12 -> 6 warps x 2), and one of Q4E_ATTN_GSPLITS
 * slices of the row's token list; partials merge in the combine kernel. */
#define Q4E_ATTN_GSPLITS 8u
extern "C" __global__ void __launch_bounds__(256, 2)
q4e_qsa_attention_gather_kernel(
        float *part, const __half *k_cache, const __half *v_cache, const float *q,
        const int32_t *tokens, const int32_t *n_sel, const int32_t *pages, uint32_t width,
        uint32_t n_head, uint32_t n_head_kv, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_ATTN_SPLIT_HD;
    constexpr uint32_t KT = Q4E_ATTN_KT;
    constexpr uint32_t C8 = HD / 8u;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t t = blockIdx.z, hkv = blockIdx.y, split = blockIdx.x;
    const uint32_t G = n_head / n_head_kv;
    const float scale = rsqrtf((float)HD);
    const int32_t *list = tokens + (uint64_t)t * width;
    const uint32_t n_list = (uint32_t)n_sel[t];
    const uint32_t n_tiles = (n_list + KT - 1u) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t l_begin = split * tiles_per_split * KT;
    uint32_t l_end = l_begin + tiles_per_split * KT;
    if (l_end > n_list) l_end = n_list;

    extern __shared__ __align__(16) uint4 q4e_gather_smem[];
    uint4 *s_k8 = q4e_gather_smem;
    uint4 *s_v8 = s_k8 + (size_t)C8 * KT;
    uint4 *s_q8 = s_v8 + (size_t)KT * C8;
    float *s_w  = (float *)(s_q8 + (size_t)Q4E_ATTN_QG * C8);
    __shared__ int32_t s_idx[KT];

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[s][j] = 0.0f;
    }

    if (l_begin < l_end) {
        for (uint32_t idx = tid; idx < G * C8; idx += blockDim.x) {
            const uint32_t gh = idx / C8, c = idx % C8;
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + hkv * G + gh) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            union { uint4 u; __half2 h2[4]; } pk;
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
            s_q8[(size_t)gh * C8 + c] = pk.u;
        }
        __syncthreads();
        for (uint32_t p0 = l_begin; p0 < l_end; p0 += KT) {
            const uint32_t n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
            /* The token list is arbitrary positions, so translation is per key
             * -- but it happens once here, in the same staging step that
             * already reads the list, and the tile then indexes cache rows. */
            if (tid < KT) {
                s_idx[tid] = tid < n_k ? (int32_t)q4e_kv_row(pages, (uint32_t)list[p0 + tid]) : 0;
            }
            __syncthreads();
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_k8[(size_t)c * KT + key] = *(const uint4 *)(k_cache +
                            (((uint64_t)s_idx[key] * n_head_kv + hkv) * HD + 8u * c));
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_v8[(size_t)key * C8 + c] = *(const uint4 *)(v_cache +
                            (((uint64_t)s_idx[key] * n_head_kv + hkv) * HD + 8u * c));
                } else {
                    s_v8[(size_t)key * C8 + c] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();

            const uint32_t qi0 = warp * 2u;
            const uint32_t nq_w = (qi0 >= G) ? 0u : ((G - qi0 >= 2u) ? 2u : 1u);
            if (nq_w == 0u) { __syncthreads(); continue; }
            const uint4 *qrow0 = s_q8 + (size_t)qi0 * C8;
            const uint4 *qrow1 = s_q8 + (size_t)(qi0 + (nq_w - 1u)) * C8;
            float dot[2] = { -INFINITY, -INFINITY };
            if (lane < n_k) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll 4
                for (uint32_t c = 0; c < C8; c++) {
                    union { uint4 u; __half2 h2[4]; } kk, qa, qb;
                    kk.u = s_k8[(size_t)c * KT + lane];
                    qa.u = qrow0[c];
                    qb.u = qrow1[c];
                    const float2 k0 = __half22float2(kk.h2[0]), k1 = __half22float2(kk.h2[1]);
                    const float2 k2 = __half22float2(kk.h2[2]), k3 = __half22float2(kk.h2[3]);
                    const float2 x0 = __half22float2(qa.h2[0]), x1 = __half22float2(qa.h2[1]);
                    const float2 x2 = __half22float2(qa.h2[2]), x3 = __half22float2(qa.h2[3]);
                    const float2 y0 = __half22float2(qb.h2[0]), y1 = __half22float2(qb.h2[1]);
                    const float2 y2 = __half22float2(qb.h2[2]), y3 = __half22float2(qb.h2[3]);
                    a0 = fmaf(x0.x, k0.x, a0); a1 = fmaf(x0.y, k0.y, a1);
                    a2 = fmaf(x1.x, k1.x, a2); a3 = fmaf(x1.y, k1.y, a3);
                    a0 = fmaf(x2.x, k2.x, a0); a1 = fmaf(x2.y, k2.y, a1);
                    a2 = fmaf(x3.x, k3.x, a2); a3 = fmaf(x3.y, k3.y, a3);
                    b0 = fmaf(y0.x, k0.x, b0); b1 = fmaf(y0.y, k0.y, b1);
                    b2 = fmaf(y1.x, k1.x, b2); b3 = fmaf(y1.y, k1.y, b3);
                    b0 = fmaf(y2.x, k2.x, b0); b1 = fmaf(y2.y, k2.y, b1);
                    b2 = fmaf(y3.x, k3.x, b2); b3 = fmaf(y3.y, k3.y, b3);
                }
                dot[0] = (a0 + a1) + (a2 + a3);
                dot[1] = (b0 + b1) + (b2 + b3);
            }
#pragma unroll
            for (int s = 0; s < 2; s++) {
                if ((uint32_t)s >= nq_w) break;
                const float d = dot[s];
                float tile_max = d;
                for (int off = 16; off > 0; off >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
                if (tile_max == -INFINITY) continue;
                const float m_new = fmaxf(m[s], tile_max);
                const float corr = __expf(m[s] - m_new);
                const float w = __expf(d - m_new);
                s_w[(size_t)warp * KT + lane] = w;
                float wsum = w;
                for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
                l[s] = l[s] * corr + wsum;
                m[s] = m_new;
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; j++) acc[s][j] *= corr;
                const float *wrow = s_w + (size_t)warp * KT;
                const uint4 *vcol = s_v8 + lane;
#pragma unroll 4
                for (uint32_t kk = 0; kk < KT; kk++) {
                    const float wk = wrow[kk];
                    union { uint4 u; __half2 h2[4]; } vv;
                    vv.u = vcol[(size_t)kk * C8];
                    const float2 v0 = __half22float2(vv.h2[0]), v1 = __half22float2(vv.h2[1]);
                    const float2 v2 = __half22float2(vv.h2[2]), v3 = __half22float2(vv.h2[3]);
                    acc[s][0] = fmaf(wk, v0.x, acc[s][0]); acc[s][1] = fmaf(wk, v0.y, acc[s][1]);
                    acc[s][2] = fmaf(wk, v1.x, acc[s][2]); acc[s][3] = fmaf(wk, v1.y, acc[s][3]);
                    acc[s][4] = fmaf(wk, v2.x, acc[s][4]); acc[s][5] = fmaf(wk, v2.y, acc[s][5]);
                    acc[s][6] = fmaf(wk, v3.x, acc[s][6]); acc[s][7] = fmaf(wk, v3.y, acc[s][7]);
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const uint32_t gh = warp * 2u + (uint32_t)s;
        if (gh >= G) break;
        const uint64_t qg = (uint64_t)t * n_head + hkv * G + gh;
        float *prow = part + (qg * n_splits + split) * Q4E_ATTN_PART_STRIDE;
        if (lane == 0u) { prow[0] = m[s]; prow[1] = l[s]; }
        float4 *dst = (float4 *)(prow + 8u + 8u * lane);
        dst[0] = make_float4(acc[s][0], acc[s][1], acc[s][2], acc[s][3]);
        dst[1] = make_float4(acc[s][4], acc[s][5], acc[s][6], acc[s][7]);
    }
}

/* Tensor-core attention over a per-row key list (or, with tokens == NULL, the
 * dense causal range 0..pos[t]), replacing the fmaf inner loops of the gather
 * kernel above: at a 2048-token chunk those ran ~37 ms per layer, about 1.4
 * TFLOPS, for what is 51 GFLOP of GEMM.
 *
 * A block owns one row t, one kv head and one slice of that row's key list.
 * The MMA's M dimension is the kv group -- the twelve query heads that share
 * the head's K and V, padded to 16 with zero rows -- so both products are
 * ordinary m16n8k16 fp16 GEMMs with fp32 accumulation: S = Q K^T over 64-key
 * tiles (warp w scores keys 8w..8w+7 of the tile against all 16 rows), and
 * O += P V with warp w owning 32 of the 256 output dims.  K, V and Q tiles
 * sit in shared memory as [row][256] fp16 with the 16-byte chunk index XORed
 * with (row & 7), which keeps both the gather stores and every ldmatrix
 * conflict free; V is read with ldmatrix.trans so the same [key][dim] tile
 * serves as the col-major B operand.  The softmax is the online form per row,
 * done by the warp that owns rows 2w and 2w+1 once the tile's scores are in
 * shared memory; P goes back as fp16 and the per-row rescale factors with it.
 *
 * Numerics: scores and the running max/sum are fp32; P is rounded to fp16
 * before the PV product where the scalar kernel kept it fp32 -- the same
 * rounding a Flash-style kernel makes, ~1e-3 relative on the output, below
 * the fp16 K/V cache's own quantization.  Partials go to `part` in the layout
 * q4e_qsa_attention_combine_kernel merges; with one split the normalized
 * result is written to `out` directly. */
#define Q4E_TC_KT 64u
#define Q4E_TC_HD 256u
#define Q4E_TC_CH (Q4E_TC_HD / 8u)          /* 16-byte chunks per row */
#define Q4E_TC_SMEM ((16u + 2u * Q4E_TC_KT) * Q4E_TC_HD * 2u + 16u * Q4E_TC_KT * 4u + 16u * Q4E_TC_KT * 2u + 16u * 4u)
__device__ __forceinline__ static uint32_t q4e_tc_off(uint32_t row, uint32_t chunk) {
    return (row * Q4E_TC_CH + (chunk ^ (row & 7u))) * 16u;
}
extern "C" __global__ void __launch_bounds__(256)
q4e_qsa_attention_tc_kernel(
        float *out, float *part, const __half *k_cache, const __half *v_cache, const float *q,
        const int32_t *tokens, const int32_t *n_sel, const int32_t *pos, const int32_t *pages,
        uint32_t width, uint32_t n_head, uint32_t n_head_kv, uint32_t n_splits) {
    constexpr uint32_t KT = Q4E_TC_KT, HD = Q4E_TC_HD, CH = Q4E_TC_CH;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t t = blockIdx.z, hkv = blockIdx.y, split = blockIdx.x;
    const uint32_t G = n_head / n_head_kv;
    const float scale = rsqrtf((float)HD);

    extern __shared__ __align__(16) uint8_t q4e_tc_smem[];
    uint8_t *s_q = q4e_tc_smem;                          /* [16][256] f16 */
    uint8_t *s_k = s_q + 16u * HD * 2u;                  /* [64][256] f16 */
    uint8_t *s_v = s_k + KT * HD * 2u;                   /* [64][256] f16 */
    float   *s_s = (float *)(s_v + KT * HD * 2u);        /* [16][64] f32 scores */
    __half  *s_p = (__half *)(s_s + 16u * KT);           /* [16][64] f16, chunk-swizzled */
    float   *s_corr = (float *)(s_p + 16u * KT);         /* [16] */
    __shared__ int32_t s_idx[Q4E_TC_KT];
    __shared__ float s_m[16], s_l[16];

    const int32_t *list = tokens ? tokens + (uint64_t)t * width : NULL;
    const uint32_t n_list = tokens ? (uint32_t)n_sel[t] : (uint32_t)pos[t] + 1u;
    const uint32_t n_tiles = (n_list + KT - 1u) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t l_begin = split * tiles_per_split * KT;
    uint32_t l_end = l_begin + tiles_per_split * KT;
    if (l_end > n_list) l_end = n_list;

    /* Q: the group's heads as rows, pre-scaled, rows G..15 zero. */
    for (uint32_t i = tid; i < 16u * CH; i += 256u) {
        const uint32_t r = i / CH, c = i % CH;
        union { uint4 u; __half2 h2[4]; } pk;
        if (r < G) {
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + hkv * G + r) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
        } else {
            pk.u = make_uint4(0u, 0u, 0u, 0u);
        }
        *(uint4 *)(s_q + q4e_tc_off(r, c)) = pk.u;
    }
    float m_r[2] = { -INFINITY, -INFINITY }, l_r[2] = { 0.0f, 0.0f };
    float o[4][4];
#pragma unroll
    for (int f = 0; f < 4; f++) { o[f][0] = o[f][1] = o[f][2] = o[f][3] = 0.0f; }

    /* The next tile's K and V rows are fetched into registers (eight 16-byte
     * pieces of each per thread) before the current tile is computed, so the
     * gather's latency overlaps the MMAs instead of serializing with them; a
     * block holds 78 KB of shared memory, so it is alone on its SM and has
     * nothing else to hide behind.  Keys past the list are zero so their
     * (masked, zero-weight) columns stay finite. */
    constexpr uint32_t PF = KT * CH / 256u;
    uint4 pk[PF], pv[PF];
    auto fetch = [&](uint32_t p0) {
        const uint32_t n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
        if (tid < KT) {
            s_idx[tid] = tid < n_k
                ? (int32_t)q4e_kv_row(pages, tokens ? (uint32_t)list[p0 + tid] : p0 + tid) : -1;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t j = 0; j < PF; j++) {
            const uint32_t i = tid + 256u * j;
            const uint32_t key = i / CH, c = i % CH;
            const int32_t row = s_idx[key];
            if (row >= 0) {
                const uint64_t base = ((uint64_t)row * n_head_kv + hkv) * HD + 8u * c;
                pk[j] = *(const uint4 *)(k_cache + base);
                pv[j] = *(const uint4 *)(v_cache + base);
            } else {
                pk[j] = make_uint4(0u, 0u, 0u, 0u);
                pv[j] = pk[j];
            }
        }
    };
    if (l_begin < l_end) fetch(l_begin);
    __syncthreads();

    for (uint32_t p0 = l_begin; p0 < l_end; p0 += KT) {
        const uint32_t n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
#pragma unroll
        for (uint32_t j = 0; j < PF; j++) {
            const uint32_t i = tid + 256u * j;
            const uint32_t key = i / CH, c = i % CH;
            *(uint4 *)(s_k + q4e_tc_off(key, c)) = pk[j];
            *(uint4 *)(s_v + q4e_tc_off(key, c)) = pv[j];
        }
        __syncthreads();
        if (p0 + KT < l_end) fetch(p0 + KT);

        /* S[16][8w..8w+8) = Q K^T over the 256 dims. */
        float sacc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 4
        for (uint32_t ks = 0; ks < HD / 16u; ks++) {
            uint32_t a[4], b[2];
            tt_ldmatrix_x4_addr(a, tt_smem_addr(s_q + q4e_tc_off(lane & 15u, ks * 2u + (lane >> 4u))));
            tt_ldmatrix_x2_addr(b, tt_smem_addr(s_k + q4e_tc_off(warp * 8u + (lane & 7u),
                                                                 ks * 2u + ((lane >> 3u) & 1u))));
            tt_mma_m16n8k16_f16_f32(sacc, a, b);
        }
        {
            const uint32_t r0 = lane >> 2u, c0 = warp * 8u + 2u * (lane & 3u);
            s_s[r0 * KT + c0] = sacc[0];
            s_s[r0 * KT + c0 + 1u] = sacc[1];
            s_s[(r0 + 8u) * KT + c0] = sacc[2];
            s_s[(r0 + 8u) * KT + c0 + 1u] = sacc[3];
        }
        __syncthreads();

        /* Online softmax for rows 2w and 2w+1: lane covers keys lane, lane + 32. */
#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const uint32_t r = warp * 2u + (uint32_t)rr;
            const float v0 = lane < n_k ? s_s[r * KT + lane] : -INFINITY;
            const float v1 = lane + 32u < n_k ? s_s[r * KT + lane + 32u] : -INFINITY;
            float mx = fmaxf(v0, v1);
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            float corr = 1.0f, p0f = 0.0f, p1f = 0.0f;
            if (mx != -INFINITY) {
                const float m_new = fmaxf(m_r[rr], mx);
                corr = __expf(m_r[rr] - m_new);
                p0f = v0 == -INFINITY ? 0.0f : __expf(v0 - m_new);
                p1f = v1 == -INFINITY ? 0.0f : __expf(v1 - m_new);
                float sum = p0f + p1f;
                for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
                l_r[rr] = l_r[rr] * corr + sum;
                m_r[rr] = m_new;
            }
            s_p[r * KT + ((lane >> 3u) ^ (r & 7u)) * 8u + (lane & 7u)] = __float2half_rn(p0f);
            s_p[r * KT + (((lane + 32u) >> 3u) ^ (r & 7u)) * 8u + (lane & 7u)] = __float2half_rn(p1f);
            if (lane == 0u) s_corr[r] = corr;
        }
        __syncthreads();

        /* O[16][32w..32w+32) = corr * O + P V. */
        {
            const float c_lo = s_corr[lane >> 2u], c_hi = s_corr[(lane >> 2u) + 8u];
#pragma unroll
            for (int f = 0; f < 4; f++) {
                o[f][0] *= c_lo; o[f][1] *= c_lo; o[f][2] *= c_hi; o[f][3] *= c_hi;
            }
        }
#pragma unroll
        for (uint32_t ks = 0; ks < KT / 16u; ks++) {
            uint32_t a[4];
            {
                const uint32_t r = lane & 15u, c = ks * 2u + (lane >> 4u);
                tt_ldmatrix_x4_addr(a, tt_smem_addr(s_p + r * KT + (c ^ (r & 7u)) * 8u));
            }
            const uint32_t key = ks * 16u + (lane & 7u) + 8u * ((lane >> 3u) & 1u);
#pragma unroll
            for (uint32_t f = 0; f < 4; f++) {
                uint32_t b[2];
                tt_ldmatrix_x2_trans_addr(b, tt_smem_addr(s_v + q4e_tc_off(key, warp * 4u + f)));
                tt_mma_m16n8k16_f16_f32(o[f], a, b);
            }
        }
        __syncthreads();
    }

    if (lane == 0u) {
        s_m[warp * 2u] = m_r[0]; s_m[warp * 2u + 1u] = m_r[1];
        s_l[warp * 2u] = l_r[0]; s_l[warp * 2u + 1u] = l_r[1];
    }
    __syncthreads();
#pragma unroll
    for (int half_ = 0; half_ < 2; half_++) {
        const uint32_t r = (lane >> 2u) + 8u * (uint32_t)half_;
        if (r >= G) continue;
        const uint64_t qg = (uint64_t)t * n_head + hkv * G + r;
        const uint32_t d0 = warp * 32u + 2u * (lane & 3u);
        if (part) {
            float *prow = part + (qg * n_splits + split) * Q4E_ATTN_PART_STRIDE;
            if (warp == 0u && (lane & 3u) == 0u) { prow[0] = s_m[r]; prow[1] = s_l[r]; }
#pragma unroll
            for (uint32_t f = 0; f < 4; f++) {
                prow[8u + d0 + 8u * f] = o[f][2 * half_];
                prow[8u + d0 + 8u * f + 1u] = o[f][2 * half_ + 1];
            }
        } else {
            const float inv = s_l[r] > 0.0f ? 1.0f / s_l[r] : 0.0f;
            float *orow = out + qg * HD;
#pragma unroll
            for (uint32_t f = 0; f < 4; f++) {
                orow[d0 + 8u * f] = o[f][2 * half_] * inv;
                orow[d0 + 8u * f + 1u] = o[f][2 * half_ + 1] * inv;
            }
        }
    }
}
