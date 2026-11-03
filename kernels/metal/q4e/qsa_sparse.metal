// q4e qsa_sparse entries: Metal port of kernels/cuda/q4e/qsa_sparse.cu (see q4e_common.h for conventions).
//
// The split-key attention (decode and the verify batch), the combine that
// folds its partial slabs, and the gathered attention over the indexer's
// selected positions. All three are specialised to head_dim 256.
//
// THREADGROUP MEMORY, RESTRUCTURED. The CUDA split and gather kernels stage a
// key tile [32 chunks][32 keys], a value tile [32 keys][32 chunks], the
// group's queries and a weight row side by side: 41984 bytes (the engine's
// `qsa_sparse_smem()`), past this device's 32768. Here the key and value
// tiles share ONE 16 KiB buffer: a tile's keys are staged, every lane's two
// dot products are taken into registers, a barrier retires the keys, and only
// then are the values staged over them. The CUDA body also computes both dot
// products before it touches the values, so every score, weight and
// accumulator is the same arithmetic in the same order; only the staging
// schedule moved. Footprint: 16384 (K/V) + 8192 (queries) + 1024 (weights) =
// 25600 bytes, declared statically (the gather adds 128 bytes for its row
// indices). The engine still requests CUDA's 41984; because neither kernel
// declares a `[[threadgroup(0)]]` binding, the backend does not apply it.
//
// Not ported: q4e_qsa_attention_tc_kernel (tensor-core MMA).

#define Q4E_ATTN_KT       32u               // keys per tile: one per lane
#define Q4E_ATTN_QG       16u               // queries per block: 2 per simdgroup
#define Q4E_ATTN_SPLIT_HD 256u
#define Q4E_ATTN_PART_STRIDE (Q4E_ATTN_SPLIT_HD + 8u)   // m, l, pad, acc[256]

// 16-byte f16 chunks per 256-wide row.
constant constexpr uint Q4E_QSA_C8 = Q4E_ATTN_SPLIT_HD / 8u;

// The split and gather kernels' staging, in 16-byte words: the shared K/V tile
// (C8 x KT), the query block (QG x C8) and one weight row per simdgroup of the
// 256-thread block (8 x KT floats) - 25600 bytes. Declared as a static array in
// each kernel rather than bound at `[[threadgroup(0)]]`, so the backend knows
// the CUDA launcher's larger dynamic request (41984 bytes) describes nothing
// here and does not apply it.
constant constexpr uint Q4E_QSA_SPARSE_SMEM_U4 =
    Q4E_QSA_C8 * Q4E_ATTN_KT + Q4E_ATTN_QG * Q4E_QSA_C8 + (8u * Q4E_ATTN_KT * 4u) / 16u;

/// Whether `x` is exactly negative infinity, by bit pattern. Fast math is the
/// library default and may fold a floating compare against an infinity; the
/// CUDA `== -INFINITY` tests are control flow (a fully masked tile, an empty
/// split), so they must not be optimized away.
static inline bool q4e_qsa_is_neg_inf(float x) {
    return as_type<uint>(x) == 0xff800000u;
}

/// Two f16 lanes of a staged chunk, widened.
static inline float2 q4e_qsa_h2(uint u) {
    return float2(as_type<half2>(u));
}

/// Two floats rounded to nearest-even halves and packed low-first, as
/// `__floats2half2_rn` does.
static inline uint q4e_qsa_pack(float a, float b) {
    return as_type<uint>(half2(float2(a, b)));
}

/// Eight scaled query dims `src[0..8)` as one staged chunk.
static inline uint4 q4e_qsa_stage_q(device const float *src, float scale) {
    const float4 v0 = *((device const float4 *)src);
    const float4 v1 = *((device const float4 *)(src + 4));
    return uint4(q4e_qsa_pack(v0.x * scale, v0.y * scale),
                 q4e_qsa_pack(v0.z * scale, v0.w * scale),
                 q4e_qsa_pack(v1.x * scale, v1.y * scale),
                 q4e_qsa_pack(v1.z * scale, v1.w * scale));
}

/// Key `lane` of the staged chunk-major key tile against two staged queries:
/// four partial sums each, folded `(a0 + a1) + (a2 + a3)`, exactly the CUDA
/// score loop.
static inline void q4e_qsa_tile_dots(threadgroup const uint4 *s_k8,
                                     threadgroup const uint4 *qrow0,
                                     threadgroup const uint4 *qrow1,
                                     uint lane, thread float &dot0, thread float &dot1) {
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
    for (uint c = 0; c < Q4E_QSA_C8; c++) {
        const uint4 kk = s_k8[c * Q4E_ATTN_KT + lane];
        const uint4 qa = qrow0[c];
        const uint4 qb = qrow1[c];
        const float2 k0 = q4e_qsa_h2(kk.x), k1 = q4e_qsa_h2(kk.y);
        const float2 k2 = q4e_qsa_h2(kk.z), k3 = q4e_qsa_h2(kk.w);
        const float2 x0 = q4e_qsa_h2(qa.x), x1 = q4e_qsa_h2(qa.y);
        const float2 x2 = q4e_qsa_h2(qa.z), x3 = q4e_qsa_h2(qa.w);
        const float2 y0 = q4e_qsa_h2(qb.x), y1 = q4e_qsa_h2(qb.y);
        const float2 y2 = q4e_qsa_h2(qb.z), y3 = q4e_qsa_h2(qb.w);
        a0 = fma(x0.x, k0.x, a0); a1 = fma(x0.y, k0.y, a1);
        a2 = fma(x1.x, k1.x, a2); a3 = fma(x1.y, k1.y, a3);
        a0 = fma(x2.x, k2.x, a0); a1 = fma(x2.y, k2.y, a1);
        a2 = fma(x3.x, k3.x, a2); a3 = fma(x3.y, k3.y, a3);
        b0 = fma(y0.x, k0.x, b0); b1 = fma(y0.y, k0.y, b1);
        b2 = fma(y1.x, k1.x, b2); b3 = fma(y1.y, k1.y, b3);
        b0 = fma(y2.x, k2.x, b0); b1 = fma(y2.y, k2.y, b1);
        b2 = fma(y3.x, k3.x, b2); b3 = fma(y3.y, k3.y, b3);
    }
    dot0 = (a0 + a1) + (a2 + a3);
    dot1 = (b0 + b1) + (b2 + b3);
}

/// One query's online-softmax step over a staged tile: `d` is this lane's
/// score (-inf where masked), `wrow` the simdgroup's weight row, `s_v8` the
/// key-major value tile (zero past the tile's keys). Lane owns dims
/// `8 * lane .. 8 * lane + 7`. A fully masked tile changes nothing.
static inline void q4e_qsa_online(float d, thread float &m, thread float &l,
                                  thread float *acc, threadgroup float *wrow,
                                  threadgroup const uint4 *s_v8, uint lane) {
    const float tile_max = q4e_warp_allmax(d);
    if (q4e_qsa_is_neg_inf(tile_max)) {
        return;
    }
    const float m_new = fmax(m, tile_max);
    const float corr = exp(m - m_new);
    const float w = exp(d - m_new);            // 0 where masked
    wrow[lane] = w;
    const float wsum = q4e_warp_allsum(w);
    l = l * corr + wsum;
    m = m_new;
    simdgroup_barrier(mem_flags::mem_threadgroup);

    for (uint j = 0; j < 8u; j++) {
        acc[j] *= corr;
    }
    threadgroup const uint4 *vcol = s_v8 + lane;
    for (uint k = 0; k < Q4E_ATTN_KT; k++) {
        const float wk = wrow[k];
        const uint4 vv = vcol[k * Q4E_QSA_C8];
        const float2 v0 = q4e_qsa_h2(vv.x), v1 = q4e_qsa_h2(vv.y);
        const float2 v2 = q4e_qsa_h2(vv.z), v3 = q4e_qsa_h2(vv.w);
        acc[0] = fma(wk, v0.x, acc[0]);
        acc[1] = fma(wk, v0.y, acc[1]);
        acc[2] = fma(wk, v1.x, acc[2]);
        acc[3] = fma(wk, v1.y, acc[3]);
        acc[4] = fma(wk, v2.x, acc[4]);
        acc[5] = fma(wk, v2.y, acc[5]);
        acc[6] = fma(wk, v3.x, acc[6]);
        acc[7] = fma(wk, v3.y, acc[7]);
    }
    // The next query's step overwrites `wrow`.
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

/// A query's result: normalized into `out` at one split, a partial slab
/// `(m, l, pad, acc[256])` for the combine otherwise.
static inline void q4e_qsa_emit(device float *out, device float *part, ulong qg,
                                uint n_splits, uint split, float m, float l,
                                thread const float *acc, uint lane, bool direct) {
    if (direct) {
        const float inv = 1.0f / l;
        device float4 *dst = (device float4 *)(out + qg * Q4E_ATTN_SPLIT_HD + 8u * lane);
        dst[0] = float4(acc[0] * inv, acc[1] * inv, acc[2] * inv, acc[3] * inv);
        dst[1] = float4(acc[4] * inv, acc[5] * inv, acc[6] * inv, acc[7] * inv);
    } else {
        device float *prow = part + (qg * n_splits + split) * Q4E_ATTN_PART_STRIDE;
        if (lane == 0u) {
            prow[0] = m;
            prow[1] = l;
        }
        device float4 *dst = (device float4 *)(prow + 8u + 8u * lane);
        dst[0] = float4(acc[0], acc[1], acc[2], acc[3]);
        dst[1] = float4(acc[4], acc[5], acc[6], acc[7]);
    }
}

/// `qsa_sparse.cu:53` `q4e_qsa_attention_split_kernel`: split-key causal GQA.
///
/// Grid (n_tok, n_splits, n_head_kv), 256 threads, 25600 bytes of dynamic
/// memory (see the file comment; CUDA asks 41984). Threadgroup x is row x
/// with all G = n_head / n_head_kv of its query heads (two per simdgroup,
/// G <= 16), cutting that row's own key range `0..=pos[x]` into `n_splits`
/// runs of whole 32-key tiles; a split past the end writes the empty partial
/// (-inf, 0, 0). Buffers: 0 out [t][h][256], 1 part
/// [t*n_head + h][n_splits][264], 2 k_cache, 3 v_cache (f16), 4 q
/// [t][h][256], 5 n_head, 6 n_head_kv, 7 pos [t], 8 pages, 9 n_tok,
/// 10 n_splits. With `n_splits == 1` it writes `out` itself.
///
/// The queries are staged as f16 after scaling by 1/16, as in CUDA; the
/// accumulation is f32.
kernel void q4e_qsa_attention_split_kernel(
        device float *out              [[buffer(0)]],
        device float *part             [[buffer(1)]],
        device const half *k_cache     [[buffer(2)]],
        device const half *v_cache     [[buffer(3)]],
        device const float *q          [[buffer(4)]],
        constant uint &n_head          [[buffer(5)]],
        constant uint &n_head_kv       [[buffer(6)]],
        device const int *pos          [[buffer(7)]],
        device const int *pages        [[buffer(8)]],
        constant uint &n_tok           [[buffer(9)]],
        constant uint &n_splits        [[buffer(10)]],
        uint3 bid                      [[threadgroup_position_in_grid]],
        uint3 tid3                     [[thread_position_in_threadgroup]],
        uint3 ntg                      [[threads_per_threadgroup]]) {
    // Static rather than dynamic: see Q4E_QSA_SPARSE_SMEM_U4.
    threadgroup uint4 smem[Q4E_QSA_SPARSE_SMEM_U4];
    constexpr uint HD = Q4E_ATTN_SPLIT_HD;
    constexpr uint KT = Q4E_ATTN_KT;
    constexpr uint C8 = Q4E_QSA_C8;
    const uint tid = tid3.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    const uint hkv = bid.z;
    const uint split = bid.y;
    const uint G = n_head / n_head_kv;
    const uint NQ = G * n_tok;
    const uint q0 = bid.x * G;
    if (q0 >= NQ || G > Q4E_ATTN_QG) {
        return;
    }
    const uint n_q = G;
    const float scale = rsqrt(float(HD));

    threadgroup uint4 *s_kv8 = smem;                               // K [C8][KT], then V [KT][C8]
    threadgroup uint4 *s_q8 = s_kv8 + C8 * KT;                     // [QG][C8]
    threadgroup float *s_w = (threadgroup float *)(s_q8 + Q4E_ATTN_QG * C8);   // [warps][KT]

    const uint last = uint(pos[bid.x]);
    const uint n_tiles = (last + KT) / KT;
    const uint tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint k_begin = split * tiles_per_split * KT;
    uint k_end = k_begin + tiles_per_split * KT;
    if (k_end > last + 1u) {
        k_end = last + 1u;
    }

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
    for (uint s = 0; s < 2u; s++) {
        for (uint j = 0; j < 8u; j++) {
            acc[s][j] = 0.0f;
        }
    }

    if (k_begin < k_end) {
        for (uint idx = tid; idx < n_q * C8; idx += ntg.x) {
            const uint qi = idx / C8, c = idx % C8;
            const uint t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
            s_q8[qi * C8 + c] = q4e_qsa_stage_q(q + (ulong(t) * n_head + h) * HD + 8u * c, scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint qi0 = warp * 2u;
        const uint nq_w = (qi0 >= n_q) ? 0u : ((n_q - qi0 >= 2u) ? 2u : 1u);
        for (uint p0 = k_begin; p0 < k_end; p0 += KT) {
            const uint n_k = (k_end - p0 < KT) ? (k_end - p0) : KT;
            // k_begin and the stride are multiples of KT, so the tile lies in
            // one page and one lookup covers it.
            const uint krow = q4e_kv_row(pages, p0);
            for (uint i = 0; i < 4u; i++) {
                const uint key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_kv8[c * KT + key] = *((device const uint4 *)(k_cache +
                            ((ulong(krow + key) * n_head_kv + hkv) * HD + 8u * c)));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dot[2] = { -INFINITY, -INFINITY };
            if (nq_w != 0u && lane < n_k) {
                q4e_qsa_tile_dots(s_kv8, s_q8 + qi0 * C8, s_q8 + (qi0 + (nq_w - 1u)) * C8,
                                  lane, dot[0], dot[1]);
            }
            // Every score read of the keys is done before the values land on
            // top of them.
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 4u; i++) {
                const uint key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_kv8[key * C8 + c] = *((device const uint4 *)(v_cache +
                            ((ulong(krow + key) * n_head_kv + hkv) * HD + 8u * c)));
                } else {
                    // The accumulation runs the whole tile with zero weights
                    // on the tail, and 0 * NaN from stale memory is NaN.
                    s_kv8[key * C8 + c] = uint4(0u);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = 0; s < nq_w; s++) {
                const uint qi = qi0 + s;
                const uint qpos = uint(pos[(q0 + qi) / G]);
                const float d = (lane < n_k && p0 + lane <= qpos) ? dot[s] : -INFINITY;
                q4e_qsa_online(d, m[s], l[s], acc[s], s_w + warp * KT, s_kv8, lane);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint s = 0; s < 2u; s++) {
        const uint qi = warp * 2u + s;
        if (qi >= n_q) {
            break;
        }
        const uint t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
        const ulong qg = ulong(t) * n_head + h;
        q4e_qsa_emit(out, part, qg, n_splits, split, m[s], l[s], acc[s], lane, n_splits == 1u);
    }
}

/// `qsa_sparse.cu:259` `q4e_qsa_attention_combine_kernel`: folds the split
/// partials. One threadgroup per (row, head), thread d merges dim d.
///
/// Grid n_tok * n_head, 256 threads, no threadgroup memory. Buffers: 0 out
/// [t*n_head + h][256], 1 part, 2 n_splits. An empty partial (m = -inf) is
/// skipped.
kernel void q4e_qsa_attention_combine_kernel(
        device float *out              [[buffer(0)]],
        device const float *part       [[buffer(1)]],
        constant uint &n_splits        [[buffer(2)]],
        uint bid                       [[threadgroup_position_in_grid]],
        uint tid                       [[thread_position_in_threadgroup]]) {
    constexpr uint HD = Q4E_ATTN_SPLIT_HD;
    const ulong qg = bid;
    device const float *base = part + qg * n_splits * Q4E_ATTN_PART_STRIDE;
    const uint d = tid;
    float gm = -INFINITY;
    for (uint sp = 0; sp < n_splits; sp++) {
        gm = fmax(gm, base[ulong(sp) * Q4E_ATTN_PART_STRIDE]);
    }
    float den = 0.0f, num = 0.0f;
    for (uint sp = 0; sp < n_splits; sp++) {
        device const float *prow = base + ulong(sp) * Q4E_ATTN_PART_STRIDE;
        const float ms = prow[0];
        if (q4e_qsa_is_neg_inf(ms)) {
            continue;
        }
        const float wgt = exp(ms - gm);
        den = fma(prow[1], wgt, den);
        num = fma(prow[8u + d], wgt, num);
    }
    out[qg * HD + d] = num / den;
}

// Slices of a row's token list the gathered attention runs.
#define Q4E_ATTN_GSPLITS 8u

/// `qsa_sparse.cu:284` `q4e_qsa_attention_gather_kernel`: attention over a
/// per-row list of positions (the indexer's selection), no causal mask.
///
/// Grid (n_splits, n_head_kv, n_tok), 256 threads, 25600 bytes of dynamic
/// memory (CUDA asks 41984; see the file comment) plus 128 static. A
/// threadgroup owns row t, one kv head, all G of its query heads, and one of
/// `n_splits` slices (whole 32-entry tiles) of `tokens[t][0 .. n_sel[t])`.
/// Writes only partial slabs; the combine writes the destination. Buffers:
/// 0 part, 1 k_cache, 2 v_cache (f16), 3 q [t][h][256], 4 tokens
/// [t][width], 5 n_sel [t], 6 pages, 7 width, 8 n_head, 9 n_head_kv,
/// 10 n_splits.
///
/// Each listed position is translated through the page table once, while the
/// tile's list entries are staged.
kernel void q4e_qsa_attention_gather_kernel(
        device float *part             [[buffer(0)]],
        device const half *k_cache     [[buffer(1)]],
        device const half *v_cache     [[buffer(2)]],
        device const float *q          [[buffer(3)]],
        device const int *tokens       [[buffer(4)]],
        device const int *n_sel        [[buffer(5)]],
        device const int *pages        [[buffer(6)]],
        constant uint &width           [[buffer(7)]],
        constant uint &n_head          [[buffer(8)]],
        constant uint &n_head_kv       [[buffer(9)]],
        constant uint &n_splits        [[buffer(10)]],
        uint3 bid                      [[threadgroup_position_in_grid]],
        uint3 tid3                     [[thread_position_in_threadgroup]],
        uint3 ntg                      [[threads_per_threadgroup]]) {
    // Static rather than dynamic: see Q4E_QSA_SPARSE_SMEM_U4.
    threadgroup uint4 smem[Q4E_QSA_SPARSE_SMEM_U4];
    constexpr uint HD = Q4E_ATTN_SPLIT_HD;
    constexpr uint KT = Q4E_ATTN_KT;
    constexpr uint C8 = Q4E_QSA_C8;
    threadgroup int s_idx[Q4E_ATTN_KT];
    const uint tid = tid3.x, lane = tid & 31u, warp = tid >> 5u;
    const uint t = bid.z, hkv = bid.y, split = bid.x;
    const uint G = n_head / n_head_kv;
    const float scale = rsqrt(float(HD));
    device const int *list = tokens + ulong(t) * width;
    const uint n_list = uint(n_sel[t]);
    const uint n_tiles = (n_list + KT - 1u) / KT;
    const uint tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint l_begin = split * tiles_per_split * KT;
    uint l_end = l_begin + tiles_per_split * KT;
    if (l_end > n_list) {
        l_end = n_list;
    }

    threadgroup uint4 *s_kv8 = smem;                               // K [C8][KT], then V [KT][C8]
    threadgroup uint4 *s_q8 = s_kv8 + C8 * KT;                     // [QG][C8]
    threadgroup float *s_w = (threadgroup float *)(s_q8 + Q4E_ATTN_QG * C8);   // [warps][KT]

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
    for (uint s = 0; s < 2u; s++) {
        for (uint j = 0; j < 8u; j++) {
            acc[s][j] = 0.0f;
        }
    }

    if (l_begin < l_end) {
        for (uint idx = tid; idx < G * C8; idx += ntg.x) {
            const uint gh = idx / C8, c = idx % C8;
            s_q8[gh * C8 + c] =
                q4e_qsa_stage_q(q + (ulong(t) * n_head + hkv * G + gh) * HD + 8u * c, scale);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint qi0 = warp * 2u;
        const uint nq_w = (qi0 >= G) ? 0u : ((G - qi0 >= 2u) ? 2u : 1u);
        for (uint p0 = l_begin; p0 < l_end; p0 += KT) {
            const uint n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
            if (tid < KT) {
                s_idx[tid] = tid < n_k ? int(q4e_kv_row(pages, uint(list[p0 + tid]))) : 0;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 4u; i++) {
                const uint key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_kv8[c * KT + key] = *((device const uint4 *)(k_cache +
                            ((ulong(s_idx[key]) * n_head_kv + hkv) * HD + 8u * c)));
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float dot[2] = { -INFINITY, -INFINITY };
            if (nq_w != 0u && lane < n_k) {
                q4e_qsa_tile_dots(s_kv8, s_q8 + qi0 * C8, s_q8 + (qi0 + (nq_w - 1u)) * C8,
                                  lane, dot[0], dot[1]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint i = 0; i < 4u; i++) {
                const uint key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_kv8[key * C8 + c] = *((device const uint4 *)(v_cache +
                            ((ulong(s_idx[key]) * n_head_kv + hkv) * HD + 8u * c)));
                } else {
                    s_kv8[key * C8 + c] = uint4(0u);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (uint s = 0; s < nq_w; s++) {
                q4e_qsa_online(dot[s], m[s], l[s], acc[s], s_w + warp * KT, s_kv8, lane);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    for (uint s = 0; s < 2u; s++) {
        const uint gh = warp * 2u + s;
        if (gh >= G) {
            break;
        }
        const ulong qg = ulong(t) * n_head + hkv * G + gh;
        q4e_qsa_emit(part, part, qg, n_splits, split, m[s], l[s], acc[s], lane, false);
    }
}
