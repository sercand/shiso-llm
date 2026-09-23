/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels);
 * D2 (shiso): q4e_idx_topk_kernel's compaction writes the selected blocks in
 * index order through a block-wide scan, and admits threshold ties lowest
 * index first, where the source appended them by atomic arrival.  The set was
 * the same up to exact ties; the order was not, and the attention sums the
 * selected tokens in that order - so the same prompt decoded to different
 * text from one run to the next once its context passed the budget.
 * No other line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * QSA indexer (Qwen Sparse Attention).  Per the Qwen3.8-Flash-Next report
 * §2.1.2: an MQA indexer with H=4 query heads of 128 dims and one key head;
 * keys are mean-pooled over blocks of r=4 tokens, RMS-normed and partially
 * rotated (64 dims, block start position); queries are normed and rotated at
 * their own position; a block's score is sum_h relu(<q_h, k_b>) over blocks
 * fully observed by the query; the top ceil(2048/4)=512 blocks plus the tail
 * tokens of the incomplete block are what the core attention attends to.
 * Below 512 complete blocks the selection is every visible token, so the
 * dense kernels above are used there (and stay bit-identical).
 * ------------------------------------------------------------------------ */
#define Q4E_IDX_DIM   128u
#define Q4E_IDX_R     4u
#define Q4E_IDX_TAIL  (Q4E_IDX_R - 1u)

/* Raw indexer keys into the f16 cache at the rows' paged positions. */
extern "C" __global__ void q4e_idx_store_k_kernel(__half *cache, const float *k, const int32_t *pos,
                                              const int32_t *pages, uint32_t n_tok) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;
    const uint64_t dst = (uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * Q4E_IDX_DIM;
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        cache[dst + i] = __float2half(k[(uint64_t)t * Q4E_IDX_DIM + i]);
    }
}

/* Batched idx key store: one launch over all rows, each row's own page table.
 * grid.x = row (block per token, as the scalar kernel), pages from the packed
 * array.  Numerically identical to the per-row scalar path (a plain store). */
extern "C" __global__ void q4e_idx_store_k_batch_kernel(
        __half *cache, const float *k, const int32_t *pos,
        const int32_t *const *pages_arr, uint32_t n_tok) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;
    const int32_t *pages = pages_arr[t];
    const uint64_t dst = (uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * Q4E_IDX_DIM;
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        cache[dst + i] = __float2half(k[(uint64_t)t * Q4E_IDX_DIM + i]);
    }
}

/* One block per key block b in [b0, b1): mean of its r raw keys, RMSNorm with
 * the (Gemma +1 baked) k-norm weight, RoPE at position r*b, stored f16.
 *
 * A page holds a whole number of key blocks, so block b's r raw keys are
 * contiguous inside one page and its pooled row sits in that page's pooled
 * slice: both addresses come from the one translation of position r*b. */
static_assert(SHS_Q4E_PAGE_TOKENS % Q4E_IDX_R == 0u,
              "a pooled key block must not straddle a KV page");
extern "C" __global__ void q4e_idx_pool_kernel(__half *pooled, const __half *cache, const float *w,
                                           const int32_t *pos, const int32_t *pages, uint32_t n_tok,
                                           uint32_t n_rot, float rope_base,
                                           const int32_t *mrope_all,
                                           uint32_t sec_t, uint32_t sec_h, uint32_t sec_w,
                                           float eps) {
    /* Blocks completed by this forward: [pos0/r, (pos_last+1)/r).  Derived on
     * the device so a captured graph stays right as the context grows. */
    const uint32_t b0 = (uint32_t)pos[0] / Q4E_IDX_R;
    const uint32_t b1 = ((uint32_t)pos[n_tok - 1u] + 1u) / Q4E_IDX_R;
    const uint32_t b = b0 + blockIdx.x;
    if (b >= b1) return;
    const uint32_t krow = q4e_kv_row(pages, b * Q4E_IDX_R);
    const uint32_t brow = krow / Q4E_IDX_R;          /* pooled row of block b */
    __shared__ float s_k[Q4E_IDX_DIM];
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t j = 0; j < Q4E_IDX_R; j++) {
            acc += __half2float(cache[((uint64_t)krow + j) * Q4E_IDX_DIM + i]);
        }
        acc *= 1.0f / (float)Q4E_IDX_R;
        s_k[i] = acc;
        sum += acc * acc;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)Q4E_IDX_DIM + eps);
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) s_k[i] *= scale * w[i];
    __syncthreads();
    /* This block's start position, unlike every other rope site, is derived
     * arithmetically rather than read out of pos[], and b0 rounds pos[0] down --
     * so the start can sit up to Q4E_IDX_R-1 slots BEFORE this chunk's first
     * token.  That is why mrope_all carries SHS_Q4E_MROPE_BACK rows in front of
     * token 0, and why the row index below can reach back into them.  Leaving
     * this kernel on scalar rope while the query went mrope would make the
     * indexer's inner product inconsistent, and only past the 2048-token budget
     * where the indexer starts being consulted at all. */
    const int32_t block_pos = (int32_t)(b * Q4E_IDX_R);
    const uint32_t mrow = (uint32_t)((int32_t)SHS_Q4E_MROPE_BACK + block_pos - pos[0]);
    q4e_rope_mrope(s_k, n_rot, rope_base,
                   mrope_all[3u * mrow], mrope_all[3u * mrow + 1u], mrope_all[3u * mrow + 2u],
                   sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        pooled[(uint64_t)brow * Q4E_IDX_DIM + i] = __float2half(s_k[i]);
    }
}

/* Per (row, head): RMSNorm + RoPE of the indexer query, stored f16. */
extern "C" __global__ void q4e_idx_q_kernel(__half *qn, const float *q, const float *w,
                                        const int32_t *mrope, uint32_t sec_t,
                                        uint32_t sec_h, uint32_t sec_w, uint32_t n_head,
                                        uint32_t n_rot, float rope_base, float eps) {
    const uint32_t t = blockIdx.y, h = blockIdx.x;
    const uint64_t base = ((uint64_t)t * n_head + h) * Q4E_IDX_DIM;
    __shared__ float s_q[Q4E_IDX_DIM];
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) {
        const float v = q[base + i];
        s_q[i] = v;
        sum += v * v;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)Q4E_IDX_DIM + eps);
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) s_q[i] *= scale * w[i];
    __syncthreads();
    q4e_rope_mrope(s_q, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < Q4E_IDX_DIM; i += blockDim.x) qn[base + i] = __float2half(s_q[i]);
}

/* Block scores: score[t][b] = sum_h relu(<q_th, k_b>) for blocks fully seen
 * by row t (r*b + r - 1 <= pos_t), else -inf.  A thread block handles 16
 * rows x 64 key blocks with both operands staged as f16. */
#define Q4E_IDX_QT 16u
#define Q4E_IDX_BT 64u
/* A key-block tile covers Q4E_IDX_BT * Q4E_IDX_R positions (today exactly one
 * page) and tiles start at multiples of that, so a tile's pooled rows sit in
 * one page, contiguous behind one page-table lookup. */
static_assert(SHS_Q4E_PAGE_TOKENS % (Q4E_IDX_BT * Q4E_IDX_R) == 0u,
              "a key-block tile must not straddle a KV page");
extern "C" __global__ void __launch_bounds__(256) q4e_idx_score_kernel(
        float *score, const __half *qn, const __half *pooled, const int32_t *pos,
        const int32_t *pages, uint32_t n_head, uint32_t n_tok,
        uint32_t n_blocks_stride, const int32_t *pos_last) {
    /* Complete blocks so far, from the last row's position (device-derived
     * so captured graphs track the growing context); the row stride of
     * `score` is the fixed n_blocks_stride. */
    const uint32_t n_blocks = ((uint32_t)pos_last[0] + 1u) / Q4E_IDX_R;
    const uint32_t t0 = blockIdx.y * Q4E_IDX_QT;
    const uint32_t b0 = blockIdx.x * Q4E_IDX_BT;
    if (b0 >= n_blocks) return;
    extern __shared__ __align__(16) __half q4e_idx_smem[];
    __half *s_q = q4e_idx_smem;                                   /* [QT][n_head][128] */
    __half *s_k = s_q + (size_t)Q4E_IDX_QT * n_head * Q4E_IDX_DIM; /* [BT][128] */
    const uint32_t n_q = (n_tok - t0 < Q4E_IDX_QT) ? (n_tok - t0) : Q4E_IDX_QT;
    const uint32_t n_b = (n_blocks - b0 < Q4E_IDX_BT) ? (n_blocks - b0) : Q4E_IDX_BT;
    for (uint32_t idx = threadIdx.x; idx < n_q * n_head * Q4E_IDX_DIM / 8u; idx += blockDim.x) {
        ((uint4 *)s_q)[idx] = ((const uint4 *)(qn + (uint64_t)t0 * n_head * Q4E_IDX_DIM))[idx];
    }
    const uint32_t brow = q4e_kv_row(pages, b0 * Q4E_IDX_R) / Q4E_IDX_R;
    for (uint32_t idx = threadIdx.x; idx < n_b * Q4E_IDX_DIM / 8u; idx += blockDim.x) {
        ((uint4 *)s_k)[idx] = ((const uint4 *)(pooled + (uint64_t)brow * Q4E_IDX_DIM))[idx];
    }
    __syncthreads();
    /* 1024 (row, block) pairs over 256 threads: thread owns 4 blocks of one row
     * group so its q reads are shared. */
    for (uint32_t pair = threadIdx.x; pair < Q4E_IDX_QT * Q4E_IDX_BT; pair += blockDim.x) {
        const uint32_t qi = pair / Q4E_IDX_BT, bi = pair % Q4E_IDX_BT;
        if (qi >= n_q) break;
        const uint32_t t = t0 + qi, b = b0 + bi;
        float total = -INFINITY;
        if (bi < n_b && b * Q4E_IDX_R + Q4E_IDX_R - 1u <= (uint32_t)pos[t]) {
            total = 0.0f;
            const __half2 *k2 = (const __half2 *)(s_k + (size_t)bi * Q4E_IDX_DIM);
            for (uint32_t h = 0; h < n_head; h++) {
                const __half2 *q2 = (const __half2 *)(s_q + ((size_t)qi * n_head + h) * Q4E_IDX_DIM);
                float d = 0.0f;
#pragma unroll 8
                for (uint32_t i = 0; i < Q4E_IDX_DIM / 2u; i++) {
                    const float2 a = __half22float2(q2[i]), c = __half22float2(k2[i]);
                    d = fmaf(a.x, c.x, d);
                    d = fmaf(a.y, c.y, d);
                }
                total += fmaxf(d, 0.0f);
            }
        }
        score[(uint64_t)t * n_blocks_stride + b] = total;
    }
}

/* The same scores on tensor cores: a block owns 16 rows x 128 key blocks; for
 * each of the four indexer heads warp w forms S_h = Q_h K^T for its 16 key
 * blocks (two n8 fragments, eight k16 steps over the 128 dims) and folds
 * relu(S_h) into the fp32 total.  Q and K tiles are [row][128] fp16 in shared
 * memory with the 16-byte chunk index XORed by (row & 7) so ldmatrix is
 * conflict free.  Same inputs and masks as the fmaf kernel above; only the
 * summation order inside each dot differs. */
#define Q4E_IDX_TC_BT 128u
#define Q4E_IDX_TC_SMEM ((16u * 4u + Q4E_IDX_TC_BT) * Q4E_IDX_DIM * 2u)
__device__ __forceinline__ static uint32_t q4e_idx_tc_off(uint32_t row, uint32_t chunk) {
    return (row * 16u + (chunk ^ (row & 7u))) * 16u;
}
extern "C" __global__ void __launch_bounds__(256) q4e_idx_score_tc_kernel(
        float *score, const __half *qn, const __half *pooled, const int32_t *pos,
        const int32_t *pages, uint32_t n_tok, uint32_t n_blocks_stride, const int32_t *pos_last) {
    constexpr uint32_t NH = 4u, BT = Q4E_IDX_TC_BT;
    const uint32_t n_blocks = ((uint32_t)pos_last[0] + 1u) / Q4E_IDX_R;
    const uint32_t t0 = blockIdx.y * Q4E_IDX_QT;
    const uint32_t b0 = blockIdx.x * BT;
    if (b0 >= n_blocks) return;
    extern __shared__ __align__(16) uint8_t q4e_idx_tc_smem[];
    uint8_t *s_q = q4e_idx_tc_smem;                       /* [h][16][128] f16 */
    uint8_t *s_k = s_q + 16u * NH * Q4E_IDX_DIM * 2u;     /* [128][128] f16 */
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t n_q = (n_tok - t0 < Q4E_IDX_QT) ? (n_tok - t0) : Q4E_IDX_QT;
    const uint32_t n_b = (n_blocks - b0 < BT) ? (n_blocks - b0) : BT;

    /* Q rows: qn is [t][h][128]; rows past n_q are zero. */
    for (uint32_t i = tid; i < 16u * NH * 16u; i += 256u) {
        const uint32_t row = i >> 4u, c = i & 15u;           /* row = qi * NH + h */
        const uint32_t qi = row / NH, h = row - qi * NH;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (qi < n_q) v = ((const uint4 *)(qn + ((uint64_t)(t0 + qi) * NH + h) * Q4E_IDX_DIM))[c];
        *(uint4 *)(s_q + q4e_idx_tc_off(h * 16u + qi, c)) = v;
    }
    /* K rows: the tile's pooled keys are contiguous inside one page (the tile
     * is half a page); rows past n_b are zero. */
    const uint32_t brow = q4e_kv_row(pages, b0 * Q4E_IDX_R) / Q4E_IDX_R;
    for (uint32_t i = tid; i < BT * 16u; i += 256u) {
        const uint32_t bi = i >> 4u, c = i & 15u;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (bi < n_b) v = ((const uint4 *)(pooled + (uint64_t)(brow + bi) * Q4E_IDX_DIM))[c];
        *(uint4 *)(s_k + q4e_idx_tc_off(bi, c)) = v;
    }
    __syncthreads();

    float total[2][4];
#pragma unroll
    for (int f = 0; f < 2; f++) { total[f][0] = total[f][1] = total[f][2] = total[f][3] = 0.0f; }
#pragma unroll
    for (uint32_t h = 0; h < NH; h++) {
        float acc[2][4];
#pragma unroll
        for (int f = 0; f < 2; f++) { acc[f][0] = acc[f][1] = acc[f][2] = acc[f][3] = 0.0f; }
#pragma unroll
        for (uint32_t ks = 0; ks < Q4E_IDX_DIM / 16u; ks++) {
            uint32_t a[4];
            tt_ldmatrix_x4_addr(a, tt_smem_addr(s_q + q4e_idx_tc_off(h * 16u + (lane & 15u),
                                                                      ks * 2u + (lane >> 4u))));
#pragma unroll
            for (uint32_t f = 0; f < 2; f++) {
                uint32_t b[2];
                const uint32_t brow_s = warp * 16u + f * 8u + (lane & 7u);
                tt_ldmatrix_x2_addr(b, tt_smem_addr(s_k + q4e_idx_tc_off(brow_s, ks * 2u + ((lane >> 3u) & 1u))));
                tt_mma_m16n8k16_f16_f32(acc[f], a, b);
            }
        }
#pragma unroll
        for (int f = 0; f < 2; f++) {
            total[f][0] += fmaxf(acc[f][0], 0.0f); total[f][1] += fmaxf(acc[f][1], 0.0f);
            total[f][2] += fmaxf(acc[f][2], 0.0f); total[f][3] += fmaxf(acc[f][3], 0.0f);
        }
    }
    /* C layout: rows lane/4 and lane/4 + 8, columns 2 * (lane % 4), +1. */
#pragma unroll
    for (int half_ = 0; half_ < 2; half_++) {
        const uint32_t qi = (lane >> 2u) + 8u * (uint32_t)half_;
        if (qi >= n_q) continue;
        const uint32_t t = t0 + qi;
        const uint32_t p = (uint32_t)pos[t];
        float *row = score + (uint64_t)t * n_blocks_stride;
#pragma unroll
        for (uint32_t f = 0; f < 2; f++) {
#pragma unroll
            for (uint32_t j = 0; j < 2; j++) {
                const uint32_t bi = warp * 16u + f * 8u + 2u * (lane & 3u) + j;
                const uint32_t b = b0 + bi;
                const bool vis = bi < n_b && b * Q4E_IDX_R + Q4E_IDX_R - 1u <= p;
                row[b] = vis ? total[f][2 * half_ + j] : -INFINITY;
            }
        }
    }
}

/* Top-k blocks per row by radix select on the float bit pattern (4 passes of
 * 8 bits), then compaction.  -inf (invisible) blocks are never selected;
 * ties at the threshold enter lowest index first.  Output: sel[t][0..cnt)
 * block indices in ascending order, cnt <= k (D2). */
extern "C" __global__ void __launch_bounds__(1024) q4e_idx_topk_kernel(
        int32_t *sel, int32_t *cnt, const float *score, uint32_t n_blocks_stride,
        const int32_t *pos_last, uint32_t k) {
    const uint32_t t = blockIdx.x;
    const uint32_t n_blocks = ((uint32_t)pos_last[0] + 1u) / Q4E_IDX_R;
    const float *row = score + (uint64_t)t * n_blocks_stride;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_remaining, s_take;
    __shared__ int32_t s_out;
    const uint32_t tid = threadIdx.x;

    /* Monotonic key: larger float -> larger key. */
    auto key_of = [](float f) -> uint32_t {
        uint32_t u = __float_as_uint(f);
        return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    };
    const uint32_t neg_inf_key = key_of(-INFINITY);

    if (tid == 0u) { s_prefix = 0u; s_remaining = k; }
    __syncthreads();
    uint32_t prefix = 0u, remaining = k;
    for (int pass = 3; pass >= 0; pass--) {
        const uint32_t shift = 8u * (uint32_t)pass;
        const uint32_t mask_hi = pass == 3 ? 0u : (0xFFFFFFFFu << (shift + 8u));
        for (uint32_t i = tid; i < 256u; i += blockDim.x) hist[i] = 0u;
        __syncthreads();
        for (uint32_t i = tid; i < n_blocks; i += blockDim.x) {
            const uint32_t u = key_of(row[i]);
            if (u == neg_inf_key) continue;
            if ((u & mask_hi) == prefix) atomicAdd(&hist[(u >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t acc = 0u;
            int chosen = -1;
            for (int bin = 255; bin >= 0; bin--) {
                if (acc + hist[bin] >= remaining) { chosen = bin; break; }
                acc += hist[bin];
            }
            if (chosen < 0) {
                /* Fewer visible blocks than k: take everything visible. */
                s_remaining = remaining - acc; s_prefix = prefix; s_take = 0xFFFFFFFFu;
            } else {
                s_remaining = remaining - acc;      /* still needed inside the bin */
                s_prefix = prefix | ((uint32_t)chosen << shift);
                s_take = 0u;
            }
        }
        __syncthreads();
        if (s_take == 0xFFFFFFFFu) { prefix = 0u; remaining = 0u; break; }
        prefix = s_prefix;
        remaining = s_remaining;
        __syncthreads();
    }
    /* prefix is now the full threshold key T (when a bin was found in every
     * pass); everything > T is in, and `remaining` of the == T ties.
     *
     * Compaction in index order (D2): a tile of blockDim candidates at a time,
     * each thread's rank among the tile's taken blocks and tied blocks from a
     * block-wide scan, so the output order and the ties admitted depend on
     * the scores alone. */
    const bool take_all = (s_take == 0xFFFFFFFFu);
    __shared__ uint32_t s_warp_take[32], s_warp_eq[32];
    uint32_t base_out = 0u, base_eq = 0u;
    int32_t *out = sel + (uint64_t)t * k;
    const uint32_t lane = tid & 31u, warp = tid >> 5u, n_warp = (blockDim.x + 31u) >> 5u;
    for (uint32_t i0 = 0; i0 < n_blocks; i0 += blockDim.x) {
        const uint32_t i = i0 + tid;
        bool gt = false, eq = false;
        if (i < n_blocks) {
            const uint32_t u = key_of(row[i]);
            if (u != neg_inf_key) {
                gt = take_all || u > prefix;
                eq = !gt && u == prefix;
            }
        }
        const uint32_t beq = __ballot_sync(0xffffffffu, eq);
        const uint32_t below = (1u << lane) - 1u;
        if (lane == 0u) s_warp_eq[warp] = __popc(beq);
        __syncthreads();
        uint32_t off_eq = 0u, tot_eq = 0u;
        for (uint32_t w = 0; w < n_warp; w++) {
            if (w < warp) off_eq += s_warp_eq[w];
            tot_eq += s_warp_eq[w];
        }
        const uint32_t eq_rank = base_eq + off_eq + __popc(beq & below);
        const bool tie_in = eq && eq_rank < remaining;
        const uint32_t take_b = __ballot_sync(0xffffffffu, gt || tie_in);
        __syncthreads();
        if (lane == 0u) s_warp_take[warp] = __popc(take_b);
        __syncthreads();
        uint32_t off_take = 0u, tot_take = 0u;
        for (uint32_t w = 0; w < n_warp; w++) {
            if (w < warp) off_take += s_warp_take[w];
            tot_take += s_warp_take[w];
        }
        if (gt || tie_in) {
            const uint32_t o = base_out + off_take + __popc(take_b & below);
            if (o < k) out[o] = (int32_t)i;
        }
        base_out += tot_take;
        base_eq += tot_eq;
        __syncthreads();
    }
    if (tid == 0u) s_out = (int32_t)base_out;
    __syncthreads();
    if (tid == 0u) cnt[t] = s_out < (int32_t)k ? s_out : (int32_t)k;
}

/* Selected blocks -> token indices, plus the tail of the incomplete block. */
extern "C" __global__ void q4e_idx_expand_kernel(int32_t *tokens, int32_t *n_sel, const int32_t *sel,
                                             const int32_t *cnt, const int32_t *pos,
                                             uint32_t k, uint32_t width) {
    const uint32_t t = blockIdx.x;
    const int32_t c = cnt[t];
    int32_t *out = tokens + (uint64_t)t * width;
    for (uint32_t i = threadIdx.x; i < (uint32_t)c * Q4E_IDX_R; i += blockDim.x) {
        out[i] = sel[(uint64_t)t * k + i / Q4E_IDX_R] * (int32_t)Q4E_IDX_R + (int32_t)(i % Q4E_IDX_R);
    }
    const uint32_t p = (uint32_t)pos[t];
    const uint32_t tail0 = ((p + 1u) / Q4E_IDX_R) * Q4E_IDX_R;   /* first token of the incomplete block */
    const uint32_t n_tail = p + 1u - tail0;                        /* 0..r-1 */
    if (threadIdx.x < n_tail) out[(uint32_t)c * Q4E_IDX_R + threadIdx.x] = (int32_t)(tail0 + threadIdx.x);
    if (threadIdx.x == 0u) n_sel[t] = c * (int32_t)Q4E_IDX_R + (int32_t)n_tail;
}
