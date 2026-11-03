// q4e idx entries: Metal port of kernels/cuda/q4e/idx.cu (see q4e_common.h for conventions).
//
// The indexer that picks which of the context an attention block reads: an MQA
// tower of four 128-wide query heads over one key head, keys mean-pooled four
// positions to a block, a block scored as sum_h relu(<q_h, k_b>), the best
// blocks kept and expanded back into positions.
//
// Ported: q4e_idx_store_k_kernel, q4e_idx_pool_kernel, q4e_idx_q_kernel,
// q4e_idx_score_kernel (the scalar scorer), q4e_idx_topk_kernel,
// q4e_idx_expand_kernel. Not ported: q4e_idx_store_k_batch_kernel (it reads a
// device array of page-table pointers, which the Metal binding cannot pack
// yet) and q4e_idx_score_tc_kernel (tensor-core fragments).
//
// Every entry fits the 32 KiB threadgroup budget as written. The scalar scorer
// is the tight one: at four heads its dynamic slab is (16 * 4 + 64) * 128 * 2 =
// 32768 bytes, exactly the limit, and it declares no static threadgroup memory
// so the two together stay inside it. A head count above four would ask for
// more than this device grants, and run.rs refuses that launch.

#define Q4E_IDX_DIM   128u
#define Q4E_IDX_R     4u
#define Q4E_IDX_QT    16u
#define Q4E_IDX_BT    64u

static_assert(Q4E_PAGE_TOKENS % Q4E_IDX_R == 0u,
              "a pooled key block must not straddle a KV page");
static_assert(Q4E_PAGE_TOKENS % (Q4E_IDX_BT * Q4E_IDX_R) == 0u,
              "a key-block tile must not straddle a KV page");

/// `q4e_idx_store_k_kernel` (idx.cu:32): raw indexer keys into the f16 cache at
/// the rows' paged positions. One threadgroup per token.
///
/// Buffers: 0 `cache` (f16 plane), 1 `k` (f32 `[n_tok][128]`), 2 `pos`,
/// 3 `pages`, 4 `n_tok`.
kernel void q4e_idx_store_k_kernel(device half *cache [[buffer(0)]],
                                   device const float *k [[buffer(1)]],
                                   device const int *pos [[buffer(2)]],
                                   device const int *pages [[buffer(3)]],
                                   constant uint &n_tok [[buffer(4)]],
                                   uint3 bid [[threadgroup_position_in_grid]],
                                   uint3 tid3 [[thread_position_in_threadgroup]],
                                   uint3 bdim [[threads_per_threadgroup]]) {
    const uint t = bid.x;
    if (t >= n_tok) return;
    const ulong dst = ulong(q4e_kv_row(pages, uint(pos[t]))) * Q4E_IDX_DIM;
    for (uint i = tid3.x; i < Q4E_IDX_DIM; i += bdim.x) {
        cache[dst + i] = half(k[ulong(t) * Q4E_IDX_DIM + i]);
    }
}

/// `q4e_idx_pool_kernel` (idx.cu:65): one threadgroup per key block `b` in
/// `[pos[0] / r, (pos[n_tok - 1] + 1) / r)`: the mean of its `r` raw keys,
/// RMSNorm with the k-norm weight, mRoPE at the block's start position, stored
/// f16 at the block's pooled row.
///
/// Buffers: 0 `pooled`, 1 `cache`, 2 `w`, 3 `pos`, 4 `pages`, 5 `n_tok`,
/// 6 `n_rot`, 7 `rope_base`, 8 `mrope_all`, 9 `sec_t`, 10 `sec_h`, 11 `sec_w`,
/// 12 `eps`.
///
/// The block range is derived on the device from the positions, and the host
/// launches `n_tok / r + 1` threadgroups (a truncating divide plus one), so
/// the surplus ones exit on `b >= b1`. The start position `r * b` can sit up
/// to `r - 1` slots before `pos[0]`, which is why `mrope_all` carries
/// `Q4E_MROPE_BACK` rows in front of token 0 and the row index reaches back
/// into them.
kernel void q4e_idx_pool_kernel(device half *pooled [[buffer(0)]],
                                device const half *cache [[buffer(1)]],
                                device const float *w [[buffer(2)]],
                                device const int *pos [[buffer(3)]],
                                device const int *pages [[buffer(4)]],
                                constant uint &n_tok [[buffer(5)]],
                                constant uint &n_rot [[buffer(6)]],
                                constant float &rope_base [[buffer(7)]],
                                device const int *mrope_all [[buffer(8)]],
                                constant uint &sec_t [[buffer(9)]],
                                constant uint &sec_h [[buffer(10)]],
                                constant uint &sec_w [[buffer(11)]],
                                constant float &eps [[buffer(12)]],
                                uint3 bid [[threadgroup_position_in_grid]],
                                uint3 tid3 [[thread_position_in_threadgroup]],
                                uint3 bdim [[threads_per_threadgroup]]) {
    const uint tid = tid3.x;
    const uint b0 = uint(pos[0]) / Q4E_IDX_R;
    const uint b1 = (uint(pos[n_tok - 1u]) + 1u) / Q4E_IDX_R;
    const uint b = b0 + bid.x;
    if (b >= b1) return;
    const uint krow = q4e_kv_row(pages, b * Q4E_IDX_R);
    const uint brow = krow / Q4E_IDX_R;
    threadgroup float s_k[Q4E_IDX_DIM];
    threadgroup float s_red[Q4E_SLOTS];
    float sum = 0.0f;
    for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) {
        float acc = 0.0f;
        for (uint j = 0; j < Q4E_IDX_R; j++) {
            acc += float(cache[(ulong(krow) + j) * Q4E_IDX_DIM + i]);
        }
        acc *= 1.0f / float(Q4E_IDX_R);
        s_k[i] = acc;
        sum += acc * acc;
    }
    const float scale = rsqrt(q4e_block_sum(sum, s_red, tid, bdim.x) / float(Q4E_IDX_DIM) + eps);
    for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) s_k[i] *= scale * w[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int block_pos = int(b * Q4E_IDX_R);
    const uint mrow = uint(int(Q4E_MROPE_BACK) + block_pos - pos[0]);
    q4e_rope_mrope(s_k, n_rot, rope_base,
                   mrope_all[3u * mrow], mrope_all[3u * mrow + 1u], mrope_all[3u * mrow + 2u],
                   sec_t, sec_h, sec_w, tid, bdim.x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) {
        pooled[ulong(brow) * Q4E_IDX_DIM + i] = half(s_k[i]);
    }
}

/// `q4e_idx_q_kernel` (idx.cu:113): per (head, row) threadgroup, RMSNorm and
/// mRoPE of the indexer query at the row's own position, stored f16.
/// `grid.x` is the head and `grid.y` the row.
///
/// Buffers: 0 `qn`, 1 `q`, 2 `w`, 3 `mrope` (row 0 is token 0, no lookback),
/// 4 `sec_t`, 5 `sec_h`, 6 `sec_w`, 7 `n_head`, 8 `n_rot`, 9 `rope_base`,
/// 10 `eps`.
kernel void q4e_idx_q_kernel(device half *qn [[buffer(0)]],
                             device const float *q [[buffer(1)]],
                             device const float *w [[buffer(2)]],
                             device const int *mrope [[buffer(3)]],
                             constant uint &sec_t [[buffer(4)]],
                             constant uint &sec_h [[buffer(5)]],
                             constant uint &sec_w [[buffer(6)]],
                             constant uint &n_head [[buffer(7)]],
                             constant uint &n_rot [[buffer(8)]],
                             constant float &rope_base [[buffer(9)]],
                             constant float &eps [[buffer(10)]],
                             uint3 bid [[threadgroup_position_in_grid]],
                             uint3 tid3 [[thread_position_in_threadgroup]],
                             uint3 bdim [[threads_per_threadgroup]]) {
    const uint tid = tid3.x;
    const uint t = bid.y, h = bid.x;
    const ulong base = (ulong(t) * n_head + h) * Q4E_IDX_DIM;
    threadgroup float s_q[Q4E_IDX_DIM];
    threadgroup float s_red[Q4E_SLOTS];
    float sum = 0.0f;
    for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) {
        const float v = q[base + i];
        s_q[i] = v;
        sum += v * v;
    }
    const float scale = rsqrt(q4e_block_sum(sum, s_red, tid, bdim.x) / float(Q4E_IDX_DIM) + eps);
    for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) s_q[i] *= scale * w[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q4e_rope_mrope(s_q, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, tid, bdim.x);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) qn[base + i] = half(s_q[i]);
}

/// `q4e_idx_score_kernel` (idx.cu:146): the scalar block scorer.
/// `score[t][b] = sum_h relu(<q_th, k_b>)` for a block row `t` has fully seen
/// (`r * b + r - 1 <= pos[t]`), else `-inf`. A threadgroup covers 16 rows x 64
/// key blocks with both operands staged f16 in the dynamic slab
/// (`[16][n_head][128]` queries, then `[64][128]` keys).
///
/// Buffers: 0 `score`, 1 `qn`, 2 `pooled`, 3 `pos`, 4 `pages`, 5 `n_head`,
/// 6 `n_tok`, 7 `n_blocks_stride`, 8 `pos_last` (a one-element slice of the
/// position row). Threadgroup memory 0: `(16 * n_head + 64) * 128 * 2` bytes.
///
/// A key tile is 64 blocks = 256 positions = one page, and tiles start at page
/// boundaries, so one page-table lookup covers the whole tile.
kernel void q4e_idx_score_kernel(device float *score [[buffer(0)]],
                                 device const half *qn [[buffer(1)]],
                                 device const half *pooled [[buffer(2)]],
                                 device const int *pos [[buffer(3)]],
                                 device const int *pages [[buffer(4)]],
                                 constant uint &n_head [[buffer(5)]],
                                 constant uint &n_tok [[buffer(6)]],
                                 constant uint &n_blocks_stride [[buffer(7)]],
                                 device const int *pos_last [[buffer(8)]],
                                 threadgroup half *smem [[threadgroup(0)]],
                                 uint3 bid [[threadgroup_position_in_grid]],
                                 uint3 tid3 [[thread_position_in_threadgroup]],
                                 uint3 bdim [[threads_per_threadgroup]]) {
    const uint tid = tid3.x;
    const uint n_blocks = (uint(pos_last[0]) + 1u) / Q4E_IDX_R;
    const uint t0 = bid.y * Q4E_IDX_QT;
    const uint b0 = bid.x * Q4E_IDX_BT;
    if (b0 >= n_blocks) return;
    threadgroup half *s_q = smem;
    threadgroup half *s_k = s_q + Q4E_IDX_QT * n_head * Q4E_IDX_DIM;
    const uint n_q = (n_tok - t0 < Q4E_IDX_QT) ? (n_tok - t0) : Q4E_IDX_QT;
    const uint n_b = (n_blocks - b0 < Q4E_IDX_BT) ? (n_blocks - b0) : Q4E_IDX_BT;
    {
        device const uint4 *src = (device const uint4 *)(qn + ulong(t0) * n_head * Q4E_IDX_DIM);
        threadgroup uint4 *dst = (threadgroup uint4 *)s_q;
        for (uint idx = tid; idx < n_q * n_head * Q4E_IDX_DIM / 8u; idx += bdim.x) dst[idx] = src[idx];
    }
    const uint brow = q4e_kv_row(pages, b0 * Q4E_IDX_R) / Q4E_IDX_R;
    {
        device const uint4 *src = (device const uint4 *)(pooled + ulong(brow) * Q4E_IDX_DIM);
        threadgroup uint4 *dst = (threadgroup uint4 *)s_k;
        for (uint idx = tid; idx < n_b * Q4E_IDX_DIM / 8u; idx += bdim.x) dst[idx] = src[idx];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pair = tid; pair < Q4E_IDX_QT * Q4E_IDX_BT; pair += bdim.x) {
        const uint qi = pair / Q4E_IDX_BT, bi = pair % Q4E_IDX_BT;
        if (qi >= n_q) break;
        const uint t = t0 + qi, b = b0 + bi;
        float total = -INFINITY;
        if (bi < n_b && b * Q4E_IDX_R + Q4E_IDX_R - 1u <= uint(pos[t])) {
            total = 0.0f;
            threadgroup const half2 *k2 = (threadgroup const half2 *)(s_k + bi * Q4E_IDX_DIM);
            for (uint h = 0; h < n_head; h++) {
                threadgroup const half2 *q2 =
                    (threadgroup const half2 *)(s_q + (qi * n_head + h) * Q4E_IDX_DIM);
                float d = 0.0f;
                for (uint i = 0; i < Q4E_IDX_DIM / 2u; i++) {
                    const float2 a = float2(q2[i]), c = float2(k2[i]);
                    d = fma(a.x, c.x, d);
                    d = fma(a.y, c.y, d);
                }
                total += fmax(d, 0.0f);
            }
        }
        score[ulong(t) * n_blocks_stride + b] = total;
    }
}

/// Monotonic key of a float's bit pattern: a larger float is a larger key.
static inline uint q4e_idx_key_of(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

/// `q4e_idx_topk_kernel` (idx.cu:300): top-`k` blocks per row by a radix select
/// on the key of the float bit pattern (four passes of 8 bits), then an
/// index-order compaction. `-inf` blocks are never selected; ties at the
/// threshold enter lowest index first. Output: `sel[t][0 .. cnt)` block
/// indices ascending, `cnt <= k`. One 1024-thread threadgroup per row.
///
/// Buffers: 0 `sel` (`[rows][k]`), 1 `cnt`, 2 `score`, 3 `n_blocks_stride`,
/// 4 `pos_last`, 5 `k`.
///
/// The compaction's warp ballots are simdgroup ballots of the same 32 lanes:
/// the threadgroup is one-dimensional, so simdgroup `tid / 32` holds exactly
/// the threads CUDA's warp `tid / 32` does, and the output order depends on
/// the scores alone.
kernel void q4e_idx_topk_kernel(device int *sel [[buffer(0)]],
                                device int *cnt [[buffer(1)]],
                                device const float *score [[buffer(2)]],
                                constant uint &n_blocks_stride [[buffer(3)]],
                                device const int *pos_last [[buffer(4)]],
                                constant uint &k [[buffer(5)]],
                                uint3 bid [[threadgroup_position_in_grid]],
                                uint3 tid3 [[thread_position_in_threadgroup]],
                                uint3 bdim [[threads_per_threadgroup]]) {
    const uint t = bid.x;
    const uint n_blocks = (uint(pos_last[0]) + 1u) / Q4E_IDX_R;
    device const float *row = score + ulong(t) * n_blocks_stride;
    threadgroup atomic_uint hist[256];
    threadgroup uint s_prefix, s_remaining, s_take;
    threadgroup int s_out;
    const uint tid = tid3.x;
    const uint nthreads = bdim.x;

    const uint neg_inf_key = q4e_idx_key_of(-INFINITY);

    if (tid == 0u) { s_prefix = 0u; s_remaining = k; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint prefix = 0u, remaining = k;
    for (int pass = 3; pass >= 0; pass--) {
        const uint shift = 8u * uint(pass);
        const uint mask_hi = pass == 3 ? 0u : (0xFFFFFFFFu << (shift + 8u));
        for (uint i = tid; i < 256u; i += nthreads) {
            atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n_blocks; i += nthreads) {
            const uint u = q4e_idx_key_of(row[i]);
            if (u == neg_inf_key) continue;
            if ((u & mask_hi) == prefix) {
                atomic_fetch_add_explicit(&hist[(u >> shift) & 0xFFu], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint acc = 0u;
            int chosen = -1;
            for (int bin = 255; bin >= 0; bin--) {
                const uint hb = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (acc + hb >= remaining) { chosen = bin; break; }
                acc += hb;
            }
            if (chosen < 0) {
                // Fewer visible blocks than k: take everything visible.
                s_remaining = remaining - acc; s_prefix = prefix; s_take = 0xFFFFFFFFu;
            } else {
                s_remaining = remaining - acc;
                s_prefix = prefix | (uint(chosen) << shift);
                s_take = 0u;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s_take == 0xFFFFFFFFu) { prefix = 0u; remaining = 0u; break; }
        prefix = s_prefix;
        remaining = s_remaining;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const bool take_all = (s_take == 0xFFFFFFFFu);
    threadgroup uint s_warp_take[32], s_warp_eq[32];
    uint base_out = 0u, base_eq = 0u;
    device int *out = sel + ulong(t) * k;
    const uint lane = tid & 31u, warp = tid >> 5u, n_warp = (nthreads + 31u) >> 5u;
    for (uint i0 = 0; i0 < n_blocks; i0 += nthreads) {
        const uint i = i0 + tid;
        bool gt = false, eq = false;
        if (i < n_blocks) {
            const uint u = q4e_idx_key_of(row[i]);
            if (u != neg_inf_key) {
                gt = take_all || u > prefix;
                eq = !gt && u == prefix;
            }
        }
        const uint beq = uint(ulong(simd_vote::vote_t(simd_ballot(eq))) & 0xFFFFFFFFul);
        const uint below = (1u << lane) - 1u;
        if (lane == 0u) s_warp_eq[warp] = popcount(beq);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint off_eq = 0u, tot_eq = 0u;
        for (uint w = 0; w < n_warp; w++) {
            if (w < warp) off_eq += s_warp_eq[w];
            tot_eq += s_warp_eq[w];
        }
        const uint eq_rank = base_eq + off_eq + popcount(beq & below);
        const bool tie_in = eq && eq_rank < remaining;
        const uint take_b =
            uint(ulong(simd_vote::vote_t(simd_ballot(gt || tie_in))) & 0xFFFFFFFFul);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0u) s_warp_take[warp] = popcount(take_b);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint off_take = 0u, tot_take = 0u;
        for (uint w = 0; w < n_warp; w++) {
            if (w < warp) off_take += s_warp_take[w];
            tot_take += s_warp_take[w];
        }
        if (gt || tie_in) {
            const uint o = base_out + off_take + popcount(take_b & below);
            if (o < k) out[o] = int(i);
        }
        base_out += tot_take;
        base_eq += tot_eq;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0u) s_out = int(base_out);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) cnt[t] = s_out < int(k) ? s_out : int(k);
}

/// `q4e_idx_expand_kernel` (idx.cu:410): selected blocks to token indices,
/// then the tail of the row's incomplete block. One threadgroup per row.
///
/// Buffers: 0 `tokens` (`[rows][width]`), 1 `n_sel`, 2 `sel` (`[rows][k]`),
/// 3 `cnt`, 4 `pos`, 5 `k`, 6 `width`.
kernel void q4e_idx_expand_kernel(device int *tokens [[buffer(0)]],
                                  device int *n_sel [[buffer(1)]],
                                  device const int *sel [[buffer(2)]],
                                  device const int *cnt [[buffer(3)]],
                                  device const int *pos [[buffer(4)]],
                                  constant uint &k [[buffer(5)]],
                                  constant uint &width [[buffer(6)]],
                                  uint3 bid [[threadgroup_position_in_grid]],
                                  uint3 tid3 [[thread_position_in_threadgroup]],
                                  uint3 bdim [[threads_per_threadgroup]]) {
    const uint t = bid.x;
    const uint tid = tid3.x;
    const int c = cnt[t];
    device int *out = tokens + ulong(t) * width;
    for (uint i = tid; i < uint(c) * Q4E_IDX_R; i += bdim.x) {
        out[i] = sel[ulong(t) * k + i / Q4E_IDX_R] * int(Q4E_IDX_R) + int(i % Q4E_IDX_R);
    }
    const uint p = uint(pos[t]);
    const uint tail0 = ((p + 1u) / Q4E_IDX_R) * Q4E_IDX_R;
    const uint n_tail = p + 1u - tail0;
    if (tid < n_tail) out[uint(c) * Q4E_IDX_R + tid] = int(tail0 + tid);
    if (tid == 0u) n_sel[t] = c * int(Q4E_IDX_R) + int(n_tail);
}
