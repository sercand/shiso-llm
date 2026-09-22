/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels).
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * Gated DeltaNet (the 36 linear-attention layers).
 * ------------------------------------------------------------------------ */

/* Causal depthwise conv over the q|k|v channel block, then silu.
 *
 * The kernel window reaches SHS_N_GDN_CONV - 1 tokens back, so the tail of the
 * previous chunk is carried in a per-sequence state of that many columns.  The
 * state is updated to the new tail here, which is what makes a chunked prefill
 * produce the same result as a single-shot one.
 *
 * `ckpt`, when given, receives the window as it stands after each token t <
 * min(ckpt_cap, n_tok - 1), laid out like `state` at ckpt[t].  Speculative
 * verify runs several tokens in one call and, when the target rejects a
 * suffix, restores the checkpoint of the last accepted one. */
extern "C" __global__ void q4e_gdn_conv_kernel(
        float *out, float *state, const float *x, const float *w,
        uint32_t channels, uint32_t kernel, uint32_t n_tok,
        float *ckpt, uint32_t ckpt_cap) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const uint32_t hist = kernel - 1u;

    /* Gather this channel's history followed by the new tokens. */
    float win[8];
    for (uint32_t i = 0; i < hist; i++) win[i] = state[(uint64_t)i * channels + c];

    const float *wc = w + (uint64_t)c * kernel;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float cur = x[(uint64_t)t * channels + c];
        float acc = 0.0f;
        for (uint32_t k = 0; k < hist; k++) acc += wc[k] * win[k];
        acc += wc[hist] * cur;
        out[(uint64_t)t * channels + c] = acc / (1.0f + __expf(-acc));
        for (uint32_t k = 0; k + 1u < hist; k++) win[k] = win[k + 1];
        if (hist > 0) win[hist - 1] = cur;
        if (ckpt && t < ckpt_cap && t + 1u < n_tok) {
            float *ck = ckpt + (uint64_t)t * hist * channels;
            for (uint32_t i = 0; i < hist; i++) ck[(uint64_t)i * channels + c] = win[i];
        }
    }

    for (uint32_t i = 0; i < hist; i++) state[(uint64_t)i * channels + c] = win[i];
}

/* L2-normalize each 128-wide q and k head in place.  This is an L2 norm, not
 * an RMS norm: there is no weight and no epsilon-scaled mean. */
extern "C" __global__ void q4e_gdn_l2norm_kernel(
        float *qk, uint32_t head_dim, uint32_t n_heads, uint32_t stride, uint32_t offset) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    float *v = qk + (uint64_t)t * stride + offset + (uint64_t)h * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) sum += v[i] * v[i];
    sum = q4e_block_sum(sum);
    const float inv = rsqrtf(sum + 1e-12f);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) v[i] *= inv;
    (void)n_heads;
}

/* decay = softplus(alpha + dt_bias) * ssm_a and beta = sigmoid(beta_proj).
 * ssm_a already holds -exp(A_log), so decay comes out negative and the
 * recurrence below exponentiates it directly. */
extern "C" __global__ void q4e_gdn_gates_kernel(
        float *decay, float *beta,
        const float *alpha_proj, const float *beta_proj,
        const float *dt_bias, const float *a, uint32_t n_head, uint32_t n_tok) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (uint64_t)n_head * n_tok) return;
    const uint32_t h = (uint32_t)(i % n_head);
    const float x = alpha_proj[i] + dt_bias[h];
    /* softplus, guarded the way the reference is so large x does not overflow */
    const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
    decay[i] = sp * a[h];
    beta[i] = 1.0f / (1.0f + __expf(-beta_proj[i]));
}

/* The gated delta rule, one block per value head.
 *
 * The 128x128 state lives in shared memory for the whole chunk, so a prefill
 * reads and writes it once instead of once per token.  It is stored
 * transposed, m[j][i] = S[i][j], so the three inner products below all walk a
 * contiguous row.
 *
 * Value head h reads query/key head h % n_head_k -- modulo, not divide.  That
 * detail is easy to get backwards and produces plausible garbage.
 *
 * Per token:   S *= exp(decay)
 *              delta[j] = (v[j] - <S[:,j], k>) * beta
 *              S[:,j]  += delta[j] * k
 *              out[j]   = <S[:,j], q> / sqrt(head_dim)
 */
#define Q4E_GDN_SPLIT 4u   /* lanes per state column */
extern "C" __global__ void __launch_bounds__(512) q4e_gdn_recurrent_kernel(
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta,
        uint32_t head_dim, uint32_t n_head_k, uint32_t n_head_v,
        uint32_t k_offset, uint32_t v_offset, uint32_t stride, uint32_t n_tok,
        float *ckpt, uint32_t ckpt_cap) {
    extern __shared__ float q4e_gdn_smem[];    /* head_dim * (head_dim + 1) */
    float *s_state = q4e_gdn_smem;
    const uint32_t h = blockIdx.x;
    const uint32_t hk = h % n_head_k;
    /* Q4E_GDN_SPLIT adjacent lanes share one column j of S (row j of the
     * transposed state); each owns a quarter of the 128-long inner products
     * and the quarters meet in a shuffle.  One thread per column left the SM
     * with four warps and a 128-deep dependent FMA chain per token: 10.8 ms
     * for a 2048-token chunk.  The split reassociates the sums (a change of
     * the same kind as the chunk width, see trap 4). */
    const uint32_t j = threadIdx.x / Q4E_GDN_SPLIT;
    const uint32_t sub = threadIdx.x % Q4E_GDN_SPLIT;
    const uint32_t seg = head_dim / Q4E_GDN_SPLIT;          /* 32 */
    const float scale = rsqrtf((float)head_dim);
    const uint32_t row_stride = head_dim + 1u;

    float *m = state + (uint64_t)h * head_dim * head_dim;
    for (uint32_t idx = threadIdx.x; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        s_state[r * row_stride + c] = m[idx];
    }
    __syncthreads();

    __shared__ float s_k[128];
    __shared__ float s_q[128];
    __shared__ float s_kq;

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *base = qkv + (uint64_t)t * stride;
        const float *q_d = base + (uint64_t)hk * head_dim;
        const float *k_d = base + k_offset + (uint64_t)hk * head_dim;
        const float *v_d = base + v_offset + (uint64_t)h * head_dim;
        float kq_part = 0.0f;
        if (sub == 0u && j < head_dim) {
            const float kv = k_d[j];
            const float qv = q_d[j];
            s_k[j] = kv;
            s_q[j] = qv;
            kq_part = kv * qv;
        }
        kq_part = q4e_block_sum(kq_part);
        if (threadIdx.x == 0u) s_kq = kq_part;
        __syncthreads();

        const uint64_t gi = (uint64_t)t * n_head_v + h;
        const float dec = __expf(decay[gi]);
        const float bt = beta[gi];

        /* This lane's quarter of the row, walked from a rotated start so the
         * four lanes of a column, and the eight columns of a warp, hit
         * different banks. */
        float *row = s_state + (uint64_t)j * row_stride;
        const uint32_t i0 = sub * seg;
        float sk = 0.0f, sq = 0.0f;
        for (uint32_t n = 0; n < seg; n++) {
            const uint32_t i = i0 + ((n + 8u * sub) & (seg - 1u));
            const float mv = row[i];
            sk = fmaf(mv, s_k[i], sk);
            sq = fmaf(mv, s_q[i], sq);
        }
        sk += __shfl_xor_sync(0xffffffffu, sk, 1);
        sk += __shfl_xor_sync(0xffffffffu, sk, 2);
        sq += __shfl_xor_sync(0xffffffffu, sq, 1);
        sq += __shfl_xor_sync(0xffffffffu, sq, 2);
        const float delta = (v_d[j] - dec * sk) * bt;

        if (sub == 0u) {
            attn_out[((uint64_t)t * n_head_v + h) * head_dim + j] =
                (dec * sq + delta * s_kq) * scale;
        }
        for (uint32_t n = 0; n < seg; n++) {
            const uint32_t i = i0 + ((n + 8u * sub) & (seg - 1u));
            row[i] = fmaf(row[i], dec, delta * s_k[i]);
        }
        __syncthreads();
        if (ckpt && t < ckpt_cap && t + 1u < n_tok) {
            float *ck = ckpt + ((uint64_t)t * n_head_v + h) * head_dim * head_dim;
            for (uint32_t r = sub; r < head_dim; r += Q4E_GDN_SPLIT) ck[r * head_dim + j] = s_state[r * row_stride + j];
        }
    }

    for (uint32_t idx = threadIdx.x; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        m[idx] = s_state[r * row_stride + c];
    }
}

/* Chunked-parallel gated delta rule, for prefill.
 *
 * The recurrence above is exact but sequential: one block per value head
 * walks the tokens one at a time, and every token pays a dependent pass over
 * the whole 128x128 state.  At a 2048-token chunk that is 322 ms of a 26k
 * step.  The chunked-parallel (FLA / FlashQLA) form rewrites a run of C
 * tokens as matmuls against the state at the run's start, so the state is
 * touched twice per run instead of C times, and the only sequential work left
 * is the run-to-run carry.
 *
 * With g_t = log decay_t, G_t = sum_{s<=t} g_s inside the run and S_0 the
 * state at its start, unrolling S_t = e^{g_t}(I - b_t k_t k_t^T) S_{t-1} +
 * b_t k_t v_t^T gives
 *
 *     S_t   = e^{G_t} S_0 + sum_{u<=t} e^{G_t - G_u} k_u d_u^T
 *     d_t   = b_t (v_t - e^{G_t} S_0^T k_t) - sum_{u<t} A[t][u] d_u
 *     A[t][u] = b_t e^{G_t - G_u} (k_u . k_t)                       (u < t)
 *     o_t   = (e^{G_t} S_0^T q_t + sum_{u<=t} P[t][u] d_u) / sqrt(D)
 *     P[t][u] = e^{G_t - G_u} (q_t . k_u)                           (u <= t)
 *
 * Every exponent is a difference G_t - G_u with u <= t, i.e. a decay in
 * [0, 1]: no 1/e^G rescaling and nothing to overflow, which is the reason for
 * this arrangement rather than the textbook "divide the keys by the cumulative
 * decay" one.
 *
 * C = 64 because the shared-memory working set is what bounds it: keys,
 * deltas, the C x C coefficient matrix and a staged q tile come to 97 KiB at
 * 64, against the 99 KiB a block may opt into here, and the per-token
 * arithmetic grows as 3D + C (65 k MAC per token per head at C = 64 against
 * the sequential form's 49 k), so a wider run buys fewer carries at a rising
 * cost and does not fit beside the state anyway.
 *
 * The state stays in registers: thread (j, sub) holds the 16 keys
 * [sub*16, sub*16+16) of value column j, so the block is 128 columns x
 * Q4E_GDN_CSPLIT lanes = 1024 threads and the eight lanes of a column reduce
 * through shuffles.  Each lane reads its slice as float4 in a rotated order
 * (the sequential kernel's trick) so the eight lanes of a column hit eight
 * different shared-memory bank groups.  Phases, per run: keys in,
 * coefficients, right-hand sides, the C-step forward substitution, outputs (q
 * staged in tiles so the key rows can stay resident), and the carry into the
 * next run.  Only the staging phases and the two gram matrices need block
 * barriers -- a column's deltas are written and read inside one warp, so the
 * substitution's C dependent steps use __syncwarp().
 *
 * What it costs and why, measured on a 2048-token chunk with a standalone
 * microbenchmark against the sequential kernel (same inputs, agreement 4e-5%
 * relative L2): 8.7 -> 5.3 ms per layer, 1.65x.  The remainder is
 * shared-memory traffic, not arithmetic: every token's keys and queries are
 * re-read by all 32 warps (each owns one value column and 16 keys of it), so
 * a run moves ~19 MB through shared memory for ~4 M multiply-adds.  Cutting
 * that further means giving each thread several value columns so one shared
 * read feeds several products; the two arrangements tried (four columns per
 * thread with 32-lane reductions, and the wider gram tiles) both lost more to
 * the longer reductions than they saved, so this is where it stands.
 *
 * Numerics differ from the sequential kernel -- these are different sums, not
 * a reassociation of the same ones -- so this path is gated on the oracle,
 * not on the prefill fingerprint.  It also produces no per-token state, so it
 * cannot fill the speculative-rollback checkpoints; the launcher keeps runs
 * that need them on the sequential kernel. */
#define Q4E_GDN_CHUNK 64u    /* tokens per run */
#define Q4E_GDN_QTILE 32u    /* q rows staged at a time in the output phase */
#define Q4E_GDN_CSPLIT 8u    /* lanes per value column: 1024 threads, 32 warps */

extern "C" __global__ void __launch_bounds__(128u * Q4E_GDN_CSPLIT) q4e_gdn_chunk_kernel(
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta,
        uint32_t n_head_k, uint32_t n_head_v,
        uint32_t k_offset, uint32_t v_offset, uint32_t stride, uint32_t n_tok) {
    const uint32_t D = 128u;                  /* head_dim, checked by the launcher */
    const uint32_t C = Q4E_GDN_CHUNK;
    const uint32_t DS = D + 1u;               /* delta row stride: +1 so the
                                               * per-column reads below spread
                                               * over all 32 banks */
    extern __shared__ float q4e_gdn_chunk_smem[];
    float *s_K = q4e_gdn_chunk_smem;          /* [C][D]  keys */
    float *s_D = s_K + (uint64_t)C * D;       /* [C][DS] rhs, then deltas */
    float *s_A = s_D + (uint64_t)C * DS;      /* [C][C]  coefficients, then P */
    float *s_Q = s_A + (uint64_t)C * C;       /* [QTILE][D] staged queries */
    float *s_G = s_Q + (uint64_t)Q4E_GDN_QTILE * D;   /* [C] cumulative log decay */
    float *s_b = s_G + C;                     /* [C] beta */

    const uint32_t h = blockIdx.x;
    const uint32_t hk = h % n_head_k;
    const uint32_t NV4 = D / (4u * Q4E_GDN_CSPLIT);   /* float4 slots per lane */
    const uint32_t j = threadIdx.x / Q4E_GDN_CSPLIT;
    const uint32_t sub = threadIdx.x % Q4E_GDN_CSPLIT;
    const uint32_t i0 = sub * (D / Q4E_GDN_CSPLIT);
    const uint32_t rot = (sub * NV4) / Q4E_GDN_CSPLIT;  /* float4 rotation, see above */
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t n_warp = blockDim.x >> 5u;
    const float scale = rsqrtf((float)D);

    float4 *m4 = (float4 *)(state + ((uint64_t)h * D + j) * D + i0);
    float4 sreg[NV4];
#pragma unroll
    for (uint32_t n4 = 0; n4 < NV4; n4++) sreg[n4] = m4[(n4 + rot) & (NV4 - 1u)];

    for (uint32_t t0 = 0; t0 < n_tok; t0 += C) {
        const uint32_t n = (n_tok - t0 < C) ? (n_tok - t0) : C;

        /* Gates in parallel, then the inclusive prefix sum of the log decays
         * by one thread over shared memory: 64 dependent adds are cheaper
         * than a scan's barriers, but 128 strided global loads on that same
         * serial path are not. */
        for (uint32_t t = threadIdx.x; t < n; t += blockDim.x) {
            const uint64_t gi = (uint64_t)(t0 + t) * n_head_v + h;
            s_G[t] = decay[gi];
            s_b[t] = beta[gi];
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
            float acc = 0.0f;
            for (uint32_t t = 0; t < n; t++) { acc += s_G[t]; s_G[t] = acc; }
        }
        for (uint32_t idx = threadIdx.x; idx < n * D; idx += blockDim.x) {
            const uint32_t t = idx / D, i = idx - t * D;
            s_K[idx] = qkv[(uint64_t)(t0 + t) * stride + k_offset + hk * D + i];
        }
        __syncthreads();

        /* A[t][u] = b_t e^{G_t - G_u} (k_u . k_t), strictly below the
         * diagonal.  One warp per entry: its 32 lanes take four keys each so
         * both rows are read as contiguous float4. */
        for (uint32_t rr = warp; rr < C; rr += n_warp) {
            /* One warp per row, its own key row hoisted into registers: each
             * column then costs one shared read instead of two.  Rows are
             * handed out in low/high pairs (rr and C-1-rr) because row t has
             * t columns, so a warp that takes both gets a constant C-1.
             *
             * The bound is C and not the run length: the pairing sends the
             * second half of the rows to rr >= n_warp, so stopping at a short
             * run's n left rows [32, 96-n) of a partial run unwritten -- the
             * substitution then read stale shared memory for every run whose
             * length fell in [33, 63].  Rows past n are skipped below, one
             * predicate instead of a truncated loop. */
            const uint32_t t = ((rr / n_warp) % 2u == 0u) ? rr : (C - 1u - (rr - n_warp));
            if (t >= n) continue;
            const float4 kt = ((const float4 *)(s_K + t * D))[lane];
            const float bt = s_b[t], gt = s_G[t];
            for (uint32_t u = 0; u < t; u++) {
                const float4 ku = ((const float4 *)(s_K + u * D))[lane];
                float dot = kt.x * ku.x + kt.y * ku.y + kt.z * ku.z + kt.w * ku.w;
                dot = warp_sum_f32(dot);
                if (lane == 0u) s_A[t * C + u] = bt * __expf(gt - s_G[u]) * dot;
            }
        }
        __syncthreads();

        /* rhs_t = b_t (v_t - e^{G_t} S^T k_t). */
        for (uint32_t t = 0; t < n; t++) {
            const float4 *k4 = (const float4 *)(s_K + t * D + i0);
            float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
            for (uint32_t n4 = 0; n4 < NV4; n4++) {
                const float4 kv = k4[(n4 + rot) & (NV4 - 1u)];
                p0 = fmaf(sreg[n4].x, kv.x, p0);
                p1 = fmaf(sreg[n4].y, kv.y, p1);
                p2 = fmaf(sreg[n4].z, kv.z, p2);
                p3 = fmaf(sreg[n4].w, kv.w, p3);
            }
            float p = (p0 + p1) + (p2 + p3);
#pragma unroll
            for (uint32_t o = 1u; o < Q4E_GDN_CSPLIT; o <<= 1) p += __shfl_xor_sync(0xffffffffu, p, o);
            if (sub == 0u) {
                const float v = qkv[(uint64_t)(t0 + t) * stride + v_offset + h * D + j];
                s_D[t * DS + j] = s_b[t] * (v - p * __expf(s_G[t]));
            }
        }
        __syncwarp();

        /* Forward substitution: d_t = rhs_t - sum_{u<t} A[t][u] d_u.  C
         * dependent steps, but each is one rank-1 update over the value
         * dimension instead of a pass over the state -- that trade is what
         * the chunk buys. */
        for (uint32_t t = 1u; t < n; t++) {
            float acc = 0.0f;
            for (uint32_t u = sub; u < t; u += Q4E_GDN_CSPLIT) {
                acc = fmaf(s_A[t * C + u], s_D[u * DS + j], acc);
            }
#pragma unroll
            for (uint32_t o = 1u; o < Q4E_GDN_CSPLIT; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (sub == 0u) s_D[t * DS + j] -= acc;
            __syncwarp();
        }

        /* Outputs.  The queries are staged a tile at a time so the keys stay
         * resident: P needs both, and both at full width do not fit. */
        for (uint32_t tb = 0; tb < n; tb += Q4E_GDN_QTILE) {
            const uint32_t nt = (n - tb < Q4E_GDN_QTILE) ? (n - tb) : Q4E_GDN_QTILE;
            for (uint32_t idx = threadIdx.x; idx < nt * D; idx += blockDim.x) {
                const uint32_t tt = idx / D, i = idx - tt * D;
                s_Q[idx] = qkv[(uint64_t)(t0 + tb + tt) * stride + hk * D + i];
            }
            __syncthreads();

            for (uint32_t pair = warp; pair < nt * C; pair += n_warp) {
                const uint32_t tt = pair / C, u = pair - tt * C;
                const uint32_t t = tb + tt;
                float v = 0.0f;
                if (u <= t) {
                    const float4 a = ((const float4 *)(s_Q + tt * D))[lane];
                    const float4 b = ((const float4 *)(s_K + u * D))[lane];
                    float dot = a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                    dot = warp_sum_f32(dot);
                    v = __expf(s_G[t] - s_G[u]) * dot;
                }
                if (lane == 0u) s_A[pair] = v;
            }
            __syncthreads();

            for (uint32_t tt = 0; tt < nt; tt++) {
                const uint32_t t = tb + tt;
                const float4 *q4 = (const float4 *)(s_Q + tt * D + i0);
                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
#pragma unroll
                for (uint32_t n4 = 0; n4 < NV4; n4++) {
                    const float4 qv = q4[(n4 + rot) & (NV4 - 1u)];
                    p0 = fmaf(sreg[n4].x, qv.x, p0);
                    p1 = fmaf(sreg[n4].y, qv.y, p1);
                    p2 = fmaf(sreg[n4].z, qv.z, p2);
                    p3 = fmaf(sreg[n4].w, qv.w, p3);
                }
                const float p = (p0 + p1) + (p2 + p3);
                float acc = 0.0f;
                for (uint32_t u = sub; u <= t; u += Q4E_GDN_CSPLIT) {
                    acc = fmaf(s_A[tt * C + u], s_D[u * DS + j], acc);
                }
                acc = fmaf(p, __expf(s_G[t]), acc);
#pragma unroll
                for (uint32_t o = 1u; o < Q4E_GDN_CSPLIT; o <<= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
                if (sub == 0u) {
                    attn_out[((uint64_t)(t0 + t) * n_head_v + h) * D + j] = acc * scale;
                }
            }
            __syncthreads();
        }

        /* Carry: S <- e^{G_last} S + sum_t e^{G_last - G_t} k_t d_t^T. */
        const float g_last = s_G[n - 1u];
        const float dec = __expf(g_last);
#pragma unroll
        for (uint32_t n4 = 0; n4 < NV4; n4++) {
            sreg[n4].x *= dec; sreg[n4].y *= dec;
            sreg[n4].z *= dec; sreg[n4].w *= dec;
        }
        for (uint32_t t = 0; t < n; t++) {
            const float d = s_D[t * DS + j] * __expf(g_last - s_G[t]);
            const float4 *k4 = (const float4 *)(s_K + t * D + i0);
#pragma unroll
            for (uint32_t n4 = 0; n4 < NV4; n4++) {
                const float4 kv = k4[(n4 + rot) & (NV4 - 1u)];
                sreg[n4].x = fmaf(kv.x, d, sreg[n4].x);
                sreg[n4].y = fmaf(kv.y, d, sreg[n4].y);
                sreg[n4].z = fmaf(kv.z, d, sreg[n4].z);
                sreg[n4].w = fmaf(kv.w, d, sreg[n4].w);
            }
        }
        __syncthreads();   /* s_K / s_D are rewritten by the next run */
    }

#pragma unroll
    for (uint32_t n4 = 0; n4 < NV4; n4++) m4[(n4 + rot) & (NV4 - 1u)] = sreg[n4];
}

/* Per-head RMSNorm of the delta-rule output, then the sigmoid output gate.
 * ssm_norm is the one norm in the model the converter does NOT offset by one,
 * so it is applied raw here just like every other weight. */
extern "C" __global__ void q4e_gdn_out_gate_kernel(
        float *out, const float *attn, const float *z, const float *w,
        uint32_t head_dim, uint32_t n_head, float eps) {
    const uint32_t group = blockIdx.x;         /* (token, head) */
    const float *src = attn + (uint64_t)group * head_dim;
    const float *zr = z + (uint64_t)group * head_dim;
    float *dst = out + (uint64_t)group * head_dim;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) sum += src[i] * src[i];
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float g = 1.0f / (1.0f + __expf(-zr[i]));
        dst[i] = src[i] * scale * w[i] * g;
    }
    (void)n_head;
}

/* ---------------------------------------------------------------------------
 * Batched-decode variants of the three recurrent-state kernels.
 *
 * A batched decode passes one token per sequence (row) through the model in a
 * single forward.  These three kernels each carry per-sequence state, so the
 * scalar path loops them once per row on the host.  The batched variants add a
 * grid.y = row dimension and take a device array of per-row state pointers
 * (built once per step by shs_gpu_q4e_ptrs_pack), so one launch serves every
 * row.  Each row is exactly one token, so there is no chained scan and no
 * checkpoint -- the math is the single-token body of the scalar kernel with
 * every I/O offset by its row.  All threads in a block share blockIdx.y, so a
 * NULL row-state guard cannot deadlock a __syncthreads.
 * ------------------------------------------------------------------------ */
extern "C" __global__ void q4e_gdn_conv_batch_kernel(
        float *out, float * const *states, const float *x, const float *w,
        uint32_t channels, uint32_t kernel) {
    const uint32_t row = blockIdx.y;
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float *state = states[row];
    if (!state) return;
    const uint32_t hist = kernel - 1u;
    const float *xr = x + (uint64_t)row * channels;
    float *outr = out + (uint64_t)row * channels;
    float win[8];
    for (uint32_t i = 0; i < hist; i++) win[i] = state[(uint64_t)i * channels + c];
    const float *wc = w + (uint64_t)c * kernel;
    const float cur = xr[c];
    float acc = 0.0f;
    for (uint32_t k = 0; k < hist; k++) acc += wc[k] * win[k];
    acc += wc[hist] * cur;
    outr[c] = acc / (1.0f + __expf(-acc));
    for (uint32_t k = 0; k + 1u < hist; k++) win[k] = win[k + 1];
    if (hist > 0) win[hist - 1] = cur;
    for (uint32_t i = 0; i < hist; i++) state[(uint64_t)i * channels + c] = win[i];
}

extern "C" __global__ void __launch_bounds__(512) q4e_gdn_recurrent_batch_kernel(
        float *attn_out, float * const *states,
        const float *qkv, const float *decay, const float *beta,
        uint32_t head_dim, uint32_t n_head_k, uint32_t n_head_v,
        uint32_t k_offset, uint32_t v_offset, uint32_t stride) {
    extern __shared__ float q4e_gdn_smem[];
    float *s_state = q4e_gdn_smem;
    const uint32_t row = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const uint32_t hk = h % n_head_k;
    const uint32_t j = threadIdx.x / Q4E_GDN_SPLIT;
    const uint32_t sub = threadIdx.x % Q4E_GDN_SPLIT;
    const uint32_t seg = head_dim / Q4E_GDN_SPLIT;
    const float scale = rsqrtf((float)head_dim);
    const uint32_t row_stride = head_dim + 1u;
    float *state = states[row];
    if (!state) return;

    float *m = state + (uint64_t)h * head_dim * head_dim;
    for (uint32_t idx = threadIdx.x; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        s_state[r * row_stride + c] = m[idx];
    }
    __syncthreads();

    __shared__ float s_k[128];
    __shared__ float s_q[128];
    __shared__ float s_kq;

    const float *base = qkv + (uint64_t)row * stride;
    const float *q_d = base + (uint64_t)hk * head_dim;
    const float *k_d = base + k_offset + (uint64_t)hk * head_dim;
    const float *v_d = base + v_offset + (uint64_t)h * head_dim;
    float kq_part = 0.0f;
    if (sub == 0u && j < head_dim) {
        const float kv = k_d[j];
        const float qv = q_d[j];
        s_k[j] = kv;
        s_q[j] = qv;
        kq_part = kv * qv;
    }
    kq_part = q4e_block_sum(kq_part);
    if (threadIdx.x == 0u) s_kq = kq_part;
    __syncthreads();

    const uint64_t gi = (uint64_t)row * n_head_v + h;
    const float dec = __expf(decay[gi]);
    const float bt = beta[gi];

    float *rowp = s_state + (uint64_t)j * row_stride;
    const uint32_t i0 = sub * seg;
    float sk = 0.0f, sq = 0.0f;
    for (uint32_t n = 0; n < seg; n++) {
        const uint32_t i = i0 + ((n + 8u * sub) & (seg - 1u));
        const float mv = rowp[i];
        sk = fmaf(mv, s_k[i], sk);
        sq = fmaf(mv, s_q[i], sq);
    }
    sk += __shfl_xor_sync(0xffffffffu, sk, 1);
    sk += __shfl_xor_sync(0xffffffffu, sk, 2);
    sq += __shfl_xor_sync(0xffffffffu, sq, 1);
    sq += __shfl_xor_sync(0xffffffffu, sq, 2);
    const float delta = (v_d[j] - dec * sk) * bt;

    if (sub == 0u) {
        attn_out[((uint64_t)row * n_head_v + h) * head_dim + j] =
            (dec * sq + delta * s_kq) * scale;
    }
    for (uint32_t n = 0; n < seg; n++) {
        const uint32_t i = i0 + ((n + 8u * sub) & (seg - 1u));
        rowp[i] = fmaf(rowp[i], dec, delta * s_k[i]);
    }
    __syncthreads();

    for (uint32_t idx = threadIdx.x; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        m[idx] = s_state[r * row_stride + c];
    }
}
