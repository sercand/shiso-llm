/* This family's own variants of two ported paths: the tensor-core attention
 * with its softmax weights carried exactly into the P V product, and the
 * whole of a gated recurrent layer's decode row after its convolution in one
 * launch.
 *
 * Not an upstream unit. Every other file in this directory but `modelopt.cu`
 * is a byte copy of the source its kernels came from, and a change to their
 * arithmetic is a defect, not a cleanup - which is why these are a module of
 * their own (`q4e_exact`) rather than edits to `qsa_sparse.cu` and `gdn.cu`.
 * Both are derived from those units: each body below is the ported kernel's,
 * changed only where its comment says.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * Tensor-core attention with exact weights.
 *
 * `q4e_qsa_attention_tc_kernel` (qsa_sparse.cu) with one change: the softmax
 * weight p of each (row, key) goes into the P V product as two fp16 halves,
 *
 *     hi = fp16(p),  lo = fp16(p - hi),
 *
 * and every P V fragment is two MMAs on the same V fragment, hi first and lo
 * second, in that order on every tile. The ported kernel rounds p to fp16
 * once, so the numerator carries a relative error of up to 2^-11 a weight
 * that the denominator - which sums the fp32 p - does not share.
 *
 * p is in [0, 1], so neither half leaves fp16's range. p - hi is exact in
 * fp32 (the two are within an fp16 ulp of each other), and lo rounds it to
 * fp16, whose subnormals reach 2^-24: hi + lo is p to within about 2^-24
 * absolute. What is left between the numerator and the fp32 weights is V's
 * own fp16 (the cache's format) and the fp32 accumulation order. Each row's
 * arithmetic still reads only its own keys, so batch invariance is unchanged.
 *
 * Cost against the ported kernel: 48 MMAs a warp a tile instead of 32 (V
 * fragments are loaded once and used twice) and 2 KiB more shared memory,
 * 81,984 bytes, which still leaves the block alone on its SM as before. The
 * launch geometry and the argument list are the ported kernel's.
 * ------------------------------------------------------------------------ */
#define Q4E_EX_TC_KT 64u
#define Q4E_EX_TC_HD 256u
#define Q4E_EX_TC_CH (Q4E_EX_TC_HD / 8u)    /* 16-byte chunks per row */
#define Q4E_EX_TC_SMEM ((16u + 2u * Q4E_EX_TC_KT) * Q4E_EX_TC_HD * 2u + 16u * Q4E_EX_TC_KT * 4u + 2u * 16u * Q4E_EX_TC_KT * 2u + 16u * 4u)
/* The partial slab q4e_qsa_attention_combine_kernel folds: m, l, six pad, then
 * the 256 accumulators (qsa_sparse.cu's Q4E_ATTN_PART_STRIDE). */
#define Q4E_EX_PART_STRIDE (Q4E_EX_TC_HD + 8u)
__device__ __forceinline__ static uint32_t q4e_ex_tc_off(uint32_t row, uint32_t chunk) {
    return (row * Q4E_EX_TC_CH + (chunk ^ (row & 7u))) * 16u;
}
extern "C" __global__ void __launch_bounds__(256)
q4e_qsa_attention_tc_exact_kernel(
        float *out, float *part, const __half *k_cache, const __half *v_cache, const float *q,
        const int32_t *tokens, const int32_t *n_sel, const int32_t *pos, const int32_t *pages,
        uint32_t width, uint32_t n_head, uint32_t n_head_kv, uint32_t n_splits) {
    constexpr uint32_t KT = Q4E_EX_TC_KT, HD = Q4E_EX_TC_HD, CH = Q4E_EX_TC_CH;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t t = blockIdx.z, hkv = blockIdx.y, split = blockIdx.x;
    const uint32_t G = n_head / n_head_kv;
    const float scale = rsqrtf((float)HD);

    extern __shared__ __align__(16) uint8_t q4e_ex_tc_smem[];
    uint8_t *s_q = q4e_ex_tc_smem;                       /* [16][256] f16 */
    uint8_t *s_k = s_q + 16u * HD * 2u;                  /* [64][256] f16 */
    uint8_t *s_v = s_k + KT * HD * 2u;                   /* [64][256] f16 */
    float   *s_s = (float *)(s_v + KT * HD * 2u);        /* [16][64] f32 scores */
    __half  *s_p = (__half *)(s_s + 16u * KT);           /* [16][64] f16, chunk-swizzled */
    __half  *s_p_lo = s_p + 16u * KT;                    /* [16][64] f16, same swizzle */
    float   *s_corr = (float *)(s_p_lo + 16u * KT);      /* [16] */
    __shared__ int32_t s_idx[Q4E_EX_TC_KT];
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
        *(uint4 *)(s_q + q4e_ex_tc_off(r, c)) = pk.u;
    }
    float m_r[2] = { -INFINITY, -INFINITY }, l_r[2] = { 0.0f, 0.0f };
    float o[4][4];
#pragma unroll
    for (int f = 0; f < 4; f++) { o[f][0] = o[f][1] = o[f][2] = o[f][3] = 0.0f; }

    /* The next tile's K and V rows are fetched into registers (eight 16-byte
     * pieces of each per thread) before the current tile is computed, so the
     * gather's latency overlaps the MMAs instead of serializing with them; a
     * block holds 80 KB of shared memory, so it is alone on its SM and has
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
            *(uint4 *)(s_k + q4e_ex_tc_off(key, c)) = pk[j];
            *(uint4 *)(s_v + q4e_ex_tc_off(key, c)) = pv[j];
        }
        __syncthreads();
        if (p0 + KT < l_end) fetch(p0 + KT);

        /* S[16][8w..8w+8) = Q K^T over the 256 dims. */
        float sacc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 4
        for (uint32_t ks = 0; ks < HD / 16u; ks++) {
            uint32_t a[4], b[2];
            tt_ldmatrix_x4_addr(a, tt_smem_addr(s_q + q4e_ex_tc_off(lane & 15u, ks * 2u + (lane >> 4u))));
            tt_ldmatrix_x2_addr(b, tt_smem_addr(s_k + q4e_ex_tc_off(warp * 8u + (lane & 7u),
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
            {
                const uint32_t i0 = r * KT + ((lane >> 3u) ^ (r & 7u)) * 8u + (lane & 7u);
                const uint32_t i1 = r * KT + (((lane + 32u) >> 3u) ^ (r & 7u)) * 8u + (lane & 7u);
                const __half h0 = __float2half_rn(p0f), h1 = __float2half_rn(p1f);
                s_p[i0] = h0;
                s_p[i1] = h1;
                s_p_lo[i0] = __float2half_rn(p0f - __half2float(h0));
                s_p_lo[i1] = __float2half_rn(p1f - __half2float(h1));
            }
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
            uint32_t a[4], a_lo[4];
            {
                const uint32_t r = lane & 15u, c = ks * 2u + (lane >> 4u);
                tt_ldmatrix_x4_addr(a, tt_smem_addr(s_p + r * KT + (c ^ (r & 7u)) * 8u));
                tt_ldmatrix_x4_addr(a_lo, tt_smem_addr(s_p_lo + r * KT + (c ^ (r & 7u)) * 8u));
            }
            const uint32_t key = ks * 16u + (lane & 7u) + 8u * ((lane >> 3u) & 1u);
#pragma unroll
            for (uint32_t f = 0; f < 4; f++) {
                uint32_t b[2];
                tt_ldmatrix_x2_trans_addr(b, tt_smem_addr(s_v + q4e_ex_tc_off(key, warp * 4u + f)));
                tt_mma_m16n8k16_f16_f32(o[f], a, b);
                tt_mma_m16n8k16_f16_f32(o[f], a_lo, b);
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
            float *prow = part + (qg * n_splits + split) * Q4E_EX_PART_STRIDE;
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

/* ---------------------------------------------------------------------------
 * One decode row of a gated recurrent layer, after its convolution.
 *
 * A single-session decode row of a recurrent layer ran six launches between
 * its projections: the convolution, two L2 norms, the gates, the recurrence
 * and the output norm and gate. This kernel is the last five in one launch,
 * one block per value head, and gives the same bits as the five do:
 *
 *   1. the L2 norms of this head's q and k (q4e_gdn_l2norm_kernel),
 *   2. its decay and write strength (q4e_gdn_gates_kernel),
 *   3. the delta rule over its state (q4e_gdn_recurrent_run at n_tok = 1),
 *   4. the RMS norm and sigmoid gate of the result (q4e_gdn_out_gate_kernel).
 *
 * Each phase is the standalone kernel's expressions in its order, over the
 * same values: the recurrence keeps its 512-thread (j, sub) layout and its
 * shared state tile, and the three 128-wide reductions keep the 128-thread
 * tree the standalone kernels fold through (see q4e_ex_sum128). Nothing the
 * five wrote and nothing else read - the normalized q and k, the decay and
 * beta, the unnormalized attention row - is written.
 *
 * The convolution is NOT fused, and that is load-bearing. Value heads h,
 * h + n_head_k, ... share query/key head h % n_head_k, so a block per value
 * head would convolve that head's q and k channels in several blocks while
 * one of them rewrites their window: the others could read the window before
 * or after the rewrite, depending on which block the device ran first. Blocks
 * of one launch have no order, so the convolution stays its own launch and
 * this one reads its output.
 *
 * A one-row pass records no rollback state (the recurrence records only when
 * n_tok > 1), so this kernel takes no checkpoint slab; the host never plans it
 * for a verify window. The host also plans it only at head_dim 128, which the
 * shared rows below are sized for - it does not check.
 *
 * Launch: grid n_head_v, block 4 * head_dim = 512, dynamic shared memory
 * head_dim * (head_dim + 1) * 4 = 66,048 bytes (the recurrence's tile).
 * ------------------------------------------------------------------------ */
#define Q4E_EX_GDN_SPLIT 4u   /* lanes per state column, gdn.cu's Q4E_GDN_SPLIT */
#define Q4E_EX_GDN_D 128u     /* the head width the shared rows are sized for */

/* q4e_block_sum as a 128-thread block runs it, inside a wider block: the
 * first four warps fold their lanes, then warp 0 folds the four partials with
 * the remaining lanes zero. That is the tree q4e_gdn_l2norm_kernel and
 * q4e_gdn_out_gate_kernel reduce through, so the sum is the same bits. (The
 * whole-block q4e_block_sum with zero terms past thread 128 happens to give
 * them too - its extra partials are exact zeros in the same shuffle positions
 * - but this states the tree rather than relying on that.) Every thread of
 * the block calls it; `red` holds five floats and is not reused by a later
 * call. */
__device__ __forceinline__ static float q4e_ex_sum128(float v, float *red) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    if (warp < 4u) {
        v = warp_sum_f32(v);
        if (lane == 0u) red[warp] = v;
    }
    __syncthreads();
    if (warp == 0u) {
        v = (lane < 4u) ? red[lane] : 0.0f;
        v = warp_sum_f32(v);
        if (lane == 0u) red[4] = v;
    }
    __syncthreads();
    return red[4];
}

extern "C" __global__ void __launch_bounds__(512) q4e_gdn_step_kernel(
        float *gated_out, float *state, const float *conv,
        const float *alpha_proj, const float *beta_proj,
        const float *dt_bias, const float *a, const float *z, const float *norm_w,
        uint32_t head_dim, uint32_t n_head_k, uint32_t n_head_v,
        uint32_t k_offset, uint32_t v_offset, float eps) {
    extern __shared__ float q4e_ex_gdn_smem[];   /* head_dim * (head_dim + 1) */
    float *s_state = q4e_ex_gdn_smem;
    __shared__ float s_q[Q4E_EX_GDN_D], s_k[Q4E_EX_GDN_D], s_attn[Q4E_EX_GDN_D];
    __shared__ float s_red_q[5], s_red_k[5], s_red_o[5];
    __shared__ float s_dec, s_bt, s_kq;
    const uint32_t tid = threadIdx.x;
    const uint32_t h = blockIdx.x;
    const uint32_t hk = h % n_head_k;
    const float *q_d = conv + (uint64_t)hk * head_dim;
    const float *k_d = conv + k_offset + (uint64_t)hk * head_dim;
    const float *v_d = conv + v_offset + (uint64_t)h * head_dim;
    (void)n_head_v;

    /* 1. L2 norms: q4e_gdn_l2norm_kernel's body over the first 128 threads,
     * once at the query offset and once at the key offset. */
    float sum_q = 0.0f, sum_k = 0.0f;
    if (tid < 128u) {
        for (uint32_t i = tid; i < head_dim; i += 128u) sum_q += q_d[i] * q_d[i];
        for (uint32_t i = tid; i < head_dim; i += 128u) sum_k += k_d[i] * k_d[i];
    }
    const float inv_q = rsqrtf(q4e_ex_sum128(sum_q, s_red_q) + 1e-12f);
    const float inv_k = rsqrtf(q4e_ex_sum128(sum_k, s_red_k) + 1e-12f);
    if (tid < 128u) {
        for (uint32_t i = tid; i < head_dim; i += 128u) {
            s_q[i] = q_d[i] * inv_q;
            s_k[i] = k_d[i] * inv_k;
        }
    }

    /* 2. Gates: q4e_gdn_gates_kernel's expressions for (row 0, head h), and
     * the recurrence's exponential of the decay. */
    if (tid == 0u) {
        const float x = alpha_proj[h] + dt_bias[h];
        const float sp = (x > 20.0f) ? x : log1pf(__expf(x));
        const float decay = sp * a[h];
        s_dec = __expf(decay);
        s_bt = 1.0f / (1.0f + __expf(-beta_proj[h]));
    }

    /* 3. The delta rule: q4e_gdn_recurrent_run at n_tok = 1, q and k read
     * from shared memory instead of from the conv row. */
    const uint32_t j = tid / Q4E_EX_GDN_SPLIT;
    const uint32_t sub = tid % Q4E_EX_GDN_SPLIT;
    const uint32_t seg = head_dim / Q4E_EX_GDN_SPLIT;          /* 32 */
    const float scale = rsqrtf((float)head_dim);
    const uint32_t row_stride = head_dim + 1u;
    float *m = state + (uint64_t)h * head_dim * head_dim;
    for (uint32_t idx = tid; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        s_state[r * row_stride + c] = m[idx];
    }
    __syncthreads();

    float kq_part = 0.0f;
    if (sub == 0u && j < head_dim) kq_part = s_k[j] * s_q[j];
    kq_part = q4e_block_sum(kq_part);
    if (tid == 0u) s_kq = kq_part;
    __syncthreads();

    const float dec = s_dec;
    const float bt = s_bt;
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

    if (sub == 0u) s_attn[j] = (dec * sq + delta * s_kq) * scale;
    for (uint32_t n = 0; n < seg; n++) {
        const uint32_t i = i0 + ((n + 8u * sub) & (seg - 1u));
        row[i] = fmaf(row[i], dec, delta * s_k[i]);
    }
    __syncthreads();

    for (uint32_t idx = tid; idx < head_dim * head_dim; idx += blockDim.x) {
        const uint32_t r = idx / head_dim, c = idx % head_dim;
        m[idx] = s_state[r * row_stride + c];
    }

    /* 4. q4e_gdn_out_gate_kernel's body over the first 128 threads. */
    const float *zr = z + (uint64_t)h * head_dim;
    float *dst = gated_out + (uint64_t)h * head_dim;
    float sum = 0.0f;
    if (tid < 128u) {
        for (uint32_t i = tid; i < head_dim; i += 128u) sum += s_attn[i] * s_attn[i];
    }
    const float oscale = rsqrtf(q4e_ex_sum128(sum, s_red_o) / (float)head_dim + eps);
    if (tid < 128u) {
        for (uint32_t i = tid; i < head_dim; i += 128u) {
            const float g = 1.0f / (1.0f + __expf(-zr[i]));
            dst[i] = s_attn[i] * oscale * norm_w[i] * g;
        }
    }
}
