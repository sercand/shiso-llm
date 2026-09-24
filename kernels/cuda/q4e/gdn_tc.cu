/*
 * The gated delta rule for a prefill chunk, on tensor cores.
 *
 * shiso's own, not a copy: the unit beside it (`gdn.cu`) carries the upstream
 * kernels, and this one replaces its chunked prefill kernel with the same
 * arithmetic arranged so that almost all of it is matrix multiplies.
 *
 * `q4e_gdn_chunk_kernel` gives each value head one block and walks the runs of
 * a chunk one after the other, on CUDA cores: 48 blocks for 48 heads, 17 ms a
 * layer at an 8192-row chunk on a GB10, a sixth of a prefill.  Almost none of
 * its work depends on the run before: with G_t the cumulative log decay inside
 * a run of C tokens and S_0 the state at its start (keys by values),
 *
 *     A[t][u] = b_t e^{G_t - G_u} (k_u . k_t)          u < t
 *     T       = (I + A)^-1                            unit lower triangular
 *     u       = T (b o V)                             C x D
 *     w       = T (b e^G o K)                         C x D
 *     d       = u - w S_0                             the deltas
 *     o_t     = s e^{G_t} q_t S_0 + sum_{u<=t} P[t][u] d_u,
 *               P[t][u] = s e^{G_t - G_u} (q_t . k_u)
 *     S_next  = e^{G_last} S_0 + sum_t e^{G_last - G_t} k_t d_t^T
 *
 * with s = 1 / sqrt(D).  Everything but the last three lines is independent
 * of S_0, so `q4e_gdn_tc_prep_kernel` computes it for every (run, head) at
 * once, and `q4e_gdn_tc_scan_kernel` carries the state: one block per head,
 * its 128 x 128 state held in the warps' accumulator registers for the whole
 * chunk, writing each run's outputs as it goes - so no per-run state is ever
 * stored.
 *
 * Numerics: the multiplies take half-precision operands and accumulate in
 * single precision; the gates, the triangular solve and the carried state are
 * single precision throughout.  Every operand here is small - unit keys and
 * queries, gates in (0, 1], deltas of the order of the values - so the half
 * operand costs what a TF32 one would and runs at twice the rate.  These are
 * different sums from the CUDA-core kernel, so this is checked against it
 * rather than expected to match it bit for bit.  It writes no per-position
 * state, so the launcher keeps runs that a speculative rollback could reach on
 * the sequential kernel, exactly as it does for the chunked one.
 *
 * Every tile a multiply reads is zero past the run's last token: a partial run
 * is a full run whose tail contributes nothing, so no multiply needs a bound.
 *
 * The float-to-half copies between accumulator fragments of one shape rely on
 * the two sharing an element order, which every architecture these kernels
 * are built for does; the comparison against the CUDA-core kernel is what
 * would catch one that did not.
 */

#include "q4e.cuh"
#include <mma.h>

using namespace nvcuda;

#define Q4E_GT_C 64u      /* tokens per run */
#define Q4E_GT_D 128u     /* head width, checked by the launcher */
#define Q4E_GT_LH 136u    /* row stride, in halves, of a [.][D] half tile */
#define Q4E_GT_LP 72u     /* row stride, in halves, of a [C][C] half tile */
#define Q4E_GT_LA 68u     /* row stride, in floats, of a [C][C] float tile */

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q4e_gt_a_row;
typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::col_major> q4e_gt_a_col;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> q4e_gt_b_row;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> q4e_gt_b_col;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> q4e_gt_acc;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, __half> q4e_gt_acc_h;

/* Stores a float accumulator as halves. */
__device__ __forceinline__ static void q4e_gt_store_h(__half *dst, const q4e_gt_acc &acc, uint32_t ld) {
    q4e_gt_acc_h h;
#pragma unroll
    for (int i = 0; i < h.num_elements; i++) h.x[i] = __float2half_rn(acc.x[i]);
    wmma::store_matrix_sync(dst, h, ld, wmma::mem_row_major);
}

/* One [C][D] tile of the fused qkv rows into shared memory as halves, zero
 * past n, each row scaled by `row_scale[t]` when given. */
__device__ __forceinline__ static void q4e_gt_stage(
        __half *dst, const float *qkv, uint32_t t0, uint32_t n, uint32_t stride, uint32_t col,
        const float *row_scale) {
    for (uint32_t idx = threadIdx.x; idx < Q4E_GT_C * (Q4E_GT_D / 4u); idx += blockDim.x) {
        const uint32_t t = idx / (Q4E_GT_D / 4u), i4 = idx % (Q4E_GT_D / 4u);
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (t < n) {
            v = *(const float4 *)(qkv + (uint64_t)(t0 + t) * stride + col + i4 * 4u);
            if (row_scale) {
                const float s = row_scale[t];
                v.x *= s; v.y *= s; v.z *= s; v.w *= s;
            }
        }
        __half2 *o = (__half2 *)(dst + t * Q4E_GT_LH + i4 * 4u);
        o[0] = __floats2half2_rn(v.x, v.y);
        o[1] = __floats2half2_rn(v.z, v.w);
    }
}

/* out[C][C] (float) = X Y^T over D, both [C][D] half tiles. */
__device__ __forceinline__ static void q4e_gt_gram(float *out, const __half *x, const __half *y) {
    const uint32_t warp = threadIdx.x >> 5u;
    for (uint32_t tile = warp; tile < 16u; tile += blockDim.x >> 5u) {
        const uint32_t tr = tile / 4u, tc = tile % 4u;
        q4e_gt_acc acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (uint32_t k = 0; k < Q4E_GT_D; k += 16u) {
            q4e_gt_a_row a;
            q4e_gt_b_col b;
            wmma::load_matrix_sync(a, x + tr * 16u * Q4E_GT_LH + k, Q4E_GT_LH);
            wmma::load_matrix_sync(b, y + tc * 16u * Q4E_GT_LH + k, Q4E_GT_LH);
            wmma::mma_sync(acc, a, b, acc);
        }
        wmma::store_matrix_sync(out + tr * 16u * Q4E_GT_LA + tc * 16u, acc, Q4E_GT_LA, wmma::mem_row_major);
    }
}

/* T X for T [C][C] and X [C][D], both half tiles in shared memory, stored to
 * `out` (row stride D) as floats or as halves. */
template <typename OUT>
__device__ __forceinline__ static void q4e_gt_apply(OUT *out, const __half *t_mat, const __half *x) {
    const uint32_t warp = threadIdx.x >> 5u;
    for (uint32_t tile = warp; tile < 32u; tile += blockDim.x >> 5u) {
        const uint32_t tr = tile / 8u, tc = tile % 8u;
        q4e_gt_acc acc;
        wmma::fill_fragment(acc, 0.0f);
#pragma unroll
        for (uint32_t k = 0; k < Q4E_GT_C; k += 16u) {
            q4e_gt_a_row a;
            q4e_gt_b_row b;
            wmma::load_matrix_sync(a, t_mat + tr * 16u * Q4E_GT_LP + k, Q4E_GT_LP);
            wmma::load_matrix_sync(b, x + k * Q4E_GT_LH + tc * 16u, Q4E_GT_LH);
            wmma::mma_sync(acc, a, b, acc);
        }
        if constexpr (sizeof(OUT) == sizeof(float)) {
            wmma::store_matrix_sync((float *)out + tr * 16u * Q4E_GT_D + tc * 16u, acc, Q4E_GT_D,
                                    wmma::mem_row_major);
        } else {
            q4e_gt_store_h((__half *)out + tr * 16u * Q4E_GT_D + tc * 16u, acc, Q4E_GT_D);
        }
    }
}

/* Everything a run needs that does not depend on the state at its start.
 *
 * Grid (value heads, runs), 512 threads - heads fastest, so the three value
 * heads that share a key head, and every head of one run, are resident
 * together and read the run's keys and queries out of L2 rather than DRAM.
 * The kernel is bound by what it moves, not by what it multiplies.  Writes,
 * per (run, head): u and w [C][D] and P [C][C] (scaled, masked) as halves,
 * and the cumulative log decays [C] the scan rescales by. */
extern "C" __global__ void __launch_bounds__(512) q4e_gdn_tc_prep_kernel(
        __half *u_out, __half *w_out, __half *p_out, float *g_out,
        const float *qkv, const float *decay, const float *beta,
        uint32_t n_head_k, uint32_t n_head_v,
        uint32_t k_offset, uint32_t v_offset, uint32_t stride, uint32_t n_tok) {
    extern __shared__ __align__(32) unsigned char q4e_gt_prep_smem[];
    /* Shared memory is what bounds how many of these a multiprocessor holds,
     * so T takes the keys' place once both grams have read them. */
    __half *s_K = (__half *)q4e_gt_prep_smem;                        /* [C][LH] keys, then T [C][LP] */
    __half *s_X = s_K + Q4E_GT_C * Q4E_GT_LH;                        /* [C][LH] queries, then the right-hand sides */
    __half *s_T = s_K;
    float *s_A = (float *)(s_X + Q4E_GT_C * Q4E_GT_LH);              /* [C][LA] grams, then the solve */
    float *s_G = s_A + Q4E_GT_C * Q4E_GT_LA;                         /* [C] cumulative log decay */
    float *s_b = s_G + Q4E_GT_C;                                     /* [C] beta */
    float *s_e = s_b + Q4E_GT_C;                                     /* [C] row scales */

    const uint32_t h = blockIdx.x, run = blockIdx.y, hk = h % n_head_k;
    const uint32_t t0 = run * Q4E_GT_C;
    const uint32_t n = (n_tok - t0 < Q4E_GT_C) ? (n_tok - t0) : Q4E_GT_C;
    const uint64_t slot = (uint64_t)run * n_head_v + h;
    const uint32_t tid = threadIdx.x;
    const float scale = rsqrtf((float)Q4E_GT_D);

    if (tid < Q4E_GT_C) {
        float g = 0.0f, b = 0.0f;
        if (tid < n) {
            const uint64_t gi = (uint64_t)(t0 + tid) * n_head_v + h;
            g = decay[gi];
            b = beta[gi];
        }
        s_G[tid] = g;
        s_b[tid] = b;
    }
    __syncthreads();
    /* Inclusive prefix sum by one warp: 64 values, a shuffle scan per half. */
    if (tid < 32u) {
        float lo = s_G[tid], hi = s_G[tid + 32u];
#pragma unroll
        for (uint32_t o = 1u; o < 32u; o <<= 1) {
            const float a = __shfl_up_sync(0xffffffffu, lo, o);
            const float b = __shfl_up_sync(0xffffffffu, hi, o);
            if (tid >= o) { lo += a; hi += b; }
        }
        hi += __shfl_sync(0xffffffffu, lo, 31);
        s_G[tid] = lo;
        s_G[tid + 32u] = hi;
    }
    q4e_gt_stage(s_K, qkv, t0, n, stride, k_offset + hk * Q4E_GT_D, NULL);
    q4e_gt_stage(s_X, qkv, t0, n, stride, hk * Q4E_GT_D, NULL);
    __syncthreads();
    if (tid < Q4E_GT_C) g_out[slot * Q4E_GT_C + tid] = s_G[tid];

    /* P = s e^{G_t - G_u} Q K^T on and below the diagonal. */
    q4e_gt_gram(s_A, s_X, s_K);
    __syncthreads();
    for (uint32_t idx = tid; idx < Q4E_GT_C * Q4E_GT_C; idx += blockDim.x) {
        const uint32_t t = idx / Q4E_GT_C, u = idx % Q4E_GT_C;
        const float v = (u <= t && t < n) ? scale * __expf(s_G[t] - s_G[u]) * s_A[t * Q4E_GT_LA + u] : 0.0f;
        p_out[slot * Q4E_GT_C * Q4E_GT_C + idx] = __float2half_rn(v);
    }
    __syncthreads();

    /* A = b_t e^{G_t - G_u} K K^T strictly below the diagonal. */
    q4e_gt_gram(s_A, s_K, s_K);
    __syncthreads();
    for (uint32_t idx = tid; idx < Q4E_GT_C * Q4E_GT_C; idx += blockDim.x) {
        const uint32_t t = idx / Q4E_GT_C, u = idx % Q4E_GT_C;
        const float a = s_A[t * Q4E_GT_LA + u];
        s_A[t * Q4E_GT_LA + u] = (u < t && t < n) ? s_b[t] * __expf(s_G[t] - s_G[u]) * a : 0.0f;
    }
    __syncthreads();

    /* T = (I + A)^-1, in place, by 16 x 16 blocks.  Row-at-a-time forward
     * substitution costs two block barriers per row, 126 a run, and those
     * barriers were most of this kernel's time.  Blocked, the four diagonal
     * blocks are inverted by one warp each with warp-local ordering only,
     * and the off-diagonal blocks then follow one block row at a time:
     *
     *     T_ij = -T_ii sum_{j<=k<i} A_ik T_kj                       (i > j)
     *
     * which is two barriers per block row.  The queries are no longer needed,
     * so their tile holds the sums. */
    {
        const uint32_t lane = tid & 31u, warp = tid >> 5u;
        if (warp < 4u) {
            /* T_bb for the unit lower triangular diagonal block b: lane c owns
             * column c, and row r only reads rows above it, so the rows are
             * ordered by the warp alone. */
            float *blk = s_A + warp * 16u * Q4E_GT_LA + warp * 16u;
            const uint32_t c = lane;
            for (uint32_t r = 1u; r < 16u; r++) {
                float v = 0.0f;
                if (c < r) {
                    v = -blk[r * Q4E_GT_LA + c];
                    for (uint32_t u = c + 1u; u < r; u++) v = fmaf(-blk[r * Q4E_GT_LA + u], blk[u * Q4E_GT_LA + c], v);
                }
                __syncwarp();
                if (c < r) blk[r * Q4E_GT_LA + c] = v;
                __syncwarp();
            }
        }
        __syncthreads();
        float *s_Y = (float *)s_X;                    /* [3][16][16] the sums of one block row */
        for (uint32_t bi = 1u; bi < 4u; bi++) {
            /* Y_j = sum_{j<=k<i} A_ik T_kj for every j < i; T_jj's diagonal
             * is the implied 1. */
            for (uint32_t idx = tid; idx < bi * 256u; idx += blockDim.x) {
                const uint32_t bj = idx / 256u, r = (idx % 256u) / 16u, cc = idx % 16u;
                const uint32_t row = bi * 16u + r, col = bj * 16u + cc;
                float y = 0.0f;
                for (uint32_t k = bj * 16u; k < bi * 16u; k++) {
                    const float t_kc = (k == col) ? 1.0f : (k > col ? s_A[k * Q4E_GT_LA + col] : 0.0f);
                    y = fmaf(s_A[row * Q4E_GT_LA + k], t_kc, y);
                }
                s_Y[idx] = y;
            }
            __syncthreads();
            /* T_ij = -T_ii Y_j. */
            for (uint32_t idx = tid; idx < bi * 256u; idx += blockDim.x) {
                const uint32_t bj = idx / 256u, r = (idx % 256u) / 16u, cc = idx % 16u;
                const uint32_t row = bi * 16u + r;
                float v = -s_Y[bj * 256u + r * 16u + cc];
                for (uint32_t u = 0; u < r; u++) {
                    v = fmaf(-s_A[row * Q4E_GT_LA + bi * 16u + u], s_Y[bj * 256u + u * 16u + cc], v);
                }
                s_A[row * Q4E_GT_LA + bj * 16u + cc] = v;
            }
            __syncthreads();
        }
    }
    for (uint32_t idx = tid; idx < Q4E_GT_C * Q4E_GT_C; idx += blockDim.x) {
        const uint32_t t = idx / Q4E_GT_C, u = idx % Q4E_GT_C;
        s_T[t * Q4E_GT_LP + u] = __float2half_rn(t == u ? 1.0f : s_A[t * Q4E_GT_LA + u]);
    }

    /* u = T (b o V). */
    q4e_gt_stage(s_X, qkv, t0, n, stride, v_offset + h * Q4E_GT_D, s_b);
    __syncthreads();
    q4e_gt_apply(u_out + slot * Q4E_GT_C * Q4E_GT_D, s_T, s_X);
    if (tid < Q4E_GT_C) s_e[tid] = s_b[tid] * __expf(s_G[tid]);
    __syncthreads();

    /* w = T (b e^G o K). */
    q4e_gt_stage(s_X, qkv, t0, n, stride, k_offset + hk * Q4E_GT_D, s_e);
    __syncthreads();
    q4e_gt_apply(w_out + slot * Q4E_GT_C * Q4E_GT_D, s_T, s_X);
}

/* The run-to-run carry, and every run's outputs.
 *
 * Grid (value heads), 512 threads.  The head's state [D keys][D values] is
 * held in the sixteen warps' accumulators - four 16 x 16 tiles each - for the
 * whole chunk, and copied to shared memory as halves once per run for the two
 * products that read it.
 *
 * `snap`, when not null, receives the state as it stands before run
 * `snap_run`, laid out exactly as `state` is: the checkpoint a later prompt
 * resumes from when this chunk ends between two runs, taken on the run grid
 * where a cold prefill of that longer prompt would pass through the same bits.
 * No run's arithmetic reads it. */
extern "C" __global__ void __launch_bounds__(512) q4e_gdn_tc_scan_kernel(
        float *attn_out, float *state,
        const float *qkv, const __half *u_in, const __half *w_in, const __half *p_in, const float *g_in,
        uint32_t n_head_k, uint32_t n_head_v, uint32_t k_offset, uint32_t stride, uint32_t n_tok,
        float *snap, uint32_t snap_run) {
    extern __shared__ __align__(32) unsigned char q4e_gt_scan_smem[];
    __half *s_S = (__half *)q4e_gt_scan_smem;                        /* [D][LH] the state, as halves */
    __half *s_X = s_S + Q4E_GT_D * Q4E_GT_LH;                        /* [C][LH] scaled q, then scaled k */
    __half *s_D = s_X + Q4E_GT_C * Q4E_GT_LH;                        /* [C][LH] deltas */
    float *s_G = (float *)(s_D + Q4E_GT_C * Q4E_GT_LH);              /* [C] cumulative log decay */
    float *s_e = s_G + Q4E_GT_C;                                     /* [C] row scales */
    /* A partial run's outputs are staged over the half state, which nothing
     * reads between the outputs and the next run's copy. */
    float *s_O = (float *)s_S;

    const uint32_t h = blockIdx.x, hk = h % n_head_k;
    const uint32_t tid = threadIdx.x, warp = tid >> 5u;
    const float scale = rsqrtf((float)Q4E_GT_D);
    const uint64_t out_ld = (uint64_t)n_head_v * Q4E_GT_D;
    float *state_h = state + (uint64_t)h * Q4E_GT_D * Q4E_GT_D;

    /* The state tiles this warp owns: rows (keys) sr, columns (values) 4
     * consecutive tiles.  The stored state is value-major, [value][key], which
     * read column-major is exactly keys by values. */
    const uint32_t sr = warp / 2u, sc0 = (warp % 2u) * 4u;
    q4e_gt_acc s_acc[4];
#pragma unroll
    for (uint32_t q = 0; q < 4u; q++) {
        wmma::load_matrix_sync(s_acc[q], state_h + (sc0 + q) * 16u * Q4E_GT_D + sr * 16u, Q4E_GT_D,
                               wmma::mem_col_major);
    }
    /* The [C][D] product tiles this warp computes: row tile pr, two columns. */
    const uint32_t pr = warp / 4u, pc0 = (warp % 4u) * 2u;

    for (uint32_t t0 = 0; t0 < n_tok; t0 += Q4E_GT_C) {
        const uint32_t run = t0 / Q4E_GT_C;
        const uint32_t n = (n_tok - t0 < Q4E_GT_C) ? (n_tok - t0) : Q4E_GT_C;
        const uint64_t slot = (uint64_t)run * n_head_v + h;
        const __half *w_run = w_in + slot * Q4E_GT_C * Q4E_GT_D;
        const __half *p_run = p_in + slot * Q4E_GT_C * Q4E_GT_C;
        const __half *u_run = u_in + slot * Q4E_GT_C * Q4E_GT_D;

        if (snap && run == snap_run) {
            float *snap_h = snap + (uint64_t)h * Q4E_GT_D * Q4E_GT_D;
#pragma unroll
            for (uint32_t q = 0; q < 4u; q++) {
                wmma::store_matrix_sync(snap_h + (sc0 + q) * 16u * Q4E_GT_D + sr * 16u, s_acc[q], Q4E_GT_D,
                                        wmma::mem_col_major);
            }
        }
#pragma unroll
        for (uint32_t q = 0; q < 4u; q++) {
            q4e_gt_store_h(s_S + sr * 16u * Q4E_GT_LH + (sc0 + q) * 16u, s_acc[q], Q4E_GT_LH);
        }
        if (tid < Q4E_GT_C) {
            const float g = g_in[slot * Q4E_GT_C + tid];
            s_G[tid] = g;
            s_e[tid] = scale * __expf(g);
        }
        __syncthreads();

        /* d = u - w S, and the scaled queries staged behind it. */
        q4e_gt_stage(s_X, qkv, t0, n, stride, hk * Q4E_GT_D, s_e);
#pragma unroll
        for (uint32_t q = 0; q < 2u; q++) {
            const uint32_t pc = pc0 + q;
            q4e_gt_acc acc;
            {
                q4e_gt_acc_h u_h;
                wmma::load_matrix_sync(u_h, u_run + pr * 16u * Q4E_GT_D + pc * 16u, Q4E_GT_D, wmma::mem_row_major);
#pragma unroll
                for (int i = 0; i < acc.num_elements; i++) acc.x[i] = __half2float(u_h.x[i]);
            }
#pragma unroll
            for (uint32_t k = 0; k < Q4E_GT_D; k += 16u) {
                q4e_gt_a_row a;
                q4e_gt_b_row b;
                wmma::load_matrix_sync(a, w_run + pr * 16u * Q4E_GT_D + k, Q4E_GT_D);
                wmma::load_matrix_sync(b, s_S + k * Q4E_GT_LH + pc * 16u, Q4E_GT_LH);
#pragma unroll
                for (int i = 0; i < a.num_elements; i++) a.x[i] = __hneg(a.x[i]);
                wmma::mma_sync(acc, a, b, acc);
            }
            q4e_gt_store_h(s_D + pr * 16u * Q4E_GT_LH + pc * 16u, acc, Q4E_GT_LH);
        }
        __syncthreads();

        /* o = (s e^G o Q) S + P d. */
        q4e_gt_acc o_acc[2];
#pragma unroll
        for (uint32_t q = 0; q < 2u; q++) {
            const uint32_t pc = pc0 + q;
            wmma::fill_fragment(o_acc[q], 0.0f);
#pragma unroll
            for (uint32_t k = 0; k < Q4E_GT_D; k += 16u) {
                q4e_gt_a_row a;
                q4e_gt_b_row b;
                wmma::load_matrix_sync(a, s_X + pr * 16u * Q4E_GT_LH + k, Q4E_GT_LH);
                wmma::load_matrix_sync(b, s_S + k * Q4E_GT_LH + pc * 16u, Q4E_GT_LH);
                wmma::mma_sync(o_acc[q], a, b, o_acc[q]);
            }
#pragma unroll
            for (uint32_t k = 0; k < Q4E_GT_C; k += 16u) {
                q4e_gt_a_row a;
                q4e_gt_b_row b;
                wmma::load_matrix_sync(a, p_run + pr * 16u * Q4E_GT_C + k, Q4E_GT_C);
                wmma::load_matrix_sync(b, s_D + k * Q4E_GT_LH + pc * 16u, Q4E_GT_LH);
                wmma::mma_sync(o_acc[q], a, b, o_acc[q]);
            }
        }
        const float g_last = s_G[n - 1u];
        __syncthreads();   /* every read of s_S and s_X above is done */

        if (n == Q4E_GT_C) {
#pragma unroll
            for (uint32_t q = 0; q < 2u; q++) {
                wmma::store_matrix_sync(attn_out + ((uint64_t)(t0 + pr * 16u) * n_head_v + h) * Q4E_GT_D +
                                            (pc0 + q) * 16u,
                                        o_acc[q], (uint32_t)out_ld, wmma::mem_row_major);
            }
        } else {
#pragma unroll
            for (uint32_t q = 0; q < 2u; q++) {
                wmma::store_matrix_sync(s_O + pr * 16u * Q4E_GT_D + (pc0 + q) * 16u, o_acc[q], Q4E_GT_D,
                                        wmma::mem_row_major);
            }
        }
        if (tid < Q4E_GT_C) s_e[tid] = (tid < n) ? __expf(g_last - s_G[tid]) : 0.0f;
        __syncthreads();
        if (n != Q4E_GT_C) {
            for (uint32_t idx = tid; idx < n * Q4E_GT_D; idx += blockDim.x) {
                const uint32_t t = idx / Q4E_GT_D, j = idx % Q4E_GT_D;
                attn_out[((uint64_t)(t0 + t) * n_head_v + h) * Q4E_GT_D + j] = s_O[idx];
            }
        }

        /* The carry: S <- e^{G_last} S + (e^{G_last - G} o K)^T d. */
        q4e_gt_stage(s_X, qkv, t0, n, stride, k_offset + hk * Q4E_GT_D, s_e);
        __syncthreads();
        const float dec = __expf(g_last);
#pragma unroll
        for (uint32_t q = 0; q < 4u; q++) {
#pragma unroll
            for (int i = 0; i < s_acc[q].num_elements; i++) s_acc[q].x[i] *= dec;
#pragma unroll
            for (uint32_t k = 0; k < Q4E_GT_C; k += 16u) {
                q4e_gt_a_col a;
                q4e_gt_b_row b;
                /* K'^T[i][t] = K'[t][i]: the staged rows read column-major. */
                wmma::load_matrix_sync(a, s_X + k * Q4E_GT_LH + sr * 16u, Q4E_GT_LH);
                wmma::load_matrix_sync(b, s_D + k * Q4E_GT_LH + (sc0 + q) * 16u, Q4E_GT_LH);
                wmma::mma_sync(s_acc[q], a, b, s_acc[q]);
            }
        }
        __syncthreads();   /* s_X, s_D and the staged outputs are rewritten next run */
    }

#pragma unroll
    for (uint32_t q = 0; q < 4u; q++) {
        wmma::store_matrix_sync(state_h + (sc0 + q) * 16u * Q4E_GT_D + sr * 16u, s_acc[q], Q4E_GT_D,
                                wmma::mem_col_major);
    }
}

/* The causal convolution of a prefill chunk, parallel over tokens.
 *
 * `q4e_gdn_conv_kernel` gives each channel one thread and walks every token
 * of the chunk in order: 40 blocks, and a dependent chain 8192 tokens long.
 * The convolution has no recurrence - each output reads the last few inputs,
 * not the last output - so here a block covers 256 channels of 64 tokens and
 * reads its window's history from the inputs before it, or from the carried
 * state for the first tile.  The state is rewritten by
 * `q4e_gdn_conv_tail_kernel`, launched after, because the first tile reads it
 * and no block of one launch may write what another reads. */
#define Q4E_GC_TT 64u

extern "C" __global__ void __launch_bounds__(256) q4e_gdn_conv_par_kernel(
        float *out, const float *state, const float *x, const float *w,
        uint32_t channels, uint32_t kernel, uint32_t n_tok) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const uint32_t hist = kernel - 1u;
    const uint32_t t0 = blockIdx.y * Q4E_GC_TT;
    const uint32_t t1 = (t0 + Q4E_GC_TT < n_tok) ? t0 + Q4E_GC_TT : n_tok;
    float win[8];
    for (uint32_t i = 0; i < hist; i++) {
        const int64_t p = (int64_t)t0 - (int64_t)hist + (int64_t)i;
        win[i] = (p >= 0) ? x[(uint64_t)p * channels + c] : state[(uint64_t)((int64_t)hist + p) * channels + c];
    }
    const float *wc = w + (uint64_t)c * kernel;
    for (uint32_t t = t0; t < t1; t++) {
        const float cur = x[(uint64_t)t * channels + c];
        float acc = 0.0f;
        for (uint32_t k = 0; k < hist; k++) acc += wc[k] * win[k];
        acc += wc[hist] * cur;
        out[(uint64_t)t * channels + c] = acc / (1.0f + __expf(-acc));
        for (uint32_t k = 0; k + 1u < hist; k++) win[k] = win[k + 1];
        if (hist > 0) win[hist - 1] = cur;
    }
}

/* The carried window after the chunk: its last `kernel - 1` inputs, reaching
 * back into the previous state when the chunk is shorter than that. */
extern "C" __global__ void __launch_bounds__(256) q4e_gdn_conv_tail_kernel(
        float *state, const float *x, uint32_t channels, uint32_t kernel, uint32_t n_tok) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const uint32_t hist = kernel - 1u;
    /* Ascending, so a read from the old state (index hist + p > i) always
     * precedes the write that would replace it. */
    for (uint32_t i = 0; i < hist; i++) {
        const int64_t p = (int64_t)n_tok - (int64_t)hist + (int64_t)i;
        state[(uint64_t)i * channels + c] =
            (p >= 0) ? x[(uint64_t)p * channels + c] : state[(uint64_t)((int64_t)hist + p) * channels + c];
    }
}
