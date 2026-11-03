/*
 * The gated delta rule's prompt path, fused: two kernels where the ported chain
 * takes seven.
 *
 * shiso's own.  Over a prefill chunk the ported chain runs the parallel
 * convolution (`q4e_gdn_conv_par_kernel`), two in-place L2 norms, the gates,
 * the tensor-core prep and scan (`gdn_tc.cu`), and the output gate - each a
 * full pass over a [T, 10240] or [T, 6144] f32 activation, every one of them
 * already near the device's bandwidth.  So the path gets faster only by moving
 * fewer bytes, and the bytes that can go are the intermediates:
 *
 *   - `q4e_gdf_prep_kernel` reads the projection's raw q|k|v rows and does the
 *     convolution, the silu, the two norms and the gates itself, for one key
 *     head and the three value heads that read it.  The convolved rows never
 *     reach memory except the normalized queries and keys, once per key head,
 *     in f32.  One block per (key head, run) rather than per (value head,
 *     run): the three value heads share both of the key head's grams, so each
 *     gram is computed once, and their three triangular solves run side by
 *     side under one set of barriers.  Per (run, value head) it writes u, T,
 *     P and the decays - T instead of the ported w, because the scan forms w
 *     from T and the key rows it reads anyway, and T is a third of w's size.
 *
 *   - `q4e_gdf_scan_kernel` is the carry with the output gate folded in.  A
 *     run's outputs are a whole head row per token, which is exactly the span
 *     the per-head RMS norm reduces over, so the block normalizes and gates
 *     its own outputs and writes the gated row; the ungated output is never
 *     stored.  See the section comment above it for how the carry is laid out.
 *
 * Measured on a GB10 at a 4096-row chunk, per layer, against the ported chain
 * (7.0 ms: conv 1.6, norms 0.6, prep 2.0, scan 1.55, gate 1.3): prep 1.55 ms
 * and scan 1.8 ms, 3.3 ms in all.  Both are bound by DRAM traffic, which fell
 * from about 1.5 GB to about 0.68 GB a layer; T and P are stored as their ten
 * lower 16 x 16 blocks (`Q4E_GF_TRI`) for the same reason.
 *
 * Both take a row window (`row0`, `n_tok`) over the chunk, so a host may cut
 * one chunk into several prep/scan pairs sharing one window of per-run
 * scratch; the state goes through memory in f32 between windows, which is
 * exact, so where windows are cut moves no bit.  (Cutting does not pay on a
 * GB10 today: the scan gains from scratch that stays in L2, but a prep of a
 * few runs is one under-filled wave.  The engine launches one window.)
 *
 * Numerics: every value either kernel produces is the ported chain's value,
 * bit for bit.  Each elementwise step is the same expression in the same order
 * (the multiply-add contraction the compiler applies is the same for the same
 * expression), each reduction reproduces the ported kernel's tree - a norm
 * folds its row through four 32-lane shuffle trees and combines the partials
 * as `(s0 + s2) + (s1 + s3)`, which is what the ported 128-thread block sum
 * does - and every product is the same sum of the same half-precision operands
 * in the same order of 16-wide steps.  The comparison against the ported chain
 * (`shiso-kerneltest`'s `gdn_fused` cases) is what holds this; a change that
 * breaks it is a defect even if the result is close.
 *
 * Both are written for the one geometry this family has: a 128-wide head, a
 * four-tap convolution, and three value heads per key head.  The planner only
 * takes this path when the shape is that one.
 */

#include "q4e.cuh"
#include <mma.h>

using namespace nvcuda;

#define Q4E_GF_C 64u      /* tokens per run, the ported kernels' run */
#define Q4E_GF_D 128u     /* head width */
#define Q4E_GF_R 3u       /* value heads per key head */
#define Q4E_GF_HIST 3u    /* convolution history: four taps */
#define Q4E_GF_LH 136u    /* row stride, in halves, of a [.][D] half tile */
#define Q4E_GF_LP 72u     /* row stride, in halves, of a [C][C] half tile */
#define Q4E_GF_LA 68u     /* row stride, in floats, of a [C][C] float tile */

typedef wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q4e_gf_a_row;
typedef wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> q4e_gf_b_col;
typedef wmma::fragment<wmma::accumulator, 16, 16, 16, float> q4e_gf_acc;

/* 16 bytes from global `src` to shared `dst`, asynchronously. */
__device__ __forceinline__ static void q4e_gf_cp16(void *dst, const void *src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"((uint32_t)__cvta_generic_to_shared(dst)),
                 "l"(src));
}
__device__ __forceinline__ static void q4e_gf_cp_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ static void q4e_gf_cp_wait() { asm volatile("cp.async.wait_all;\n" ::); }

/* The 128-thread block sum of the ported norms, for a row one warp holds as
 * four 32-element quarters (`v[j]` is element `lane + 32 j`): each quarter
 * folds through the same shuffle tree a warp of that block runs, and the four
 * partials then combine the way that block's second fold combines its four
 * warps (lanes 0..3 of one warp, the rest zero, folded by halves).  Every lane
 * returns the total. */
__device__ __forceinline__ static float q4e_gf_row_sum(const float v[4]) {
    const float s0 = warp_sum_f32(v[0]);
    const float s1 = warp_sum_f32(v[1]);
    const float s2 = warp_sum_f32(v[2]);
    const float s3 = warp_sum_f32(v[3]);
    const float total = (s0 + s2) + (s1 + s3);
    return __shfl_sync(0xffffffffu, total, 0);
}

/* The convolution and silu of one channel at one token, from its four taps
 * oldest first: `q4e_gdn_conv_par_kernel`'s expression, unrolled. */
__device__ __forceinline__ static float q4e_gf_conv(const float wc[4], float x0, float x1, float x2, float x3) {
    float acc = 0.0f;
    acc += wc[0] * x0;
    acc += wc[1] * x1;
    acc += wc[2] * x2;
    acc += wc[3] * x3;
    return acc / (1.0f + __expf(-acc));
}

/* Raw input of channel c at chunk row p: before the chunk (p < 0) it is in the
 * carried window, past the chunk's last row it reads as zero. */
__device__ __forceinline__ static float q4e_gf_in(
        const float *x, const float *conv_state, int64_t p, int64_t n_tok, uint32_t stride, uint32_t c) {
    const float *src = (p >= 0) ? x + (uint64_t)p * stride
                                : conv_state + (uint64_t)((int64_t)Q4E_GF_HIST + p) * stride;
    return (p < n_tok) ? src[c] : 0.0f;
}

/* out[C][C] (float) = X Y^T over D, both [C][D] half tiles; `q4e_gt_gram`'s
 * tiles and order, one tile a warp. */
__device__ __forceinline__ static void q4e_gf_gram(float *out, const __half *x, const __half *y, uint32_t tile) {
    const uint32_t tr = tile / 4u, tc = tile % 4u;
    q4e_gf_acc acc;
    wmma::fill_fragment(acc, 0.0f);
#pragma unroll
    for (uint32_t k = 0; k < Q4E_GF_D; k += 16u) {
        q4e_gf_a_row a;
        q4e_gf_b_col b;
        wmma::load_matrix_sync(a, x + tr * 16u * Q4E_GF_LH + k, Q4E_GF_LH);
        wmma::load_matrix_sync(b, y + tc * 16u * Q4E_GF_LH + k, Q4E_GF_LH);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(out + tr * 16u * Q4E_GF_LA + tc * 16u, acc, Q4E_GF_LA, wmma::mem_row_major);
}

/* Two floats as a packed pair of halves, each rounded to nearest. */
__device__ __forceinline__ static uint32_t q4e_gf_h2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *)&h;
}

/* c += a b for one m16n8k16 step, f16 operands, f32 accumulators.  Lane
 * geometry throughout: g = lane / 4, c = lane % 4; A holds rows g and g + 8 at
 * columns 2c, 2c + 1 and 2c + 8, 2c + 9; B holds rows 2c, 2c + 1 and 2c + 8,
 * 2c + 9 of column g; C holds rows g and g + 8 at columns 2c and 2c + 1.  A
 * 16 x 16 x 16 WMMA product is two of these, so a sum taken this way is the
 * WMMA sum of the same operands, bit for bit. */
__device__ __forceinline__ static void q4e_gf_mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

/* Where element (t, v) of a run's u goes in the scan's fragment order: thread
 * (warp v / 16, lane 4 (v % 8) + (t % 8) / 2) reads its 32 halves as one
 * contiguous 64-byte run - its accumulator initialization, with no shared
 * memory and no gather. */
__device__ __forceinline__ static uint32_t q4e_gf_u_at(uint32_t t, uint32_t v) {
    const uint32_t thread = (v / 16u) * 32u + (v % 8u) * 4u + (t % 8u) / 2u;
    return thread * 32u + (t / 8u) * 4u + ((v / 8u) % 2u) * 2u + (t % 2u);
}

/* Halves a lower-triangular [C][C] tile is stored in: its ten 16 x 16 blocks
 * on and below the diagonal, block (bi, bj) at bi (bi + 1) / 2 + bj, each
 * row-major.  T and P are zero above the diagonal, so the six blocks above it
 * carry nothing a reader needs. */
#define Q4E_GF_TRI 2560u

/* Where element (t, u), u's block at or left of t's, goes in that layout. */
__device__ __forceinline__ static uint32_t q4e_gf_tri_at(uint32_t t, uint32_t u) {
    const uint32_t bi = t / 16u, bj = u / 16u;
    return ((bi * (bi + 1u)) / 2u + bj) * 256u + (t % 16u) * 16u + (u % 16u);
}

/* u = T X for T a padded [C][C] half tile and X a padded [C][D] half tile,
 * stored as halves in the scan's fragment order: `q4e_gt_apply`'s sums in its
 * order.  Warp w computes rows 16 (w / 4) .. + 15, columns 32 (w % 4) .. + 31. */
__device__ __forceinline__ static void q4e_gf_apply_u(__half *out, const __half *t_mat, const __half *x) {
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    const uint32_t g = lane >> 2u, c = lane & 3u;
    const uint32_t m0 = (warp >> 2u) * 16u, n0 = (warp & 3u) * 32u;
    float acc[4][4];
#pragma unroll
    for (uint32_t nt = 0; nt < 4u; nt++) {
#pragma unroll
        for (uint32_t i = 0; i < 4u; i++) acc[nt][i] = 0.0f;
    }
#pragma unroll
    for (uint32_t ku = 0; ku < Q4E_GF_C / 16u; ku++) {
        uint32_t a[4];
        a[0] = *(const uint32_t *)(t_mat + (m0 + g) * Q4E_GF_LP + 16u * ku + 2u * c);
        a[1] = *(const uint32_t *)(t_mat + (m0 + g + 8u) * Q4E_GF_LP + 16u * ku + 2u * c);
        a[2] = *(const uint32_t *)(t_mat + (m0 + g) * Q4E_GF_LP + 16u * ku + 2u * c + 8u);
        a[3] = *(const uint32_t *)(t_mat + (m0 + g + 8u) * Q4E_GF_LP + 16u * ku + 2u * c + 8u);
        const uint32_t r = 16u * ku + 2u * c;
#pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
            const uint32_t col = n0 + 8u * nt + g;
            const __half2 b0 = __halves2half2(x[r * Q4E_GF_LH + col], x[(r + 1u) * Q4E_GF_LH + col]);
            const __half2 b1 = __halves2half2(x[(r + 8u) * Q4E_GF_LH + col], x[(r + 9u) * Q4E_GF_LH + col]);
            q4e_gf_mma(acc[nt], a, *(const uint32_t *)&b0, *(const uint32_t *)&b1);
        }
    }
#pragma unroll
    for (uint32_t nt = 0; nt < 4u; nt++) {
        const uint32_t v = n0 + 8u * nt + 2u * c, t = m0 + g;
        out[q4e_gf_u_at(t, v)] = __float2half_rn(acc[nt][0]);
        out[q4e_gf_u_at(t, v + 1u)] = __float2half_rn(acc[nt][1]);
        out[q4e_gf_u_at(t + 8u, v)] = __float2half_rn(acc[nt][2]);
        out[q4e_gf_u_at(t + 8u, v + 1u)] = __float2half_rn(acc[nt][3]);
    }
}

/* Shared memory of the prep kernel, in bytes. */
#define Q4E_GF_PREP_SMEM                                                         \
    (2u * Q4E_GF_C * Q4E_GF_LH * 2u + Q4E_GF_R * Q4E_GF_C * Q4E_GF_LA * 4u +     \
     2u * Q4E_GF_R * Q4E_GF_C * 4u)

/* Everything a run needs that does not depend on the state at its start, for
 * one key head and its three value heads, from the projection's raw rows.
 *
 * Grid (key heads, runs of this window), 512 threads, 16 warps.  Run
 * `blockIdx.y` covers chunk rows `row0 + 64 blockIdx.y ..`.  Its outputs for
 * value head h go to slot `blockIdx.y * n_head_v + h`, those for the key head
 * to `blockIdx.y * n_head_k + hk` - slots of this window, so a host can reuse
 * one window's scratch for the next.  Per value-head slot: u [C][D] as halves
 * in the scan's fragment order (`q4e_gf_u_at`), T and P [C][C] as halves, and
 * the cumulative log decays and the write scales b e^G [2][C].  Per key-head
 * slot: the normalized query and key rows [C][D] in f32.  Everything is zero
 * past the run's last token.
 *
 * `conv_state` is the window as it stood before the chunk: the host rewrites
 * it only after every window's prep has read it.
 *
 * Warp w owns rows 4w .. 4w + 3 of the query and key tiles, lane l channels
 * l + 32 j; a value head's rows are convolved where they are staged, a head
 * ahead of their use. */
extern "C" __global__ void __launch_bounds__(512) q4e_gdf_prep_kernel(
        __half *u_out, __half *t_out, __half *p_out, float *ge_out, float *q_out, float *k_out,
        const float *x, const float *conv_state, const float *conv_w,
        const float *alpha_proj, const float *beta_proj, const float *dt_bias, const float *a,
        uint32_t n_head_k, uint32_t n_head_v, uint32_t stride, uint32_t row0, uint32_t n_tok) {
    extern __shared__ __align__(32) unsigned char q4e_gf_prep_smem[];
    __half *s_Q = (__half *)q4e_gf_prep_smem;                    /* [C][LH] queries, then Y, then T */
    __half *s_K = s_Q + Q4E_GF_C * Q4E_GF_LH;                    /* [C][LH] keys, then the right-hand sides */
    float *s_A = (float *)(s_K + Q4E_GF_C * Q4E_GF_LH);          /* [R][C][LA] grams, then the solves */
    float *s_G = s_A + Q4E_GF_R * Q4E_GF_C * Q4E_GF_LA;          /* [R][C] cumulative log decay */
    float *s_b = s_G + Q4E_GF_R * Q4E_GF_C;                      /* [R][C] beta */
    __half *s_T = s_Q;
    __half *s_X = s_K;
    float *s_Y = (float *)s_Q;                                   /* [R][3][16][16] block-row sums */

    const uint32_t hk = blockIdx.x, run = blockIdx.y;
    const uint32_t t0 = row0 + run * Q4E_GF_C;
    const uint32_t n = (n_tok - t0 < Q4E_GF_C) ? (n_tok - t0) : Q4E_GF_C;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t k_offset = n_head_k * Q4E_GF_D, v_offset = 2u * n_head_k * Q4E_GF_D;
    const float scale = rsqrtf((float)Q4E_GF_D);
    const uint32_t r0 = warp * 4u;

    /* Value head j's raw rows for this thread - channel vc, rows 16 vg - 3 ..
     * 16 vg + 15 - and its taps.  Issued a head ahead of their use, the first
     * head's here, because their latency is otherwise exposed once a head. */
    const uint32_t vc = tid % Q4E_GF_D, vg = tid / Q4E_GF_D;
    float xin[19];
    float4 vw;
    auto load_v = [&](uint32_t j) {
        const uint32_t c = v_offset + (hk + j * n_head_k) * Q4E_GF_D + vc;
        vw = *(const float4 *)(conv_w + (uint64_t)c * 4u);
#pragma unroll
        for (uint32_t i = 0; i < 19u; i++) {
            const int64_t p = (int64_t)t0 + (int64_t)(vg * 16u) + (int64_t)i - (int64_t)Q4E_GF_HIST;
            xin[i] = q4e_gf_in(x, conv_state, p, (int64_t)n_tok, stride, c);
        }
    };
    load_v(0);

    /* The gates of the three value heads, `q4e_gdn_gates_kernel`'s
     * expressions, zero past the run's last token. */
    if (tid < Q4E_GF_R * Q4E_GF_C) {
        const uint32_t j = tid / Q4E_GF_C, t = tid % Q4E_GF_C;
        const uint32_t h = hk + j * n_head_k;
        float g = 0.0f, b = 0.0f;
        if (t < n) {
            const uint64_t i = (uint64_t)(t0 + t) * n_head_v + h;
            const float xg = alpha_proj[i] + dt_bias[h];
            const float sp = (xg > 20.0f) ? xg : log1pf(__expf(xg));
            g = sp * a[h];
            b = 1.0f / (1.0f + __expf(-beta_proj[i]));
        }
        s_G[j * Q4E_GF_C + t] = g;
        s_b[j * Q4E_GF_C + t] = b;
    }

    /* The convolution, silu and L2 norm of this key head's query (m = 0) and
     * key (m = 1) rows.  Every load of a matrix is issued before any of its
     * arithmetic. */
#pragma unroll
    for (uint32_t m = 0; m < 2u; m++) {
        const uint32_t col0 = (m == 0u ? 0u : k_offset) + hk * Q4E_GF_D;
        float wc[4][4], win[4][7];
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            const uint32_t c = col0 + lane + 32u * j;
            const float4 w4 = *(const float4 *)(conv_w + (uint64_t)c * 4u);
            wc[j][0] = w4.x; wc[j][1] = w4.y; wc[j][2] = w4.z; wc[j][3] = w4.w;
#pragma unroll
            for (uint32_t i = 0; i < 7u; i++) {
                const int64_t p = (int64_t)t0 + (int64_t)r0 + (int64_t)i - (int64_t)Q4E_GF_HIST;
                win[j][i] = q4e_gf_in(x, conv_state, p, (int64_t)n_tok, stride, c);
            }
        }
        __half *tile = (m == 0u) ? s_Q : s_K;
#pragma unroll
        for (uint32_t i = 0; i < 4u; i++) {
            const uint32_t t = r0 + i;
            float v[4], sq[4];
#pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                v[j] = q4e_gf_conv(wc[j], win[j][i], win[j][i + 1u], win[j][i + 2u], win[j][i + 3u]);
                sq[j] = v[j] * v[j];
            }
            const float inv = rsqrtf(q4e_gf_row_sum(sq) + 1e-12f);
#pragma unroll
            for (uint32_t j = 0; j < 4u; j++) {
                const float o = (t < n) ? v[j] * inv : 0.0f;
                tile[t * Q4E_GF_LH + lane + 32u * j] = __float2half_rn(o);
                ((m == 0u) ? q_out : k_out)[((uint64_t)run * n_head_k + hk) * Q4E_GF_C * Q4E_GF_D +
                                            t * Q4E_GF_D + lane + 32u * j] = o;
            }
        }
    }
    __syncthreads();

    /* Inclusive prefix sums of the three heads' decays, one warp each, as
     * the ported prep scans its one. */
    if (warp < Q4E_GF_R) {
        float *gj = s_G + warp * Q4E_GF_C;
        float lo = gj[lane], hi = gj[lane + 32u];
#pragma unroll
        for (uint32_t o = 1u; o < 32u; o <<= 1) {
            const float pa = __shfl_up_sync(0xffffffffu, lo, o);
            const float pb = __shfl_up_sync(0xffffffffu, hi, o);
            if (lane >= o) { lo += pa; hi += pb; }
        }
        hi += __shfl_sync(0xffffffffu, lo, 31);
        gj[lane] = lo;
        gj[lane + 32u] = hi;
        const uint64_t slot = (uint64_t)run * n_head_v + hk + warp * n_head_k;
        const float *bj = s_b + warp * Q4E_GF_C;
        ge_out[slot * 2u * Q4E_GF_C + lane] = lo;
        ge_out[slot * 2u * Q4E_GF_C + lane + 32u] = hi;
        ge_out[slot * 2u * Q4E_GF_C + Q4E_GF_C + lane] = bj[lane] * __expf(lo);
        ge_out[slot * 2u * Q4E_GF_C + Q4E_GF_C + lane + 32u] = bj[lane + 32u] * __expf(hi);
    }

    /* Q K^T into the second solve tile and K K^T into the first. */
    q4e_gf_gram(s_A + Q4E_GF_C * Q4E_GF_LA, s_Q, s_K, warp);
    q4e_gf_gram(s_A, s_K, s_K, warp);
    __syncthreads();

    /* P for each head straight out, and each head's A in its own tile: every
     * element reads both grams before any of the three writes over them. */
    for (uint32_t idx = tid; idx < Q4E_GF_C * Q4E_GF_C; idx += blockDim.x) {
        const uint32_t t = idx / Q4E_GF_C, u = idx % Q4E_GF_C;
        const float kk = s_A[t * Q4E_GF_LA + u];
        const float qq = s_A[Q4E_GF_C * Q4E_GF_LA + t * Q4E_GF_LA + u];
#pragma unroll
        for (uint32_t j = 0; j < Q4E_GF_R; j++) {
            const float *gj = s_G + j * Q4E_GF_C;
            const uint64_t slot = (uint64_t)run * n_head_v + hk + j * n_head_k;
            const float pv = (u <= t && t < n) ? scale * __expf(gj[t] - gj[u]) * qq : 0.0f;
            if (u / 16u <= t / 16u) p_out[slot * Q4E_GF_TRI + q4e_gf_tri_at(t, u)] = __float2half_rn(pv);
            s_A[j * Q4E_GF_C * Q4E_GF_LA + t * Q4E_GF_LA + u] =
                (u < t && t < n) ? s_b[j * Q4E_GF_C + t] * __expf(gj[t] - gj[u]) * kk : 0.0f;
        }
    }
    __syncthreads();

    /* T = (I + A)^-1 for the three heads at once, `q4e_gdn_tc_prep_kernel`'s
     * blocked substitution: warps 4j .. 4j + 3 invert head j's four diagonal
     * blocks, then the block rows follow under barriers shared by all three. */
    if (warp < 4u * Q4E_GF_R) {
        float *blk = s_A + (warp / 4u) * Q4E_GF_C * Q4E_GF_LA + (warp % 4u) * 16u * Q4E_GF_LA + (warp % 4u) * 16u;
        const uint32_t c = lane;
        for (uint32_t r = 1u; r < 16u; r++) {
            float v = 0.0f;
            if (c < r) {
                v = -blk[r * Q4E_GF_LA + c];
                for (uint32_t u = c + 1u; u < r; u++) v = fmaf(-blk[r * Q4E_GF_LA + u], blk[u * Q4E_GF_LA + c], v);
            }
            __syncwarp();
            if (c < r) blk[r * Q4E_GF_LA + c] = v;
            __syncwarp();
        }
    }
    __syncthreads();
    for (uint32_t bi = 1u; bi < 4u; bi++) {
        const uint32_t per = bi * 256u;
        for (uint32_t idx = tid; idx < Q4E_GF_R * per; idx += blockDim.x) {
            const uint32_t j = idx / per, rest = idx % per;
            const float *sa = s_A + j * Q4E_GF_C * Q4E_GF_LA;
            const uint32_t bj = rest / 256u, r = (rest % 256u) / 16u, cc = rest % 16u;
            const uint32_t row = bi * 16u + r, col = bj * 16u + cc;
            float y = 0.0f;
            for (uint32_t k = bj * 16u; k < bi * 16u; k++) {
                const float t_kc = (k == col) ? 1.0f : (k > col ? sa[k * Q4E_GF_LA + col] : 0.0f);
                y = fmaf(sa[row * Q4E_GF_LA + k], t_kc, y);
            }
            s_Y[j * 768u + rest] = y;
        }
        __syncthreads();
        for (uint32_t idx = tid; idx < Q4E_GF_R * per; idx += blockDim.x) {
            const uint32_t j = idx / per, rest = idx % per;
            float *sa = s_A + j * Q4E_GF_C * Q4E_GF_LA;
            const float *sy = s_Y + j * 768u;
            const uint32_t bj = rest / 256u, r = (rest % 256u) / 16u, cc = rest % 16u;
            const uint32_t row = bi * 16u + r;
            float v = -sy[bj * 256u + r * 16u + cc];
            for (uint32_t u = 0; u < r; u++) {
                v = fmaf(-sa[row * Q4E_GF_LA + bi * 16u + u], sy[bj * 256u + u * 16u + cc], v);
            }
            sa[row * Q4E_GF_LA + bj * 16u + cc] = v;
        }
        __syncthreads();
    }

    /* Per head: T, and u = T (b o V) in the scan's fragment order. */
#pragma unroll
    for (uint32_t j = 0; j < Q4E_GF_R; j++) {
        const uint32_t h = hk + j * n_head_k;
        const uint64_t slot = (uint64_t)run * n_head_v + h;
        const float *sa = s_A + j * Q4E_GF_C * Q4E_GF_LA;
        const float *bj = s_b + j * Q4E_GF_C;
        /* T as halves, kept for the product and stored for the scan; then
         * the value rows, convolved here, and the next head's loads. */
        {
            const float wc[4] = {vw.x, vw.y, vw.z, vw.w};
            for (uint32_t idx = tid; idx < Q4E_GF_C * Q4E_GF_C; idx += blockDim.x) {
                const uint32_t t = idx / Q4E_GF_C, u = idx % Q4E_GF_C;
                const __half tv = __float2half_rn(t == u ? 1.0f : sa[t * Q4E_GF_LA + u]);
                s_T[t * Q4E_GF_LP + u] = tv;
                if (u / 16u <= t / 16u) t_out[slot * Q4E_GF_TRI + q4e_gf_tri_at(t, u)] = tv;
            }
#pragma unroll
            for (uint32_t i = 0; i < 16u; i++) {
                const uint32_t t = vg * 16u + i;
                const float v = (t < n) ? q4e_gf_conv(wc, xin[i], xin[i + 1u], xin[i + 2u], xin[i + 3u]) * bj[t] : 0.0f;
                s_X[t * Q4E_GF_LH + vc] = __float2half_rn(v);
            }
            if (j + 1u < Q4E_GF_R) load_v(j + 1u);
        }
        __syncthreads();
        q4e_gf_apply_u(u_out + slot * Q4E_GF_C * Q4E_GF_D, s_T, s_X);
        __syncthreads();
    }
}

/* ---------------------------------------------------------------------------
 * The scan.
 *
 * The ported scan holds the state in WMMA accumulators, and since an
 * accumulator is not an operand it copies the whole state to shared memory as
 * halves every run, stages the deltas there too, and separates the three
 * products with block barriers.  Here the state is held transposed, S^T
 * [value][key], in raw m16n8k16 accumulators, and every product is written in
 * that orientation:
 *
 *     w   = T half(b e^G o K)
 *     d^T = u^T - S^T w^T        o^T = S^T q'^T + d^T P^T
 *     S^T <- e^{G_last} S^T + d^T k'
 *
 * so the state and the deltas are always the left operand, whose fragment is
 * the accumulator's own register layout converted to halves.  A warp owns
 * sixteen values - sixteen rows of S^T, d^T and o^T - and needs no other
 * warp's anything between a run's first product and its last.  Each output
 * element is the ported sum of the same half-precision products in the same
 * order of 16-wide steps; the right operands q' = half(s e^G q) and
 * k' = half(e^{G_last - G} k) are rounded from the f32 rows as they are read,
 * which is the rounding the ported scan staged them with.
 *
 * w is formed here from T and the f32 key rows rather than read: T is half
 * its size, and the key rows are read anyway.  The run tiles arrive by
 * `cp.async` into unpadded tiles whose 16-byte pieces are permuted by row
 * (`q4e_gf_sw*`), each buffer refilled with the next run's tile as soon as
 * this run's last reader is done with it.
 * ------------------------------------------------------------------------ */

/* Element (r, col) of a [.][D] float tile, its 16-byte pieces permuted within
 * each row by `piece ^ (r % 8)`: the rows a fragment load touches land on
 * different banks, which is what padding buys, without its bytes - and a
 * piece moves whole, so `cp.async` fills the tile as it lies. */
__device__ __forceinline__ static uint32_t q4e_gf_swf(uint32_t r, uint32_t col) {
    return r * Q4E_GF_D + ((((col >> 2u) ^ (r & 7u)) << 2u) | (col & 3u));
}

/* Element (r, col) of a [.][D] half tile, permuted the same way. */
__device__ __forceinline__ static uint32_t q4e_gf_swh(uint32_t r, uint32_t col) {
    return r * Q4E_GF_D + ((((col >> 3u) ^ (r & 7u)) << 3u) | (col & 7u));
}

/* Element (r, col) of a [.][C] half tile, permuted the same way. */
__device__ __forceinline__ static uint32_t q4e_gf_swc(uint32_t r, uint32_t col) {
    return r * Q4E_GF_C + ((((col >> 3u) ^ (r & 7u)) << 3u) | (col & 7u));
}

/* Shared memory of the scan kernel, in bytes: the f32 query and key rows
 * [C][D], w [C][D] (and, after the deltas, half a run of staged outputs),
 * T and P [C][C] halves, and the decays and write scales [2][C]. */
#define Q4E_GF_SCAN_SMEM                                                         \
    (2u * Q4E_GF_C * Q4E_GF_D * 4u + Q4E_GF_C * Q4E_GF_D * 2u + 2u * Q4E_GF_C * Q4E_GF_C * 2u + \
     2u * Q4E_GF_C * 4u)

/* The run-to-run carry over this window's runs, and every run's gated output.
 *
 * Grid (value heads), 256 threads.  The carry of `q4e_gdn_tc_scan_kernel` over
 * chunk rows `row0 .. n_tok` (row0 on the run grid), reading what
 * `q4e_gdf_prep_kernel` wrote for the same window: per slot u (in this
 * kernel's fragment order), T and P, the decays and the write scales; per
 * (run, key head) the normalized f32 query and key rows.  Each run's outputs
 * are staged in shared memory, half a run at a time, and normalized and gated
 * with `q4e_gdn_out_gate_kernel`'s arithmetic: `out` receives the gated rows
 * [T][n_head_v * D], and `z` and `norm_w` are that kernel's gate rows and norm
 * weight.
 *
 * `snap`, when not null, receives the state before chunk run `snap_run`
 * (counted from the chunk's first row, not the window's), laid out as `state`
 * is - the ported scan's checkpoint. */
extern "C" __global__ void __launch_bounds__(256) q4e_gdf_scan_kernel(
        float *out, float *state,
        const __half *u_in, const __half *t_in, const __half *p_in, const float *ge_in,
        const float *q_in, const float *k_in, const float *z, const float *norm_w,
        uint32_t n_head_k, uint32_t n_head_v, uint32_t row0, uint32_t n_tok, float eps,
        float *snap, uint32_t snap_run) {
    extern __shared__ __align__(32) unsigned char q4e_gf_scan_smem[];
    float *s_Q = (float *)q4e_gf_scan_smem;                      /* [C][D] query rows */
    float *s_K = s_Q + Q4E_GF_C * Q4E_GF_D;                      /* [C][D] key rows */
    __half *s_W = (__half *)(s_K + Q4E_GF_C * Q4E_GF_D);         /* [C][D] w */
    __half *s_T = s_W + Q4E_GF_C * Q4E_GF_D;                     /* [C][C] T */
    __half *s_P = s_T + Q4E_GF_C * Q4E_GF_C;                     /* [C][C] P */
    float *s_G = (float *)(s_P + Q4E_GF_C * Q4E_GF_C);           /* [C] cumulative log decay */
    float *s_E = s_G + Q4E_GF_C;                                 /* [C] b e^G */
    float *s_O = (float *)s_W;                                   /* [C/2][D] staged outputs, after the deltas */

    const uint32_t h = blockIdx.x, hk = h % n_head_k;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t g = lane >> 2u, c = lane & 3u;
    const uint32_t v0 = warp * 16u;               /* this warp's sixteen values */
    const uint64_t out_ld = (uint64_t)n_head_v * Q4E_GF_D;
    float *state_h = state + (uint64_t)h * Q4E_GF_D * Q4E_GF_D;
    const uint64_t tile_d = (uint64_t)Q4E_GF_C * Q4E_GF_D;
    const float scale = rsqrtf((float)Q4E_GF_D);

    /* The input tiles of run `rl` of this window, in the groups their buffers
     * free in. */
    auto fetch_f32 = [&](float *dst, const float *src) {
#pragma unroll
        for (uint32_t q = 0; q < 8u; q++) {
            const uint32_t idx = tid + q * 256u;
            const uint32_t r = idx / 32u, pc = idx % 32u;
            q4e_gf_cp16(dst + r * Q4E_GF_D + ((pc ^ (r & 7u)) << 2u), src + r * Q4E_GF_D + pc * 4u);
        }
    };
    /* A packed lower-triangular tile into a full one: 320 16-byte pieces;
     * the pieces of the blocks above the diagonal are never written, and stay
     * the zeros the tile was cleared to. */
    auto fetch_tri = [&](__half *dst, const __half *src) {
#pragma unroll
        for (uint32_t q = 0; q < 2u; q++) {
            const uint32_t idx = tid + q * 256u;
            if (idx < Q4E_GF_TRI / 8u) {
                const uint32_t b = idx / 32u, r = (idx % 32u) / 2u, hp = idx % 2u;
                const uint32_t bi = (b >= 6u) ? 3u : (b >= 3u) ? 2u : (b >= 1u) ? 1u : 0u;
                const uint32_t bj = b - (bi * (bi + 1u)) / 2u;
                const uint32_t row = 16u * bi + r, pc = 2u * bj + hp;
                q4e_gf_cp16(dst + row * Q4E_GF_C + ((pc ^ (row & 7u)) << 3u), src + b * 256u + r * 16u + hp * 8u);
            }
        }
    };
    auto fetch_t = [&](uint32_t rl) {
        fetch_tri(s_T, t_in + ((uint64_t)rl * n_head_v + h) * Q4E_GF_TRI);
        q4e_gf_cp_commit();
    };
    auto fetch_qp = [&](uint32_t rl) {
        fetch_f32(s_Q, q_in + ((uint64_t)rl * n_head_k + hk) * tile_d);
        fetch_tri(s_P, p_in + ((uint64_t)rl * n_head_v + h) * Q4E_GF_TRI);
        q4e_gf_cp_commit();
    };
    auto fetch_kg = [&](uint32_t rl) {
        fetch_f32(s_K, k_in + ((uint64_t)rl * n_head_k + hk) * tile_d);
        if (tid < 2u * Q4E_GF_C / 4u) {
            q4e_gf_cp16(s_G + tid * 4u, ge_in + ((uint64_t)rl * n_head_v + h) * 2u * Q4E_GF_C + tid * 4u);
        }
        q4e_gf_cp_commit();
    };
    /* This thread's 32 halves of u, as four 16-byte loads. */
    uint4 ur[4];
    auto load_u = [&](uint32_t rl) {
        const uint4 *src = (const uint4 *)(u_in + ((uint64_t)rl * n_head_v + h) * tile_d + tid * 32u);
#pragma unroll
        for (uint32_t q = 0; q < 4u; q++) ur[q] = src[q];
    };

    /* S^T rows v0 + g and v0 + g + 8, keys 8 j + 2 c and + 1: the stored
     * state is value-major, so S^T is read as it lies. */
    float sacc[16][4];
#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) {
        const float2 lo = *(const float2 *)(state_h + (uint64_t)(v0 + g) * Q4E_GF_D + 8u * j + 2u * c);
        const float2 hi = *(const float2 *)(state_h + (uint64_t)(v0 + g + 8u) * Q4E_GF_D + 8u * j + 2u * c);
        sacc[j][0] = lo.x; sacc[j][1] = lo.y; sacc[j][2] = hi.x; sacc[j][3] = hi.y;
    }
    /* The norm weight of the four channels 4 lane .. 4 lane + 3 this lane
     * gates. */
    const float4 nw = *(const float4 *)(norm_w + 4u * lane);

    /* T and P are filled block by block; the blocks above their diagonals
     * are these zeros for the whole launch. */
    for (uint32_t idx = tid; idx < 2u * Q4E_GF_C * Q4E_GF_C / 8u; idx += blockDim.x) {
        *(uint4 *)(s_T + idx * 8u) = make_uint4(0u, 0u, 0u, 0u);
    }
    __syncthreads();
    if (row0 < n_tok) {
        fetch_t(0);
        fetch_qp(0);
        fetch_kg(0);
        load_u(0);
    }

    for (uint32_t t0 = row0; t0 < n_tok; t0 += Q4E_GF_C) {
        const uint32_t run = t0 / Q4E_GF_C, rl = (t0 - row0) / Q4E_GF_C;
        const uint32_t n = (n_tok - t0 < Q4E_GF_C) ? (n_tok - t0) : Q4E_GF_C;
        const bool more = t0 + Q4E_GF_C < n_tok;

        /* This run's gate rows, in flight through the products: warp w gates
         * rows 4w .. 4w + 3 of each half run, lane l channels 4l .. 4l + 3. */
        float4 zr[2][4];
#pragma unroll
        for (uint32_t hf = 0; hf < 2u; hf++) {
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t t = hf * 32u + warp * 4u + i;
                zr[hf][i] = (t < n) ? __ldcs((const float4 *)(z + (uint64_t)(t0 + t) * out_ld + h * Q4E_GF_D + 4u * lane))
                                    : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        if (snap && run == snap_run) {
            float *snap_h = snap + (uint64_t)h * Q4E_GF_D * Q4E_GF_D;
#pragma unroll
            for (uint32_t j = 0; j < 16u; j++) {
                *(float2 *)(snap_h + (uint64_t)(v0 + g) * Q4E_GF_D + 8u * j + 2u * c) = make_float2(sacc[j][0], sacc[j][1]);
                *(float2 *)(snap_h + (uint64_t)(v0 + g + 8u) * Q4E_GF_D + 8u * j + 2u * c) = make_float2(sacc[j][2], sacc[j][3]);
            }
        }
        q4e_gf_cp_wait();
        __syncthreads();

        /* w = T half(b e^G o K): warp w rows 16 (w / 2) .. + 15, keys
         * 64 (w % 2) .. + 63. */
        {
            const uint32_t m0 = (warp >> 1u) * 16u, n0 = (warp & 1u) * 64u;
            float wacc[8][4];
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; nt++) {
#pragma unroll
                for (uint32_t i = 0; i < 4u; i++) wacc[nt][i] = 0.0f;
            }
#pragma unroll
            for (uint32_t ku = 0; ku < 4u; ku++) {
                uint32_t a[4];
                a[0] = *(const uint32_t *)(s_T + q4e_gf_swc(m0 + g, 16u * ku + 2u * c));
                a[1] = *(const uint32_t *)(s_T + q4e_gf_swc(m0 + g + 8u, 16u * ku + 2u * c));
                a[2] = *(const uint32_t *)(s_T + q4e_gf_swc(m0 + g, 16u * ku + 2u * c + 8u));
                a[3] = *(const uint32_t *)(s_T + q4e_gf_swc(m0 + g + 8u, 16u * ku + 2u * c + 8u));
                const uint32_t r = 16u * ku + 2u * c;
                const float e0 = s_E[r], e1 = s_E[r + 1u], e8 = s_E[r + 8u], e9 = s_E[r + 9u];
#pragma unroll
                for (uint32_t nt = 0; nt < 8u; nt++) {
                    const uint32_t col = n0 + 8u * nt + g;
                    const uint32_t b0 = q4e_gf_h2(s_K[q4e_gf_swf(r, col)] * e0, s_K[q4e_gf_swf(r + 1u, col)] * e1);
                    const uint32_t b1 = q4e_gf_h2(s_K[q4e_gf_swf(r + 8u, col)] * e8, s_K[q4e_gf_swf(r + 9u, col)] * e9);
                    q4e_gf_mma(wacc[nt], a, b0, b1);
                }
            }
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; nt++) {
                const uint32_t col = n0 + 8u * nt + 2u * c;
                *(uint32_t *)(s_W + q4e_gf_swh(m0 + g, col)) = q4e_gf_h2(wacc[nt][0], wacc[nt][1]);
                *(uint32_t *)(s_W + q4e_gf_swh(m0 + g + 8u, col)) = q4e_gf_h2(wacc[nt][2], wacc[nt][3]);
            }
        }
        __syncthreads();   /* w is complete; T is read */
        if (more) fetch_t(rl + 1u);

        /* The state as left operands, one per 16-key step. */
        uint32_t sa[8][4];
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
            sa[kk][0] = q4e_gf_h2(sacc[2u * kk][0], sacc[2u * kk][1]);
            sa[kk][1] = q4e_gf_h2(sacc[2u * kk][2], sacc[2u * kk][3]);
            sa[kk][2] = q4e_gf_h2(sacc[2u * kk + 1u][0], sacc[2u * kk + 1u][1]);
            sa[kk][3] = q4e_gf_h2(sacc[2u * kk + 1u][2], sacc[2u * kk + 1u][3]);
        }

        /* d^T = u^T - S^T w^T, tokens 8 nt .. 8 nt + 7 per tile. */
        float dacc[8][4];
        {
            const __half *uh = (const __half *)ur;
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; nt++) {
#pragma unroll
                for (uint32_t i = 0; i < 4u; i++) dacc[nt][i] = __half2float(uh[nt * 4u + i]);
            }
        }
        if (more) load_u(rl + 1u);
#pragma unroll
        for (uint32_t kk = 0; kk < 8u; kk++) {
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; nt++) {
                const uint32_t t = 8u * nt + g;
                __half2 b0 = __hneg2(*(const __half2 *)(s_W + q4e_gf_swh(t, 16u * kk + 2u * c)));
                __half2 b1 = __hneg2(*(const __half2 *)(s_W + q4e_gf_swh(t, 16u * kk + 2u * c + 8u)));
                q4e_gf_mma(dacc[nt], sa[kk], *(const uint32_t *)&b0, *(const uint32_t *)&b1);
            }
        }
        /* The deltas as left operands, one per 16-token step. */
        uint32_t da[4][4];
#pragma unroll
        for (uint32_t ku = 0; ku < 4u; ku++) {
            da[ku][0] = q4e_gf_h2(dacc[2u * ku][0], dacc[2u * ku][1]);
            da[ku][1] = q4e_gf_h2(dacc[2u * ku][2], dacc[2u * ku][3]);
            da[ku][2] = q4e_gf_h2(dacc[2u * ku + 1u][0], dacc[2u * ku + 1u][1]);
            da[ku][3] = q4e_gf_h2(dacc[2u * ku + 1u][2], dacc[2u * ku + 1u][3]);
        }

        /* The queries' row scales, read before the decays' buffer is
         * refilled: the output product runs after the carry. */
        float eq[8];
#pragma unroll
        for (uint32_t nt = 0; nt < 8u; nt++) eq[nt] = scale * __expf(s_G[8u * nt + g]);

        /* S^T <- e^{G_last} S^T + d^T k', k' = half(e^{G_last - G} k) as it
         * is read.  Before the output product, which reads the state as it
         * stood at the run's start - the halves in `sa` - so that the key
         * rows' buffer frees early and the next run's rows have the output
         * product and the gate to arrive in. */
        {
            const float g_last = s_G[n - 1u];
            const float dec = __expf(g_last);
#pragma unroll
            for (uint32_t j = 0; j < 16u; j++) {
#pragma unroll
                for (uint32_t i = 0; i < 4u; i++) sacc[j][i] *= dec;
            }
#pragma unroll
            for (uint32_t ku = 0; ku < 4u; ku++) {
                const uint32_t r = 16u * ku + 2u * c;
                float ek[4];
#pragma unroll
                for (uint32_t i = 0; i < 4u; i++) {
                    const uint32_t t = r + (i & 1u) + 8u * (i >> 1u);
                    ek[i] = (t < n) ? __expf(g_last - s_G[t]) : 0.0f;
                }
#pragma unroll
                for (uint32_t j = 0; j < 16u; j++) {
                    const uint32_t col = 8u * j + g;
                    const uint32_t b0 = q4e_gf_h2(s_K[q4e_gf_swf(r, col)] * ek[0], s_K[q4e_gf_swf(r + 1u, col)] * ek[1]);
                    const uint32_t b1 = q4e_gf_h2(s_K[q4e_gf_swf(r + 8u, col)] * ek[2], s_K[q4e_gf_swf(r + 9u, col)] * ek[3]);
                    q4e_gf_mma(sacc[j], da[ku], b0, b1);
                }
            }
        }

        __syncthreads();   /* every read of w, the key rows and the decays is done */
        if (more) fetch_kg(rl + 1u);

        /* o^T = S^T q'^T + d^T P^T, q' = half(s e^G q) as it is read. */
        float oacc[8][4];
        {
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; nt++) {
#pragma unroll
                for (uint32_t i = 0; i < 4u; i++) oacc[nt][i] = 0.0f;
            }
#pragma unroll
            for (uint32_t kk = 0; kk < 8u; kk++) {
#pragma unroll
                for (uint32_t nt = 0; nt < 8u; nt++) {
                    const uint32_t t = 8u * nt + g;
                    const float2 q0 = *(const float2 *)(s_Q + q4e_gf_swf(t, 16u * kk + 2u * c));
                    const float2 q1 = *(const float2 *)(s_Q + q4e_gf_swf(t, 16u * kk + 2u * c + 8u));
                    q4e_gf_mma(oacc[nt], sa[kk], q4e_gf_h2(q0.x * eq[nt], q0.y * eq[nt]),
                               q4e_gf_h2(q1.x * eq[nt], q1.y * eq[nt]));
                }
            }
        }
#pragma unroll
        for (uint32_t ku = 0; ku < 4u; ku++) {
#pragma unroll
            for (uint32_t nt = 0; nt < 8u; nt++) {
                const uint32_t t = 8u * nt + g;
                q4e_gf_mma(oacc[nt], da[ku], *(const uint32_t *)(s_P + q4e_gf_swc(t, 16u * ku + 2u * c)),
                           *(const uint32_t *)(s_P + q4e_gf_swc(t, 16u * ku + 2u * c + 8u)));
            }
        }
        __syncthreads();   /* every read of the query rows and P is done */
        if (more) fetch_qp(rl + 1u);

        /* The output gate, half a run at a time: per-row RMS norm over the
         * head, then the sigmoid gate, each row reduced as the ported
         * 128-thread block reduces it. */
#pragma unroll
        for (uint32_t hf = 0; hf < 2u; hf++) {
#pragma unroll
            for (uint32_t q = 0; q < 4u; q++) {
                const uint32_t nt = hf * 4u + q;
                const uint32_t t = 8u * q + 2u * c;
                s_O[q4e_gf_swf(t, v0 + g)] = oacc[nt][0];
                s_O[q4e_gf_swf(t + 1u, v0 + g)] = oacc[nt][1];
                s_O[q4e_gf_swf(t, v0 + g + 8u)] = oacc[nt][2];
                s_O[q4e_gf_swf(t + 1u, v0 + g + 8u)] = oacc[nt][3];
            }
            __syncthreads();
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t tr = warp * 4u + i, t = hf * 32u + tr;
                /* The sum reads the row in the ported block's order, lane l
                 * channels l + 32 j; the gate writes it four channels a
                 * lane. */
                float sq[4];
#pragma unroll
                for (uint32_t j = 0; j < 4u; j++) {
                    const float v = s_O[q4e_gf_swf(tr, lane + 32u * j)];
                    sq[j] = v * v;
                }
                const float rs = rsqrtf(q4e_gf_row_sum(sq) / (float)Q4E_GF_D + eps);
                if (t < n) {
                    const float4 src = *(const float4 *)(s_O + q4e_gf_swf(tr, 4u * lane));
                    const float4 zz = zr[hf][i];
                    float4 o;
                    o.x = src.x * rs * nw.x * (1.0f / (1.0f + __expf(-zz.x)));
                    o.y = src.y * rs * nw.y * (1.0f / (1.0f + __expf(-zz.y)));
                    o.z = src.z * rs * nw.z * (1.0f / (1.0f + __expf(-zz.z)));
                    o.w = src.w * rs * nw.w * (1.0f / (1.0f + __expf(-zz.w)));
                    __stcs((float4 *)(out + (uint64_t)(t0 + t) * out_ld + h * Q4E_GF_D + 4u * lane), o);
                }
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (uint32_t j = 0; j < 16u; j++) {
        *(float2 *)(state_h + (uint64_t)(v0 + g) * Q4E_GF_D + 8u * j + 2u * c) = make_float2(sacc[j][0], sacc[j][1]);
        *(float2 *)(state_h + (uint64_t)(v0 + g + 8u) * Q4E_GF_D + 8u * j + 2u * c) = make_float2(sacc[j][2], sacc[j][3]);
    }
}
