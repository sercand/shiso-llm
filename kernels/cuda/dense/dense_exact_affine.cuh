/* Row-exact MLX affine matmul: 4- and 8-bit weights in groups of 64, f32 rows.
 *
 * The CUDA twin of `kernels/metal/dense/dense_exact_affine.metal`,
 * under the same entry names and launch geometry, so the engine's plan in
 * `shiso_engine::projection::exact_affine_plan` drives both. What it keeps is
 * the property, not the Metal bits: a row's output does not depend on how
 * many rows share the call, so a verify window of drafted tokens can be
 * checked against serial decoding by equality. Bits across backends are not
 * required to match; within this one they are.
 *
 * The arithmetic, identical in both kernels. Per output `n` of row `r`:
 *
 *   chunk c in 0..8 takes groups c, c + 8, c + 16, ...
 *   per group g:  P = the FMA chain over the group's 64 values in order,
 *                     P = fma(q_i, x_i, P)
 *                 S = the left-to-right sum of the 64 inputs
 *                 acc_c = fma(bias, S, fma(scale, P, acc_c))
 *   out = ((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))
 *
 * Every step is an explicit round-to-nearest intrinsic, so the compiler can
 * neither contract nor reassociate it, and the chunking is a constant rather
 * than a function of the call. The matvec combines its chunks by butterfly
 * shuffles and the matrix kernel through shared memory, in the same tree:
 * float addition is commutative, so the two lanes of a butterfly pair hold
 * the same bits.
 *
 * Inputs need K % 64 == 0 and N % 8 == 0, which the host plan checks.
 */
#ifndef SHISO_DENSE_EXACT_AFFINE_CUH
#define SHISO_DENSE_EXACT_AFFINE_CUH

#include "dense_common.cuh"

/* Chunks a row's groups are dealt to: `DENSE_EXACT_SPLITS` on Metal. */
#define DENSE_EXACT_SPLITS 8u

/* Rows one matrix block may carry: 8 * rt with rt at most 2. */
#define DENSE_EXACT_MAX_ROWS 16u

/* Outputs one matrix block may carry: 8 * nt with nt at most 4. */
#define DENSE_EXACT_MAX_OUTS 32u

struct dense_args_exact_affine {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_rows;
    uint32_t pad;
};
static_assert(sizeof(struct dense_args_exact_affine) == 16, "ExactAffineArgs is 16 bytes");

/* One group's step for one (row, output): `words` are the output's 64 / (32 /
 * BITS) packed words of the group, `x` the row's 64 inputs of it. */
template <unsigned BITS>
__device__ __forceinline__ float dense_exact_group(
        float acc, const uint32_t *__restrict__ words, const float *__restrict__ x,
        float scale, float bias) {
    constexpr unsigned PER = 32u / BITS;
    constexpr unsigned WPG = 64u / PER;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    float p = 0.0f;
    float s = 0.0f;
#pragma unroll
    for (unsigned w = 0; w < WPG; ++w) {
        const uint32_t word = words[w];
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) {
            const float xv = x[w * PER + i];
            p = __fmaf_rn((float) ((word >> (BITS * i)) & MASK), xv, p);
            s = __fadd_rn(s, xv);
        }
    }
    return __fmaf_rn(bias, s, __fmaf_rn(scale, p, acc));
}

/* One chunk's accumulator for one (row, output). */
template <unsigned BITS>
__device__ __forceinline__ float dense_exact_chunk(
        const uint32_t *__restrict__ w_row, const unsigned short *__restrict__ s_row,
        const unsigned short *__restrict__ b_row, const float *__restrict__ x_row,
        unsigned groups, unsigned c) {
    constexpr unsigned WPG = 64u / (32u / BITS);
    float acc = 0.0f;
    for (unsigned g = c; g < groups; g += DENSE_EXACT_SPLITS) {
        acc = dense_exact_group<BITS>(acc, w_row + g * WPG, x_row + g * 64u,
                                      dense_bf16(s_row + g), dense_bf16(b_row + g));
    }
    return acc;
}

/* The fixed tree over eight chunk accumulators. */
__device__ __forceinline__ float dense_exact_tree(const float a[DENSE_EXACT_SPLITS]) {
    const float t01 = __fadd_rn(a[0], a[1]);
    const float t23 = __fadd_rn(a[2], a[3]);
    const float t45 = __fadd_rn(a[4], a[5]);
    const float t67 = __fadd_rn(a[6], a[7]);
    return __fadd_rn(__fadd_rn(t01, t23), __fadd_rn(t45, t67));
}

/* One row. Lane (chunk c = lane % 8, slot j = lane / 8) of warp w in block b
 * owns outputs n0 + 4u, u < NR, with n0 = (b * warps + w) * 4 * NR + j; the
 * chunks meet by butterfly shuffles in the tree's order. Grid and block are the
 * Metal entry's: `ceil(N / (warps * 4 * NR))` blocks of `warps * 32`. */
template <unsigned BITS, unsigned NR>
__device__ __forceinline__ void dense_exact_qmv(
        const struct dense_args_exact_affine args, const uint32_t *__restrict__ w,
        const unsigned short *__restrict__ sc, const unsigned short *__restrict__ bi,
        const float *__restrict__ x, float *__restrict__ out) {
    constexpr unsigned WPR_DIV = 32u / BITS;
    const unsigned k = args.in_dim;
    const unsigned n = args.out_dim;
    const unsigned groups = k / 64u;
    const unsigned lane = threadIdx.x % 32u;
    const unsigned warp = threadIdx.x / 32u;
    const unsigned warps = blockDim.x / 32u;
    const unsigned c = lane % DENSE_EXACT_SPLITS;
    const unsigned n0 = (blockIdx.x * warps + warp) * (4u * NR) + lane / DENSE_EXACT_SPLITS;
#pragma unroll
    for (unsigned u = 0; u < NR; ++u) {
        const unsigned nn = n0 + 4u * u;
        const unsigned row = min(nn, n - 1u);
        float v = dense_exact_chunk<BITS>(w + (uint64_t) row * (k / WPR_DIV),
                                          sc + (uint64_t) row * groups,
                                          bi + (uint64_t) row * groups, x, groups, c);
        // The tree, by butterflies: chunk pairs, then pairs of pairs.
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 1));
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 2));
        v = __fadd_rn(v, __shfl_xor_sync(0xffffffffu, v, 4));
        if (c == 0u && nn < n) {
            out[nn] = v;
        }
    }
}

/* Any row count. Block (bx, by) covers outputs bx * 8 * NT .. and rows
 * by * 8 * RT ..; warp w is chunk w and takes that chunk's groups. Per group
 * the warp stages the rows' 64 inputs and their sums in shared memory - each
 * sum computed once, as the same left-to-right chain the matvec computes -
 * and a lane then unpacks each of its output's values once and carries it
 * into every one of its rows' chains, in the value order the matvec uses.
 * The chunks meet in shared memory, in the tree's order. Block: 256 threads. */
template <unsigned BITS, unsigned NT, unsigned RT>
__device__ __forceinline__ void dense_exact_qmm(
        const struct dense_args_exact_affine args, const uint32_t *__restrict__ w,
        const unsigned short *__restrict__ sc, const unsigned short *__restrict__ bi,
        const float *__restrict__ x, float *__restrict__ out) {
    constexpr unsigned PER = 32u / BITS;
    constexpr unsigned WPG = 64u / PER;
    constexpr uint32_t MASK = (1u << BITS) - 1u;
    constexpr unsigned OUTS = 8u * NT;
    constexpr unsigned ROWS = 8u * RT;
    constexpr unsigned LANES_PER_OUT = 32u / OUTS;
    constexpr unsigned ROWS_PER_LANE = ROWS / LANES_PER_OUT;
    /* The staged inputs during the sweep, then the chunk partials in the same
     * memory: 33 KiB, where the two apart would pass the 48 KiB static
     * limit. */
    __shared__ float smem[DENSE_EXACT_SPLITS * DENSE_EXACT_MAX_ROWS * (64 + 1)];
    static_assert(DENSE_EXACT_SPLITS * DENSE_EXACT_MAX_ROWS * DENSE_EXACT_MAX_OUTS
                          <= DENSE_EXACT_SPLITS * DENSE_EXACT_MAX_ROWS * (64 + 1),
                  "the partials fit in the staging they reuse");
    float (*stage)[DENSE_EXACT_MAX_ROWS][64 + 1] =
            reinterpret_cast<float (*)[DENSE_EXACT_MAX_ROWS][64 + 1]>(smem);
    float (*part)[DENSE_EXACT_MAX_ROWS][DENSE_EXACT_MAX_OUTS] =
            reinterpret_cast<float (*)[DENSE_EXACT_MAX_ROWS][DENSE_EXACT_MAX_OUTS]>(smem);

    const unsigned k = args.in_dim;
    const unsigned n = args.out_dim;
    const unsigned rows = args.n_rows;
    const unsigned groups = k / 64u;
    const unsigned lane = threadIdx.x % 32u;
    const unsigned c = threadIdx.x / 32u;
    const unsigned o = lane % OUTS;
    const unsigned r_lane = lane / OUTS;
    const unsigned nn = blockIdx.x * OUTS + o;
    const unsigned out_row = min(nn, n - 1u);
    const unsigned r0 = blockIdx.y * ROWS;

    const uint32_t *w_row = w + (uint64_t) out_row * groups * WPG;
    const unsigned short *s_row = sc + (uint64_t) out_row * groups;
    const unsigned short *b_row = bi + (uint64_t) out_row * groups;
    float (*xs)[64 + 1] = stage[c];

    float acc[ROWS_PER_LANE];
#pragma unroll
    for (unsigned j = 0; j < ROWS_PER_LANE; ++j) {
        acc[j] = 0.0f;
    }
    for (unsigned g = c; g < groups; g += DENSE_EXACT_SPLITS) {
        // The rows' inputs of this group, two per lane per row.
        for (unsigned t = lane; t < ROWS * 64u; t += 32u) {
            const unsigned rr = t / 64u;
            const unsigned r = min(r0 + rr, rows - 1u);
            xs[rr][t % 64u] = x[(uint64_t) r * k + g * 64u + t % 64u];
        }
        __syncwarp();
        // Each row's sum, once, in the matvec's order; kept in the pad slot.
        if (lane < ROWS) {
            float sum = 0.0f;
            for (unsigned i = 0; i < 64u; ++i) {
                sum = __fadd_rn(sum, xs[lane][i]);
            }
            xs[lane][64] = sum;
        }
        __syncwarp();
        float p[ROWS_PER_LANE];
#pragma unroll
        for (unsigned j = 0; j < ROWS_PER_LANE; ++j) {
            p[j] = 0.0f;
        }
        const uint32_t *gw = w_row + g * WPG;
#pragma unroll
        for (unsigned q = 0; q < WPG; ++q) {
            const uint32_t word = gw[q];
#pragma unroll
            for (unsigned i = 0; i < PER; ++i) {
                const float qv = (float) ((word >> (BITS * i)) & MASK);
#pragma unroll
                for (unsigned j = 0; j < ROWS_PER_LANE; ++j) {
                    p[j] = __fmaf_rn(qv, xs[r_lane + j * LANES_PER_OUT][q * PER + i], p[j]);
                }
            }
        }
        const float scale = dense_bf16(s_row + g);
        const float bias = dense_bf16(b_row + g);
#pragma unroll
        for (unsigned j = 0; j < ROWS_PER_LANE; ++j) {
            const float sum = xs[r_lane + j * LANES_PER_OUT][64];
            acc[j] = __fmaf_rn(bias, sum, __fmaf_rn(scale, p[j], acc[j]));
        }
        // The next group's staging overwrites this one's.
        __syncwarp();
    }
    // Every warp is done with its staging before any partial overwrites it.
    __syncthreads();
#pragma unroll
    for (unsigned j = 0; j < ROWS_PER_LANE; ++j) {
        part[c][r_lane + j * LANES_PER_OUT][o] = acc[j];
    }
    __syncthreads();
    for (unsigned t = threadIdx.x; t < ROWS * OUTS; t += blockDim.x) {
        const unsigned rr = t / OUTS;
        const unsigned oo = t % OUTS;
        const unsigned r = r0 + rr;
        const unsigned col = blockIdx.x * OUTS + oo;
        if (r < rows && col < n) {
            float a[DENSE_EXACT_SPLITS];
#pragma unroll
            for (unsigned q = 0; q < DENSE_EXACT_SPLITS; ++q) {
                a[q] = part[q][rr][oo];
            }
            out[(uint64_t) r * n + col] = dense_exact_tree(a);
        }
    }
}

#define DENSE_EXACT_QMV(BITS, NR)                                                        \
    extern "C" __global__ void kernel_dense_exact_qmv_##BITS##_nr##NR##_f32(            \
            const struct dense_args_exact_affine args, const uint32_t *__restrict__ w,  \
            const unsigned short *__restrict__ sc, const unsigned short *__restrict__ bi, \
            const float *__restrict__ x, float *__restrict__ out) {                     \
        dense_exact_qmv<BITS, NR>(args, w, sc, bi, x, out);                             \
    }

#define DENSE_EXACT_QMM(BITS, NT, RT)                                                    \
    extern "C" __global__ void kernel_dense_exact_qmm_##BITS##_nt##NT##_rt##RT##_f32(   \
            const struct dense_args_exact_affine args, const uint32_t *__restrict__ w,  \
            const unsigned short *__restrict__ sc, const unsigned short *__restrict__ bi, \
            const float *__restrict__ x, float *__restrict__ out) {                     \
        dense_exact_qmm<BITS, NT, RT>(args, w, sc, bi, x, out);                         \
    }

DENSE_EXACT_QMV(4, 1)
DENSE_EXACT_QMV(4, 2)
DENSE_EXACT_QMV(8, 1)
DENSE_EXACT_QMV(8, 2)
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

#endif
