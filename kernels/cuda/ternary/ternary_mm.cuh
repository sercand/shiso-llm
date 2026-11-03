/* Matrix-matrix products over ternary and 2-bit affine weights; the twins of
 * `kernels/metal/ternary/ternary_mm.metal` under the same entry
 * names and argument order. The Metal pair feeds the simdgroup matrix units;
 * these are plain shared-memory tiles with FMA cores, the simplest kernels
 * that read the same weights the same way: 32 output rows by `TN` tokens a
 * block of 128 threads, the weights of each 64-input step decoded once into
 * shared memory (exact in f32) and every thread accumulating `32 * TN / 128`
 * outputs. `kernel_ternary_mmv_*` is the eight-token tile. */
#ifndef SHISO_TERNARY_MM_CUH
#define SHISO_TERNARY_MM_CUH

#include "ternary_mv.cuh"

#define TERNARY_MM_ROWS 32u
#define TERNARY_MM_K 64u

__device__ __forceinline__ unsigned ternary_ptq1_code(const unsigned char *blk, unsigned e) {
    if (e < 80u) {
        return (unsigned) ternary_trit(blk[e % 16u], e / 16u);
    }
    if (e < 120u) {
        return (unsigned) ternary_trit(blk[16u + (e - 80u) % 8u], (e - 80u) / 8u);
    }
    return (unsigned) ternary_trit(blk[24u + (e - 120u) % 2u], (e - 120u) / 2u);
}

template <unsigned FMT, unsigned TN>
__device__ __forceinline__ void ternary_mm(const struct ternary_args_mv &args,
                                           const unsigned char *w,
                                           const unsigned short *scales,
                                           const unsigned short *biases, const float *x,
                                           float *y) {
    __shared__ float wt[TERNARY_MM_ROWS * TERNARY_MM_K];
    __shared__ float xt[TN * TERNARY_MM_K];
    const unsigned in_dim = args.in_dim;
    const unsigned out_dim = args.out_dim;
    const unsigned groups = in_dim / TERNARY_GROUP;
    const uint64_t row_bytes = ternary_row_bytes(FMT, in_dim);
    const unsigned o0 = blockIdx.x * TERNARY_MM_ROWS;
    const unsigned t0 = blockIdx.y * TN;
    const unsigned tid = threadIdx.x;
    constexpr unsigned PER = TERNARY_MM_ROWS * TN / 128u;
    float acc[PER];
#pragma unroll
    for (unsigned i = 0; i < PER; ++i) {
        acc[i] = 0.0f;
    }
    const unsigned wr = tid / 4u;
    const unsigned wk = (tid % 4u) * 16u;
    const unsigned row = o0 + wr;
    for (unsigned k0 = 0; k0 < in_dim; k0 += TERNARY_MM_K) {
        __syncthreads();
        {
            const unsigned k = k0 + wk;
            const unsigned g = k / TERNARY_GROUP;
            const unsigned e0 = k % TERNARY_GROUP;
            float *dst = wt + wr * TERNARY_MM_K + wk;
            if (row >= out_dim) {
#pragma unroll
                for (unsigned i = 0; i < 16u; ++i) {
                    dst[i] = 0.0f;
                }
            } else if (FMT == TERNARY_PTQ1) {
                const unsigned char *blk = w + (uint64_t) row * row_bytes + (uint64_t) g * 28u;
                const float sc = ternary_half(*(const unsigned short *) (blk + 26u));
                for (unsigned i = 0; i < 16u; ++i) {
                    dst[i] = sc * (float) ternary_ptq1_code(blk, e0 + i) - sc;
                }
            } else {
                float sc;
                float bi;
                unsigned word;
                if (FMT == TERNARY_PQ2) {
                    const unsigned short *p = (const unsigned short *) (w + (uint64_t) row * row_bytes +
                                                                        (uint64_t) g * 34u);
                    sc = ternary_half(p[0]);
                    bi = -sc;
                    word = (unsigned) p[1u + e0 / 8u] | ((unsigned) p[2u + e0 / 8u] << 16u);
                } else {
                    const unsigned *p =
                            (const unsigned *) (w + (uint64_t) row * row_bytes + (uint64_t) g * 32u);
                    sc = ternary_half(scales[(uint64_t) row * groups + g]);
                    bi = ternary_half(biases[(uint64_t) row * groups + g]);
                    word = p[e0 / 16u];
                }
#pragma unroll
                for (unsigned i = 0; i < 16u; ++i) {
                    dst[i] = sc * (float) ((word >> (2u * i)) & 3u) + bi;
                }
            }
        }
        for (unsigned i = tid; i < TN * TERNARY_MM_K; i += 128u) {
            const unsigned t = i / TERNARY_MM_K;
            const unsigned k = i % TERNARY_MM_K;
            xt[i] = t0 + t < args.n_tok ? x[(uint64_t) (t0 + t) * in_dim + k0 + k] : 0.0f;
        }
        __syncthreads();
#pragma unroll
        for (unsigned i = 0; i < PER; ++i) {
            const unsigned o = (tid + 128u * i) / TN;
            const unsigned t = (tid + 128u * i) % TN;
            float a = acc[i];
            for (unsigned k = 0; k < TERNARY_MM_K; ++k) {
                a = fmaf(wt[o * TERNARY_MM_K + k], xt[t * TERNARY_MM_K + k], a);
            }
            acc[i] = a;
        }
    }
#pragma unroll
    for (unsigned i = 0; i < PER; ++i) {
        const unsigned o = (tid + 128u * i) / TN;
        const unsigned t = (tid + 128u * i) % TN;
        if (o0 + o < out_dim && t0 + t < args.n_tok) {
            y[(uint64_t) (t0 + t) * out_dim + o0 + o] = acc[i];
        }
    }
}

#define TERNARY_MM_ENTRY(name, fmt, tn, entry)                                                    \
    extern "C" __global__ void entry(                                                            \
            const struct ternary_args_mv args, const unsigned char *__restrict__ w,              \
            const unsigned short *__restrict__ scales, const unsigned short *__restrict__ biases, \
            const float *__restrict__ x, float *__restrict__ y) {                                \
        ternary_mm<fmt, tn>(args, w, scales, biases, x, y);                                      \
    }

TERNARY_MM_ENTRY(pq2_0, TERNARY_PQ2, 8, kernel_ternary_mm_pq2_0_f32_n8)
TERNARY_MM_ENTRY(pq2_0, TERNARY_PQ2, 32, kernel_ternary_mm_pq2_0_f32_n32)
TERNARY_MM_ENTRY(mlx2, TERNARY_MLX2, 8, kernel_ternary_mm_mlx2_f32_n8)
TERNARY_MM_ENTRY(mlx2, TERNARY_MLX2, 32, kernel_ternary_mm_mlx2_f32_n32)
TERNARY_MM_ENTRY(ptq1_0, TERNARY_PTQ1, 8, kernel_ternary_mm_ptq1_0_f32_n8)
TERNARY_MM_ENTRY(ptq1_0, TERNARY_PTQ1, 32, kernel_ternary_mm_ptq1_0_f32_n32)
TERNARY_MM_ENTRY(pq2_0, TERNARY_PQ2, 8, kernel_ternary_mmv_pq2_0_f32)
TERNARY_MM_ENTRY(mlx2, TERNARY_MLX2, 8, kernel_ternary_mmv_mlx2_f32)
TERNARY_MM_ENTRY(ptq1_0, TERNARY_PTQ1, 8, kernel_ternary_mmv_ptq1_0_f32)

#endif
