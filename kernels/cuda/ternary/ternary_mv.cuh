/* Matrix-vector products over ternary and 2-bit affine weights, one
 * activation row a block (only `T = 1` is instantiated), and the row gather; the twins of
 * `kernels/metal/ternary/ternary_mv.metal`, whose header comment
 * states the formats, the arithmetic and the geometry. A warp here is a Metal
 * simdgroup: 32 lanes, four output rows, eight lanes a 128-input group. */
#ifndef SHISO_TERNARY_MV_CUH
#define SHISO_TERNARY_MV_CUH

#include <cuda_fp16.h>

#include "ternary_common.cuh"

#define TERNARY_GROUP 128u
#define TERNARY_ROWS 8u
#define TERNARY_SG_ROWS 4u
#define TERNARY_SWEEP (4u * TERNARY_GROUP)
#define TERNARY_MAX_SPLITS 16u

#define TERNARY_PQ2 0u
#define TERNARY_MLX2 1u
#define TERNARY_PTQ1 2u

__device__ __forceinline__ float ternary_half(unsigned short bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float ternary_load_x16(const float *x, float (&xt)[16]) {
    const float4 *xv = (const float4 *) x;
    /* 4^-j for the code's position j in its half-word, j = 0..3 and 4..7. */
    const float pre[8] = {1.0f, 1.0f / 4.0f, 1.0f / 16.0f, 1.0f / 64.0f,
                          1.0f / 256.0f, 1.0f / 1024.0f, 1.0f / 4096.0f, 1.0f / 16384.0f};
    float sum = 0.0f;
#pragma unroll
    for (unsigned i = 0; i < 4u; ++i) {
        const float4 v = xv[i];
        sum += (v.x + v.y) + (v.z + v.w);
        const unsigned j = (4u * i) % 8u;
        xt[4u * i + 0u] = v.x * pre[j + 0u];
        xt[4u * i + 1u] = v.y * pre[j + 1u];
        xt[4u * i + 2u] = v.z * pre[j + 2u];
        xt[4u * i + 3u] = v.w * pre[j + 3u];
    }
    return sum;
}

__device__ __forceinline__ float ternary_qdot16(unsigned short lo, unsigned short hi,
                                                const float (&xt)[16]) {
    float acc = 0.0f;
#pragma unroll
    for (unsigned j = 0; j < 8u; ++j) {
        const unsigned mask = 3u << (2u * j);
        acc += xt[j] * (float) (lo & mask) + xt[8u + j] * (float) (hi & mask);
    }
    return acc;
}

__device__ __forceinline__ float ternary_trit(unsigned byte, unsigned n) {
    const unsigned pow3[5] = {1u, 3u, 9u, 27u, 81u};
    return (float) ((((byte * pow3[n]) & 255u) * 3u) >> 8u);
}

__device__ __forceinline__ float ternary_ptq1_slot(const unsigned char *blk, const float *x,
                                                   unsigned j, float &xsum) {
    float acc = 0.0f;
    float sum = 0.0f;
    const unsigned b0 = blk[2u * j];
    const unsigned b1 = blk[2u * j + 1u];
#pragma unroll
    for (unsigned n = 0; n < 5u; ++n) {
        const float2 v = *(const float2 *) (x + 16u * n + 2u * j);
        acc += v.x * ternary_trit(b0, n) + v.y * ternary_trit(b1, n);
        sum += v.x + v.y;
    }
    const unsigned b2 = blk[16u + j];
#pragma unroll
    for (unsigned n = 0; n < 5u; ++n) {
        const float v = x[80u + 8u * n + j];
        acc += v * ternary_trit(b2, n);
        sum += v;
    }
    if (j < 2u) {
        const unsigned b3 = blk[24u + j];
#pragma unroll
        for (unsigned n = 0; n < 4u; ++n) {
            const float v = x[120u + 2u * n + j];
            acc += v * ternary_trit(b3, n);
            sum += v;
        }
    }
    xsum = sum;
    return acc;
}

__device__ __forceinline__ uint64_t ternary_row_bytes(unsigned fmt, unsigned in_dim) {
    const uint64_t groups = in_dim / TERNARY_GROUP;
    return fmt == TERNARY_PQ2 ? groups * 34u : fmt == TERNARY_PTQ1 ? groups * 28u : in_dim / 4u;
}

template <unsigned FMT, unsigned T>
__device__ __forceinline__ void ternary_project(const struct ternary_args_mv &args,
                                                const unsigned char *w,
                                                const unsigned short *scales,
                                                const unsigned short *biases, const float *x,
                                                float *y) {
    __shared__ float part[TERNARY_MAX_SPLITS * TERNARY_ROWS * T];
    const unsigned in_dim = args.in_dim;
    const unsigned out_dim = args.out_dim;
    const unsigned r0 = blockIdx.y * T;
    if (r0 >= args.n_tok) {
        return;
    }
    const unsigned lane = threadIdx.x % 32u;
    const unsigned simd = threadIdx.x / 32u;
    const unsigned splits = args.splits;
    const unsigned split = simd / 2u;
    const unsigned sub = (simd % 2u) * TERNARY_SG_ROWS;
    const unsigned rows_here = min(T, args.n_tok - r0);
    const unsigned groups = in_dim / TERNARY_GROUP;
    const unsigned o = blockIdx.x * TERNARY_ROWS + sub;
    const unsigned rows_out = o < out_dim ? min(TERNARY_SG_ROWS, out_dim - o) : 0u;
    const uint64_t row_bytes = ternary_row_bytes(FMT, in_dim);
    const unsigned char *wr = w + (uint64_t) o * row_bytes;
    const unsigned short *sr = scales + (uint64_t) o * groups;
    const unsigned short *br = biases + (uint64_t) o * groups;
    const float *xr = x + (uint64_t) r0 * in_dim;

    float acc[TERNARY_SG_ROWS][T];
#pragma unroll
    for (unsigned r = 0; r < TERNARY_SG_ROWS; ++r) {
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    const unsigned slot = lane % 8u;
    for (unsigned k0 = split * TERNARY_SWEEP; k0 < in_dim; k0 += TERNARY_SWEEP * splits) {
        const unsigned g = k0 / TERNARY_GROUP + lane / 8u;
        if (g >= groups) {
            continue;
        }
        float sc[TERNARY_SG_ROWS];
        float bi[TERNARY_SG_ROWS];
        unsigned short lo[TERNARY_SG_ROWS];
        unsigned short hi[TERNARY_SG_ROWS];
        const unsigned char *blk[TERNARY_SG_ROWS];
#pragma unroll
        for (unsigned r = 0; r < TERNARY_SG_ROWS; ++r) {
            if (r >= rows_out) {
                continue;
            }
            const unsigned char *row = wr + (uint64_t) r * row_bytes;
            if (FMT == TERNARY_PQ2) {
                const unsigned short *p = (const unsigned short *) (row + (uint64_t) g * 34u);
                sc[r] = ternary_half(p[0]);
                bi[r] = -sc[r];
                lo[r] = p[1u + 2u * slot];
                hi[r] = p[2u + 2u * slot];
            } else if (FMT == TERNARY_MLX2) {
                const unsigned short *p = (const unsigned short *) (row + (uint64_t) g * 32u);
                sc[r] = ternary_half(sr[(uint64_t) r * groups + g]);
                bi[r] = ternary_half(br[(uint64_t) r * groups + g]);
                lo[r] = p[2u * slot];
                hi[r] = p[2u * slot + 1u];
            } else {
                blk[r] = row + (uint64_t) g * 28u;
                sc[r] = ternary_half(*(const unsigned short *) (blk[r] + 26u));
                bi[r] = -sc[r];
            }
        }
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            if (t >= rows_here) {
                continue;
            }
            const float *xg = xr + (uint64_t) t * in_dim + (uint64_t) g * TERNARY_GROUP;
            if (FMT == TERNARY_PTQ1) {
#pragma unroll
                for (unsigned r = 0; r < TERNARY_SG_ROWS; ++r) {
                    if (r < rows_out) {
                        float sum;
                        const float dot = ternary_ptq1_slot(blk[r], xg, slot, sum);
                        acc[r][t] += sc[r] * dot + bi[r] * sum;
                    }
                }
            } else {
                float xt[16];
                const float sum = ternary_load_x16(xg + 16u * slot, xt);
#pragma unroll
                for (unsigned r = 0; r < TERNARY_SG_ROWS; ++r) {
                    if (r < rows_out) {
                        acc[r][t] += sc[r] * ternary_qdot16(lo[r], hi[r], xt) + bi[r] * sum;
                    }
                }
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < TERNARY_SG_ROWS; ++r) {
#pragma unroll
        for (unsigned t = 0; t < T; ++t) {
            float v = acc[r][t];
#pragma unroll
            for (unsigned off = 16u; off > 0u; off >>= 1u) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (splits == 1u) {
                if (lane == 0u && t < rows_here && r < rows_out) {
                    y[(uint64_t) (r0 + t) * out_dim + o + r] = v;
                }
            } else if (lane == 0u) {
                part[(split * TERNARY_ROWS + sub + r) * T + t] = v;
            }
        }
    }
    if (splits == 1u) {
        return;
    }
    __syncthreads();
    const unsigned tid = threadIdx.x;
    if (tid < TERNARY_ROWS * T) {
        const unsigned r = tid / T;
        const unsigned t = tid % T;
        const unsigned oo = blockIdx.x * TERNARY_ROWS + r;
        if (oo < out_dim && t < rows_here) {
            float v = 0.0f;
            for (unsigned p = 0; p < splits; ++p) {
                v += part[(p * TERNARY_ROWS + r) * T + t];
            }
            y[(uint64_t) (r0 + t) * out_dim + oo] = v;
        }
    }
}

#define TERNARY_MV(name, fmt, t)                                                                  \
    extern "C" __global__ void kernel_ternary_mv_##name##_f32_t##t(                              \
            const struct ternary_args_mv args, const unsigned char *__restrict__ w,              \
            const unsigned short *__restrict__ scales, const unsigned short *__restrict__ biases, \
            const float *__restrict__ x, float *__restrict__ y) {                                \
        ternary_project<fmt, t>(args, w, scales, biases, x, y);                                  \
    }

TERNARY_MV(pq2_0, TERNARY_PQ2, 1)
TERNARY_MV(mlx2, TERNARY_MLX2, 1)
TERNARY_MV(ptq1_0, TERNARY_PTQ1, 1)

/* One weight row per token id, decoded to f32. Grid `(in_dim / 128, n_tok)`,
 * 128 threads. */
template <unsigned FMT>
__device__ __forceinline__ void ternary_get_rows(const struct ternary_args_rows &args,
                                                 const unsigned char *w,
                                                 const unsigned short *scales,
                                                 const unsigned short *biases, const int *ids,
                                                 float *y) {
    const unsigned i = threadIdx.x;
    const unsigned g = blockIdx.x;
    const uint64_t groups = args.in_dim / TERNARY_GROUP;
    const uint64_t id = (uint64_t) (unsigned) ids[blockIdx.y];
    float scale;
    float bias;
    unsigned code;
    if (FMT == TERNARY_PQ2) {
        const unsigned char *blk = w + (id * groups + g) * 34u;
        scale = ternary_half(*(const unsigned short *) blk);
        bias = -scale;
        code = (blk[2u + i / 4u] >> (2u * (i % 4u))) & 3u;
    } else if (FMT == TERNARY_MLX2) {
        const unsigned char *blk = w + (id * groups + g) * 32u;
        scale = ternary_half(scales[id * groups + g]);
        bias = ternary_half(biases[id * groups + g]);
        code = (blk[i / 4u] >> (2u * (i % 4u))) & 3u;
    } else {
        const unsigned char *blk = w + (id * groups + g) * 28u;
        scale = ternary_half(*(const unsigned short *) (blk + 26u));
        bias = -scale;
        if (i < 80u) {
            code = (unsigned) ternary_trit(blk[i % 16u], i / 16u);
        } else if (i < 120u) {
            code = (unsigned) ternary_trit(blk[16u + (i - 80u) % 8u], (i - 80u) / 8u);
        } else {
            code = (unsigned) ternary_trit(blk[24u + (i - 120u) % 2u], (i - 120u) / 2u);
        }
    }
    y[(uint64_t) blockIdx.y * args.in_dim + (uint64_t) g * TERNARY_GROUP + i] =
            scale * (float) code + bias;
}

#define TERNARY_ROWS_ENTRY(name, fmt)                                                             \
    extern "C" __global__ void kernel_ternary_get_rows_##name##_f32(                             \
            const struct ternary_args_rows args, const unsigned char *__restrict__ w,            \
            const unsigned short *__restrict__ scales, const unsigned short *__restrict__ biases, \
            const int *__restrict__ ids, float *__restrict__ y) {                                \
        ternary_get_rows<fmt>(args, w, scales, biases, ids, y);                                  \
    }

TERNARY_ROWS_ENTRY(pq2_0, TERNARY_PQ2)
TERNARY_ROWS_ENTRY(mlx2, TERNARY_MLX2)
TERNARY_ROWS_ENTRY(ptq1_0, TERNARY_PTQ1)

#endif
