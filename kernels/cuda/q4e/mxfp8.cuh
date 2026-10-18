/* The MXFP8 decode and the exact single-precision arithmetic every MXFP8
 * step kernel runs: `mxfp8.cu`'s helpers, in a header so another unit's
 * launches over the same weights (`ffn.cu`) run them instruction for
 * instruction, and `q4e_mx_rows`, the one-row step's body, for the same
 * reason. See `mxfp8.cu` for the format and why the arithmetic is written
 * without flushing. */

#ifndef SHISO_Q4E_MXFP8_CUH
#define SHISO_Q4E_MXFP8_CUH

#include <stdint.h>
#include <cuda_fp16.h>

/* Code bytes a warp stages per row per step: sixteen a lane. */
#define Q4E_MX_CHUNK 512u
/* Weights one `ue8m0` byte scales. */
#define Q4E_MX_GROUP 32u

/* Exact single-precision arithmetic whatever the build's flush mode. */
__device__ __forceinline__ float q4e_mx_mul(float a, float b) {
    float r;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}
__device__ __forceinline__ float q4e_mx_fma(float a, float b, float c) {
    float r;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(r) : "f"(a), "f"(b), "f"(c));
    return r;
}
__device__ __forceinline__ float q4e_mx_add(float a, float b) {
    float r;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(r) : "f"(a), "f"(b));
    return r;
}

/* `2^(s - 127)` for one `ue8m0` byte: the byte is the `f32` exponent field,
 * except 0 (2^-127, the one subnormal) and the reserved NaN code 255. */
__device__ __forceinline__ float q4e_mx_scale(uint32_t s) {
    const uint32_t bits = s == 0u ? 0x00400000u : (s == 255u ? 0x7FC00000u : s << 23);
    return __uint_as_float(bits);
}

/* Two codes (the even element in the low byte) as two halves holding each
 * value over 256: sign to the half's sign bit, the four exponent and three
 * mantissa bits to the top of the half's own fields. The exponent field is
 * then the code's, which puts the value 2^8 below it, and a zero exponent is
 * a half subnormal of the same mantissa - exactly the code's subnormal over
 * 256. The two NaN codes become half NaNs. */
__device__ __forceinline__ uint32_t q4e_mx_halves(uint32_t p) {
    uint32_t h = ((p & 0x0080u) << 8) | ((p & 0x007Fu) << 7)
               | ((p & 0x8000u) << 16) | ((p & 0x7F00u) << 15);
    if ((p & 0x007Fu) == 0x007Fu) {
        h |= 0x00007FFFu;
    }
    if ((p & 0x7F00u) == 0x7F00u) {
        h |= 0x7FFF0000u;
    }
    return h;
}

/* Two codes as two exact `f32` values, before their group scale. */
__device__ __forceinline__ float2 q4e_mx_pair(uint32_t p) {
    const uint32_t h = q4e_mx_halves(p);
    const float2 f = __half22float2(*reinterpret_cast<const __half2 *>(&h));
    /* `f` is zero or at least 2^-17, so the factor of 256 is exact. */
    return make_float2(f.x * 256.0f, f.y * 256.0f);
}

/* The decreasing-offset shuffle fold of `warp_sum_f32`, without flushing. */
__device__ __forceinline__ float q4e_mx_warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = q4e_mx_add(v, __shfl_down_sync(0xffffffffu, v, offset));
    }
    return v;
}

/* `R` output rows a warp, `NT` activation rows, `D` chunks deep. `stage` is
 * this warp's `R * 512` bytes of shared memory. Rows past `out_dim` read the
 * last row and store nothing.
 *
 * `R` and `D` move nothing in a row's arithmetic - each row is still one
 * lane-strided sequence of fused multiply-adds and one fold - so the entries
 * below choose them for bandwidth alone. */
template <int NT, int R, int D>
__device__ __forceinline__ void q4e_mx_rows(
        float *__restrict__ out, const uint8_t *__restrict__ codes,
        const uint8_t *__restrict__ scales, const float *__restrict__ x,
        uint32_t in_dim, uint32_t out_dim, uint32_t row0, uint32_t lane,
        uint8_t *__restrict__ stage) {
    const uint8_t *wr[R];
    const uint8_t *sr[R];
    float acc[R][NT];
#pragma unroll
    for (int r = 0; r < R; r++) {
        const uint64_t row = min(row0 + (uint32_t) r, out_dim - 1u);
        wr[r] = codes + row * in_dim;
        sr[r] = scales + row * (in_dim / Q4E_MX_GROUP);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            acc[r][t] = 0.0f;
        }
    }
    for (uint32_t c0 = 0; c0 < in_dim; c0 += Q4E_MX_CHUNK * D) {
        /* Every load first, so the warp's whole share of the memory system is
         * asked for before any of it is waited on. A sixteen-byte slot is
         * either wholly inside the row or wholly past it, because the row is
         * whole groups of 32. */
        uint4 buf[R][D];
#pragma unroll
        for (int d = 0; d < D; d++) {
            const uint32_t off = c0 + (uint32_t) d * Q4E_MX_CHUNK + 16u * lane;
#pragma unroll
            for (int r = 0; r < R; r++) {
                buf[r][d] = off < in_dim ? __ldg(reinterpret_cast<const uint4 *>(wr[r] + off))
                                         : make_uint4(0u, 0u, 0u, 0u);
            }
        }
#pragma unroll
        for (int d = 0; d < D; d++) {
            const uint32_t base = c0 + (uint32_t) d * Q4E_MX_CHUNK;
            if (base >= in_dim) {
                break;
            }
            __syncwarp();
#pragma unroll
            for (int r = 0; r < R; r++) {
                *reinterpret_cast<uint4 *>(stage + r * Q4E_MX_CHUNK + 16u * lane) = buf[r][d];
            }
            __syncwarp();
            /* Pair `j` of this chunk for this lane is element
             * `base + 2 lane + 64 j`: pair index `lane + 32 (base / 64 + j)`,
             * the brain-float kernel's order. */
#pragma unroll
            for (uint32_t j = 0; j < Q4E_MX_CHUNK / 64u; j++) {
                const uint32_t e = base + 2u * lane + 64u * j;
                if (e < in_dim) {
                    float2 xv[NT];
#pragma unroll
                    for (int t = 0; t < NT; t++) {
                        xv[t] = *reinterpret_cast<const float2 *>(x + (uint64_t) t * in_dim + e);
                    }
#pragma unroll
                    for (int r = 0; r < R; r++) {
                        const uint32_t p = *reinterpret_cast<const uint16_t *>(
                                stage + r * Q4E_MX_CHUNK + 2u * lane + 64u * j);
                        const float s = q4e_mx_scale(__ldg(sr[r] + e / Q4E_MX_GROUP));
                        const float2 v = q4e_mx_pair(p);
                        const float w0 = q4e_mx_mul(v.x, s);
                        const float w1 = q4e_mx_mul(v.y, s);
#pragma unroll
                        for (int t = 0; t < NT; t++) {
                            acc[r][t] = q4e_mx_fma(w0, xv[t].x, acc[r][t]);
                            acc[r][t] = q4e_mx_fma(w1, xv[t].y, acc[r][t]);
                        }
                    }
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++) {
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float v = q4e_mx_warp_sum(acc[r][t]);
            if (lane == 0u && row0 + (uint32_t) r < out_dim) {
                out[(uint64_t) t * out_dim + row0 + r] = v;
            }
        }
    }
}

#endif
