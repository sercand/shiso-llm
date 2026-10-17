/* Dense projections over an MXFP8 weight, read as stored: one `e4m3` byte a
 * weight, row-major `[out, in]`, and one `ue8m0` byte a group of 32 weights
 * along a row, `[out, in / 32]`:
 *
 *     w[o][k] = e4m3(codes[o][k]) * 2^(scales[o][k / 32] - 127)
 *
 * Not an upstream unit, for the reason `modelopt.cu` is not.
 *
 * Five entries, one argument list
 * (`out, codes, scales, x, in_dim, out_dim, rows`):
 *
 * - `q4e_mxfp8_gemv_kernel`, one activation row (a decode step), and
 *   `q4e_mxfp8_gemv_narrow_kernel`, the same for an output of at most 1024
 *   rows;
 * - `q4e_mxfp8_rows_kernel`, two to eight (a speculative verify), and
 *   `q4e_mxfp8_rows_narrow_kernel`, the same for an output of at most 1024
 *   rows;
 * - `q4e_mxfp8_gemm_kernel`, a prompt chunk of any width, on tensor cores.
 *
 * # The step kernels are the brain-float tier, bit for bit
 *
 * Every MXFP8 weight is exactly a brain float whenever it is a normal number:
 * an `e4m3` value has four significant bits and a power-of-two scale only
 * moves its exponent. So the step kernels run the arithmetic of
 * `q4e_matmul_bf16_rows_kernel` (matvec.cu) on that widened matrix, in its
 * order: lane `l` owns element pairs `l, l + 32, ...` of a row, one fused
 * multiply-add per weight in contraction order with the activation rows inner,
 * then the decreasing-offset shuffle fold. A row's output is therefore the
 * brain-float tier's output for the widened weight, and independent of how
 * many activation rows share the launch, which is what lets a verify row
 * reproduce a single-row step.
 *
 * What differs is how the bytes arrive. The brain-float kernel's lanes read
 * four contiguous bytes each; at one byte a weight that ordering would be two
 * bytes a lane, so instead each lane loads sixteen contiguous code bytes per
 * 512-byte chunk, the warp stages the chunk in shared memory, and every lane
 * reads its own pairs back out of it, every load of a step issued before the
 * first is consumed. The verify kernels also stage the activations, once a
 * block, so the weight is read once whatever the pass width.
 *
 * # Exact, and safe under flush-to-zero
 *
 * The tree builds with `--use_fast_math`, which flushes subnormal `f32`
 * operands and results. The code is decoded through half precision, where an
 * `e4m3` code shifted into place is exactly the value over 256 (an `e4m3`
 * subnormal lands on a half subnormal, which widens to a normal `f32`), then
 * scaled by 256 exactly. The group scale, the multiply-adds and the fold use
 * the round-to-nearest PTX instructions without `.ftz`, so a weight or a sum
 * in the subnormal range is kept rather than flushed. Against the brain-float
 * kernel (which flushes) that can only differ where the widened weight, a
 * product or a partial sum is below 2^-126; a brain float cannot even
 * represent some of those weights.
 *
 * # The prompt kernel holds its rows to every pass width
 *
 * A 128 x 128 tile of `out[col][o]`, 32 contraction elements a step - exactly
 * one scale group - through `mma.sync` m16n8k16 with half inputs and
 * single-precision accumulation. The codes go into the tile as their value
 * over 256, which half precision holds exactly; the activations are rounded to
 * halves, as the dense half-precision prompt tier and the NVFP4 prompt stages
 * round them. Each group's 32-term product is formed from zero, scaled by its
 * row's `256 * 2^(s - 127)` in single precision, and added to the running sum
 * in group order. A column's sum reads only its own activation row, in an
 * order fixed by the shape, so a prompt's rows do not depend on where the
 * prompt was cut or what shared its chunk.
 */

#include <stdint.h>
#include <cuda_fp16.h>

/* Warps a step-kernel block runs. */
#define Q4E_MX_WARPS 4u
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

/* One activation row, two output rows a warp, one chunk deep.
 *
 * Measured on a GB10 against one row a warp four chunks deep: 2560 x 6144
 * (out x in) 84 -> 73 us, 10240 x 2560 unchanged at 116 us; two rows a warp
 * halve the warps a wide contraction is cut into, which is what left that
 * shape's last wave of rows short of the memory system. `rows` is accepted
 * for the shared argument list and must be 1.
 *
 * Buffers: out `[1, out_dim]` f32, codes, scales, x `[1, in_dim]` f32. Grid
 * `ceil(out_dim / 8)`, 128 threads. `in_dim` a multiple of 32 and `codes`
 * sixteen-byte aligned. */
extern "C" __global__ void __launch_bounds__(Q4E_MX_WARPS * 32u) q4e_mxfp8_gemv_kernel(
        float *out, const uint8_t *codes, const uint8_t *scales, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    (void) rows;
    __shared__ __align__(16) uint8_t stage[Q4E_MX_WARPS][2u * Q4E_MX_CHUNK];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t row0 = (blockIdx.x * Q4E_MX_WARPS + warp) * 2u;
    if (row0 >= out_dim) {
        return;
    }
    q4e_mx_rows<1, 2, 1>(out, codes, scales, x, in_dim, out_dim, row0, lane, stage[warp]);
}

/* One activation row, one output row a warp, four chunks deep: the narrow
 * outputs (at most 1024 rows), which two rows a warp would spread over half
 * as many warps. Measured on a GB10 at 512 x 2560: 15.1 us against the wide
 * kernel's, 13.5 us here. Geometry as the multi-row kernel. */
extern "C" __global__ void __launch_bounds__(Q4E_MX_WARPS * 32u) q4e_mxfp8_gemv_narrow_kernel(
        float *out, const uint8_t *codes, const uint8_t *scales, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    (void) rows;
    __shared__ __align__(16) uint8_t stage[Q4E_MX_WARPS][Q4E_MX_CHUNK];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t row0 = blockIdx.x * Q4E_MX_WARPS + warp;
    if (row0 >= out_dim) {
        return;
    }
    q4e_mx_rows<1, 1, 4>(out, codes, scales, x, in_dim, out_dim, row0, lane, stage[warp]);
}

/* ---------------------------------------------------------------------------
 * The verify kernels: two to eight activation rows.
 * ------------------------------------------------------------------------ */

__device__ __forceinline__ void q4e_mx_cp16(void *smem, const void *gmem) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(gmem));
}
__device__ __forceinline__ void q4e_mx_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template <int N>
__device__ __forceinline__ void q4e_mx_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/* The widest verify pass, and the activation elements of one staged slab
 * (one code chunk). */
#define Q4E_MX_STEP_ROWS 8u
#define Q4E_MX_SLAB Q4E_MX_CHUNK

/* `NT` activation rows, `R` output rows a warp, every warp of the block
 * walking the contraction in step: the block stages each 512-element slab of
 * all `NT` activation rows in shared memory once (`xs`, `[2][NT][512]` f32,
 * double-buffered behind `cp.async`), and every warp reads its activations
 * from there. The warp-per-row kernel above reads them from the cache once
 * per output row instead, which at five rows and a 6144-wide contraction is
 * 120 KB of activations per 6 KB weight row - more traffic through the
 * load units than the weights themselves.
 *
 * A row's arithmetic is `q4e_mx_rows`'s exactly: lane `l` owns element pairs
 * `l, l + 32, ...` and runs one fused multiply-add per weight per activation
 * row in contraction order, then the same fold. Rows past `out_dim` read the
 * last row and store nothing; every warp still reaches every barrier. */
template <int NT, int R>
__device__ __forceinline__ void q4e_mx_rows_xs(
        float *__restrict__ out, const uint8_t *__restrict__ codes,
        const uint8_t *__restrict__ scales, const float *__restrict__ x,
        uint32_t in_dim, uint32_t out_dim, uint32_t row0, uint32_t lane,
        float *__restrict__ xs, uint8_t *__restrict__ stage) {
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
    const uint32_t nslab = (in_dim + Q4E_MX_SLAB - 1u) / Q4E_MX_SLAB;
    /* Slab `s` of every activation row into buffer `s % 2`; a sixteen-byte
     * piece is wholly inside the row or wholly past it (whole groups). */
    auto stage_x = [&](uint32_t s) {
        float *dst = xs + (s & 1u) * (NT * Q4E_MX_SLAB);
        const uint32_t base = s * Q4E_MX_SLAB;
#pragma unroll
        for (uint32_t i = threadIdx.x; i < NT * Q4E_MX_SLAB / 4u; i += Q4E_MX_WARPS * 32u) {
            const uint32_t t = i / (Q4E_MX_SLAB / 4u);
            const uint32_t k = (i % (Q4E_MX_SLAB / 4u)) * 4u;
            if (base + k < in_dim) {
                q4e_mx_cp16(dst + t * Q4E_MX_SLAB + k, x + (uint64_t) t * in_dim + base + k);
            }
        }
    };
    /* The codes ride one slab ahead in registers. */
    auto load_codes = [&](uint4 *buf, uint32_t base) {
        const uint32_t off = base + 16u * lane;
#pragma unroll
        for (int r = 0; r < R; r++) {
            buf[r] = off < in_dim ? __ldg(reinterpret_cast<const uint4 *>(wr[r] + off))
                                  : make_uint4(0u, 0u, 0u, 0u);
        }
    };
    uint4 buf[R];
    load_codes(buf, 0u);
    stage_x(0u);
    q4e_mx_cp_commit();
    for (uint32_t s = 0; s < nslab; s++) {
        if (s + 1u < nslab) {
            stage_x(s + 1u);
        }
        q4e_mx_cp_commit();
        q4e_mx_cp_wait<1>();
        __syncthreads();
        const uint32_t base = s * Q4E_MX_SLAB;
#pragma unroll
        for (int r = 0; r < R; r++) {
            *reinterpret_cast<uint4 *>(stage + r * Q4E_MX_CHUNK + 16u * lane) = buf[r];
        }
        load_codes(buf, base + Q4E_MX_SLAB);
        __syncwarp();
        const float *xb = xs + (s & 1u) * (NT * Q4E_MX_SLAB);
        /* Pair `j` of this slab for this lane is element
         * `base + 2 lane + 64 j`, as in `q4e_mx_rows`. */
#pragma unroll
        for (uint32_t j = 0; j < Q4E_MX_SLAB / 64u; j++) {
            const uint32_t e = base + 2u * lane + 64u * j;
            if (e < in_dim) {
                float2 xv[NT];
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    xv[t] = *reinterpret_cast<const float2 *>(xb + t * Q4E_MX_SLAB + 2u * lane + 64u * j);
                }
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const uint32_t p = *reinterpret_cast<const uint16_t *>(
                            stage + r * Q4E_MX_CHUNK + 2u * lane + 64u * j);
                    const float sc = q4e_mx_scale(__ldg(sr[r] + e / Q4E_MX_GROUP));
                    const float2 v = q4e_mx_pair(p);
                    const float w0 = q4e_mx_mul(v.x, sc);
                    const float w1 = q4e_mx_mul(v.y, sc);
#pragma unroll
                    for (int t = 0; t < NT; t++) {
                        acc[r][t] = q4e_mx_fma(w0, xv[t].x, acc[r][t]);
                        acc[r][t] = q4e_mx_fma(w1, xv[t].y, acc[r][t]);
                    }
                }
            }
        }
        /* Every warp is done with this slab's activations and its own
         * staged codes before either is overwritten. */
        __syncthreads();
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

/* One verify entry: `R` output rows a warp, the row count dispatched to its
 * instantiation. */
template <int R>
__device__ __forceinline__ void q4e_mx_verify(
        float *out, const uint8_t *codes, const uint8_t *scales, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    __shared__ __align__(16) float xs[2u * Q4E_MX_STEP_ROWS * Q4E_MX_SLAB];
    __shared__ __align__(16) uint8_t stage[Q4E_MX_WARPS][R * Q4E_MX_CHUNK];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    const uint32_t row0 = (blockIdx.x * Q4E_MX_WARPS + warp) * (uint32_t) R;
    uint8_t *st = stage[warp];
    switch (rows) {
    case 2u: q4e_mx_rows_xs<2, R>(out, codes, scales, x, in_dim, out_dim, row0, lane, xs, st); break;
    case 3u: q4e_mx_rows_xs<3, R>(out, codes, scales, x, in_dim, out_dim, row0, lane, xs, st); break;
    case 4u: q4e_mx_rows_xs<4, R>(out, codes, scales, x, in_dim, out_dim, row0, lane, xs, st); break;
    case 5u: q4e_mx_rows_xs<5, R>(out, codes, scales, x, in_dim, out_dim, row0, lane, xs, st); break;
    case 6u: q4e_mx_rows_xs<6, R>(out, codes, scales, x, in_dim, out_dim, row0, lane, xs, st); break;
    case 7u: q4e_mx_rows_xs<7, R>(out, codes, scales, x, in_dim, out_dim, row0, lane, xs, st); break;
    case 8u: q4e_mx_rows_xs<8, R>(out, codes, scales, x, in_dim, out_dim, row0, lane, xs, st); break;
    default: break;
    }
}

/* Two to eight activation rows, `[rows, in_dim]` in and `[rows, out_dim]`
 * out; any other count writes nothing. Four output rows a warp, sixteen a
 * block.
 *
 * Measured on a GB10 (wall clock, median of three runs' best batch) against
 * one row a warp reading its activations from the cache: five rows at
 * 10240 x 2560 126 -> 121 us, at 2560 x 6144 92 -> 79 us, eight rows at
 * 10240 x 2560 144 -> 127 us and at 2560 x 6144 106 -> 83 us - within a few
 * percent of the one-row kernel (120 and 74 us) at five rows, which is the
 * weight read once whatever the pass width.
 *
 * Grid `ceil(out_dim / 16)`, 128 threads. `in_dim` a multiple of 32, `codes`
 * and `x` sixteen-byte aligned. */
extern "C" __global__ void __launch_bounds__(Q4E_MX_WARPS * 32u) q4e_mxfp8_rows_kernel(
        float *out, const uint8_t *codes, const uint8_t *scales, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    q4e_mx_verify<4>(out, codes, scales, x, in_dim, out_dim, rows);
}

/* The same sums, one output row a warp: the narrow outputs (at most 1024
 * rows), which four rows a warp would spread over a quarter of the warps -
 * three blocks at 48 rows. Measured as above at five rows: 48 x 2560
 * 12.4 -> 8.3 us, 640 x 2560 20.4 -> 18.6 us. Grid `ceil(out_dim / 4)`,
 * 128 threads. */
extern "C" __global__ void __launch_bounds__(Q4E_MX_WARPS * 32u) q4e_mxfp8_rows_narrow_kernel(
        float *out, const uint8_t *codes, const uint8_t *scales, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    q4e_mx_verify<1>(out, codes, scales, x, in_dim, out_dim, rows);
}

/* ---------------------------------------------------------------------------
 * The prompt GEMM.
 * ------------------------------------------------------------------------ */

/* A 128 x 128 tile of `out[col][o]`, sixteen warps of 32 x 32. */
#define Q4E_MX_BM 128u
#define Q4E_MX_BN 128u
#define Q4E_MX_GEMM_THREADS 512u
/* Contraction steps (one scale group each) in flight behind the one being
 * multiplied. */
#define Q4E_MX_STAGES 3u
/* Output tiles a raster group spans; see the kernel. */
#define Q4E_MX_GROUP_TILES 16u
#define Q4E_MX_A_RAW (Q4E_MX_BM * Q4E_MX_GROUP)         /* code bytes a step */
#define Q4E_MX_X_RAW (Q4E_MX_BN * Q4E_MX_GROUP * 4u)    /* activation bytes a step */
#define Q4E_MX_TILE (Q4E_MX_BM * Q4E_MX_GROUP * 2u)     /* bytes of one half tile */
/* Dynamic shared memory the kernel takes: the raw ring, two half tiles of
 * each operand, two steps of scales. */
#define Q4E_MX_GEMM_SMEM \
    (Q4E_MX_STAGES * (Q4E_MX_A_RAW + Q4E_MX_X_RAW) + 4u * Q4E_MX_TILE + 2u * Q4E_MX_BM * 4u)

__device__ __forceinline__ void q4e_mx_mma(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/* The same product from zero. */
__device__ __forceinline__ void q4e_mx_mma0(float *c, const uint32_t *a, const uint32_t *b) {
    const float z = 0.0f;
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %10, %10, %10};\n"
            : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "f"(z));
}

__device__ __forceinline__ void q4e_mx_ldsm4(uint32_t *r, const void *p) {
    const uint32_t s = (uint32_t) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(s));
}

/* Four codes (the bytes of `w`) as four halves of value over 256, the even
 * ones low: `q4e_mx_halves`'s bits for every code but the two NaN codes. */
__device__ __forceinline__ uint2 q4e_mx_halves4(uint32_t w) {
    const uint32_t lo = __byte_perm(w, 0u, 0x4140u) << 7;
    const uint32_t hi = __byte_perm(w, 0u, 0x4342u) << 7;
    /* The sign lands on bit 14 of each half; adding that bit to itself
     * carries it to bit 15. */
    return make_uint2(lo + (lo & 0x40004000u), hi + (hi & 0x40004000u));
}

/* Nonzero when a byte of `w` is a NaN code (`0x7F` or `0xFF`). */
__device__ __forceinline__ uint32_t q4e_mx_has_nan(uint32_t w) {
    return ((w & 0x7F7F7F7Fu) + 0x01010101u) & 0x80808080u;
}

/* Sixteen codes as sixteen halves of value over 256. */
__device__ __forceinline__ void q4e_mx_halves16(uint4 q, uint4 &h0, uint4 &h1) {
    if (q4e_mx_has_nan(q.x) | q4e_mx_has_nan(q.y) | q4e_mx_has_nan(q.z) | q4e_mx_has_nan(q.w)) {
        h0 = make_uint4(q4e_mx_halves(q.x & 0xFFFFu), q4e_mx_halves(q.x >> 16),
                        q4e_mx_halves(q.y & 0xFFFFu), q4e_mx_halves(q.y >> 16));
        h1 = make_uint4(q4e_mx_halves(q.z & 0xFFFFu), q4e_mx_halves(q.z >> 16),
                        q4e_mx_halves(q.w & 0xFFFFu), q4e_mx_halves(q.w >> 16));
    } else {
        const uint2 a = q4e_mx_halves4(q.x);
        const uint2 b = q4e_mx_halves4(q.y);
        const uint2 c = q4e_mx_halves4(q.z);
        const uint2 d = q4e_mx_halves4(q.w);
        h0 = make_uint4(a.x, a.y, b.x, b.y);
        h1 = make_uint4(c.x, c.y, d.x, d.y);
    }
}

/* Two activations as a pair of halves, the first low. */
__device__ __forceinline__ uint32_t q4e_mx_h2(float a, float b) {
    const __half2 h = __floats2half2_rn(a, b);
    return *reinterpret_cast<const uint32_t *>(&h);
}

/* Byte offset of sixteen-byte chunk `c` (0..3) of row `r` in a half tile of
 * 64-byte rows, swizzled so the eight rows one ldmatrix phase reads land on
 * eight distinct bank groups without padding. */
__device__ __forceinline__ uint32_t q4e_mx_off(uint32_t r, uint32_t c) {
    return r * 64u + ((c ^ ((r >> 1) & 3u)) << 4);
}

/* `out[col][o] = sum_k w[o][k] x[col][k]` for a prompt chunk of `rows`
 * columns.
 *
 * Each contraction step is one scale group. Its raw codes and activations
 * arrive by `cp.async` three steps deep; the thread that copied a piece
 * converts it (codes to halves of value over 256, activations rounded to
 * halves) into a swizzled half tile, so only its own wait covers the copy and
 * one barrier a step publishes the tiles. Each warp forms the group's 32-term
 * product from zero with two m16n8k16 steps, scales it by its row's
 * `256 * 2^(s - 127)` and adds it to the running sum - the same instructions
 * on the same operands, in the same order, as the 64 x 64 four-warp tile this
 * replaces, so a column's bits are unchanged.
 *
 * Tiles run in raster groups of sixteen output tiles, every column tile of a
 * group before the next group, so the blocks resident at once share a few
 * weight tiles and a few activation tiles instead of streaming the whole
 * weight once per column tile.
 *
 * Measured on a GB10 (wall clock, median of three runs' best batch) at 2048
 * rows, against that 64 x 64 tile: 10240 x 2560 5.41 -> 2.57 ms (20 -> 42
 * TFLOP/s), 12288 x 2560 7.55 -> 3.10 ms, 2560 x 6144 2.21 -> 1.58 ms,
 * 6144 x 2560 2.06 -> 1.54 ms. The device's f16 multiply with f32
 * accumulation measures 58 TFLOP/s in a register-only loop; the per-group
 * scaling and the barrier a step are most of the rest. Tried and dropped:
 * eight warps of 64 x 32 (the same, at twice the registers), two groups a
 * step (spills), staging only through registers (2560 x 6144 at 36 TFLOP/s,
 * its loads exposed), and converting the activations per fragment instead of
 * once a tile (34 TFLOP/s).
 *
 * Buffers: out `[rows, out_dim]` f32, codes, scales, x `[rows, in_dim]` f32.
 * Grid `(ceil(out_dim / 128), ceil(rows / 128))`, 512 threads,
 * `Q4E_MX_GEMM_SMEM` bytes of dynamic shared memory. `in_dim` a multiple of
 * 32, `codes` and `x` sixteen-byte aligned. */
extern "C" __global__ void __launch_bounds__(Q4E_MX_GEMM_THREADS, 1) q4e_mxfp8_gemm_kernel(
        float *out, const uint8_t *codes, const uint8_t *scales, const float *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t rows) {
    extern __shared__ __align__(16) uint8_t q4e_mx_dyn[];
    uint8_t *a_raw = q4e_mx_dyn;                                  /* [STAGES][128][32] */
    uint8_t *x_raw = a_raw + Q4E_MX_STAGES * Q4E_MX_A_RAW;        /* [STAGES][128][32] f32 */
    uint8_t *a16 = x_raw + Q4E_MX_STAGES * Q4E_MX_X_RAW;          /* [2] half tiles */
    uint8_t *b16 = a16 + 2u * Q4E_MX_TILE;                        /* [2] half tiles */
    float *s_s = reinterpret_cast<float *>(b16 + 2u * Q4E_MX_TILE); /* [2][128] */

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint32_t wm = warp & 3u; /* 32 output rows */
    const uint32_t wn = warp >> 2; /* 32 columns */

    const uint32_t out_tiles = gridDim.x;
    const uint32_t col_tiles = gridDim.y;
    const uint32_t lin = blockIdx.y * out_tiles + blockIdx.x;
    const uint32_t per_group = Q4E_MX_GROUP_TILES * col_tiles;
    const uint32_t first = (lin / per_group) * Q4E_MX_GROUP_TILES;
    const uint32_t gsize = min(out_tiles - first, Q4E_MX_GROUP_TILES);
    const uint32_t in_group = lin % per_group;
    const uint32_t row0 = (first + in_group % gsize) * Q4E_MX_BM;
    const uint32_t col0 = (in_group / gsize) * Q4E_MX_BN;
    const uint32_t nk = in_dim / Q4E_MX_GROUP;

    /* Codes: the first 256 threads own sixteen of row `tid / 2`. */
    const bool a_on = tid < 2u * Q4E_MX_BM;
    const uint32_t ar = (tid >> 1) & (Q4E_MX_BM - 1u);
    const uint32_t ac = tid & 1u;
    const bool a_live = a_on && row0 + ar < out_dim;
    const uint8_t *a_src = codes + (uint64_t) min(row0 + ar, out_dim - 1u) * in_dim + ac * 16u;
    /* Activations: every thread owns four of column `tid / 8` and four of
     * column `tid / 8 + 64`. */
    const uint32_t bn = tid >> 3;
    const uint32_t bq = tid & 7u;
    const bool b_live0 = col0 + bn < rows;
    const bool b_live1 = col0 + bn + 64u < rows;
    const float *b_src0 = x + (uint64_t) min(col0 + bn, rows - 1u) * in_dim + bq * 4u;
    const float *b_src1 = x + (uint64_t) min(col0 + bn + 64u, rows - 1u) * in_dim + bq * 4u;
    /* Scales: the first 128 threads own row `tid`'s byte, two steps ahead
     * in registers. */
    const bool s_on = tid < Q4E_MX_BM;
    const bool s_live = s_on && row0 + tid < out_dim;
    const uint8_t *s_src = scales + (uint64_t) min(row0 + tid, out_dim - 1u) * nk;

    /* Rows and columns past the matrix stage zeros once, in every slot; a
     * row past it scales by one. */
#pragma unroll
    for (uint32_t s = 0; s < Q4E_MX_STAGES; s++) {
        if (a_on && !a_live) {
            *reinterpret_cast<uint4 *>(a_raw + s * Q4E_MX_A_RAW + tid * 16u) = make_uint4(0u, 0u, 0u, 0u);
        }
        if (!b_live0) {
            *reinterpret_cast<float4 *>(x_raw + s * Q4E_MX_X_RAW + tid * 16u) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        if (!b_live1) {
            *reinterpret_cast<float4 *>(x_raw + s * Q4E_MX_X_RAW + (tid + Q4E_MX_GEMM_THREADS) * 16u) =
                    make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    auto issue = [&](uint32_t kt) {
        const uint32_t s = kt % Q4E_MX_STAGES;
        const uint32_t k0 = kt * Q4E_MX_GROUP;
        if (a_live) {
            q4e_mx_cp16(a_raw + s * Q4E_MX_A_RAW + tid * 16u, a_src + k0);
        }
        if (b_live0) {
            q4e_mx_cp16(x_raw + s * Q4E_MX_X_RAW + tid * 16u, b_src0 + k0);
        }
        if (b_live1) {
            q4e_mx_cp16(x_raw + s * Q4E_MX_X_RAW + (tid + Q4E_MX_GEMM_THREADS) * 16u, b_src1 + k0);
        }
    };
    uint32_t sb0 = 127u, sb1 = 127u;
    if (s_live) {
        sb0 = __ldg(s_src);
        if (nk > 1u) {
            sb1 = __ldg(s_src + 1);
        }
    }

    float acc[2][4][4];
#pragma unroll
    for (uint32_t i = 0; i < 2u; i++) {
#pragma unroll
        for (uint32_t j = 0; j < 4u; j++) {
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
        }
    }

#pragma unroll
    for (uint32_t s = 0; s + 1u < Q4E_MX_STAGES; s++) {
        if (s < nk) {
            issue(s);
        }
        q4e_mx_cp_commit();
    }
    for (uint32_t kt = 0; kt < nk; kt++) {
        const uint32_t s = kt % Q4E_MX_STAGES;
        const uint32_t ab = kt & 1u;
        q4e_mx_cp_wait<Q4E_MX_STAGES - 2u>();
        /* This thread's pieces of step `kt`, converted into half tile `ab`.
         * Its last readers were step `kt - 2`'s, which every warp finished
         * before the previous barrier. */
        if (a_on) {
            const uint4 q = *reinterpret_cast<const uint4 *>(a_raw + s * Q4E_MX_A_RAW + tid * 16u);
            uint4 h0, h1;
            q4e_mx_halves16(q, h0, h1);
            uint8_t *t = a16 + ab * Q4E_MX_TILE;
            *reinterpret_cast<uint4 *>(t + q4e_mx_off(ar, 2u * ac)) = h0;
            *reinterpret_cast<uint4 *>(t + q4e_mx_off(ar, 2u * ac + 1u)) = h1;
        }
        {
            uint8_t *t = b16 + ab * Q4E_MX_TILE;
            const float4 v0 = *reinterpret_cast<const float4 *>(x_raw + s * Q4E_MX_X_RAW + tid * 16u);
            const float4 v1 = *reinterpret_cast<const float4 *>(
                    x_raw + s * Q4E_MX_X_RAW + (tid + Q4E_MX_GEMM_THREADS) * 16u);
            /* Four halves are half a chunk: chunk `bq / 2`, its `bq % 2` half. */
            *reinterpret_cast<uint2 *>(t + q4e_mx_off(bn, bq >> 1) + (bq & 1u) * 8u) =
                    make_uint2(q4e_mx_h2(v0.x, v0.y), q4e_mx_h2(v0.z, v0.w));
            *reinterpret_cast<uint2 *>(t + q4e_mx_off(bn + 64u, bq >> 1) + (bq & 1u) * 8u) =
                    make_uint2(q4e_mx_h2(v1.x, v1.y), q4e_mx_h2(v1.z, v1.w));
        }
        if (s_on) {
            s_s[ab * Q4E_MX_BM + tid] = s_live ? q4e_mx_scale(sb0) : 1.0f;
            sb0 = sb1;
            if (s_live && kt + 2u < nk) {
                sb1 = __ldg(s_src + kt + 2u);
            }
        }
        __syncthreads();
        /* The slot step `kt + 2` lands in held step `kt - 1`, which only its
         * owners read, before this barrier. */
        if (kt + Q4E_MX_STAGES - 1u < nk) {
            issue(kt + Q4E_MX_STAGES - 1u);
        }
        q4e_mx_cp_commit();

        const uint8_t *as = a16 + ab * Q4E_MX_TILE;
        const uint8_t *bs = b16 + ab * Q4E_MX_TILE;
        /* The group's product from zero, then scaled into the running sum. */
        float part[2][4][4];
#pragma unroll
        for (uint32_t kq = 0; kq < 2u; kq++) {
            uint32_t af[2][4];
            uint32_t bf[4][2];
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; mt++) {
                const uint32_t r = wm * 32u + mt * 16u + (lane & 15u);
                q4e_mx_ldsm4(af[mt], as + q4e_mx_off(r, kq * 2u + (lane >> 4)));
            }
#pragma unroll
            for (uint32_t np = 0; np < 2u; np++) {
                uint32_t rr[4];
                const uint32_t n = wn * 32u + np * 16u + (lane & 7u) + (lane >> 4) * 8u;
                q4e_mx_ldsm4(rr, bs + q4e_mx_off(n, kq * 2u + ((lane >> 3) & 1u)));
                bf[2 * np][0] = rr[0];
                bf[2 * np][1] = rr[1];
                bf[2 * np + 1][0] = rr[2];
                bf[2 * np + 1][1] = rr[3];
            }
#pragma unroll
            for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
                for (uint32_t nt = 0; nt < 4u; nt++) {
                    if (kq == 0u) {
                        q4e_mx_mma0(part[mt][nt], af[mt], bf[nt]);
                    } else {
                        q4e_mx_mma(part[mt][nt], af[mt], bf[nt]);
                    }
                }
            }
        }
        const float *ss = s_s + ab * Q4E_MX_BM;
#pragma unroll
        for (uint32_t mt = 0; mt < 2u; mt++) {
            const uint32_t r = wm * 32u + mt * 16u + gid;
            const float f0 = ss[r];
            const float f1 = ss[r + 8u];
#pragma unroll
            for (uint32_t nt = 0; nt < 4u; nt++) {
#pragma unroll
                for (uint32_t h = 0; h < 4u; h++) {
                    const float scaled = q4e_mx_mul(q4e_mx_mul(part[mt][nt][h], 256.0f), h < 2u ? f0 : f1);
                    acc[mt][nt][h] = q4e_mx_add(acc[mt][nt][h], scaled);
                }
            }
        }
    }
    /* c0, c1 at (row gid, columns 2 tig, +1); c2, c3 eight rows down. */
#pragma unroll
    for (uint32_t mt = 0; mt < 2u; mt++) {
#pragma unroll
        for (uint32_t nt = 0; nt < 4u; nt++) {
#pragma unroll
            for (uint32_t h = 0; h < 4u; h++) {
                const uint32_t col = col0 + wn * 32u + nt * 8u + tig * 2u + (h & 1u);
                const uint32_t o = row0 + wm * 32u + mt * 16u + gid + (h >> 1) * 8u;
                if (col < rows && o < out_dim) {
                    out[(uint64_t) col * out_dim + o] = acc[mt][nt][h];
                }
            }
        }
    }
}
