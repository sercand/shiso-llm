/* A few activation rows times a `q4_0` or `q8_0` weight: the projections of a
 * block drafter, whose passes are eight rows wide.
 *
 * The extended matvec family reads these shapes at 60-130 GB/s on a GB10
 * (`kernel_mul_mv_ext_q4_0_f32_r1_4_nx8`, 88 us a call on average over the
 * drafter's projections). Here a warp owns two output rows and walks the
 * contraction 128 weights at a time, lane `l` on weights 4l .. 4l + 3: it
 * decodes those four of both rows once and applies them to every activation
 * row, whose four floats the warp reads as one coalesced 512-byte load. A
 * lane on a quant block of its own instead read the activations 128 bytes
 * apart, 32 cache lines a load, and ran at a fraction of this.
 *
 * Every accumulator adds its lane's quant blocks in increasing order and the
 * warp folds lanes in a fixed butterfly, so a row's output is the same bits
 * whatever else shares the launch. It is NOT the extended family's bits:
 * which is why the engine reaches this only for weights that opted in.
 *
 * `_r2`, `_r4`, `_r8`: output rows a warp owns. A row's sum is the same
 * whichever owns it - its lanes, its order - so the engine picks by the
 * output's width: more rows a warp reads the activations fewer times, fewer
 * keep a narrow output on enough blocks.
 *
 * Grid `ceil(out / (4 * rows a warp))` blocks of 128 threads. `n_rows` is at most 8, and every
 * activation row starts sixteen-byte aligned (`in_dim` is a multiple of 32).
 */
#ifndef SHISO_DENSE_ROWS_MV_CUH
#define SHISO_DENSE_ROWS_MV_CUH

#include "dense_common.cuh"

struct dense_args_rows_mv {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_rows;
    uint32_t pad0;
};
static_assert(sizeof(struct dense_args_rows_mv) == 16, "RowsMvArgs is 16 bytes");

#define DENSE_ROWS_MV_MAX 8u
#define DENSE_ROWS_MV_WARPS 4u

/* Which block format a decode reads, as a type. */
template <unsigned BYTES>
struct dense_rows_mv_tag {};

/* The four weights `4m .. 4m + 3` of one quant block, decoded: `q4_0` keeps
 * weights 0-15 in the low nibbles of bytes 0-15 and 16-31 in their high
 * nibbles, `q8_0` one signed byte each, both behind an f16 scale. */
static __device__ __forceinline__ void dense_rows_mv_four(const uint8_t *blk, unsigned m,
                                                          const dense_rows_mv_tag<18> &,
                                                          float w[4]) {
    const uint16_t *h = reinterpret_cast<const uint16_t *>(blk);
    const float d = __half2float(__ushort_as_half(h[0]));
    const unsigned at = 1u + 2u * (m & 3u);
    const unsigned bytes = (unsigned) h[at] | ((unsigned) h[at + 1u] << 16);
    const unsigned shift = m < 4u ? 0u : 4u;
#pragma unroll
    for (unsigned e = 0; e < 4u; ++e) {
        w[e] = (float) ((int) ((bytes >> (8u * e + shift)) & 0x0fu) - 8) * d;
    }
}

static __device__ __forceinline__ void dense_rows_mv_four(const uint8_t *blk, unsigned m,
                                                          const dense_rows_mv_tag<34> &,
                                                          float w[4]) {
    const uint16_t *h = reinterpret_cast<const uint16_t *>(blk);
    const float d = __half2float(__ushort_as_half(h[0]));
    const unsigned at = 1u + 2u * m;
    const unsigned bytes = (unsigned) h[at] | ((unsigned) h[at + 1u] << 16);
#pragma unroll
    for (unsigned e = 0; e < 4u; ++e) {
        w[e] = (float) (int8_t) ((bytes >> (8u * e)) & 0xffu) * d;
    }
}

template <unsigned BYTES, unsigned DENSE_ROWS_MV_PER_WARP>
static __device__ __forceinline__ void dense_rows_mv(
        const struct dense_args_rows_mv args, const uint8_t *__restrict__ weight,
        const float *__restrict__ x, float *__restrict__ out) {
    const unsigned lane = threadIdx.x & 31u;
    const unsigned warp = threadIdx.x >> 5;
    const unsigned n_rows = args.n_rows;
    const unsigned blocks = args.in_dim / 32u;
    const uint64_t row_bytes = (uint64_t) blocks * BYTES;
    const unsigned first = (blockIdx.x * DENSE_ROWS_MV_WARPS + warp) * DENSE_ROWS_MV_PER_WARP;
    /* Lane `l` of a span of 128 weights takes weights 4l .. 4l + 3: quant
     * block l / 8 of the span, its quarter-row l % 8. */
    const unsigned sub = lane >> 3;
    const unsigned m = lane & 7u;
    const dense_rows_mv_tag<BYTES> tag;
    float acc[DENSE_ROWS_MV_PER_WARP][DENSE_ROWS_MV_MAX];
#pragma unroll
    for (unsigned r = 0; r < DENSE_ROWS_MV_PER_WARP; ++r) {
#pragma unroll
        for (unsigned n = 0; n < DENSE_ROWS_MV_MAX; ++n) {
            acc[r][n] = 0.0f;
        }
    }
    for (unsigned b0 = 0; b0 < blocks; b0 += 4u) {
        const unsigned blk = b0 + sub;
        if (blk >= blocks) {
            continue;
        }
        /* A row past the output reads row zero, which exists, and is never
         * stored. */
        float wv[DENSE_ROWS_MV_PER_WARP][4];
#pragma unroll
        for (unsigned r = 0; r < DENSE_ROWS_MV_PER_WARP; ++r) {
            const unsigned o = first + r < args.out_dim ? first + r : 0u;
            dense_rows_mv_four(weight + (uint64_t) o * row_bytes + (uint64_t) blk * BYTES, m, tag,
                               wv[r]);
        }
#pragma unroll
        for (unsigned n = 0; n < DENSE_ROWS_MV_MAX; ++n) {
            if (n < n_rows) {
                const float4 xv = *reinterpret_cast<const float4 *>(
                    x + (uint64_t) n * args.in_dim + (uint64_t) b0 * 32u + 4u * lane);
#pragma unroll
                for (unsigned r = 0; r < DENSE_ROWS_MV_PER_WARP; ++r) {
                    float a = acc[r][n];
                    a = fmaf(wv[r][0], xv.x, a);
                    a = fmaf(wv[r][1], xv.y, a);
                    a = fmaf(wv[r][2], xv.z, a);
                    a = fmaf(wv[r][3], xv.w, a);
                    acc[r][n] = a;
                }
            }
        }
    }
#pragma unroll
    for (unsigned r = 0; r < DENSE_ROWS_MV_PER_WARP; ++r) {
#pragma unroll
        for (unsigned n = 0; n < DENSE_ROWS_MV_MAX; ++n) {
            float v = acc[r][n];
#pragma unroll
            for (unsigned off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            acc[r][n] = v;
        }
    }
    if (lane == 0u) {
#pragma unroll
        for (unsigned r = 0; r < DENSE_ROWS_MV_PER_WARP; ++r) {
            const unsigned o = first + r;
            if (o >= args.out_dim) {
                continue;
            }
#pragma unroll
            for (unsigned n = 0; n < DENSE_ROWS_MV_MAX; ++n) {
                if (n < n_rows) {
                    out[(uint64_t) n * args.out_dim + o] = acc[r][n];
                }
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(128) kernel_dense_rows_mv_q4_0_f32_r2(
        const struct dense_args_rows_mv args, const uint8_t *__restrict__ weight,
        const float *__restrict__ x, float *__restrict__ out) {
    dense_rows_mv<18, 2>(args, weight, x, out);
}

extern "C" __global__ void __launch_bounds__(128) kernel_dense_rows_mv_q4_0_f32_r4(
        const struct dense_args_rows_mv args, const uint8_t *__restrict__ weight,
        const float *__restrict__ x, float *__restrict__ out) {
    dense_rows_mv<18, 4>(args, weight, x, out);
}

extern "C" __global__ void __launch_bounds__(128) kernel_dense_rows_mv_q4_0_f32_r8(
        const struct dense_args_rows_mv args, const uint8_t *__restrict__ weight,
        const float *__restrict__ x, float *__restrict__ out) {
    dense_rows_mv<18, 8>(args, weight, x, out);
}

extern "C" __global__ void __launch_bounds__(128) kernel_dense_rows_mv_q8_0_f32_r2(
        const struct dense_args_rows_mv args, const uint8_t *__restrict__ weight,
        const float *__restrict__ x, float *__restrict__ out) {
    dense_rows_mv<34, 2>(args, weight, x, out);
}

extern "C" __global__ void __launch_bounds__(128) kernel_dense_rows_mv_q8_0_f32_r4(
        const struct dense_args_rows_mv args, const uint8_t *__restrict__ weight,
        const float *__restrict__ x, float *__restrict__ out) {
    dense_rows_mv<34, 4>(args, weight, x, out);
}

extern "C" __global__ void __launch_bounds__(128) kernel_dense_rows_mv_q8_0_f32_r8(
        const struct dense_args_rows_mv args, const uint8_t *__restrict__ weight,
        const float *__restrict__ x, float *__restrict__ out) {
    dense_rows_mv<34, 8>(args, weight, x, out);
}

#endif
