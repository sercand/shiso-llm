/* The plain (unquantized) matvec, in the two instantiations the engine can
 * name on this device.
 *
 * Nothing here is copied. What is preserved from the Metal twin in
 * `crates/shiso-metal/metal/generic/mul_mv_dense.metal` is the contract: the
 * entry name, the `MulMvArgs` block, the buffer order, the `(32, nsg)` block
 * with the output rows divided by `nr0` on `x`, the sixteen-element run each
 * lane takes, the scalar tail over whatever the block tiling did not cover, and
 * the reduction - a whole-warp fold, one partial per warp into shared memory,
 * then a second fold - which is `helper_mv_reduce_and_write` step for step.
 *
 * # Why only the `_4` spellings are here, and why there are now two of them
 *
 * `crates/shiso-engine/src/launch.rs`'s `plain_mv` picks `_short` below a width
 * of 32, the `_4` variant when the width is a multiple of four, and the wide
 * one otherwise. Every unquantized weight any CUDA family here reaches has an
 * input width of the hidden size, which is a multiple of four in all of them,
 * so `_short` and the wide spelling still have no caller - and a spelling with
 * no caller would be a module claiming coverage it has no oracle for.
 *
 * The `f32` twin arrived with `qwen35moe`, which is the first CUDA family
 * with an unquantized weight that is not half precision: its router, its
 * shared expert's scalar gate and the recurrent block's two drivers are all
 * `BF16` in the file, no projection kernel here reads `BF16`, and widening
 * them lands on `f32`. `plain_mv` has named `kernel_mul_mv_f32_f32_4` for as
 * long as it has existed and nothing on this backend had it, which the driver
 * reported as `CUDA_ERROR_NOT_FOUND` at the first router of the first layer.
 *
 * The body is one template over the weight's element type now rather than two
 * copies of it. The two differ in one expression - how a stored weight becomes
 * a float - and a copy would have been two places to fix a reduction order.
 *
 * # The simdgroup count is read from the launch, not from a constant
 *
 * Metal takes it as a function constant. `crates/shiso-cuda/src/launch.rs`
 * refuses a launch-time constant outright, and it does not need one: the count
 * is the block's own second extent, which the host already sets from the same
 * number (`plain_block(nsg)`). Reading `blockDim.y` is reading the host's
 * choice rather than restating it.
 */
#ifndef SHISO_MUL_MV_DENSE_CUH
#define SHISO_MUL_MV_DENSE_CUH

#include "generic.cuh"

/* Elements one block of the k tiling covers. */
#define SHISO_MV_NB 32
/* Elements one lane takes per turn. */
#define SHISO_MV_NF 16
/* Four-element groups in that run. */
#define SHISO_MV_NF4 (SHISO_MV_NF / 4)

/* One stored weight as a float. Two overloads, which is the whole of the
 * difference between the two entries below. */
__device__ __forceinline__ float shiso_mv_weight(__half v) {
    return __half2float(v);
}
__device__ __forceinline__ float shiso_mv_weight(float v) {
    return v;
}

/* The whole-warp fold, one partial per warp, then the fold again.
 *
 * `helper_mv_reduce_and_write` in `metal/generic/mul_mv_q8_0.metal`, which both
 * plain families share there and both would share here. `shmem` is `nr0` runs
 * of one float per lane, which is the `PLAIN_SMEM_BYTES` the host asks for. */
template <int nr0>
__device__ __forceinline__ void shiso_mv_reduce_and_write(
        float *__restrict__ dst_f32, float *sumf, int r0, int ne01, unsigned lane,
        unsigned warp, unsigned nsg, float *shmem) {
#pragma unroll
    for (int row = 0; row < nr0; ++row) {
        if (warp == 0u) {
            shmem[32 * row + lane] = 0.0f;
        }
        sumf[row] = shiso_warp_all_sum(sumf[row]);
    }
    __syncthreads();
#pragma unroll
    for (int row = 0; row < nr0; ++row) {
        if (lane == 0u) {
            shmem[32 * row + warp] = sumf[row];
        }
    }
    __syncthreads();
    (void) nsg;
#pragma unroll
    for (int row = 0; row < nr0; ++row) {
        if (r0 + row >= ne01) {
            continue;
        }
        const float tot = shiso_warp_all_sum(shmem[32 * row + lane]);
        if (lane == 0u && warp == 0u) {
            dst_f32[r0 + row] = tot;
        }
    }
}

template <typename W, int nr0>
__device__ __forceinline__ void shiso_mul_mv_dense_4_impl(
        const struct shiso_args_mul_mv args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    float *shmem = (float *) shiso_generic_smem;
    const unsigned lane = threadIdx.x;
    const unsigned warp = threadIdx.y;
    const int nsg = (int) blockDim.y;

    const int nb = args.ne00 / SHISO_MV_NB;

    const int r0 = (int) blockIdx.x * nr0;
    const int r1 = (int) blockIdx.y;
    const int im = (int) blockIdx.z;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t offset1 = (uint64_t) r1 * args.nb11 + (uint64_t) i12 * args.nb12
                           + (uint64_t) i13 * args.nb13;
    const float *y = (const float *) (src1 + offset1);

    const W *ax[nr0];
#pragma unroll
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (uint64_t) (r0 + row) * args.nb01
                               + (uint64_t) (i12 / args.r2) * args.nb02
                               + (uint64_t) (i13 / args.r3) * args.nb03;
        ax[row] = (const W *) (src0 + offset0);
    }

    float sumf[nr0];
#pragma unroll
    for (int row = 0; row < nr0; ++row) {
        sumf[row] = 0.0f;
    }

    /* Two lanes share each sixteen-element run: `ix` picks which block of the k
     * tiling the lane's warp starts at and `il` which half of that block's
     * thirty-two elements it takes. */
    const int ix = (int) lane / (32 / SHISO_MV_NF);
    const int il = (int) lane % (32 / SHISO_MV_NF);
    const int ib0 = (int) warp * SHISO_MV_NF + ix;

    for (int ib = ib0; ib < nb; ib += nsg * SHISO_MV_NF) {
        const int at = ib * SHISO_MV_NB + il * SHISO_MV_NF;
        float yl[SHISO_MV_NF];
#pragma unroll
        for (int i = 0; i < SHISO_MV_NF; ++i) {
            yl[i] = y[at + i];
        }
#pragma unroll
        for (int row = 0; row < nr0; ++row) {
            const W *xb = ax[row] + at;
            float sumq = 0.0f;
#pragma unroll
            for (int g = 0; g < SHISO_MV_NF4; ++g) {
                sumq += shiso_mv_weight(xb[4 * g + 0]) * yl[4 * g + 0]
                      + shiso_mv_weight(xb[4 * g + 1]) * yl[4 * g + 1]
                      + shiso_mv_weight(xb[4 * g + 2]) * yl[4 * g + 2]
                      + shiso_mv_weight(xb[4 * g + 3]) * yl[4 * g + 3];
            }
            sumf[row] += sumq;
        }
    }

    /* Whatever the block tiling left: a row width that is not a whole number of
     * `SHISO_MV_NB` has a tail, and it is summed one element per lane. */
    for (int i = nb * SHISO_MV_NB + (int) warp * 32 + (int) lane; i < args.ne00;
         i += 32 * nsg) {
#pragma unroll
        for (int row = 0; row < nr0; ++row) {
            sumf[row] += shiso_mv_weight(ax[row][i]) * y[i];
        }
    }

    float *dst_f32 = (float *) dst
                   + (uint64_t) im * (uint64_t) args.ne0 * (uint64_t) args.ne1
                   + (uint64_t) r1 * (uint64_t) args.ne0;
    shiso_mv_reduce_and_write<nr0>(dst_f32, sumf, r0, args.ne01, lane, warp,
                                   (unsigned) nsg, shmem);
}

/* `dst = W x`, W in half and x in single, one output row block per `x` block.
 *
 * `nr0` is a launch argument rather than a name suffix, exactly as the Metal
 * twin's `kernel_mul_mv_t_t_4_disp` switches on it, and the two values the host
 * ever sets are the two arms here. A third value falls through and writes
 * nothing, which is the arm a host that invented a row blocking would meet.
 */
extern "C" __global__ void kernel_mul_mv_f16_f32_4(
        const struct shiso_args_mul_mv args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    switch (args.nr0) {
        case 2:
            shiso_mul_mv_dense_4_impl<__half, 2>(args, src0, src1, dst);
            break;
        case 4:
            shiso_mul_mv_dense_4_impl<__half, 4>(args, src0, src1, dst);
            break;
        default:
            break;
    }
}

/* The same, for a weight stored in single precision.
 *
 * The `qwen35moe` family's router, its shared expert's scalar gate and the
 * recurrent block's two drivers all arrive here: they are `BF16` in the file,
 * no projection kernel in this build reads `BF16`, and the loader widens them
 * to `f32`. */
extern "C" __global__ void kernel_mul_mv_f32_f32_4(
        const struct shiso_args_mul_mv args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    switch (args.nr0) {
        case 2:
            shiso_mul_mv_dense_4_impl<float, 2>(args, src0, src1, dst);
            break;
        case 4:
            shiso_mul_mv_dense_4_impl<float, 4>(args, src0, src1, dst);
            break;
        default:
            break;
    }
}

#endif /* SHISO_MUL_MV_DENSE_CUH */
