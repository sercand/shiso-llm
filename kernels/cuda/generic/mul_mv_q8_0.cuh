/* The `q8_0` plain matvec, which is the one decode path this codec has.
 *
 * `crates/shiso-engine/src/launch.rs`'s `ext_name_for` returns `None` for
 * `q8_0` on a backend that specializes by name, and `plain_mv` answers
 * `kernel_mul_mv_q8_0_f32` for that codec at every width. So on CUDA this entry
 * is not a fallback: below the GEMM's two-token threshold it is the only way a
 * `q8_0` weight is read at all, and a qwen35 `Q4_K_M` container keeps 48 of its
 * tensors here - `ssm_alpha` and `ssm_beta`, one pair per recurrent layer.
 *
 * Nothing here is copied. What is preserved from the Metal twin in
 * `crates/shiso-metal/metal/generic/mul_mv_q8_0.metal` is the contract: the
 * entry name, the `MulMvArgs` block, the buffer order, the `(32, nsg)` block
 * with the output rows divided by two on `x`, the eight-element run each lane
 * takes, and the reduction - a whole-warp fold, one partial per warp into
 * shared memory, then a second fold - which is `helper_mv_reduce_and_write`
 * step for step and is shared with the dense family here as it is there.
 *
 * # Two output rows, not a launch argument
 *
 * Metal's `N_R0_Q8_0` is 2 and the host's `PLAIN_ROWS` is 2, and this codec has
 * no second spelling on either side: the grid's `x` extent divides the output
 * width by that number, so a kernel that produced a different count would write
 * a band of the output and leave the rest of it holding whatever was there.
 * It is a constant here rather than a `switch` on `args.nr0` for that reason -
 * the dense family switches because its host picks between two values, and this
 * one's host does not.
 *
 * # The simdgroup count is read from the launch, not from a constant
 *
 * The same argument `mul_mv_dense.cuh` makes: Metal takes it as a function
 * constant, `crates/shiso-cuda/src/launch.rs` refuses a launch-time constant
 * outright, and the count is the block's own second extent, which the host
 * already sets from the same number (`plain_block(Q8_MV_NSG)`).
 *
 * # No tail loop
 *
 * The Metal twin has none either, and cannot need one: a `q8_0` row is a whole
 * number of 32-element blocks by the codec's own definition, so `ne00 / QK8_0`
 * covers every weight of the row.
 */
#ifndef SHISO_MUL_MV_Q8_0_CUH
#define SHISO_MUL_MV_Q8_0_CUH

#include "generic.cuh"
#include "mul_mv_dense.cuh"

/* Output rows one threadgroup produces: `N_R0_Q8_0` there, `PLAIN_ROWS` here. */
#define SHISO_MV_Q8_NR0 2
/* Elements one lane takes per turn. */
#define SHISO_MV_Q8_NQ 8

/* `dst = W x`, W in q8_0 and x in single, two output rows per block. */
extern "C" __global__ void kernel_mul_mv_q8_0_f32(
        const struct shiso_args_mul_mv args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    float *shmem = (float *) shiso_generic_smem;
    const unsigned lane = threadIdx.x;
    const unsigned warp = threadIdx.y;
    const int nsg = (int) blockDim.y;

    const int nb = args.ne00 / SHISO_QK8_0;

    const int r0 = (int) blockIdx.x * SHISO_MV_Q8_NR0;
    const int r1 = (int) blockIdx.y;
    const int im = (int) blockIdx.z;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    const uint64_t offset1 = (uint64_t) r1 * args.nb11 + (uint64_t) i12 * args.nb12
                           + (uint64_t) i13 * args.nb13;
    const float *y = (const float *) (src1 + offset1);

    const struct shiso_block_q8_0 *ax[SHISO_MV_Q8_NR0];
#pragma unroll
    for (int row = 0; row < SHISO_MV_Q8_NR0; ++row) {
        const uint64_t offset0 = (uint64_t) (r0 + row) * args.nb01
                               + (uint64_t) (i12 / args.r2) * args.nb02
                               + (uint64_t) (i13 / args.r3) * args.nb03;
        ax[row] = (const struct shiso_block_q8_0 *) (src0 + offset0);
    }

    float sumf[SHISO_MV_Q8_NR0];
#pragma unroll
    for (int row = 0; row < SHISO_MV_Q8_NR0; ++row) {
        sumf[row] = 0.0f;
    }

    /* Four lanes share each block: `ix` picks which block of the k tiling the
     * lane's warp starts at and `il` which eighth of that block's thirty-two
     * elements it takes. */
    const int ix = (int) lane / (32 / SHISO_MV_Q8_NQ);
    const int il = (int) lane % (32 / SHISO_MV_Q8_NQ);
    const int ib0 = (int) warp * SHISO_MV_Q8_NQ + ix;

    float yl[SHISO_MV_Q8_NQ];
    const float *yb = y + (int64_t) ib0 * SHISO_QK8_0 + il * SHISO_MV_Q8_NQ;

    for (int ib = ib0; ib < nb; ib += nsg * SHISO_MV_Q8_NQ) {
#pragma unroll
        for (int i = 0; i < SHISO_MV_Q8_NQ; ++i) {
            yl[i] = yb[i];
        }
#pragma unroll
        for (int row = 0; row < SHISO_MV_Q8_NR0; ++row) {
            const int8_t *qs = ax[row][ib].qs + il * SHISO_MV_Q8_NQ;
            float sumq = 0.0f;
#pragma unroll
            for (int i = 0; i < SHISO_MV_Q8_NQ; ++i) {
                sumq += (float) qs[i] * yl[i];
            }
            sumf[row] += sumq * __half2float(ax[row][ib].d);
        }
        yb += (int64_t) nsg * SHISO_MV_Q8_NQ * SHISO_QK8_0;
    }

    float *dst_f32 = (float *) dst
                   + (uint64_t) im * (uint64_t) args.ne0 * (uint64_t) args.ne1
                   + (uint64_t) r1 * (uint64_t) args.ne0;
    shiso_mv_reduce_and_write<SHISO_MV_Q8_NR0>(dst_f32, sumf, r0, args.ne01, lane, warp,
                                               (unsigned) nsg, shmem);
}

#endif /* SHISO_MUL_MV_Q8_0_CUH */
