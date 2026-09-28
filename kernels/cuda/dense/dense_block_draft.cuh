/* The two kernels a block drafter adds to the dense set: the strided row copy
 * that hands the drafter the target's tapped hidden states, and the grouped
 * dynamic causal convolution.
 *
 * Twins of `crates/shiso-metal/metal/dense/dense_block_draft.metal`, same
 * names, argument blocks and flat grids (`ceil(n / 256)` blocks of 256).
 */
#ifndef SHISO_DENSE_BLOCK_DRAFT_CUH
#define SHISO_DENSE_BLOCK_DRAFT_CUH

#include "dense_common.cuh"

struct dense_args_copy_rows {
    uint32_t n_rows;
    uint32_t width;
    uint32_t dst_stride;
    uint32_t dst_offset;
};
static_assert(sizeof(struct dense_args_copy_rows) == 16, "CopyRowsArgs is 16 bytes");

struct dense_args_dyn_conv {
    uint32_t n_rows;
    uint32_t hidden;
    uint32_t group;
    uint32_t taps;
    uint32_t part;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};
static_assert(sizeof(struct dense_args_dyn_conv) == 32, "DynConvArgs is 32 bytes");

/* Row `r` of `src` lands at `dst[r * dst_stride + dst_offset ..]`. */
extern "C" __global__ void kernel_dense_copy_rows_f32(
        const struct dense_args_copy_rows args, const float *__restrict__ src,
        float *__restrict__ dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n_rows * args.width) {
        return;
    }
    const unsigned r = i / args.width;
    const unsigned c = i % args.width;
    dst[(uint64_t) r * args.dst_stride + args.dst_offset + c] = src[(uint64_t) r * args.width + c];
}

/* `out[t][c] = sum over o < taps, o <= t, of
 *              base[part][o][c] * x[t - o][c] + dyn[t][part][o][c / group] * x[t - o][c]`,
 * the static term of an offset added before its dynamic one, offset by offset,
 * as the Metal twin folds them. */
extern "C" __global__ void kernel_dense_dyn_conv_f32(
        const struct dense_args_dyn_conv args, const float *__restrict__ x,
        const float *__restrict__ dyn, const float *__restrict__ base,
        float *__restrict__ out) {
    const unsigned hidden = args.hidden;
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n_rows * hidden) {
        return;
    }
    const unsigned t = i / hidden;
    const unsigned c = i % hidden;
    const unsigned groups = hidden / args.group;
    const unsigned g = c / args.group;
    float acc = 0.0f;
    for (unsigned o = 0; o < args.taps && o <= t; ++o) {
        const float v = x[(uint64_t) (t - o) * hidden + c];
        const float b = base[((uint64_t) args.part * args.taps + o) * hidden + c];
        const float d = dyn[(((uint64_t) t * 2u + args.part) * args.taps + o) * groups + g];
        acc = __fadd_rn(acc, __fmul_rn(b, v));
        acc = __fadd_rn(acc, __fmul_rn(d, v));
    }
    out[i] = acc;
}

#endif
