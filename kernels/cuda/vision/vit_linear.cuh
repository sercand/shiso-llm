/* The tower's projection, its normalization, its two activations and its adds.
 *
 * The CUDA twins of `metal/vision/vit_gemm.metal` and
 * `metal/vision/vit_norm_gelu.metal`. Nothing here is copied from anything; the
 * arithmetic is those two files', which is what the shared oracle in
 * `crates/shiso-kerneltest/src/vision` was written against.
 *
 * The tower normalizes with a mean and a variance, not an RMS, and carries both
 * a scale and an offset. That is the single biggest difference from the
 * language model's blocks, and it is why none of the existing norm kernels could
 * be reused: subtracting the mean is not an optimization away from the RMS form,
 * it is a different function.
 */
#ifndef SHISO_VIT_LINEAR_CUH
#define SHISO_VIT_LINEAR_CUH

#include "vit_common.cuh"

/* `out[m, n] = sum_k in[m, k] * weight[n, k] + bias[n]`.
 *
 * The weight is indexed as `[n][k]` -- output width slowest -- which is how the
 * checkpoint writes a linear layer, so no transpose pass runs anywhere in this
 * tower. The patch projection, the fused qkv, the attention output, both
 * feed-forward matrices and both merger layers all go through this entry point.
 *
 * # Why a staged tile and not a thread per output
 *
 * A thread that owned one output column would read `weight[n * k + ...]` with
 * neighbouring threads a whole row apart, which is the worst access this device
 * has. Staging a VIT_GEMM_TILE square of each operand makes both loads
 * contiguous along `k` and reuses each one across the tile.
 *
 * # The summation order is the contract
 *
 * `k` is walked strictly in increasing order: the tile loop advances by
 * VIT_GEMM_TILE and the inner loop walks the tile front to back, so the sum is
 * the same association a plain `for k in 0..K` loop makes. That is deliberate --
 * the oracle this kernel is held to is exactly that loop -- and a blocking
 * change that reorders `k` is a change to the numbers, not just to the speed.
 *
 * The block is a VIT_GEMM_TILE square and may not be anything else: a thread's
 * tile coordinates are its own `threadIdx`, and no thread returns early,
 * because every one of them reaches the two barriers inside the tile loop. */
extern "C" __global__ void kernel_vit_gemm_bias(
        const struct vit_args_gemm args,
        float *__restrict__ out,
        const float *__restrict__ in,
        const unsigned short *__restrict__ weight,
        const unsigned short *__restrict__ bias) {
    __shared__ float a_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    __shared__ float b_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];

    const unsigned tx = threadIdx.x;
    const unsigned ty = threadIdx.y;
    const unsigned row = blockIdx.y * VIT_GEMM_TILE + ty;
    const unsigned col = blockIdx.x * VIT_GEMM_TILE + tx;
    /* The weight row this thread stages, which is not the one it accumulates:
     * the tile is filled row-per-thread-y so the `k` index runs along x and the
     * load stays contiguous, then read back column-wise in the inner loop. */
    const unsigned stage_n = blockIdx.x * VIT_GEMM_TILE + ty;

    float acc = 0.0f;
    for (unsigned k0 = 0u; k0 < args.k; k0 += VIT_GEMM_TILE) {
        const unsigned k = k0 + tx;
        const bool have_k = k < args.k;
        a_tile[ty][tx] = (row < args.m && have_k)
                ? in[(uint64_t) row * args.k + k]
                : 0.0f;
        b_tile[ty][tx] = (stage_n < args.n && have_k)
                ? vit_bf16(weight + (uint64_t) stage_n * args.k + k)
                : 0.0f;
        __syncthreads();
        for (unsigned kk = 0u; kk < VIT_GEMM_TILE; kk++) {
            acc += a_tile[ty][kk] * b_tile[tx][kk];
        }
        __syncthreads();
    }

    if (row < args.m && col < args.n) {
        out[(uint64_t) row * args.n + col] = acc + vit_bf16(bias + col);
    }
}

/* `out[r, i] = (in[r, i] - mean) * rsqrt(var + eps) * weight[i] + bias[i]`.
 *
 * One block per row, two block reductions: the mean, then the variance about
 * it. The variance pass re-reads the row rather than deriving it from a sum of
 * squares, because the row is already in cache by then and the sum-of-squares
 * form loses precision exactly where a normalization matters -- a row whose
 * values are large and whose variance is small.
 *
 * Safe to run in place (`out == in`): both reductions complete before the write
 * loop starts, and the write loop touches each index once. Neither of the two
 * row pointers is `__restrict__` for that reason -- the aliased launch is a
 * documented call, and telling the compiler the two cannot overlap while a
 * caller overlaps them is undefined behaviour rather than a hint.
 *
 * The dynamic shared allocation is VIT_BLOCK_SUM_SLOTS floats, 132 bytes, which
 * is what both encoders and every case pass. */
extern "C" __global__ void kernel_vit_layer_norm(
        const struct vit_args_layer_norm args,
        float *out,
        const float *in,
        const unsigned short *__restrict__ weight,
        const unsigned short *__restrict__ bias) {
    const unsigned r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    float *scratch = (float *) vit_smem;
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;
    const float *src = in + (uint64_t) r * args.dim;
    float *dst = out + (uint64_t) r * args.dim;

    float sum = 0.0f;
    for (unsigned i = tid; i < args.dim; i += nthreads) {
        sum += src[i];
    }
    const float mean = vit_block_sum(sum, scratch, tid, nthreads) / (float) args.dim;

    float var = 0.0f;
    for (unsigned i = tid; i < args.dim; i += nthreads) {
        const float d = src[i] - mean;
        var += d * d;
    }
    const float rstd = rsqrtf(
            vit_block_sum(var, scratch, tid, nthreads) / (float) args.dim + args.eps);

    for (unsigned i = tid; i < args.dim; i += nthreads) {
        dst[i] = (src[i] - mean) * rstd * vit_bf16(weight + i) + vit_bf16(bias + i);
    }
}

/* The blocks' activation, applied in place to a feed-forward row.
 *
 * The bias is already in `x`: it rides the GEMM's epilogue rather than being
 * added here, which keeps one addition in the accumulator's own precision
 * instead of two passes over the same buffer. */
extern "C" __global__ void kernel_vit_gelu_tanh(
        const struct vit_args_unary args, float *__restrict__ x) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    x[i] = vit_gelu_tanh(x[i]);
}

/* The merger's activation, applied in place.
 *
 * A separate entry point rather than a flag on the one above: the two differ by
 * about 1e-4 per value, the choice belongs to the layer rather than to the
 * launch, and a branch that picked between them would be a branch that could be
 * got wrong at one call site without anything failing loudly. */
extern "C" __global__ void kernel_vit_gelu_erf(
        const struct vit_args_unary args, float *__restrict__ x) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    x[i] = vit_gelu_erf(x[i]);
}

/* `x[i] += add[i % period]`, in place.
 *
 * Both residual adds use it with `period == count`, which is the plain
 * elementwise form. The position embedding uses a shorter period: a clip's
 * temporal groups sit on one patch grid, so one grid's worth of resampled
 * vectors is added to every group rather than being replicated on the host and
 * uploaded several times. */
extern "C" __global__ void kernel_vit_add_inplace(
        const struct vit_args_add args,
        float *__restrict__ x,
        const float *__restrict__ add) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    x[i] += add[i % args.period];
}

#endif /* SHISO_VIT_LINEAR_CUH */
