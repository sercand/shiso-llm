/* The dense set's small kernels: the gated activation, the scaled residual
 * close, the softcap and the row scale.
 *
 * Each is the CUDA twin of an entry in `metal/dense/dense_misc.metal`, kept
 * apart from the generic set for the reason the Metal ones are: the generic
 * entries are what a qwen35 decode dispatches, and a flag on those would be a
 * branch that could be got wrong at one call site with nothing failing loudly.
 */
#ifndef SHISO_DENSE_MISC_CUH
#define SHISO_DENSE_MISC_CUH

#include "dense_common.cuh"

/* A whole-warp butterfly sum: every lane ends up holding the total.
 *
 * `kernel_dense_add_scale_rms_norm_mul_f32_4` below keeps its generic twin's
 * `simd_sum` reduction rather than the dense set's fixed-order
 * `dense_block_sum`, exactly as the Metal body does and for the same reason: it
 * is `kernel_add_rms_norm_mul_f32_4` plus one multiply, and staying the twin it
 * was copied from outranks the set's convention there. This is that reduction
 * and nothing else reaches for it. */
__device__ __forceinline__ float dense_warp_all_sum(float v) {
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_xor_sync(DENSE_FULL_WARP, v, off);
    }
    return v;
}

/* `dst[i] = dense_gelu_tanh(gate[i]) * up[i] * alpha`, flat over `ne0`.
 *
 * The gated GELU, which is what a Gemma-shaped feed-forward uses where qwen35
 * uses the gated SiLU. Same argument block, same clamp handling and the same
 * flat index as `kernel_swiglu_flat_f32`. */
extern "C" __global__ void kernel_dense_geglu_flat_f32(
        const struct dense_args_glu args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (unsigned) args.ne0) {
        return;
    }

    const float *src0_f32 = (const float *) src0 + args.i00;
    const float *src1_f32 = (const float *) src1 + args.i10;
    float *dst_f32 = (float *) dst;

    float x0 = src0_f32[i];
    float x1 = src1_f32[i];
    if (args.limit > 1.0e-6f) {
        x0 = fminf(x0, args.limit);
        x1 = fminf(fmaxf(x1, -args.limit), args.limit);
    }

    dst_f32[i] = dense_gelu_tanh(x0) * x1 * args.alpha;
}

/* `sum = (a + b) * scale`, then `norm = sum * rsqrt(mean(sum^2) + eps) * w`.
 *
 * `kernel_add_rms_norm_mul_f32_4` with one difference and no others: the scale
 * is applied to the residual sum before BOTH the store and the sum of squares,
 * so the row the next block adds into carries it and the row the next block
 * reads is normalized against the scaled magnitudes. The scale rides in its own
 * parameter rather than in the argument block because the block is the generic
 * norm's, shared with every other norm entry.
 *
 * One warp per row, as the Metal twin is launched: the work loop advances by
 * `blockDim.x` alone and never reads a warp index, so a taller block would have
 * every extra warp recompute and rewrite the same row. */
extern "C" __global__ void kernel_dense_add_scale_rms_norm_mul_f32_4(
        const struct dense_args_norm args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        const char *__restrict__ weight,
        char *__restrict__ sum_dst,
        char *__restrict__ norm_dst,
        const float scale) {
    float *shmem_f32 = (float *) dense_smem;
    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    if (warp == 0u) {
        shmem_f32[lane] = 0.0f;
    }

    const int i01 = (int) blockIdx.x;
    const int i02 = (int) blockIdx.y;
    const int i03 = (int) blockIdx.z;
    const uint64_t in_at = (uint64_t) i03 * args.nbf3[0] + (uint64_t) i02 * args.nbf2[0]
                         + (uint64_t) i01 * args.nbf1[0];
    const uint64_t w_at = (uint64_t) (i03 % args.nef3[1]) * args.nbf3[1]
                        + (uint64_t) (i02 % args.nef2[1]) * args.nbf2[1]
                        + (uint64_t) (i01 % args.nef1[1]) * args.nbf1[1];
    const uint64_t out_at = (uint64_t) i03 * args.nb3 + (uint64_t) i02 * args.nb2
                          + (uint64_t) i01 * args.nb1;

    const float4 *a = (const float4 *) (src0 + in_at);
    const float4 *b = (const float4 *) (src1 + in_at);
    const float4 *w = (const float4 *) (weight + w_at);
    float4 *sum = (float4 *) (sum_dst + out_at);
    float4 *norm = (float4 *) (norm_dst + out_at);

    float sumf = 0.0f;
    for (int i00 = (int) tid; i00 < args.ne00_t; i00 += (int) blockDim.x) {
        const float4 av = a[i00];
        const float4 bv = b[i00];
        const float4 v = make_float4((av.x + bv.x) * scale, (av.y + bv.y) * scale,
                                     (av.z + bv.z) * scale, (av.w + bv.w) * scale);
        sum[i00] = v;
        sumf += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    sumf = dense_warp_all_sum(sumf);
    __syncthreads();
    if (lane == 0u) {
        shmem_f32[warp] = sumf;
    }
    __syncthreads();
    sumf = dense_warp_all_sum(shmem_f32[lane]);

    const float mean = sumf / (float) args.ne00;
    const float norm_scale = 1.0f / sqrtf(mean + args.eps);
    for (int i00 = (int) tid; i00 < args.ne00_t; i00 += (int) blockDim.x) {
        const float4 v = sum[i00];
        const float4 wv = w[i00];
        norm[i00] = make_float4(v.x * norm_scale * wv.x, v.y * norm_scale * wv.y,
                                v.z * norm_scale * wv.z, v.w * norm_scale * wv.w);
    }
}

/* `x[i] = cap * tanh(x[i] / cap)`, in place over the first `n` elements.
 *
 * Gemma's logit softcap: a smooth clamp that leaves the middle of the range
 * nearly untouched and bounds the tails. */
extern "C" __global__ void kernel_dense_softcap_f32(
        const struct dense_args_softcap args, float *__restrict__ x) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n) {
        return;
    }
    x[i] = args.cap * tanhf(x[i] / args.cap);
}

/* `x[i] *= s`, in place over the first `n` elements.
 *
 * The embedding scale and the per-layer scalar are both one multiply over a run
 * of rows, so both reach the device through this. */
extern "C" __global__ void kernel_dense_scale_f32(
        const struct dense_args_scale args, float *__restrict__ x) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n) {
        return;
    }
    x[i] = x[i] * args.s;
}

#endif /* SHISO_DENSE_MISC_CUH */
