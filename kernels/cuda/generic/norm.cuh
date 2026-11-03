/* The two fused RMS norms every dense slot passes through.
 *
 * Nothing here is copied. What is preserved from the Metal twins in
 * `kernels/metal/generic/norm.metal` is the contract: the entry
 * names, the argument block, the buffer order, one warp per row, and - the part
 * that is arithmetic rather than plumbing - where the reduction folds. Both
 * kernels reduce with a whole-warp butterfly, spill one partial per warp to
 * shared memory and butterfly again, exactly as `simd_sum` does either side of
 * the Metal barrier, so a golden that pins a decode step pins the same order of
 * partial sums on both devices.
 *
 * # Why these live in the `generic` module and not in `dense`
 *
 * Because the engine dispatches them there:
 * `crates/shiso-engine/src/layers.rs` names both under `launch::GENERIC`, and
 * on this backend a module name is the key of a separately compiled image
 * rather than a label. A twin of a generic entry has to be in the image called
 * `generic` whatever family asks for it. The scaled variant a Gemma slot closes
 * with is a different entry in the `dense` module
 * (`kernel_dense_add_scale_rms_norm_mul_f32_4`) and is not affected.
 *
 * # The fuse depth is compiled in, and the unread binding still exists
 *
 * The Metal side is one template over a fuse depth with two operand buffers
 * declared whatever depth it was built at. `kernel_rms_norm_mul_f32_4` is the
 * depth-two instantiation: it never dereferences the second operand, and the
 * host binds `weight` twice so the destination lands in the slot it belongs in
 * (`crates/shiso-engine/src/layers.rs` says so at the call site). This twin
 * declares that parameter for the same reason and reads it for none.
 */
#ifndef SHISO_NORM_CUH
#define SHISO_NORM_CUH

#include "generic.cuh"

/* `dst = x * rsqrt(mean(x^2) + eps) * weight`, over one row per block.
 *
 * The one norm in a dense pass with no residual behind it: the first slot of a
 * chunk normalizes the embedding it was handed. `src1_1` is the second fuse
 * operand this instantiation never reads; see the header. */
extern "C" __global__ void kernel_rms_norm_mul_f32_4(
        const struct shiso_args_norm args,
        const char *__restrict__ src0,
        const char *__restrict__ src1_0,
        const char *__restrict__ src1_1,
        char *__restrict__ dst) {
    (void) src1_1;

    float *shmem_f32 = (float *) shiso_generic_smem;
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

    const float4 *x = (const float4 *) (src0 + in_at);
    const float4 *f0 = (const float4 *) (src1_0 + w_at);
    float4 *y = (float4 *) (dst + out_at);

    float sumf = 0.0f;
    for (int i00 = (int) tid; i00 < args.ne00_t; i00 += (int) blockDim.x) {
        const float4 v = x[i00];
        sumf += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    sumf = shiso_warp_all_sum(sumf);
    __syncthreads();
    if (lane == 0u) {
        shmem_f32[warp] = sumf;
    }
    __syncthreads();
    sumf = shiso_warp_all_sum(shmem_f32[lane]);

    const float mean = sumf / (float) args.ne00;
    const float scale = 1.0f / sqrtf(mean + args.eps);
    for (int i00 = (int) tid; i00 < args.ne00_t; i00 += (int) blockDim.x) {
        const float4 v = x[i00];
        const float4 wv = f0[i00];
        y[i00] = make_float4(v.x * scale * wv.x, v.y * scale * wv.y,
                             v.z * scale * wv.z, v.w * scale * wv.w);
    }
}

/* `sum = a + b`, then `norm = sum * rsqrt(mean(sum^2) + eps) * w`.
 *
 * Two destinations, because the residual the next block adds into and the
 * normalized row the next block reads are both wanted and the sum is computed
 * once. The sum is stored before it is reduced and re-read from memory in the
 * second loop, exactly as the Metal twin does: reading back what was stored is
 * what makes the two devices normalize the same bits rather than a register
 * copy of them. */
extern "C" __global__ void kernel_add_rms_norm_mul_f32_4(
        const struct shiso_args_norm args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        const char *__restrict__ weight,
        char *__restrict__ sum_dst,
        char *__restrict__ norm_dst) {
    float *shmem_f32 = (float *) shiso_generic_smem;
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
        const float4 v = make_float4(av.x + bv.x, av.y + bv.y, av.z + bv.z, av.w + bv.w);
        sum[i00] = v;
        sumf += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    }
    sumf = shiso_warp_all_sum(sumf);
    __syncthreads();
    if (lane == 0u) {
        shmem_f32[warp] = sumf;
    }
    __syncthreads();
    sumf = shiso_warp_all_sum(shmem_f32[lane]);

    const float mean = sumf / (float) args.ne00;
    const float scale = 1.0f / sqrtf(mean + args.eps);
    for (int i00 = (int) tid; i00 < args.ne00_t; i00 += (int) blockDim.x) {
        const float4 v = sum[i00];
        const float4 wv = w[i00];
        norm[i00] = make_float4(v.x * scale * wv.x, v.y * scale * wv.y,
                                v.z * scale * wv.z, v.w * scale * wv.w);
    }
}

#endif /* SHISO_NORM_CUH */
