/* The gated SiLU, which is the activation a qwen35 feed-forward is built on.
 *
 * Nothing here is copied. What is preserved from the Metal twin in
 * `kernels/metal/generic/glu.metal` is the contract: the entry name,
 * the `GluArgs` block, the buffer order, the flat index taken straight from the
 * grid, the two element offsets the row is reached through, and the clamp pair
 * - which is applied only when `limit` is above 1e-6, so that zero means "no
 * clamp" rather than "clamp everything to zero".
 *
 * `kernel_dense_geglu_flat_f32` in `kernels/cuda/dense/dense_misc.cuh` is the
 * same body at the other curve. They are two entries rather than one with a
 * flag for the reason `crates/shiso-engine/src/ffn.rs` gives: the choice
 * belongs to the checkpoint rather than to the launch, and a branch inside the
 * kernel could be got wrong at one call site with nothing failing loudly.
 *
 * # The exponential
 *
 * Written as `expf`, which is the accurate entry. `kernels/KERNEL.toml` sets
 * `--use_fast_math` for every directory but `cuda/dense`, so in this module
 * nvcc lowers it to the fast one; that is the manifest's decision and not this
 * kernel's, and `crates/shiso-kerneltest/src/generic/glu.rs` grades what it
 * actually compiled to at `TOL_F32`. Spelling the fast intrinsic here instead
 * would be an arithmetic edit made to restate a build flag.
 */
#ifndef SHISO_GLU_CUH
#define SHISO_GLU_CUH

#include "generic.cuh"

/* `dst[i] = silu(gate[i]) * up[i] * alpha`, flat over `ne0`. */
extern "C" __global__ void kernel_swiglu_flat_f32(
        const struct shiso_args_glu args,
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

    const float silu = x0 / (1.0f + expf(-x0));
    dst_f32[i] = silu * x1 * args.alpha;
}

#endif /* SHISO_GLU_CUH */
