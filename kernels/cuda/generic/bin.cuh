/* The one binary op the dense pass reaches for.
 *
 * `crates/shiso-metal/metal/generic/bin.metal` carries a whole fused binary
 * family behind function constants; the engine names exactly one of its plain
 * entries from a dense pass (`crates/shiso-engine/src/layers.rs`, under
 * `launch::GENERIC`), and only that one is twinned here. A module that exported
 * the rest without a caller would be claiming coverage it has no oracle for.
 */
#ifndef SHISO_BIN_CUH
#define SHISO_BIN_CUH

#include "generic.cuh"

/* `out[i] = a[i] + b[i]`, over the first `n` elements.
 *
 * A destination of its own rather than an add in place, because the residual
 * stream is two buffers the pass alternates between and this writes the one the
 * next slot is about to read. */
extern "C" __global__ void kernel_add2_f32(
        const struct shiso_args_add args,
        const float *__restrict__ a,
        const float *__restrict__ b,
        float *__restrict__ out) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.n) {
        return;
    }
    out[i] = a[i] + b[i];
}

#endif /* SHISO_BIN_CUH */
