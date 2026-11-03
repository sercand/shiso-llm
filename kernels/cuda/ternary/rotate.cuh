/* Signed blockwise Walsh-Hadamard rotation of f32 rows; the twin of
 * `kernels/metal/ternary/rotate.metal`, whose header comment states
 * the transform. One block per (rotation block, token), the rotation block
 * staged whole in dynamic shared memory before any write, so `src == dst` is
 * safe when nothing is permuted. */
#ifndef SHISO_TERNARY_ROTATE_CUH
#define SHISO_TERNARY_ROTATE_CUH

#include "ternary_common.cuh"

extern "C" __global__ void kernel_rotate_wht_f32(
        const struct ternary_args_rotate args,
        const float *src,
        const float *__restrict__ signs,
        float *dst) {
    extern __shared__ float stage[];
    const unsigned block = args.block;
    const unsigned base = blockIdx.x * block;
    const uint64_t row = (uint64_t) blockIdx.y * args.width;
    const unsigned tid = threadIdx.x;
    const unsigned nth = blockDim.x;

    for (unsigned i = tid; i < block; i += nth) {
        const unsigned e = base + i;
        unsigned from = e;
        if (args.rep > 1u) {
            const unsigned head = e / args.hd;
            const unsigned k = head / args.rep;
            const unsigned m = head % args.rep;
            from = (m * args.nk + k) * args.hd + e % args.hd;
        }
        float x = src[row + from];
        if (args.inverse == 0u) {
            x *= signs[e];
        }
        stage[i] = x;
    }

    for (unsigned h = 1u; h < block; h <<= 1u) {
        __syncthreads();
        for (unsigned p = tid; p < block / 2u; p += nth) {
            const unsigned i = (p / h) * (2u * h) + p % h;
            const float a = stage[i];
            const float b = stage[i + h];
            stage[i] = a + b;
            stage[i + h] = a - b;
        }
    }
    __syncthreads();

    const float scale = rsqrtf((float) block);
    for (unsigned i = tid; i < block; i += nth) {
        const unsigned e = base + i;
        float y = stage[i] * scale;
        if (args.inverse != 0u) {
            y *= signs[e];
        }
        dst[row + e] = y;
    }
}

#endif
