/* The NVFP4 projection, at one token and at a chunk.
 *
 * `dst[out + tok * out_dim] = sum_in w[out][in] * x[in + tok * in_dim]`, where
 *
 *     w[r][k] = e2m1(nibble) * (e4m3(scale[r][k / 16]) * global)
 *
 * with the four-bit code in the low nibble of byte `k / 2` when `k` is even and
 * the high nibble when it is odd.
 *
 * # Three levels, and the order the two scales multiply in
 *
 * The group scale and the per-tensor scale are multiplied TOGETHER FIRST, once
 * per group, and the product is reused across the group's sixteen weights.
 * That is what `crates/shiso-modelfile/src/nvfp4.rs` does, and it is not the same
 * arithmetic as folding the global into each decoded weight: the second form
 * rounds once per weight instead of once per group, and the two stop agreeing
 * on the last bit. The oracle compares against that decoder, so the kernel has
 * to round where the decoder rounds or the comparison is grading the rounding
 * rather than the kernel.
 *
 * **The per-tensor level is the one that is easy to drop.** This checkpoint's
 * globals are around 3e-5; a kernel that omitted `global` would be wrong by
 * about 31,458x uniformly, which is a residual of the right shape and the
 * wrong magnitude - and the next normalization very nearly removes it. It is a
 * field of the argument block rather than something the host may fold in
 * elsewhere for exactly that reason, and the oracle's fixtures are real tensor
 * corners rather than synthetic ones so that a kernel which drops it fails
 * rather than passes.
 *
 * # The low nibble is the earlier weight
 *
 * The one convention in this format that the checkpoint itself cannot settle:
 * reversing the nibbles permutes weights inside a byte and inside a group, so
 * every group scale, every magnitude histogram and every column norm is
 * unchanged. It is taken from `crates/shiso-modelfile/src/nvfp4.rs`, which takes it
 * from `vllm`'s own dequantizer, and it is pinned there by a hand-written byte
 * because no test over real weights can see it.
 */
#ifndef SHISO_Q35MOE_MV_NVFP4_CUH
#define SHISO_Q35MOE_MV_NVFP4_CUH

#include "q35moe_common.cuh"

/* Decodes weight `k` of a row whose packed bytes start at `packed` and whose
 * group scales start at `scales`, given the tensor's global scale. */
__device__ __forceinline__ float shiso_nvfp4_weight(
        const unsigned char *__restrict__ packed,
        const unsigned char *__restrict__ scales,
        int k,
        float global_scale) {
    const unsigned char byte = packed[k / SHISO_NVFP4_PER_BYTE];
    const unsigned char code = (k & 1) ? (unsigned char) (byte >> 4)
                                       : (unsigned char) (byte & 0x0Fu);
    const float scale = shiso_e4m3(scales[k / SHISO_NVFP4_GROUP]) * global_scale;
    return shiso_e2m1(code) * scale;
}

/* `dst = W x`, one output row per WARP.
 *
 * Grid `(ceil(out_dim / SHISO_Q35MOE_MV_ROWS), n_tok, 1)`, block
 * `(SHISO_Q35MOE_MV_BLOCK, 1, 1)`. Every lane walks whole sixteen-weight
 * GROUPS of its row - see `shiso_q35moe_nvfp4_row_dot`, which is the whole of
 * the arithmetic and is shared with the routed expert stages so the two cannot
 * drift apart.
 *
 * # Why this stopped being a block per row
 *
 * It was `(out_dim, n_tok)` blocks of 256 threads, each thread walking one
 * PACKED BYTE per turn and the block folding through shared memory. Measured
 * on this checkpoint at 128 decode steps:
 *
 * | tensor | bytes/step | ms/step | GB/s |
 * | --- | --- | --- | --- |
 * | `lm_head`, 248320 x 2048 | 286 MB | 2.851 | 100 |
 * | shared expert gate+up, 512 x 2048 | 47 MB | 0.687 | 68 |
 * | shared expert down, 2048 x 512 | 24 MB | 0.504 | 47 |
 *
 * against 211 GB/s for the FP8 entry beside this one on the same step, and
 * against this device's 239 GB/s measured peak. The cause is in the geometry
 * rather than the codec: a 2048-wide row hands each of 256 threads four bytes
 * and then charges a 256-wide fold for them, and a 512-wide row hands each
 * thread ONE byte. The fold, the barrier and the eight scalar activation loads
 * per packed word were the kernel; the weights were a rounding error on it.
 *
 * # The tail block is guarded, and the guard is per warp
 *
 * `out_dim` is not a multiple of four in general - 248320 happens to be, 2
 * would not - so a warp whose row is past the end returns before it reads. It
 * returns rather than masking because `shiso_q35moe_nvfp4_row_dot` folds
 * across the warp and a lane that skipped the fold would hang the others; the
 * whole warp agrees on `row`, so the return is uniform within it.
 */
extern "C" __global__ void kernel_q35moe_mv_nvfp4_f32(
        const struct shiso_args_q35moe_nvfp4 args,
        const unsigned char *__restrict__ weights,
        const unsigned char *__restrict__ scales,
        const float *__restrict__ src,
        float *__restrict__ dst) {
    const unsigned lane = threadIdx.x & 31u;
    const int row = (int) (blockIdx.x * SHISO_Q35MOE_MV_ROWS + (threadIdx.x >> 5u));
    const int tok = (int) blockIdx.y;
    if (row >= args.out_dim || tok >= args.n_tok) {
        return;
    }

    const int64_t packed_width = (int64_t) args.in_dim / SHISO_NVFP4_PER_BYTE;
    const int64_t scale_width = (int64_t) args.in_dim / SHISO_NVFP4_GROUP;
    const float sum = shiso_q35moe_nvfp4_row_dot(
            weights + (int64_t) row * packed_width,
            scales + (int64_t) row * scale_width,
            args.global_scale,
            src + (int64_t) tok * (int64_t) args.in_dim,
            args.in_dim,
            lane);
    if (lane == 0u) {
        dst[(int64_t) tok * (int64_t) args.out_dim + row] = sum;
    }
}

/* `dst = X W^T` over a chunk of tokens.
 *
 * Grid `(ceil(out_dim / 16), ceil(n_tok / 16), 1)`, block `(16, 16, 1)`, and
 * the same tiling `kernel_q35moe_mm_fp8_f32` uses - staged decoded, both edges
 * guarded, one thread per output element. The only difference is what a staged
 * weight costs to produce.
 */
extern "C" __global__ void kernel_q35moe_mm_nvfp4_f32(
        const struct shiso_args_q35moe_nvfp4 args,
        const unsigned char *__restrict__ weights,
        const unsigned char *__restrict__ scales,
        const float *__restrict__ src,
        float *__restrict__ dst) {
    __shared__ float a_tile[SHISO_Q35MOE_TILE][SHISO_Q35MOE_TILE];
    __shared__ float w_tile[SHISO_Q35MOE_TILE][SHISO_Q35MOE_TILE];

    const int row = (int) (blockIdx.x * SHISO_Q35MOE_TILE + threadIdx.x);
    const int tok = (int) (blockIdx.y * SHISO_Q35MOE_TILE + threadIdx.y);
    const int packed_width = args.in_dim / SHISO_NVFP4_PER_BYTE;
    const int groups = args.in_dim / SHISO_NVFP4_GROUP;

    float sum = 0.0f;
    for (int base = 0; base < args.in_dim; base += SHISO_Q35MOE_TILE) {
        const int a_k = base + (int) threadIdx.x;
        a_tile[threadIdx.y][threadIdx.x] =
                (tok < args.n_tok && a_k < args.in_dim)
                        ? src[(int64_t) tok * (int64_t) args.in_dim + a_k]
                        : 0.0f;

        const int w_k = base + (int) threadIdx.y;
        w_tile[threadIdx.y][threadIdx.x] =
                (row < args.out_dim && w_k < args.in_dim)
                        ? shiso_nvfp4_weight(
                                  weights + (int64_t) row * (int64_t) packed_width,
                                  scales + (int64_t) row * (int64_t) groups,
                                  w_k,
                                  args.global_scale)
                        : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < SHISO_Q35MOE_TILE; ++k) {
            sum = fmaf(a_tile[threadIdx.y][k], w_tile[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (row < args.out_dim && tok < args.n_tok) {
        dst[(int64_t) tok * (int64_t) args.out_dim + row] = sum;
    }
}

#endif /* SHISO_Q35MOE_MV_NVFP4_CUH */
