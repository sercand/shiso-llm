/* The FP8 projection, at one token and at a chunk.
 *
 * `dst[out + tok * out_dim] = sum_in e4m3(W[out][in]) * weight_scale * x[in + tok * in_dim]`,
 * which is `crates/shiso-engine/src/projection.rs`'s contract with this
 * checkpoint's codec in the weight slot.
 *
 * # This is W8A16 on a W8A8 checkpoint, and that is a ruling
 *
 * The container states an `input_scale` beside every one of these 130 targets:
 * the export quantized the activations too, and a product that used both sides
 * would accumulate `w_q . x_q` in FP8 and scale once at the end, which is what
 * a GB10's FP8 tensor cores are for. These entries do not do that. They decode
 * the weight, multiply by an `f32` activation and accumulate in `f32`.
 *
 * That is deliberate and it is the correctness path: it is exactly what the
 * host oracle does, so a disagreement between the two is a kernel defect and
 * not a difference of scheme. It is also strictly MORE precise than the
 * checkpoint's intent, because the activations keep every bit they had - so
 * the divergence from the reference, if any, is in the direction that cannot
 * be mistaken for a bug. The tensor-core path is a measurement against these,
 * not a replacement for them, and `input_scale` is carried through the loader
 * unused until that measurement is made.
 *
 * # Why the matvec and the GEMM are both here
 *
 * A matvec re-reads the whole weight matrix once per token. At a 512-token
 * prefill chunk over a 23.4 GB checkpoint that is not a tuning difference; it
 * is the difference between a prompt that prefills and one that appears to
 * hang. The host picks between them on the row count, exactly as
 * `projection.rs` already does for every other codec.
 */
#ifndef SHISO_Q35MOE_MV_FP8_CUH
#define SHISO_Q35MOE_MV_FP8_CUH

#include "q35moe_common.cuh"

/* `dst = (W x) * weight_scale`, one output row per block.
 *
 * Grid `(out_dim, n_tok, 1)`, block `(256, 1, 1)`. One block owns one (row,
 * token) pair and strides its own row by four weights per thread per turn,
 * falling back to one weight per thread per turn for the whole row when
 * `in_dim` is not a multiple of four.
 *
 * The fallback is an ALIGNMENT guard and not a count guard, which is the
 * distinction that makes it necessary. A row starts at `weights + row *
 * in_dim`, so an `in_dim` of, say, 2050 puts every odd row two bytes off a
 * word boundary and the four-byte load below faults with
 * `CUDA_ERROR_MISALIGNED_ADDRESS` - and `modelopt_dispatchable` in
 * `crates/shiso-engine/src/projection.rs` admits it, because FP8 genuinely has
 * no width constraint: one byte is one weight, so `block_elems()` is 1 and
 * every `in_dim >= 1` is dispatchable. A tail loop that only mopped up the
 * remaining `in_dim % 4` weights would read as if it had handled this and
 * would not have. `crates/shiso-kerneltest/src/qwen35moe/fp8.rs`'s
 * `fp8_matvec_matches_the_host_codec_at_an_unaligned_width` is the case that
 * fails if this guard is removed; it uses widths of 2049, 2050 and 2051, which
 * are the three residues that break it at a row count high enough for the word
 * loop to have run.
 *
 * Four and not one, which is what this took until 2026-09-21. An `e4m3`
 * weight is one byte, so a thread reading one weight per turn made a warp
 * fetch 32 bytes per load instruction - a quarter of what one can move - and
 * this entry, the largest single kernel in an Ornith decode step at 34.6% of
 * it, ran at 127.5 GB/s of this device's measured 239 GB/s peak. One
 * four-byte word per thread per turn takes the same shapes to 186-212 GB/s.
 * Measured in `docs/handoff/cuda-decode-perf.md`.
 *
 * The fold order is NOT the one this had before: a thread now accumulates four
 * adjacent weights where it used to accumulate one every 256, so the partial
 * sums differ in their last bits and the model's logits move. The measured
 * move against the `transformers` oracle is in the commit that made it; the
 * budget was not touched.
 *
 * The scale is applied ONCE, to the finished sum, rather than per weight. Both
 * orders are defensible in exact arithmetic and only one of them is what the
 * host oracle does, and matching the oracle is the entire value of the oracle.
 */
extern "C" __global__ void kernel_q35moe_mv_fp8_f32(
        const struct shiso_args_q35moe_fp8 args,
        const unsigned char *__restrict__ weights,
        const float *__restrict__ src,
        float *__restrict__ dst) {
    __shared__ float partials[SHISO_Q35MOE_WARPS];

    const int row = (int) blockIdx.x;
    const int tok = (int) blockIdx.y;
    if (row >= args.out_dim || tok >= args.n_tok) {
        return;
    }

    const unsigned char *w = weights + (int64_t) row * (int64_t) args.in_dim;
    const float *x = src + (int64_t) tok * (int64_t) args.in_dim;

    float sum = 0.0f;
    /* Zero when the width would misalign a row, which sends the whole row
     * through the byte loop below. Uniform across the block, so it costs a
     * branch and no divergence. */
    const int quads = ((args.in_dim & 3) == 0) ? args.in_dim / 4 : 0;
    for (int q = (int) threadIdx.x; q < quads; q += SHISO_Q35MOE_BLOCK) {
        const unsigned word = ((const unsigned *) w)[q];
        const float *xq = x + q * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sum = fmaf(shiso_e4m3((unsigned char) ((word >> (8u * (unsigned) j)) & 0xFFu)),
                       xq[j], sum);
        }
    }
    for (int k = quads * 4 + (int) threadIdx.x; k < args.in_dim; k += SHISO_Q35MOE_BLOCK) {
        sum = fmaf(shiso_e4m3(w[k]), x[k], sum);
    }
    sum = shiso_q35moe_block_sum(sum, partials);
    if (threadIdx.x == 0) {
        dst[(int64_t) tok * (int64_t) args.out_dim + row] = sum * args.weight_scale;
    }
}

/* `dst = (X W^T) * weight_scale` over a chunk of tokens.
 *
 * Grid `(ceil(out_dim / 16), ceil(n_tok / 16), 1)`, block `(16, 16, 1)`. One
 * thread owns one output element and the block walks the contraction in
 * sixteen-wide slabs, staging both operands in shared memory so each is read
 * from global memory once per slab rather than once per output element.
 *
 * The weight tile is staged DECODED. Staging the bytes instead would halve the
 * shared-memory bill and cost a decode per use rather than per load, which is
 * sixteen decodes of the same byte; at this tile size the decode is the
 * arithmetic, so the trade is the wrong way round.
 *
 * Both edges are guarded, because neither 248,320 output rows nor a
 * caller-chosen token count is a whole number of tiles: an unguarded tile
 * would read a row past the matrix and write a row past the output.
 */
extern "C" __global__ void kernel_q35moe_mm_fp8_f32(
        const struct shiso_args_q35moe_fp8 args,
        const unsigned char *__restrict__ weights,
        const float *__restrict__ src,
        float *__restrict__ dst) {
    __shared__ float a_tile[SHISO_Q35MOE_TILE][SHISO_Q35MOE_TILE];
    __shared__ float w_tile[SHISO_Q35MOE_TILE][SHISO_Q35MOE_TILE];

    const int row = (int) (blockIdx.x * SHISO_Q35MOE_TILE + threadIdx.x);
    const int tok = (int) (blockIdx.y * SHISO_Q35MOE_TILE + threadIdx.y);

    float sum = 0.0f;
    for (int base = 0; base < args.in_dim; base += SHISO_Q35MOE_TILE) {
        /* Thread `(x, y)` stages activation `(tok_y, base + x)` and weight
         * `(row_x, base + y)`, so both loads are contiguous across `x` within
         * a warp and neither tile is transposed on the way in. */
        const int a_k = base + (int) threadIdx.x;
        const int a_tok = (int) (blockIdx.y * SHISO_Q35MOE_TILE + threadIdx.y);
        a_tile[threadIdx.y][threadIdx.x] =
                (a_tok < args.n_tok && a_k < args.in_dim)
                        ? src[(int64_t) a_tok * (int64_t) args.in_dim + a_k]
                        : 0.0f;

        const int w_k = base + (int) threadIdx.y;
        const int w_row = (int) (blockIdx.x * SHISO_Q35MOE_TILE + threadIdx.x);
        w_tile[threadIdx.y][threadIdx.x] =
                (w_row < args.out_dim && w_k < args.in_dim)
                        ? shiso_e4m3(weights[(int64_t) w_row * (int64_t) args.in_dim + w_k])
                        : 0.0f;
        __syncthreads();

#pragma unroll
        for (int k = 0; k < SHISO_Q35MOE_TILE; ++k) {
            sum = fmaf(a_tile[threadIdx.y][k], w_tile[k][threadIdx.x], sum);
        }
        __syncthreads();
    }

    if (row < args.out_dim && tok < args.n_tok) {
        dst[(int64_t) tok * (int64_t) args.out_dim + row] = sum * args.weight_scale;
    }
}

#endif /* SHISO_Q35MOE_MV_FP8_CUH */
