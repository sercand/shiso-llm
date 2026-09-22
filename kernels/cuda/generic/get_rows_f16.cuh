/* The half-precision vocabulary gather.
 *
 * Nothing here is copied. What is preserved from the Metal twin in
 * `crates/shiso-metal/metal/generic/get_rows.metal` is the contract: the entry
 * name, the `GetRowsArgs` block, the buffer order, one block per 256-element
 * span of one row with the token index on the second grid axis, and the two
 * guards - an id outside the table leaves its row untouched, and a span past
 * the row's width writes nothing.
 *
 * # Why the engine names this and why CUDA did not have it
 *
 * `crates/shiso-engine/src/layers.rs`'s `EmbedKernel::for_dtype` has mapped
 * `DType::F16` to `kernel_get_rows_f16_f32` in module `launch::GENERIC` for as
 * long as there has been a directory checkpoint, and until now that entry
 * existed only in the Metal set: **the CUDA `generic` and `qwen35` modules
 * between them carried one gather, the `q6_K` one**. Every CUDA family so far
 * has read a `q6_K` vocabulary table, so nothing named this and nothing
 * noticed.
 *
 * `qwen35moe` is the first CUDA family whose table is not `q6_K`. It ships
 * `BF16`, which no projection kernel here reads, so the loader widens it - to
 * `f16` rather than `f32` because the table is 248,320 rows of 2,048 and the
 * wider one would cost a gigabyte of device memory for precision a gather does
 * not use. That made this the first launch of the first prefill, and it
 * refused by name: `cuModuleGetFunction`, `CUDA_ERROR_NOT_FOUND`.
 *
 * Which is the good outcome, and worth saying so. `HANDOFF.md` at the
 * repository root is twelve defects of which eleven failed SILENTLY; a missing
 * entry is the twelfth kind, and it names itself.
 *
 * # This one is the block-descriptor gather, not the strided one
 *
 * The Metal set has two entries that read a half table and they are not
 * duplicates: `kernel_get_rows_f16` takes the general strided descriptor, and
 * this one takes the block-and-token descriptor the quantized gathers share -
 * which is the only one the engine's single gather launch site builds. `half`
 * has no blocks, so the span is a plain element range rather than a block
 * index, and the bound is `n_embd` because the last span of a row whose width
 * is not a multiple of 256 is short.
 */
#ifndef SHISO_GET_ROWS_F16_CUH
#define SHISO_GET_ROWS_F16_CUH

#include "generic.cuh"

/* Elements one block gathers, which is the quantized gathers' block width and
 * is what the host sizes the grid by. */
#define SHISO_GET_ROWS_SPAN 256

/* Gathers the rows `src1`'s ids name out of the half table `src0`, one f32 row
 * per token. */
extern "C" __global__ void kernel_get_rows_f16_f32(
        const struct shiso_args_get_rows args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    const int32_t span = (int32_t) blockIdx.x;
    const int32_t tok_i = (int32_t) blockIdx.y;
    if (tok_i >= args.n_tokens) {
        return;
    }

    const int32_t token =
        ((const int32_t *) (src1 + (uint64_t) tok_i * args.token_stride))[0];
    if (token < 0 || token >= args.n_vocab) {
        return;
    }

    const __half *row = (const __half *) (src0 + (uint64_t) token * args.src_row_bytes);
    float *out = (float *) (dst + (uint64_t) tok_i * args.dst_row_bytes);

    const int32_t i0 = span * SHISO_GET_ROWS_SPAN;
    for (int32_t i = (int32_t) threadIdx.x; i < SHISO_GET_ROWS_SPAN;
         i += (int32_t) blockDim.x) {
        const int32_t idx = i0 + i;
        if (idx < args.n_embd) {
            out[idx] = __half2float(row[idx]);
        }
    }
}

#endif /* SHISO_GET_ROWS_F16_CUH */
