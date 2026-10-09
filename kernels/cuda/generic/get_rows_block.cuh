/* The vocabulary gather over block-quantized tables.
 *
 * `shiso_engine::layers::RowGather::for_dtype` has named
 * `kernel_get_rows_q8_0_f32`, `kernel_get_rows_q4_K_f32` and
 * `kernel_get_rows_q4_0_f32` in module `generic` since it was written. **On
 * this backend none of the three existed**: the module built one gather, for
 * `f16`, and the other names resolved on Metal alone. Nothing had noticed
 * because every CUDA container this tree had opened kept its vocabulary table
 * in `f16` (a directory checkpoint, decoded at load) or `q6_K` (the Gemma 4
 * QAT export, whose gather lives in the `qwen35` module because that is where
 * the dispatch points `Q6_K`).
 *
 * `unsloth`'s `Qwen3.6-35B-A3B-UD-Q4_K_XL` keeps `token_embd` in `q8_0`, and it
 * is the first kernel a prompt touches - so the whole forward pass was behind a
 * `CUDA_ERROR_NOT_FOUND` on an entry the host had been naming all along.
 *
 * # One template rather than one kernel per codec
 *
 * The gather's geometry does not depend on the codec: the grid is one block per
 * 256-element span of a row whatever the format packs into, because
 * `launch::get_rows_grid` sizes it from the row width alone. What differs is
 * one decode, and `shiso_dequant_plane_t4<codec>` is already that decode - the
 * same one the plain matvec, the prefill GEMM and the routed expert stages
 * read these formats through, graded against an oracle dot product by
 * `shiso_kerneltest::qwen35::gemv` and against a staged plane by
 * `shiso_kerneltest::generic::plane`.
 *
 * So a table gathered here and a weight multiplied there cannot disagree about
 * what a byte means. A gather that respelled the unpack could, and the way it
 * would show is an embedding that is subtly the wrong vector for the right
 * token - which stays finite, runs every layer, and answers fluent nonsense.
 *
 * `q6_K` is deliberately NOT instantiated here. Its gather already exists in
 * the `qwen35` module and the dispatch points there; a second entry under a
 * second name would be two implementations of one contract, which is the thing
 * this file exists to avoid.
 */
#ifndef SHISO_GET_ROWS_BLOCK_CUH
#define SHISO_GET_ROWS_BLOCK_CUH

#include "dequant_plane.cuh"

/* Gathers the rows `src1`'s ids name out of the block-quantized table `src0`,
 * one `f32` row per token.
 *
 * A thread owns whole FOUR-ELEMENT CHUNKS, which is what the decode produces;
 * `blockIdx.x` is the 256-element span and the chunk index is derived from it,
 * so a codec whose block is 32 elements and one whose block is 256 walk the
 * same grid.
 *
 * An id outside the table leaves its row untouched rather than reading out of
 * bounds, and an element past the row's width writes nothing: both are the
 * `q6_K` gather's guards, and both matter because the grid is sized from the
 * padded row width rather than from `n_embd`. The tail is guarded per ELEMENT
 * rather than per chunk, because a row width that is not a multiple of four
 * would otherwise drop up to three elements or write past the row.
 */
template <typename codec>
__device__ __forceinline__ void shiso_get_rows_block(
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
    const codec *row = (const codec *) (src0 + (uint64_t) token * args.src_row_bytes);
    float *out = (float *) (dst + (uint64_t) tok_i * args.dst_row_bytes);

    constexpr int per_block = shiso_dequant_plane_chunks<codec>();
    /* Four-element chunks in one 256-element span. */
    const int32_t chunks = SHISO_QK_K / 4;
    const int32_t first = span * chunks;
    for (int32_t j = (int32_t) threadIdx.x; j < chunks; j += (int32_t) blockDim.x) {
        const int32_t c = first + j;
        const int32_t idx = c * 4;
        if (idx >= args.n_embd) {
            continue;
        }
        float4 v;
        shiso_dequant_plane_t4<codec>(row + c / per_block, c % per_block, &v);
        const float parts[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
        for (int32_t e = 0; e < 4; ++e) {
            if (idx + e < args.n_embd) {
                out[idx + e] = parts[e];
            }
        }
    }
}

/* One entry per codec. C linkage and a spelled name, because the host resolves
 * a name rather than a mangling. */
#define SHISO_GET_ROWS_BLOCK_ENTRY(SUFFIX, CODEC)                             \
    extern "C" __global__ void kernel_get_rows_##SUFFIX##_f32(                \
            const struct shiso_args_get_rows args,                            \
            const char *__restrict__ src0,                                    \
            const char *__restrict__ src1,                                    \
            char *__restrict__ dst) {                                         \
        shiso_get_rows_block<struct CODEC>(args, src0, src1, dst);            \
    }

SHISO_GET_ROWS_BLOCK_ENTRY(q8_0, shiso_block_q8_0)
SHISO_GET_ROWS_BLOCK_ENTRY(q4_K, shiso_block_q4_K)
SHISO_GET_ROWS_BLOCK_ENTRY(q4_0, shiso_block_q4_0)
SHISO_GET_ROWS_BLOCK_ENTRY(q5_K, shiso_block_q5_K)

/* Gathers the rows `src1`'s ids name out an `iq1_m` table, one `f32` row per
 * token.
 *
 * `iq1_m` is NOT one of the codec entries above: those all walk through
 * `shiso_dequant_plane_t4`, and this format has no four-element unpack that
 * does not re-derive the whole block's scale first. The block f16 is not
 * stored - it is the four top nibbles of the odd scale bytes, least
 * significant first - and each 16-weight subblock carries its own 3-bit
 * scale multiplier beside it, so the assembled `d`, the per-subblock `dl`
 * and the per-quad grid-shift delta are computed per element from the block
 * header. This is the row-at-a-time twin of `deq32_iq1_m` in
 * `crates/shiso-metal/metal/generic/get_rows.metal` and of the host
 * reference in `shiso_modelfile::iq_quant`, graded bit-for-bit against it by
 * `shiso_kerneltest::dense::iq::iq1_m_get_rows_matches_the_oracle`; the
 * `dl` here for the quad `m` of group `ib` is that dequantizer's `dl0` for
 * `m < 2` and its `dl1` above it, which is the same `6 * (ib % 2)` shift
 * plus three for the upper subblock.
 *
 * The geometry is the codec gathers' - one block per 256-element span, the
 * same guards on the token id and the row width - because the grid is sized
 * from the row width alone. */
extern "C" __global__ void kernel_get_rows_iq1_m_f32(
        const struct shiso_args_get_rows args,
        const char *__restrict__ src0,
        const char *__restrict__ src1,
        char *__restrict__ dst) {
    const int32_t block = (int32_t) blockIdx.x;
    const int32_t tok_i = (int32_t) blockIdx.y;
    if (tok_i >= args.n_tokens) {
        return;
    }
    const int32_t token =
            ((const int32_t *) (src1 + (uint64_t) tok_i * args.token_stride))[0];
    if (token < 0 || token >= args.n_vocab) {
        return;
    }
    const struct shiso_block_iq1_m *row =
            (const struct shiso_block_iq1_m *) (src0 + (uint64_t) token * args.src_row_bytes);
    const struct shiso_block_iq1_m *qb = row + block;
    float *out = (float *) (dst + (uint64_t) tok_i * args.dst_row_bytes);

    const unsigned w0 = (unsigned) qb->scales[0] | ((unsigned) qb->scales[1] << 8);
    const unsigned w1 = (unsigned) qb->scales[2] | ((unsigned) qb->scales[3] << 8);
    const unsigned w2 = (unsigned) qb->scales[4] | ((unsigned) qb->scales[5] << 8);
    const unsigned w3 = (unsigned) qb->scales[6] | ((unsigned) qb->scales[7] << 8);
    const unsigned short dbits = (unsigned short) ((w0 >> 12) | ((w1 >> 8) & 0x00F0)
                                                   | ((w2 >> 4) & 0x0F00) | (w3 & 0xF000));
    const float d = __half2float(__ushort_as_half(dbits));
    const unsigned words[4] = { w0, w1, w2, w3 };

    const int32_t i0 = block * SHISO_QK_K;
    for (int32_t i = (int32_t) threadIdx.x; i < SHISO_QK_K; i += (int32_t) blockDim.x) {
        const int32_t idx = i0 + i;
        if (idx >= args.n_embd) {
            continue;
        }
        const unsigned ib = (unsigned) i >> 5u;        /* 32-weight group  */
        const unsigned m  = ((unsigned) i >> 3u) & 3u; /* quad in the group */
        const unsigned j  = (unsigned) i & 7u;         /* weight in the quad */
        const unsigned shift = 6u * (ib & 1u) + 3u * (m >> 1u);
        const float dl = d * (float) (2u * ((words[ib >> 1u] >> shift) & 7u) + 1u);
        const unsigned char qhb = qb->qh[2u * ib + (m >> 1u)];
        const unsigned grid_idx = (unsigned) qb->qs[4u * ib + m]
                                | ((((unsigned) qhb) << ((m & 1u) ? 4u : 8u)) & 0x700u);
        const float delta = (qhb & ((m & 1u) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
        const uint64_t grid = SHISO_IQ1S_GRID[grid_idx];
        out[idx] = dl * (shiso_iq_sbyte(grid, (int) j) + delta);
    }
}

#endif /* SHISO_GET_ROWS_BLOCK_CUH */
