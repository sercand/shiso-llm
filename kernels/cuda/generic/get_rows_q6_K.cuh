/* The q6_K vocabulary gather.
 *
 * Nothing here is copied. What is preserved from the Metal twin in
 * `crates/shiso-metal/metal/qwen35/gemv_ext_q5k_q6k.metal` is the contract: the
 * entry name, the `GetRowsArgs` block, the buffer order, one block per
 * 256-element span of one row with the token index on the second grid axis, and
 * the unpack - `ql` low nibbles, the two high bits out of `qh`, one signed
 * scale per sixteen weights, and the `- 32` bias folded before the multiply.
 *
 * # Why this entry is in the `qwen35` module
 *
 * Because the engine asks for it there. `crates/shiso-engine/src/layers.rs`
 * maps `DType::Q6_K` to module `launch::Q35`, and on this backend a module name
 * is the key of a separately compiled image rather than a label. The name is
 * the qwen35 family's for historical reasons and is not a claim about the rest
 * of that family, which has one other entry here and nothing else.
 *
 * # Why a Gemma 4 GGUF needs it
 *
 * The QAT `q4_0` export packs its vocabulary table at `q6_K` even though every
 * linear layer in it is `q4_0`, so this is the first kernel a prompt touches
 * and the whole forward pass is behind it.
 */
#ifndef SHISO_GET_ROWS_Q6_K_CUH
#define SHISO_GET_ROWS_Q6_K_CUH

#include "generic.cuh"

/* Gathers the rows `src1`'s ids name out of the `q6_K` table `src0`, one f32
 * row per token.
 *
 * An id outside the table leaves its row untouched rather than reading out of
 * bounds, and a block index past the row's width writes nothing: both are the
 * Metal twin's guards, and both matter because the grid is sized from the
 * padded row width rather than from `n_embd`. */
extern "C" __global__ void kernel_get_rows_q6_K_f32(
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

    const struct shiso_block_q6_K *row =
        (const struct shiso_block_q6_K *) (src0 + (uint64_t) token * args.src_row_bytes);
    const struct shiso_block_q6_K *qb = row + block;
    float *out = (float *) (dst + (uint64_t) tok_i * args.dst_row_bytes);

    const int32_t i0 = block * SHISO_QK_K;
    for (int32_t i = (int32_t) threadIdx.x; i < SHISO_QK_K; i += (int32_t) blockDim.x) {
        const int32_t idx = i0 + i;
        if (idx >= args.n_embd) {
            continue;
        }
        const unsigned n128 = (unsigned) i >> 7u;
        const unsigned r = (unsigned) i & 127u;
        const unsigned quarter = r >> 5u;
        const unsigned l = r & 31u;
        const unsigned ql_base = n128 * 64u;
        const unsigned qh_base = n128 * 32u;
        unsigned q;
        if (quarter == 0u) {
            q = ((unsigned) qb->ql[ql_base + l] & 0x0Fu)
              | ((((unsigned) qb->qh[qh_base + l] >> 0u) & 3u) << 4u);
        } else if (quarter == 1u) {
            q = ((unsigned) qb->ql[ql_base + 32u + l] & 0x0Fu)
              | ((((unsigned) qb->qh[qh_base + l] >> 2u) & 3u) << 4u);
        } else if (quarter == 2u) {
            q = ((unsigned) qb->ql[ql_base + l] >> 4u)
              | ((((unsigned) qb->qh[qh_base + l] >> 4u) & 3u) << 4u);
        } else {
            q = ((unsigned) qb->ql[ql_base + 32u + l] >> 4u)
              | ((((unsigned) qb->qh[qh_base + l] >> 6u) & 3u) << 4u);
        }
        const unsigned sc_base = n128 * 8u + quarter * 2u + (l >> 4u);
        out[idx] = __half2float(qb->d) * (float) ((int) qb->scales[sc_base])
                 * (float) ((int) q - 32);
    }
}

#endif /* SHISO_GET_ROWS_Q6_K_CUH */
