/* The per-layer embedding pair: one row of a table too large to hold, turned
 * into the vector each layer slot multiplies its gate by.
 *
 * Gemma 4 E4B carries one q6_K block per layer slot per token id, so a token's
 * whole row is `n_layers` blocks laid end to end. The table is streamed off the
 * file a row at a time and staged in front of the chunk, or held resident for a
 * caller that can afford it; the only difference the kernel sees is whether it
 * indexes `rows` by the token id or by the token's own position in the chunk.
 */
#ifndef SHISO_DENSE_PLE_CUH
#define SHISO_DENSE_PLE_CUH

#include "dense_common.cuh"

/* Normalizes the model's per-layer projection, dequantizes the token's row and
 * mixes the two.
 *
 * One threadgroup per (layer slot, token), dispatched as `(n_layers, n_tok)`.
 * Per slot:
 *
 *   p_i   = proj[t, l*dim + i] * proj_scale
 *   p_i  <- p_i * rsqrt(sum_i(p_i^2) / dim + eps) * norm_w[i]
 *   e_i   = dequant(rows + row*row_bytes + l*sizeof(block))[i] * embed_scale
 *   out_i = (p_i + e_i) * mix_scale
 *
 * where `row` is `ids[t]` for a resident table and `t` for a streamed one.
 *
 * THE SCALE IS FOLDED IN BEFORE THE SUM OF SQUARES, so it reaches the
 * magnitudes the normalization divides by as well as the value it multiplies.
 *
 * The decoded row is staged in shared memory rather than dequantized per
 * element: one `dequantize_dense_q6_K` call produces sixteen weights, so a
 * per-element decode would do sixteen times the unpacking work and throw
 * fifteen sixteenths of it away. `dense_smem` therefore holds the block sum's
 * slots first and the staged row after them. */
extern "C" __global__ void kernel_dense_ple_prepare_q6_K(
        const struct dense_args_ple args,
        float *__restrict__ out,
        const float *__restrict__ proj,
        const float *__restrict__ norm_w,
        const char *__restrict__ rows,
        const int32_t *__restrict__ ids) {
    const unsigned l = blockIdx.x;
    const unsigned t = blockIdx.y;
    if (l >= args.n_layers || t >= args.n_tok) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;

    const unsigned dim = args.dim;
    float *scratch = (float *) dense_smem;
    float *emb = scratch + DENSE_BLOCK_SUM_SLOTS;

    /* The token's row, then this slot's block inside it. The block pointer is
     * stepped by `l` rather than by a byte count, so the stride is the struct's
     * own 210 bytes and cannot drift from the layout the decoder reads. */
    const uint64_t row = (args.resident != 0u) ? (uint64_t) ids[t] : (uint64_t) t;
    const struct shs_dense_block_q6_K *blk =
        (const struct shs_dense_block_q6_K *) (rows + row * (uint64_t) args.row_bytes) + l;

    for (unsigned g = tid; g < dim / DENSE_PLE_GROUP; g += nt) {
        float reg[DENSE_PLE_GROUP];
        dequantize_dense_q6_K(blk, (int) g, reg);
        for (unsigned j = 0u; j < DENSE_PLE_GROUP; ++j) {
            emb[g * DENSE_PLE_GROUP + j] = reg[j];
        }
    }
    __syncthreads();

    const float *p = proj + ((uint64_t) t * args.n_layers + l) * dim;
    float *dst = out + ((uint64_t) t * args.n_layers + l) * dim;

    float acc = 0.0f;
    for (unsigned i = tid; i < dim; i += nt) {
        const float v = p[i] * args.proj_scale;
        acc = dense_fma(v, v, acc);
    }
    const float total = dense_block_sum(acc, scratch, tid, nt);
    const float inv = rsqrtf(total / (float) dim + args.eps);

    for (unsigned i = tid; i < dim; i += nt) {
        const float pv = p[i] * args.proj_scale * inv * norm_w[i];
        const float ev = emb[i] * args.embed_scale;
        dst[i] = (pv + ev) * args.mix_scale;
    }
}

/* The same prepare, over the mlx sidecar's affine row instead of a q6_K one.
 *
 * A row holds every slot's words, then every slot's scales, then every slot's
 * biases - the order `gemma4::ple_repack` writes them in. The three bases are
 * DERIVED from the slot count and the slot width rather than written down, so
 * this decodes whatever row stride the caller passes and not only the one the
 * shipped checkpoint happens to have. A revision that pinned that checkpoint's
 * forty-two-slot offsets as literals could only ever be tested at one shape. */
extern "C" __global__ void kernel_dense_ple_prepare_affine4(
        const struct dense_args_ple args,
        float *__restrict__ out,
        const float *__restrict__ proj,
        const float *__restrict__ norm_w,
        const char *__restrict__ rows,
        const int32_t *__restrict__ ids) {
    const unsigned l = blockIdx.x;
    const unsigned t = blockIdx.y;
    if (l >= args.n_layers || t >= args.n_tok) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;

    const unsigned dim = args.dim;
    float *scratch = (float *) dense_smem;
    float *emb = scratch + DENSE_BLOCK_SUM_SLOTS;

    const uint64_t row = (args.resident != 0u) ? (uint64_t) ids[t] : (uint64_t) t;
    const char *row_ptr = rows + row * (uint64_t) args.row_bytes;

    /* `dim` is DENSE_PLE_DIM (256) - four whole groups - and the engine asserts
     * it on the host, because a kernel cannot refuse. The divide is therefore
     * exact, and it is the SLOT COUNT that varies here: the three bases below
     * are derived from `n_layers` and the slot width rather than written down,
     * so this decodes whatever row stride the caller passes. */
    const unsigned groups_per_slot = dim / 64u;
    const unsigned words_per_slot = dim / 2u;                 /* four bits a weight */
    const unsigned scales_per_slot = groups_per_slot * 2u;    /* one bf16 a group */
    const unsigned scales_base = args.n_layers * words_per_slot;
    const unsigned biases_base = scales_base + args.n_layers * scales_per_slot;

    const uint32_t *w = (const uint32_t *) (row_ptr + l * words_per_slot);
    const unsigned short *s =
        (const unsigned short *) (row_ptr + scales_base + l * scales_per_slot);
    const unsigned short *b =
        (const unsigned short *) (row_ptr + biases_base + l * scales_per_slot);

    for (unsigned i = tid; i < dim; i += nt) {
        const unsigned g = i / 64u;
        const unsigned in_group = i % 64u;
        const unsigned word_idx = in_group / 8u;
        const unsigned shift = (in_group % 8u) * 4u;

        const uint32_t word = w[g * 8u + word_idx];
        const float q = (float) ((word >> shift) & 0x0fu);
        emb[i] = q * dense_bf16(s + g) + dense_bf16(b + g);
    }
    __syncthreads();

    const float *p = proj + ((uint64_t) t * args.n_layers + l) * dim;
    float *dst = out + ((uint64_t) t * args.n_layers + l) * dim;

    float acc = 0.0f;
    for (unsigned i = tid; i < dim; i += nt) {
        const float v = p[i] * args.proj_scale;
        acc = dense_fma(v, v, acc);
    }
    const float total = dense_block_sum(acc, scratch, tid, nt);
    const float inv = rsqrtf(total / (float) dim + args.eps);

    for (unsigned i = tid; i < dim; i += nt) {
        const float pv = p[i] * args.proj_scale * inv * norm_w[i];
        const float ev = emb[i] * args.embed_scale;
        dst[i] = (pv + ev) * args.mix_scale;
    }
}

/* `gate[t, i] = dense_gelu_tanh(gate[t, i]) * ple[t, layer, i]`, in place.
 *
 * Flat over the `n_tok * dim` gate: the slot is a field of the argument block
 * rather than a grid dimension, because one layer of the loop runs one of these
 * and the gate it multiplies is the only row in flight. In place because the
 * gate is this launch's only reader as well as its only writer. */
extern "C" __global__ void kernel_dense_ple_gate_mul(
        const struct dense_args_ple_gate args,
        float *__restrict__ gate,
        const float *__restrict__ ple) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned n = args.n_tok * args.dim;
    if (i >= n) {
        return;
    }
    const unsigned t = i / args.dim;
    const unsigned d = i - t * args.dim;
    const uint64_t at = ((uint64_t) t * args.n_layers + args.layer) * args.dim + d;
    gate[i] = dense_gelu_tanh(gate[i]) * ple[at];
}

#endif /* SHISO_DENSE_PLE_CUH */
