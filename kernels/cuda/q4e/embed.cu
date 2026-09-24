/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels).
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 *
 * The row gather that opens a pass, at every element type a vocabulary table
 * is stored in here: half precision with the residual-stream fan-out the
 * hyper-connected families use, brain float, and the block-quantized tier.
 * They belong to no kernel family - every model in this tree starts the same
 * way - which is why they are one unit of their own rather than folded into
 * one of the block families.
 */

#include "q4e.cuh"

extern "C" __global__ void embed_token_hc_kernel(float *out, const unsigned short *w, uint32_t token, uint32_t n_embd, uint32_t n_hc) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t n = n_embd * n_hc;
    if (i >= n) return;
    uint32_t e = i % n_embd;
    out[i] = __half2float(reinterpret_cast<const __half *>(w)[(uint64_t)token * n_embd + e]);
}

extern "C" __global__ void embed_tokens_hc_kernel(
        float *out,
        const int32_t *tokens,
        const __half *w,
        uint32_t n_vocab,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_hc) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint32_t d = gid % n_embd;
    uint64_t tmp = gid / n_embd;
    uint32_t t = tmp / n_hc;
    int32_t tok_i = tokens[t];
    uint32_t tok = tok_i < 0 ? 0u : (uint32_t)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = __half2float(w[(uint64_t)tok * n_embd + d]);
}

extern "C" __global__ void embed_tokens_q8_0_kernel(
        float *out,
        const int32_t *tokens,
        const unsigned char *w,
        uint32_t n_tokens,
        uint32_t n_embd) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)n_tokens * n_embd;
    if (gid >= n) return;
    uint32_t t = gid / n_embd;
    uint32_t d = gid - (uint64_t)t * n_embd;
    int32_t tok = tokens[t];
    const uint64_t row_blocks = n_embd / 32u;
    const unsigned char *blk = w + ((uint64_t)tok * row_blocks + (d >> 5)) * 34u;
    const float scale = __half2float(*(const __half *)blk);
    out[gid] = scale * (float)((const int8_t *)(blk + 2))[d & 31u];
}

extern "C" __global__ void embed_tokens_bf16_kernel(
        float *out,
        const uint16_t *weights,
        const int32_t *tokens,
        uint32_t n_tokens,
        uint32_t n_embd,
        uint32_t n_vocab) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tokens * n_embd;
    if (i >= n) return;
    const uint32_t token_row = (uint32_t)(i / n_embd);
    const uint32_t col = (uint32_t)(i - (uint64_t)token_row * n_embd);
    const int32_t token = tokens[token_row];
    if (token < 0 || (uint32_t)token >= n_vocab) {
        out[i] = 0.0f;
        return;
    }
    const uint16_t bits = weights[(uint64_t)(uint32_t)token * n_embd + col];
    out[i] = __uint_as_float((uint32_t)bits << 16);
}
