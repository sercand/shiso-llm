/* Per-head RMSNorm plus partial rope for the query and key projections.
 *
 * The CUDA twins of `crates/shiso-metal/metal/qwen35/rmsnorm_rope.metal`, over
 * the law `kernels/cuda/q4e/qsa.cu` carries as `q4e_qsa_q_norm_rope_kernel` and
 * `q4e_qsa_store_kv_kernel`.
 *
 * THE THREE THINGS THAT ARE EASY TO GET BACKWARDS, all of them arguments:
 *
 *   - The query projection emits a query and an output gate INTERLEAVED per
 *     head: head h owns `[q(head_dim) | gate(head_dim)]` inside a
 *     `2 * head_dim * n_head` row. Splitting the flat row down the middle
 *     instead leaves every head plausible and wrong. `q4e/qsa.cu` says so in
 *     the same words.
 *   - Rope covers the FIRST `n_rot` of `head_dim` in NeoX pairing
 *     (i with i + n_rot/2); the rest passes through. The generic kernel set's
 *     tail-rope layout rotates the last dimensions instead.
 *   - The key entry takes BOTH the linear position and, on the mrope variant,
 *     the three grid streams. The linear one, and only the linear one, reaches
 *     the page table. Translating the temporal stream instead would fold a
 *     whole image onto a handful of cache rows and leave every count in the
 *     request correct.
 *
 * Geometry, per `KERNELS.md` and unchanged from the Metal side: grid
 * `(n_head, n_tok)` / `(n_head_kv, n_tok)`, block 128, 1156 B of dynamic shared
 * memory - the block-sum slots plus one staged head row at head_dim 256.
 */
#ifndef SHISO_Q35_NORM_ROPE_CUH
#define SHISO_Q35_NORM_ROPE_CUH

#include "q35_common.cuh"

/* The query half: normalize one head, rotate it, and copy the output gate that
 * rides beside it in the same projection row.
 *
 * `mrope` picks the rotation and is a compile-time literal at each entry point,
 * so the one-dimensional path is not merely equal to what it was but is the
 * same instructions. When it is false only `pos_t` is read. */
__device__ __forceinline__ void q35_q_norm_rope_body(
        const struct q35_args_q_norm_rope &args,
        float *q_out, float *gate_out, const float *qkv, const float *w,
        float pos_t, float pos_h, float pos_w, bool mrope) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;
    /* One row of the q projection holds the query and its output gate. */
    const uint64_t src = ((uint64_t) t * args.n_head + h) * (2u * (uint64_t) head_dim);
    const uint64_t dst = ((uint64_t) t * args.n_head + h) * (uint64_t) head_dim;

    float *scratch = (float *) q35_smem;
    float *s_q = scratch + Q35_BLOCK_SUM_SLOTS;

    float sum = 0.0f;
    for (unsigned i = tid; i < head_dim; i += nt) {
        const float v = qkv[src + i];
        s_q[i] = v;
        sum += v * v;
        /* The gate rides beside the query and is copied raw: no activation, no
         * norm, no rope. It is consumed after attention. */
        gate_out[dst + i] = qkv[src + head_dim + i];
    }
    const float scale = rsqrtf(q35_block_sum(sum, scratch, tid, nt) / (float) head_dim + args.eps);
    for (unsigned i = tid; i < head_dim; i += nt) {
        s_q[i] *= scale * w[i];
    }
    __syncthreads();
    if (mrope) {
        q35_rope_mrope_interleaved(s_q, args.n_rot, args.rope_base, pos_t, pos_h, pos_w, tid, nt);
    } else {
        q35_rope_neox(s_q, args.n_rot, args.rope_base, pos_t, tid, nt);
    }
    /* NeoX rotates a pair on the lane owning its first dimension, so the second
     * half of every pair is left by a different lane than the one storing it. */
    __syncthreads();
    for (unsigned i = tid; i < head_dim; i += nt) {
        q_out[dst + i] = s_q[i];
    }
}

/* The key half: normalize one head, rotate it, and write it into the f16 cache
 * row the page table names, with the value copied beside it.
 *
 * `row` is the physical cache row the CALLER translated from the token's linear
 * position. The rotation's positions are separate arguments because the two are
 * different questions. */
__device__ __forceinline__ void q35_k_norm_rope_kv_body(
        const struct q35_args_k_norm_rope &args,
        __half *k_cache, __half *v_cache, const float *k, const float *v, const float *kw,
        uint64_t row, float pos_t, float pos_h, float pos_w, bool mrope) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const uint64_t src = ((uint64_t) t * args.n_head_kv + h) * (uint64_t) head_dim;
    const uint64_t slot = (row * args.n_head_kv + h) * (uint64_t) head_dim;

    float *scratch = (float *) q35_smem;
    float *s_k = scratch + Q35_BLOCK_SUM_SLOTS;

    float sum = 0.0f;
    for (unsigned i = tid; i < head_dim; i += 128u) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrtf(q35_block_sum(sum, scratch, tid, 128u) / (float) head_dim + args.eps);
    for (unsigned i = tid; i < head_dim; i += 128u) {
        s_k[i] *= scale * kw[i];
    }
    __syncthreads();
    if (mrope) {
        q35_rope_mrope_interleaved(s_k, args.n_rot, args.rope_base, pos_t, pos_h, pos_w, tid, 128u);
    } else {
        q35_rope_neox(s_k, args.n_rot, args.rope_base, pos_t, tid, 128u);
    }
    __syncthreads();
    for (unsigned i = tid; i < head_dim; i += 128u) {
        k_cache[slot + i] = __float2half(s_k[i]);
        v_cache[slot + i] = __float2half(v[src + i]);
    }
}

extern "C" __global__ void kernel_q35_q_norm_rope(
        const struct q35_args_q_norm_rope args,
        float *q_out, float *gate_out, const float *qkv, const float *w,
        const int32_t *pos) {
    const float p = (float) (uint32_t) pos[blockIdx.y];
    q35_q_norm_rope_body(args, q_out, gate_out, qkv, w, p, p, p, false);
}

/* The multimodal query rotation. Three position streams instead of one: a text
 * token carries the same number in all three, and a placeholder token carries
 * its temporal group, its row and its column. There is no linear position here
 * because nothing in this kernel addresses memory by position. */
extern "C" __global__ void kernel_q35_q_norm_rope_mrope(
        const struct q35_args_q_norm_rope args,
        float *q_out, float *gate_out, const float *qkv, const float *w,
        const int32_t *pos_t, const int32_t *pos_h, const int32_t *pos_w) {
    const unsigned t = blockIdx.y;
    q35_q_norm_rope_body(args, q_out, gate_out, qkv, w,
            (float) (uint32_t) pos_t[t],
            (float) (uint32_t) pos_h[t],
            (float) (uint32_t) pos_w[t],
            true);
}

extern "C" __global__ void kernel_q35_k_norm_rope_kv(
        const struct q35_args_k_norm_rope args,
        __half *k_cache, __half *v_cache, const float *k, const float *v, const float *kw,
        const int32_t *pos, const int32_t *pages) {
    /* One translation per stored row: the table is a handful of entries that
     * every lane of the block reads, so it is a cached broadcast load. */
    const uint64_t row = q35_kv_row(
            q35_row_pages(pages, blockIdx.y, args.page_stride),
            (uint32_t) pos[blockIdx.y], args.page_tokens);
    const float p = (float) (uint32_t) pos[blockIdx.y];
    q35_k_norm_rope_kv_body(args, k_cache, v_cache, k, v, kw, row, p, p, p, false);
}

/* The multimodal key rotation. The linear position still decides the cache row
 * - a placeholder token occupies one sequence slot like any other, and the page
 * table knows nothing about pictures - while the three grid streams decide the
 * angles. */
extern "C" __global__ void kernel_q35_k_norm_rope_kv_mrope(
        const struct q35_args_k_norm_rope args,
        __half *k_cache, __half *v_cache, const float *k, const float *v, const float *kw,
        const int32_t *pos, const int32_t *pages,
        const int32_t *pos_t, const int32_t *pos_h, const int32_t *pos_w) {
    const unsigned t = blockIdx.y;
    const uint64_t row = q35_kv_row(
            q35_row_pages(pages, t, args.page_stride),
            (uint32_t) pos[t], args.page_tokens);
    q35_k_norm_rope_kv_body(args, k_cache, v_cache, k, v, kw, row,
            (float) (uint32_t) pos_t[t],
            (float) (uint32_t) pos_h[t],
            (float) (uint32_t) pos_w[t],
            true);
}

#endif /* SHISO_Q35_NORM_ROPE_CUH */
