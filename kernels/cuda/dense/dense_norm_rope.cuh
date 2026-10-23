/* Per-head RMSNorm plus the parametric rotation, for the query half and for the
 * key/value half that lands in the cache.
 *
 * Two things separate this pair from an ordinary neox rope, and both are
 * arguments rather than constants. The rotation derives its frequency ladder
 * over `freq_dims`, which need not be the rotated width: a head of 512 whose
 * ladder is the whole head but whose leading 128 dimensions alone rotate pairs
 * (i, i + 256), not (i, i + 64), and only the first of those is the
 * checkpoint's rotation. And the value half is normalized -- by the same mean
 * square the key uses, with NO weight at all -- whenever the layer says so.
 *
 * What is NOT here is an output gate. This family's query projection is
 * `n_head * head_dim` wide and carries nothing beside the query, so the query
 * kernel writes one row and reads one row.
 */
#ifndef SHISO_DENSE_NORM_ROPE_CUH
#define SHISO_DENSE_NORM_ROPE_CUH

#include "dense_common.cuh"

/* The query half: normalize one head and rotate it.
 *
 * One threadgroup per (head, token); the head is staged in shared memory
 * because the rotation reads a pair of dimensions that no single lane owns. */
extern "C" __global__ void kernel_dense_q_norm_rope(
        const struct dense_args_q_norm_rope args,
        float *__restrict__ q_out,
        const float *__restrict__ q_in,
        const float *__restrict__ w,
        const int32_t *__restrict__ pos) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;
    /* One row of the q projection is the query and nothing else. */
    const uint64_t base = ((uint64_t) t * args.n_head + h) * head_dim;

    float *scratch = (float *) dense_smem;
    float *s_q = scratch + DENSE_BLOCK_SUM_SLOTS;

    float sum = 0.0f;
    for (unsigned i = tid; i < head_dim; i += nt) {
        const float v = q_in[base + i];
        s_q[i] = v;
        sum += v * v;
    }
    /* The weight multiplies the normalized value directly: this family stores
     * the weight itself rather than a deviation from one. */
    const float inv = rsqrtf(dense_block_sum(sum, scratch, tid, nt) / (float) head_dim + args.eps);
    for (unsigned i = tid; i < head_dim; i += nt) {
        s_q[i] *= inv * w[i];
    }
    __syncthreads();
    dense_rope_pairs(s_q, args.n_rot, args.freq_dims, args.rope_base,
                     (float) (uint32_t) pos[t], tid, nt);
    /* A pair is rotated on the lane owning its first dimension, so the second
     * half of every pair is left by a different lane than the one storing it. */
    __syncthreads();
    for (unsigned i = tid; i < head_dim; i += nt) {
        q_out[base + i] = s_q[i];
    }
}

/* The key/value half: normalize and rotate the key, normalize the value when
 * the layer asks, and write both into the f16 cache row the page table names.
 *
 * The cache is f16 because a decode step reads all of it. The two halves land
 * in one launch because a position's key and its value occupy the same row, and
 * one launch is what keeps them in step. */
extern "C" __global__ void kernel_dense_kv_norm_rope_store(
        const struct dense_args_kv_norm_rope args,
        __half *__restrict__ k_cache,
        __half *__restrict__ v_cache,
        const float *__restrict__ k_in,
        const float *__restrict__ v_in,
        const float *__restrict__ kw,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ pages) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;
    /* One translation per stored row: the table is a handful of entries every
     * lane of the block reads, so it is a cached broadcast load. */
    const uint64_t row = dense_kv_row(
            dense_row_pages(pages, t, args.page_stride),
            (uint32_t) pos[t], args.page_tokens);
    const uint64_t src = ((uint64_t) t * args.n_head_kv + h) * head_dim;
    const uint64_t slot = ((uint64_t) row * args.n_head_kv + h) * head_dim;

    float *scratch = (float *) dense_smem;
    float *s_row = scratch + DENSE_BLOCK_SUM_SLOTS;

    float ksum = 0.0f;
    for (unsigned i = tid; i < head_dim; i += nt) {
        const float x = k_in[src + i];
        s_row[i] = x;
        ksum += x * x;
    }
    const float kinv =
        rsqrtf(dense_block_sum(ksum, scratch, tid, nt) / (float) head_dim + args.eps);
    for (unsigned i = tid; i < head_dim; i += nt) {
        s_row[i] *= kinv * kw[i];
    }
    __syncthreads();
    dense_rope_pairs(s_row, args.n_rot, args.freq_dims, args.rope_base,
                     (float) (uint32_t) pos[t], tid, nt);
    __syncthreads();
    for (unsigned i = tid; i < head_dim; i += nt) {
        k_cache[slot + i] = __float2half(s_row[i]);
    }

    /* The value reuses the key's staging row, so nothing may still be reading
     * it. */
    __syncthreads();

    float vsum = 0.0f;
    for (unsigned i = tid; i < head_dim; i += nt) {
        const float x = v_in[src + i];
        s_row[i] = x;
        vsum += x * x;
    }
    /* Unweighted on purpose: this is the same mean square the key takes, and
     * the layer carries no value norm weight to apply after it. `v_norm` is
     * uniform over the block, so the reduction's barriers are reached by every
     * lane or by none. */
    float vinv = 1.0f;
    if (args.v_norm != 0u) {
        vinv = rsqrtf(dense_block_sum(vsum, scratch, tid, nt) / (float) head_dim + args.eps);
    }
    /* No barrier guards the staging row here, and none is needed: the loop
     * above and the loop below walk the same `i = tid; i += nt` index set, so
     * every lane reads back only what it wrote. The barriers earlier in this
     * kernel are there because the rotation reads a pair of dimensions that no
     * single lane owns, which is a different thing. */
    for (unsigned i = tid; i < head_dim; i += nt) {
        v_cache[slot + i] = __float2half(s_row[i] * vinv);
    }
}

/* --- rotation without a norm ----------------------------------------------
 *
 * The pair below is the two kernels above with the per-head norm taken out,
 * for a family whose attention rotates its queries and keys and normalizes
 * neither. A unit weight on the kernels above is NOT this: their division by
 * the head's root mean square is unconditional, and a unit weight only removes
 * the multiply after it.
 *
 * The argument blocks are the norm pair's, with `eps` and `v_norm` unread, and
 * the staging row sits where the norm pair's does, so the launches ask for the
 * same shared memory. These twins mirror `kernels/metal/dense/dense_norm_rope.metal`
 * and have NOT been compiled or run on a CUDA device yet. */

/* The query half: rotate one head and store it. */
extern "C" __global__ void kernel_dense_q_rope(
        const struct dense_args_q_norm_rope args,
        float *__restrict__ q_out,
        const float *__restrict__ q_in,
        const int32_t *__restrict__ pos) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;
    const uint64_t base = ((uint64_t) t * args.n_head + h) * head_dim;

    float *s_q = (float *) dense_smem + DENSE_BLOCK_SUM_SLOTS;
    for (unsigned i = tid; i < head_dim; i += nt) {
        s_q[i] = q_in[base + i];
    }
    __syncthreads();
    dense_rope_pairs(s_q, args.n_rot, args.freq_dims, args.rope_base,
                     (float) (uint32_t) pos[t], tid, nt);
    __syncthreads();
    for (unsigned i = tid; i < head_dim; i += nt) {
        q_out[base + i] = s_q[i];
    }
}

/* The key/value half: rotate the key, and write it and the raw value into the
 * f16 cache row the page table names. */
extern "C" __global__ void kernel_dense_kv_rope_store(
        const struct dense_args_kv_norm_rope args,
        __half *__restrict__ k_cache,
        __half *__restrict__ v_cache,
        const float *__restrict__ k_in,
        const float *__restrict__ v_in,
        const int32_t *__restrict__ pos,
        const int32_t *__restrict__ pages) {
    const unsigned head_dim = args.head_dim;
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;
    const uint64_t row = dense_kv_row(
            dense_row_pages(pages, t, args.page_stride),
            (uint32_t) pos[t], args.page_tokens);
    const uint64_t src = ((uint64_t) t * args.n_head_kv + h) * head_dim;
    const uint64_t slot = ((uint64_t) row * args.n_head_kv + h) * head_dim;

    float *s_row = (float *) dense_smem + DENSE_BLOCK_SUM_SLOTS;
    for (unsigned i = tid; i < head_dim; i += nt) {
        s_row[i] = k_in[src + i];
    }
    __syncthreads();
    dense_rope_pairs(s_row, args.n_rot, args.freq_dims, args.rope_base,
                     (float) (uint32_t) pos[t], tid, nt);
    __syncthreads();
    /* The value never touches the staging row, so it is stored in the same
     * loop as the key. */
    for (unsigned i = tid; i < head_dim; i += nt) {
        k_cache[slot + i] = __float2half(s_row[i]);
        v_cache[slot + i] = __float2half(v_in[src + i]);
    }
}

#endif /* SHISO_DENSE_NORM_ROPE_CUH */
