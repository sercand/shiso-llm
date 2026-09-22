/* The vision tower's dense kernels: the fused attention-plane normalization and
 * two-axis rotation, the pooler that folds a patch block into one token, and the
 * elementwise clamp every clamped linear projection is built from.
 *
 * The CUDA twins of `metal/dense/dense_vision.metal`. They live in the `dense`
 * set rather than the `vision` one for the reason the Metal originals do: they
 * are written against argument blocks and not against one checkpoint's widths,
 * and the tower they serve reuses this set's reduction helpers and its neox
 * rotation. The Qwen ViT has its own `vision`-set kernels; this is the Gemma 4
 * tower's, and the two differ in exactly the ways a second family would -- a
 * rotation over two spatial axes instead of the patch's grid position, a pooling
 * fold instead of a spatial merge, and a clamp around every projection.
 *
 * Nothing here is a byte copy. The arithmetic is the Metal twins', which is what
 * `crates/shiso-kerneltest/src/dense/vision.rs` grades both halves against.
 */
#ifndef SHISO_DENSE_VISION_CUH
#define SHISO_DENSE_VISION_CUH

#include "dense_common.cuh"

/* Per-head RMSNorm of the query, key and value planes and the two-axis rotation
 * of the query and the key, in place on the three planes.
 *
 * One block per (head, patch row). Each plane's head is staged in shared memory
 * because the rotation pairs a channel with one half a quarter-head away, which
 * no single lane owns.
 *
 * The rotation is the reference's two `ggml_rope_ext` calls, each over half the
 * head: the leading half (`[0, head_dim/2)`) rotates by the patch's column and
 * the trailing half (`[head_dim/2, head_dim)`) by its row, both pairing
 * `(j, j + head_dim/4)` at the angle `pos * base^(-2j / (head_dim/2))`. A
 * checkpoint that calls this a "partial" rotation means how the ladder is built,
 * not that any channel is left alone: every channel rotates, half by one axis
 * and half by the other.
 *
 * The value plane is normalized with no weight and never rotated -- the same
 * mean square the key takes, which is the whole of the difference between the
 * two, as in `kernel_dense_kv_norm_rope_store`.
 *
 * The dynamic shared allocation is `4 * (DENSE_BLOCK_SUM_SLOTS + 3 * head_dim)`:
 * the block reduction's scratch first, then the three staged heads. */
extern "C" __global__ void kernel_dense_vit_qkv_norm_rope2d(
        const struct dense_args_vit_rope2d args,
        float *__restrict__ q,
        float *__restrict__ k,
        float *__restrict__ v,
        const float *__restrict__ q_norm_w,
        const float *__restrict__ k_norm_w) {
    const unsigned n = blockIdx.y;
    const unsigned h = blockIdx.x;
    if (n >= args.rows) {
        return;
    }
    const unsigned hd = args.head_dim;
    const unsigned dim = args.heads * hd;
    const unsigned tid = threadIdx.x;
    const unsigned nt = blockDim.x;
    const uint64_t base = (uint64_t) n * dim + (uint64_t) h * hd;

    /* A still is one plane, so the patch's grid position is its row within the
     * whole item: column is the fast axis and row the slow one. */
    const float col = (float) (n % args.grid_w);
    const float row = (float) (n / args.grid_w);

    float *scratch = (float *) dense_smem;
    float *s_q = scratch + DENSE_BLOCK_SUM_SLOTS;
    float *s_k = s_q + hd;
    float *s_v = s_k + hd;

    float qsum = 0.0f;
    float ksum = 0.0f;
    float vsum = 0.0f;
    for (unsigned i = tid; i < hd; i += nt) {
        const float qv = q[base + i];
        const float kv = k[base + i];
        const float vv = v[base + i];
        s_q[i] = qv;
        s_k[i] = kv;
        s_v[i] = vv;
        qsum += qv * qv;
        ksum += kv * kv;
        vsum += vv * vv;
    }
    /* The weight multiplies the normalized value directly: this tower stores the
     * weight itself, not a deviation from one. */
    const float qinv = rsqrtf(dense_block_sum(qsum, scratch, tid, nt) / (float) hd + args.eps);
    const float kinv = rsqrtf(dense_block_sum(ksum, scratch, tid, nt) / (float) hd + args.eps);
    for (unsigned i = tid; i < hd; i += nt) {
        s_q[i] *= qinv * q_norm_w[i];
        s_k[i] *= kinv * k_norm_w[i];
    }
    /* The value carries no weight: it is divided by its own root mean square and
     * nothing else. */
    const float vinv = rsqrtf(dense_block_sum(vsum, scratch, tid, nt) / (float) hd + args.eps);
    for (unsigned i = tid; i < hd; i += nt) {
        s_v[i] *= vinv;
    }
    __syncthreads();

    /* Two neox rotations over the two halves of the head, each over a ladder
     * built for half the head's width. The offset slice for the trailing half is
     * a pointer into the same staged row; the helper only ever touches the
     * `n_rot` dimensions it is handed. */
    const unsigned half_hd = hd / 2u;
    dense_rope_pairs(s_q, half_hd, half_hd, args.rope_base, col, tid, nt);
    dense_rope_pairs(s_q + half_hd, half_hd, half_hd, args.rope_base, row, tid, nt);
    dense_rope_pairs(s_k, half_hd, half_hd, args.rope_base, col, tid, nt);
    dense_rope_pairs(s_k + half_hd, half_hd, half_hd, args.rope_base, row, tid, nt);
    __syncthreads();

    for (unsigned i = tid; i < hd; i += nt) {
        q[base + i] = s_q[i];
        k[base + i] = s_k[i];
        v[base + i] = s_v[i];
    }
}

/* A `3 x 3` average pool over the patch grid, scaled by `args.scale`.
 *
 * One output row folds a `3 x 3` block of patch rows into one row of the same
 * width, taking the mean of the nine and then multiplying by the scale. The
 * reference reaches the same number by `ggml_pool_2d(AVG, 3, 3, ...)` followed
 * by `ggml_scale(sqrt(n_embd))`; the sum is accumulated in the same row-major
 * order (outer row, inner column) so the division by nine and the scale land on
 * the same f32 the reference produces.
 *
 * Both grid extents are whole multiples of three by the resize rule, so the
 * nine-source index never runs off the grid and no padding is needed. */
extern "C" __global__ void kernel_dense_vit_avgpool3_scale(
        const struct dense_args_vit_pool args,
        float *__restrict__ out,
        const float *__restrict__ in) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned hidden = args.hidden;
    const unsigned out_w = args.grid_w / 3u;
    const unsigned total = (args.grid_h / 3u) * out_w * hidden;
    if (i >= total) {
        return;
    }
    const unsigned orow = i / hidden;
    const unsigned c = i % hidden;
    const unsigned oy = orow / out_w;
    const unsigned ox = orow % out_w;
    float sum = 0.0f;
    for (unsigned dy = 0u; dy < 3u; dy++) {
        for (unsigned dx = 0u; dx < 3u; dx++) {
            const unsigned sy = oy * 3u + dy;
            const unsigned sx = ox * 3u + dx;
            sum += in[((uint64_t) sy * args.grid_w + sx) * hidden + c];
        }
    }
    out[i] = (sum / 9.0f) * args.scale;
}

/* `out[i] = clamp(in[i], args.min, args.max)` over `args.count` elements.
 *
 * Elementwise and out-of-place, so the same buffer may be given as both `out`
 * and `in` to clamp a projection's result in place: each element is read once
 * and written once by the same thread, so an aliasing pair cannot race.
 *
 * Neither pointer is `__restrict__`, and that is not an omission: the aliased
 * launch is a documented call, and telling the compiler the two cannot overlap
 * while a caller overlaps them is undefined behaviour rather than a hint. */
extern "C" __global__ void kernel_dense_clamp_f32(
        const struct dense_args_clamp args,
        float *out,
        const float *in) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    out[i] = x < args.min ? args.min : (x > args.max ? args.max : x);
}

#endif /* SHISO_DENSE_VISION_CUH */
