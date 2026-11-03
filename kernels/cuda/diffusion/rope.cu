/* Rotary-embedding entries of the `diffusion` module: the 4-axis interleaved
 * rotation of the DiT blocks and the rotate-half rotation of the text encoder.
 * Include-only piece of `diffusion.cu`; ported from the Metal twin.
 *
 * Needs accurate sinf/cosf: the angles reach thousands of radians, which is why
 * KERNEL.toml drops `--use_fast_math` for this module.
 */

/* One section's interleaved-pair rotation: pair j of axis a turns by
 * positions[token*4 + a] * freqs[a*freqs_per_axis + j], the pairs adjacent
 * within the axis (i0 = a*axis_dim + 2j). `buf + row` is the section start. */
static __device__ __forceinline__ void dit_rope_section(
        float* buf, u64 row, const float* positions, const float* freqs, u32 token,
        u32 head_dim, u32 freqs_per_axis, u32 tid) {
    const u32 axis_dim = head_dim / 4u;
    const u32 pairs = 4u * freqs_per_axis;
    for (u32 p = tid; p < pairs; p += DIT_BLOCK) {
        const u32 a = p / freqs_per_axis;
        const u32 j = p - a * freqs_per_axis;
        const float angle = positions[token * 4u + a] * freqs[a * freqs_per_axis + j];
        const float c = cosf(angle);
        const float s = sinf(angle);
        const u64 i0 = row + a * axis_dim + j * 2u;
        const float x0 = buf[i0];
        const float x1 = buf[i0 + 1u];
        buf[i0] = x0 * c - x1 * s;
        buf[i0 + 1u] = x0 * s + x1 * c;
    }
}

/* 4-axis interleaved RoPE on Q and K in one launch, in place, rows of
 * row_stride elements (packed or fused qkv). Grid (tokens, heads) x 64: one
 * block rotates the q section at q_offset + h*head_dim in the q binding and the
 * k section at k_offset + h*head_dim in the k binding. The transformer may bind
 * the same fused buffer to both with different offsets; the sections a block
 * writes are disjoint from every other block's, so q rotates once, k once, and
 * v is never touched. */
extern "C" __global__ void kernel_dit_rope_4d(
        const dit_args_rope_4d args, float* q, float* k, const float* positions,
        const float* freqs) {
    const u32 t = blockIdx.x;
    const u32 h = blockIdx.y;
    const u32 tid = threadIdx.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    dit_rope_section(q, (u64)t * args.row_stride + args.q_offset + (u64)h * args.head_dim,
                     positions, freqs, t, args.head_dim, args.freqs_per_axis, tid);
    dit_rope_section(k, (u64)t * args.row_stride + args.k_offset + (u64)h * args.head_dim,
                     positions, freqs, t, args.head_dim, args.freqs_per_axis, tid);
}

/* Rotate-half 1-axis RoPE on one packed (tokens, heads, head_dim) tensor, token
 * index as the position; rows from pinned_from on use pinned_pos instead
 * (pinned_from == tokens pins nothing). Pair (i, i + head_dim/2) turns by
 * position * freqs[i]. Grid (tokens, heads) x 64. */
extern "C" __global__ void kernel_dit_rope_half(
        const dit_args_rope_half args, float* tensor, const float* freqs) {
    const u32 t = blockIdx.x;
    const u32 h = blockIdx.y;
    const u32 tid = threadIdx.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    const u32 fold = args.head_dim / 2u;
    const u32 position = t >= args.pinned_from ? args.pinned_pos : t;
    float* row = tensor + ((u64)t * args.heads + h) * args.head_dim;
    for (u32 i = tid; i < fold; i += DIT_BLOCK) {
        const float angle = (float)position * freqs[i];
        const float c = cosf(angle);
        const float s = sinf(angle);
        const float x0 = row[i];
        const float x1 = row[i + fold];
        row[i] = x0 * c - x1 * s;
        row[i + fold] = x1 * c + x0 * s;
    }
}
