/* Activation, QK-norm+rope, group-norm, channel-norm and upsample entries of
 * the `diffusion` module. Include-only piece of `diffusion.cu`; ported from the
 * Metal twin.
 *
 * tanh and the GELU cubic need the accurate `tanhf` (KERNEL.toml drops
 * `--use_fast_math`): the cubic drives the argument past the fast form's
 * overflow point and hands back NaN from a row the transformer produces
 * routinely, and the gate tanh multiplies every block's whole branch.
 */

/* out[i] = tanh(in[i]). */
extern "C" __global__ void kernel_dit_tanh_f32(
        const dit_args_unary args, float* out, const float* in) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    out[i] = tanhf(in[i]);
}

/* out[i] = 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))). */
extern "C" __global__ void kernel_dit_gelu_tanh_f32(
        const dit_args_unary args, float* out, const float* in) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    out[i] = 0.5f * x * (1.0f + tanhf(inner));
}

/* Per-head RMS norm with a per-dimension scale, then interleaved RoPE from a
 * per-token angle table, in place over SEPARATE packed q and k buffers (a block
 * whose projections are three linears). Grid (tokens, heads) x 64. Each thread
 * owns whole pairs (x[2i], x[2i+1]) - adjacent, not the two halves of the head -
 * so the rotation reads only values its own thread normalized. cos/sin are
 * (tokens, head_dim/2); q and k rows are q_stride / k_stride elements apart. */
extern "C" __global__ void kernel_dit_qk_norm_rope_f32(
        const dit_args_qk_norm_rope args, float* q, float* k, const float* q_scale,
        const float* k_scale, const float* cos_t, const float* sin_t) {
    __shared__ float scratch[3];
    const u32 t = blockIdx.x;
    const u32 h = blockIdx.y;
    const u32 tid = threadIdx.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    const u32 half_d = args.head_dim / 2u;
    const float* ct = cos_t + (u64)t * half_d;
    const float* st = sin_t + (u64)t * half_d;
    for (u32 section = 0u; section < 2u; section++) {
        float* row = section == 0u
                ? q + (u64)t * args.q_stride + (u64)h * args.head_dim
                : k + (u64)t * args.k_stride + (u64)h * args.head_dim;
        const float* scale = section == 0u ? q_scale : k_scale;
        float part = 0.0f;
        for (u32 d = tid; d < args.head_dim; d += DIT_BLOCK) {
            const float v = row[d];
            part += v * v;
        }
        const float ms = dit_block_sum(part, scratch, tid) / (float)args.head_dim;
        const float inv = 1.0f / sqrtf(ms + args.eps);
        for (u32 p = tid; p < half_d; p += DIT_BLOCK) {
            const float x0 = row[2u * p] * inv * scale[2u * p];
            const float x1 = row[2u * p + 1u] * inv * scale[2u * p + 1u];
            const float c = ct[p];
            const float s = st[p];
            row[2u * p] = x0 * c - x1 * s;
            row[2u * p + 1u] = x0 * s + x1 * c;
        }
        /* The k section's reduction reuses the scratch. */
        __syncthreads();
    }
}

/* Elements [start, end) of a group's slice c of `chunks`. */
static __device__ __forceinline__ void dit_gn_slice(
        u32 per_group, u32 chunks, u32 c, u32* start, u32* end) {
    const u32 len = (per_group + chunks - 1u) / chunks;
    const u32 s = min(c * len, per_group);
    *start = s;
    *end = min(s + len, per_group);
}

/* Group norm pass 1 of 2: the mean and the centered sum of squares (M2) of one
 * slice of one (batch, group), written to partials[(g*chunks + c)*2 + {0,1}].
 * Grid (batch*groups, chunks) x 64. */
extern "C" __global__ void kernel_dit_group_norm_stats_f32(
        const dit_args_group_norm args, float* partials, const float* in) {
    __shared__ float scratch[3];
    const u32 g = blockIdx.x;
    const u32 c = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 per_group = args.channels_per_group * args.spatial;
    if (g >= args.batch * args.num_groups || c >= args.chunks) {
        return;
    }
    u32 lo, hi;
    dit_gn_slice(per_group, args.chunks, c, &lo, &hi);
    const float* src = in + (u64)g * per_group;
    const float n = (float)(hi - lo);
    float part = 0.0f;
    for (u32 i = lo + tid; i < hi; i += DIT_BLOCK) {
        part += src[i];
    }
    const float mean = n > 0.0f ? dit_block_sum(part, scratch, tid) / n : 0.0f;
    float m2 = 0.0f;
    for (u32 i = lo + tid; i < hi; i += DIT_BLOCK) {
        const float dv = src[i] - mean;
        m2 += dv * dv;
    }
    m2 = dit_block_sum(m2, scratch, tid);
    if (tid == 0u) {
        partials[((u64)g * args.chunks + c) * 2u] = mean;
        partials[((u64)g * args.chunks + c) * 2u + 1u] = m2;
    }
}

/* Group norm pass 2 of 2: combines the group's slice statistics (Chan's
 * pairwise update, in slice order, so every run agrees) and writes
 * (x - mean)/sqrt(var + eps) * weight[ch] + bias[ch] over one slice. Element i
 * of a group belongs to channel (g % groups)*channels_per_group + i/spatial;
 * the affine buffers hold `channels` elements. Same grid as pass 1. */
extern "C" __global__ void kernel_dit_group_norm_apply_f32(
        const dit_args_group_norm args, float* out, const float* in, const float* weight,
        const float* bias, const float* partials) {
    const u32 g = blockIdx.x;
    const u32 c = blockIdx.y;
    const u32 tid = threadIdx.x;
    const u32 per_group = args.channels_per_group * args.spatial;
    if (g >= args.batch * args.num_groups || c >= args.chunks) {
        return;
    }
    float count = 0.0f;
    float mean = 0.0f;
    float m2 = 0.0f;
    for (u32 j = 0; j < args.chunks; ++j) {
        u32 lo, hi;
        dit_gn_slice(per_group, args.chunks, j, &lo, &hi);
        const float nb = (float)(hi - lo);
        if (nb == 0.0f) {
            continue;
        }
        const float mb = partials[((u64)g * args.chunks + j) * 2u];
        const float m2b = partials[((u64)g * args.chunks + j) * 2u + 1u];
        const float total = count + nb;
        const float delta = mb - mean;
        mean += delta * (nb / total);
        m2 += m2b + delta * delta * (count * nb / total);
        count = total;
    }
    const float inv = 1.0f / sqrtf(m2 / (float)per_group + args.eps);
    u32 lo, hi;
    dit_gn_slice(per_group, args.chunks, c, &lo, &hi);
    const u64 base = (u64)g * per_group;
    const u32 ch0 = (g % args.num_groups) * args.channels_per_group;
    for (u32 i = lo + tid; i < hi; i += DIT_BLOCK) {
        const u32 ch = ch0 + i / args.spatial;
        out[base + i] = (in[base + i] - mean) * inv * weight[ch] + bias[ch];
    }
}

/* Nearest-neighbour 2x upsampling (batch, channels, h, w) -> (.., 2h, 2w).
 * Grid (w, h, batch*channels) x 64; thread 0 fans one source pixel out to its
 * four destinations (the host launches 64 threads; the rest exit). */
extern "C" __global__ void kernel_dit_upsample_nearest_f32(
        const dit_args_upsample args, float* out, const float* in) {
    if (threadIdx.x != 0u) {
        return;
    }
    const u32 j = blockIdx.x;
    const u32 i = blockIdx.y;
    const u32 nc = blockIdx.z;
    if (j >= args.w || i >= args.h) {
        return;
    }
    const float v = in[((u64)nc * args.h + i) * args.w + j];
    float* dst = out + (u64)nc * (4u * args.h * args.w);
    for (u32 di = 0u; di < 2u; di++) {
        for (u32 dj = 0u; dj < 2u; dj++) {
            dst[((2u * i + di) * 2u * args.w) + 2u * j + dj] = v;
        }
    }
}

/* Channel RMS norm pass 1: inv[s] = scale / max(sqrt(sum_c x[c,s]^2), eps), the
 * squares folded in channel order. One thread per pixel (adjacent threads read
 * adjacent pixels of a channel plane: coalesced). Grid ceil(spatial/64) x 64. */
extern "C" __global__ void kernel_dit_channel_rms_inv_f32(
        const dit_args_channel_rms args, float* inv, const float* in) {
    const u32 s = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (s >= args.spatial) {
        return;
    }
    float ss = 0.0f;
    for (u32 c = 0; c < args.channels; ++c) {
        const float v = in[(u64)c * args.spatial + s];
        ss += v * v;
    }
    inv[s] = args.scale / fmaxf(sqrtf(ss), args.eps);
}
