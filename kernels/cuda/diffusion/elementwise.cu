/* Elementwise, row-wise and gather entries of the `diffusion` module.
 *
 * Include-only piece of `diffusion.cu` (see KERNEL.toml). Ported from the
 * Metal twin, same index math and same fold order where the numbers depend on
 * it. Every entry is `extern "C"` with its name spelled out in full, because
 * the host resolves entries by those literal names (`REQUIRED_KERNELS`).
 */

/* `out[i] = silu(in[i])`. Grid ceil(count/64) x 64. */
extern "C" __global__ void kernel_dit_silu_f32(
        const dit_args_unary args, float* out, const float* in) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    out[i] = dit_silu(in[i]);
}

/* SiLU gate over a fused row: out[r,i] = silu(in[r,i]) * in[r,inner+i]. The
 * gate is the FIRST half. `in` may start mid-row (rows are `stride` wide); the
 * output is packed (rows, inner). Grid (ceil(inner/64), rows) x 64. */
extern "C" __global__ void kernel_dit_swiglu_f32(
        const dit_args_swiglu args, float* out, const float* in) {
    const u32 r = blockIdx.y;
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (r >= args.rows || i >= args.inner) {
        return;
    }
    const float* row = in + (u64)r * args.stride;
    const float g = row[i];
    out[(u64)r * args.inner + i] = dit_silu(g) * row[args.inner + i];
}

/* out[i] = silu(a[i]) * b[i] over two packed buffers (`a` is the gate). */
extern "C" __global__ void kernel_dit_swiglu2_f32(
        const dit_args_unary args, float* out, const float* a, const float* b) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    out[i] = dit_silu(a[i]) * b[i];
}

/* out[i] = x[i] * (1 + scale[i % period]) + shift[i % period]. */
extern "C" __global__ void kernel_dit_modulate_f32(
        const dit_args_modulate args, float* out, const float* x,
        const float* scale, const float* shift) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const u32 d = i % args.period;
    out[i] = x[i] * (1.0f + scale[d]) + shift[d];
}

/* out = a + gate[i % period] * b. */
extern "C" __global__ void kernel_dit_add_gated_f32(
        const dit_args_add_gated args, float* out, const float* a,
        const float* gate, const float* b) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    out[i] = a[i] + gate[i % args.period] * b[i];
}

/* out[i] = a[i] + b[i]. */
extern "C" __global__ void kernel_dit_add_f32(
        const dit_args_unary args, float* out, const float* a, const float* b) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    out[i] = a[i] + b[i];
}

/* out[i] = x[i] + dt * v[i]: the Euler update. Not fused into an FMA: the host
 * twin rounds the product first, and the sampler's loop is compared against it. */
extern "C" __global__ void kernel_dit_axpy_f32(
        const dit_args_axpy args, float* out, const float* x, const float* v) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    out[i] = __fadd_rn(x[i], __fmul_rn(args.dt, v[i]));
}

/* RMS norm of each row: out = x * (1/sqrt(mean(x^2)+eps)) * w.
 * Grid rows x 64; one block per row. */
extern "C" __global__ void kernel_dit_rmsnorm_f32(
        const dit_args_norm args, float* out, const float* in, const float* weight) {
    __shared__ float scratch[3];
    const u32 r = blockIdx.x;
    const u32 tid = threadIdx.x;
    if (r >= args.rows) {
        return;
    }
    const float* row = in + (u64)r * args.dim;
    float* dst = out + (u64)r * args.dim;
    float part = 0.0f;
    for (u32 d = tid; d < args.dim; d += DIT_BLOCK) {
        const float v = row[d];
        part += v * v;
    }
    const float ms = dit_block_sum(part, scratch, tid) / (float)args.dim;
    const float inv = 1.0f / sqrtf(ms + args.eps);
    for (u32 d = tid; d < args.dim; d += DIT_BLOCK) {
        dst[d] = row[d] * inv * weight[d];
    }
}

/* Row softmax of scale * x, in place; one 64-thread block per row. `scale`
 * must be positive (the max of the scaled row is the scaled max). */
extern "C" __global__ void kernel_dit_softmax_rows_f32(
        const dit_args_softmax args, float* x) {
    __shared__ float scratch[3];
    const u32 r = blockIdx.x;
    const u32 tid = threadIdx.x;
    if (r >= args.rows) {
        return;
    }
    float* row = x + (u64)r * args.cols;
    float m = -DIT_INF;
    for (u32 d = tid; d < args.cols; d += DIT_BLOCK) {
        m = fmaxf(m, row[d]);
    }
    m = dit_block_max(m, scratch, tid);
    float part = 0.0f;
    for (u32 d = tid; d < args.cols; d += DIT_BLOCK) {
        const float e = expf((row[d] - m) * args.scale);
        row[d] = e;
        part += e;
    }
    const float inv = 1.0f / dit_block_sum(part, scratch, tid);
    for (u32 d = tid; d < args.cols; d += DIT_BLOCK) {
        row[d] *= inv;
    }
}

/* Weightless LayerNorm of each row: out = (x - mean) / sqrt(var + eps). */
extern "C" __global__ void kernel_dit_layer_norm_f32(
        const dit_args_norm args, float* out, const float* in) {
    __shared__ float scratch[3];
    const u32 r = blockIdx.x;
    const u32 tid = threadIdx.x;
    if (r >= args.rows) {
        return;
    }
    const float* row = in + (u64)r * args.dim;
    float* dst = out + (u64)r * args.dim;
    float part = 0.0f;
    for (u32 d = tid; d < args.dim; d += DIT_BLOCK) {
        part += row[d];
    }
    const float mean = dit_block_sum(part, scratch, tid) / (float)args.dim;
    float var = 0.0f;
    for (u32 d = tid; d < args.dim; d += DIT_BLOCK) {
        const float dv = row[d] - mean;
        var += dv * dv;
    }
    const float inv = 1.0f / sqrtf(dit_block_sum(var, scratch, tid) / (float)args.dim + args.eps);
    for (u32 d = tid; d < args.dim; d += DIT_BLOCK) {
        dst[d] = (row[d] - mean) * inv;
    }
}

/* Per-head RMS norm with a per-dimension scale over the q and k head sections
 * of fused qkv rows, in place. Grid (tokens, heads) x 64: one block per
 * (token, head); q at token*row_stride + head*head_dim, k heads*head_dim
 * further along the row. Blocks own disjoint sections, so in-place is safe. */
extern "C" __global__ void kernel_dit_qk_norm_f32(
        const dit_args_qk_norm args, float* qkv, const float* q_scale,
        const float* k_scale) {
    __shared__ float scratch[3];
    const u32 t = blockIdx.x;
    const u32 h = blockIdx.y;
    const u32 tid = threadIdx.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    const u64 base = (u64)t * args.row_stride + (u64)h * args.head_dim;
    const u32 k_base = args.heads * args.head_dim;
    for (u32 section = 0u; section < 2u; section++) {
        float* row = qkv + base + (section == 0u ? 0u : k_base);
        const float* scale = (section == 0u) ? q_scale : k_scale;
        float part = 0.0f;
        for (u32 d = tid; d < args.head_dim; d += DIT_BLOCK) {
            const float v = row[d];
            part += v * v;
        }
        const float ms = dit_block_sum(part, scratch, tid) / (float)args.head_dim;
        const float inv = 1.0f / sqrtf(ms + args.eps);
        for (u32 d = tid; d < args.head_dim; d += DIT_BLOCK) {
            row[d] *= inv * scale[d];
        }
        /* The second section's reduction reuses the scratch. */
        __syncthreads();
    }
}

/* out[r] = [a[r], b[r]], both packed. Grid rows x 64. */
extern "C" __global__ void kernel_dit_concat_rows_f32(
        const dit_args_concat args, float* out, const float* a, const float* b) {
    const u32 r = blockIdx.x;
    const u32 tid = threadIdx.x;
    if (r >= args.rows) {
        return;
    }
    const u32 width = args.cols_a + args.cols_b;
    const float* ra = a + (u64)r * args.cols_a;
    const float* rb = b + (u64)r * args.cols_b;
    float* dst = out + (u64)r * width;
    for (u32 i = tid; i < width; i += DIT_BLOCK) {
        dst[i] = (i < args.cols_a) ? ra[i] : rb[i - args.cols_a];
    }
}

/* Transpose a (rows, cols) f32 matrix: out[c*rows + r] = in[r*cols + c].
 * The host reuses the norm block; eps unused. Grid ceil(rows*cols/64) x 64.
 * The index is u32 like the Metal twin: a matrix past 2^32 elements is out of
 * contract. */
extern "C" __global__ void kernel_dit_transpose_f32(
        const dit_args_norm args, float* out, const float* in) {
    const u32 i = blockIdx.x * DIT_BLOCK + threadIdx.x;
    const u32 total = args.rows * args.dim;
    if (i >= total) {
        return;
    }
    const u32 r = i / args.dim;
    const u32 c = i - r * args.dim;
    out[(u64)c * args.rows + r] = in[i];
}

/* Embedding gather: token t takes bf16 table row ids[t], widened to f32.
 * Grid (tokens, ceil(hidden/64)) x 64. */
extern "C" __global__ void kernel_dit_embed_bf16_f32(
        const dit_args_embed args, float* out, const u16* table, const u32* ids) {
    const u32 t = blockIdx.x;
    const u32 d = blockIdx.y * DIT_BLOCK + threadIdx.x;
    if (t >= args.tokens || d >= args.hidden) {
        return;
    }
    const u32 id = ids[t];
    out[(u64)t * args.hidden + d] = dit_bf16(table + ((u64)id * args.hidden + d));
}

/* The residual VAE up block's parameter-free shortcut, added in place:
 * out[c,y,x] += in[(c*factor + offset + (y&1)*2 + (x&1)) / repeats, y/2, x/2].
 * Grid (ceil(4hw/64), out_c) x 64. */
extern "C" __global__ void kernel_dit_dup_up_add_f32(
        const dit_args_dup_up args, float* out, const float* in) {
    const u32 ow = 2u * args.w;
    const u32 plane = ow * 2u * args.h;
    const u32 p = blockIdx.x * DIT_BLOCK + threadIdx.x;
    const u32 c = blockIdx.y;
    if (p >= plane || c >= args.out_c) {
        return;
    }
    const u32 y = p / ow;
    const u32 x = p - y * ow;
    const u32 src = (c * args.factor + args.offset + (y & 1u) * 2u + (x & 1u)) / args.repeats;
    out[(u64)c * plane + p] +=
            in[((u64)src * args.h + (y >> 1u)) * args.w + (x >> 1u)];
}
