// The flux2-klein kernel pack: every device operation the family's driver
// (`crates/shiso-models/src/flux2/ops.rs`) launches, under module `flux2`.
//
// Each kernel computes exactly what the driver's host twin computes - same
// index math, same section offsets, same accumulation order where the
// numbers depend on it - because the host twin is both the fallback for a
// device without this set and the arithmetic the tests grade against
// (`tests/flux2_kernels.rs`). The fused-row contracts (sdpa's per-operand
// q/k/v offsets, rope_4d's q/k section offsets on an aliased binding, the
// gate-first swiglu) are the ones the driver's call sites marshal; the
// oracle tests in `shiso-models/src/flux2/oracle_tests.rs` pin the same
// shapes on the host side.
//
// `flux2_common.h` is concatenated ahead of this file.

// --- elementwise -------------------------------------------------------------

/// `out[i] = silu(in[i])`.
kernel void kernel_flux2_silu_f32(
        constant flux2_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * FLUX2_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = flux2_silu(in[i]);
}

/// SiLU gate over a fused row: `out[r,i] = silu(in[r,i]) * in[r,inner+i]`.
///
/// The gate is the FIRST half (the reference's gating chunks the
/// row in two and silus the first chunk). The input is one strided buffer
/// whose view may start mid-row (the single block's halves begin at column
/// 3h of the fused row); the output is packed `(rows, inner)`. The driver
/// launches one threadgroup per output element, so thread 0 of each group
/// owns it.
kernel void kernel_flux2_swiglu_f32(
        constant flux2_args_swiglu & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    if (tpitg.x != 0u) {
        return;
    }
    const uint r = tgpig.y;
    const uint i = tgpig.x;
    if (r >= args.rows || i >= args.inner) {
        return;
    }
    device const float *row = in + (uint64_t)r * args.stride;
    const float g = row[i];
    out[(uint64_t)r * args.inner + i] = flux2_silu(g) * row[args.inner + i];
}

/// `out[i] = silu(a[i]) * b[i]` over two packed buffers: the text encoder's
/// separate gate/up projections, `a` the gate - the same first-half gating
/// the fused-row swiglu applies.
kernel void kernel_flux2_swiglu2_f32(
        constant flux2_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *a            [[buffer(2)]],
        device const float *b            [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * FLUX2_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const float g = a[i];
    out[i] = flux2_silu(g) * b[i];
}

/// `out[i] = x[i] * (1 + scale[i % period]) + shift[i % period]`: the AdaLN
/// modulation with the per-dimension vectors broadcast over the rows.
kernel void kernel_flux2_modulate_f32(
        constant flux2_args_modulate & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *x                [[buffer(2)]],
        device const float *scale            [[buffer(3)]],
        device const float *shift            [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * FLUX2_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const uint d = i % args.period;
    out[i] = x[i] * (1.0f + scale[d]) + shift[d];
}

/// `out = a + gate * b`, `gate` broadcast over `period`: the gated residual.
kernel void kernel_flux2_add_gated_f32(
        constant flux2_args_add_gated & args [[buffer(0)]],
        device float *out                     [[buffer(1)]],
        device const float *a                 [[buffer(2)]],
        device const float *gate              [[buffer(3)]],
        device const float *b                 [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * FLUX2_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = a[i] + gate[i % args.period] * b[i];
}

/// `out[i] = a[i] + b[i]`.
kernel void kernel_flux2_add_f32(
        constant flux2_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *a            [[buffer(2)]],
        device const float *b            [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * FLUX2_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = a[i] + b[i];
}

/// `out[i] = x[i] + dt * v[i]`: the Euler latent update.
kernel void kernel_flux2_axpy_f32(
        constant flux2_args_axpy & args [[buffer(0)]],
        device float *out               [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const float *v           [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * FLUX2_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = x[i] + args.dt * v[i];
}

// --- gemm ---------------------------------------------------------------------

/// The stored-width decoders the tile body calls: the weight stays in the
/// width the checkpoint stored it and is widened on read (`ushort`
/// brain-float, `uchar` E4M3, or `float`), never through a copy.
static inline float flux2_weight_value(float w) {
    return w;
}

static inline float flux2_weight_value(ushort w) {
    return as_type<float>((uint)w << 16);
}

static inline float flux2_weight_value(uchar w) {
    return flux2_e4m3(uint(w));
}

/// The staged 16x16 tile body every weight width shares.
///
/// `k` is walked strictly in increasing order - the tile loop advances by
/// FLUX2_GEMM_TILE and the inner loop walks the tile front to back - so the
/// sum is the same association the driver's host twin makes with a plain
/// `for k in 0..K` loop. That is deliberate: a blocking change that reorders
/// `k` is a change to the numbers, not just to the speed.
///
/// The scale multiplies the accumulator in the epilogue, never the weight
/// in memory. The bias is per output column, applied only when
/// `args.has_bias` says the binding is a real vector; the driver clears
/// the flag and binds its zero stand-in when the linear carries none.
template <typename W>
static inline void flux2_gemm_body(
        constant flux2_args_gemm & args,
        device float *out,
        device const float *in,
        device const W *weight,
        float scale,
        device const float *bias,
        threadgroup float (*a_tile)[FLUX2_GEMM_TILE],
        threadgroup float (*b_tile)[FLUX2_GEMM_TILE],
        uint3 tgpig,
        ushort3 tpitg) {
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const uint row = tgpig.y * FLUX2_GEMM_TILE + ty;
    const uint col = tgpig.x * FLUX2_GEMM_TILE + tx;
    // The weight row this thread stages, which is not the one it
    // accumulates: the tile is filled row-per-thread-y so `k` runs along x
    // and the load stays contiguous, then read back column-wise.
    const uint stage_n = tgpig.x * FLUX2_GEMM_TILE + ty;
    const bool has_bias = args.has_bias != 0u;

    float acc = 0.0f;
    for (uint k0 = 0u; k0 < args.k; k0 += FLUX2_GEMM_TILE) {
        const uint k = k0 + tx;
        const bool have_k = k < args.k;
        a_tile[ty][tx] = (row < args.m && have_k)
                ? in[(uint64_t)row * args.k + k]
                : 0.0f;
        const uint64_t w_idx = (uint64_t)stage_n * args.k + k;
        b_tile[ty][tx] = (stage_n < args.n && have_k)
                ? flux2_weight_value(weight[w_idx])
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < FLUX2_GEMM_TILE; kk++) {
            acc += a_tile[ty][kk] * b_tile[tx][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < args.m && col < args.n) {
        out[(uint64_t)row * args.n + col] =
                acc * scale + (has_bias ? bias[col] : 0.0f);
    }
}

/// `out[m,n] = in[m,k] . weight[n,k] * scale + bias[n]`, weight F32.
kernel void kernel_flux2_gemm_f32(
        constant flux2_args_gemm & args    [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const float *weight         [[buffer(3)]],
        constant float &weight_scale       [[buffer(4)]],
        device const float *bias           [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    // Static threadgroup arrays, not `[[threadgroup(0)]]` parameters: the
    // driver launches these with no dynamic threadgroup request.
    threadgroup float a_tile[FLUX2_GEMM_TILE][FLUX2_GEMM_TILE];
    threadgroup float b_tile[FLUX2_GEMM_TILE][FLUX2_GEMM_TILE];
    flux2_gemm_body<float>(args, out, in, weight, weight_scale, bias,
                           a_tile, b_tile, tgpig, tpitg);
}

/// The same product over an FP8 E4M3 weight: the bytes are read directly and
/// dequantized in registers during the dot, and the accumulated sum is
/// multiplied by the tensor's `weight_scale` in the epilogue.
kernel void kernel_flux2_gemm_fp8(
        constant flux2_args_gemm & args    [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const uchar *weight_fp8     [[buffer(3)]],
        constant float &weight_scale       [[buffer(4)]],
        device const float *bias           [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[FLUX2_GEMM_TILE][FLUX2_GEMM_TILE];
    threadgroup float b_tile[FLUX2_GEMM_TILE][FLUX2_GEMM_TILE];
    flux2_gemm_body<uchar>(args, out, in, weight_fp8, weight_scale, bias,
                           a_tile, b_tile, tgpig, tpitg);
}

// The brain-float GEMM is a template with two named instantiations because
// the driver decides kernel-vs-host by probing `module_entries("flux2")` for
// a name containing `kernel_flux2_gemm_bf16`, and Metal's table lists only
// CUDA-style mangled names (`device.rs::module_entries`). The plain host
// name is the launch entry; the `_Z...` instantiation is the name nvcc
// would give the same kernel, so the probe sees the set and the Metal build
// runs it instead of the host fallback. The tag parameter is otherwise
// unused: both instantiations are the same code.
#define FLUX2_GEMM_BF16_PARAMS \
        constant flux2_args_gemm & args    [[buffer(0)]], \
        device float *out                  [[buffer(1)]], \
        device const float *in             [[buffer(2)]], \
        device const ushort *weight        [[buffer(3)]], \
        constant float &weight_scale       [[buffer(4)]], \
        device const float *bias           [[buffer(5)]], \
        uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort3 tpitg[[thread_position_in_threadgroup]]

template <bool ProbeAlias>
kernel void kernel_flux2_gemm_bf16_kernel(FLUX2_GEMM_BF16_PARAMS) {
    threadgroup float a_tile[FLUX2_GEMM_TILE][FLUX2_GEMM_TILE];
    threadgroup float b_tile[FLUX2_GEMM_TILE][FLUX2_GEMM_TILE];
    flux2_gemm_body<ushort>(args, out, in, weight, weight_scale, bias,
                            a_tile, b_tile, tgpig, tpitg);
}

template [[host_name("kernel_flux2_gemm_bf16")]]
kernel void kernel_flux2_gemm_bf16_kernel<false>(FLUX2_GEMM_BF16_PARAMS);

template [[host_name("_Z22kernel_flux2_gemm_bf16v")]]
kernel void kernel_flux2_gemm_bf16_kernel<true>(FLUX2_GEMM_BF16_PARAMS);

// --- norms ----------------------------------------------------------------------

/// RMS normalization of each row: `out = x * rsqrt(mean(x^2) + eps) * w`.
kernel void kernel_flux2_rmsnorm_f32(
        constant flux2_args_norm & args  [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        device const float *weight       [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint r = tgpig.x;
    const uint tid = tpitg.x;
    if (r >= args.rows) {
        return;
    }
    device const float *row = in + (uint64_t)r * args.dim;
    device float *dst = out + (uint64_t)r * args.dim;
    float part = 0.0f;
    for (uint d = tid; d < args.dim; d += FLUX2_BLOCK) {
        const float v = row[d];
        part += v * v;
    }
    const float ms = flux2_block_sum(part, scratch, tid) / float(args.dim);
    const float inv = 1.0f / sqrt(ms + args.eps);
    for (uint d = tid; d < args.dim; d += FLUX2_BLOCK) {
        dst[d] = row[d] * inv * weight[d];
    }
}

/// Weightless LayerNorm of each row: `out = (x - mean) / sqrt(var + eps)`.
///
/// Mean-subtracting, unlike rmsnorm, and with no affine parameters: the
/// transformer's block norms are `LayerNorm(affine=False)`, whose shift and
/// scale come from the AdaLN modulation instead.
kernel void kernel_flux2_layer_norm_f32(
        constant flux2_args_norm & args  [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint r = tgpig.x;
    const uint tid = tpitg.x;
    if (r >= args.rows) {
        return;
    }
    device const float *row = in + (uint64_t)r * args.dim;
    device float *dst = out + (uint64_t)r * args.dim;
    float part = 0.0f;
    for (uint d = tid; d < args.dim; d += FLUX2_BLOCK) {
        part += row[d];
    }
    const float mean = flux2_block_sum(part, scratch, tid) / float(args.dim);
    float var = 0.0f;
    for (uint d = tid; d < args.dim; d += FLUX2_BLOCK) {
        const float dv = row[d] - mean;
        var += dv * dv;
    }
    const float inv = 1.0f / sqrt(flux2_block_sum(var, scratch, tid) / float(args.dim) + args.eps);
    for (uint d = tid; d < args.dim; d += FLUX2_BLOCK) {
        dst[d] = (row[d] - mean) * inv;
    }
}

/// Per-head RMS normalization with a per-dimension scale over the q and k
/// head sections of fused qkv rows, in place: the QK-norm the reference
/// applies before RoPE.
///
/// One threadgroup per (token, head). The q section starts at
/// `token * row_stride + head * head_dim` and the k section
/// `heads * head_dim` further along the row; the v section, which no offset
/// points at, is never touched. Different threadgroups own disjoint head
/// sections, so the in-place writes cannot race.
kernel void kernel_flux2_qk_norm_f32(
        constant flux2_args_qk_norm & args [[buffer(0)]],
        device float *qkv                   [[buffer(1)]],
        device const float *q_scale         [[buffer(2)]],
        device const float *k_scale         [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    const uint base = t * args.row_stride + h * args.head_dim;
    const uint k_base = args.heads * args.head_dim;
    for (uint section = 0u; section < 2u; section++) {
        device float *row = qkv + base + (section == 0u ? 0u : k_base);
        device const float *scale = (section == 0u) ? q_scale : k_scale;
        float part = 0.0f;
        for (uint d = tid; d < args.head_dim; d += FLUX2_BLOCK) {
            const float v = row[d];
            part += v * v;
        }
        const float ms = flux2_block_sum(part, scratch, tid) / float(args.head_dim);
        const float inv = 1.0f / sqrt(ms + args.eps);
        for (uint d = tid; d < args.head_dim; d += FLUX2_BLOCK) {
            row[d] *= inv * scale[d];
        }
        // The second section's reduction reuses the scratch; hold until
        // every thread has consumed the first broadcast.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// Row-wise concatenation: `out[r] = [a[r], b[r]]`, both sides packed (the
/// single block's `linear2` input, the reference's concatenation of attention
/// and MLP output).
kernel void kernel_flux2_concat_rows_f32(
        constant flux2_args_concat & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *a              [[buffer(2)]],
        device const float *b              [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint r = tgpig.x;
    const uint tid = tpitg.x;
    if (r >= args.rows) {
        return;
    }
    const uint width = args.cols_a + args.cols_b;
    device const float *ra = a + (uint64_t)r * args.cols_a;
    device const float *rb = b + (uint64_t)r * args.cols_b;
    device float *dst = out + (uint64_t)r * width;
    for (uint i = tid; i < width; i += FLUX2_BLOCK) {
        dst[i] = (i < args.cols_a) ? ra[i] : rb[i - args.cols_a];
    }
}

// --- attention --------------------------------------------------------------------

/// One joint-attention dot: head `h` of key row `s` against the query head,
/// reduced across the block. `krow` is the operand-relative start of the
/// key head; the per-operand offsets are already folded into it by the
/// caller's index math.
static inline float flux2_head_dot(
        device const float *q, uint qrow,
        device const float *k, uint krow,
        uint head_dim, uint tid, threadgroup float *scratch) {
    float part = 0.0f;
    for (uint d = tid; d < head_dim; d += FLUX2_BLOCK) {
        part += q[qrow + d] * k[krow + d];
    }
    return flux2_block_sum(part, scratch, tid);
}

/// Joint-sequence scaled dot-product attention over `[txt, img, cache]` key
/// rows, reading strided fused-qkv rows where the driver asks.
///
/// One threadgroup per (query row, head); query rows below `txt_tokens`
/// come from the txt stream and write `out_txt`, the rest from the img
/// stream and write `out_img`. Head `h` of a row starts at
/// `row * stride + offset + h * head_dim` for each operand's OWN offset -
/// the transformer binds one fused buffer as q, k and v three times, and
/// reading k or v from the q offset would silently compute
/// `softmax(q q^T) q`. Cached rows carry the same layout against
/// `cache_stride`; when `cache_tokens == 0` the cache bindings are the
/// driver's stand-in and are never read. The key order is irrelevant to the
/// answer (the softmax is order-invariant); the scores are max-subtracted
/// and normalized exactly as the host twin does.
kernel void kernel_flux2_sdpa_f32(
        constant flux2_args_sdpa & args [[buffer(0)]],
        device float *out_txt            [[buffer(1)]],
        device float *out_img            [[buffer(2)]],
        device const float *q_txt        [[buffer(3)]],
        device const float *k_txt        [[buffer(4)]],
        device const float *v_txt        [[buffer(5)]],
        device const float *q_img        [[buffer(6)]],
        device const float *k_img        [[buffer(7)]],
        device const float *v_img        [[buffer(8)]],
        device const float *cache_k      [[buffer(9)]],
        device const float *cache_v      [[buffer(10)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg.x;
    if (t >= args.txt_tokens + args.img_tokens || h >= args.heads) {
        return;
    }
    const uint hd = args.head_dim;
    const uint keys = args.txt_tokens + args.img_tokens + args.cache_tokens;
    device const float *q;
    uint qrow;
    if (t < args.txt_tokens) {
        q = q_txt;
        qrow = t * args.txt_stride + args.q_offset + h * hd;
    } else {
        q = q_img;
        qrow = (t - args.txt_tokens) * args.img_stride + args.q_offset + h * hd;
    }
    const float scale = 1.0f / sqrt(float(hd));

    // Pass 1: the running maximum over every key's score.
    float max_score = -INFINITY;
    for (uint s = 0u; s < keys; s++) {
        device const float *k;
        uint krow;
        if (s < args.txt_tokens) {
            k = k_txt;
            krow = s * args.txt_stride + args.k_offset + h * hd;
        } else if (s < args.txt_tokens + args.img_tokens) {
            k = k_img;
            krow = (s - args.txt_tokens) * args.img_stride + args.k_offset + h * hd;
        } else {
            k = cache_k;
            krow = (s - args.txt_tokens - args.img_tokens) * args.cache_stride
                    + args.k_offset + h * hd;
        }
        const float score = flux2_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
        max_score = max(max_score, score);
    }

    // Pass 2: the exponentials, their sum, and the value mix. Every thread
    // recomputes the same scores (the dot is block-broadcast), so `sum`
    // needs no second reduction; each thread owns its slice of the head.
    float acc[FLUX2_ATTN_SLOTS];
    for (uint slot = 0u; slot < FLUX2_ATTN_SLOTS; slot++) {
        acc[slot] = 0.0f;
    }
    float sum = 0.0f;
    for (uint s = 0u; s < keys; s++) {
        device const float *k;
        uint krow;
        if (s < args.txt_tokens) {
            k = k_txt;
            krow = s * args.txt_stride + args.k_offset + h * hd;
        } else if (s < args.txt_tokens + args.img_tokens) {
            k = k_img;
            krow = (s - args.txt_tokens) * args.img_stride + args.k_offset + h * hd;
        } else {
            k = cache_k;
            krow = (s - args.txt_tokens - args.img_tokens) * args.cache_stride
                    + args.k_offset + h * hd;
        }
        const float score = flux2_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
        const float p = exp(score - max_score);
        sum += p;
        device const float *v;
        uint vrow;
        if (s < args.txt_tokens) {
            v = v_txt;
            vrow = s * args.txt_stride + args.v_offset + h * hd;
        } else if (s < args.txt_tokens + args.img_tokens) {
            v = v_img;
            vrow = (s - args.txt_tokens) * args.img_stride + args.v_offset + h * hd;
        } else {
            v = cache_v;
            vrow = (s - args.txt_tokens - args.img_tokens) * args.cache_stride
                    + args.v_offset + h * hd;
        }
        for (uint slot = 0u; slot < FLUX2_ATTN_SLOTS; slot++) {
            const uint d = tid + slot * FLUX2_BLOCK;
            if (d < hd) {
                acc[slot] += p * v[vrow + d];
            }
        }
    }

    const float inv = 1.0f / sum;
    const uint width = args.heads * hd;
    device float *dst;
    uint base;
    if (t < args.txt_tokens) {
        dst = out_txt;
        base = t * width + h * hd;
    } else {
        dst = out_img;
        base = (t - args.txt_tokens) * width + h * hd;
    }
    for (uint slot = 0u; slot < FLUX2_ATTN_SLOTS; slot++) {
        const uint d = tid + slot * FLUX2_BLOCK;
        if (d < hd) {
            dst[base + d] = acc[slot] * inv;
        }
    }
}

/// Causal grouped-query attention over packed rows (the text encoder's
/// prefill): query head `h` reads kv head `h / (q_heads / kv_heads)` and
/// attends to keys `0..=t`.
kernel void kernel_flux2_gqa_f32(
        constant flux2_args_gqa & args [[buffer(0)]],
        device float *out               [[buffer(1)]],
        device const float *q           [[buffer(2)]],
        device const float *k           [[buffer(3)]],
        device const float *v           [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg.x;
    if (t >= args.tokens || h >= args.q_heads) {
        return;
    }
    const uint hd = args.head_dim;
    const uint group = args.q_heads / args.kv_heads;
    const uint kvh = h / group;
    const uint qrow = (t * args.q_heads + h) * hd;
    const float scale = 1.0f / sqrt(float(hd));

    float max_score = -INFINITY;
    for (uint s = 0u; s <= t; s++) {
        const uint krow = (s * args.kv_heads + kvh) * hd;
        const float score = flux2_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
        max_score = max(max_score, score);
    }

    float acc[FLUX2_ATTN_SLOTS];
    for (uint slot = 0u; slot < FLUX2_ATTN_SLOTS; slot++) {
        acc[slot] = 0.0f;
    }
    float sum = 0.0f;
    for (uint s = 0u; s <= t; s++) {
        const uint krow = (s * args.kv_heads + kvh) * hd;
        const float score = flux2_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
        const float p = exp(score - max_score);
        sum += p;
        const uint vrow = (s * args.kv_heads + kvh) * hd;
        for (uint slot = 0u; slot < FLUX2_ATTN_SLOTS; slot++) {
            const uint d = tid + slot * FLUX2_BLOCK;
            if (d < hd) {
                acc[slot] += p * v[vrow + d];
            }
        }
    }
    const float inv = 1.0f / sum;
    for (uint slot = 0u; slot < FLUX2_ATTN_SLOTS; slot++) {
        const uint d = tid + slot * FLUX2_BLOCK;
        if (d < hd) {
            out[qrow + d] = acc[slot] * inv;
        }
    }
}

// --- rope ------------------------------------------------------------------------

/// One section's interleaved-pair rotation: pair `j` of axis `a` turns by
/// `pos_a * freq[a * freqs_per_axis + j]`, the pairs adjacent within the
/// axis (`i0 = a * axis_dim + 2j`), exactly the reference's rotary embedding
/// and the host twin's loop.
static inline void flux2_rope_section(
        device float *buf, uint row,
        device const float *positions, device const float *freqs,
        uint token, uint head_dim, uint freqs_per_axis, uint tid) {
    const uint axis_dim = head_dim / 4u;
    const uint pairs = 4u * freqs_per_axis;
    for (uint p = tid; p < pairs; p += FLUX2_BLOCK) {
        const uint a = p / freqs_per_axis;
        const uint j = p - a * freqs_per_axis;
        const float angle = positions[token * 4u + a] * freqs[a * freqs_per_axis + j];
        const float c = cos(angle);
        const float s = sin(angle);
        const uint i0 = row + a * axis_dim + j * 2u;
        const float x0 = buf[i0];
        const float x1 = buf[i0 + 1u];
        buf[i0] = x0 * c - x1 * s;
        buf[i0 + 1u] = x0 * s + x1 * c;
    }
}

/// 4-axis interleaved RoPE on Q and K in one launch, in place, on rows of
/// `row_stride` elements (packed or fused-qkv).
///
/// One threadgroup per (token, head): it rotates the q section at
/// `q_offset + h * head_dim` in the q binding and the k section at
/// `k_offset + h * head_dim` in the k binding. The transformer binds the
/// same fused buffer to both with different offsets, so the two sections
/// this group writes are disjoint from every other group's - q rotates
/// exactly once, k exactly once, and the v section, which no offset points
/// at, is never touched.
kernel void kernel_flux2_rope_4d(
        constant flux2_args_rope_4d & args [[buffer(0)]],
        device float *q                     [[buffer(1)]],
        device float *k                     [[buffer(2)]],
        device const float *positions       [[buffer(3)]],
        device const float *freqs           [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    flux2_rope_section(
            q, t * args.row_stride + args.q_offset + h * args.head_dim,
            positions, freqs, t, args.head_dim, args.freqs_per_axis, tid);
    flux2_rope_section(
            k, t * args.row_stride + args.k_offset + h * args.head_dim,
            positions, freqs, t, args.head_dim, args.freqs_per_axis, tid);
}

/// Rotate-half 1-axis RoPE on one packed (tokens, heads, head_dim) tensor,
/// with the token index as the position: pair `(i, i + head_dim/2)` turns
/// by `token * freqs[i]`.
kernel void kernel_flux2_rope_half(
        constant flux2_args_rope_half & args [[buffer(0)]],
        device float *tensor                  [[buffer(1)]],
        device const float *freqs             [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    // `half` is a reserved type name in MSL, so the split point is `fold`.
    const uint fold = args.head_dim / 2u;
    device float *row = tensor + (uint64_t)(t * args.heads + h) * args.head_dim;
    for (uint i = tid; i < fold; i += FLUX2_BLOCK) {
        const float angle = float(t) * freqs[i];
        const float c = cos(angle);
        const float s = sin(angle);
        const float x0 = row[i];
        const float x1 = row[i + fold];
        row[i] = x0 * c - x1 * s;
        row[i + fold] = x1 * c + x0 * s;
    }
}

// --- vae shapes ---------------------------------------------------------------------

/// One NCHW f32 convolution. One threadgroup per output pixel
/// (`x` = out column, `y` = out channel, `z` = batch row `n * out_h + oh`);
/// the threads stride over the `in_c * k_h * k_w` taps and reduce. Out-of-
/// field taps read as zero, the same skip the host twin makes.
kernel void kernel_flux2_conv2d_f32(
        constant flux2_args_conv2d & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const float *weight         [[buffer(3)]],
        device const float *bias           [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint ow = tgpig.x;
    const uint oc = tgpig.y;
    const uint n = tgpig.z / args.out_h;
    const uint oh = tgpig.z % args.out_h;
    const uint tid = tpitg.x;
    if (ow >= args.out_w || oc >= args.out_c || tgpig.z >= args.batch * args.out_h) {
        return;
    }
    const uint taps = args.in_c * args.k_h * args.k_w;
    float acc = 0.0f;
    for (uint tap = tid; tap < taps; tap += FLUX2_BLOCK) {
        const uint kw = tap % args.k_w;
        const uint kh = (tap / args.k_w) % args.k_h;
        const uint ic = tap / (args.k_w * args.k_h);
        const uint ih = oh * args.stride + kh;
        const uint iw = ow * args.stride + kw;
        if (ih < args.pad || iw < args.pad
                || ih - args.pad >= args.in_h || iw - args.pad >= args.in_w) {
            continue;
        }
        const uint xi = ((n * args.in_c + ic) * args.in_h + (ih - args.pad))
                * args.in_w + (iw - args.pad);
        const uint wi = ((oc * args.in_c + ic) * args.k_h + kh) * args.k_w + kw;
        acc += in[xi] * weight[wi];
    }
    acc = flux2_block_sum(acc, scratch, tid);
    if (tid == 0u) {
        out[((n * args.out_c + oc) * args.out_h + oh) * args.out_w + ow] =
                acc + (args.has_bias != 0u ? bias[oc] : 0.0f);
    }
}

/// Group normalization with per-channel affine parameters. One threadgroup
/// per (batch, group): the mean and variance run over the group's
/// `channels_per_group * spatial` elements, and element `i` of the group
/// belongs to channel `(g % groups) * channels_per_group + i / spatial` -
/// the affine buffers hold `channels` elements, not one per group element.
kernel void kernel_flux2_group_norm_f32(
        constant flux2_args_group_norm & args [[buffer(0)]],
        device float *out                      [[buffer(1)]],
        device const float *in                 [[buffer(2)]],
        device const float *weight             [[buffer(3)]],
        device const float *bias               [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint g = tgpig.x;
    const uint tid = tpitg.x;
    const uint per_group = args.channels_per_group * args.spatial;
    if (g >= args.batch * args.num_groups) {
        return;
    }
    const uint base = g * per_group;
    device const float *src = in + base;
    device float *dst = out + base;
    float part = 0.0f;
    for (uint i = tid; i < per_group; i += FLUX2_BLOCK) {
        part += src[i];
    }
    const float mean = flux2_block_sum(part, scratch, tid) / float(per_group);
    float var = 0.0f;
    for (uint i = tid; i < per_group; i += FLUX2_BLOCK) {
        const float dv = src[i] - mean;
        var += dv * dv;
    }
    const float inv = 1.0f / sqrt(flux2_block_sum(var, scratch, tid) / float(per_group) + args.eps);
    for (uint i = tid; i < per_group; i += FLUX2_BLOCK) {
        const uint ch = (g % args.num_groups) * args.channels_per_group + i / args.spatial;
        dst[i] = (src[i] - mean) * inv * weight[ch] + bias[ch];
    }
}

/// Nearest-neighbour 2x spatial upsampling: (batch, channels, h, w) to
/// (batch, channels, 2h, 2w). One threadgroup per source pixel; thread 0
/// fans it out to the four destinations.
kernel void kernel_flux2_upsample_nearest_f32(
        constant flux2_args_upsample & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *in               [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    if (tpitg.x != 0u) {
        return;
    }
    const uint j = tgpig.x;
    const uint i = tgpig.y;
    const uint nc = tgpig.z;
    if (j >= args.w || i >= args.h) {
        return;
    }
    const float v = in[((uint64_t)nc * args.h + i) * args.w + j];
    device float *dst = out + (uint64_t)nc * (4u * args.h * args.w);
    for (uint di = 0u; di < 2u; di++) {
        for (uint dj = 0u; dj < 2u; dj++) {
            dst[((2u * i + di) * 2u * args.w) + 2u * j + dj] = v;
        }
    }
}

/// Transpose a (rows, cols) f32 matrix: `out[c * rows + r] = in[r * cols + c]`.
/// The driver reuses the norm block with `eps` unused.
kernel void kernel_flux2_transpose_f32(
        constant flux2_args_norm & args  [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * FLUX2_BLOCK + tpitg.x;
    const uint total = args.rows * args.dim;
    if (i >= total) {
        return;
    }
    const uint r = i / args.dim;
    const uint c = i - r * args.dim;
    out[c * args.rows + r] = in[i];
}

/// Embedding gather: token `t` takes table row `ids[t]`, widened from BF16
/// to f32 on write. One threadgroup per (token, hidden chunk).
kernel void kernel_flux2_embed_bf16_f32(
        constant flux2_args_embed & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device const ushort *table        [[buffer(2)]],
        device const uint *ids            [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint t = tgpig.x;
    const uint d = tgpig.y * FLUX2_BLOCK + tpitg.x;
    if (t >= args.tokens || d >= args.hidden) {
        return;
    }
    const uint id = ids[t];
    out[(uint64_t)t * args.hidden + d] = flux2_bf16(table + ((uint64_t)id * args.hidden + d));
}
