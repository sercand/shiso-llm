// The diffusion DiT kernel pack: every device operation the family's driver
// (`crates/shiso-models/src/diffusion/ops.rs`) launches, under module `diffusion`.
//
// Each kernel computes exactly what the driver's host twin computes - same
// index math, same section offsets, same accumulation order where the
// numbers depend on it - because the host twin is both the fallback for a
// device without this set and the arithmetic the tests grade against
// (`tests/dit_kernels.rs`). The fused-row contracts (sdpa's per-operand
// q/k/v offsets, rope_4d's q/k section offsets on an aliased binding, the
// gate-first swiglu) are the ones the driver's call sites marshal; the
// oracle tests in `shiso-models/src/flux2/oracle_tests.rs` pin the same
// shapes on the host side.
//
// `diffusion_common.h` is concatenated ahead of this file.

// --- elementwise -------------------------------------------------------------

/// `out[i] = silu(in[i])`.
kernel void kernel_dit_silu_f32(
        constant dit_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = dit_silu(in[i]);
}

/// `out[i] = tanh(in[i])`: the shared modulation's residual gates. The
/// precise form, because the library compiles under fast math and a gate
/// multiplies every block's whole branch.
kernel void kernel_dit_tanh_f32(
        constant dit_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = precise::tanh(in[i]);
}

/// `out[i] = gelu(in[i])` in the tanh form:
/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
kernel void kernel_dit_gelu_tanh_f32(
        constant dit_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    const float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
    out[i] = 0.5f * x * (1.0f + precise::tanh(inner));
}

/// SiLU gate over a fused row: `out[r,i] = silu(in[r,i]) * in[r,inner+i]`.
///
/// The gate is the FIRST half (the reference's gating chunks the
/// row in two and silus the first chunk). The input is one strided buffer
/// whose view may start mid-row (the single block's halves begin at column
/// 3h of the fused row); the output is packed `(rows, inner)`. One thread
/// per output element: grid `(ceil(inner / 64), rows)`.
kernel void kernel_dit_swiglu_f32(
        constant dit_args_swiglu & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint r = tgpig.y;
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (r >= args.rows || i >= args.inner) {
        return;
    }
    device const float *row = in + (uint64_t)r * args.stride;
    const float g = row[i];
    out[(uint64_t)r * args.inner + i] = dit_silu(g) * row[args.inner + i];
}

/// `out[i] = silu(a[i]) * b[i]` over two packed buffers: the text encoder's
/// separate gate/up projections, `a` the gate - the same first-half gating
/// the fused-row swiglu applies.
kernel void kernel_dit_swiglu2_f32(
        constant dit_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *a            [[buffer(2)]],
        device const float *b            [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const float g = a[i];
    out[i] = dit_silu(g) * b[i];
}

/// `out[i] = x[i] * (1 + scale[i % period]) + shift[i % period]`: the AdaLN
/// modulation with the per-dimension vectors broadcast over the rows.
kernel void kernel_dit_modulate_f32(
        constant dit_args_modulate & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *x                [[buffer(2)]],
        device const float *scale            [[buffer(3)]],
        device const float *shift            [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const uint d = i % args.period;
    out[i] = x[i] * (1.0f + scale[d]) + shift[d];
}

/// `out = a + gate * b`, `gate` broadcast over `period`: the gated residual.
kernel void kernel_dit_add_gated_f32(
        constant dit_args_add_gated & args [[buffer(0)]],
        device float *out                     [[buffer(1)]],
        device const float *a                 [[buffer(2)]],
        device const float *gate              [[buffer(3)]],
        device const float *b                 [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = a[i] + gate[i % args.period] * b[i];
}

/// `out[i] = a[i] + b[i]`.
kernel void kernel_dit_add_f32(
        constant dit_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *a            [[buffer(2)]],
        device const float *b            [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = a[i] + b[i];
}

/// `out[i] = x[i] + dt * v[i]`: the Euler latent update.
kernel void kernel_dit_axpy_f32(
        constant dit_args_axpy & args [[buffer(0)]],
        device float *out               [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        device const float *v           [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = x[i] + args.dt * v[i];
}

// --- gemm ---------------------------------------------------------------------

/// The stored-width decoders the tile body calls: the weight stays in the
/// width the checkpoint stored it and is widened on read (`ushort`
/// brain-float, `uchar` E4M3, or `float`), never through a copy.
static inline float dit_weight_value(float w) {
    return w;
}

static inline float dit_weight_value(ushort w) {
    return as_type<float>((uint)w << 16);
}

static inline float dit_weight_value(uchar w) {
    return dit_e4m3(uint(w));
}

static inline float dit_weight_value(char w) {
    return float(w);
}

/// The epilogue multiplier of a linear: one value for the whole tensor (the
/// FP8 `weight_scale`, or 1.0), or one per weight row (the INT8 export's
/// `weight_scale [out, 1]`), indexed by the output column the row produces.
struct dit_scale_tensor {
    float s;
    float at(uint) const { return s; }
};

struct dit_scale_rows {
    device const float *s;
    float at(uint col) const { return s[col]; }
};

/// The staged 16x16 tile body every weight width shares.
///
/// `k` is walked strictly in increasing order - the tile loop advances by
/// DIT_GEMM_TILE and the inner loop walks the tile front to back - so the
/// sum is the same association the driver's host twin makes with a plain
/// `for k in 0..K` loop. That is deliberate: a blocking change that reorders
/// `k` is a change to the numbers, not just to the speed.
///
/// The scale multiplies the accumulator in the epilogue, never the weight
/// in memory. The bias is per output column, applied only when
/// `args.has_bias` says the binding is a real vector; the driver clears
/// the flag and binds its zero stand-in when the linear carries none.
template <typename W, typename SC>
static inline void dit_gemm_body(
        constant dit_args_gemm & args,
        device float *out,
        device const float *in,
        device const W *weight,
        const SC scale,
        device const float *bias,
        threadgroup float (*a_tile)[DIT_GEMM_TILE],
        threadgroup float (*b_tile)[DIT_GEMM_TILE],
        uint3 tgpig,
        ushort3 tpitg) {
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const uint row = tgpig.y * DIT_GEMM_TILE + ty;
    const uint col = tgpig.x * DIT_GEMM_TILE + tx;
    // The weight row this thread stages, which is not the one it
    // accumulates: the tile is filled row-per-thread-y so `k` runs along x
    // and the load stays contiguous, then read back column-wise.
    const uint stage_n = tgpig.x * DIT_GEMM_TILE + ty;
    const bool has_bias = args.has_bias != 0u;

    float acc = 0.0f;
    for (uint k0 = 0u; k0 < args.k; k0 += DIT_GEMM_TILE) {
        const uint k = k0 + tx;
        const bool have_k = k < args.k;
        a_tile[ty][tx] = (row < args.m && have_k)
                ? in[(uint64_t)row * args.k + k]
                : 0.0f;
        const uint64_t w_idx = (uint64_t)stage_n * args.k + k;
        b_tile[ty][tx] = (stage_n < args.n && have_k)
                ? dit_weight_value(weight[w_idx])
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < DIT_GEMM_TILE; kk++) {
            acc += a_tile[ty][kk] * b_tile[tx][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < args.m && col < args.n) {
        out[(uint64_t)row * args.n + col] =
                acc * scale.at(col) + (has_bias ? bias[col] : 0.0f);
    }
}

/// `out[m,n] = in[m,k] . weight[n,k] * scale + bias[n]`, weight F32.
kernel void kernel_dit_gemm_f32(
        constant dit_args_gemm & args    [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const float *weight         [[buffer(3)]],
        constant float &weight_scale       [[buffer(4)]],
        device const float *bias           [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    // Static threadgroup arrays, not `[[threadgroup(0)]]` parameters: the
    // driver launches these with no dynamic threadgroup request.
    threadgroup float a_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    threadgroup float b_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    dit_gemm_body(args, out, in, weight, dit_scale_tensor{weight_scale}, bias,
                           a_tile, b_tile, tgpig, tpitg);
}

/// The same product over an FP8 E4M3 weight: the bytes are read directly and
/// dequantized in registers during the dot, and the accumulated sum is
/// multiplied by the tensor's `weight_scale` in the epilogue.
kernel void kernel_dit_gemm_fp8(
        constant dit_args_gemm & args    [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const uchar *weight_fp8     [[buffer(3)]],
        constant float &weight_scale       [[buffer(4)]],
        device const float *bias           [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    threadgroup float b_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    dit_gemm_body(args, out, in, weight_fp8, dit_scale_tensor{weight_scale}, bias,
                           a_tile, b_tile, tgpig, tpitg);
}

/// The same product over a per-row INT8 weight: the signed bytes are widened
/// in registers and `row_scale[n]` - one value per weight row, which is one
/// per output column - multiplies the accumulator in the epilogue.
kernel void kernel_dit_gemm_i8(
        constant dit_args_gemm & args    [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const char *weight_i8       [[buffer(3)]],
        device const float *row_scale      [[buffer(4)]],
        device const float *bias           [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    threadgroup float b_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    dit_gemm_body(args, out, in, weight_i8, dit_scale_rows{row_scale}, bias,
                  a_tile, b_tile, tgpig, tpitg);
}

// The brain-float GEMM is a template with two named instantiations because
// the driver decides kernel-vs-host by probing `module_entries("diffusion")` for
// a name containing `kernel_dit_gemm_bf16`, and Metal's table lists only
// CUDA-style mangled names (`device.rs::module_entries`). The plain host
// name is the launch entry; the `_Z...` instantiation is the name nvcc
// would give the same kernel, so the probe sees the set and the Metal build
// runs it instead of the host fallback. The tag parameter is otherwise
// unused: both instantiations are the same code.
#define DIT_GEMM_BF16_PARAMS \
        constant dit_args_gemm & args    [[buffer(0)]], \
        device float *out                  [[buffer(1)]], \
        device const float *in             [[buffer(2)]], \
        device const ushort *weight        [[buffer(3)]], \
        constant float &weight_scale       [[buffer(4)]], \
        device const float *bias           [[buffer(5)]], \
        uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort3 tpitg[[thread_position_in_threadgroup]]

template <bool ProbeAlias>
kernel void kernel_dit_gemm_bf16_kernel(DIT_GEMM_BF16_PARAMS) {
    threadgroup float a_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    threadgroup float b_tile[DIT_GEMM_TILE][DIT_GEMM_TILE];
    dit_gemm_body(args, out, in, weight, dit_scale_tensor{weight_scale}, bias,
                            a_tile, b_tile, tgpig, tpitg);
}

template [[host_name("kernel_dit_gemm_bf16")]]
kernel void kernel_dit_gemm_bf16_kernel<false>(DIT_GEMM_BF16_PARAMS);

template [[host_name("_Z22kernel_dit_gemm_bf16v")]]
kernel void kernel_dit_gemm_bf16_kernel<true>(DIT_GEMM_BF16_PARAMS);

// --- norms ----------------------------------------------------------------------

/// RMS normalization of each row: `out = x * rsqrt(mean(x^2) + eps) * w`.
kernel void kernel_dit_rmsnorm_f32(
        constant dit_args_norm & args  [[buffer(0)]],
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
    for (uint d = tid; d < args.dim; d += DIT_BLOCK) {
        const float v = row[d];
        part += v * v;
    }
    const float ms = dit_block_sum(part, scratch, tid) / float(args.dim);
    const float inv = 1.0f / sqrt(ms + args.eps);
    for (uint d = tid; d < args.dim; d += DIT_BLOCK) {
        dst[d] = row[d] * inv * weight[d];
    }
}

/// Row softmax of `scale * x`, in place; one 64-thread group per row.
///
/// The decoder's single-head attention runs as two GEMMs around this: at a
/// 1024x1024 image that is 16384 x 16384 scores over 512-wide rows, which the
/// per-row attention kernel spent 3.4 s on (on an M1 Max) and the tiled
/// GEMMs take a fraction of. `scale` must be positive, so the max of the
/// scaled row is the scaled max.
kernel void kernel_dit_softmax_rows_f32(
        constant dit_args_softmax & args [[buffer(0)]],
        device float *x                     [[buffer(1)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint r = tgpig.x;
    const uint tid = tpitg.x;
    if (r >= args.rows) {
        return;
    }
    device float *row = x + (uint64_t) r * args.cols;
    float m = -INFINITY;
    for (uint d = tid; d < args.cols; d += DIT_BLOCK) {
        m = max(m, row[d]);
    }
    m = dit_block_max(m, scratch, tid);
    float part = 0.0f;
    for (uint d = tid; d < args.cols; d += DIT_BLOCK) {
        const float e = exp((row[d] - m) * args.scale);
        row[d] = e;
        part += e;
    }
    const float inv = 1.0f / dit_block_sum(part, scratch, tid);
    for (uint d = tid; d < args.cols; d += DIT_BLOCK) {
        row[d] *= inv;
    }
}

/// Weightless LayerNorm of each row: `out = (x - mean) / sqrt(var + eps)`.
///
/// Mean-subtracting, unlike rmsnorm, and with no affine parameters: the
/// transformer's block norms are `LayerNorm(affine=False)`, whose shift and
/// scale come from the AdaLN modulation instead.
kernel void kernel_dit_layer_norm_f32(
        constant dit_args_norm & args  [[buffer(0)]],
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
    for (uint d = tid; d < args.dim; d += DIT_BLOCK) {
        part += row[d];
    }
    const float mean = dit_block_sum(part, scratch, tid) / float(args.dim);
    float var = 0.0f;
    for (uint d = tid; d < args.dim; d += DIT_BLOCK) {
        const float dv = row[d] - mean;
        var += dv * dv;
    }
    const float inv = 1.0f / sqrt(dit_block_sum(var, scratch, tid) / float(args.dim) + args.eps);
    for (uint d = tid; d < args.dim; d += DIT_BLOCK) {
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
kernel void kernel_dit_qk_norm_f32(
        constant dit_args_qk_norm & args [[buffer(0)]],
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
        for (uint d = tid; d < args.head_dim; d += DIT_BLOCK) {
            const float v = row[d];
            part += v * v;
        }
        const float ms = dit_block_sum(part, scratch, tid) / float(args.head_dim);
        const float inv = 1.0f / sqrt(ms + args.eps);
        for (uint d = tid; d < args.head_dim; d += DIT_BLOCK) {
            row[d] *= inv * scale[d];
        }
        // The second section's reduction reuses the scratch; hold until
        // every thread has consumed the first broadcast.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// Per-head RMS normalization with a per-dimension scale, then interleaved
/// RoPE from a per-token angle table, over separate packed q and k buffers,
/// in place: the QK-norm and rotation of a block whose projections are
/// three separate linears.
///
/// One threadgroup per (token, head); each thread owns whole pairs
/// `(x[2i], x[2i+1])`, so the rotation reads only values its own thread
/// normalized. The pairs are adjacent (interleaved), not the two halves of
/// the head. `cos`/`sin` are `(tokens, head_dim / 2)`.
kernel void kernel_dit_qk_norm_rope_f32(
        constant dit_args_qk_norm_rope & args [[buffer(0)]],
        device float *q                          [[buffer(1)]],
        device float *k                          [[buffer(2)]],
        device const float *q_scale              [[buffer(3)]],
        device const float *k_scale              [[buffer(4)]],
        device const float *cos_t                [[buffer(5)]],
        device const float *sin_t                [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint t = tgpig.x;
    const uint h = tgpig.y;
    const uint tid = tpitg.x;
    if (t >= args.tokens || h >= args.heads) {
        return;
    }
    const uint half_d = args.head_dim / 2u;
    device const float *ct = cos_t + (uint64_t) t * half_d;
    device const float *st = sin_t + (uint64_t) t * half_d;
    for (uint section = 0u; section < 2u; section++) {
        device float *row = section == 0u
                ? q + (uint64_t) t * args.q_stride + h * args.head_dim
                : k + (uint64_t) t * args.k_stride + h * args.head_dim;
        device const float *scale = section == 0u ? q_scale : k_scale;
        float part = 0.0f;
        for (uint d = tid; d < args.head_dim; d += DIT_BLOCK) {
            const float v = row[d];
            part += v * v;
        }
        const float ms = dit_block_sum(part, scratch, tid) / float(args.head_dim);
        const float inv = 1.0f / sqrt(ms + args.eps);
        for (uint p = tid; p < half_d; p += DIT_BLOCK) {
            const float x0 = row[2u * p] * inv * scale[2u * p];
            const float x1 = row[2u * p + 1u] * inv * scale[2u * p + 1u];
            const float c = ct[p];
            const float s = st[p];
            row[2u * p] = x0 * c - x1 * s;
            row[2u * p + 1u] = x0 * s + x1 * c;
        }
        // The k section's reduction reuses the scratch; hold until every
        // thread has consumed the q broadcast.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

/// Row-wise concatenation: `out[r] = [a[r], b[r]]`, both sides packed (the
/// single block's `linear2` input, the reference's concatenation of attention
/// and MLP output).
kernel void kernel_dit_concat_rows_f32(
        constant dit_args_concat & args [[buffer(0)]],
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
    for (uint i = tid; i < width; i += DIT_BLOCK) {
        dst[i] = (i < args.cols_a) ? ra[i] : rb[i - args.cols_a];
    }
}

// --- attention --------------------------------------------------------------------

/// One joint-attention dot: head `h` of key row `s` against the query head,
/// reduced across the block. `krow` is the operand-relative start of the
/// key head; the per-operand offsets are already folded into it by the
/// caller's index math.
static inline float dit_head_dot(
        device const float *q, uint qrow,
        device const float *k, uint krow,
        uint head_dim, uint tid, threadgroup float *scratch) {
    float part = 0.0f;
    for (uint d = tid; d < head_dim; d += DIT_BLOCK) {
        part += q[qrow + d] * k[krow + d];
    }
    return dit_block_sum(part, scratch, tid);
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
kernel void kernel_dit_sdpa_f32(
        constant dit_args_sdpa & args [[buffer(0)]],
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
        const float score = dit_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
        max_score = max(max_score, score);
    }

    // Pass 2: the exponentials, their sum, and the value mix. Every thread
    // recomputes the same scores (the dot is block-broadcast), so `sum`
    // needs no second reduction; each thread owns its slice of the head.
    float acc[DIT_ATTN_SLOTS];
    for (uint slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
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
        const float score = dit_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
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
        for (uint slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
            const uint d = tid + slot * DIT_BLOCK;
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
    for (uint slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
        const uint d = tid + slot * DIT_BLOCK;
        if (d < hd) {
            dst[base + d] = acc[slot] * inv;
        }
    }
}

/// Causal grouped-query attention over packed rows (the text encoder's
/// prefill): query head `h` reads kv head `h / (q_heads / kv_heads)` and
/// attends to keys `0..=t`.
kernel void kernel_dit_gqa_f32(
        constant dit_args_gqa & args [[buffer(0)]],
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
        const float score = dit_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
        max_score = max(max_score, score);
    }

    float acc[DIT_ATTN_SLOTS];
    for (uint slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
        acc[slot] = 0.0f;
    }
    float sum = 0.0f;
    for (uint s = 0u; s <= t; s++) {
        const uint krow = (s * args.kv_heads + kvh) * hd;
        const float score = dit_head_dot(q, qrow, k, krow, hd, tid, scratch) * scale;
        const float p = exp(score - max_score);
        sum += p;
        const uint vrow = (s * args.kv_heads + kvh) * hd;
        for (uint slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
            const uint d = tid + slot * DIT_BLOCK;
            if (d < hd) {
                acc[slot] += p * v[vrow + d];
            }
        }
    }
    const float inv = 1.0f / sum;
    for (uint slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
        const uint d = tid + slot * DIT_BLOCK;
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
static inline void dit_rope_section(
        device float *buf, uint row,
        device const float *positions, device const float *freqs,
        uint token, uint head_dim, uint freqs_per_axis, uint tid) {
    const uint axis_dim = head_dim / 4u;
    const uint pairs = 4u * freqs_per_axis;
    for (uint p = tid; p < pairs; p += DIT_BLOCK) {
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
kernel void kernel_dit_rope_4d(
        constant dit_args_rope_4d & args [[buffer(0)]],
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
    dit_rope_section(
            q, t * args.row_stride + args.q_offset + h * args.head_dim,
            positions, freqs, t, args.head_dim, args.freqs_per_axis, tid);
    dit_rope_section(
            k, t * args.row_stride + args.k_offset + h * args.head_dim,
            positions, freqs, t, args.head_dim, args.freqs_per_axis, tid);
}

/// Rotate-half 1-axis RoPE on one packed (tokens, heads, head_dim) tensor,
/// with the token index as the position: pair `(i, i + head_dim/2)` turns
/// by `token * freqs[i]`.
kernel void kernel_dit_rope_half(
        constant dit_args_rope_half & args [[buffer(0)]],
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
    for (uint i = tid; i < fold; i += DIT_BLOCK) {
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
kernel void kernel_dit_conv2d_f32(
        constant dit_args_conv2d & args [[buffer(0)]],
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
    for (uint tap = tid; tap < taps; tap += DIT_BLOCK) {
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
    acc = dit_block_sum(acc, scratch, tid);
    if (tid == 0u) {
        out[((n * args.out_c + oc) * args.out_h + oh) * args.out_w + ow] =
                acc + (args.has_bias != 0u ? bias[oc] : 0.0f);
    }
}

/// Elements `[start, end)` of a group's slice `c` of `chunks`.
static inline uint2 dit_gn_slice(uint per_group, uint chunks, uint c) {
    const uint len = (per_group + chunks - 1u) / chunks;
    const uint start = min(c * len, per_group);
    return uint2(start, min(start + len, per_group));
}

/// Group normalization, pass 1 of 2: the mean and the centered sum of
/// squares (`M2`) of one slice of one (batch, group), written as
/// `partials[(g * chunks + c) * 2 + {0, 1}]`. Grid `(batch * groups,
/// chunks)` x 64 threads.
///
/// One threadgroup per whole group kept 32 threadgroups streaming a
/// 1024x1024 decoder field: 70 ms per norm on an M1 Max, ~1.5 s of a
/// decode. Slicing the group puts the whole GPU on it; the slice statistics
/// are combined in pass 2 by Chan's formula, in slice order, so the result
/// is the same on every run.
kernel void kernel_dit_group_norm_stats_f32(
        constant dit_args_group_norm & args [[buffer(0)]],
        device float *partials                 [[buffer(1)]],
        device const float *in                 [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float scratch[3];
    const uint g = tgpig.x;
    const uint c = tgpig.y;
    const uint tid = tpitg.x;
    const uint per_group = args.channels_per_group * args.spatial;
    if (g >= args.batch * args.num_groups || c >= args.chunks) {
        return;
    }
    const uint2 span = dit_gn_slice(per_group, args.chunks, c);
    device const float *src = in + (uint64_t) g * per_group;
    const float n = float(span.y - span.x);
    float part = 0.0f;
    for (uint i = span.x + tid; i < span.y; i += DIT_BLOCK) {
        part += src[i];
    }
    const float mean = n > 0.0f ? dit_block_sum(part, scratch, tid) / n : 0.0f;
    float m2 = 0.0f;
    for (uint i = span.x + tid; i < span.y; i += DIT_BLOCK) {
        const float dv = src[i] - mean;
        m2 += dv * dv;
    }
    m2 = dit_block_sum(m2, scratch, tid);
    if (tid == 0u) {
        partials[((uint64_t) g * args.chunks + c) * 2u] = mean;
        partials[((uint64_t) g * args.chunks + c) * 2u + 1u] = m2;
    }
}

/// Group normalization, pass 2 of 2: combines the group's slice statistics
/// and writes `(x - mean) / sqrt(var + eps) * weight[ch] + bias[ch]` over
/// one slice. Element `i` of a group belongs to channel
/// `(g % groups) * channels_per_group + i / spatial` - the affine buffers
/// hold `channels` elements, not one per group element. Same grid as pass 1.
kernel void kernel_dit_group_norm_apply_f32(
        constant dit_args_group_norm & args [[buffer(0)]],
        device float *out                      [[buffer(1)]],
        device const float *in                 [[buffer(2)]],
        device const float *weight             [[buffer(3)]],
        device const float *bias               [[buffer(4)]],
        device const float *partials           [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint g = tgpig.x;
    const uint c = tgpig.y;
    const uint tid = tpitg.x;
    const uint per_group = args.channels_per_group * args.spatial;
    if (g >= args.batch * args.num_groups || c >= args.chunks) {
        return;
    }
    // Every thread folds the same slices in the same order (Chan et al.'s
    // pairwise update), so the whole group agrees on one mean and variance.
    float count = 0.0f;
    float mean = 0.0f;
    float m2 = 0.0f;
    for (uint j = 0; j < args.chunks; ++j) {
        const uint2 span = dit_gn_slice(per_group, args.chunks, j);
        const float nb = float(span.y - span.x);
        if (nb == 0.0f) {
            continue;
        }
        const float mb = partials[((uint64_t) g * args.chunks + j) * 2u];
        const float m2b = partials[((uint64_t) g * args.chunks + j) * 2u + 1u];
        const float total = count + nb;
        const float delta = mb - mean;
        mean += delta * (nb / total);
        m2 += m2b + delta * delta * (count * nb / total);
        count = total;
    }
    const float inv = 1.0f / sqrt(m2 / float(per_group) + args.eps);
    const uint2 span = dit_gn_slice(per_group, args.chunks, c);
    const uint64_t base = (uint64_t) g * per_group;
    const uint ch0 = (g % args.num_groups) * args.channels_per_group;
    for (uint i = span.x + tid; i < span.y; i += DIT_BLOCK) {
        const uint ch = ch0 + i / args.spatial;
        out[base + i] = (in[base + i] - mean) * inv * weight[ch] + bias[ch];
    }
}

/// Nearest-neighbour 2x spatial upsampling: (batch, channels, h, w) to
/// (batch, channels, 2h, 2w). One threadgroup per source pixel; thread 0
/// fans it out to the four destinations.
kernel void kernel_dit_upsample_nearest_f32(
        constant dit_args_upsample & args [[buffer(0)]],
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
kernel void kernel_dit_transpose_f32(
        constant dit_args_norm & args  [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * DIT_BLOCK + tpitg.x;
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
kernel void kernel_dit_embed_bf16_f32(
        constant dit_args_embed & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device const ushort *table        [[buffer(2)]],
        device const uint *ids            [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint t = tgpig.x;
    const uint d = tgpig.y * DIT_BLOCK + tpitg.x;
    if (t >= args.tokens || d >= args.hidden) {
        return;
    }
    const uint id = ids[t];
    out[(uint64_t)t * args.hidden + d] = dit_bf16(table + ((uint64_t)id * args.hidden + d));
}

// --- simdgroup-matrix gemm --------------------------------------------------------
//
// The 16x16 kernels above compute one output per thread with scalar
// multiply-adds; on the transformer's 1024-row products they run at a few
// hundred GFLOP/s, which made the denoising step take ~11 s of GEMM on an
// M1 Max. This tile is the audio set's 64x64 layout: four simdgroups in a
// 2x2 grid, each owning a 32x32 quarter as a 4x4 grid of 8x8 accumulators,
// k staged 16 at a time with the activations as `[row][k]` and the weights
// transposed to `[k][col]`, each staged row padded by four floats against
// bank conflicts. The next slice's global reads are issued after the current
// slice's multiplies so they overlap the math.
//
// The operands are reached through small reader structs, so the same body
// serves a dense linear (rows of `in`) and the VAE's convolutions (an im2col
// gather of the input field, never materialized). Every output element is
// the multiply-accumulates of its row and column over 8-k steps in k order
// from zero; the association differs from the 16x16 kernels' scalar fold,
// so the two agree to rounding, not bit for bit.

/// Dense activation rows: element (r, k) of `x[m][k]`. `packed_float4` so a
/// view that starts on any 4-byte boundary is read correctly.
struct dit_rows_f32 {
    device const float *x;
    uint k;
    float4 load4(uint r, uint kk) const {
        return float4(*((device const packed_float4 *) (x + (uint64_t) r * k + kk)));
    }
};

/// Weight rows in their stored width, widened on read. The packed vector
/// types carry the element's own alignment, which is all a mapped
/// checkpoint's tensor offsets promise.
struct dit_wrows_f32 {
    device const float *w;
    uint k;
    float4 load4(uint n, uint kk) const {
        return float4(*((device const packed_float4 *) (w + (uint64_t) n * k + kk)));
    }
};

struct dit_wrows_bf16 {
    device const ushort *w;
    uint k;
    float4 load4(uint n, uint kk) const {
        const packed_ushort4 b = *((device const packed_ushort4 *) (w + (uint64_t) n * k + kk));
        return float4(as_type<float>((uint) b.x << 16), as_type<float>((uint) b.y << 16),
                      as_type<float>((uint) b.z << 16), as_type<float>((uint) b.w << 16));
    }
};

/// Signed INT8 codes, widened exactly (every code is an integer f32 holds);
/// the per-row scale rides the epilogue.
struct dit_wrows_i8 {
    device const char *w;
    uint k;
    float4 load4(uint n, uint kk) const {
        const packed_char4 b = *((device const packed_char4 *) (w + (uint64_t) n * k + kk));
        return float4(b);
    }
};

struct dit_wrows_fp8 {
    device const uchar *w;
    uint k;
    float4 load4(uint n, uint kk) const {
        const packed_uchar4 b = *((device const packed_uchar4 *) (w + (uint64_t) n * k + kk));
        return float4(dit_e4m3_bits(b.x), dit_e4m3_bits(b.y),
                      dit_e4m3_bits(b.z), dit_e4m3_bits(b.w));
    }
};

/// GGUF k-quant weight rows: a row of `k` weights is `k / 256` blocks.
///
/// Q4_K (144 bytes): `half d, dmin; uchar scales[12]; uchar qs[128]`. The
/// 256 weights are four runs of 64, each the low then the high nibbles of
/// 32 bytes of `qs`; each run of 32 has a 6-bit scale and min packed in
/// `scales`, and the weight is `d * sc * q - dmin * m`. Four weights from
/// a 4-aligned `kk` share one run, so one load decodes one scale/min pair
/// and four adjacent nibbles.
struct dit_wrows_q4k {
    device const uchar *w;
    uint k;
    float4 load4(uint n, uint kk) const {
        device const uchar *b = w + (uint64_t) n * (k / 256u) * 144u + (kk / 256u) * 144u;
        const uint i = kk % 256u;
        const uint chunk = i / 64u;
        const uint within = i % 64u;
        const uint sub = 2u * chunk + (within >= 32u ? 1u : 0u);
        device const uchar *sc = b + 4;
        uint scale;
        uint min;
        if (sub < 4u) {
            scale = sc[sub] & 63u;
            min = sc[sub + 4u] & 63u;
        } else {
            scale = (sc[sub + 4u] & 0xFu) | ((sc[sub - 4u] >> 6u) << 4u);
            min = (sc[sub + 4u] >> 4u) | ((sc[sub] >> 6u) << 4u);
        }
        const float d = float(*((device const half *) b)) * float(scale);
        const float m = float(*((device const half *) (b + 2))) * float(min);
        const uchar4 q = uchar4(*((device const packed_uchar4 *) (b + 16 + 32u * chunk + within % 32u)));
        const uint4 code = within >= 32u ? uint4(q) >> 4u : uint4(q) & 0xFu;
        return d * float4(code) - m;
    }
};

/// Q6_K (210 bytes): `uchar ql[128]; uchar qh[64]; char scales[16]; half
/// d`. Each half of 128 weights is four runs of 32: run `r` takes the low
/// (runs 0, 1) or high (runs 2, 3) nibbles of `ql[32 * (r & 1) ..]` and
/// bits `2r, 2r + 1` of `qh`, minus 32, with one int8 scale per 16
/// weights; the weight is `d * scale * q`.
struct dit_wrows_q6k {
    device const uchar *w;
    uint k;
    float4 load4(uint n, uint kk) const {
        device const uchar *b = w + (uint64_t) n * (k / 256u) * 210u + (kk / 256u) * 210u;
        const uint i = kk % 256u;
        const uint half_ = i / 128u;
        const uint r = (i % 128u) / 32u;
        const uint l = i % 32u;
        const uchar4 ql = uchar4(*((device const packed_uchar4 *)
                (b + 64u * half_ + 32u * (r & 1u) + l)));
        const uchar4 qh = uchar4(*((device const packed_uchar4 *) (b + 128u + 32u * half_ + l)));
        const uint4 lo = r >= 2u ? uint4(ql) >> 4u : uint4(ql) & 0xFu;
        const uint4 hi = (uint4(qh) >> (2u * r)) & 3u;
        const int4 q = int4(lo | (hi << 4u)) - 32;
        const float scale = float(((device const char *) (b + 192u))[8u * half_ + l / 16u + 2u * r]);
        const float d = float(*((device const half *) (b + 208u)));
        return d * scale * float4(q);
    }
};

/// A linear's epilogue: `out[r][c] = acc * scale(c) + bias[c]`, the bias read
/// only when the driver's presence flag says the binding is real.
template <typename SC>
struct dit_store_rows {
    device float *out;
    uint n;
    SC scale;
    device const float *bias;
    bool has_bias;
    void store(uint r, uint c, float v) const {
        out[(uint64_t) r * n + c] = v * scale.at(c) + (has_bias ? bias[c] : 0.0f);
    }
};

/// Whether a left-operand reader gathers along the pixels instead of along
/// k (`load4m(r, kk)`: four consecutive rows of one column), and whether an
/// epilogue wants the fragment drained row-fastest. Both are for the
/// channel-first convolutions: four adjacent output pixels of one tap are
/// adjacent input values, where four taps of one pixel are four scattered
/// ones, and adjacent pixels of one output channel are adjacent stores.
template <typename AR> struct dit_reads_along_m { enum { value = 0 }; };
template <typename EP> struct dit_stores_rows_fast { enum { value = 0 }; };
/// The pixel-wise gather, for readers that declare one; this overload is
/// the stand-in the tile compiles for every other reader and never calls.
template <typename AR>
static inline float4 dit_load4m(const AR a, uint r, uint kk, uint m) {
    return float4(0.0f);
}

/// Threadgroup floats the tile stages through: `[64][16+4]` activations and
/// `[16][64+4]` weights.
#define DIT_G64_SMEM (DIT_G64_TILE * (DIT_G64_K + 4u) + DIT_G64_K * (DIT_G64_TILE + 4u))

template <typename AR, typename WR, typename EP>
static inline void dit_gemm64_tile(
        uint M, uint N, uint K,
        const AR a, const WR w, const EP ep,
        threadgroup float *smem,
        uint3 tgpig, ushort tid, ushort lane, ushort sg) {
    const uint BM = DIT_G64_TILE;
    const uint BN = DIT_G64_TILE;
    const uint BK = DIT_G64_K;
    const uint NT = DIT_G64_THREADS;
    const uint LDA = BK + 4u;
    const uint LDB = BN + 4u;
    threadgroup float *sa = smem;            // [BM rows][BK k]
    threadgroup float *sb = smem + BM * LDA; // [BK k][BN cols]
    const uint n0 = tgpig.x * BN;
    const uint m0 = tgpig.y * BM;
    const uint r0 = (sg >> 1u) * 32u;
    const uint c0 = (sg & 1u) * 32u;
    // Two float4 slots per thread per operand: slot e is row e / 4, k (e % 4) * 4.
    const uint SLOTS = BM * BK / 4u / NT;
    float4 ra[SLOTS];
    float4 rb[SLOTS];
    const bool along_m = dit_reads_along_m<AR>::value != 0;
    auto fetch = [&](uint k0) {
        DIT_UNROLL for (uint q = 0; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            if (along_m) {
                // Slot e is column e / (BM / 4), rows (e % (BM / 4)) * 4 + 0..3.
                const uint ka = e / (BM / 4u);
                const uint ma = (e - ka * (BM / 4u)) * 4u;
                ra[q] = dit_load4m(a, m0 + ma, k0 + ka, M);
            } else {
                ra[q] = m0 + rr < M ? a.load4(m0 + rr, k0 + kk) : float4(0.0f);
            }
            rb[q] = n0 + rr < N ? w.load4(n0 + rr, k0 + kk) : float4(0.0f);
        }
    };

    simdgroup_float8x8 acc[4][4];
    DIT_UNROLL for (uint i = 0; i < 4u; ++i) {
        DIT_UNROLL for (uint j = 0; j < 4u; ++j) {
            acc[i][j] = simdgroup_float8x8(0.0f);
        }
    }
    fetch(0);
    for (uint k0 = 0; k0 < K; k0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        DIT_UNROLL for (uint q = 0; q < SLOTS; ++q) {
            const uint e = tid + q * NT;
            const uint rr = e / (BK / 4u);
            const uint kk = (e - rr * (BK / 4u)) * 4u;
            if (along_m) {
                const uint ka = e / (BM / 4u);
                const uint ma = (e - ka * (BM / 4u)) * 4u;
                sa[(ma + 0u) * LDA + ka] = ra[q].x;
                sa[(ma + 1u) * LDA + ka] = ra[q].y;
                sa[(ma + 2u) * LDA + ka] = ra[q].z;
                sa[(ma + 3u) * LDA + ka] = ra[q].w;
            } else {
                *((threadgroup float4 *) (sa + rr * LDA + kk)) = ra[q];
            }
            sb[(kk + 0u) * LDB + rr] = rb[q].x;
            sb[(kk + 1u) * LDB + rr] = rb[q].y;
            sb[(kk + 2u) * LDB + rr] = rb[q].z;
            sb[(kk + 3u) * LDB + rr] = rb[q].w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        DIT_UNROLL for (uint kk = 0; kk < BK; kk += 8u) {
            simdgroup_float8x8 af[4];
            simdgroup_float8x8 bf[4];
            DIT_UNROLL for (uint i = 0; i < 4u; ++i) {
                simdgroup_load(af[i], sa + (r0 + 8u * i) * LDA + kk, LDA);
            }
            DIT_UNROLL for (uint j = 0; j < 4u; ++j) {
                simdgroup_load(bf[j], sb + kk * LDB + c0 + 8u * j, LDB);
            }
            DIT_UNROLL for (uint i = 0; i < 4u; ++i) {
                DIT_UNROLL for (uint j = 0; j < 4u; ++j) {
                    simdgroup_multiply_accumulate(acc[i][j], af[i], bf[j], acc[i][j]);
                }
            }
        }
        if (k0 + BK < K) {
            fetch(k0 + BK);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The staged operands are done with; each simdgroup drains its fragments
    // through its own 8x8 slot at the start of the staging area.
    threadgroup float *slot = smem + sg * 64u;
    DIT_UNROLL for (uint i = 0; i < 4u; ++i) {
        DIT_UNROLL for (uint j = 0; j < 4u; ++j) {
            simdgroup_store(acc[i][j], slot, 8u);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            for (uint e = lane; e < 64u; e += 32u) {
                // Row-fastest drains let adjacent lanes store adjacent rows.
                const bool rows_fast = dit_stores_rows_fast<EP>::value != 0;
                const uint fr = rows_fast ? (e & 7u) : (e >> 3u);
                const uint fc = rows_fast ? (e >> 3u) : (e & 7u);
                const uint gm = m0 + r0 + 8u * i + fr;
                const uint gn = n0 + c0 + 8u * j + fc;
                if (gm < M && gn < N) {
                    ep.store(gm, gn, slot[fr * 8u + fc]);
                }
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}

#define DIT_GEMM64_PARAMS(WT) \
        constant dit_args_gemm & args    [[buffer(0)]], \
        device float *out                  [[buffer(1)]], \
        device const float *in             [[buffer(2)]], \
        device const WT *weight            [[buffer(3)]], \
        constant float &weight_scale       [[buffer(4)]], \
        device const float *bias           [[buffer(5)]], \
        uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort tid [[thread_index_in_threadgroup]], \
        ushort lane [[thread_index_in_simdgroup]], \
        ushort sg [[simdgroup_index_in_threadgroup]]

/// `out[m,n] = in[m,k] . weight[n,k] * scale + bias[n]` on the simdgroup
/// tile, `k % 16 == 0`. Grid `(ceil(n/64), ceil(m/64))` x 128 threads.
kernel void kernel_dit_gemm64_f32(DIT_GEMM64_PARAMS(float)) {
    threadgroup float smem[DIT_G64_SMEM];
    dit_gemm64_tile(args.m, args.n, args.k,
                      dit_rows_f32{in, args.k}, dit_wrows_f32{weight, args.k},
                      dit_store_rows<dit_scale_tensor>{out, args.n, {weight_scale}, bias, args.has_bias != 0u},
                      smem, tgpig, tid, lane, sg);
}

/// The same tile over a brain-float weight.
kernel void kernel_dit_gemm64_bf16(DIT_GEMM64_PARAMS(ushort)) {
    threadgroup float smem[DIT_G64_SMEM];
    dit_gemm64_tile(args.m, args.n, args.k,
                      dit_rows_f32{in, args.k}, dit_wrows_bf16{weight, args.k},
                      dit_store_rows<dit_scale_tensor>{out, args.n, {weight_scale}, bias, args.has_bias != 0u},
                      smem, tgpig, tid, lane, sg);
}

/// The same tile over an E4M3 weight, decoded while staging; the tensor's
/// scale multiplies the accumulator in the epilogue.
kernel void kernel_dit_gemm64_fp8(DIT_GEMM64_PARAMS(uchar)) {
    threadgroup float smem[DIT_G64_SMEM];
    dit_gemm64_tile(args.m, args.n, args.k,
                      dit_rows_f32{in, args.k}, dit_wrows_fp8{weight, args.k},
                      dit_store_rows<dit_scale_tensor>{out, args.n, {weight_scale}, bias, args.has_bias != 0u},
                      smem, tgpig, tid, lane, sg);
}

/// The same tile over GGUF Q4_K and Q6_K weights, decoded while staging
/// (`k % 256 == 0`); binding 4 is the unit scale.
kernel void kernel_dit_gemm64_q4k(DIT_GEMM64_PARAMS(uchar)) {
    threadgroup float smem[DIT_G64_SMEM];
    dit_gemm64_tile(args.m, args.n, args.k,
                      dit_rows_f32{in, args.k}, dit_wrows_q4k{weight, args.k},
                      dit_store_rows<dit_scale_tensor>{out, args.n, {weight_scale}, bias, args.has_bias != 0u},
                      smem, tgpig, tid, lane, sg);
}

kernel void kernel_dit_gemm64_q6k(DIT_GEMM64_PARAMS(uchar)) {
    threadgroup float smem[DIT_G64_SMEM];
    dit_gemm64_tile(args.m, args.n, args.k,
                      dit_rows_f32{in, args.k}, dit_wrows_q6k{weight, args.k},
                      dit_store_rows<dit_scale_tensor>{out, args.n, {weight_scale}, bias, args.has_bias != 0u},
                      smem, tgpig, tid, lane, sg);
}

/// The same tile over a per-row INT8 weight, widened while staging; the
/// row's scale multiplies the accumulator in the epilogue. Binding 4 is the
/// `n`-long scale vector, not a scalar.
kernel void kernel_dit_gemm64_i8(
        constant dit_args_gemm & args    [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const char *weight          [[buffer(3)]],
        device const float *row_scale      [[buffer(4)]],
        device const float *bias           [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[DIT_G64_SMEM];
    dit_gemm64_tile(args.m, args.n, args.k,
                      dit_rows_f32{in, args.k}, dit_wrows_i8{weight, args.k},
                      dit_store_rows<dit_scale_rows>{out, args.n, {row_scale}, bias, args.has_bias != 0u},
                      smem, tgpig, tid, lane, sg);
}

// --- implicit-gemm convolution -----------------------------------------------------

/// The convolution's left operand without an im2col buffer: row `r` is
/// output pixel `(r / out_w, r % out_w)` of one batch image and column `t` is
/// tap `(ic, kh, kw)` in the weight's own `[ic][kh][kw]` order, so the weight
/// `[out_c][in_c][kh][kw]` is already the `[n][k]` matrix the tile reads.
/// Taps outside the padded field read as zero.
struct dit_im2col {
    device const float *in; // this batch image's (in_c, in_h, in_w) field
    uint in_h;
    uint in_w;
    uint out_w;
    uint k_h;
    uint k_w;
    uint stride;
    uint pad;
    /// Four consecutive taps of one output pixel. The pixel and the first
    /// tap are decomposed once and the tap is then stepped (kw, then kh,
    /// then ic): integer division is the slow path on the GPU, and
    /// decomposing every tap from scratch - sixteen divisions per load - held
    /// the decoder's convolutions to a third of the plain GEMM's rate on the
    /// same tile.
    float4 load4(uint r, uint kk) const {
        const uint oh = r / out_w;
        const uint ow = r - oh * out_w;
        const uint khw = k_h * k_w;
        uint ic = kk / khw;
        const uint rem = kk - ic * khw;
        uint kh = rem / k_w;
        uint kw = rem - kh * k_w;
        const int ih0 = int(oh * stride) - int(pad);
        const int iw0 = int(ow * stride) - int(pad);
        float v[4];
        DIT_UNROLL for (uint c = 0; c < 4u; ++c) {
            const int ih = ih0 + int(kh);
            const int iw = iw0 + int(kw);
            v[c] = (ih < 0 || iw < 0 || ih >= int(in_h) || iw >= int(in_w))
                    ? 0.0f
                    : in[((uint64_t) ic * in_h + uint(ih)) * in_w + uint(iw)];
            if (++kw == k_w) {
                kw = 0u;
                if (++kh == k_h) {
                    kh = 0u;
                    ++ic;
                }
            }
        }
        return float4(v[0], v[1], v[2], v[3]);
    }
};

/// NCHW epilogue: column `c` is the output channel, row `r` the pixel.
struct dit_store_nchw {
    device float *out; // this batch image's (out_c, out_h * out_w) field
    uint plane;
    device const float *bias;
    bool has_bias;
    void store(uint r, uint c, float v) const {
        out[(uint64_t) c * plane + r] = v + (has_bias ? bias[c] : 0.0f);
    }
};

/// One NCHW f32 convolution as an implicit GEMM on the simdgroup tile:
/// M = out_h * out_w pixels, N = out_c, K = in_c * k_h * k_w taps, which the
/// driver guarantees is a multiple of 16. Grid
/// `(ceil(out_c/64), ceil(out_h*out_w/64), batch)` x 128 threads.
///
/// The per-pixel kernel above spent ~97 s of a 512x512 decode on an M1 Max:
/// one 64-thread group per output value, each re-reading its whole receptive
/// field. Here every staged input value feeds 64 output channels.
kernel void kernel_dit_conv2d_gemm_f32(
        constant dit_args_conv2d & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        device const float *weight         [[buffer(3)]],
        device const float *bias           [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[DIT_G64_SMEM];
    const uint b = tgpig.z;
    const uint plane = args.out_h * args.out_w;
    const uint taps = args.in_c * args.k_h * args.k_w;
    const dit_im2col a{
            in + (uint64_t) b * args.in_c * args.in_h * args.in_w,
            args.in_h, args.in_w, args.out_w, args.k_h, args.k_w, args.stride, args.pad};
    const dit_store_nchw ep{
            out + (uint64_t) b * args.out_c * plane, plane, bias, args.has_bias != 0u};
    dit_gemm64_tile(plane, args.out_c, taps, a, dit_wrows_f32{weight, taps}, ep,
                      smem, tgpig, tid, lane, sg);
}

// --- fused attention -----------------------------------------------------------------

/// Where head `h` of joint row `t` starts for one operand, with that
/// operand's own section offset (see `kernel_dit_sdpa_f32`).
static inline device const float *dit_joint_row(
        constant dit_args_sdpa & args, uint t, uint h, uint offset,
        device const float *txt, device const float *img, device const float *cache) {
    const uint hd = args.head_dim;
    if (t < args.txt_tokens) {
        return txt + (uint64_t) t * args.txt_stride + offset + h * hd;
    }
    if (t < args.txt_tokens + args.img_tokens) {
        return img + (uint64_t) (t - args.txt_tokens) * args.img_stride + offset + h * hd;
    }
    return cache + (uint64_t) (t - args.txt_tokens - args.img_tokens) * args.cache_stride
            + offset + h * hd;
}

/// `v` narrowed to half, saturating: a value past half's range stages as
/// the largest finite half, not as an infinity that would turn the whole
/// row's softmax into NaN.
static inline half4 dit_half4_sat(float4 v) {
    return half4(clamp(v, float4(-65504.0f), float4(65504.0f)));
}

/// Joint-sequence attention in one pass over the keys with an online
/// softmax, on simdgroup matrices: the same contract, bindings and output as
/// `kernel_dit_sdpa_f32`, for `head_dim % 8 == 0` and `head_dim <= 128`.
///
/// A threadgroup takes 32 query rows of one head (eight per simdgroup, held
/// in registers as 8x8 fragments, pre-scaled by 1/sqrt(head_dim)) and
/// streams the keys 32 at a time through threadgroup memory, the next block
/// loading into registers while the current one is multiplied. The softmax
/// never leaves registers: a lane holds two columns of one row of each score
/// fragment, the four lanes of a row (lane bits 0 and 3) fold the running
/// maximum and denominator with two shuffles, and the value accumulators,
/// laid out the same way, are rescaled lane by lane.
///
/// q, k, v and the probabilities are staged as half; every product
/// accumulates in f32. Half carries more mantissa than the bf16 the
/// reference attends in, q and k come out of `QK-norm`, and the staging
/// saturates (`dit_half4_sat`). Against the f32 version of this kernel it
/// halves the threadgroup traffic that bounds it: 4115 joint rows x 24 heads
/// of 128 (a 1024x1024 image) take 115 ms per layer on an M1 Max, from 228 ms
/// for the previous kernel, which kept the scores in f32 and handed them
/// through threadgroup memory four barriers per 16 keys.
///
/// Keys past the end are masked to -inf, so ragged sequence lengths need no
/// special case. The softmax is the max-subtracted form the per-row kernel
/// computes, accumulated block by block.
///
/// The causal instantiation (`kernel_dit_flash_sdpa_causal_f32`) masks every
/// key after the query's own joint row (`s > t`) and stops streaming at the
/// tile's last row, so a prefix of P rows costs half the score blocks of the
/// bidirectional pass. The driver sends it no cached rows: a cached key
/// sits after every live row and would always be masked.
#define DIT_FLASH_SDPA_PARAMS \
        constant dit_args_sdpa & args [[buffer(0)]], \
        device float *out_txt            [[buffer(1)]], \
        device float *out_img            [[buffer(2)]], \
        device const float *q_txt        [[buffer(3)]], \
        device const float *k_txt        [[buffer(4)]], \
        device const float *v_txt        [[buffer(5)]], \
        device const float *q_img        [[buffer(6)]], \
        device const float *k_img        [[buffer(7)]], \
        device const float *v_img        [[buffer(8)]], \
        device const float *cache_k      [[buffer(9)]], \
        device const float *cache_v      [[buffer(10)]], \
        uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort tid [[thread_index_in_threadgroup]], \
        ushort lane [[thread_index_in_simdgroup]], \
        ushort sg [[simdgroup_index_in_threadgroup]]

template <bool CAUSAL, uint RB, uint NSG, uint HD>
kernel void kernel_dit_flash_sdpa(DIT_FLASH_SDPA_PARAMS) {
    const uint QR = 8u * RB * NSG;
    const uint KB = DIT_FA_KEYS;
    const uint NB = KB / 8u;
    const uint NT = 32u * NSG;
    const uint LDK = DIT_FA_MAX_D + 8u;
    // Only the K and V blocks pass through threadgroup memory; each row is
    // padded by eight halves so the transposed K loads do not land eight rows
    // on one bank.
    threadgroup half kv[2u * DIT_FA_KEYS * (DIT_FA_MAX_D + 8u)];

    const uint h = tgpig.y;
    const uint q0 = tgpig.x * QR;
    // HD == 0 reads the head width from the arguments; a fixed width lets
    // every head loop unroll, so the fragment arrays stay in registers.
    const uint hd = HD == 0u ? args.head_dim : HD;
    const uint nd = hd / 8u;
    const uint rows = args.txt_tokens + args.img_tokens;
    // A causal tile's last row sees no key past itself, so the stream stops
    // there; rows past the end load zero queries and are never written.
    const uint keys = CAUSAL ? min(rows, q0 + QR) : rows + args.cache_tokens;
    if (h >= args.heads) {
        return;
    }
    const float scale = 1.0f / sqrt(float(hd));

    // The lane's place in every 8x8 fragment: columns `fn, fn + 1` of row
    // `fr`; the four lanes sharing a row differ in lane bits 0 and 3.
    const ushort fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    const ushort fr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    uint qrow[RB];
    DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
        qrow[rb] = q0 + (sg * RB + rb) * 8u + fr;
    }

    // The query fragments, pre-scaled and narrowed, read straight into the
    // lane's own fragment elements (a fragment's element layout is the same
    // for every element type).
    simdgroup_half8x8 qf[RB][DIT_FA_MAX_D / 8u];
    simdgroup_float8x8 of[RB][DIT_FA_MAX_D / 8u];
    DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
        const bool live = qrow[rb] < rows;
        device const float *qp = live
                ? dit_joint_row(args, qrow[rb], h, args.q_offset, q_txt, q_img, q_txt)
                : q_txt;
        DIT_UNROLL for (uint i = 0; i < (HD == 0u ? nd : HD / 8u); ++i) {
            const float2 qv = live ? float2(qp[8u * i + fn], qp[8u * i + fn + 1u]) * scale
                                   : float2(0.0f);
            const half2 qh = half2(clamp(qv, float2(-65504.0f), float2(65504.0f)));
            qf[rb][i].thread_elements()[0] = qh.x;
            qf[rb][i].thread_elements()[1] = qh.y;
            of[rb][i] = simdgroup_float8x8(0.0f);
        }
    }

    float run_max[RB];
    float run_sum[RB];
    DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
        run_max[rb] = -INFINITY;
        run_sum[rb] = 0.0f;
    }
    threadgroup half *sk = kv;
    threadgroup half *sv = kv + KB * LDK;
    // Four consecutive head values per load, narrowed as they arrive so the
    // prefetch holds halves; a key past the end stages as zero and is
    // masked below.
    const uint SL = (KB * DIT_FA_MAX_D / 4u + NT - 1u) / NT;
    const uint hd4 = hd / 4u;
    half4 rk[SL];
    half4 rv[SL];
    auto fetch = [&](uint s0) {
        DIT_UNROLL for (uint q = 0; q < SL; ++q) {
            const uint e = tid + q * NT;
            const uint j = e / hd4;
            const uint d = (e - j * hd4) * 4u;
            const uint s = s0 + j;
            rk[q] = half4(0.0h);
            rv[q] = half4(0.0h);
            if (e < KB * hd4 && s < keys) {
                rk[q] = dit_half4_sat(float4(*((device const packed_float4 *)
                        (dit_joint_row(args, s, h, args.k_offset, k_txt, k_img, cache_k) + d))));
                rv[q] = dit_half4_sat(float4(*((device const packed_float4 *)
                        (dit_joint_row(args, s, h, args.v_offset, v_txt, v_img, cache_v) + d))));
            }
        }
    };
    fetch(0);
    for (uint s0 = 0; s0 < keys; s0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        DIT_UNROLL for (uint q = 0; q < SL; ++q) {
            const uint e = tid + q * NT;
            if (e < KB * hd4) {
                const uint j = e / hd4;
                const uint d = (e - j * hd4) * 4u;
                *((threadgroup half4 *) (sk + j * LDK + d)) = rk[q];
                *((threadgroup half4 *) (sv + j * LDK + d)) = rv[q];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s0 + KB < keys) {
            fetch(s0 + KB);
        }

        // Scores: S (8 x KB) = Q (8 x hd) . K^T per row block, one fragment
        // per 8 keys; each K fragment is loaded once for every row block.
        simdgroup_float8x8 sf[RB][NB];
        DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
            DIT_UNROLL for (uint b = 0; b < NB; ++b) {
                sf[rb][b] = simdgroup_float8x8(0.0f);
            }
        }
        DIT_UNROLL for (uint i = 0; i < (HD == 0u ? nd : HD / 8u); ++i) {
            DIT_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_half8x8 kt;
                // kt(r, c) = K[8b + c][8i + r]: the transposed load.
                simdgroup_load(kt, sk + 8u * b * LDK + 8u * i, LDK, ulong2(0, 0), true);
                DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
                    simdgroup_multiply_accumulate(sf[rb][b], qf[rb][i], kt, sf[rb][b]);
                }
            }
        }

        simdgroup_half8x8 pf[RB][NB];
        float alpha[RB];
        DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
            float block_max = -INFINITY;
            DIT_UNROLL for (uint b = 0; b < NB; ++b) {
                DIT_UNROLL for (uint c = 0; c < 2u; ++c) {
                    const uint key = s0 + 8u * b + fn + c;
                    if (key >= keys || (CAUSAL && key > qrow[rb])) {
                        sf[rb][b].thread_elements()[c] = -INFINITY;
                    }
                    block_max = max(block_max, sf[rb][b].thread_elements()[c]);
                }
            }
            block_max = max(block_max, simd_shuffle_xor(block_max, 1));
            block_max = max(block_max, simd_shuffle_xor(block_max, 8));
            const float new_max = max(run_max[rb], block_max);
            alpha[rb] = exp(run_max[rb] - new_max);
            float block_sum = 0.0f;
            DIT_UNROLL for (uint b = 0; b < NB; ++b) {
                DIT_UNROLL for (uint c = 0; c < 2u; ++c) {
                    const float p = exp(sf[rb][b].thread_elements()[c] - new_max);
                    pf[rb][b].thread_elements()[c] = half(p);
                    block_sum += p;
                }
            }
            block_sum += simd_shuffle_xor(block_sum, 1);
            block_sum += simd_shuffle_xor(block_sum, 8);
            run_sum[rb] = run_sum[rb] * alpha[rb] + block_sum;
            run_max[rb] = new_max;
        }
        DIT_UNROLL for (uint i = 0; i < (HD == 0u ? nd : HD / 8u); ++i) {
            DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
                of[rb][i].thread_elements()[0] *= alpha[rb];
                of[rb][i].thread_elements()[1] *= alpha[rb];
            }
            DIT_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_half8x8 vf;
                simdgroup_load(vf, sv + 8u * b * LDK + 8u * i, LDK);
                DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
                    simdgroup_multiply_accumulate(of[rb][i], pf[rb][b], vf, of[rb][i]);
                }
            }
        }
    }

    // Normalize by the denominators and write each lane's two elements of
    // every fragment straight to its row.
    const uint width = args.heads * hd;
    DIT_UNROLL for (uint rb = 0; rb < RB; ++rb) {
        const uint t = qrow[rb];
        if (t >= rows) {
            continue;
        }
        const float inv = 1.0f / run_sum[rb];
        device float *op = t < args.txt_tokens
                ? out_txt + (uint64_t) t * width + h * hd
                : out_img + (uint64_t) (t - args.txt_tokens) * width + h * hd;
        DIT_UNROLL for (uint i = 0; i < (HD == 0u ? nd : HD / 8u); ++i) {
            op[8u * i + fn] = of[rb][i].thread_elements()[0] * inv;
            op[8u * i + fn + 1u] = of[rb][i].thread_elements()[1] * inv;
        }
    }
}

template [[host_name("kernel_dit_flash_sdpa_f32")]]
kernel void kernel_dit_flash_sdpa<false, DIT_FA_RB, DIT_FA_SG, 0u>(DIT_FLASH_SDPA_PARAMS);

template [[host_name("kernel_dit_flash_sdpa_causal_f32")]]
kernel void kernel_dit_flash_sdpa<true, DIT_FA_RB, DIT_FA_SG, 0u>(DIT_FLASH_SDPA_PARAMS);

template [[host_name("kernel_dit_flash_sdpa_d128_f32")]]
kernel void kernel_dit_flash_sdpa<false, DIT_FA_RB, DIT_FA_SG, 128u>(DIT_FLASH_SDPA_PARAMS);

template [[host_name("kernel_dit_flash_sdpa_causal_d128_f32")]]
kernel void kernel_dit_flash_sdpa<true, DIT_FA_RB, DIT_FA_SG, 128u>(DIT_FLASH_SDPA_PARAMS);

// --- fused linear: swiglu/concat input, gated-residual output --------------------

/// The input row of a block's output linear without materializing it:
/// columns `[0, cols_a)` are `a`'s row, columns past it are
/// `silu(g) * u` of the fused gate/up row. A float4 never straddles the two
/// parts because both widths are multiples of four.
struct dit_rows_swiglu {
    device const float *a;
    device const float *gu;
    uint cols_a;
    uint inner;
    uint gu_stride;
    uint gu_offset;
    float4 load4(uint r, uint kk) const {
        if (kk < cols_a) {
            return float4(*((device const packed_float4 *) (a + (uint64_t) r * cols_a + kk)));
        }
        device const float *g = gu + (uint64_t) r * gu_stride + gu_offset + (kk - cols_a);
        const float4 gv = float4(*((device const packed_float4 *) g));
        const float4 uv = float4(*((device const packed_float4 *) (g + inner)));
        return float4(dit_silu(gv.x), dit_silu(gv.y), dit_silu(gv.z), dit_silu(gv.w)) * uv;
    }
};

/// The gated residual in the epilogue: each output element is read and
/// written by the one thread that owns it, so updating the stream in place
/// is race-free.
template <typename SC>
struct dit_store_gated {
    device float *out;
    uint n;
    SC scale;
    device const float *bias;
    bool has_bias;
    device const float *gate;
    void store(uint r, uint c, float v) const {
        const uint64_t i = (uint64_t) r * n + c;
        out[i] += gate[c] * (v * scale.at(c) + (has_bias ? bias[c] : 0.0f));
    }
};

#define DIT_GEMM_FUSED_PARAMS_SCALED(WT, SCALE) \
        constant dit_args_gemm_fused & args [[buffer(0)]], \
        device float *out                     [[buffer(1)]], \
        device const float *a                 [[buffer(2)]], \
        device const float *gu                [[buffer(3)]], \
        device const WT *weight               [[buffer(4)]], \
        SCALE                                 [[buffer(5)]], \
        device const float *bias              [[buffer(6)]], \
        device const float *gate              [[buffer(7)]], \
        uint3 tgpig[[threadgroup_position_in_grid]], \
        ushort tid [[thread_index_in_threadgroup]], \
        ushort lane [[thread_index_in_simdgroup]], \
        ushort sg [[simdgroup_index_in_threadgroup]]

#define DIT_GEMM_FUSED_PARAMS(WT) DIT_GEMM_FUSED_PARAMS_SCALED(WT, constant float &weight_scale)

#define DIT_GEMM_FUSED_BODY_SCALED(WR, SC, SCALE) \
    threadgroup float smem[DIT_G64_SMEM]; \
    const uint k = args.cols_a + args.inner; \
    dit_gemm64_tile(args.m, args.n, k, \
                      dit_rows_swiglu{a, gu, args.cols_a, args.inner, args.gu_stride, args.gu_offset}, \
                      WR{weight, k}, \
                      dit_store_gated<SC>{out, args.n, {SCALE}, bias, args.has_bias != 0u, gate}, \
                      smem, tgpig, tid, lane, sg);

#define DIT_GEMM_FUSED_BODY(WR) DIT_GEMM_FUSED_BODY_SCALED(WR, dit_scale_tensor, weight_scale)

/// `out += gate * linear([a | swiglu(gu)])` on the simdgroup tile. Grid
/// `(ceil(n/64), ceil(m/64))` x 128 threads.
///
/// One launch replaces four in every transformer block: the swiglu pass,
/// the concat, the linear into a scratch buffer and the gated add - and the
/// two full-width intermediates they wrote and read back.
kernel void kernel_dit_gemm64_fused_f32(DIT_GEMM_FUSED_PARAMS(float)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_f32)
}

kernel void kernel_dit_gemm64_fused_bf16(DIT_GEMM_FUSED_PARAMS(ushort)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_bf16)
}

kernel void kernel_dit_gemm64_fused_fp8(DIT_GEMM_FUSED_PARAMS(uchar)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_fp8)
}

kernel void kernel_dit_gemm64_fused_q4k(DIT_GEMM_FUSED_PARAMS(uchar)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q4k)
}

kernel void kernel_dit_gemm64_fused_q6k(DIT_GEMM_FUSED_PARAMS(uchar)) {
    DIT_GEMM_FUSED_BODY(dit_wrows_q6k)
}

/// The same fused linear over a per-row INT8 weight; binding 5 is the
/// `n`-long row-scale vector. A ConvRot weight's input must already be
/// rotated: the driver writes `[a | swiglu(gu)]` rotated into a packed
/// scratch row with `kernel_dit_hadamard256_f32` and binds it as `a`.
kernel void kernel_dit_gemm64_fused_i8(DIT_GEMM_FUSED_PARAMS_SCALED(char, device const float *row_scale)) {
    DIT_GEMM_FUSED_BODY_SCALED(dit_wrows_i8, dit_scale_rows, row_scale)
}

// --- ConvRot activation rotation ----------------------------------------------
//
// `x . H` for each 256-wide group of a row, `H = kron(H4, H4, H4, H4) / 16`
// with `H4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]`. Element `i` of a
// group has base-4 digits (i3 i2 i1 i0) and `H[i][j] = prod_d H4[i_d][j_d]`,
// so the product is four radix-4 butterflies, one per digit, each applying
// the symmetric H4 to the four elements that differ only in that digit:
// `y_j = S - 2 * v_(3-j)` with `S` their sum. Every factor is +-1 until the
// final 1/16, which is a power of two, so each output is a fixed-order sum
// of signed inputs scaled exactly.

/// The input element (r, kk) of a rotated linear: a plain row, or the
/// `[a | silu(g) * u]` row the fused linear forms.
static inline float dit_rot_input(constant dit_args_hadamard & args,
                                  device const float *a, device const float *gu,
                                  uint r, uint kk) {
    if (kk < args.cols_a) {
        return a[(uint64_t) r * args.cols_a + kk];
    }
    device const float *g = gu + (uint64_t) r * args.gu_stride + args.gu_offset + (kk - args.cols_a);
    return dit_silu(g[0]) * g[args.inner];
}

static inline float4 dit_h4(float4 v) {
    const float s = (v.x + v.y) + (v.z + v.w);
    return float4(s - 2.0f * v.w, s - 2.0f * v.z, s - 2.0f * v.y, s - 2.0f * v.x);
}

/// `out[r, :] = rot([a | swiglu(gu)][r, :])`, packed `cols_a + inner` wide,
/// a multiple of 256. Grid `(width / 256, rows)` x 64 threads: one
/// threadgroup per (row, group), four elements per thread per stage.
kernel void kernel_dit_hadamard256_f32(
        constant dit_args_hadamard & args [[buffer(0)]],
        device float *out                   [[buffer(1)]],
        device const float *a               [[buffer(2)]],
        device const float *gu              [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[256];
    const uint r = tgpig.y;
    const uint base = tgpig.x * 256u;
    const uint width = args.cols_a + args.inner;
    // Digit 0: the four elements 4t..4t+3 are this thread's own, so the
    // first butterfly runs on the loaded values before anything is staged.
    float4 x = float4(dit_rot_input(args, a, gu, r, base + 4u * tid),
                      dit_rot_input(args, a, gu, r, base + 4u * tid + 1u),
                      dit_rot_input(args, a, gu, r, base + 4u * tid + 2u),
                      dit_rot_input(args, a, gu, r, base + 4u * tid + 3u));
    x = dit_h4(x);
    *((threadgroup float4 *) (v + 4u * tid)) = x;
    // Digits 1 and 2 (strides 4 and 16) through threadgroup memory: thread t
    // owns the tuple whose free digit is the stage's, at `hi * 4s + lo`.
    DIT_UNROLL for (uint stride = 4u; stride <= 16u; stride *= 4u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint lo = tid % stride;
        const uint b = (tid - lo) * 4u + lo;
        float4 y = dit_h4(float4(v[b], v[b + stride], v[b + 2u * stride], v[b + 3u * stride]));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v[b] = y.x;
        v[b + stride] = y.y;
        v[b + 2u * stride] = y.z;
        v[b + 3u * stride] = y.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Digit 3 (stride 64): thread t owns t, t+64, t+128, t+192 and writes
    // them straight out - consecutive threads, consecutive addresses.
    float4 y = dit_h4(float4(v[tid], v[tid + 64u], v[tid + 128u], v[tid + 192u])) * 0.0625f;
    device float *o = out + (uint64_t) r * width + base;
    o[tid] = y.x;
    o[tid + 64u] = y.y;
    o[tid + 128u] = y.z;
    o[tid + 192u] = y.w;
}

// --- residual VAE decoder: channel RMS norm, fused convolutions, dup-up -------
//
// The residual VAE decoder normalizes over the channel axis of each pixel
// (`x / max(||x[:, s]||, eps) * sqrt(C) * gamma`, no mean, no groups) and
// always feeds the normalized field straight into a convolution. So the
// norm is split: one pass writes a per-pixel inverse norm, and the
// convolution's gather applies it (with the gain and, in the resnets, the
// SiLU) as it stages each tap. The normalized field - the decoder's largest
// tensor at full resolution - is never written. The upsamplers' nearest 2x
// interpolation rides the gather the same way.

/// Channel RMS norm, pass 1: `inv[s] = scale / max(sqrt(sum_c x[c, s]^2),
/// eps)`, the squares folded in channel order. One thread per pixel;
/// adjacent threads read adjacent pixels of one channel plane, so every load
/// is coalesced. Grid `ceil(spatial / 64)` x 64 threads.
kernel void kernel_dit_channel_rms_inv_f32(
        constant dit_args_channel_rms & args [[buffer(0)]],
        device float *inv                      [[buffer(1)]],
        device const float *in                 [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint s = tgpig.x * DIT_BLOCK + tpitg.x;
    if (s >= args.spatial) {
        return;
    }
    float ss = 0.0f;
    for (uint c = 0; c < args.channels; ++c) {
        const float v = in[(uint64_t) c * args.spatial + s];
        ss += v * v;
    }
    inv[s] = args.scale / max(sqrt(ss), args.eps);
}

/// `n / d` for `n < 2^24`: a float reciprocal estimate, which is off by at
/// most one there, then corrected exactly. The integer divide is a long
/// software sequence on the GPU, and three of them per gathered vector cost
/// the VAE convolution 8% (2.36 against 2.55 TFLOPS with the divisors
/// constant, on an M4).
static inline uint dit_div(uint n, uint d) {
    uint q = uint(float(n) / float(d));
    if (q * d > n) {
        --q;
    } else if ((q + 1u) * d <= n) {
        ++q;
    }
    return q;
}

/// The gather transforms of the VAE convolutions.
#define DIT_VAE_RMS 0u
#define DIT_VAE_UP2 1u

/// The convolution's left operand, as `dit_im2col` (stride 1, square
/// kernel) but with the taps ordered channel-fastest (see `load4m`), with
/// the input transform applied per tap:
///
/// - `DIT_VAE_RMS`: the tap reads `x[ic, p] * inv[p] * gamma[ic]`, then
///   SiLU when `silu` is set - the reference's norm -> (silu) -> conv;
/// - `DIT_VAE_UP2`: the convolved field is the stored one upsampled 2x by
///   nearest neighbour (`(f_h, f_w) == 2 * (in_h, in_w)`), so tap
///   `(ih, iw)` reads stored pixel `(ih / 2, iw / 2)`.
///
/// Taps outside the convolved field read as zero: the reference pads the
/// transformed field, so the padding is never normalized or activated.
template <uint MODE>
struct dit_im2col_vae {
    device const float *in;
    device const float *inv;
    device const float *gamma;
    uint in_h;
    uint in_w;
    uint f_h;
    uint f_w;
    uint out_w;
    uint k;
    uint pad;
    bool silu;
    uint in_c_;
    /// The tile reads this operand through `load4m` only
    /// (`dit_reads_along_m`); the k-wise gather is never called.
    float4 load4(uint r, uint kk) const {
        return float4(0.0f);
    }
    /// Column `kk` of the four output pixels `r..r+3`, zero past row `m`.
    /// The fused convolutions order their taps channel-fastest, `kk = (kh *
    /// k + kw) * in_c + ic` (the driver stores the weight `[o][kh][kw][i]`),
    /// so a 16-wide k slice is sixteen channels of one tap and the
    /// decomposition below runs once per column, not per value. Four
    /// pixels of one output row whose taps all land inside the field are
    /// four adjacent inputs, read as one vector; the edges take the
    /// per-value path.
    float4 load4m(uint r, uint kk, uint m) const {
        const uint tap = dit_div(kk, in_c_);
        const uint ic = kk - tap * in_c_;
        const uint kh = dit_div(tap, k);
        const uint kw = tap - kh * k;
        const uint plane = in_h * in_w;
        const float g = MODE == DIT_VAE_RMS ? gamma[ic] : 1.0f;
        const uint oh = dit_div(r, out_w);
        const uint ow = r - oh * out_w;
        const int ih = int(oh) - int(pad) + int(kh);
        const int iw = int(ow) - int(pad) + int(kw);
        float4 v;
        if (r + 3u < m && ow + 3u < out_w && ih >= 0 && ih < int(f_h) && iw >= 0
                && iw + 3 < int(f_w)) {
            if (MODE == DIT_VAE_UP2) {
                // Columns iw..iw+3 of the doubled field are stored columns
                // iw/2 .. (iw+3)/2: two or three values, each repeated.
                device const float *row = in + (uint64_t) ic * plane + (uint(ih) >> 1u) * in_w;
                const uint c0 = uint(iw);
                v = float4(row[c0 >> 1u], row[(c0 + 1u) >> 1u], row[(c0 + 2u) >> 1u],
                           row[(c0 + 3u) >> 1u]);
            } else {
                const uint p = uint(ih) * in_w + uint(iw);
                const float4 x = float4(*((device const packed_float4 *) (in + (uint64_t) ic * plane + p)))
                        * float4(*((device const packed_float4 *) (inv + p))) * g;
                v = silu ? float4(dit_silu(x.x), dit_silu(x.y), dit_silu(x.z), dit_silu(x.w)) : x;
            }
            return v;
        }
        uint ph = oh;
        uint pw = ow;
        float e[4];
        DIT_UNROLL for (uint c = 0; c < 4u; ++c) {
            const int yh = int(ph) - int(pad) + int(kh);
            const int xw = int(pw) - int(pad) + int(kw);
            if (r + c >= m || yh < 0 || xw < 0 || yh >= int(f_h) || xw >= int(f_w)) {
                e[c] = 0.0f;
            } else if (MODE == DIT_VAE_UP2) {
                e[c] = in[(uint64_t) ic * plane + (uint(yh) >> 1u) * in_w + (uint(xw) >> 1u)];
            } else {
                const uint p = uint(yh) * in_w + uint(xw);
                const float x = in[(uint64_t) ic * plane + p] * inv[p] * g;
                e[c] = silu ? dit_silu(x) : x;
            }
            if (++pw == out_w) {
                pw = 0u;
                ++ph;
            }
        }
        return float4(e[0], e[1], e[2], e[3]);
    }
};
template <uint MODE> struct dit_reads_along_m<dit_im2col_vae<MODE>> { enum { value = 1 }; };
template <uint MODE>
static inline float4 dit_load4m(const dit_im2col_vae<MODE> a, uint r, uint kk, uint m) {
    return a.load4m(r, kk, m);
}

/// NCHW epilogue with the residual add: `out = conv + bias`, or
/// `out += conv + bias` when `accumulate` is set. Each output element is
/// read and written by the one thread that stores it, so accumulating into
/// the residual stream in place is race-free.
struct dit_store_nchw_acc {
    device float *out;
    uint plane;
    device const float *bias;
    bool has_bias;
    bool accumulate;
    void store(uint r, uint c, float v) const {
        const uint64_t i = (uint64_t) c * plane + r;
        const float y = v + (has_bias ? bias[c] : 0.0f);
        out[i] = accumulate ? out[i] + y : y;
    }
};
template <> struct dit_stores_rows_fast<dit_store_nchw_acc> { enum { value = 1 }; };

/// norm -> (silu) -> conv as one implicit GEMM: M = out_h * out_w pixels,
/// N = out_c, K = in_c * k * k taps (a multiple of 16, which the driver
/// guarantees). `inv` is pass 1's per-pixel inverse norm of `in`, `gamma`
/// the norm's per-channel gain. Grid `(ceil(out_c/64), ceil(M/64))` x 128.
kernel void kernel_dit_conv2d_rms_gemm_f32(
        constant dit_args_conv_vae & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *in               [[buffer(2)]],
        device const float *weight           [[buffer(3)]],
        device const float *bias             [[buffer(4)]],
        device const float *inv              [[buffer(5)]],
        device const float *gamma            [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[DIT_G64_SMEM];
    const uint plane = args.out_h * args.out_w;
    const uint taps = args.in_c * args.k * args.k;
    const dit_im2col_vae<DIT_VAE_RMS> a{
            in, inv, gamma, args.in_h, args.in_w, args.in_h, args.in_w,
            args.out_w, args.k, args.pad, args.silu != 0u, args.in_c};
    const dit_store_nchw_acc ep{out, plane, bias, args.has_bias != 0u, args.accumulate != 0u};
    dit_gemm64_tile(plane, args.out_c, taps, a, dit_wrows_f32{weight, taps}, ep,
                      smem, tgpig, tid, lane, sg);
}

/// nearest-2x -> conv as one implicit GEMM over the upsampled field, which
/// is never materialized: `(in_h, in_w)` is the stored field, `(out_h,
/// out_w)` the convolved one (twice it, same padding). Grid and K contract
/// as `kernel_dit_conv2d_rms_gemm_f32`; `silu` is ignored.
kernel void kernel_dit_conv2d_up2_gemm_f32(
        constant dit_args_conv_vae & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *in               [[buffer(2)]],
        device const float *weight           [[buffer(3)]],
        device const float *bias             [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float smem[DIT_G64_SMEM];
    const uint plane = args.out_h * args.out_w;
    const uint taps = args.in_c * args.k * args.k;
    const dit_im2col_vae<DIT_VAE_UP2> a{
            in, in, in, args.in_h, args.in_w, 2u * args.in_h, 2u * args.in_w,
            args.out_w, args.k, args.pad, false, args.in_c};
    const dit_store_nchw_acc ep{out, plane, bias, args.has_bias != 0u, args.accumulate != 0u};
    dit_gemm64_tile(plane, args.out_c, taps, a, dit_wrows_f32{weight, taps}, ep,
                      smem, tgpig, tid, lane, sg);
}

/// The up block's parameter-free shortcut, added in place into the
/// upsampler's output (`dit_args_dup_up`): every source value is
/// duplicated across `repeats` channels and spread over a 2x2 (and, for
/// a temporal factor, a frame) neighbourhood, of which a single frame keeps
/// the slot `offset` selects. Grid `(ceil(4hw / 64), out_c)` x 64 threads.
kernel void kernel_dit_dup_up_add_f32(
        constant dit_args_dup_up & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device const float *in            [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint ow = 2u * args.w;
    const uint plane = ow * 2u * args.h;
    const uint p = tgpig.x * DIT_BLOCK + tpitg.x;
    const uint c = tgpig.y;
    if (p >= plane || c >= args.out_c) {
        return;
    }
    const uint y = p / ow;
    const uint x = p - y * ow;
    const uint src = (c * args.factor + args.offset + (y & 1u) * 2u + (x & 1u)) / args.repeats;
    out[(uint64_t) c * plane + p] +=
            in[((uint64_t) src * args.h + (y >> 1u)) * args.w + (x >> 1u)];
}
