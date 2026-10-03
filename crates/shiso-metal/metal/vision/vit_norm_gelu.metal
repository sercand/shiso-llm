// Normalization, the two activations and the residual add.
//
// The tower normalizes with a mean and a variance, not an RMS, and the
// normalization carries both a scale and an offset. That is the single biggest
// difference from the language model's blocks, and it is why none of the
// existing norm kernels could be reused: subtracting the mean is not an
// optimization away from the RMS form, it is a different function.

/// `out[r, i] = (in[r, i] - mean) * rsqrt(var + eps) * weight[i] + bias[i]`.
///
/// One threadgroup per row, two block reductions: the mean, then the variance
/// about it. The variance pass re-reads the row rather than deriving it from a
/// sum of squares, because the row is already in cache by then and the
/// sum-of-squares form loses precision exactly where a normalization matters --
/// a row whose values are large and whose variance is small.
///
/// Safe to run in place (`out == in`): both reductions complete before the
/// write loop starts, and the write loop touches each index once.
template <typename W>
static inline void vit_layer_norm_body(
        constant vit_args_layer_norm & args,
        device float *out,
        device const float *in,
        device const ushort *weight,
        device const ushort *bias,
        threadgroup float *scratch,
        uint3 tgpig,
        ushort3 tpitg,
        ushort3 ntg) {
    const uint r = tgpig.x;
    if (r >= args.rows) {
        return;
    }
    const uint nthreads = ntg.x;
    device const float *src = in + (uint64_t) r * args.dim;
    device float *dst = out + (uint64_t) r * args.dim;

    float sum = 0.0f;
    for (uint i = tpitg.x; i < args.dim; i += nthreads) {
        sum += src[i];
    }
    const float mean = vit_block_sum(sum, scratch, tpitg.x, nthreads) / (float) args.dim;

    float var = 0.0f;
    for (uint i = tpitg.x; i < args.dim; i += nthreads) {
        const float d = src[i] - mean;
        var += d * d;
    }
    const float rstd = rsqrt(
            vit_block_sum(var, scratch, tpitg.x, nthreads) / (float) args.dim + args.eps);

    for (uint i = tpitg.x; i < args.dim; i += nthreads) {
        dst[i] = (src[i] - mean) * rstd * W::load(weight + i) + W::load(bias + i);
    }
}

// The brain-float tower's entry and the half-precision tower's twin.
kernel void kernel_vit_layer_norm(
        constant vit_args_layer_norm & args [[buffer(0)]],
        device float *out                   [[buffer(1)]],
        device const float *in              [[buffer(2)]],
        device const ushort *weight         [[buffer(3)]],
        device const ushort *bias           [[buffer(4)]],
        threadgroup float *scratch          [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    vit_layer_norm_body<vit_w_bf16>(args, out, in, weight, bias, scratch, tgpig, tpitg, ntg);
}

kernel void kernel_vit_layer_norm_f16(
        constant vit_args_layer_norm & args [[buffer(0)]],
        device float *out                   [[buffer(1)]],
        device const float *in              [[buffer(2)]],
        device const ushort *weight         [[buffer(3)]],
        device const ushort *bias           [[buffer(4)]],
        threadgroup float *scratch          [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    vit_layer_norm_body<vit_w_f16>(args, out, in, weight, bias, scratch, tgpig, tpitg, ntg);
}

/// The blocks' activation, applied in place to a feed-forward row.
///
/// The bias is already in `x`: it rides the GEMM's epilogue rather than being
/// added here, which keeps one addition in the accumulator's own precision
/// instead of two passes over the same buffer.
kernel void kernel_vit_gelu_tanh(
        constant vit_args_unary & args [[buffer(0)]],
        device float *x                [[buffer(1)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * 256u + tpitg.x;
    if (i >= args.count) {
        return;
    }
    x[i] = vit_gelu_tanh(x[i]);
}

/// The merger's activation, applied in place.
///
/// A separate entry point rather than a flag on the one above: the two differ
/// by about 1e-4 per value, the choice belongs to the layer rather than to the
/// launch, and a branch that picked between them would be a branch that could
/// be got wrong at one call site without anything failing loudly.
kernel void kernel_vit_gelu_erf(
        constant vit_args_unary & args [[buffer(0)]],
        device float *x                [[buffer(1)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * 256u + tpitg.x;
    if (i >= args.count) {
        return;
    }
    x[i] = vit_gelu_erf(x[i]);
}

/// `x[i] += add[i % period]`, in place.
///
/// Both residual adds use it with `period == count`, which is the plain
/// elementwise form. The position embedding uses a shorter period: a clip's
/// temporal groups sit on one patch grid, so one grid's worth of resampled
/// vectors is added to every group rather than being replicated on the host and
/// uploaded several times.
kernel void kernel_vit_add_inplace(
        constant vit_args_add & args [[buffer(0)]],
        device float *x              [[buffer(1)]],
        device const float *add      [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * 256u + tpitg.x;
    if (i >= args.count) {
        return;
    }
    x[i] += add[i % args.period];
}
