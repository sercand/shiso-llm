// Elementwise arithmetic of the audio encoders: activations, the scaled
// residual add, the gated linear unit and a decoder MLP's SiLU gate.

/// `out[i] = f(in[i])` for the contract's op codes; safe in place.
kernel void kernel_audio_unary_f32(
        constant audio_args_unary & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *in           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    float y;
    switch (args.op) {
        case AUDIO_OP_RELU:     y = max(x, 0.0f); break;
        case AUDIO_OP_SILU:     y = x * audio_sigmoid(x); break;
        case AUDIO_OP_SIGMOID:  y = audio_sigmoid(x); break;
        case AUDIO_OP_GELU_ERF: y = 0.5f * x * (1.0f + audio_erf(x * 0.70710678118654752f)); break;
        case AUDIO_OP_TANH:     y = precise::tanh(x); break;
        case AUDIO_OP_CLAMP:    y = clamp(x, args.a, args.b); break;
        case AUDIO_OP_SCALE:    y = x * args.a; break;
        default:                y = x; break;
    }
    out[i] = y;
}

/// `y[i] += alpha * x[i]`: the half-step feed-forward residual and the plain one.
kernel void kernel_audio_axpy_f32(
        constant audio_args_axpy & args [[buffer(0)]],
        device float *y                 [[buffer(1)]],
        device const float *x           [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    y[i] = y[i] + args.alpha * x[i];
}

/// `out[r][c] = in[r][c] * sigmoid(in[r][C + c])`: the first half of each
/// row is the value, the second half its gate.
kernel void kernel_audio_glu_f32(
        constant audio_args_glu & args [[buffer(0)]],
        device float *out              [[buffer(1)]],
        device const float *in         [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.rows * args.channels) {
        return;
    }
    const uint r = i / args.channels;
    const uint c = i - r * args.channels;
    device const float *row = in + (uint64_t) r * 2u * args.channels;
    out[i] = row[c] * audio_sigmoid(row[args.channels + c]);
}

/// `out[i] = silu(gate[i]) * up[i]`: a SiLU-gated MLP's product, the gate
/// and the value from two projections; safe with `out` bound to `gate`.
kernel void kernel_audio_silu_mul_f32(
        constant audio_args_silu_mul & args [[buffer(0)]],
        device float *out                   [[buffer(1)]],
        device const float *gate            [[buffer(2)]],
        device const float *up              [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const float g = gate[i];
    out[i] = g * audio_sigmoid(g) * up[i];
}

/// `y[r][c] += b[c]`: a linear's bias after a projection that carries none.
kernel void kernel_audio_bias_rows_f32(
        constant audio_args_bias_rows & args [[buffer(0)]],
        device float *y                      [[buffer(1)]],
        device const float *b                [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.rows * args.cols) {
        return;
    }
    y[i] = y[i] + b[i % args.cols];
}

/// The larger of two (value, index) candidates; a tie keeps the lower index.
static inline void audio_argmax_pick(thread float &v, thread uint &i, float ov, uint oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

/// Per row, the first index of the largest value, one threadgroup of
/// AUDIO_ARGMAX_BLOCK threads per row. A NaN never wins and a row with
/// nothing above negative infinity answers 0, so the answer is the one a
/// strictly-greater scan from index 0 gives.
kernel void kernel_audio_argmax_rows_f32(
        constant audio_args_argmax & args [[buffer(0)]],
        device uint *out                  [[buffer(1)]],
        device const float *in            [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float best_v[AUDIO_ARGMAX_BLOCK / 32u];
    threadgroup uint best_i[AUDIO_ARGMAX_BLOCK / 32u];
    const uint r = tgpig.x;
    if (r >= args.rows) {
        return;
    }
    device const float *row = in + (uint64_t) r * args.cols;
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint c = tpitg.x; c < args.cols; c += AUDIO_ARGMAX_BLOCK) {
        // Strictly greater: within a thread's stride the first one stays.
        if (row[c] > v) {
            v = row[c];
            i = c;
        }
    }
    for (ushort off = 16; off > 0; off >>= 1) {
        audio_argmax_pick(v, i, simd_shuffle_xor(v, off), simd_shuffle_xor(i, off));
    }
    if (lane == 0) {
        best_v[sg] = v;
        best_i[sg] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tpitg.x == 0) {
        for (uint s = 1; s < AUDIO_ARGMAX_BLOCK / 32u; ++s) {
            audio_argmax_pick(v, i, best_v[s], best_i[s]);
        }
        out[r] = i == 0xffffffffu ? 0u : i;
    }
}
