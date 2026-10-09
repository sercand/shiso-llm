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

/// `out[r][c] = gelu_tanh(gate[r][c]) * up[r * up_stride + up_offset + c]`: a
/// gated-GELU MLP's product of its gate and value projections, the value read from a
/// strided slice so a per-layer column block of a wider matrix needs no copy.
/// The tanh form of GELU, `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`,
/// with the tanh evaluated precisely as the unary entry's is. `out` may be
/// `gate`.
kernel void kernel_audio_gelu_tanh_mul_f32(
        constant audio_args_gelu_mul & args [[buffer(0)]],
        device float *out                   [[buffer(1)]],
        device const float *gate            [[buffer(2)]],
        device const float *up              [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.rows * args.cols) {
        return;
    }
    const uint r = i / args.cols;
    const uint c = i - r * args.cols;
    const float g = gate[i];
    const float t = precise::tanh(0.7978845608028654f * (g + 0.044715f * g * g * g));
    out[i] = 0.5f * g * (1.0f + t) * up[(uint64_t) r * args.up_stride + args.up_offset + c];
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

/// One wide row's argmax in two launches, for a row too long for one
/// threadgroup to scan quickly (a vocabulary's logits). The first launch:
/// threadgroup `p` scans columns `[p * span, (p + 1) * span)` as the one-pass
/// kernel scans a row and writes its (value, index) candidate to
/// `parts[p]` - the value's bits, then the index, `0xffffffff` for none. The
/// answer does not depend on how the row is cut: the largest value with the
/// lowest index among equals, a NaN never winning.
kernel void kernel_audio_argmax_split_f32(
        constant audio_args_argmax_split & args [[buffer(0)]],
        device uint *parts                      [[buffer(1)]],
        device const float *in                  [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float best_v[AUDIO_ARGMAX_BLOCK / 32u];
    threadgroup uint best_i[AUDIO_ARGMAX_BLOCK / 32u];
    const uint p = tgpig.x;
    if (p >= args.parts) {
        return;
    }
    const uint lo = p * args.span;
    const uint hi = min(args.cols, lo + args.span);
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint c = lo + tpitg.x; c < hi; c += AUDIO_ARGMAX_BLOCK) {
        if (in[c] > v) {
            v = in[c];
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
        parts[2u * p] = as_type<uint>(v);
        parts[2u * p + 1u] = i;
    }
}

/// The second launch: one threadgroup picks among the `parts` candidates and
/// writes the index, 0 when no candidate rose above negative infinity.
kernel void kernel_audio_argmax_merge_f32(
        constant audio_args_argmax_split & args [[buffer(0)]],
        device uint *out                        [[buffer(1)]],
        device const uint *parts                [[buffer(2)]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float best_v[AUDIO_ARGMAX_BLOCK / 32u];
    threadgroup uint best_i[AUDIO_ARGMAX_BLOCK / 32u];
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint p = tpitg.x; p < args.parts; p += AUDIO_ARGMAX_BLOCK) {
        audio_argmax_pick(v, i, as_type<float>(parts[2u * p]), parts[2u * p + 1u]);
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
        out[0] = i == 0xffffffffu ? 0u : i;
    }
}
