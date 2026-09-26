// Convolutions and layout moves of the audio encoders' front ends.

/// Stride-2 3x3 conv2d over channels-last `[t][f][c]`, full or depthwise, with
/// explicit leading pads, an input valid-row cut and an output one. See
/// KERNELS.md for the formula.
kernel void kernel_audio_conv2d_s2_f32(
        constant audio_args_conv2d & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device const float *in            [[buffer(2)]],
        device const float *w             [[buffer(3)]],
        device const float *bias          [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    const uint total = args.t_out * args.f_out * args.c_out;
    if (i >= total) {
        return;
    }
    const uint co = i % args.c_out;
    const uint f = (i / args.c_out) % args.f_out;
    const uint t = i / (args.c_out * args.f_out);
    if (t >= args.out_valid) {
        out[i] = 0.0f;
        return;
    }
    const uint rows = min(args.t_in, args.in_valid);
    const uint per = args.depthwise != 0u ? 1u : args.c_in;
    float acc = args.has_bias != 0u ? bias[co] : 0.0f;
    for (uint kt = 0; kt < 3u; ++kt) {
        const int ti = (int) (2u * t + kt) - (int) args.pad_t;
        if (ti < 0 || (uint) ti >= rows) {
            continue;
        }
        for (uint kf = 0; kf < 3u; ++kf) {
            const int fi = (int) (2u * f + kf) - (int) args.pad_f;
            if (fi < 0 || (uint) fi >= args.f_in) {
                continue;
            }
            device const float *px = in + ((uint64_t) ti * args.f_in + (uint) fi) * args.c_in;
            for (uint g = 0; g < per; ++g) {
                const uint ci = args.depthwise != 0u ? co : g;
                acc += w[((co * per + g) * 3u + kt) * 3u + kf] * px[ci];
            }
        }
    }
    out[i] = args.relu != 0u ? max(acc, 0.0f) : acc;
}

/// Depthwise conv1d over rows the caller has already left-padded (or prefixed
/// with a streaming cache): `out[t][c] = sum_j w[c][j] * in[t + j][c]`.
kernel void kernel_audio_dwconv1d_f32(
        constant audio_args_dwconv1d & args [[buffer(0)]],
        device float *out                   [[buffer(1)]],
        device const float *in              [[buffer(2)]],
        device const float *w               [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.rows * args.channels) {
        return;
    }
    const uint c = i % args.channels;
    const uint t = i / args.channels;
    float acc = 0.0f;
    for (uint j = 0; j < args.taps; ++j) {
        acc += w[c * args.taps + j] * in[(uint64_t) (t + j) * args.channels + c];
    }
    out[i] = acc;
}

/// `out[t][j*C + c] = in[t + j - pad][c]`, zero outside the rows.
kernel void kernel_audio_unfold_f32(
        constant audio_args_unfold & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device const float *in            [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    const uint width = args.taps * args.channels;
    if (i >= args.rows * width) {
        return;
    }
    const uint t = i / width;
    const uint j = (i - t * width) / args.channels;
    const uint c = i % args.channels;
    const int src = (int) (t + j) - (int) args.pad;
    out[i] = (src < 0 || (uint) src >= args.rows) ? 0.0f : in[(uint64_t) src * args.channels + c];
}

/// `out[b][c][r] = in[b][r][c]`.
kernel void kernel_audio_transpose_f32(
        constant audio_args_transpose & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *in               [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    const uint plane = args.rows * args.cols;
    if (i >= args.batch * plane) {
        return;
    }
    const uint b = i / plane;
    const uint r = (i - b * plane) / args.cols;
    const uint c = i % args.cols;
    out[(uint64_t) b * plane + (uint64_t) c * args.rows + r] = in[i];
}
