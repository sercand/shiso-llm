// Normalizations over f32 rows with f32 parameters.

// The layer norm is compiled with exact floating point: its precision rests on
// forming each deviation as (x - pivot) - dmean, and fast math is free to
// reassociate that into x - (pivot + dmean), which rounds the mean back to the
// row's magnitude and undoes the whole point.
#pragma METAL fp math_mode(safe)
/// `out = (in - mean) * rsqrt(var + eps) * w (+ b)`, two-pass so a row with a
/// large mean and a small variance keeps its precision. Safe in place.
kernel void kernel_audio_layer_norm_f32(
        constant audio_args_norm & args [[buffer(0)]],
        device float *out               [[buffer(1)]],
        device const float *in          [[buffer(2)]],
        device const float *weight      [[buffer(3)]],
        device const float *bias        [[buffer(4)]],
        threadgroup float *scratch      [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint r = tgpig.x;
    if (r >= args.rows) {
        return;
    }
    const uint nt = ntg.x;
    device const float *src = in + (uint64_t) r * args.dim;
    device float *dst = out + (uint64_t) r * args.dim;
    // Everything is taken about the row's first value, never about zero. A row
    // near 1e4 has an f32 ulp of 1e-3 there, so a mean *stored* at that
    // magnitude is already off by a tenth of a deviation of order 1e-2, and
    // no summation order can recover it. `x - pivot` is exact for values that
    // close to each other, and the shifted mean `dmean` stays deviation-sized.
    const float pivot = src[0];
    float sum = 0.0f;
    for (uint i = tpitg.x; i < args.dim; i += nt) {
        sum += src[i] - pivot;
    }
    const float dmean = audio_block_sum(sum, scratch, tpitg.x, nt) / (float) args.dim;
    float var = 0.0f;
    for (uint i = tpitg.x; i < args.dim; i += nt) {
        const float d = (src[i] - pivot) - dmean;
        var += d * d;
    }
    const float rstd = rsqrt(audio_block_sum(var, scratch, tpitg.x, nt) / (float) args.dim + args.eps);
    const bool has_bias = (args.flags & AUDIO_NORM_HAS_BIAS) != 0u;
    for (uint i = tpitg.x; i < args.dim; i += nt) {
        const float y = ((src[i] - pivot) - dmean) * rstd * weight[i];
        dst[i] = has_bias ? y + bias[i] : y;
    }
}

#pragma METAL fp math_mode(fast)

/// `out = in * rsqrt(mean(in^2) + eps) (* w)`. Safe in place.
kernel void kernel_audio_rms_norm_f32(
        constant audio_args_norm & args [[buffer(0)]],
        device float *out               [[buffer(1)]],
        device const float *in          [[buffer(2)]],
        device const float *weight      [[buffer(3)]],
        threadgroup float *scratch      [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint r = tgpig.x;
    if (r >= args.rows) {
        return;
    }
    const uint nt = ntg.x;
    device const float *src = in + (uint64_t) r * args.dim;
    device float *dst = out + (uint64_t) r * args.dim;
    float sq = 0.0f;
    for (uint i = tpitg.x; i < args.dim; i += nt) {
        sq += src[i] * src[i];
    }
    const float rms = rsqrt(audio_block_sum(sq, scratch, tpitg.x, nt) / (float) args.dim + args.eps);
    const bool has_weight = (args.flags & AUDIO_NORM_HAS_WEIGHT) != 0u;
    for (uint i = tpitg.x; i < args.dim; i += nt) {
        const float y = src[i] * rms;
        dst[i] = has_weight ? y * weight[i] : y;
    }
}

/// `x += add`, then `out = x * rsqrt(mean(x^2) + eps) (* w)`: a residual add
/// and the RMS norm after it in one pass, each sum and the reduction formed
/// exactly as the axpy (at alpha 1) and the RMS norm above form them. `out`
/// may be `add` (each thread reads and writes only its own elements); it may
/// not be `x`.
kernel void kernel_audio_add_rms_norm_f32(
        constant audio_args_norm & args [[buffer(0)]],
        device float *out               [[buffer(1)]],
        device float *x                 [[buffer(2)]],
        device const float *add         [[buffer(3)]],
        device const float *weight      [[buffer(4)]],
        threadgroup float *scratch      [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint r = tgpig.x;
    if (r >= args.rows) {
        return;
    }
    const uint nt = ntg.x;
    device float *src = x + (uint64_t) r * args.dim;
    device const float *a = add + (uint64_t) r * args.dim;
    device float *dst = out + (uint64_t) r * args.dim;
    // A thread's values are loaded before any is stored, eight at a time:
    // the stores may alias the loads as far as the compiler knows, and a
    // row of 2048 read one dependent round trip at a time took 11 us.
    const uint R = AUDIO_ADD_RMS_RUN;
    const bool has_weight = (args.flags & AUDIO_NORM_HAS_WEIGHT) != 0u;
    float sq = 0.0f;
    for (uint base = tpitg.x; base < args.dim; base += R * nt) {
        float v[AUDIO_ADD_RMS_RUN];
        for (uint j = 0; j < R; ++j) {
            const uint i = base + j * nt;
            v[j] = i < args.dim ? src[i] + a[i] : 0.0f;
        }
        for (uint j = 0; j < R; ++j) {
            const uint i = base + j * nt;
            if (i < args.dim) {
                src[i] = v[j];
                sq += v[j] * v[j];
            }
        }
    }
    const float rms = rsqrt(audio_block_sum(sq, scratch, tpitg.x, nt) / (float) args.dim + args.eps);
    for (uint base = tpitg.x; base < args.dim; base += R * nt) {
        float v[AUDIO_ADD_RMS_RUN];
        float g[AUDIO_ADD_RMS_RUN];
        for (uint j = 0; j < R; ++j) {
            const uint i = base + j * nt;
            v[j] = i < args.dim ? src[i] : 0.0f;
            g[j] = i < args.dim && has_weight ? weight[i] : 1.0f;
        }
        for (uint j = 0; j < R; ++j) {
            const uint i = base + j * nt;
            if (i < args.dim) {
                const float y = v[j] * rms;
                dst[i] = has_weight ? y * g[j] : y;
            }
        }
    }
}
