// The gated short convolution: `out = c * conv(b * x)`, a depthwise causal
// convolution of a few taps between two elementwise gates, with no bias and no
// activation.
//
// The input projection's rows are read in place, `3 * channels` floats wide
// with the two gates first and the convolved input last (`[b | c | x]`): a
// copy of any third would be a launch and a buffer for rows this launch reads
// once.
//
// The session's memory is one record, the last `taps - 1` gated inputs
// `b * x`, `[taps - 1, channels]` oldest first, f32. It is rewritten in place
// by the launch that reads it, so a chunk leaves it where the next one starts.
// The gate is applied before the window is saved because the convolution reads
// the gated product, never `x` alone: saving `x` would make the next chunk
// re-gate old inputs with nothing to gate them by.

#include <metal_stdlib>
using namespace metal;

/// Widest convolution the window registers hold.
constant constexpr uint DENSE_SHORT_CONV_MAX_TAPS = 8u;

/// One thread a channel, walking the chunk's tokens in order.
///
/// `out[t, ch] = c[t, ch] * sum_j w[ch, j] * window_t[j]`, where the window is
/// the last `taps` gated inputs `b * x` ending at token `t`: the session's
/// saved ones first, then the chunk's own.
///
/// Buffers: args, proj `[n_tok, 3 * channels]`, window `[taps - 1, channels]`,
/// weight `[channels, taps]`, out `[n_tok, channels]`.
kernel void kernel_dense_short_conv_f32(
        constant dense_args_short_conv & args [[buffer(0)]],
        device const float * proj [[buffer(1)]],
        device float * window [[buffer(2)]],
        device const float * w [[buffer(3)]],
        device float * out [[buffer(4)]],
        uint gid [[thread_position_in_grid]]) {
    const uint ch = gid;
    const uint channels = args.channels;
    const uint k = args.taps;
    if (ch >= channels) {
        return;
    }
    const uint64_t stride = 3ull * channels;
    float win[DENSE_SHORT_CONV_MAX_TAPS];
    for (uint j = 0; j + 1u < k; ++j) {
        win[j] = window[j * channels + ch];
    }
    float wc[DENSE_SHORT_CONV_MAX_TAPS];
    for (uint j = 0; j < k; ++j) {
        wc[j] = w[ch * k + j];
    }
    for (uint t = 0; t < args.n_tok; ++t) {
        device const float * row = proj + (uint64_t) t * stride;
        win[k - 1u] = row[ch] * row[2u * channels + ch];
        float acc = 0.0f;
        for (uint j = 0; j < k; ++j) {
            acc += wc[j] * win[j];
        }
        out[(uint64_t) t * channels + ch] = row[channels + ch] * acc;
        for (uint j = 0; j + 1u < k; ++j) {
            win[j] = win[j + 1u];
        }
    }
    for (uint j = 0; j + 1u < k; ++j) {
        window[j * channels + ch] = win[j];
    }
}
