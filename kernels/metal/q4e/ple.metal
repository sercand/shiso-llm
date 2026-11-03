// q4e ple entries: Metal port of kernels/cuda/q4e/ple.cu (see q4e_common.h for conventions).
//
// The n-gram embedding block's gate and its dilated convolution. Ported:
// q4e_ple_gated_value_kernel, q4e_ple_conv_kernel. Not ported: the two row
// decoders (q4e_ple_dequant_kernel, q4e_ple_dequant_exl3_kernel), whose row
// format the Metal path replaces, and q4e_ple_conv_batch_kernel, which reads a
// device array of per-row window pointers the Metal binding cannot pack yet.

/// `q4e_ple_gated_value_kernel` (ple.cu:84): per (token, stream) threadgroup,
/// `gate = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))` with
/// `s = <key[t][h], query[t][h]> / sqrt(n_embd)`, then the token's value times
/// the gate into stream `h`. `grid.x` is the token and `grid.y` the stream.
///
/// Buffers: 0 `gv` (`[t][n_hc][n_embd]`), 1 `key`, 2 `query` (both
/// `[t][n_hc][n_embd]`), 3 `value` (`[t][n_embd]`), 4 `n_embd`, 5 `n_hc`.
///
/// The signed square root before the sigmoid is easy to miss and changes the
/// gate's shape completely.
kernel void q4e_ple_gated_value_kernel(device float *gv [[buffer(0)]],
                                       device const float *key [[buffer(1)]],
                                       device const float *query [[buffer(2)]],
                                       device const float *value [[buffer(3)]],
                                       constant uint &n_embd [[buffer(4)]],
                                       constant uint &n_hc [[buffer(5)]],
                                       uint3 bid [[threadgroup_position_in_grid]],
                                       uint3 tid3 [[thread_position_in_threadgroup]],
                                       uint3 bdim [[threads_per_threadgroup]]) {
    const uint tid = tid3.x;
    const uint t = bid.x;
    const uint h = bid.y;
    const ulong base = (ulong(t) * n_hc + h) * n_embd;
    threadgroup float s_red[Q4E_SLOTS];

    float dot = 0.0f;
    for (uint e = tid; e < n_embd; e += bdim.x) {
        dot += key[base + e] * query[base + e];
    }
    dot = q4e_block_sum(dot, s_red, tid, bdim.x) * rsqrt(float(n_embd));
    const float mag = sqrt(fmax(fabs(dot), 1e-6f));
    const float g = 1.0f / (1.0f + exp(-copysign(mag, dot)));

    device const float *v = value + ulong(t) * n_embd;
    for (uint e = tid; e < n_embd; e += bdim.x) gv[base + e] = v[e] * g;
}

/// `q4e_ple_conv_kernel` (ple.cu:107): depthwise causal convolution dilated by
/// the n-gram size (taps at t, t-3, t-6, t-9 for the shipped kernel of 4),
/// silu on the result, one thread per channel walking the chunk's tokens. The
/// `(kernel - 1) * dilation` history columns are carried per sequence in
/// `state`, so a chunked prefill matches a single-shot one.
///
/// Buffers: 0 `out`, 1 `state` (`[hist][channels]`), 2 `x` (`[n_tok][channels]`),
/// 3 `w` (`[channels][kernel]`), 4 `channels`, 5 `kernel`, 6 `dilation`,
/// 7 `n_tok`, 8 `ckpt` (`[ckpt_cap][hist][channels]`), 9 `ckpt_cap`.
///
/// The host binds `out` and `x` to the same buffer. Each element is read and
/// then written by the one thread that owns its channel, so the alias is
/// safe, and `x` is therefore not declared `const`.
///
/// `ckpt` is null on the host whenever `ckpt_cap` is zero, and on Metal a null
/// arrives as an 8-byte zero bound with setBytes rather than a null pointer, so
/// the checkpoint write is gated on `ckpt_cap` alone (`t < ckpt_cap` is false
/// for every `t` at zero); the pointer test CUDA also makes is dropped.
///
/// History is capped at 16 columns, as in CUDA; the shipped geometry uses 9.
kernel void q4e_ple_conv_kernel(device float *out [[buffer(0)]],
                                device float *state [[buffer(1)]],
                                device float *x [[buffer(2)]],
                                device const float *w [[buffer(3)]],
                                constant uint &channels [[buffer(4)]],
                                constant uint &kernel_w [[buffer(5)]],
                                constant uint &dilation [[buffer(6)]],
                                constant uint &n_tok [[buffer(7)]],
                                device float *ckpt [[buffer(8)]],
                                constant uint &ckpt_cap [[buffer(9)]],
                                uint3 bid [[threadgroup_position_in_grid]],
                                uint3 tid3 [[thread_position_in_threadgroup]],
                                uint3 bdim [[threads_per_threadgroup]]) {
    const uint c = bid.x * bdim.x + tid3.x;
    if (c >= channels) return;
    const uint hist = (kernel_w - 1u) * dilation;

    float win[16];
    for (uint i = 0; i < hist; i++) win[i] = state[ulong(i) * channels + c];

    device const float *wc = w + ulong(c) * kernel_w;
    for (uint t = 0; t < n_tok; t++) {
        const float cur = x[ulong(t) * channels + c];
        float acc = wc[kernel_w - 1u] * cur;
        for (uint k = 0; k + 1u < kernel_w; k++) {
            // tap k reads (kernel - 1 - k) * dilation positions back
            const uint back = (kernel_w - 1u - k) * dilation;
            acc += wc[k] * win[hist - back];
        }
        out[ulong(t) * channels + c] = acc / (1.0f + exp(-acc));
        for (uint i = 0; i + 1u < hist; i++) win[i] = win[i + 1];
        if (hist > 0) win[hist - 1] = cur;
        if (t < ckpt_cap && t + 1u < n_tok) {
            device float *ck = ckpt + ulong(t) * hist * channels;
            for (uint i = 0; i < hist; i++) ck[ulong(i) * channels + c] = win[i];
        }
    }

    for (uint i = 0; i < hist; i++) state[ulong(i) * channels + c] = win[i];
}
