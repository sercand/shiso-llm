// GDN input mixing: causal depthwise conv + silu, the two L2 norms, the decay
// and beta gates, and the output norm-and-gate.
//
// Ported from the CUDA q4e_gdn_conv_kernel / q4e_gdn_l2norm_kernel /
// q4e_gdn_gates_kernel / q4e_gdn_out_gate_kernel.  Every summation below walks
// the same lanes in the same order as the source; the shapes are the contract.

/// Causal depthwise conv over the q|k|v channels plus silu, one thread per
/// channel.  Handles decode (n_tok 1) and chunked prefill: the window carries
/// the tail of the previous call, so a chunked prefill matches a single-shot one.
kernel void kernel_q35_gdn_conv(
        constant q35_args_gdn_conv & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device float *state               [[buffer(2)]],
        device const float *x             [[buffer(3)]],
        device const float *w             [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint c = tgpig.x * 256u + tpitg.x;
    if (c >= args.channels) {
        return;
    }
    const uint hist = args.conv_kernel - 1u;

    // This channel's history, oldest first.
    float win[8];
    for (uint i = 0; i < hist; i++) {
        win[i] = state[(uint64_t) i * args.channels + c];
    }

    device const float *wc = w + (uint64_t) c * args.conv_kernel;
    for (uint t = 0; t < args.n_tok; t++) {
        const float cur = x[(uint64_t) t * args.channels + c];
        float acc = 0.0f;
        for (uint k = 0; k < hist; k++) {
            acc += wc[k] * win[k];
        }
        acc += wc[hist] * cur;
        out[(uint64_t) t * args.channels + c] = q35_silu(acc);
        for (uint k = 0; k + 1u < hist; k++) {
            win[k] = win[k + 1u];
        }
        if (hist > 0u) {
            win[hist - 1u] = cur;
        }
    }

    for (uint i = 0; i < hist; i++) {
        state[(uint64_t) i * args.channels + c] = win[i];
    }
}

/// `kernel_q35_gdn_conv`, also writing the window after every row but the last
/// into a trail of `n_tok - 1` window records: a verify pass's per-row windows
/// in one launch. The arithmetic is the plain kernel's, row for row.
kernel void kernel_q35_gdn_conv_trail(
        constant q35_args_gdn_conv & args [[buffer(0)]],
        device float *out                 [[buffer(1)]],
        device float *state               [[buffer(2)]],
        device const float *x             [[buffer(3)]],
        device const float *w             [[buffer(4)]],
        device float *trail               [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint c = tgpig.x * 256u + tpitg.x;
    if (c >= args.channels) {
        return;
    }
    const uint hist = args.conv_kernel - 1u;

    // This channel's history, oldest first.
    float win[8];
    for (uint i = 0; i < hist; i++) {
        win[i] = state[(uint64_t) i * args.channels + c];
    }

    device const float *wc = w + (uint64_t) c * args.conv_kernel;
    for (uint t = 0; t < args.n_tok; t++) {
        const float cur = x[(uint64_t) t * args.channels + c];
        float acc = 0.0f;
        for (uint k = 0; k < hist; k++) {
            acc += wc[k] * win[k];
        }
        acc += wc[hist] * cur;
        out[(uint64_t) t * args.channels + c] = q35_silu(acc);
        for (uint k = 0; k + 1u < hist; k++) {
            win[k] = win[k + 1u];
        }
        if (hist > 0u) {
            win[hist - 1u] = cur;
        }
        // The window after this row, into its record - every row but the last,
        // whose record is the window the launch leaves in `state`.
        if (t + 1u < args.n_tok) {
            device float *rec = trail + (uint64_t) t * hist * args.channels;
            for (uint i = 0; i < hist; i++) {
                rec[(uint64_t) i * args.channels + c] = win[i];
            }
        }
    }

    for (uint i = 0; i < hist; i++) {
        state[(uint64_t) i * args.channels + c] = win[i];
    }
}

/// In-place L2 norm of one head of one token.  Not an RMS norm: no weight, no
/// mean, and the epsilon is added to the raw sum of squares.
kernel void kernel_q35_gdn_l2norm(
        constant q35_args_gdn_l2norm & args [[buffer(0)]],
        device float *qk               [[buffer(1)]],
        threadgroup float *scratch     [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    device float *v = qk + (uint64_t) t * args.stride + args.offset
            + (uint64_t) h * args.head_dim;

    float sum = 0.0f;
    for (uint i = tpitg.x; i < args.head_dim; i += 128u) {
        sum += v[i] * v[i];
    }
    const float total = q35_block_sum(sum, scratch, tpitg.x, 128u);
    const float inv = rsqrt(total + 1e-12f);
    for (uint i = tpitg.x; i < args.head_dim; i += 128u) {
        v[i] *= inv;
    }
}

/// decay = softplus(alpha_proj + dt_bias) * ssm_a, beta = sigmoid(beta_proj).
/// ssm_a already holds -exp(A_log), so decay is negative and both consumers
/// exponentiate it directly.
kernel void kernel_q35_gdn_gates(
        constant q35_args_gdn_gates & args [[buffer(0)]],
        device float *decay            [[buffer(1)]],
        device float *beta             [[buffer(2)]],
        device const float *alpha_proj   [[buffer(3)]],
        device const float *beta_proj  [[buffer(4)]],
        device const float *dt_bias    [[buffer(5)]],
        device const float *a          [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint64_t i = (uint64_t) tgpig.x * 256u + tpitg.x;
    if (i >= (uint64_t) args.n_head * args.n_tok) {
        return;
    }
    const uint h = i % args.n_head;
    const float x = alpha_proj[i] + dt_bias[h];
    decay[i] = q35_softplus(x) * a[h];
    beta[i] = q35_sigmoid(beta_proj[i]);
}

/// Per-head RMSNorm of the delta-rule output, then the output gate.
/// The norm weight is applied raw: it is the one weight the converter does not
/// offset by one.
///
/// The gate is `silu`, not `sigmoid`. The two differ by a factor of the gate's own
/// value, which is enough to turn the model's output into punctuation: with the
/// plain sigmoid this checkpoint answers a factual prompt in fragments, and with
/// `silu` it answers in English. The C reference this family was read from applies
/// a sigmoid, but it describes a different member of the family - one with a
/// different value-head count - and it is not this checkpoint's gate.
kernel void kernel_q35_gdn_out_gate(
        constant q35_args_gdn_out_gate & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *attn         [[buffer(2)]],
        device const float *z            [[buffer(3)]],
        device const float *w            [[buffer(4)]],
        threadgroup float *scratch       [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint group = tgpig.x;
    const uint base = group * args.head_dim;

    float sum = 0.0f;
    for (uint i = tpitg.x; i < args.head_dim; i += 128u) {
        const float v = attn[base + i];
        sum += v * v;
    }
    const float scale = rsqrt(q35_block_sum(sum, scratch, tpitg.x, 128u)
            / (float) args.head_dim + args.eps);
    for (uint i = tpitg.x; i < args.head_dim; i += 128u) {
        out[base + i] = attn[base + i] * scale * w[i] * q35_silu(z[base + i]);
    }
}
