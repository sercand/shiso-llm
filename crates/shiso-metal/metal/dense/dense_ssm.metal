// The selective state-space block: a depthwise causal convolution over the
// mixed channels and both projections, a diagonal recurrence per head, and the
// gated grouped norm before the output projection.
//
// Every kernel here reads the input projection's rows in place, `in_stride`
// floats apart, with the gate first, the convolved channels next and one step
// size per head last. A copy of any of the three would be a launch and a
// buffer for rows the next launch reads once.
//
// The session's memory is two records: the last `taps - 1` convolution
// inputs, `[kernel - 1, channels]` oldest first, and the state,
// `[heads, head_dim, state]`. Both are f32 and both are rewritten in place by
// the launch that reads them, so a chunk leaves them where the next one starts.

#include <metal_stdlib>
using namespace metal;

/// Widest convolution the window registers hold.
constant constexpr uint DENSE_SSM_MAX_KERNEL = 8u;
/// Widest state a lane carries: `state / 32` values, at most this many.
constant constexpr uint DENSE_SSM_MAX_PER_LANE = 8u;

/// One thread a channel, walking the chunk's tokens in order.
///
/// `out[t, c] = silu(bias[c] + sum_j w[c, j] * window_t[j])`, where the window
/// is the last `taps` inputs ending at token `t`: the session's saved ones
/// first, then the chunk's own. The saved window is rewritten with the last
/// `taps - 1` inputs the chunk saw.
///
/// Buffers: args, proj `[n_tok, in_stride]`, window `[kernel - 1, channels]`,
/// weight `[channels, kernel]`, bias `[channels]`, out `[n_tok, channels]`.
kernel void kernel_dense_ssm_conv_f32(
        constant dense_args_ssm_conv & args [[buffer(0)]],
        device const float * proj [[buffer(1)]],
        device float * window [[buffer(2)]],
        device const float * w [[buffer(3)]],
        device const float * bias [[buffer(4)]],
        device float * out [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    const uint c = gid;
    const uint ch = args.channels;
    const uint k = args.taps;
    if (c >= ch) {
        return;
    }
    float win[DENSE_SSM_MAX_KERNEL];
    for (uint j = 0; j + 1u < k; ++j) {
        win[j] = window[j * ch + c];
    }
    float wc[DENSE_SSM_MAX_KERNEL];
    for (uint j = 0; j < k; ++j) {
        wc[j] = w[c * k + j];
    }
    const float b = bias[c];
    for (uint t = 0; t < args.n_tok; ++t) {
        win[k - 1u] = proj[(uint64_t) t * args.in_stride + args.in_offset + c];
        float acc = 0.0f;
        for (uint j = 0; j < k; ++j) {
            acc += wc[j] * win[j];
        }
        acc += b;
        out[(uint64_t) t * ch + c] = acc / (1.0f + exp(-acc));
        for (uint j = 0; j + 1u < k; ++j) {
            win[j] = win[j + 1u];
        }
    }
    for (uint j = 0; j + 1u < k; ++j) {
        window[j * ch + c] = win[j];
    }
}

/// `log(1 + exp(x))`, in the form that neither overflows nor loses the small
/// end: `max(x, 0) + log1p(exp(-|x|))`.
static inline float dense_softplus(float x) {
    return max(x, 0.0f) + log(1.0f + exp(-fabs(x)));
}

/// The diagonal recurrence, one simdgroup a (head, channel), the chunk's
/// tokens in order.
///
/// Per token, with `dt = softplus(dt_raw + dt_bias[h])` and
/// `dA = exp(-exp(a_log[h]) * dt)`:
///
///   state[n] = dA * state[n] + x * dt * B[g, n]
///   y        = sum_n state[n] * C[g, n] + D[h] * x
///
/// where `g = h / (heads / groups)`: consecutive heads share a group's two
/// projections. A lane holds `state / 32` of the channel's state values in
/// registers across the whole chunk, so the state is read once and written once
/// per launch however many tokens it carries.
///
/// Grid: `(head_dim / 8, heads)`, eight simdgroups a threadgroup.
///
/// Buffers: args, conv out `[n_tok, channels]`, proj `[n_tok, in_stride]`,
/// dt_bias `[heads]`, a_log `[heads]`, D `[heads]`, state
/// `[heads, head_dim, state]`, y `[n_tok, heads * head_dim]`.
kernel void kernel_dense_ssm_scan_f32(
        constant dense_args_ssm_scan & args [[buffer(0)]],
        device const float * xbc [[buffer(1)]],
        device const float * proj [[buffer(2)]],
        device const float * dt_bias [[buffer(3)]],
        device const float * a_log [[buffer(4)]],
        device const float * d [[buffer(5)]],
        device float * st [[buffer(6)]],
        device float * y [[buffer(7)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint lane [[thread_index_in_simdgroup]],
        uint sg [[simdgroup_index_in_threadgroup]],
        uint n_sg [[simdgroups_per_threadgroup]]) {
    const uint h = tgpig.y;
    const uint p = tgpig.x * n_sg + sg;
    if (p >= args.head_dim) {
        return;
    }
    const uint per = args.state / 32u;
    const uint inner = args.heads * args.head_dim;
    const uint g = h / (args.heads / args.groups);
    const uint b_off = inner + g * args.state;
    const uint c_off = inner + (args.groups + g) * args.state;
    device float * srow = st + ((uint64_t) h * args.head_dim + p) * args.state;

    float s[DENSE_SSM_MAX_PER_LANE];
    for (uint i = 0; i < per; ++i) {
        s[i] = srow[lane * per + i];
    }
    const float a = -exp(a_log[h]);
    const float dh = d[h];
    const float bias = dt_bias[h];
    for (uint t = 0; t < args.n_tok; ++t) {
        const float dt = dense_softplus(proj[(uint64_t) t * args.in_stride + args.dt_offset + h] + bias);
        const float da = exp(a * dt);
        device const float * row = xbc + (uint64_t) t * args.channels;
        const float x = row[h * args.head_dim + p];
        float acc = 0.0f;
        for (uint i = 0; i < per; ++i) {
            const uint n = lane * per + i;
            s[i] = da * s[i] + x * dt * row[b_off + n];
            acc += s[i] * row[c_off + n];
        }
        acc = simd_sum(acc);
        if (lane == 0u) {
            y[(uint64_t) t * inner + h * args.head_dim + p] = acc + x * dh;
        }
    }
    for (uint i = 0; i < per; ++i) {
        srow[lane * per + i] = s[i];
    }
}

/// `y = w * rms_norm(silu(z) * y)` over groups of `group_size` channels, in
/// place, with no weight inside the norm: the weight multiplies after it.
///
/// Grid: `(inner / group_size, n_tok)`; the group is staged in threadgroup
/// memory, `group_size` floats after the reduction's slots.
///
/// Buffers: args, y `[n_tok, inner]`, proj `[n_tok, in_stride]` (the gate is
/// its first `inner` floats), w `[inner]`.
kernel void kernel_dense_ssm_gate_norm_f32(
        constant dense_args_ssm_norm & args [[buffer(0)]],
        device float * y [[buffer(1)]],
        device const float * proj [[buffer(2)]],
        device const float * w [[buffer(3)]],
        threadgroup float * tbuf [[threadgroup(0)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const uint grp = tgpig.x;
    const uint t = tgpig.y;
    const uint tid = tpitg.x;
    const uint nt = ntg.x;
    const uint gs = args.group_size;
    threadgroup float * scratch = tbuf;
    threadgroup float * row = tbuf + DENSE_BLOCK_SUM_SLOTS;
    device float * yr = y + (uint64_t) t * args.inner + grp * gs;
    device const float * zr = proj + (uint64_t) t * args.in_stride + grp * gs;
    float sum = 0.0f;
    for (uint i = tid; i < gs; i += nt) {
        const float z = zr[i];
        const float v = z / (1.0f + exp(-z)) * yr[i];
        row[i] = v;
        sum += v * v;
    }
    const float inv = rsqrt(dense_block_sum(sum, scratch, tid, nt) / (float) gs + args.eps);
    for (uint i = tid; i < gs; i += nt) {
        yr[i] = w[grp * gs + i] * (row[i] * inv);
    }
}
