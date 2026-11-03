// The three element-wise stages of the routed feed-forward: the activation,
// the combine and the shared-expert add.
//
// Ported from `kernels/cuda/q4e/moe.cu`, whose bodies are fidelity-gated and
// must not be edited. The same host code (`crates/shiso-engine/src/moe.rs`)
// launches these and the CUDA ones by the same entry names, and one host
// reference grades both, so the arithmetic here is the CUDA arithmetic and not
// an equal-looking rewrite of it.
//
// `q35moe_common.h` is concatenated ahead of this file by `SOURCES` in
// `src/compile.rs`, so there is nothing to include: the library is one source
// string and a quote-include of a sibling would not resolve.
//
// No `#pragma METAL fp math_mode` here. Fast math is the library's build
// default and this file holds at it.

/// `out = silu(gate) * up` over `n` elements.
///
/// `kernels/cuda/q4e/moe.cu:83`. The reference divides rather than calling a
/// sigmoid, and the division is kept: `g / (1 + exp(-g))` and `g * sigmoid(g)`
/// do not round alike.
///
/// Buffers: out, gate, up, n.
kernel void q4e_swiglu_kernel(
        device float *out [[buffer(0)]],
        device const float *gate [[buffer(1)]],
        device const float *up [[buffer(2)]],
        constant ulong &n [[buffer(3)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]]) {
    const ulong i = ulong(gid) * tpg + tid;
    if (i >= n) {
        return;
    }
    const float g = gate[i];
    out[i] = (g / (1.0f + precise::exp(-g))) * up[i];
}

/// The weighted sum of a token's `n_used` columns.
///
/// `kernels/cuda/q4e/moe.cu:93`. Column `t * n_used + s` is slot `s` of token
/// `t`, which is the third of the four seam conventions: the expert matmul
/// writes column-major over (token, slot) and this is the reader that pins it.
///
/// Buffers: out, down, weights, n_embd, n_used.
kernel void q4e_moe_combine_kernel(
        device float *out [[buffer(0)]],
        device const float *down [[buffer(1)]],
        device const float *weights [[buffer(2)]],
        constant uint &n_embd [[buffer(3)]],
        constant uint &n_used [[buffer(4)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 grid [[threadgroups_per_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint t = group.y;
    const uint tid = tpitg.x;
    const uint tpg = ntg.x;
    for (uint e = group.x * tpg + tid; e < n_embd; e += grid.x * tpg) {
        float acc = 0.0f;
        for (uint s = 0; s < n_used; ++s) {
            const ulong col = ulong(t) * n_used + s;
            acc += down[col * n_embd + e] * weights[col];
        }
        out[ulong(t) * n_embd + e] = acc;
    }
}

/// `out += shared * sigmoid(logit[t])`, one scalar gate per token.
///
/// `kernels/cuda/q4e/moe.cu:110`. The shared expert carries a single scalar
/// gate per token, not a per-channel one, and the store is an add rather than a
/// write because the routed experts have already written this row.
///
/// Buffers: out, shared, logit, n_embd.
kernel void q4e_shared_add_kernel(
        device float *out [[buffer(0)]],
        device const float *shared [[buffer(1)]],
        device const float *logit [[buffer(2)]],
        constant uint &n_embd [[buffer(3)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint3 grid [[threadgroups_per_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint t = group.y;
    const uint tid = tpitg.x;
    const uint tpg = ntg.x;
    const float g = 1.0f / (1.0f + precise::exp(-logit[t]));
    for (uint e = group.x * tpg + tid; e < n_embd; e += grid.x * tpg) {
        out[ulong(t) * n_embd + e] += shared[ulong(t) * n_embd + e] * g;
    }
}
