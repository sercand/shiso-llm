// q4e hc entries: Metal port of kernels/cuda/q4e/hc.cu (see q4e_common.h for conventions).
//
// The hyper-connection streams: the residual broadcast that opens a pass, the
// per-stream group RMSNorm, the scale-and-activate between the mix's two
// projections, the collapse into the block input, and the fused residual
// update + next-block norm that closes a block. Every expression and its
// order is the CUDA one; the launch geometry is the engine's
// (`crates/shiso-engine/src/sparse/kernels.rs`: `hc_init`, `hc_norm`,
// `hc_collapse`, `hc_combine`, `elementwise`), all 256 threads wide.
//
// NULL POINTER (q4e_hc_combine_norm_kernel `w`): the CUDA body's only guard is
// the pointer test itself. On Metal the engine's `Operand::Null` arrives as
// eight zero bytes bound with setBytes, i.e. a valid address whose first two
// floats read as +0.0. There is no companion count or flag in the argument
// list, so this port treats "the first two words are all-zero bits" as NULL
// (see `q4e_hc_null_arg`). A real norm weight whose first two elements are
// both exactly +0.0 would be misread as absent; the host should replace this
// with an explicit flag (reported with the port).

/// Values one thread of the combine carries: ceil(n_embd / threads), 10 at
/// the shipped 2560 with 256 threads. Mirrors `Q4E_HC_FUSE_MAX` in hc.cu and
/// `kernels::HC_FUSE_MAX` on the host.
#define Q4E_HC_FUSE_MAX 16u

/// Whether a pointer argument is the engine's `Operand::Null`: eight zero bytes
/// bound by value. Reads exactly those eight bytes, which is in bounds for
/// both a Null binding and any weight of two or more floats.
static inline bool q4e_hc_null_arg(device const float *p) {
    device const uint *bits = (device const uint *)p;
    return (bits[0] | bits[1]) == 0u;
}

/// hc.cu:22 `q4e_hc_init_kernel`: `res[t][h][e] = embed[t][e]` for every
/// stream `h`. Buffers: 0 `res` (f32, n_tok*n_hc*n_embd), 1 `embed` (f32,
/// n_tok*n_embd), 2 `n_embd`, 3 `n_hc`, 4 `n_tok`. Flat, one element a thread.
kernel void q4e_hc_init_kernel(
        device float *res [[buffer(0)]],
        device const float *embed [[buffer(1)]],
        constant uint &n_embd [[buffer(2)]],
        constant uint &n_hc [[buffer(3)]],
        constant uint &n_tok [[buffer(4)]],
        uint bid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const ulong idx = (ulong)bid * bdim + tid;
    const ulong total = (ulong)n_tok * n_hc * n_embd;
    if (idx >= total) return;
    const uint e = (uint)(idx % n_embd);
    const ulong t = idx / ((ulong)n_hc * n_embd);
    res[idx] = embed[t * n_embd + e];
}

/// hc.cu:34 `q4e_hc_init_add_kernel`: the broadcast plus a per-stream term
/// (the draft head's input). Buffers: 0 `res`, 1 `embed` (per token), 2 `h`
/// (per stream, same shape as `res`), 3 `n_embd`, 4 `n_hc`, 5 `n_tok`.
kernel void q4e_hc_init_add_kernel(
        device float *res [[buffer(0)]],
        device const float *embed [[buffer(1)]],
        device const float *h [[buffer(2)]],
        constant uint &n_embd [[buffer(3)]],
        constant uint &n_hc [[buffer(4)]],
        constant uint &n_tok [[buffer(5)]],
        uint bid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const ulong idx = (ulong)bid * bdim + tid;
    const ulong total = (ulong)n_tok * n_hc * n_embd;
    if (idx >= total) return;
    const uint e = (uint)(idx % n_embd);
    const ulong t = idx / ((ulong)n_hc * n_embd);
    res[idx] = embed[t * n_embd + e] + h[idx];
}

/// hc.cu:50 `q4e_hc_norm_kernel<OUT>`: group RMSNorm, one threadgroup per
/// (token, stream) pair; the mean square is over one `n_embd` stream while the
/// weight spans `n_hc * n_embd`. Buffers: 0 `out` (OUT), 1 `res` (f32), 2 `w`
/// (f32, n_hc*n_embd), 3 `n_embd`, 4 `n_hc`, 5 `eps`.
template <typename OUT>
kernel void q4e_hc_norm_kernel(
        device OUT *out [[buffer(0)]],
        device const float *res [[buffer(1)]],
        device const float *w [[buffer(2)]],
        constant uint &n_embd [[buffer(3)]],
        constant uint &n_hc [[buffer(4)]],
        constant float &eps [[buffer(5)]],
        uint group [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    threadgroup float scratch[Q4E_SLOTS];
    const uint h = group % n_hc;
    device const float *src = res + (ulong)group * n_embd;
    device OUT *dst = out + (ulong)group * n_embd;
    device const float *wh = w + (ulong)h * n_embd;

    float sum = 0.0f;
    for (uint i = tid; i < n_embd; i += bdim) {
        const float v = src[i];
        sum += v * v;
    }
    sum = q4e_block_sum(sum, scratch, tid, bdim);
    const float scale = rsqrt(sum / (float)n_embd + eps);
    for (uint i = tid; i < n_embd; i += bdim) {
        q4e_store(dst + i, src[i] * scale * wh[i]);
    }
}

template [[host_name("_Z18q4e_hc_norm_kernelIfEv")]] kernel void q4e_hc_norm_kernel<float>(
        device float *, device const float *, device const float *,
        constant uint &, constant uint &, constant float &, uint, uint, uint);
template [[host_name("_Z18q4e_hc_norm_kernelI6__halfEv")]] kernel void q4e_hc_norm_kernel<half>(
        device half *, device const float *, device const float *,
        constant uint &, constant uint &, constant float &, uint, uint, uint);

/// hc.cu:73 `q4e_scale_silu_kernel`: `x = silu(x * inv_hc)` in place.
/// Buffers: 0 `x` (f32), 1 `inv_hc` (f32), 2 `n` (u64).
kernel void q4e_scale_silu_kernel(
        device float *x [[buffer(0)]],
        constant float &inv_hc [[buffer(1)]],
        constant ulong &n [[buffer(2)]],
        uint bid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const ulong i = (ulong)bid * bdim + tid;
    if (i >= n) return;
    const float v = x[i] * inv_hc;
    x[i] = v / (1.0f + exp(-v));
}

/// hc.cu:82 `q4e_hc_collapse_kernel<XT>`: the mean over streams of
/// `xn * sigmoid(up)`, one threadgroup per token. Buffers: 0 `out` (f32,
/// n_tok*n_embd), 1 `xn` (XT), 2 `up` (XT), 3 `n_embd`, 4 `n_hc`.
template <typename XT>
kernel void q4e_hc_collapse_kernel(
        device float *out [[buffer(0)]],
        device const XT *xn [[buffer(1)]],
        device const XT *up [[buffer(2)]],
        constant uint &n_embd [[buffer(3)]],
        constant uint &n_hc [[buffer(4)]],
        uint t [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const float inv = 1.0f / (float)n_hc;
    device const XT *xn_t = xn + (ulong)t * n_hc * n_embd;
    device const XT *up_t = up + (ulong)t * n_hc * n_embd;
    device float *out_t = out + (ulong)t * n_embd;
    for (uint e = tid; e < n_embd; e += bdim) {
        float acc = 0.0f;
        for (uint h = 0; h < n_hc; h++) {
            const uint o = h * n_embd + e;
            const float g = 1.0f / (1.0f + exp(-q4e_load(up_t + o)));
            acc += q4e_load(xn_t + o) * g;
        }
        out_t[e] = acc * inv;
    }
}

template [[host_name("_Z22q4e_hc_collapse_kernelIfEv")]] kernel void q4e_hc_collapse_kernel<float>(
        device float *, device const float *, device const float *,
        constant uint &, constant uint &, uint, uint, uint);
template [[host_name("_Z22q4e_hc_collapse_kernelI6__halfEv")]] kernel void q4e_hc_collapse_kernel<half>(
        device float *, device const half *, device const half *,
        constant uint &, constant uint &, uint, uint, uint);

/// hc.cu:105 `q4e_hc_collapse_f16_dual_kernel`: the half-stream collapse that
/// also writes the block input as halves (round to nearest even). Buffers: 0
/// `out` (f32), 1 `out_h` (f16), 2 `xn` (f16), 3 `up` (f16), 4 `n_embd`, 5
/// `n_hc`.
kernel void q4e_hc_collapse_f16_dual_kernel(
        device float *out [[buffer(0)]],
        device half *out_h [[buffer(1)]],
        device const half *xn [[buffer(2)]],
        device const half *up [[buffer(3)]],
        constant uint &n_embd [[buffer(4)]],
        constant uint &n_hc [[buffer(5)]],
        uint t [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const float inv = 1.0f / (float)n_hc;
    device const half *xn_t = xn + (ulong)t * n_hc * n_embd;
    device const half *up_t = up + (ulong)t * n_hc * n_embd;
    device float *out_t = out + (ulong)t * n_embd;
    device half *out_h_t = out_h + (ulong)t * n_embd;
    for (uint e = tid; e < n_embd; e += bdim) {
        float acc = 0.0f;
        for (uint h = 0; h < n_hc; h++) {
            const uint o = h * n_embd + e;
            const float g = 1.0f / (1.0f + exp(-(float)up_t[o]));
            acc += (float)xn_t[o] * g;
        }
        out_t[e] = acc * inv;
        out_h_t[e] = half(acc * inv);
    }
}

/// hc.cu:153 `q4e_hc_combine_norm_kernel<XN>`: the residual update that closes
/// a block (`res += block_out * 2*sigmoid(inject / n_hc)` per stream), fused
/// with the next mix's group RMSNorm into `xn`. Grid `(n_tok, y)`: the streams
/// are walked from `blockIdx.y` in steps of `gridDim.y` (the host picks
/// `y = n_hc` below 256 tokens, 1 above). Buffers: 0 `res` (f32, in/out), 1
/// `xn` (XN), 2 `block_out` (f32, n_tok*n_embd), 3 `inject` (f32,
/// n_tok*n_hc), 4 `w` (f32, n_hc*n_embd, or the engine's Null: then only the
/// update runs and `xn` is untouched), 5 `n_embd` (<= 256 * 16), 6 `n_hc`,
/// 7 `eps`.
///
/// The three per-thread arrays stay in registers under the fixed, unrolled
/// bound exactly as in CUDA; no threadgroup memory beyond the reduction
/// scratch.
template <typename XN>
kernel void q4e_hc_combine_norm_kernel(
        device float *res [[buffer(0)]],
        device XN *xn [[buffer(1)]],
        device const float *block_out [[buffer(2)]],
        device const float *inject [[buffer(3)]],
        device const float *w [[buffer(4)]],
        constant uint &n_embd [[buffer(5)]],
        constant uint &n_hc [[buffer(6)]],
        constant float &eps [[buffer(7)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim3 [[threads_per_threadgroup]],
        uint3 grid3 [[threadgroups_per_grid]]) {
    threadgroup float scratch[Q4E_SLOTS];
    const uint tid = tid3.x;
    const uint bdim = bdim3.x;
    const uint t = bid.x;
    device const float *src = block_out + (ulong)t * n_embd;
    const uint h0 = bid.y;
    const uint hstep = grid3.y;
    const bool has_w = !q4e_hc_null_arg(w);

    float sv[Q4E_HC_FUSE_MAX];
#pragma unroll
    for (uint n = 0; n < Q4E_HC_FUSE_MAX; n++) {
        const uint i = tid + n * bdim;
        sv[n] = (i < n_embd) ? src[i] : 0.0f;
    }

    for (uint h = h0; h < n_hc; h += hstep) {
        device float *dst = res + ((ulong)t * n_hc + h) * n_embd;
        device const float *wh = w + (has_w ? (ulong)h * n_embd : 0ul);
        float vv[Q4E_HC_FUSE_MAX], wv[Q4E_HC_FUSE_MAX];
#pragma unroll
        for (uint n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint i = tid + n * bdim;
            vv[n] = (i < n_embd) ? dst[i] : 0.0f;
            wv[n] = (has_w && i < n_embd) ? wh[i] : 0.0f;
        }
        const float wgt =
            2.0f / (1.0f + exp(-inject[(ulong)t * n_hc + h] / (float)n_hc));
        float sum = 0.0f;
#pragma unroll
        for (uint n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint i = tid + n * bdim;
            if (i < n_embd) {
                float v = vv[n];
                v += sv[n] * wgt;
                dst[i] = v;
                vv[n] = v;
                sum += v * v;
            }
        }
        if (!has_w) continue;
        sum = q4e_block_sum(sum, scratch, tid, bdim);
        const float scale = rsqrt(sum / (float)n_embd + eps);
        device XN *o = xn + ((ulong)t * n_hc + h) * n_embd;
#pragma unroll
        for (uint n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint i = tid + n * bdim;
            if (i < n_embd) q4e_store(o + i, vv[n] * scale * wv[n]);
        }
    }
}

template [[host_name("_Z26q4e_hc_combine_norm_kernelIfEv")]] kernel void q4e_hc_combine_norm_kernel<float>(
        device float *, device float *, device const float *, device const float *,
        device const float *, constant uint &, constant uint &, constant float &,
        uint3, uint3, uint3, uint3);
template [[host_name("_Z26q4e_hc_combine_norm_kernelI6__halfEv")]] kernel void q4e_hc_combine_norm_kernel<half>(
        device float *, device half *, device const float *, device const float *,
        device const float *, constant uint &, constant uint &, constant float &,
        uint3, uint3, uint3, uint3);
