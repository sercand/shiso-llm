// q4e matvec entries: Metal port of kernels/cuda/q4e/matvec.cu (see q4e_common.h for conventions).
//
// The dense projections of the sparse stack: the brain-float and
// half-precision warp-per-row kernels, their K-split fold, the single-precision
// split mat-vec and its fold, and the two element-width conversions the BLAS
// tiers stage through.
//
// Not ported here: the Q8_0 kernels (the engine refuses block-quantized dense
// weights in this kernel set), the twelve-bit packed projection and its
// expansion, and the EXL3 head-slice kernels.
//
// Every kernel keeps the CUDA per-lane accumulation order (one `fma` per
// weight, the activation rows inner) and the CUDA `shfl_down` fold, so an
// output differs from the CUDA one only where the two compilers contract or
// round differently. None of these kernels uses threadgroup memory beyond the
// 33-float block-sum scratch; the engine requests no dynamic smem for any of
// them.
//
// The half-precision rows kernel's split writes `n_split` single-precision
// (or `OT`) slabs of `NT * out_dim`, split-major; `q4e_matvec_combine_t_kernel`
// folds them with its width argument set to `NT * out_dim`. At one split the
// slab is the destination and no fold runs. The single-precision mat-vec is
// the same shape at one activation row, with the slab stride read from the
// grid width (`gridDim.x == out_dim`).

/// `q4e_matmul_bf16_rows_kernel<NT>` (matvec.cu:109): `NT` activation rows
/// against a brain-float weight, one simdgroup per output row.
///
/// Buffers: 0 `out` f32 `[NT][out_dim]`, 1 `w` bf16 bits `[out_dim][in_dim]`,
/// 2 `x` f32 `[NT][in_dim]`, 3 `in_dim` (even), 4 `out_dim`. Grid
/// `ceil(out_dim / 8)`, block 256: eight simdgroups, one row each. A lane
/// reads eight weights a step over whole 256-wide stretches and weight pairs
/// over the rest, so the per-lane order differs from the CUDA body's pairs:
/// the same products, summed in a different order.
template <int NT>
kernel void q4e_matmul_bf16_rows_kernel(
        device float *out [[buffer(0)]],
        device const ushort *w [[buffer(1)]],
        device const float *x [[buffer(2)]],
        constant uint &in_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim [[threads_per_threadgroup]]) {
    const uint lane = tid3.x & 31u;
    const uint row = bid.x * (bdim.x >> 5u) + (tid3.x >> 5u);
    // The whole simdgroup shares `row`, so it leaves together and the fold
    // below never shuffles with a departed lane.
    if (row >= out_dim) {
        return;
    }
    device const ushort2 *wr = (device const ushort2 *)(w + (ulong)row * in_dim);
    float acc[NT];
    for (int t = 0; t < NT; t++) {
        acc[t] = 0.0f;
    }
    // Sixteen bytes a lane a step - eight weights - where the width allows:
    // a simdgroup then keeps half a kibibyte of the row in flight rather than
    // the pair loads' 128 bytes, which on this device is what separates a
    // brain-float decode step bound by bandwidth from one bound by latency.
    // Whatever the wide loop leaves is summed by the pair loop below.
    const uint wide = (in_dim / 256u) * 256u;
    device const ushort4 *w8 = (device const ushort4 *)(w + (ulong)row * in_dim);
    for (uint j = lane * 8u; j < wide; j += 256u) {
        const ushort4 a = w8[j / 4u];
        const ushort4 b = w8[j / 4u + 1u];
        const float wv[8] = {q4e_bf16(a.x), q4e_bf16(a.y), q4e_bf16(a.z), q4e_bf16(a.w),
                             q4e_bf16(b.x), q4e_bf16(b.y), q4e_bf16(b.z), q4e_bf16(b.w)};
        for (int t = 0; t < NT; t++) {
            device const float4 *xv = (device const float4 *)(x + (ulong)t * in_dim + j);
            const float4 x0 = xv[0];
            const float4 x1 = xv[1];
            float a0 = acc[t];
            a0 = fma(wv[0], x0.x, a0);
            a0 = fma(wv[1], x0.y, a0);
            a0 = fma(wv[2], x0.z, a0);
            a0 = fma(wv[3], x0.w, a0);
            a0 = fma(wv[4], x1.x, a0);
            a0 = fma(wv[5], x1.y, a0);
            a0 = fma(wv[6], x1.z, a0);
            acc[t] = fma(wv[7], x1.w, a0);
        }
    }
    for (uint i = wide / 2u + lane; i < in_dim / 2u; i += 32u) {
        const ushort2 wb = wr[i];
        const float wx = q4e_bf16(wb.x);
        const float wy = q4e_bf16(wb.y);
        for (int t = 0; t < NT; t++) {
            const float2 xv = *(device const float2 *)(x + (ulong)t * in_dim + 2u * i);
            acc[t] = fma(wx, xv.x, acc[t]);
            acc[t] = fma(wy, xv.y, acc[t]);
        }
    }
    for (int t = 0; t < NT; t++) {
        const float v = q4e_warp_sum(acc[t]);
        if (lane == 0u) {
            out[(ulong)t * out_dim + row] = v;
        }
    }
}

/// `q4e_load8` (matvec.cu:146): eight single-precision activations.
static inline void q4e_mv_load8(device const float *p, thread float *v) {
    device const float4 *xp = (device const float4 *)p;
    const float4 xa = xp[0];
    const float4 xb = xp[1];
    v[0] = xa.x; v[1] = xa.y; v[2] = xa.z; v[3] = xa.w;
    v[4] = xb.x; v[5] = xb.y; v[6] = xb.z; v[7] = xb.w;
}

/// `q4e_load8` (matvec.cu:152): eight half-precision activations, widened
/// exactly.
static inline void q4e_mv_load8(device const half *p, thread float *v) {
    device const half4 *xp = (device const half4 *)p;
    const float4 xa = float4(xp[0]);
    const float4 xb = float4(xp[1]);
    v[0] = xa.x; v[1] = xa.y; v[2] = xa.z; v[3] = xa.w;
    v[4] = xb.x; v[5] = xb.y; v[6] = xb.z; v[7] = xb.w;
}

/// `q4e_matmul_f16_rows_kernel<NT, XT, OT>` (matvec.cu:163): `NT` activation
/// rows against a half-precision weight, one simdgroup per output row, the
/// contraction split across `grid.y`.
///
/// Buffers: 0 `out` `OT` `[grid.y][NT][out_dim]` (the destination itself at
/// one split), 1 `w` f16 `[out_dim][in_dim]`, 2 `x` `XT` `[NT][in_dim]`,
/// 3 `in_dim` (a multiple of 8), 4 `out_dim`, 5 `chunk` (a multiple of 256,
/// so a split boundary never cuts an eight-wide load). Grid
/// `(ceil(out_dim / 8), n_split)`, block 256. A split whose range starts past
/// `in_dim` writes zeros, which the fold adds harmlessly.
template <int NT, typename XT, typename OT>
kernel void q4e_matmul_f16_rows_kernel(
        device OT *out [[buffer(0)]],
        device const half *w [[buffer(1)]],
        device const XT *x [[buffer(2)]],
        constant uint &in_dim [[buffer(3)]],
        constant uint &out_dim [[buffer(4)]],
        constant uint &chunk [[buffer(5)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim [[threads_per_threadgroup]]) {
    const uint lane = tid3.x & 31u;
    const uint row = bid.x * (bdim.x >> 5u) + (tid3.x >> 5u);
    if (row >= out_dim) {
        return;
    }
    const uint i0 = bid.y * chunk;
    const uint i1 = min(i0 + chunk, in_dim);
    device const half *wr = w + (ulong)row * in_dim;
    float acc[NT];
    for (int t = 0; t < NT; t++) {
        acc[t] = 0.0f;
    }
    for (uint i = i0 + lane * 8u; i < i1; i += 256u) {
        device const half4 *wp = (device const half4 *)(wr + i);
        const float4 wa = float4(wp[0]);
        const float4 wb = float4(wp[1]);
        const float wv[8] = {wa.x, wa.y, wa.z, wa.w, wb.x, wb.y, wb.z, wb.w};
        for (int t = 0; t < NT; t++) {
            float xv[8];
            q4e_mv_load8(x + (ulong)t * in_dim + i, xv);
            for (int j = 0; j < 8; j++) {
                acc[t] = fma(wv[j], xv[j], acc[t]);
            }
        }
    }
    device OT *dst = out + (ulong)bid.y * NT * out_dim;
    for (int t = 0; t < NT; t++) {
        const float v = q4e_warp_sum(acc[t]);
        if (lane == 0u) {
            q4e_store(dst + (ulong)t * out_dim + row, v);
        }
    }
}

/// `q4e_matvec_combine_t_kernel<OT>` (matvec.cu:355): adds `n_split`
/// single-precision partial slabs, in split order, into `out`.
///
/// Buffers: 0 `out` `OT` `[out_dim]`, 1 `partial` f32 `[n_split][out_dim]`,
/// 2 `out_dim` (the projection's row count times its output width), 3
/// `n_split`. Grid `ceil(out_dim / 128)`, block 128.
template <typename OT>
kernel void q4e_matvec_combine_t_kernel(
        device OT *out [[buffer(0)]],
        device const float *partial [[buffer(1)]],
        constant uint &out_dim [[buffer(2)]],
        constant uint &n_split [[buffer(3)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim [[threads_per_threadgroup]]) {
    const uint row = bid.x * bdim.x + tid3.x;
    if (row >= out_dim) {
        return;
    }
    float acc = 0.0f;
    for (uint s = 0; s < n_split; s++) {
        acc += partial[(ulong)s * out_dim + row];
    }
    q4e_store(out + row, acc);
}

/// `q4e_f32_to_f16_kernel` (matvec.cu:364): rounds to half precision,
/// nearest-even (Metal's default float-to-half rounding, as `__float2half_rn`).
///
/// Buffers: 0 `out` f16 `[n]`, 1 `in` f32 `[n]`, 2 `n` (u64). Grid
/// `ceil(n / 256)`, block 256.
kernel void q4e_f32_to_f16_kernel(
        device half *out [[buffer(0)]],
        device const float *in [[buffer(1)]],
        constant ulong &n [[buffer(2)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim [[threads_per_threadgroup]]) {
    const ulong i = (ulong)bid.x * bdim.x + tid3.x;
    if (i < n) {
        out[i] = half(in[i]);
    }
}

/// `q4e_f16_to_f32_kernel` (matvec.cu:372): the exact widening a half weight
/// takes before a single-precision multiply.
///
/// Buffers: 0 `out` f32 `[n]`, 1 `in` f16 `[n]`, 2 `n` (u64). Grid
/// `ceil(n / 256)`, block 256.
kernel void q4e_f16_to_f32_kernel(
        device float *out [[buffer(0)]],
        device const half *in [[buffer(1)]],
        constant ulong &n [[buffer(2)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim [[threads_per_threadgroup]]) {
    const ulong i = (ulong)bid.x * bdim.x + tid3.x;
    if (i < n) {
        out[i] = float(in[i]);
    }
}

/// `q4e_matvec_f32_kernel` (matvec.cu:413): a single-precision mat-vec, one
/// threadgroup per (output row, contraction split).
///
/// Buffers: 0 `out` f32 `[n_split][out_dim]` (the destination at one split),
/// 1 `w` f32 `[out_dim][in_dim]`, 2 `x` f32 `[in_dim]`, 3 `in_dim`, 4 `chunk`.
/// Grid `(out_dim, n_split)`, block 256; the slab stride is `grid.x`, so the
/// grid width must be exactly `out_dim`. The multiply-add is written as `fma`
/// because that is what the CUDA compiler contracts `acc += w * x` to.
kernel void q4e_matvec_f32_kernel(
        device float *out [[buffer(0)]],
        device const float *w [[buffer(1)]],
        device const float *x [[buffer(2)]],
        constant uint &in_dim [[buffer(3)]],
        constant uint &chunk [[buffer(4)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim [[threads_per_threadgroup]],
        uint3 grid [[threadgroups_per_grid]]) {
    threadgroup float scratch[Q4E_SLOTS];
    const uint row = bid.x;
    const uint sp = bid.y;
    const uint i0 = sp * chunk;
    const uint i1 = min(i0 + chunk, in_dim);
    device const float *wr = w + (ulong)row * in_dim;
    float acc = 0.0f;
    for (uint i = i0 + tid3.x; i < i1; i += bdim.x) {
        acc = fma(wr[i], x[i], acc);
    }
    acc = q4e_block_sum(acc, scratch, tid3.x, bdim.x);
    if (tid3.x == 0u) {
        out[(ulong)sp * grid.x + row] = acc;
    }
}

/// `q4e_matvec_f32_combine_kernel` (matvec.cu:427): adds the mat-vec's
/// `n_split` partial slabs, in split order.
///
/// Buffers: 0 `out` f32 `[out_dim]`, 1 `partial` f32 `[n_split][out_dim]`,
/// 2 `out_dim`, 3 `n_split`. Grid `ceil(out_dim / 128)`, block 128.
kernel void q4e_matvec_f32_combine_kernel(
        device float *out [[buffer(0)]],
        device const float *partial [[buffer(1)]],
        constant uint &out_dim [[buffer(2)]],
        constant uint &n_split [[buffer(3)]],
        uint3 bid [[threadgroup_position_in_grid]],
        uint3 tid3 [[thread_position_in_threadgroup]],
        uint3 bdim [[threads_per_threadgroup]]) {
    const uint row = bid.x * bdim.x + tid3.x;
    if (row >= out_dim) {
        return;
    }
    float acc = 0.0f;
    for (uint s = 0; s < n_split; s++) {
        acc += partial[(ulong)s * out_dim + row];
    }
    out[row] = acc;
}

// Explicit instantiations: exactly the argument sets matvec.cu instantiates,
// under the Itanium names the CUDA build exports, so the engine's symbol
// decoder resolves both backends' entries to the same (template, args).

#define Q4E_MV_BF16_PARAMS \
        device float *out [[buffer(0)]], device const ushort *w [[buffer(1)]], \
        device const float *x [[buffer(2)]], constant uint &in_dim [[buffer(3)]], \
        constant uint &out_dim [[buffer(4)]], uint3 bid [[threadgroup_position_in_grid]], \
        uint3 tid3 [[thread_position_in_threadgroup]], uint3 bdim [[threads_per_threadgroup]]

#define Q4E_MV_F16_PARAMS(XT, OT) \
        device OT *out [[buffer(0)]], device const half *w [[buffer(1)]], \
        device const XT *x [[buffer(2)]], constant uint &in_dim [[buffer(3)]], \
        constant uint &out_dim [[buffer(4)]], constant uint &chunk [[buffer(5)]], \
        uint3 bid [[threadgroup_position_in_grid]], \
        uint3 tid3 [[thread_position_in_threadgroup]], uint3 bdim [[threads_per_threadgroup]]

#define Q4E_MV_COMBINE_PARAMS(OT) \
        device OT *out [[buffer(0)]], device const float *partial [[buffer(1)]], \
        constant uint &out_dim [[buffer(2)]], constant uint &n_split [[buffer(3)]], \
        uint3 bid [[threadgroup_position_in_grid]], \
        uint3 tid3 [[thread_position_in_threadgroup]], uint3 bdim [[threads_per_threadgroup]]

template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi1EEv")]] kernel void q4e_matmul_bf16_rows_kernel<1>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi2EEv")]] kernel void q4e_matmul_bf16_rows_kernel<2>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi3EEv")]] kernel void q4e_matmul_bf16_rows_kernel<3>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi4EEv")]] kernel void q4e_matmul_bf16_rows_kernel<4>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi5EEv")]] kernel void q4e_matmul_bf16_rows_kernel<5>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi6EEv")]] kernel void q4e_matmul_bf16_rows_kernel<6>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi7EEv")]] kernel void q4e_matmul_bf16_rows_kernel<7>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi8EEv")]] kernel void q4e_matmul_bf16_rows_kernel<8>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi9EEv")]] kernel void q4e_matmul_bf16_rows_kernel<9>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi10EEv")]] kernel void q4e_matmul_bf16_rows_kernel<10>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi11EEv")]] kernel void q4e_matmul_bf16_rows_kernel<11>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi12EEv")]] kernel void q4e_matmul_bf16_rows_kernel<12>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi13EEv")]] kernel void q4e_matmul_bf16_rows_kernel<13>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi14EEv")]] kernel void q4e_matmul_bf16_rows_kernel<14>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi15EEv")]] kernel void q4e_matmul_bf16_rows_kernel<15>(Q4E_MV_BF16_PARAMS);
template [[host_name("_Z27q4e_matmul_bf16_rows_kernelILi16EEv")]] kernel void q4e_matmul_bf16_rows_kernel<16>(Q4E_MV_BF16_PARAMS);

template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi1EffEv")]] kernel void q4e_matmul_f16_rows_kernel<1, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi2EffEv")]] kernel void q4e_matmul_f16_rows_kernel<2, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi3EffEv")]] kernel void q4e_matmul_f16_rows_kernel<3, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi4EffEv")]] kernel void q4e_matmul_f16_rows_kernel<4, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi5EffEv")]] kernel void q4e_matmul_f16_rows_kernel<5, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi6EffEv")]] kernel void q4e_matmul_f16_rows_kernel<6, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi7EffEv")]] kernel void q4e_matmul_f16_rows_kernel<7, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi8EffEv")]] kernel void q4e_matmul_f16_rows_kernel<8, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi9EffEv")]] kernel void q4e_matmul_f16_rows_kernel<9, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi10EffEv")]] kernel void q4e_matmul_f16_rows_kernel<10, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi11EffEv")]] kernel void q4e_matmul_f16_rows_kernel<11, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi12EffEv")]] kernel void q4e_matmul_f16_rows_kernel<12, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi13EffEv")]] kernel void q4e_matmul_f16_rows_kernel<13, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi14EffEv")]] kernel void q4e_matmul_f16_rows_kernel<14, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi15EffEv")]] kernel void q4e_matmul_f16_rows_kernel<15, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi16EffEv")]] kernel void q4e_matmul_f16_rows_kernel<16, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi17EffEv")]] kernel void q4e_matmul_f16_rows_kernel<17, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi18EffEv")]] kernel void q4e_matmul_f16_rows_kernel<18, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi19EffEv")]] kernel void q4e_matmul_f16_rows_kernel<19, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi20EffEv")]] kernel void q4e_matmul_f16_rows_kernel<20, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi21EffEv")]] kernel void q4e_matmul_f16_rows_kernel<21, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi22EffEv")]] kernel void q4e_matmul_f16_rows_kernel<22, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi23EffEv")]] kernel void q4e_matmul_f16_rows_kernel<23, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi24EffEv")]] kernel void q4e_matmul_f16_rows_kernel<24, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi25EffEv")]] kernel void q4e_matmul_f16_rows_kernel<25, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi26EffEv")]] kernel void q4e_matmul_f16_rows_kernel<26, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi27EffEv")]] kernel void q4e_matmul_f16_rows_kernel<27, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi28EffEv")]] kernel void q4e_matmul_f16_rows_kernel<28, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi29EffEv")]] kernel void q4e_matmul_f16_rows_kernel<29, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi30EffEv")]] kernel void q4e_matmul_f16_rows_kernel<30, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi31EffEv")]] kernel void q4e_matmul_f16_rows_kernel<31, float, float>(Q4E_MV_F16_PARAMS(float, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi32EffEv")]] kernel void q4e_matmul_f16_rows_kernel<32, float, float>(Q4E_MV_F16_PARAMS(float, float));

template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi1E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<1, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi2E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<2, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi3E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<3, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi4E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<4, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi5E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<5, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi6E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<6, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi7E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<7, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi8E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<8, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi9E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<9, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi10E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<10, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi11E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<11, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi12E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<12, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi13E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<13, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi14E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<14, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi15E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<15, half, float>(Q4E_MV_F16_PARAMS(half, float));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi16E6__halffEv")]] kernel void q4e_matmul_f16_rows_kernel<16, half, float>(Q4E_MV_F16_PARAMS(half, float));

template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi1Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<1, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi2Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<2, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi3Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<3, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi4Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<4, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi5Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<5, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi6Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<6, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi7Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<7, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi8Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<8, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi9Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<9, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi10Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<10, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi11Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<11, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi12Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<12, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi13Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<13, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi14Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<14, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi15Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<15, float, half>(Q4E_MV_F16_PARAMS(float, half));
template [[host_name("_Z26q4e_matmul_f16_rows_kernelILi16Ef6__halfEv")]] kernel void q4e_matmul_f16_rows_kernel<16, float, half>(Q4E_MV_F16_PARAMS(float, half));

template [[host_name("_Z27q4e_matvec_combine_t_kernelIfEv")]] kernel void q4e_matvec_combine_t_kernel<float>(Q4E_MV_COMBINE_PARAMS(float));
template [[host_name("_Z27q4e_matvec_combine_t_kernelI6__halfEv")]] kernel void q4e_matvec_combine_t_kernel<half>(Q4E_MV_COMBINE_PARAMS(half));

#undef Q4E_MV_BF16_PARAMS
#undef Q4E_MV_F16_PARAMS
#undef Q4E_MV_COMBINE_PARAMS
