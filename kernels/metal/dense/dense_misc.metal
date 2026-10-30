// The dense set's small kernels: the gated activation, the scaled residual
// close, the softcap and the row scale.
//
// Each one is a parametric twin of something the generic set already has, kept
// apart from it rather than folded into it with a flag: the generic entries are
// the ones a qwen35 decode dispatches, and a flag on those would be a branch
// that could be got wrong at one call site with nothing failing loudly.
//
// `kernel_dense_add_scale_rms_norm_mul_f32_4` keeps its generic twin's
// `simd_sum` reduction on purpose rather than the dense set's fixed-order
// `dense_block_sum`: it is `kernel_add_rms_norm_mul_f32_4` plus one multiply,
// and staying bit-for-bit the twin it was copied from outranks the set's
// convention here.

/// `dst[i] = gelu_tanh(gate[i]) * up[i] * alpha`, flat over `ne0` elements.
///
/// The gated GELU, which is what a Gemma-shaped feed-forward uses where qwen35
/// uses the gated SiLU.  Same argument block, same clamp handling and the same
/// flat index as `kernel_swiglu_flat_f32`, so the two are interchangeable at a
/// call site that knows which activation the checkpoint asked for.
kernel void kernel_dense_geglu_flat_f32(
        constant shs_metal_args_glu & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint i [[thread_position_in_grid]]) {
    if (i >= (uint)args.ne0) return;

    device const float * src0_f32 = (device const float *) src0 + args.i00;
    device const float * src1_f32 = (device const float *) src1 + args.i10;
    device       float * dst_f32  = (device       float *) dst;

    float x0 = src0_f32[i];
    float x1 = src1_f32[i];
    if (args.limit > 1.0e-6f) {
        x0 = min(x0, args.limit);
        x1 = clamp(x1, -args.limit, args.limit);
    }

    dst_f32[i] = dense_gelu_tanh(x0)*x1*args.alpha;
}

/// `sum = (a + b) * scale`, then `norm = sum * rsqrt(mean(sum^2) + eps) * w`.
///
/// `kernel_add_rms_norm_mul_f32_4` with one difference and no others: the scale
/// is applied to the residual sum before BOTH the store and the sum of squares,
/// so the row the next block adds into carries it and the row the next block
/// reads is normalized against the scaled magnitudes.  The scale rides in its
/// own buffer rather than in the argument block because the block is the
/// generic norm's, shared with every other norm entry in the library.
kernel void kernel_dense_add_scale_rms_norm_mul_f32_4(
        constant shs_metal_args_norm & args,
        device const char * src0,
        device const char * src1,
        device const char * weight,
        device       char * sum_dst,
        device       char * norm_dst,
        constant float & scale,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) shmem_f32[tiisg] = 0.0f;

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    device const float4 *a = (device const float4 *)
        (src0 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);
    device const float4 *b = (device const float4 *)
        (src1 + i03*args.nbf3[0] + i02*args.nbf2[0] + i01*args.nbf1[0]);
    device const float4 *w = (device const float4 *)
        (weight + (i03%args.nef3[1])*args.nbf3[1] + (i02%args.nef2[1])*args.nbf2[1] + (i01%args.nef1[1])*args.nbf1[1]);
    device float4 *sum = (device float4 *)
        (sum_dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);
    device float4 *norm = (device float4 *)
        (norm_dst + i03*args.nb3 + i02*args.nb2 + i01*args.nb1);

    float sumf = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        const float4 v = (a[i00] + b[i00]) * scale;
        sum[i00] = v;
        sumf += dot(v, v);
    }
    sumf = simd_sum(sumf);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) shmem_f32[sgitg] = sumf;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sumf = simd_sum(shmem_f32[tiisg]);

    const float mean = sumf / args.ne00;
    const float norm_scale = 1.0f / sqrt(mean + args.eps);
    for (int i00 = tpitg.x; i00 < args.ne00_t; i00 += ntg.x) {
        norm[i00] = (sum[i00] * norm_scale) * w[i00];
    }
}

/// `x[i] = cap * tanh(x[i] / cap)`, in place over the first `n` elements.
///
/// Gemma's logit softcap: a smooth clamp that leaves the middle of the range
/// nearly untouched and bounds the tails, applied to the logits row before it
/// is sampled from.
///
/// The library's default `tanh` - the fast rational form in `exp(2y)` - is kept
/// here deliberately, unlike `dense_gelu_tanh` in `dense_common.h`, which
/// switches to `precise::tanh` for exactly this reason. The difference is the
/// argument: this kernel's is `x / cap` with `cap` at 30, not a cubic that
/// amplifies a moderate input into a large one, so the fast form's overflow
/// point - an argument past about 44, same threshold `vit_common.h`'s
/// `VIT_GELU_LINEAR` guard is sized against - needs a pre-softcap logit past
/// about 1320. Nothing this model produces gets there, so the guard the GELU
/// needs would cost a kernel here without ever firing.
kernel void kernel_dense_softcap_f32(
        constant dense_args_softcap & args,
        device float * x,
        uint i [[thread_position_in_grid]]) {
    if (i >= args.n) return;
    x[i] = args.cap * tanh(x[i] / args.cap);
}

/// `x[i] *= s`, in place over the first `n` elements.
///
/// The embedding scale and the per-layer scalar are both one multiply over a
/// run of rows, so both reach the device through this.
kernel void kernel_dense_scale_f32(
        constant dense_args_scale & args,
        device float * x,
        uint i [[thread_position_in_grid]]) {
    if (i >= args.n) return;
    x[i] = x[i] * args.s;
}

// The layer norm is compiled with exact floating point: its precision rests on
// forming each deviation as (x - pivot) - dmean, and fast math is free to
// reassociate that into x - (pivot + dmean), which rounds the mean back to the
// row's magnitude and undoes the whole point.
#pragma METAL fp math_mode(safe)
/// `dst[r] = (src[r] - mean(src[r])) / sqrt(var(src[r]) + eps) * w`, the
/// variance biased, one threadgroup a row. Both sums go through
/// `dense_block_sum`, so a row's result is the same whatever else the launch
/// carries.
///
/// Buffers: args, src, w, dst. Grid `rows`, 256 threads.
kernel void kernel_dense_layer_norm_mul_f32(
        constant dense_args_layer_norm & args [[buffer(0)]],
        device const float * src [[buffer(1)]],
        device const float * w [[buffer(2)]],
        device float * dst [[buffer(3)]],
        uint row [[threadgroup_position_in_grid]],
        uint tid [[thread_index_in_threadgroup]],
        uint nt [[threads_per_threadgroup]]) {
    threadgroup float scratch[DENSE_BLOCK_SUM_SLOTS];
    device const float * x = src + (uint64_t) row * args.dim;
    device float * y = dst + (uint64_t) row * args.dim;
    // Everything is taken about the row's first value, never about zero: the
    // hidden states this normalizes can carry large-magnitude dimensions, and
    // a row near 1e4 has an f32 ulp of 1e-3 there, so a mean *stored* at that
    // magnitude is already off by a tenth of a deviation of order 1e-2. `x -
    // pivot` is exact for values that close to each other, and the shifted
    // mean `dmean` stays deviation-sized.
    const float pivot = x[0];
    float sum = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        sum += x[i] - pivot;
    }
    const float dmean = dense_block_sum(sum, scratch, tid, nt) / (float) args.dim;
    float sq = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        const float d = (x[i] - pivot) - dmean;
        sq += d * d;
    }
    const float var = dense_block_sum(sq, scratch, tid, nt) / (float) args.dim;
    const float inv = rsqrt(var + args.eps);
    for (uint i = tid; i < args.dim; i += nt) {
        y[i] = ((x[i] - pivot) - dmean) * inv * w[i];
    }
}
#pragma METAL fp math_mode(fast)
