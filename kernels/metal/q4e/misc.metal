// q4e misc entries: Metal port of kernels/cuda/q4e/misc.cu (see q4e_common.h for conventions).
//
// The two leftovers that belong to no family: the two-term residual add that
// closes the n-gram block, and the on-device greedy pick over logits rows.

/// misc.cu:10 `q4e_add2_kernel`: `res[i] += a[i] + b[i]` (the sum of the two
/// branches first, then into the residual, as CUDA associates it). Buffers:
/// 0 `res` (f32, in/out), 1 `a` (f32), 2 `b` (f32), 3 `n` (u64). Flat.
kernel void q4e_add2_kernel(
        device float *res [[buffer(0)]],
        device const float *a [[buffer(1)]],
        device const float *b [[buffer(2)]],
        constant ulong &n [[buffer(3)]],
        uint bid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const ulong i = (ulong)bid * bdim + tid;
    if (i < n) res[i] += a[i] + b[i];
}

/// misc.cu:19 `q4e_argmax_rows_kernel`: the argmax of each logits row, one
/// threadgroup per row (1024 threads on the host), ties to the lower index.
/// Buffers: 0 `out` (i32, one per row), 1 `logits` (f32, rows*n_vocab), 2
/// `n_vocab`. The draft head passes a fourth argument (`n_rows`) that the
/// CUDA signature does not declare; it is bound at buffer(3) and ignored here
/// exactly as CUDA ignores it.
///
/// A row of all NaN or all -inf answers -1, as in CUDA.
kernel void q4e_argmax_rows_kernel(
        device int *out [[buffer(0)]],
        device const float *logits [[buffer(1)]],
        constant uint &n_vocab [[buffer(2)]],
        uint row_i [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    threadgroup float s_v[32];
    threadgroup int s_i[32];
    device const float *row = logits + (ulong)row_i * n_vocab;
    float best = -INFINITY;
    int best_i = -1;
    for (uint i = tid; i < n_vocab; i += bdim) {
        const float v = row[i];
        if (v > best) { best = v; best_i = (int)i; }
    }
    float ov;
    int oi;
    q4e_block_argmax(best, best_i, &ov, &oi, s_v, s_i, tid, bdim);
    if (tid == 0u) out[row_i] = oi;
}
