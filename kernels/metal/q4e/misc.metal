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

/// misc.cu `q4e_stream_mean_rows_kernel`: the mean over the hyper-connection
/// streams of each residual row, into one slot of a wider row. Buffers: 0
/// `dst` (f32), 1 `res` (f32, rows*n_hc*n_embd), 2 `n_embd`, 3 `n_hc`, 4
/// `n_tok`, 5 `dst_stride`, 6 `dst_offset`. Flat, one thread an element.
kernel void q4e_stream_mean_rows_kernel(
        device float *dst [[buffer(0)]],
        device const float *res [[buffer(1)]],
        constant uint &n_embd [[buffer(2)]],
        constant uint &n_hc [[buffer(3)]],
        constant uint &n_tok [[buffer(4)]],
        constant uint &dst_stride [[buffer(5)]],
        constant uint &dst_offset [[buffer(6)]],
        uint bid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const ulong idx = (ulong)bid * bdim + tid;
    if (idx >= (ulong)n_tok * n_embd) return;
    const uint e = (uint)(idx % n_embd);
    const ulong t = idx / n_embd;
    device const float *row = res + t * n_hc * n_embd + e;
    float sum = 0.0f;
    for (uint h = 0; h < n_hc; h++) sum += row[(ulong)h * n_embd];
    dst[t * dst_stride + dst_offset + e] = sum / (float)n_hc;
}

static inline uint q4e_order_key(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static inline bool q4e_top_better(uint ka, uint ia, uint kb, uint ib) {
    return ka > kb || (ka == kb && ia < ib);
}

/// misc.cu `q4e_top_slices_kernel`: the `k` highest entries of each slice of
/// each logits row, highest first, ties to the lower id. Threadgroup grid
/// `(slices, rows)` of 256. Buffers: 0 `logits`, 1 `n_vocab`, 2 `k`, 3
/// `slices`, 4 `values` (f32, [row][slice][k]), 5 `ids` (u32, same).
kernel void q4e_top_slices_kernel(
        device const float *logits [[buffer(0)]],
        constant uint &n_vocab [[buffer(1)]],
        constant uint &k [[buffer(2)]],
        constant uint &slices [[buffer(3)]],
        device float *values [[buffer(4)]],
        device uint *ids [[buffer(5)]],
        uint2 group [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    threadgroup uint s_key[32];
    threadgroup uint s_id[32];
    threadgroup uint t_key;
    threadgroup uint t_id;
    const uint row = group.y;
    const uint slice = group.x;
    const uint span = (n_vocab + slices - 1u) / slices;
    const uint lo = slice * span;
    const uint hi = min(lo + span, n_vocab);
    device const float *r = logits + (ulong)row * n_vocab;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5;
    uint th_key = 0u, th_id = 0u;
    for (uint t = 0; t < k; t++) {
        uint bk = 0u, bi = 0xffffffffu;
        for (uint i = lo + tid; i < hi; i += bdim) {
            const uint key = q4e_order_key(r[i]);
            const bool below = t == 0u || key < th_key || (key == th_key && i > th_id);
            if (below && q4e_top_better(key, i, bk, bi)) { bk = key; bi = i; }
        }
        for (ushort off = 16; off > 0; off >>= 1) {
            const uint ok = simd_shuffle_xor(bk, off);
            const uint oi = simd_shuffle_xor(bi, off);
            if (q4e_top_better(ok, oi, bk, bi)) { bk = ok; bi = oi; }
        }
        if (lane == 0u) { s_key[warp] = bk; s_id[warp] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint ck = s_key[0], ci = s_id[0];
            for (uint w = 1u; w < (bdim >> 5); w++) {
                if (q4e_top_better(s_key[w], s_id[w], ck, ci)) { ck = s_key[w]; ci = s_id[w]; }
            }
            const ulong at = ((ulong)row * slices + slice) * k + t;
            values[at] = ci < n_vocab ? r[ci] : -INFINITY;
            ids[at] = ci;
            t_key = ck;
            t_id = ci;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        th_key = t_key;
        th_id = t_id;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
