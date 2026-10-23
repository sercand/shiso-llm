// Per-head RMSNorm plus the parametric rotation, for the query half and for the
// key/value half that lands in the cache.
//
// Two things separate this pair from metal/qwen35/rmsnorm_rope.metal, and both
// are arguments rather than constants.  The rotation derives its frequency
// ladder over `freq_dims`, which need not be the rotated width: a head of 512
// whose ladder is the whole head but whose leading 128 dimensions alone rotate
// pairs (i, i + 256), not (i, i + 64), and only the first of those is the
// checkpoint's rotation.  And the value half is normalized -- by the same mean
// square the key uses, with no weight at all -- whenever the layer says so,
// where qwen35 always copies the value raw.
//
// What is NOT here is the output gate.  This family's query projection is
// `n_head * head_dim` wide and carries nothing beside the query, so the query
// kernel writes one row and reads one row.

/// The query half: normalize one head and rotate it.
///
/// One threadgroup per (head, token); the head is staged in threadgroup memory
/// because the rotation reads a pair of dimensions that no single lane owns.
kernel void kernel_dense_q_norm_rope(
        constant dense_args_q_norm_rope & args [[buffer(0)]],
        device float *q_out [[buffer(1)]],
        device const float *q_in [[buffer(2)]],
        device const float *w [[buffer(3)]],
        device const int32_t *pos [[buffer(4)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint tid = tpitg.x;
    const uint nt = ntg.x;
    // One row of the q projection is the query and nothing else.
    const uint64_t base = ((uint64_t) t * args.n_head + h) * head_dim;

    threadgroup float *scratch = tbuf;
    threadgroup float *s_q = tbuf + DENSE_BLOCK_SUM_SLOTS;

    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nt) {
        const float v = q_in[base + i];
        s_q[i] = v;
        sum += v * v;
    }
    // The weight multiplies the normalized value directly: this family stores
    // the weight itself rather than a deviation from one.
    const float inv = rsqrt(dense_block_sum(sum, scratch, tid, nt)
            / (float) head_dim + args.eps);
    for (uint i = tid; i < head_dim; i += nt) {
        s_q[i] *= inv * w[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dense_rope_pairs(s_q, args.n_rot, args.freq_dims, args.rope_base,
            (float) (uint32_t) pos[t], tid, nt);
    // A pair is rotated on the lane owning its first dimension, so the second
    // half of every pair is left by a different lane than the one storing it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < head_dim; i += nt) {
        q_out[base + i] = s_q[i];
    }
}

/// The key/value half: normalize and rotate the key, normalize the value when
/// the layer asks, and write both into the f16 cache row the page table names.
///
/// The cache is f16 because a decode step reads all of it.  The two halves land
/// in one launch because a position's key and its value occupy the same row, and
/// one launch is what keeps them in step.
kernel void kernel_dense_kv_norm_rope_store(
        constant dense_args_kv_norm_rope & args [[buffer(0)]],
        device half *k_cache [[buffer(1)]],
        device half *v_cache [[buffer(2)]],
        device const float *k_in [[buffer(3)]],
        device const float *v_in [[buffer(4)]],
        device const float *kw [[buffer(5)]],
        device const int32_t *pos [[buffer(6)]],
        device const int32_t *pages [[buffer(7)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint tid = tpitg.x;
    const uint nt = ntg.x;
    // One translation per stored row: the table is a handful of entries every
    // lane of the threadgroup reads, so it is a cached broadcast load.
    const uint64_t row = dense_kv_row(dense_row_pages(pages, t, args.page_stride),
                                      (uint32_t) pos[t], args.page_tokens);
    const uint64_t src = ((uint64_t) t * args.n_head_kv + h) * head_dim;
    const uint64_t slot = ((uint64_t) row * args.n_head_kv + h) * head_dim;

    threadgroup float *scratch = tbuf;
    threadgroup float *s_row = tbuf + DENSE_BLOCK_SUM_SLOTS;

    float ksum = 0.0f;
    for (uint i = tid; i < head_dim; i += nt) {
        const float x = k_in[src + i];
        s_row[i] = x;
        ksum += x * x;
    }
    const float kinv = rsqrt(dense_block_sum(ksum, scratch, tid, nt)
            / (float) head_dim + args.eps);
    for (uint i = tid; i < head_dim; i += nt) {
        s_row[i] *= kinv * kw[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dense_rope_pairs(s_row, args.n_rot, args.freq_dims, args.rope_base,
            (float) (uint32_t) pos[t], tid, nt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < head_dim; i += nt) {
        k_cache[slot + i] = half(s_row[i]);
    }

    // The value reuses the key's staging row, so nothing may still be reading it.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float vsum = 0.0f;
    for (uint i = tid; i < head_dim; i += nt) {
        const float x = v_in[src + i];
        s_row[i] = x;
        vsum += x * x;
    }
    // Unweighted on purpose: this is the same mean square the key takes, and the
    // layer carries no value norm weight to apply after it.  `v_norm` is uniform
    // over the threadgroup, so the reduction's barriers are reached by every
    // lane or by none.
    float vinv = 1.0f;
    if (args.v_norm != 0u) {
        vinv = rsqrt(dense_block_sum(vsum, scratch, tid, nt)
                / (float) head_dim + args.eps);
    }
    for (uint i = tid; i < head_dim; i += nt) {
        v_cache[slot + i] = half(s_row[i] * vinv);
    }
}

// -- rotation without a norm -------------------------------------------------
//
// The pair below is the two kernels above with the per-head norm taken out,
// for a family whose attention rotates its queries and keys and normalizes
// neither.  A unit weight on the kernels above is NOT this: their division by
// the head's root mean square is unconditional, and a unit weight only removes
// the multiply after it.
//
// The argument blocks are the norm pair's, with `eps` unread and `v_norm`
// unread: a slot that normalizes nothing has no epsilon, and the engine
// refuses a value norm on a slot without the key norm whose mean square it
// would share.  The staging row is the same width, so the launches ask for the
// same threadgroup memory.

/// The query half: rotate one head and store it.
///
/// One threadgroup per (head, token), the head staged in threadgroup memory
/// for the same reason as `kernel_dense_q_norm_rope`: a pair's two dimensions
/// belong to different lanes.
kernel void kernel_dense_q_rope(
        constant dense_args_q_norm_rope & args [[buffer(0)]],
        device float *q_out [[buffer(1)]],
        device const float *q_in [[buffer(2)]],
        device const int32_t *pos [[buffer(3)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint tid = tpitg.x;
    const uint nt = ntg.x;
    const uint64_t base = ((uint64_t) t * args.n_head + h) * head_dim;

    threadgroup float *s_q = tbuf + DENSE_BLOCK_SUM_SLOTS;
    for (uint i = tid; i < head_dim; i += nt) {
        s_q[i] = q_in[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dense_rope_pairs(s_q, args.n_rot, args.freq_dims, args.rope_base,
            (float) (uint32_t) pos[t], tid, nt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < head_dim; i += nt) {
        q_out[base + i] = s_q[i];
    }
}

/// The key/value half: rotate the key, and write it and the raw value into the
/// f16 cache row the page table names.
kernel void kernel_dense_kv_rope_store(
        constant dense_args_kv_norm_rope & args [[buffer(0)]],
        device half *k_cache [[buffer(1)]],
        device half *v_cache [[buffer(2)]],
        device const float *k_in [[buffer(3)]],
        device const float *v_in [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        device const int32_t *pages [[buffer(6)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint tid = tpitg.x;
    const uint nt = ntg.x;
    const uint64_t row = dense_kv_row(dense_row_pages(pages, t, args.page_stride),
                                      (uint32_t) pos[t], args.page_tokens);
    const uint64_t src = ((uint64_t) t * args.n_head_kv + h) * head_dim;
    const uint64_t slot = ((uint64_t) row * args.n_head_kv + h) * head_dim;

    threadgroup float *s_row = tbuf + DENSE_BLOCK_SUM_SLOTS;
    for (uint i = tid; i < head_dim; i += nt) {
        s_row[i] = k_in[src + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    dense_rope_pairs(s_row, args.n_rot, args.freq_dims, args.rope_base,
            (float) (uint32_t) pos[t], tid, nt);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The value never touches the staging row, so it is stored in the same
    // loop as the key.
    for (uint i = tid; i < head_dim; i += nt) {
        k_cache[slot + i] = half(s_row[i]);
        v_cache[slot + i] = half(v_in[src + i]);
    }
}
