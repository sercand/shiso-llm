// Per-head RMSNorm plus partial RoPE for the query and key projections.
//
// RoPE rotates the FIRST n_rot dimensions as neox pairs (i, i + n_rot/2), which
// is what the reference implementation and the CUDA source of truth both do.
// The tail-rope layout used by the generic kernel set rotates the last
// dimensions instead; using it here would leave every head plausible and wrong.
//
// Each of the two families has two entry points over one body: a
// one-dimensional one, where a token's position is a single number, and a
// multimodal one, where it is three grid coordinates and the rotary pairs are
// dealt out among them.  The body is shared because the normalization, the
// gate split and the cache addressing are the same arithmetic either way and a
// second copy of them could drift; the rotation is the only branch, it is
// resolved at compile time from a literal at each entry point, and a chunk with
// no media is dispatched to the one-dimensional entry so that path is not
// merely equal to what it was but is the same instructions.

/// The query half: normalize one head, rotate it, and copy the output gate
/// that rides beside it in the same projection row.
///
/// `mrope` picks the rotation. When it is false only `pos_t` is read, and it
/// carries the token's single position.
static inline void q35_q_norm_rope_body(
        constant q35_args_q_norm_rope & args,
        device float *q_out,
        device float *gate_out,
        device const float *qkv,
        device const float *w,
        threadgroup float *tbuf,
        float pos_t,
        float pos_h,
        float pos_w,
        bool mrope,
        uint3 tgpig,
        ushort3 tpitg,
        ushort3 ntg) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint tid = tpitg.x;
    const uint nt = ntg.x;
    // One row of the q projection holds the query and its output gate.
    const uint64_t src = ((uint64_t) t * args.n_head + h) * (2u * head_dim);
    const uint64_t dst = ((uint64_t) t * args.n_head + h) * head_dim;

    threadgroup float *scratch = tbuf;
    threadgroup float *s_q = tbuf + Q35_BLOCK_SUM_SLOTS;
    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nt) {
        const float v = qkv[src + i];
        s_q[i] = v;
        sum += v * v;
        // The gate rides beside the query and is copied raw: no activation, no
        // norm, no rope.  It is consumed after attention.
        gate_out[dst + i] = qkv[src + head_dim + i];
    }
    const float scale = rsqrt(q35_block_sum(sum, scratch, tid, nt)
            / (float) head_dim + args.eps);
    for (uint i = tid; i < head_dim; i += nt) {
        s_q[i] *= scale * w[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (mrope) {
        q35_rope_mrope_interleaved(s_q, args.n_rot, args.rope_base,
                pos_t, pos_h, pos_w, tid, nt);
    } else {
        q35_rope_neox(s_q, args.n_rot, args.rope_base, pos_t, tid, nt);
    }
    // NeoX rotates a pair on the lane owning its first dimension, so the second
    // half of every pair is left by a different lane than the one storing it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < head_dim; i += nt) {
        q_out[dst + i] = s_q[i];
    }
}

/// The key half: normalize one head, rotate it, and write it into the f16
/// cache row the page table names, with the value copied beside it.
///
/// `row` is the physical cache row the caller translated from the token's
/// LINEAR position. The rotation's positions are a separate argument because
/// the two are different questions: a placeholder token sits at one place in
/// the sequence and at another in the picture, and only the first of those
/// addresses memory.
static inline void q35_k_norm_rope_kv_body(
        constant q35_args_k_norm_rope & args,
        device half *k_cache,
        device half *v_cache,
        device const float *k,
        device const float *v,
        device const float *kw,
        threadgroup float *tbuf,
        uint64_t row,
        float pos_t,
        float pos_h,
        float pos_w,
        bool mrope,
        uint3 tgpig,
        ushort3 tpitg) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint64_t src = ((uint64_t) t * args.n_head_kv + h) * head_dim;
    const uint64_t slot = ((uint64_t) row * args.n_head_kv + h) * head_dim;

    threadgroup float *scratch = tbuf;
    threadgroup float *s_k = tbuf + Q35_BLOCK_SUM_SLOTS;

    float sum = 0.0f;
    for (uint i = tpitg.x; i < head_dim; i += 128u) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrt(q35_block_sum(sum, scratch, tpitg.x, 128u)
            / (float) head_dim + args.eps);
    for (uint i = tpitg.x; i < head_dim; i += 128u) {
        s_k[i] *= scale * kw[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (mrope) {
        q35_rope_mrope_interleaved(s_k, args.n_rot, args.rope_base,
                pos_t, pos_h, pos_w, tpitg.x, 128u);
    } else {
        q35_rope_neox(s_k, args.n_rot, args.rope_base, pos_t, tpitg.x, 128u);
    }
    // NeoX rotates a pair on the lane owning its first dimension, so the second
    // half of every pair is left by a different lane than the one storing it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tpitg.x; i < head_dim; i += 128u) {
        k_cache[slot + i] = half(s_k[i]);
        v_cache[slot + i] = half(v[src + i]);
    }
}

kernel void kernel_q35_q_norm_rope(
        constant q35_args_q_norm_rope & args [[buffer(0)]],
        device float *q_out [[buffer(1)]],
        device float *gate_out [[buffer(2)]],
        device const float *qkv [[buffer(3)]],
        device const float *w [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const float p = (float) (uint32_t) pos[tgpig.y];
    q35_q_norm_rope_body(args, q_out, gate_out, qkv, w, tbuf,
            p, p, p, false, tgpig, tpitg, ntg);
}

// The multimodal query rotation.  Three position streams instead of one: a
// text token carries the same number in all three, and a placeholder token
// carries its temporal group, its row and its column.  There is no linear
// position here because nothing in this kernel addresses memory by position.
kernel void kernel_q35_q_norm_rope_mrope(
        constant q35_args_q_norm_rope & args [[buffer(0)]],
        device float *q_out [[buffer(1)]],
        device float *gate_out [[buffer(2)]],
        device const float *qkv [[buffer(3)]],
        device const float *w [[buffer(4)]],
        device const int32_t *pos_t [[buffer(5)]],
        device const int32_t *pos_h [[buffer(6)]],
        device const int32_t *pos_w [[buffer(7)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint t = tgpig.y;
    q35_q_norm_rope_body(args, q_out, gate_out, qkv, w, tbuf,
            (float) (uint32_t) pos_t[t],
            (float) (uint32_t) pos_h[t],
            (float) (uint32_t) pos_w[t],
            true, tgpig, tpitg, ntg);
}

// Key norm + RoPE written straight into the f16 KV cache, with the value copied
// beside it.  The cache is f16 because a decode step reads all of it; the value
// is stored raw because nothing downstream can use precision the softmax throws
// away.  The cache slot of a position is its physical row, which the session
// page table supplies.
kernel void kernel_q35_k_norm_rope_kv(
        constant q35_args_k_norm_rope & args [[buffer(0)]],
        device half *k_cache [[buffer(1)]],
        device half *v_cache [[buffer(2)]],
        device const float *k [[buffer(3)]],
        device const float *v [[buffer(4)]],
        device const float *kw [[buffer(5)]],
        device const int32_t *pos [[buffer(6)]],
        device const int32_t *pages [[buffer(7)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    // One translation per stored row: the table is a handful of entries that every
    // lane of the threadgroup reads, so it is a cached broadcast load.
    const uint64_t row = q35_kv_row(q35_row_pages(pages, tgpig.y, args.page_stride),
                                    (uint32_t) pos[tgpig.y], args.page_tokens);
    const float p = (float) (uint32_t) pos[tgpig.y];
    q35_k_norm_rope_kv_body(args, k_cache, v_cache, k, v, kw, tbuf,
            row, p, p, p, false, tgpig, tpitg);
}

// The multimodal key rotation.  The linear position still decides the cache
// row -- a placeholder token occupies one sequence slot like any other, and
// the page table knows nothing about pictures -- while the three grid streams
// decide the angles.  Feeding the temporal stream to the page table instead
// would fold a whole image onto one cache row and leave every count correct.
kernel void kernel_q35_k_norm_rope_kv_mrope(
        constant q35_args_k_norm_rope & args [[buffer(0)]],
        device half *k_cache [[buffer(1)]],
        device half *v_cache [[buffer(2)]],
        device const float *k [[buffer(3)]],
        device const float *v [[buffer(4)]],
        device const float *kw [[buffer(5)]],
        device const int32_t *pos [[buffer(6)]],
        device const int32_t *pages [[buffer(7)]],
        device const int32_t *pos_t [[buffer(8)]],
        device const int32_t *pos_h [[buffer(9)]],
        device const int32_t *pos_w [[buffer(10)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint t = tgpig.y;
    const uint64_t row = q35_kv_row(q35_row_pages(pages, t, args.page_stride),
                                    (uint32_t) pos[t], args.page_tokens);
    q35_k_norm_rope_kv_body(args, k_cache, v_cache, k, v, kw, tbuf,
            row,
            (float) (uint32_t) pos_t[t],
            (float) (uint32_t) pos_h[t],
            (float) (uint32_t) pos_w[t],
            true, tgpig, tpitg);
}
