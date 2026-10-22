// Causal GQA attention over the paged f16 KV cache.
//
// Positions are spread over the threadgroup's simdgroups rather than over its
// lanes: one simdgroup owns whole positions and reduces each score with a
// shuffle butterfly, so the position loop never needs a barrier.  Walking
// positions with a whole-threadgroup reduction per step is the obvious shape and
// is several times slower for the same result.
//
// The softmax is online (running max + running denominator) and the per-simdgroup
// partials are merged at the end by re-referencing every partial against the
// global max, which is what makes the merge order-independent.
kernel void kernel_q35_attn_decode(
        constant q35_args_attn_decode & args [[buffer(0)]],
        device float *out [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const half *k_cache [[buffer(3)]],
        device const half *v_cache [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        device const int32_t *pages [[buffer(6)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint lane = tpitg.x & 31u;
    const uint warp = tpitg.x >> 5u;
    // 24 query heads share 2 key/value heads: an integer divide, not a modulo.
    const uint hkv = h / (args.n_head / args.n_head_kv);
    const float scale = rsqrt((float) head_dim);
    const uint slots = (head_dim + 31u) / 32u;

    // s_v[Q35_ATTN_WARPS][head_dim], then s_m and s_l below it.
    threadgroup float *s_v = tbuf;
    threadgroup float *s_m = tbuf + Q35_ATTN_WARPS * Q35_ATTN_HD;
    threadgroup float *s_l = s_m + Q35_ATTN_WARPS;

    const uint64_t qbase = ((uint64_t) t * args.n_head + h) * head_dim;
    float qv[Q35_ATTN_LANE_SLOTS];
    float acc[Q35_ATTN_LANE_SLOTS];
    for (uint j = 0u; j < slots; j++) {
        const uint i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    // The causal bound is the position of the token itself. The rows under it
// are translated one at a time: a walk over positions crosses a page
// boundary wherever the session happened to start each page.
    const uint last = (uint32_t) pos[t];
    // This ROW's page table, not the launch's. A batched decode step carries
    // one row per session and two sessions share no page, so a table selected
    // once for the whole launch would answer every row past the first out of
    // row zero's conversation.
    device const int32_t *row_pages = q35_row_pages(pages, t, args.page_stride);
    for (uint p = warp; p <= last; p += Q35_ATTN_WARPS) {
        const uint64_t kb =
            ((uint64_t) q35_kv_row(row_pages, p, args.page_tokens) * args.n_head_kv + hkv)
            * head_dim;
        float dot = 0.0f;
        for (uint j = 0u; j < slots; j++) {
            const uint i = lane + 32u * j;
            if (i < head_dim) {
                dot = q35_fma(qv[j], (float) k_cache[kb + i], dot);
            }
        }
        // Butterfly, so every lane ends up holding the whole score.
        for (uint off = 16u; off > 0u; off >>= 1u) {
            dot += simd_shuffle_xor(dot, off);
        }

        const float m_new = max(m, dot);
        const float corr = exp(m - m_new);
        const float w = exp(dot - m_new);
        for (uint j = 0u; j < slots; j++) {
            const uint i = lane + 32u * j;
            const float v = (i < head_dim) ? (float) v_cache[kb + i] : 0.0f;
            acc[j] = acc[j] * corr + w * v;
        }
        l = l * corr + w;
        m = m_new;
    }

    for (uint j = 0u; j < slots; j++) {
        const uint i = lane + 32u * j;
        if (i < head_dim) {
            s_v[warp * Q35_ATTN_HD + i] = acc[j];
        }
    }
    if (lane == 0u) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -INFINITY;
    for (uint w = 0u; w < Q35_ATTN_WARPS; w++) {
        gm = max(gm, s_m[w]);
    }
    float den = 0.0f;
    for (uint w = 0u; w < Q35_ATTN_WARPS; w++) {
        den += s_l[w] * exp(s_m[w] - gm);
    }
    const float inv = 1.0f / den;
    for (uint i = tpitg.x; i < head_dim; i += 256u) {
        float num = 0.0f;
        for (uint w = 0u; w < Q35_ATTN_WARPS; w++) {
            num += s_v[w * Q35_ATTN_HD + i] * exp(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

// The attention output gate: x *= sigmoid(gate).  Applied after attention and
// before o_proj, so it never sees a normalized or scaled value.
kernel void kernel_q35_attn_gate_mul(
        constant q35_args_attn_gate & args [[buffer(0)]],
        device float *x [[buffer(1)]],
        device const float *gate [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint64_t i = (uint64_t) tgpig.x * 256u + tpitg.x;
    if (i >= (uint64_t) args.n_tok * args.n_head * args.head_dim) {
        return;
    }
    x[i] *= q35_sigmoid(gate[i]);
}
