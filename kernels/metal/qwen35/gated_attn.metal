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

// Split-K decode attention: one threadgroup per (key/value head, row, split).
//
// The single-pass kernel above gives each query head its own threadgroup and
// walks every position with it, so a launch is n_head threadgroups whatever the
// context, and the G query heads of one key/value head each read that head's
// rows from memory. Measured on an M4 at a 4000-position context (Qwen3.5-4B,
// 16 query heads over 4 key/value heads): 1.99 ms a layer, ~8 GB/s against
// the ~16 MB of cache it reads.
//
// Here a threadgroup takes one contiguous chunk of positions and every query
// head of its group, so a cache row is loaded once for all G of them, and a
// row's positions are spread over up to Q35_SPLIT_MAX threadgroups. A lane owns
// DPL consecutive dimensions, so one row is one coalesced read of the whole
// simdgroup. Each split leaves an unnormalized partial (its own running max,
// denominator and weighted value sum); kernel_q35_attn_decode_merge folds them
// in split order.
//
// Order is fixed throughout: positions go to simdgroups round-robin inside the
// split, the simdgroups are merged in index order, and the splits in split
// order, and the split count depends on the row's own position alone.
template <uint G, uint DPL>
kernel void kernel_q35_attn_decode_split(
        constant q35_args_attn_decode & args [[buffer(0)]],
        device float *part_acc [[buffer(1)]],
        device float *part_ml [[buffer(2)]],
        device const float *q [[buffer(3)]],
        device const half *k_cache [[buffer(4)]],
        device const half *v_cache [[buffer(5)]],
        device const int32_t *pos [[buffer(6)]],
        device const int32_t *pages [[buffer(7)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    constexpr uint head_dim = DPL * 32u;
    const uint hkv = tgpig.x;
    const uint t = tgpig.y;
    const uint s = tgpig.z;
    const uint tid = tpitg.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;

    const uint n_pos = (uint32_t) pos[t] + 1u;
    const uint n_split = q35_attn_splits(n_pos);
    if (s >= n_split) {
        return;
    }
    const uint chunk = (n_pos + n_split - 1u) / n_split;
    const uint begin = s * chunk;
    const uint end = min(n_pos, begin + chunk);

    const float scale = rsqrt((float) head_dim);
    const uint h0 = hkv * G;
    float qv[G][DPL];
    float acc[G][DPL];
    float m[G];
    float l[G];
    for (uint g = 0u; g < G; g++) {
        const uint64_t qbase = ((uint64_t) t * args.n_head + h0 + g) * head_dim + lane * DPL;
        for (uint j = 0u; j < DPL; j++) {
            qv[g][j] = q[qbase + j] * scale;
            acc[g][j] = 0.0f;
        }
        m[g] = -INFINITY;
        l[g] = 0.0f;
    }

    device const int32_t *row_pages = q35_row_pages(pages, t, args.page_stride);
    for (uint p = begin + warp; p < end; p += Q35_SPLIT_WARPS) {
        const uint64_t kb =
            ((uint64_t) q35_kv_row(row_pages, p, args.page_tokens) * args.n_head_kv + hkv)
            * head_dim + lane * DPL;
        float kf[DPL];
        float vf[DPL];
        for (uint j = 0u; j < DPL; j += 4u) {
            const float4 k4 = float4(*((device const packed_half4 *) (k_cache + kb + j)));
            const float4 v4 = float4(*((device const packed_half4 *) (v_cache + kb + j)));
            kf[j] = k4.x; kf[j + 1u] = k4.y; kf[j + 2u] = k4.z; kf[j + 3u] = k4.w;
            vf[j] = v4.x; vf[j + 1u] = v4.y; vf[j + 2u] = v4.z; vf[j + 3u] = v4.w;
        }
        for (uint g = 0u; g < G; g++) {
            float dot = 0.0f;
            for (uint j = 0u; j < DPL; j++) {
                dot = q35_fma(qv[g][j], kf[j], dot);
            }
            // Butterfly, so every lane ends up holding the whole score.
            for (uint off = 16u; off > 0u; off >>= 1u) {
                dot += simd_shuffle_xor(dot, off);
            }
            const float m_new = max(m[g], dot);
            const float corr = exp(m[g] - m_new);
            const float w = exp(dot - m_new);
            for (uint j = 0u; j < DPL; j++) {
                acc[g][j] = acc[g][j] * corr + w * vf[j];
            }
            l[g] = l[g] * corr + w;
            m[g] = m_new;
        }
    }

    // s_acc[Q35_SPLIT_WARPS][head_dim], then s_m and s_l, G slots a simdgroup.
    // One head's sums are staged at a time, so the threadgroup memory is one
    // head's worth whatever G is.
    threadgroup float *s_acc = tbuf;
    threadgroup float *s_m = tbuf + Q35_SPLIT_WARPS * head_dim;
    threadgroup float *s_l = s_m + Q35_SPLIT_WARPS * G;
    if (lane == 0u) {
        for (uint g = 0u; g < G; g++) {
            s_m[warp * G + g] = m[g];
            s_l[warp * G + g] = l[g];
        }
    }
    for (uint g = 0u; g < G; g++) {
        for (uint j = 0u; j < DPL; j++) {
            s_acc[warp * head_dim + lane * DPL + j] = acc[g][j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // A simdgroup that drew no position holds m = -inf and l = 0, and
        // exp(-inf - gm) is zero; simdgroup 0 always draws `begin`, so gm is
        // finite.
        float gm = -INFINITY;
        for (uint w = 0u; w < Q35_SPLIT_WARPS; w++) {
            gm = max(gm, s_m[w * G + g]);
        }
        const uint64_t slot =
            ((uint64_t) t * args.n_head + h0 + g) * Q35_SPLIT_MAX + s;
        for (uint i = tid; i < head_dim; i += Q35_SPLIT_WARPS * 32u) {
            float num = 0.0f;
            for (uint w = 0u; w < Q35_SPLIT_WARPS; w++) {
                num += s_acc[w * head_dim + i] * exp(s_m[w * G + g] - gm);
            }
            part_acc[slot * head_dim + i] = num;
        }
        if (tid == 0u) {
            float den = 0.0f;
            for (uint w = 0u; w < Q35_SPLIT_WARPS; w++) {
                den += s_l[w * G + g] * exp(s_m[w * G + g] - gm);
            }
            part_ml[slot * 2u] = gm;
            part_ml[slot * 2u + 1u] = den;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// G query heads a key/value head serves, at head_dim 256: the 4 of Qwen3.5-4B
// and 9B, the 6 of 27B, and the 8 of the 35B mixture. Any other shape keeps
// the single-pass kernel.
template [[host_name("kernel_q35_attn_decode_split_g4")]] kernel void
kernel_q35_attn_decode_split<4u, 8u>(
        constant q35_args_attn_decode &, device float *, device float *,
        device const float *, device const half *, device const half *,
        device const int32_t *, device const int32_t *, threadgroup float *,
        uint3, ushort3);
template [[host_name("kernel_q35_attn_decode_split_g6")]] kernel void
kernel_q35_attn_decode_split<6u, 8u>(
        constant q35_args_attn_decode &, device float *, device float *,
        device const float *, device const half *, device const half *,
        device const int32_t *, device const int32_t *, threadgroup float *,
        uint3, ushort3);
template [[host_name("kernel_q35_attn_decode_split_g8")]] kernel void
kernel_q35_attn_decode_split<8u, 8u>(
        constant q35_args_attn_decode &, device float *, device float *,
        device const float *, device const half *, device const half *,
        device const int32_t *, device const int32_t *, threadgroup float *,
        uint3, ushort3);

// Folds a row's split partials, in split order, into its attention output.
//
// One threadgroup per (head, row) and one lane per dimension. Every lane
// re-derives the global max and the denominator from the same values in the
// same order, so they agree bit for bit without a broadcast.
kernel void kernel_q35_attn_decode_merge(
        constant q35_args_attn_decode & args [[buffer(0)]],
        device float *out [[buffer(1)]],
        device const float *part_acc [[buffer(2)]],
        device const float *part_ml [[buffer(3)]],
        device const int32_t *pos [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint h = tgpig.x;
    const uint t = tgpig.y;
    const uint n_split = q35_attn_splits((uint32_t) pos[t] + 1u);
    const uint64_t row = (uint64_t) t * args.n_head + h;
    const uint64_t slot0 = row * Q35_SPLIT_MAX;

    float gm = -INFINITY;
    for (uint s = 0u; s < n_split; s++) {
        gm = max(gm, part_ml[(slot0 + s) * 2u]);
    }
    float den = 0.0f;
    for (uint s = 0u; s < n_split; s++) {
        den += part_ml[(slot0 + s) * 2u + 1u] * exp(part_ml[(slot0 + s) * 2u] - gm);
    }
    const float inv = 1.0f / den;
    for (uint i = tpitg.x; i < head_dim; i += 256u) {
        float num = 0.0f;
        for (uint s = 0u; s < n_split; s++) {
            num += part_acc[(slot0 + s) * head_dim + i] * exp(part_ml[(slot0 + s) * 2u] - gm);
        }
        out[row * head_dim + i] = num * inv;
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
