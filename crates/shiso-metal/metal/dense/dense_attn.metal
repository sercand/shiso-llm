// Windowed, bounded and scaled GQA over the paged f16 cache.
//
// The two entry points are the shapes metal/qwen35/gated_attn.metal and
// metal/qwen35/gated_attn_prefill.metal have -- one simdgroup per position for a
// short chunk, a tile of queries against a tile of transposed keys for a long
// one -- with the causal bound replaced by a span.
//
// The span is the whole of what makes this set parametric.  qwen35 derives what
// a query may see from the query's own position: everything at or before it.
// Here the low end is a sliding window (`pos + 1 - window`, or 0 when the layer
// has none) and the high end is a per-token bound the caller supplies, which for
// a text token is its own position and for a bidirectional prefix reaches past
// it.  Neither end is derivable from the other, and a kernel that walked
// `0..=pos` would be right on exactly the layers that have no window.
//
// The score scale is an argument for the same reason: a layer of this family
// passes 1.0 where qwen35 passes `1 / sqrt(head_dim)`, and it is applied to the
// reduced score rather than folded into the query, so the number in the argument
// block is the number the softmax sees.

/// The first key position a query at `pos` may read under `window`.
///
/// `0` is no window at all.  The comparison is written as `pos + 1 > window`
/// rather than `pos >= window` so that the subtraction below never underflows an
/// unsigned position.
static inline uint dense_attn_lo(uint pos, uint window) {
    return (window != 0u && pos + 1u > window) ? (pos + 1u - window) : 0u;
}

/// The decode softmax: one threadgroup per (head, token), positions spread over
/// the threadgroup's simdgroups.
///
/// One simdgroup owns whole positions and reduces each score with a shuffle
/// butterfly, so the position loop never needs a barrier.  The softmax is online
/// (running max + running denominator) and the per-simdgroup partials are merged
/// at the end by re-referencing every partial against the global max, which is
/// what makes the merge order-independent.
kernel void kernel_dense_attn_decode(
        constant dense_args_attn & args [[buffer(0)]],
        device float *out [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const half *k_cache [[buffer(3)]],
        device const half *v_cache [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        device const int32_t *bound [[buffer(6)]],
        device const int32_t *pages [[buffer(7)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    const uint lane = tpitg.x & 31u;
    const uint warp = tpitg.x >> 5u;
    // Query heads share key/value heads in whole groups: an integer divide, not
    // a modulo.
    const uint hkv = h / (args.n_head / args.n_head_kv);
    const uint slots = (head_dim + 31u) / 32u;

    // s_v[DENSE_ATTN_WARPS][DENSE_ATTN_HD], then s_m and s_l below it.
    threadgroup float *s_v = tbuf;
    threadgroup float *s_m = tbuf + DENSE_ATTN_WARPS * DENSE_ATTN_HD;
    threadgroup float *s_l = s_m + DENSE_ATTN_WARPS;

    const uint64_t qbase = ((uint64_t) t * args.n_head + h) * head_dim;
    float qv[DENSE_ATTN_LANE_SLOTS];
    float acc[DENSE_ATTN_LANE_SLOTS];
    for (uint j = 0u; j < slots; j++) {
        const uint i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    // The span, in logical positions.  The rows under it are translated one at a
    // time: a walk over positions crosses a page boundary wherever the session
    // happened to start each page.
    const uint p_lo = dense_attn_lo((uint32_t) pos[t], args.window);
    const uint p_hi = (uint32_t) bound[t];
    // This ROW's page table, not the launch's. A batched decode step carries
    // one row per session and two sessions share no page, so a table selected
    // once for the whole launch would answer every row past the first out of
    // row zero's conversation.
    device const int32_t *row_pages = dense_row_pages(pages, t, args.page_stride);
    for (uint p = p_lo + warp; p <= p_hi; p += DENSE_ATTN_WARPS) {
        const uint64_t kb =
            ((uint64_t) dense_kv_row(row_pages, p, args.page_tokens) * args.n_head_kv + hkv)
            * head_dim;
        float dot = 0.0f;
        for (uint j = 0u; j < slots; j++) {
            const uint i = lane + 32u * j;
            if (i < head_dim) {
                dot = dense_fma(qv[j], (float) k_cache[kb + i], dot);
            }
        }
        // Butterfly, so every lane ends up holding the whole score.
        for (uint off = 16u; off > 0u; off >>= 1u) {
            dot += simd_shuffle_xor(dot, off);
        }
        dot *= args.scale;

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
            s_v[warp * DENSE_ATTN_HD + i] = acc[j];
        }
    }
    if (lane == 0u) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -INFINITY;
    for (uint w = 0u; w < DENSE_ATTN_WARPS; w++) {
        gm = max(gm, s_m[w]);
    }
    // An empty span -- a bound below the window's low end -- would otherwise
    // reference every partial against -INFINITY and write a row of NaN into the
    // residual stream.  Zero is what an attention over no keys contributes.
    if (gm == -INFINITY) {
        for (uint i = tpitg.x; i < head_dim; i += 256u) {
            out[qbase + i] = 0.0f;
        }
        return;
    }
    float den = 0.0f;
    for (uint w = 0u; w < DENSE_ATTN_WARPS; w++) {
        den += s_l[w] * exp(s_m[w] - gm);
    }
    const float inv = 1.0f / den;
    for (uint i = tpitg.x; i < head_dim; i += 256u) {
        float num = 0.0f;
        for (uint w = 0u; w < DENSE_ATTN_WARPS; w++) {
            num += s_v[w * DENSE_ATTN_HD + i] * exp(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

/// The prefill softmax: DENSE_ATTN_QT queries per threadgroup, DENSE_ATTN_KT
/// keys per tile, one key per lane.
///
/// Two shapes carry this kernel and both come from the qwen35 prefill it copies.
/// The first is that lane l scores key p0 + l entirely on its own -- there is no
/// per-key reduction -- so the tile's softmax needs a max butterfly over the
/// whole simdgroup instead.  The second is that the key weights of a tile are
/// warp-private: the register broadcast is a shuffle from the lane that computed
/// the weight, which needs no barrier at all.  That also keeps every barrier in
/// this kernel outside the query loop, where a short final tile would otherwise
/// leave some simdgroups unable to reach one.
///
/// The tile's key range is the union of its queries' spans -- the lowest window
/// start against the highest bound -- and the per-query mask inside the tile is
/// that query's own span.  A tile whose queries disagree therefore stages keys
/// some of them cannot see, which is a wasted load rather than a wrong answer.
///
/// Threadgroup memory is the transposed keys alone: 16 KiB at head_dim 512.  The
/// values are read straight from the cache because that copy would be a
/// bandwidth cache rather than arithmetic, and the device holds the same f16
/// values in the same order.
// One table for the whole launch, and `args.page_stride` is not read here.
// This entry is reached only by a chunk of DENSE_ATTN_QT tokens or more, and a
// chunk is one session's own consecutive positions by construction - a decode
// batch is at most MAX_DECODE_BATCH rows and takes the decode entry instead.
// The host refuses a per-row table on this path rather than letting a tile
// silently read row t0's.
kernel void kernel_dense_attn_prefill(
        constant dense_args_attn & args [[buffer(0)]],
        device float *out [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const half *k_cache [[buffer(3)]],
        device const half *v_cache [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        device const int32_t *bound [[buffer(6)]],
        device const int32_t *pages [[buffer(7)]],
        threadgroup char *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint head_dim = args.head_dim;
    const uint h = tgpig.x;
    const uint t0 = tgpig.y * DENSE_ATTN_QT;
    if (t0 >= args.n_tok) {
        return;
    }
    const uint tid = tpitg.x;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    const uint hkv = h / (args.n_head / args.n_head_kv);
    const uint slots = head_dim / 32u;

    threadgroup half *s_k = (threadgroup half *) tbuf;
    const uint n_q = (args.n_tok - t0 < DENSE_ATTN_QT) ? (args.n_tok - t0) : DENSE_ATTN_QT;

    // The union of the tile's spans: the lowest window start any query of this
    // tile has, against the highest bound any of them carries.
    uint tile_lo = 0xFFFFFFFFu;
    uint tile_hi = 0u;
    for (uint i = 0u; i < n_q; i++) {
        const uint lo = dense_attn_lo((uint32_t) pos[t0 + i], args.window);
        const uint hi = (uint32_t) bound[t0 + i];
        tile_lo = min(tile_lo, lo);
        tile_hi = max(tile_hi, hi);
    }

    // Two queries per simdgroup, each with its own running max, running
    // denominator and head_dim/32 accumulator slots.
    float acc[2][DENSE_ATTN_LANE_SLOTS];
    float m[2], l[2];
    for (uint s = 0u; s < 2u; s++) {
        m[s] = -INFINITY;
        l[s] = 0.0f;
        for (uint j = 0u; j < slots; j++) {
            acc[s][j] = 0.0f;
        }
    }

    for (uint p0 = tile_lo; p0 <= tile_hi; p0 += DENSE_ATTN_KT) {
        const uint n_k = (tile_hi + 1u - p0 < DENSE_ATTN_KT) ? (tile_hi + 1u - p0) : DENSE_ATTN_KT;
        for (uint idx = tid; idx < n_k * head_dim; idx += ntg.x) {
            const uint k = idx / head_dim;
            const uint d = idx - k * head_dim;
            const uint64_t row = dense_kv_row(pages, p0 + k, args.page_tokens);
            const uint64_t src = ((uint64_t) row * args.n_head_kv + hkv) * head_dim + d;
            s_k[(size_t) d * DENSE_ATTN_KT + k] = k_cache[src];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint s = 0u; s < 2u; s++) {
            const uint qi = warp * 2u + s;
            if (qi >= n_q) {
                break;
            }
            const uint q_lo = dense_attn_lo((uint32_t) pos[t0 + qi], args.window);
            const uint q_hi = (uint32_t) bound[t0 + qi];
            const uint64_t qbase = ((uint64_t) (t0 + qi) * args.n_head + h) * head_dim;

            float dot = 0.0f;
            if (lane < n_k && p0 + lane >= q_lo && p0 + lane <= q_hi) {
                for (uint d = 0u; d < head_dim; d++) {
                    dot = dense_fma(q[qbase + d], (float) s_k[(size_t) d * DENSE_ATTN_KT + lane], dot);
                }
                dot *= args.scale;
            } else {
                dot = -INFINITY;
            }

            float tile_max = dot;
            for (uint off = 16u; off > 0u; off >>= 1u) {
                tile_max = max(tile_max, simd_shuffle_xor(tile_max, off));
            }
            if (tile_max == -INFINITY) {
                continue;  // every key of this tile is masked for this query
            }

            const float m_new = max(m[s], tile_max);
            const float corr = exp(m[s] - m_new);
            const float w = exp(dot - m_new);  // 0 wherever the key was masked
            float wsum = w;
            for (uint off = 16u; off > 0u; off >>= 1u) {
                wsum += simd_shuffle_xor(wsum, off);
            }
            l[s] = l[s] * corr + wsum;
            m[s] = m_new;

            for (uint j = 0u; j < slots; j++) {
                const uint d = lane + 32u * j;
                float a = acc[s][j] * corr;
                for (uint k = 0u; k < n_k; k++) {
                    const uint64_t vrow = dense_kv_row(pages, p0 + k, args.page_tokens);
                    const uint64_t vb = ((uint64_t) vrow * args.n_head_kv + hkv) * head_dim + d;
                    a = dense_fma(simd_shuffle(w, (int) k), (float) v_cache[vb], a);
                }
                acc[s][j] = a;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint s = 0u; s < 2u; s++) {
        const uint qi = warp * 2u + s;
        if (qi >= n_q) {
            break;
        }
        // An empty span contributes nothing rather than a row of NaN; see the
        // decode kernel's guard.
        const float inv = (l[s] > 0.0f) ? (1.0f / l[s]) : 0.0f;
        const uint64_t base = ((uint64_t) (t0 + qi) * args.n_head + h) * head_dim;
        // Unguarded, and therefore a precondition: this entry point requires
        // head_dim % 32 == 0.  `slots` is head_dim / 32 and the 32 lanes cover
        // the row between them, so a head that is not a whole number of lanes
        // would leave its tail unwritten AND scatter part of it past the row.
        // The decode entry has no such requirement -- it guards every access
        // with `i < head_dim` -- so this is the prefill entry's alone, and the
        // engine validates it on the host.
        for (uint j = 0u; j < slots; j++) {
            out[base + lane + 32u * j] = acc[s][j] * inv;
        }
    }
}
