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
template <uint SL>
kernel void kernel_dense_attn_decode_impl(
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
    // Compile-time, so the per-lane arrays below stay in registers; a
    // runtime trip count over them spills both to memory.
    constexpr uint slots = SL;

    // s_v[DENSE_ATTN_WARPS][DENSE_ATTN_HD], then s_m and s_l below it.
    threadgroup float *s_v = tbuf;
    threadgroup float *s_m = tbuf + DENSE_ATTN_WARPS * DENSE_ATTN_HD;
    threadgroup float *s_l = s_m + DENSE_ATTN_WARPS;

    const uint64_t qbase = ((uint64_t) t * args.n_head + h) * head_dim;
    float qv[SL];
    float acc[SL];
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

typedef decltype(kernel_dense_attn_decode_impl<16>) kernel_dense_attn_decode_t;

/// Any head dimension up to DENSE_ATTN_HD.
template [[host_name("kernel_dense_attn_decode")]] kernel kernel_dense_attn_decode_t kernel_dense_attn_decode_impl<16>;
/// Head dimensions the slot count fits exactly: the same arithmetic, fewer
/// registers a lane.
template [[host_name("kernel_dense_attn_decode_hd128")]] kernel kernel_dense_attn_decode_t kernel_dense_attn_decode_impl<4>;
template [[host_name("kernel_dense_attn_decode_hd256")]] kernel kernel_dense_attn_decode_t kernel_dense_attn_decode_impl<8>;

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

/// The prefill softmax on simdgroup matrices, for a head `HD` wide: the same
/// span, the same scale and the same paged cache as `kernel_dense_attn_prefill`,
/// with the products on the matrix units.
///
/// The scalar entry above scores one key per lane and blends the values one
/// key at a time, re-translating every value row through the page table for
/// every query. On an M1 Max that held a 3B hybrid's eight attention layers to
/// about 40 GFLOP/s, so a 512-token chunk at position 3584 cost 2.4 s more
/// than the same chunk at position 0. This is the vision tower's flash shape
/// (`metal/vision/vit_attn.metal`) over the cache: a threadgroup takes 64
/// query rows of one head - eight per simdgroup, held as 8x8 fragments - and
/// streams the union of their spans through threadgroup memory `KB` keys at a
/// time, the next block loading into registers while the current one is
/// multiplied, with an online softmax per row.
///
/// Queries, staged keys and values, scores and probabilities are all f32; the
/// cache's f16 values widen exactly on the way in. A score is therefore the
/// scalar entry's dot product summed in another order, and the softmax is the
/// same max-subtracted form accumulated block by block.
///
/// Each row is masked by its own span, so a row whose span holds no key of a
/// block - or no key at all - is carried through with its running maximum at
/// -inf and contributes a row of zeros, as the scalar entry does.
///
/// grid `(n_head, ceil(n_tok / 64))`, block 256, no dynamic threadgroup
/// memory. `head_dim` is the instantiation's `HD`; the host picks the entry by
/// it.
template <uint HD, uint KB>
kernel void kernel_dense_attn_flash(
        constant dense_args_attn & args [[buffer(0)]],
        device float *out [[buffer(1)]],
        device const float *q [[buffer(2)]],
        device const half *k_cache [[buffer(3)]],
        device const half *v_cache [[buffer(4)]],
        device const int32_t *pos [[buffer(5)]],
        device const int32_t *bound [[buffer(6)]],
        device const int32_t *pages [[buffer(7)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint QR = DENSE_FLASH_QROWS;
    const uint NT = DENSE_FLASH_THREADS;
    const uint NB = KB / 8u;
    const uint ND = HD / 8u;
    // Each staged row padded by four floats so the transposed key loads do
    // not land eight rows on one bank.
    const uint LDK = HD + 4u;
    threadgroup float kv[2u * KB * (HD + 4u)];

    const uint h = tgpig.x;
    const uint t0 = tgpig.y * QR;
    // Uniform across the threadgroup, so no barrier below is skipped by part
    // of it.
    if (t0 >= args.n_tok) {
        return;
    }
    const uint n_q = min(args.n_tok - t0, QR);
    const uint hkv = h / (args.n_head / args.n_head_kv);

    // The union of the tile's spans, as the scalar entry takes it: a block
    // some of the tile's rows cannot see is a wasted load, not a wrong answer.
    uint tile_lo = 0xFFFFFFFFu;
    uint tile_hi = 0u;
    for (uint i = 0u; i < n_q; i++) {
        tile_lo = min(tile_lo, dense_attn_lo((uint32_t) pos[t0 + i], args.window));
        tile_hi = max(tile_hi, (uint32_t) bound[t0 + i]);
    }

    // The lane's place in every 8x8 fragment: columns `fn, fn + 1` of row
    // `fr`; the four lanes sharing a row differ in lane bits 0 and 3.
    const ushort fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    const ushort fr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    const uint row = sg * 8u + fr;
    const bool live = row < n_q;
    const uint qrow = t0 + (live ? row : 0u);
    // A row past the chunk reads nothing: its span is empty.
    const uint q_lo = live ? dense_attn_lo((uint32_t) pos[qrow], args.window) : 1u;
    const uint q_hi = live ? (uint32_t) bound[qrow] : 0u;

    simdgroup_float8x8 qf[ND];
    simdgroup_float8x8 of[ND];
    device const float *qp = q + ((uint64_t) qrow * args.n_head + h) * HD;
    DENSE_UNROLL for (uint i = 0; i < ND; ++i) {
        qf[i].thread_elements()[0] = live ? qp[8u * i + fn] : 0.0f;
        qf[i].thread_elements()[1] = live ? qp[8u * i + fn + 1u] : 0.0f;
        of[i] = simdgroup_float8x8(0.0f);
    }

    float run_max = -INFINITY;
    float run_sum = 0.0f;
    threadgroup float *sk = kv;
    threadgroup float *sv = kv + KB * LDK;
    // Four consecutive head values per load. A key past the tile's last
    // stages as zero and is masked below.
    const uint HD4 = HD / 4u;
    const uint SL = (KB * HD4 + NT - 1u) / NT;
    float4 rk[SL];
    float4 rv[SL];
    auto fetch = [&](uint s0) {
        DENSE_UNROLL for (uint t = 0; t < SL; ++t) {
            const uint e = tid + t * NT;
            const uint j = e / HD4;
            const uint d = (e - j * HD4) * 4u;
            const uint s = s0 + j;
            rk[t] = float4(0.0f);
            rv[t] = float4(0.0f);
            if (e < KB * HD4 && s <= tile_hi) {
                const uint64_t r = dense_kv_row(pages, s, args.page_tokens);
                const uint64_t src = (r * args.n_head_kv + hkv) * HD + d;
                rk[t] = float4(*((device const half4 *) (k_cache + src)));
                rv[t] = float4(*((device const half4 *) (v_cache + src)));
            }
        }
    };
    if (tile_lo <= tile_hi) {
        fetch(tile_lo);
    }
    for (uint s0 = tile_lo; s0 <= tile_hi; s0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        DENSE_UNROLL for (uint t = 0; t < SL; ++t) {
            const uint e = tid + t * NT;
            if (e < KB * HD4) {
                const uint j = e / HD4;
                const uint d = (e - j * HD4) * 4u;
                *((threadgroup float4 *) (sk + j * LDK + d)) = rk[t];
                *((threadgroup float4 *) (sv + j * LDK + d)) = rv[t];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s0 + KB <= tile_hi) {
            fetch(s0 + KB);
        }

        // Scores: S (8 x KB) = Q (8 x HD) . K^T, one fragment per 8 keys.
        simdgroup_float8x8 sf[NB];
        DENSE_UNROLL for (uint b = 0; b < NB; ++b) {
            sf[b] = simdgroup_float8x8(0.0f);
        }
        DENSE_UNROLL for (uint i = 0; i < ND; ++i) {
            DENSE_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_float8x8 kt;
                // kt(r, c) = K[8b + c][8i + r]: the transposed load.
                simdgroup_load(kt, sk + 8u * b * LDK + 8u * i, LDK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[b], qf[i], kt, sf[b]);
            }
        }

        float block_max = -INFINITY;
        DENSE_UNROLL for (uint b = 0; b < NB; ++b) {
            DENSE_UNROLL for (uint c = 0; c < 2u; ++c) {
                const uint s = s0 + 8u * b + fn + c;
                const bool seen = s >= q_lo && s <= q_hi;
                const float score = seen ? sf[b].thread_elements()[c] * args.scale : -INFINITY;
                sf[b].thread_elements()[c] = score;
                block_max = max(block_max, score);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float new_max = max(run_max, block_max);
        // A row that has seen no key yet keeps its maximum at -inf; the
        // guards keep -inf - -inf from turning it into NaN.
        const bool empty = new_max == -INFINITY;
        const float alpha = empty ? 1.0f : exp(run_max - new_max);
        float block_sum = 0.0f;
        DENSE_UNROLL for (uint b = 0; b < NB; ++b) {
            DENSE_UNROLL for (uint c = 0; c < 2u; ++c) {
                const float score = sf[b].thread_elements()[c];
                const float p = score == -INFINITY ? 0.0f : exp(score - new_max);
                sf[b].thread_elements()[c] = p;
                block_sum += p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        run_sum = run_sum * alpha + block_sum;
        run_max = new_max;

        DENSE_UNROLL for (uint i = 0; i < ND; ++i) {
            of[i].thread_elements()[0] *= alpha;
            of[i].thread_elements()[1] *= alpha;
            DENSE_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, sv + 8u * b * LDK + 8u * i, LDK);
                simdgroup_multiply_accumulate(of[i], sf[b], vf, of[i]);
            }
        }
    }

    // Normalize by the denominator and write each lane's two elements of
    // every fragment straight to its row. An empty span writes zeros.
    if (!live) {
        return;
    }
    const float inv = (run_sum > 0.0f) ? (1.0f / run_sum) : 0.0f;
    device float *op = out + ((uint64_t) qrow * args.n_head + h) * HD;
    DENSE_UNROLL for (uint i = 0; i < ND; ++i) {
        op[8u * i + fn] = of[i].thread_elements()[0] * inv;
        op[8u * i + fn + 1u] = of[i].thread_elements()[1] * inv;
    }
}

typedef decltype(kernel_dense_attn_flash<64u, 32u>) kernel_dense_attn_flash_t;

// One instantiation per head width a dense family reads at this shape. The
// key block narrows as the head widens so the two staged f32 blocks stay
// under 18 KiB of threadgroup memory: 2 x 32 x 68 x 4 and 2 x 16 x 132 x 4.
template [[host_name("kernel_dense_attn_flash_hd64")]] kernel kernel_dense_attn_flash_t kernel_dense_attn_flash<64u, 32u>;
template [[host_name("kernel_dense_attn_flash_hd128")]] kernel kernel_dense_attn_flash_t kernel_dense_attn_flash<128u, 16u>;
