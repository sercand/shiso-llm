// Attention over a bounded key window with an optional relative-position term,
// causal attention over a KV cache (many queries, and the one query of a
// decode step), the cache append, and the half-split rotary embedding.

/// The relative-position half of the attention logits, once per (query,
/// head, distance): `scores[i][h][t] = (q'[i] + bias_v) . rel[t]` over head
/// `h`'s dims, with `q' = q * q_dim_scale`. A simdgroup takes 8 queries by 4
/// distance rows; lanes stride the head's dims, so every read is a row
/// segment the lanes take side by side, and one reduction closes each score.
kernel void kernel_audio_rel_scores_f32(
        constant audio_args_window_attn & args [[buffer(0)]],
        device float *scores                   [[buffer(1)]],
        device const float *q                  [[buffer(2)]],
        device const float *bias_v             [[buffer(3)]],
        device const float *q_dim_scale        [[buffer(4)]],
        device const float *rel                [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint QB = AUDIO_ATTN_QB;
    const uint TR = AUDIO_REL_ROWS;
    const uint t0 = tgpig.x * TR;
    const uint h = tgpig.y;
    const uint i0 = tgpig.z * QB;
    if (t0 >= args.n_d || h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint nq = min(QB, args.n_q - i0);
    const uint nt = min(TR, args.n_d - t0);
    float acc[AUDIO_ATTN_QB][AUDIO_REL_ROWS];
    for (uint r = 0; r < QB; ++r) {
        for (uint t = 0; t < TR; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (uint e = lane; e < hd; e += 32u) {
        const float scale = args.has_q_dim_scale != 0u ? q_dim_scale[e] : 1.0f;
        const float bv = args.has_bias_uv != 0u ? bias_v[hoff + e] : 0.0f;
        float rv[AUDIO_REL_ROWS];
        for (uint t = 0; t < TR; ++t) {
            rv[t] = t < nt ? rel[(uint64_t) (t0 + t) * w + hoff + e] : 0.0f;
        }
        for (uint r = 0; r < nq; ++r) {
            const float qv = q[(uint64_t) (i0 + r) * w + hoff + e] * scale + bv;
            for (uint t = 0; t < TR; ++t) {
                acc[r][t] += qv * rv[t];
            }
        }
    }
    for (uint r = 0; r < nq; ++r) {
        for (uint t = 0; t < nt; ++t) {
            const float total = simd_sum(acc[r][t]);
            if (lane == 0u) {
                scores[((uint64_t) (i0 + r) * args.heads + h) * args.n_d + t0 + t] = total;
            }
        }
    }
}

/// Attention for a block of AUDIO_ATTN_QB queries of one head per simdgroup.
///
/// The block's queries (content bias and per-dim scale applied) sit in
/// threadgroup memory. Keys stream through 32 at a time, one per lane: each
/// lane scores its key against every query of the block, the relative term is
/// read from the scores the pre-pass computed, the block's running max and
/// sum move once per 32 keys per query, and the value rows are accumulated
/// with each lane owning a slice of the head's dims and the probabilities
/// broadcast across lanes. Visibility is decided per (query, key), so a block
/// may straddle chunks; the keys visited are the union its queries may see.
kernel void kernel_audio_window_attn_f32(
        constant audio_args_window_attn & args [[buffer(0)]],
        device float *out                      [[buffer(1)]],
        device const float *q                  [[buffer(2)]],
        device const float *k                  [[buffer(3)]],
        device const float *v                  [[buffer(4)]],
        device const float *bias_u             [[buffer(5)]],
        device const float *q_dim_scale        [[buffer(6)]],
        device const float *rel_scores         [[buffer(7)]],
        threadgroup float *qa                  [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint QB = AUDIO_ATTN_QB;
    const uint h = tgpig.x;
    const uint i0 = tgpig.y * QB;
    if (h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint nq = min(QB, args.n_q - i0);
    const bool has_rel = args.n_d > 0u;
    for (uint e = lane; e < QB * hd; e += 32u) {
        const uint r = e / hd;
        const uint d = e - r * hd;
        float qv = 0.0f;
        if (r < nq) {
            qv = q[(uint64_t) (i0 + r) * w + hoff + d];
            if (args.has_q_dim_scale != 0u) {
                qv *= q_dim_scale[d];
            }
            if (args.has_bias_uv != 0u) {
                qv += bias_u[hoff + d];
            }
        }
        qa[e] = qv;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint qfirst = args.q_abs0 + i0;
    const uint qlast = qfirst + nq - 1u;
    const uint c_first = qfirst / args.chunk;
    const uint c_last = qlast / args.chunk;
    const uint lo_chunk = c_first > args.left_chunks ? c_first - args.left_chunks : 0u;
    const uint lo_abs = max(lo_chunk * args.chunk, args.k_abs0);
    const uint hi_abs = min(min((c_last + 1u) * args.chunk, args.k_end), args.k_abs0 + args.n_k);

    float m[AUDIO_ATTN_QB];
    float l[AUDIO_ATTN_QB];
    float acc[AUDIO_ATTN_QB][AUDIO_ATTN_SLOTS];
    for (uint r = 0; r < QB; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (uint base = lo_abs; base < hi_abs; base += 32u) {
        const uint ka = base + lane;
        const bool key_in = ka < hi_abs;
        const uint j = ka - args.k_abs0;
        const uint kc = ka / args.chunk;
        float sc[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            sc[r] = 0.0f;
        }
        if (key_in) {
            device const float *krow = k + (uint64_t) j * w + hoff;
            for (uint d = 0; d < hd; ++d) {
                const float kd = krow[d];
                for (uint r = 0; r < QB; ++r) {
                    sc[r] += qa[r * hd + d] * kd;
                }
            }
        }
        for (uint r = 0; r < QB; ++r) {
            const uint qabs = qfirst + r;
            const uint qc = qabs / args.chunk;
            const bool visible = key_in && r < nq && kc <= qc && qc - kc <= args.left_chunks;
            float s_r = -INFINITY;
            if (visible) {
                float bd = 0.0f;
                if (has_rel) {
                    const int dr = (int) qabs - (int) ka - args.d_min;
                    if (dr >= 0 && (uint) dr < args.n_d) {
                        bd = rel_scores[((uint64_t) (i0 + r) * args.heads + h) * args.n_d + (uint) dr];
                    }
                }
                s_r = args.scale * (args.ac_scale * sc[r] + bd);
                if (args.softcap > 0.0f) {
                    s_r = args.softcap * precise::tanh(s_r / args.softcap);
                }
            }
            sc[r] = s_r;
        }
        float p[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            const float block_max = simd_max(sc[r]);
            const float m_new = max(m[r], block_max);
            const float corr = m_new == -INFINITY ? 1.0f : precise::exp(m[r] - m_new);
            p[r] = sc[r] == -INFINITY ? 0.0f : precise::exp(sc[r] - m_new);
            l[r] = l[r] * corr + simd_sum(p[r]);
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                acc[r][s] *= corr;
            }
            m[r] = m_new;
        }
        const uint keys_here = min(32u, hi_abs - base);
        for (uint jj = 0; jj < keys_here; ++jj) {
            const uint jv = base + jj - args.k_abs0;
            device const float *vrow = v + (uint64_t) jv * w + hoff;
            float vv[AUDIO_ATTN_SLOTS];
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                const uint e = lane + 32u * s;
                vv[s] = e < hd ? vrow[e] : 0.0f;
            }
            for (uint r = 0; r < QB; ++r) {
                const float pj = simd_shuffle(p[r], (ushort) jj);
                for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                    acc[r][s] += pj * vv[s];
                }
            }
        }
    }
    for (uint r = 0; r < nq; ++r) {
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            const uint e = lane + 32u * s;
            if (e < hd) {
                out[(uint64_t) (i0 + r) * w + hoff + e] = acc[r][s] * inv;
            }
        }
    }
}

/// Half-split rotary embedding, in place: for m < hd/2,
/// (x[m], x[m+hd/2]) rotates by (pos0 + row) * theta^(-2m/hd).
kernel void kernel_audio_rope_half_f32(
        constant audio_args_rope & args [[buffer(0)]],
        device float *x                 [[buffer(1)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint half_dim = args.head_dim / 2u;
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.rows * args.heads * half_dim) {
        return;
    }
    const uint m = i % half_dim;
    const uint rh = i / half_dim;
    const uint row = rh / args.heads;
    device float *p = x + (uint64_t) rh * args.head_dim;
    const float inv_freq = precise::pow(args.theta, -2.0f * (float) m / (float) args.head_dim);
    const float ang = (float) (args.pos0 + row) * inv_freq;
    float c;
    const float s = precise::sincos(ang, c);
    const float a = p[m];
    const float b = p[m + half_dim];
    p[m] = a * c - b * s;
    p[m + half_dim] = b * c + a * s;
}

/// The rotation above at the decoder state's position: row `row` turns by
/// `(state[POS] + pos0 + row) * theta^(-2m/hd)`, so a batch of decode steps
/// rotates each step's row where the step before left the position.
kernel void kernel_audio_rope_half_state_f32(
        constant audio_args_rope & args [[buffer(0)]],
        device float *x                 [[buffer(1)]],
        device const uint *state        [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint half_dim = args.head_dim / 2u;
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.rows * args.heads * half_dim) {
        return;
    }
    const uint m = i % half_dim;
    const uint rh = i / half_dim;
    const uint row = rh / args.heads;
    device float *p = x + (uint64_t) rh * args.head_dim;
    const float inv_freq = precise::pow(args.theta, -2.0f * (float) m / (float) args.head_dim);
    const float ang = (float) (state[AUDIO_DECODE_POS] + args.pos0 + row) * inv_freq;
    float c;
    const float s = precise::sincos(ang, c);
    const float a = p[m];
    const float b = p[m + half_dim];
    p[m] = a * c - b * s;
    p[m + half_dim] = b * c + a * s;
}

/// `rows` new key and value rows, `[rows][kv_heads * hd]`, into one layer's
/// cache planes `[kv_heads][cap][hd]` from the decoder state's position on.
/// A row that would land at or past `cap` is dropped: the host sizes the
/// cache before it queues the rows. The position is left for the step kernel
/// to move, because every layer appends at the same one.
kernel void kernel_audio_kv_append_f32(
        constant audio_args_kv_append & args [[buffer(0)]],
        device float *k_cache                [[buffer(1)]],
        device float *v_cache                [[buffer(2)]],
        device const float *k                [[buffer(3)]],
        device const float *v                [[buffer(4)]],
        device const uint *state             [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    const uint hd = args.head_dim;
    if (i >= args.rows * args.kv_heads * hd) {
        return;
    }
    const uint e = i % hd;
    const uint rh = i / hd;
    const uint h = rh % args.kv_heads;
    const uint r = rh / args.kv_heads;
    const uint pos = state[AUDIO_DECODE_POS] + r;
    if (pos >= args.cap) {
        return;
    }
    const uint64_t dst = ((uint64_t) h * args.cap + pos) * hd + e;
    k_cache[dst] = k[i];
    v_cache[dst] = v[i];
}

/// Causal grouped-query attention of a block of AUDIO_ATTN_QB queries of one
/// head over a layer's cache planes `[kv_heads][cap][hd]`, one simdgroup per
/// block: the blocking of the window attention above. Query `i` sits at
/// `pos0 + i` and sees cache rows `0..=pos0 + i` of kv head
/// `h / (heads / kv_heads)`; the rows visited stop at the block's last
/// query, so nothing past it is ever read.
kernel void kernel_audio_causal_attn_f32(
        constant audio_args_cache_attn & args [[buffer(0)]],
        device float *out                     [[buffer(1)]],
        device const float *q                 [[buffer(2)]],
        device const float *k_cache           [[buffer(3)]],
        device const float *v_cache           [[buffer(4)]],
        threadgroup float *qa                 [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint QB = AUDIO_ATTN_QB;
    const uint h = tgpig.x;
    const uint i0 = tgpig.y * QB;
    if (h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint kvh = h / (args.heads / args.kv_heads);
    device const float *kp = k_cache + (uint64_t) kvh * args.cap * hd;
    device const float *vp = v_cache + (uint64_t) kvh * args.cap * hd;
    const uint nq = min(QB, args.n_q - i0);
    for (uint e = lane; e < QB * hd; e += 32u) {
        const uint r = e / hd;
        const uint d = e - r * hd;
        qa[e] = r < nq ? q[(uint64_t) (i0 + r) * w + hoff + d] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint qfirst = args.pos0 + i0;
    const uint hi = min(qfirst + nq, args.cap);

    float m[AUDIO_ATTN_QB];
    float l[AUDIO_ATTN_QB];
    float acc[AUDIO_ATTN_QB][AUDIO_ATTN_SLOTS];
    for (uint r = 0; r < QB; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (uint base = 0; base < hi; base += 32u) {
        const uint ka = base + lane;
        const bool key_in = ka < hi;
        float sc[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            sc[r] = 0.0f;
        }
        if (key_in) {
            device const float *krow = kp + (uint64_t) ka * hd;
            for (uint d = 0; d < hd; ++d) {
                const float kd = krow[d];
                for (uint r = 0; r < QB; ++r) {
                    sc[r] += qa[r * hd + d] * kd;
                }
            }
        }
        for (uint r = 0; r < QB; ++r) {
            const bool visible = key_in && r < nq && ka <= qfirst + r;
            sc[r] = visible ? args.scale * sc[r] : -INFINITY;
        }
        float p[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            const float block_max = simd_max(sc[r]);
            const float m_new = max(m[r], block_max);
            const float corr = m_new == -INFINITY ? 1.0f : precise::exp(m[r] - m_new);
            p[r] = sc[r] == -INFINITY ? 0.0f : precise::exp(sc[r] - m_new);
            l[r] = l[r] * corr + simd_sum(p[r]);
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                acc[r][s] *= corr;
            }
            m[r] = m_new;
        }
        const uint keys_here = min(32u, hi - base);
        for (uint jj = 0; jj < keys_here; ++jj) {
            device const float *vrow = vp + (uint64_t) (base + jj) * hd;
            float vv[AUDIO_ATTN_SLOTS];
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                const uint e = lane + 32u * s;
                vv[s] = e < hd ? vrow[e] : 0.0f;
            }
            for (uint r = 0; r < QB; ++r) {
                const float pj = simd_shuffle(p[r], (ushort) jj);
                for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                    acc[r][s] += pj * vv[s];
                }
            }
        }
    }
    for (uint r = 0; r < nq; ++r) {
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            const uint e = lane + 32u * s;
            if (e < hd) {
                out[(uint64_t) (i0 + r) * w + hoff + e] = acc[r][s] * inv;
            }
        }
    }
}

/// The one query of a decode step, per head, over cache rows
/// `0..=state[POS]` of its kv head, split over the rows: one simdgroup per
/// (kv head, split, tile of AUDIO_DECODE_GT query heads of its group), four
/// splits to a threadgroup of 128 threads, grid `(kv_heads, ceil(splits /
/// 4), ceil(group / GT))`. The rows present are cut into
/// `splits` runs of `ceil(len / splits)` rounded up to a whole block, and a
/// simdgroup scores its run against each query head of its tile, so a
/// cached row is read once per tile. Lanes own four consecutive dims, so a
/// key or value row is one contiguous 512-byte read; rows go AUDIO_DECODE_KB
/// at a time, the running max and sum moving once per block. Each (head,
/// split) leaves its unnormalised accumulator at `partials[(h * splits +
/// split) * hd]` and its (max, sum) pair after all the accumulators, at
/// `heads * splits * hd + 2 (h * splits + split)`; a split with no rows
/// leaves (-inf, 0) and zeros. The merge entry below closes them.
/// `head_dim` is a multiple of four, at most 128.
///
/// The exponentials here are the fast ones, unlike the window attention's:
/// with them the kernel took 66 us over 4096 cached rows on an M1 Max
/// against 153 us precise, and it runs once per layer per decoded token. The
/// fast `exp(x)` is within `3 + 2|x|` ulp, and every argument here is at most
/// zero, so a weight's absolute error stays within 3 ulp of 1: the weights
/// that matter have x near 0, and the ones with a large |x| are too small to.
kernel void kernel_audio_decode_attn_f32(
        constant audio_args_cache_attn & args [[buffer(0)]],
        device float *partials                [[buffer(1)]],
        device const float *q                 [[buffer(2)]],
        device const float *k_cache           [[buffer(3)]],
        device const float *v_cache           [[buffer(4)]],
        device const uint *state              [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint KB = AUDIO_DECODE_KB;
    const uint GT = AUDIO_DECODE_GT;
    const uint kvh = tgpig.x;
    const uint split = tgpig.y * AUDIO_DECODE_SPLITS_PER_TG + sg;
    const uint group = args.heads / args.kv_heads;
    const uint g0 = tgpig.z * GT;
    if (kvh >= args.kv_heads || split >= args.splits || g0 >= group) {
        return;
    }
    const uint ng = min(GT, group - g0);
    const uint h0 = kvh * group + g0;
    const uint hd = args.head_dim;
    const uint n4 = hd / 4u;
    const bool owns = lane < n4;
    device const float4 *kp = (device const float4 *) (k_cache + (uint64_t) kvh * args.cap * hd);
    device const float4 *vp = (device const float4 *) (v_cache + (uint64_t) kvh * args.cap * hd);
    float4 qv[AUDIO_DECODE_GT];
    float m[AUDIO_DECODE_GT];
    float l[AUDIO_DECODE_GT];
    float4 acc[AUDIO_DECODE_GT];
    for (uint g = 0; g < GT; ++g) {
        qv[g] = (g < ng && owns) ? ((device const float4 *) (q + (uint64_t) (h0 + g) * hd))[lane] : float4(0.0f);
        m[g] = -INFINITY;
        l[g] = 0.0f;
        acc[g] = float4(0.0f);
    }
    const uint len = min(state[AUDIO_DECODE_POS] + 1u, args.cap);
    const uint per = ((len + args.splits - 1u) / args.splits + KB - 1u) / KB * KB;
    const uint start = split * per;
    const uint end = min(start + per, len);
    for (uint base = start; base < end; base += KB) {
        const uint n = min(KB, end - base);
        float4 kv[AUDIO_DECODE_KB];
        float4 vv[AUDIO_DECODE_KB];
        for (uint b = 0; b < KB; ++b) {
            const bool in = b < n && owns;
            kv[b] = in ? kp[(uint64_t) (base + b) * n4 + lane] : float4(0.0f);
            vv[b] = in ? vp[(uint64_t) (base + b) * n4 + lane] : float4(0.0f);
        }
        for (uint g = 0; g < GT; ++g) {
            float p[AUDIO_DECODE_KB];
            float mb = -INFINITY;
            for (uint b = 0; b < KB; ++b) {
                const float4 d4 = qv[g] * kv[b];
                const float s = args.scale * simd_sum((d4.x + d4.y) + (d4.z + d4.w));
                p[b] = b < n ? s : -INFINITY;
                mb = max(mb, p[b]);
            }
            const float m_new = max(m[g], mb);
            const float corr = m[g] == -INFINITY ? 0.0f : exp(m[g] - m_new);
            float sum = 0.0f;
            float4 a = acc[g] * corr;
            for (uint b = 0; b < KB; ++b) {
                const float pb = b < n ? exp(p[b] - m_new) : 0.0f;
                sum += pb;
                a += pb * vv[b];
            }
            l[g] = l[g] * corr + sum;
            acc[g] = a;
            m[g] = m_new;
        }
    }
    const uint64_t ml0 = (uint64_t) args.heads * args.splits * hd;
    for (uint g = 0; g < ng; ++g) {
        const uint64_t slot = (uint64_t) (h0 + g) * args.splits + split;
        if (owns) {
            ((device float4 *) (partials + slot * hd))[lane] = acc[g];
        }
        if (lane == 0u) {
            partials[ml0 + 2u * slot] = m[g];
            partials[ml0 + 2u * slot + 1u] = l[g];
        }
    }
}

/// Closes decode attention, one threadgroup of AUDIO_DECODE_MERGE_THREADS
/// per head: `out = sum_s acc_s e^(m_s - M) / sum_s l_s e^(m_s - M)` with
/// `M` the largest split max. The splits' weights are formed once into
/// threadgroup memory and their sum taken by the block reduction; simdgroup
/// `g` then sums the accumulators of splits `g, g + 8, ...` in order, a lane
/// per four dims, and the eight partial sums are added in simdgroup order,
/// so the result does not depend on how anything was scheduled.
kernel void kernel_audio_decode_attn_merge_f32(
        constant audio_args_cache_attn & args [[buffer(0)]],
        device float *out                     [[buffer(1)]],
        device const float *partials          [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float wts[AUDIO_DECODE_MAX_SPLITS];
    threadgroup float scratch[AUDIO_BLOCK_SUM_SLOTS];
    threadgroup float4 part[AUDIO_DECODE_MERGE_SG][32];
    const uint NT = AUDIO_DECODE_MERGE_THREADS;
    const uint h = tgpig.x;
    if (h >= args.heads) {
        return;
    }
    const uint hd = args.head_dim;
    const uint n4 = hd / 4u;
    const bool owns = lane < n4;
    const uint splits = min(args.splits, AUDIO_DECODE_MAX_SPLITS);
    device const float *ml = partials + (uint64_t) args.heads * args.splits * hd
            + 2u * (uint64_t) h * args.splits;
    device const float4 *accs = (device const float4 *) (partials + (uint64_t) h * args.splits * hd);
    float mx = -INFINITY;
    for (uint s = tid; s < splits; s += NT) {
        mx = max(mx, ml[2u * s]);
    }
    mx = simd_max(mx);
    if (lane == 0u) {
        scratch[sg] = mx;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mx = -INFINITY;
    for (uint g = 0; g < AUDIO_DECODE_MERGE_SG; ++g) {
        mx = max(mx, scratch[g]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = 0.0f;
    for (uint s = tid; s < splits; s += NT) {
        const float ms = ml[2u * s];
        const float ws = ms == -INFINITY ? 0.0f : precise::exp(ms - mx);
        wts[s] = ws;
        t += ml[2u * s + 1u] * ws;
    }
    const float total = audio_block_sum(t, scratch, tid, NT);
    float4 a = float4(0.0f);
    if (owns) {
        for (uint s = sg; s < splits; s += AUDIO_DECODE_MERGE_SG) {
            a += accs[(uint64_t) s * n4 + lane] * wts[s];
        }
    }
    part[sg][lane] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0u && owns) {
        float4 sum = float4(0.0f);
        for (uint g = 0; g < AUDIO_DECODE_MERGE_SG; ++g) {
            sum += part[g][lane];
        }
        const float inv = total > 0.0f ? 1.0f / total : 0.0f;
        ((device float4 *) (out + (uint64_t) h * hd))[lane] = sum * inv;
    }
}
