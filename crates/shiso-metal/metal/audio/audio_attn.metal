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

/// The rotation above at each row's own position: row `row` turns by
/// `(pos0 + positions[row]) * theta^(-2m/hd)`, so rows of several packed
/// sequences each count from their own start.
kernel void kernel_audio_rope_half_pos_f32(
        constant audio_args_rope & args [[buffer(0)]],
        device float *x                 [[buffer(1)]],
        device const uint *positions    [[buffer(2)]],
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
    const float ang = (float) (args.pos0 + positions[row]) * inv_freq;
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

/// One threadgroup's head of the fused norm, rotation and append below, over
/// one row: `q`, `k` and `v` are the row's, `k_plane` and `v_plane` its
/// sequence's cache planes of `cap` rows, and `pos` the row's position.
/// Every entry that fuses these steps runs this body, so a batched launch
/// forms each value exactly as the single-sequence one does. With `append`
/// false the key head writes nothing (the query head is rotated regardless).
static inline void audio_qk_rope_append_head(
        uint heads, uint hd, float eps, float theta,
        device float *q, device const float *k, device const float *v,
        device const float *q_norm, device const float *k_norm,
        device float *k_plane, device float *v_plane, uint cap, uint pos, bool append,
        uint h, uint t, uint nt,
        threadgroup float *scratch, threadgroup float *normed) {
    const uint half_dim = hd / 2u;
    const bool is_q = h < heads;
    const uint kh = h - heads;
    device const float *src = is_q ? q + (uint64_t) h * hd : k + (uint64_t) kh * hd;
    device const float *weight = is_q ? q_norm : k_norm;
    float sq = 0.0f;
    for (uint i = t; i < hd; i += nt) {
        sq += src[i] * src[i];
    }
    const float rms = rsqrt(audio_block_sum(sq, scratch, t, nt) / (float) hd + eps);
    for (uint i = t; i < hd; i += nt) {
        const float y = src[i] * rms;
        normed[i] = y * weight[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (is_q) {
        device float *p = q + (uint64_t) h * hd;
        for (uint m = t; m < half_dim; m += nt) {
            const float inv_freq = precise::pow(theta, -2.0f * (float) m / (float) hd);
            const float ang = (float) pos * inv_freq;
            float c;
            const float s = precise::sincos(ang, c);
            const float a = normed[m];
            const float b = normed[m + half_dim];
            p[m] = a * c - b * s;
            p[m + half_dim] = b * c + a * s;
        }
        return;
    }
    if (!append || pos >= cap) {
        return;
    }
    const uint64_t at = ((uint64_t) kh * cap + pos) * hd;
    for (uint m = t; m < half_dim; m += nt) {
        const float inv_freq = precise::pow(theta, -2.0f * (float) m / (float) hd);
        const float ang = (float) pos * inv_freq;
        float c;
        const float s = precise::sincos(ang, c);
        const float a = normed[m];
        const float b = normed[m + half_dim];
        k_plane[at + m] = a * c - b * s;
        k_plane[at + m + half_dim] = b * c + a * s;
    }
    device const float *vs = v + (uint64_t) kh * hd;
    for (uint e = t; e < hd; e += nt) {
        v_plane[at + e] = vs[e];
    }
}

/// A decoder step's per-head query and key norms, their rotation and the
/// cache append in one launch, each value formed exactly as the RMS norm
/// (256 threads over a head's dims), the state-positioned rotation and the
/// append form it. Threadgroup `(h, r)` of row `r`: `h < heads` normalises
/// query head `h` by `q_norm` and rotates it in place; the next `kv_heads`
/// normalise key head `h - heads` by `k_norm`, rotate it and write it, with
/// value head `h - heads` as it is, into the cache at `state[POS] + r`. A row
/// at or past `cap` is not written, as the append drops it.
kernel void kernel_audio_qk_norm_rope_append_f32(
        constant audio_args_qk_rope & args [[buffer(0)]],
        device float *q                    [[buffer(1)]],
        device const float *k              [[buffer(2)]],
        device const float *v              [[buffer(3)]],
        device const float *q_norm         [[buffer(4)]],
        device const float *k_norm         [[buffer(5)]],
        device float *k_cache              [[buffer(6)]],
        device float *v_cache              [[buffer(7)]],
        device const uint *state           [[buffer(8)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float scratch[AUDIO_BLOCK_SUM_SLOTS];
    threadgroup float normed[AUDIO_ATTN_SLOTS * 32u];
    const uint row = tgpig.y;
    const uint hd = args.head_dim;
    audio_qk_rope_append_head(
            args.heads, hd, args.eps, args.theta,
            q + (uint64_t) row * args.heads * hd,
            k + (uint64_t) row * args.kv_heads * hd,
            v + (uint64_t) row * args.kv_heads * hd,
            q_norm, k_norm, k_cache, v_cache, args.cap,
            state[AUDIO_DECODE_POS] + row, true,
            tgpig.x, tpitg.x, ntg.x, scratch, normed);
}

/// The fused norm, rotation and append over a segmented prefill's stacked
/// rows `[rows][heads * hd]`: threadgroup `(h, r)` finds the sequence whose
/// table entry owns stacked row `r` (rows `[row0, row0 + rows)`) and works at
/// position `pos0 + (r - row0)` into that sequence's planes of the shared
/// buffers. A row no sequence owns is left alone; a row at or past its
/// sequence's capacity is not appended.
kernel void kernel_audio_qk_norm_rope_append_seg_f32(
        constant audio_args_qk_rope_seq & args [[buffer(0)]],
        device float *q                        [[buffer(1)]],
        device const float *k                  [[buffer(2)]],
        device const float *v                  [[buffer(3)]],
        device const float *q_norm             [[buffer(4)]],
        device const float *k_norm             [[buffer(5)]],
        device float *k_cache                  [[buffer(6)]],
        device float *v_cache                  [[buffer(7)]],
        device const uint *table               [[buffer(8)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float scratch[AUDIO_BLOCK_SUM_SLOTS];
    threadgroup float normed[AUDIO_ATTN_SLOTS * 32u];
    const uint row = tgpig.y;
    if (row >= args.rows) {
        return;
    }
    uint s = 0u;
    while (s < args.seqs) {
        device const uint *e = table + (uint64_t) s * AUDIO_SEQ_WORDS;
        if (row >= e[AUDIO_SEQ_ROW0] && row - e[AUDIO_SEQ_ROW0] < e[AUDIO_SEQ_ROWS]) {
            break;
        }
        ++s;
    }
    if (s >= args.seqs) {
        return;
    }
    device const uint *e = table + (uint64_t) s * AUDIO_SEQ_WORDS;
    const uint hd = args.head_dim;
    const uint cap = e[AUDIO_SEQ_CAP];
    const uint64_t plane = (uint64_t) e[AUDIO_SEQ_PLANE] * hd;
    audio_qk_rope_append_head(
            args.heads, hd, args.eps, args.theta,
            q + (uint64_t) row * args.heads * hd,
            k + (uint64_t) row * args.kv_heads * hd,
            v + (uint64_t) row * args.kv_heads * hd,
            q_norm, k_norm, k_cache + plane, v_cache + plane, cap,
            e[AUDIO_SEQ_POS0] + (row - e[AUDIO_SEQ_ROW0]), true,
            tgpig.x, tpitg.x, ntg.x, scratch, normed);
}

/// The fused norm, rotation and append of one decode step over a batch:
/// row `s` of `[seqs][heads * hd]` is sequence `s`'s, at its own state's
/// position (state block `s`), appended into its own planes. A finished
/// sequence (its state done) appends nothing.
kernel void kernel_audio_qk_norm_rope_append_batch_f32(
        constant audio_args_qk_rope_seq & args [[buffer(0)]],
        device float *q                        [[buffer(1)]],
        device const float *k                  [[buffer(2)]],
        device const float *v                  [[buffer(3)]],
        device const float *q_norm             [[buffer(4)]],
        device const float *k_norm             [[buffer(5)]],
        device float *k_cache                  [[buffer(6)]],
        device float *v_cache                  [[buffer(7)]],
        device const uint *states              [[buffer(8)]],
        device const uint *table               [[buffer(9)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    threadgroup float scratch[AUDIO_BLOCK_SUM_SLOTS];
    threadgroup float normed[AUDIO_ATTN_SLOTS * 32u];
    const uint s = tgpig.y;
    if (s >= args.seqs) {
        return;
    }
    device const uint *st = states + (uint64_t) s * AUDIO_DECODE_STATE_WORDS;
    device const uint *e = table + (uint64_t) s * AUDIO_SEQ_WORDS;
    const uint hd = args.head_dim;
    const uint64_t plane = (uint64_t) e[AUDIO_SEQ_PLANE] * hd;
    audio_qk_rope_append_head(
            args.heads, hd, args.eps, args.theta,
            q + (uint64_t) s * args.heads * hd,
            k + (uint64_t) s * args.kv_heads * hd,
            v + (uint64_t) s * args.kv_heads * hd,
            q_norm, k_norm, k_cache + plane, v_cache + plane, e[AUDIO_SEQ_CAP],
            st[AUDIO_DECODE_POS], st[AUDIO_DECODE_DONE] == 0u,
            tgpig.x, tpitg.x, ntg.x, scratch, normed);
}

/// One simdgroup's block of causal attention: queries `i0 ..` of head `h`
/// among one sequence's `n_q` rows (`q` and `out` at its row 0), at
/// positions from `pos0`, over its kv head's rows `kp` and `vp` of a plane of
/// `cap`. The single-sequence and segmented entries both run this body.
static inline void audio_causal_attn_block(
        device float *out, device const float *q,
        device const float *kp, device const float *vp,
        threadgroup float *qa,
        uint n_q, uint heads, uint hd, uint h, uint i0, uint pos0, uint cap, float scale,
        ushort lane) {
    const uint QB = AUDIO_ATTN_QB;
    const uint w = heads * hd;
    const uint hoff = h * hd;
    const uint nq = min(QB, n_q - i0);
    for (uint e = lane; e < QB * hd; e += 32u) {
        const uint r = e / hd;
        const uint d = e - r * hd;
        qa[e] = r < nq ? q[(uint64_t) (i0 + r) * w + hoff + d] : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    const uint qfirst = pos0 + i0;
    const uint hi = min(qfirst + nq, cap);

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
            sc[r] = visible ? scale * sc[r] : -INFINITY;
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
    const uint h = tgpig.x;
    const uint i0 = tgpig.y * AUDIO_ATTN_QB;
    if (h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint kvh = h / (args.heads / args.kv_heads);
    audio_causal_attn_block(
            out, q,
            k_cache + (uint64_t) kvh * args.cap * hd,
            v_cache + (uint64_t) kvh * args.cap * hd,
            qa, args.n_q, args.heads, hd, h, i0, args.pos0, args.cap, args.scale, lane);
}

/// Causal attention over a segmented prefill: sequence `s` of the table owns
/// stacked query rows `[row0, row0 + rows)` of `[rows][heads * hd]`, at
/// positions from `pos0`, and attends over its own plane of the shared
/// cache buffers exactly as the single-sequence entry does over a plane of
/// its own. Grid `(heads, ceil(max rows / 8), seqs)`: a block past its
/// sequence's rows returns, and a stacked row no sequence owns is never
/// written.
kernel void kernel_audio_causal_attn_seg_f32(
        constant audio_args_batch_attn & args [[buffer(0)]],
        device float *out                     [[buffer(1)]],
        device const float *q                 [[buffer(2)]],
        device const float *k_cache           [[buffer(3)]],
        device const float *v_cache           [[buffer(4)]],
        device const uint *table              [[buffer(5)]],
        threadgroup float *qa                 [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]]) {
    const uint h = tgpig.x;
    const uint i0 = tgpig.y * AUDIO_ATTN_QB;
    const uint s = tgpig.z;
    if (h >= args.heads || s >= args.seqs) {
        return;
    }
    device const uint *e = table + (uint64_t) s * AUDIO_SEQ_WORDS;
    const uint rows = e[AUDIO_SEQ_ROWS];
    if (i0 >= rows) {
        return;
    }
    const uint hd = args.head_dim;
    const uint cap = e[AUDIO_SEQ_CAP];
    const uint kvh = h / (args.heads / args.kv_heads);
    const uint64_t plane = ((uint64_t) e[AUDIO_SEQ_PLANE] + (uint64_t) kvh * cap) * hd;
    const uint64_t row0 = (uint64_t) e[AUDIO_SEQ_ROW0] * args.heads * hd;
    audio_causal_attn_block(
            out + row0, q + row0, k_cache + plane, v_cache + plane,
            qa, rows, args.heads, hd, h, i0, e[AUDIO_SEQ_POS0], cap, args.scale, lane);
}

/// One simdgroup's split of decode attention: the query heads `h0 ..
/// h0 + ng` of one sequence's query row `q` over rows `0..len` of their kv
/// head's key and value rows `kp` and `vp`, left as partials under heads
/// `slot0 ..` of a partial buffer whose (max, sum) pairs start at `ml0`. The
/// single-sequence and batched entries both run this body, so a batched
/// launch forms each value exactly as the single one does.
static inline void audio_decode_attn_split(
        device float *partials, device const float *q,
        device const float4 *kp, device const float4 *vp,
        uint len, uint split, uint splits, uint h0, uint slot0, uint ng, uint hd,
        float scale, uint64_t ml0, ushort lane) {
    const uint KB = AUDIO_DECODE_KB;
    const uint GT = AUDIO_DECODE_GT;
    const uint n4 = hd / 4u;
    const bool owns = lane < n4;
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
    const uint per = ((len + splits - 1u) / splits + KB - 1u) / KB * KB;
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
                const float s = scale * simd_sum((d4.x + d4.y) + (d4.z + d4.w));
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
    for (uint g = 0; g < ng; ++g) {
        const uint64_t slot = (uint64_t) (slot0 + g) * splits + split;
        if (owns) {
            ((device float4 *) (partials + slot * hd))[lane] = acc[g];
        }
        if (lane == 0u) {
            partials[ml0 + 2u * slot] = m[g];
            partials[ml0 + 2u * slot + 1u] = l[g];
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
    const uint GT = AUDIO_DECODE_GT;
    const uint kvh = tgpig.x;
    const uint split = tgpig.y * AUDIO_DECODE_SPLITS_PER_TG + sg;
    const uint group = args.heads / args.kv_heads;
    const uint g0 = tgpig.z * GT;
    if (kvh >= args.kv_heads || split >= args.splits || g0 >= group) {
        return;
    }
    const uint h0 = kvh * group + g0;
    const uint hd = args.head_dim;
    const uint64_t plane = (uint64_t) kvh * args.cap * hd;
    audio_decode_attn_split(
            partials, q,
            (device const float4 *) (k_cache + plane), (device const float4 *) (v_cache + plane),
            min(state[AUDIO_DECODE_POS] + 1u, args.cap), split, args.splits, h0, h0,
            min(GT, group - g0), hd, args.scale,
            (uint64_t) args.heads * args.splits * hd, lane);
}

/// Decode attention's first launch over a batch: the one query row of each
/// sequence, `[seqs][heads * hd]`, over its own plane of the shared cache
/// buffers and its own length, `state[POS] + 1` from its state block
/// (clamped to its capacity; none at all for a finished sequence, whose row
/// then merges to zeros). Grid `(kv_heads, ceil(splits / 4), ceil(group /
/// GT) * seqs)`, 128 threads: `z` is sequence-major. The partials are laid
/// out as the single entry's for `seqs * heads` heads, sequence `s`'s head
/// `h` at `s * heads + h`, so `kernel_audio_decode_attn_merge_f32` over
/// `seqs * heads` heads closes the batch into `[seqs][heads * hd]`.
kernel void kernel_audio_decode_attn_batch_f32(
        constant audio_args_batch_attn & args [[buffer(0)]],
        device float *partials                [[buffer(1)]],
        device const float *q                 [[buffer(2)]],
        device const float *k_cache           [[buffer(3)]],
        device const float *v_cache           [[buffer(4)]],
        device const uint *states             [[buffer(5)]],
        device const uint *table              [[buffer(6)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint GT = AUDIO_DECODE_GT;
    const uint kvh = tgpig.x;
    const uint split = tgpig.y * AUDIO_DECODE_SPLITS_PER_TG + sg;
    const uint group = args.heads / args.kv_heads;
    const uint tiles = (group + GT - 1u) / GT;
    const uint s = tgpig.z / tiles;
    const uint g0 = (tgpig.z - s * tiles) * GT;
    if (s >= args.seqs || kvh >= args.kv_heads || split >= args.splits || g0 >= group) {
        return;
    }
    device const uint *st = states + (uint64_t) s * AUDIO_DECODE_STATE_WORDS;
    device const uint *e = table + (uint64_t) s * AUDIO_SEQ_WORDS;
    const uint cap = e[AUDIO_SEQ_CAP];
    const uint len = st[AUDIO_DECODE_DONE] != 0u ? 0u : min(st[AUDIO_DECODE_POS] + 1u, cap);
    const uint h0 = kvh * group + g0;
    const uint hd = args.head_dim;
    const uint64_t plane = ((uint64_t) e[AUDIO_SEQ_PLANE] + (uint64_t) kvh * cap) * hd;
    audio_decode_attn_split(
            partials, q + (uint64_t) s * args.heads * hd,
            (device const float4 *) (k_cache + plane), (device const float4 *) (v_cache + plane),
            len, split, args.splits, h0, s * args.heads + h0,
            min(GT, group - g0), hd, args.scale,
            (uint64_t) args.seqs * args.heads * args.splits * hd, lane);
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

/// Where a lane's two elements of an 8x8 simdgroup matrix sit: row `y`,
/// columns `x` and `x + 1`. The lanes of one row differ only in bits 0 and 3,
/// so a row's reduction is the shuffles across 1 and 8.
static inline ushort2 audio_frag_xy(ushort lane) {
    const ushort quad = lane >> 2;
    return ushort2(((quad & 2u) << 1) + ((lane & 1u) << 1), (quad & 4u) + ((lane >> 1) & 3u));
}

/// Span attention on simdgroup matrices: one threadgroup per (kv head, tile
/// of `8 * row_groups` query items), an item being one (row, query head) of
/// the kv head's group, so the group's heads share every key and value the
/// tile reads. Simdgroup `(rg, cs)` takes the tile's items `8rg .. 8rg + 8`
/// over head columns `cs*W .. cs*W + W`: it holds those queries and its slice
/// of the output in registers, and a 512-wide head is several slices rather
/// than one simdgroup's 128 + 128 floats a lane, which would spill.
///
/// The keys come `KB` at a time from the union of the tile's visible ranges.
/// Each simdgroup forms its slice's partial transposed scores `K.Q^T`, so a
/// key fragment's lane reads two neighbouring floats of one key row, as a
/// value fragment's does. The partial scores go through threadgroup memory,
/// which sums the slices and hands each lane the scores of its own item row,
/// the layout the probabilities need as a left operand; every simdgroup of
/// the row group then runs the same online softmax on the same sums (the
/// summation order is fixed, so the copies agree to the bit) and adds its
/// probabilities times its slice of the values. The fragments are read
/// straight from device memory, each lane its own two elements, with the key
/// row clamped into the input; the mask gives a clamped or invisible key no
/// weight.
///
/// Measured on an M1 Max over 8192 rows, four query heads, per launch: a
/// 1025-key window at 256 wide over two kv heads, and 512 wide over one
/// kv head unbounded. Here (64 columns, 16 keys, 8 simdgroups): 14.7 and 282
/// ms. 128 columns: 15.7 and 296; 32 keys: 27.5 and 496, 8 keys 19.1 and 368
/// (32 keys' fragments cost the threadgroups a core); the scores' keys as the
/// fragments' columns, a lane's two keys a row apart, 32.7 and 561 at 32
/// keys; two item fragments a simdgroup, each key read feeding two products,
/// no better (17.3 and 267 at its best); keys and values staged through
/// threadgroup memory in 8-key blocks, 18.6 and 343 (four barriers a block).
///
/// The scores and the products are f32 throughout (the fragments' multiply
/// is an f32 FMA), and the exponential is the precise one.
template <uint W, uint KB>
static inline void audio_span_flash(
        constant audio_args_span_attn & args,
        device float *out, device const float *q,
        device const float *k, device const float *v,
        device const uint *segments, threadgroup float *part,
        threadgroup uint *bounds, uint3 tgpig, ushort lane, ushort sg) {
    const uint n = args.n;
    const uint hd = args.head_dim;
    const uint group = args.heads / args.kv_heads;
    const uint slices = (hd + W - 1u) / W;
    const uint rg = sg / slices;
    const uint cs = sg - rg * slices;
    const uint tile = args.row_groups * 8u;
    const uint items = n * group;
    const uint item0 = tgpig.y * tile;
    const uint kvh = tgpig.x;
    const uint w = args.heads * hd;
    const uint kvw = args.kv_heads * hd;

    // Each item's visible keys `[lo, hi)` and, past them, the tile's union.
    // An item past the input, or a row whose segment is empty, sees nothing.
    if (sg == 0) {
        uint lo = 0xffffffffu;
        uint hi = 0u;
        if (lane < tile) {
            const uint item = item0 + lane;
            uint ilo = 1u;
            uint ihi = 0u;
            if (item < items) {
                const uint i = item / group;
                uint s = 0u;
                uint e = n;
                if (args.segmented != 0u) {
                    s = segments[2u * i];
                    e = min(segments[2u * i + 1u], n);
                }
                // The host clamps `left` and `right` to `n`: no sum wraps.
                ilo = max(s, i > args.left ? i - args.left : 0u);
                ihi = min(e, i + args.right + 1u);
            }
            bounds[2u * lane] = ilo;
            bounds[2u * lane + 1u] = ihi;
            if (ilo < ihi) {
                lo = ilo;
                hi = ihi;
            }
        }
        lo = simd_min(lo);
        hi = simd_max(hi);
        if (lane == 0) {
            bounds[2u * AUDIO_SPAN_MAX_ITEMS] = lo;
            bounds[2u * AUDIO_SPAN_MAX_ITEMS + 1u] = hi;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ushort2 xy = audio_frag_xy(lane);
    const uint t0 = rg * 8u;
    // This lane's item row `y`: its visible keys.
    const uint my_lo = bounds[2u * (t0 + xy.y)];
    const uint my_hi = bounds[2u * (t0 + xy.y) + 1u];
    const uint jlo = bounds[2u * AUDIO_SPAN_MAX_ITEMS];
    const uint jhi = bounds[2u * AUDIO_SPAN_MAX_ITEMS + 1u];
    const uint d0 = cs * W;
    // Fragments of this slice inside the head: the head is a multiple of 32
    // wide, so a slice is whole fragments, and only the last may be short.
    const uint frags = min(W, hd - d0) / 8u;
    device const float *kp = k + kvh * hd + d0;
    device const float *vp = v + kvh * hd + d0;

    // The queries transposed, `Q^T(d, item)`: this lane holds items `x` and
    // `x + 1` at head column `8c + y`.
    simdgroup_float8x8 qf[W / 8u];
    simdgroup_float8x8 of[W / 8u];
    {
        device const float *qcol[2];
        AUDIO_UNROLL for (uint e = 0; e < 2u; ++e) {
            const uint it = min(item0 + t0 + xy.x + e, items - 1u);
            const uint r = it / group;
            qcol[e] = q + (uint64_t) r * w + (kvh * group + (it - r * group)) * hd + d0 + xy.y;
        }
        AUDIO_UNROLL for (uint c = 0; c < W / 8u; ++c) {
            float2 qv = float2(0.0f);
            if (c < frags) {
                qv = float2(qcol[0][8u * c], qcol[1][8u * c]) * args.scale;
            }
            qf[c].thread_elements()[0] = qv.x;
            qf[c].thread_elements()[1] = qv.y;
            of[c] = simdgroup_float8x8(0.0f);
        }
    }

    float m = -INFINITY;
    float l = 0.0f;
    threadgroup float *mine = part + sg * (8u * KB);
    threadgroup const float *peers = part + rg * slices * (8u * KB);
    for (uint base = jlo; base < jhi; base += KB) {
        simdgroup_float8x8 sf[KB / 8u];
        AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
            sf[b] = simdgroup_float8x8(0.0f);
        }
        // K(j, d): this lane's key `y` of each fragment, head columns `x` and
        // `x + 1`. Slice column outer, so the block's accumulators are
        // independent chains (key fragment outer measured 37.7 and 680 ms).
        device const float *krow[KB / 8u];
        AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
            krow[b] = kp + (uint64_t) min(base + 8u * b + xy.y, n - 1u) * kvw + xy.x;
        }
        AUDIO_UNROLL for (uint c = 0; c < W / 8u; ++c) {
            if (c < frags) {
                AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
                    const float2 kv = float2(*((device const packed_float2 *) (krow[b] + 8u * c)));
                    simdgroup_float8x8 kf;
                    kf.thread_elements()[0] = kv.x;
                    kf.thread_elements()[1] = kv.y;
                    simdgroup_multiply_accumulate(sf[b], kf, qf[c], sf[b]);
                }
            }
        }
        // `S^T(key, item)` in, `S(item, key)` back: every lane reads exactly
        // what the same lane of each slice's simdgroup wrote. Threadgroup
        // barriers, not simdgroup ones, around the hand-off (see
        // `kernel_flux2_flash_sdpa_f32` for what a simdgroup barrier let
        // through on an M1 Max); the second keeps the next block's writes
        // from overtaking a slow reader.
        AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
            mine[xy.x * KB + 8u * b + xy.y] = sf[b].thread_elements()[0];
            mine[(xy.x + 1u) * KB + 8u * b + xy.y] = sf[b].thread_elements()[1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float s[KB / 8u][2];
        AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
            const uint at = xy.y * KB + 8u * b + xy.x;
            float2 sum = float2(0.0f);
            for (uint p = 0; p < slices; ++p) {
                sum += float2(peers[p * 8u * KB + at], peers[p * 8u * KB + at + 1u]);
            }
            s[b][0] = sum.x;
            s[b][1] = sum.y;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float block_max = -INFINITY;
        AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
            AUDIO_UNROLL for (uint e = 0; e < 2u; ++e) {
                const uint j = base + 8u * b + xy.x + e;
                s[b][e] = j >= my_lo && j < my_hi ? s[b][e] : -INFINITY;
                block_max = max(block_max, s[b][e]);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float m_new = max(m, block_max);
        // A row that has seen only masked keys keeps a maximum of minus
        // infinity; its rescale is one so nothing becomes NaN.
        const float alpha = m_new == -INFINITY ? 1.0f : precise::exp(m - m_new);
        float block_sum = 0.0f;
        simdgroup_float8x8 pf[KB / 8u];
        AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
            AUDIO_UNROLL for (uint e = 0; e < 2u; ++e) {
                const float p = s[b][e] == -INFINITY ? 0.0f : precise::exp(s[b][e] - m_new);
                block_sum += p;
                pf[b].thread_elements()[e] = p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        l = l * alpha + block_sum;
        m = m_new;

        // An output fragment's lane holds the same item row as its
        // probability fragment's, so the rescale is per lane, with no
        // diagonal product.
        AUDIO_UNROLL for (uint c = 0; c < W / 8u; ++c) {
            of[c].thread_elements()[0] *= alpha;
            of[c].thread_elements()[1] *= alpha;
        }
        AUDIO_UNROLL for (uint b = 0; b < KB / 8u; ++b) {
            device const float *vrow = vp + (uint64_t) min(base + 8u * b + xy.y, n - 1u) * kvw + xy.x;
            AUDIO_UNROLL for (uint c = 0; c < W / 8u; ++c) {
                if (c < frags) {
                    const float2 vv = float2(*((device const packed_float2 *) (vrow + 8u * c)));
                    simdgroup_float8x8 vf;
                    vf.thread_elements()[0] = vv.x;
                    vf.thread_elements()[1] = vv.y;
                    simdgroup_multiply_accumulate(of[c], pf[b], vf, of[c]);
                }
            }
        }
    }

    const uint t = t0 + xy.y;
    if (item0 + t < items) {
        const uint it = item0 + t;
        const uint r = it / group;
        const uint h = kvh * group + (it - r * group);
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        device float *orow = out + (uint64_t) r * w + h * hd + d0 + xy.x;
        AUDIO_UNROLL for (uint c = 0; c < W / 8u; ++c) {
            if (c < frags) {
                *((device packed_float2 *) (orow + 8u * c)) = packed_float2(
                        of[c].thread_elements()[0] * inv, of[c].thread_elements()[1] * inv);
            }
        }
    }
}

/// Bidirectional grouped-query attention over a window of each row's
/// segment, with no cache: `audio_span_flash` at `AUDIO_SPAN_W`-column slices
/// and `AUDIO_SPAN_KB`-key blocks. Grid `(kv_heads, ceil(n * heads/kv_heads /
/// (8 * row_groups)))`, `row_groups * ceil(hd/W)` simdgroups.
kernel void kernel_audio_span_attn_f32(
        constant audio_args_span_attn & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *q                [[buffer(2)]],
        device const float *k                [[buffer(3)]],
        device const float *v                [[buffer(4)]],
        device const uint *segments          [[buffer(5)]],
        threadgroup float *part              [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint bounds[2u * AUDIO_SPAN_MAX_ITEMS + 2u];
    audio_span_flash<AUDIO_SPAN_W, AUDIO_SPAN_KB>(
            args, out, q, k, v, segments, part, bounds, tgpig, lane, sg);
}
