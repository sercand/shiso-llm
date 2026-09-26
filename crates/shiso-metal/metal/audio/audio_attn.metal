// Attention over a bounded key window with an optional relative-position term,
// and the half-split rotary embedding.

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
