// Chunked-parallel gated delta rule.
//
// One body, four threadgroup widths, one summation order.
//
// Same recurrence as kernel_q35_gdn_recurrent, unrolled C tokens at a time:
//
//   G_t     = sum_{s<=t} log decay_s                     (cumulative log decay)
//   A[t][u] = b_t e^{G_t - G_u} (k_u . k_t)              (u < t)
//   d_t     = b_t (v_t - e^{G_t} S0^T k_t) - sum_{u<t} A[t][u] d_u
//   P[t][u] = e^{G_t - G_u} (q_t . k_u)                  (u <= t)
//   o_t     = (e^{G_t} S0^T q_t + sum_{u<=t} P[t][u] d_u) / sqrt(D)
//   S       <- e^{G_last} S0 + sum_t e^{G_last - G_t} k_t d_t^T
//
// Every exponent is a difference G_t - G_u with u <= t, i.e. a decay in [0, 1]:
// nothing to overflow, which is the reason for this arrangement rather than the
// textbook "divide the keys by the cumulative decay" one.
//
// The CUDA original stages the keys, the queries and the deltas in shared
// memory.  Here the keys and queries are read from the device instead: the
// staging arrays are a bandwidth cache in CUDA, not arithmetic, and the device
// holds the same values.  s_A is the only array that must be threadgroup
// resident, because every warp reads every other warp's entries.  The deltas
// become register-owned: lane `sub` keeps the deltas for t == sub (mod CSPLIT),
// which is exactly the set the substitution and the output phase read back per
// lane; the carry needs every t, so it reaches them through a broadcast.
//
// --- what makes CSPLIT a knob -----------------------------------------------
//
// The canonical unit of this kernel is the CSPLIT = 8 lane's chain: four float4
// slots of the state row, walked from a rotated start, with the four float4
// components kept as four separate partials and folded (p0 + p1) + (p2 + p3).
// That per-component grouping is not an implementation detail -- it is what
// reproduces the CUDA original's float4 arithmetic -- so it is part of the
// order every variant must reproduce.  A 128-float row holds Q35_GDN_CSPLIT of
// those four-slot units, and a lane has to hold a WHOLE number of them, so
// CSPLIT may be 1, 2, 4 or 8 (128, 256, 512 or 1024 lanes) and no more: 16
// would split a unit in half and there would be no CUDA order left to copy.
//
// A lane at CSPLIT < 8 therefore runs `units = 8 / CSPLIT` independent
// four-slot chains and combines them the way the cross-lane butterfly would
// have.  Which lane gets which unit is the whole of the parameterization, and
// it is INTERLEAVED: lane `sub` owns units sub, sub + CSPLIT, sub + 2*CSPLIT,
// ...  so that a unit's index has `sub` in its low bits and the lane's own unit
// counter `ui` in its high ones.  That matters because the reference
// combination is a recursive-doubling xor butterfly, which pairs by bit 0
// first, then bit 1, then bit 2; with the interleaved assignment those first
// bits are exactly the lane bits, so the butterfly runs first and the lane's
// own units fold afterwards, and the tree is the CSPLIT = 8 tree at every
// width.  Blocking the units instead (lane `sub` taking units sub*units ..) is
// the same tree read the other way round, but it puts the deltas of token t in
// a lane that no longer satisfies `sub == t % CSPLIT`, and the output phase
// combines a lane's delta sum with its state-row sum BEFORE the butterfly --
// so the two would have to be shuffled back together.  Interleaved keeps them
// in the same lane, which is why `dreg` can stay indexed by t / CSPLIT.
//
// The other two phases need no parameterization at all.  The A matrix and the
// P tile reduce inside one 32-lane simdgroup through q35_warp_sum, whose tree
// is fixed by the simdgroup and not by the threadgroup, and the row-to-warp
// handout only decides WHICH warp computes a value, never what it is.  The
// strided gate loop is a load, not an accumulation.  So they are order-safe at
// any width and appear here unchanged.

// The order above is a claim about the ADDITIONS, and the library's default
// fast math lets the optimizer reassociate one without the source changing:
// at CSPLIT 1 a single thread holds eight independent chains, which is exactly
// the shape a vectorizer likes to re-fold, and the per-component p0..p3
// partials are that shape again one level down. Most of the arithmetic here
// goes through the helpers in q35_common.h, which are already compiled safe;
// this covers the additions written directly in this file, so the guarantee
// does not rest on the optimizer declining an opportunity. The default is
// restored below the instantiations -- and `fast` there means the build's
// default, not a saved state; see compile_library in
// shiso/crates/shiso-metal/src/compile.rs.
#pragma METAL fp math_mode(safe)

/// Slots in the canonical four-slot unit.  Four float4 slots, sixteen floats,
/// and Q35_GDN_CSPLIT units tile one Q35_GDN_D row.
#define Q35_GDN_UNIT_SLOTS 4u

/// Fold this lane's per-unit partials into the total for its value column.
///
/// Unit index `u = ui * CSPLIT + sub`, so `sub` occupies the low bits of `u`
/// and `ui` the high ones.  The reference combination at CSPLIT = 8 is a
/// recursive-doubling xor butterfly over the eight units, which pairs by bit 0,
/// then bit 1, then bit 2; splitting that by where the bits live gives the two
/// loops below, in this order and only in this order.  `part` is clobbered.
///
/// Both loops vanish at the ends of the range -- no butterfly at CSPLIT 1, no
/// fold at CSPLIT 8 -- so the reference width really does compile to the
/// original three shuffles and nothing else.
template <uint CSPLIT>
static inline __attribute__((always_inline)) float q35_gdn_chunk_fold(
        thread float *part) {
    const uint units = Q35_GDN_CSPLIT / CSPLIT;
    for (uint ui = 0u; ui < units; ui++) {
        for (uint o = 1u; o < CSPLIT; o <<= 1u) {
            part[ui] += simd_shuffle_xor(part[ui], o);
        }
    }
    for (uint step = 1u; step < units; step <<= 1u) {
        for (uint i = 0u; i + step < units; i += 2u * step) {
            part[i] += part[i + step];
        }
    }
    return part[0];
}

/// The chunk body, parameterized on the lane split and on where the deltas live.
///
/// `dreg` is the per-column delta store, either a `thread float *` into the
/// caller's register array or a `device float *` into a scratch buffer, so the
/// body is written once and the hybrid does not become a second copy of it that
/// can drift.  `DSTRIDE` is the distance between two of this lane's delta slots:
/// 1 for the register array, and the row pitch for a device scratch, where
/// laying the deltas out as [token][column] is what keeps the 128 lanes of a
/// CSPLIT = 1 threadgroup reading one contiguous span instead of 128 lines.
///
/// `always_inline` is load-bearing for the register variants rather than a
/// hint: `dreg` is an array in the caller, and only inlining lets it stop
/// being an addressable object and go back to being registers.
template <uint CSPLIT, uint DSTRIDE, typename DREG>
static inline __attribute__((always_inline)) void q35_gdn_chunk_impl(
        constant q35_args_gdn_chunk & args,
        device float *attn_out,
        device float *state,
        device const float *qkv,
        device const float *decay,
        device const float *beta,
        DREG dreg,
        threadgroup char *tbuf,
        uint3 tgpig,
        ushort3 tpitg,
        ushort3 ntg) {
    // 1, 2, 4 or 8 lanes per value column, and nothing else. At CSPLIT 16
    // `units` would be 0, every unit loop would be skipped, the fold would
    // read part[0] uninitialised and the answer would be wrong with no
    // diagnostic. Caught at compile time so that a future instantiation has to
    // come back here and say what its order is.
    static_assert(CSPLIT == 1u || CSPLIT == 2u || CSPLIT == 4u || CSPLIT == 8u,
                  "q35_gdn_chunk_impl is written for CSPLIT in {1, 2, 4, 8}");
    const uint tid = tpitg.x;
    const uint D = args.head_dim;
    // The four-slot unit is sixteen floats, an eighth of a 128-float row, and
    // the unit count Q35_GDN_CSPLIT is derived from that row: at any other
    // head_dim a lane would own a fractional number of units and there would be
    // no CSPLIT = 8 order to reproduce. The recurrent kernel refuses the same
    // way.
    if (D != Q35_GDN_D) {
        return;
    }
    const uint C = Q35_GDN_CHUNK;
    const uint n_tok = args.n_tok;
    const uint stride = args.stride;
    const uint h = tgpig.x;
    const uint hk = h % args.n_head_k;
    const uint n_head_v = args.n_head_v;
    const uint j = tid / CSPLIT;
    const uint sub = tid % CSPLIT;
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    const uint n_warp = ntg.x >> 5u;
    const float scale = rsqrt((float) D);
    // Four-slot units per lane, interleaved: this lane walks units `sub`,
    // `sub + CSPLIT`, ... so the unit index carries `sub` in its low bits.
    const uint units = Q35_GDN_CSPLIT / CSPLIT;

    threadgroup float *smem = (threadgroup float *) tbuf;
    threadgroup float *s_A = smem;          // [C][C]
    threadgroup float *s_G = smem + C * C;  // inclusive prefix sum of log decay
    threadgroup float *s_b = s_G + C;

    device const float *k_base = qkv + args.k_offset + (uint64_t) hk * D;
    device const float *q_base = qkv + (uint64_t) hk * D;
    device const float *v_base = qkv + args.v_offset;

    // This lane's slice of the state: `units` four-slot units of its value
    // column's row, each walked in the order that unit's rotation gives, so
    // that the sequence belongs to the UNIT and not to the lane that happens
    // to hold it -- which is what makes CSPLIT free.  The slots are loaded
    // component-wise rather than as a float4 because the host may hand out a
    // buffer view at any 4-byte offset; the arithmetic grouping is unchanged
    // by that.
    device float *m_row = state + ((uint64_t) h * D + j) * D;
    float4 sreg[Q35_GDN_UNIT_SLOTS * (Q35_GDN_CSPLIT / CSPLIT)];
    for (uint ui = 0u; ui < units; ui++) {
        const uint usub = ui * CSPLIT + sub;
        const uint urot = usub / 2u;
        for (uint n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
            const uint slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
            device const float *m = m_row
                    + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
            sreg[ui * Q35_GDN_UNIT_SLOTS + n4] =
                    float4{m[0u], m[1u], m[2u], m[3u]};
        }
    }

    for (uint t0 = 0u; t0 < n_tok; t0 += C) {
        const uint n = (n_tok - t0 < C) ? (n_tok - t0) : C;

        // Gates in parallel, then the inclusive prefix sum of the log decays by
        // one thread over threadgroup memory: 64 dependent adds are cheaper
        // than a scan's barriers, but 64 strided global loads on that same
        // serial path are not.  The strided handout is a load, so the width it
        // is strided by does not reach the arithmetic.
        for (uint t = tid; t < n; t += ntg.x) {
            const uint64_t gi = (uint64_t) (t0 + t) * n_head_v + h;
            s_G[t] = decay[gi];
            s_b[t] = beta[gi];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            float acc = 0.0f;
            for (uint t = 0u; t < n; t++) {
                acc += s_G[t];
                s_G[t] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A[t][u] = b_t e^{G_t - G_u} (k_u . k_t), strictly below the diagonal.
        // One warp per row, its own key row hoisted into registers, so each
        // column costs one read of the other row.  Rows are handed out low/high
        // so a warp that gets both halves of the run costs the same amount.
        // The reduction is q35_warp_sum over one simdgroup, so the value of an
        // entry does not depend on how many warps the threadgroup has; only on
        // which warp computes it, and that is a schedule, not a sum.
        for (uint rr = warp; rr < C; rr += n_warp) {
            const uint t = ((rr / n_warp) % 2u == 0u) ? rr : (C - 1u - (rr - n_warp));
            if (t >= n) {
                continue;
            }
            device const float *k_t = k_base + (uint64_t) (t0 + t) * stride;
            const float bt = s_b[t];
            const float gt = s_G[t];
            for (uint u = 0u; u < t; u++) {
                device const float *k_u = k_base + (uint64_t) (t0 + u) * stride;
                const uint i = 4u * lane;
                float dot = k_t[i] * k_u[i];
                dot += k_t[i + 1u] * k_u[i + 1u];
                dot += k_t[i + 2u] * k_u[i + 2u];
                dot += k_t[i + 3u] * k_u[i + 3u];
                dot = q35_warp_sum(dot);
                if (lane == 0u) {
                    s_A[t * C + u] = bt * exp(gt - s_G[u]) * dot;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // rhs_t = b_t (v_t - e^{G_t} S0^T k_t).  One four-slot chain per unit
        // this lane owns, each with its four float4 components kept as four
        // partials, so the grouping CUDA gets from float4 arithmetic is
        // reproduced rather than re-associated; then the units fold in the
        // butterfly's order.  The fold leaves the same total in every lane, so
        // the owner test below is free to be any partition of the tokens.
        for (uint t = 0u; t < n; t++) {
            device const float *k_row = k_base + (uint64_t) (t0 + t) * stride;
            device const float *v_row = v_base + (uint64_t) (t0 + t) * stride + (uint64_t) h * D;
            float part[Q35_GDN_CSPLIT / CSPLIT];
            for (uint ui = 0u; ui < units; ui++) {
                const uint usub = ui * CSPLIT + sub;
                const uint urot = usub / 2u;
                float p0 = 0.0f;
                float p1 = 0.0f;
                float p2 = 0.0f;
                float p3 = 0.0f;
                for (uint n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
                    const uint slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
                    device const float *k4 = k_row
                            + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
                    const float4 sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
                    p0 = q35_fma(sv.x, k4[0u], p0);
                    p1 = q35_fma(sv.y, k4[1u], p1);
                    p2 = q35_fma(sv.z, k4[2u], p2);
                    p3 = q35_fma(sv.w, k4[3u], p3);
                }
                part[ui] = q35_canonical4(p0, p1, p2, p3);
            }
            const float p = q35_gdn_chunk_fold<CSPLIT>(part);
            if (sub == (t & (CSPLIT - 1u))) {
                dreg[(t / CSPLIT) * DSTRIDE] = s_b[t] * (v_row[j] - p * exp(s_G[t]));
            }
        }

        // Forward substitution: d_t = rhs_t - sum_{u<t} A[t][u] d_u.  C
        // dependent steps, but each is one rank-1 update over the value
        // dimension instead of a pass through the state, and that trade is what
        // the chunk buys.  Lane sub owns the deltas for t == sub (mod CSPLIT),
        // which is exactly the set this loop reads back for this lane, so the
        // __syncwarp() the shared-memory version needs between the steps has
        // nothing left to order and is gone rather than translated.
        //
        // The chains are the CSPLIT = 8 chains: one per unit index modulo
        // Q35_GDN_CSPLIT, never one long chain over the lane's whole residue
        // class.  At CSPLIT 8 the outer loop runs once and this is the
        // original `u = sub; u += 8` sweep.
        for (uint t = 1u; t < n; t++) {
            float part[Q35_GDN_CSPLIT / CSPLIT];
            for (uint ui = 0u; ui < units; ui++) {
                const uint usub = ui * CSPLIT + sub;
                float acc = 0.0f;
                for (uint u = usub; u < t; u += Q35_GDN_CSPLIT) {
                    acc = q35_fma(s_A[t * C + u], dreg[(u / CSPLIT) * DSTRIDE], acc);
                }
                part[ui] = acc;
            }
            const float acc = q35_gdn_chunk_fold<CSPLIT>(part);
            if (sub == (t & (CSPLIT - 1u))) {
                dreg[(t / CSPLIT) * DSTRIDE] -= acc;
            }
        }

        // The tile below is staged into the same s_A array the substitution above
        // has just finished reading, so a barrier is needed between the two
        // phases: the query staging that carried it in CUDA is gone, but the
        // aliasing is the same, and without it a warp that reaches the tile
        // first overwrites rows another warp is still substituting.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Outputs, one query tile at a time.  CUDA stages the queries so the
        // keys can stay resident; both reads go to the device here, which holds
        // the same values, and P goes through the same s_A array.
        for (uint tb = 0u; tb < n; tb += Q35_GDN_QTILE) {
            const uint nt = (n - tb < Q35_GDN_QTILE) ? (n - tb) : Q35_GDN_QTILE;
            for (uint pair = warp; pair < nt * C; pair += n_warp) {
                const uint tt = pair / C;
                const uint u = pair - tt * C;
                const uint t = tb + tt;
                float v = 0.0f;
                if (u <= t) {
                    device const float *q_row = q_base + (uint64_t) (t0 + t) * stride;
                    device const float *k_u = k_base + (uint64_t) (t0 + u) * stride;
                    const uint i = 4u * lane;
                    float dot = q_row[i] * k_u[i];
                    dot += q_row[i + 1u] * k_u[i + 1u];
                    dot += q_row[i + 2u] * k_u[i + 2u];
                    dot += q_row[i + 3u] * k_u[i + 3u];
                    dot = q35_warp_sum(dot);
                    v = exp(s_G[t] - s_G[u]) * dot;
                }
                if (lane == 0u) {
                    s_A[tt * C + u] = v;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // The state term and the delta term of one unit meet in a single
            // fma BEFORE anything crosses, so both have to belong to the same
            // unit index: the interleaved handout is what puts the unit's
            // four-slot chain and the deltas of `u == usub (mod 8)` in the
            // same lane, and it is the reason this file interleaves rather
            // than blocks.
            for (uint tt = 0u; tt < nt; tt++) {
                const uint t = tb + tt;
                device const float *q_row = q_base + (uint64_t) (t0 + t) * stride;
                const float eg = exp(s_G[t]);
                float part[Q35_GDN_CSPLIT / CSPLIT];
                for (uint ui = 0u; ui < units; ui++) {
                    const uint usub = ui * CSPLIT + sub;
                    const uint urot = usub / 2u;
                    float p0 = 0.0f;
                    float p1 = 0.0f;
                    float p2 = 0.0f;
                    float p3 = 0.0f;
                    for (uint n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
                        const uint slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
                        device const float *q4 = q_row
                                + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
                        const float4 sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
                        p0 = q35_fma(sv.x, q4[0u], p0);
                        p1 = q35_fma(sv.y, q4[1u], p1);
                        p2 = q35_fma(sv.z, q4[2u], p2);
                        p3 = q35_fma(sv.w, q4[3u], p3);
                    }
                    const float p = q35_canonical4(p0, p1, p2, p3);
                    float acc = 0.0f;
                    for (uint u = usub; u <= t; u += Q35_GDN_CSPLIT) {
                        acc = q35_fma(s_A[tt * C + u], dreg[(u / CSPLIT) * DSTRIDE], acc);
                    }
                    part[ui] = q35_fma(p, eg, acc);
                }
                const float acc = q35_gdn_chunk_fold<CSPLIT>(part);
                if (sub == 0u) {
                    attn_out[((uint64_t) (t0 + t) * n_head_v + h) * D + j] = acc * scale;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        // Carry: S <- e^{G_last} S0 + sum_t e^{G_last - G_t} k_t d_t^T.  The
        // delta for token t lives in the lane with sub == t % CSPLIT, so it is
        // read back through a broadcast instead of a shared array.  The
        // broadcast stays inside the CSPLIT lanes of one value column, so the
        // shuffle only rewrites the low bits of the lane index.
        //
        // Every slot of the carry is its own independent fma chain, so this
        // phase has no cross-unit order to preserve -- only the pairing of a
        // register with the key component it was read against, which is why it
        // walks the same rotated per-unit sequence as the phases above.
        const float g_last = s_G[n - 1u];
        const float dec = exp(g_last);
        for (uint i = 0u; i < Q35_GDN_UNIT_SLOTS * units; i++) {
            sreg[i].x *= dec;
            sreg[i].y *= dec;
            sreg[i].z *= dec;
            sreg[i].w *= dec;
        }
        for (uint t = 0u; t < n; t++) {
            const uint owner = (lane & ~(CSPLIT - 1u)) | (t & (CSPLIT - 1u));
            const float d = simd_shuffle(dreg[(t / CSPLIT) * DSTRIDE], (int) owner)
                    * exp(g_last - s_G[t]);
            device const float *k_row = k_base + (uint64_t) (t0 + t) * stride;
            for (uint ui = 0u; ui < units; ui++) {
                const uint usub = ui * CSPLIT + sub;
                const uint urot = usub / 2u;
                for (uint n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
                    const uint slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
                    device const float *k4 = k_row
                            + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
                    thread float4 &sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
                    sv.x = q35_fma(k4[0u], d, sv.x);
                    sv.y = q35_fma(k4[1u], d, sv.y);
                    sv.z = q35_fma(k4[2u], d, sv.z);
                    sv.w = q35_fma(k4[3u], d, sv.w);
                }
            }
        }
        // The next run rewrites s_G, s_b and s_A while the carry above is
        // still reading s_G and the last tile is still reading s_A, so the two
        // runs need a barrier between them rather than only the one at the top
        // of the next one.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint ui = 0u; ui < units; ui++) {
        const uint usub = ui * CSPLIT + sub;
        const uint urot = usub / 2u;
        for (uint n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
            const uint slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
            device float *m = m_row
                    + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
            const float4 sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
            m[0u] = sv.x;
            m[1u] = sv.y;
            m[2u] = sv.z;
            m[3u] = sv.w;
        }
    }
}

// One chunk body, instantiated per lane split.  CSPLIT lanes share a value
// column and each lane owns Q35_GDN_CSPLIT / CSPLIT of the row's four-slot
// units, so the threadgroup is 128 * CSPLIT lanes wide.  The deltas are a
// register array here, one slot per token this lane owns, which is the whole
// reason the widths differ in cost: at CSPLIT 1 that array is 64 floats beside
// a 128-float state slice.
#define Q35_GDN_CHUNK_BODY(NAME, CSPLIT)                                      \
kernel void NAME(                                                             \
        constant q35_args_gdn_chunk & args [[buffer(0)]],                     \
        device float *attn_out [[buffer(1)]],                                 \
        device float *state [[buffer(2)]],                                    \
        device const float *qkv [[buffer(3)]],                                \
        device const float *decay [[buffer(4)]],                              \
        device const float *beta [[buffer(5)]],                               \
        threadgroup char *tbuf [[threadgroup(0)]],                            \
        uint3 tgpig[[threadgroup_position_in_grid]],                          \
        ushort3 tpitg[[thread_position_in_threadgroup]],                      \
        ushort3 ntg[[threads_per_threadgroup]]) {                             \
    float dreg[Q35_GDN_CHUNK / (CSPLIT)];                                     \
    q35_gdn_chunk_impl<CSPLIT, 1u>(                                           \
            args, attn_out, state, qkv, decay, beta, dreg, tbuf,              \
            tgpig, tpitg, ntg);                                               \
}

Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c1, 1u)
Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c2, 2u)
Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c4, 4u)
Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c8, 8u)

// Back to the library-wide default for the files concatenated after this one.
#pragma METAL fp math_mode(fast)
