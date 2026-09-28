// Sequential GDN delta-rule recurrence: decode and short prefill.
//
// One body, several threadgroup widths, one summation order.
//
// The CUDA original keeps the 128x129 transposed state in shared memory. Here
// the same state lives in registers: SPLIT lanes share one value column and
// lane (j, sub) owns SEG = 128 / SPLIT floats of that column's state row, so a
// threadgroup of 128 * SPLIT lanes holds exactly one full state. SPLIT is the
// only thing that differs between the instantiations at the bottom of this
// file -- 1, 2 and 4 lanes per column, 128, 256 and 512 threads per group --
// and it is a performance knob, not a numerical one.
//
// What makes it a knob is that the state row is walked as four quarters of 32
// and the rotation of each quarter is keyed to the QUARTER index, never to the
// lane that happens to be walking it. Every variant therefore evaluates the
// same four 32-FMA chains over the same four index sequences and combines them
// with the same pairing:
//
//   inner(q) = 32-FMA chain over quarter q from the rotated start (n + 8q) % 32
//   total    = (inner(0) + inner(1)) + (inner(2) + inner(3))
//
// A lane that owns four quarters folds them with q35_canonical4; a lane that
// owns two adds its pair and crosses one xor step; a lane that owns one crosses
// two. All three are the same tree, so all three produce the same bits.
//
//   dec     = exp(decay[t][h])
//   sk      = S_row . k        (four quarter chains, paired as above)
//   delta_j = (v[j] - dec * sk) * beta
//   out[j]  = (dec * sum(S q) + delta * <k,q>) * rsqrt(head_dim)
//   S[i]    = S * exp(decay) + delta * k
//
// Value head h reads query/key head h % n_head_k: modulo, not divide.
//
// SEG sizes `row`, so it has to be a compile-time template parameter rather
// than a runtime value: a runtime bound would move the state out of registers,
// which is the one thing this kernel exists to avoid.

// The order above is a claim about the ADDITIONS, and the library's default
// fast math lets the optimizer reassociate one without the source changing:
// at 128 lanes a single thread holds four independent chains, which is
// exactly the shape a vectorizer likes to re-fold. Most of the arithmetic
// here goes through the helpers in q35_common.h, which are already compiled
// safe; this covers the few additions written directly in this file, so the
// guarantee does not rest on the optimizer declining an opportunity. The
// default is restored below the instantiations -- and `fast` there means
// the build's default, not a saved state; see compile_library in
// shiso/crates/shiso-metal/src/compile.rs.
#pragma METAL fp math_mode(safe)

template <uint SPLIT, uint SEG, bool TRAIL>
static inline void q35_gdn_recurrent_impl(
        constant q35_args_gdn_recurrent & args,
        device float *attn_out,
        device float *state,
        device const float *qkv,
        device const float *decay,
        device const float *beta,
        device float *trail,
        threadgroup float *tbuf,
        uint3 tgpig,
        ushort3 tpitg) {
    const uint head_dim = args.head_dim;
    // Both the register layout and the rotated sweep assume one quarter of 128.
    if (head_dim != Q35_GDN_D) {
        return;
    }
    const uint tid = tpitg.x;
    const uint h = tgpig.x;
    // Value head h reads query/key head h % n_head_k: modulo, not divide.
    const uint hk = h % args.n_head_k;
    const uint seg = head_dim / SPLIT;
    const uint j = tid / SPLIT;
    const uint sub = tid % SPLIT;
    const uint i0 = sub * seg;
    // Quarters of the state row this lane owns, and the global index of the
    // first of them. At SPLIT 4 this is one quarter and the quarter loops run
    // once, which is the reference shape; at SPLIT 1 one lane owns all four.
    // 1, 2 or 4 lanes per column, and nothing else: at SPLIT 8 `quarters`
    // would be 0, both quarter loops would be skipped, `part_k[0]` would be
    // read uninitialised and the combination below would fall into the 2-xor
    // arm -- a wrong answer with no diagnostic. Caught at compile time so a
    // future instantiation has to come back here and say what its order is.
    static_assert(SPLIT == 1u || SPLIT == 2u || SPLIT == 4u,
                  "q35_gdn_recurrent_impl is written for SPLIT in {1, 2, 4}");
    const uint quarters = 4u / SPLIT;
    const uint q_first = sub * quarters;
    const float scale = rsqrt((float) head_dim);

    threadgroup float *scratch = tbuf;                  // Q35_BLOCK_SUM_SLOTS
    threadgroup float *s_k = tbuf + Q35_BLOCK_SUM_SLOTS;
    threadgroup float *s_q = tbuf + Q35_BLOCK_SUM_SLOTS + Q35_GDN_D;

    // One quarter of the transposed state per lane, in registers.
    device float *S = state + (uint64_t) h * head_dim * head_dim
            + (uint64_t) j * head_dim + i0;
    float row[SEG];
    for (uint n = 0u; n < seg; n++) {
        // The base pointer already carries the lane's quarter offset, so the
        // register index is relative to it: the CUDA twin indexes its unoffset
        // shared row with the absolute key index, which is the same address.
        row[n] = S[n];
    }
    for (uint t = 0u; t < args.n_tok; t++) {
        device const float *base = qkv + (uint64_t) t * args.stride;
        device const float *q_d = base + (uint64_t) hk * head_dim;
        device const float *k_d = base + args.k_offset + (uint64_t) hk * head_dim;
        device const float *v_d = base + args.v_offset + (uint64_t) h * head_dim;

        // The sub == 0 lanes own one dimension each and publish k and q with it.
        if (sub == 0u && j < head_dim) {
            s_k[j] = k_d[j];
            s_q[j] = q_d[j];
        }
        // <k,q> is needed by every lane, and its summation order has to be the
        // same one at every width. The block sum's tree is fixed by WHICH lane
        // holds which product, so the products are laid down on the first
        // Q35_GDN_D lanes -- a dense prefix that does not move when SPLIT does
        // -- and the reduction is told that length rather than the threadgroup's.
        // Keyed to `sub` instead, dimension d would sit at lane d * SPLIT and
        // the tree would be a different tree at every width; that is the same
        // mistake the state sweep makes when its rotation is keyed to the lane.
        const float kq_part = (tid < Q35_GDN_D) ? (k_d[tid] * q_d[tid]) : 0.0f;
        const float s_kq = q35_block_sum(kq_part, scratch, tid, Q35_GDN_D);

        const uint64_t gi = (uint64_t) t * args.n_head_v + h;
        const float dec = exp(decay[gi]);
        const float bt = beta[gi];

        // One independent 32-FMA chain per quarter this lane owns, each walked
        // from a start rotated by the QUARTER index. Keying the rotation to
        // the quarter rather than to `sub` is what makes the chain the same
        // sequence whichever lane walks it, and so what makes SPLIT free.
        float part_k[4];
        float part_q[4];
        for (uint qi = 0u; qi < quarters; qi++) {
            const uint q = q_first + qi;                  // quarter, not lane
            const uint qbase = q * Q35_GDN_QUARTER;
            // The register index is relative to S, which already carries this
            // lane's i0; the published rows are indexed absolutely.
            const uint r0 = qi * Q35_GDN_QUARTER;
            float sk_q = 0.0f;
            float sq_q = 0.0f;
            for (uint n = 0u; n < Q35_GDN_QUARTER; n++) {
                const uint rot = (n + 8u * q) & (Q35_GDN_QUARTER - 1u);
                const float mv = row[r0 + rot];
                sk_q = q35_fma(mv, s_k[qbase + rot], sk_q);
                sq_q = q35_fma(mv, s_q[qbase + rot], sq_q);
            }
            part_k[qi] = sk_q;
            part_q[qi] = sq_q;
        }
        // Combine what this lane holds, then cross the remaining lanes. Both
        // halves use the same pairing, so the tree is the same tree however
        // the quarters are distributed over lanes. SPLIT is a compile-time
        // parameter, so exactly one arm survives compilation.
        float sk;
        float sq;
        if (SPLIT == 1u) {
            sk = q35_canonical4(part_k[0], part_k[1], part_k[2], part_k[3]);
            sq = q35_canonical4(part_q[0], part_q[1], part_q[2], part_q[3]);
        } else if (SPLIT == 2u) {
            sk = part_k[0] + part_k[1];
            sq = part_q[0] + part_q[1];
            sk += simd_shuffle_xor(sk, 1);
            sq += simd_shuffle_xor(sq, 1);
        } else {
            sk = part_k[0];
            sq = part_q[0];
            sk += simd_shuffle_xor(sk, 1);
            sk += simd_shuffle_xor(sk, 2);
            sq += simd_shuffle_xor(sq, 1);
            sq += simd_shuffle_xor(sq, 2);
        }
        const float delta = (v_d[j] - dec * sk) * bt;

        if (sub == 0u) {
            attn_out[((uint64_t) t * args.n_head_v + h) * head_dim + j] =
                    (dec * sq + delta * s_kq) * scale;
        }
        // The carry walks the same rotated per-quarter sequence as the read
        // above; anything else would pair a register with the wrong key.
        for (uint qi = 0u; qi < quarters; qi++) {
            const uint q = q_first + qi;
            const uint qbase = q * Q35_GDN_QUARTER;
            const uint r0 = qi * Q35_GDN_QUARTER;
            for (uint n = 0u; n < Q35_GDN_QUARTER; n++) {
                const uint rot = (n + 8u * q) & (Q35_GDN_QUARTER - 1u);
                row[r0 + rot] = q35_fma(row[r0 + rot], dec,
                                        delta * s_k[qbase + rot]);
            }
        }
        // The state after this row, into its record of the trail - every row
        // but the last, whose record is the state the launch leaves behind.
        // The registers are exactly what a launch over rows 0..=t would have
        // written back, so a record here is bit for bit the per-row walk's.
        if (TRAIL && t + 1u < args.n_tok) {
            device float *R = trail + (uint64_t) t * args.n_head_v * head_dim * head_dim
                    + (uint64_t) h * head_dim * head_dim + (uint64_t) j * head_dim + i0;
            for (uint n = 0u; n < seg; n++) {
                R[n] = row[n];
            }
        }
        // k and q are rewritten by the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint n = 0u; n < seg; n++) {
        S[n] = row[n];
    }
}

// One recurrent body, instantiated per lane split. SPLIT lanes share a value
// column; each lane owns SEG = Q35_GDN_D / SPLIT floats of the state row.
#define Q35_GDN_RECURRENT_BODY(NAME, SPLIT, SEG)                              \
kernel void NAME(                                                             \
        constant q35_args_gdn_recurrent & args [[buffer(0)]],                 \
        device float *attn_out [[buffer(1)]],                                 \
        device float *state [[buffer(2)]],                                    \
        device const float *qkv [[buffer(3)]],                                \
        device const float *decay [[buffer(4)]],                              \
        device const float *beta [[buffer(5)]],                               \
        threadgroup float *tbuf [[threadgroup(0)]],                           \
        uint3 tgpig[[threadgroup_position_in_grid]],                          \
        ushort3 tpitg[[thread_position_in_threadgroup]]) {                    \
    q35_gdn_recurrent_impl<SPLIT, SEG, false>(                                \
            args, attn_out, state, qkv, decay, beta, nullptr, tbuf, tgpig, tpitg); \
}

// The same body, also writing the state after every row but the last into a
// trail of `n_tok - 1` state records: what a verify pass needs to wind back to
// any row, in one launch instead of one per row.
#define Q35_GDN_RECURRENT_TRAIL_BODY(NAME, SPLIT, SEG)                        \
kernel void NAME(                                                             \
        constant q35_args_gdn_recurrent & args [[buffer(0)]],                 \
        device float *attn_out [[buffer(1)]],                                 \
        device float *state [[buffer(2)]],                                    \
        device const float *qkv [[buffer(3)]],                                \
        device const float *decay [[buffer(4)]],                              \
        device const float *beta [[buffer(5)]],                               \
        device float *trail [[buffer(6)]],                                    \
        threadgroup float *tbuf [[threadgroup(0)]],                           \
        uint3 tgpig[[threadgroup_position_in_grid]],                          \
        ushort3 tpitg[[thread_position_in_threadgroup]]) {                    \
    q35_gdn_recurrent_impl<SPLIT, SEG, true>(                                 \
            args, attn_out, state, qkv, decay, beta, trail, tbuf, tgpig, tpitg); \
}

Q35_GDN_RECURRENT_BODY(kernel_q35_gdn_recurrent_s1, 1u, 128u)
Q35_GDN_RECURRENT_BODY(kernel_q35_gdn_recurrent_s2, 2u, 64u)
Q35_GDN_RECURRENT_BODY(kernel_q35_gdn_recurrent_s4, 4u, 32u)
Q35_GDN_RECURRENT_TRAIL_BODY(kernel_q35_gdn_recurrent_trail_s1, 1u, 128u)
Q35_GDN_RECURRENT_TRAIL_BODY(kernel_q35_gdn_recurrent_trail_s2, 2u, 64u)
Q35_GDN_RECURRENT_TRAIL_BODY(kernel_q35_gdn_recurrent_trail_s4, 4u, 32u)

// Back to the library-wide default for the files concatenated after this one.
#pragma METAL fp math_mode(fast)
