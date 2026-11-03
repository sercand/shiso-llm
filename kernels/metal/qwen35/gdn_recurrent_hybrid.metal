// A recurrent variant that fits every device: one quarter of the state
// staged in threadgroup memory, the other three in registers, 128 lanes.
//
// It is the Task 3 body (gdn_recurrent.metal) at SPLIT = 1: one lane per
// value column, the state row walked as four quarters, the rotation keyed
// to the quarter index, the same q35_canonical4 pairing. The one thing that
// differs is WHERE one of the four quarters lives. Quarter 0 sits in
// threadgroup memory (128 lanes * 32 floats * 4 B = 16384 B) and quarters
// 1, 2 and 3 stay in registers, so a device that cannot afford the
// all-register 128-float row per lane can still afford this one -- and so
// that the staged quarter plus the 1156 B of scratch this kernel already uses
// (17540 B total) stays comfortably inside the 32768 B threadgroup memory
// every Apple Silicon GPU (Mac GPU family 2) is built to, rather than
// against whatever one development machine happens to accept. Staging two
// quarters (32768 B of staging alone, 33924 B total) is over that family
// limit on every M-series part, not just small ones: `run.rs`'s dispatch
// path now refuses an over-limit smem request instead of handing an
// unvalidated length to the driver, so getting this wrong no longer means
// "it happened to run on the machine in the room."
//
// This exists so that no device is refused. It is expected to lose to the
// wider register-resident shapes, since a threadgroup read costs more than a
// register read on every FMA that touches the staged quarter -- but the data
// does not back "every benchmark": measured on this M1 Max, at `n_tok = 32`
// this is the *fastest* recurrent shape, 782.46 us against `_s1`'s 944.96,
// and at `n_tok = 1` the two are within the measurement floor's own noise.

// The order this file claims is the same order gdn_recurrent.metal claims,
// and the library's default fast math is just as free to reassociate the
// additions written directly in this file as it is there. See the note at
// #pragma METAL fp math_mode(safe) in gdn_recurrent.metal, and at
// compile_library in shiso/crates/shiso-metal/src/compile.rs, before
// touching either pragma: `fast` below restores the BUILD DEFAULT, not a
// saved state, and full_source concatenates files in a fixed order, so a
// stray `fast` here would silently turn fast math back on for every q35 and
// vision source that follows this file.
#pragma METAL fp math_mode(safe)

/// The Task 3 SPLIT = 1 body, with quarter 0 of the state row staged in
/// threadgroup memory instead of kept in registers.
///
/// `staged` is per-lane private: lane `tid` owns exactly
/// `staged[tid * 32 .. tid * 32 + 32)` and no other lane ever reads or
/// writes that span, so no barrier is needed around it beyond the one this
/// function already takes at the end of every token for `s_k`/`s_q`, which
/// every lane publishes to and every lane reads back.
static inline void q35_gdn_recurrent_staged_impl(
        constant q35_args_gdn_recurrent & args,
        device float *attn_out,
        device float *state,
        device const float *qkv,
        device const float *decay,
        device const float *beta,
        threadgroup float *tbuf,
        threadgroup float *staged,
        uint3 tgpig,
        ushort3 tpitg,
        ushort3 ntg) {
    (void) ntg;
    const uint head_dim = args.head_dim;
    // Both the register layout and the rotated sweep assume one quarter of 128.
    if (head_dim != Q35_GDN_D) {
        return;
    }
    const uint tid = tpitg.x;
    const uint h = tgpig.x;
    // Value head h reads query/key head h % n_head_k: modulo, not divide.
    const uint hk = h % args.n_head_k;
    // SPLIT is fixed at 1 here: one lane owns the whole column, all four
    // quarters of it.
    const uint j = tid;
    const float scale = rsqrt((float) head_dim);

    threadgroup float *scratch = tbuf;                  // Q35_BLOCK_SUM_SLOTS
    threadgroup float *s_k = tbuf + Q35_BLOCK_SUM_SLOTS;
    threadgroup float *s_q = tbuf + Q35_BLOCK_SUM_SLOTS + Q35_GDN_D;
    // This lane's slice of the staged quarter: quarter 0 of its row.
    threadgroup float *my_staged = staged + tid * Q35_GDN_QUARTER;

    device float *S = state + (uint64_t) h * head_dim * head_dim
            + (uint64_t) j * head_dim;
    // Quarter 0 in threadgroup memory, quarters 1, 2 and 3 in registers.
    for (uint n = 0u; n < Q35_GDN_QUARTER; n++) {
        my_staged[n] = S[n];
    }
    float row_hi[96];
    for (uint n = 0u; n < 96u; n++) {
        row_hi[n] = S[Q35_GDN_QUARTER + n];
    }
    for (uint t = 0u; t < args.n_tok; t++) {
        device const float *base = qkv + (uint64_t) t * args.stride;
        device const float *q_d = base + (uint64_t) hk * head_dim;
        device const float *k_d = base + args.k_offset + (uint64_t) hk * head_dim;
        device const float *v_d = base + args.v_offset + (uint64_t) h * head_dim;

        // Every lane owns one dimension at SPLIT 1, so every lane publishes.
        if (j < head_dim) {
            s_k[j] = k_d[j];
            s_q[j] = q_d[j];
        }
        // Same dense-prefix product layout and block reduction the
        // all-register body uses: fixed to Q35_GDN_D lanes so the tree does
        // not move with the variant.
        const float kq_part = (tid < Q35_GDN_D) ? (k_d[tid] * q_d[tid]) : 0.0f;
        const float s_kq = q35_block_sum(kq_part, scratch, tid, Q35_GDN_D);

        const uint64_t gi = (uint64_t) t * args.n_head_v + h;
        const float dec = exp(decay[gi]);
        const float bt = beta[gi];

        // One independent 32-FMA chain per quarter, walked from a start
        // rotated by the QUARTER index -- byte for byte the same sequence
        // gdn_recurrent.metal walks at SPLIT 1. Only the source of `mv`
        // differs: quarter 0 reads the staged slot, quarters 1, 2 and 3
        // read registers.
        float part_k[4];
        float part_q[4];
        for (uint qi = 0u; qi < 4u; qi++) {
            const uint qbase = qi * Q35_GDN_QUARTER;
            float sk_q = 0.0f;
            float sq_q = 0.0f;
            for (uint n = 0u; n < Q35_GDN_QUARTER; n++) {
                const uint rot = (n + 8u * qi) & (Q35_GDN_QUARTER - 1u);
                float mv;
                if (qi == 0u) {
                    mv = my_staged[rot];
                } else {
                    mv = row_hi[(qi - 1u) * Q35_GDN_QUARTER + rot];
                }
                sk_q = q35_fma(mv, s_k[qbase + rot], sk_q);
                sq_q = q35_fma(mv, s_q[qbase + rot], sq_q);
            }
            part_k[qi] = sk_q;
            part_q[qi] = sq_q;
        }
        // The SPLIT == 1 arm of the all-register body: one lane holds all
        // four quarters, so they combine with q35_canonical4 directly.
        const float sk = q35_canonical4(part_k[0], part_k[1], part_k[2], part_k[3]);
        const float sq = q35_canonical4(part_q[0], part_q[1], part_q[2], part_q[3]);
        const float delta = (v_d[j] - dec * sk) * bt;

        attn_out[((uint64_t) t * args.n_head_v + h) * head_dim + j] =
                (dec * sq + delta * s_kq) * scale;

        // The carry walks the same rotated per-quarter sequence as the read
        // above; anything else would pair a register (or staged slot) with
        // the wrong key.
        for (uint qi = 0u; qi < 4u; qi++) {
            const uint qbase = qi * Q35_GDN_QUARTER;
            for (uint n = 0u; n < Q35_GDN_QUARTER; n++) {
                const uint rot = (n + 8u * qi) & (Q35_GDN_QUARTER - 1u);
                const float delta_k = delta * s_k[qbase + rot];
                if (qi == 0u) {
                    my_staged[rot] = q35_fma(my_staged[rot], dec, delta_k);
                } else {
                    const uint idx = (qi - 1u) * Q35_GDN_QUARTER + rot;
                    row_hi[idx] = q35_fma(row_hi[idx], dec, delta_k);
                }
            }
        }
        // k and q are rewritten by the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (uint n = 0u; n < Q35_GDN_QUARTER; n++) {
        S[n] = my_staged[n];
    }
    for (uint n = 0u; n < 96u; n++) {
        S[Q35_GDN_QUARTER + n] = row_hi[n];
    }
}

// One quarter staged, three in registers: the always-fits variant.
//
// 128 lanes, four quarters each. Quarter 0 lives in threadgroup memory
// (128 lanes * 32 floats * 4 B = 16384 B, plus the 1156 B of scratch this
// kernel already uses -- 17540 B total, inside the 32768 B every Apple
// Silicon GPU allows) and quarters 1, 2 and 3 stay in registers.
//
// This exists so that no device is refused. It is expected to lose to the
// wider register-resident shapes, and the tuner will say so -- though not
// always: this shape has measured fastest of the whole recurrent family at
// `n_tok = 32` on real hardware, so "always loses" is not a bar it clears.
kernel void kernel_q35_gdn_recurrent_hybrid(
        constant q35_args_gdn_recurrent & args [[buffer(0)]],
        device float *attn_out [[buffer(1)]],
        device float *state [[buffer(2)]],
        device const float *qkv [[buffer(3)]],
        device const float *decay [[buffer(4)]],
        device const float *beta [[buffer(5)]],
        threadgroup float *tbuf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    // The staged quarter follows the scratch, k and q rows the shared body uses.
    threadgroup float *staged = tbuf + Q35_GDN_REC_SLOTS;
    q35_gdn_recurrent_staged_impl(
            args, attn_out, state, qkv, decay, beta, tbuf, staged,
            tgpig, tpitg, ntg);
}

// Back to the library-wide default for the files concatenated after this
// one -- `fast` means the build's default, not a saved state; see
// compile_library in shiso/crates/shiso-metal/src/compile.rs.
#pragma METAL fp math_mode(fast)
