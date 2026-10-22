// q4e gdn entries: Metal port of kernels/cuda/q4e/gdn.cu (see q4e_common.h for conventions).
//
// The gated DeltaNet block of the sparse family's linear-attention layers:
// the causal convolution, the q/k L2 norm, the decay/beta gates, the
// sequential delta-rule recurrence (and its rollback replay) and the gated
// output norm. Every entry binds the CUDA parameter at position `i` to
// `[[buffer(i)]]` and reads grid and block as CUDA does.
//
// Not ported, and why:
// - `q4e_gdn_conv_seg_kernel`, `q4e_gdn_recurrent_seg_kernel` and the three
//   `*_batch_*` kernels take device arrays of per-session pointers
//   (`float * const *`), which this backend cannot pack yet.
// - `q4e_gdn_chunk_kernel` stages ~97 KiB of threadgroup memory against this
//   device's 32 KiB; the host keeps Metal off the chunked path.
//
// THE RECURRENCE IS RESTRUCTURED. The CUDA recurrence (and the replay, which
// is the same body) stages the head_dim x (head_dim + 1) state tile in dynamic
// shared memory: 66 048 bytes at head_dim 128, twice what a threadgroup may
// hold here. The port keeps the state in registers instead. The CUDA thread
// (j, sub) of the 512-thread block already touches exactly the 32 floats
// row[i0 + ((n + 8 sub) & 31)], n = 0..31, of row j of the transposed state and
// nothing else, so the port's thread (j, sub) holds those same 32 floats in
// `reg[n]`, in that same walk order. Every inner product and every update is
// the CUDA expression over the same operands in the same order (an `fma` chain
// over n, then the two xor folds), so each output and each state element is
// the CUDA arithmetic unchanged; only where the state is parked moved. The
// engine still requests the 66 048 bytes (`kernels::gdn_recurrent`), which the
// Metal run path refuses; this port declares no `[[threadgroup(0)]]` and needs
// none, so the host should pass 0 on Metal.
//
// The 32 state registers a lane holds push the compiler's default register
// budget below 512 threads (the pipeline reported a 384-thread maximum), so
// both state kernels carry `max_total_threads_per_threadgroup(512)`, which
// makes the compiler fit the engine's 512-thread block.
//
// A pointer the host may pass as NULL (`Operand::Null`) arrives here as an
// 8-byte zero bound with setBytes, i.e. a valid pointer to eight zero bytes,
// never a null one. Every such access is gated on the companion count the
// host pairs with it (`ckpt_cap == 0` for an absent rollback slab).

/// The state columns one lane walks: `head_dim / Q4E_GDN_SPLIT` at the only
/// head width the register layout is written for.
#define Q4E_GDN_SPLIT 4u
#define Q4E_GDN_D 128u
#define Q4E_GDN_SEG 32u

/// `log1pf` for x >= 0 (the argument is always an `exp`). The shading
/// language has none, and log(1 + x) loses the argument once x falls toward
/// an ulp of 1, which is exactly the range a strongly negative decay input
/// reaches. The textbook fix, `x * log(u) / (u - 1)` with u = 1 + x, is not
/// safe here: under the library's fast math the compiler may fold
/// `(1 + x) - 1` back to x, and on this device that form (with an accurate
/// log) still left a 0.2% error in a tiny decay; with the fast log, 3%. So below 0.1 the series is summed
/// directly (the first omitted term, x^9/9, is under 2^-24 of x there), and
/// above it 1 + x rounds with a relative error in the result under 6e-7.
static inline float q4e_gdn_log1p(float x) {
    if (x < 0.1f) {
        // x - x^2/2 + x^3/3 - ... - x^8/8, Horner form.
        float p = -1.0f / 8.0f;
        p = fma(p, x, 1.0f / 7.0f);
        p = fma(p, x, -1.0f / 6.0f);
        p = fma(p, x, 1.0f / 5.0f);
        p = fma(p, x, -1.0f / 4.0f);
        p = fma(p, x, 1.0f / 3.0f);
        p = fma(p, x, -1.0f / 2.0f);
        p = fma(p, x, 1.0f);
        return p * x;
    }
    return precise::log(1.0f + x);
}

/// `q4e_gdn_conv_kernel` (gdn.cu:58, body gdn.cu:28): causal depthwise conv
/// over the q|k|v channel block, then silu, one thread per channel.
///
/// Buffers in bind order: `out, state, x, w, channels, kernel, n_tok, ckpt,
/// ckpt_cap`. `state` holds the last `kernel - 1` inputs of each channel and
/// is rewritten to the new tail. `ckpt` receives the window after each token
/// t < min(ckpt_cap, n_tok - 1); the host passes it NULL with `ckpt_cap == 0`
/// on a plain pass, and that count is the only gate here (see the file note).
kernel void q4e_gdn_conv_kernel(
        device float *out [[buffer(0)]],
        device float *state [[buffer(1)]],
        device const float *x [[buffer(2)]],
        device const float *w [[buffer(3)]],
        constant uint &channels [[buffer(4)]],
        constant uint &kernel_len [[buffer(5)]],
        constant uint &n_tok [[buffer(6)]],
        device float *ckpt [[buffer(7)]],
        constant uint &ckpt_cap [[buffer(8)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const uint c = tgpig.x * ntg.x + tpitg.x;
    if (c >= channels) {
        return;
    }
    const uint hist = kernel_len - 1u;

    // This channel's history followed by the new tokens.
    float win[8];
    for (uint i = 0; i < hist; i++) {
        win[i] = state[ulong(i) * channels + c];
    }

    device const float *wc = w + ulong(c) * kernel_len;
    for (uint t = 0; t < n_tok; t++) {
        const float cur = x[ulong(t) * channels + c];
        float acc = 0.0f;
        for (uint k = 0; k < hist; k++) {
            acc += wc[k] * win[k];
        }
        acc += wc[hist] * cur;
        out[ulong(t) * channels + c] = acc / (1.0f + exp(-acc));
        for (uint k = 0; k + 1u < hist; k++) {
            win[k] = win[k + 1];
        }
        if (hist > 0) {
            win[hist - 1] = cur;
        }
        if (t < ckpt_cap && t + 1u < n_tok) {
            device float *ck = ckpt + ulong(t) * hist * channels;
            for (uint i = 0; i < hist; i++) {
                ck[ulong(i) * channels + c] = win[i];
            }
        }
    }

    for (uint i = 0; i < hist; i++) {
        state[ulong(i) * channels + c] = win[i];
    }
}

/// `q4e_gdn_l2norm_kernel` (gdn.cu:96): L2-normalize one head_dim-wide q or k
/// head in place, one threadgroup per (head = x, token = y). No weight and no
/// mean: `v *= rsqrt(sum v^2 + 1e-12)`.
///
/// Buffers in bind order: `qk, head_dim, n_heads, stride, offset`
/// (`n_heads` is unused, as in CUDA).
kernel void q4e_gdn_l2norm_kernel(
        device float *qk [[buffer(0)]],
        constant uint &head_dim [[buffer(1)]],
        constant uint &n_heads [[buffer(2)]],
        constant uint &stride [[buffer(3)]],
        constant uint &offset [[buffer(4)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[Q4E_SLOTS];
    const uint t = tgpig.y;
    const uint h = tgpig.x;
    device float *v = qk + ulong(t) * stride + offset + ulong(h) * head_dim;
    float sum = 0.0f;
    for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
        sum += v[i] * v[i];
    }
    sum = q4e_block_sum(sum, scratch, tpitg.x, ntg.x);
    const float inv = rsqrt(sum + 1e-12f);
    for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
        v[i] *= inv;
    }
    (void)n_heads;
}

/// `q4e_gdn_gates_kernel` (gdn.cu:112): `decay = softplus(alpha + dt_bias) *
/// a` (a already holds -exp(A_log), so decay is negative) and `beta =
/// sigmoid(beta_proj)`, one thread per (token, head), heads fastest.
///
/// Buffers in bind order: `decay, beta, alpha_proj, beta_proj, dt_bias, a,
/// n_head, n_tok`.
kernel void q4e_gdn_gates_kernel(
        device float *decay [[buffer(0)]],
        device float *beta [[buffer(1)]],
        device const float *alpha_proj [[buffer(2)]],
        device const float *beta_proj [[buffer(3)]],
        device const float *dt_bias [[buffer(4)]],
        device const float *a [[buffer(5)]],
        constant uint &n_head [[buffer(6)]],
        constant uint &n_tok [[buffer(7)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    const ulong i = ulong(tgpig.x) * ntg.x + tpitg.x;
    if (i >= ulong(n_head) * n_tok) {
        return;
    }
    const uint h = uint(i % n_head);
    const float x = alpha_proj[i] + dt_bias[h];
    // softplus, guarded so a large x does not overflow
    const float sp = (x > 20.0f) ? x : q4e_gdn_log1p(exp(x));
    decay[i] = sp * a[h];
    beta[i] = 1.0f / (1.0f + exp(-beta_proj[i]));
}

/// The rollback slab's row section (gdn.cu:149): after the start-of-pass
/// state of every value head.
static inline device float *q4e_gdn_rec_rows(device float *ckpt, uint head_dim, uint n_head_v) {
    return ckpt + ulong(n_head_v) * head_dim * head_dim;
}

/// Where `q4e_gdn_recurrent_run` reads and writes; `outputs == false` is the
/// replay (CUDA's `attn_out == NULL`), which advances the state only.
struct q4e_gdn_rec_io {
    device float *attn_out;
    device const float *state_in;
    device float *state_out;
    device const float *qkv;
    device const float *decay;
    device const float *beta;
    device float *ckpt;
    bool outputs;
};

/// The recurrence's scalar shape, as CUDA passes it.
struct q4e_gdn_rec_shape {
    uint head_dim;
    uint n_head_k;
    uint n_head_v;
    uint k_offset;
    uint v_offset;
    uint stride;
    uint n_tok;
    uint ckpt_cap;
};

/// `q4e_gdn_recurrent_run` (gdn.cu:153) with the state in registers (see the
/// file note). Thread (j, sub) owns `reg[n] = S^T[j][i0 + ((n + 8 sub) & 31)]`.
///
/// Per token:   S *= exp(decay)
///              delta[j] = (v[j] - <S[:,j], k>) * beta
///              S[:,j]  += delta[j] * k
///              out[j]   = <S[:,j], q> / sqrt(head_dim)
///
/// Value head h reads query/key head h % n_head_k: modulo, not divide.
static inline void q4e_gdn_recurrent_run(
        q4e_gdn_rec_io io, q4e_gdn_rec_shape sh,
        threadgroup float *s_k, threadgroup float *s_q, threadgroup float *s_kq,
        threadgroup float *scratch, uint h, uint tid, uint nthreads) {
    const uint head_dim = sh.head_dim;
    const uint hk = h % sh.n_head_k;
    const uint j = tid / Q4E_GDN_SPLIT;
    const uint sub = tid % Q4E_GDN_SPLIT;
    const uint seg = Q4E_GDN_SEG;
    const float scale = rsqrt(float(head_dim));
    const bool outputs = io.outputs;
    // Only threads of a real column (j < head_dim) own state; at the engine's
    // 512 threads and head_dim 128 that is every thread.
    const bool owner = j < head_dim;

    device const float *m_in = io.state_in + ulong(h) * head_dim * head_dim;
    device float *m = io.state_out + ulong(h) * head_dim * head_dim;
    // The rollback record: this head's state as the pass found it, and the
    // inputs of each row a rollback could keep. Gated on the count alone: an
    // absent slab is an 8-byte stand-in with `ckpt_cap == 0`.
    const uint recorded = (sh.ckpt_cap > 0u && sh.n_tok > 1u) ? min(sh.ckpt_cap, sh.n_tok) : 0u;
    device float *s0 = io.ckpt + ulong(h) * head_dim * head_dim;

    const uint i0 = sub * seg;
    // The walk's rotation is a multiple of four, so four consecutive `n` are
    // four consecutive `i`: the state moves as 16-byte vectors, in the same
    // walk order - a scalar load here left the recurrence at a fifth of the
    // device's bandwidth.
    float reg[Q4E_GDN_SEG];
    for (uint n = 0; n < Q4E_GDN_SEG; n += 4u) {
        const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
        const float4 v = owner ? *(device const float4 *)(m_in + ulong(j) * head_dim + i)
                               : float4(0.0f);
        reg[n] = v.x;
        reg[n + 1u] = v.y;
        reg[n + 2u] = v.z;
        reg[n + 3u] = v.w;
        if (recorded && owner) {
            *(device float4 *)(s0 + ulong(j) * head_dim + i) = v;
        }
    }
    if (recorded) {
        device float *rows = q4e_gdn_rec_rows(io.ckpt, head_dim, sh.n_head_v);
        device float *rdec = rows + ulong(sh.ckpt_cap) * sh.stride;
        device float *rbeta = rdec + ulong(sh.ckpt_cap) * sh.n_head_v;
        for (uint t = 0; t < recorded; t++) {
            device const float *base = io.qkv + ulong(t) * sh.stride;
            device float *dst = rows + ulong(t) * sh.stride;
            for (uint i = tid; i < head_dim; i += nthreads) {
                dst[sh.v_offset + ulong(h) * head_dim + i] = base[sh.v_offset + ulong(h) * head_dim + i];
                if (h < sh.n_head_k) {
                    dst[sh.k_offset + ulong(h) * head_dim + i] = base[sh.k_offset + ulong(h) * head_dim + i];
                }
            }
            if (tid == 0u) {
                rdec[ulong(t) * sh.n_head_v + h] = io.decay[ulong(t) * sh.n_head_v + h];
                rbeta[ulong(t) * sh.n_head_v + h] = io.beta[ulong(t) * sh.n_head_v + h];
            }
        }
    }

    for (uint t = 0; t < sh.n_tok; t++) {
        device const float *base = io.qkv + ulong(t) * sh.stride;
        device const float *q_d = base + ulong(hk) * head_dim;
        device const float *k_d = base + sh.k_offset + ulong(hk) * head_dim;
        device const float *v_d = base + sh.v_offset + ulong(h) * head_dim;
        float kq_part = 0.0f;
        if (sub == 0u && owner) {
            const float kv = k_d[j];
            s_k[j] = kv;
            if (outputs) {
                const float qv = q_d[j];
                s_q[j] = qv;
                kq_part = kv * qv;
            }
        }
        if (outputs) {
            kq_part = q4e_block_sum(kq_part, scratch, tid, nthreads);
            if (tid == 0u) {
                *s_kq = kq_part;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const ulong gi = ulong(t) * sh.n_head_v + h;
        const float dec = exp(io.decay[gi]);
        const float bt = io.beta[gi];

        // This lane's quarter of the row, in the CUDA walk order.
        float sk = 0.0f, sq = 0.0f;
        for (uint n = 0; n < Q4E_GDN_SEG; n++) {
            const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
            const float mv = reg[n];
            sk = fma(mv, s_k[i], sk);
            if (outputs) {
                sq = fma(mv, s_q[i], sq);
            }
        }
        sk += simd_shuffle_xor(sk, ushort(1));
        sk += simd_shuffle_xor(sk, ushort(2));
        sq += simd_shuffle_xor(sq, ushort(1));
        sq += simd_shuffle_xor(sq, ushort(2));
        const float delta = owner ? (v_d[j] - dec * sk) * bt : 0.0f;

        if (outputs && sub == 0u && owner) {
            io.attn_out[(ulong(t) * sh.n_head_v + h) * head_dim + j] = (dec * sq + delta * (*s_kq)) * scale;
        }
        for (uint n = 0; n < Q4E_GDN_SEG; n++) {
            const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
            reg[n] = fma(reg[n], dec, delta * s_k[i]);
        }
        // s_k / s_q / s_kq are rewritten by the next token.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (owner) {
        for (uint n = 0; n < Q4E_GDN_SEG; n += 4u) {
            const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
            *(device float4 *)(m + ulong(j) * head_dim + i) =
                float4(reg[n], reg[n + 1u], reg[n + 2u], reg[n + 3u]);
        }
    }
}

/// `q4e_gdn_recurrent_kernel` (gdn.cu:274): the gated delta rule, one
/// threadgroup of `head_dim * 4` = 512 threads per value head, state in
/// registers (see the file note: the CUDA 66 048-byte dynamic tile is gone,
/// and no `[[threadgroup(0)]]` is declared).
///
/// Buffers in bind order: `attn_out, state, qkv, decay, beta, head_dim,
/// n_head_k, n_head_v, k_offset, v_offset, stride, n_tok, ckpt, ckpt_cap`.
/// `head_dim` must be 128 (the register layout is sized for it; the host's
/// `GDN_STATE_HEAD_DIM`); any other width returns without touching memory.
/// `ckpt` is NULL with `ckpt_cap == 0` on a plain pass, gated on the count.
[[max_total_threads_per_threadgroup(512)]]
kernel void q4e_gdn_recurrent_kernel(
        device float *attn_out [[buffer(0)]],
        device float *state [[buffer(1)]],
        device const float *qkv [[buffer(2)]],
        device const float *decay [[buffer(3)]],
        device const float *beta [[buffer(4)]],
        constant uint &head_dim [[buffer(5)]],
        constant uint &n_head_k [[buffer(6)]],
        constant uint &n_head_v [[buffer(7)]],
        constant uint &k_offset [[buffer(8)]],
        constant uint &v_offset [[buffer(9)]],
        constant uint &stride [[buffer(10)]],
        constant uint &n_tok [[buffer(11)]],
        device float *ckpt [[buffer(12)]],
        constant uint &ckpt_cap [[buffer(13)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    threadgroup float s_k[Q4E_GDN_D];
    threadgroup float s_q[Q4E_GDN_D];
    threadgroup float s_kq[1];
    threadgroup float scratch[Q4E_SLOTS];
    if (head_dim != Q4E_GDN_D) {
        return;
    }
    const q4e_gdn_rec_io io = {attn_out, state, state, qkv, decay, beta, ckpt, true};
    const q4e_gdn_rec_shape sh = {head_dim, n_head_k, n_head_v, k_offset, v_offset, stride, n_tok, ckpt_cap};
    q4e_gdn_recurrent_run(io, sh, s_k, s_q, s_kq, scratch, tgpig.x, tpitg.x, ntg.x);
}

/// `q4e_gdn_replay_kernel` (gdn.cu:309): a rollback. The state after the
/// first `n_keep` rows of the last verify pass, rebuilt from the slab's
/// start-of-pass state and recorded rows by the recurrence's own body, so bit
/// for bit what the pass held after those rows. Same geometry as the
/// recurrence (the engine launches it with `kernels::gdn_recurrent`, whose
/// 66 048-byte smem request this port does not need).
///
/// Buffers in bind order: `state, ckpt, head_dim, n_head_k, n_head_v,
/// k_offset, v_offset, stride, n_keep, ckpt_cap`.
[[max_total_threads_per_threadgroup(512)]]
kernel void q4e_gdn_replay_kernel(
        device float *state [[buffer(0)]],
        device float *ckpt [[buffer(1)]],
        constant uint &head_dim [[buffer(2)]],
        constant uint &n_head_k [[buffer(3)]],
        constant uint &n_head_v [[buffer(4)]],
        constant uint &k_offset [[buffer(5)]],
        constant uint &v_offset [[buffer(6)]],
        constant uint &stride [[buffer(7)]],
        constant uint &n_keep [[buffer(8)]],
        constant uint &ckpt_cap [[buffer(9)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    threadgroup float s_k[Q4E_GDN_D];
    threadgroup float s_q[Q4E_GDN_D];
    threadgroup float s_kq[1];
    threadgroup float scratch[Q4E_SLOTS];
    if (head_dim != Q4E_GDN_D) {
        return;
    }
    device float *rows = q4e_gdn_rec_rows(ckpt, head_dim, n_head_v);
    device float *rdec = rows + ulong(ckpt_cap) * stride;
    device float *rbeta = rdec + ulong(ckpt_cap) * n_head_v;
    // No outputs and no record: CUDA passes attn_out = NULL and ckpt = NULL,
    // 0; `outputs` and the zero cap stand for both here.
    const q4e_gdn_rec_io io = {state, ckpt, state, rows, rdec, rbeta, ckpt, false};
    const q4e_gdn_rec_shape sh = {head_dim, n_head_k, n_head_v, k_offset, v_offset, stride, n_keep, 0u};
    q4e_gdn_recurrent_run(io, sh, s_k, s_q, s_kq, scratch, tgpig.x, tpitg.x, ntg.x);
}

/// `q4e_gdn_out_gate_kernel` (gdn.cu:590): per-head RMSNorm of the delta-rule
/// output (weight applied raw, no +1 offset), then the sigmoid output gate;
/// one threadgroup per (token, head).
///
/// Buffers in bind order: `out, attn, z, w, head_dim, n_head, eps`
/// (`n_head` is unused, as in CUDA).
kernel void q4e_gdn_out_gate_kernel(
        device float *out [[buffer(0)]],
        device const float *attn [[buffer(1)]],
        device const float *z [[buffer(2)]],
        device const float *w [[buffer(3)]],
        constant uint &head_dim [[buffer(4)]],
        constant uint &n_head [[buffer(5)]],
        constant float &eps [[buffer(6)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    threadgroup float scratch[Q4E_SLOTS];
    const uint group = tgpig.x;
    device const float *src = attn + ulong(group) * head_dim;
    device const float *zr = z + ulong(group) * head_dim;
    device float *dst = out + ulong(group) * head_dim;

    float sum = 0.0f;
    for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
        sum += src[i] * src[i];
    }
    const float scale = rsqrt(q4e_block_sum(sum, scratch, tpitg.x, ntg.x) / float(head_dim) + eps);
    for (uint i = tpitg.x; i < head_dim; i += ntg.x) {
        const float g = 1.0f / (1.0f + exp(-zr[i]));
        dst[i] = src[i] * scale * w[i] * g;
    }
    (void)n_head;
}
