// q4e exact entries: Metal port of kernels/cuda/q4e/exact.cu (see q4e_common.h
// for conventions).
//
// Only `q4e_gdn_step_kernel` is ported. `q4e_qsa_attention_tc_exact_kernel` is
// the tensor-core attention's twin, and this backend ports neither: the host
// keeps Metal off both tensor-core paths.
//
// `q4e_gdn_step_kernel` is one decode row of a recurrent layer after its
// convolution, in one launch: the q and k L2 norms, the gates, the recurrence
// and the output norm and gate of `gdn.metal`, each phase that file's
// expressions over the same values in the same order, so the step gives the
// bits the five launches give on this device. The state stays in registers in
// the recurrence's layout (see `gdn.metal`'s file note), so the 66 048 bytes
// the host requests for the CUDA tile describe nothing here and no
// `[[threadgroup(0)]]` is declared; threadgroup memory is three 128-float rows
// and a reduction scratch, about 2 KiB.
//
// The three 128-wide sums call `q4e_block_sum` with `nthreads = 128` from all
// 512 threads: only the first four simdgroups' partials are folded, which is
// the tree the 128-thread `q4e_gdn_l2norm_kernel` and
// `q4e_gdn_out_gate_kernel` reduce through. The kq sum folds over all 512
// threads, as the recurrence's does.
//
// The convolution is not fused: value heads that share a key head would read
// that head's window while another threadgroup rewrites it, and threadgroups
// of one dispatch have no order (see exact.cu). The host plans this kernel
// only for one row of one session at head_dim 128, which the rows below are
// sized for; it does not check.
//
// The decay passes through threadgroup memory before its `exp`, as it passes
// through device memory between the gates and the recurrence in the chain, so
// the exponent is the stored float and not an expression fast math may fold.

/// `q4e_gdn_step_kernel` (exact.cu): one threadgroup of `4 * head_dim` = 512
/// threads per value head.
///
/// Buffers in bind order: `gated_out, state, conv, alpha_proj, beta_proj,
/// dt_bias, a, z, norm_w, head_dim, n_head_k, n_head_v, k_offset, v_offset,
/// eps` (`n_head_v` is unused, as in CUDA).
[[max_total_threads_per_threadgroup(512)]]
kernel void q4e_gdn_step_kernel(
        device float *gated_out [[buffer(0)]],
        device float *state [[buffer(1)]],
        device const float *conv [[buffer(2)]],
        device const float *alpha_proj [[buffer(3)]],
        device const float *beta_proj [[buffer(4)]],
        device const float *dt_bias [[buffer(5)]],
        device const float *a [[buffer(6)]],
        device const float *z [[buffer(7)]],
        device const float *norm_w [[buffer(8)]],
        constant uint &head_dim [[buffer(9)]],
        constant uint &n_head_k [[buffer(10)]],
        constant uint &n_head_v [[buffer(11)]],
        constant uint &k_offset [[buffer(12)]],
        constant uint &v_offset [[buffer(13)]],
        constant float &eps [[buffer(14)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint3 tpitg [[thread_position_in_threadgroup]],
        uint3 ntg [[threads_per_threadgroup]]) {
    threadgroup float s_q[Q4E_GDN_D];
    threadgroup float s_k[Q4E_GDN_D];
    threadgroup float s_attn[Q4E_GDN_D];
    threadgroup float s_gates[2];   // decay, beta
    threadgroup float scratch[Q4E_SLOTS];
    const uint tid = tpitg.x;
    const uint nthreads = ntg.x;
    const uint h = tgpig.x;
    const uint hk = h % n_head_k;
    device const float *q_d = conv + ulong(hk) * head_dim;
    device const float *k_d = conv + k_offset + ulong(hk) * head_dim;
    device const float *v_d = conv + v_offset + ulong(h) * head_dim;
    (void)n_head_v;

    // 1. L2 norms: `q4e_gdn_l2norm_kernel` over the first 128 threads, at the
    // query offset and then at the key offset.
    float sum_q = 0.0f;
    float sum_k = 0.0f;
    if (tid < 128u) {
        for (uint i = tid; i < head_dim; i += 128u) {
            sum_q += q_d[i] * q_d[i];
        }
        for (uint i = tid; i < head_dim; i += 128u) {
            sum_k += k_d[i] * k_d[i];
        }
    }
    sum_q = q4e_block_sum(sum_q, scratch, tid, 128u);
    const float inv_q = rsqrt(sum_q + 1e-12f);
    sum_k = q4e_block_sum(sum_k, scratch, tid, 128u);
    const float inv_k = rsqrt(sum_k + 1e-12f);
    if (tid < 128u) {
        for (uint i = tid; i < head_dim; i += 128u) {
            s_q[i] = q_d[i] * inv_q;
            s_k[i] = k_d[i] * inv_k;
        }
    }

    // 2. Gates: `q4e_gdn_gates_kernel` for (row 0, head h).
    if (tid == 0u) {
        const float x = alpha_proj[h] + dt_bias[h];
        const float sp = (x > 20.0f) ? x : q4e_gdn_log1p(exp(x));
        s_gates[0] = sp * a[h];
        s_gates[1] = 1.0f / (1.0f + exp(-beta_proj[h]));
    }

    // 3. The delta rule: `q4e_gdn_recurrent_run` at n_tok = 1, q and k read
    // from threadgroup memory. Thread (j, sub) owns
    // `reg[n] = S^T[j][i0 + ((n + 8 sub) & 31)]`.
    const uint j = tid / Q4E_GDN_SPLIT;
    const uint sub = tid % Q4E_GDN_SPLIT;
    const uint seg = Q4E_GDN_SEG;
    const float scale = rsqrt(float(head_dim));
    const bool owner = j < head_dim;
    device float *m = state + ulong(h) * head_dim * head_dim;
    const uint i0 = sub * seg;
    float reg[Q4E_GDN_SEG];
    for (uint n = 0; n < Q4E_GDN_SEG; n += 4u) {
        const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
        const float4 v = owner ? *(device const float4 *)(m + ulong(j) * head_dim + i)
                               : float4(0.0f);
        reg[n] = v.x;
        reg[n + 1u] = v.y;
        reg[n + 2u] = v.z;
        reg[n + 3u] = v.w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float kq_part = 0.0f;
    if (sub == 0u && owner) {
        kq_part = s_k[j] * s_q[j];
    }
    const float kq = q4e_block_sum(kq_part, scratch, tid, nthreads);

    const float dec = exp(s_gates[0]);
    const float bt = s_gates[1];
    float sk = 0.0f, sq = 0.0f;
    for (uint n = 0; n < Q4E_GDN_SEG; n++) {
        const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
        const float mv = reg[n];
        sk = fma(mv, s_k[i], sk);
        sq = fma(mv, s_q[i], sq);
    }
    sk += simd_shuffle_xor(sk, ushort(1));
    sk += simd_shuffle_xor(sk, ushort(2));
    sq += simd_shuffle_xor(sq, ushort(1));
    sq += simd_shuffle_xor(sq, ushort(2));
    const float delta = owner ? (v_d[j] - dec * sk) * bt : 0.0f;

    if (sub == 0u && owner) {
        s_attn[j] = (dec * sq + delta * kq) * scale;
    }
    for (uint n = 0; n < Q4E_GDN_SEG; n++) {
        const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
        reg[n] = fma(reg[n], dec, delta * s_k[i]);
    }
    if (owner) {
        for (uint n = 0; n < Q4E_GDN_SEG; n += 4u) {
            const uint i = i0 + ((n + 8u * sub) & (seg - 1u));
            *(device float4 *)(m + ulong(j) * head_dim + i) =
                float4(reg[n], reg[n + 1u], reg[n + 2u], reg[n + 3u]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. `q4e_gdn_out_gate_kernel` over the first 128 threads.
    device const float *zr = z + ulong(h) * head_dim;
    device float *dst = gated_out + ulong(h) * head_dim;
    float sum = 0.0f;
    if (tid < 128u) {
        for (uint i = tid; i < head_dim; i += 128u) {
            sum += s_attn[i] * s_attn[i];
        }
    }
    const float oscale = rsqrt(q4e_block_sum(sum, scratch, tid, 128u) / float(head_dim) + eps);
    if (tid < 128u) {
        for (uint i = tid; i < head_dim; i += 128u) {
            const float g = 1.0f / (1.0f + exp(-zr[i]));
            dst[i] = s_attn[i] * oscale * norm_w[i] * g;
        }
    }
}
