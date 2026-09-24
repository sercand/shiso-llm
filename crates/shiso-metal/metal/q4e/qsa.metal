// q4e qsa entries: Metal port of kernels/cuda/q4e/qsa.cu (see q4e_common.h for conventions).
//
// The attention block's query-side norm and rotary, the key/value store into
// the paged f16 cache, the always-correct dense attention and the output gate.
//
// The query projection emits a query and an output gate interleaved per head:
// head h owns [q(head_dim) | gate(head_dim)] inside a 2 * head_dim * n_head
// row. Splitting the flat row down the middle instead is wrong.
//
// Not ported, deliberately:
// - q4e_qsa_store_kv_batch_kernel and q4e_qsa_attention_batch_kernel take a
//   device array of page-table pointers, which this backend cannot bind yet.
// - q4e_qsa_attention_tiled_kernel: its layout needs 50176 bytes of
//   threadgroup memory at head_dim 256 (and the engine requests exactly that),
//   past this device's 32768. The engine only takes that path when the device
//   grants the tiled kernel its memory, so on this device prefill runs the
//   dense kernel below.

/// `qsa.cu:24` `q4e_qsa_q_norm_rope_kernel`: per-head RMSNorm then partial
/// mRoPE, reading the query out of its strided slot and writing it contiguous.
///
/// Grid (n_head, n_tok), 128 threads, `head_dim * 4` bytes of dynamic memory.
/// Buffers: 0 q_out [t][h][head_dim], 1 gate_out (same layout), 2 qkv
/// [t][h][q | gate], 3 w [head_dim], 4 head_dim, 5 n_head, 6 n_rot,
/// 7 rope_base, 8 mrope [t][3], 9 sec_t, 10 sec_h, 11 sec_w, 12 eps.
kernel void q4e_qsa_q_norm_rope_kernel(
        device float *q_out            [[buffer(0)]],
        device float *gate_out         [[buffer(1)]],
        device const float *qkv        [[buffer(2)]],
        device const float *w          [[buffer(3)]],
        constant uint &head_dim        [[buffer(4)]],
        constant uint &n_head          [[buffer(5)]],
        constant uint &n_rot           [[buffer(6)]],
        constant float &rope_base      [[buffer(7)]],
        device const int *mrope        [[buffer(8)]],
        constant uint &sec_t           [[buffer(9)]],
        constant uint &sec_h           [[buffer(10)]],
        constant uint &sec_w           [[buffer(11)]],
        constant float &eps            [[buffer(12)]],
        threadgroup float *s_q         [[threadgroup(0)]],
        uint3 bid                      [[threadgroup_position_in_grid]],
        uint3 tid3                     [[thread_position_in_threadgroup]],
        uint3 ntg                      [[threads_per_threadgroup]]) {
    threadgroup float red[Q4E_SLOTS];
    const uint tid = tid3.x;
    const uint nthreads = ntg.x;
    const uint t = bid.y;
    const uint h = bid.x;
    const ulong src = ulong(t) * n_head * 2u * head_dim + ulong(h) * 2u * head_dim;
    const ulong dst = (ulong(t) * n_head + h) * head_dim;

    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nthreads) {
        const float v = qkv[src + i];
        s_q[i] = v;
        sum += v * v;
        gate_out[dst + i] = qkv[src + head_dim + i];
    }
    const float scale = rsqrt(q4e_block_sum(sum, red, tid, nthreads) / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += nthreads) {
        s_q[i] *= scale * w[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q4e_rope_mrope(s_q, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, tid, nthreads);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < head_dim; i += nthreads) {
        q_out[dst + i] = s_q[i];
    }
}

/// `qsa.cu:56` `q4e_qsa_store_kv_kernel`: key norm + mRoPE straight into the
/// f16 cache, and the value beside it.
///
/// Grid (n_head_kv, n_tok), 128 threads, `head_dim * 4` bytes of dynamic
/// memory. Buffers: 0 k_cache, 1 v_cache ([pool row][kv head][head_dim] f16),
/// 2 k, 3 v ([t][kv head][head_dim]), 4 kw, 5 head_dim, 6 n_head_kv, 7 n_rot,
/// 8 rope_base, 9 pos [t], 10 mrope [t][3], 11 sec_t, 12 sec_h, 13 sec_w,
/// 14 pages, 15 pool_slots (cache ROWS, not pages), 16 eps.
///
/// The angle comes from `mrope`, the cache slot from `pos`: the same number
/// for text, different inside an image. A slot past `pool_slots` is refused
/// rather than written, which keeps a wrong page id off another sequence.
kernel void q4e_qsa_store_kv_kernel(
        device half *k_cache           [[buffer(0)]],
        device half *v_cache           [[buffer(1)]],
        device const float *k          [[buffer(2)]],
        device const float *v          [[buffer(3)]],
        device const float *kw         [[buffer(4)]],
        constant uint &head_dim        [[buffer(5)]],
        constant uint &n_head_kv       [[buffer(6)]],
        constant uint &n_rot           [[buffer(7)]],
        constant float &rope_base      [[buffer(8)]],
        device const int *pos          [[buffer(9)]],
        device const int *mrope        [[buffer(10)]],
        constant uint &sec_t           [[buffer(11)]],
        constant uint &sec_h           [[buffer(12)]],
        constant uint &sec_w           [[buffer(13)]],
        device const int *pages        [[buffer(14)]],
        constant uint &pool_slots      [[buffer(15)]],
        constant float &eps            [[buffer(16)]],
        threadgroup float *s_k         [[threadgroup(0)]],
        uint3 bid                      [[threadgroup_position_in_grid]],
        uint3 tid3                     [[thread_position_in_threadgroup]],
        uint3 ntg                      [[threads_per_threadgroup]]) {
    threadgroup float red[Q4E_SLOTS];
    const uint tid = tid3.x;
    const uint nthreads = ntg.x;
    const uint t = bid.y;
    const uint h = bid.x;
    const ulong src = (ulong(t) * n_head_kv + h) * head_dim;

    float sum = 0.0f;
    for (uint i = tid; i < head_dim; i += nthreads) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrt(q4e_block_sum(sum, red, tid, nthreads) / float(head_dim) + eps);
    for (uint i = tid; i < head_dim; i += nthreads) {
        s_k[i] *= scale * kw[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    q4e_rope_mrope(s_k, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, tid, nthreads);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const ulong slot = (ulong(q4e_kv_row(pages, uint(pos[t]))) * n_head_kv + h) * head_dim;
    if (slot + head_dim > ulong(pool_slots) * n_head_kv * head_dim) {
        return;
    }
    for (uint i = tid; i < head_dim; i += nthreads) {
        k_cache[slot + i] = half(s_k[i]);
        v_cache[slot + i] = half(v[src + i]);
    }
}

// Head dims one lane may hold (`head_dim <= 32 * this`), and the warps of the
// dense attention block.
#define Q4E_ATTN_LANE_SLOTS 8u
#define Q4E_ATTN_WARPS      8u

/// `qsa.cu:110` `q4e_qsa_attention_kernel`: causal GQA over the resident cache
/// with online softmax, one threadgroup per (query head, token).
///
/// Grid (n_head, n_tok), 256 threads (8 simdgroups), `8 * head_dim * 4` bytes
/// of dynamic memory for the per-warp accumulators. Buffers: 0 out
/// [t][h][head_dim], 1 k_cache, 2 v_cache (f16), 3 q [t][h][head_dim],
/// 4 head_dim, 5 n_head, 6 n_head_kv, 7 pos [t], 8 pages, 9 n_tok.
///
/// A simdgroup owns whole positions `warp, warp + 8, ...` and reduces each dot
/// product with a butterfly, so the position loop has no barrier; the eight
/// (max, denominator, accumulator) partials are merged at the end in simdgroup
/// order. A simdgroup that owns no position leaves (-inf, 0, 0), which the
/// merge weighs by `exp(-inf) = 0`.
kernel void q4e_qsa_attention_kernel(
        device float *out              [[buffer(0)]],
        device const half *k_cache     [[buffer(1)]],
        device const half *v_cache     [[buffer(2)]],
        device const float *q          [[buffer(3)]],
        constant uint &head_dim        [[buffer(4)]],
        constant uint &n_head          [[buffer(5)]],
        constant uint &n_head_kv       [[buffer(6)]],
        device const int *pos          [[buffer(7)]],
        device const int *pages        [[buffer(8)]],
        constant uint &n_tok           [[buffer(9)]],
        threadgroup float *s_acc       [[threadgroup(0)]],
        uint3 bid                      [[threadgroup_position_in_grid]],
        uint3 tid3                     [[thread_position_in_threadgroup]],
        uint3 ntg                      [[threads_per_threadgroup]]) {
    threadgroup float s_m[Q4E_ATTN_WARPS];
    threadgroup float s_l[Q4E_ATTN_WARPS];
    const uint tid = tid3.x;
    const uint h = bid.x;
    const uint t = bid.y;
    const uint hkv = h / (n_head / n_head_kv);
    const uint last = uint(pos[t]);
    const float scale = rsqrt(float(head_dim));
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    const uint slots = (head_dim + 31u) / 32u;
    const ulong qbase = (ulong(t) * n_head + h) * head_dim;

    float qv[Q4E_ATTN_LANE_SLOTS];
    float acc[Q4E_ATTN_LANE_SLOTS];
    for (uint j = 0; j < slots; j++) {
        const uint i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (uint p = warp; p <= last; p += Q4E_ATTN_WARPS) {
        const ulong kb = (ulong(q4e_kv_row(pages, p)) * n_head_kv + hkv) * head_dim;
        float dot = 0.0f;
        for (uint j = 0; j < slots; j++) {
            const uint i = lane + 32u * j;
            if (i < head_dim) {
                dot += qv[j] * float(k_cache[kb + i]);
            }
        }
        dot = q4e_warp_allsum(dot);

        const float m_new = fmax(m, dot);
        const float corr = exp(m - m_new);
        const float w = exp(dot - m_new);
        for (uint j = 0; j < slots; j++) {
            const uint i = lane + 32u * j;
            const float vv = (i < head_dim) ? float(v_cache[kb + i]) : 0.0f;
            acc[j] = acc[j] * corr + w * vv;
        }
        l = l * corr + w;
        m = m_new;
    }

    for (uint j = 0; j < slots; j++) {
        const uint i = lane + 32u * j;
        if (i < head_dim) {
            s_acc[ulong(warp) * head_dim + i] = acc[j];
        }
    }
    if (lane == 0u) {
        s_m[warp] = m;
        s_l[warp] = l;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float gm = -INFINITY;
    for (uint w = 0; w < Q4E_ATTN_WARPS; w++) {
        gm = fmax(gm, s_m[w]);
    }
    float den = 0.0f;
    for (uint w = 0; w < Q4E_ATTN_WARPS; w++) {
        den += s_l[w] * exp(s_m[w] - gm);
    }
    const float inv = 1.0f / den;
    for (uint i = tid; i < head_dim; i += ntg.x) {
        float num = 0.0f;
        for (uint w = 0; w < Q4E_ATTN_WARPS; w++) {
            num += s_acc[ulong(w) * head_dim + i] * exp(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

/// `qsa.cu:428` `q4e_qsa_gate_kernel`: `x *= sigmoid(gate)` in place.
///
/// Linear grid of 256-thread groups over `n`. Buffers: 0 x, 1 gate, 2 n (u64).
kernel void q4e_qsa_gate_kernel(
        device float *x                [[buffer(0)]],
        device const float *gate       [[buffer(1)]],
        constant ulong &n              [[buffer(2)]],
        uint bid                       [[threadgroup_position_in_grid]],
        uint tid                       [[thread_position_in_threadgroup]],
        uint ntg                       [[threads_per_threadgroup]]) {
    const ulong i = ulong(bid) * ntg + tid;
    if (i >= n) {
        return;
    }
    x[i] *= 1.0f / (1.0f + exp(-gate[i]));
}
