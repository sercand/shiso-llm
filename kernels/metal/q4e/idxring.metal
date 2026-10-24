// q4e idxring entry: Metal twin of kernels/cuda/q4e/idxring.cu (see
// q4e_common.h for conventions).
//
// The indexer's pooled pass over a per-session ring of raw keys, so a page
// carries no raw key row per position. The arithmetic is
// `q4e_idx_pool_kernel`'s (idx.metal) unchanged; a raw key is read from the
// pass's own f32 rows, rounded to half as the cache store rounded it, or -
// before the pass's first position - from the ring. Threadgroup 0 always
// exists (the host launches `n_tok / r + 1`), and after its own pool it makes
// the two writes, behind a barrier: the tap into a checkpoint's ring section
// when `snap_row` asks for one, then the pass's last RING rows into the ring.
//
// Ring geometry and the bound on how far a pass may start behind the highest
// position written are the CUDA unit's; see its header.

#define Q4E_IR_RING   16u
#define Q4E_IR_NO_TAP 0xFFFFFFFFu

static_assert(Q4E_IR_RING % Q4E_IDX_R == 0u && Q4E_IR_RING >= 3u * Q4E_IDX_R,
              "the ring must hold a whole block and a verify's rewind behind it");

/// The half raw key of position `x`, element `i`.
static inline half q4e_ir_raw(device const half *ring, device const float *k,
                              uint p0, uint x, uint i) {
    return x >= p0 ? half(k[ulong(x - p0) * Q4E_IDX_DIM + i])
                   : ring[ulong(x % Q4E_IR_RING) * Q4E_IDX_DIM + i];
}

/// `q4e_idx_ring_pool_kernel` (idxring.cu).
///
/// Buffers: 0 `pooled`, 1 `ring`, 2 `k` (f32 `[n_tok][128]`), 3 `w`, 4 `pos`,
/// 5 `pages`, 6 `n_tok`, 7 `n_rot`, 8 `rope_base`, 9 `mrope_all`, 10 `sec_t`,
/// 11 `sec_h`, 12 `sec_w`, 13 `eps`, 14 `snap`, 15 `snap_row`.
kernel void q4e_idx_ring_pool_kernel(device half *pooled [[buffer(0)]],
                                     device half *ring [[buffer(1)]],
                                     device const float *k [[buffer(2)]],
                                     device const float *w [[buffer(3)]],
                                     device const int *pos [[buffer(4)]],
                                     device const int *pages [[buffer(5)]],
                                     constant uint &n_tok [[buffer(6)]],
                                     constant uint &n_rot [[buffer(7)]],
                                     constant float &rope_base [[buffer(8)]],
                                     device const int *mrope_all [[buffer(9)]],
                                     constant uint &sec_t [[buffer(10)]],
                                     constant uint &sec_h [[buffer(11)]],
                                     constant uint &sec_w [[buffer(12)]],
                                     constant float &eps [[buffer(13)]],
                                     device half *snap [[buffer(14)]],
                                     constant uint &snap_row [[buffer(15)]],
                                     uint3 bid [[threadgroup_position_in_grid]],
                                     uint3 tid3 [[thread_position_in_threadgroup]],
                                     uint3 bdim [[threads_per_threadgroup]]) {
    const uint tid = tid3.x;
    const uint p0 = uint(pos[0]);
    const uint b0 = p0 / Q4E_IDX_R;
    const uint b1 = (uint(pos[n_tok - 1u]) + 1u) / Q4E_IDX_R;
    const uint b = b0 + bid.x;
    threadgroup float s_k[Q4E_IDX_DIM];
    threadgroup float s_red[Q4E_SLOTS];
    // Threadgroup-uniform, so every barrier inside is reached by all or none.
    if (b < b1) {
        const uint krow = q4e_kv_row(pages, b * Q4E_IDX_R);
        const uint brow = krow / Q4E_IDX_R;
        float sum = 0.0f;
        for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) {
            float acc = 0.0f;
            for (uint j = 0; j < Q4E_IDX_R; j++) {
                acc += float(q4e_ir_raw(ring, k, p0, b * Q4E_IDX_R + j, i));
            }
            acc *= 1.0f / float(Q4E_IDX_R);
            s_k[i] = acc;
            sum += acc * acc;
        }
        const float scale = rsqrt(q4e_block_sum(sum, s_red, tid, bdim.x) / float(Q4E_IDX_DIM) + eps);
        for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) s_k[i] *= scale * w[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const int block_pos = int(b * Q4E_IDX_R);
        const uint mrow = uint(int(Q4E_MROPE_BACK) + block_pos - pos[0]);
        q4e_rope_mrope(s_k, n_rot, rope_base,
                       mrope_all[3u * mrow], mrope_all[3u * mrow + 1u], mrope_all[3u * mrow + 2u],
                       sec_t, sec_h, sec_w, tid, bdim.x);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) {
            pooled[ulong(brow) * Q4E_IDX_DIM + i] = half(s_k[i]);
        }
    }
    if (bid.x != 0u) return;
    // Every read of the ring above happens before any write below.
    threadgroup_barrier(mem_flags::mem_device);
    if (snap_row != Q4E_IR_NO_TAP) {
        const uint sp = p0 + snap_row;
        for (uint x = (sp / Q4E_IDX_R) * Q4E_IDX_R; x < sp; x++) {
            for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) {
                snap[ulong(x % Q4E_IR_RING) * Q4E_IDX_DIM + i] = q4e_ir_raw(ring, k, p0, x, i);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
    const uint first = n_tok > Q4E_IR_RING ? n_tok - Q4E_IR_RING : 0u;
    for (uint t = first; t < n_tok; t++) {
        const uint x = uint(pos[t]);
        for (uint i = tid; i < Q4E_IDX_DIM; i += bdim.x) {
            ring[ulong(x % Q4E_IR_RING) * Q4E_IDX_DIM + i] = half(k[ulong(t) * Q4E_IDX_DIM + i]);
        }
    }
}
