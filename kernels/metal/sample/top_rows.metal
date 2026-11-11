// sample entries: Metal twin of `sample_top_rows_kernel` in
// kernels/cuda/sample/sample.cu.
//
// The top of a logits row: its `k` highest entries, ascending by token id, for
// a host draw that reads a top-k stage's worth of the row instead of the whole
// of it. "Highest" orders values the way the host's total order does (larger
// first, ties to the lower id), so the set is the one the host's own top-k
// over the full row keeps; the host draws from these exactly what it would
// from the row whenever its top-k is at most `k`.
//
// One threadgroup per row. A radix select, eight bits at a time from the top
// of an order-preserving key, finds the k-th highest key T and how many
// entries equal to T the set still needs; a compaction in index order then
// writes every entry above T and the first `need` entries equal to T. Index
// order is what makes the output ascending by id and the ties the lowest ids.
//
// The keys are compared as bits, so nothing here depends on the math mode: an
// infinity a suppress stage wrote orders below every finite logit.
//
// Buffers: 0 `logits` (f32, rows * n_vocab), 1 `n_vocab`, 2 `k`, 3 `values`
// (f32, rows * k), 4 `ids` (u32, rows * k). A row shorter than `k` writes all
// of its entries and leaves the rest of its slots alone. Grid `rows`, block a
// whole number of simdgroups, 1024 threads at most.

#include <metal_stdlib>
using namespace metal;

/// Larger float, larger key: every bit of a negative flipped, the sign bit of
/// a positive set. NaNs land at the ends as they do in a total order.
static inline uint sample_order_key(float f) {
    const uint u = as_type<uint>(f);
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

kernel void sample_top_rows_kernel(
        device const float *logits [[buffer(0)]],
        constant uint &n_vocab [[buffer(1)]],
        constant uint &k [[buffer(2)]],
        device float *values [[buffer(3)]],
        device uint *ids [[buffer(4)]],
        uint row_i [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]],
        uint lane [[thread_index_in_simdgroup]],
        uint warp [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    // The threshold's bits so far and what the set still takes at them.
    threadgroup uint s_state[2];
    threadgroup uint s_ties[32];
    threadgroup uint s_taken[32];

    device const float *row = logits + ulong(row_i) * n_vocab;
    device float *out_v = values + ulong(row_i) * k;
    device uint *out_i = ids + ulong(row_i) * k;

    // After the pass at `shift`, `prefix` holds the top bits of T and `need`
    // how many entries matching those bits the set takes.
    uint prefix = 0u;
    uint need = min(k, n_vocab);
    if (need == 0u) {
        return;
    }
    for (int shift = 24; shift >= 0; shift -= 8) {
        const uint above = shift == 24 ? 0u : (0xFFFFFFFFu << uint(shift + 8));
        for (uint b = tid; b < 256u; b += bdim) {
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < n_vocab; i += bdim) {
            const uint key = sample_order_key(row[i]);
            if ((key & above) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> uint(shift)) & 0xFFu], 1u,
                                          memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0u) {
            uint taken = 0u;
            int bin = 255;
            for (; bin > 0; bin--) {
                const uint n = atomic_load_explicit(&hist[bin], memory_order_relaxed);
                if (taken + n >= need) {
                    break;
                }
                taken += n;
            }
            s_state[0] = prefix | (uint(bin) << uint(shift));
            s_state[1] = need - taken;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        prefix = s_state[0];
        need = s_state[1];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Compaction in index order, a tile of `bdim` entries at a time: each
    // entry's rank among the tile's ties and among its taken entries comes
    // from its simdgroup's prefix count and the counts of the simdgroups
    // before it.
    const uint warps = (bdim + 31u) >> 5u;
    uint written = 0u;
    uint ties_seen = 0u;
    for (uint base = 0u; base < n_vocab; base += bdim) {
        const uint i = base + tid;
        float v = 0.0f;
        bool over = false;
        bool tie = false;
        if (i < n_vocab) {
            v = row[i];
            const uint key = sample_order_key(v);
            over = key > prefix;
            tie = key == prefix;
        }
        const uint tie_one = tie ? 1u : 0u;
        const uint ties_lane = simd_prefix_exclusive_sum(tie_one);
        const uint ties_warp = simd_sum(tie_one);
        if (lane == 0u) {
            s_ties[warp] = ties_warp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint ties_before = 0u;
        uint ties_tile = 0u;
        for (uint w = 0u; w < warps; w++) {
            const uint n = s_ties[w];
            ties_before += w < warp ? n : 0u;
            ties_tile += n;
        }
        const bool take = over || (tie && ties_seen + ties_before + ties_lane < need);
        const uint take_one = take ? 1u : 0u;
        const uint taken_lane = simd_prefix_exclusive_sum(take_one);
        const uint taken_warp = simd_sum(take_one);
        if (lane == 0u) {
            s_taken[warp] = taken_warp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint taken_before = 0u;
        uint taken_tile = 0u;
        for (uint w = 0u; w < warps; w++) {
            const uint n = s_taken[w];
            taken_before += w < warp ? n : 0u;
            taken_tile += n;
        }
        if (take) {
            const uint at = written + taken_before + taken_lane;
            if (at < k) {
                out_v[at] = v;
                out_i[at] = i;
            }
        }
        written += taken_tile;
        ties_seen += ties_tile;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
