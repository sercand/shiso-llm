// The router and the counting sort that groups the routed columns by expert.
//
// Ported from `kernels/cuda/q4e/moe.cu`, whose bodies are fidelity-gated and
// must not be edited. The same host code (`crates/shiso-engine/src/moe.rs`)
// launches these and the CUDA ones by the same entry names, and one host
// reference grades both.
//
// `q35moe_common.h` is concatenated ahead of this file by `SOURCES` in
// `src/compile.rs`, so its block reductions are already in scope and there is
// nothing to include: the library is one source string and a quote-include of a
// sibling would not resolve.
//
// No `#pragma METAL fp math_mode` here. Fast math is the library's build
// default and this file holds at it.

/// Histogram of routed columns per expert.
///
/// `kernels/cuda/q4e/moe.cu:203`. `counts` is cleared by the host before this
/// runs: it is accumulated atomically and outlives the layer, so an uncleared
/// row would carry every earlier layer's counts and grow an expert's span past
/// what the sorted map holds.
///
/// Buffers: counts, ids, n_rows, n_expert.
kernel void q4e_moe_sort_count_kernel(
        device atomic_int *counts [[buffer(0)]],
        device const int *ids [[buffer(1)]],
        constant uint &n_rows [[buffer(2)]],
        constant uint &n_expert [[buffer(3)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]]) {
    const uint i = gid * tpg + tid;
    if (i >= n_rows) {
        return;
    }
    const int e = ids[i];
    if (e >= 0 && uint(e) < n_expert) {
        atomic_fetch_add_explicit(&counts[e], 1, memory_order_relaxed);
    }
}

/// Exclusive prefix sums over the expert axis, and the list of experts that
/// received a column.
///
/// `kernels/cuda/q4e/moe.cu:211`. One threadgroup, one thread, serial: a few
/// hundred entries once per layer is far below the point where a parallel scan
/// would pay for itself. The number of active experts goes in `active[n_expert]`
/// and the column total in `bounds[n_expert]`.
///
/// Buffers: bounds, active, counts, n_expert.
kernel void q4e_moe_sort_scan_kernel(
        device int *bounds [[buffer(0)]],
        device int *active [[buffer(1)]],
        device const int *counts [[buffer(2)]],
        constant uint &n_expert [[buffer(3)]],
        uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0u) {
        return;
    }
    int total = 0;
    int n_active = 0;
    for (uint e = 0; e < n_expert; ++e) {
        bounds[e] = total;
        total += counts[e];
        if (counts[e] > 0) {
            active[n_active++] = int(e);
        }
    }
    bounds[n_expert] = total;
    active[n_expert] = n_active;
}

/// Places each routed column at its expert's next free slot.
///
/// `kernels/cuda/q4e/moe.cu:228`. The atomic leaves the order inside one
/// expert's span unspecified, which does not affect the result: every column
/// writes its own output row. `cursor` is cleared by the host, like `counts`.
///
/// Buffers: sorted, cursor, bounds, ids, n_rows, n_expert.
kernel void q4e_moe_sort_scatter_kernel(
        device int *sorted [[buffer(0)]],
        device atomic_int *cursor [[buffer(1)]],
        device const int *bounds [[buffer(2)]],
        device const int *ids [[buffer(3)]],
        constant uint &n_rows [[buffer(4)]],
        constant uint &n_expert [[buffer(5)]],
        uint gid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]]) {
    const uint i = gid * tpg + tid;
    if (i >= n_rows) {
        return;
    }
    const int e = ids[i];
    if (e < 0 || uint(e) >= n_expert) {
        return;
    }
    const int slot = atomic_fetch_add_explicit(&cursor[e], 1, memory_order_relaxed);
    sorted[bounds[e] + slot] = int(i);
}

/// Softmax over every expert, the best `n_used` of them, renormalized over
/// those.
///
/// `kernels/cuda/q4e/moe.cu:23`. One threadgroup per token. Three details are
/// load bearing and each is easy to lose in translation:
///
///   * the logit row is staged in `smem` -- the dynamic threadgroup allocation
///     the host sizes at `n_expert * 4` bytes -- BEFORE the selection rounds,
///     because each round masks the winner off by writing `-INFINITY` into it
///     and the device logits must not be touched;
///   * the argmax keeps the LOWER expert id among equal values. `e` only grows
///     across a thread's strided sweep, so keeping the first of equal values
///     keeps the lower id, and `q35moe_block_argmax` carries the same rule
///     across threads;
///   * the divisor is clamped BEFORE the division, not after. `fmax(wsum,
///     1e-20f)` then `w / wsum`; clamping the quotient instead would hand back
///     a different number for every row whose selected mass underflows.
///
/// The selection is `n_used` rounds of block argmax rather than one serial
/// thread: the obvious serial form spills the running top-k arrays to local
/// memory and leaves the rest of the threadgroup idle.
///
/// Buffers: ids, weights, logits, n_expert, n_used. Threadgroup: `n_expert`
/// floats at index 0.
kernel void q4e_moe_route_kernel(
        device int *ids [[buffer(0)]],
        device float *weights [[buffer(1)]],
        device const float *logits [[buffer(2)]],
        constant uint &n_expert [[buffer(3)]],
        constant uint &n_used [[buffer(4)]],
        threadgroup float *smem [[threadgroup(0)]],
        uint t [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]]) {
    device const float *row = logits + ulong(t) * n_expert;
    threadgroup float reduce[Q35MOE_BLOCK_SLOTS];
    threadgroup int reduce_i[Q35MOE_BLOCK_SLOTS];

    float m = -INFINITY;
    for (uint e = tid; e < n_expert; e += tpg) {
        const float v = row[e];
        smem[e] = v;
        m = fmax(m, v);
    }
    // The reduction's barriers are also what publish `smem` to the threadgroup.
    const float s_max = q35moe_block_max(m, reduce, tid, tpg);

    float sum = 0.0f;
    for (uint e = tid; e < n_expert; e += tpg) {
        sum += precise::exp(smem[e] - s_max);
    }
    const float inv_sum = 1.0f / q35moe_block_sum(sum, reduce, tid, tpg);

    // 16, the reference's bound, which covers qwen3.5-moe's 8 with room to
    // spare. The CUDA body says its entry point rejects anything wider;
    // `crates/shiso-engine/src/moe.rs` has no such check today, so a shape with
    // `n_used > 16` would overrun these two arrays here and the `__shared__`
    // pair there, identically. The guard belongs on the host, where one check
    // covers both backends, and it is not in this file's gift to add.
    threadgroup float chosen_w[16];
    threadgroup int chosen_i[16];
    for (uint s = 0; s < n_used; ++s) {
        float best = -INFINITY;
        int best_i = -1;
        // `e` only grows, so keeping the first of equal values is the lower id.
        for (uint e = tid; e < n_expert; e += tpg) {
            const float v = smem[e];
            if (v > best) {
                best = v;
                best_i = int(e);
            }
        }
        float top = 0.0f;
        int top_i = -1;
        q35moe_block_argmax(best, best_i, &top, &top_i, reduce, reduce_i, tid, tpg);
        if (tid == 0u) {
            chosen_w[s] = top;
            chosen_i[s] = top_i;
            if (top_i >= 0) {
                smem[top_i] = -INFINITY;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (tid == 0u) {
        float wsum = 0.0f;
        for (uint s = 0; s < n_used; ++s) {
            chosen_w[s] = precise::exp(chosen_w[s] - s_max) * inv_sum;
            wsum += chosen_w[s];
        }
        // The reference clamps the divisor before dividing.
        wsum = fmax(wsum, 1e-20f);
        for (uint s = 0; s < n_used; ++s) {
            ids[ulong(t) * n_used + s] = chosen_i[s];
            weights[ulong(t) * n_used + s] = chosen_w[s] / wsum;
        }
    }
}
