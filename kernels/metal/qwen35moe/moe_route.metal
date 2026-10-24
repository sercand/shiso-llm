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
/// `kernels/cuda/q4e/moe.cu:381`. One simdgroup; lane `l` owns a contiguous
/// run of the experts, so a lane's exclusive prefix over its neighbours'
/// totals is where its run starts, for the bounds and for the active list
/// alike. Both come out exactly as a serial walk writes them: ascending expert
/// order. The serial walk this replaced cost 60 us a layer on an M1 Max. The
/// number of active experts goes in `active[n_expert]` and the column total in
/// `bounds[n_expert]`.
///
/// Buffers: bounds, active, counts, n_expert.
kernel void q4e_moe_sort_scan_kernel(
        device int *bounds [[buffer(0)]],
        device int *active [[buffer(1)]],
        device const int *counts [[buffer(2)]],
        constant uint &n_expert [[buffer(3)]],
        uint tid [[thread_position_in_threadgroup]]) {
    const uint lane = tid;
    if (lane >= 32u) {
        return;
    }
    const uint per = (n_expert + 31u) / 32u;
    const uint lo = min(lane * per, n_expert);
    const uint hi = min(lo + per, n_expert);
    int rows = 0;
    int live = 0;
    for (uint e = lo; e < hi; ++e) {
        const int c = counts[e];
        rows += c;
        live += c > 0 ? 1 : 0;
    }
    int rows_at = rows;
    int live_at = live;
    for (ushort d = 1; d < 32; d <<= 1) {
        const int r = simd_shuffle_up(rows_at, d);
        const int a = simd_shuffle_up(live_at, d);
        if (lane >= d) {
            rows_at += r;
            live_at += a;
        }
    }
    int total = rows_at - rows;
    int n_active = live_at - live;
    for (uint e = lo; e < hi; ++e) {
        const int c = counts[e];
        bounds[e] = total;
        total += c;
        if (c > 0) {
            active[n_active++] = int(e);
        }
    }
    if (lane == 31u) {
        bounds[n_expert] = total;
        active[n_expert] = n_active;
    }
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

// Tokens, experts and routed rows the one-launch selection and sort covers
// (`kernels/cuda/q4e/moe.cu`'s `Q4E_ROUTE_SORT_*`), and the threadgroup
// `q4e_moe_route_kernel` runs at, whose sum order it reproduces.
#define Q4E_ROUTE_SORT_TOKENS 32u
#define Q4E_ROUTE_SORT_EXPERTS 512u
#define Q4E_ROUTE_SORT_ROWS (Q4E_ROUTE_SORT_TOKENS * 16u)
#define Q4E_ROUTE_BLOCK 256u

/// The routed selection, the sort by expert and the sort's inverse in one
/// threadgroup: `q4e_moe_route_kernel` over every token followed by
/// `q4e_moe_sort_count_kernel`, `q4e_moe_sort_scan_kernel` and
/// `q4e_moe_sort_scatter_kernel`, bit for bit.
///
/// `kernels/cuda/q4e/moe.cu:107`. One simdgroup a token, its logits in
/// registers:
///
///   * the max and the argmax rounds are order-independent (the argmax is a
///     total order: value, then the lower id);
///   * the softmax sum is `q35moe_block_sum`'s order over the 256-thread block
///     the route kernel runs at, emulated: virtual thread `vt = lane + 32 v`
///     sums `e = vt, vt + 256, ...`, each virtual simdgroup folds with
///     `q35moe_simd_sum`, then the eight simdgroup sums fold again;
///   * the chosen weights are the same serial loop, `precise::exp` included;
///   * bounds and the active list are the scan's ascending walk, and a
///     column's place inside its expert is its rank among that expert's
///     columns, which the atomic scatter left to chance (nothing reads it).
///
/// The counts live in threadgroup memory, so the two clears the three-pass
/// sort needs are gone with it. `n_expert` at most 512 and a multiple of 32,
/// `n_tok` at most 32, `n_used` at most 16.
///
/// Buffers: ids, weights, logits, bounds, active, sorted, pos, n_tok,
/// n_expert, n_used. One threadgroup of `32 * 32` threads.
[[max_total_threads_per_threadgroup(1024)]]
kernel void q4e_moe_route_sort_small_kernel(
        device int *ids [[buffer(0)]],
        device float *weights [[buffer(1)]],
        device const float *logits [[buffer(2)]],
        device int *bounds [[buffer(3)]],
        device int *active [[buffer(4)]],
        device int *sorted [[buffer(5)]],
        device int *pos [[buffer(6)]],
        constant uint &n_tok [[buffer(7)]],
        constant uint &n_expert [[buffer(8)]],
        constant uint &n_used [[buffer(9)]],
        uint tid [[thread_position_in_threadgroup]],
        uint tpg [[threads_per_threadgroup]]) {
    threadgroup int s_ids[Q4E_ROUTE_SORT_ROWS];
    threadgroup int s_bounds[Q4E_ROUTE_SORT_EXPERTS + 1u];
    threadgroup atomic_int s_counts[Q4E_ROUTE_SORT_EXPERTS];
    threadgroup float s_cw[Q4E_ROUTE_SORT_TOKENS][16];
    threadgroup int s_ci[Q4E_ROUTE_SORT_TOKENS][16];
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    constexpr uint PER = Q4E_ROUTE_SORT_EXPERTS / 32u;

    for (uint e = tid; e < n_expert; e += tpg) {
        atomic_store_explicit(&s_counts[e], 0, memory_order_relaxed);
    }

    if (warp < n_tok) {
        device const float *row = logits + ulong(warp) * n_expert;
        float v[PER];
        float m = -INFINITY;
        for (uint j = 0; j < PER; ++j) {
            const uint e = lane + 32u * j;
            v[j] = e < n_expert ? row[e] : -INFINITY;
            if (e < n_expert) {
                m = fmax(m, v[j]);
            }
        }
        const float s_max = simd_max(m);
        // `q35moe_block_sum` over a 256-thread block: virtual simdgroup `vw`'s
        // lane `l` is virtual thread `l + 32 vw`, which sums registers
        // `vw + 8 k`.
        float warp_total = 0.0f;
        for (uint vw = 0; vw < Q4E_ROUTE_BLOCK / 32u; ++vw) {
            float sum = 0.0f;
            for (uint k = 0; k < PER / (Q4E_ROUTE_BLOCK / 32u); ++k) {
                const uint j = vw + (Q4E_ROUTE_BLOCK / 32u) * k;
                if (lane + 32u * j < n_expert) {
                    sum += precise::exp(v[j] - s_max);
                }
            }
            sum = q35moe_simd_sum(sum);
            const float s_w = simd_broadcast(sum, ushort(0));
            if (lane == vw) {
                warp_total = s_w;
            }
        }
        warp_total = q35moe_simd_sum(warp_total);
        const float inv_sum = 1.0f / simd_broadcast(warp_total, ushort(0));

        threadgroup float *chosen_w = s_cw[warp];
        threadgroup int *chosen_i = s_ci[warp];
        for (uint s = 0; s < 16u; ++s) {
            if (s >= n_used) {
                break;
            }
            float best = -INFINITY;
            int best_i = -1;
            for (uint j = 0; j < PER; ++j) {
                const uint e = lane + 32u * j;
                if (e < n_expert && v[j] > best) {
                    best = v[j];
                    best_i = int(e);
                }
            }
            for (ushort d = 16; d > 0; d >>= 1) {
                const float ov = simd_shuffle_down(best, d);
                const int oi = simd_shuffle_down(best_i, d);
                if (ov > best || (ov == best && oi >= 0 && (best_i < 0 || oi < best_i))) {
                    best = ov;
                    best_i = oi;
                }
            }
            const float top = simd_broadcast(best, ushort(0));
            const int top_i = simd_broadcast(best_i, ushort(0));
            if (lane == 0u) {
                chosen_w[s] = top;
                chosen_i[s] = top_i;
            }
            if (top_i >= 0 && uint(top_i) % 32u == lane) {
                for (uint j = 0; j < PER; ++j) {
                    if (uint(top_i) / 32u == j) {
                        v[j] = -INFINITY;
                    }
                }
            }
        }
        if (lane == 0u) {
            float wsum = 0.0f;
            for (uint s = 0; s < n_used; ++s) {
                chosen_w[s] = precise::exp(chosen_w[s] - s_max) * inv_sum;
                wsum += chosen_w[s];
            }
            wsum = fmax(wsum, 1e-20f);
            for (uint s = 0; s < n_used; ++s) {
                ids[ulong(warp) * n_used + s] = chosen_i[s];
                weights[ulong(warp) * n_used + s] = chosen_w[s] / wsum;
                s_ids[warp * n_used + s] = chosen_i[s];
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint n_rows = n_tok * n_used;
    for (uint i = tid; i < n_rows; i += tpg) {
        const int e = s_ids[i];
        if (e >= 0 && uint(e) < n_expert) {
            atomic_fetch_add_explicit(&s_counts[e], 1, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (warp == 0u) {
        // `q4e_moe_sort_scan_kernel`'s walk: lane `l` owns a contiguous run.
        const uint per = (n_expert + 31u) / 32u;
        const uint lo = min(lane * per, n_expert);
        const uint hi = min(lo + per, n_expert);
        int rows = 0;
        int live = 0;
        for (uint e = lo; e < hi; ++e) {
            const int c = atomic_load_explicit(&s_counts[e], memory_order_relaxed);
            rows += c;
            live += c > 0 ? 1 : 0;
        }
        int rows_at = rows;
        int live_at = live;
        for (ushort d = 1; d < 32; d <<= 1) {
            const int r = simd_shuffle_up(rows_at, d);
            const int a = simd_shuffle_up(live_at, d);
            if (lane >= d) {
                rows_at += r;
                live_at += a;
            }
        }
        int total = rows_at - rows;
        int n_active = live_at - live;
        for (uint e = lo; e < hi; ++e) {
            const int c = atomic_load_explicit(&s_counts[e], memory_order_relaxed);
            s_bounds[e] = total;
            bounds[e] = total;
            total += c;
            if (c > 0) {
                active[n_active++] = int(e);
            }
        }
        if (lane == 31u) {
            s_bounds[n_expert] = total;
            bounds[n_expert] = total;
            active[n_expert] = n_active;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < n_rows; i += tpg) {
        const int e = s_ids[i];
        if (e < 0 || uint(e) >= n_expert) {
            continue;
        }
        int rank = 0;
        for (uint j = 0; j < i; ++j) {
            rank += s_ids[j] == e ? 1 : 0;
        }
        const int at = s_bounds[e] + rank;
        sorted[at] = int(i);
        pos[i] = at;
    }
}
