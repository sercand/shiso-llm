/* The suppress stage: every listed token id gets -inf in every logits row, so
 * no pick taken afterwards - the greedy one on the device, or any host draw
 * over a row read back - can land on it.
 *
 * The list is `[n, id_0, ..., id_{n-1}]` in one device buffer, and the count
 * is read from the buffer rather than passed as an argument on purpose: a
 * captured graph replays its launch arguments verbatim, so a count baked into
 * the launch would freeze whatever list existed when the graph was captured.
 * Reading it here lets the list be written once, after the plans exist, with
 * no plan rebuilt and no graph invalidated. An id past the vocabulary is
 * skipped rather than written: the host refuses such a list, and this is the
 * second line that keeps a bad one from writing outside the row. */

#include <stdint.h>
#include <math.h>

extern "C" __global__ void sample_suppress_rows_kernel(float *logits, const uint32_t *list, uint32_t n_vocab) {
    const uint32_t n = list[0];
    float *row = logits + (uint64_t)blockIdx.x * n_vocab;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const uint32_t id = list[1u + i];
        if (id < n_vocab) row[id] = -INFINITY;
    }
}

/* The top of a logits row: its `k` highest entries, ascending by token id,
 * for a host draw that reads a top-k stage's worth of the row instead of the
 * whole of it.  "Highest" orders values the way the host's total order does
 * (larger first, ties to the lower id), so the set is the one the host's own
 * top-k over the full row keeps; the host draws from these exactly what it
 * would from the row whenever its top-k is at most `k`.
 *
 * One block per row.  A radix select, eight bits at a time from the top of an
 * order-preserving key, finds the k-th highest key T and how many entries
 * equal to T the set still needs; a compaction in index order then writes
 * every entry above T and the first `need` entries equal to T.  Index order is
 * what makes the output ascending by id and the ties the lowest ids.
 *
 * `values` and `ids` are `k` entries a row.  A row shorter than `k` writes
 * all of its entries and leaves the rest of its slots alone. */

static __device__ __forceinline__ uint32_t sample_order_key(float f) {
    /* Larger float, larger key: flip every bit of a negative, the sign bit of
     * a positive.  NaNs land at the ends as they do in a total order. */
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

extern "C" __global__ void __launch_bounds__(1024) sample_top_rows_kernel(
        const float *logits, uint32_t n_vocab, uint32_t k, float *values, uint32_t *ids) {
    const float *row = logits + (uint64_t)blockIdx.x * n_vocab;
    float *out_v = values + (uint64_t)blockIdx.x * k;
    uint32_t *out_i = ids + (uint64_t)blockIdx.x * k;
    const uint32_t tid = threadIdx.x;
    __shared__ uint32_t hist[256];
    __shared__ uint32_t s_prefix, s_need;

    /* The threshold: after the pass at `shift`, `prefix` holds the top bits
     * of T and `need` how many entries matching those bits the set takes. */
    uint32_t prefix = 0u;
    uint32_t need = k < n_vocab ? k : n_vocab;
    if (need == 0u) return;
    for (int shift = 24; shift >= 0; shift -= 8) {
        const uint32_t above = shift == 24 ? 0u : (0xFFFFFFFFu << (uint32_t)(shift + 8));
        for (uint32_t b = tid; b < 256u; b += blockDim.x) hist[b] = 0u;
        __syncthreads();
        for (uint32_t i = tid; i < n_vocab; i += blockDim.x) {
            const uint32_t key = sample_order_key(row[i]);
            if ((key & above) == prefix) atomicAdd(&hist[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        if (tid == 0u) {
            uint32_t taken = 0u;
            int bin = 255;
            for (; bin > 0; bin--) {
                if (taken + hist[bin] >= need) break;
                taken += hist[bin];
            }
            s_need = need - taken;
            s_prefix = prefix | ((uint32_t)bin << shift);
        }
        __syncthreads();
        prefix = s_prefix;
        need = s_need;
        __syncthreads();
    }

    /* Compaction in index order, a tile of blockDim entries at a time: each
     * entry's rank among the tile's ties and among its taken entries comes
     * from warp ballots and the per-warp counts before it. */
    __shared__ uint32_t s_ties[32], s_taken[32];
    const uint32_t lane = tid & 31u, warp = tid >> 5u, warps = blockDim.x >> 5u;
    const uint32_t below = (1u << lane) - 1u;
    uint32_t written = 0u, ties_seen = 0u;
    for (uint32_t base = 0u; base < n_vocab; base += blockDim.x) {
        const uint32_t i = base + tid;
        float v = 0.0f;
        bool over = false, tie = false;
        if (i < n_vocab) {
            v = row[i];
            const uint32_t key = sample_order_key(v);
            over = key > prefix;
            tie = key == prefix;
        }
        const uint32_t tie_bits = __ballot_sync(0xFFFFFFFFu, tie);
        if (lane == 0u) s_ties[warp] = __popc(tie_bits);
        __syncthreads();
        uint32_t ties_before = 0u, ties_tile = 0u;
        for (uint32_t w = 0u; w < warps; w++) {
            if (w < warp) ties_before += s_ties[w];
            ties_tile += s_ties[w];
        }
        const uint32_t tie_rank = ties_seen + ties_before + __popc(tie_bits & below);
        const bool take = over || (tie && tie_rank < need);
        const uint32_t take_bits = __ballot_sync(0xFFFFFFFFu, take);
        if (lane == 0u) s_taken[warp] = __popc(take_bits);
        __syncthreads();
        uint32_t taken_before = 0u, taken_tile = 0u;
        for (uint32_t w = 0u; w < warps; w++) {
            if (w < warp) taken_before += s_taken[w];
            taken_tile += s_taken[w];
        }
        if (take) {
            const uint32_t at = written + taken_before + __popc(take_bits & below);
            if (at < k) {
                out_v[at] = v;
                out_i[at] = i;
            }
        }
        written += taken_tile;
        ties_seen += ties_tile;
        __syncthreads();
    }
}
