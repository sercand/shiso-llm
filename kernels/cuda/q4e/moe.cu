/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels), D2 (explicit template instantiations),
 * D3 (external linkage on the template kernels: an explicit instantiation of a `static` template is emitted as a non-`.visible` entry the driver cannot resolve).
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * Mixture of experts.
 * ------------------------------------------------------------------------ */

/* Softmax over all experts, then the top n_used by probability, then
 * renormalize over the chosen ones.
 *
 * The selection is n_used rounds of block argmax over a shared copy of the
 * logits, each round masking off what it took.  Doing it serially in one
 * thread instead -- the obvious way for k = 10 -- spills the running top-k
 * arrays to local memory and leaves 255 threads idle, which measured at 72 us
 * per layer, over 3 ms of every decode step. */
extern "C" __global__ void q4e_moe_route_kernel(
        int32_t *ids, float *weights, const float *logits,
        uint32_t n_expert, uint32_t n_used) {
    const uint32_t t = blockIdx.x;
    const float *row = logits + (uint64_t)t * n_expert;
    extern __shared__ float q4e_route_smem[];      /* n_expert */

    float m = -INFINITY;
    for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) {
        const float v = row[e];
        q4e_route_smem[e] = v;
        m = fmaxf(m, v);
    }
    const float s_max = q4e_block_max(m);

    float sum = 0.0f;
    for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) {
        sum += __expf(q4e_route_smem[e] - s_max);
    }
    const float inv_sum = 1.0f / q4e_block_sum(sum);

    /* 16 covers SHS_MAX_EXPERT_USED (10) with room to spare; the entry point
     * below rejects anything wider. */
    __shared__ float chosen_w[16];
    __shared__ int32_t chosen_i[16];
    for (uint32_t s = 0; s < n_used; s++) {
        float best = -INFINITY;
        int32_t best_i = -1;
        /* e only grows, so keeping the first of equal values is the lower id. */
        for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) {
            const float v = q4e_route_smem[e];
            if (v > best) { best = v; best_i = (int32_t)e; }
        }
        float top = 0.0f;
        int32_t top_i = -1;
        q4e_block_argmax(best, best_i, &top, &top_i);
        if (threadIdx.x == 0) {
            chosen_w[s] = top;
            chosen_i[s] = top_i;
            if (top_i >= 0) q4e_route_smem[top_i] = -INFINITY;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float wsum = 0.0f;
        for (uint32_t s = 0; s < n_used; s++) {
            chosen_w[s] = __expf(chosen_w[s] - s_max) * inv_sum;
            wsum += chosen_w[s];
        }
        /* The reference clamps the divisor before dividing. */
        wsum = fmaxf(wsum, 1e-20f);
        for (uint32_t s = 0; s < n_used; s++) {
            ids[(uint64_t)t * n_used + s] = chosen_i[s];
            weights[(uint64_t)t * n_used + s] = chosen_w[s] / wsum;
        }
    }
}

/* q4e_moe_route_kernel, q4e_moe_sort_*_kernel and the sort's inverse in one
 * block, for a pass of at most Q4E_ROUTE_SORT_TOKENS tokens (shiso).
 *
 * At a decode step the router's selection and the counting sort were seven
 * launches of a block or less each - 17 us a layer of dependent latency in
 * front of the routed GEMMs.  Here warp w routes token w with the logits in
 * its registers, and the block then sorts every routed column in shared
 * memory.  Same results, bit for bit:
 *   - the max and the argmax rounds are order-independent (the argmax is a
 *     total order: value, then the lower id);
 *   - the softmax sum is q4e_block_sum's exact order, the 256-thread block
 *     emulated: virtual thread vt = lane + 32 v sums e = vt, vt + 256, ...,
 *     each virtual warp reduces with warp_sum_f32, then the eight warp sums;
 *   - the chosen weights are the same serial loop;
 *   - bounds and the active list are the scan's ascending walk, and a column's
 *     place inside its expert is its rank among that expert's columns, which
 *     the atomic scatter left to chance (nothing reads that order).
 * n_expert is at most Q4E_ROUTE_SORT_EXPERTS and a multiple of 32. */
#define Q4E_ROUTE_SORT_TOKENS 32u
#define Q4E_ROUTE_SORT_EXPERTS 512u
#define Q4E_ROUTE_SORT_ROWS (Q4E_ROUTE_SORT_TOKENS * 16u)
#define Q4E_ROUTE_BLOCK 256u  /* the block q4e_moe_route_kernel runs at */

extern "C" __global__ void __launch_bounds__(32u * Q4E_ROUTE_SORT_TOKENS)
q4e_moe_route_sort_small_kernel(
        int32_t *ids, float *weights, const float *logits,
        int32_t *bounds, int32_t *active, int32_t *sorted, int32_t *pos,
        uint32_t n_tok, uint32_t n_expert, uint32_t n_used) {
    __shared__ int32_t s_ids[Q4E_ROUTE_SORT_ROWS];
    __shared__ int32_t s_bounds[Q4E_ROUTE_SORT_EXPERTS + 1u];
    __shared__ int32_t s_counts[Q4E_ROUTE_SORT_EXPERTS];
    const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    constexpr uint32_t PER = Q4E_ROUTE_SORT_EXPERTS / 32u;

    for (uint32_t e = threadIdx.x; e < n_expert; e += blockDim.x) s_counts[e] = 0;

    if (warp < n_tok) {
        const float *row = logits + (uint64_t)warp * n_expert;
        float v[PER];
        float m = -INFINITY;
#pragma unroll
        for (uint32_t j = 0; j < PER; j++) {
            const uint32_t e = lane + 32u * j;
            v[j] = e < n_expert ? row[e] : -INFINITY;
            if (e < n_expert) m = fmaxf(m, v[j]);
        }
        m = warp_max_f32(m);
        const float s_max = __shfl_sync(0xffffffffu, m, 0);
        /* q4e_block_sum over a 256-thread block: warp sums, then their sum. */
        float warp_total = 0.0f;
        for (uint32_t vw = 0; vw < Q4E_ROUTE_BLOCK / 32u; vw++) {
            const uint32_t vt = lane + 32u * vw;
            float sum = 0.0f;
            for (uint32_t e = vt; e < n_expert; e += Q4E_ROUTE_BLOCK) {
                sum += __expf(v[e / 32u] - s_max);
            }
            sum = warp_sum_f32(sum);
            const float s_w = __shfl_sync(0xffffffffu, sum, 0);
            if (lane == vw) warp_total = s_w;
        }
        warp_total = warp_sum_f32(warp_total);
        const float inv_sum = 1.0f / __shfl_sync(0xffffffffu, warp_total, 0);

        float chosen_w[16];
        int32_t chosen_i[16];
        for (uint32_t s = 0; s < n_used; s++) {
            float best = -INFINITY;
            int32_t best_i = -1;
#pragma unroll
            for (uint32_t j = 0; j < PER; j++) {
                const uint32_t e = lane + 32u * j;
                if (e < n_expert && v[j] > best) { best = v[j]; best_i = (int32_t)e; }
            }
            for (uint32_t d = 16u; d > 0u; d >>= 1) {
                const float ov = __shfl_down_sync(0xffffffffu, best, d);
                const int32_t oi = __shfl_down_sync(0xffffffffu, best_i, d);
                if (ov > best || (ov == best && oi >= 0 && (best_i < 0 || oi < best_i))) { best = ov; best_i = oi; }
            }
            const float top = __shfl_sync(0xffffffffu, best, 0);
            const int32_t top_i = __shfl_sync(0xffffffffu, best_i, 0);
            chosen_w[s] = top;
            chosen_i[s] = top_i;
            if (top_i >= 0 && (uint32_t)top_i % 32u == lane) {
#pragma unroll
                for (uint32_t j = 0; j < PER; j++)
                    if ((uint32_t)top_i / 32u == j) v[j] = -INFINITY;
            }
        }
        if (lane == 0u) {
            float wsum = 0.0f;
            for (uint32_t s = 0; s < n_used; s++) {
                chosen_w[s] = __expf(chosen_w[s] - s_max) * inv_sum;
                wsum += chosen_w[s];
            }
            wsum = fmaxf(wsum, 1e-20f);
            for (uint32_t s = 0; s < n_used; s++) {
                ids[(uint64_t)warp * n_used + s] = chosen_i[s];
                weights[(uint64_t)warp * n_used + s] = chosen_w[s] / wsum;
                s_ids[warp * n_used + s] = chosen_i[s];
            }
        }
    }
    __syncthreads();

    const uint32_t n_rows = n_tok * n_used;
    for (uint32_t i = threadIdx.x; i < n_rows; i += blockDim.x) {
        const int32_t e = s_ids[i];
        if (e >= 0 && (uint32_t)e < n_expert) atomicAdd(&s_counts[e], 1);
    }
    __syncthreads();
    if (warp == 0u) {
        /* q4e_moe_sort_scan_kernel's walk: lane l owns a contiguous run. */
        const uint32_t per = (n_expert + 31u) / 32u;
        const uint32_t lo = min(lane * per, n_expert), hi = min(lo + per, n_expert);
        int32_t rows = 0, live = 0;
        for (uint32_t e = lo; e < hi; e++) {
            const int32_t c = s_counts[e];
            rows += c;
            live += c > 0;
        }
        int32_t rows_at = rows, live_at = live;
        for (int d = 1; d < 32; d <<= 1) {
            const int32_t r = __shfl_up_sync(0xffffffffu, rows_at, d);
            const int32_t a = __shfl_up_sync(0xffffffffu, live_at, d);
            if ((int)lane >= d) { rows_at += r; live_at += a; }
        }
        int32_t total = rows_at - rows, n_active = live_at - live;
        for (uint32_t e = lo; e < hi; e++) {
            const int32_t c = s_counts[e];
            s_bounds[e] = total;
            bounds[e] = total;
            total += c;
            if (c > 0) active[n_active++] = (int32_t)e;
        }
        if (lane == 31u) {
            s_bounds[n_expert] = total;
            bounds[n_expert] = total;
            active[n_expert] = n_active;
        }
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n_rows; i += blockDim.x) {
        const int32_t e = s_ids[i];
        if (e < 0 || (uint32_t)e >= n_expert) continue;
        int32_t rank = 0;
        for (uint32_t j = 0; j < i; j++) rank += s_ids[j] == e;
        const int32_t at = s_bounds[e] + rank;
        sorted[at] = (int32_t)i;
        pos[i] = at;
    }
}

/* out = silu(gate) * up over the routed intermediate. */
extern "C" __global__ void q4e_swiglu_kernel(float *out, const float *gate, const float *up, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float g = gate[i];
    out[i] = (g / (1.0f + __expf(-g))) * up[i];
}

/* Weighted sum of the n_used expert outputs for each token.  The MoE matmul
 * writes column-major over (token, slot), so slot s of token t is column
 * t * n_used + s. */
extern "C" __global__ void q4e_moe_combine_kernel(
        float *out, const float *down, const float *weights,
        uint32_t n_embd, uint32_t n_used) {
    const uint32_t t = blockIdx.y;
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n_embd;
         e += gridDim.x * blockDim.x) {
        float acc = 0.0f;
        for (uint32_t s = 0; s < n_used; s++) {
            const uint64_t col = (uint64_t)t * n_used + s;
            acc += down[col * n_embd + e] * weights[col];
        }
        out[(uint64_t)t * n_embd + e] = acc;
    }
}

/* moe_out += shared_out * sigmoid(shared_logit) -- the shared expert carries a
 * single scalar gate per token, not a per-channel one. */
extern "C" __global__ void q4e_shared_add_kernel(
        float *out, const float *shared, const float *logit, uint32_t n_embd) {
    const uint32_t t = blockIdx.y;
    const float g = 1.0f / (1.0f + __expf(-logit[t]));
    for (uint32_t e = blockIdx.x * blockDim.x + threadIdx.x; e < n_embd;
         e += gridDim.x * blockDim.x) {
        out[(uint64_t)t * n_embd + e] += shared[(uint64_t)t * n_embd + e] * g;
    }
}

/* Rows of x replicated once per routing slot: mgemm reads one input slab per
 * slot, and slot t * n_used + s is token t. */
extern "C" __global__ void q4e_moe_rows_per_slot_kernel(
        float *out, const float *x, uint32_t in_dim, uint32_t n_used, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t slot = i / in_dim;
    out[i] = x[(slot / n_used) * in_dim + (i - slot * in_dim)];
}

/* One dequantized weight, for the register-resident batched kernel below.
 * Lane `lane` of block `blk` is weight `lane` of that block. */
template <int TYPE>
__device__ __forceinline__ static float q4e_down_weight(const uint8_t *blk, uint32_t lane) {
    if (TYPE == 7) {
        /* Q5_1: scale and minimum, four bits in qs plus a fifth in qh.  Lane l
         * wants weight l, which lives in nibble (l >> 4) of qs[l & 15] with its
         * top bit at position l of qh. */
        const float d = __half2float(*(const __half *)blk);
        const float mn = __half2float(*(const __half *)(blk + 2));
        uint32_t qh;
        memcpy(&qh, blk + 4, sizeof(qh));
        const uint8_t packed = blk[8u + (lane & 15u)];
        const uint32_t nib = (lane < 16u) ? (packed & 0x0Fu) : (packed >> 4);
        return fmaf(d, (float)(nib | (((qh >> lane) & 1u) << 4)), mn);
    }
    const float d = __half2float(*(const __half *)blk);
    return d * (float)((const int8_t *)(blk + 2))[lane];
}

/* Expert-grouped routed down projection, for prefill.
 *
 * The per-(token, slot) kernel below reads a whole expert row for every column
 * that selected it.  At one token that is exactly the traffic the model needs;
 * at a 512-token chunk it is up to 512 times too much, and it is why prefill
 * measured 120 tok/s.  Here the columns are sorted by expert first, so a warp
 * loads its output row's weights into registers once and then sweeps every
 * column routed to that expert.
 *
 * BLOCKS is a template parameter so the weight array stays in registers; a
 * runtime bound would spill it to local memory and undo the point. */
template <int TYPE, int BLOCKS>
__global__ void q4e_moe_down_grouped_kernel(
        float *dst, const uint8_t *W, const float *x,
        const int32_t *sorted_cols, const int32_t *bounds, const int32_t *active,
        uint32_t n_expert, uint32_t out_dim, uint32_t in_dim, uint32_t expert_bytes) {
    /* blockIdx.y indexes the experts that received a column, not all 512: a
     * verify batch touches ~40 of them, and 300k empty blocks per layer were
     * half the kernel's time. */
    if ((int32_t)blockIdx.y >= active[n_expert]) return;
    const uint32_t expert = (uint32_t)active[blockIdx.y];
    const int32_t c0 = bounds[expert];
    const int32_t c1 = bounds[expert + 1u];
    if (c1 <= c0) return;

    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const uint32_t block_bytes = (TYPE == 7) ? 24u : 34u;
    const uint8_t *wrow = W + (uint64_t)expert * expert_bytes +
                          (uint64_t)row * BLOCKS * block_bytes;

    float wv[BLOCKS];
#pragma unroll
    for (int b = 0; b < BLOCKS; b++) {
        wv[b] = q4e_down_weight<TYPE>(wrow + (uint64_t)b * block_bytes, lane);
    }

    for (int32_t c = c0; c < c1; c++) {
        const uint32_t col = (uint32_t)sorted_cols[c];
        const float *xrow = x + (uint64_t)col * in_dim;
        float acc = 0.0f;
#pragma unroll
        for (int b = 0; b < BLOCKS; b++) acc = fmaf(wv[b], xrow[b * 32u + lane], acc);
        acc = warp_sum_f32(acc);
        if (lane == 0u) dst[(uint64_t)col * out_dim + row] = acc;
    }
}

/* Counting sort of the (token, slot) columns by expert.  Three tiny kernels:
 * histogram, exclusive scan over the expert axis, scatter.  The scatter's
 * atomicAdd leaves the order inside one expert unspecified, which does not
 * affect the result: every column writes its own output row. */
extern "C" __global__ void q4e_moe_sort_count_kernel(
        int32_t *counts, const int32_t *ids, uint32_t n_rows, uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    const int32_t e = ids[i];
    if (e >= 0 && (uint32_t)e < n_expert) atomicAdd(&counts[e], 1);
}

extern "C" __global__ void q4e_moe_sort_scan_kernel(
        int32_t *bounds, int32_t *active, const int32_t *counts, uint32_t n_expert) {
    /* One warp; lane l owns a contiguous run of the experts, so a lane's
     * exclusive prefix over its neighbours' totals is where its run starts,
     * for the bounds and for the active list alike.  Both come out exactly as
     * a serial walk writes them: ascending expert order.  The serial walk this
     * replaced cost 28 us a layer, one dependent global load per expert,
     * which was 5% of a decode step. */
    const uint32_t lane = threadIdx.x;
    if (lane >= 32u) return;
    const uint32_t per = (n_expert + 31u) / 32u;
    const uint32_t lo = min(lane * per, n_expert), hi = min(lo + per, n_expert);
    int32_t rows = 0, live = 0;
    for (uint32_t e = lo; e < hi; e++) {
        const int32_t c = counts[e];
        rows += c;
        live += c > 0;
    }
    int32_t rows_at = rows, live_at = live;
    for (int d = 1; d < 32; d <<= 1) {
        const int32_t r = __shfl_up_sync(0xffffffffu, rows_at, d);
        const int32_t a = __shfl_up_sync(0xffffffffu, live_at, d);
        if ((int)lane >= d) { rows_at += r; live_at += a; }
    }
    int32_t total = rows_at - rows, n_active = live_at - live;
    for (uint32_t e = lo; e < hi; e++) {
        const int32_t c = counts[e];
        bounds[e] = total;
        total += c;
        if (c > 0) active[n_active++] = (int32_t)e;
    }
    if (lane == 31u) {
        bounds[n_expert] = total;
        active[n_expert] = n_active;
    }
}

extern "C" __global__ void q4e_moe_sort_scatter_kernel(
        int32_t *sorted, int32_t *cursor, const int32_t *bounds,
        const int32_t *ids, uint32_t n_rows, uint32_t n_expert) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rows) return;
    const int32_t e = ids[i];
    if (e < 0 || (uint32_t)e >= n_expert) return;
    const int32_t slot = atomicAdd(&cursor[e], 1);
    sorted[bounds[e] + slot] = (int32_t)i;
}

/* Routed down projection.
 *
 * The down experts have K = moe_ff (640): a whole number of 32-weight legacy
 * blocks but not of a 256-weight super-block, and each (token, slot) pair
 * needs its own input row rather than one row broadcast across a token's
 * experts.  The vendored mmvq path does not handle that shape correctly for
 * Q5_1, so this walks the blocks directly.
 *
 * One warp owns one output row of one (token, slot), and lane l takes weight
 * l of every block, so the warp consumes each block in one pass and its loads
 * coalesce.  Splitting the 20 blocks across threads instead leaves 20 of 128
 * threads doing all the work, which measured at 69 GB/s -- a third of what
 * this machine can read.  dst is column major over (token, slot), matching
 * what the swiglu stage produced. */
template <int TYPE>
__global__ void q4e_moe_down_kernel(
        float *dst, const uint8_t *W, const float *x, const int32_t *ids,
        uint32_t out_dim, uint32_t in_dim, uint32_t n_used, uint32_t expert_bytes) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    const uint32_t col = blockIdx.y;                 /* token * n_used + slot */
    if (row >= out_dim) return;
    const int32_t expert = ids[col];
    const uint32_t blocks = in_dim / 32u;
    const uint32_t block_bytes = (TYPE == 7) ? 24u : 34u;

    const uint8_t *wrow = W + (uint64_t)expert * expert_bytes +
                          (uint64_t)row * blocks * block_bytes;
    const float *xrow = x + (uint64_t)col * in_dim;

    float acc = 0.0f;
    for (uint32_t b = 0; b < blocks; b++) {
        const uint8_t *blk = wrow + (uint64_t)b * block_bytes;
        const float xv = xrow[b * 32u + lane];
        if (TYPE == 7) {
            /* Q5_1: scale and minimum, four bits in qs plus a fifth in qh.
             * Lane l wants weight l, which lives in nibble (l >> 4) of
             * qs[l & 15] with its top bit at position l of qh. */
            const float d = __half2float(*(const __half *)blk);
            const float mn = __half2float(*(const __half *)(blk + 2));
            uint32_t qh;
            memcpy(&qh, blk + 4, sizeof(qh));
            const uint8_t packed = blk[8u + (lane & 15u)];
            const uint32_t nib = (lane < 16u) ? (packed & 0x0Fu) : (packed >> 4);
            const uint32_t q = nib | (((qh >> lane) & 1u) << 4);
            acc = fmaf(fmaf(d, (float)q, mn), xv, acc);
        } else {
            const float d = __half2float(*(const __half *)blk);
            const int8_t q = ((const int8_t *)(blk + 2))[lane];
            acc = fmaf(d * (float)q, xv, acc);
        }
    }
    acc = warp_sum_f32(acc);
    if (lane == 0u) dst[(uint64_t)col * out_dim + row] = acc;
}

/* Explicit instantiations for exactly the argument sets the launchers
 * use.  `extern "C"` is illegal on a template, so these entries keep
 * their mangled names and are resolved by mangled name. */
template __global__ void q4e_moe_down_grouped_kernel<7, 20>(
        float *, const uint8_t *, const float *, const int32_t *,
        const int32_t *, const int32_t *,
        uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_moe_down_grouped_kernel<8, 20>(
        float *, const uint8_t *, const float *, const int32_t *,
        const int32_t *, const int32_t *,
        uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_moe_down_kernel<7>(
        float *, const uint8_t *, const float *, const int32_t *,
        uint32_t, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_moe_down_kernel<8>(
        float *, const uint8_t *, const float *, const int32_t *,
        uint32_t, uint32_t, uint32_t, uint32_t);
