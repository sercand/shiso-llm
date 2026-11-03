/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels).
 * D2 (shiso): q4e_argmax_rows_kernel reads its row sixteen floats a thread at a
 * time; the comparisons and the tie rule are unchanged, and so is the index it
 * returns (see the kernel).
 * No other line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 *
 * One exception, which has no source: q4e_copy_row_kernel at the end of this
 * file is this tree's own, written for the prompt-role segmented pass's
 * gathered vocabulary projection.
 */

#include "q4e.cuh"

extern "C" __global__ void q4e_add2_kernel(float *res, const float *a, const float *b, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) res[i] += a[i] + b[i];
}

/* Greedy choice per logits row, on the device.  Reading a 9-row verify batch
 * (8.9 MB) back to pageable host memory cost 12-83 ms per step at 26k
 * context; the step only needs the argmax of every row and the logits of the
 * one row it commits.  Ties go to the lower index, as the host sampler's. */
/* D2 (shiso): one block a row reads a 248,320-wide vocabulary in 243 dependent
 * rounds of four bytes a thread, 110 us a decode step on a GB10 (9 GB/s). Here
 * a thread holds four float4 loads in flight a round. A thread still walks its
 * own elements in increasing index order with the strict `>`, so it keeps the
 * lowest index of its largest value; the block fold keeps the lowest index of
 * equal values; together that is the lowest index whose value equals the row's
 * maximum, whichever elements each thread saw - the original's answer. */
#define Q4E_ARGMAX_DEPTH 4u
extern "C" __global__ void q4e_argmax_rows_kernel(int32_t *out, const float *logits, uint32_t n_vocab) {
    const float *row = logits + (uint64_t)blockIdx.x * n_vocab;
    float best = -INFINITY;
    int32_t best_i = -1;
    /* A row is sixteen-byte aligned when the width is a multiple of four. */
    const uint32_t n4 = (n_vocab & 3u) == 0u ? n_vocab / 4u : 0u;
    const float4 *row4 = reinterpret_cast<const float4 *>(row);
    for (uint32_t c = threadIdx.x; c < n4; c += blockDim.x * Q4E_ARGMAX_DEPTH) {
        float4 q[Q4E_ARGMAX_DEPTH];
#pragma unroll
        for (uint32_t d = 0; d < Q4E_ARGMAX_DEPTH; d++) {
            const uint32_t at = c + d * blockDim.x;
            q[d] = at < n4 ? row4[at] : make_float4(-INFINITY, -INFINITY, -INFINITY, -INFINITY);
        }
#pragma unroll
        for (uint32_t d = 0; d < Q4E_ARGMAX_DEPTH; d++) {
            const int32_t i = (int32_t)(4u * (c + d * blockDim.x));
            if (q[d].x > best) { best = q[d].x; best_i = i; }
            if (q[d].y > best) { best = q[d].y; best_i = i + 1; }
            if (q[d].z > best) { best = q[d].z; best_i = i + 2; }
            if (q[d].w > best) { best = q[d].w; best_i = i + 3; }
        }
    }
    for (uint32_t i = 4u * n4 + threadIdx.x; i < n_vocab; i += blockDim.x) {
        const float v = row[i];
        if (v > best) { best = v; best_i = (int32_t)i; }
    }
    float ov;
    int32_t oi;
    q4e_block_argmax(best, best_i, &ov, &oi);
    if (threadIdx.x == 0u) out[blockIdx.x] = oi;
}

// One row copied within a buffer: a prompt-role segmented pass gathers each
// run's last collapsed row to the front of the block, in run order, so one
// projection covers them all. Run r's last row is at or past row r, so no copy
// reads a row an earlier copy wrote.
extern "C" __global__ void q4e_copy_row_kernel(float *dst, const float *src, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i];
}

// The mean over the hyper-connection streams of a pass's residual rows, into
// one slot of a wider row: a block drafter reads a tapped layer's hidden state
// as that mean, every tap of one token side by side. Also this tree's own.
// res[t][h][e]; dst[t * dst_stride + dst_offset + e]. The streams are summed
// in order and divided once, as a mean over the stream axis is.
extern "C" __global__ void q4e_stream_mean_rows_kernel(
        float *dst, const float *res, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok,
        uint32_t dst_stride, uint32_t dst_offset) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (uint64_t)n_tok * n_embd) return;
    const uint32_t e = (uint32_t)(idx % n_embd);
    const uint64_t t = idx / n_embd;
    const float *row = res + t * n_hc * n_embd + e;
    float sum = 0.0f;
    for (uint32_t h = 0; h < n_hc; h++) sum += row[(uint64_t)h * n_embd];
    dst[t * dst_stride + dst_offset + e] = sum / (float)n_hc;
}

// The `k` highest entries of each slice of each logits row, highest first,
// ties to the lower id - the order a total order on the values gives. One block
// a (slice, row); `k` rounds, each taking the best entry strictly below the
// last one taken, so nothing but the threshold is kept between rounds and the
// answer does not depend on which thread saw what. A slice with fewer than `k`
// entries pads with id 0xffffffff and -inf. Also this tree's own: a block
// drafter reads sixteen candidates from each of seven rows, which the one
// block a row top pass reads in 715 us on a GB10 and this in a few dozen.
// values/ids: [row][slice][k].
static __device__ __forceinline__ uint32_t q4e_order_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static __device__ __forceinline__ bool q4e_top_better(uint32_t ka, uint32_t ia, uint32_t kb, uint32_t ib) {
    return ka > kb || (ka == kb && ia < ib);
}

extern "C" __global__ void __launch_bounds__(256) q4e_top_slices_kernel(
        const float *logits, uint32_t n_vocab, uint32_t k, uint32_t slices,
        float *values, uint32_t *ids) {
    __shared__ uint32_t s_key[8];
    __shared__ uint32_t s_id[8];
    __shared__ uint32_t t_key;
    __shared__ uint32_t t_id;
    const uint32_t row = blockIdx.y;
    const uint32_t slice = blockIdx.x;
    const uint32_t span = (n_vocab + slices - 1u) / slices;
    const uint32_t lo = slice * span;
    const uint32_t hi = min(lo + span, n_vocab);
    const float *r = logits + (uint64_t)row * n_vocab;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    uint32_t th_key = 0u, th_id = 0u;
    for (uint32_t t = 0; t < k; t++) {
        uint32_t bk = 0u, bi = 0xffffffffu;
        for (uint32_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            const uint32_t key = q4e_order_key(r[i]);
            const bool below = t == 0u || key < th_key || (key == th_key && i > th_id);
            if (below && q4e_top_better(key, i, bk, bi)) { bk = key; bi = i; }
        }
        for (uint32_t off = 16u; off > 0u; off >>= 1) {
            const uint32_t ok = __shfl_xor_sync(0xffffffffu, bk, off);
            const uint32_t oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (q4e_top_better(ok, oi, bk, bi)) { bk = ok; bi = oi; }
        }
        if (lane == 0u) { s_key[warp] = bk; s_id[warp] = bi; }
        __syncthreads();
        if (threadIdx.x == 0u) {
            uint32_t ck = s_key[0], ci = s_id[0];
            for (uint32_t w = 1u; w < (blockDim.x >> 5); w++) {
                if (q4e_top_better(s_key[w], s_id[w], ck, ci)) { ck = s_key[w]; ci = s_id[w]; }
            }
            const uint64_t at = ((uint64_t)row * slices + slice) * k + t;
            values[at] = ci < n_vocab ? r[ci] : -INFINITY;
            ids[at] = ci;
            t_key = ck;
            t_id = ci;
        }
        __syncthreads();
        th_key = t_key;
        th_id = t_id;
        // Read before the next round's thread 0 may write it again.
        __syncthreads();
    }
}
