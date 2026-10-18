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
