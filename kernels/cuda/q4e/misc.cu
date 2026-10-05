/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels).
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
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
extern "C" __global__ void q4e_argmax_rows_kernel(int32_t *out, const float *logits, uint32_t n_vocab) {
    const float *row = logits + (uint64_t)blockIdx.x * n_vocab;
    float best = -INFINITY;
    int32_t best_i = -1;
    for (uint32_t i = threadIdx.x; i < n_vocab; i += blockDim.x) {
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
