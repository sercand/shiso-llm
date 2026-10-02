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
