// sample entries: Metal twin of kernels/cuda/sample/sample.cu.

/// sample.cu `sample_suppress_rows_kernel`: writes -inf at every listed id of
/// every logits row, one threadgroup per row. Buffers: 0 `logits` (f32,
/// rows*n_vocab, in/out), 1 `list` (u32: the count, then that many ids), 2
/// `n_vocab`. The count is read from the list rather than bound as a scalar so
/// a list written after a plan was built is the one every later launch reads.
/// An id past the vocabulary is skipped.
kernel void sample_suppress_rows_kernel(
        device float *logits [[buffer(0)]],
        device const uint *list [[buffer(1)]],
        constant uint &n_vocab [[buffer(2)]],
        uint row_i [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    const uint n = list[0];
    device float *row = logits + (ulong)row_i * n_vocab;
    for (uint i = tid; i < n; i += bdim) {
        const uint id = list[1u + i];
        if (id < n_vocab) row[id] = -INFINITY;
    }
}
