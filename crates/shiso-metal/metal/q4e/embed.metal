// q4e embed entries: Metal port of kernels/cuda/q4e/embed.cu (see q4e_common.h for conventions).
//
// The half-precision row gather with the residual-stream fan-out. The two
// other storage formats' gathers in embed.cu (the block-quantized and the
// brain-float one) are outside this port and not here. The
// sparse stack's vocabulary table is stored as f16 (`DType::F16` in
// `crates/shiso-engine/src/sparse/gather.rs`), so the half entry is the one it
// launches, with `n_hc = 1`.

/// embed.cu:17 `embed_token_hc_kernel`: one token's f16 row, widened and
/// tiled across `n_hc` streams. Buffers: 0 `out` (f32, n_hc*n_embd), 1 `w`
/// (f16 bits, n_vocab*n_embd), 2 `token`, 3 `n_embd`, 4 `n_hc`. No range check
/// on `token`, as in CUDA: the caller owns it. 32-bit flat index, as in CUDA.
kernel void embed_token_hc_kernel(
        device float *out [[buffer(0)]],
        device const half *w [[buffer(1)]],
        constant uint &token [[buffer(2)]],
        constant uint &n_embd [[buffer(3)]],
        constant uint &n_hc [[buffer(4)]],
        uint bid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    uint i = bid * bdim + tid;
    uint n = n_embd * n_hc;
    if (i >= n) return;
    uint e = i % n_embd;
    out[i] = (float)w[(ulong)token * n_embd + e];
}

/// embed.cu:25 `embed_tokens_hc_kernel`: every token's f16 row, widened and
/// tiled across `n_hc` streams; an id below zero or at/above `n_vocab` reads
/// row 0. Buffers: 0 `out` (f32, n_tokens*n_hc*n_embd), 1 `tokens` (i32), 2
/// `w` (f16), 3 `n_vocab`, 4 `n_tokens`, 5 `n_embd`, 6 `n_hc`.
kernel void embed_tokens_hc_kernel(
        device float *out [[buffer(0)]],
        device const int *tokens [[buffer(1)]],
        device const half *w [[buffer(2)]],
        constant uint &n_vocab [[buffer(3)]],
        constant uint &n_tokens [[buffer(4)]],
        constant uint &n_embd [[buffer(5)]],
        constant uint &n_hc [[buffer(6)]],
        uint bid [[threadgroup_position_in_grid]],
        uint tid [[thread_position_in_threadgroup]],
        uint bdim [[threads_per_threadgroup]]) {
    ulong gid = (ulong)bid * bdim + tid;
    ulong n = (ulong)n_tokens * n_hc * n_embd;
    if (gid >= n) return;
    uint d = gid % n_embd;
    ulong tmp = gid / n_embd;
    uint t = tmp / n_hc;
    int tok_i = tokens[t];
    uint tok = tok_i < 0 ? 0u : (uint)tok_i;
    if (tok >= n_vocab) tok = 0;
    out[gid] = (float)w[(ulong)tok * n_embd + d];
}
