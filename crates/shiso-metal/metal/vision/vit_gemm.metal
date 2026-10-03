// The one matrix multiply the vision tower needs.
//
// Every projection in the tower is the same shape of operation: f32 activation
// rows times a bf16 weight held transposed, accumulating in f32, with a bf16
// bias broadcast over the output columns. The patch projection, the fused qkv,
// the attention output, both feed-forward matrices and both merger layers all
// go through this entry point; there is no second GEMM to keep in step with it.
//
// Weights stay in the width the checkpoint stores them and are widened on read.
// The activations are f32 because that is what the residual stream this tower
// feeds is, end to end: nothing here narrows a value that later has to be
// added to an f32 hidden state.

/// `out[m, n] = sum_k in[m, k] * weight[n, k] + bias[n]`.
///
/// The weight is indexed as `[n][k]` -- output width slowest -- which is how
/// the checkpoint writes a linear layer, so no transpose pass runs anywhere in
/// this tower.
///
/// # Why a staged tile and not a thread per output
///
/// A thread that owned one output column would read `weight[n * k + ...]` with
/// neighbouring threads a whole row apart, which is the worst access this
/// device has. Staging a VIT_GEMM_TILE square of each operand makes both loads
/// contiguous along `k` and reuses each one across the tile.
///
/// # The summation order is the contract
///
/// `k` is walked strictly in increasing order: the tile loop advances by
/// VIT_GEMM_TILE and the inner loop walks the tile front to back, so the sum is
/// the same association a plain `for k in 0..K` loop makes. That is deliberate
/// -- the oracle this kernel is held to is exactly that loop -- and a blocking
/// change that reorders `k` is a change to the numbers, not just to the speed.
template <typename W>
static inline void vit_gemm_bias_body(
        constant vit_args_gemm & args,
        device float *out,
        device const float *in,
        device const ushort *weight,
        device const ushort *bias,
        threadgroup float (*a_tile)[VIT_GEMM_TILE],
        threadgroup float (*b_tile)[VIT_GEMM_TILE],
        uint3 tgpig,
        ushort3 tpitg) {
    const uint tx = tpitg.x;
    const uint ty = tpitg.y;
    const uint row = tgpig.y * VIT_GEMM_TILE + ty;
    const uint col = tgpig.x * VIT_GEMM_TILE + tx;
    // The weight row this thread stages, which is not the one it accumulates:
    // the tile is filled row-per-thread-y so the `k` index runs along x and the
    // load stays contiguous, then read back column-wise in the inner loop.
    const uint stage_n = tgpig.x * VIT_GEMM_TILE + ty;

    float acc = 0.0f;
    for (uint k0 = 0u; k0 < args.k; k0 += VIT_GEMM_TILE) {
        const uint k = k0 + tx;
        const bool have_k = k < args.k;
        a_tile[ty][tx] = (row < args.m && have_k)
                ? in[(uint64_t) row * args.k + k]
                : 0.0f;
        b_tile[ty][tx] = (stage_n < args.n && have_k)
                ? W::load(weight + (uint64_t) stage_n * args.k + k)
                : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0u; kk < VIT_GEMM_TILE; kk++) {
            acc += a_tile[ty][kk] * b_tile[tx][kk];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < args.m && col < args.n) {
        out[(uint64_t) row * args.n + col] = acc + W::load(bias + col);
    }
}

// The brain-float tower's entry and the half-precision tower's twin. The
// tiles are declared here because threadgroup memory belongs to a kernel.
kernel void kernel_vit_gemm_bias(
        constant vit_args_gemm & args [[buffer(0)]],
        device float *out             [[buffer(1)]],
        device const float *in        [[buffer(2)]],
        device const ushort *weight   [[buffer(3)]],
        device const ushort *bias     [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    threadgroup float b_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    vit_gemm_bias_body<vit_w_bf16>(args, out, in, weight, bias, a_tile, b_tile, tgpig, tpitg);
}

kernel void kernel_vit_gemm_bias_f16(
        constant vit_args_gemm & args [[buffer(0)]],
        device float *out             [[buffer(1)]],
        device const float *in        [[buffer(2)]],
        device const ushort *weight   [[buffer(3)]],
        device const ushort *bias     [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    threadgroup float a_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    threadgroup float b_tile[VIT_GEMM_TILE][VIT_GEMM_TILE];
    vit_gemm_bias_body<vit_w_f16>(args, out, in, weight, bias, a_tile, b_tile, tgpig, tpitg);
}
