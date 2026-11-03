// The qwen35moe expert stages.
//
// `q35moe_common.h` is concatenated ahead of this file by `SOURCES` in
// `src/compile.rs`, so its argument block and its affine tile are already in
// scope and there is nothing to include: the library is one source string and
// a quote-include would not resolve.
//
// No `#pragma METAL fp math_mode` here. Fast math is the library's build
// default and this file holds at it.

// Grades `q35moe_affine_tile` against the host product. One threadgroup,
// `Rows x TileN` outputs, the whole of `in_dim`, written row-major over
// `args.out_dim`.
//
// This entry exists only to grade the tile: it is the smallest launch that
// exercises the group-major slab layout, the per-group scale/bias epilogue and
// the input row sums at once.
//
// Buffers: args, input, weights, scales, biases, out.
kernel void kernel_q35moe_tile_probe(
        constant q35moe_expert_args &args [[buffer(0)]],
        device const bfloat *input [[buffer(1)]],
        device const uchar *weights [[buffer(2)]],
        device const bfloat *scales [[buffer(3)]],
        device const bfloat *biases [[buffer(4)]],
        device float *out [[buffer(5)]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]) {
    constexpr uint Rows = 8;
    constexpr uint TileN = 32;
    threadgroup float row_sums[Rows];
    q35moe_affine_tile<Rows, TileN, 8>(input, weights, scales, biases,
                                       uint(args.in_dim), out,
                                       uint(args.out_dim), 0, row_sums,
                                       simd_lane, simd_group);
}

// ---------------------------------------------------------------------------
// The two routed expert stages over the q4-affine codec, in two activation
// dtypes.
//
// Ported from `kernels/cuda/qwen35moe/experts.cuh`, whose NVFP4 pair is the
// same block structure over a different codec: `group.y` indexes the ACTIVE
// experts through the list the scan pass left behind, `group.x` picks four
// output rows, each simdgroup owns one of them, and the block sweeps that
// expert's whole span of the sorted column map so the weight row stays in cache
// across the columns routed to it. One path, no per-column variant: a threshold
// at which a model starts answering differently is the defect the grouping
// exists to avoid.
//
// `Act` is the float the ACTIVATIONS cross the launch boundary in -- the input
// rows and the outputs both. The weights are `uchar` codes and the scale and
// bias are `bfloat` whatever `Act` is, because those are container widths and
// not an activation choice.
//
// Four conventions live here and none of them is checkable by reading one
// kernel; `crates/shiso-kerneltest/src/qwen35moe/q4affine.rs` grades all four:
//
//   1. a routed column is `token * n_used + slot`, so `col / n_used` is the
//      token whose activation row gate and up read;
//   2. the down projection reads its OWN column's row of the activated
//      intermediate, not its token's -- that row is what this (token, slot)
//      pair's own expert produced;
//   3. down writes `[column, out_dim]` row-major, which is column-major over
//      (token, slot), because that is what `q4e_moe_combine_kernel` reads at
//      `down[col * n_embd + e]`. The other way round is a block of the right
//      size holding a transpose;
//   4. the per-expert parameters are indexed by expert id. The affine codec has
//      no per-tensor scalar to get wrong the way NVFP4's `globals[e]` is, but
//      the slab strides are the same mistake in a different place.
//
// No `#pragma METAL fp math_mode` below either. Fast math is the library's
// build default and these bodies are written against it; the measured
// divergences are in `KERNELS.md`.

/// Gate and up for one tile of one active expert's columns.
///
/// One launch rather than two because the two projections share everything but
/// the matrix -- the same activation row, the same column list, the same
/// expert -- and the activation row is the operand that is not in cache.
///
/// Buffers, matching `moe.rs::expert_stage`: args, gate codes/scales/biases,
/// up codes/scales/biases, input, sorted, bounds, active, out_gate, out_up.
template <typename Act>
inline void q35moe_gate_up(constant q35moe_expert_args &args,
                           device const uchar *gate_w,
                           device const bfloat *gate_s,
                           device const bfloat *gate_b,
                           device const uchar *up_w,
                           device const bfloat *up_s,
                           device const bfloat *up_b,
                           device const Act *input,
                           device const int *sorted,
                           device const int *bounds,
                           device const int *active,
                           device Act *out_gate,
                           device Act *out_up,
                           uint3 group,
                           uint simd_lane,
                           uint simd_group) {
    // `grid.y` is `min(columns, n_expert)` and the active experts are however
    // many of those received a column, so the tail of the extent returns here.
    if (int(group.y) >= active[args.n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const uint in_dim = uint(args.in_dim);
    const uint out_dim = uint(args.out_dim);
    const uint row = group.x * Q35MOE_EXPERT_ROWS + simd_group;
    if (row >= out_dim) {
        return;
    }

    const uint groups = in_dim / Q35MOE_GROUP;
    const ulong code_base = ulong(uint(expert)) * out_dim * in_dim / 2u;
    const ulong parameter_base = ulong(uint(expert)) * out_dim * groups;
    const uint n_used = uint(args.n_used);

    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        // Convention 1: every slot of a token reads the same activation row.
        device const Act *xrow = input + ulong(col / n_used) * in_dim;
        const float g = q35moe_affine_row_dot(gate_w, gate_s, gate_b, code_base,
                                              parameter_base, row, in_dim, xrow,
                                              simd_lane);
        const float u = q35moe_affine_row_dot(up_w, up_s, up_b, code_base,
                                              parameter_base, row, in_dim, xrow,
                                              simd_lane);
        if (simd_lane == 0u) {
            out_gate[ulong(col) * out_dim + row] = Act(g);
            out_up[ulong(col) * out_dim + row] = Act(u);
        }
    }
}

/// The routed down projection for one tile of one active expert's columns.
///
/// Buffers, matching `moe.rs::expert_stage`: args, down codes/scales/biases,
/// mid, sorted, bounds, active, out.
template <typename Act>
inline void q35moe_down(constant q35moe_expert_args &args,
                        device const uchar *down_w,
                        device const bfloat *down_s,
                        device const bfloat *down_b,
                        device const Act *mid,
                        device const int *sorted,
                        device const int *bounds,
                        device const int *active,
                        device Act *out,
                        uint3 group,
                        uint simd_lane,
                        uint simd_group) {
    if (int(group.y) >= active[args.n_expert]) {
        return;
    }
    const int expert = active[group.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    if (c1 <= c0) {
        return;
    }
    const uint in_dim = uint(args.in_dim);
    const uint out_dim = uint(args.out_dim);
    const uint row = group.x * Q35MOE_EXPERT_ROWS + simd_group;
    if (row >= out_dim) {
        return;
    }

    const uint groups = in_dim / Q35MOE_GROUP;
    const ulong code_base = ulong(uint(expert)) * out_dim * in_dim / 2u;
    const ulong parameter_base = ulong(uint(expert)) * out_dim * groups;

    for (int c = c0; c < c1; ++c) {
        const uint col = uint(sorted[c]);
        // Convention 2: this column's own row, not `col / n_used`'s. `n_used`
        // is deliberately unread in this body.
        device const Act *xrow = mid + ulong(col) * in_dim;
        const float v = q35moe_affine_row_dot(down_w, down_s, down_b, code_base,
                                              parameter_base, row, in_dim, xrow,
                                              simd_lane);
        if (simd_lane == 0u) {
            // Convention 3: `[column, out_dim]` row-major.
            out[ulong(col) * out_dim + row] = Act(v);
        }
    }
}

/// Gate and up over q4-affine experts, `f32` activations.
kernel void kernel_q35moe_experts_gate_up_q4affine_f32(
        constant q35moe_expert_args &args [[buffer(0)]],
        device const uchar *gate_w [[buffer(1)]],
        device const bfloat *gate_s [[buffer(2)]],
        device const bfloat *gate_b [[buffer(3)]],
        device const uchar *up_w [[buffer(4)]],
        device const bfloat *up_s [[buffer(5)]],
        device const bfloat *up_b [[buffer(6)]],
        device const float *input [[buffer(7)]],
        device const int *sorted [[buffer(8)]],
        device const int *bounds [[buffer(9)]],
        device const int *active [[buffer(10)]],
        device float *out_gate [[buffer(11)]],
        device float *out_up [[buffer(12)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]) {
    q35moe_gate_up<float>(args, gate_w, gate_s, gate_b, up_w, up_s, up_b, input,
                          sorted, bounds, active, out_gate, out_up, group,
                          simd_lane, simd_group);
}

/// The routed down projection over q4-affine experts, `f32` activations.
kernel void kernel_q35moe_experts_down_q4affine_f32(
        constant q35moe_expert_args &args [[buffer(0)]],
        device const uchar *down_w [[buffer(1)]],
        device const bfloat *down_s [[buffer(2)]],
        device const bfloat *down_b [[buffer(3)]],
        device const float *mid [[buffer(4)]],
        device const int *sorted [[buffer(5)]],
        device const int *bounds [[buffer(6)]],
        device const int *active [[buffer(7)]],
        device float *out [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]) {
    q35moe_down<float>(args, down_w, down_s, down_b, mid, sorted, bounds, active,
                       out, group, simd_lane, simd_group);
}

/// Gate and up over q4-affine experts, `bfloat` activations.
kernel void kernel_q35moe_experts_gate_up_q4affine_bf16(
        constant q35moe_expert_args &args [[buffer(0)]],
        device const uchar *gate_w [[buffer(1)]],
        device const bfloat *gate_s [[buffer(2)]],
        device const bfloat *gate_b [[buffer(3)]],
        device const uchar *up_w [[buffer(4)]],
        device const bfloat *up_s [[buffer(5)]],
        device const bfloat *up_b [[buffer(6)]],
        device const bfloat *input [[buffer(7)]],
        device const int *sorted [[buffer(8)]],
        device const int *bounds [[buffer(9)]],
        device const int *active [[buffer(10)]],
        device bfloat *out_gate [[buffer(11)]],
        device bfloat *out_up [[buffer(12)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]) {
    q35moe_gate_up<bfloat>(args, gate_w, gate_s, gate_b, up_w, up_s, up_b, input,
                           sorted, bounds, active, out_gate, out_up, group,
                           simd_lane, simd_group);
}

/// The routed down projection over q4-affine experts, `bfloat` activations.
kernel void kernel_q35moe_experts_down_q4affine_bf16(
        constant q35moe_expert_args &args [[buffer(0)]],
        device const uchar *down_w [[buffer(1)]],
        device const bfloat *down_s [[buffer(2)]],
        device const bfloat *down_b [[buffer(3)]],
        device const bfloat *mid [[buffer(4)]],
        device const int *sorted [[buffer(5)]],
        device const int *bounds [[buffer(6)]],
        device const int *active [[buffer(7)]],
        device bfloat *out [[buffer(8)]],
        uint3 group [[threadgroup_position_in_grid]],
        uint simd_lane [[thread_index_in_simdgroup]],
        uint simd_group [[simdgroup_index_in_threadgroup]]) {
    q35moe_down<bfloat>(args, down_w, down_s, down_b, mid, sorted, bounds,
                        active, out, group, simd_lane, simd_group);
}
