// The two kernels a block drafter adds to the dense set.
//
// A block drafter proposes several tokens in one pass from the target's own
// hidden states at a few of its layers. Everything else it runs - norms,
// projections, rotation, attention, the gated feed-forward - is the dense set's
// already; these two are what it has that nothing else does.

#include <metal_stdlib>
using namespace metal;

/// Copies `n_rows` rows of `width` floats into a strided destination: row `r`
/// lands at `dst[r * dst_stride + dst_offset ..]`.
///
/// How a target pass hands the drafter its hidden states: each tapped layer's
/// residual rows are written into their slot of a `[row][tap][hidden]` block,
/// which is the concatenation the drafter's input projection reads.
kernel void kernel_dense_copy_rows_f32(
        constant dense_args_copy_rows & args,
        device const float * src,
        device float * dst,
        uint i [[thread_position_in_grid]]) {
    if (i >= args.n_rows * args.width) return;
    const uint r = i / args.width;
    const uint c = i % args.width;
    dst[(uint64_t) r * args.dst_stride + args.dst_offset + c] = src[(uint64_t) r * args.width + c];
}

/// Grouped dynamic causal convolution along a block's rows.
///
/// `out[t][c] = sum over o < taps with t - o >= 0 of
///              (base[half][o][c] + dyn[t][half][o][c / group]) * x[t - o][c]`
///
/// `dyn` is the per-row kernel a projection of the block's normed input
/// predicted, laid out `[row][2][taps][channels / group]`; `part` picks the
/// first half (applied before the block's attention or feed-forward) or the
/// second (applied to what came out of it). The static and the dynamic term of
/// each offset are added in that order, offset by offset, as the reference
/// folds them.
kernel void kernel_dense_dyn_conv_f32(
        constant dense_args_dyn_conv & args,
        device const float * x,
        device const float * dyn,
        device const float * base,
        device float * out,
        uint i [[thread_position_in_grid]]) {
    const uint hidden = args.hidden;
    if (i >= args.n_rows * hidden) return;
    const uint t = i / hidden;
    const uint c = i % hidden;
    const uint groups = hidden / args.group;
    const uint g = c / args.group;
    float acc = 0.0f;
    for (uint o = 0; o < args.taps; o++) {
        if (o > t) break;
        const float v = x[(uint64_t) (t - o) * hidden + c];
        const float b = base[((uint64_t) args.part * args.taps + o) * hidden + c];
        const float d = dyn[(((uint64_t) t * 2u + args.part) * args.taps + o) * groups + g];
        acc = acc + b * v;
        acc = acc + d * v;
    }
    out[i] = acc;
}
