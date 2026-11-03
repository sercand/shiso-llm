// The spatial merge: a square block of neighbouring patches becomes one row.
//
// This is the step that turns patch rows into the tokens the language model
// sees, and its only content is an address mapping -- but the mapping is the
// whole contract between three things that must agree: the order the
// preprocessor emitted patches in, the grid positions the rotation used, and
// which four vectors end up concatenated into one token. Get it wrong and every
// shape downstream is still correct, so nothing fails; the picture the model
// describes is just rearranged.
//
// Patch rows arrive grid-major -- row by row across the frame, frames back to
// back -- so the four patches of a merge block are NOT adjacent in the input.
// They are gathered here rather than being reshaped for free, which is the
// price of keeping the preprocessor's order the natural one.

/// `out[o, p * hidden + d] = in[src(o, p), d]`, where `p` walks the merge block
/// row-major and `src` is the patch that block position names in the grid.
///
/// One threadgroup per output row. The merged width is `merge^2 * hidden`, and
/// the output rows of a clip are grouped by frame in the same order the input
/// rows are, so a later frame's tokens follow an earlier frame's.
kernel void kernel_vit_spatial_merge(
        constant vit_args_merge & args [[buffer(0)]],
        device float *out              [[buffer(1)]],
        device const float *in         [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint o = tgpig.x;
    if (o >= args.merged_rows) {
        return;
    }
    const uint nthreads = ntg.x;
    const uint m = args.merge;
    const uint width = m * m * args.hidden;
    const uint out_gh = args.grid_h / m;
    const uint out_gw = args.grid_w / m;
    const uint per_group = out_gh * out_gw;
    const uint plane = args.grid_h * args.grid_w;

    const uint group = o / per_group;
    const uint local = o % per_group;
    const uint oh = local / out_gw;
    const uint ow = local % out_gw;

    for (uint i = tpitg.x; i < width; i += nthreads) {
        const uint slot = i / args.hidden;
        const uint d = i % args.hidden;
        const uint ph = oh * m + slot / m;
        const uint pw = ow * m + slot % m;
        const uint64_t src = ((uint64_t) group * plane + (uint64_t) ph * args.grid_w + pw)
                * args.hidden + d;
        out[(uint64_t) o * width + i] = in[src];
    }
}
