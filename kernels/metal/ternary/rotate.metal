// Signed blockwise Walsh-Hadamard rotation of f32 rows.
//
// Forward: y = H(P(x) * signs), the transform a projection's input passes
// through before it meets a weight the rotation was folded into. Inverse:
// y = signs * H(x), which takes a row gathered from a rotated embedding table
// back to the basis the residual stream lives in. H is the Sylvester-ordered
// transform over each consecutive `block` elements, scaled by 1/sqrt(block) so
// it is its own inverse; P is the optional head permutation of
// `ternary_args_rotate`.
//
// One threadgroup per (block, token). The block is staged whole in threadgroup
// memory before any write, so `src == dst` is safe when nothing is permuted:
// each threadgroup reads and writes only its own block. The host refuses an
// in-place call that permutes.

kernel void kernel_rotate_wht_f32(
        constant ternary_args_rotate & args [[buffer(0)]],
        device const float * src            [[buffer(1)]],
        device const float * signs          [[buffer(2)]],
        device       float * dst            [[buffer(3)]],
        threadgroup  float * stage          [[threadgroup(0)]],
        uint2 tg  [[threadgroup_position_in_grid]],
        uint2 tid2 [[thread_position_in_threadgroup]],
        uint2 ntg  [[threads_per_threadgroup]]) {
    const uint tid = tid2.x;
    const uint nth = ntg.x;
    const uint block = args.block;
    const uint base  = tg.x * block;
    const ulong row  = (ulong) tg.y * args.width;

    for (uint i = tid; i < block; i += nth) {
        const uint e = base + i;
        uint from = e;
        if (args.rep > 1u) {
            const uint head = e / args.hd;
            const uint k    = head / args.rep;
            const uint m    = head % args.rep;
            from = (m * args.nk + k) * args.hd + e % args.hd;
        }
        float x = src[row + from];
        if (args.inverse == 0u) {
            x *= signs[e];
        }
        stage[i] = x;
    }

    for (uint h = 1u; h < block; h <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint p = tid; p < block / 2u; p += nth) {
            const uint i = (p / h) * (2u * h) + p % h;
            const float a = stage[i];
            const float b = stage[i + h];
            stage[i]     = a + b;
            stage[i + h] = a - b;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float scale = rsqrt((float) block);
    for (uint i = tid; i < block; i += nth) {
        const uint e = base + i;
        float y = stage[i] * scale;
        if (args.inverse != 0u) {
            y *= signs[e];
        }
        dst[row + e] = y;
    }
}
