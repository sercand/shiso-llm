/* The tower's attention pair and the merge that ends it.
 *
 * The CUDA twins of `metal/vision/vit_attn.metal` and
 * `metal/vision/vit_merge.metal`. Nothing here is copied; the arithmetic is
 * those two files'.
 *
 * Two things here are easy to get subtly wrong and impossible to see
 * afterwards, so both are spelled out rather than inferred at the call site.
 *
 * The rotation is two-dimensional. The head's channels split in half and pair
 * (i, i + head_dim/2), which is the same pairing the language model's blocks
 * use; what differs is the angle. The first quarter of the channels rotate by
 * the patch's ROW in the grid and the second quarter by its COLUMN, both over
 * the same frequency ladder built for a width of head_dim/2. A checkpoint that
 * describes this as a "partial" rotation is describing how those frequencies are
 * built, not a subset of channels being left alone: every channel rotates.
 *
 * The attention is bidirectional and bounded by the frame. There is no causal
 * mask -- a patch sees every other patch -- but a clip's frames are separate
 * spans, so a query never attends across a temporal group boundary.
 *
 * A patch's grid position is its offset inside its own temporal group, derived
 * grid-major: `local = n % plane`, row `local / grid_w`, column
 * `local % grid_w`. That is NOT the merge-block order `qwen3vl_patch_rc` in
 * `vision.cuh` derives, and the difference is the one that fails silently --
 * every buffer the right size, every launch succeeding, and the picture
 * described rearranged. The two sets are different decompositions; see
 * `KERNEL_NAMES.md`.
 */
#ifndef SHISO_VIT_TOWER_CUH
#define SHISO_VIT_TOWER_CUH

#include "vit_common.cuh"

/* Splits the fused projection into three planes and rotates the first two.
 *
 * The bias is already in `qkv`: it came out of the GEMM's epilogue, so this
 * kernel only reshapes and rotates. Values are read from the fused row rather
 * than written out and read back, which is what lets the rotation run without a
 * barrier between the value copy and the rotation.
 *
 * `powf`, `sinf` and `cosf` are the accurate library entries, as the Metal twin
 * uses the accurate builtins. That is enforced by this directory's manifest
 * dropping `--use_fast_math` rather than by a spelling here, because under that
 * flag nvcc rewrites all three calls and no source spelling opts back out. */
extern "C" __global__ void kernel_vit_qkv_rope(
        const struct vit_args_qkv_rope args,
        float *__restrict__ q,
        float *__restrict__ k,
        float *__restrict__ v,
        const float *__restrict__ qkv) {
    const unsigned n = blockIdx.y;
    const unsigned h = blockIdx.x;
    if (n >= args.rows) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;
    const unsigned hd = args.head_dim;
    const unsigned dim = args.heads * hd;
    const unsigned half_hd = hd / 2u;
    /* Channels per spatial axis: the row axis owns the first quarter of the
     * head's channels and the column axis the second, and each is paired with
     * its partner half a head away. */
    const unsigned per_axis = half_hd / 2u;

    const unsigned local = n % args.plane;
    const float row = (float) (local / args.grid_w);
    const float col = (float) (local % args.grid_w);

    const float *src = qkv + (uint64_t) n * 3u * dim;
    const unsigned off = h * hd;
    float *qd = q + (uint64_t) n * dim + off;
    float *kd = k + (uint64_t) n * dim + off;
    float *vd = v + (uint64_t) n * dim + off;

    for (unsigned i = tid; i < hd; i += nthreads) {
        vd[i] = src[2u * dim + off + i];
    }
    for (unsigned i = tid; i < half_hd; i += nthreads) {
        const unsigned freq = (i < per_axis) ? i : (i - per_axis);
        const float pos = (i < per_axis) ? row : col;
        const float inv = powf(args.rope_base, -2.0f * (float) freq / (float) half_hd);
        const float theta = pos * inv;
        const float s = sinf(theta);
        const float c = cosf(theta);
        const float q0 = src[off + i];
        const float q1 = src[off + i + half_hd];
        qd[i] = q0 * c - q1 * s;
        qd[i + half_hd] = q1 * c + q0 * s;
        const float k0 = src[dim + off + i];
        const float k1 = src[dim + off + i + half_hd];
        kd[i] = k0 * c - k1 * s;
        kd[i + half_hd] = k1 * c + k0 * s;
    }
}

/* Bidirectional attention over one temporal group, one block per
 * (patch row, head).
 *
 * Warps own whole key positions and stride over the group, each keeping its own
 * running softmax maximum, denominator and value accumulator; the combine at the
 * end rescales every warp's partial onto the largest maximum. That shape is what
 * keeps the exponentials bounded without a second pass over the keys, and it is
 * why a lane's accumulator has to be a register array rather than shared memory:
 * it is touched once per key.
 *
 * The dynamic shared allocation holds, in order, the query row (head_dim
 * floats), one accumulator per warp (warps * head_dim), then the per-warp
 * maximum and denominator (one each). The host sizes it from the same formula.
 *
 * `expf` is the accurate entry rather than `__expf`: the Metal twin calls the
 * accurate builtin, and this directory's manifest drops `--use_fast_math` so
 * that stays true. */
extern "C" __global__ void kernel_vit_attention(
        const struct vit_args_attention args,
        float *__restrict__ out,
        const float *__restrict__ q,
        const float *__restrict__ k,
        const float *__restrict__ v) {
    const unsigned n = blockIdx.y;
    const unsigned h = blockIdx.x;
    const unsigned hd = args.head_dim;
    /* A head wider than the per-lane accumulator can hold would be silently
     * truncated, which is the one failure this tower cannot detect downstream. */
    if (n >= args.rows || hd > VIT_ATTN_SLOTS * 32u) {
        return;
    }
    const unsigned dim = args.heads * hd;
    const unsigned off = h * hd;
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const unsigned warps = nthreads >> 5u;
    const float scale = args.scale;

    float *s_q = (float *) vit_smem;
    float *s_acc = s_q + hd;
    float *s_max = s_acc + (uint64_t) warps * hd;
    float *s_den = s_max + warps;

    const float *qd = q + (uint64_t) n * dim + off;
    for (unsigned i = tid; i < hd; i += nthreads) {
        s_q[i] = qd[i];
    }
    __syncthreads();

    const unsigned base = (n / args.plane) * args.plane;
    const unsigned end = base + args.plane;

    float run_max = -INFINITY;
    float run_den = 0.0f;
    float acc[VIT_ATTN_SLOTS];
    for (unsigned s = 0u; s < VIT_ATTN_SLOTS; s++) {
        acc[s] = 0.0f;
    }

    for (unsigned p = base + warp; p < end; p += warps) {
        const float *kd = k + (uint64_t) p * dim + off;
        float dot = 0.0f;
        for (unsigned d = lane; d < hd; d += 32u) {
            dot += s_q[d] * kd[d];
        }
        dot = vit_warp_sum_all(dot);
        const float score = dot * scale;
        const float new_max = fmaxf(run_max, score);
        const float rescale = expf(run_max - new_max);
        const float weight = expf(score - new_max);
        run_den = run_den * rescale + weight;
        run_max = new_max;
        const float *vd = v + (uint64_t) p * dim + off;
        unsigned slot = 0u;
        for (unsigned d = lane; d < hd; d += 32u, slot++) {
            acc[slot] = acc[slot] * rescale + weight * vd[d];
        }
    }

    {
        unsigned slot = 0u;
        for (unsigned d = lane; d < hd; d += 32u, slot++) {
            s_acc[(uint64_t) warp * hd + d] = acc[slot];
        }
        if (lane == 0u) {
            s_max[warp] = run_max;
            s_den[warp] = run_den;
        }
    }
    __syncthreads();

    for (unsigned d = tid; d < hd; d += nthreads) {
        float m = -INFINITY;
        for (unsigned w = 0u; w < warps; w++) {
            m = fmaxf(m, s_max[w]);
        }
        float den = 0.0f;
        float num = 0.0f;
        for (unsigned w = 0u; w < warps; w++) {
            const float r = expf(s_max[w] - m);
            den += s_den[w] * r;
            num += s_acc[(uint64_t) w * hd + d] * r;
        }
        out[(uint64_t) n * dim + off + d] = (den > 0.0f) ? (num / den) : 0.0f;
    }
}

/* `out[o, p * hidden + d] = in[src(o, p), d]`, where `p` walks the merge block
 * row-major and `src` is the patch that block position names in the grid.
 *
 * This is the step that turns patch rows into the tokens the language model
 * sees, and its only content is an address mapping -- but the mapping is the
 * whole contract between three things that must agree: the order the
 * preprocessor emitted patches in, the grid positions the rotation used, and
 * which vectors end up concatenated into one token. Get it wrong and every shape
 * downstream is still correct, so nothing fails; the picture the model describes
 * is just rearranged.
 *
 * One block per output row. The merged width is `merge^2 * hidden`, and the
 * output rows of a clip are grouped by frame in the same order the input rows
 * are, so a later frame's tokens follow an earlier frame's. The merge factor is
 * an argument: this checkpoint family uses two and a pooling projector uses
 * three. */
extern "C" __global__ void kernel_vit_spatial_merge(
        const struct vit_args_merge args,
        float *__restrict__ out,
        const float *__restrict__ in) {
    const unsigned o = blockIdx.x;
    if (o >= args.merged_rows) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;
    const unsigned m = args.merge;
    const unsigned width = m * m * args.hidden;
    const unsigned out_gh = args.grid_h / m;
    const unsigned out_gw = args.grid_w / m;
    const unsigned per_group = out_gh * out_gw;
    const unsigned plane = args.grid_h * args.grid_w;

    const unsigned group = o / per_group;
    const unsigned local = o % per_group;
    const unsigned oh = local / out_gw;
    const unsigned ow = local % out_gw;

    for (unsigned i = tid; i < width; i += nthreads) {
        const unsigned slot = i / args.hidden;
        const unsigned d = i % args.hidden;
        const unsigned ph = oh * m + slot / m;
        const unsigned pw = ow * m + slot % m;
        const uint64_t src = ((uint64_t) group * plane + (uint64_t) ph * args.grid_w + pw)
                * args.hidden + d;
        out[(uint64_t) o * width + i] = in[src];
    }
}

#endif /* SHISO_VIT_TOWER_CUH */
