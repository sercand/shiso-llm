/* The tower's tensor-core path: attention on matrix fragments, and the two
 * elementwise kernels that let its projections run through the device's BLAS.
 *
 * The parametric set beside this one is written for any backend and runs every
 * multiply on the scalar units, one output per thread. For a 896-pixel picture
 * that is 3136 patch rows through two dozen blocks, and it cost 5.3 s on the
 * GB10 - 61% of it in the attention, which streamed every key and value out of
 * global memory once per query. These entries are CUDA-only. The host looks
 * for them in the module's entry list and uses the scalar set when they are
 * absent, so the Metal twin needs nothing new.
 *
 * # Brain-float operands, single-precision everything else
 *
 * The fragments multiply brain-float operands and accumulate in single
 * precision, which is how the checkpoint's reference runs this tower. The
 * weights are stored brain-float, so a projection loses nothing on its weight
 * side; its activations are rounded once on the way in. The softmax statistics,
 * the accumulators, the residual stream and every output stay single
 * precision.
 */
#ifndef SHISO_VIT_FAST_CUH
#define SHISO_VIT_FAST_CUH

#include "vit_common.cuh"

#include <cuda_bf16.h>

/* Queries per block, sixteen per warp: eight warps share each staged tile. */
#define VIT_FA_BQ 128u
/* Keys per tile. */
#define VIT_FA_BK 64u
/* The head width the fragments see, a whole number of 16-wide steps. A head
 * this narrow or narrower runs here; the host sends a wider one to the scalar
 * kernel. Zero columns past the real width contribute nothing to a score and
 * are never stored. */
#define VIT_FA_HD 80u
/* Row padding of the two staged tiles, in elements, chosen so a warp's
 * fragment loads fall on 32 distinct banks. */
#define VIT_FA_PAD 8u

/* `in[i]` as two brain-floats whose sum it is to about sixteen bits: `hi[i]`
 * the nearest brain-float, `lo[i]` the nearest to what `hi` left over. `lo`
 * starts `count` elements after `hi`, in the same buffer.
 *
 * One rounding to brain-float keeps eight bits of an activation, and a tower
 * two dozen blocks deep compounds that to a percent of its largest output.
 * The weights are brain-float as stored, so multiplying them by both halves
 * and summing recovers a single-precision product at twice the multiplies. */
extern "C" __global__ void kernel_vit_to_bf16(
        const struct vit_args_unary args,
        __nv_bfloat16 *__restrict__ out,
        const float *__restrict__ in) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    const __nv_bfloat16 hi = __float2bfloat16_rn(x);
    out[i] = hi;
    out[(uint64_t) args.count + i] = __float2bfloat16_rn(x - __bfloat162float(hi));
}

/* `x[i] += bias[i % period]`, the bias brain-float: a projection's epilogue
 * when the multiply itself ran in the device's BLAS. */
extern "C" __global__ void kernel_vit_add_bias_bf16(
        const struct vit_args_add args,
        float *__restrict__ x,
        const unsigned short *__restrict__ bias) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    x[i] += vit_bf16(bias + (i % args.period));
}

/* Two floats as one register of two brain-floats, the first in the low half:
 * the order a fragment register holds consecutive columns in. */
static __device__ __forceinline__ uint32_t vit_pack_bf16(float lo, float hi) {
    const __nv_bfloat162 pair = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&pair);
}

/* The same pair as its nearest brain-floats (`*big`) and what those left over
 * (`*small`), packed alike. See `kernel_vit_to_bf16` for why the remainder is
 * kept. */
static __device__ __forceinline__ void vit_split_bf16(float x, float y, uint32_t *big,
                                                      uint32_t *small) {
    const __nv_bfloat162 hi = __floats2bfloat162_rn(x, y);
    const float2 back = __bfloat1622float2(hi);
    const __nv_bfloat162 lo = __floats2bfloat162_rn(x - back.x, y - back.y);
    *big = *reinterpret_cast<const uint32_t *>(&hi);
    *small = *reinterpret_cast<const uint32_t *>(&lo);
}

/* `d += a * b` for single-precision operands carried as brain-float pairs:
 * the big halves' product and both cross terms. The two small halves'
 * product is below what the accumulator keeps and is left out. */
static __device__ __forceinline__ void vit_mma_bf16x3(float d[4], const uint32_t ah[4],
                                                       const uint32_t al[4], uint32_t bh0,
                                                       uint32_t bh1, uint32_t bl0,
                                                       uint32_t bl1);

/* `d += a * b` over one 16x8x16 brain-float fragment, single-precision
 * accumulate. The register layouts are the instruction's, per the PTX ISA:
 * with `g = lane / 4` and `t = lane % 4`, `a` holds rows `g` and `g + 8` at
 * columns `2t, 2t+1` and `2t+8, 2t+9`; `b` holds rows `2t, 2t+1` and
 * `2t+8, 2t+9` of column `g`; `d` holds rows `g` and `g + 8` at columns
 * `2t, 2t+1`. */
static __device__ __forceinline__ void vit_mma_bf16(float d[4], const uint32_t a[4],
                                                     uint32_t b0, uint32_t b1) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

static __device__ __forceinline__ void vit_mma_bf16x3(float d[4], const uint32_t ah[4],
                                                       const uint32_t al[4], uint32_t bh0,
                                                       uint32_t bh1, uint32_t bl0,
                                                       uint32_t bl1) {
    vit_mma_bf16(d, al, bh0, bh1);
    vit_mma_bf16(d, ah, bl0, bl1);
    vit_mma_bf16(d, ah, bh0, bh1);
}

/* A query row's value at head column `d`, or zero past the row's end or the
 * head's width. */
static __device__ __forceinline__ float vit_fa_q(const float *row, bool live, unsigned d,
                                                 unsigned hd) {
    return (live && d < hd) ? row[d] : 0.0f;
}

/* The tensor-core attention's arguments: the scalar kernel's, plus the padded
 * group length its staged planes are laid out in. */
struct vit_args_attention_mma {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t plane;
    /* `plane` rounded up to a whole number of key tiles. */
    uint32_t plane_pad;
    float scale;
};
static_assert(sizeof(struct vit_args_attention_mma) == 24, "vit_args_attention_mma is 24 bytes");

/* The key and value planes as the tensor-core attention stages them, once per
 * layer: each as brain-float pairs (the nearest value, then its remainder),
 * head-major, every temporal group padded to a whole number of key tiles with
 * zeros, the values transposed.
 *
 *   kp[half][head][group * plane_pad + key][VIT_FA_HD]
 *   vp[half][head][VIT_FA_HD][group * plane_pad + key]
 *
 * Done here rather than inside the attention because every one of a head's
 * query blocks stages every one of its key tiles: converting in the attention
 * repeated this work once per 64 queries, and it was most of the kernel's
 * time. With it done once, a tile is a straight sixteen-byte copy.
 *
 * One block per padded key row, one thread per (head, column). */
extern "C" __global__ void kernel_vit_split_kv(
        const struct vit_args_attention_mma args,
        __nv_bfloat16 *__restrict__ kp,
        __nv_bfloat16 *__restrict__ vp,
        const float *__restrict__ k,
        const float *__restrict__ v) {
    const unsigned prow = blockIdx.x;
    const unsigned groups = args.rows / args.plane;
    const unsigned padded = groups * args.plane_pad;
    if (prow >= padded) {
        return;
    }
    const unsigned group = prow / args.plane_pad;
    const unsigned key = prow % args.plane_pad;
    const bool real = key < args.plane;
    const unsigned row = group * args.plane + key;
    const unsigned hd = args.head_dim;
    const unsigned dim = args.heads * hd;
    const uint64_t kplane = (uint64_t) args.heads * padded * VIT_FA_HD;
    for (unsigned i = threadIdx.x; i < args.heads * VIT_FA_HD; i += blockDim.x) {
        const unsigned h = i / VIT_FA_HD;
        const unsigned d = i % VIT_FA_HD;
        const bool in = real && d < hd;
        const uint64_t at = (uint64_t) row * dim + (uint64_t) h * hd + d;
        const float kx = in ? k[at] : 0.0f;
        const float vx = in ? v[at] : 0.0f;
        const __nv_bfloat16 kh = __float2bfloat16_rn(kx);
        const __nv_bfloat16 vh = __float2bfloat16_rn(vx);
        const uint64_t ko = ((uint64_t) h * padded + prow) * VIT_FA_HD + d;
        const uint64_t vo = ((uint64_t) h * VIT_FA_HD + d) * padded + prow;
        kp[ko] = kh;
        kp[kplane + ko] = __float2bfloat16_rn(kx - __bfloat162float(kh));
        vp[vo] = vh;
        vp[kplane + vo] = __float2bfloat16_rn(vx - __bfloat162float(vh));
    }
}

/* Bidirectional attention over one temporal group, on matrix fragments.
 *
 * The same contract as `kernel_vit_attention` - the same argument block, the
 * same planes it reads through `kernel_vit_split_kv`, the same output -
 * launched differently: one block of eight warps per (head, 128 queries of
 * one group), grid `(heads, groups * ceil(plane / 128))`, no dynamic shared
 * memory. Each warp owns sixteen query rows. Keys and values are staged a
 * 64-key tile at a time in shared memory, the values transposed, so each tile
 * is read once per 128 queries rather than every query reading every key from
 * global memory. Eight warps rather than four measured 390 -> 358 ms for a
 * 896-pixel picture on the GB10.
 *
 * The softmax is the online form: a running maximum and denominator per query
 * row, the accumulator rescaled whenever the maximum moves. Scores are taken
 * in base two, the scale folded into the queries with a factor of log2(e), so
 * the exponentials are `exp2f`.
 *
 * Every operand - queries, keys, probabilities, values - is carried as a pair
 * of brain-floats, the nearest value and its remainder, and multiplied as
 * three products (`vit_mma_bf16x3`). One brain-float rounding per operand
 * moved this tower's output by up to 5e-3 against the single-precision
 * kernel, five times the budget its reference is held to. */
extern "C" __global__ void __launch_bounds__(256) kernel_vit_attention_mma(
        const struct vit_args_attention_mma args,
        float *__restrict__ out,
        const float *__restrict__ q,
        const __nv_bfloat16 *__restrict__ kp,
        const __nv_bfloat16 *__restrict__ vp) {
    /* Each tile twice, as its nearest brain-floats and as what they left over. */
    __shared__ __align__(16) __nv_bfloat16 ks[2][VIT_FA_BK][VIT_FA_HD + VIT_FA_PAD];
    __shared__ __align__(16) __nv_bfloat16 vt[2][VIT_FA_HD][VIT_FA_BK + VIT_FA_PAD];

    const unsigned hd = args.head_dim;
    if (hd > VIT_FA_HD) {
        return;
    }
    const unsigned h = blockIdx.x;
    const unsigned tiles = (args.plane + VIT_FA_BQ - 1u) / VIT_FA_BQ;
    const unsigned group = blockIdx.y / tiles;
    const unsigned base = group * args.plane;
    const unsigned end = base + args.plane;
    const unsigned q0 = base + (blockIdx.y % tiles) * VIT_FA_BQ;
    const unsigned dim = args.heads * hd;
    const unsigned off = h * hd;
    const unsigned padded = (args.rows / args.plane) * args.plane_pad;
    const uint64_t kplane = (uint64_t) args.heads * padded * VIT_FA_HD;
    /* This head's planes, at this group's first padded key. */
    const __nv_bfloat16 *kh_src = kp + ((uint64_t) h * padded + group * args.plane_pad) * VIT_FA_HD;
    const __nv_bfloat16 *vh_src = vp + (uint64_t) h * VIT_FA_HD * padded + group * args.plane_pad;

    const unsigned tid = threadIdx.x;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const unsigned g = lane >> 2u;
    const unsigned t = lane & 3u;

    /* This lane's two query rows, and the warp's query fragments over the
     * whole (padded) head, scaled into base two once. */
    const unsigned r0 = q0 + warp * 16u + g;
    const unsigned r1 = r0 + 8u;
    const bool live0 = r0 < end && r0 < args.rows;
    const bool live1 = r1 < end && r1 < args.rows;
    const float qscale = args.scale * 1.4426950408889634f;
    const float *qr0 = q + (uint64_t) (live0 ? r0 : base) * dim + off;
    const float *qr1 = q + (uint64_t) (live1 ? r1 : base) * dim + off;
    uint32_t qa[VIT_FA_HD / 16u][4];
    uint32_t ql[VIT_FA_HD / 16u][4];
#pragma unroll
    for (unsigned s = 0u; s < VIT_FA_HD / 16u; s++) {
        const unsigned c = s * 16u + 2u * t;
        vit_split_bf16(vit_fa_q(qr0, live0, c, hd) * qscale,
                       vit_fa_q(qr0, live0, c + 1u, hd) * qscale, &qa[s][0], &ql[s][0]);
        vit_split_bf16(vit_fa_q(qr1, live1, c, hd) * qscale,
                       vit_fa_q(qr1, live1, c + 1u, hd) * qscale, &qa[s][1], &ql[s][1]);
        vit_split_bf16(vit_fa_q(qr0, live0, c + 8u, hd) * qscale,
                       vit_fa_q(qr0, live0, c + 9u, hd) * qscale, &qa[s][2], &ql[s][2]);
        vit_split_bf16(vit_fa_q(qr1, live1, c + 8u, hd) * qscale,
                       vit_fa_q(qr1, live1, c + 9u, hd) * qscale, &qa[s][3], &ql[s][3]);
    }

    float o[VIT_FA_HD / 8u][4];
#pragma unroll
    for (unsigned j = 0u; j < VIT_FA_HD / 8u; j++) {
        o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    }
    float m0 = -INFINITY, m1 = -INFINITY;
    /* Thread-partial denominators: every lane of a row rescales by the same
     * factor, so the partials are summed across the row's four lanes once, at
     * the end, rather than every tile. */
    float l0 = 0.0f, l1 = 0.0f;

    for (unsigned k0 = base; k0 < end; k0 += VIT_FA_BK) {
        __syncthreads();
        /* Stage the tile from the split planes, sixteen bytes at a time: keys
         * as 64 rows of VIT_FA_HD, values as VIT_FA_HD rows of 64. Both halves
         * of each. Past the group's end the planes hold zeros, and the mask
         * below gives those keys no weight. */
        const unsigned tk = k0 - base;
        constexpr unsigned KV = VIT_FA_HD / 8u;
        constexpr unsigned VV = VIT_FA_BK / 8u;
        for (unsigned i = tid; i < 2u * VIT_FA_BK * KV; i += blockDim.x) {
            const unsigned half = i / (VIT_FA_BK * KV);
            const unsigned r = (i / KV) % VIT_FA_BK;
            const unsigned c = i % KV;
            const uint4 *src = reinterpret_cast<const uint4 *>(
                    kh_src + half * kplane + (uint64_t) (tk + r) * VIT_FA_HD) + c;
            *reinterpret_cast<uint4 *>(&ks[half][r][c * 8u]) = *src;
        }
        for (unsigned i = tid; i < 2u * VIT_FA_HD * VV; i += blockDim.x) {
            const unsigned half = i / (VIT_FA_HD * VV);
            const unsigned d = (i / VV) % VIT_FA_HD;
            const unsigned c = i % VV;
            const uint4 *src = reinterpret_cast<const uint4 *>(
                    vh_src + half * kplane + (uint64_t) d * padded + tk) + c;
            *reinterpret_cast<uint4 *>(&vt[half][d][c * 8u]) = *src;
        }
        __syncthreads();

        /* Scores for this warp's sixteen rows against the tile's 64 keys. */
        float s[VIT_FA_BK / 8u][4];
#pragma unroll
        for (unsigned n = 0u; n < VIT_FA_BK / 8u; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (unsigned st = 0u; st < VIT_FA_HD / 16u; st++) {
                const __nv_bfloat16 *kh = &ks[0][n * 8u + g][st * 16u + 2u * t];
                const __nv_bfloat16 *kl = &ks[1][n * 8u + g][st * 16u + 2u * t];
                vit_mma_bf16x3(s[n], qa[st], ql[st],
                               *reinterpret_cast<const uint32_t *>(kh),
                               *reinterpret_cast<const uint32_t *>(kh + 8),
                               *reinterpret_cast<const uint32_t *>(kl),
                               *reinterpret_cast<const uint32_t *>(kl + 8));
            }
            /* Keys past the group's end take no weight. */
            const unsigned key = k0 + n * 8u + 2u * t;
            if (key >= end) {
                s[n][0] = s[n][2] = -INFINITY;
            }
            if (key + 1u >= end) {
                s[n][1] = s[n][3] = -INFINITY;
            }
        }

        /* The online softmax, per row: the new maximum across the row's four
         * lanes, the rescale of what came before, the tile's weights. */
        float t0 = m0, t1 = m1;
#pragma unroll
        for (unsigned n = 0u; n < VIT_FA_BK / 8u; n++) {
            t0 = fmaxf(t0, fmaxf(s[n][0], s[n][1]));
            t1 = fmaxf(t1, fmaxf(s[n][2], s[n][3]));
        }
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, 1));
        t0 = fmaxf(t0, __shfl_xor_sync(0xffffffffu, t0, 2));
        t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, 1));
        t1 = fmaxf(t1, __shfl_xor_sync(0xffffffffu, t1, 2));
        /* A row that has seen only masked keys keeps a maximum of minus
         * infinity; its rescale is taken as one so nothing becomes NaN. */
        const float a0 = (t0 == -INFINITY) ? 1.0f : exp2f(m0 - t0);
        const float a1 = (t1 == -INFINITY) ? 1.0f : exp2f(m1 - t1);
        m0 = t0;
        m1 = t1;
        l0 *= a0;
        l1 *= a1;
#pragma unroll
        for (unsigned j = 0u; j < VIT_FA_HD / 8u; j++) {
            o[j][0] *= a0;
            o[j][1] *= a0;
            o[j][2] *= a1;
            o[j][3] *= a1;
        }
#pragma unroll
        for (unsigned n = 0u; n < VIT_FA_BK / 8u; n++) {
            s[n][0] = (m0 == -INFINITY) ? 0.0f : exp2f(s[n][0] - m0);
            s[n][1] = (m0 == -INFINITY) ? 0.0f : exp2f(s[n][1] - m0);
            s[n][2] = (m1 == -INFINITY) ? 0.0f : exp2f(s[n][2] - m1);
            s[n][3] = (m1 == -INFINITY) ? 0.0f : exp2f(s[n][3] - m1);
            l0 += s[n][0] + s[n][1];
            l1 += s[n][2] + s[n][3];
        }

        /* The weights times the values: a score fragment pair is exactly the
         * left operand of one 16-key step, so no shuffle is needed. */
#pragma unroll
        for (unsigned st = 0u; st < VIT_FA_BK / 16u; st++) {
            uint32_t pa[4];
            uint32_t pl[4];
            vit_split_bf16(s[2u * st][0], s[2u * st][1], &pa[0], &pl[0]);
            vit_split_bf16(s[2u * st][2], s[2u * st][3], &pa[1], &pl[1]);
            vit_split_bf16(s[2u * st + 1u][0], s[2u * st + 1u][1], &pa[2], &pl[2]);
            vit_split_bf16(s[2u * st + 1u][2], s[2u * st + 1u][3], &pa[3], &pl[3]);
#pragma unroll
            for (unsigned j = 0u; j < VIT_FA_HD / 8u; j++) {
                const __nv_bfloat16 *vh = &vt[0][j * 8u + g][st * 16u + 2u * t];
                const __nv_bfloat16 *vl = &vt[1][j * 8u + g][st * 16u + 2u * t];
                vit_mma_bf16x3(o[j], pa, pl,
                               *reinterpret_cast<const uint32_t *>(vh),
                               *reinterpret_cast<const uint32_t *>(vh + 8),
                               *reinterpret_cast<const uint32_t *>(vl),
                               *reinterpret_cast<const uint32_t *>(vl + 8));
            }
        }
    }

    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float inv0 = (l0 > 0.0f) ? 1.0f / l0 : 0.0f;
    const float inv1 = (l1 > 0.0f) ? 1.0f / l1 : 0.0f;
#pragma unroll
    for (unsigned j = 0u; j < VIT_FA_HD / 8u; j++) {
        const unsigned d = j * 8u + 2u * t;
        if (live0) {
            float *dst = out + (uint64_t) r0 * dim + off;
            if (d < hd) dst[d] = o[j][0] * inv0;
            if (d + 1u < hd) dst[d + 1u] = o[j][1] * inv0;
        }
        if (live1) {
            float *dst = out + (uint64_t) r1 * dim + off;
            if (d < hd) dst[d] = o[j][2] * inv1;
            if (d + 1u < hd) dst[d + 1u] = o[j][3] * inv1;
        }
    }
}

#endif /* SHISO_VIT_FAST_CUH */
