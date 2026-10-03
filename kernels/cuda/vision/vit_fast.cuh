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

/* The element a split operand is stored in, by the tower's weight width: the
 * BLAS multiplies the operand against the weights, and both sides of one call
 * are one type. A half-precision plane holds every operand this tower
 * multiplies - each is a normalized row, an attention output or an activated
 * one, far inside the half's range. A pair of halves keeps about 22 bits of a
 * value, against 16 for a pair of brain floats, down to an absolute floor of
 * 2^-25 where the remainder falls below the smallest subnormal half: an
 * operand that small is noise in a sum over the tower's widths. */
template <typename W> struct vit_plane;
template <> struct vit_plane<vit_w_bf16> {
    typedef __nv_bfloat16 t;
    static __device__ __forceinline__ t round(float x) { return __float2bfloat16_rn(x); }
    static __device__ __forceinline__ float back(t h) { return __bfloat162float(h); }
};
template <> struct vit_plane<vit_w_f16> {
    typedef __half t;
    static __device__ __forceinline__ t round(float x) { return __float2half_rn(x); }
    static __device__ __forceinline__ float back(t h) { return __half2float(h); }
};

/* Writes `x` at `i` of an operand split into pairs of the plane's element: the
 * nearest value at `out[i]`, the remainder `count` elements further on. The
 * layout `kernel_vit_to_bf16` writes and a BLAS projection reads. */
template <typename W>
static __device__ __forceinline__ void vit_store_split(typename vit_plane<W>::t *out,
                                                       uint64_t count, uint64_t i, float x) {
    const typename vit_plane<W>::t hi = vit_plane<W>::round(x);
    out[i] = hi;
    out[count + i] = vit_plane<W>::round(x - vit_plane<W>::back(hi));
}

/* `in[i]` as two brain-floats whose sum it is to about sixteen bits: `hi[i]`
 * the nearest brain-float, `lo[i]` the nearest to what `hi` left over. `lo`
 * starts `count` elements after `hi`, in the same buffer.
 *
 * One rounding to brain-float keeps eight bits of an activation, and a tower
 * two dozen blocks deep compounds that to a percent of its largest output.
 * The weights are brain-float as stored, so multiplying them by both halves
 * and summing recovers a single-precision product at twice the multiplies. */
template <typename W>
static __device__ __forceinline__ void kernel_vit_to_bf16_body(
        const struct vit_args_unary args,
        typename vit_plane<W>::t *__restrict__ out,
        const float *__restrict__ in) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    vit_store_split<W>(out, args.count, i, in[i]);
}

/* The brain-float split and the half-precision one; see `vit_plane`. */
extern "C" __global__ void kernel_vit_to_bf16(
        const struct vit_args_unary args,
        __nv_bfloat16 *__restrict__ out,
        const float *__restrict__ in) {
    kernel_vit_to_bf16_body<vit_w_bf16>(args, out, in);
}

extern "C" __global__ void kernel_vit_to_f16(
        const struct vit_args_unary args,
        __half *__restrict__ out,
        const float *__restrict__ in) {
    kernel_vit_to_bf16_body<vit_w_f16>(args, out, in);
}

/* `x[i] += bias[i % period]`, the bias brain-float: a projection's epilogue
 * when the multiply itself ran in the device's BLAS. */
template <typename W>
static __device__ __forceinline__ void kernel_vit_add_bias_bf16_body(
        const struct vit_args_add args,
        float *__restrict__ x,
        const unsigned short *__restrict__ bias) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    x[i] += W::load(bias + (i % args.period));
}

/* The brain-float tower's entry and the half-precision tower's twin; see
 * `vit_w_bf16`. */
extern "C" __global__ void kernel_vit_add_bias_bf16(
        const struct vit_args_add args,
        float *__restrict__ x,
        const unsigned short *__restrict__ bias) {
    kernel_vit_add_bias_bf16_body<vit_w_bf16>(args, x, bias);
}

extern "C" __global__ void kernel_vit_add_bias_f16(
        const struct vit_args_add args,
        float *__restrict__ x,
        const unsigned short *__restrict__ bias) {
    kernel_vit_add_bias_bf16_body<vit_w_f16>(args, x, bias);
}

/* `kernel_vit_layer_norm` writing its rows as the next projection's split
 * operand instead of as single-precision rows: the normalized row is only ever
 * read by that projection, so writing it once as f32 and then again as the
 * pair was a whole pass over the hidden stream for nothing. The arithmetic is
 * the unfused kernel's, value for value. */
template <typename W>
static __device__ __forceinline__ void kernel_vit_layer_norm_split_body(
        const struct vit_args_layer_norm args,
        typename vit_plane<W>::t *__restrict__ out,
        const float *__restrict__ in,
        const unsigned short *__restrict__ weight,
        const unsigned short *__restrict__ bias) {
    const unsigned r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    float *scratch = (float *) vit_smem;
    const unsigned tid = threadIdx.x;
    const unsigned nthreads = blockDim.x;
    const float *src = in + (uint64_t) r * args.dim;
    const uint64_t count = (uint64_t) args.rows * args.dim;

    float sum = 0.0f;
    for (unsigned i = tid; i < args.dim; i += nthreads) {
        sum += src[i];
    }
    const float mean = vit_block_sum(sum, scratch, tid, nthreads) / (float) args.dim;
    float var = 0.0f;
    for (unsigned i = tid; i < args.dim; i += nthreads) {
        const float d = src[i] - mean;
        var += d * d;
    }
    const float rstd = rsqrtf(
            vit_block_sum(var, scratch, tid, nthreads) / (float) args.dim + args.eps);
    for (unsigned i = tid; i < args.dim; i += nthreads) {
        const float y = (src[i] - mean) * rstd * W::load(weight + i) + W::load(bias + i);
        vit_store_split<W>(out, count, (uint64_t) r * args.dim + i, y);
    }
}

/* The brain-float tower's entry and the half-precision tower's twin; see
 * `vit_w_bf16`. */
extern "C" __global__ void kernel_vit_layer_norm_split(
        const struct vit_args_layer_norm args,
        __nv_bfloat16 *__restrict__ out,
        const float *__restrict__ in,
        const unsigned short *__restrict__ weight,
        const unsigned short *__restrict__ bias) {
    kernel_vit_layer_norm_split_body<vit_w_bf16>(args, out, in, weight, bias);
}

extern "C" __global__ void kernel_vit_layer_norm_split_f16(
        const struct vit_args_layer_norm args,
        __half *__restrict__ out,
        const float *__restrict__ in,
        const unsigned short *__restrict__ weight,
        const unsigned short *__restrict__ bias) {
    kernel_vit_layer_norm_split_body<vit_w_f16>(args, out, in, weight, bias);
}

/* The first feed-forward layer's epilogue and the second's operand in one
 * pass: `gelu_tanh(x[i] + bias[i % period])`, written as a split operand. The
 * activated row is read by nothing but the second layer, so it is never
 * stored as single precision. */
template <typename W>
static __device__ __forceinline__ void kernel_vit_bias_gelu_split_body(
        const struct vit_args_add args,
        typename vit_plane<W>::t *__restrict__ out,
        const float *__restrict__ x,
        const unsigned short *__restrict__ bias) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    vit_store_split<W>(out, args.count, i, vit_gelu_tanh(x[i] + W::load(bias + (i % args.period))));
}

/* The brain-float tower's entry and the half-precision tower's twin; see
 * `vit_w_bf16`. */
extern "C" __global__ void kernel_vit_bias_gelu_split(
        const struct vit_args_add args,
        __nv_bfloat16 *__restrict__ out,
        const float *__restrict__ x,
        const unsigned short *__restrict__ bias) {
    kernel_vit_bias_gelu_split_body<vit_w_bf16>(args, out, x, bias);
}

extern "C" __global__ void kernel_vit_bias_gelu_split_f16(
        const struct vit_args_add args,
        __half *__restrict__ out,
        const float *__restrict__ x,
        const unsigned short *__restrict__ bias) {
    kernel_vit_bias_gelu_split_body<vit_w_f16>(args, out, x, bias);
}

/* A projection's bias and the block's residual in one pass:
 * `x[i] = (x[i] + bias[i % period]) + residual[i]`, in the order the two
 * unfused kernels added them. */
template <typename W>
static __device__ __forceinline__ void kernel_vit_add_bias_residual_body(
        const struct vit_args_add args,
        float *__restrict__ x,
        const unsigned short *__restrict__ bias,
        const float *__restrict__ residual) {
    const unsigned i = blockIdx.x * VIT_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    x[i] = (x[i] + W::load(bias + (i % args.period))) + residual[i];
}

/* The brain-float tower's entry and the half-precision tower's twin; see
 * `vit_w_bf16`. */
extern "C" __global__ void kernel_vit_add_bias_residual(
        const struct vit_args_add args,
        float *__restrict__ x,
        const unsigned short *__restrict__ bias,
        const float *__restrict__ residual) {
    kernel_vit_add_bias_residual_body<vit_w_bf16>(args, x, bias, residual);
}

extern "C" __global__ void kernel_vit_add_bias_residual_f16(
        const struct vit_args_add args,
        float *__restrict__ x,
        const unsigned short *__restrict__ bias,
        const float *__restrict__ residual) {
    kernel_vit_add_bias_residual_body<vit_w_f16>(args, x, bias, residual);
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

/* `2^x` from the special-function unit. Its error, two units in the last place,
 * is some thousand times below what the brain-float pairs keep, and the
 * accurate `exp2f` sequence was a fifth of this kernel's instructions. A
 * large negative argument flushes to zero, which is the weight it stands for. */
static __device__ __forceinline__ float vit_ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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

/* `kernel_vit_qkv_rope_split`'s arguments: the rotation's, plus the padded
 * group length of the split planes it writes. */
struct vit_args_qkv_rope_split {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    uint32_t grid_w;
    uint32_t plane;
    uint32_t plane_pad;
    float rope_base;
};
static_assert(sizeof(struct vit_args_qkv_rope_split) == 28, "vit_args_qkv_rope_split is 28 bytes");

/* The fused projection's bias, the two-dimensional rotation and the key/value
 * split in one pass over the fused row: `q` comes out rotated in single
 * precision, the keys rotated and the values as-is, both straight into the
 * split planes `kernel_vit_attention_mma` stages from.
 *
 * The arithmetic is `kernel_vit_add_bias_bf16`, then `kernel_vit_qkv_rope`,
 * then `kernel_vit_split_kv`, value for value; what is gone is two
 * single-precision planes written and read back once per layer. Only real
 * key rows are written, so the planes' padding rows must be zero before the
 * first layer - the host clears them once per encode - and the padding
 * columns past the head's width are written as zeros here.
 *
 * One block per (head, row), as the rotation. */
template <typename W>
static __device__ __forceinline__ void kernel_vit_qkv_rope_split_body(
        const struct vit_args_qkv_rope_split args,
        float *__restrict__ q,
        __nv_bfloat16 *__restrict__ kp,
        __nv_bfloat16 *__restrict__ vp,
        const float *__restrict__ qkv,
        const unsigned short *__restrict__ bias) {
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
    const unsigned per_axis = half_hd / 2u;

    const unsigned local = n % args.plane;
    const float row = (float) (local / args.grid_w);
    const float col = (float) (local % args.grid_w);
    const unsigned padded = (args.rows / args.plane) * args.plane_pad;
    const unsigned prow = (n / args.plane) * args.plane_pad + local;
    const uint64_t kplane = (uint64_t) args.heads * padded * VIT_FA_HD;

    const float *src = qkv + (uint64_t) n * 3u * dim;
    const unsigned off = h * hd;
    float *qd = q + (uint64_t) n * dim + off;
    __nv_bfloat16 *kd = kp + ((uint64_t) h * padded + prow) * VIT_FA_HD;
    __nv_bfloat16 *vd = vp + (uint64_t) h * VIT_FA_HD * padded + prow;

    for (unsigned i = tid; i < VIT_FA_HD; i += nthreads) {
        const float x = (i < hd) ? src[2u * dim + off + i] + W::load(bias + 2u * dim + off + i)
                                 : 0.0f;
        const __nv_bfloat16 hi = __float2bfloat16_rn(x);
        vd[(uint64_t) i * padded] = hi;
        vd[kplane + (uint64_t) i * padded] = __float2bfloat16_rn(x - __bfloat162float(hi));
        if (i >= hd) {
            kd[i] = __float2bfloat16_rn(0.0f);
            kd[kplane + i] = __float2bfloat16_rn(0.0f);
        }
    }
    for (unsigned i = tid; i < half_hd; i += nthreads) {
        const unsigned freq = (i < per_axis) ? i : (i - per_axis);
        const float pos = (i < per_axis) ? row : col;
        const float inv = powf(args.rope_base, -2.0f * (float) freq / (float) half_hd);
        const float theta = pos * inv;
        const float s = sinf(theta);
        const float c = cosf(theta);
        const float q0 = src[off + i] + W::load(bias + off + i);
        const float q1 = src[off + i + half_hd] + W::load(bias + off + i + half_hd);
        qd[i] = q0 * c - q1 * s;
        qd[i + half_hd] = q1 * c + q0 * s;
        const float k0 = src[dim + off + i] + W::load(bias + dim + off + i);
        const float k1 = src[dim + off + i + half_hd] + W::load(bias + dim + off + i + half_hd);
        vit_store_split<vit_w_bf16>(kd, kplane, i, k0 * c - k1 * s);
        vit_store_split<vit_w_bf16>(kd, kplane, i + half_hd, k1 * c + k0 * s);
    }
}

/* The brain-float tower's entry and the half-precision tower's twin: only the
 * bias is read in the tower's width. The key and value planes are the
 * attention's operands, brain-float pairs whatever the weights are. */
extern "C" __global__ void kernel_vit_qkv_rope_split(
        const struct vit_args_qkv_rope_split args,
        float *__restrict__ q,
        __nv_bfloat16 *__restrict__ kp,
        __nv_bfloat16 *__restrict__ vp,
        const float *__restrict__ qkv,
        const unsigned short *__restrict__ bias) {
    kernel_vit_qkv_rope_split_body<vit_w_bf16>(args, q, kp, vp, qkv, bias);
}

extern "C" __global__ void kernel_vit_qkv_rope_split_f16(
        const struct vit_args_qkv_rope_split args,
        float *__restrict__ q,
        __nv_bfloat16 *__restrict__ kp,
        __nv_bfloat16 *__restrict__ vp,
        const float *__restrict__ qkv,
        const unsigned short *__restrict__ bias) {
    kernel_vit_qkv_rope_split_body<vit_w_f16>(args, q, kp, vp, qkv, bias);
}

/* One staged key/value tile: each as its nearest brain-floats and as what
 * they left over, the values transposed. The attention holds two, one being
 * filled while the other is read. */
struct vit_fa_tile {
    __nv_bfloat16 ks[2][VIT_FA_BK][VIT_FA_HD + VIT_FA_PAD];
    __nv_bfloat16 vt[2][VIT_FA_HD][VIT_FA_BK + VIT_FA_PAD];
};
/* The attention's dynamic shared memory, which the host passes at launch. */
#define VIT_FA_SMEM (2u * (unsigned) sizeof(struct vit_fa_tile))

/* Sixteen bytes from global to shared memory without passing through a
 * register, completing asynchronously. */
static __device__ __forceinline__ void vit_cp_async16(void *dst, const void *src) {
    const unsigned at = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(at), "l"(src));
}

/* Issues the copies that fill `dst` with the key tile starting `tk` keys into
 * the group, and closes them as one group. */
static __device__ __forceinline__ void vit_fa_stage(struct vit_fa_tile *dst,
                                                    const __nv_bfloat16 *kh_src,
                                                    const __nv_bfloat16 *vh_src, uint64_t kplane,
                                                    unsigned padded, unsigned tk) {
    constexpr unsigned KV = VIT_FA_HD / 8u;
    constexpr unsigned VV = VIT_FA_BK / 8u;
    for (unsigned i = threadIdx.x; i < 2u * VIT_FA_BK * KV; i += blockDim.x) {
        const unsigned half = i / (VIT_FA_BK * KV);
        const unsigned r = (i / KV) % VIT_FA_BK;
        const unsigned c = i % KV;
        vit_cp_async16(&dst->ks[half][r][c * 8u],
                       kh_src + half * kplane + (uint64_t) (tk + r) * VIT_FA_HD + c * 8u);
    }
    for (unsigned i = threadIdx.x; i < 2u * VIT_FA_HD * VV; i += blockDim.x) {
        const unsigned half = i / (VIT_FA_HD * VV);
        const unsigned d = (i / VV) % VIT_FA_HD;
        const unsigned c = i % VV;
        vit_cp_async16(&dst->vt[half][d][c * 8u],
                       vh_src + half * kplane + (uint64_t) d * padded + tk + c * 8u);
    }
    asm volatile("cp.async.commit_group;\n" ::);
}

/* Bidirectional attention over one temporal group, on matrix fragments.
 *
 * The same contract as `kernel_vit_attention` - the same argument block, the
 * same planes it reads through `kernel_vit_split_kv`, the same output -
 * launched differently: one block of eight warps per (head, 128 queries of
 * one group), grid `(heads, groups * ceil(plane / 128))`, VIT_FA_SMEM bytes
 * of dynamic shared memory. Each warp owns sixteen query rows. Keys and
 * values are staged a 64-key tile at a time, the values transposed, so each
 * tile is read once per 128 queries rather than every query reading every key
 * from global memory; two tiles are held, and the next one's asynchronous
 * copies run while this one is multiplied. Measured on the GB10 for a
 * 896-pixel picture's whole tower: eight warps rather than four, 390 -> 358
 * ms; the second tile, 294 -> 239 ms.
 *
 * The softmax is the online form: a running maximum and denominator per query
 * row, the accumulator rescaled whenever the maximum moves. Scores are taken
 * in base two, the scale folded into the queries with a factor of log2(e), so
 * the exponentials are `vit_ex2`.
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
    /* Two tiles, VIT_FA_SMEM bytes of dynamic shared memory: the next key
     * tile's copies are in flight while this one's fragments are multiplied. */
    struct vit_fa_tile *staged = reinterpret_cast<struct vit_fa_tile *>(vit_smem);

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

    /* Past the group's end the planes hold zeros, and the mask below gives
     * those keys no weight. */
    const unsigned ntiles = args.plane_pad / VIT_FA_BK;
    vit_fa_stage(&staged[0], kh_src, vh_src, kplane, padded, 0u);
    for (unsigned it = 0u; it < ntiles; it++) {
        const unsigned k0 = base + it * VIT_FA_BK;
        if (it + 1u < ntiles) {
            vit_fa_stage(&staged[(it + 1u) & 1u], kh_src, vh_src, kplane, padded,
                         (it + 1u) * VIT_FA_BK);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
        const struct vit_fa_tile &tile = staged[it & 1u];

        /* Scores for this warp's sixteen rows against the tile's 64 keys. */
        float s[VIT_FA_BK / 8u][4];
#pragma unroll
        for (unsigned n = 0u; n < VIT_FA_BK / 8u; n++) {
            s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.0f;
#pragma unroll
            for (unsigned st = 0u; st < VIT_FA_HD / 16u; st++) {
                const __nv_bfloat16 *kh = &tile.ks[0][n * 8u + g][st * 16u + 2u * t];
                const __nv_bfloat16 *kl = &tile.ks[1][n * 8u + g][st * 16u + 2u * t];
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
        const float a0 = (t0 == -INFINITY) ? 1.0f : vit_ex2(m0 - t0);
        const float a1 = (t1 == -INFINITY) ? 1.0f : vit_ex2(m1 - t1);
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
            s[n][0] = (m0 == -INFINITY) ? 0.0f : vit_ex2(s[n][0] - m0);
            s[n][1] = (m0 == -INFINITY) ? 0.0f : vit_ex2(s[n][1] - m0);
            s[n][2] = (m1 == -INFINITY) ? 0.0f : vit_ex2(s[n][2] - m1);
            s[n][3] = (m1 == -INFINITY) ? 0.0f : vit_ex2(s[n][3] - m1);
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
                const __nv_bfloat16 *vh = &tile.vt[0][j * 8u + g][st * 16u + 2u * t];
                const __nv_bfloat16 *vl = &tile.vt[1][j * 8u + g][st * 16u + 2u * t];
                vit_mma_bf16x3(o[j], pa, pl,
                               *reinterpret_cast<const uint32_t *>(vh),
                               *reinterpret_cast<const uint32_t *>(vh + 8),
                               *reinterpret_cast<const uint32_t *>(vl),
                               *reinterpret_cast<const uint32_t *>(vl + 8));
            }
        }
        /* The next iteration refills the buffer just read. */
        __syncthreads();
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
