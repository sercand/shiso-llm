/* The `audio` module: the audio encoders' kernel set.
 *
 * Nothing here is copied. The arithmetic is the Metal twin's in
 * `crates/shiso-metal/metal/audio`, the launch contract is that directory's
 * KERNELS.md, and the entry names are the host's `&'static str` literals in
 * `crates/shiso-models/src/audio_ops`, so every entry carries C linkage.
 *
 * Launch geometry is the contract's: the element-wise entries take a flat
 * index over 256-thread blocks, the norms one block per row, the GEMM a 32x64
 * output tile per 128-thread block with 12288 bytes of dynamic shared memory
 * (the 64-row GEMM 64x64 per 256 threads with 20480),
 * and the window attention one warp per (head, query).
 */

#include "audio_common.cuh"

/* ---- element-wise --------------------------------------------------------- */

extern "C" __global__ void kernel_audio_unary_f32(
        const struct audio_args_unary args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    float y;
    switch (args.op) {
        case AUDIO_OP_RELU:     y = fmaxf(x, 0.0f); break;
        case AUDIO_OP_SILU:     y = x * audio_sigmoid(x); break;
        case AUDIO_OP_SIGMOID:  y = audio_sigmoid(x); break;
        case AUDIO_OP_GELU_ERF: y = 0.5f * x * (1.0f + audio_erf(x * 0.70710678118654752f)); break;
        case AUDIO_OP_TANH:     y = tanhf(x); break;
        case AUDIO_OP_CLAMP:    y = fminf(fmaxf(x, args.a), args.b); break;
        case AUDIO_OP_SCALE:    y = x * args.a; break;
        default:                y = x; break;
    }
    out[i] = y;
}

extern "C" __global__ void kernel_audio_axpy_f32(
        const struct audio_args_axpy args, float *y, const float *x) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    y[i] = y[i] + args.alpha * x[i];
}

extern "C" __global__ void kernel_audio_glu_f32(
        const struct audio_args_glu args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.channels) {
        return;
    }
    const uint r = i / args.channels;
    const uint c = i - r * args.channels;
    const float *row = in + (uint64_t) r * 2u * args.channels;
    out[i] = row[c] * audio_sigmoid(row[args.channels + c]);
}

extern "C" __global__ void kernel_audio_bias_rows_f32(
        const struct audio_args_bias_rows args, float *y, const float *b) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.cols) {
        return;
    }
    y[i] = y[i] + b[i % args.cols];
}

/* The larger of two (value, index) candidates; a tie keeps the lower index. */
static __device__ __forceinline__ void audio_argmax_pick(float &v, uint &i, float ov, uint oi) {
    if (ov > v || (ov == v && oi < i)) {
        v = ov;
        i = oi;
    }
}

/* Per row, the first index of the largest value, one block of
 * AUDIO_ARGMAX_BLOCK threads per row. A NaN never wins and a row with nothing
 * above negative infinity answers 0. */
extern "C" __global__ void kernel_audio_argmax_rows_f32(
        const struct audio_args_argmax args, uint *out, const float *in) {
    __shared__ float best_v[AUDIO_ARGMAX_BLOCK / 32u];
    __shared__ uint best_i[AUDIO_ARGMAX_BLOCK / 32u];
    const uint r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    const float *row = in + (uint64_t) r * args.cols;
    float v = -INFINITY;
    uint i = 0xffffffffu;
    for (uint c = threadIdx.x; c < args.cols; c += AUDIO_ARGMAX_BLOCK) {
        if (row[c] > v) {
            v = row[c];
            i = c;
        }
    }
    for (uint off = 16u; off > 0u; off >>= 1u) {
        audio_argmax_pick(v, i, __shfl_xor_sync(0xffffffffu, v, off),
                          __shfl_xor_sync(0xffffffffu, i, off));
    }
    if ((threadIdx.x & 31u) == 0u) {
        best_v[threadIdx.x >> 5u] = v;
        best_i[threadIdx.x >> 5u] = i;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
        for (uint s = 1; s < AUDIO_ARGMAX_BLOCK / 32u; ++s) {
            audio_argmax_pick(v, i, best_v[s], best_i[s]);
        }
        out[r] = i == 0xffffffffu ? 0u : i;
    }
}

/* ---- norms ---------------------------------------------------------------- */

/* Two-pass about the row's first value, as the Metal twin: a row with a large
 * mean and a small variance keeps its precision. Safe in place. */
extern "C" __global__ void kernel_audio_layer_norm_f32(
        const struct audio_args_norm args, float *out, const float *in,
        const float *weight, const float *bias) {
    extern __shared__ float scratch[];
    const uint r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    const uint nt = blockDim.x;
    const uint tid = threadIdx.x;
    const float *src = in + (uint64_t) r * args.dim;
    float *dst = out + (uint64_t) r * args.dim;
    const float pivot = src[0];
    float sum = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        sum += src[i] - pivot;
    }
    const float dmean = audio_block_sum(sum, scratch, tid, nt) / (float) args.dim;
    float var = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        const float d = (src[i] - pivot) - dmean;
        var += d * d;
    }
    const float rstd = rsqrtf(audio_block_sum(var, scratch, tid, nt) / (float) args.dim + args.eps);
    const bool has_bias = (args.flags & AUDIO_NORM_HAS_BIAS) != 0u;
    for (uint i = tid; i < args.dim; i += nt) {
        const float y = ((src[i] - pivot) - dmean) * rstd * weight[i];
        dst[i] = has_bias ? y + bias[i] : y;
    }
}

extern "C" __global__ void kernel_audio_rms_norm_f32(
        const struct audio_args_norm args, float *out, const float *in, const float *weight) {
    extern __shared__ float scratch[];
    const uint r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    const uint nt = blockDim.x;
    const uint tid = threadIdx.x;
    const float *src = in + (uint64_t) r * args.dim;
    float *dst = out + (uint64_t) r * args.dim;
    float sq = 0.0f;
    for (uint i = tid; i < args.dim; i += nt) {
        sq += src[i] * src[i];
    }
    const float rms = rsqrtf(audio_block_sum(sq, scratch, tid, nt) / (float) args.dim + args.eps);
    const bool has_weight = (args.flags & AUDIO_NORM_HAS_WEIGHT) != 0u;
    for (uint i = tid; i < args.dim; i += nt) {
        const float y = src[i] * rms;
        dst[i] = has_weight ? y * weight[i] : y;
    }
}

/* ---- the GEMM ------------------------------------------------------------- */

struct audio_w_f32 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        return ((const float *) w)[i];
    }
};
struct audio_w_bf16 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        return audio_bf16((const unsigned short *) w + i);
    }
};
struct audio_w_f16 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        return __half2float(((const __half *) w)[i]);
    }
};
/* GGUF Q8_0: 34-byte blocks of an f16 scale then 32 signed bytes along each
 * row, dequantised in f32. The scale is read as two bytes so the load makes no
 * assumption about the base the buffer was imported at; blocks are 2-byte but
 * not 4-byte aligned relative to it. */
struct audio_w_q8_0 {
    static __device__ __forceinline__ float load(const void *w, uint64_t i) {
        const unsigned char *block = (const unsigned char *) w + (i >> 5u) * 34u;
        const unsigned short bits = (unsigned short) (block[0] | (block[1] << 8));
        const float d = __half2float(__ushort_as_half(bits));
        return d * (float) ((const signed char *) block)[2u + (i & 31u)];
    }
};

/* y = x * W^T (+ b): a 32-row by 64-column output tile per block of 128
 * threads, both operands staged through shared memory with zero fill past
 * every edge; each thread accumulates sixteen outputs of one row in f32. With
 * more than one split, block z covers its slice of k and writes a partial
 * tile into plane z of `y`, which the split-sum pass adds up. */
/* BM rows by 64 columns per block of 4 * BM threads, each thread one row by
 * 16 columns: the 32-row tile on 128 threads, and the 64-row tile the host
 * picks for more than 32 rows on 256. */
template <typename W, uint BM>
static __device__ __forceinline__ void audio_gemm(
        const struct audio_args_gemm args, float *y, const float *x, const void *w,
        const float *bias) {
    extern __shared__ float smem[];
    const uint NT = 4u * BM;
    const uint BN = AUDIO_GEMM_TILE_N;
    const uint BK = AUDIO_GEMM_TILE_K;
    float *sa = smem;           /* [BM rows][32 k] */
    float *sb = smem + BM * BK; /* [64 cols][32 k] */
    const uint tid = threadIdx.x;
    const uint n0 = blockIdx.x * BN;
    const uint m0 = blockIdx.y * BM;
    const uint k_begin = blockIdx.z * args.k_chunk;
    const uint k_end = min(args.k, k_begin + args.k_chunk);
    float *out = y + (uint64_t) blockIdx.z * args.m * args.n;
    const uint row = tid >> 2u;         /* 0..BM-1 */
    const uint col0 = (tid & 3u) * 16u; /* 0, 16, 32, 48 */
    float acc[16];
    for (uint j = 0; j < 16u; ++j) {
        acc[j] = 0.0f;
    }
    for (uint k0 = k_begin; k0 < k_end; k0 += BK) {
        for (uint e = tid; e < BM * BK; e += NT) {
            const uint rr = e / BK;
            const uint kk = e - rr * BK;
            const uint gk = k0 + kk;
            const uint gm = m0 + rr;
            sa[e] = (gm < args.m && gk < k_end) ? x[(uint64_t) gm * args.k + gk] : 0.0f;
        }
        for (uint e = tid; e < BN * BK; e += NT) {
            const uint rr = e / BK;
            const uint kk = e - rr * BK;
            const uint gk = k0 + kk;
            const uint gn = n0 + rr;
            sb[e] = (gn < args.n && gk < k_end) ? W::load(w, (uint64_t) gn * args.k + gk) : 0.0f;
        }
        __syncthreads();
        for (uint kk = 0; kk < BK; ++kk) {
            const float a = sa[row * BK + kk];
            for (uint j = 0; j < 16u; ++j) {
                acc[j] += a * sb[(col0 + j) * BK + kk];
            }
        }
        __syncthreads();
    }
    /* A partial tile carries no bias: the split-sum pass adds it once. */
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u && args.splits <= 1u;
    const uint gm = m0 + row;
    if (gm >= args.m) {
        return;
    }
    for (uint j = 0; j < 16u; ++j) {
        const uint gn = n0 + col0 + j;
        if (gn < args.n) {
            out[(uint64_t) gm * args.n + gn] = acc[j] + (has_bias ? bias[gn] : 0.0f);
        }
    }
}

/* y = x * W^T (+ b) for a handful of rows: each warp owns 4 output columns of
 * up to 8 rows and strides its columns' weights along k, so a block of four
 * warps covers 16 columns and the grid is `n / 16` by `m / 8`. */
template <typename W>
static __device__ __forceinline__ void audio_gemv(
        const struct audio_args_gemm args, float *y, const float *x, const void *w,
        const float *bias) {
    const uint MT = AUDIO_GEMV_ROWS;
    const uint NC = AUDIO_GEMV_COLS;
    const uint lane = threadIdx.x & 31u;
    const uint warp = threadIdx.x >> 5u;
    const uint n0 = (blockIdx.x * 4u + warp) * NC;
    const uint m0 = blockIdx.y * MT;
    if (n0 >= args.n || m0 >= args.m) {
        return;
    }
    const uint rows = min(MT, args.m - m0);
    const uint cols = min(NC, args.n - n0);
    float acc[AUDIO_GEMV_ROWS][AUDIO_GEMV_COLS];
    for (uint r = 0; r < MT; ++r) {
        for (uint c = 0; c < NC; ++c) {
            acc[r][c] = 0.0f;
        }
    }
    for (uint kk = lane; kk < args.k; kk += 32u) {
        float wv[AUDIO_GEMV_COLS];
        for (uint c = 0; c < NC; ++c) {
            wv[c] = c < cols ? W::load(w, (uint64_t) (n0 + c) * args.k + kk) : 0.0f;
        }
        for (uint r = 0; r < rows; ++r) {
            const float xv = x[(uint64_t) (m0 + r) * args.k + kk];
            for (uint c = 0; c < NC; ++c) {
                acc[r][c] += xv * wv[c];
            }
        }
    }
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
    for (uint r = 0; r < rows; ++r) {
        for (uint c = 0; c < cols; ++c) {
            const float total = audio_warp_sum(acc[r][c]);
            if (lane == 0u) {
                y[(uint64_t) (m0 + r) * args.n + n0 + c] = total + (has_bias ? bias[n0 + c] : 0.0f);
            }
        }
    }
}

#define AUDIO_GEMM_ENTRIES(SUFFIX, W) \
    extern "C" __global__ void kernel_audio_gemm_##SUFFIX( \
            const struct audio_args_gemm args, float *y, const float *x, const void *w, const float *bias) { \
        audio_gemm<W, AUDIO_GEMM_TILE_M>(args, y, x, w, bias); \
    } \
    extern "C" __global__ void kernel_audio_gemm64_##SUFFIX( \
            const struct audio_args_gemm args, float *y, const float *x, const void *w, const float *bias) { \
        audio_gemm<W, AUDIO_GEMM64_TILE_M>(args, y, x, w, bias); \
    } \
    extern "C" __global__ void kernel_audio_gemv_##SUFFIX( \
            const struct audio_args_gemm args, float *y, const float *x, const void *w, const float *bias) { \
        audio_gemv<W>(args, y, x, w, bias); \
    }

AUDIO_GEMM_ENTRIES(f32w, audio_w_f32)
AUDIO_GEMM_ENTRIES(bf16w, audio_w_bf16)
AUDIO_GEMM_ENTRIES(f16w, audio_w_f16)
AUDIO_GEMM_ENTRIES(q8_0w, audio_w_q8_0)

/* y[i] = sum_z parts[z][i] (+ b[i % n]): closes a split-k GEMM, summing the
 * planes in order so the result does not depend on scheduling. */
extern "C" __global__ void kernel_audio_split_sum_f32(
        const struct audio_args_gemm args, float *y, const float *parts, const float *bias) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint count = args.m * args.n;
    if (i >= count) {
        return;
    }
    float total = 0.0f;
    for (uint z = 0; z < args.splits; ++z) {
        total += parts[(uint64_t) z * count + i];
    }
    const bool has_bias = (args.flags & AUDIO_GEMM_HAS_BIAS) != 0u;
    y[i] = total + (has_bias ? bias[i % args.n] : 0.0f);
}

/* ---- convolutions and layout moves --------------------------------------- */

extern "C" __global__ void kernel_audio_conv2d_s2_f32(
        const struct audio_args_conv2d args, float *out, const float *in, const float *w,
        const float *bias) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint plane = args.t_out * args.f_out * args.c_out;
    if (i >= args.batch * plane) {
        return;
    }
    // Planes are independent images: each one's padding is zeros, never the
    // neighbouring plane's rows.
    const uint b = i / plane;
    const uint j = i - b * plane;
    const uint co = j % args.c_out;
    const uint f = (j / args.c_out) % args.f_out;
    const uint t = j / (args.c_out * args.f_out);
    const float *in_b = in + (uint64_t) b * args.t_in * args.f_in * args.c_in;
    if (t >= args.out_valid) {
        out[i] = 0.0f;
        return;
    }
    const uint rows = min(args.t_in, args.in_valid);
    const uint per = args.depthwise != 0u ? 1u : args.c_in;
    float acc = args.has_bias != 0u ? bias[co] : 0.0f;
    for (uint kt = 0; kt < 3u; ++kt) {
        const int ti = (int) (2u * t + kt) - (int) args.pad_t;
        if (ti < 0 || (uint) ti >= rows) {
            continue;
        }
        for (uint kf = 0; kf < 3u; ++kf) {
            const int fi = (int) (2u * f + kf) - (int) args.pad_f;
            if (fi < 0 || (uint) fi >= args.f_in) {
                continue;
            }
            const float *px = in_b + ((uint64_t) ti * args.f_in + (uint) fi) * args.c_in;
            for (uint g = 0; g < per; ++g) {
                const uint ci = args.depthwise != 0u ? co : g;
                acc += w[((co * per + g) * 3u + kt) * 3u + kf] * px[ci];
            }
        }
    }
    out[i] = args.relu != 0u ? fmaxf(acc, 0.0f) : acc;
}

// The patches of a stride-2 3x3 conv2d, so a GEMM can take the conv:
// `out[b][t][f][(kt*3 + kf)*c_in + ci] = in[b][2t+kt-pad_t][2f+kf-pad_f][ci]`,
// zero outside the plane, past `in_valid` and on output rows past
// `out_valid`. Reads the conv's argument block; `c_out`, `depthwise`,
// `has_bias` and `relu` are unread.
extern "C" __global__ void kernel_audio_im2col_s2_f32(
        const struct audio_args_conv2d args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint taps = 9u * args.c_in;
    const uint plane = args.t_out * args.f_out * taps;
    if (i >= args.batch * plane) {
        return;
    }
    const uint b = i / plane;
    const uint j = i - b * plane;
    const uint ci = j % args.c_in;
    const uint tap = (j / args.c_in) % 9u;
    const uint f = (j / taps) % args.f_out;
    const uint t = j / (taps * args.f_out);
    const int ti = (int) (2u * t + tap / 3u) - (int) args.pad_t;
    const int fi = (int) (2u * f + tap % 3u) - (int) args.pad_f;
    const uint rows = min(args.t_in, args.in_valid);
    float v = 0.0f;
    if (t < args.out_valid && ti >= 0 && (uint) ti < rows && fi >= 0 && (uint) fi < args.f_in) {
        v = in[(((uint64_t) b * args.t_in + (uint) ti) * args.f_in + (uint) fi) * args.c_in + ci];
    }
    out[i] = v;
}

extern "C" __global__ void kernel_audio_dwconv1d_f32(
        const struct audio_args_dwconv1d args, float *out, const float *in, const float *w) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.channels) {
        return;
    }
    const uint c = i % args.channels;
    const uint t = i / args.channels;
    float acc = 0.0f;
    for (uint j = 0; j < args.taps; ++j) {
        acc += w[c * args.taps + j] * in[(uint64_t) (t + j) * args.channels + c];
    }
    out[i] = acc;
}

extern "C" __global__ void kernel_audio_unfold_f32(
        const struct audio_args_unfold args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint width = args.taps * args.channels;
    if (i >= args.rows * width) {
        return;
    }
    const uint t = i / width;
    const uint j = (i - t * width) / args.channels;
    const uint c = i % args.channels;
    const int src = (int) (t + j) - (int) args.pad;
    out[i] = (src < 0 || (uint) src >= args.rows) ? 0.0f : in[(uint64_t) src * args.channels + c];
}

extern "C" __global__ void kernel_audio_transpose_f32(
        const struct audio_args_transpose args, float *out, const float *in) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint plane = args.rows * args.cols;
    if (i >= args.batch * plane) {
        return;
    }
    const uint b = i / plane;
    const uint r = (i - b * plane) / args.cols;
    const uint c = i % args.cols;
    out[(uint64_t) b * plane + (uint64_t) c * args.rows + r] = in[i];
}

/* ---- attention and rotation ---------------------------------------------- */

/* The relative-position half of the attention logits, once per (query, head,
 * distance): scores[i][h][t] = (q'[i] + bias_v) . rel[t] over head h's dims,
 * with q' = q * q_dim_scale. A warp takes AUDIO_ATTN_QB queries by
 * AUDIO_REL_ROWS distance rows; lanes stride the head's dims. */
extern "C" __global__ void kernel_audio_rel_scores_f32(
        const struct audio_args_window_attn args, float *scores, const float *q,
        const float *bias_v, const float *q_dim_scale, const float *rel) {
    const uint QB = AUDIO_ATTN_QB;
    const uint TR = AUDIO_REL_ROWS;
    const uint t0 = blockIdx.x * TR;
    const uint h = blockIdx.y;
    const uint i0 = blockIdx.z * QB;
    const uint lane = threadIdx.x;
    if (t0 >= args.n_d || h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint nq = min(QB, args.n_q - i0);
    const uint nt = min(TR, args.n_d - t0);
    float acc[AUDIO_ATTN_QB][AUDIO_REL_ROWS];
    for (uint r = 0; r < QB; ++r) {
        for (uint t = 0; t < TR; ++t) {
            acc[r][t] = 0.0f;
        }
    }
    for (uint e = lane; e < hd; e += 32u) {
        const float scale = args.has_q_dim_scale != 0u ? q_dim_scale[e] : 1.0f;
        const float bv = args.has_bias_uv != 0u ? bias_v[hoff + e] : 0.0f;
        float rv[AUDIO_REL_ROWS];
        for (uint t = 0; t < TR; ++t) {
            rv[t] = t < nt ? rel[(uint64_t) (t0 + t) * w + hoff + e] : 0.0f;
        }
        for (uint r = 0; r < QB; ++r) {
            if (r < nq) {
                const float qv = q[(uint64_t) (i0 + r) * w + hoff + e] * scale + bv;
                for (uint t = 0; t < TR; ++t) {
                    acc[r][t] += qv * rv[t];
                }
            }
        }
    }
    for (uint r = 0; r < QB; ++r) {
        for (uint t = 0; t < TR; ++t) {
            const float total = audio_warp_sum_all(acc[r][t]);
            if (lane == 0u && r < nq && t < nt) {
                scores[((uint64_t) (i0 + r) * args.heads + h) * args.n_d + t0 + t] = total;
            }
        }
    }
}

/* Attention for a block of AUDIO_ATTN_QB queries of one head per warp. The
 * block's queries (content bias and per-dim scale applied) sit in shared
 * memory; keys stream through 32 at a time, one per lane, with the relative
 * term read from the pre-pass's scores, the running max and sum moving once
 * per 32 keys, and each lane owning a slice of the head's dims for the value
 * sum. Visibility is decided per (query, key), so a block may straddle
 * chunks. */
extern "C" __global__ void kernel_audio_window_attn_f32(
        const struct audio_args_window_attn args, float *out, const float *q, const float *k,
        const float *v, const float *bias_u, const float *q_dim_scale, const float *rel_scores) {
    extern __shared__ float qa[];
    const uint QB = AUDIO_ATTN_QB;
    const uint h = blockIdx.x;
    const uint i0 = blockIdx.y * QB;
    const uint lane = threadIdx.x;
    if (h >= args.heads || i0 >= args.n_q) {
        return;
    }
    const uint hd = args.head_dim;
    const uint w = args.heads * hd;
    const uint hoff = h * hd;
    const uint nq = min(QB, args.n_q - i0);
    const bool has_rel = args.n_d > 0u;
    for (uint e = lane; e < QB * hd; e += 32u) {
        const uint r = e / hd;
        const uint d = e - r * hd;
        float qv = 0.0f;
        if (r < nq) {
            qv = q[(uint64_t) (i0 + r) * w + hoff + d];
            if (args.has_q_dim_scale != 0u) {
                qv *= q_dim_scale[d];
            }
            if (args.has_bias_uv != 0u) {
                qv += bias_u[hoff + d];
            }
        }
        qa[e] = qv;
    }
    __syncwarp();

    const uint qfirst = args.q_abs0 + i0;
    const uint qlast = qfirst + nq - 1u;
    const uint c_first = qfirst / args.chunk;
    const uint c_last = qlast / args.chunk;
    const uint lo_chunk = c_first > args.left_chunks ? c_first - args.left_chunks : 0u;
    const uint lo_abs = max(lo_chunk * args.chunk, args.k_abs0);
    const uint hi_abs = min(min((c_last + 1u) * args.chunk, args.k_end), args.k_abs0 + args.n_k);

    float m[AUDIO_ATTN_QB];
    float l[AUDIO_ATTN_QB];
    float acc[AUDIO_ATTN_QB][AUDIO_ATTN_SLOTS];
    for (uint r = 0; r < QB; ++r) {
        m[r] = -INFINITY;
        l[r] = 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            acc[r][s] = 0.0f;
        }
    }
    for (uint base = lo_abs; base < hi_abs; base += 32u) {
        const uint ka = base + lane;
        const bool key_in = ka < hi_abs;
        const uint j = ka - args.k_abs0;
        const uint kc = ka / args.chunk;
        float sc[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            sc[r] = 0.0f;
        }
        if (key_in) {
            const float *krow = k + (uint64_t) j * w + hoff;
            for (uint d = 0; d < hd; ++d) {
                const float kd = krow[d];
                for (uint r = 0; r < QB; ++r) {
                    sc[r] += qa[r * hd + d] * kd;
                }
            }
        }
        for (uint r = 0; r < QB; ++r) {
            const uint qabs = qfirst + r;
            const uint qc = qabs / args.chunk;
            const bool visible = key_in && r < nq && kc <= qc && qc - kc <= args.left_chunks;
            float s_r = -INFINITY;
            if (visible) {
                float bd = 0.0f;
                if (has_rel) {
                    const int dr = (int) qabs - (int) ka - args.d_min;
                    if (dr >= 0 && (uint) dr < args.n_d) {
                        bd = rel_scores[((uint64_t) (i0 + r) * args.heads + h) * args.n_d + (uint) dr];
                    }
                }
                s_r = args.scale * (args.ac_scale * sc[r] + bd);
                if (args.softcap > 0.0f) {
                    s_r = args.softcap * tanhf(s_r / args.softcap);
                }
            }
            sc[r] = s_r;
        }
        float p[AUDIO_ATTN_QB];
        for (uint r = 0; r < QB; ++r) {
            const float block_max = audio_warp_max_all(sc[r]);
            const float m_new = fmaxf(m[r], block_max);
            const float corr = m_new == -INFINITY ? 1.0f : expf(m[r] - m_new);
            p[r] = sc[r] == -INFINITY ? 0.0f : expf(sc[r] - m_new);
            l[r] = l[r] * corr + audio_warp_sum_all(p[r]);
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                acc[r][s] *= corr;
            }
            m[r] = m_new;
        }
        const uint keys_here = min(32u, hi_abs - base);
        for (uint jj = 0; jj < keys_here; ++jj) {
            const uint jv = base + jj - args.k_abs0;
            const float *vrow = v + (uint64_t) jv * w + hoff;
            float vv[AUDIO_ATTN_SLOTS];
            for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                const uint e = lane + 32u * s;
                vv[s] = e < hd ? vrow[e] : 0.0f;
            }
            for (uint r = 0; r < QB; ++r) {
                const float pj = __shfl_sync(0xffffffffu, p[r], jj);
                for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
                    acc[r][s] += pj * vv[s];
                }
            }
        }
    }
    for (uint r = 0; r < nq; ++r) {
        const float inv = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
        for (uint s = 0; s < AUDIO_ATTN_SLOTS; ++s) {
            const uint e = lane + 32u * s;
            if (e < hd) {
                out[(uint64_t) (i0 + r) * w + hoff + e] = acc[r][s] * inv;
            }
        }
    }
}

/* Half-split rotary embedding, in place: for m < hd/2, (x[m], x[m+hd/2])
 * rotates by (pos0 + row) * theta^(-2m/hd). */
extern "C" __global__ void kernel_audio_rope_half_f32(
        const struct audio_args_rope args, float *x) {
    const uint half_dim = args.head_dim / 2u;
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.heads * half_dim) {
        return;
    }
    const uint m = i % half_dim;
    const uint rh = i / half_dim;
    const uint row = rh / args.heads;
    float *p = x + (uint64_t) rh * args.head_dim;
    const float inv_freq = powf(args.theta, -2.0f * (float) m / (float) args.head_dim);
    const float ang = (float) (args.pos0 + row) * inv_freq;
    float s;
    float c;
    sincosf(ang, &s, &c);
    const float a = p[m];
    const float b = p[m + half_dim];
    p[m] = a * c - b * s;
    p[m + half_dim] = b * c + a * s;
}

/* ---- the recurrent cell --------------------------------------------------- */

extern "C" __global__ void kernel_audio_lstm_cell_f32(
        const struct audio_args_lstm args, float *h_out, float *c, const float *gates) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint H = args.hidden;
    if (u >= H) {
        return;
    }
    const float gi = audio_sigmoid(gates[u]);
    const float gf = audio_sigmoid(gates[H + u]);
    const float gg = tanhf(gates[2u * H + u]);
    const float go = audio_sigmoid(gates[3u * H + u]);
    const float cn = gf * c[u] + gi * gg;
    c[u] = cn;
    h_out[u] = go * tanhf(cn);
}

/* ---- greedy transducer search on the device ------------------------------ */

/* out[r][c] = relu(pending[t + r][c] + dec[c]) for the window's rows, 0 for
 * rows at or past avail; t is read from the search state. */
extern "C" __global__ void kernel_audio_greedy_window_f32(
        const struct audio_args_greedy_window args, float *out, const float *pending,
        const float *dec, const uint *state) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.rows * args.cols) {
        return;
    }
    const uint r = i / args.cols;
    const uint c = i - r * args.cols;
    const uint src = state[AUDIO_GREEDY_T] + r;
    out[i] = src < args.avail ? fmaxf(pending[(uint64_t) src * args.cols + c] + dec[c], 0.0f) : 0.0f;
}

/* One decision of greedy search over the window's argmax ids, on one thread,
 * as the Metal twin. */
extern "C" __global__ void kernel_audio_greedy_step_u32(
        const struct audio_args_greedy_step args, uint *state, uint *out, const uint *ids) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) {
        return;
    }
    uint t = state[AUDIO_GREEDY_T];
    uint sym = state[AUDIO_GREEDY_SYM];
    out[2u * args.slot] = AUDIO_GREEDY_NONE;
    out[2u * args.slot + 1u] = 0u;
    state[AUDIO_GREEDY_EMIT] = 0u;
    if (t >= args.avail) {
        state[AUDIO_GREEDY_DONE] = 1u;
        return;
    }
    const uint m = min(args.rows, args.avail - t);
    uint j = m;
    for (uint r = 0; r < m; ++r) {
        if (ids[r] != args.blank) {
            j = r;
            break;
        }
    }
    if (j == m) {
        t += m;
        sym = 0u;
    } else {
        if (j > 0u) {
            t += j;
            sym = 0u;
        }
        const uint k = ids[j];
        out[2u * args.slot] = k;
        out[2u * args.slot + 1u] = t;
        state[AUDIO_GREEDY_TOKEN] = k;
        state[AUDIO_GREEDY_EMIT] = 1u;
        sym += 1u;
        if (sym >= args.max_symbols) {
            t += 1u;
            sym = 0u;
        }
    }
    state[AUDIO_GREEDY_T] = t;
    state[AUDIO_GREEDY_SYM] = sym;
    state[AUDIO_GREEDY_DONE] = t >= args.avail ? 1u : 0u;
}

/* out = table[token], the state's last token's row; zeros past the table. */
extern "C" __global__ void kernel_audio_embed_row_f32(
        const struct audio_args_embed_row args, float *out, const float *table, const uint *state) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (u >= args.width) {
        return;
    }
    const uint k = state[AUDIO_GREEDY_TOKEN];
    out[u] = k < args.vocab ? table[(uint64_t) k * args.width + u] : 0.0f;
}

/* The cell above, keeping the new state only under the emit flag; x_out
 * always takes h'. */
extern "C" __global__ void kernel_audio_lstm_cell_gated_f32(
        const struct audio_args_lstm args, float *x_out, float *h, float *c, const float *gates,
        const uint *state) {
    const uint u = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    const uint H = args.hidden;
    if (u >= H) {
        return;
    }
    const float gi = audio_sigmoid(gates[u]);
    const float gf = audio_sigmoid(gates[H + u]);
    const float gg = tanhf(gates[2u * H + u]);
    const float go = audio_sigmoid(gates[3u * H + u]);
    const float cn = gf * c[u] + gi * gg;
    const float hn = go * tanhf(cn);
    x_out[u] = hn;
    if (state[AUDIO_GREEDY_EMIT] != 0u) {
        c[u] = cn;
        h[u] = hn;
    }
}

/* dst = src when the last step emitted, else nothing. */
extern "C" __global__ void kernel_audio_keep_f32(
        const struct audio_args_keep args, float *dst, const float *src, const uint *state) {
    const uint i = blockIdx.x * AUDIO_LINEAR_BLOCK + threadIdx.x;
    if (i >= args.count || state[AUDIO_GREEDY_EMIT] == 0u) {
        return;
    }
    dst[i] = src[i];
}
