/*
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels), D2 (explicit template instantiations),
 * D3 (external linkage on the template kernels: an explicit instantiation of a `static` template is emitted as a non-`.visible` entry the driver cannot resolve).
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 */

#include "q4e.cuh"

/* Q8_0 mat-vec for a narrow output.
 *
 * The stock decode kernel gives one warp per output row, so a 320-row
 * projection fills 40 thread blocks and leaves most of the memory system
 * idle.  Both hyper-connection down projections have exactly that shape and
 * run 97 times per token, which made them the single largest phase of a
 * decode step.  Here one block owns one row and its threads split that row's
 * q8 blocks, so the grid is out_dim blocks wide and each row is read as one
 * contiguous stream.  Restricted to in_dim % 32 == 0, which every qwen4exp
 * projection satisfies. */
extern "C" __global__ void q4e_matvec_q8_0_narrow_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t blocks) {
    const unsigned char *wr = w + (uint64_t)blockIdx.x * blocks * 34;
    float acc = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        const __half *scale_h = (const __half *)(wr + b * 34);
        const int8_t *qs = (const int8_t *)(wr + b * 34 + 2);
        const int8_t *xqb = xq + b * 32;
        int dot = 0;
#pragma unroll
        for (uint32_t i = 0; i < 32u; i += 4u) {
            dot = __dp4a(load_i8x4_i32_unaligned(qs + i),
                         load_i8x4_i32_aligned(xqb + i), dot);
        }
        acc += __half2float(*scale_h) * xscale[b] * (float)dot;
    }
    acc = q4e_block_sum(acc);
    if (threadIdx.x == 0u) out[blockIdx.x] = acc;
}

/* Q8_0 matmul for a handful of activation rows (2..16) -- the speculative
 * verify batch, 1 + K tokens.  */
template <int NT>
__device__ __forceinline__ static void q4e_q8_0_rows_block(
        const unsigned char *wb, const int8_t *xq, const float *xscale,
        uint64_t b, uint64_t blocks, float *acc) {
    const float scale = __half2float(*(const __half *)wb);
    const int8_t *qs = (const int8_t *)(wb + 2);
    int32_t wq[8];
#pragma unroll
    for (uint32_t i = 0; i < 8u; i++) wq[i] = load_i8x4_i32_unaligned(qs + 4u * i);
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const int8_t *xqb = xq + ((uint64_t)t * blocks + b) * 32;
        int dot = 0;
#pragma unroll
        for (uint32_t i = 0; i < 8u; i++) {
            dot = __dp4a(wq[i], load_i8x4_i32_aligned(xqb + 4u * i), dot);
        }
        acc[t] += scale * xscale[(uint64_t)t * blocks + b] * (float)dot;
    }
}

template <int NT>
__global__ void q4e_matmul_q8_0_rows_narrow_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t blocks, uint32_t out_dim) {
    const unsigned char *wr = w + (uint64_t)blockIdx.x * blocks * 34;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint64_t b = threadIdx.x; b < blocks; b += blockDim.x) {
        q4e_q8_0_rows_block<NT>(wr + b * 34, xq, xscale, b, blocks, acc);
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = q4e_block_sum(acc[t]);
        if (threadIdx.x == 0u) out[(uint64_t)t * out_dim + blockIdx.x] = v;
    }
}

template <int NT>
__global__ void q4e_matmul_q8_0_rows_warp_kernel(
        float *out, const unsigned char *w, const int8_t *xq,
        const float *xscale, uint64_t blocks, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const unsigned char *wr = w + (uint64_t)row * blocks * 34;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        q4e_q8_0_rows_block<NT>(wr + b * 34, xq, xscale, b, blocks, acc);
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * out_dim + row] = v;
    }
}

/* BF16 matmul for 1..16 activation rows: a warp per output row, lane-strided
 * bf16x2 weight loads dotted with every row's f32 activation.  The indexer
 * projections (2560 -> 128 and 2560 -> 512) are BF16 and run on every
 * forward; at a few rows the cuBLAS GemmEx path (with its f32 -> bf16
 * conversion) cost more than the arithmetic. */
template <int NT>
__global__ void q4e_matmul_bf16_rows_kernel(
        float *out, const __nv_bfloat16 *w, const float *x,
        uint32_t in_dim, uint32_t out_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const __nv_bfloat162 *wr = (const __nv_bfloat162 *)(w + (uint64_t)row * in_dim);
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint32_t i = lane; i < in_dim / 2u; i += 32u) {
        const float2 wv = __bfloat1622float2(wr[i]);
#pragma unroll
        for (int t = 0; t < NT; t++) {
            const float2 xv = *(const float2 *)(x + (uint64_t)t * in_dim + 2u * i);
            acc[t] = fmaf(wv.x, xv.x, acc[t]);
            acc[t] = fmaf(wv.y, xv.y, acc[t]);
        }
    }
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = warp_sum_f32(acc[t]);
        if (lane == 0u) out[(uint64_t)t * out_dim + row] = v;
    }
}

/* fp16 weights: the EXL3 repack keeps the checkpoint's unquantized tensors
 * (hyper-connection mix, router, PLE projections, GDN alpha/beta, injection
 * logits) as fp16.  Up to 16 rows: a warp per output row and K-split, each
 * lane reading eight weights (16 bytes) at a time and dotting them with every
 * row's f32 activation.  The split exists for the narrow outputs -- the
 * hyper-connection down projections are 10240 inputs into 320 outputs and
 * the injection logits 10240 into 4, both twice a layer -- which on a warp
 * per row alone left most of the GPU idle; partials are split-major and a
 * second pass adds them, so at one split the kernel writes the final layout.
 * Above 16 rows (prefill) cuBLAS. */
__device__ __forceinline__ static void q4e_load8(const float *p, float *v) {
    const float4 *xp = (const float4 *)p;
    const float4 xa = xp[0], xb = xp[1];
    v[0] = xa.x; v[1] = xa.y; v[2] = xa.z; v[3] = xa.w;
    v[4] = xb.x; v[5] = xb.y; v[6] = xb.z; v[7] = xb.w;
}
__device__ __forceinline__ static void q4e_load8(const __half *p, float *v) {
    const uint4 raw = *(const uint4 *)p;
    const __half2 *h2 = (const __half2 *)&raw;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 f = __half22float2(h2[j]);
        v[2 * j] = f.x;
        v[2 * j + 1] = f.y;
    }
}

template <int NT, typename XT, typename OT>
__global__ void q4e_matmul_f16_rows_kernel(
        OT *out, const __half *w, const XT *x,
        uint32_t in_dim, uint32_t out_dim, uint32_t chunk) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = blockIdx.x * (blockDim.x >> 5u) + (threadIdx.x >> 5u);
    if (row >= out_dim) return;
    const uint32_t i0 = blockIdx.y * chunk;
    const uint32_t i1 = min(i0 + chunk, in_dim);
    const __half *wr = w + (uint64_t)row * in_dim;
    float acc[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) acc[t] = 0.0f;
    for (uint32_t i = i0 + lane * 8u; i < i1; i += 256u) {
        const uint4 raw = *(const uint4 *)(wr + i);
        const __half2 *h2 = (const __half2 *)&raw;
        float wv[8];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float2 f = __half22float2(h2[j]);
            wv[2 * j] = f.x;
            wv[2 * j + 1] = f.y;
        }
#pragma unroll
        for (int t = 0; t < NT; t++) {
            float xv[8];
            q4e_load8(x + (uint64_t)t * in_dim + i, xv);
#pragma unroll
            for (int j = 0; j < 8; j++) acc[t] = fmaf(wv[j], xv[j], acc[t]);
        }
    }
    OT *dst = out + (uint64_t)blockIdx.y * NT * out_dim;
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const float v = warp_sum_f32(acc[t]);
        if (lane == 0u) q4e_store(dst + (uint64_t)t * out_dim + row, v);
    }
}

/* The K-split partials are fp32 whatever the output type. */
template <typename OT>
__global__ void q4e_matvec_combine_t_kernel(
        OT *out, const float *partial, uint32_t out_dim, uint32_t n_split) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_split; s++) acc += partial[(uint64_t)s * out_dim + row];
    q4e_store(out + row, acc);
}

extern "C" __global__ void q4e_f32_to_f16_kernel(__half *out, const float *in, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half_rn(in[i]);
}

/* The widening a half-precision weight takes before a single-precision
 * multiply.  Exact: every half is a float, so the product the multiply forms
 * is the one the per-row kernel forms, and only the summation order differs. */
extern "C" __global__ void q4e_f16_to_f32_kernel(float *out, const __half *in, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __half2float(in[i]);
}

/* Draft-head slice (see q4e_head_slice in ds4.c).  The selected 128-row
 * blocks of an EXL3 vocabulary head are gathered into a payload with the
 * layout of a (k, n_sel) tensor -- tiles, then suh[k], then svh[n_sel] -- so
 * shs_exl3_gemm runs it as any dense EXL3 tensor.  Blocks of 128 rather than
 * single rows because the output Hadamard mixes every 128 consecutive output
 * columns: a logit depends on all eight n16 tiles of its block, and only whole
 * blocks can be lifted out intact. */
extern "C" __global__ void q4e_exl3_slice_tiles_kernel(
        uint8_t *dst, const uint8_t *src, const int32_t *blocks, uint32_t n_blocks,
        uint32_t src_blocks_n, uint32_t tile_bytes, uint64_t n_pieces) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pieces) return;
    const uint32_t per_tile = tile_bytes / 16u;
    const uint64_t tile = i / per_tile, piece = i - tile * per_tile;
    const uint32_t row = (uint32_t)(tile / ((uint64_t)n_blocks * 8u));     /* k16 block */
    const uint32_t rem = (uint32_t)(tile - (uint64_t)row * n_blocks * 8u);
    const uint32_t b = rem >> 3u, j = rem & 7u;
    const uint64_t s = ((uint64_t)row * src_blocks_n + (uint64_t)blocks[b] * 8u + j) * tile_bytes;
    ((uint4 *)(dst + tile * tile_bytes))[piece] = ((const uint4 *)(src + s))[piece];
}

extern "C" __global__ void q4e_exl3_slice_svh_kernel(
        __half *dst, const __half *src, const int32_t *blocks, uint32_t n_blocks) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_blocks * 128u) return;
    dst[i] = src[(uint64_t)blocks[i >> 7u] * 128u + (i & 127u)];
}

/* F32 mat-vec with the contraction split across blocks.
 *
 * The generic F32 matmul gives one block per output row, which is fine for the
 * 512-wide router but leaves the hyper-connection injection logits -- 10240
 * inputs into 4 outputs, 97 times a token -- running on four blocks.  Here the
 * K range is cut into n_split pieces when the output is narrow, and a second
 * pass adds the pieces up.  Partials are split-major, so at n_split == 1 the
 * first kernel already writes the final layout and the second is skipped. */
extern "C" __global__ void q4e_matvec_f32_kernel(
        float *out, const float *w, const float *x,
        uint32_t in_dim, uint32_t chunk) {
    const uint32_t row = blockIdx.x;
    const uint32_t sp = blockIdx.y;
    const uint32_t i0 = sp * chunk;
    const uint32_t i1 = min(i0 + chunk, in_dim);
    const float *wr = w + (uint64_t)row * in_dim;
    float acc = 0.0f;
    for (uint32_t i = i0 + threadIdx.x; i < i1; i += blockDim.x) acc += wr[i] * x[i];
    acc = q4e_block_sum(acc);
    if (threadIdx.x == 0u) out[(uint64_t)sp * gridDim.x + row] = acc;
}

extern "C" __global__ void q4e_matvec_f32_combine_kernel(
        float *out, const float *partial, uint32_t out_dim, uint32_t n_split) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t s = 0; s < n_split; s++) acc += partial[(uint64_t)s * out_dim + row];
    out[row] = acc;
}

/* Explicit instantiations for exactly the argument sets the launchers
 * use.  `extern "C"` is illegal on a template, so these entries keep
 * their mangled names and are resolved by mangled name. */
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<2>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<3>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<4>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<5>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<6>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<7>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<8>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<9>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<10>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<11>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<12>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<13>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<14>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<15>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_narrow_kernel<16>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<2>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<3>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<4>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<5>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<6>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<7>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<8>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<9>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<10>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<11>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<12>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<13>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<14>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<15>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_q8_0_rows_warp_kernel<16>(
        float *, const unsigned char *, const int8_t *, const float *,
        uint64_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<1>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<2>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<3>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<4>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<5>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<6>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<7>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<8>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<9>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<10>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<11>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<12>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<13>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<14>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<15>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_bf16_rows_kernel<16>(
        float *, const __nv_bfloat16 *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<1, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<2, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<3, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<4, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<5, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<6, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<7, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<8, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<9, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<10, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<11, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<12, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<13, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<14, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<15, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<16, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
/* 17 to 32 rows (shiso): a batched verify pass stays on the per-row kernel. */
template __global__ void q4e_matmul_f16_rows_kernel<17, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<18, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<19, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<20, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<21, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<22, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<23, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<24, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<25, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<26, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<27, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<28, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<29, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<30, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<31, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<32, float, float>(
        float *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<1, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<2, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<3, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<4, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<5, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<6, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<7, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<8, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<9, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<10, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<11, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<12, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<13, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<14, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<15, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<16, __half, float>(
        float *, const __half *, const __half *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<1, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<2, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<3, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<4, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<5, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<6, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<7, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<8, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<9, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<10, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<11, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<12, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<13, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<14, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<15, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matmul_f16_rows_kernel<16, float, __half>(
        __half *, const __half *, const float *, uint32_t, uint32_t, uint32_t);
template __global__ void q4e_matvec_combine_t_kernel<float>(
        float *, const float *, uint32_t, uint32_t);
template __global__ void q4e_matvec_combine_t_kernel<__half>(
        __half *, const float *, uint32_t, uint32_t);
