/* Argument blocks and helpers of the `audio` set, as the Metal twin's
 * `audio_common.h` declares them; the blocks are the ones
 * `crates/shiso-models/src/audio_ops` packs, field for field. */
#ifndef SHISO_AUDIO_COMMON_CUH
#define SHISO_AUDIO_COMMON_CUH

#include <cuda_fp16.h>
#include <stdint.h>

#define AUDIO_LINEAR_BLOCK 256u
#define AUDIO_BLOCK_SUM_SLOTS 33u
#define AUDIO_GEMM_TILE 32u
#define AUDIO_GEMM_TILE_M 32u
#define AUDIO_GEMM_TILE_N 64u
#define AUDIO_GEMM_TILE_K 32u
#define AUDIO_GEMM64_TILE_M 64u
#define AUDIO_GEMV_ROWS 8u
#define AUDIO_GEMV_COLS 4u
#define AUDIO_ATTN_SLOTS 4u
#define AUDIO_ATTN_QB 8u
#define AUDIO_REL_ROWS 4u
#define AUDIO_ARGMAX_BLOCK 256u

#define AUDIO_OP_RELU 0u
#define AUDIO_OP_SILU 1u
#define AUDIO_OP_SIGMOID 2u
#define AUDIO_OP_GELU_ERF 3u
#define AUDIO_OP_TANH 4u
#define AUDIO_OP_CLAMP 5u
#define AUDIO_OP_SCALE 6u

#define AUDIO_NORM_HAS_WEIGHT 1u
#define AUDIO_NORM_HAS_BIAS 2u
#define AUDIO_GEMM_HAS_BIAS 1u

typedef unsigned int uint;

struct audio_args_unary { uint count; uint op; float a; float b; };
struct audio_args_axpy { uint count; float alpha; };
struct audio_args_bias_rows { uint rows; uint cols; };
struct audio_args_argmax { uint rows; uint cols; };
struct audio_args_glu { uint rows; uint channels; };
struct audio_args_norm { uint rows; uint dim; float eps; uint flags; };
struct audio_args_gemm { uint m; uint n; uint k; uint flags; uint k_chunk; uint splits; };
struct audio_args_conv2d {
    uint t_in; uint f_in; uint c_in;
    uint t_out; uint f_out; uint c_out;
    uint pad_t; uint pad_f;
    uint depthwise; uint has_bias; uint relu;
    uint in_valid; uint out_valid;
    uint batch;
};
struct audio_args_dwconv1d { uint rows; uint channels; uint taps; };
struct audio_args_unfold { uint rows; uint channels; uint taps; uint pad; };
struct audio_args_transpose { uint batch; uint rows; uint cols; };
struct audio_args_window_attn {
    uint n_q; uint n_k; uint heads; uint head_dim;
    uint q_abs0; uint k_abs0; uint k_end;
    uint chunk; uint left_chunks;
    int d_min; uint n_d;
    float scale; float ac_scale; float softcap;
    uint has_bias_uv; uint has_q_dim_scale;
};
struct audio_args_rope { uint rows; uint heads; uint head_dim; uint pos0; float theta; };
struct audio_args_lstm { uint hidden; };
struct audio_args_greedy_window { uint rows; uint cols; uint avail; };
struct audio_args_greedy_step { uint rows; uint avail; uint blank; uint max_symbols; uint slot; };
struct audio_args_embed_row { uint width; uint vocab; };
struct audio_args_keep { uint count; };

/* Greedy search's state on the device, as the Metal twin's. */
#define AUDIO_GREEDY_T 0u
#define AUDIO_GREEDY_SYM 1u
#define AUDIO_GREEDY_DONE 2u
#define AUDIO_GREEDY_TOKEN 3u
#define AUDIO_GREEDY_EMIT 4u
#define AUDIO_GREEDY_NONE 0xffffffffu

/* One brain-float, widened exactly (the 16 bits become the high half). */
static __device__ __forceinline__ float audio_bf16(const unsigned short *p) {
    return __uint_as_float(((uint) *p) << 16);
}

/* Warp sum folded in decreasing-offset order; lane 0 holds the total. */
static __device__ __forceinline__ float audio_warp_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_down_sync(0xffffffffu, v, off);
    }
    return v;
}

/* Warp sum, butterfly-folded so every lane holds the total. */
static __device__ __forceinline__ float audio_warp_sum_all(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_xor_sync(0xffffffffu, v, off);
    }
    return v;
}

/* Warp max, butterfly-folded so every lane holds it. */
static __device__ __forceinline__ float audio_warp_max_all(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
    }
    return v;
}

/* Block-wide sum over at most 32 warps; `scratch` holds
 * AUDIO_BLOCK_SUM_SLOTS floats. */
static __device__ __forceinline__ float audio_block_sum(float v, float *scratch, uint tid, uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = audio_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    __syncthreads();
    const uint n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = audio_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[AUDIO_BLOCK_SUM_SLOTS - 1u] = v;
    }
    __syncthreads();
    const float total = scratch[AUDIO_BLOCK_SUM_SLOTS - 1u];
    /* The slots are reused by the next reduction of the same launch. */
    __syncthreads();
    return total;
}

/* The error function, rational-times-Gaussian form with error below 1.5e-7:
 * the same form as the Metal twin, so the two backends agree on it. */
static __device__ __forceinline__ float audio_erf(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float a = fabsf(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
            + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - poly * expf(-a * a));
}

static __device__ __forceinline__ float audio_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

#endif /* SHISO_AUDIO_COMMON_CUH */
