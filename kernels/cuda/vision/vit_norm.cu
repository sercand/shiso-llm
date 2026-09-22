/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the
 * kernels, which were `static` in a header written to be included into one
 * translation unit and now have to be reachable from a module registry).  No
 * line of kernel arithmetic was edited, reordered or reformatted; a change to
 * any of it is a defect, not a cleanup.
 *
 * The normalization and the three pointwise epilogues.  They are one family:
 * every block of the tower runs the normalization twice and one of the two
 * activations once, and all four widen their affine parameters out of the same
 * brain-float rows.
 */

#include "vision.cuh"

/* LayerNorm (mean and variance, not RMS) with weight and bias. */
extern "C" __global__ void qwen3vl_layernorm_kernel(
        float *out, const float *in, const uint16_t *weight,
        const uint16_t *bias, uint32_t dim) {
    const uint32_t r = blockIdx.x;
    const float *src = in + (size_t)r * dim;
    float *dst = out + (size_t)r * dim;
    __shared__ float s_mean, s_rstd;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) sum += src[i];
    sum = q4e_block_sum(sum);
    if (threadIdx.x == 0) s_mean = sum / (float)dim;
    __syncthreads();
    const float mean = s_mean;
    float var = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        const float d = src[i] - mean;
        var += d * d;
    }
    var = q4e_block_sum(var);
    if (threadIdx.x == 0) s_rstd = rsqrtf(var / (float)dim + SHS_QWEN3VL_VISION_EPS);
    __syncthreads();
    const float rstd = s_rstd;
    for (uint32_t i = threadIdx.x; i < dim; i += blockDim.x) {
        dst[i] = (src[i] - mean) * rstd * qwen3vl_bf16(weight + i) +
                 qwen3vl_bf16(bias + i);
    }
}

/* out[i] += bias[i % dim], and optionally a residual. */
extern "C" __global__ void qwen3vl_bias_kernel(
        float *x, const uint16_t *bias, const float *residual,
        uint64_t n, uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = x[i] + qwen3vl_bf16(bias + (uint32_t)(i % dim));
    if (residual) v += residual[i];
    x[i] = v;
}

/* gelu_pytorch_tanh, the blocks' activation. */
extern "C" __global__ void qwen3vl_gelu_tanh_kernel(
        float *x, const uint16_t *bias, uint64_t n, uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] + qwen3vl_bf16(bias + (uint32_t)(i % dim));
    const float c = 0.7978845608028654f;   /* sqrt(2/pi) */
    x[i] = 0.5f * v * (1.0f + tanhf(c * (v + 0.044715f * v * v * v)));
}

/* Exact erf GELU: nn.GELU() in the merger, which is NOT the tanh
 * approximation the blocks use.  erff is exact and the merger runs once. */
extern "C" __global__ void qwen3vl_gelu_erf_kernel(
        float *x, const uint16_t *bias, uint64_t n, uint32_t dim) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] + qwen3vl_bf16(bias + (uint32_t)(i % dim));
    x[i] = 0.5f * v * (1.0f + erff(v * 0.7071067811865476f));
}
