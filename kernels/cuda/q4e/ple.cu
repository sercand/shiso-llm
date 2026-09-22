/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels).
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * PLE n-gram block (layer SHS_PLE_LAYER only).
 * ------------------------------------------------------------------------ */

/* Decode the streamed IQ4_NL rows into the flat per-token embedding.  Row r of
 * the gather belongs to token r / n_heads and occupies head slot r % n_heads,
 * so the heads land head-slowest inside the 2560-wide result, matching the
 * reference's reshape of a [head_dim, n_heads * n_tok] gather. */
extern "C" __global__ void q4e_ple_dequant_kernel(
        float *out, const uint8_t *rows, uint32_t head_dim, uint32_t n_heads,
        uint32_t row_bytes, uint32_t n_rows) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) return;
    const uint8_t *src = rows + (uint64_t)r * row_bytes;
    const uint32_t t = r / n_heads;
    const uint32_t h = r % n_heads;
    float *dst = out + (uint64_t)t * n_heads * head_dim + (uint64_t)h * head_dim;

    const int8_t kv[16] = { -127, -104, -83, -65, -49, -35, -22, -10,
                              1,   13,  25,  38,  53,  69,  89, 113 };
    for (uint32_t b = threadIdx.x; b < head_dim / 32u; b += blockDim.x) {
        const uint8_t *blk = src + (uint64_t)b * 18u;
        const float d = __half2float(*(const __half *)blk);
        for (uint32_t j = 0; j < 16u; j++) {
            const uint8_t q = blk[2 + j];
            dst[b * 32u + j]       = d * (float)kv[q & 0x0Fu];
            dst[b * 32u + j + 16u] = d * (float)kv[q >> 4];
        }
    }
}

/* The EXL3 checkpoint's row codec, one 160-thread block per row: bit m of
 * weight i sits at ring position ((i - m/6) mod 160) * 6 + m%6 of a 6-bit
 * tail-biting trellis; the 16-bit state decodes through the mul1 codebook,
 * then the row's fp16 scale and the hash head's bias.  Port of exllamav3's
 * ngram_dequant_kernel, rounding where it rounds so rows match its table. */
extern "C" __global__ void q4e_ple_dequant_exl3_kernel(
        float *out, const uint8_t *rows, const __half *bias,
        uint32_t head_dim, uint32_t n_heads, uint32_t row_bytes, uint32_t n_rows) {
    const uint32_t r = blockIdx.x;
    const uint32_t i = threadIdx.x;
    if (r >= n_rows) return;
    const uint32_t words = row_bytes / 2u;
    __shared__ uint16_t sw[64];
    if (i < words) sw[i] = ((const uint16_t *)(rows + (uint64_t)r * row_bytes))[i];
    __syncthreads();
    if (i >= head_dim) return;

    const float scale = __half2float(__ushort_as_half(sw[0]));
    uint32_t state = 0;
#pragma unroll 4
    for (uint32_t m = 0; m < 16u; m++) {
        int pos = (int)i - (int)(m / 6u);
        if (pos < 0) pos += (int)head_dim;
        const uint32_t sb = (uint32_t)pos * 6u + m % 6u;
        state |= ((uint32_t)(sw[1 + (sb >> 4)] >> (sb & 15u)) & 1u) << m;
    }
    const uint32_t prod = state * 0x83DCD12Du;
    const float h = 1024.0f + (float)((prod & 0xFFu) + ((prod >> 8) & 0xFFu) +
                                      ((prod >> 16) & 0xFFu) + ((prod >> 24) & 0xFFu));
    const float k_inv = __half2float(__ushort_as_half(0x1eee));
    const float k_bias = __half2float(__ushort_as_half(0xc931));
    const float cb = __half2float(__float2half_rn(h * k_inv + k_bias));

    const uint32_t t = r / n_heads;
    const uint32_t hd = r % n_heads;
    const float b = __half2float(bias[(uint64_t)hd * head_dim + i]);
    out[((uint64_t)t * n_heads + hd) * head_dim + i] = __half2float(__float2half_rn(fmaf(cb, scale, b)));
}

/* gate[t][h] = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6))) with
 * s = <key[t][h], query[t][h]> / sqrt(n_embd), then broadcast the value across
 * the streams.  The signed square root before the sigmoid is easy to miss and
 * changes the gate's shape completely. */
extern "C" __global__ void q4e_ple_gated_value_kernel(
        float *gv, const float *key, const float *query, const float *value,
        uint32_t n_embd, uint32_t n_hc) {
    const uint32_t t = blockIdx.x;
    const uint32_t h = blockIdx.y;
    const uint64_t base = ((uint64_t)t * n_hc + h) * n_embd;

    float dot = 0.0f;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        dot += key[base + e] * query[base + e];
    }
    dot = q4e_block_sum(dot) * rsqrtf((float)n_embd);
    const float mag = sqrtf(fmaxf(fabsf(dot), 1e-6f));
    const float g = 1.0f / (1.0f + __expf(-copysignf(mag, dot)));

    const float *v = value + (uint64_t)t * n_embd;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) gv[base + e] = v[e] * g;
}

/* Depthwise causal conv dilated by the n-gram size: taps sit at t, t-3, t-6,
 * t-9 for the shipped kernel of 4.  The history is (kernel - 1) * dilation
 * columns, carried per sequence so a chunked prefill matches a single-shot
 * one, and silu is applied to the result. */
extern "C" __global__ void q4e_ple_conv_kernel(
        float *out, float *state, const float *x, const float *w,
        uint32_t channels, uint32_t kernel, uint32_t dilation, uint32_t n_tok,
        float *ckpt, uint32_t ckpt_cap) {
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    const uint32_t hist = (kernel - 1u) * dilation;

    float win[16];
    for (uint32_t i = 0; i < hist; i++) win[i] = state[(uint64_t)i * channels + c];

    const float *wc = w + (uint64_t)c * kernel;
    for (uint32_t t = 0; t < n_tok; t++) {
        const float cur = x[(uint64_t)t * channels + c];
        float acc = wc[kernel - 1u] * cur;
        for (uint32_t k = 0; k + 1u < kernel; k++) {
            /* tap k reads (kernel - 1 - k) * dilation positions back */
            const uint32_t back = (kernel - 1u - k) * dilation;
            acc += wc[k] * win[hist - back];
        }
        out[(uint64_t)t * channels + c] = acc / (1.0f + __expf(-acc));
        for (uint32_t i = 0; i + 1u < hist; i++) win[i] = win[i + 1];
        if (hist > 0) win[hist - 1] = cur;
        if (ckpt && t < ckpt_cap && t + 1u < n_tok) {
            float *ck = ckpt + (uint64_t)t * hist * channels;
            for (uint32_t i = 0; i < hist; i++) ck[(uint64_t)i * channels + c] = win[i];
        }
    }

    for (uint32_t i = 0; i < hist; i++) state[(uint64_t)i * channels + c] = win[i];
}

extern "C" __global__ void q4e_ple_conv_batch_kernel(
        float *out, float * const *states, const float *x, const float *w,
        uint32_t channels, uint32_t kernel, uint32_t dilation) {
    const uint32_t row = blockIdx.y;
    const uint32_t c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float *state = states[row];
    if (!state) return;
    const uint32_t hist = (kernel - 1u) * dilation;
    const float *xr = x + (uint64_t)row * channels;
    float *outr = out + (uint64_t)row * channels;
    float win[16];
    for (uint32_t i = 0; i < hist; i++) win[i] = state[(uint64_t)i * channels + c];
    const float *wc = w + (uint64_t)c * kernel;
    const float cur = xr[c];
    float acc = wc[kernel - 1u] * cur;
    for (uint32_t k = 0; k + 1u < kernel; k++) {
        const uint32_t back = (kernel - 1u - k) * dilation;
        acc += wc[k] * win[hist - back];
    }
    outr[c] = acc / (1.0f + __expf(-acc));
    for (uint32_t i = 0; i + 1u < hist; i++) win[i] = win[i + 1];
    if (hist > 0) win[hist - 1] = cur;
    for (uint32_t i = 0; i < hist; i++) state[(uint64_t)i * channels + c] = win[i];
}
