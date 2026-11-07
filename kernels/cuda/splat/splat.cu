/* The `splat` module: the background remover's operations the diffusion
 * pack does not carry. The entries, their argument blocks and their
 * arithmetic are the Metal twin's (`kernels/metal/splat/splat.metal`), entry
 * for entry; see that file for what each computes and why.
 */

#include <stdint.h>

typedef uint32_t u32;
typedef uint64_t u64;

#define SPLAT_BLOCK 64u
#define SPLAT_MAX_WINDOW_TOKENS 256u

struct splat_args_unary {
    u32 count;
};
static_assert(sizeof(splat_args_unary) == 4, "splat_args_unary layout");

struct splat_args_resize {
    u32 channels;
    u32 in_h;
    u32 in_w;
    u32 out_h;
    u32 out_w;
};
static_assert(sizeof(splat_args_resize) == 20, "splat_args_resize layout");

struct splat_args_merge {
    u32 h;
    u32 w;
    u32 channels;
};
static_assert(sizeof(splat_args_merge) == 12, "splat_args_merge layout");

struct splat_args_window {
    u32 h;
    u32 w;
    u32 hp;
    u32 wp;
    u32 window;
    u32 shift;
    u32 heads;
    u32 head_dim;
    float scale;
};
static_assert(sizeof(splat_args_window) == 36, "splat_args_window layout");

struct splat_args_deform {
    u32 channels;
    u32 in_h;
    u32 in_w;
    u32 out_h;
    u32 out_w;
    u32 k_h;
    u32 k_w;
    u32 stride;
    u32 pad;
    u32 row0;
    u32 rows;
};
static_assert(sizeof(splat_args_deform) == 44, "splat_args_deform layout");

extern "C" __global__ void kernel_splat_relu_f32(
        const splat_args_unary args, float* out, const float* in) {
    const u32 i = blockIdx.x * SPLAT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    out[i] = fmaxf(in[i], 0.0f);
}

extern "C" __global__ void kernel_splat_sigmoid_f32(
        const splat_args_unary args, float* out, const float* in) {
    const u32 i = blockIdx.x * SPLAT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    if (x >= 0.0f) {
        out[i] = 1.0f / (1.0f + expf(-x));
    } else {
        const float e = expf(x);
        out[i] = e / (1.0f + e);
    }
}

/* `1 + erf(x)`: the Metal twin's Chebyshev fit, so both backends answer the
 * same numbers (the CUDA library's own `erff` would differ in the last bits). */
__device__ __forceinline__ float splat_one_plus_erf(float x) {
    const float z = fabsf(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float poly = -z * z - 1.26551223f
            + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f
            + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f
            + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    const float erfc_z = t * expf(poly);
    return x >= 0.0f ? 2.0f - erfc_z : erfc_z;
}

extern "C" __global__ void kernel_splat_gelu_erf_f32(
        const splat_args_unary args, float* out, const float* in) {
    const u32 i = blockIdx.x * SPLAT_BLOCK + threadIdx.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    out[i] = 0.5f * x * splat_one_plus_erf(x * 0.70710678118654752f);
}

extern "C" __global__ void kernel_splat_resize_bilinear_f32(
        const splat_args_resize args, float* out, const float* in) {
    const u32 x = blockIdx.x * SPLAT_BLOCK + threadIdx.x;
    const u32 y = blockIdx.y;
    const u32 c = blockIdx.z;
    if (x >= args.out_w || y >= args.out_h || c >= args.channels) {
        return;
    }
    const float sy = args.out_h > 1u ? (float)(args.in_h - 1u) / (float)(args.out_h - 1u) : 0.0f;
    const float sx = args.out_w > 1u ? (float)(args.in_w - 1u) / (float)(args.out_w - 1u) : 0.0f;
    const float fy = sy * (float)y;
    const float fx = sx * (float)x;
    const u32 y0 = min((u32)fy, args.in_h - 1u);
    const u32 x0 = min((u32)fx, args.in_w - 1u);
    const u32 y1 = y0 + (y0 < args.in_h - 1u ? 1u : 0u);
    const u32 x1 = x0 + (x0 < args.in_w - 1u ? 1u : 0u);
    const float ly = fy - (float)y0;
    const float lx = fx - (float)x0;
    const float* plane = in + (u64)c * args.in_h * args.in_w;
    const float top = (1.0f - lx) * plane[y0 * args.in_w + x0] + lx * plane[y0 * args.in_w + x1];
    const float bottom = (1.0f - lx) * plane[y1 * args.in_w + x0] + lx * plane[y1 * args.in_w + x1];
    out[((u64)c * args.out_h + y) * args.out_w + x] = (1.0f - ly) * top + ly * bottom;
}

extern "C" __global__ void kernel_splat_merge2x2_f32(
        const splat_args_merge args, float* out, const float* in) {
    const u32 col = blockIdx.x * SPLAT_BLOCK + threadIdx.x;
    const u32 token = blockIdx.y;
    const u32 ow = args.w / 2u;
    if (col >= 4u * args.channels || token >= (args.h / 2u) * ow) {
        return;
    }
    const u32 i = token / ow;
    const u32 j = token - i * ow;
    const u32 part = col / args.channels;
    const u32 ch = col - part * args.channels;
    const u32 y = 2u * i + (part & 1u);
    const u32 x = 2u * j + (part >> 1u);
    out[(u64)token * 4u * args.channels + col] = in[((u64)y * args.w + x) * args.channels + ch];
}

__device__ __forceinline__ u32 splat_region(u32 v, u32 n, u32 window, u32 shift) {
    return v < n - window ? 0u : (v < n - shift ? 1u : 2u);
}

extern "C" __global__ void kernel_splat_window_attn_f32(
        const splat_args_window args, float* out, const float* q, const float* k,
        const float* v, const float* k_pad, const float* v_pad, const float* table) {
    const u32 ws = args.window;
    const u32 tokens = ws * ws;
    const u32 per_row = args.wp / ws;
    const u32 win = blockIdx.x;
    const u32 head = blockIdx.y;
    if (win >= (args.hp / ws) * per_row || head >= args.heads || tokens > SPLAT_MAX_WINDOW_TOKENS) {
        return;
    }
    const u32 wy = win / per_row;
    const u32 wx = win - wy * per_row;
    const u32 width = args.heads * args.head_dim;
    const u32 span = 2u * ws - 1u;
    float scores[SPLAT_MAX_WINDOW_TOKENS];
    for (u32 qi = threadIdx.x; qi < tokens; qi += SPLAT_BLOCK) {
        const u32 ly = qi / ws;
        const u32 lx = qi - ly * ws;
        const u32 ry = wy * ws + ly;
        const u32 rx = wx * ws + lx;
        const u32 oy = (ry + args.shift) % args.hp;
        const u32 ox = (rx + args.shift) % args.wp;
        if (oy >= args.h || ox >= args.w) {
            continue;
        }
        const float* qrow = q + ((u64)oy * args.w + ox) * width + head * args.head_dim;
        const u32 q_region = 3u * splat_region(ry, args.hp, ws, args.shift)
                + splat_region(rx, args.wp, ws, args.shift);
        float max_score = -3.402823466e38f;
        for (u32 kj = 0u; kj < tokens; kj++) {
            const u32 ky = kj / ws;
            const u32 kx = kj - ky * ws;
            const u32 kry = wy * ws + ky;
            const u32 krx = wx * ws + kx;
            const u32 koy = (kry + args.shift) % args.hp;
            const u32 kox = (krx + args.shift) % args.wp;
            const float* krow = (koy < args.h && kox < args.w)
                    ? k + ((u64)koy * args.w + kox) * width + head * args.head_dim
                    : k_pad + head * args.head_dim;
            float dot = 0.0f;
            for (u32 d = 0u; d < args.head_dim; d++) {
                dot += (qrow[d] * args.scale) * krow[d];
            }
            const u32 rel = (ly + ws - 1u - ky) * span + (lx + ws - 1u - kx);
            float score = dot + table[rel * args.heads + head];
            if (args.shift > 0u) {
                const u32 k_region = 3u * splat_region(kry, args.hp, ws, args.shift)
                        + splat_region(krx, args.wp, ws, args.shift);
                if (k_region != q_region) {
                    score -= 100.0f;
                }
            }
            scores[kj] = score;
            max_score = fmaxf(max_score, score);
        }
        float sum = 0.0f;
        for (u32 kj = 0u; kj < tokens; kj++) {
            scores[kj] = expf(scores[kj] - max_score);
            sum += scores[kj];
        }
        const float inv = 1.0f / sum;
        float* orow = out + ((u64)oy * args.w + ox) * width + head * args.head_dim;
        for (u32 d = 0u; d < args.head_dim; d++) {
            float acc = 0.0f;
            for (u32 kj = 0u; kj < tokens; kj++) {
                const u32 ky = kj / ws;
                const u32 kx = kj - ky * ws;
                const u32 koy = (wy * ws + ky + args.shift) % args.hp;
                const u32 kox = (wx * ws + kx + args.shift) % args.wp;
                const float value = (koy < args.h && kox < args.w)
                        ? v[((u64)koy * args.w + kox) * width + head * args.head_dim + d]
                        : v_pad[head * args.head_dim + d];
                acc += scores[kj] * value;
            }
            orow[d] = acc * inv;
        }
    }
}

extern "C" __global__ void kernel_splat_deform_im2col_f32(
        const splat_args_deform args, float* cols, const float* in, const float* offset,
        const float* mask) {
    const u32 taps = args.k_h * args.k_w;
    const u32 width = args.channels * taps;
    const u32 col = blockIdx.x * SPLAT_BLOCK + threadIdx.x;
    const u32 r = blockIdx.y;
    if (col >= width || r >= args.rows) {
        return;
    }
    const u32 plane = args.out_h * args.out_w;
    const u32 p = args.row0 + r;
    const u32 oh = p / args.out_w;
    const u32 ow = p - oh * args.out_w;
    const u32 c = col / taps;
    const u32 t = col - c * taps;
    const u32 ti = t / args.k_w;
    const u32 tj = t - ti * args.k_w;
    const float dy = offset[(u64)(2u * t) * plane + p];
    const float dx = offset[(u64)(2u * t + 1u) * plane + p];
    const float m = mask[(u64)t * plane + p];
    const float y = (float)((int)(oh * args.stride) - (int)args.pad + (int)ti) + dy;
    const float x = (float)((int)(ow * args.stride) - (int)args.pad + (int)tj) + dx;
    float value = 0.0f;
    const float fh = (float)args.in_h;
    const float fw = (float)args.in_w;
    if (y > -1.0f && y < fh && x > -1.0f && x < fw) {
        const float y_low = floorf(y);
        const float x_low = floorf(x);
        const int h_low = (int)y_low;
        const int w_low = (int)x_low;
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;
        const float lh = y - y_low;
        const float lw = x - x_low;
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;
        const float* src = in + (u64)c * args.in_h * args.in_w;
        const int ih = (int)args.in_h;
        const int iw = (int)args.in_w;
        const float v1 = (h_low >= 0 && w_low >= 0) ? src[h_low * iw + w_low] : 0.0f;
        const float v2 = (h_low >= 0 && w_high <= iw - 1) ? src[h_low * iw + w_high] : 0.0f;
        const float v3 = (h_high <= ih - 1 && w_low >= 0) ? src[h_high * iw + w_low] : 0.0f;
        const float v4 = (h_high <= ih - 1 && w_high <= iw - 1) ? src[h_high * iw + w_high] : 0.0f;
        value = hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
    }
    cols[(u64)r * width + col] = m * value;
}
