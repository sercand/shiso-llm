// The 3D splat family's own kernel set, module `splat`.
//
// The family's transformers run on the diffusion set; these are the
// operations its background remover needs that the diffusion set does not
// carry: Swin's shifted-window attention, the modulated deformable
// convolution's sampling, bilinear resampling with aligned corners, Swin's
// 2x2 patch merge, and three elementwise activations (ReLU, the logistic
// sigmoid, and GELU in its exact erf form).
//
// Every argument block's field order is the binding contract
// `crates/shiso-models/src/idol/splat_ops.rs` mirrors with `#[repr(C)]` Pods;
// the sizes are asserted here and there. The CUDA twin is
// `kernels/cuda/splat/splat.cu`; both are graded by the cases in
// `crates/shiso-kerneltest/src/splat/`.

// Threads per threadgroup, as the driver launches every entry here.
#define SPLAT_BLOCK 64u
// The largest window the attention keeps a score row for (16 x 16 tokens).
#define SPLAT_MAX_WINDOW_TOKENS 256u

struct splat_args_unary {
    uint count;
};
static_assert(sizeof(splat_args_unary) == 4, "splat_args_unary layout");

struct splat_args_resize {
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
};
static_assert(sizeof(splat_args_resize) == 20, "splat_args_resize layout");

struct splat_args_merge {
    uint h;
    uint w;
    uint channels;
};
static_assert(sizeof(splat_args_merge) == 12, "splat_args_merge layout");

struct splat_args_window {
    uint h;
    uint w;
    uint hp;
    uint wp;
    uint window;
    uint shift;
    uint heads;
    uint head_dim;
    float scale;
};
static_assert(sizeof(splat_args_window) == 36, "splat_args_window layout");

struct splat_args_deform {
    uint channels;
    uint in_h;
    uint in_w;
    uint out_h;
    uint out_w;
    uint k_h;
    uint k_w;
    uint stride;
    uint pad;
    uint row0;
    uint rows;
};
static_assert(sizeof(splat_args_deform) == 44, "splat_args_deform layout");

// `out[i] = max(in[i], 0)`.
kernel void kernel_splat_relu_f32(
        constant splat_args_unary & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * SPLAT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    out[i] = max(in[i], 0.0f);
}

// `out[i] = 1 / (1 + exp(-in[i]))`, in the form that does not overflow.
kernel void kernel_splat_sigmoid_f32(
        constant splat_args_unary & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * SPLAT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    if (x >= 0.0f) {
        out[i] = 1.0f / (1.0f + precise::exp(-x));
    } else {
        const float e = precise::exp(x);
        out[i] = e / (1.0f + e);
    }
}

// `1 + erf(x)` from the complementary function's Chebyshev fit (fractional
// error below 1.2e-7 everywhere). Taken as `erfc(-x)` on the negative side
// rather than `1 + erf(x)`, which would cancel exactly where GELU is near
// zero. The language has no `erf`, and GELU's exact form is what the
// background remover was trained with.
static inline float splat_one_plus_erf(float x) {
    const float z = fabs(x);
    const float t = 1.0f / (1.0f + 0.5f * z);
    const float poly = -z * z - 1.26551223f
            + t * (1.00002368f + t * (0.37409196f + t * (0.09678418f
            + t * (-0.18628806f + t * (0.27886807f + t * (-1.13520398f
            + t * (1.48851587f + t * (-0.82215223f + t * 0.17087277f))))))));
    const float erfc_z = t * precise::exp(poly);
    return x >= 0.0f ? 2.0f - erfc_z : erfc_z;
}

// `out[i] = 0.5 x (1 + erf(x / sqrt(2)))`: GELU in its exact form.
kernel void kernel_splat_gelu_erf_f32(
        constant splat_args_unary & args [[buffer(0)]],
        device float *out                  [[buffer(1)]],
        device const float *in             [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * SPLAT_BLOCK + tpitg.x;
    if (i >= args.count) {
        return;
    }
    const float x = in[i];
    out[i] = 0.5f * x * splat_one_plus_erf(x * 0.70710678118654752f);
}

// Bilinear resampling of a `(channels, in_h, in_w)` field to
// `(channels, out_h, out_w)` with aligned corners: output pixel `y` reads
// source row `y * (in_h - 1) / (out_h - 1)` (row 0 when `out_h` is 1), the
// reference's `align_corners=True` arithmetic. One thread per output value.
kernel void kernel_splat_resize_bilinear_f32(
        constant splat_args_resize & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *in               [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint x = tgpig.x * SPLAT_BLOCK + tpitg.x;
    const uint y = tgpig.y;
    const uint c = tgpig.z;
    if (x >= args.out_w || y >= args.out_h || c >= args.channels) {
        return;
    }
    const float sy = args.out_h > 1u ? float(args.in_h - 1u) / float(args.out_h - 1u) : 0.0f;
    const float sx = args.out_w > 1u ? float(args.in_w - 1u) / float(args.out_w - 1u) : 0.0f;
    const float fy = sy * float(y);
    const float fx = sx * float(x);
    const uint y0 = min(uint(fy), args.in_h - 1u);
    const uint x0 = min(uint(fx), args.in_w - 1u);
    const uint y1 = y0 + (y0 < args.in_h - 1u ? 1u : 0u);
    const uint x1 = x0 + (x0 < args.in_w - 1u ? 1u : 0u);
    const float ly = fy - float(y0);
    const float lx = fx - float(x0);
    device const float *plane = in + (uint64_t)c * args.in_h * args.in_w;
    const float top = (1.0f - lx) * plane[y0 * args.in_w + x0] + lx * plane[y0 * args.in_w + x1];
    const float bottom = (1.0f - lx) * plane[y1 * args.in_w + x0] + lx * plane[y1 * args.in_w + x1];
    out[((uint64_t)c * args.out_h + y) * args.out_w + x] = (1.0f - ly) * top + ly * bottom;
}

// Swin's patch merge over `(h * w, channels)` token rows: output token
// `(i, j)` is the concatenation of input tokens `(2i, 2j)`, `(2i + 1, 2j)`,
// `(2i, 2j + 1)` and `(2i + 1, 2j + 1)`, in that order. One thread per
// output value.
kernel void kernel_splat_merge2x2_f32(
        constant splat_args_merge & args [[buffer(0)]],
        device float *out                   [[buffer(1)]],
        device const float *in              [[buffer(2)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint col = tgpig.x * SPLAT_BLOCK + tpitg.x;
    const uint token = tgpig.y;
    const uint ow = args.w / 2u;
    if (col >= 4u * args.channels || token >= (args.h / 2u) * ow) {
        return;
    }
    const uint i = token / ow;
    const uint j = token - i * ow;
    const uint part = col / args.channels;
    const uint ch = col - part * args.channels;
    const uint y = 2u * i + (part & 1u);
    const uint x = 2u * j + (part >> 1u);
    out[(uint64_t)token * 4u * args.channels + col] = in[((uint64_t)y * args.w + x) * args.channels + ch];
}

// The region of a rolled coordinate for the shifted window's mask: the
// reference labels `[0, n - window)`, `[n - window, n - shift)` and
// `[n - shift, n)` as three regions per axis.
static inline uint splat_region(uint v, uint n, uint window, uint shift) {
    return v < n - window ? 0u : (v < n - shift ? 1u : 2u);
}

// Swin's shifted-window self attention over an `h x w` token grid, padded
// to `hp x wp` (multiples of `window`) and rolled by `shift`.
///
// `q`, `k`, `v` and `out` are `(h * w, heads * head_dim)` rows of the
// unpadded grid. A padded position's key and value are `k_pad` and `v_pad`
// (the projection of the zero the reference pads with, which is its bias);
// a padded position's query is computed by nobody, as the reference
// discards it. Window `(wy, wx)`'s token `(ly, lx)` sits at rolled
// coordinate `(wy * window + ly, wx * window + lx)`, which is unrolled
// coordinate `((ry + shift) % hp, (rx + shift) % wp)`; the answer is written
// back there, which is the reference's partition, roll and reverse in one
// index. The score adds the relative position bias `table[((ly - ky) +
// window - 1) * (2 window - 1) + (lx - kx) + window - 1][head]` and, when
// shifted, -100 between tokens of different regions.
///
// Grid `(windows, heads)` of 64 threads, each thread taking every 64th
// query of the window.
kernel void kernel_splat_window_attn_f32(
        constant splat_args_window & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *q                [[buffer(2)]],
        device const float *k                [[buffer(3)]],
        device const float *v                [[buffer(4)]],
        device const float *k_pad            [[buffer(5)]],
        device const float *v_pad            [[buffer(6)]],
        device const float *table            [[buffer(7)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint ws = args.window;
    const uint tokens = ws * ws;
    const uint per_row = args.wp / ws;
    const uint win = tgpig.x;
    const uint head = tgpig.y;
    if (win >= (args.hp / ws) * per_row || head >= args.heads || tokens > SPLAT_MAX_WINDOW_TOKENS) {
        return;
    }
    const uint wy = win / per_row;
    const uint wx = win - wy * per_row;
    const uint width = args.heads * args.head_dim;
    const uint span = 2u * ws - 1u;
    float scores[SPLAT_MAX_WINDOW_TOKENS];
    for (uint qi = tpitg.x; qi < tokens; qi += SPLAT_BLOCK) {
        const uint ly = qi / ws;
        const uint lx = qi - ly * ws;
        const uint ry = wy * ws + ly;
        const uint rx = wx * ws + lx;
        const uint oy = (ry + args.shift) % args.hp;
        const uint ox = (rx + args.shift) % args.wp;
        if (oy >= args.h || ox >= args.w) {
            continue;
        }
        device const float *qrow = q + ((uint64_t)oy * args.w + ox) * width + head * args.head_dim;
        const uint q_region = 3u * splat_region(ry, args.hp, ws, args.shift)
                + splat_region(rx, args.wp, ws, args.shift);
        float max_score = -INFINITY;
        for (uint kj = 0u; kj < tokens; kj++) {
            const uint ky = kj / ws;
            const uint kx = kj - ky * ws;
            const uint kry = wy * ws + ky;
            const uint krx = wx * ws + kx;
            const uint koy = (kry + args.shift) % args.hp;
            const uint kox = (krx + args.shift) % args.wp;
            device const float *krow = (koy < args.h && kox < args.w)
                    ? k + ((uint64_t)koy * args.w + kox) * width + head * args.head_dim
                    : k_pad + head * args.head_dim;
            float dot = 0.0f;
            for (uint d = 0u; d < args.head_dim; d++) {
                dot += (qrow[d] * args.scale) * krow[d];
            }
            const uint rel = (ly + ws - 1u - ky) * span + (lx + ws - 1u - kx);
            float score = dot + table[rel * args.heads + head];
            if (args.shift > 0u) {
                const uint k_region = 3u * splat_region(kry, args.hp, ws, args.shift)
                        + splat_region(krx, args.wp, ws, args.shift);
                if (k_region != q_region) {
                    score -= 100.0f;
                }
            }
            scores[kj] = score;
            max_score = max(max_score, score);
        }
        float sum = 0.0f;
        for (uint kj = 0u; kj < tokens; kj++) {
            scores[kj] = precise::exp(scores[kj] - max_score);
            sum += scores[kj];
        }
        const float inv = 1.0f / sum;
        device float *orow = out + ((uint64_t)oy * args.w + ox) * width + head * args.head_dim;
        for (uint d = 0u; d < args.head_dim; d++) {
            float acc = 0.0f;
            for (uint kj = 0u; kj < tokens; kj++) {
                const uint ky = kj / ws;
                const uint kx = kj - ky * ws;
                const uint koy = (wy * ws + ky + args.shift) % args.hp;
                const uint kox = (wx * ws + kx + args.shift) % args.wp;
                const float value = (koy < args.h && kox < args.w)
                        ? v[((uint64_t)koy * args.w + kox) * width + head * args.head_dim + d]
                        : v_pad[head * args.head_dim + d];
                acc += scores[kj] * value;
            }
            orow[d] = acc * inv;
        }
    }
}

// The modulated deformable convolution's columns: for output pixels
// `row0 .. row0 + rows` (row-major over `out_h x out_w`) and every
// `(channel, tap)`, `cols[r][c * taps + t] = mask[t][p] * sample(c, y, x)`
// where tap `t = (i, j)` samples at `y = oh * stride - pad + i + dy`,
// `x = ow * stride - pad + j + dx`, the offsets `(dy, dx)` read from planes
// `2t` and `2t + 1` of `offset` (the reference's `(y, x)` order), and
// `sample` is bilinear with every corner outside the field reading zero and
// nothing read at all past one pixel outside it. A GEMM against the weight's
// `(out, channels * taps)` rows then finishes the convolution.
///
// Grid `(ceil(channels * taps / 64), rows)` of 64 threads, one per value.
kernel void kernel_splat_deform_im2col_f32(
        constant splat_args_deform & args [[buffer(0)]],
        device float *cols                   [[buffer(1)]],
        device const float *in               [[buffer(2)]],
        device const float *offset           [[buffer(3)]],
        device const float *mask             [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint taps = args.k_h * args.k_w;
    const uint width = args.channels * taps;
    const uint col = tgpig.x * SPLAT_BLOCK + tpitg.x;
    const uint r = tgpig.y;
    if (col >= width || r >= args.rows) {
        return;
    }
    const uint plane = args.out_h * args.out_w;
    const uint p = args.row0 + r;
    const uint oh = p / args.out_w;
    const uint ow = p - oh * args.out_w;
    const uint c = col / taps;
    const uint t = col - c * taps;
    const uint ti = t / args.k_w;
    const uint tj = t - ti * args.k_w;
    const float dy = offset[(uint64_t)(2u * t) * plane + p];
    const float dx = offset[(uint64_t)(2u * t + 1u) * plane + p];
    const float m = mask[(uint64_t)t * plane + p];
    const float y = float(int(oh * args.stride) - int(args.pad) + int(ti)) + dy;
    const float x = float(int(ow * args.stride) - int(args.pad) + int(tj)) + dx;
    float value = 0.0f;
    const float fh = float(args.in_h);
    const float fw = float(args.in_w);
    if (y > -1.0f && y < fh && x > -1.0f && x < fw) {
        const float y_low = floor(y);
        const float x_low = floor(x);
        const int h_low = int(y_low);
        const int w_low = int(x_low);
        const int h_high = h_low + 1;
        const int w_high = w_low + 1;
        const float lh = y - y_low;
        const float lw = x - x_low;
        const float hh = 1.0f - lh;
        const float hw = 1.0f - lw;
        device const float *src = in + (uint64_t)c * args.in_h * args.in_w;
        const int ih = int(args.in_h);
        const int iw = int(args.in_w);
        const float v1 = (h_low >= 0 && w_low >= 0) ? src[h_low * iw + w_low] : 0.0f;
        const float v2 = (h_low >= 0 && w_high <= iw - 1) ? src[h_low * iw + w_high] : 0.0f;
        const float v3 = (h_high <= ih - 1 && w_low >= 0) ? src[h_high * iw + w_low] : 0.0f;
        const float v4 = (h_high <= ih - 1 && w_high <= iw - 1) ? src[h_high * iw + w_high] : 0.0f;
        value = hh * hw * v1 + hh * lw * v2 + lh * hw * v3 + lh * lw * v4;
    }
    cols[(uint64_t)r * width + col] = m * value;
}
