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
// The fused attention's tile (`kernel_splat_attn_f32`): a fixed 64-wide head,
// 64 query rows per threadgroup of eight simdgroups (eight rows each), keys
// streamed 32 at a time.
#define SPLAT_ATTN_HEAD 64u
#define SPLAT_ATTN_SG 8u
#define SPLAT_ATTN_QROWS (8u * SPLAT_ATTN_SG)
#define SPLAT_ATTN_THREADS (32u * SPLAT_ATTN_SG)
#define SPLAT_ATTN_KEYS 32u
#define SPLAT_UNROLL _Pragma("clang loop unroll(full)")

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

struct splat_args_attn {
    uint queries;
    uint keys;
    uint heads;
    uint width;
    float scale;
    // One key that stands for many identical rows: its score gets
    // `heavy_bias` (the log of the count) added. `heavy_key >= keys` for none.
    uint heavy_key;
    float heavy_bias;
};
static_assert(sizeof(splat_args_attn) == 28, "splat_args_attn layout");

struct splat_args_rope {
    uint row0;
    uint rows;
    uint heads;
    float eps;
};
static_assert(sizeof(splat_args_rope) == 16, "splat_args_rope layout");

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

// Attention over 64-wide heads in one pass over the keys, with an online
// softmax on simdgroup matrices: for every head `h`,
// `out[r][h*64 + d] = sum_j softmax_j(scale * q[r][h] . k[j][h]) v[j][h*64 + d]`,
// q `(queries, width)`, k and v `(keys, width)`, out `(queries, width)`, all
// row-major with `width = heads * 64` floats per row. Queries and keys may
// differ in number (the decoders attend from anchors to latent tokens).
//
// A threadgroup takes 64 query rows of one head, eight per simdgroup, held
// in registers as 8x8 fragments pre-multiplied by `scale`, and streams the
// keys 32 at a time through threadgroup memory, the next block loading into
// registers while the current one is multiplied. A lane holds two columns of
// one row of each score fragment; the four lanes sharing a row (lane bits 0
// and 3) fold the running maximum and denominator with two shuffles, and
// the value accumulators, laid out the same way, are rescaled lane by lane.
//
// Everything stays f32: q, k, v, the probabilities and every product. The
// diffusion set's fused kernel stages them as half, which moved the image
// tower's tokens by 7.5e-3 over its 32 layers against a 5e-4 bound. K and V
// together take 17.4 KB of threadgroup memory, so the tile is single-buffered
// (a second copy would pass the 32 KB limit) and the prefetch rides in
// registers instead.
//
// Keys past the end score -inf and are never read; query rows past the end
// load zero and are never written.
//
// `heavy_key` is one key standing for `n` identical rows of the sequence:
// `n` equal keys with equal values contribute `n exp(s) v` to the sum, which
// is one key scoring `s + log(n)`. The denoiser's unconditional pass has 4101
// such rows (a zero conditioning refines to one row repeated), and attends
// 8194 rows instead of 12294 this way.
//
// The exponentials are the fast form: every argument is at most zero (a score
// less its running maximum), where it agrees with the precise one to the
// cases' 1e-5, and -inf still answers zero. `precise::exp` cost a third of
// the launch.
//
// Measured on an M1 Max at the denoiser's joint attention (12294 rows, 16
// heads), cool: 130 ms per launch (4.75 TFLOP/s), against 193 ms with
// `precise::exp` and 800 ms for the same attention as transposes, score
// GEMMs, softmax and P.V GEMMs. Two row blocks per simdgroup (the K fragment
// reused across 16 rows) spilled: 230 ms. On an M4 the same launch is 207 ms
// (2.99 TFLOP/s); with every fragment a `simdgroup_half8x8` it was 199 ms and
// with only K and V staged as half 216 ms, so the f32 form gives up nothing
// worth its rounding.
//
// Grid `(ceil(queries / 64), heads)` of 256 threads.
kernel void kernel_splat_attn_f32(
        constant splat_args_attn & args [[buffer(0)]],
        device float *out                [[buffer(1)]],
        device const float *q            [[buffer(2)]],
        device const float *k            [[buffer(3)]],
        device const float *v            [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tid [[thread_index_in_threadgroup]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint HD = SPLAT_ATTN_HEAD;
    const uint ND = HD / 8u;
    const uint KB = SPLAT_ATTN_KEYS;
    const uint NB = KB / 8u;
    const uint NT = SPLAT_ATTN_THREADS;
    // Each staged row is padded by four floats so the transposed K loads do
    // not land on one bank.
    const uint LDK = HD + 4u;
    threadgroup float kv[2u * SPLAT_ATTN_KEYS * (SPLAT_ATTN_HEAD + 4u)];

    const uint h = tgpig.y;
    if (h >= args.heads) {
        return;
    }
    const uint keys = args.keys;
    const uint width = args.width;

    // The lane's place in every 8x8 fragment: columns `fn, fn + 1` of row `fr`.
    const ushort fn = ((lane / 4u) & 2u) * 2u + (lane % 2u) * 2u;
    const ushort fr = ((lane / 4u) & 4u) + ((lane / 2u) % 4u);
    const uint qrow = tgpig.x * SPLAT_ATTN_QROWS + sg * 8u + fr;
    const bool live = qrow < args.queries;

    simdgroup_float8x8 qf[ND];
    simdgroup_float8x8 of[ND];
    device const float *qp = q + (uint64_t) (live ? qrow : 0u) * width + h * HD;
    SPLAT_UNROLL for (uint i = 0; i < ND; ++i) {
        const float2 qv = live ? float2(qp[8u * i + fn], qp[8u * i + fn + 1u]) * args.scale
                               : float2(0.0f);
        qf[i].thread_elements()[0] = qv.x;
        qf[i].thread_elements()[1] = qv.y;
        of[i] = simdgroup_float8x8(0.0f);
    }

    float run_max = -INFINITY;
    float run_sum = 0.0f;
    threadgroup float *sk = kv;
    threadgroup float *sv = kv + KB * LDK;
    // Four consecutive head values per load; a key past the end stages as
    // zero and is masked below.
    const uint HD4 = HD / 4u;
    const uint SL = KB * HD4 / NT;
    float4 rk[SL];
    float4 rv[SL];
    auto fetch = [&](uint s0) {
        SPLAT_UNROLL for (uint c = 0; c < SL; ++c) {
            const uint e = tid + c * NT;
            const uint j = e / HD4;
            const uint d = (e - j * HD4) * 4u;
            const uint s = s0 + j;
            rk[c] = float4(0.0f);
            rv[c] = float4(0.0f);
            if (s < keys) {
                const uint64_t o = (uint64_t) s * width + h * HD + d;
                rk[c] = float4(*((device const packed_float4 *) (k + o)));
                rv[c] = float4(*((device const packed_float4 *) (v + o)));
            }
        }
    };
    fetch(0);
    for (uint s0 = 0; s0 < keys; s0 += KB) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        SPLAT_UNROLL for (uint c = 0; c < SL; ++c) {
            const uint e = tid + c * NT;
            const uint j = e / HD4;
            const uint d = (e - j * HD4) * 4u;
            *((threadgroup float4 *) (sk + j * LDK + d)) = rk[c];
            *((threadgroup float4 *) (sv + j * LDK + d)) = rv[c];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (s0 + KB < keys) {
            fetch(s0 + KB);
        }

        // Scores: S (8 x KB) = Q (8 x 64) . K^T, one fragment per 8 keys.
        simdgroup_float8x8 sf[NB];
        SPLAT_UNROLL for (uint b = 0; b < NB; ++b) {
            sf[b] = simdgroup_float8x8(0.0f);
        }
        SPLAT_UNROLL for (uint i = 0; i < ND; ++i) {
            SPLAT_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_float8x8 kt;
                // kt(r, c) = K[8b + c][8i + r]: the transposed load.
                simdgroup_load(kt, sk + 8u * b * LDK + 8u * i, LDK, ulong2(0, 0), true);
                simdgroup_multiply_accumulate(sf[b], qf[i], kt, sf[b]);
            }
        }

        float block_max = -INFINITY;
        SPLAT_UNROLL for (uint b = 0; b < NB; ++b) {
            SPLAT_UNROLL for (uint c = 0; c < 2u; ++c) {
                const uint key = s0 + 8u * b + fn + c;
                if (key >= keys) {
                    sf[b].thread_elements()[c] = -INFINITY;
                } else if (key == args.heavy_key) {
                    sf[b].thread_elements()[c] += args.heavy_bias;
                }
                block_max = max(block_max, sf[b].thread_elements()[c]);
            }
        }
        block_max = max(block_max, simd_shuffle_xor(block_max, 1));
        block_max = max(block_max, simd_shuffle_xor(block_max, 8));
        const float new_max = max(run_max, block_max);
        const float alpha = exp(run_max - new_max);
        float block_sum = 0.0f;
        SPLAT_UNROLL for (uint b = 0; b < NB; ++b) {
            SPLAT_UNROLL for (uint c = 0; c < 2u; ++c) {
                const float p = exp(sf[b].thread_elements()[c] - new_max);
                sf[b].thread_elements()[c] = p;
                block_sum += p;
            }
        }
        block_sum += simd_shuffle_xor(block_sum, 1);
        block_sum += simd_shuffle_xor(block_sum, 8);
        run_sum = run_sum * alpha + block_sum;
        run_max = new_max;

        SPLAT_UNROLL for (uint i = 0; i < ND; ++i) {
            of[i].thread_elements()[0] *= alpha;
            of[i].thread_elements()[1] *= alpha;
            SPLAT_UNROLL for (uint b = 0; b < NB; ++b) {
                simdgroup_float8x8 vf;
                simdgroup_load(vf, sv + 8u * b * LDK + 8u * i, LDK);
                simdgroup_multiply_accumulate(of[i], sf[b], vf, of[i]);
            }
        }
    }

    if (!live) {
        return;
    }
    const float inv = 1.0f / run_sum;
    device float *op = out + (uint64_t) qrow * width + h * HD;
    SPLAT_UNROLL for (uint i = 0; i < ND; ++i) {
        op[8u * i + fn] = of[i].thread_elements()[0] * inv;
        op[8u * i + fn + 1u] = of[i].thread_elements()[1] * inv;
    }
}

// One pair of one head's row, turned by `(c, s)` and scaled by the head's
// RMS norm: the lane's two values of `x`, in place. Called by every lane of
// a simdgroup at once; the sum of squares folds across the 32 lanes in a
// fixed order.
static inline void splat_rope_norm_pair(device float *x, device const float *gamma,
                                        float c, float s, float eps) {
    const float a = x[0];
    const float b = x[1];
    const float ra = a * c - b * s;
    const float rb = b * c + a * s;
    float ss = ra * ra + rb * rb;
    ss += simd_shuffle_xor(ss, 1);
    ss += simd_shuffle_xor(ss, 2);
    ss += simd_shuffle_xor(ss, 4);
    ss += simd_shuffle_xor(ss, 8);
    ss += simd_shuffle_xor(ss, 16);
    const float inv = 1.0f / sqrt(ss / float(SPLAT_ATTN_HEAD) + eps);
    x[0] = ra * inv * gamma[0];
    x[1] = rb * inv * gamma[1];
}

// The denoiser's rotary turn and per-head norm of its queries and keys, in
// place, from its predicted positions: for row `r`, head `h` (64 wide) and
// pair `j < 32`, with `angle = offsets[(r * heads + h) * 3 + axis[j]] * rates[j]`,
// the pair `(a, b)` at columns `(2j, 2j + 1)` of the head becomes
// `(a cos - b sin, b cos + a sin)`, and the turned head `x` then
// `x / sqrt(mean(x^2) + eps) * gamma[h]`. `q` and `k` are
// `(rows, heads * 64)` row-major and turn by the same angles; `gamma_q` and
// `gamma_k` are `(heads, 64)`.
//
// One launch in place of the tables this used to write (`cos - 1` and `sin`,
// two buffers the size of `q`), the two extra projections that formed the
// turned partner of `q` and `k`, four gated adds, two RMS norms and two more
// gated adds for gamma: measured on an M1 Max at the joint shape, 48 GEMMs
// of 7 ms and 4 ms of elementwise launches a pass, for 0.6 ms a launch here.
//
// The angles reach the hundreds, where the fast sine is off by far more than
// an f32 `sin_cos`, hence `precise::`. `rates` are the pair frequencies
// already multiplied by pi on the host: the library is built with fast math,
// which may reassociate a three-way product, and one multiply leaves the
// compiler no order to choose.
//
// Grid `(ceil(heads / 2), rows in this launch)` of 64 threads: a simdgroup
// is one head, a lane one pair. This launch's first row is `row0`, so the
// driver can split a long sequence across launches.
kernel void kernel_splat_rope_norm_f32(
        constant splat_args_rope & args [[buffer(0)]],
        device float *q                    [[buffer(1)]],
        device float *k                    [[buffer(2)]],
        device const float *offsets        [[buffer(3)]],
        device const float *rates          [[buffer(4)]],
        device const uint *axis            [[buffer(5)]],
        device const float *gamma_q        [[buffer(6)]],
        device const float *gamma_k        [[buffer(7)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort lane [[thread_index_in_simdgroup]],
        ushort sg [[simdgroup_index_in_threadgroup]]) {
    const uint HD = SPLAT_ATTN_HEAD;
    const uint head = tgpig.x * 2u + sg;
    const uint row = args.row0 + tgpig.y;
    if (head >= args.heads || row >= args.rows) {
        return;
    }
    const float angle = offsets[((uint64_t) row * args.heads + head) * 3u + axis[lane]] * rates[lane];
    const float c = precise::cos(angle);
    const float s = precise::sin(angle);
    const uint g = head * HD + 2u * lane;
    const uint64_t at = (uint64_t) row * (args.heads * HD) + g;
    splat_rope_norm_pair(q + at, gamma_q + g, c, s, args.eps);
    splat_rope_norm_pair(k + at, gamma_k + g, c, s, args.eps);
}
