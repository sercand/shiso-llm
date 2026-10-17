/* Convolution entries of the `diffusion` module: the per-pixel NCHW f32
 * convolution, the same convolution as an implicit GEMM on the 64x64 tile, and
 * the residual VAE decoder's two fused variants (channel-RMS-norm -> (SiLU) ->
 * conv, and nearest-2x -> conv, with an optional accumulate epilogue).
 *
 * Include-only piece of `diffusion.cu` (needs `gemm.cu`'s tile and readers);
 * ported from the Metal twin.
 *
 * Contract the hosts rely on (and the FLUX.3 autoencoder's encoder added): a
 * tap that falls outside the input - above, left, below OR right of the field -
 * reads as zero, and `out_h` / `out_w` come from the arguments, never derived
 * from in/pad/stride. The encoder's stride-2 downsample passes `pad 0, stride 2,
 * out = H/2`, so its last tap row/column lies past the bottom/right edge
 * (the reference pads one zero there).
 *
 * LIMIT: the implicit-GEMM grids put output pixels on gridDim.y
 * (ceil(out_h*out_w/64)), which CUDA caps at 65535: a convolved field above
 * 4,194,240 pixels (2048x2048 is 4,194,304) cannot launch as the host's
 * `Extent3` is shaped. Metal has no such cap. Fixing it is a host change (split
 * the launch over row ranges), not a kernel one.
 */

/* One NCHW f32 convolution, per pixel. Grid (out_w, out_c, batch*out_h) x 64:
 * one block per output value (x = out column, y = out channel, z = batch row
 * n*out_h + oh); the threads stride over the in_c*k_h*k_w taps and reduce. */
extern "C" __global__ void kernel_dit_conv2d_f32(
        const dit_args_conv2d args, float* out, const float* in, const float* weight,
        const float* bias) {
    __shared__ float scratch[3];
    const u32 ow = blockIdx.x;
    const u32 oc = blockIdx.y;
    const u32 n = blockIdx.z / args.out_h;
    const u32 oh = blockIdx.z % args.out_h;
    const u32 tid = threadIdx.x;
    if (ow >= args.out_w || oc >= args.out_c || blockIdx.z >= args.batch * args.out_h) {
        return;
    }
    const u32 taps = args.in_c * args.k_h * args.k_w;
    float acc = 0.0f;
    for (u32 tap = tid; tap < taps; tap += DIT_BLOCK) {
        const u32 kw = tap % args.k_w;
        const u32 kh = (tap / args.k_w) % args.k_h;
        const u32 ic = tap / (args.k_w * args.k_h);
        const u32 ih = oh * args.stride + kh;
        const u32 iw = ow * args.stride + kw;
        if (ih < args.pad || iw < args.pad || ih - args.pad >= args.in_h
                || iw - args.pad >= args.in_w) {
            continue;
        }
        const u64 xi = (((u64)n * args.in_c + ic) * args.in_h + (ih - args.pad)) * args.in_w
                + (iw - args.pad);
        const u64 wi = (((u64)oc * args.in_c + ic) * args.k_h + kh) * args.k_w + kw;
        acc += in[xi] * weight[wi];
    }
    acc = dit_block_sum(acc, scratch, tid);
    if (tid == 0u) {
        out[(((u64)n * args.out_c + oc) * args.out_h + oh) * args.out_w + ow] =
                acc + (args.has_bias != 0u ? bias[oc] : 0.0f);
    }
}

/* The convolution's left operand without an im2col buffer: row r is output
 * pixel (r / out_w, r % out_w) of one batch image, column t is tap (ic, kh, kw)
 * in the weight's own [ic][kh][kw] order, so the weight [out_c][in_c][kh][kw]
 * is already the [n][k] matrix the tile reads. */
struct dit_im2col {
    static constexpr bool kAlongM = false;
    const float* in; /* this batch image's (in_c, in_h, in_w) field */
    u32 in_h, in_w, out_w, k_h, k_w, stride, pad;
    /* Four consecutive taps of one output pixel (kw, then kh, then ic). */
    __device__ float4 load4(u32 r, u32 kk) const {
        const u32 oh = r / out_w;
        const u32 ow = r - oh * out_w;
        const u32 khw = k_h * k_w;
        u32 ic = kk / khw;
        const u32 rem = kk - ic * khw;
        u32 kh = rem / k_w;
        u32 kw = rem - kh * k_w;
        const int ih0 = (int)(oh * stride) - (int)pad;
        const int iw0 = (int)(ow * stride) - (int)pad;
        float v[4];
#pragma unroll
        for (u32 c = 0; c < 4u; ++c) {
            const int ih = ih0 + (int)kh;
            const int iw = iw0 + (int)kw;
            v[c] = (ih < 0 || iw < 0 || ih >= (int)in_h || iw >= (int)in_w)
                    ? 0.0f
                    : in[((u64)ic * in_h + (u32)ih) * in_w + (u32)iw];
            if (++kw == k_w) {
                kw = 0u;
                if (++kh == k_h) {
                    kh = 0u;
                    ++ic;
                }
            }
        }
        return make_float4(v[0], v[1], v[2], v[3]);
    }
};

/* NCHW epilogue: column c is the output channel, row r the pixel. Rows-fast so
 * a warp's stores are consecutive pixels of one channel plane. */
struct dit_store_nchw {
    static constexpr bool kRowsFast = true;
    float* out; /* this batch image's (out_c, out_h*out_w) field */
    u32 plane;
    const float* bias;
    bool has_bias;
    __device__ void store(u32 r, u32 c, float v) const {
        out[(u64)c * plane + r] = v + (has_bias ? bias[c] : 0.0f);
    }
};

/* One NCHW f32 convolution as an implicit GEMM: M = out_h*out_w pixels,
 * N = out_c, K = in_c*k_h*k_w taps (a multiple of 16; the host sends other
 * shapes to the per-pixel entry). Grid (ceil(out_c/64), ceil(out_h*out_w/64),
 * batch) x 128. */
extern "C" __global__ void kernel_dit_conv2d_gemm_f32(
        const dit_args_conv2d args, float* out, const float* in, const float* weight,
        const float* bias) {
    __shared__ float smem[DIT_G64_SMEM];
    const u32 b = blockIdx.z;
    const u32 plane = args.out_h * args.out_w;
    const u32 taps = args.in_c * args.k_h * args.k_w;
    const dit_im2col a{in + (u64)b * args.in_c * args.in_h * args.in_w, args.in_h, args.in_w,
                       args.out_w, args.k_h, args.k_w, args.stride, args.pad};
    const dit_store_nchw ep{out + (u64)b * args.out_c * plane, plane, bias, args.has_bias != 0u};
    dit_gemm64_tile(plane, args.out_c, taps, a, dit_wrows_f32{weight, taps}, ep, smem);
}

/* --- residual VAE decoder: fused channel-RMS / upsample convolutions --------- */

/* The gather transforms. */
#define DIT_VAE_RMS 0u
#define DIT_VAE_UP2 1u

/* The convolution's left operand like dit_im2col (stride 1, square kernel) but
 * with the taps ordered CHANNEL-FASTEST, kk = (kh*k + kw)*in_c + ic (the host
 * stores the weight [o][kh][kw][i]), the input transform applied per tap:
 *
 * - DIT_VAE_RMS: the tap reads x[ic,p] * inv[p] * gamma[ic], then SiLU when
 *   `silu` is set (the reference's norm -> (silu) -> conv);
 * - DIT_VAE_UP2: the convolved field is the stored one upsampled 2x by nearest
 *   neighbour ((f_h, f_w) == 2*(in_h, in_w)), so tap (ih, iw) reads stored
 *   pixel (ih/2, iw/2).
 *
 * Taps outside the convolved field read as zero: the reference pads the
 * transformed field, so padding is never normalized or activated. The tile
 * reads this operand along M (load4m: column kk of four consecutive pixels).
 * Unlike the Metal twin there is no vectorised in-field fast path: every value
 * takes the per-value route, which is the same arithmetic. */
template <u32 MODE>
struct dit_im2col_vae {
    static constexpr bool kAlongM = true;
    const float* in;
    const float* inv;
    const float* gamma;
    u32 in_h, in_w, f_h, f_w, out_w, k, pad;
    bool silu;
    u32 in_c_;
    /* Column kk of the four output pixels r..r+3, zero past row m. */
    __device__ float4 load4m(u32 r, u32 kk, u32 m) const {
        const u32 tap = kk / in_c_;
        const u32 ic = kk - tap * in_c_;
        const u32 kh = tap / k;
        const u32 kw = tap - kh * k;
        const u64 plane = (u64)in_h * in_w;
        const float g = MODE == DIT_VAE_RMS ? gamma[ic] : 1.0f;
        u32 ph = r / out_w;
        u32 pw = r - ph * out_w;
        float e[4];
#pragma unroll
        for (u32 c = 0; c < 4u; ++c) {
            const int yh = (int)ph - (int)pad + (int)kh;
            const int xw = (int)pw - (int)pad + (int)kw;
            if (r + c >= m || yh < 0 || xw < 0 || yh >= (int)f_h || xw >= (int)f_w) {
                e[c] = 0.0f;
            } else if (MODE == DIT_VAE_UP2) {
                e[c] = in[ic * plane + (u64)((u32)yh >> 1u) * in_w + ((u32)xw >> 1u)];
            } else {
                const u64 p = (u64)(u32)yh * in_w + (u32)xw;
                const float x = in[ic * plane + p] * inv[p] * g;
                e[c] = silu ? dit_silu(x) : x;
            }
            if (++pw == out_w) {
                pw = 0u;
                ++ph;
            }
        }
        return make_float4(e[0], e[1], e[2], e[3]);
    }
};

/* NCHW epilogue with the residual add: out = conv + bias, or out += conv + bias
 * when `accumulate` is set. Each element is read and written by the one thread
 * that stores it, so accumulating into the residual stream in place is
 * race-free. */
struct dit_store_nchw_acc {
    static constexpr bool kRowsFast = true;
    float* out;
    u32 plane;
    const float* bias;
    bool has_bias;
    bool accumulate;
    __device__ void store(u32 r, u32 c, float v) const {
        const u64 i = (u64)c * plane + r;
        const float y = v + (has_bias ? bias[c] : 0.0f);
        out[i] = accumulate ? out[i] + y : y;
    }
};

/* norm -> (silu) -> conv as one implicit GEMM: M = out_h*out_w, N = out_c,
 * K = in_c*k*k (a multiple of 16). `inv` is channel_rms_inv's per-pixel inverse
 * norm of `in`, `gamma` the norm's per-channel gain. Grid (ceil(out_c/64),
 * ceil(M/64)) x 128. Binding order: args, out, in, weight, bias, inv, gamma. */
extern "C" __global__ void kernel_dit_conv2d_rms_gemm_f32(
        const dit_args_conv_vae args, float* out, const float* in, const float* weight,
        const float* bias, const float* inv, const float* gamma) {
    __shared__ float smem[DIT_G64_SMEM];
    const u32 plane = args.out_h * args.out_w;
    const u32 taps = args.in_c * args.k * args.k;
    const dit_im2col_vae<DIT_VAE_RMS> a{in, inv, gamma, args.in_h, args.in_w, args.in_h,
                                        args.in_w, args.out_w, args.k, args.pad,
                                        args.silu != 0u, args.in_c};
    const dit_store_nchw_acc ep{out, plane, bias, args.has_bias != 0u, args.accumulate != 0u};
    dit_gemm64_tile(plane, args.out_c, taps, a, dit_wrows_f32{weight, taps}, ep, smem);
}

/* nearest-2x -> conv as one implicit GEMM over the upsampled field, which is
 * never materialized: (in_h, in_w) is the stored field, (out_h, out_w) the
 * convolved one (twice it, same padding). Grid and K contract as the rms entry;
 * `silu` is ignored. Binding order: args, out, in, weight, bias. */
extern "C" __global__ void kernel_dit_conv2d_up2_gemm_f32(
        const dit_args_conv_vae args, float* out, const float* in, const float* weight,
        const float* bias) {
    __shared__ float smem[DIT_G64_SMEM];
    const u32 plane = args.out_h * args.out_w;
    const u32 taps = args.in_c * args.k * args.k;
    const dit_im2col_vae<DIT_VAE_UP2> a{in, in, in, args.in_h, args.in_w, 2u * args.in_h,
                                        2u * args.in_w, args.out_w, args.k, args.pad,
                                        false, args.in_c};
    const dit_store_nchw_acc ep{out, plane, bias, args.has_bias != 0u, args.accumulate != 0u};
    dit_gemm64_tile(plane, args.out_c, taps, a, dit_wrows_f32{weight, taps}, ep, smem);
}
