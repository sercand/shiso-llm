/*
 *
 * Deltas against the source, and nothing else: D1 (extern "C" linkage on the non-template kernels), D2 (explicit template instantiations),
 * D3 (external linkage on the template kernels: an explicit instantiation of a `static` template is emitted as a non-`.visible` entry the driver cannot resolve).
 * D4 (shiso): q4e_hc_combine_norm_kernel issues its loads before its arithmetic; the expressions and their order are unchanged.
 * No line of kernel arithmetic was edited, reordered or
 * reformatted; a change to any of it is a defect, not a cleanup.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * Hyper-connections.
 * ------------------------------------------------------------------------ */

/* Values one thread of q4e_hc_combine_norm_kernel carries: ceil(n_embd /
 * threads), 10 at the shipped 2560 with 256 threads. */
#define Q4E_HC_FUSE_MAX 16u

/* res[t][h][e] = embed[t][e] for every stream h.  The residual starts as the
 * embedding tiled across the streams, not zero padded. */
extern "C" __global__ void q4e_hc_init_kernel(
        float *res, const float *embed, uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * n_hc * n_embd;
    if (idx >= total) return;
    const uint32_t e = (uint32_t)(idx % n_embd);
    const uint64_t t = idx / ((uint64_t)n_hc * n_embd);
    res[idx] = embed[t * n_embd + e];
}

/* res = embed broadcast over the streams, plus a per-stream term.  This is the
 * MTP draft input: fc_embedding(emb) added to every stream of fc_hidden(res). */
extern "C" __global__ void q4e_hc_init_add_kernel(
        float *res, const float *embed, const float *h,
        uint32_t n_embd, uint32_t n_hc, uint32_t n_tok) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_tok * n_hc * n_embd;
    if (idx >= total) return;
    const uint32_t e = (uint32_t)(idx % n_embd);
    const uint64_t t = idx / ((uint64_t)n_hc * n_embd);
    res[idx] = embed[t * n_embd + e] + h[idx];
}

/* The normed residual `xn` may be stored as fp16 (OUT = __half): its only
 * readers are the fp16 hyper-connection GEMMs, which convert it to fp16
 * anyway, and the four-term stream mix, so the rounding lands where the
 * model already rounds; the residual itself stays fp32.  See hc_f16 in ds4.c. */
template <typename OUT>
__global__ void q4e_hc_norm_kernel(
        OUT *out, const float *res, const float *w,
        uint32_t n_embd, uint32_t n_hc, float eps) {
    const uint32_t group = blockIdx.x;            /* one (token, stream) pair */
    const uint32_t h = group % n_hc;
    const float *src = res + (uint64_t)group * n_embd;
    OUT *dst = out + (uint64_t)group * n_embd;
    const float *wh = w + (uint64_t)h * n_embd;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        const float v = src[i];
        sum += v * v;
    }
    sum = q4e_block_sum(sum);
    const float scale = rsqrtf(sum / (float)n_embd + eps);
    for (uint32_t i = threadIdx.x; i < n_embd; i += blockDim.x) {
        q4e_store(dst + i, src[i] * scale * wh[i]);
    }
}

/* y = silu(x / n_hc).  The division sits inside the activation, before the up
 * projection, and dropping it changes the gate materially. */
extern "C" __global__ void q4e_scale_silu_kernel(float *x, float inv_hc, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i] * inv_hc;
    x[i] = v / (1.0f + __expf(-v));
}

/* Collapse the gated streams into the block input: mean over hc, not sum. */
template <typename XT>
__global__ void q4e_hc_collapse_kernel(
        float *out, const XT *xn, const XT *up,
        uint32_t n_embd, uint32_t n_hc) {
    const uint32_t t = blockIdx.x;
    const float inv = 1.0f / (float)n_hc;
    const XT *xn_t = xn + (uint64_t)t * n_hc * n_embd;
    const XT *up_t = up + (uint64_t)t * n_hc * n_embd;
    float *out_t = out + (uint64_t)t * n_embd;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            const uint32_t o = h * n_embd + e;
            const float g = 1.0f / (1.0f + __expf(-q4e_load(up_t + o)));
            acc += q4e_load(xn_t + o) * g;
        }
        out_t[e] = acc * inv;
    }
}

/* The same collapse over half-precision streams, also writing the block input
 * as halves: every projection of a prefill chunk that reads the block input
 * reads it on the half-precision tier, and this is the one place it is
 * produced, so rounding it here replaces one rounding pass per projection. */
extern "C" __global__ void q4e_hc_collapse_f16_dual_kernel(
        float *out, __half *out_h, const __half *xn, const __half *up,
        uint32_t n_embd, uint32_t n_hc) {
    const uint32_t t = blockIdx.x;
    const float inv = 1.0f / (float)n_hc;
    const __half *xn_t = xn + (uint64_t)t * n_hc * n_embd;
    const __half *up_t = up + (uint64_t)t * n_hc * n_embd;
    float *out_t = out + (uint64_t)t * n_embd;
    __half *out_h_t = out_h + (uint64_t)t * n_embd;
    for (uint32_t e = threadIdx.x; e < n_embd; e += blockDim.x) {
        float acc = 0.0f;
        for (uint32_t h = 0; h < n_hc; h++) {
            const uint32_t o = h * n_embd + e;
            const float g = 1.0f / (1.0f + __expf(-__half2float(up_t[o])));
            acc += __half2float(xn_t[o]) * g;
        }
        out_t[e] = acc * inv;
        out_h_t[e] = __float2half_rn(acc * inv);
    }
}

/* The residual update that closes a block, fused with the group-RMSNorm that
 * opens the next one.
 *
 * Both traverse the whole hc * n_embd residual, which at a prefill chunk is
 * the widest stream in the model (2048 tokens x 40 KiB), and the update's
 * result is exactly the norm's input.  Run apart they cost four passes over
 * it -- update reads and writes it, the norm reads it and writes the normed
 * copy -- and the block output once per stream; run together, three, and the
 * block output once.  Measured at 26k: 1.08 + 0.78 ms -> 1.17 ms per layer
 * and per direction.
 *
 * One block per token, not per (token, stream): that is what lets the block
 * output be read once for all n_hc streams.  The per-stream reduction, the
 * element-to-thread mapping and every expression are the ones the two
 * separate kernels used, so the result is bit-identical to them.
 *
 * The streams are walked grid-strided, so the launcher can spread them over
 * gridDim.y blocks instead: at one token, four blocks that each read the
 * block output are worth more than one block that reads it once and then
 * reduces four times in sequence (decode measured 22.6 against 23.9 tok/s
 * before this was a grid stride).
 *
 * `w` is the next mix's norm weight, or NULL where no mix follows in this
 * pass (the last layer) or something else reads the residual in between (the
 * PLE block); then this is the residual update alone and the mix norms its
 * own input. */
template <typename XN>
__global__ void q4e_hc_combine_norm_kernel(
        float *res, XN *xn, const float *block_out, const float *inject,
        const float *w, uint32_t n_embd, uint32_t n_hc, float eps) {
    const uint32_t t = blockIdx.x;
    const float *src = block_out + (uint64_t)t * n_embd;
    const uint32_t h0 = blockIdx.y;
    const uint32_t hstep = gridDim.y;

    /* The block output, held for every stream, and the updated residual, held
     * for the norm's second pass.  Both loops are unrolled over a fixed bound
     * with a guard rather than run to a runtime trip count: a dynamic index
     * into a local array is a local-memory array, and the spill traffic cost
     * more than the pass it saved (1.51 ms against 1.17 ms per call at 26k).
     * The launcher refuses shapes above the bound. */
    float sv[Q4E_HC_FUSE_MAX];
#pragma unroll
    for (uint32_t n = 0; n < Q4E_HC_FUSE_MAX; n++) {
        const uint32_t i = threadIdx.x + n * blockDim.x;
        sv[n] = (i < n_embd) ? src[i] : 0.0f;
    }

    for (uint32_t h = h0; h < n_hc; h += hstep) {
        /* Every load of this stream is issued before any of the arithmetic
         * (shiso): the residual row, the norm weight and the injection logit
         * are independent of each other, and issued in turn behind the
         * dependent chain they were a decode step's layer paid four DRAM
         * round trips here, 10 us, twice a layer.  The expressions and their
         * order are unchanged. */
        float *dst = res + ((uint64_t)t * n_hc + h) * n_embd;
        const float *wh = w ? w + (uint64_t)h * n_embd : nullptr;
        float vv[Q4E_HC_FUSE_MAX], wv[Q4E_HC_FUSE_MAX];
#pragma unroll
        for (uint32_t n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint32_t i = threadIdx.x + n * blockDim.x;
            vv[n] = (i < n_embd) ? dst[i] : 0.0f;
            wv[n] = (wh && i < n_embd) ? wh[i] : 0.0f;
        }
        const float wgt =
            2.0f / (1.0f + __expf(-inject[(uint64_t)t * n_hc + h] / (float)n_hc));
        float sum = 0.0f;
#pragma unroll
        for (uint32_t n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint32_t i = threadIdx.x + n * blockDim.x;
            if (i < n_embd) {
                float v = vv[n];
                v += sv[n] * wgt;
                dst[i] = v;
                vv[n] = v;
                sum += v * v;
            }
        }
        if (!w) continue;
        sum = q4e_block_sum(sum);
        const float scale = rsqrtf(sum / (float)n_embd + eps);
        XN *out = xn + ((uint64_t)t * n_hc + h) * n_embd;
#pragma unroll
        for (uint32_t n = 0; n < Q4E_HC_FUSE_MAX; n++) {
            const uint32_t i = threadIdx.x + n * blockDim.x;
            if (i < n_embd) q4e_store(out + i, vv[n] * scale * wv[n]);
        }
    }
}

/* Explicit instantiations for exactly the argument sets the launchers
 * use.  `extern "C"` is illegal on a template, so these entries keep
 * their mangled names and are resolved by mangled name. */
template __global__ void q4e_hc_norm_kernel<float>(
        float *, const float *, const float *, uint32_t, uint32_t, float);
template __global__ void q4e_hc_norm_kernel<__half>(
        __half *, const float *, const float *, uint32_t, uint32_t, float);
template __global__ void q4e_hc_collapse_kernel<float>(
        float *, const float *, const float *, uint32_t, uint32_t);
template __global__ void q4e_hc_collapse_kernel<__half>(
        float *, const __half *, const __half *, uint32_t, uint32_t);
template __global__ void q4e_hc_combine_norm_kernel<float>(
        float *, float *, const float *, const float *, const float *,
        uint32_t, uint32_t, float);
template __global__ void q4e_hc_combine_norm_kernel<__half>(
        float *, __half *, const float *, const float *, const float *,
        uint32_t, uint32_t, float);
