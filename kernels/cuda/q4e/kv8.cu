/* The key/value cache stored as one byte an element: FP8 E4M3 at a static
 * scale of one.
 *
 * Not an upstream unit. Every kernel below is a ported one from `qsa.cu`,
 * `qsa_sparse.cu` or `exact.cu` with the cache's element type changed and
 * nothing else: the store rounds its f32 row to E4M3 where the ported store
 * rounds it to f16, and every reader turns the byte back into the f16 the
 * ported reader would have loaded, then runs the ported arithmetic on it
 * unchanged. They are a module of their own (`q4e_kv8`) so that no ported unit
 * is edited.
 *
 * # Why the decode is exact, and why that is the whole design
 *
 * Every finite E4M3 value is an f16 value: E4M3's exponent range (2^-9 at the
 * smallest subnormal, 448 at the top) sits well inside f16's normal range, and
 * its three mantissa bits are a prefix of f16's ten. So a reader can
 * reconstruct the stored value exactly as an f16, and from there on it is the
 * f16 kernel reading an f16 cache that happens to hold those values. The store
 * is the only place this module rounds anything, which buys three properties
 * the f16 cache already had and a lossy reader would lose:
 *
 *   - the dense, tiled, split, gathered and tensor-core readers still agree
 *     with each other exactly as their f16 twins do, so the rules that pick
 *     between them (verify = single step, batch invariance) keep holding;
 *   - a page's bytes depend only on the rows written into it, never on how the
 *     prompt was chunked, so a prefix page is reusable across requests;
 *   - each reader is bit-equal to its f16 twin run on an f16 cache holding the
 *     decoded values, which is what the kernel cases check.
 *
 * A static scale is what keeps the second property. A per-page scale would
 * make a page's bytes depend on which rows shared it when it was first
 * written; a per-row scale would not, and is the escape hatch if the value
 * plane turns out to need more range than one scale gives.
 *
 * # The codec
 *
 * Encode (f32 -> E4M3): round to nearest, ties to even, on the E4M3 grid;
 * anything at or past 448 - infinities and NaN included - is written as +-448.
 * The NaN code (0x7F / 0xFF) is never written, so a cache this module filled
 * holds only finite values. Done in integer arithmetic on the f32 bit pattern,
 * so `--use_fast_math`'s flush-to-zero cannot move a subnormal input.
 *
 * Decode (E4M3 -> f16): shifting the byte's exponent and mantissa up by seven
 * bits lands them in f16's fields, and the result is the stored value times
 * 2^-8 - exactly, subnormals included, because f16 has the exponent range for
 * both. Multiplying by 256 in f16 then restores the value; f16 arithmetic keeps
 * subnormals and a power-of-two product that is representable is exact. Two
 * integer operations and one paired f16 multiply per two elements, no table
 * and no sm_89 conversion instruction, so the module stays at compute_80. The
 * never-written NaN code decodes to +-480, the grid step above 448.
 */

#include "q4e.cuh"

/* ---------------------------------------------------------------------------
 * The codec.
 * ------------------------------------------------------------------------ */

/* f32 -> E4M3 code: saturating, round to nearest even, never the NaN code. */
__device__ __forceinline__ static uint8_t q4e_kv8_encode(float x) {
    const uint32_t u = __float_as_uint(x);
    const uint32_t s = (u >> 24) & 0x80u;
    const uint32_t a = u & 0x7FFFFFFFu;
    uint32_t code;
    if (a >= 0x43E00000u) {
        /* 448 and past, infinities and NaN. */
        code = 0x7Eu;
    } else if (a <= 0x3A800000u) {
        /* At most 2^-10, half the smallest subnormal: the tie goes to zero,
         * which is the even neighbour. */
        code = 0u;
    } else if (a >= 0x3C800000u) {
        /* Normal (|x| >= 2^-6): drop twenty mantissa bits with the tie to
         * even, then rebias the exponent from 127 to 7. A carry out of the
         * mantissa is the next binade, which the rebias handles as is. */
        const uint32_t r = (a + 0x7FFFFu + ((a >> 20) & 1u)) >> 20;
        code = r - ((127u - 7u) << 3);
        if (code > 0x7Eu) code = 0x7Eu;
    } else {
        /* Subnormal steps of 2^-9: the value in those units, rounded to
         * nearest even. A round up to 8 is 2^-6, the smallest normal, whose
         * code is 8 - the two grids meet without a seam. */
        const uint32_t e = a >> 23;                          /* 117..120 */
        const uint32_t mant = (a & 0x7FFFFFu) | 0x800000u;
        const uint32_t shift = 141u - e;                     /* 21..24 */
        code = (mant + (1u << (shift - 1u)) - 1u + ((mant >> shift) & 1u)) >> shift;
    }
    return (uint8_t)(s | code);
}

/* The f16 pattern of the value 2^-8 * e4m3(b), for the codes `u` holds in its
 * bytes 0 and 2. */
__device__ __forceinline__ static uint32_t q4e_kv8_shift2(uint32_t u) {
    return ((u & 0x00800080u) << 8) | ((u & 0x007F007Fu) << 7);
}

/* Two codes, chosen out of `w` by the byte-permute selector `sel`, to an f16
 * pair holding exactly their values. */
__device__ __forceinline__ static uint32_t q4e_kv8_dec2(uint32_t w, uint32_t sel) {
    union { uint32_t u; __half2 h; } v;
    v.u = q4e_kv8_shift2(__byte_perm(w, 0u, sel));
    v.h = __hmul2(v.h, __float2half2_rn(256.0f));
    return v.u;
}

/* Eight codes (one 8-byte load) to the eight f16 values a 16-byte f16 load of
 * the same row would have carried, in the same order. */
__device__ __forceinline__ static uint4 q4e_kv8_dec8(uint2 w) {
    /* Selector nibbles pick byte 0 then a zero byte, byte 1 then a zero byte
     * (0x4140), or bytes 2 and 3 likewise (0x4342). */
    return make_uint4(q4e_kv8_dec2(w.x, 0x4140u), q4e_kv8_dec2(w.x, 0x4342u),
                      q4e_kv8_dec2(w.y, 0x4140u), q4e_kv8_dec2(w.y, 0x4342u));
}

/* One code to its f16 value. */
__device__ __forceinline__ static __half q4e_kv8_dec1(uint8_t b) {
    const uint32_t c = b;
    __half_raw r;
    r.x = (unsigned short)(((c & 0x80u) << 8) | ((c & 0x7Fu) << 7));
    return __hmul(__half(r), __float2half_rn(256.0f));
}

/* ---------------------------------------------------------------------------
 * Stores: `q4e_qsa_store_kv_kernel` and `..._batch_kernel` (qsa.cu), with the
 * final write rounding to E4M3 where theirs rounds to f16.
 * ------------------------------------------------------------------------ */

extern "C" __global__ void q4e_qsa_store_kv_fp8_kernel(
        uint8_t *k_cache, uint8_t *v_cache, const float *k, const float *v,
        const float *kw, uint32_t head_dim, uint32_t n_head_kv, uint32_t n_rot,
        float rope_base, const int32_t *pos, const int32_t *mrope,
        uint32_t sec_t, uint32_t sec_h, uint32_t sec_w, const int32_t *pages,
        uint32_t pool_slots, float eps) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const uint64_t src = ((uint64_t)t * n_head_kv + h) * head_dim;

    extern __shared__ float q4e_kv8_k_smem[];
    float *s_k = q4e_kv8_k_smem;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) s_k[i] *= scale * kw[i];
    __syncthreads();
    q4e_rope_mrope(s_k, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
    __syncthreads();

    const uint64_t slot = ((uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * n_head_kv + h) * head_dim;
    if (slot + head_dim > (uint64_t)pool_slots * n_head_kv * head_dim) return;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        k_cache[slot + i] = q4e_kv8_encode(s_k[i]);
        v_cache[slot + i] = q4e_kv8_encode(v[src + i]);
    }
}

extern "C" __global__ void q4e_qsa_store_kv_batch_fp8_kernel(
        uint8_t *k_cache, uint8_t *v_cache, const float *k, const float *v,
        const float *kw, uint32_t head_dim, uint32_t n_head_kv, uint32_t n_rot,
        float rope_base, const int32_t *pos, const int32_t *mrope,
        uint32_t sec_t, uint32_t sec_h, uint32_t sec_w,
        const int32_t *const *pages_arr, uint32_t pool_slots, float eps) {
    const uint32_t t = blockIdx.y;
    const uint32_t h = blockIdx.x;
    const int32_t *pages = pages_arr[t];
    const uint64_t src = ((uint64_t)t * n_head_kv + h) * head_dim;

    extern __shared__ float q4e_kv8_k_smem[];
    float *s_k = q4e_kv8_k_smem;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float x = k[src + i];
        s_k[i] = x;
        sum += x * x;
    }
    const float scale = rsqrtf(q4e_block_sum(sum) / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) s_k[i] *= scale * kw[i];
    __syncthreads();
    q4e_rope_mrope(s_k, n_rot, rope_base,
                   mrope[3u * t], mrope[3u * t + 1u], mrope[3u * t + 2u],
                   sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
    __syncthreads();
    const uint64_t slot = ((uint64_t)q4e_kv_row(pages, (uint32_t)pos[t]) * n_head_kv + h) * head_dim;
    if (slot + head_dim > (uint64_t)pool_slots * n_head_kv * head_dim) return;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        k_cache[slot + i] = q4e_kv8_encode(s_k[i]);
        v_cache[slot + i] = q4e_kv8_encode(v[src + i]);
    }
}

/* ---------------------------------------------------------------------------
 * Dense readers: `q4e_qsa_attention_kernel`, `..._batch_kernel` and
 * `..._tiled_kernel` (qsa.cu), each cache load replaced by a byte load and its
 * exact decode.
 * ------------------------------------------------------------------------ */
#define Q4E_KV8_LANE_SLOTS 8u          /* head_dim <= 32 * this */
#define Q4E_KV8_WARPS      8u

extern "C" __global__ void q4e_qsa_attention_fp8_kernel(
        float *out, const uint8_t *k_cache, const uint8_t *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t last = (uint32_t)pos[t];
    const float scale = rsqrtf((float)head_dim);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = (head_dim + 31u) / 32u;
    const uint64_t qbase = ((uint64_t)t * n_head + h) * head_dim;

    float qv[Q4E_KV8_LANE_SLOTS];
    float acc[Q4E_KV8_LANE_SLOTS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (uint32_t p = warp; p <= last; p += Q4E_KV8_WARPS) {
        const uint64_t kb = ((uint64_t)q4e_kv_row(pages, p) * n_head_kv + hkv) * head_dim;
        float dot = 0.0f;
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            if (i < head_dim) dot += qv[j] * __half2float(q4e_kv8_dec1(k_cache[kb + i]));
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);

        const float m_new = fmaxf(m, dot);
        const float corr = __expf(m - m_new);
        const float w = __expf(dot - m_new);
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            const float v = (i < head_dim) ? __half2float(q4e_kv8_dec1(v_cache[kb + i])) : 0.0f;
            acc[j] = acc[j] * corr + w * v;
        }
        l = l * corr + w;
        m = m_new;
    }

    extern __shared__ float q4e_kv8_attn_smem[];
    float *s_acc = q4e_kv8_attn_smem;                   /* warps * head_dim */
    __shared__ float s_m[Q4E_KV8_WARPS];
    __shared__ float s_l[Q4E_KV8_WARPS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        if (i < head_dim) s_acc[(uint64_t)warp * head_dim + i] = acc[j];
    }
    if (lane == 0u) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();

    float gm = -INFINITY;
    for (uint32_t w = 0; w < Q4E_KV8_WARPS; w++) gm = fmaxf(gm, s_m[w]);
    float den = 0.0f;
    for (uint32_t w = 0; w < Q4E_KV8_WARPS; w++) den += s_l[w] * __expf(s_m[w] - gm);
    const float inv = 1.0f / den;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float num = 0.0f;
        for (uint32_t w = 0; w < Q4E_KV8_WARPS; w++) {
            num += s_acc[(uint64_t)w * head_dim + i] * __expf(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

extern "C" __global__ void q4e_qsa_attention_batch_fp8_kernel(
        float *out, const uint8_t *k_cache, const uint8_t *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *const *pages_arr, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const int32_t *pages = pages_arr[t];
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t last = (uint32_t)pos[t];
    const float scale = rsqrtf((float)head_dim);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = (head_dim + 31u) / 32u;
    const uint64_t qbase = ((uint64_t)t * n_head + h) * head_dim;

    float qv[Q4E_KV8_LANE_SLOTS];
    float acc[Q4E_KV8_LANE_SLOTS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        qv[j] = (i < head_dim) ? q[qbase + i] * scale : 0.0f;
        acc[j] = 0.0f;
    }
    float m = -INFINITY;
    float l = 0.0f;

    for (uint32_t p = warp; p <= last; p += Q4E_KV8_WARPS) {
        const uint64_t kb = ((uint64_t)q4e_kv_row(pages, p) * n_head_kv + hkv) * head_dim;
        float dot = 0.0f;
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            if (i < head_dim) dot += qv[j] * __half2float(q4e_kv8_dec1(k_cache[kb + i]));
        }
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);

        const float m_new = fmaxf(m, dot);
        const float corr = __expf(m - m_new);
        const float w = __expf(dot - m_new);
        for (uint32_t j = 0; j < slots; j++) {
            const uint32_t i = lane + 32u * j;
            const float v = (i < head_dim) ? __half2float(q4e_kv8_dec1(v_cache[kb + i])) : 0.0f;
            acc[j] = acc[j] * corr + w * v;
        }
        l = l * corr + w;
        m = m_new;
    }

    extern __shared__ float q4e_kv8_attn_smem[];
    float *s_acc = q4e_kv8_attn_smem;
    __shared__ float s_m[Q4E_KV8_WARPS];
    __shared__ float s_l[Q4E_KV8_WARPS];
    for (uint32_t j = 0; j < slots; j++) {
        const uint32_t i = lane + 32u * j;
        if (i < head_dim) s_acc[(uint64_t)warp * head_dim + i] = acc[j];
    }
    if (lane == 0u) { s_m[warp] = m; s_l[warp] = l; }
    __syncthreads();

    float gm = -INFINITY;
    for (uint32_t w = 0; w < Q4E_KV8_WARPS; w++) gm = fmaxf(gm, s_m[w]);
    float den = 0.0f;
    for (uint32_t w = 0; w < Q4E_KV8_WARPS; w++) den += s_l[w] * __expf(s_m[w] - gm);
    const float inv = 1.0f / den;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        float num = 0.0f;
        for (uint32_t w = 0; w < Q4E_KV8_WARPS; w++) {
            num += s_acc[(uint64_t)w * head_dim + i] * __expf(s_m[w] - gm);
        }
        out[qbase + i] = num * inv;
    }
}

#define Q4E_KV8_QT 16u          /* queries per block */
#define Q4E_KV8_KT 32u          /* keys per tile: one per lane */

static_assert(SHS_Q4E_PAGE_TOKENS % Q4E_KV8_KT == 0u,
              "a key tile must not straddle a KV page");

/* The shared tiles stay f16, laid out as the ported kernel lays them, so the
 * launch reserves the same bytes and the inner loops are the ported ones. */
extern "C" __global__ void q4e_qsa_attention_tiled_fp8_kernel(
        float *out, const uint8_t *k_cache, const uint8_t *v_cache, const float *q,
        uint32_t head_dim, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok) {
    const uint32_t h = blockIdx.x;
    const uint32_t t0 = blockIdx.y * Q4E_KV8_QT;
    if (t0 >= n_tok) return;
    const uint32_t hkv = h / (n_head / n_head_kv);
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t slots = head_dim / 32u;          /* dims per lane */
    const float scale = rsqrtf((float)head_dim);

    extern __shared__ char q4e_kv8_tile_smem[];
    __half *s_k = (__half *)q4e_kv8_tile_smem;                    /* [dim][KT] */
    __half *s_v = s_k + (size_t)head_dim * Q4E_KV8_KT;            /* [KT][dim] */
    float  *s_q = (float *)(s_v + (size_t)head_dim * Q4E_KV8_KT); /* [QT][dim] */
    float  *s_w = s_q + (size_t)Q4E_KV8_QT * head_dim;            /* [warps][KT] */

    const uint32_t n_q = (n_tok - t0 < Q4E_KV8_QT) ? (n_tok - t0) : Q4E_KV8_QT;
    uint32_t last = 0;
    for (uint32_t i = 0; i < n_q; i++) {
        const uint32_t p = (uint32_t)pos[t0 + i];
        if (p > last) last = p;
    }
    for (uint32_t idx = threadIdx.x; idx < n_q * head_dim; idx += blockDim.x) {
        const uint32_t qi = idx / head_dim, d = idx % head_dim;
        s_q[qi * head_dim + d] = q[((uint64_t)(t0 + qi) * n_head + h) * head_dim + d] * scale;
    }

    float acc[2][Q4E_KV8_LANE_SLOTS];
    float m[2], l[2];
    for (int s = 0; s < 2; s++) {
        m[s] = -INFINITY;
        l[s] = 0.0f;
        for (uint32_t j = 0; j < slots; j++) acc[s][j] = 0.0f;
    }
    __syncthreads();

    for (uint32_t p0 = 0; p0 <= last; p0 += Q4E_KV8_KT) {
        const uint32_t n_k = (last + 1u - p0 < Q4E_KV8_KT) ? (last + 1u - p0) : Q4E_KV8_KT;
        const uint32_t krow = q4e_kv_row(pages, p0);     /* one lookup per tile */
        for (uint32_t idx = threadIdx.x; idx < n_k * head_dim; idx += blockDim.x) {
            const uint32_t k = idx / head_dim, d = idx % head_dim;
            const uint64_t src = ((uint64_t)(krow + k) * n_head_kv + hkv) * head_dim + d;
            s_k[(size_t)d * Q4E_KV8_KT + k] = q4e_kv8_dec1(k_cache[src]);
            s_v[(size_t)k * head_dim + d] = q4e_kv8_dec1(v_cache[src]);
        }
        __syncthreads();

        for (int s = 0; s < 2; s++) {
            const uint32_t qi = warp * 2u + (uint32_t)s;
            if (qi >= n_q) break;
            const uint32_t qpos = (uint32_t)pos[t0 + qi];
            const float *qrow = s_q + (size_t)qi * head_dim;

            float dot = 0.0f;
            if (lane < n_k && p0 + lane <= qpos) {
                for (uint32_t d = 0; d < head_dim; d++) {
                    dot = fmaf(qrow[d], __half2float(s_k[(size_t)d * Q4E_KV8_KT + lane]), dot);
                }
            } else {
                dot = -INFINITY;
            }

            float tile_max = dot;
            for (int off = 16; off > 0; off >>= 1) {
                tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
            }
            if (tile_max == -INFINITY) continue;          /* fully masked tile */
            const float m_new = fmaxf(m[s], tile_max);
            const float corr = __expf(m[s] - m_new);
            const float w = __expf(dot - m_new);          /* 0 where masked */
            s_w[(size_t)warp * Q4E_KV8_KT + lane] = w;

            float wsum = w;
            for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
            l[s] = l[s] * corr + wsum;
            m[s] = m_new;

            __syncwarp();
            for (uint32_t j = 0; j < slots; j++) {
                const uint32_t d = lane + 32u * j;
                float a = acc[s][j] * corr;
                for (uint32_t k = 0; k < n_k; k++) {
                    a = fmaf(s_w[(size_t)warp * Q4E_KV8_KT + k],
                             __half2float(s_v[(size_t)k * head_dim + d]), a);
                }
                acc[s][j] = a;
            }
            __syncwarp();
        }
        __syncthreads();
    }

    for (int s = 0; s < 2; s++) {
        const uint32_t qi = warp * 2u + (uint32_t)s;
        if (qi >= n_q) break;
        const float inv = 1.0f / l[s];
        const uint64_t base = ((uint64_t)(t0 + qi) * n_head + h) * head_dim;
        for (uint32_t j = 0; j < slots; j++) {
            out[base + lane + 32u * j] = acc[s][j] * inv;
        }
    }
}

/* ---------------------------------------------------------------------------
 * Split and gathered readers: `q4e_qsa_attention_split_kernel` and
 * `..._gather_kernel` (qsa_sparse.cu). Their tiles are f16 in 16-byte chunks;
 * here each chunk is one 8-byte load of codes decoded into the same 16 bytes,
 * so the shared layout, the reservation and every loop after the staging are
 * the ported ones. Partials go to the slab layout
 * `q4e_qsa_attention_combine_kernel` folds, unchanged.
 * ------------------------------------------------------------------------ */
#define Q4E_KV8_QG      16u          /* queries per block: 2 per warp */
#define Q4E_KV8_SPLIT_HD 256u
#define Q4E_KV8_PART_STRIDE (Q4E_KV8_SPLIT_HD + 8u)   /* m, l, pad, acc[256] */

extern "C" __global__ void __launch_bounds__(256, 2)
q4e_qsa_attention_split_fp8_kernel(
        float *out, float *part, const uint8_t *k_cache, const uint8_t *v_cache,
        const float *q, uint32_t n_head, uint32_t n_head_kv,
        const int32_t *pos, const int32_t *pages, uint32_t n_tok, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_KV8_SPLIT_HD;
    constexpr uint32_t KT = Q4E_KV8_KT;
    constexpr uint32_t C8 = HD / 8u;                    /* 8-element chunks per row */
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t hkv = blockIdx.z;
    const uint32_t split = blockIdx.y;
    const uint32_t G = n_head / n_head_kv;
    const uint32_t NQ = G * n_tok;
    const uint32_t q0 = blockIdx.x * G;
    if (q0 >= NQ || G > Q4E_KV8_QG) return;
    const uint32_t n_q = G;
    const float scale = rsqrtf((float)HD);

    extern __shared__ __align__(16) uint4 q4e_kv8_split_smem[];
    uint4 *s_k8 = q4e_kv8_split_smem;                      /* [C8][KT]: chunk-major */
    uint4 *s_v8 = s_k8 + (size_t)C8 * KT;                  /* [KT][C8]: key-major */
    uint4 *s_q8 = s_v8 + (size_t)KT * C8;                  /* [QG][C8] */
    float *s_w  = (float *)(s_q8 + (size_t)Q4E_KV8_QG * C8); /* [warps][KT] */

    const uint32_t last = (uint32_t)pos[blockIdx.x];
    const uint32_t n_tiles = (last + KT) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t k_begin = split * tiles_per_split * KT;
    uint32_t k_end = k_begin + tiles_per_split * KT;
    if (k_end > last + 1u) k_end = last + 1u;

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[s][j] = 0.0f;
    }

    if (k_begin < k_end) {
        for (uint32_t idx = tid; idx < n_q * C8; idx += blockDim.x) {
            const uint32_t qi = idx / C8, c = idx % C8;
            const uint32_t t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + h) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            union { uint4 u; __half2 h2[4]; } pk;
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
            s_q8[(size_t)qi * C8 + c] = pk.u;
        }
        __syncthreads();

        for (uint32_t p0 = k_begin; p0 < k_end; p0 += KT) {
            const uint32_t n_k = (k_end - p0 < KT) ? (k_end - p0) : KT;
            const uint32_t krow = q4e_kv_row(pages, p0);
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_k8[(size_t)c * KT + key] = q4e_kv8_dec8(*(const uint2 *)(k_cache +
                            (((uint64_t)(krow + key) * n_head_kv + hkv) * HD + 8u * c)));
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_v8[(size_t)key * C8 + c] = q4e_kv8_dec8(*(const uint2 *)(v_cache +
                            (((uint64_t)(krow + key) * n_head_kv + hkv) * HD + 8u * c)));
                } else {
                    /* The accumulation runs the whole tile with zero weights on
                     * the tail, and 0 * NaN from stale memory is NaN. */
                    s_v8[(size_t)key * C8 + c] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();

            const uint32_t qi0 = warp * 2u;
            const uint32_t nq_w = (qi0 >= n_q) ? 0u : ((n_q - qi0 >= 2u) ? 2u : 1u);
            if (nq_w == 0u) { __syncthreads(); continue; }
            const uint4 *qrow0 = s_q8 + (size_t)qi0 * C8;
            const uint4 *qrow1 = s_q8 + (size_t)(qi0 + (nq_w - 1u)) * C8;
            float dot[2] = { -INFINITY, -INFINITY };
            if (lane < n_k) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll 4
                for (uint32_t c = 0; c < C8; c++) {
                    union { uint4 u; __half2 h2[4]; } kk, qa, qb;
                    kk.u = s_k8[(size_t)c * KT + lane];
                    qa.u = qrow0[c];
                    qb.u = qrow1[c];
                    const float2 k0 = __half22float2(kk.h2[0]), k1 = __half22float2(kk.h2[1]);
                    const float2 k2 = __half22float2(kk.h2[2]), k3 = __half22float2(kk.h2[3]);
                    const float2 x0 = __half22float2(qa.h2[0]), x1 = __half22float2(qa.h2[1]);
                    const float2 x2 = __half22float2(qa.h2[2]), x3 = __half22float2(qa.h2[3]);
                    const float2 y0 = __half22float2(qb.h2[0]), y1 = __half22float2(qb.h2[1]);
                    const float2 y2 = __half22float2(qb.h2[2]), y3 = __half22float2(qb.h2[3]);
                    a0 = fmaf(x0.x, k0.x, a0); a1 = fmaf(x0.y, k0.y, a1);
                    a2 = fmaf(x1.x, k1.x, a2); a3 = fmaf(x1.y, k1.y, a3);
                    a0 = fmaf(x2.x, k2.x, a0); a1 = fmaf(x2.y, k2.y, a1);
                    a2 = fmaf(x3.x, k3.x, a2); a3 = fmaf(x3.y, k3.y, a3);
                    b0 = fmaf(y0.x, k0.x, b0); b1 = fmaf(y0.y, k0.y, b1);
                    b2 = fmaf(y1.x, k1.x, b2); b3 = fmaf(y1.y, k1.y, b3);
                    b0 = fmaf(y2.x, k2.x, b0); b1 = fmaf(y2.y, k2.y, b1);
                    b2 = fmaf(y3.x, k3.x, b2); b3 = fmaf(y3.y, k3.y, b3);
                }
                dot[0] = (a0 + a1) + (a2 + a3);
                dot[1] = (b0 + b1) + (b2 + b3);
            }

#pragma unroll
            for (int s = 0; s < 2; s++) {
                if ((uint32_t)s >= nq_w) break;
                const uint32_t qi = qi0 + (uint32_t)s;
                const uint32_t qpos = (uint32_t)pos[(q0 + qi) / G];
                const float d = (lane < n_k && p0 + lane <= qpos) ? dot[s] : -INFINITY;
                float tile_max = d;
                for (int off = 16; off > 0; off >>= 1) {
                    tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
                }
                if (tile_max == -INFINITY) continue;
                const float m_new = fmaxf(m[s], tile_max);
                const float corr = __expf(m[s] - m_new);
                const float w = __expf(d - m_new);          /* 0 where masked */
                s_w[(size_t)warp * KT + lane] = w;
                float wsum = w;
                for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
                l[s] = l[s] * corr + wsum;
                m[s] = m_new;
                __syncwarp();

#pragma unroll
                for (int j = 0; j < 8; j++) acc[s][j] *= corr;
                const float *wrow = s_w + (size_t)warp * KT;
                const uint4 *vcol = s_v8 + lane;
#pragma unroll 4
                for (uint32_t k = 0; k < KT; k++) {
                    const float wk = wrow[k];
                    union { uint4 u; __half2 h2[4]; } vv;
                    vv.u = vcol[(size_t)k * C8];
                    const float2 v0 = __half22float2(vv.h2[0]), v1 = __half22float2(vv.h2[1]);
                    const float2 v2 = __half22float2(vv.h2[2]), v3 = __half22float2(vv.h2[3]);
                    acc[s][0] = fmaf(wk, v0.x, acc[s][0]);
                    acc[s][1] = fmaf(wk, v0.y, acc[s][1]);
                    acc[s][2] = fmaf(wk, v1.x, acc[s][2]);
                    acc[s][3] = fmaf(wk, v1.y, acc[s][3]);
                    acc[s][4] = fmaf(wk, v2.x, acc[s][4]);
                    acc[s][5] = fmaf(wk, v2.y, acc[s][5]);
                    acc[s][6] = fmaf(wk, v3.x, acc[s][6]);
                    acc[s][7] = fmaf(wk, v3.y, acc[s][7]);
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }

#pragma unroll
    for (int s = 0; s < 2; s++) {
        const uint32_t qi = warp * 2u + (uint32_t)s;
        if (qi >= n_q) break;
        const uint32_t t = (q0 + qi) / G, h = hkv * G + (q0 + qi) % G;
        const uint64_t qg = (uint64_t)t * n_head + h;
        if (n_splits == 1u) {
            const float inv = 1.0f / l[s];
            float4 *dst = (float4 *)(out + qg * HD + 8u * lane);
            dst[0] = make_float4(acc[s][0] * inv, acc[s][1] * inv, acc[s][2] * inv, acc[s][3] * inv);
            dst[1] = make_float4(acc[s][4] * inv, acc[s][5] * inv, acc[s][6] * inv, acc[s][7] * inv);
        } else {
            float *prow = part + (qg * n_splits + split) * Q4E_KV8_PART_STRIDE;
            if (lane == 0u) { prow[0] = m[s]; prow[1] = l[s]; }
            float4 *dst = (float4 *)(prow + 8u + 8u * lane);
            dst[0] = make_float4(acc[s][0], acc[s][1], acc[s][2], acc[s][3]);
            dst[1] = make_float4(acc[s][4], acc[s][5], acc[s][6], acc[s][7]);
        }
    }
}

extern "C" __global__ void __launch_bounds__(256, 2)
q4e_qsa_attention_gather_fp8_kernel(
        float *part, const uint8_t *k_cache, const uint8_t *v_cache, const float *q,
        const int32_t *tokens, const int32_t *n_sel, const int32_t *pages, uint32_t width,
        uint32_t n_head, uint32_t n_head_kv, uint32_t n_splits) {
    constexpr uint32_t HD = Q4E_KV8_SPLIT_HD;
    constexpr uint32_t KT = Q4E_KV8_KT;
    constexpr uint32_t C8 = HD / 8u;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t t = blockIdx.z, hkv = blockIdx.y, split = blockIdx.x;
    const uint32_t G = n_head / n_head_kv;
    const float scale = rsqrtf((float)HD);
    const int32_t *list = tokens + (uint64_t)t * width;
    const uint32_t n_list = (uint32_t)n_sel[t];
    const uint32_t n_tiles = (n_list + KT - 1u) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t l_begin = split * tiles_per_split * KT;
    uint32_t l_end = l_begin + tiles_per_split * KT;
    if (l_end > n_list) l_end = n_list;

    extern __shared__ __align__(16) uint4 q4e_kv8_gather_smem[];
    uint4 *s_k8 = q4e_kv8_gather_smem;
    uint4 *s_v8 = s_k8 + (size_t)C8 * KT;
    uint4 *s_q8 = s_v8 + (size_t)KT * C8;
    float *s_w  = (float *)(s_q8 + (size_t)Q4E_KV8_QG * C8);
    __shared__ int32_t s_idx[KT];

    float acc[2][8];
    float m[2] = { -INFINITY, -INFINITY };
    float l[2] = { 0.0f, 0.0f };
#pragma unroll
    for (int s = 0; s < 2; s++) {
#pragma unroll
        for (int j = 0; j < 8; j++) acc[s][j] = 0.0f;
    }

    if (l_begin < l_end) {
        for (uint32_t idx = tid; idx < G * C8; idx += blockDim.x) {
            const uint32_t gh = idx / C8, c = idx % C8;
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + hkv * G + gh) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            union { uint4 u; __half2 h2[4]; } pk;
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
            s_q8[(size_t)gh * C8 + c] = pk.u;
        }
        __syncthreads();
        for (uint32_t p0 = l_begin; p0 < l_end; p0 += KT) {
            const uint32_t n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
            if (tid < KT) {
                s_idx[tid] = tid < n_k ? (int32_t)q4e_kv_row(pages, (uint32_t)list[p0 + tid]) : 0;
            }
            __syncthreads();
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = lane, c = warp + 8u * i;
                if (key < n_k) {
                    s_k8[(size_t)c * KT + key] = q4e_kv8_dec8(*(const uint2 *)(k_cache +
                            (((uint64_t)s_idx[key] * n_head_kv + hkv) * HD + 8u * c)));
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < 4u; i++) {
                const uint32_t key = warp + 8u * i, c = lane;
                if (key < n_k) {
                    s_v8[(size_t)key * C8 + c] = q4e_kv8_dec8(*(const uint2 *)(v_cache +
                            (((uint64_t)s_idx[key] * n_head_kv + hkv) * HD + 8u * c)));
                } else {
                    s_v8[(size_t)key * C8 + c] = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            __syncthreads();

            const uint32_t qi0 = warp * 2u;
            const uint32_t nq_w = (qi0 >= G) ? 0u : ((G - qi0 >= 2u) ? 2u : 1u);
            if (nq_w == 0u) { __syncthreads(); continue; }
            const uint4 *qrow0 = s_q8 + (size_t)qi0 * C8;
            const uint4 *qrow1 = s_q8 + (size_t)(qi0 + (nq_w - 1u)) * C8;
            float dot[2] = { -INFINITY, -INFINITY };
            if (lane < n_k) {
                float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
                float b0 = 0.0f, b1 = 0.0f, b2 = 0.0f, b3 = 0.0f;
#pragma unroll 4
                for (uint32_t c = 0; c < C8; c++) {
                    union { uint4 u; __half2 h2[4]; } kk, qa, qb;
                    kk.u = s_k8[(size_t)c * KT + lane];
                    qa.u = qrow0[c];
                    qb.u = qrow1[c];
                    const float2 k0 = __half22float2(kk.h2[0]), k1 = __half22float2(kk.h2[1]);
                    const float2 k2 = __half22float2(kk.h2[2]), k3 = __half22float2(kk.h2[3]);
                    const float2 x0 = __half22float2(qa.h2[0]), x1 = __half22float2(qa.h2[1]);
                    const float2 x2 = __half22float2(qa.h2[2]), x3 = __half22float2(qa.h2[3]);
                    const float2 y0 = __half22float2(qb.h2[0]), y1 = __half22float2(qb.h2[1]);
                    const float2 y2 = __half22float2(qb.h2[2]), y3 = __half22float2(qb.h2[3]);
                    a0 = fmaf(x0.x, k0.x, a0); a1 = fmaf(x0.y, k0.y, a1);
                    a2 = fmaf(x1.x, k1.x, a2); a3 = fmaf(x1.y, k1.y, a3);
                    a0 = fmaf(x2.x, k2.x, a0); a1 = fmaf(x2.y, k2.y, a1);
                    a2 = fmaf(x3.x, k3.x, a2); a3 = fmaf(x3.y, k3.y, a3);
                    b0 = fmaf(y0.x, k0.x, b0); b1 = fmaf(y0.y, k0.y, b1);
                    b2 = fmaf(y1.x, k1.x, b2); b3 = fmaf(y1.y, k1.y, b3);
                    b0 = fmaf(y2.x, k2.x, b0); b1 = fmaf(y2.y, k2.y, b1);
                    b2 = fmaf(y3.x, k3.x, b2); b3 = fmaf(y3.y, k3.y, b3);
                }
                dot[0] = (a0 + a1) + (a2 + a3);
                dot[1] = (b0 + b1) + (b2 + b3);
            }
#pragma unroll
            for (int s = 0; s < 2; s++) {
                if ((uint32_t)s >= nq_w) break;
                const float d = dot[s];
                float tile_max = d;
                for (int off = 16; off > 0; off >>= 1) tile_max = fmaxf(tile_max, __shfl_xor_sync(0xffffffffu, tile_max, off));
                if (tile_max == -INFINITY) continue;
                const float m_new = fmaxf(m[s], tile_max);
                const float corr = __expf(m[s] - m_new);
                const float w = __expf(d - m_new);
                s_w[(size_t)warp * KT + lane] = w;
                float wsum = w;
                for (int off = 16; off > 0; off >>= 1) wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
                l[s] = l[s] * corr + wsum;
                m[s] = m_new;
                __syncwarp();
#pragma unroll
                for (int j = 0; j < 8; j++) acc[s][j] *= corr;
                const float *wrow = s_w + (size_t)warp * KT;
                const uint4 *vcol = s_v8 + lane;
#pragma unroll 4
                for (uint32_t kk = 0; kk < KT; kk++) {
                    const float wk = wrow[kk];
                    union { uint4 u; __half2 h2[4]; } vv;
                    vv.u = vcol[(size_t)kk * C8];
                    const float2 v0 = __half22float2(vv.h2[0]), v1 = __half22float2(vv.h2[1]);
                    const float2 v2 = __half22float2(vv.h2[2]), v3 = __half22float2(vv.h2[3]);
                    acc[s][0] = fmaf(wk, v0.x, acc[s][0]); acc[s][1] = fmaf(wk, v0.y, acc[s][1]);
                    acc[s][2] = fmaf(wk, v1.x, acc[s][2]); acc[s][3] = fmaf(wk, v1.y, acc[s][3]);
                    acc[s][4] = fmaf(wk, v2.x, acc[s][4]); acc[s][5] = fmaf(wk, v2.y, acc[s][5]);
                    acc[s][6] = fmaf(wk, v3.x, acc[s][6]); acc[s][7] = fmaf(wk, v3.y, acc[s][7]);
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }
#pragma unroll
    for (int s = 0; s < 2; s++) {
        const uint32_t gh = warp * 2u + (uint32_t)s;
        if (gh >= G) break;
        const uint64_t qg = (uint64_t)t * n_head + hkv * G + gh;
        float *prow = part + (qg * n_splits + split) * Q4E_KV8_PART_STRIDE;
        if (lane == 0u) { prow[0] = m[s]; prow[1] = l[s]; }
        float4 *dst = (float4 *)(prow + 8u + 8u * lane);
        dst[0] = make_float4(acc[s][0], acc[s][1], acc[s][2], acc[s][3]);
        dst[1] = make_float4(acc[s][4], acc[s][5], acc[s][6], acc[s][7]);
    }
}

/* ---------------------------------------------------------------------------
 * Tensor-core reader: `q4e_qsa_attention_tc_exact_kernel` (exact.cu).
 *
 * The prefetch holds the next tile's codes - one 8-byte piece of K and of V
 * per thread per chunk, half the registers the f16 kernel's 16-byte pieces
 * take - and decodes them into the same swizzled f16 tile as it stores them.
 * Every ldmatrix and MMA after that is the exact kernel's; the shared
 * reservation, the grid and the argument list are too.
 * ------------------------------------------------------------------------ */
#define Q4E_KV8_TC_KT 64u
#define Q4E_KV8_TC_HD 256u
#define Q4E_KV8_TC_CH (Q4E_KV8_TC_HD / 8u)    /* 8-element chunks per row */
__device__ __forceinline__ static uint32_t q4e_kv8_tc_off(uint32_t row, uint32_t chunk) {
    return (row * Q4E_KV8_TC_CH + (chunk ^ (row & 7u))) * 16u;
}
extern "C" __global__ void __launch_bounds__(256)
q4e_qsa_attention_tc_exact_fp8_kernel(
        float *out, float *part, const uint8_t *k_cache, const uint8_t *v_cache, const float *q,
        const int32_t *tokens, const int32_t *n_sel, const int32_t *pos, const int32_t *pages,
        uint32_t width, uint32_t n_head, uint32_t n_head_kv, uint32_t n_splits) {
    constexpr uint32_t KT = Q4E_KV8_TC_KT, HD = Q4E_KV8_TC_HD, CH = Q4E_KV8_TC_CH;
    const uint32_t tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5u;
    const uint32_t t = blockIdx.z, hkv = blockIdx.y, split = blockIdx.x;
    const uint32_t G = n_head / n_head_kv;
    const float scale = rsqrtf((float)HD);

    extern __shared__ __align__(16) uint8_t q4e_kv8_tc_smem[];
    uint8_t *s_q = q4e_kv8_tc_smem;                      /* [16][256] f16 */
    uint8_t *s_k = s_q + 16u * HD * 2u;                  /* [64][256] f16 */
    uint8_t *s_v = s_k + KT * HD * 2u;                   /* [64][256] f16 */
    float   *s_s = (float *)(s_v + KT * HD * 2u);        /* [16][64] f32 scores */
    __half  *s_p = (__half *)(s_s + 16u * KT);           /* [16][64] f16, chunk-swizzled */
    __half  *s_p_lo = s_p + 16u * KT;                    /* [16][64] f16, same swizzle */
    float   *s_corr = (float *)(s_p_lo + 16u * KT);      /* [16] */
    __shared__ int32_t s_idx[Q4E_KV8_TC_KT];
    __shared__ float s_m[16], s_l[16];

    const int32_t *list = tokens ? tokens + (uint64_t)t * width : NULL;
    const uint32_t n_list = tokens ? (uint32_t)n_sel[t] : (uint32_t)pos[t] + 1u;
    const uint32_t n_tiles = (n_list + KT - 1u) / KT;
    const uint32_t tiles_per_split = (n_tiles + n_splits - 1u) / n_splits;
    const uint32_t l_begin = split * tiles_per_split * KT;
    uint32_t l_end = l_begin + tiles_per_split * KT;
    if (l_end > n_list) l_end = n_list;

    for (uint32_t i = tid; i < 16u * CH; i += 256u) {
        const uint32_t r = i / CH, c = i % CH;
        union { uint4 u; __half2 h2[4]; } pk;
        if (r < G) {
            const float4 *src = (const float4 *)(q + ((uint64_t)t * n_head + hkv * G + r) * HD + 8u * c);
            const float4 v0 = src[0], v1 = src[1];
            pk.h2[0] = __floats2half2_rn(v0.x * scale, v0.y * scale);
            pk.h2[1] = __floats2half2_rn(v0.z * scale, v0.w * scale);
            pk.h2[2] = __floats2half2_rn(v1.x * scale, v1.y * scale);
            pk.h2[3] = __floats2half2_rn(v1.z * scale, v1.w * scale);
        } else {
            pk.u = make_uint4(0u, 0u, 0u, 0u);
        }
        *(uint4 *)(s_q + q4e_kv8_tc_off(r, c)) = pk.u;
    }
    float m_r[2] = { -INFINITY, -INFINITY }, l_r[2] = { 0.0f, 0.0f };
    float o[4][4];
#pragma unroll
    for (int f = 0; f < 4; f++) { o[f][0] = o[f][1] = o[f][2] = o[f][3] = 0.0f; }

    /* Keys past the list are zero codes, which decode to zero, so their
     * (masked, zero-weight) columns stay finite as in the f16 kernel. */
    constexpr uint32_t PF = KT * CH / 256u;
    uint2 pk[PF], pv[PF];
    auto fetch = [&](uint32_t p0) {
        const uint32_t n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
        if (tid < KT) {
            s_idx[tid] = tid < n_k
                ? (int32_t)q4e_kv_row(pages, tokens ? (uint32_t)list[p0 + tid] : p0 + tid) : -1;
        }
        __syncthreads();
#pragma unroll
        for (uint32_t j = 0; j < PF; j++) {
            const uint32_t i = tid + 256u * j;
            const uint32_t key = i / CH, c = i % CH;
            const int32_t row = s_idx[key];
            if (row >= 0) {
                const uint64_t base = ((uint64_t)row * n_head_kv + hkv) * HD + 8u * c;
                pk[j] = *(const uint2 *)(k_cache + base);
                pv[j] = *(const uint2 *)(v_cache + base);
            } else {
                pk[j] = make_uint2(0u, 0u);
                pv[j] = pk[j];
            }
        }
    };
    if (l_begin < l_end) fetch(l_begin);
    __syncthreads();

    for (uint32_t p0 = l_begin; p0 < l_end; p0 += KT) {
        const uint32_t n_k = (l_end - p0 < KT) ? (l_end - p0) : KT;
#pragma unroll
        for (uint32_t j = 0; j < PF; j++) {
            const uint32_t i = tid + 256u * j;
            const uint32_t key = i / CH, c = i % CH;
            *(uint4 *)(s_k + q4e_kv8_tc_off(key, c)) = q4e_kv8_dec8(pk[j]);
            *(uint4 *)(s_v + q4e_kv8_tc_off(key, c)) = q4e_kv8_dec8(pv[j]);
        }
        __syncthreads();
        if (p0 + KT < l_end) fetch(p0 + KT);

        float sacc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#pragma unroll 4
        for (uint32_t ks = 0; ks < HD / 16u; ks++) {
            uint32_t a[4], b[2];
            tt_ldmatrix_x4_addr(a, tt_smem_addr(s_q + q4e_kv8_tc_off(lane & 15u, ks * 2u + (lane >> 4u))));
            tt_ldmatrix_x2_addr(b, tt_smem_addr(s_k + q4e_kv8_tc_off(warp * 8u + (lane & 7u),
                                                                  ks * 2u + ((lane >> 3u) & 1u))));
            tt_mma_m16n8k16_f16_f32(sacc, a, b);
        }
        {
            const uint32_t r0 = lane >> 2u, c0 = warp * 8u + 2u * (lane & 3u);
            s_s[r0 * KT + c0] = sacc[0];
            s_s[r0 * KT + c0 + 1u] = sacc[1];
            s_s[(r0 + 8u) * KT + c0] = sacc[2];
            s_s[(r0 + 8u) * KT + c0 + 1u] = sacc[3];
        }
        __syncthreads();

#pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const uint32_t r = warp * 2u + (uint32_t)rr;
            const float v0 = lane < n_k ? s_s[r * KT + lane] : -INFINITY;
            const float v1 = lane + 32u < n_k ? s_s[r * KT + lane + 32u] : -INFINITY;
            float mx = fmaxf(v0, v1);
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            float corr = 1.0f, p0f = 0.0f, p1f = 0.0f;
            if (mx != -INFINITY) {
                const float m_new = fmaxf(m_r[rr], mx);
                corr = __expf(m_r[rr] - m_new);
                p0f = v0 == -INFINITY ? 0.0f : __expf(v0 - m_new);
                p1f = v1 == -INFINITY ? 0.0f : __expf(v1 - m_new);
                float sum = p0f + p1f;
                for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
                l_r[rr] = l_r[rr] * corr + sum;
                m_r[rr] = m_new;
            }
            {
                const uint32_t i0 = r * KT + ((lane >> 3u) ^ (r & 7u)) * 8u + (lane & 7u);
                const uint32_t i1 = r * KT + (((lane + 32u) >> 3u) ^ (r & 7u)) * 8u + (lane & 7u);
                const __half h0 = __float2half_rn(p0f), h1 = __float2half_rn(p1f);
                s_p[i0] = h0;
                s_p[i1] = h1;
                s_p_lo[i0] = __float2half_rn(p0f - __half2float(h0));
                s_p_lo[i1] = __float2half_rn(p1f - __half2float(h1));
            }
            if (lane == 0u) s_corr[r] = corr;
        }
        __syncthreads();

        {
            const float c_lo = s_corr[lane >> 2u], c_hi = s_corr[(lane >> 2u) + 8u];
#pragma unroll
            for (int f = 0; f < 4; f++) {
                o[f][0] *= c_lo; o[f][1] *= c_lo; o[f][2] *= c_hi; o[f][3] *= c_hi;
            }
        }
#pragma unroll
        for (uint32_t ks = 0; ks < KT / 16u; ks++) {
            uint32_t a[4], a_lo[4];
            {
                const uint32_t r = lane & 15u, c = ks * 2u + (lane >> 4u);
                tt_ldmatrix_x4_addr(a, tt_smem_addr(s_p + r * KT + (c ^ (r & 7u)) * 8u));
                tt_ldmatrix_x4_addr(a_lo, tt_smem_addr(s_p_lo + r * KT + (c ^ (r & 7u)) * 8u));
            }
            const uint32_t key = ks * 16u + (lane & 7u) + 8u * ((lane >> 3u) & 1u);
#pragma unroll
            for (uint32_t f = 0; f < 4; f++) {
                uint32_t b[2];
                tt_ldmatrix_x2_trans_addr(b, tt_smem_addr(s_v + q4e_kv8_tc_off(key, warp * 4u + f)));
                tt_mma_m16n8k16_f16_f32(o[f], a, b);
                tt_mma_m16n8k16_f16_f32(o[f], a_lo, b);
            }
        }
        __syncthreads();
    }

    if (lane == 0u) {
        s_m[warp * 2u] = m_r[0]; s_m[warp * 2u + 1u] = m_r[1];
        s_l[warp * 2u] = l_r[0]; s_l[warp * 2u + 1u] = l_r[1];
    }
    __syncthreads();
#pragma unroll
    for (int half_ = 0; half_ < 2; half_++) {
        const uint32_t r = (lane >> 2u) + 8u * (uint32_t)half_;
        if (r >= G) continue;
        const uint64_t qg = (uint64_t)t * n_head + hkv * G + r;
        const uint32_t d0 = warp * 32u + 2u * (lane & 3u);
        if (part) {
            float *prow = part + (qg * n_splits + split) * Q4E_KV8_PART_STRIDE;
            if (warp == 0u && (lane & 3u) == 0u) { prow[0] = s_m[r]; prow[1] = s_l[r]; }
#pragma unroll
            for (uint32_t f = 0; f < 4; f++) {
                prow[8u + d0 + 8u * f] = o[f][2 * half_];
                prow[8u + d0 + 8u * f + 1u] = o[f][2 * half_ + 1];
            }
        } else {
            const float inv = s_l[r] > 0.0f ? 1.0f / s_l[r] : 0.0f;
            float *orow = out + qg * HD;
#pragma unroll
            for (uint32_t f = 0; f < 4; f++) {
                orow[d0 + 8u * f] = o[f][2 * half_] * inv;
                orow[d0 + 8u * f + 1u] = o[f][2 * half_ + 1] * inv;
            }
        }
    }
}
