/* GGUF block-quantized weight-row readers for the 64x64 GEMM tile.
 *
 * Include-only piece of `diffusion.cu`. Each reader is the Metal row loader
 * (`dit_wrows_q*` in `diffusion.metal`) decoded byte for byte: same block
 * layouts, same scale/min packing, same bit positions. The only difference is
 * the memory access: Metal reads `packed_uchar4`; here every byte is read
 * individually, because a mapped checkpoint promises only element alignment
 * (a Q4_K block is 144 bytes into a tensor whose own start is only 2-aligned
 * at best) and CUDA vector loads fault on a misaligned address.
 *
 * Contract shared by all of them: `load4(n, kk)` returns weights
 * `kk .. kk+3` of weight row `n` (a row is `k` weights, `k / 256` blocks, or
 * `k / 32` for Q8_0), and `kk` is a multiple of 4, so the four never straddle
 * a scale group or a block. The tile never asks for a row >= N or a column
 * >= K. Rows are contiguous block arrays with no padding: row `n` starts at
 * `n * (k / block) * block_bytes`.
 *
 * Block layouts (all little-endian):
 *   Q2_K  84 B: u8 scales[16]; u8 qs[64]; f16 d, dmin
 *   Q3_K 110 B: u8 hmask[32]; u8 qs[64]; u8 scales[12]; f16 d
 *   Q4_K 144 B: f16 d, dmin; u8 scales[12]; u8 qs[128]
 *   Q5_K 176 B: f16 d, dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]
 *   Q6_K 210 B: u8 ql[128]; u8 qh[64]; i8 scales[16]; f16 d
 *   Q8_0  34 B: f16 d; i8 qs[32]
 * A shape error here is a plausible-looking wrong picture, not a crash: the
 * shared `shiso-kerneltest` k-quant cases grade each against an oracle written
 * from the format definition.
 */

/* Q4_K / Q5_K 6-bit scale and min of sub-block `sub` (0..8) from the 12 packed
 * scale bytes. */
static __device__ __forceinline__ void dit_kq_scale_min(
        const u8* sc, u32 sub, u32* scale, u32* min) {
    if (sub < 4u) {
        *scale = sc[sub] & 63u;
        *min = sc[sub + 4u] & 63u;
    } else {
        *scale = (sc[sub + 4u] & 0xFu) | ((sc[sub - 4u] >> 6u) << 4u);
        *min = (sc[sub + 4u] >> 4u) | ((sc[sub] >> 6u) << 4u);
    }
}

/* Q4_K: 256 weights are four runs of 64, each the low then the high nibbles of
 * 32 bytes of `qs`; each run of 32 has its own scale/min; weight = d*sc*q - dmin*m. */
struct dit_wrows_q4k {
    const u8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u8* b = w + (u64)n * (k / 256u) * 144u + (u64)(kk / 256u) * 144u;
        const u32 i = kk % 256u;
        const u32 chunk = i / 64u;
        const u32 within = i % 64u;
        const u32 sub = 2u * chunk + (within >= 32u ? 1u : 0u);
        u32 scale, min;
        dit_kq_scale_min(b + 4, sub, &scale, &min);
        const float d = dit_f16(b) * (float)scale;
        const float m = dit_f16(b + 2) * (float)min;
        const u8* q = b + 16 + 32u * chunk + within % 32u;
        const u32 shift = within >= 32u ? 4u : 0u;
        return make_float4(d * (float)((q[0] >> shift) & 0xFu) - m,
                           d * (float)((q[1] >> shift) & 0xFu) - m,
                           d * (float)((q[2] >> shift) & 0xFu) - m,
                           d * (float)((q[3] >> shift) & 0xFu) - m);
    }
};

/* Q6_K: each half of 128 weights is four runs of 32; run r takes the low
 * (r = 0, 1) or high (r = 2, 3) nibble of ql[32*(r&1)..] and bits 2r..2r+1 of
 * qh, minus 32, one i8 scale per 16 weights; weight = d*scale*q. */
struct dit_wrows_q6k {
    const u8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u8* b = w + (u64)n * (k / 256u) * 210u + (u64)(kk / 256u) * 210u;
        const u32 i = kk % 256u;
        const u32 half_ = i / 128u;
        const u32 r = (i % 128u) / 32u;
        const u32 l = i % 32u;
        const u8* ql = b + 64u * half_ + 32u * (r & 1u) + l;
        const u8* qh = b + 128u + 32u * half_ + l;
        const u32 lo_shift = r >= 2u ? 4u : 0u;
        const float scale = (float)(((const i8*)(b + 192u))[8u * half_ + l / 16u + 2u * r]);
        const float d = dit_f16(b + 208u) * scale;
        float v[4];
#pragma unroll
        for (u32 c = 0; c < 4u; ++c) {
            const u32 lo = (ql[c] >> lo_shift) & 0xFu;
            const u32 hi = ((u32)qh[c] >> (2u * r)) & 3u;
            v[c] = d * (float)((int)(lo | (hi << 4u)) - 32);
        }
        return make_float4(v[0], v[1], v[2], v[3]);
    }
};

/* Q2_K: two planes of 128 weights, each four runs of 32 at bit shifts 0,2,4,6
 * of the plane's 32 qs bytes; every 16 weights have a scale nibble (low) and a
 * min nibble (high) in scales[8*plane + 2*run + half]; weight = d*sc*q - dmin*m. */
struct dit_wrows_q2k {
    const u8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u8* b = w + (u64)n * (k / 256u) * 84u + (u64)(kk / 256u) * 84u;
        const u32 i = kk % 256u;
        const u32 plane = i / 128u;
        const u32 run = (i % 128u) / 32u;
        const u32 within = i % 32u;
        const u32 sc = b[8u * plane + 2u * run + within / 16u];
        const float d = dit_f16(b + 80) * (float)(sc & 0xFu);
        const float m = dit_f16(b + 82) * (float)(sc >> 4u);
        const u8* q = b + 16u + 32u * plane + within;
        const u32 sh = 2u * run;
        return make_float4(d * (float)((q[0] >> sh) & 3u) - m,
                           d * (float)((q[1] >> sh) & 3u) - m,
                           d * (float)((q[2] >> sh) & 3u) - m,
                           d * (float)((q[3] >> sh) & 3u) - m);
    }
};

/* Q3_K: 2-bit codes as Q2_K; the third bit is the INVERTED hmask bit
 * 4*plane+run of byte `within` (a clear bit subtracts 4). The sixteen 6-bit
 * scales (minus 32) keep low nibbles in scales[0..8] (low nibble for scales
 * 0..8, high nibble for 8..16) and top two bits in scales[8..12]. */
struct dit_wrows_q3k {
    const u8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u8* b = w + (u64)n * (k / 256u) * 110u + (u64)(kk / 256u) * 110u;
        const u32 i = kk % 256u;
        const u32 plane = i / 128u;
        const u32 run = (i % 128u) / 32u;
        const u32 within = i % 32u;
        const u32 is = 8u * plane + 2u * run + within / 16u;
        const u8* sc = b + 96;
        const u32 low = is < 8u ? ((u32)sc[is] & 0xFu) : ((u32)sc[is - 8u] >> 4u);
        const u32 high = ((u32)sc[8u + (is & 3u)] >> (2u * (is >> 2u))) & 3u;
        const float d = dit_f16(b + 108) * (float)((int)(low | (high << 4u)) - 32);
        const u8* q = b + 32u + 32u * plane + within;
        const u8* h = b + within;
        float v[4];
#pragma unroll
        for (u32 c = 0; c < 4u; ++c) {
            const int lo = (int)(((u32)q[c] >> (2u * run)) & 3u);
            const int set = (int)(((u32)h[c] >> (4u * plane + run)) & 1u);
            v[c] = d * (float)(lo - 4 + 4 * set);
        }
        return make_float4(v[0], v[1], v[2], v[3]);
    }
};

/* Q5_K: Q4_K's runs and scale/min packing plus a fifth bit: bit 2*chunk (low
 * nibbles) or 2*chunk+1 (high nibbles) of qh[l] adds 16 to the code. */
struct dit_wrows_q5k {
    const u8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u8* b = w + (u64)n * (k / 256u) * 176u + (u64)(kk / 256u) * 176u;
        const u32 i = kk % 256u;
        const u32 chunk = i / 64u;
        const u32 within = i % 64u;
        const u32 high_half = within >= 32u ? 1u : 0u;
        const u32 sub = 2u * chunk + high_half;
        u32 scale, min;
        dit_kq_scale_min(b + 4, sub, &scale, &min);
        const float d = dit_f16(b) * (float)scale;
        const float m = dit_f16(b + 2) * (float)min;
        const u32 l = within % 32u;
        const u8* q = b + 48u + 32u * chunk + l;
        const u8* qh = b + 16u + l;
        const u32 shift = high_half != 0u ? 4u : 0u;
        float v[4];
#pragma unroll
        for (u32 c = 0; c < 4u; ++c) {
            const u32 nib = ((u32)q[c] >> shift) & 0xFu;
            const u32 bit = ((u32)qh[c] >> (2u * chunk + high_half)) & 1u;
            v[c] = d * (float)(nib | (bit << 4u)) - m;
        }
        return make_float4(v[0], v[1], v[2], v[3]);
    }
};

/* Q8_0: weight = d * q. Rows are k/32 blocks; a 4-aligned kk never straddles. */
struct dit_wrows_q8_0 {
    const u8* w;
    u32 k;
    __device__ float4 load4(u32 n, u32 kk) const {
        const u8* b = w + (u64)n * (k / 32u) * 34u + (u64)(kk / 32u) * 34u;
        const float d = dit_f16(b);
        const i8* q = (const i8*)(b + 2u + kk % 32u);
        return make_float4(d * (float)q[0], d * (float)q[1], d * (float)q[2], d * (float)q[3]);
    }
};
