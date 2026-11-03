/* What the two low-precision prompt units share: exact scalar decodes, the
 * activation quantizers' rounding, and the `cp.async` / `ldmatrix` plumbing
 * their tiles are staged with.
 *
 * Not an upstream unit. Written for this repository.
 *
 * Everything that rounds is spelled out instruction by instruction, and the
 * directory builds without `--use_fast_math`, because the quantizers are held
 * bit for bit to a host restatement (`shiso-kerneltest`): a division that
 * became a reciprocal approximation, or a subnormal flushed to zero, would move
 * an `e2m1` code off its tie and show as a one-code disagreement in a test
 * whose whole point is that there are none.
 */
#pragma once

#include <stdint.h>

/* `e4m3` to `f32`, exact. The NaN bytes (`0x7F`, `0xFF`) come back NaN. */
__device__ __forceinline__ float lp_e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 0x0Fu;
    const uint32_t m = b & 0x07u;
    float v;
    if ((b & 0x7Fu) == 0x7Fu) {
        v = __int_as_float(0x7fffffff);
    } else if (e == 0u) {
        v = (float) m * 0.001953125f; /* m * 2^-9 */
    } else {
        v = __int_as_float((int) (((e + 120u) << 23) | (m << 20)));
    }
    return (b & 0x80u) ? -v : v;
}

/* `f32` to `e4m3`, round to nearest even, saturating at 448 (NaN stays NaN):
 * the hardware conversion, one value in the low byte. */
__device__ __forceinline__ uint32_t lp_to_e4m3(float x) {
    uint16_t r;
    asm("{\n\t.reg .b16 t;\n\tcvt.rn.satfinite.e4m3x2.f32 t, %1, %2;\n\tmov.b16 %0, t;\n\t}"
        : "=h"(r)
        : "f"(0.0f), "f"(x));
    return (uint32_t) r & 0xFFu;
}

/* Four of a block's sixteen activations as NVFP4 against the global scale
 * `gs`, given the block's `amax`:
 *
 *     sf   = e4m3_rn(min(amax * gs / 6, 448))
 *     code = the e2m1 value nearest |x| / (e4m3(sf) / gs), ties to even
 *
 * The nearest code is found the way the reference this path is matched to
 * finds it - against seven thresholds scaled by the dequantized step, each a
 * single rounded product - rather than by dividing every value, so a value
 * sitting exactly on a scaled midpoint lands where the reference puts it.
 * The sign is set only on a nonzero code. A block whose scale or global
 * rounds to zero is all zero codes. Value `i` lands in nibble `i` of the
 * result; the block's scale byte in `sbyte`. */
__device__ __forceinline__ uint32_t lp_fp4_codes4(const float *v, float amax, float gs,
                                                  uint32_t &sbyte) {
    const float sf = fminf(__fdiv_rn(__fmul_rn(amax, gs), 6.0f), 448.0f);
    sbyte = lp_to_e4m3(sf);
    const float qs = lp_e4m3(sbyte);
    uint32_t out = 0u;
    if (qs != 0.0f && gs != 0.0f) {
        const float vs = __fdiv_rn(qs, gs);
        const float t025 = __fmul_rn(vs, 0.25f);
        const float t075 = __fmul_rn(vs, 0.75f);
        const float t125 = __fmul_rn(vs, 1.25f);
        const float t175 = __fmul_rn(vs, 1.75f);
        const float t250 = __fmul_rn(vs, 2.5f);
        const float t350 = __fmul_rn(vs, 3.5f);
        const float t500 = __fmul_rn(vs, 5.0f);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float q = v[i];
            const float mag = fabsf(q);
            uint32_t n = 0u;
            if (mag > t025 && mag < t075) {
                n = 1u;
            } else if (mag >= t075 && mag <= t125) {
                n = 2u;
            } else if (mag > t125 && mag < t175) {
                n = 3u;
            } else if (mag >= t175 && mag <= t250) {
                n = 4u;
            } else if (mag > t250 && mag < t350) {
                n = 5u;
            } else if (mag >= t350 && mag <= t500) {
                n = 6u;
            } else if (mag > t500) {
                n = 7u;
            }
            if (n != 0u && q < 0.0f) {
                n |= 8u;
            }
            out |= n << (4 * i);
        }
    }
    return out;
}

/* A brain float widened exactly. */
__device__ __forceinline__ float lp_bf16(uint32_t bits) {
    return __uint_as_float(bits << 16);
}

/* `f32` to a brain float, round to nearest even (NaN stays NaN). */
__device__ __forceinline__ uint32_t lp_to_bf16(float x) {
    uint16_t r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(x));
    return r;
}

/* Four consecutive activations as floats, from `f32` or brain-float rows. */
__device__ __forceinline__ void lp_load4(const float *p, float *v) {
    const float4 t = *(const float4 *) p;
    v[0] = t.x;
    v[1] = t.y;
    v[2] = t.z;
    v[3] = t.w;
}

__device__ __forceinline__ void lp_load4(const uint16_t *p, float *v) {
    const uint2 t = *(const uint2 *) p;
    v[0] = lp_bf16(t.x & 0xFFFFu);
    v[1] = lp_bf16(t.x >> 16);
    v[2] = lp_bf16(t.y & 0xFFFFu);
    v[3] = lp_bf16(t.y >> 16);
}

__device__ __forceinline__ uint32_t lp_smem(const void *p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

/* `cp.async` of 16 / 8 bytes; the destination is zero-filled when `!valid`
 * (a source size of zero reads nothing). */
__device__ __forceinline__ void lp_cp16(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(lp_smem(dst)), "l"(src),
                 "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void lp_cp8(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(lp_smem(dst)), "l"(src),
                 "r"(valid ? 8 : 0));
}

__device__ __forceinline__ void lp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void lp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

__device__ __forceinline__ void lp_ldsm4(uint32_t *r, uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

/* Byte offset of 16-byte chunk `c` (0..3) of row `r` in a tile of 64-byte
 * rows, swizzled so the eight rows one `ldmatrix` phase reads sit on eight
 * distinct bank groups. */
__device__ __forceinline__ uint32_t lp_off64(uint32_t r, uint32_t c) {
    return r * 64u + ((c ^ ((r >> 1) & 3u)) << 4);
}

/* The same for 128-byte rows, chunks 0..7. */
__device__ __forceinline__ uint32_t lp_off128(uint32_t r, uint32_t c) {
    return r * 128u + ((c ^ (r & 7u)) << 4);
}
