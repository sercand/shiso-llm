/* The `modelopt` codecs of this family's NVIDIA export: the routed stages over
 * an NVFP4 expert bank (the target's) and a block-scaled FP8 one (the draft
 * head's), and the decode of FP8 and NVFP4 n-gram rows.
 *
 * Not an upstream unit. Every other file in this directory is a byte copy of
 * the source the family's kernels came from, and these entries are this
 * repository's own - which is why they are a module of their own
 * (`q4e_modelopt`) rather than lines added to `moe.cu` or `ple.cu`. Their
 * argument lists are the Metal set's `q4e/fp8.metal` ones, with the NVFP4
 * bank's two extra arrays where the FP8 bank has its block scales, so the
 * host plans one shape of stage for every codec.
 *
 * # Every decode here survives a flushed subnormal
 *
 * This module builds with the tree's `--use_fast_math`, which flushes
 * subnormals to zero. An `e4m3` scale byte with a zero exponent field IS a
 * subnormal (`m * 2^-9`), and this export's group scales reach them; a decode
 * that formed it as a subnormal `f32` and rescaled would read zero. So the
 * zero-exponent case is computed as a normal `f32` constant times the
 * mantissa, and every other case is assembled directly at its own exponent.
 *
 * # NVFP4, as the rest of the tree reads it
 *
 * `w[r][k] = e2m1(nibble) * (e4m3(scale[r][k / 16]) * global)`, the low nibble
 * of byte `k / 2` holding the even weight. The group scale and the global are
 * multiplied together first, once a group, and a group's sixteen products are
 * summed before that scale is applied - `kernels/cuda/qwen35moe`'s order.
 */

#include <stdint.h>

#define Q4E_NVFP4_GROUP 16u
#define Q4E_NVFP4_ROWS 8u

/* `e4m3` to `f32`, exact and safe under flush-to-zero. */
__device__ __forceinline__ float q4e_mo_e4m3(uint32_t b) {
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

/* `e2m1` to `f32`, exact: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6. */
__device__ __forceinline__ float q4e_mo_e2m1(uint32_t c) {
    const uint32_t e = (c >> 1) & 3u;
    const uint32_t m = c & 1u;
    const float v = e ? __int_as_float((int) (((e + 126u) << 23) | (m << 22)))
                      : (m ? 0.5f : 0.0f);
    return (c & 8u) ? -v : v;
}

__device__ __forceinline__ float q4e_mo_warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        v += __shfl_xor_sync(0xffffffffu, v, o);
    }
    return v;
}

/* Eight packed weights against eight activations, from zero, in order. */
__device__ __forceinline__ float q4e_mo_word_dot(uint32_t w, float4 a, float4 b) {
    float s = 0.0f;
    s = fmaf(q4e_mo_e2m1(w & 0xFu), a.x, s);
    s = fmaf(q4e_mo_e2m1((w >> 4) & 0xFu), a.y, s);
    s = fmaf(q4e_mo_e2m1((w >> 8) & 0xFu), a.z, s);
    s = fmaf(q4e_mo_e2m1((w >> 12) & 0xFu), a.w, s);
    s = fmaf(q4e_mo_e2m1((w >> 16) & 0xFu), b.x, s);
    s = fmaf(q4e_mo_e2m1((w >> 20) & 0xFu), b.y, s);
    s = fmaf(q4e_mo_e2m1((w >> 24) & 0xFu), b.z, s);
    s = fmaf(q4e_mo_e2m1((w >> 28) & 0xFu), b.w, s);
    return s;
}

/* One lane's share of the dot of one NVFP4 row with `x`: lane `l` takes
 * groups `l`, `l + 32`, ... The widths this family stores (2560 and 640) are
 * whole groups, and a row starts at a multiple of eight bytes, so a group is
 * one eight-byte load and four sixteen-byte ones. */
__device__ __forceinline__ float q4e_mo_nvfp4_row(
        const uint8_t *__restrict__ w, const uint8_t *__restrict__ s, float global,
        const float *__restrict__ x, uint32_t in_dim, uint32_t lane) {
    float acc = 0.0f;
    const uint32_t groups = in_dim / Q4E_NVFP4_GROUP;
    for (uint32_t g = lane; g < groups; g += 32u) {
        const uint2 word = ((const uint2 *) w)[g];
        const float scale = q4e_mo_e4m3(s[g]) * global;
        const float4 *xg = (const float4 *) (x + (uint64_t) g * Q4E_NVFP4_GROUP);
        const float part = q4e_mo_word_dot(word.x, xg[0], xg[1])
                         + q4e_mo_word_dot(word.y, xg[2], xg[3]);
        acc = fmaf(part, scale, acc);
    }
    return acc;
}

#include <cuda_fp16.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * The routed stages of a step or a verify pass, a warp per output row.
 *
 * A step routes one to a few columns to each expert, so these are weight
 * streams: the bytes of the distinct experts touched are the whole cost, and
 * the work per byte is one decode and a few multiply-adds. What they keep
 * from a plain warp-per-row loop is its arithmetic, operation for operation;
 * what they change is how the bytes arrive:
 *
 * - A warp owns the same row of gate and up, or four rows of down taken one
 *   at a time, and issues every weight load of a row - codes and scales -
 *   before the first multiply. The bookkeeping chain in front of them (active
 *   list, bounds, slot map: three dependent reads) is paid once for four down
 *   rows instead of once for each 360-byte one, which is where the old down
 *   stage lost its bandwidth.
 * - A row's weights sit in registers while its expert's columns are walked,
 *   so a token group that shares an expert reads that expert once, and each
 *   column's activations are loaded once for both gate and up.
 * - A row's scale bytes are read as one word a lane and handed out by
 *   shuffle: whole lines instead of 32-byte pieces.
 * - The four-bit codes are widened through half precision bit placement
 *   rather than a branchy per-code decode: eight codes in ~30 integer and
 *   half operations instead of ~80.
 * - Blocks are four warps, and the registers stay near fifty, so many blocks
 *   share an SM: holding several rows' weights at once (more bytes in flight
 *   a warp) measured slower than the occupancy it cost.
 *
 * Every column is still the same fixed sequence - lane `l` takes groups `l`,
 * `l + 32`, ...; a group's sixteen products summed from zero in order and then
 * scaled; the lanes folded by one butterfly - so a row's result depends on
 * its token and expert alone, never on how many columns share the launch: a
 * verify row equals the single step bit for bit. The sums are bit for bit the
 * previous one-row-a-warp kernels' too.
 *
 * GB10, the kernel bench's Zipf routing over 512 experts, five tokens (a
 * speculative verify), median of three: gate+up 399 -> 388 us (~220 GB/s of
 * distinct expert bytes, 80% of peak), down 270 -> 212 us (160 -> 205 GB/s).
 * ------------------------------------------------------------------------ */

/* Rows a warp owns in the step stages, and warps a block. */
#define Q4E_NVFP4_GU_ROWS 1u
#define Q4E_NVFP4_DN_ROWS 4u
#define Q4E_NVFP4_WARPS 4u
/* Groups a lane holds in registers: 32 lanes x 3 groups x 16 weights covers a
 * contraction up to 1,536 wide; a wider one - the hidden width here - takes
 * the reload path, which re-reads the row once per column. */
#define Q4E_NVFP4_LANE_GROUPS 3u
/* The same for the held entry: five groups a lane covers the 2,560-wide
 * hidden row. */
#define Q4E_NVFP4_HELD_GROUPS 5u

/* Eight codes of `w` widened exactly to floats, `v[j]` the code in bits
 * `4j .. 4j + 3`. Codes `j` and `j + 4` share a half2: the sign goes to the
 * half's sign bit and the three magnitude bits to the exponent's low two bits
 * and the mantissa's top bit, which reads as the code's value times 2^-14 -
 * a subnormal half for the two smallest magnitudes, which half arithmetic
 * keeps (flush-to-zero applies to single precision only). One multiply by
 * 2^14 makes every value exact and normal, and the widening to single
 * precision is exact. */
__device__ __forceinline__ void q4e_mo_e2m1x8(uint32_t w, float *v) {
    const __half2 k = __float2half2_rn(16384.0f);
#pragma unroll
    for (uint32_t j = 0; j < 4u; ++j) {
        const uint32_t t = w >> (4u * j);
        const uint32_t h = ((t << 9) & 0x0E000E00u) | ((t << 12) & 0x80008000u);
        __half2 hh;
        memcpy(&hh, &h, 4);
        const float2 f = __half22float2(__hmul2(hh, k));
        v[j] = f.x;
        v[j + 4u] = f.y;
    }
}

/* `q4e_mo_word_dot` over codes widened by `q4e_mo_e2m1x8`: the same values,
 * the same order, the same bits. */
__device__ __forceinline__ float q4e_mo_word_dot_fast(uint32_t w, float4 a, float4 b) {
    float v[8];
    q4e_mo_e2m1x8(w, v);
    float s = 0.0f;
    s = fmaf(v[0], a.x, s);
    s = fmaf(v[1], a.y, s);
    s = fmaf(v[2], a.z, s);
    s = fmaf(v[3], a.w, s);
    s = fmaf(v[4], b.x, s);
    s = fmaf(v[5], b.y, s);
    s = fmaf(v[6], b.z, s);
    s = fmaf(v[7], b.w, s);
    return s;
}

/* One step stage over `M` matrices of `R` rows each, a warp's share.
 * `w[m]`, `s[m]` point at matrix `m`'s first row of this warp, `global[m]` is
 * its slot's scale and `out[m]` its output; `row` is the first row's index
 * and `rows` how many of the `R` exist. Columns `c0 .. c1` of `sorted` read
 * activation row `col / div`. */
/* `silu(gate) * up`, exactly as `q4e_swiglu_kernel` (moe.cu, built with the
 * same flags) computes it from the two stored rows. */
__device__ __forceinline__ float q4e_mo_swiglu(float g, float u) {
    return (g / (1.0f + __expf(-g))) * u;
}

/* Stores one row's `M` sums for column `col`: each matrix's own output, or
 * with `ACT` (two matrices, gate then up) the activated product into
 * `out[0]`, the value the activation launch would have written from the two
 * stored rows. Every lane holds the folded sums; lane 0 stores. */
template <uint32_t M, bool ACT>
__device__ __forceinline__ void q4e_mo_store_row(
        float *const *out, const float *v, uint64_t at, uint32_t lane) {
    if (lane != 0u) {
        return;
    }
    if (ACT) {
        out[0][at] = q4e_mo_swiglu(v[0], v[1]);
    } else {
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
            out[m][at] = v[m];
        }
    }
}

template <uint32_t M, uint32_t R, uint32_t P, uint32_t G, bool ACT = false>
__device__ __forceinline__ void q4e_mo_nvfp4_step_rows(
        float *const *out, const uint8_t *const *w, const uint8_t *const *s,
        const float *global, const float *__restrict__ x, const int *__restrict__ sorted,
        int c0, int c1, uint32_t div, uint32_t row, uint32_t rows, uint32_t in_dim,
        uint32_t out_dim, uint32_t lane) {
    const uint32_t groups = in_dim / Q4E_NVFP4_GROUP;
    const uint64_t packed = in_dim / 2u;
    if (groups > 32u * G || (groups & 3u) != 0u) {
        /* Too wide to hold: re-read the rows for every column, in the same
         * order, from the cache. */
        for (int c = c0; c < c1; ++c) {
            const uint32_t col = (uint32_t) sorted[c];
            const float *xr = x + (uint64_t) (col / div) * in_dim;
#pragma unroll
            for (uint32_t r = 0; r < R; ++r) {
                if (r < rows) {
                    float v[M];
#pragma unroll
                    for (uint32_t m = 0; m < M; ++m) {
                        v[m] = q4e_mo_warp_sum(q4e_mo_nvfp4_row(
                                w[m] + r * packed, s[m] + r * groups, global[m], xr, in_dim,
                                lane));
                    }
                    q4e_mo_store_row<M, ACT>(out, v, (uint64_t) col * out_dim + row + r, lane);
                }
            }
        }
        return;
    }
    /* `P` rows at a time: their weight loads all issued before the first
     * multiply, then every column against them. */
#pragma unroll 1
    for (uint32_t r0 = 0; r0 < rows; r0 += P) {
        uint2 code[M][P][G];
        float scale[M][P][G];
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
            for (uint32_t p = 0; p < P; ++p) {
#pragma unroll
                for (uint32_t q = 0; q < G; ++q) {
                    const uint32_t g = lane + 32u * q;
                    const uint32_t r = r0 + p;
                    code[m][p][q] = (r < rows && g < groups)
                                            ? ((const uint2 *) (w[m] + r * packed))[g]
                                            : make_uint2(0u, 0u);
                }
            }
        }
#pragma unroll
        for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
            for (uint32_t p = 0; p < P; ++p) {
                /* The row's scale bytes as one word a lane - whole lines,
                 * not a byte a lane - handed to the lanes that use them. */
                const uint32_t r = r0 + p;
                /* Word `v` of lane `l` holds groups `4 (l + 32 v) .. + 3`:
                 * one word a lane covers 128 groups, and a wider row (the
                 * hidden width, 160) takes a second. */
                constexpr uint32_t W = (G + 3u) / 4u;
                uint32_t word[W];
#pragma unroll
                for (uint32_t v = 0; v < W; ++v) {
                    const uint32_t at = lane + 32u * v;
                    word[v] = (r < rows && at * 4u < groups)
                                      ? ((const uint32_t *) (s[m] + r * groups))[at]
                                      : 0u;
                }
#pragma unroll
                for (uint32_t q = 0; q < G; ++q) {
                    const uint32_t g = lane + 32u * q;
                    /* Group `g` is byte `g % 4` of word `q / 4` of lane
                     * `(g / 4) % 32`. */
                    const uint32_t b = __shfl_sync(0xffffffffu, word[q >> 2], (g >> 2) & 31u);
                    scale[m][p][q] = (r < rows && g < groups)
                                             ? q4e_mo_e4m3((b >> (8u * (g & 3u))) & 0xFFu)
                                                       * global[m]
                                             : 0.0f;
                }
            }
        }
        for (int c = c0; c < c1; ++c) {
            const uint32_t col = (uint32_t) sorted[c];
            const float *xr = x + (uint64_t) (col / div) * in_dim;
            float acc[M][P];
#pragma unroll
            for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
                for (uint32_t p = 0; p < P; ++p) {
                    acc[m][p] = 0.0f;
                }
            }
#pragma unroll
            for (uint32_t q = 0; q < G; ++q) {
                const uint32_t g = lane + 32u * q;
                if (g < groups) {
                    const float4 *xg = (const float4 *) (xr + (uint64_t) g * Q4E_NVFP4_GROUP);
                    const float4 x0 = xg[0];
                    const float4 x1 = xg[1];
                    const float4 x2 = xg[2];
                    const float4 x3 = xg[3];
#pragma unroll
                    for (uint32_t m = 0; m < M; ++m) {
#pragma unroll
                        for (uint32_t p = 0; p < P; ++p) {
                            /* Opaque to the optimiser, so the decode stays
                             * in the column loop instead of being hoisted
                             * into sixteen live registers a word. */
                            uint32_t lo = code[m][p][q].x;
                            uint32_t hi = code[m][p][q].y;
                            asm volatile("" : "+r"(lo), "+r"(hi));
                            const float part = q4e_mo_word_dot_fast(lo, x0, x1)
                                             + q4e_mo_word_dot_fast(hi, x2, x3);
                            acc[m][p] = fmaf(part, scale[m][p][q], acc[m][p]);
                        }
                    }
                }
            }
#pragma unroll
            for (uint32_t p = 0; p < P; ++p) {
                float v[M];
#pragma unroll
                for (uint32_t m = 0; m < M; ++m) {
                    v[m] = q4e_mo_warp_sum(acc[m][p]);
                }
                if (r0 + p < rows) {
                    q4e_mo_store_row<M, ACT>(out, v, (uint64_t) col * out_dim + row + r0 + p,
                                             lane);
                }
            }
        }
    }
}

/* `q4e_mo_nvfp4_step_rows<1, R, 1, G>` - one matrix, `R` rows taken one at a
 * time - with the next row's codes and scale words requested before this
 * row's columns are summed, so a warp's rows are one stream instead of `R`
 * round trips. The same values in the same order: only the loads move.
 *
 * Measured on a GB10 (graph of 48 layers' down stages, ten random experts of
 * 512, compute_80 JIT, three runs): 44.4-45.0 -> 42.7-43.9 us a layer.
 * Holding all four rows' loads at once (`P = 4`) measured 49.4-50.5 us: the
 * registers it takes cost more blocks than the bytes in flight bought. */
template <uint32_t R, uint32_t G>
__device__ __forceinline__ void q4e_mo_nvfp4_step_rows_pipe(
        float *__restrict__ out, const uint8_t *__restrict__ w, const uint8_t *__restrict__ s,
        float global, const float *__restrict__ x, const int *__restrict__ sorted, int c0,
        int c1, uint32_t div, uint32_t row, uint32_t rows, uint32_t in_dim, uint32_t out_dim,
        uint32_t lane) {
    const uint32_t groups = in_dim / Q4E_NVFP4_GROUP;
    const uint64_t packed = in_dim / 2u;
    if (groups > 32u * G || (groups & 3u) != 0u) {
        const uint8_t *const wm[1] = {w};
        const uint8_t *const sm[1] = {s};
        float *const om[1] = {out};
        q4e_mo_nvfp4_step_rows<1u, R, 1u, G>(om, wm, sm, &global, x, sorted, c0, c1, div, row,
                                             rows, in_dim, out_dim, lane);
        return;
    }
    constexpr uint32_t W = (G + 3u) / 4u;
    uint2 code[2][G];
    uint32_t word[2][W];
    /* Row `r`'s codes and scale words, as `q4e_mo_nvfp4_step_rows` loads them. */
    auto load = [&](uint32_t r, uint2 *c, uint32_t *wd) {
#pragma unroll
        for (uint32_t q = 0; q < G; ++q) {
            const uint32_t g = lane + 32u * q;
            c[q] = (r < rows && g < groups) ? ((const uint2 *) (w + r * packed))[g]
                                            : make_uint2(0u, 0u);
        }
#pragma unroll
        for (uint32_t v = 0; v < W; ++v) {
            const uint32_t at = lane + 32u * v;
            wd[v] = (r < rows && at * 4u < groups) ? ((const uint32_t *) (s + r * groups))[at] : 0u;
        }
    };
    load(0u, code[0], word[0]);
#pragma unroll
    for (uint32_t r = 0; r < R; ++r) {
        const uint32_t cur = r & 1u;
        if (r + 1u < R) {
            load(r + 1u, code[cur ^ 1u], word[cur ^ 1u]);
        }
        if (r >= rows) {
            continue;
        }
        float scale[G];
#pragma unroll
        for (uint32_t q = 0; q < G; ++q) {
            const uint32_t g = lane + 32u * q;
            const uint32_t b = __shfl_sync(0xffffffffu, word[cur][q >> 2], (g >> 2) & 31u);
            scale[q] = g < groups ? q4e_mo_e4m3((b >> (8u * (g & 3u))) & 0xFFu) * global : 0.0f;
        }
        for (int c = c0; c < c1; ++c) {
            const uint32_t col = (uint32_t) sorted[c];
            const float *xr = x + (uint64_t) (col / div) * in_dim;
            float acc = 0.0f;
#pragma unroll
            for (uint32_t q = 0; q < G; ++q) {
                const uint32_t g = lane + 32u * q;
                if (g < groups) {
                    const float4 *xg = (const float4 *) (xr + (uint64_t) g * Q4E_NVFP4_GROUP);
                    const float4 x0 = xg[0];
                    const float4 x1 = xg[1];
                    const float4 x2 = xg[2];
                    const float4 x3 = xg[3];
                    uint32_t lo = code[cur][q].x;
                    uint32_t hi = code[cur][q].y;
                    asm volatile("" : "+r"(lo), "+r"(hi));
                    const float part = q4e_mo_word_dot_fast(lo, x0, x1)
                                     + q4e_mo_word_dot_fast(hi, x2, x3);
                    acc = fmaf(part, scale[q], acc);
                }
            }
            const float v = q4e_mo_warp_sum(acc);
            if (lane == 0u) {
                out[(uint64_t) col * out_dim + row + r] = v;
            }
        }
    }
}

/* Gate and up for one active expert's columns, `Q4E_NVFP4_GU_ROWS` rows of
 * each a warp: the body of both gate-and-up entries. `out[0]` and `out[1]`
 * are the gate and up outputs, or with `ACT` `out[0]` is the activated
 * intermediate. */
template <uint32_t G, bool ACT>
__device__ __forceinline__ void q4e_mo_nvfp4_gate_up(
        float *const *out,
        const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    /* Both reads at once: the count and this block's entry are independent. */
    const int n_active = active[n_expert];
    const int expert = active[blockIdx.y];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = (blockIdx.x * Q4E_NVFP4_WARPS + (threadIdx.x >> 5)) * Q4E_NVFP4_GU_ROWS;
    if ((int) blockIdx.y >= n_active || row >= out_dim) {
        return;
    }
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint64_t slot = map[expert];
    if (c1 <= c0) {
        return;
    }
    const uint64_t first = slot * out_dim + row;
    const uint8_t *w[2] = {gate_w + first * (in_dim / 2u), up_w + first * (in_dim / 2u)};
    const uint8_t *s[2] = {gate_s + first * (in_dim / Q4E_NVFP4_GROUP),
                           up_s + first * (in_dim / Q4E_NVFP4_GROUP)};
    const float g[2] = {gate_g[slot], up_g[slot]};
    q4e_mo_nvfp4_step_rows<2u, Q4E_NVFP4_GU_ROWS, 1u, G, ACT>(
            out, w, s, g, x, sorted, c0, c1, n_used, row,
            min(Q4E_NVFP4_GU_ROWS, out_dim - row), in_dim, out_dim, lane);
}

/* Gate and up for one active expert's columns: `Q4E_NVFP4_GU_ROWS` rows of
 * each a warp, `Q4E_NVFP4_WARPS` warps a block, the expert's span of the sorted column
 * list walked with the rows held. `map[e]` is the slot expert `e`'s matrices
 * sit at; a column's activation row is its token's, `col / n_used`.
 *
 * Buffers: out_gate, out_up, gate_w, gate_s, gate_g, up_w, up_s, up_g, x,
 * sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used. Grid
 * `(ceil(out_dim / 4), min(columns, n_expert))`, 128 threads. */
extern "C" __global__ void __launch_bounds__(32u * Q4E_NVFP4_WARPS) q4e_nvfp4_experts_gate_up_kernel(
        float *out_gate, float *out_up,
        const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    float *const out[2] = {out_gate, out_up};
    q4e_mo_nvfp4_gate_up<Q4E_NVFP4_LANE_GROUPS, false>(
            out, gate_w, gate_s, gate_g, up_w, up_s, up_g, x, sorted, bounds, active, map,
            in_dim, out_dim, n_expert, n_used);
}

/* The same rows with the activation applied as they are stored: each
 * column's `silu(gate) * up` into `out_mid`, `[column, out_dim]` - the bits
 * `q4e_swiglu_kernel` writes from the two rows `q4e_nvfp4_experts_gate_up_kernel`
 * stores - without the two intermediate rows or the activation launch.
 * The decode step's entry: one column an expert.
 *
 * Buffers: out_mid, gate_w, gate_s, gate_g, up_w, up_s, up_g, x, sorted,
 * bounds, active, map, in_dim, out_dim, n_expert, n_used. Grid as the
 * unactivated entry's. */
extern "C" __global__ void __launch_bounds__(32u * Q4E_NVFP4_WARPS) q4e_nvfp4_experts_gate_up_act_kernel(
        float *out_mid,
        const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    float *const out[2] = {out_mid, out_mid};
    q4e_mo_nvfp4_gate_up<Q4E_NVFP4_LANE_GROUPS, true>(
            out, gate_w, gate_s, gate_g, up_w, up_s, up_g, x, sorted, bounds, active, map,
            in_dim, out_dim, n_expert, n_used);
}

/* The activated rows with a 2560-wide row held in registers (five groups a
 * lane) while an expert's columns are walked, rather than re-read per
 * column: the verify's entry. The same sums - the register and the reload
 * paths are one order - so a verify row is still the decode row's bits.
 *
 * Measured on a GB10 (kernel bench, Zipf routing over 512 experts, wall
 * clock, medians of three): five tokens 391-427 -> 378-388 us; one token
 * 84-87 -> 88-89 us, which is why the decode keeps the reload path.
 * Arguments and grid as `q4e_nvfp4_experts_gate_up_act_kernel`'s. */
extern "C" __global__ void __launch_bounds__(32u * Q4E_NVFP4_WARPS) q4e_nvfp4_experts_gate_up_act_held_kernel(
        float *out_mid,
        const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    float *const out[2] = {out_mid, out_mid};
    q4e_mo_nvfp4_gate_up<Q4E_NVFP4_HELD_GROUPS, true>(
            out, gate_w, gate_s, gate_g, up_w, up_s, up_g, x, sorted, bounds, active, map,
            in_dim, out_dim, n_expert, n_used);
}

/* The routed down projection over an NVFP4 bank: each column reads its own row
 * of the activated intermediate and writes `[column, out_dim]`;
 * `Q4E_NVFP4_DN_ROWS` rows a warp, taken one at a time with the next one's
 * loads in flight (`q4e_mo_nvfp4_step_rows_pipe`; two or four held at once
 * measured 3-6% slower on a GB10).
 *
 * Buffers: out, down_w, down_s, down_g, mid, sorted, bounds, active, map,
 * in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 16), min(columns,
 * n_expert))`, 128 threads. */
extern "C" __global__ void __launch_bounds__(32u * Q4E_NVFP4_WARPS) q4e_nvfp4_experts_down_kernel(
        float *out, const uint8_t *down_w, const uint8_t *down_s, const float *down_g,
        const float *mid, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    (void) n_used;
    const int n_active = active[n_expert];
    const int expert = active[blockIdx.y];
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t row = (blockIdx.x * Q4E_NVFP4_WARPS + (threadIdx.x >> 5)) * Q4E_NVFP4_DN_ROWS;
    if ((int) blockIdx.y >= n_active || row >= out_dim) {
        return;
    }
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint64_t slot = map[expert];
    if (c1 <= c0) {
        return;
    }
    const uint64_t first = slot * out_dim + row;
    q4e_mo_nvfp4_step_rows_pipe<Q4E_NVFP4_DN_ROWS, 2u>(
            out, down_w + first * (in_dim / 2u), down_s + first * (in_dim / Q4E_NVFP4_GROUP),
            down_g[slot], mid, sorted, c0, c1, 1u, row, min(Q4E_NVFP4_DN_ROWS, out_dim - row),
            in_dim, out_dim, lane);
}
/* ---------------------------------------------------------------------------
 * The routed stages of a prompt pass, on tensor cores.
 *
 * A prompt routes tens of columns to every expert (a median near twenty at a
 * 2,048-token chunk, a few experts past three hundred, some none), and every
 * expert of the layer is read: 1.4 GB of NVFP4 weights a layer, about 6 ms of
 * DRAM on a GB10 whatever the arithmetic does. So the shape here is a weight
 * stream that computes in its shadow:
 *
 * - A block owns 256 weight rows - for gate and up the same 128 rows of both,
 *   so one staged activation tile feeds both products and each activation
 *   row is gathered five times a layer, not twenty - and walks its expert's
 *   columns 64 at a time, the column tiles of one expert split over
 *   `Q4E_MG_SPLIT` neighbouring blocks so a heavy expert is not one long
 *   block, while its weight tile is still read from DRAM about once.
 * - Weights arrive packed, by `cp.async`, a stage of 256 contraction elements
 *   ahead: 128 code bytes of every row, whole lines. The first version staged
 *   32 bytes of each of 128 rows a step, and the DRAM served those scattered
 *   pieces at about half the rate it streams lines (measured with the
 *   arithmetic removed: 7.2 ms against 4.3 for the same bytes).
 * - Activations arrive `Q4E_MG_AHEAD` steps ahead through registers and are
 *   rounded to halves on the way into shared memory, where `ldmatrix` reads
 *   them as `mma` fragments. One step ahead left every step waiting out a
 *   gathered read.
 * - Each warp decodes its own 16 rows straight into `mma.sync` m16n8k16 A
 *   fragments: a k16 step is exactly one 16-weight group, so a fragment
 *   register is one code byte under one group scale.
 * - Column groups of eight past the end of an expert's list are skipped, so
 *   an expert of twenty columns pays for twenty-four, not sixty-four.
 * - The output tile leaves through shared memory as whole lines.
 *
 * The weights decode to halves EXACTLY, and are carried at 2^-8 of their
 * value: an `e2m1` code placed into a half's bit pattern reads as its value
 * times 2^-14 (subnormal for the two smallest), an `e4m3` scale placed the
 * same way reads as its value times 2^-8 and is lifted by 2^14, and their
 * product - at most six significant bits, between 2^-18 and 10.5 - is exact.
 * The finished single-precision sum is lifted by 2^8 and only then times the
 * per-expert global, which at around 1e-4 would have pushed products into the
 * subnormals. Every product and partial sum is the unscaled one times 2^-8,
 * so the result is bit for bit the one of weights decoded at full value. The
 * activations are rounded to halves, as the dense half-precision prompt tier
 * rounds them.
 *
 * A column's sum is a fixed sequence of 16-wide steps in contraction order,
 * whatever else shares its tile, so a prompt's rows do not depend on how many
 * columns its chunk routed to the same expert.
 *
 * GB10, the kernel bench's Zipf routing over 512 experts, a 2,048-token chunk,
 * median of three: gate+up 11.7 -> 5.7 ms, down 11.6 -> 3.6 ms (both now
 * within about 15% of their DRAM floor: weights, activations and outputs at
 * ~230 GB/s); a 4,096-token chunk 7.5 and 5.1 ms.
 * ------------------------------------------------------------------------ */

/* Weight rows a block covers, and the half of them a matrix contributes. */
#define Q4E_MG_BM 256u
#define Q4E_MG_HALF 128u
#define Q4E_MG_BN 64u
/* Contraction one weight stage covers - 128 code bytes a row, a whole line -
 * and one activation step. */
#define Q4E_MG_SK 256u
#define Q4E_MG_BK 64u
#define Q4E_MG_THREADS 512u
/* Blocks one (expert, row tile) is split over, column tile `z` of every
 * `Q4E_MG_SPLIT` going to block `z`. */
#define Q4E_MG_SPLIT 4u
/* Halves a staged activation row takes: the 64 of a step plus 8, so the eight
 * rows of an `ldmatrix` phase land on distinct banks. */
#define Q4E_MG_BLD (Q4E_MG_BK + 8u)
/* Floats a staged output column takes: 256 rows plus 4, so the fragment
 * stores of a warp land on distinct banks. */
#define Q4E_MG_OLD (Q4E_MG_BM + 4u)
/* Dynamic shared memory a block takes: `sizeof(q4e_mo_gemm_smem)`. */
#define Q4E_MG_SMEM 92416u

__device__ __forceinline__ void q4e_mo_mma(float *c, const uint32_t *a, const uint32_t *b) {
    asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
            : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ uint32_t q4e_mo_smem(const void *p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

/* `bytes` (4 or 16) from `src` to shared `dst`; zeros when `!valid`. */
__device__ __forceinline__ void q4e_mo_cp_async16(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(q4e_mo_smem(dst)),
                 "l"(src), "r"(valid ? 16 : 0));
}

__device__ __forceinline__ void q4e_mo_cp_async4(void *dst, const void *src, bool valid) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(q4e_mo_smem(dst)),
                 "l"(src), "r"(valid ? 4 : 0));
}

__device__ __forceinline__ void q4e_mo_cp_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void q4e_mo_cp_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

/* An `e4m3` byte placed into a half's bit pattern: its value times 2^-8,
 * exact (the subnormal bytes become subnormal halves). The NaN byte becomes a
 * half NaN, as the single-precision decode makes it. */
__device__ __forceinline__ uint32_t q4e_mo_e4m3_tiny(uint32_t b) {
    return ((b & 0x7Fu) == 0x7Fu) ? 0x7E00u : (((b & 0x7Fu) << 7) | ((b & 0x80u) << 8));
}

/* One code byte as an A fragment register: the low code at the lower k, as
 * halves times 2^-14, times `scale`. */
__device__ __forceinline__ uint32_t q4e_mo_a_frag(uint32_t byte, __half2 scale) {
    const uint32_t t = byte | (byte << 12);
    const uint32_t h = ((t << 9) & 0x0E000E00u) | ((t << 12) & 0x80008000u);
    __half2 hh;
    memcpy(&hh, &h, 4);
    const __half2 r = __hmul2(hh, scale);
    uint32_t out;
    memcpy(&out, &r, 4);
    return out;
}

/* One half of a block's weight rows: `Q4E_MG_HALF` rows of one matrix. */
struct q4e_mo_half {
    const uint8_t *w; /* the matrix's slot base */
    const uint8_t *s;
    float global;
    float *out;
    uint32_t row0; /* first row of the half */
};

/* A block's shared memory, `Q4E_MG_SMEM` bytes of it dynamic. The weight ring
 * and the activation steps are dead once the contraction ends, and the output
 * tile is staged over them. */
struct q4e_mo_gemm_smem {
    union {
        struct {
            /* Two weight stages, a row's 128 code bytes each, its eight
             * 16-byte granules stored at `granule ^ (row & 7)` so the eight
             * rows a fragment read touches sit on distinct banks. */
            uint8_t codes[2][Q4E_MG_BM][Q4E_MG_SK / 2u];
            uint8_t scales[2][Q4E_MG_BM][Q4E_MG_SK / Q4E_NVFP4_GROUP];
            __half b[2][Q4E_MG_BN][Q4E_MG_BLD];
        } k;
        float out[Q4E_MG_BN][Q4E_MG_OLD];
    };
    int cols[Q4E_MG_BN];
};

/* Activation steps a thread holds in registers ahead of their use. */
#define Q4E_MG_AHEAD 2u

/* One block's work: rows `h[0].row0 ..+HALF` of one matrix and
 * `h[1].row0 ..+HALF` of another (or the same), every `Q4E_MG_SPLIT`th
 * 64-column tile of the expert's columns `c0 .. c1` starting at tile `z`. `x`
 * is read at row `col / div`. `in_dim` is a multiple of 128. */
__device__ __forceinline__ void q4e_mo_nvfp4_gemm_block(
        const q4e_mo_half *h, const float *__restrict__ x, const int *__restrict__ sorted,
        int c0, int c1, uint32_t div, uint32_t in_dim, uint32_t out_dim, uint32_t z) {
    extern __shared__ __align__(16) uint8_t q4e_mo_dyn_smem[];
    q4e_mo_gemm_smem &sm = *reinterpret_cast<q4e_mo_gemm_smem *>(q4e_mo_dyn_smem);

    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5;
    const uint32_t gid = lane >> 2;
    const uint32_t tig = lane & 3u;
    const uint64_t packed = in_dim / 2u;
    const uint64_t groups = in_dim / Q4E_NVFP4_GROUP;
    const uint32_t steps = in_dim / Q4E_MG_BK;
    const uint32_t supers = (in_dim + Q4E_MG_SK - 1u) / Q4E_MG_SK;

    /* This thread's share of a weight stage, fixed for the block: code rows
     * `c_row + 64 i` - passes 0 and 1 in the first half, 2 and 3 in the
     * second - eight threads a row, so one instruction asks for whole 128-byte
     * lines (rows read 32 bytes at a time ran the DRAM at half speed); and
     * scale rows `s_row + 128 i`, one pass a half, in 4-byte pieces. */
    const uint32_t c_part = tid & 7u;
    const uint32_t c_row = tid >> 3;
    const uint32_t s_part = tid & 3u;
    const uint32_t s_row = tid >> 2;
    const uint8_t *c_src[4];
    bool c_ok[4];
#pragma unroll
    for (uint32_t i = 0; i < 4u; ++i) {
        const q4e_mo_half &hh = h[i / 2u];
        const uint32_t grow = hh.row0 + c_row + 64u * (i & 1u);
        c_ok[i] = grow < out_dim;
        c_src[i] = hh.w + (uint64_t) (c_ok[i] ? grow : 0u) * packed + c_part * 16u;
    }
    const uint8_t *s_src[2];
    bool s_ok[2];
#pragma unroll
    for (uint32_t i = 0; i < 2u; ++i) {
        const q4e_mo_half &hh = h[i];
        const uint32_t grow = hh.row0 + s_row;
        s_ok[i] = grow < out_dim;
        s_src[i] = hh.s + (uint64_t) (s_ok[i] ? grow : 0u) * groups + s_part * 4u;
    }
    auto load_super = [&](uint32_t sup, uint32_t buf) {
        const uint32_t k0 = sup * Q4E_MG_SK;
        const bool k_ok = k0 + c_part * 32u < in_dim;
#pragma unroll
        for (uint32_t i = 0; i < 4u; ++i) {
            const uint32_t row = c_row + 64u * i;
            const bool ok = c_ok[i] && k_ok;
            q4e_mo_cp_async16(&sm.k.codes[buf][row][(c_part ^ (row & 7u)) * 16u],
                              c_src[i] + (ok ? k0 / 2u : 0u), ok);
        }
        const uint32_t g = k0 / Q4E_NVFP4_GROUP + s_part * 4u;
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const bool ok = s_ok[i] && g < groups;
            q4e_mo_cp_async4(&sm.k.scales[buf][s_row + Q4E_MG_HALF * i][s_part * 4u],
                             s_src[i] + (ok ? k0 / Q4E_NVFP4_GROUP : 0u), ok);
        }
    };

    /* This warp's 16 rows: the first half's warps, then the second's. */
    const uint32_t w_half = warp / (Q4E_MG_HALF / 16u);
    const float w_global = w_half ? h[1].global : h[0].global;
    const float lift = 256.0f;
    const uint32_t r0 = warp * 16u + gid;
    const uint32_t swz = gid; /* (r0 & 7) == (r0 + 8 & 7) == gid */

    /* Activation staging: chunk `b_q` (four floats) of columns `b_c` and
     * `b_c + 32`. */
    const uint32_t b_q = tid & 15u;
    const uint32_t b_c = tid >> 4;

    const __half2 k14 = __float2half2_rn(16384.0f);

    for (int n_base = c0 + (int) (z * Q4E_MG_BN); n_base < c1;
         n_base += (int) (Q4E_MG_SPLIT * Q4E_MG_BN)) {
        __syncthreads(); /* the previous tile is done with every buffer */
        if (tid < Q4E_MG_BN) {
            const int c = n_base + (int) tid;
            sm.cols[tid] = c < c1 ? sorted[c] : -1;
        }
        load_super(0u, 0u);
        q4e_mo_cp_commit();
        if (supers > 1u) {
            load_super(1u, 1u);
        }
        q4e_mo_cp_commit();
        __syncthreads();
        const int n_valid = min((int) Q4E_MG_BN, c1 - n_base);
        const float *b_src[2];
#pragma unroll
        for (uint32_t i = 0; i < 2u; ++i) {
            const int col = sm.cols[b_c + 32u * i];
            b_src[i] = col >= 0 ? x + (uint64_t) ((uint32_t) col / div) * in_dim + b_q * 4u
                                : nullptr;
        }

        float acc[Q4E_MG_BN / 8u][4];
#pragma unroll
        for (uint32_t i = 0; i < Q4E_MG_BN / 8u; ++i) {
            acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.0f;
        }

        /* Activations `Q4E_MG_AHEAD` steps ahead, one register set a step:
         * a step of arithmetic is much shorter than a gathered read. */
        float4 b_reg[Q4E_MG_AHEAD][2];
        auto load_b = [&](uint32_t kb, float4 *set) {
#pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                set[i] = b_src[i] ? *(const float4 *) (b_src[i] + kb * Q4E_MG_BK)
                                  : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        };
#pragma unroll
        for (uint32_t i = 0; i < Q4E_MG_AHEAD; ++i) {
            load_b(i, b_reg[i]); /* `steps` >= 2: `in_dim` >= 128 */
        }

        auto step = [&](uint32_t kb, float4 *set) {
            const uint32_t sup = kb / (Q4E_MG_SK / Q4E_MG_BK);
            const uint32_t sub = kb % (Q4E_MG_SK / Q4E_MG_BK);
            const uint32_t buf = sup & 1u;
            const uint32_t bb = kb & 1u;
            if (sub == 0u) {
                if (sup == 0u) {
                    q4e_mo_cp_wait<1>();
                } else {
                    q4e_mo_cp_wait<0>();
                }
            }
#pragma unroll
            for (uint32_t i = 0; i < 2u; ++i) {
                const __half2 lo = __floats2half2_rn(set[i].x, set[i].y);
                const __half2 hi = __floats2half2_rn(set[i].z, set[i].w);
                uint2 v;
                memcpy(&v.x, &lo, 4);
                memcpy(&v.y, &hi, 4);
                *(uint2 *) &sm.k.b[bb][b_c + 32u * i][b_q * 4u] = v;
            }
            __syncthreads();
            if (sub == 0u && sup >= 1u && sup + 1u < supers) {
                /* Into the stage the previous super step read, which every
                 * warp has finished with: they all passed the barrier. */
                load_super(sup + 1u, (sup + 1u) & 1u);
                q4e_mo_cp_commit();
            }
            if (kb + Q4E_MG_AHEAD < steps) {
                load_b(kb + Q4E_MG_AHEAD, set);
            }

            /* This step's two 16-byte granules of the warp's two rows:
             * groups 4 sub .. 4 sub + 3, eight code bytes each. */
            const uint4 g0 = *(const uint4 *) &sm.k.codes[buf][r0][((2u * sub) ^ swz) * 16u];
            const uint4 g1 = *(const uint4 *) &sm.k.codes[buf][r0 + 8u][((2u * sub) ^ swz) * 16u];
            const uint4 g2 = *(const uint4 *) &sm.k.codes[buf][r0][((2u * sub + 1u) ^ swz) * 16u];
            const uint4 g3 =
                    *(const uint4 *) &sm.k.codes[buf][r0 + 8u][((2u * sub + 1u) ^ swz) * 16u];
            const uint32_t s0 = *(const uint32_t *) &sm.k.scales[buf][r0][sub * 4u];
            const uint32_t s1 = *(const uint32_t *) &sm.k.scales[buf][r0 + 8u][sub * 4u];
            const uint32_t lo0[4] = {g0.x, g0.z, g2.x, g2.z};
            const uint32_t hi0[4] = {g0.y, g0.w, g2.y, g2.w};
            const uint32_t lo1[4] = {g1.x, g1.z, g3.x, g3.z};
            const uint32_t hi1[4] = {g1.y, g1.w, g3.y, g3.w};
#pragma unroll
            for (uint32_t j = 0; j < 4u; ++j) {
                /* Group j of the step: the two rows' scales, lifted to their
                 * value times 2^6, one row a half. */
                const uint32_t sp = q4e_mo_e4m3_tiny((s0 >> (8u * j)) & 0xFFu)
                                  | (q4e_mo_e4m3_tiny((s1 >> (8u * j)) & 0xFFu) << 16);
                __half2 spair;
                memcpy(&spair, &sp, 4);
                spair = __hmul2(spair, k14);
                const __half2 sc0 = __low2half2(spair);
                const __half2 sc1 = __high2half2(spair);
                uint32_t a[4];
                a[0] = q4e_mo_a_frag((lo0[j] >> (8u * tig)) & 0xFFu, sc0);
                a[1] = q4e_mo_a_frag((lo1[j] >> (8u * tig)) & 0xFFu, sc1);
                a[2] = q4e_mo_a_frag((hi0[j] >> (8u * tig)) & 0xFFu, sc0);
                a[3] = q4e_mo_a_frag((hi1[j] >> (8u * tig)) & 0xFFu, sc1);
#pragma unroll
                for (uint32_t np = 0; np < Q4E_MG_BN / 16u; ++np) {
                    if ((int) (np * 16u) < n_valid) {
                        const uint32_t n = np * 16u + (lane & 7u) + ((lane >> 4) << 3);
                        const uint32_t k = j * 16u + ((lane >> 3) & 1u) * 8u;
                        uint32_t b[4];
                        asm volatile(
                                "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3])
                                : "r"(q4e_mo_smem(&sm.k.b[bb][n][k])));
                        q4e_mo_mma(acc[2u * np], a, b);
                        if ((int) (np * 16u + 8u) < n_valid) {
                            q4e_mo_mma(acc[2u * np + 1u], a, b + 2);
                        }
                    }
                }
            }
        };
        /* `steps` is a multiple of `Q4E_MG_AHEAD`: `in_dim` of 128. */
        for (uint32_t kb = 0; kb < steps; kb += Q4E_MG_AHEAD) {
#pragma unroll
            for (uint32_t i = 0; i < Q4E_MG_AHEAD; ++i) {
                step(kb + i, b_reg[i]);
            }
        }

        /* The output tile through shared memory, so each column's rows leave
         * as whole lines: scattered 32-byte pieces cost the down stage as long
         * as its weights did. c0, c1 sit at (row gid, columns 2 tig, +1); c2,
         * c3 eight rows down. */
        __syncthreads(); /* every warp is done with the weight and activation buffers */
#pragma unroll
        for (uint32_t nt = 0; nt < Q4E_MG_BN / 8u; ++nt) {
            if ((int) (nt * 8u) < n_valid) {
#pragma unroll
                for (uint32_t e = 0; e < 4u; ++e) {
                    const uint32_t n = nt * 8u + tig * 2u + (e & 1u);
                    sm.out[n][r0 + (e >> 1) * 8u] = (acc[nt][e] * lift) * w_global;
                }
            }
        }
        __syncthreads();
        {
            /* A column's rows as float4s, `Q4E_MG_BM / 4` threads a column. */
            const uint32_t q = tid % (Q4E_MG_BM / 4u); /* rows 4q .. 4q + 3 */
            const q4e_mo_half hh = q < Q4E_MG_HALF / 4u ? h[0] : h[1];
            const uint32_t o = hh.row0 + (q * 4u) % Q4E_MG_HALF;
            for (int n = (int) (tid / (Q4E_MG_BM / 4u)); n < n_valid;
                 n += (int) (Q4E_MG_THREADS / (Q4E_MG_BM / 4u))) {
                const int col = sm.cols[n];
                const float4 v = *(const float4 *) &sm.out[n][q * 4u];
                float *dst = hh.out + (uint64_t) col * out_dim + o;
                if ((out_dim & 3u) == 0u && o + 3u < out_dim) {
                    *(float4 *) dst = v;
                } else {
                    const float vv[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (uint32_t e = 0; e < 4u; ++e) {
                        if (o + e < out_dim) {
                            dst[e] = vv[e];
                        }
                    }
                }
            }
        }
    }
}

static_assert(sizeof(q4e_mo_gemm_smem) == Q4E_MG_SMEM, "the host passes Q4E_MG_SMEM");

/* Gate and up of a prompt pass over an NVFP4 bank: a block covers the same 64
 * rows of both, row tile `blockIdx.x / Q4E_MG_SPLIT`, column tiles from
 * `blockIdx.x % Q4E_MG_SPLIT`; one active expert a block on the grid's second
 * axis.
 *
 * Buffers: out_gate, out_up, gate_w, gate_s, gate_g, up_w, up_s, up_g, x,
 * sorted, bounds, active, map, in_dim, out_dim, n_expert, n_used. Grid
 * `(ceil(out_dim / 128) * 4, min(columns, n_expert))`, 512 threads,
 * `Q4E_MG_SMEM` bytes of dynamic shared memory; `in_dim` a multiple of 128. */
extern "C" __global__ void __launch_bounds__(Q4E_MG_THREADS, 1) q4e_nvfp4_experts_gemm_gate_up_kernel(
        float *out_gate, float *out_up,
        const uint8_t *gate_w, const uint8_t *gate_s, const float *gate_g,
        const uint8_t *up_w, const uint8_t *up_s, const float *up_g,
        const float *x, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    const int n_active = active[n_expert];
    const int expert = active[blockIdx.y];
    if ((int) blockIdx.y >= n_active) {
        return;
    }
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t z = blockIdx.x % Q4E_MG_SPLIT;
    if (c1 - c0 <= (int) (z * Q4E_MG_BN)) {
        return;
    }
    const uint32_t row0 = (blockIdx.x / Q4E_MG_SPLIT) * Q4E_MG_HALF;
    const uint64_t slot = map[expert];
    const uint64_t first = slot * out_dim;
    const q4e_mo_half h[2] = {
            {gate_w + first * (in_dim / 2u), gate_s + first * (in_dim / Q4E_NVFP4_GROUP),
             gate_g[slot], out_gate, row0},
            {up_w + first * (in_dim / 2u), up_s + first * (in_dim / Q4E_NVFP4_GROUP),
             up_g[slot], out_up, row0},
    };
    q4e_mo_nvfp4_gemm_block(h, x, sorted, c0, c1, n_used, in_dim, out_dim, z);
}

/* The routed down projection of a prompt pass over an NVFP4 bank: 128 rows a
 * block, as two halves of the same matrix.
 *
 * Buffers: out, down_w, down_s, down_g, mid, sorted, bounds, active, map,
 * in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 256) * 4,
 * min(columns, n_expert))`, 512 threads, `Q4E_MG_SMEM` bytes of dynamic shared
 * memory; `in_dim` a multiple of 128. */
extern "C" __global__ void __launch_bounds__(Q4E_MG_THREADS, 1) q4e_nvfp4_experts_gemm_down_kernel(
        float *out, const uint8_t *down_w, const uint8_t *down_s, const float *down_g,
        const float *mid, const int *sorted, const int *bounds, const int *active,
        const uint32_t *map, uint32_t in_dim, uint32_t out_dim, uint32_t n_expert,
        uint32_t n_used) {
    (void) n_used;
    const int n_active = active[n_expert];
    const int expert = active[blockIdx.y];
    if ((int) blockIdx.y >= n_active) {
        return;
    }
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t z = blockIdx.x % Q4E_MG_SPLIT;
    if (c1 - c0 <= (int) (z * Q4E_MG_BN)) {
        return;
    }
    const uint32_t row0 = (blockIdx.x / Q4E_MG_SPLIT) * Q4E_MG_BM;
    const uint64_t slot = map[expert];
    const uint64_t first = slot * out_dim;
    const uint8_t *w = down_w + first * (in_dim / 2u);
    const uint8_t *s = down_s + first * (in_dim / Q4E_NVFP4_GROUP);
    const float g = down_g[slot];
    const q4e_mo_half h[2] = {
            {w, s, g, out, row0},
            {w, s, g, out, row0 + Q4E_MG_HALF},
    };
    q4e_mo_nvfp4_gemm_block(h, mid, sorted, c0, c1, 1u, in_dim, out_dim, z);
}


/* A brain float widened exactly. */
__device__ __forceinline__ float q4e_mo_bf16(uint16_t bits) {
    return __int_as_float((int) ((uint32_t) bits << 16));
}

#define Q4E_FP8_BLOCK 128u

/* One lane's share of the dot of one block-scaled FP8 row with `x`: lane `l`
 * reads bytes `4l .. 4l + 3` of every 128-wide stretch, so a warp consumes one
 * scale block's row segment per step - the Metal set's `q4e_fp8_row`. */
__device__ __forceinline__ float q4e_mo_fp8_row(
        const uint8_t *__restrict__ w, const uint16_t *__restrict__ scales,
        const float *__restrict__ x, uint32_t in_dim, uint32_t lane) {
    float acc = 0.0f;
    for (uint32_t k0 = 0u; k0 < in_dim; k0 += Q4E_FP8_BLOCK) {
        const float s = q4e_mo_bf16(scales[k0 / Q4E_FP8_BLOCK]);
        const uint32_t k = k0 + lane * 4u;
        if (k + 3u < in_dim) {
            const uint32_t q = *(const uint32_t *) (w + k);
            const float4 xv = *(const float4 *) (x + k);
            const float part = q4e_mo_e4m3(q & 0xFFu) * xv.x
                             + q4e_mo_e4m3((q >> 8) & 0xFFu) * xv.y
                             + q4e_mo_e4m3((q >> 16) & 0xFFu) * xv.z
                             + q4e_mo_e4m3(q >> 24) * xv.w;
            acc += s * part;
        }
    }
    return acc;
}

/* Gate and up over a block-scaled FP8 bank - the draft head's routed experts.
 *
 * Buffers: out_gate, out_up, gate_w, gate_s, up_w, up_s, x, sorted, bounds,
 * active, map, in_dim, out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8),
 * min(columns, n_expert))`, 256 threads, a warp per output row. */
extern "C" __global__ void q4e_fp8_experts_gate_up_kernel(
        float *out_gate, float *out_up, const uint8_t *gate_w, const uint16_t *gate_s,
        const uint8_t *up_w, const uint16_t *up_s, const float *x, const int *sorted,
        const int *bounds, const int *active, const uint32_t *map, uint32_t in_dim,
        uint32_t out_dim, uint32_t n_expert, uint32_t n_used) {
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t o = blockIdx.x * Q4E_NVFP4_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint64_t slot = map[expert];
    const uint32_t blocks = (in_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint32_t block_rows = (out_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint64_t row = slot * out_dim + o;
    const uint64_t srow = (slot * block_rows + o / Q4E_FP8_BLOCK) * blocks;
    const uint8_t *gw = gate_w + row * in_dim;
    const uint8_t *uw = up_w + row * in_dim;
    for (int c = c0; c < c1; ++c) {
        const uint32_t col = (uint32_t) sorted[c];
        const float *xr = x + (uint64_t) (col / n_used) * in_dim;
        const float g = q4e_mo_warp_sum(q4e_mo_fp8_row(gw, gate_s + srow, xr, in_dim, lane));
        const float u = q4e_mo_warp_sum(q4e_mo_fp8_row(uw, up_s + srow, xr, in_dim, lane));
        if (lane == 0u) {
            out_gate[(uint64_t) col * out_dim + o] = g;
            out_up[(uint64_t) col * out_dim + o] = u;
        }
    }
}

/* The routed down projection over a block-scaled FP8 bank.
 *
 * Buffers: out, down_w, down_s, mid, sorted, bounds, active, map, in_dim,
 * out_dim, n_expert, n_used. Grid `(ceil(out_dim / 8), min(columns,
 * n_expert))`, 256 threads. */
extern "C" __global__ void q4e_fp8_experts_down_kernel(
        float *out, const uint8_t *down_w, const uint16_t *down_s, const float *mid,
        const int *sorted, const int *bounds, const int *active, const uint32_t *map,
        uint32_t in_dim, uint32_t out_dim, uint32_t n_expert, uint32_t n_used) {
    (void) n_used;
    if ((int) blockIdx.y >= active[n_expert]) {
        return;
    }
    const uint32_t lane = threadIdx.x & 31u;
    const int expert = active[blockIdx.y];
    const int c0 = bounds[expert];
    const int c1 = bounds[expert + 1];
    const uint32_t o = blockIdx.x * Q4E_NVFP4_ROWS + (threadIdx.x >> 5);
    if (c1 <= c0 || o >= out_dim) {
        return;
    }
    const uint64_t slot = map[expert];
    const uint32_t blocks = (in_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint32_t block_rows = (out_dim + Q4E_FP8_BLOCK - 1u) / Q4E_FP8_BLOCK;
    const uint8_t *dw = down_w + (slot * out_dim + o) * in_dim;
    const uint64_t srow = (slot * block_rows + o / Q4E_FP8_BLOCK) * blocks;
    for (int c = c0; c < c1; ++c) {
        const uint32_t col = (uint32_t) sorted[c];
        const float v = q4e_mo_warp_sum(
                q4e_mo_fp8_row(dw, down_s + srow, mid + (uint64_t) col * in_dim, in_dim, lane));
        if (lane == 0u) {
            out[(uint64_t) col * out_dim + o] = v;
        }
    }
}

/* Decodes gathered FP8 n-gram rows: `head_dim` E4M3 bytes a row, one scale for
 * the whole table. Row `r` lands at token `r / n_heads`, head slot
 * `r % n_heads`, head-slowest - `q4e_ple_dequant_kernel`'s placement.
 *
 * Buffers: out, rows, head_dim, n_heads, row_bytes, n_rows, scale (one float
 * on the device). Grid `n_rows`, a block of `head_dim` threads. */
extern "C" __global__ void q4e_ple_dequant_fp8_kernel(
        float *out, const uint8_t *rows, uint32_t head_dim, uint32_t n_heads,
        uint32_t row_bytes, uint32_t n_rows, const float *scale) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }
    const uint8_t *src = rows + (uint64_t) r * row_bytes;
    const uint32_t t = r / n_heads;
    const uint32_t h = r % n_heads;
    float *dst = out + (uint64_t) t * n_heads * head_dim + (uint64_t) h * head_dim;
    const float s = scale[0];
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dst[i] = q4e_mo_e4m3(src[i]) * s;
    }
}

/* A float rounded to the nearest brain float, ties to even, and widened back;
 * a NaN passes through so the carry cannot walk it into an infinity. Written
 * out rather than taken from the intrinsic so the host codec
 * (`shiso_ple::codec::round_to_bf16`) can restate it line for line. */
__device__ __forceinline__ float q4e_mo_round_bf16(float v) {
    const uint32_t bits = (uint32_t) __float_as_int(v);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) {
        return v;
    }
    return __int_as_float((int) ((bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u));
}

/* Decodes gathered NVFP4 n-gram rows: `head_dim / 2` code bytes (the low
 * nibble the earlier weight), then `head_dim / 16` `e4m3` group scales, one
 * `f32` scale for the whole table. Placement as `q4e_ple_dequant_fp8_kernel`.
 *
 * `bf16((e2m1 * e4m3) * global)`: the reference table's order and its one
 * rounding, NOT the projections' `e2m1 * (e4m3 * global)` - the code-times-
 * scale product is exact, so this order rounds once in single precision and
 * the other twice. Both multiplies are spelled `__fmul_rn` so no contraction
 * can move them. The smallest nonzero product is `2^-10 * global`, a normal
 * float for any table scale above about `2^-116`, so the module's
 * flush-to-zero never reaches it.
 *
 * Buffers: out, rows, head_dim, n_heads, row_bytes, n_rows, global (one float
 * on the device). Grid `n_rows`, a block of `head_dim` threads. */
extern "C" __global__ void q4e_ple_dequant_nvfp4_kernel(
        float *out, const uint8_t *rows, uint32_t head_dim, uint32_t n_heads,
        uint32_t row_bytes, uint32_t n_rows, const float *global) {
    const uint32_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }
    const uint8_t *src = rows + (uint64_t) r * row_bytes;
    const uint8_t *scales = src + head_dim / 2u;
    const uint32_t t = r / n_heads;
    const uint32_t h = r % n_heads;
    float *dst = out + (uint64_t) t * n_heads * head_dim + (uint64_t) h * head_dim;
    const float g = global[0];
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const uint32_t byte = src[i / 2u];
        const uint32_t code = (i & 1u) ? (byte >> 4) : (byte & 0x0Fu);
        const float exact = __fmul_rn(q4e_mo_e2m1(code), q4e_mo_e4m3(scales[i / Q4E_NVFP4_GROUP]));
        dst[i] = q4e_mo_round_bf16(__fmul_rn(exact, g));
    }
}
