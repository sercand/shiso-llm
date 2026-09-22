/* Shared declarations for the `qwen35moe` module: the two codecs' element
 * decodes and the argument blocks every entry here takes.
 *
 * Nothing in this directory is a byte copy of anything, so no file carries a
 * provenance header and `scripts/q4e_ptx_oracle.sh --mode=fidelity` reports
 * them as generated. The arithmetic is `crates/shiso-modelfile/src/fp8.rs` and
 * `crates/shiso-modelfile/src/nvfp4.rs`, which are in this tree and are themselves
 * graded bit-for-bit against torch 2.14.
 *
 * # The decodes are written out rather than taken from `<cuda_fp8.h>`
 *
 * CUDA ships `__nv_fp8_e4m3` and a conversion for it. It is not used here, for
 * two reasons and neither is taste. First, this module is pinned at
 * `compute_80` like every other CUDA family in this tree, and the hardware
 * conversion is an `sm_89` instruction, so the header would silently hand back
 * a software emulation whose rounding is its own business rather than ours.
 * Second, and this is the one that matters: these kernels are graded against
 * the host decoders above, and the only way that grading means anything is if
 * the two implement the same law rather than two laws that agree on the data
 * the fixture happened to carry. The bodies below are those functions line for
 * line.
 */
#ifndef SHISO_Q35MOE_COMMON_CUH
#define SHISO_Q35MOE_COMMON_CUH

#include <cuda_runtime.h>
#include <stdint.h>

/* Weights sharing one `e4m3` scale byte in NVFP4. */
#define SHISO_NVFP4_GROUP 16
/* Weights packed into one NVFP4 storage byte. */
#define SHISO_NVFP4_PER_BYTE 2

/* Lanes in a warp, which is also the width of every block's `x` extent here. */
#define SHISO_Q35MOE_WARP 32
/* Warps per block for the matvec entries: 256 threads. */
#define SHISO_Q35MOE_WARPS 8
/* Threads per block for the matvec entries. */
#define SHISO_Q35MOE_BLOCK (SHISO_Q35MOE_WARP * SHISO_Q35MOE_WARPS)

/* Output rows one block of the NVFP4 matvec covers, at one warp each.
 *
 * The entry this serves used to give one 256-thread block to one output row
 * and fold it with `shiso_q35moe_block_sum`. At this family's widths that is
 * the wrong shape twice over: a 2048-wide row is 1024 packed bytes, so each of
 * the 256 threads got FOUR BYTES and then paid a 256-wide reduction across
 * shared memory and a `__syncthreads` to fold them. The reduction was the
 * kernel.
 *
 * One warp per row folds in five shuffles with no shared memory and no
 * barrier, and four rows to a block keeps the block at 128 threads - wide
 * enough to hide the remaining latency, narrow enough that the 512-row
 * projections in this family still fill the device. `out_dim` is not required
 * to be a multiple of this: the last block's spare warps return.
 */
#define SHISO_Q35MOE_MV_ROWS 4
/* Threads per block of the NVFP4 matvec: one warp per output row. */
#define SHISO_Q35MOE_MV_BLOCK (SHISO_Q35MOE_WARP * SHISO_Q35MOE_MV_ROWS)

/* Square tile the GEMM entries walk, on all three axes. 16x16 output elements
 * per block at one thread each, over 16-wide slabs of the contraction. */
#define SHISO_Q35MOE_TILE 16

/* Two to a small signed power, exactly.
 *
 * The host's `exp2` in `fp8.rs`, and for the same reason: `exp2f` is not exact
 * for every input on every target and `--use_fast_math` would rewrite it to an
 * intrinsic that is less so, while the exponents here span -6..=8 and building
 * the bit pattern is both exact and obviously so. */
__device__ __forceinline__ float shiso_exp2i(int power) {
    return __int_as_float((power + 127) << 23);
}

/* Decodes one `e4m3` byte.
 *
 * One sign bit, a four-bit exponent biased by seven, three mantissa bits, no
 * infinities, `0x7f`/`0xff` reserved for NaN, and subnormals reaching down to
 * 2^-9. This checkpoint's NVFP4 scale arrays really do use the subnormal range
 * - `experts.0.gate_proj`'s smallest scale byte is `0x01` - so the branch is
 * load-bearing rather than defensive: without it whole groups of a weight
 * matrix decode to zero, which fails nowhere and answers wrongly. */
__device__ __forceinline__ float shiso_e4m3(unsigned char byte) {
    const float sign = (byte & 0x80u) ? -1.0f : 1.0f;
    const int exponent = (int) ((byte >> 3) & 0x0Fu);
    const float mantissa = (float) (byte & 0x07u);
    if (exponent == 0x0F) {
        /* The all-ones exponent is NaN in this format; there is no infinity. */
        if ((byte & 0x07u) == 0x07u) {
            return __int_as_float(0x7fffffff);
        }
        return sign * (256.0f * (1.0f + mantissa * 0.125f));
    }
    if (exponent == 0) {
        /* Subnormal: no implicit leading one, and the exponent is the smallest
         * normal one rather than the bias. */
        return sign * (mantissa * 0.125f) * shiso_exp2i(-6);
    }
    return sign * (1.0f + mantissa * 0.125f) * shiso_exp2i(exponent - 7);
}

/* The eight magnitudes the three low bits of an `e2m1` code select.
 *
 * Written out rather than computed, exactly as the host table is: the format's
 * two exponent bits and one mantissa bit give {0, 0.5} for the subnormal
 * exponent and {1, 1.5, 2, 3, 4, 6} above it, which is a rule with two special
 * cases at the bottom, and a table of eight is the whole of it and cannot be
 * off by one. */
__device__ __constant__ float SHISO_E2M1[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

/* Decodes one four-bit code, sign bit included. */
__device__ __forceinline__ float shiso_e2m1(unsigned char code) {
    const float magnitude = SHISO_E2M1[code & 0x07u];
    return (code & 0x08u) ? -magnitude : magnitude;
}

/* The widths and the one scale an FP8 projection is launched over.
 *
 * Sixteen bytes, one struct, passed BY VALUE - and that is the whole of the
 * first parameter as the PTX declares it. `crates/shiso-cuda/src/launch.rs`
 * hands the driver one pointer per DECLARED parameter, so four `i32`/`f32`
 * scalar operands here would give the entry a four-byte first parameter where
 * it declares sixteen, and every launch would succeed reading a pointer out of
 * a slot holding a quarter of a struct. `grep '\.visible \.entry'` on the
 * generated PTX is the check, and `docs/superpowers/plans/` records this
 * costing a session once already. */
struct shiso_args_q35moe_fp8 {
    /* Elements in one weight row, which is the projection's input width. */
    int32_t in_dim;
    /* Weight rows, which is the projection's output width. */
    int32_t out_dim;
    /* Activation rows this launch covers. */
    int32_t n_tok;
    /* The one `f32` every weight of this tensor is multiplied by. */
    float weight_scale;
};

/* The same, for NVFP4, which needs one more number.
 *
 * `in_dim` is the LOGICAL width: the packed row is `in_dim / 2` bytes and the
 * scale row is `in_dim / 16` bytes, both derived here rather than passed, so a
 * caller cannot state a packed width that disagrees with the logical one. */
struct shiso_args_q35moe_nvfp4 {
    /* Elements in one weight row, counted in weights and not in bytes. */
    int32_t in_dim;
    /* Weight rows, which is the projection's output width. */
    int32_t out_dim;
    /* Activation rows this launch covers. */
    int32_t n_tok;
    /* The whole tensor's scale, which the per-group `e4m3` bytes are relative
     * to. Dropping it rescales the tensor by about 31,458x, and the next
     * normalization very nearly hides that - so it is a field of the argument
     * block rather than something a caller may or may not fold in. */
    float global_scale;
};

/* Folds `value` across a whole warp. */
__device__ __forceinline__ float shiso_q35moe_warp_sum(float value) {
#pragma unroll
    for (int offset = SHISO_Q35MOE_WARP / 2; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset, SHISO_Q35MOE_WARP);
    }
    return value;
}

/* Folds one block's per-warp partials into lane zero of warp zero.
 *
 * `shmem` is at least `SHISO_Q35MOE_WARPS` floats. The value is meaningful in
 * thread zero only; every other thread's return is whatever its lane held. */
__device__ __forceinline__ float shiso_q35moe_block_sum(float value, float *shmem) {
    const unsigned lane = threadIdx.x % SHISO_Q35MOE_WARP;
    const unsigned warp = threadIdx.x / SHISO_Q35MOE_WARP;
    value = shiso_q35moe_warp_sum(value);
    if (lane == 0) {
        shmem[warp] = value;
    }
    __syncthreads();
    if (warp != 0) {
        return value;
    }
    float total = (lane < SHISO_Q35MOE_WARPS) ? shmem[lane] : 0.0f;
#pragma unroll
    for (int offset = SHISO_Q35MOE_WARPS / 2; offset > 0; offset >>= 1) {
        total += __shfl_xor_sync(0xffffffffu, total, offset, SHISO_Q35MOE_WARP);
    }
    return total;
}

/* Decodes one four-bit code without touching memory.
 *
 * **Bit-identical to `SHISO_E2M1[code & 7]` with the sign applied**, for all
 * sixteen codes, and `every_e2m1_code_decodes_the_way_the_host_table_does`
 * in `crates/shiso-kerneltest/src/qwen35moe/nvfp4.rs` grades that claim over
 * all sixteen codes rather than trusting it. It is here because the table is the wrong
 * storage for this access pattern, not because the table is wrong.
 *
 * `__constant__` is broadcast storage: a warp whose lanes hold ONE address
 * reads it in a cycle, and a warp whose lanes hold eight different addresses
 * serializes into eight. Every lane of a matvec holds a different weight by
 * construction, so the table was serializing on every decode - and that access
 * is an MIO transaction, which is the queue `ncu` attributed 41.4% of this
 * family's warp stall cycles to (`short scoreboard`, 11.3 of 27.3 cycles
 * between issues, measured on `kernel_q35moe_experts_gate_up_nvfp4_f32`).
 * Arithmetic has no such queue and runs on the pipe that was idle.
 *
 * The three low bits are a two-bit exponent and a one-bit mantissa. Above the
 * subnormal exponent the value is `(1 + m/2) * 2^(e-1)`, which is an f32 whose
 * exponent field is `126 + e` and whose mantissa is `m` in the top bit; `e ==
 * 0` is the format's subnormal pair `{0, 0.5}` and is the one case that
 * formula does not reach, so it is selected rather than computed. The compiler
 * emits that select as a predicated move, so the branch costs no divergence.
 */
__device__ __forceinline__ float shiso_e2m1_fast(unsigned code) {
    const unsigned e = (code >> 1) & 3u;
    const unsigned m = code & 1u;
    unsigned bits = ((126u + e) << 23) | (m << 22);
    if (e == 0u) {
        bits = m ? 0x3F000000u : 0u;
    }
    /* The sign bit of the code is the sign bit of the float, twenty-eight
     * places up. Code 8 is therefore -0.0, which is what the table gives too. */
    return __int_as_float((int) (bits | ((code & 8u) << 28)));
}

/* Eight NVFP4 weights, one packed word, two `float4` of activation.
 *
 * The low nibble of a byte is the earlier weight - see `mv_nvfp4.cuh` for why
 * that convention is pinned by hand - so byte `j` of `word` contributes to
 * lanes `2j` and `2j + 1` of the eight activations, which are `a.xyzw` then
 * `b.xyzw`. The scale is NOT applied here: the caller applies one scale to the
 * finished group, which is what makes this eight fused multiply-adds and no
 * multiplies.
 */
__device__ __forceinline__ float shiso_nvfp4_word_dot(unsigned word, float4 a, float4 b) {
    float s = 0.0f;
    s = fmaf(shiso_e2m1_fast(word & 0x0Fu), a.x, s);
    s = fmaf(shiso_e2m1_fast((word >> 4) & 0x0Fu), a.y, s);
    s = fmaf(shiso_e2m1_fast((word >> 8) & 0x0Fu), a.z, s);
    s = fmaf(shiso_e2m1_fast((word >> 12) & 0x0Fu), a.w, s);
    s = fmaf(shiso_e2m1_fast((word >> 16) & 0x0Fu), b.x, s);
    s = fmaf(shiso_e2m1_fast((word >> 20) & 0x0Fu), b.y, s);
    s = fmaf(shiso_e2m1_fast((word >> 24) & 0x0Fu), b.z, s);
    s = fmaf(shiso_e2m1_fast((word >> 28) & 0x0Fu), b.w, s);
    return s;
}

/* One warp's dot of one NVFP4 weight row against one activation row.
 *
 * **A lane owns whole sixteen-weight GROUPS, strided by the warp.** That is
 * the one decomposition this format hands over free: a group is exactly eight
 * packed bytes, exactly one scale byte and exactly four `float4` of
 * activation, so a lane's weights are one aligned `uint2`, its scale is one
 * byte it reads once instead of eight times, and nothing ever straddles.
 *
 * What that buys, per sixteen weights per lane, against the walk this replaced
 * (one four-byte word per lane per turn, one scale byte per word, one scalar
 * `float` per weight):
 *
 * | | loads | scale decodes | FMUL | FFMA |
 * | --- | --- | --- | --- | --- |
 * | before | 2 x (1 + 1 + 8) = 20 | 2 | 16 | 16 |
 * | here | 1 + 1 + 4 = 6 | 1 | 1 | 16 |
 *
 * 3.3x fewer memory instructions and a sixteenth of the multiplies. The
 * kernels this serves were reading at 60-119 GB/s of this device's measured
 * 239 GB/s while `ncu` reported 12% memory throughput against 50% `mem pipes
 * busy` and a 98% L1 hit rate - the signature of a kernel spending its issue
 * slots on small loads of bytes it already has, which is exactly what the left
 * column is.
 *
 * # The scale multiplies once per group, and that is a change of answer
 *
 * `sum_k (w_k * scale) * x_k` became `scale * sum_k (w_k * x_k)`. Identical in
 * exact arithmetic, different in the last bits, and the SECOND form is the
 * better-conditioned one: it is a two-level sum where the flat walk was one
 * level, and torch's own reduction - which is what the oracle runs - is itself
 * blocked. The group scale and the per-tensor global still multiply together
 * FIRST and once per group, which is the ordering `nvfp4.rs` defines and the
 * one thing here that is not free to reassociate.
 *
 * The move against the `transformers` oracle is recorded in the commit that
 * made this; the budget was not touched.
 *
 * # The narrow path is not dead code
 *
 * `wide` is uniform across the warp - it reads the widths and the two base
 * pointers, which every lane agrees on - so the branch costs no divergence. It
 * is false only for a row this format cannot produce (a width that is not a
 * whole number of groups) or a slab a future loader aligned differently, and
 * it answers identically rather than faster. A kernel that was silently wrong
 * on such a row would be worse than one that is slow on it.
 */
__device__ __forceinline__ float shiso_q35moe_nvfp4_row_dot(
        const unsigned char *__restrict__ packed,
        const unsigned char *__restrict__ scales,
        float global_scale,
        const float *__restrict__ x,
        int in_dim,
        unsigned lane) {
    float acc = 0.0f;
    const int groups = in_dim / SHISO_NVFP4_GROUP;
    const bool wide = (in_dim % SHISO_NVFP4_GROUP) == 0
                   && ((uintptr_t) packed & 7u) == 0u
                   && ((uintptr_t) x & 15u) == 0u;
    if (wide) {
        for (int g = (int) lane; g < groups; g += SHISO_Q35MOE_WARP) {
            const uint2 w = ((const uint2 *) packed)[g];
            const float scale = shiso_e4m3(scales[g]) * global_scale;
            const float4 *xg = (const float4 *) (x + (int64_t) g * SHISO_NVFP4_GROUP);
            const float part = shiso_nvfp4_word_dot(w.x, xg[0], xg[1])
                             + shiso_nvfp4_word_dot(w.y, xg[2], xg[3]);
            acc = fmaf(part, scale, acc);
        }
    } else {
        for (int k = (int) lane; k < in_dim; k += SHISO_Q35MOE_WARP) {
            const unsigned char byte = packed[k / SHISO_NVFP4_PER_BYTE];
            const unsigned code = (k & 1) ? (unsigned) (byte >> 4)
                                          : (unsigned) (byte & 0x0Fu);
            const float scale =
                    shiso_e4m3(scales[k / SHISO_NVFP4_GROUP]) * global_scale;
            acc = fmaf(shiso_e2m1_fast(code) * scale, x[k], acc);
        }
    }
    return shiso_q35moe_warp_sum(acc);
}

#endif /* SHISO_Q35MOE_COMMON_CUH */
