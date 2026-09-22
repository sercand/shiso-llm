/* The argument blocks and the q4_0 block layout the generic family reads.
 *
 * Nothing here is copied. The two argument structs are the field order
 * `crates/shiso-engine/src/launch.rs` declares, transcribed so the kernels
 * below read the same bytes the host packs; the `q4_0` block is ggml's 18-byte
 * format, which is a file layout rather than anybody's source; and the
 * dequantizer is the arithmetic of the format, derived from the Metal twin in
 * `crates/shiso-metal/metal/generic/mul_mv_ext.metal` so both devices round the
 * same way.
 */
#ifndef SHISO_GENERIC_CUH
#define SHISO_GENERIC_CUH

#include <cuda_fp16.h>
#include <stdint.h>

/* `shiso_engine::launch::MulMvExtArgs`, field for field.
 *
 * The two explicit pads are the four bytes the C layout leaves before each
 * 64-bit stride, spelled out on the Rust side so `bytemuck::Pod` can hold the
 * struct and spelled out here so the two declarations cannot drift. The static
 * assertion below is the whole of what keeps them honest: a launch whose
 * argument block is one field out reads a stride as an extent and computes a
 * product that is confidently wrong. */
struct shiso_args_mul_mv_ext {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    int32_t  pad0;
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    int32_t  pad1;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int8_t   pad2[4];
};
static_assert(sizeof(struct shiso_args_mul_mv_ext) == 112,
              "MulMvExtArgs is 112 bytes on the host and must be here too");

/* `shiso_engine::launch::MulMmArgs`, field for field. */
struct shiso_args_mul_mm {
    int32_t  ne00;
    int32_t  ne02;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne12;
    int32_t  pad0;
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int16_t  r2;
    int16_t  r3;
    int8_t   pad2[4];
};
static_assert(sizeof(struct shiso_args_mul_mm) == 88,
              "MulMmArgs is 88 bytes on the host and must be here too");

/* `shiso_engine::launch::MulMvArgs`, field for field. A different struct from
 * the extended family's, not a prefix of it: that one carries the input-split
 * and row-blocking fields the plain matvec has no use for, and `nr0` here is
 * the output rows one block produces. */
struct shiso_args_mul_mv {
    int32_t  ne00;
    int32_t  ne01;
    int32_t  ne02;
    uint8_t  pad0[4];
    uint64_t nb00;
    uint64_t nb01;
    uint64_t nb02;
    uint64_t nb03;
    int32_t  ne10;
    int32_t  ne11;
    int32_t  ne12;
    uint8_t  pad1[4];
    uint64_t nb10;
    uint64_t nb11;
    uint64_t nb12;
    uint64_t nb13;
    int32_t  ne0;
    int32_t  ne1;
    int32_t  nr0;
    int16_t  r2;
    int16_t  r3;
};
static_assert(sizeof(struct shiso_args_mul_mv) == 112,
              "MulMvArgs is 112 bytes on the host and must be here too");

/* `shiso_engine::launch::NormArgs`, field for field. Shared by both fused norms
 * in `norm.cuh` and by `kernel_dense_add_scale_rms_norm_mul_f32_4` in the dense
 * module, which declares its own copy for the same reason: a module's headers
 * reach only its own directory. */
struct shiso_args_norm {
    int32_t  ne00;
    int32_t  ne00_t;
    uint64_t nb1;
    uint64_t nb2;
    uint64_t nb3;
    float    eps;
    int32_t  nef1[3];
    int32_t  nef2[3];
    int32_t  nef3[3];
    uint64_t nbf1[3];
    uint64_t nbf2[3];
    uint64_t nbf3[3];
};
static_assert(sizeof(struct shiso_args_norm) == 144,
              "NormArgs is 144 bytes on the host and must be here too");

/* `shiso_engine::launch::GluArgs` (`metal/generic/glu.metal`).
 *
 * The three explicit pads are the four bytes the C layout leaves before each
 * 64-bit stride, spelled out on the Rust side so `bytemuck::Pod` can hold the
 * struct and spelled out here so the two declarations cannot drift.
 *
 * Declared again in `kernels/cuda/dense/dense_common.cuh` as
 * `dense_args_glu`, because the two modules are separately compiled images and
 * neither includes the other's headers. Both carry the same static assertion
 * against the same host struct, which is what keeps the pair honest. */
struct shiso_args_glu {
    int32_t  ne00;
    int32_t  pad0;
    uint64_t nb01;
    int32_t  ne10;
    int32_t  pad1;
    uint64_t nb11;
    int32_t  ne0;
    int32_t  pad2;
    uint64_t nb1;
    int32_t  i00;
    int32_t  i10;
    float    alpha;
    float    limit;
};
static_assert(sizeof(struct shiso_args_glu) == 64, "GluArgs is 64 bytes");

/* `shiso_engine::launch::AddArgs`. */
struct shiso_args_add {
    uint32_t n;
};
static_assert(sizeof(struct shiso_args_add) == 4, "AddArgs is 4 bytes");

/* `shiso_engine::launch::GetRowsArgs`. The pad is the four bytes the C layout
 * leaves before the stride run, spelled out on both sides so neither can
 * drift. */
struct shiso_args_get_rows {
    int32_t  n_embd;
    int32_t  n_vocab;
    int32_t  n_tokens;
    int32_t  pad0;
    uint64_t src_row_bytes;
    uint64_t dst_row_bytes;
    uint64_t token_stride;
};
static_assert(sizeof(struct shiso_args_get_rows) == 40,
              "GetRowsArgs is 40 bytes on the host and must be here too");

/* Every lane of a warp. */
#define SHISO_FULL_WARP 0xffffffffu

/* A whole-warp butterfly sum: every lane ends up holding the total.
 *
 * The reduction the fused norms and the plain matvec both fold with, and the
 * same one `kernel_dense_add_scale_rms_norm_mul_f32_4` keeps in
 * `kernels/cuda/dense/dense_misc.cuh`: a golden that pins a decode step pins
 * the order its partial sums were added in, so there is one of these rather
 * than one per caller. */
__device__ __forceinline__ float shiso_warp_all_sum(float v) {
    for (unsigned off = 16u; off > 0u; off >>= 1u) {
        v += __shfl_xor_sync(SHISO_FULL_WARP, v, off);
    }
    return v;
}

/* The dynamic shared allocation a launch asks for. One declaration per
 * translation unit, so a kernel that wants scratch takes it from here rather
 * than declaring a second `extern __shared__` of a different type - which nvcc
 * would alias to the same address and a reader would not. */
extern __shared__ __align__(16) char shiso_generic_smem[];

/* Weights in one q6_K block. */
#define SHISO_QK_K 256

/* One q6_K block: low nibbles, the two high bits per weight, one signed scale
 * per sixteen weights and an fp16 block scale. 210 bytes, matching the loader's
 * `BlockQ6K`, the Metal block struct in
 * `crates/shiso-metal/metal/qwen35/gemv_ext_q5k_q6k.metal` and the dense
 * module's copy in `kernels/cuda/dense/dense_common.cuh`. */
struct shiso_block_q6_K {
    unsigned char ql[SHISO_QK_K / 2];
    unsigned char qh[SHISO_QK_K / 4];
    signed char   scales[SHISO_QK_K / 16];
    __half        d;
};
static_assert(sizeof(struct shiso_block_q6_K) == 210, "a q6_K block is 210 bytes");

/* Four-element chunks in one q6_K block, which is what the matvec walks. */
#define SHISO_Q6_K_CHUNKS (SHISO_QK_K / 4)

/* Four consecutive bytes of a quantized block, as one word.
 *
 * Two 16-bit loads, not four 8-bit ones. The four weights a `_t4` unpack keeps
 * always live in four adjacent bytes, and the matvec issues one `_t4` per
 * thread per chunk, so the byte-at-a-time spelling is what left these kernels
 * issuing four times the memory instructions they need - the defect this
 * exists to prevent, measured in `docs/handoff/cuda-decode-perf.md`.
 *
 * Every block struct here carries an `__half`, so a block pointer is already
 * two-byte aligned and every offset the callers below add to it is even. A
 * single 32-bit load is faster still - that file records how much - but a
 * q6_K block is 210 bytes, so every odd block starts two bytes off a word
 * boundary and a word load would fault on it. */
__device__ __forceinline__ unsigned shiso_load4(const unsigned char *p) {
    const unsigned short *h = (const unsigned short *) p;
    return (unsigned) h[0] | ((unsigned) h[1] << 16u);
}

/* Four consecutive elements of a q6_K block, `il` running 0..63.
 *
 * The four that are kept, and only those. The Metal twin - and this function
 * until 2026-09-21 - unpacked the whole sixteen-element group and threw twelve
 * away, on the argument that the group's scale is the same work either way.
 * The scale is; the twelve extra `ql` and `qh` byte loads are not, and at one
 * `_t4` per thread per chunk they were four times the memory instructions the
 * kernel needs. Dropping them is bit-exact - every kept element is built by
 * the same expression from the same bytes, folded in the same order - and took
 * this entry from 121.5 to 149.1 GB/s on the 497 MB output head. */
__device__ __forceinline__ void shiso_dequantize_q6_K_t4(
        const struct shiso_block_q6_K *xb, int il, float4 *reg) {
    const int il16 = il / 4;
    const int row = il & 3;
    const int n128 = il16 / 8;
    const int il128 = il16 & 7;
    const int quarter = il128 / 2;
    const int l0 = (il128 & 1) * 16;
    /* The odd quarters read the second 32 bytes of the half; the high quarters
     * read the upper nibble of the same bytes the low ones read. */
    const unsigned ql_base = (unsigned) n128 * 64u + (((quarter & 1) != 0) ? 32u : 0u);
    const unsigned qh_base = (unsigned) n128 * 32u;
    const unsigned sc_base = (unsigned) n128 * 8u + (unsigned) quarter * 2u
                           + (unsigned) (il128 & 1);
    const float d = __half2float(xb->d) * (float) ((int) xb->scales[sc_base]);
    const unsigned l = (unsigned) l0 + (unsigned) (4 * row);
    const unsigned ql4 = shiso_load4(xb->ql + ql_base + l);
    const unsigned qh4 = shiso_load4(xb->qh + qh_base + l);
    const unsigned hi_shift = (unsigned) quarter * 2u;
    float out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned lo = (ql4 >> (8u * (unsigned) i)) & 0xFFu;
        const unsigned hi = (qh4 >> (8u * (unsigned) i)) & 0xFFu;
        const unsigned v = ((quarter < 2) ? (lo & 0x0Fu) : (lo >> 4u))
                         | (((hi >> hi_shift) & 3u) << 4u);
        out[i] = d * (float) ((int) v - 32);
    }
    reg->x = out[0];
    reg->y = out[1];
    reg->z = out[2];
    reg->w = out[3];
}

/* --- the two k-quant super-block formats the qwen35 container carries ------
 *
 * `q4_K` for the bulk of its linear layers and `q5_K` for 48 of its tensors.
 * Both are 256 weights under an fp16 scale and an fp16 minimum, with six 6-bit
 * scale/minimum pairs packed into twelve bytes; they differ in whether there is
 * a fifth-bit plane.
 *
 * The arithmetic below is `dequantize_dense_q4_K` and `dequantize_dense_q5_K`
 * from `crates/shiso-metal/metal/generic/mul_mv_ext.metal` and
 * `crates/shiso-metal/metal/qwen35/gemv_ext_q5k_q6k.metal`, which is what
 * produced the committed goldens. Neither the element order nor the fold order
 * is adjusted: element `16*il + i` is built as `dl * q - ml` with the same
 * left-to-right association. That is the contract, not a style choice.
 *
 * What *is* adjusted is how many elements a caller pays for. The `_t4` entries
 * build the four they return and no more; only `shiso_dequantize_q4_K_16`
 * still fills a whole 16-element group, because the prefill GEMM consumes one.
 * Every element either spelling produces is bit-identical to the other's. */

/* One q4_K super-block: 144 bytes, matching the loader's `BlockQ4K`. */
struct shiso_block_q4_K {
    __half        d;
    __half        dmin;
    unsigned char scales[12];
    unsigned char qs[SHISO_QK_K / 2];
};
static_assert(sizeof(struct shiso_block_q4_K) == 144, "a q4_K block is 144 bytes");

/* One q5_K super-block: 176 bytes, matching the loader's `BlockQ5K`. The fifth
 * bit of every weight lives in its own plane, which is the whole difference. */
struct shiso_block_q5_K {
    __half        d;
    __half        dmin;
    unsigned char scales[12];
    unsigned char qh[SHISO_QK_K / 8];
    unsigned char qs[SHISO_QK_K / 2];
};
static_assert(sizeof(struct shiso_block_q5_K) == 176, "a q5_K block is 176 bytes");

/* Four-element chunks in one q4_K block, which is what the matvec walks. */
#define SHISO_Q4_K_CHUNKS (SHISO_QK_K / 4)

/* Four-element chunks in one q5_K block, which is what the matvec walks. */
#define SHISO_Q5_K_CHUNKS (SHISO_QK_K / 4)

/* The (scale, minimum) pair of group `j`, offset by `k` bytes.
 *
 * `shs_dense_q4_K_scale_min` verbatim in shape: the first four groups take six
 * bits out of one byte each, and the last four splice the low or high nibble of
 * one byte with the top two bits of another. Shared by both formats because the
 * twelve scale bytes are packed the same way in both. */
__device__ __forceinline__ void shiso_q4_K_scale_min(
        int j, int k, const unsigned char *q, unsigned char *scale, unsigned char *min) {
    if (j < 4) {
        *scale = (unsigned char) (q[j + 0 + k] & 63u);
        *min   = (unsigned char) (q[j + 4 + k] & 63u);
    } else {
        *scale = (unsigned char) ((q[j + 4 + k] & 0x0Fu) | ((q[j - 4 + k] & 0xC0u) >> 2u));
        *min   = (unsigned char) ((q[j + 4 + k] >> 4u)   | ((q[j - 0 + k] & 0xC0u) >> 2u));
    }
}

/* Elements `16*il .. 16*il+15` of a q4_K block, in element order.
 *
 * `il` walks sixteen 16-value groups. `is` picks the scale pair, the byte span
 * is `(il / 4) * 32 + 16 * (il & 1)`, and the low two bits of `il` decide both
 * the nibble half and whether the scale is divided by sixteen. */
__device__ __forceinline__ void shiso_dequantize_q4_K_16(
        const struct shiso_block_q4_K *xb, int il, float *reg) {
    const int is = (il / 4) * 2;
    const unsigned char *q = xb->qs + (il / 4) * 32 + 16 * (il & 1);
    const int ill = il & 3;
    unsigned char sc, mn;
    shiso_q4_K_scale_min(is, ill / 2, xb->scales, &sc, &mn);
    const float d = (ill < 2) ? __half2float(xb->d) : __half2float(xb->d) * (1.0f / 16.0f);
    const float dl = d * (float) sc;
    const float ml = __half2float(xb->dmin) * (float) mn;
    const unsigned mask = (ill < 2) ? 0x0Fu : 0xF0u;

#pragma unroll
    for (int i = 0; i < 16; ++i) {
        reg[i] = dl * (float) ((unsigned) q[i] & mask) - ml;
    }
}

/* Four consecutive elements of a q4_K block, `il` running 0..63.
 *
 * The four that are kept, and only those, for the reason
 * `shiso_dequantize_q6_K_t4` gives above. `shiso_dequantize_q4_K_16` stays
 * because the prefill GEMM in `mul_mm.cuh` consumes a whole group and pays
 * nothing for it; this is therefore the one format in this file spelled twice,
 * and the two spellings are held together by
 * `shiso_kerneltest::qwen35::gemv`'s `q4_k_matvec_matches_the_oracle_dot` and
 * `q4_k_prefill_gemm_matches_a_triple_loop`, which grade them against the same
 * oracle dot product. */
__device__ __forceinline__ void shiso_dequantize_q4_K_t4(
        const struct shiso_block_q4_K *xb, int il, float4 *reg) {
    const int il16 = il / 4;
    const int row = il & 3;
    const int is = (il16 / 4) * 2;
    const unsigned char *q = xb->qs + (il16 / 4) * 32 + 16 * (il16 & 1);
    const int ill = il16 & 3;
    unsigned char sc, mn;
    shiso_q4_K_scale_min(is, ill / 2, xb->scales, &sc, &mn);
    const float d = (ill < 2) ? __half2float(xb->d) : __half2float(xb->d) * (1.0f / 16.0f);
    const float dl = d * (float) sc;
    const float ml = __half2float(xb->dmin) * (float) mn;
    const unsigned mask = (ill < 2) ? 0x0Fu : 0xF0u;
    const unsigned qs4 = shiso_load4(q + 4 * row);
    float out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        out[i] = dl * (float) (((qs4 >> (8u * (unsigned) i)) & 0xFFu) & mask) - ml;
    }
    reg->x = out[0];
    reg->y = out[1];
    reg->z = out[2];
    reg->w = out[3];
}

/* Four consecutive elements of a q5_K block, `il` running 0..63.
 *
 * The four that are kept, and only those, for the reason
 * `shiso_dequantize_q6_K_t4` gives above. This format gains the most of the
 * three, because it reads two byte planes rather than one: 121.1 to
 * 147.0 GB/s on the qwen35 container's 8192x2560 tensor. */
__device__ __forceinline__ void shiso_dequantize_q5_K_t4(
        const struct shiso_block_q5_K *xb, int il, float4 *reg) {
    const int il16 = il / 4;
    const int row = il & 3;
    const int group = il16 / 2;
    const int l0 = (il16 & 1) * 16;
    unsigned char sc, mn;
    shiso_q4_K_scale_min(group, 0, xb->scales, &sc, &mn);
    const float dl = __half2float(xb->d) * (float) sc;
    const float ml = __half2float(xb->dmin) * (float) mn;
    const unsigned char *q = xb->qs + (group / 2) * 32 + l0;
    const unsigned char *qh = xb->qh + l0;
    const unsigned shift = (unsigned) (group & 1) * 4u;
    const unsigned hmask = 1u << (unsigned) group;
    const unsigned qs4 = shiso_load4(q + 4 * row);
    const unsigned qh4 = shiso_load4(qh + 4 * row);
    float out[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        unsigned v = (((qs4 >> (8u * (unsigned) i)) & 0xFFu) >> shift) & 0x0Fu;
        v += (((qh4 >> (8u * (unsigned) i)) & 0xFFu) & hmask) ? 16u : 0u;
        out[i] = dl * (float) v - ml;
    }
    reg->x = out[0];
    reg->y = out[1];
    reg->z = out[2];
    reg->w = out[3];
}

/* Elements in one q8_0 block. */
#define SHISO_QK8_0 32

/* One q8_0 block: a half scale and 32 signed bytes, in element order.
 *
 * `block_q8_0` in `crates/shiso-metal/metal/generic/00_common.metal`, which is
 * ggml's file layout rather than anybody's source. A qwen35 `Q4_K_M` container
 * keeps this family's two gate projections - `ssm_alpha` and `ssm_beta`, 24 of
 * each - at this codec, so the dense hybrid cannot decode a token without it. */
struct shiso_block_q8_0 {
    __half d;
    int8_t qs[SHISO_QK8_0];
};
static_assert(sizeof(struct shiso_block_q8_0) == 34, "a q8_0 block is 34 bytes");

/* Sixteen consecutive elements of a q8_0 block, `il` picking which half.
 *
 * `dequantize_q8_0` in `crates/shiso-metal/metal/generic/mul_mv_q8_0.metal`,
 * including where it rounds: the byte widens to float and the scale widens to
 * float, and the product is taken there. The GEMM's staging loop narrows the
 * result to half afterwards, which is the template's `S0` and not this
 * function's business. */
__device__ __forceinline__ void shiso_dequantize_q8_0_16(
        const struct shiso_block_q8_0 *xb, int il, float *out) {
    const float d = __half2float(xb->d);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        out[i] = (float) xb->qs[i + 16 * il] * d;
    }
}

/* Elements in one q4_0 block. */
#define SHISO_QK4_0 32
/* Four-element chunks in one q4_0 block, which is what the matvec walks. */
#define SHISO_Q4_0_CHUNKS (SHISO_QK4_0 / 4)

/* One q4_0 block: a half scale and 32 four-bit weights biased by -8, the low
 * nibble of byte j carrying element j and the high nibble element j + 16. */
struct shiso_block_q4_0 {
    __half  d;
    uint8_t qs[SHISO_QK4_0 / 2];
};
static_assert(sizeof(struct shiso_block_q4_0) == 18, "a q4_0 block is 18 bytes");

/* Four consecutive elements of a q4_0 block, `il` running 0..7 over the block's
 * eight four-element chunks.
 *
 * The arithmetic is the Metal twin's, including where it rounds: the `/16` and
 * the `-8 *` are done in half, exactly as `dequantize_q4_0_t4` does them, and
 * only then widened. Both are exact in half for any finite scale, but doing
 * them in float would be a second implementation of the format rather than the
 * same one, and the point of a twin is that it is not.
 *
 * Reading the block as 16-bit words is what lets one masked multiply produce
 * two elements: `mask0` takes a nibble of the low byte and `mask1` the same
 * nibble of the high byte, and the 1/256 in `d2` undoes that byte's shift.
 * `il/4` selects the nibble half, so chunks 0..3 are elements 0..15 and chunks
 * 4..7 are elements 16..31 - which makes chunk `il` exactly elements
 * `4*il .. 4*il+3` of the block, in element order. */
__device__ __forceinline__ void shiso_dequantize_q4_0_t4(
        const struct shiso_block_q4_0 *xb, int il, float4 *reg) {
    const unsigned short *qs = (const unsigned short *) xb + 1;
    const __half dh = xb->d;
    const float d1 = (il / 4) ? __half2float(__hdiv(dh, __float2half(16.0f)))
                              : __half2float(dh);
    const float d2 = d1 / 256.0f;
    const float md = __half2float(__hmul(__float2half(-8.0f), dh));
    const unsigned short mask0 = (il / 4) ? 0x00F0 : 0x000F;
    const unsigned short mask1 = (unsigned short) (mask0 << 8);

    const unsigned short w0 = qs[2 * (il % 4) + 0];
    const unsigned short w1 = qs[2 * (il % 4) + 1];
    reg->x = d1 * (float) (unsigned short) (w0 & mask0) + md;
    reg->y = d2 * (float) (unsigned short) (w0 & mask1) + md;
    reg->z = d1 * (float) (unsigned short) (w1 & mask0) + md;
    reg->w = d2 * (float) (unsigned short) (w1 & mask1) + md;
}

/* The whole 32-element block, in element order. The GEMM wants it this way
 * because its k tile is one block wide. */
__device__ __forceinline__ void shiso_dequantize_q4_0_block(
        const struct shiso_block_q4_0 *xb, float *out) {
#pragma unroll
    for (int il = 0; il < SHISO_Q4_0_CHUNKS; ++il) {
        float4 chunk;
        shiso_dequantize_q4_0_t4(xb, il, &chunk);
        out[4 * il + 0] = chunk.x;
        out[4 * il + 1] = chunk.y;
        out[4 * il + 2] = chunk.z;
        out[4 * il + 3] = chunk.w;
    }
}

#endif /* SHISO_GENERIC_CUH */
