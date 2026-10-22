/* The argument blocks, the block layouts and the dequantizers the generic
 * family reads.
 *
 * Nothing here is copied. The two argument structs are the field order
 * `crates/shiso-engine/src/launch.rs` declares, transcribed so the kernels
 * below read the same bytes the host packs; the block structs are the GGUF
 * container's file layouts, which are file data rather than anybody's source; the codebook
 * tables at the bottom are format data transcribed from
 * `crates/shiso-modelfile/src/quant_tables.rs`; and the dequantizers are the
 * arithmetic of the formats, derived from the Metal twins in
 * `kernels/metal/generic/mul_mv_ext.metal` so both devices round
 * the same way.
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
 * `kernels/metal/qwen35/gemv_ext_q5k_q6k.metal` and the dense
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
 * from `kernels/metal/generic/mul_mv_ext.metal` and
 * `kernels/metal/qwen35/gemv_ext_q5k_q6k.metal`, which is what
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
 * `block_q8_0` in `kernels/metal/generic/00_common.metal`, which is
 * the GGUF container's file layout rather than anybody's source. A qwen35 `Q4_K_M` container
 * keeps this family's two gate projections - `ssm_alpha` and `ssm_beta`, 24 of
 * each - at this codec, so the dense hybrid cannot decode a token without it. */
struct shiso_block_q8_0 {
    __half d;
    int8_t qs[SHISO_QK8_0];
};
static_assert(sizeof(struct shiso_block_q8_0) == 34, "a q8_0 block is 34 bytes");

/* Sixteen consecutive elements of a q8_0 block, `il` picking which half.
 *
 * `dequantize_q8_0` in `kernels/metal/generic/mul_mv_q8_0.metal`,
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

/* --- the mixed i-quant set and the 2-bit K-quant -------------------------
 *
 * The nine formats the Saluki export mixes - `q2_K`, `iq2_xxs`, `iq2_xs`,
 * `iq2_s`, `iq3_xxs`, `iq3_s`, `iq1_s`, `iq1_m` and `iq4_xs` - and the
 * codebook tables they decode through. Every one of the nine blocks is 256
 * weights, so the extended matvec's `kq` walk in `mul_mv_ext.cuh` needs no
 * per-format geometry: lane `s` of a block owns weights `32*s .. 32*s+32`,
 * which for the i-quants is exactly one of their eight 32-weight groups (and
 * for `q2_K` two of its sixteen 16-weight subblocks).
 *
 * The dequantizers below are the device twins of `deq32_*` in
 * `kernels/metal/generic/mul_mv_ext.metal` and of the host
 * reference decoders in `shiso_modelfile::iq_quant`: same block bytes in,
 * same f32 values out, in the same multiplication order. That is the
 * contract, not a style choice - a golden that pins a decode step pins the
 * order these products were folded in, and the three spellings are graded
 * against each other by `shiso_kerneltest::dense::iq`.
 *
 * The i-quants are non-linear: a weight is a codebook byte times a sign, not
 * an affine image of a small integer code, so there is no `(a, b)` pair to
 * fold the dot into. Each dequantizer therefore writes the fully dequantized
 * weights and hands back `ab = (1, 0)`; the walk's `b * sum(y)` leg then
 * contributes nothing and `a * sum(q * y)` is the plain dot of the weights.
 * `q2_K` is affine per 16-weight subblock but the two subblocks of a
 * 32-weight lane carry different scale/min pairs, so it takes the same route.
 *
 * The element order is stated once here rather than nine times: every
 * dequantizer fills `qv` through a `float` view, and the Metal index
 * `qv[2*l + j/4][j % 4]` is exactly element `8*l + j` of that view (and
 * `qv[h*4 + l/4][l % 4]` exactly element `16*h + l`). The two spellings
 * address the same slot; the flat one is what makes the correspondence
 * checkable by eye.
 *
 * The tables are format data transcribed from
 * `crates/shiso-modelfile/src/quant_tables.rs` - the same source the Metal
 * copies in `iq_tables.metal` were transcribed from, and the values here and
 * there were generated from it together. They are `static const __device__`
 * rather than `__constant__` because a matvec warp's lanes hold nine
 * different grid indices at once, and constant memory serializes a warp
 * across distinct addresses; global reads of a read-only array hit the
 * cache instead. `kernel_iq_table_copy` in `generic.cu` hands them to a
 * buffer so a test can byte-compare them against the host tables. */

/* Weights in one q2_K block: a 2-bit plane per 128 weights, one scale byte
 * per 16 weights, and the fp16 scale and minimum at the end. 84 bytes,
 * matching the loader's `BlockQ2K`. */
struct shiso_block_q2_K {
    unsigned char scales[SHISO_QK_K / 16];
    unsigned char qs[SHISO_QK_K / 4];
    __half        d;
    __half        dmin;
};
static_assert(sizeof(struct shiso_block_q2_K) == 84, "a q2_K block is 84 bytes");

/* Weights in one iq2_xxs block: an fp16 scale and 32 words of four packed
 * grid indices and sign-and-scale patterns. 66 bytes, matching the loader's
 * `BlockIq2Xxs`. The words are read as `unsigned short` pairs because a
 * 66-byte block only ever lands on a 2-byte boundary. */
struct shiso_block_iq2_xxs {
    __half         d;
    unsigned short qs[32];
};
static_assert(sizeof(struct shiso_block_iq2_xxs) == 66, "an iq2_xxs block is 66 bytes");

/* Weights in one iq2_xs block: nine index bits and seven sign bits a word,
 * two 4-bit scales a byte. 74 bytes, matching the loader's `BlockIq2Xs`. */
struct shiso_block_iq2_xs {
    __half         d;
    unsigned short qs[32];
    unsigned char  scales[8];
};
static_assert(sizeof(struct shiso_block_iq2_xs) == 74, "an iq2_xs block is 74 bytes");

/* Weights in one iq2_s block: the grid indices in the first half of `qs`,
 * the per-weight signs in the second, two high index bits per quad in `qh`.
 * 82 bytes, matching the loader's `BlockIq2S`. */
struct shiso_block_iq2_s {
    __half        d;
    unsigned char qs[64];
    unsigned char qh[8];
    unsigned char scales[8];
};
static_assert(sizeof(struct shiso_block_iq2_s) == 82, "an iq2_s block is 82 bytes");

/* Weights in one iq3_xxs block: 64 grid-index bytes, then one
 * sign-and-scale word per 32-weight group after them. 98 bytes, matching
 * the loader's `BlockIq3Xxs`. */
struct shiso_block_iq3_xxs {
    __half        d;
    unsigned char qs[96];
};
static_assert(sizeof(struct shiso_block_iq3_xxs) == 98, "an iq3_xxs block is 98 bytes");

/* Weights in one iq3_s block: odd 4-bit scales (`1 + 2 * nibble`) one byte
 * per two groups, the high index bit of each index in `qh`, and one sign
 * byte per quad. 110 bytes, matching the loader's `BlockIq3S`. */
struct shiso_block_iq3_s {
    __half        d;
    unsigned char qs[64];
    unsigned char qh[8];
    unsigned char signs[32];
    unsigned char scales[4];
};
static_assert(sizeof(struct shiso_block_iq3_s) == 110, "an iq3_s block is 110 bytes");

/* Weights in one iq1_s block: 11-bit indices into the ternary grid, split
 * between a byte in `qs` and three high bits per quad in `qh`, whose bits
 * 12..14 carry a scale multiplier and bit 15 the sign of the grid shift.
 * 50 bytes, matching the loader's `BlockIq1S`. */
struct shiso_block_iq1_s {
    __half         d;
    unsigned char  qs[32];
    unsigned short qh[8];
};
static_assert(sizeof(struct shiso_block_iq1_s) == 50, "an iq1_s block is 50 bytes");

/* Weights in one iq1_m block: the same ternary grid, but no stored `d` -
 * the fp16 is the four top nibbles of the odd scale bytes, least significant
 * first - and a 3-bit scale multiplier per 16-weight subblock. 56 bytes,
 * matching the loader's `BlockIq1M`. */
struct shiso_block_iq1_m {
    unsigned char qs[32];
    unsigned char qh[16];
    unsigned char scales[8];
};
static_assert(sizeof(struct shiso_block_iq1_m) == 56, "an iq1_m block is 56 bytes");

/* Weights in one iq4_xs block: a 6-bit subblock scale split across two
 * planes, offset so 32 means no scaling, and nibble indices into the
 * non-linear codebook. 136 bytes, matching the loader's `BlockIq4Xs`. */
struct shiso_block_iq4_xs {
    __half         d;
    unsigned short scales_h;
    unsigned char  scales_l[4];
    unsigned char  qs[128];
};
static_assert(sizeof(struct shiso_block_iq4_xs) == 136, "an iq4_xs block is 136 bytes");

/* Weight grid for the 2-bit extra-extra-small format: eight packed weights an index.
 * Transcribed from `IQ2XXS_GRID` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) uint64_t SHISO_IQ2XXS_GRID[256] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x08080808082b0808ull,
    0x08080808082b082bull, 0x08080808082b2b08ull, 0x08080808082b2b2bull, 0x0808080819080819ull,
    0x0808080819081908ull, 0x0808080819190808ull, 0x0808080819192b08ull, 0x08080808192b0819ull,
    0x08080808192b1908ull, 0x080808082b080808ull, 0x080808082b08082bull, 0x080808082b082b2bull,
    0x080808082b2b082bull, 0x0808081908080819ull, 0x0808081908081908ull, 0x0808081908190808ull,
    0x0808081908191919ull, 0x0808081919080808ull, 0x080808192b081908ull, 0x080808192b192b08ull,
    0x0808082b08080808ull, 0x0808082b0808082bull, 0x0808082b082b082bull, 0x0808082b2b08082bull,
    0x0808190808080819ull, 0x0808190808081908ull, 0x0808190808190808ull, 0x08081908082b0819ull,
    0x08081908082b1908ull, 0x0808190819080808ull, 0x080819081908082bull, 0x0808190819082b08ull,
    0x08081908192b0808ull, 0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b190808ull,
    0x080819082b2b1908ull, 0x0808191908080808ull, 0x080819190808082bull, 0x0808191908082b08ull,
    0x08081919082b0808ull, 0x080819191908192bull, 0x08081919192b2b19ull, 0x080819192b080808ull,
    0x080819192b190819ull, 0x0808192b08082b19ull, 0x0808192b08190808ull, 0x0808192b19080808ull,
    0x0808192b2b081908ull, 0x0808192b2b2b1908ull, 0x08082b0808080808ull, 0x08082b0808081919ull,
    0x08082b0808082b08ull, 0x08082b0808191908ull, 0x08082b08082b2b08ull, 0x08082b0819080819ull,
    0x08082b0819081908ull, 0x08082b0819190808ull, 0x08082b081919082bull, 0x08082b082b082b08ull,
    0x08082b1908081908ull, 0x08082b1919080808ull, 0x08082b2b0808082bull, 0x08082b2b08191908ull,
    0x0819080808080819ull, 0x0819080808081908ull, 0x0819080808190808ull, 0x08190808082b0819ull,
    0x0819080819080808ull, 0x08190808192b0808ull, 0x081908082b081908ull, 0x081908082b190808ull,
    0x081908082b191919ull, 0x0819081908080808ull, 0x0819081908082b08ull, 0x08190819082b0808ull,
    0x0819081919190808ull, 0x0819081919192b2bull, 0x081908192b080808ull, 0x0819082b082b1908ull,
    0x0819082b19081919ull, 0x0819190808080808ull, 0x0819190808082b08ull, 0x08191908082b0808ull,
    0x08191908082b1919ull, 0x0819190819082b19ull, 0x081919082b080808ull, 0x0819191908192b08ull,
    0x08191919192b082bull, 0x0819192b08080808ull, 0x0819192b0819192bull, 0x08192b0808080819ull,
    0x08192b0808081908ull, 0x08192b0808190808ull, 0x08192b0819080808ull, 0x08192b082b080819ull,
    0x08192b1908080808ull, 0x08192b1908081919ull, 0x08192b192b2b0808ull, 0x08192b2b19190819ull,
    0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808082b2bull, 0x082b080819081908ull,
    0x082b0808192b0819ull, 0x082b08082b080808ull, 0x082b08082b08082bull, 0x082b0819082b2b19ull,
    0x082b081919082b08ull, 0x082b082b08080808ull, 0x082b082b0808082bull, 0x082b190808080819ull,
    0x082b190808081908ull, 0x082b190808190808ull, 0x082b190819080808ull, 0x082b19081919192bull,
    0x082b191908080808ull, 0x082b191919080819ull, 0x082b1919192b1908ull, 0x082b192b2b190808ull,
    0x082b2b0808082b08ull, 0x082b2b08082b0808ull, 0x082b2b082b191908ull, 0x082b2b2b19081908ull,
    0x1908080808080819ull, 0x1908080808081908ull, 0x1908080808190808ull, 0x1908080808192b08ull,
    0x19080808082b0819ull, 0x19080808082b1908ull, 0x1908080819080808ull, 0x1908080819082b08ull,
    0x190808081919192bull, 0x19080808192b0808ull, 0x190808082b080819ull, 0x190808082b081908ull,
    0x190808082b190808ull, 0x1908081908080808ull, 0x19080819082b0808ull, 0x19080819192b0819ull,
    0x190808192b080808ull, 0x190808192b081919ull, 0x1908082b08080819ull, 0x1908082b08190808ull,
    0x1908082b19082b08ull, 0x1908082b1919192bull, 0x1908082b192b2b08ull, 0x1908190808080808ull,
    0x1908190808082b08ull, 0x19081908082b0808ull, 0x190819082b080808ull, 0x190819082b192b19ull,
    0x190819190819082bull, 0x19081919082b1908ull, 0x1908192b08080808ull, 0x19082b0808080819ull,
    0x19082b0808081908ull, 0x19082b0808190808ull, 0x19082b0819080808ull, 0x19082b0819081919ull,
    0x19082b1908080808ull, 0x19082b1919192b08ull, 0x19082b19192b0819ull, 0x19082b192b08082bull,
    0x19082b2b19081919ull, 0x19082b2b2b190808ull, 0x1919080808080808ull, 0x1919080808082b08ull,
    0x1919080808190819ull, 0x1919080808192b19ull, 0x19190808082b0808ull, 0x191908082b080808ull,
    0x191908082b082b08ull, 0x1919081908081908ull, 0x191908191908082bull, 0x191908192b2b1908ull,
    0x1919082b2b190819ull, 0x191919082b190808ull, 0x191919082b19082bull, 0x1919191908082b2bull,
    0x1919192b08080819ull, 0x1919192b19191908ull, 0x19192b0808080808ull, 0x19192b0808190819ull,
    0x19192b0808192b19ull, 0x19192b08192b1908ull, 0x19192b1919080808ull, 0x19192b2b08082b08ull,
    0x192b080808081908ull, 0x192b080808190808ull, 0x192b080819080808ull, 0x192b0808192b2b08ull,
    0x192b081908080808ull, 0x192b081919191919ull, 0x192b082b08192b08ull, 0x192b082b192b0808ull,
    0x192b190808080808ull, 0x192b190808081919ull, 0x192b191908190808ull, 0x192b19190819082bull,
    0x192b19192b081908ull, 0x192b2b081908082bull, 0x2b08080808080808ull, 0x2b0808080808082bull,
    0x2b08080808082b2bull, 0x2b08080819080819ull, 0x2b0808082b08082bull, 0x2b08081908081908ull,
    0x2b08081908192b08ull, 0x2b08081919080808ull, 0x2b08082b08190819ull, 0x2b08190808080819ull,
    0x2b08190808081908ull, 0x2b08190808190808ull, 0x2b08190808191919ull, 0x2b08190819080808ull,
    0x2b081908192b0808ull, 0x2b08191908080808ull, 0x2b0819191908192bull, 0x2b0819192b191908ull,
    0x2b08192b08082b19ull, 0x2b08192b19080808ull, 0x2b08192b192b0808ull, 0x2b082b080808082bull,
    0x2b082b1908081908ull, 0x2b082b2b08190819ull, 0x2b19080808081908ull, 0x2b19080808190808ull,
    0x2b190808082b1908ull, 0x2b19080819080808ull, 0x2b1908082b2b0819ull, 0x2b1908190819192bull,
    0x2b1908192b080808ull, 0x2b19082b19081919ull, 0x2b19190808080808ull, 0x2b191908082b082bull,
    0x2b19190819081908ull, 0x2b19191919190819ull, 0x2b192b082b080819ull, 0x2b192b19082b0808ull,
    0x2b2b08080808082bull, 0x2b2b080819190808ull, 0x2b2b08082b081919ull, 0x2b2b081908082b19ull,
    0x2b2b082b08080808ull, 0x2b2b190808192b08ull, 0x2b2b2b0819190808ull, 0x2b2b2b1908081908ull,
};

/* Weight grid for the 2-bit small format: eight packed weights an index.
 * Transcribed from `IQ2XS_GRID` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) uint64_t SHISO_IQ2XS_GRID[512] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x080808080819192bull,
    0x0808080808192b19ull, 0x08080808082b0808ull, 0x08080808082b082bull, 0x08080808082b1919ull,
    0x08080808082b2b08ull, 0x0808080819080819ull, 0x0808080819081908ull, 0x080808081908192bull,
    0x0808080819082b19ull, 0x0808080819190808ull, 0x080808081919082bull, 0x0808080819191919ull,
    0x0808080819192b08ull, 0x08080808192b0819ull, 0x08080808192b1908ull, 0x080808082b080808ull,
    0x080808082b08082bull, 0x080808082b081919ull, 0x080808082b082b08ull, 0x080808082b190819ull,
    0x080808082b191908ull, 0x080808082b192b19ull, 0x080808082b2b0808ull, 0x0808081908080819ull,
    0x0808081908081908ull, 0x080808190808192bull, 0x0808081908082b19ull, 0x0808081908190808ull,
    0x080808190819082bull, 0x0808081908191919ull, 0x0808081908192b08ull, 0x0808081908192b2bull,
    0x08080819082b0819ull, 0x08080819082b1908ull, 0x0808081919080808ull, 0x080808191908082bull,
    0x0808081919081919ull, 0x0808081919082b08ull, 0x0808081919190819ull, 0x0808081919191908ull,
    0x08080819192b0808ull, 0x08080819192b2b08ull, 0x080808192b080819ull, 0x080808192b081908ull,
    0x080808192b190808ull, 0x0808082b08080808ull, 0x0808082b0808082bull, 0x0808082b08081919ull,
    0x0808082b08082b08ull, 0x0808082b08190819ull, 0x0808082b08191908ull, 0x0808082b082b0808ull,
    0x0808082b19080819ull, 0x0808082b19081908ull, 0x0808082b19190808ull, 0x0808082b19191919ull,
    0x0808082b2b080808ull, 0x0808082b2b082b2bull, 0x0808190808080819ull, 0x0808190808081908ull,
    0x080819080808192bull, 0x0808190808082b19ull, 0x0808190808190808ull, 0x080819080819082bull,
    0x0808190808191919ull, 0x0808190808192b08ull, 0x08081908082b0819ull, 0x08081908082b1908ull,
    0x0808190819080808ull, 0x080819081908082bull, 0x0808190819081919ull, 0x0808190819082b08ull,
    0x0808190819190819ull, 0x0808190819191908ull, 0x080819081919192bull, 0x08081908192b0808ull,
    0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b190808ull, 0x0808191908080808ull,
    0x080819190808082bull, 0x0808191908081919ull, 0x0808191908082b08ull, 0x0808191908190819ull,
    0x0808191908191908ull, 0x08081919082b0808ull, 0x0808191919080819ull, 0x0808191919081908ull,
    0x0808191919190808ull, 0x08081919192b0819ull, 0x080819192b080808ull, 0x0808192b08080819ull,
    0x0808192b08081908ull, 0x0808192b08190808ull, 0x0808192b082b192bull, 0x0808192b19080808ull,
    0x0808192b1908082bull, 0x0808192b2b081908ull, 0x08082b0808080808ull, 0x08082b080808082bull,
    0x08082b0808081919ull, 0x08082b0808082b08ull, 0x08082b0808082b2bull, 0x08082b0808190819ull,
    0x08082b0808191908ull, 0x08082b08082b0808ull, 0x08082b08082b1919ull, 0x08082b0819080819ull,
    0x08082b0819081908ull, 0x08082b0819190808ull, 0x08082b0819192b08ull, 0x08082b082b080808ull,
    0x08082b082b2b0808ull, 0x08082b082b2b2b2bull, 0x08082b1908080819ull, 0x08082b1908081908ull,
    0x08082b1908190808ull, 0x08082b1919080808ull, 0x08082b192b080819ull, 0x08082b192b082b19ull,
    0x08082b2b08080808ull, 0x08082b2b082b0808ull, 0x08082b2b082b2b08ull, 0x08082b2b2b19192bull,
    0x08082b2b2b2b0808ull, 0x0819080808080819ull, 0x0819080808081908ull, 0x081908080808192bull,
    0x0819080808082b19ull, 0x0819080808190808ull, 0x081908080819082bull, 0x0819080808191919ull,
    0x0819080808192b08ull, 0x08190808082b0819ull, 0x08190808082b1908ull, 0x0819080819080808ull,
    0x081908081908082bull, 0x0819080819081919ull, 0x0819080819082b08ull, 0x0819080819190819ull,
    0x0819080819191908ull, 0x08190808192b0808ull, 0x08190808192b2b2bull, 0x081908082b080819ull,
    0x081908082b081908ull, 0x081908082b190808ull, 0x0819081908080808ull, 0x081908190808082bull,
    0x0819081908081919ull, 0x0819081908082b08ull, 0x0819081908190819ull, 0x0819081908191908ull,
    0x08190819082b0808ull, 0x0819081919080819ull, 0x0819081919081908ull, 0x0819081919190808ull,
    0x081908192b080808ull, 0x081908192b191908ull, 0x081908192b19192bull, 0x0819082b08080819ull,
    0x0819082b08081908ull, 0x0819082b0808192bull, 0x0819082b08190808ull, 0x0819082b19080808ull,
    0x0819082b192b0808ull, 0x0819190808080808ull, 0x081919080808082bull, 0x0819190808081919ull,
    0x0819190808082b08ull, 0x0819190808190819ull, 0x0819190808191908ull, 0x08191908082b0808ull,
    0x0819190819080819ull, 0x0819190819081908ull, 0x0819190819082b19ull, 0x0819190819190808ull,
    0x08191908192b1908ull, 0x081919082b080808ull, 0x0819191908080819ull, 0x0819191908081908ull,
    0x0819191908190808ull, 0x0819191919080808ull, 0x0819192b08080808ull, 0x0819192b08191908ull,
    0x0819192b19082b19ull, 0x08192b0808080819ull, 0x08192b0808081908ull, 0x08192b0808190808ull,
    0x08192b080819082bull, 0x08192b0819080808ull, 0x08192b0819191908ull, 0x08192b082b08192bull,
    0x08192b1908080808ull, 0x08192b1908081919ull, 0x08192b19192b192bull, 0x08192b2b19190819ull,
    0x08192b2b2b2b2b19ull, 0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808081919ull,
    0x082b080808082b08ull, 0x082b080808082b2bull, 0x082b080808190819ull, 0x082b080808191908ull,
    0x082b0808082b0808ull, 0x082b080819080819ull, 0x082b080819081908ull, 0x082b080819190808ull,
    0x082b08082b080808ull, 0x082b08082b2b0808ull, 0x082b081908080819ull, 0x082b081908081908ull,
    0x082b081908190808ull, 0x082b081919080808ull, 0x082b081919082b08ull, 0x082b0819192b1919ull,
    0x082b082b08080808ull, 0x082b082b082b082bull, 0x082b082b2b080808ull, 0x082b082b2b2b2b08ull,
    0x082b190808080819ull, 0x082b190808081908ull, 0x082b190808190808ull, 0x082b1908082b2b19ull,
    0x082b190819080808ull, 0x082b191908080808ull, 0x082b191919080819ull, 0x082b19191919082bull,
    0x082b19192b192b19ull, 0x082b192b08080819ull, 0x082b192b08192b2bull, 0x082b192b2b2b192bull,
    0x082b2b0808080808ull, 0x082b2b0808082b08ull, 0x082b2b0808082b2bull, 0x082b2b08082b0808ull,
    0x082b2b0819191919ull, 0x082b2b082b082b08ull, 0x082b2b082b2b082bull, 0x082b2b19192b2b08ull,
    0x082b2b192b190808ull, 0x082b2b2b08082b08ull, 0x082b2b2b082b0808ull, 0x082b2b2b2b08082bull,
    0x082b2b2b2b082b08ull, 0x082b2b2b2b082b2bull, 0x1908080808080819ull, 0x1908080808081908ull,
    0x190808080808192bull, 0x1908080808082b19ull, 0x1908080808190808ull, 0x190808080819082bull,
    0x1908080808191919ull, 0x1908080808192b08ull, 0x19080808082b0819ull, 0x19080808082b1908ull,
    0x1908080819080808ull, 0x190808081908082bull, 0x1908080819081919ull, 0x1908080819082b08ull,
    0x1908080819082b2bull, 0x1908080819190819ull, 0x1908080819191908ull, 0x19080808192b0808ull,
    0x19080808192b1919ull, 0x190808082b080819ull, 0x190808082b081908ull, 0x190808082b190808ull,
    0x1908081908080808ull, 0x190808190808082bull, 0x1908081908081919ull, 0x1908081908082b08ull,
    0x1908081908190819ull, 0x1908081908191908ull, 0x19080819082b0808ull, 0x1908081919080819ull,
    0x1908081919081908ull, 0x1908081919190808ull, 0x190808192b080808ull, 0x190808192b081919ull,
    0x190808192b2b082bull, 0x1908082b08080819ull, 0x1908082b08081908ull, 0x1908082b08190808ull,
    0x1908082b0819082bull, 0x1908082b082b2b19ull, 0x1908082b19080808ull, 0x1908190808080808ull,
    0x190819080808082bull, 0x1908190808081919ull, 0x1908190808082b08ull, 0x1908190808190819ull,
    0x1908190808191908ull, 0x1908190808192b19ull, 0x19081908082b0808ull, 0x1908190819080819ull,
    0x1908190819081908ull, 0x1908190819190808ull, 0x190819082b080808ull, 0x190819082b191908ull,
    0x1908191908080819ull, 0x1908191908081908ull, 0x1908191908190808ull, 0x19081919082b1908ull,
    0x1908191919080808ull, 0x190819192b192b2bull, 0x1908192b08080808ull, 0x1908192b08082b2bull,
    0x1908192b19081908ull, 0x1908192b19190808ull, 0x19082b0808080819ull, 0x19082b0808081908ull,
    0x19082b0808190808ull, 0x19082b0819080808ull, 0x19082b0819081919ull, 0x19082b0819191908ull,
    0x19082b08192b082bull, 0x19082b1908080808ull, 0x19082b1908190819ull, 0x19082b1919081908ull,
    0x19082b1919190808ull, 0x19082b19192b2b19ull, 0x19082b2b08081908ull, 0x1919080808080808ull,
    0x191908080808082bull, 0x1919080808081919ull, 0x1919080808082b08ull, 0x1919080808190819ull,
    0x1919080808191908ull, 0x19190808082b0808ull, 0x19190808082b2b08ull, 0x1919080819080819ull,
    0x1919080819081908ull, 0x1919080819190808ull, 0x191908082b080808ull, 0x1919081908080819ull,
    0x1919081908081908ull, 0x1919081908190808ull, 0x1919081908191919ull, 0x1919081919080808ull,
    0x191908191908082bull, 0x1919082b08080808ull, 0x1919082b19081908ull, 0x1919082b2b2b2b2bull,
    0x1919190808080819ull, 0x1919190808081908ull, 0x1919190808190808ull, 0x19191908082b0819ull,
    0x1919190819080808ull, 0x19191908192b0808ull, 0x191919082b080819ull, 0x191919082b2b0819ull,
    0x1919191908080808ull, 0x1919191908082b08ull, 0x191919192b080808ull, 0x191919192b082b08ull,
    0x1919192b082b0819ull, 0x1919192b192b2b08ull, 0x1919192b2b2b0819ull, 0x19192b0808080808ull,
    0x19192b0808191908ull, 0x19192b0819080819ull, 0x19192b0819190808ull, 0x19192b082b192b19ull,
    0x19192b1908192b2bull, 0x19192b1919080808ull, 0x19192b191908082bull, 0x19192b2b2b081919ull,
    0x192b080808080819ull, 0x192b080808081908ull, 0x192b080808190808ull, 0x192b080819080808ull,
    0x192b080819191908ull, 0x192b0808192b082bull, 0x192b08082b08192bull, 0x192b08082b2b2b19ull,
    0x192b081908080808ull, 0x192b082b082b1908ull, 0x192b082b19082b2bull, 0x192b082b2b19082bull,
    0x192b190808080808ull, 0x192b19080819192bull, 0x192b191908190808ull, 0x192b191919080808ull,
    0x192b191919081919ull, 0x192b19192b2b1908ull, 0x192b2b0808080819ull, 0x192b2b08192b2b2bull,
    0x192b2b19082b1919ull, 0x192b2b2b0808192bull, 0x192b2b2b19191908ull, 0x192b2b2b192b082bull,
    0x2b08080808080808ull, 0x2b0808080808082bull, 0x2b08080808081919ull, 0x2b08080808082b08ull,
    0x2b08080808190819ull, 0x2b08080808191908ull, 0x2b080808082b0808ull, 0x2b080808082b2b2bull,
    0x2b08080819080819ull, 0x2b08080819081908ull, 0x2b08080819190808ull, 0x2b0808082b080808ull,
    0x2b0808082b08082bull, 0x2b0808082b2b2b08ull, 0x2b0808082b2b2b2bull, 0x2b08081908080819ull,
    0x2b08081908081908ull, 0x2b0808190808192bull, 0x2b08081908190808ull, 0x2b08081919080808ull,
    0x2b08081919190819ull, 0x2b08081919192b19ull, 0x2b08082b08080808ull, 0x2b08082b082b0808ull,
    0x2b08082b2b080808ull, 0x2b08082b2b08082bull, 0x2b08082b2b2b0808ull, 0x2b08082b2b2b2b08ull,
    0x2b08190808080819ull, 0x2b08190808081908ull, 0x2b08190808190808ull, 0x2b0819080819082bull,
    0x2b08190808191919ull, 0x2b08190819080808ull, 0x2b081908192b0808ull, 0x2b0819082b082b19ull,
    0x2b08191908080808ull, 0x2b08191919081908ull, 0x2b0819192b2b1919ull, 0x2b08192b08192b08ull,
    0x2b08192b192b2b2bull, 0x2b082b0808080808ull, 0x2b082b0808082b08ull, 0x2b082b08082b1919ull,
    0x2b082b0819192b2bull, 0x2b082b082b080808ull, 0x2b082b082b08082bull, 0x2b082b082b2b2b08ull,
    0x2b082b190808192bull, 0x2b082b2b082b082bull, 0x2b082b2b2b080808ull, 0x2b082b2b2b082b08ull,
    0x2b082b2b2b19192bull, 0x2b082b2b2b2b2b08ull, 0x2b19080808080819ull, 0x2b19080808081908ull,
    0x2b19080808190808ull, 0x2b19080819080808ull, 0x2b1908081919192bull, 0x2b1908082b081908ull,
    0x2b19081908080808ull, 0x2b190819082b082bull, 0x2b190819192b1908ull, 0x2b19082b1919192bull,
    0x2b19082b2b082b19ull, 0x2b19190808080808ull, 0x2b19190808081919ull, 0x2b19190819081908ull,
    0x2b19190819190808ull, 0x2b19190819192b08ull, 0x2b191919082b2b19ull, 0x2b1919192b190808ull,
    0x2b1919192b19082bull, 0x2b19192b19080819ull, 0x2b192b0819190819ull, 0x2b192b082b2b192bull,
    0x2b192b1919082b19ull, 0x2b192b2b08191919ull, 0x2b192b2b192b0808ull, 0x2b2b080808080808ull,
    0x2b2b08080808082bull, 0x2b2b080808082b08ull, 0x2b2b080808082b2bull, 0x2b2b0808082b0808ull,
    0x2b2b0808082b2b2bull, 0x2b2b08082b2b0808ull, 0x2b2b081919190819ull, 0x2b2b081919192b19ull,
    0x2b2b08192b2b192bull, 0x2b2b082b08080808ull, 0x2b2b082b0808082bull, 0x2b2b082b08082b08ull,
    0x2b2b082b082b2b2bull, 0x2b2b082b2b080808ull, 0x2b2b082b2b2b0808ull, 0x2b2b190819080808ull,
    0x2b2b19082b191919ull, 0x2b2b192b192b1919ull, 0x2b2b192b2b192b08ull, 0x2b2b2b0808082b2bull,
    0x2b2b2b08082b0808ull, 0x2b2b2b08082b082bull, 0x2b2b2b08082b2b08ull, 0x2b2b2b082b2b0808ull,
    0x2b2b2b082b2b2b08ull, 0x2b2b2b1908081908ull, 0x2b2b2b192b081908ull, 0x2b2b2b192b08192bull,
    0x2b2b2b2b082b2b08ull, 0x2b2b2b2b082b2b2bull, 0x2b2b2b2b2b190819ull, 0x2b2b2b2b2b2b2b2bull,
};

/* Weight grid for the 2-bit small-with-signs format: eight packed weights an index.
 * Transcribed from `IQ2S_GRID` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) uint64_t SHISO_IQ2S_GRID[1024] = {
    0x0808080808080808ull, 0x080808080808082bull, 0x0808080808081919ull, 0x0808080808082b08ull,
    0x0808080808082b2bull, 0x0808080808190819ull, 0x0808080808191908ull, 0x080808080819192bull,
    0x0808080808192b19ull, 0x08080808082b0808ull, 0x08080808082b082bull, 0x08080808082b1919ull,
    0x08080808082b2b08ull, 0x0808080819080819ull, 0x0808080819081908ull, 0x080808081908192bull,
    0x0808080819082b19ull, 0x0808080819190808ull, 0x080808081919082bull, 0x0808080819191919ull,
    0x0808080819192b08ull, 0x08080808192b0819ull, 0x08080808192b1908ull, 0x08080808192b192bull,
    0x08080808192b2b19ull, 0x080808082b080808ull, 0x080808082b08082bull, 0x080808082b081919ull,
    0x080808082b082b08ull, 0x080808082b190819ull, 0x080808082b191908ull, 0x080808082b2b0808ull,
    0x080808082b2b1919ull, 0x080808082b2b2b2bull, 0x0808081908080819ull, 0x0808081908081908ull,
    0x080808190808192bull, 0x0808081908082b19ull, 0x0808081908190808ull, 0x080808190819082bull,
    0x0808081908191919ull, 0x0808081908192b08ull, 0x08080819082b0819ull, 0x08080819082b1908ull,
    0x0808081919080808ull, 0x080808191908082bull, 0x0808081919081919ull, 0x0808081919082b08ull,
    0x0808081919190819ull, 0x0808081919191908ull, 0x080808191919192bull, 0x0808081919192b19ull,
    0x08080819192b0808ull, 0x08080819192b1919ull, 0x08080819192b2b08ull, 0x080808192b080819ull,
    0x080808192b081908ull, 0x080808192b190808ull, 0x080808192b19082bull, 0x080808192b191919ull,
    0x080808192b2b0819ull, 0x080808192b2b1908ull, 0x0808082b08080808ull, 0x0808082b0808082bull,
    0x0808082b08081919ull, 0x0808082b08082b08ull, 0x0808082b08190819ull, 0x0808082b08191908ull,
    0x0808082b082b0808ull, 0x0808082b082b2b2bull, 0x0808082b19080819ull, 0x0808082b19081908ull,
    0x0808082b1908192bull, 0x0808082b19082b19ull, 0x0808082b19190808ull, 0x0808082b19191919ull,
    0x0808082b2b080808ull, 0x0808082b2b081919ull, 0x0808082b2b082b2bull, 0x0808082b2b191908ull,
    0x0808082b2b2b082bull, 0x0808190808080819ull, 0x0808190808081908ull, 0x080819080808192bull,
    0x0808190808082b19ull, 0x0808190808190808ull, 0x080819080819082bull, 0x0808190808191919ull,
    0x0808190808192b08ull, 0x08081908082b0819ull, 0x08081908082b1908ull, 0x08081908082b192bull,
    0x08081908082b2b19ull, 0x0808190819080808ull, 0x080819081908082bull, 0x0808190819081919ull,
    0x0808190819082b08ull, 0x0808190819082b2bull, 0x0808190819190819ull, 0x0808190819191908ull,
    0x080819081919192bull, 0x0808190819192b19ull, 0x08081908192b0808ull, 0x08081908192b082bull,
    0x08081908192b1919ull, 0x080819082b080819ull, 0x080819082b081908ull, 0x080819082b08192bull,
    0x080819082b082b19ull, 0x080819082b190808ull, 0x080819082b191919ull, 0x080819082b192b08ull,
    0x080819082b2b0819ull, 0x080819082b2b1908ull, 0x0808191908080808ull, 0x080819190808082bull,
    0x0808191908081919ull, 0x0808191908082b08ull, 0x0808191908082b2bull, 0x0808191908190819ull,
    0x0808191908191908ull, 0x080819190819192bull, 0x0808191908192b19ull, 0x08081919082b0808ull,
    0x08081919082b1919ull, 0x08081919082b2b08ull, 0x0808191919080819ull, 0x0808191919081908ull,
    0x080819191908192bull, 0x0808191919082b19ull, 0x0808191919190808ull, 0x080819191919082bull,
    0x0808191919191919ull, 0x0808191919192b08ull, 0x08081919192b0819ull, 0x08081919192b1908ull,
    0x080819192b080808ull, 0x080819192b08082bull, 0x080819192b081919ull, 0x080819192b082b08ull,
    0x080819192b190819ull, 0x080819192b191908ull, 0x080819192b2b0808ull, 0x0808192b08080819ull,
    0x0808192b08081908ull, 0x0808192b0808192bull, 0x0808192b08082b19ull, 0x0808192b08190808ull,
    0x0808192b08191919ull, 0x0808192b19080808ull, 0x0808192b19081919ull, 0x0808192b19082b08ull,
    0x0808192b19190819ull, 0x0808192b19191908ull, 0x0808192b192b0808ull, 0x0808192b2b080819ull,
    0x0808192b2b081908ull, 0x0808192b2b190808ull, 0x08082b0808080808ull, 0x08082b080808082bull,
    0x08082b0808081919ull, 0x08082b0808082b08ull, 0x08082b0808190819ull, 0x08082b0808191908ull,
    0x08082b080819192bull, 0x08082b0808192b19ull, 0x08082b08082b0808ull, 0x08082b08082b1919ull,
    0x08082b08082b2b2bull, 0x08082b0819080819ull, 0x08082b0819081908ull, 0x08082b081908192bull,
    0x08082b0819082b19ull, 0x08082b0819190808ull, 0x08082b081919082bull, 0x08082b0819191919ull,
    0x08082b0819192b08ull, 0x08082b08192b0819ull, 0x08082b08192b1908ull, 0x08082b082b080808ull,
    0x08082b082b081919ull, 0x08082b082b191908ull, 0x08082b082b2b2b2bull, 0x08082b1908080819ull,
    0x08082b1908081908ull, 0x08082b1908190808ull, 0x08082b190819082bull, 0x08082b1908191919ull,
    0x08082b1908192b08ull, 0x08082b19082b0819ull, 0x08082b1919080808ull, 0x08082b1919081919ull,
    0x08082b1919082b08ull, 0x08082b1919190819ull, 0x08082b1919191908ull, 0x08082b19192b0808ull,
    0x08082b192b080819ull, 0x08082b192b190808ull, 0x08082b2b08080808ull, 0x08082b2b08190819ull,
    0x08082b2b08191908ull, 0x08082b2b082b082bull, 0x08082b2b082b2b08ull, 0x08082b2b082b2b2bull,
    0x08082b2b19190808ull, 0x08082b2b2b192b19ull, 0x0819080808080819ull, 0x0819080808081908ull,
    0x081908080808192bull, 0x0819080808082b19ull, 0x0819080808190808ull, 0x081908080819082bull,
    0x0819080808191919ull, 0x0819080808192b08ull, 0x08190808082b0819ull, 0x08190808082b1908ull,
    0x08190808082b192bull, 0x0819080819080808ull, 0x081908081908082bull, 0x0819080819081919ull,
    0x0819080819082b08ull, 0x0819080819190819ull, 0x0819080819191908ull, 0x081908081919192bull,
    0x0819080819192b19ull, 0x08190808192b0808ull, 0x08190808192b082bull, 0x08190808192b1919ull,
    0x08190808192b2b08ull, 0x081908082b080819ull, 0x081908082b081908ull, 0x081908082b08192bull,
    0x081908082b190808ull, 0x081908082b191919ull, 0x081908082b192b08ull, 0x081908082b2b0819ull,
    0x081908082b2b1908ull, 0x0819081908080808ull, 0x081908190808082bull, 0x0819081908081919ull,
    0x0819081908082b08ull, 0x0819081908082b2bull, 0x0819081908190819ull, 0x0819081908191908ull,
    0x081908190819192bull, 0x0819081908192b19ull, 0x08190819082b0808ull, 0x08190819082b082bull,
    0x08190819082b1919ull, 0x08190819082b2b08ull, 0x0819081919080819ull, 0x0819081919081908ull,
    0x081908191908192bull, 0x0819081919082b19ull, 0x0819081919190808ull, 0x081908191919082bull,
    0x0819081919191919ull, 0x0819081919192b08ull, 0x08190819192b0819ull, 0x08190819192b1908ull,
    0x081908192b080808ull, 0x081908192b08082bull, 0x081908192b081919ull, 0x081908192b082b08ull,
    0x081908192b190819ull, 0x081908192b191908ull, 0x0819082b08080819ull, 0x0819082b08081908ull,
    0x0819082b08082b19ull, 0x0819082b08190808ull, 0x0819082b08191919ull, 0x0819082b082b0819ull,
    0x0819082b082b1908ull, 0x0819082b19080808ull, 0x0819082b19081919ull, 0x0819082b19190819ull,
    0x0819082b19191908ull, 0x0819082b2b080819ull, 0x0819082b2b081908ull, 0x0819082b2b190808ull,
    0x0819190808080808ull, 0x081919080808082bull, 0x0819190808081919ull, 0x0819190808082b08ull,
    0x0819190808190819ull, 0x0819190808191908ull, 0x081919080819192bull, 0x0819190808192b19ull,
    0x08191908082b0808ull, 0x08191908082b1919ull, 0x08191908082b2b08ull, 0x0819190819080819ull,
    0x0819190819081908ull, 0x081919081908192bull, 0x0819190819082b19ull, 0x0819190819190808ull,
    0x081919081919082bull, 0x0819190819191919ull, 0x0819190819192b08ull, 0x08191908192b0819ull,
    0x08191908192b1908ull, 0x081919082b080808ull, 0x081919082b08082bull, 0x081919082b081919ull,
    0x081919082b082b08ull, 0x081919082b190819ull, 0x081919082b191908ull, 0x081919082b2b0808ull,
    0x0819191908080819ull, 0x0819191908081908ull, 0x081919190808192bull, 0x0819191908082b19ull,
    0x0819191908190808ull, 0x081919190819082bull, 0x0819191908191919ull, 0x0819191908192b08ull,
    0x08191919082b0819ull, 0x08191919082b1908ull, 0x0819191919080808ull, 0x081919191908082bull,
    0x0819191919081919ull, 0x0819191919082b08ull, 0x0819191919190819ull, 0x0819191919191908ull,
    0x08191919192b0808ull, 0x081919192b080819ull, 0x081919192b081908ull, 0x081919192b190808ull,
    0x0819192b08080808ull, 0x0819192b08081919ull, 0x0819192b08082b08ull, 0x0819192b08190819ull,
    0x0819192b08191908ull, 0x0819192b082b0808ull, 0x0819192b19080819ull, 0x0819192b19081908ull,
    0x0819192b19190808ull, 0x0819192b2b080808ull, 0x0819192b2b2b2b2bull, 0x08192b0808080819ull,
    0x08192b0808081908ull, 0x08192b080808192bull, 0x08192b0808082b19ull, 0x08192b0808190808ull,
    0x08192b0808191919ull, 0x08192b0808192b08ull, 0x08192b08082b0819ull, 0x08192b0819080808ull,
    0x08192b081908082bull, 0x08192b0819081919ull, 0x08192b0819082b08ull, 0x08192b0819190819ull,
    0x08192b0819191908ull, 0x08192b08192b0808ull, 0x08192b082b080819ull, 0x08192b082b081908ull,
    0x08192b1908080808ull, 0x08192b190808082bull, 0x08192b1908081919ull, 0x08192b1908082b08ull,
    0x08192b1908190819ull, 0x08192b1908191908ull, 0x08192b19082b0808ull, 0x08192b1919080819ull,
    0x08192b1919081908ull, 0x08192b1919190808ull, 0x08192b19192b2b19ull, 0x08192b192b2b082bull,
    0x08192b2b08081908ull, 0x08192b2b08190808ull, 0x08192b2b19080808ull, 0x08192b2b1919192bull,
    0x082b080808080808ull, 0x082b08080808082bull, 0x082b080808081919ull, 0x082b080808082b08ull,
    0x082b080808190819ull, 0x082b080808191908ull, 0x082b08080819192bull, 0x082b080808192b19ull,
    0x082b0808082b0808ull, 0x082b0808082b1919ull, 0x082b0808082b2b2bull, 0x082b080819080819ull,
    0x082b080819081908ull, 0x082b080819190808ull, 0x082b08081919082bull, 0x082b080819191919ull,
    0x082b0808192b1908ull, 0x082b08082b080808ull, 0x082b08082b082b2bull, 0x082b08082b191908ull,
    0x082b08082b2b2b2bull, 0x082b081908080819ull, 0x082b081908081908ull, 0x082b081908190808ull,
    0x082b08190819082bull, 0x082b081908191919ull, 0x082b0819082b0819ull, 0x082b081919080808ull,
    0x082b08191908082bull, 0x082b081919081919ull, 0x082b081919190819ull, 0x082b081919191908ull,
    0x082b0819192b0808ull, 0x082b08192b080819ull, 0x082b08192b081908ull, 0x082b08192b190808ull,
    0x082b082b08080808ull, 0x082b082b08082b2bull, 0x082b082b082b082bull, 0x082b082b082b2b08ull,
    0x082b082b082b2b2bull, 0x082b082b19081908ull, 0x082b082b19190808ull, 0x082b082b2b082b08ull,
    0x082b082b2b082b2bull, 0x082b082b2b2b2b08ull, 0x082b190808080819ull, 0x082b190808081908ull,
    0x082b19080808192bull, 0x082b190808082b19ull, 0x082b190808190808ull, 0x082b190808191919ull,
    0x082b190808192b08ull, 0x082b1908082b0819ull, 0x082b1908082b1908ull, 0x082b190819080808ull,
    0x082b19081908082bull, 0x082b190819081919ull, 0x082b190819082b08ull, 0x082b190819190819ull,
    0x082b190819191908ull, 0x082b1908192b0808ull, 0x082b19082b080819ull, 0x082b19082b081908ull,
    0x082b19082b190808ull, 0x082b191908080808ull, 0x082b191908081919ull, 0x082b191908082b08ull,
    0x082b191908190819ull, 0x082b191908191908ull, 0x082b1919082b0808ull, 0x082b191919080819ull,
    0x082b191919081908ull, 0x082b191919190808ull, 0x082b1919192b192bull, 0x082b19192b080808ull,
    0x082b192b08080819ull, 0x082b192b08081908ull, 0x082b192b08190808ull, 0x082b192b19080808ull,
    0x082b192b19192b19ull, 0x082b2b0808080808ull, 0x082b2b0808081919ull, 0x082b2b0808190819ull,
    0x082b2b0808191908ull, 0x082b2b0819080819ull, 0x082b2b0819081908ull, 0x082b2b0819190808ull,
    0x082b2b082b082b2bull, 0x082b2b082b2b2b2bull, 0x082b2b1908080819ull, 0x082b2b1908081908ull,
    0x082b2b1908190808ull, 0x082b2b192b191919ull, 0x082b2b2b08082b2bull, 0x082b2b2b082b082bull,
    0x082b2b2b192b1908ull, 0x082b2b2b2b082b08ull, 0x082b2b2b2b082b2bull, 0x1908080808080819ull,
    0x1908080808081908ull, 0x190808080808192bull, 0x1908080808082b19ull, 0x1908080808190808ull,
    0x190808080819082bull, 0x1908080808191919ull, 0x1908080808192b08ull, 0x1908080808192b2bull,
    0x19080808082b0819ull, 0x19080808082b1908ull, 0x19080808082b192bull, 0x1908080819080808ull,
    0x190808081908082bull, 0x1908080819081919ull, 0x1908080819082b08ull, 0x1908080819082b2bull,
    0x1908080819190819ull, 0x1908080819191908ull, 0x190808081919192bull, 0x1908080819192b19ull,
    0x19080808192b0808ull, 0x19080808192b082bull, 0x19080808192b1919ull, 0x190808082b080819ull,
    0x190808082b081908ull, 0x190808082b190808ull, 0x190808082b191919ull, 0x190808082b192b08ull,
    0x190808082b2b0819ull, 0x190808082b2b1908ull, 0x1908081908080808ull, 0x190808190808082bull,
    0x1908081908081919ull, 0x1908081908082b08ull, 0x1908081908190819ull, 0x1908081908191908ull,
    0x190808190819192bull, 0x1908081908192b19ull, 0x19080819082b0808ull, 0x19080819082b082bull,
    0x19080819082b1919ull, 0x1908081919080819ull, 0x1908081919081908ull, 0x190808191908192bull,
    0x1908081919082b19ull, 0x1908081919190808ull, 0x190808191919082bull, 0x1908081919191919ull,
    0x1908081919192b08ull, 0x19080819192b0819ull, 0x19080819192b1908ull, 0x190808192b080808ull,
    0x190808192b08082bull, 0x190808192b081919ull, 0x190808192b082b08ull, 0x190808192b190819ull,
    0x190808192b191908ull, 0x190808192b2b0808ull, 0x1908082b08080819ull, 0x1908082b08081908ull,
    0x1908082b08190808ull, 0x1908082b0819082bull, 0x1908082b08191919ull, 0x1908082b08192b08ull,
    0x1908082b082b1908ull, 0x1908082b19080808ull, 0x1908082b19081919ull, 0x1908082b19082b08ull,
    0x1908082b19190819ull, 0x1908082b19191908ull, 0x1908082b192b0808ull, 0x1908082b2b080819ull,
    0x1908082b2b081908ull, 0x1908190808080808ull, 0x190819080808082bull, 0x1908190808081919ull,
    0x1908190808082b08ull, 0x1908190808082b2bull, 0x1908190808190819ull, 0x1908190808191908ull,
    0x190819080819192bull, 0x1908190808192b19ull, 0x19081908082b0808ull, 0x19081908082b082bull,
    0x19081908082b1919ull, 0x19081908082b2b08ull, 0x1908190819080819ull, 0x1908190819081908ull,
    0x190819081908192bull, 0x1908190819082b19ull, 0x1908190819190808ull, 0x190819081919082bull,
    0x1908190819191919ull, 0x1908190819192b08ull, 0x19081908192b0819ull, 0x19081908192b1908ull,
    0x190819082b080808ull, 0x190819082b08082bull, 0x190819082b081919ull, 0x190819082b082b08ull,
    0x190819082b190819ull, 0x190819082b191908ull, 0x190819082b2b0808ull, 0x1908191908080819ull,
    0x1908191908081908ull, 0x190819190808192bull, 0x1908191908082b19ull, 0x1908191908190808ull,
    0x190819190819082bull, 0x1908191908191919ull, 0x1908191908192b08ull, 0x19081919082b0819ull,
    0x19081919082b1908ull, 0x1908191919080808ull, 0x190819191908082bull, 0x1908191919081919ull,
    0x1908191919082b08ull, 0x1908191919190819ull, 0x1908191919191908ull, 0x19081919192b0808ull,
    0x19081919192b2b2bull, 0x190819192b080819ull, 0x190819192b081908ull, 0x190819192b190808ull,
    0x1908192b08080808ull, 0x1908192b0808082bull, 0x1908192b08081919ull, 0x1908192b08082b08ull,
    0x1908192b08190819ull, 0x1908192b08191908ull, 0x1908192b082b0808ull, 0x1908192b19080819ull,
    0x1908192b19081908ull, 0x1908192b19190808ull, 0x1908192b2b080808ull, 0x1908192b2b2b1919ull,
    0x19082b0808080819ull, 0x19082b0808081908ull, 0x19082b0808082b19ull, 0x19082b0808190808ull,
    0x19082b080819082bull, 0x19082b0808191919ull, 0x19082b0808192b08ull, 0x19082b08082b0819ull,
    0x19082b08082b1908ull, 0x19082b0819080808ull, 0x19082b081908082bull, 0x19082b0819081919ull,
    0x19082b0819082b08ull, 0x19082b0819190819ull, 0x19082b0819191908ull, 0x19082b08192b0808ull,
    0x19082b082b081908ull, 0x19082b082b190808ull, 0x19082b1908080808ull, 0x19082b190808082bull,
    0x19082b1908081919ull, 0x19082b1908082b08ull, 0x19082b1908190819ull, 0x19082b1908191908ull,
    0x19082b19082b0808ull, 0x19082b1919080819ull, 0x19082b1919081908ull, 0x19082b1919190808ull,
    0x19082b192b080808ull, 0x19082b192b19192bull, 0x19082b2b08080819ull, 0x19082b2b08081908ull,
    0x19082b2b08190808ull, 0x19082b2b19080808ull, 0x1919080808080808ull, 0x191908080808082bull,
    0x1919080808081919ull, 0x1919080808082b08ull, 0x1919080808190819ull, 0x1919080808191908ull,
    0x191908080819192bull, 0x1919080808192b19ull, 0x19190808082b0808ull, 0x19190808082b082bull,
    0x19190808082b1919ull, 0x19190808082b2b08ull, 0x1919080819080819ull, 0x1919080819081908ull,
    0x191908081908192bull, 0x1919080819082b19ull, 0x1919080819190808ull, 0x191908081919082bull,
    0x1919080819191919ull, 0x1919080819192b08ull, 0x19190808192b0819ull, 0x19190808192b1908ull,
    0x191908082b080808ull, 0x191908082b08082bull, 0x191908082b081919ull, 0x191908082b082b08ull,
    0x191908082b190819ull, 0x191908082b191908ull, 0x1919081908080819ull, 0x1919081908081908ull,
    0x191908190808192bull, 0x1919081908082b19ull, 0x1919081908190808ull, 0x191908190819082bull,
    0x1919081908191919ull, 0x1919081908192b08ull, 0x19190819082b0819ull, 0x19190819082b1908ull,
    0x1919081919080808ull, 0x191908191908082bull, 0x1919081919081919ull, 0x1919081919082b08ull,
    0x1919081919190819ull, 0x1919081919191908ull, 0x19190819192b0808ull, 0x191908192b080819ull,
    0x191908192b081908ull, 0x191908192b190808ull, 0x1919082b08080808ull, 0x1919082b08081919ull,
    0x1919082b08082b08ull, 0x1919082b08190819ull, 0x1919082b08191908ull, 0x1919082b082b0808ull,
    0x1919082b19080819ull, 0x1919082b19081908ull, 0x1919082b19190808ull, 0x1919082b192b2b19ull,
    0x1919082b2b080808ull, 0x1919190808080819ull, 0x1919190808081908ull, 0x191919080808192bull,
    0x1919190808082b19ull, 0x1919190808190808ull, 0x191919080819082bull, 0x1919190808191919ull,
    0x1919190808192b08ull, 0x19191908082b0819ull, 0x19191908082b1908ull, 0x1919190819080808ull,
    0x191919081908082bull, 0x1919190819081919ull, 0x1919190819082b08ull, 0x1919190819190819ull,
    0x1919190819191908ull, 0x19191908192b0808ull, 0x191919082b080819ull, 0x191919082b081908ull,
    0x191919082b190808ull, 0x1919191908080808ull, 0x191919190808082bull, 0x1919191908081919ull,
    0x1919191908082b08ull, 0x1919191908190819ull, 0x1919191908191908ull, 0x19191919082b0808ull,
    0x1919191919080819ull, 0x1919191919081908ull, 0x1919191919190808ull, 0x191919192b080808ull,
    0x1919192b08080819ull, 0x1919192b08081908ull, 0x1919192b08190808ull, 0x1919192b082b192bull,
    0x1919192b19080808ull, 0x19192b0808080808ull, 0x19192b080808082bull, 0x19192b0808081919ull,
    0x19192b0808082b08ull, 0x19192b0808190819ull, 0x19192b0808191908ull, 0x19192b08082b0808ull,
    0x19192b0819080819ull, 0x19192b0819081908ull, 0x19192b0819190808ull, 0x19192b0819192b2bull,
    0x19192b082b080808ull, 0x19192b1908080819ull, 0x19192b1908081908ull, 0x19192b1908190808ull,
    0x19192b1919080808ull, 0x19192b2b08080808ull, 0x19192b2b08192b19ull, 0x19192b2b2b081919ull,
    0x19192b2b2b2b2b08ull, 0x192b080808080819ull, 0x192b080808081908ull, 0x192b08080808192bull,
    0x192b080808190808ull, 0x192b08080819082bull, 0x192b080808191919ull, 0x192b080808192b08ull,
    0x192b0808082b0819ull, 0x192b0808082b1908ull, 0x192b080819080808ull, 0x192b080819081919ull,
    0x192b080819082b08ull, 0x192b080819190819ull, 0x192b080819191908ull, 0x192b0808192b0808ull,
    0x192b08082b081908ull, 0x192b08082b190808ull, 0x192b081908080808ull, 0x192b08190808082bull,
    0x192b081908081919ull, 0x192b081908082b08ull, 0x192b081908190819ull, 0x192b081908191908ull,
    0x192b0819082b0808ull, 0x192b081919080819ull, 0x192b081919081908ull, 0x192b081919190808ull,
    0x192b08192b080808ull, 0x192b08192b192b19ull, 0x192b082b08081908ull, 0x192b082b08190808ull,
    0x192b082b19080808ull, 0x192b082b1919192bull, 0x192b082b2b2b0819ull, 0x192b190808080808ull,
    0x192b190808081919ull, 0x192b190808082b08ull, 0x192b190808190819ull, 0x192b190808191908ull,
    0x192b1908082b0808ull, 0x192b190819080819ull, 0x192b190819081908ull, 0x192b190819190808ull,
    0x192b19082b080808ull, 0x192b191908080819ull, 0x192b191908081908ull, 0x192b191908190808ull,
    0x192b191919080808ull, 0x192b191919082b2bull, 0x192b1919192b2b08ull, 0x192b19192b19082bull,
    0x192b192b08080808ull, 0x192b192b2b191908ull, 0x192b2b0808080819ull, 0x192b2b0808081908ull,
    0x192b2b0808190808ull, 0x192b2b08192b1919ull, 0x192b2b082b192b08ull, 0x192b2b1908080808ull,
    0x192b2b19082b2b2bull, 0x192b2b2b1908082bull, 0x192b2b2b2b2b0819ull, 0x2b08080808080808ull,
    0x2b0808080808082bull, 0x2b08080808081919ull, 0x2b08080808082b08ull, 0x2b08080808190819ull,
    0x2b08080808191908ull, 0x2b08080808192b19ull, 0x2b080808082b0808ull, 0x2b080808082b1919ull,
    0x2b08080819080819ull, 0x2b08080819081908ull, 0x2b08080819190808ull, 0x2b0808081919082bull,
    0x2b08080819191919ull, 0x2b08080819192b08ull, 0x2b080808192b0819ull, 0x2b0808082b080808ull,
    0x2b0808082b081919ull, 0x2b0808082b190819ull, 0x2b0808082b191908ull, 0x2b08081908080819ull,
    0x2b08081908081908ull, 0x2b08081908082b19ull, 0x2b08081908190808ull, 0x2b0808190819082bull,
    0x2b08081908191919ull, 0x2b08081908192b08ull, 0x2b080819082b0819ull, 0x2b080819082b1908ull,
    0x2b08081919080808ull, 0x2b0808191908082bull, 0x2b08081919081919ull, 0x2b08081919082b08ull,
    0x2b08081919190819ull, 0x2b08081919191908ull, 0x2b0808192b080819ull, 0x2b0808192b081908ull,
    0x2b0808192b190808ull, 0x2b0808192b2b2b19ull, 0x2b08082b08080808ull, 0x2b08082b08081919ull,
    0x2b08082b08082b2bull, 0x2b08082b08190819ull, 0x2b08082b08191908ull, 0x2b08082b19080819ull,
    0x2b08082b19081908ull, 0x2b08082b19190808ull, 0x2b08190808080819ull, 0x2b08190808081908ull,
    0x2b0819080808192bull, 0x2b08190808082b19ull, 0x2b08190808190808ull, 0x2b0819080819082bull,
    0x2b08190808191919ull, 0x2b08190808192b08ull, 0x2b081908082b0819ull, 0x2b08190819080808ull,
    0x2b0819081908082bull, 0x2b08190819081919ull, 0x2b08190819082b08ull, 0x2b08190819190819ull,
    0x2b08190819191908ull, 0x2b081908192b0808ull, 0x2b0819082b080819ull, 0x2b0819082b081908ull,
    0x2b0819082b190808ull, 0x2b08191908080808ull, 0x2b0819190808082bull, 0x2b08191908081919ull,
    0x2b08191908082b08ull, 0x2b08191908190819ull, 0x2b08191908191908ull, 0x2b081919082b0808ull,
    0x2b08191919080819ull, 0x2b08191919081908ull, 0x2b08191919190808ull, 0x2b0819192b080808ull,
    0x2b0819192b082b2bull, 0x2b08192b08080819ull, 0x2b08192b08081908ull, 0x2b08192b08190808ull,
    0x2b08192b082b2b19ull, 0x2b08192b19080808ull, 0x2b082b0808080808ull, 0x2b082b0808081919ull,
    0x2b082b0808190819ull, 0x2b082b0808191908ull, 0x2b082b0819080819ull, 0x2b082b0819081908ull,
    0x2b082b0819190808ull, 0x2b082b082b2b082bull, 0x2b082b1908080819ull, 0x2b082b1908081908ull,
    0x2b082b1919080808ull, 0x2b082b19192b1919ull, 0x2b082b2b082b082bull, 0x2b082b2b19192b08ull,
    0x2b082b2b19192b2bull, 0x2b082b2b2b08082bull, 0x2b082b2b2b2b082bull, 0x2b19080808080819ull,
    0x2b19080808081908ull, 0x2b19080808082b19ull, 0x2b19080808190808ull, 0x2b1908080819082bull,
    0x2b19080808191919ull, 0x2b19080808192b08ull, 0x2b190808082b1908ull, 0x2b19080819080808ull,
    0x2b1908081908082bull, 0x2b19080819081919ull, 0x2b19080819082b08ull, 0x2b19080819190819ull,
    0x2b19080819191908ull, 0x2b190808192b0808ull, 0x2b1908082b080819ull, 0x2b1908082b081908ull,
    0x2b1908082b190808ull, 0x2b19081908080808ull, 0x2b19081908081919ull, 0x2b19081908190819ull,
    0x2b19081908191908ull, 0x2b19081919080819ull, 0x2b19081919081908ull, 0x2b19081919190808ull,
    0x2b19081919192b2bull, 0x2b19082b08080819ull, 0x2b19082b08081908ull, 0x2b19082b08190808ull,
    0x2b19082b19080808ull, 0x2b19082b2b2b192bull, 0x2b19190808080808ull, 0x2b1919080808082bull,
    0x2b19190808081919ull, 0x2b19190808082b08ull, 0x2b19190808190819ull, 0x2b19190808191908ull,
    0x2b191908082b0808ull, 0x2b19190819080819ull, 0x2b19190819081908ull, 0x2b19190819190808ull,
    0x2b1919082b080808ull, 0x2b1919082b19192bull, 0x2b19191908080819ull, 0x2b19191908081908ull,
    0x2b19191908190808ull, 0x2b19191919080808ull, 0x2b1919192b192b08ull, 0x2b1919192b2b0819ull,
    0x2b19192b08080808ull, 0x2b19192b1908192bull, 0x2b19192b192b1908ull, 0x2b192b0808080819ull,
    0x2b192b0808081908ull, 0x2b192b0808190808ull, 0x2b192b08082b192bull, 0x2b192b0819080808ull,
    0x2b192b082b2b2b19ull, 0x2b192b1908080808ull, 0x2b192b1919082b19ull, 0x2b192b191919082bull,
    0x2b192b2b2b190808ull, 0x2b2b080808080808ull, 0x2b2b080808081919ull, 0x2b2b080808082b2bull,
    0x2b2b080808191908ull, 0x2b2b0808082b082bull, 0x2b2b0808082b2b2bull, 0x2b2b080819080819ull,
    0x2b2b080819081908ull, 0x2b2b080819190808ull, 0x2b2b08082b2b082bull, 0x2b2b08082b2b2b2bull,
    0x2b2b081919080808ull, 0x2b2b0819192b1919ull, 0x2b2b082b0808082bull, 0x2b2b082b08082b2bull,
    0x2b2b082b082b082bull, 0x2b2b082b082b2b08ull, 0x2b2b082b082b2b2bull, 0x2b2b082b2b08082bull,
    0x2b2b082b2b082b08ull, 0x2b2b082b2b082b2bull, 0x2b2b082b2b2b2b08ull, 0x2b2b190808080819ull,
    0x2b2b190808081908ull, 0x2b2b190808190808ull, 0x2b2b190819080808ull, 0x2b2b19082b082b19ull,
    0x2b2b19082b2b1908ull, 0x2b2b191908080808ull, 0x2b2b191908192b19ull, 0x2b2b192b19190819ull,
    0x2b2b2b0808082b2bull, 0x2b2b2b08082b2b08ull, 0x2b2b2b082b2b082bull, 0x2b2b2b1919191908ull,
    0x2b2b2b192b08192bull, 0x2b2b2b2b08082b08ull, 0x2b2b2b2b08082b2bull, 0x2b2b2b2b082b0808ull,
    0x2b2b2b2b082b082bull, 0x2b2b2b2b082b2b08ull, 0x2b2b2b2b2b082b08ull, 0x2b2b2b2b2b2b2b2bull,
};

/* Weight grid for the 3-bit extra-extra-small format: four packed weights an index.
 * Transcribed from `IQ3XXS_GRID` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) uint32_t SHISO_IQ3XXS_GRID[256] = {
    0x04040404u, 0x04040414u, 0x04040424u, 0x04040c0cu,
    0x04040c1cu, 0x04040c3eu, 0x04041404u, 0x04041414u,
    0x04041c0cu, 0x04042414u, 0x04043e1cu, 0x04043e2cu,
    0x040c040cu, 0x040c041cu, 0x040c0c04u, 0x040c0c14u,
    0x040c140cu, 0x040c142cu, 0x040c1c04u, 0x040c1c14u,
    0x040c240cu, 0x040c2c24u, 0x040c3e04u, 0x04140404u,
    0x04140414u, 0x04140424u, 0x04140c0cu, 0x04141404u,
    0x04141414u, 0x04141c0cu, 0x04141c1cu, 0x04141c3eu,
    0x04142c0cu, 0x04142c3eu, 0x04143e2cu, 0x041c040cu,
    0x041c043eu, 0x041c0c04u, 0x041c0c14u, 0x041c142cu,
    0x041c3e04u, 0x04240c1cu, 0x04241c3eu, 0x04242424u,
    0x04242c3eu, 0x04243e1cu, 0x04243e2cu, 0x042c040cu,
    0x042c043eu, 0x042c1c14u, 0x042c2c14u, 0x04341c2cu,
    0x04343424u, 0x043e0c04u, 0x043e0c24u, 0x043e0c34u,
    0x043e241cu, 0x043e340cu, 0x0c04040cu, 0x0c04041cu,
    0x0c040c04u, 0x0c040c14u, 0x0c04140cu, 0x0c04141cu,
    0x0c041c04u, 0x0c041c14u, 0x0c041c24u, 0x0c04243eu,
    0x0c042c04u, 0x0c0c0404u, 0x0c0c0414u, 0x0c0c0c0cu,
    0x0c0c1404u, 0x0c0c1414u, 0x0c14040cu, 0x0c14041cu,
    0x0c140c04u, 0x0c140c14u, 0x0c14140cu, 0x0c141c04u,
    0x0c143e14u, 0x0c1c0404u, 0x0c1c0414u, 0x0c1c1404u,
    0x0c1c1c0cu, 0x0c1c2434u, 0x0c1c3434u, 0x0c24040cu,
    0x0c24042cu, 0x0c242c04u, 0x0c2c1404u, 0x0c2c1424u,
    0x0c2c2434u, 0x0c2c3e0cu, 0x0c34042cu, 0x0c3e1414u,
    0x0c3e2404u, 0x14040404u, 0x14040414u, 0x14040c0cu,
    0x14040c1cu, 0x14041404u, 0x14041414u, 0x14041434u,
    0x14041c0cu, 0x14042414u, 0x140c040cu, 0x140c041cu,
    0x140c042cu, 0x140c0c04u, 0x140c0c14u, 0x140c140cu,
    0x140c1c04u, 0x140c341cu, 0x140c343eu, 0x140c3e04u,
    0x14140404u, 0x14140414u, 0x14140c0cu, 0x14140c3eu,
    0x14141404u, 0x14141414u, 0x14141c3eu, 0x14142404u,
    0x14142c2cu, 0x141c040cu, 0x141c0c04u, 0x141c0c24u,
    0x141c3e04u, 0x141c3e24u, 0x14241c2cu, 0x14242c1cu,
    0x142c041cu, 0x142c143eu, 0x142c240cu, 0x142c3e24u,
    0x143e040cu, 0x143e041cu, 0x143e0c34u, 0x143e242cu,
    0x1c04040cu, 0x1c040c04u, 0x1c040c14u, 0x1c04140cu,
    0x1c04141cu, 0x1c042c04u, 0x1c04342cu, 0x1c043e14u,
    0x1c0c0404u, 0x1c0c0414u, 0x1c0c1404u, 0x1c0c1c0cu,
    0x1c0c2424u, 0x1c0c2434u, 0x1c14040cu, 0x1c14041cu,
    0x1c140c04u, 0x1c14142cu, 0x1c142c14u, 0x1c143e14u,
    0x1c1c0c0cu, 0x1c1c1c1cu, 0x1c241c04u, 0x1c24243eu,
    0x1c243e14u, 0x1c2c0404u, 0x1c2c0434u, 0x1c2c1414u,
    0x1c2c2c2cu, 0x1c340c24u, 0x1c341c34u, 0x1c34341cu,
    0x1c3e1c1cu, 0x1c3e3404u, 0x24040424u, 0x24040c3eu,
    0x24041c2cu, 0x24041c3eu, 0x24042c1cu, 0x24042c3eu,
    0x240c3e24u, 0x24141404u, 0x24141c3eu, 0x24142404u,
    0x24143404u, 0x24143434u, 0x241c043eu, 0x241c242cu,
    0x24240424u, 0x24242c0cu, 0x24243424u, 0x242c142cu,
    0x242c241cu, 0x242c3e04u, 0x243e042cu, 0x243e0c04u,
    0x243e0c14u, 0x243e1c04u, 0x2c040c14u, 0x2c04240cu,
    0x2c043e04u, 0x2c0c0404u, 0x2c0c0434u, 0x2c0c1434u,
    0x2c0c2c2cu, 0x2c140c24u, 0x2c141c14u, 0x2c143e14u,
    0x2c1c0414u, 0x2c1c2c1cu, 0x2c240c04u, 0x2c24141cu,
    0x2c24143eu, 0x2c243e14u, 0x2c2c0414u, 0x2c2c1c0cu,
    0x2c342c04u, 0x2c3e1424u, 0x2c3e2414u, 0x34041424u,
    0x34042424u, 0x34042434u, 0x34043424u, 0x340c140cu,
    0x340c340cu, 0x34140c3eu, 0x34143424u, 0x341c1c04u,
    0x341c1c34u, 0x34242424u, 0x342c042cu, 0x342c2c14u,
    0x34341c1cu, 0x343e041cu, 0x343e140cu, 0x3e04041cu,
    0x3e04042cu, 0x3e04043eu, 0x3e040c04u, 0x3e041c14u,
    0x3e042c14u, 0x3e0c1434u, 0x3e0c2404u, 0x3e140c14u,
    0x3e14242cu, 0x3e142c14u, 0x3e1c0404u, 0x3e1c0c2cu,
    0x3e1c1c1cu, 0x3e1c3404u, 0x3e24140cu, 0x3e24240cu,
    0x3e2c0404u, 0x3e2c0414u, 0x3e2c1424u, 0x3e341c04u,
};

/* Weight grid for the 3-bit small format: four packed weights an index.
 * Transcribed from `IQ3S_GRID` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) uint32_t SHISO_IQ3S_GRID[512] = {
    0x01010101u, 0x01010103u, 0x01010105u, 0x0101010bu,
    0x0101010fu, 0x01010301u, 0x01010303u, 0x01010305u,
    0x01010309u, 0x0101030du, 0x01010501u, 0x01010503u,
    0x0101050bu, 0x01010707u, 0x01010901u, 0x01010905u,
    0x0101090bu, 0x0101090fu, 0x01010b03u, 0x01010b07u,
    0x01010d01u, 0x01010d05u, 0x01010f03u, 0x01010f09u,
    0x01010f0fu, 0x01030101u, 0x01030103u, 0x01030105u,
    0x01030109u, 0x01030301u, 0x01030303u, 0x0103030bu,
    0x01030501u, 0x01030507u, 0x0103050fu, 0x01030703u,
    0x0103070bu, 0x01030909u, 0x01030d03u, 0x01030d0bu,
    0x01030f05u, 0x01050101u, 0x01050103u, 0x0105010bu,
    0x0105010fu, 0x01050301u, 0x01050307u, 0x0105030du,
    0x01050503u, 0x0105050bu, 0x01050701u, 0x01050709u,
    0x01050905u, 0x0105090bu, 0x0105090fu, 0x01050b03u,
    0x01050b07u, 0x01050f01u, 0x01050f07u, 0x01070107u,
    0x01070303u, 0x0107030bu, 0x01070501u, 0x01070505u,
    0x01070703u, 0x01070707u, 0x0107070du, 0x01070909u,
    0x01070b01u, 0x01070b05u, 0x01070d0fu, 0x01070f03u,
    0x01070f0bu, 0x01090101u, 0x01090307u, 0x0109030fu,
    0x01090503u, 0x01090509u, 0x01090705u, 0x01090901u,
    0x01090907u, 0x01090b03u, 0x01090f01u, 0x010b0105u,
    0x010b0109u, 0x010b0501u, 0x010b0505u, 0x010b050du,
    0x010b0707u, 0x010b0903u, 0x010b090bu, 0x010b090fu,
    0x010b0d0du, 0x010b0f07u, 0x010d010du, 0x010d0303u,
    0x010d0307u, 0x010d0703u, 0x010d0b05u, 0x010d0f03u,
    0x010f0101u, 0x010f0105u, 0x010f0109u, 0x010f0501u,
    0x010f0505u, 0x010f050du, 0x010f0707u, 0x010f0b01u,
    0x010f0b09u, 0x03010101u, 0x03010103u, 0x03010105u,
    0x03010109u, 0x03010301u, 0x03010303u, 0x03010307u,
    0x0301030bu, 0x0301030fu, 0x03010501u, 0x03010505u,
    0x03010703u, 0x03010709u, 0x0301070du, 0x03010b09u,
    0x03010b0du, 0x03010d03u, 0x03010f05u, 0x03030101u,
    0x03030103u, 0x03030107u, 0x0303010du, 0x03030301u,
    0x03030309u, 0x03030503u, 0x03030701u, 0x03030707u,
    0x03030903u, 0x03030b01u, 0x03030b05u, 0x03030f01u,
    0x03030f0du, 0x03050101u, 0x03050305u, 0x0305030bu,
    0x0305030fu, 0x03050501u, 0x03050509u, 0x03050705u,
    0x03050901u, 0x03050907u, 0x03050b0bu, 0x03050d01u,
    0x03050f05u, 0x03070103u, 0x03070109u, 0x0307010fu,
    0x03070301u, 0x03070307u, 0x03070503u, 0x0307050fu,
    0x03070701u, 0x03070709u, 0x03070903u, 0x03070d05u,
    0x03070f01u, 0x03090107u, 0x0309010bu, 0x03090305u,
    0x03090309u, 0x03090703u, 0x03090707u, 0x03090905u,
    0x0309090du, 0x03090b01u, 0x03090b09u, 0x030b0103u,
    0x030b0301u, 0x030b0307u, 0x030b0503u, 0x030b0701u,
    0x030b0705u, 0x030b0b03u, 0x030d0501u, 0x030d0509u,
    0x030d050fu, 0x030d0909u, 0x030d090du, 0x030f0103u,
    0x030f0107u, 0x030f0301u, 0x030f0305u, 0x030f0503u,
    0x030f070bu, 0x030f0903u, 0x030f0d05u, 0x030f0f01u,
    0x05010101u, 0x05010103u, 0x05010107u, 0x0501010bu,
    0x0501010fu, 0x05010301u, 0x05010305u, 0x05010309u,
    0x0501030du, 0x05010503u, 0x05010507u, 0x0501050fu,
    0x05010701u, 0x05010705u, 0x05010903u, 0x05010907u,
    0x0501090bu, 0x05010b01u, 0x05010b05u, 0x05010d0fu,
    0x05010f01u, 0x05010f07u, 0x05010f0bu, 0x05030101u,
    0x05030105u, 0x05030301u, 0x05030307u, 0x0503030fu,
    0x05030505u, 0x0503050bu, 0x05030703u, 0x05030709u,
    0x05030905u, 0x05030b03u, 0x05050103u, 0x05050109u,
    0x0505010fu, 0x05050503u, 0x05050507u, 0x05050701u,
    0x0505070fu, 0x05050903u, 0x05050b07u, 0x05050b0fu,
    0x05050f03u, 0x05050f09u, 0x05070101u, 0x05070105u,
    0x0507010bu, 0x05070303u, 0x05070505u, 0x05070509u,
    0x05070703u, 0x05070707u, 0x05070905u, 0x05070b01u,
    0x05070d0du, 0x05090103u, 0x0509010fu, 0x05090501u,
    0x05090507u, 0x05090705u, 0x0509070bu, 0x05090903u,
    0x05090f05u, 0x05090f0bu, 0x050b0109u, 0x050b0303u,
    0x050b0505u, 0x050b070fu, 0x050b0901u, 0x050b0b07u,
    0x050b0f01u, 0x050d0101u, 0x050d0105u, 0x050d010fu,
    0x050d0503u, 0x050d0b0bu, 0x050d0d03u, 0x050f010bu,
    0x050f0303u, 0x050f050du, 0x050f0701u, 0x050f0907u,
    0x050f0b01u, 0x07010105u, 0x07010303u, 0x07010307u,
    0x0701030bu, 0x0701030fu, 0x07010505u, 0x07010703u,
    0x07010707u, 0x0701070bu, 0x07010905u, 0x07010909u,
    0x0701090fu, 0x07010b03u, 0x07010d07u, 0x07010f03u,
    0x07030103u, 0x07030107u, 0x0703010bu, 0x07030309u,
    0x07030503u, 0x07030507u, 0x07030901u, 0x07030d01u,
    0x07030f05u, 0x07030f0du, 0x07050101u, 0x07050305u,
    0x07050501u, 0x07050705u, 0x07050709u, 0x07050b01u,
    0x07070103u, 0x07070301u, 0x07070309u, 0x07070503u,
    0x07070507u, 0x0707050fu, 0x07070701u, 0x07070903u,
    0x07070907u, 0x0707090fu, 0x07070b0bu, 0x07070f07u,
    0x07090107u, 0x07090303u, 0x0709030du, 0x07090505u,
    0x07090703u, 0x07090b05u, 0x07090d01u, 0x07090d09u,
    0x070b0103u, 0x070b0301u, 0x070b0305u, 0x070b050bu,
    0x070b0705u, 0x070b0909u, 0x070b0b0du, 0x070b0f07u,
    0x070d030du, 0x070d0903u, 0x070f0103u, 0x070f0107u,
    0x070f0501u, 0x070f0505u, 0x070f070bu, 0x09010101u,
    0x09010109u, 0x09010305u, 0x09010501u, 0x09010509u,
    0x0901050fu, 0x09010705u, 0x09010903u, 0x09010b01u,
    0x09010f01u, 0x09030105u, 0x0903010fu, 0x09030303u,
    0x09030307u, 0x09030505u, 0x09030701u, 0x0903070bu,
    0x09030907u, 0x09030b03u, 0x09030b0bu, 0x09050103u,
    0x09050107u, 0x09050301u, 0x0905030bu, 0x09050503u,
    0x09050707u, 0x09050901u, 0x09050b0fu, 0x09050d05u,
    0x09050f01u, 0x09070109u, 0x09070303u, 0x09070307u,
    0x09070501u, 0x09070505u, 0x09070703u, 0x0907070bu,
    0x09090101u, 0x09090105u, 0x09090509u, 0x0909070fu,
    0x09090901u, 0x09090f03u, 0x090b010bu, 0x090b010fu,
    0x090b0503u, 0x090b0d05u, 0x090d0307u, 0x090d0709u,
    0x090d0d01u, 0x090f0301u, 0x090f030bu, 0x090f0701u,
    0x090f0907u, 0x090f0b03u, 0x0b010105u, 0x0b010301u,
    0x0b010309u, 0x0b010505u, 0x0b010901u, 0x0b010909u,
    0x0b01090fu, 0x0b010b05u, 0x0b010d0du, 0x0b010f09u,
    0x0b030103u, 0x0b030107u, 0x0b03010bu, 0x0b030305u,
    0x0b030503u, 0x0b030705u, 0x0b030f05u, 0x0b050101u,
    0x0b050303u, 0x0b050507u, 0x0b050701u, 0x0b05070du,
    0x0b050b07u, 0x0b070105u, 0x0b07010fu, 0x0b070301u,
    0x0b07050fu, 0x0b070909u, 0x0b070b03u, 0x0b070d0bu,
    0x0b070f07u, 0x0b090103u, 0x0b090109u, 0x0b090501u,
    0x0b090705u, 0x0b09090du, 0x0b0b0305u, 0x0b0b050du,
    0x0b0b0b03u, 0x0b0b0b07u, 0x0b0d0905u, 0x0b0f0105u,
    0x0b0f0109u, 0x0b0f0505u, 0x0d010303u, 0x0d010307u,
    0x0d01030bu, 0x0d010703u, 0x0d010707u, 0x0d010d01u,
    0x0d030101u, 0x0d030501u, 0x0d03050fu, 0x0d030d09u,
    0x0d050305u, 0x0d050709u, 0x0d050905u, 0x0d050b0bu,
    0x0d050d05u, 0x0d050f01u, 0x0d070101u, 0x0d070309u,
    0x0d070503u, 0x0d070901u, 0x0d09050bu, 0x0d090907u,
    0x0d090d05u, 0x0d0b0101u, 0x0d0b0107u, 0x0d0b0709u,
    0x0d0b0d01u, 0x0d0d010bu, 0x0d0d0901u, 0x0d0f0303u,
    0x0d0f0307u, 0x0f010101u, 0x0f010109u, 0x0f01010fu,
    0x0f010501u, 0x0f010505u, 0x0f01070du, 0x0f010901u,
    0x0f010b09u, 0x0f010d05u, 0x0f030105u, 0x0f030303u,
    0x0f030509u, 0x0f030907u, 0x0f03090bu, 0x0f050103u,
    0x0f050109u, 0x0f050301u, 0x0f05030du, 0x0f050503u,
    0x0f050701u, 0x0f050b03u, 0x0f070105u, 0x0f070705u,
    0x0f07070bu, 0x0f070b07u, 0x0f090103u, 0x0f09010bu,
    0x0f090307u, 0x0f090501u, 0x0f090b01u, 0x0f0b0505u,
    0x0f0b0905u, 0x0f0d0105u, 0x0f0d0703u, 0x0f0f0101u,
};

/* Weight grid for the 1-bit formats: eight signed ternary weights an index.
 * Transcribed from `IQ1S_GRID` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) uint64_t SHISO_IQ1S_GRID[2048] = {
    0xffffffffffffffffull, 0xffffffffffffff01ull, 0xffffffffffff0000ull, 0xffffffffffff01ffull,
    0xffffffffffff0101ull, 0xffffffffff00ff00ull, 0xffffffffff000000ull, 0xffffffffff01ffffull,
    0xffffffffff01ff01ull, 0xffffffffff0101ffull, 0xffffffffff010101ull, 0xffffffff00ff0000ull,
    0xffffffff0000ff00ull, 0xffffffff000000ffull, 0xffffffff00000001ull, 0xffffffff00010000ull,
    0xffffffff01ffffffull, 0xffffffff01ffff01ull, 0xffffffff01ff01ffull, 0xffffffff01ff0101ull,
    0xffffffff01000000ull, 0xffffffff0101ffffull, 0xffffffff0101ff01ull, 0xffffffff010101ffull,
    0xffffffff01010101ull, 0xffffff00ffff00ffull, 0xffffff00ffff0000ull, 0xffffff00ff00ff00ull,
    0xffffff00ff0000ffull, 0xffffff00ff000001ull, 0xffffff00ff000100ull, 0xffffff00ff000101ull,
    0xffffff00ff010000ull, 0xffffff0000ffff00ull, 0xffffff0000ff0001ull, 0xffffff0000ff0100ull,
    0xffffff000000ff01ull, 0xffffff0000000000ull, 0xffffff0000000101ull, 0xffffff000001ff00ull,
    0xffffff00000100ffull, 0xffffff0000010001ull, 0xffffff00000101ffull, 0xffffff0001ff0000ull,
    0xffffff000100ff00ull, 0xffffff00010000ffull, 0xffffff0001000001ull, 0xffffff0001010000ull,
    0xffffff01ffffffffull, 0xffffff01ffffff01ull, 0xffffff01ffff01ffull, 0xffffff01ffff0101ull,
    0xffffff01ff000000ull, 0xffffff01ff01ffffull, 0xffffff01ff01ff01ull, 0xffffff01ff0101ffull,
    0xffffff01ff010101ull, 0xffffff0100ff0000ull, 0xffffff010000ff00ull, 0xffffff0100000100ull,
    0xffffff01000100ffull, 0xffffff0100010100ull, 0xffffff0101ffffffull, 0xffffff0101ffff01ull,
    0xffffff0101ff01ffull, 0xffffff0101ff0101ull, 0xffffff010100ff00ull, 0xffffff0101000000ull,
    0xffffff0101000100ull, 0xffffff010101ffffull, 0xffffff010101ff01ull, 0xffffff01010101ffull,
    0xffffff0101010101ull, 0xffff00ffff00ff00ull, 0xffff00ffff0000ffull, 0xffff00ffff000001ull,
    0xffff00ffff010000ull, 0xffff00ff00ffff00ull, 0xffff00ff00ff0100ull, 0xffff00ff00000000ull,
    0xffff00ff00000101ull, 0xffff00ff000100ffull, 0xffff00ff00010000ull, 0xffff00ff0100ff00ull,
    0xffff00ff01000100ull, 0xffff00ff01010000ull, 0xffff0000ffffff00ull, 0xffff0000ffff00ffull,
    0xffff0000ffff0000ull, 0xffff0000ffff0001ull, 0xffff0000ff000000ull, 0xffff0000ff0001ffull,
    0xffff0000ff000101ull, 0xffff0000ff010100ull, 0xffff000000ffffffull, 0xffff000000ff0000ull,
    0xffff000000ff0101ull, 0xffff00000000ffffull, 0xffff00000000ff00ull, 0xffff0000000000ffull,
    0xffff000000000000ull, 0xffff000000000001ull, 0xffff000000000100ull, 0xffff00000001ffffull,
    0xffff00000001ff01ull, 0xffff000000010000ull, 0xffff0000000101ffull, 0xffff000000010101ull,
    0xffff000001ffff00ull, 0xffff00000100ff00ull, 0xffff000001000000ull, 0xffff0000010001ffull,
    0xffff000001000101ull, 0xffff00000101ff00ull, 0xffff0000010100ffull, 0xffff000001010000ull,
    0xffff000001010001ull, 0xffff000001010100ull, 0xffff0001ff0000ffull, 0xffff0001ff000100ull,
    0xffff000100ffff00ull, 0xffff000100ff00ffull, 0xffff00010000ffffull, 0xffff00010000ff01ull,
    0xffff000100000000ull, 0xffff0001000001ffull, 0xffff00010001ffffull, 0xffff00010001ff00ull,
    0xffff000100010001ull, 0xffff000100010100ull, 0xffff000101ff0000ull, 0xffff00010100ff00ull,
    0xffff0001010000ffull, 0xffff000101000100ull, 0xffff01ffffffffffull, 0xffff01ffffffff01ull,
    0xffff01ffffff01ffull, 0xffff01ffffff0101ull, 0xffff01ffff000000ull, 0xffff01ffff01ffffull,
    0xffff01ffff01ff01ull, 0xffff01ffff0101ffull, 0xffff01ffff010101ull, 0xffff01ff00ff0000ull,
    0xffff01ff0000ff00ull, 0xffff01ff00000001ull, 0xffff01ff00010000ull, 0xffff01ff01ffffffull,
    0xffff01ff01ffff01ull, 0xffff01ff01ff01ffull, 0xffff01ff01ff0101ull, 0xffff01ff01000000ull,
    0xffff01ff0101ffffull, 0xffff01ff0101ff01ull, 0xffff01ff010101ffull, 0xffff01ff01010101ull,
    0xffff0100ffff0000ull, 0xffff0100ff00ff00ull, 0xffff0100ff0000ffull, 0xffff0100ff000100ull,
    0xffff0100ff0100ffull, 0xffff0100ff010000ull, 0xffff010000ffff00ull, 0xffff01000000ffffull,
    0xffff01000000ff00ull, 0xffff010000000000ull, 0xffff01000001ff00ull, 0xffff0100000100ffull,
    0xffff010000010100ull, 0xffff01000100ff00ull, 0xffff0100010000ffull, 0xffff010001000001ull,
    0xffff010001000100ull, 0xffff010001010000ull, 0xffff0101ffffffffull, 0xffff0101ffffff01ull,
    0xffff0101ffff01ffull, 0xffff0101ffff0101ull, 0xffff0101ff000000ull, 0xffff0101ff01ffffull,
    0xffff0101ff01ff01ull, 0xffff0101ff0101ffull, 0xffff0101ff010101ull, 0xffff010100ff0000ull,
    0xffff01010000ff00ull, 0xffff010100000100ull, 0xffff01010001ff00ull, 0xffff010100010000ull,
    0xffff010101ffffffull, 0xffff010101ffff01ull, 0xffff010101ff0000ull, 0xffff010101ff01ffull,
    0xffff010101ff0101ull, 0xffff010101000000ull, 0xffff01010101ffffull, 0xffff01010101ff01ull,
    0xffff0101010101ffull, 0xffff010101010101ull, 0xff00ffffff00ffffull, 0xff00ffffff00ff00ull,
    0xff00ffffff0000ffull, 0xff00ffffff000100ull, 0xff00ffffff0100ffull, 0xff00ffffff010000ull,
    0xff00ffff00ffff00ull, 0xff00ffff00ff00ffull, 0xff00ffff0000ffffull, 0xff00ffff00000000ull,
    0xff00ffff000001ffull, 0xff00ffff0001ff00ull, 0xff00ffff000100ffull, 0xff00ffff00010000ull,
    0xff00ffff00010100ull, 0xff00ffff0100ff00ull, 0xff00ffff010000ffull, 0xff00ffff01000001ull,
    0xff00ffff0101ff00ull, 0xff00ffff01010000ull, 0xff00ff00ffffff00ull, 0xff00ff00ffff00ffull,
    0xff00ff00ffff0001ull, 0xff00ff00ffff0100ull, 0xff00ff00ff00ffffull, 0xff00ff00ff00ff01ull,
    0xff00ff00ff000000ull, 0xff00ff00ff0001ffull, 0xff00ff00ff01ff00ull, 0xff00ff00ff0100ffull,
    0xff00ff00ff010100ull, 0xff00ff0000ff0000ull, 0xff00ff0000ff0101ull, 0xff00ff000000ffffull,
    0xff00ff000000ff00ull, 0xff00ff000000ff01ull, 0xff00ff00000000ffull, 0xff00ff0000000000ull,
    0xff00ff0000000001ull, 0xff00ff0000000100ull, 0xff00ff000001ffffull, 0xff00ff0000010000ull,
    0xff00ff0001ff00ffull, 0xff00ff000100ff01ull, 0xff00ff0001000000ull, 0xff00ff000101ff00ull,
    0xff00ff00010100ffull, 0xff00ff01ff00ff00ull, 0xff00ff01ff0000ffull, 0xff00ff01ff000001ull,
    0xff00ff01ff010000ull, 0xff00ff0100ffffffull, 0xff00ff0100ff0001ull, 0xff00ff0100ff0100ull,
    0xff00ff010000ff01ull, 0xff00ff0100000000ull, 0xff00ff01000001ffull, 0xff00ff0100000101ull,
    0xff00ff01000100ffull, 0xff00ff0100010001ull, 0xff00ff0101ff0000ull, 0xff00ff010100ff00ull,
    0xff00ff01010000ffull, 0xff00ff0101000001ull, 0xff00ff0101010000ull, 0xff0000ffffffff00ull,
    0xff0000ffffff0001ull, 0xff0000ffffff0100ull, 0xff0000ffff0000ffull, 0xff0000ffff000000ull,
    0xff0000ffff0001ffull, 0xff0000ffff000100ull, 0xff0000ffff01ff00ull, 0xff0000ffff010001ull,
    0xff0000ff00ffff00ull, 0xff0000ff00ff0000ull, 0xff0000ff00ff0001ull, 0xff0000ff00ff01ffull,
    0xff0000ff00ff0101ull, 0xff0000ff0000ff00ull, 0xff0000ff000000ffull, 0xff0000ff00000000ull,
    0xff0000ff00000001ull, 0xff0000ff00000100ull, 0xff0000ff0001ff01ull, 0xff0000ff00010000ull,
    0xff0000ff000101ffull, 0xff0000ff01ff00ffull, 0xff0000ff01ff0100ull, 0xff0000ff0100ffffull,
    0xff0000ff010000ffull, 0xff0000ff01000000ull, 0xff0000ff010001ffull, 0xff0000ff01000100ull,
    0xff0000ff01000101ull, 0xff0000ff0101ff00ull, 0xff0000ff010100ffull, 0xff0000ff01010000ull,
    0xff0000ff01010100ull, 0xff000000ffffff01ull, 0xff000000ffff0000ull, 0xff000000ffff0101ull,
    0xff000000ff00ff00ull, 0xff000000ff0000ffull, 0xff000000ff000000ull, 0xff000000ff000001ull,
    0xff000000ff000100ull, 0xff000000ff01ffffull, 0xff000000ff01ff01ull, 0xff000000ff010000ull,
    0xff000000ff0101ffull, 0xff000000ff010101ull, 0xff00000000ffff00ull, 0xff00000000ff00ffull,
    0xff00000000ff0000ull, 0xff00000000ff0001ull, 0xff0000000000ff00ull, 0xff0000000000ff01ull,
    0xff000000000000ffull, 0xff00000000000000ull, 0xff00000000000001ull, 0xff00000000000100ull,
    0xff00000000000101ull, 0xff0000000001ff00ull, 0xff000000000100ffull, 0xff00000000010000ull,
    0xff00000000010001ull, 0xff00000000010100ull, 0xff00000001ffffffull, 0xff00000001ffff01ull,
    0xff00000001ff00ffull, 0xff00000001ff0000ull, 0xff00000001ff01ffull, 0xff00000001ff0101ull,
    0xff0000000100ffffull, 0xff0000000100ff00ull, 0xff000000010000ffull, 0xff00000001000000ull,
    0xff00000001000001ull, 0xff00000001000100ull, 0xff00000001000101ull, 0xff0000000101ffffull,
    0xff0000000101ff01ull, 0xff00000001010000ull, 0xff000001ffffff00ull, 0xff000001ffff00ffull,
    0xff000001ffff0000ull, 0xff000001ffff0001ull, 0xff000001ff000000ull, 0xff000001ff000001ull,
    0xff000001ff0001ffull, 0xff000001ff000101ull, 0xff000001ff01ff00ull, 0xff000001ff010001ull,
    0xff00000100ffffffull, 0xff00000100ffff01ull, 0xff00000100ff00ffull, 0xff00000100ff0000ull,
    0xff00000100ff01ffull, 0xff00000100ff0101ull, 0xff0000010000ff00ull, 0xff00000100000000ull,
    0xff00000100000001ull, 0xff000001000001ffull, 0xff00000100000100ull, 0xff0000010001ff00ull,
    0xff000001000100ffull, 0xff00000100010000ull, 0xff000001000101ffull, 0xff00000100010100ull,
    0xff00000100010101ull, 0xff00000101ff0001ull, 0xff00000101ff0101ull, 0xff0000010100ff01ull,
    0xff00000101000000ull, 0xff000001010100ffull, 0xff00000101010100ull, 0xff0001ffff00ff00ull,
    0xff0001ffff000001ull, 0xff0001ffff010000ull, 0xff0001ff00ffff00ull, 0xff0001ff00ff00ffull,
    0xff0001ff00ff0001ull, 0xff0001ff00ff0100ull, 0xff0001ff0000ffffull, 0xff0001ff00000000ull,
    0xff0001ff000001ffull, 0xff0001ff00000101ull, 0xff0001ff0001ffffull, 0xff0001ff0001ff00ull,
    0xff0001ff000100ffull, 0xff0001ff00010001ull, 0xff0001ff00010100ull, 0xff0001ff01ff0000ull,
    0xff0001ff0100ff00ull, 0xff0001ff010000ffull, 0xff0001ff01010000ull, 0xff000100ff00ffffull,
    0xff000100ff00ff01ull, 0xff000100ff000000ull, 0xff000100ff000101ull, 0xff000100ff01ff00ull,
    0xff000100ff010000ull, 0xff00010000ffff01ull, 0xff00010000ff00ffull, 0xff00010000ff0000ull,
    0xff00010000ff01ffull, 0xff0001000000ff00ull, 0xff000100000000ffull, 0xff00010000000000ull,
    0xff00010000000001ull, 0xff00010000000100ull, 0xff00010000000101ull, 0xff0001000001ffffull,
    0xff00010000010000ull, 0xff00010000010101ull, 0xff00010001ff0100ull, 0xff0001000100ff00ull,
    0xff0001000100ff01ull, 0xff00010001000000ull, 0xff000100010001ffull, 0xff0001000101ff00ull,
    0xff00010001010001ull, 0xff00010001010100ull, 0xff000101ffff0100ull, 0xff000101ff000001ull,
    0xff000101ff0100ffull, 0xff000101ff010001ull, 0xff00010100ff00ffull, 0xff00010100ff0001ull,
    0xff00010100ff0100ull, 0xff0001010000ffffull, 0xff0001010000ff01ull, 0xff00010100000000ull,
    0xff000101000001ffull, 0xff0001010001ff00ull, 0xff00010100010001ull, 0xff00010100010100ull,
    0xff00010101ff0000ull, 0xff0001010100ff00ull, 0xff00010101000001ull, 0xff00010101000101ull,
    0xff01ffffffffffffull, 0xff01ffffffffff01ull, 0xff01ffffffff01ffull, 0xff01ffffffff0101ull,
    0xff01ffffff000000ull, 0xff01ffffff01ffffull, 0xff01ffffff01ff01ull, 0xff01ffffff010000ull,
    0xff01ffffff0101ffull, 0xff01ffffff010101ull, 0xff01ffff00ff0000ull, 0xff01ffff0000ff00ull,
    0xff01ffff00000100ull, 0xff01ffff0001ff00ull, 0xff01ffff00010000ull, 0xff01ffff01ffffffull,
    0xff01ffff01ffff01ull, 0xff01ffff01ff01ffull, 0xff01ffff01ff0101ull, 0xff01ffff01000000ull,
    0xff01ffff0101ffffull, 0xff01ffff0101ff01ull, 0xff01ffff01010000ull, 0xff01ffff010101ffull,
    0xff01ffff01010101ull, 0xff01ff00ffff0000ull, 0xff01ff00ff00ff00ull, 0xff01ff00ff0000ffull,
    0xff01ff00ff000100ull, 0xff01ff00ff010000ull, 0xff01ff0000ffff01ull, 0xff01ff0000ff00ffull,
    0xff01ff0000ff0100ull, 0xff01ff0000000000ull, 0xff01ff00000001ffull, 0xff01ff0000000101ull,
    0xff01ff000001ff00ull, 0xff01ff00000100ffull, 0xff01ff0000010000ull, 0xff01ff0000010001ull,
    0xff01ff0001ff0000ull, 0xff01ff000100ffffull, 0xff01ff0001000001ull, 0xff01ff0001000100ull,
    0xff01ff0001010000ull, 0xff01ff01ffffff00ull, 0xff01ff01ffff01ffull, 0xff01ff01ffff0101ull,
    0xff01ff01ff00ff00ull, 0xff01ff01ff000000ull, 0xff01ff01ff01ffffull, 0xff01ff01ff01ff01ull,
    0xff01ff01ff0101ffull, 0xff01ff01ff010101ull, 0xff01ff0100ff0000ull, 0xff01ff010000ff00ull,
    0xff01ff0100000001ull, 0xff01ff0100000100ull, 0xff01ff0100010000ull, 0xff01ff0101ffff00ull,
    0xff01ff0101ff01ffull, 0xff01ff0101ff0101ull, 0xff01ff010100ff00ull, 0xff01ff0101000000ull,
    0xff01ff010101ffffull, 0xff01ff010101ff01ull, 0xff01ff01010101ffull, 0xff01ff0101010101ull,
    0xff0100ffffff0000ull, 0xff0100ffff0000ffull, 0xff0100ffff000001ull, 0xff0100ffff000100ull,
    0xff0100ffff010000ull, 0xff0100ff00ff00ffull, 0xff0100ff00ff0000ull, 0xff0100ff00ff0001ull,
    0xff0100ff00ff0100ull, 0xff0100ff0000ff01ull, 0xff0100ff00000000ull, 0xff0100ff000001ffull,
    0xff0100ff00000101ull, 0xff0100ff00010001ull, 0xff0100ff01ff0000ull, 0xff0100ff0100ff00ull,
    0xff0100ff010000ffull, 0xff0100ff01000100ull, 0xff0100ff0101ff00ull, 0xff0100ff01010000ull,
    0xff010000ffff0100ull, 0xff010000ff000000ull, 0xff010000ff01ff00ull, 0xff010000ff010100ull,
    0xff01000000ffffffull, 0xff01000000ff0000ull, 0xff01000000ff01ffull, 0xff0100000000ff00ull,
    0xff010000000000ffull, 0xff01000000000000ull, 0xff01000000000100ull, 0xff0100000001ff01ull,
    0xff01000000010000ull, 0xff010000000101ffull, 0xff01000001ff0100ull, 0xff0100000100ffffull,
    0xff010000010000ffull, 0xff01000001000000ull, 0xff010000010001ffull, 0xff01000001000101ull,
    0xff0100000101ff00ull, 0xff010000010100ffull, 0xff01000001010001ull, 0xff01000001010100ull,
    0xff010001ffff0000ull, 0xff010001ff00ffffull, 0xff010001ff00ff01ull, 0xff010001ff000100ull,
    0xff010001ff010000ull, 0xff01000100ffff00ull, 0xff01000100ff0100ull, 0xff01000100000000ull,
    0xff0100010001ffffull, 0xff0100010001ff00ull, 0xff01000100010100ull, 0xff01000101ff00ffull,
    0xff01000101ff0001ull, 0xff0100010100ffffull, 0xff01000101000101ull, 0xff0101ffffffffffull,
    0xff0101ffffffff01ull, 0xff0101ffffff01ffull, 0xff0101ffffff0101ull, 0xff0101ffff000000ull,
    0xff0101ffff01ffffull, 0xff0101ffff01ff01ull, 0xff0101ffff0101ffull, 0xff0101ffff010101ull,
    0xff0101ff00ff0000ull, 0xff0101ff0000ff00ull, 0xff0101ff000000ffull, 0xff0101ff00010000ull,
    0xff0101ff01ffffffull, 0xff0101ff01ffff01ull, 0xff0101ff01ff01ffull, 0xff0101ff01ff0101ull,
    0xff0101ff0101ffffull, 0xff0101ff0101ff01ull, 0xff0101ff010101ffull, 0xff0101ff01010101ull,
    0xff010100ffff0100ull, 0xff010100ff00ff00ull, 0xff010100ff0000ffull, 0xff010100ff000100ull,
    0xff010100ff010000ull, 0xff01010000ff0001ull, 0xff01010000ff0100ull, 0xff0101000000ff01ull,
    0xff01010000000000ull, 0xff0101000001ff00ull, 0xff010100000100ffull, 0xff01010000010001ull,
    0xff01010000010100ull, 0xff01010001ff0000ull, 0xff0101000100ffffull, 0xff01010001000001ull,
    0xff01010001000100ull, 0xff010100010100ffull, 0xff01010001010000ull, 0xff010101ffffffffull,
    0xff010101ffffff01ull, 0xff010101ffff01ffull, 0xff010101ffff0101ull, 0xff010101ff01ffffull,
    0xff010101ff01ff01ull, 0xff010101ff0101ffull, 0xff010101ff010101ull, 0xff01010100ff0000ull,
    0xff0101010000ff00ull, 0xff01010100000001ull, 0xff01010100000100ull, 0xff01010100010000ull,
    0xff01010101ffffffull, 0xff01010101ffff01ull, 0xff01010101ff01ffull, 0xff01010101ff0101ull,
    0xff01010101000000ull, 0xff0101010101ffffull, 0xff0101010101ff01ull, 0xff010101010101ffull,
    0xff01010101010101ull, 0x00ffffffffff0000ull, 0x00ffffffff00ff00ull, 0x00ffffffff000001ull,
    0x00ffffffff010000ull, 0x00ffffff00ff0100ull, 0x00ffffff0000ff01ull, 0x00ffffff00000000ull,
    0x00ffffff000001ffull, 0x00ffffff00000101ull, 0x00ffffff0001ff00ull, 0x00ffffff000100ffull,
    0x00ffffff00010001ull, 0x00ffffff010000ffull, 0x00ffffff01000100ull, 0x00ffffff0101ff00ull,
    0x00ffffff01010001ull, 0x00ffff00ffffffffull, 0x00ffff00ffffff00ull, 0x00ffff00ffff00ffull,
    0x00ffff00ffff0001ull, 0x00ffff00ffff0100ull, 0x00ffff00ff00ff01ull, 0x00ffff00ff000000ull,
    0x00ffff00ff000001ull, 0x00ffff00ff0001ffull, 0x00ffff00ff000101ull, 0x00ffff00ff01ff00ull,
    0x00ffff00ff010001ull, 0x00ffff00ff010100ull, 0x00ffff0000ff0000ull, 0x00ffff0000ff01ffull,
    0x00ffff0000ff0101ull, 0x00ffff000000ff00ull, 0x00ffff00000000ffull, 0x00ffff0000000000ull,
    0x00ffff0000000001ull, 0x00ffff0000000100ull, 0x00ffff0000000101ull, 0x00ffff0000010000ull,
    0x00ffff00000101ffull, 0x00ffff0000010101ull, 0x00ffff0001ffff00ull, 0x00ffff0001ff00ffull,
    0x00ffff0001ff0001ull, 0x00ffff000100ffffull, 0x00ffff000100ff01ull, 0x00ffff0001000000ull,
    0x00ffff000101ffffull, 0x00ffff000101ff00ull, 0x00ffff000101ff01ull, 0x00ffff01ffff0000ull,
    0x00ffff01ff00ff00ull, 0x00ffff01ff0000ffull, 0x00ffff01ff000001ull, 0x00ffff01ff010000ull,
    0x00ffff0100ffff00ull, 0x00ffff010000ff01ull, 0x00ffff0100000000ull, 0x00ffff0100000101ull,
    0x00ffff01000100ffull, 0x00ffff0100010100ull, 0x00ffff0101ff0100ull, 0x00ffff01010000ffull,
    0x00ffff0101010000ull, 0x00ff00ffffffff00ull, 0x00ff00ffff000000ull, 0x00ff00ffff000100ull,
    0x00ff00ffff010100ull, 0x00ff00ff00ff0000ull, 0x00ff00ff00ff01ffull, 0x00ff00ff00ff0101ull,
    0x00ff00ff0000ff00ull, 0x00ff00ff000000ffull, 0x00ff00ff00000000ull, 0x00ff00ff00000001ull,
    0x00ff00ff0001ff00ull, 0x00ff00ff0001ff01ull, 0x00ff00ff00010000ull, 0x00ff00ff000101ffull,
    0x00ff00ff00010101ull, 0x00ff00ff01ffff00ull, 0x00ff00ff01ff0001ull, 0x00ff00ff01ff0100ull,
    0x00ff00ff0100ffffull, 0x00ff00ff0100ff01ull, 0x00ff00ff01000000ull, 0x00ff00ff0101ffffull,
    0x00ff00ff0101ff00ull, 0x00ff00ff01010100ull, 0x00ff0000ffffff00ull, 0x00ff0000ffffff01ull,
    0x00ff0000ffff0000ull, 0x00ff0000ffff0101ull, 0x00ff0000ff00ff00ull, 0x00ff0000ff0000ffull,
    0x00ff0000ff000000ull, 0x00ff0000ff000001ull, 0x00ff0000ff000100ull, 0x00ff0000ff01ffffull,
    0x00ff0000ff010000ull, 0x00ff0000ff010101ull, 0x00ff000000ffff00ull, 0x00ff000000ff00ffull,
    0x00ff000000ff0000ull, 0x00ff000000ff0001ull, 0x00ff000000ff0100ull, 0x00ff00000000ffffull,
    0x00ff00000000ff00ull, 0x00ff0000000000ffull, 0x00ff000000000000ull, 0x00ff000000000001ull,
    0x00ff0000000001ffull, 0x00ff000000000100ull, 0x00ff00000001ff00ull, 0x00ff0000000100ffull,
    0x00ff000000010000ull, 0x00ff000000010001ull, 0x00ff000000010100ull, 0x00ff000001ffff01ull,
    0x00ff000001ff00ffull, 0x00ff000001ff0000ull, 0x00ff000001ff01ffull, 0x00ff00000100ff00ull,
    0x00ff0000010000ffull, 0x00ff000001000000ull, 0x00ff000001000001ull, 0x00ff000001000100ull,
    0x00ff000001000101ull, 0x00ff000001010000ull, 0x00ff0000010101ffull, 0x00ff000001010101ull,
    0x00ff0001ffffff00ull, 0x00ff0001ffff0000ull, 0x00ff0001ffff0100ull, 0x00ff0001ff0000ffull,
    0x00ff0001ff000000ull, 0x00ff0001ff0001ffull, 0x00ff0001ff000101ull, 0x00ff0001ff01ff00ull,
    0x00ff0001ff0100ffull, 0x00ff0001ff010100ull, 0x00ff000100ffffffull, 0x00ff000100ffff01ull,
    0x00ff000100ff0000ull, 0x00ff000100ff01ffull, 0x00ff00010000ffffull, 0x00ff00010000ff00ull,
    0x00ff00010000ff01ull, 0x00ff000100000000ull, 0x00ff000100000001ull, 0x00ff000100000100ull,
    0x00ff00010001ff01ull, 0x00ff000100010000ull, 0x00ff0001000101ffull, 0x00ff000101ffff00ull,
    0x00ff000101ff0000ull, 0x00ff000101ff0101ull, 0x00ff0001010000ffull, 0x00ff000101000000ull,
    0x00ff00010101ff00ull, 0x00ff0001010100ffull, 0x00ff000101010001ull, 0x00ff01ffffff0000ull,
    0x00ff01ffff00ff00ull, 0x00ff01ffff000000ull, 0x00ff01ffff000101ull, 0x00ff01ffff010000ull,
    0x00ff01ff00ffff01ull, 0x00ff01ff00ff0100ull, 0x00ff01ff0000ffffull, 0x00ff01ff00000000ull,
    0x00ff01ff000001ffull, 0x00ff01ff0001ff00ull, 0x00ff01ff000100ffull, 0x00ff01ff00010001ull,
    0x00ff01ff00010100ull, 0x00ff01ff01ff0000ull, 0x00ff01ff0100ff00ull, 0x00ff01ff010000ffull,
    0x00ff01ff01000001ull, 0x00ff01ff01000100ull, 0x00ff01ff01010000ull, 0x00ff0100ffffff00ull,
    0x00ff0100ffff0000ull, 0x00ff0100ffff0001ull, 0x00ff0100ffff0101ull, 0x00ff0100ff00ffffull,
    0x00ff0100ff0000ffull, 0x00ff0100ff000000ull, 0x00ff0100ff0001ffull, 0x00ff0100ff01ff00ull,
    0x00ff0100ff0100ffull, 0x00ff0100ff010001ull, 0x00ff010000ffffffull, 0x00ff010000ff0000ull,
    0x00ff010000ff0101ull, 0x00ff01000000ff00ull, 0x00ff01000000ff01ull, 0x00ff0100000000ffull,
    0x00ff010000000000ull, 0x00ff010000000001ull, 0x00ff010000000100ull, 0x00ff01000001ffffull,
    0x00ff01000001ff01ull, 0x00ff010000010000ull, 0x00ff010000010001ull, 0x00ff010000010101ull,
    0x00ff010001ff0001ull, 0x00ff010001ff0100ull, 0x00ff01000100ff01ull, 0x00ff010001000000ull,
    0x00ff010001000001ull, 0x00ff0100010001ffull, 0x00ff01000101ff00ull, 0x00ff0100010100ffull,
    0x00ff010001010001ull, 0x00ff010001010100ull, 0x00ff0101ff000001ull, 0x00ff010100ff00ffull,
    0x00ff010100ff0001ull, 0x00ff010100ff0100ull, 0x00ff010100000000ull, 0x00ff0101000001ffull,
    0x00ff010100000101ull, 0x00ff0101000100ffull, 0x00ff010100010100ull, 0x00ff0101010000ffull,
    0x00ff010101010000ull, 0x0000ffffffffff00ull, 0x0000ffffffff00ffull, 0x0000ffffffff0000ull,
    0x0000ffffffff0001ull, 0x0000ffffffff0100ull, 0x0000ffffff00ff01ull, 0x0000ffffff000000ull,
    0x0000ffffff000101ull, 0x0000ffffff01ff00ull, 0x0000ffffff0100ffull, 0x0000ffffff010100ull,
    0x0000ffff00ffffffull, 0x0000ffff00ff0000ull, 0x0000ffff00ff01ffull, 0x0000ffff0000ff00ull,
    0x0000ffff000000ffull, 0x0000ffff00000000ull, 0x0000ffff00000001ull, 0x0000ffff00000100ull,
    0x0000ffff00010000ull, 0x0000ffff000101ffull, 0x0000ffff01ff0001ull, 0x0000ffff01ff0100ull,
    0x0000ffff01000000ull, 0x0000ffff010001ffull, 0x0000ffff0101ffffull, 0x0000ffff0101ff00ull,
    0x0000ffff01010001ull, 0x0000ffff01010100ull, 0x0000ff00ffff0000ull, 0x0000ff00ffff01ffull,
    0x0000ff00ffff0100ull, 0x0000ff00ffff0101ull, 0x0000ff00ff00ff00ull, 0x0000ff00ff0000ffull,
    0x0000ff00ff000000ull, 0x0000ff00ff000001ull, 0x0000ff00ff0001ffull, 0x0000ff00ff000100ull,
    0x0000ff00ff01ffffull, 0x0000ff00ff010000ull, 0x0000ff00ff010001ull, 0x0000ff00ff0101ffull,
    0x0000ff00ff010101ull, 0x0000ff0000ffff00ull, 0x0000ff0000ff00ffull, 0x0000ff0000ff0000ull,
    0x0000ff0000ff0001ull, 0x0000ff0000ff0100ull, 0x0000ff000000ffffull, 0x0000ff000000ff00ull,
    0x0000ff000000ff01ull, 0x0000ff00000000ffull, 0x0000ff0000000000ull, 0x0000ff0000000001ull,
    0x0000ff00000001ffull, 0x0000ff0000000100ull, 0x0000ff0000000101ull, 0x0000ff000001ff00ull,
    0x0000ff00000100ffull, 0x0000ff0000010000ull, 0x0000ff0000010001ull, 0x0000ff0000010100ull,
    0x0000ff0001ffff01ull, 0x0000ff0001ff0000ull, 0x0000ff000100ff00ull, 0x0000ff00010000ffull,
    0x0000ff0001000000ull, 0x0000ff0001000001ull, 0x0000ff0001000100ull, 0x0000ff000101ffffull,
    0x0000ff0001010000ull, 0x0000ff0001010101ull, 0x0000ff01ffffff00ull, 0x0000ff01ffff0001ull,
    0x0000ff01ff00ff01ull, 0x0000ff01ff000000ull, 0x0000ff01ff000101ull, 0x0000ff01ff01ff00ull,
    0x0000ff01ff0100ffull, 0x0000ff0100ffff01ull, 0x0000ff0100ff0000ull, 0x0000ff0100ff0101ull,
    0x0000ff010000ff00ull, 0x0000ff01000000ffull, 0x0000ff0100000000ull, 0x0000ff0100000001ull,
    0x0000ff0100000100ull, 0x0000ff010001ff01ull, 0x0000ff0100010000ull, 0x0000ff0101ff0000ull,
    0x0000ff010100ffffull, 0x0000ff010100ff01ull, 0x0000ff0101000000ull, 0x0000ff0101000100ull,
    0x0000ff0101000101ull, 0x0000ff01010100ffull, 0x000000ffffff00ffull, 0x000000ffffff0000ull,
    0x000000ffff00ff00ull, 0x000000ffff0000ffull, 0x000000ffff000000ull, 0x000000ffff000001ull,
    0x000000ffff0001ffull, 0x000000ffff000100ull, 0x000000ffff01ff00ull, 0x000000ffff010000ull,
    0x000000ffff0101ffull, 0x000000ffff010101ull, 0x000000ff00ffff00ull, 0x000000ff00ff00ffull,
    0x000000ff00ff0000ull, 0x000000ff00ff0001ull, 0x000000ff00ff0100ull, 0x000000ff00ff0101ull,
    0x000000ff0000ffffull, 0x000000ff0000ff00ull, 0x000000ff000000ffull, 0x000000ff00000000ull,
    0x000000ff00000001ull, 0x000000ff000001ffull, 0x000000ff00000100ull, 0x000000ff00000101ull,
    0x000000ff0001ff00ull, 0x000000ff0001ff01ull, 0x000000ff000100ffull, 0x000000ff00010000ull,
    0x000000ff00010001ull, 0x000000ff00010100ull, 0x000000ff01ffffffull, 0x000000ff01ff01ffull,
    0x000000ff01ff0101ull, 0x000000ff0100ff00ull, 0x000000ff010000ffull, 0x000000ff01000000ull,
    0x000000ff01000001ull, 0x000000ff01000100ull, 0x000000ff0101ff00ull, 0x000000ff010100ffull,
    0x000000ff01010000ull, 0x000000ff01010101ull, 0x00000000ffffff00ull, 0x00000000ffffff01ull,
    0x00000000ffff00ffull, 0x00000000ffff0000ull, 0x00000000ffff0001ull, 0x00000000ffff0100ull,
    0x00000000ff00ffffull, 0x00000000ff00ff00ull, 0x00000000ff00ff01ull, 0x00000000ff0000ffull,
    0x00000000ff000000ull, 0x00000000ff000001ull, 0x00000000ff000100ull, 0x00000000ff000101ull,
    0x00000000ff01ff00ull, 0x00000000ff0100ffull, 0x00000000ff010000ull, 0x00000000ff010001ull,
    0x00000000ff010100ull, 0x0000000000ffffffull, 0x0000000000ffff00ull, 0x0000000000ffff01ull,
    0x0000000000ff00ffull, 0x0000000000ff0000ull, 0x0000000000ff0001ull, 0x0000000000ff01ffull,
    0x0000000000ff0100ull, 0x000000000000ffffull, 0x000000000000ff00ull, 0x000000000000ff01ull,
    0x00000000000000ffull, 0x0000000000000000ull, 0x0000000000000001ull, 0x00000000000001ffull,
    0x0000000000000100ull, 0x0000000000000101ull, 0x000000000001ffffull, 0x000000000001ff00ull,
    0x00000000000100ffull, 0x0000000000010000ull, 0x0000000000010001ull, 0x00000000000101ffull,
    0x0000000000010100ull, 0x0000000000010101ull, 0x0000000001ffff00ull, 0x0000000001ff00ffull,
    0x0000000001ff0000ull, 0x0000000001ff0100ull, 0x0000000001ff0101ull, 0x000000000100ffffull,
    0x000000000100ff00ull, 0x00000000010000ffull, 0x0000000001000000ull, 0x0000000001000001ull,
    0x00000000010001ffull, 0x0000000001000100ull, 0x000000000101ff00ull, 0x00000000010100ffull,
    0x0000000001010000ull, 0x0000000001010001ull, 0x0000000001010100ull, 0x00000001ffffffffull,
    0x00000001ffffff00ull, 0x00000001ffffff01ull, 0x00000001ffff00ffull, 0x00000001ffff0001ull,
    0x00000001ffff01ffull, 0x00000001ffff0100ull, 0x00000001ff00ff00ull, 0x00000001ff0000ffull,
    0x00000001ff000000ull, 0x00000001ff0001ffull, 0x00000001ff000100ull, 0x00000001ff01ffffull,
    0x00000001ff01ff00ull, 0x00000001ff01ff01ull, 0x00000001ff0100ffull, 0x00000001ff010000ull,
    0x00000001ff010001ull, 0x00000001ff0101ffull, 0x00000001ff010100ull, 0x0000000100ffff00ull,
    0x0000000100ff0000ull, 0x0000000100ff0001ull, 0x0000000100ff01ffull, 0x0000000100ff0100ull,
    0x0000000100ff0101ull, 0x000000010000ffffull, 0x000000010000ff00ull, 0x000000010000ff01ull,
    0x00000001000000ffull, 0x0000000100000000ull, 0x0000000100000001ull, 0x00000001000001ffull,
    0x0000000100000100ull, 0x0000000100000101ull, 0x000000010001ff00ull, 0x00000001000100ffull,
    0x0000000100010000ull, 0x0000000100010100ull, 0x0000000101ffff01ull, 0x0000000101ff0000ull,
    0x0000000101ff0001ull, 0x0000000101ff01ffull, 0x0000000101ff0100ull, 0x0000000101ff0101ull,
    0x000000010100ff00ull, 0x0000000101000000ull, 0x0000000101000101ull, 0x000000010101ff01ull,
    0x0000000101010000ull, 0x0000000101010001ull, 0x00000001010101ffull, 0x0000000101010100ull,
    0x000001ffffff00ffull, 0x000001ffffff0000ull, 0x000001ffffff0001ull, 0x000001ffffff0100ull,
    0x000001ffff00ffffull, 0x000001ffff000000ull, 0x000001ffff0001ffull, 0x000001ffff01ff00ull,
    0x000001ffff010101ull, 0x000001ff00ff0000ull, 0x000001ff00ff01ffull, 0x000001ff00ff0101ull,
    0x000001ff0000ff00ull, 0x000001ff000000ffull, 0x000001ff00000000ull, 0x000001ff00000001ull,
    0x000001ff000001ffull, 0x000001ff00000100ull, 0x000001ff0001ffffull, 0x000001ff0001ff01ull,
    0x000001ff000100ffull, 0x000001ff00010000ull, 0x000001ff01ffff01ull, 0x000001ff01ff0100ull,
    0x000001ff0100ffffull, 0x000001ff0100ff01ull, 0x000001ff01000000ull, 0x000001ff010001ffull,
    0x000001ff0101ff00ull, 0x000001ff01010100ull, 0x00000100ffffff00ull, 0x00000100ffffff01ull,
    0x00000100ffff0000ull, 0x00000100ffff0101ull, 0x00000100ff00ff00ull, 0x00000100ff0000ffull,
    0x00000100ff000000ull, 0x00000100ff000001ull, 0x00000100ff000100ull, 0x00000100ff010000ull,
    0x0000010000ffff00ull, 0x0000010000ff00ffull, 0x0000010000ff0000ull, 0x0000010000ff0001ull,
    0x0000010000ff0100ull, 0x000001000000ffffull, 0x000001000000ff00ull, 0x000001000000ff01ull,
    0x00000100000000ffull, 0x0000010000000000ull, 0x0000010000000001ull, 0x00000100000001ffull,
    0x0000010000000100ull, 0x0000010000000101ull, 0x000001000001ff00ull, 0x00000100000100ffull,
    0x0000010000010000ull, 0x0000010000010001ull, 0x0000010000010100ull, 0x0000010001ffff00ull,
    0x0000010001ff0000ull, 0x0000010001ff0100ull, 0x000001000100ff00ull, 0x00000100010000ffull,
    0x0000010001000000ull, 0x0000010001000001ull, 0x00000100010001ffull, 0x0000010001000100ull,
    0x0000010001010000ull, 0x00000101ffff00ffull, 0x00000101ffff01ffull, 0x00000101ff000000ull,
    0x00000101ff000101ull, 0x00000101ff01ffffull, 0x00000101ff010000ull, 0x00000101ff010001ull,
    0x00000101ff010100ull, 0x0000010100ff0000ull, 0x0000010100ff01ffull, 0x0000010100ff0100ull,
    0x000001010000ff00ull, 0x0000010100000000ull, 0x0000010100000001ull, 0x00000101000001ffull,
    0x0000010100000100ull, 0x000001010001ff01ull, 0x0000010100010000ull, 0x00000101000101ffull,
    0x0000010100010101ull, 0x0000010101ffff00ull, 0x0000010101ff0101ull, 0x000001010100ff01ull,
    0x0000010101000000ull, 0x0000010101000001ull, 0x00000101010001ffull, 0x0000010101000101ull,
    0x000001010101ff00ull, 0x0001ffffffff0000ull, 0x0001ffffff0000ffull, 0x0001ffffff000001ull,
    0x0001ffffff000100ull, 0x0001ffffff010000ull, 0x0001ffff00ff00ffull, 0x0001ffff0000ffffull,
    0x0001ffff00000000ull, 0x0001ffff00000001ull, 0x0001ffff000001ffull, 0x0001ffff00000101ull,
    0x0001ffff0001ff00ull, 0x0001ffff000100ffull, 0x0001ffff00010001ull, 0x0001ffff00010100ull,
    0x0001ffff01ffff00ull, 0x0001ffff01000001ull, 0x0001ffff01010000ull, 0x0001ff00ffffff00ull,
    0x0001ff00ffff00ffull, 0x0001ff00ffff0001ull, 0x0001ff00ffff0100ull, 0x0001ff00ff00ff01ull,
    0x0001ff00ff000000ull, 0x0001ff00ff01ff00ull, 0x0001ff00ff01ff01ull, 0x0001ff00ff010001ull,
    0x0001ff00ff010100ull, 0x0001ff0000ff0000ull, 0x0001ff0000ff0100ull, 0x0001ff000000ff00ull,
    0x0001ff0000000000ull, 0x0001ff0000000001ull, 0x0001ff0000000100ull, 0x0001ff0000010000ull,
    0x0001ff0000010001ull, 0x0001ff0000010101ull, 0x0001ff0001ff00ffull, 0x0001ff0001ff0101ull,
    0x0001ff000100ff01ull, 0x0001ff0001000000ull, 0x0001ff000101ff00ull, 0x0001ff0001010001ull,
    0x0001ff0001010100ull, 0x0001ff01ff00ff00ull, 0x0001ff01ff000001ull, 0x0001ff01ff000100ull,
    0x0001ff0100ffffffull, 0x0001ff0100ffff00ull, 0x0001ff0100ff0001ull, 0x0001ff0100000000ull,
    0x0001ff0100000001ull, 0x0001ff01000001ffull, 0x0001ff010001ffffull, 0x0001ff0101ff0000ull,
    0x0001ff010100ff00ull, 0x0001ff0101000001ull, 0x0001ff0101010000ull, 0x000100ffff00ff00ull,
    0x000100ffff00ff01ull, 0x000100ffff000000ull, 0x000100ffff000001ull, 0x000100ffff000101ull,
    0x000100ffff01ff00ull, 0x000100ffff010001ull, 0x000100ffff010100ull, 0x000100ff00ffffffull,
    0x000100ff00ffff01ull, 0x000100ff00ff0000ull, 0x000100ff00ff01ffull, 0x000100ff00ff0101ull,
    0x000100ff0000ff00ull, 0x000100ff000000ffull, 0x000100ff00000000ull, 0x000100ff00000001ull,
    0x000100ff00000100ull, 0x000100ff00000101ull, 0x000100ff0001ffffull, 0x000100ff0001ff01ull,
    0x000100ff00010000ull, 0x000100ff01ff00ffull, 0x000100ff01ff0000ull, 0x000100ff01ff0100ull,
    0x000100ff0100ffffull, 0x000100ff0100ff01ull, 0x000100ff010000ffull, 0x000100ff01000000ull,
    0x000100ff01000001ull, 0x000100ff010001ffull, 0x000100ff01000101ull, 0x000100ff0101ff00ull,
    0x000100ff010100ffull, 0x000100ff01010100ull, 0x00010000ffff0000ull, 0x00010000ffff01ffull,
    0x00010000ffff0101ull, 0x00010000ff00ff00ull, 0x00010000ff000000ull, 0x00010000ff000001ull,
    0x00010000ff000100ull, 0x0001000000ff00ffull, 0x0001000000ff0000ull, 0x0001000000ff0001ull,
    0x0001000000ff0100ull, 0x000100000000ffffull, 0x000100000000ff00ull, 0x00010000000000ffull,
    0x0001000000000000ull, 0x0001000000000001ull, 0x0001000000000100ull, 0x000100000001ff00ull,
    0x00010000000100ffull, 0x0001000000010000ull, 0x0001000000010001ull, 0x0001000000010100ull,
    0x0001000001ff0001ull, 0x0001000001ff0100ull, 0x0001000001ff0101ull, 0x000100000100ff00ull,
    0x0001000001000000ull, 0x0001000001000001ull, 0x0001000001000100ull, 0x0001000001000101ull,
    0x000100000101ff01ull, 0x0001000001010000ull, 0x0001000001010001ull, 0x00010000010101ffull,
    0x00010001ffffff01ull, 0x00010001ffff0100ull, 0x00010001ff000000ull, 0x00010001ff01ffffull,
    0x00010001ff010001ull, 0x00010001ff0101ffull, 0x00010001ff010100ull, 0x0001000100ffffffull,
    0x0001000100ff0000ull, 0x0001000100ff01ffull, 0x0001000100ff0101ull, 0x000100010000ff00ull,
    0x00010001000000ffull, 0x0001000100000000ull, 0x0001000100000001ull, 0x00010001000001ffull,
    0x0001000100000101ull, 0x000100010001ffffull, 0x0001000100010000ull, 0x00010001000101ffull,
    0x0001000101ffffffull, 0x0001000101ffff01ull, 0x0001000101ff0000ull, 0x0001000101ff0101ull,
    0x00010001010000ffull, 0x0001000101000001ull, 0x00010001010001ffull, 0x0001000101000100ull,
    0x000100010101ffffull, 0x00010001010100ffull, 0x0001000101010001ull, 0x0001000101010101ull,
    0x000101ffff000001ull, 0x000101ffff000100ull, 0x000101ffff010000ull, 0x000101ff00ffff00ull,
    0x000101ff0000ff01ull, 0x000101ff00000000ull, 0x000101ff00000101ull, 0x000101ff0001ff00ull,
    0x000101ff00010100ull, 0x000101ff01ff0000ull, 0x000101ff0100ff00ull, 0x000101ff010001ffull,
    0x000101ff01010001ull, 0x00010100ffffff00ull, 0x00010100ffff00ffull, 0x00010100ff00ffffull,
    0x00010100ff000000ull, 0x00010100ff01ff00ull, 0x00010100ff0100ffull, 0x00010100ff010001ull,
    0x00010100ff010100ull, 0x0001010000ffffffull, 0x0001010000ffff00ull, 0x0001010000ff0000ull,
    0x0001010000ff0001ull, 0x0001010000ff01ffull, 0x000101000000ff00ull, 0x00010100000000ffull,
    0x0001010000000000ull, 0x0001010000000001ull, 0x0001010000000100ull, 0x000101000001ffffull,
    0x0001010000010000ull, 0x0001010000010101ull, 0x0001010001ffff01ull, 0x0001010001ff00ffull,
    0x0001010001ff0101ull, 0x0001010001000000ull, 0x000101000101ff00ull, 0x00010100010100ffull,
    0x0001010001010000ull, 0x0001010001010100ull, 0x00010101ff00ff00ull, 0x00010101ff000001ull,
    0x00010101ff0001ffull, 0x0001010100ffff00ull, 0x0001010100ff00ffull, 0x0001010100ff0100ull,
    0x000101010000ffffull, 0x0001010100000000ull, 0x00010101000001ffull, 0x0001010100000101ull,
    0x00010101000100ffull, 0x0001010100010000ull, 0x0001010100010100ull, 0x0001010101ff0001ull,
    0x00010101010000ffull, 0x00010101010001ffull, 0x0001010101000101ull, 0x0001010101010001ull,
    0x01ffffffffffffffull, 0x01ffffffffffff01ull, 0x01ffffffffff01ffull, 0x01ffffffffff0101ull,
    0x01ffffffff01ffffull, 0x01ffffffff01ff01ull, 0x01ffffffff0101ffull, 0x01ffffffff010101ull,
    0x01ffffff00ff0000ull, 0x01ffffff0000ffffull, 0x01ffffff0000ff00ull, 0x01ffffff000000ffull,
    0x01ffffff00000001ull, 0x01ffffff00000100ull, 0x01ffffff00010000ull, 0x01ffffff01ffffffull,
    0x01ffffff01ffff01ull, 0x01ffffff01ff01ffull, 0x01ffffff01ff0101ull, 0x01ffffff01000000ull,
    0x01ffffff0101ffffull, 0x01ffffff0101ff01ull, 0x01ffffff010101ffull, 0x01ffffff01010101ull,
    0x01ffff00ffff0000ull, 0x01ffff00ff00ff00ull, 0x01ffff00ff0000ffull, 0x01ffff00ff000001ull,
    0x01ffff00ff000100ull, 0x01ffff00ff010000ull, 0x01ffff0000ffff00ull, 0x01ffff0000ff00ffull,
    0x01ffff0000ff0100ull, 0x01ffff000000ffffull, 0x01ffff000000ff01ull, 0x01ffff0000000000ull,
    0x01ffff0000000001ull, 0x01ffff00000001ffull, 0x01ffff0000000100ull, 0x01ffff00000100ffull,
    0x01ffff0000010001ull, 0x01ffff0000010100ull, 0x01ffff0001ff0000ull, 0x01ffff0001ff0100ull,
    0x01ffff00010000ffull, 0x01ffff0001000001ull, 0x01ffff0001000100ull, 0x01ffff0001010000ull,
    0x01ffff01ffffffffull, 0x01ffff01ffffff01ull, 0x01ffff01ffff01ffull, 0x01ffff01ffff0101ull,
    0x01ffff01ff000000ull, 0x01ffff01ff01ffffull, 0x01ffff01ff01ff01ull, 0x01ffff01ff0101ffull,
    0x01ffff01ff010101ull, 0x01ffff010000ff00ull, 0x01ffff01000000ffull, 0x01ffff0100000100ull,
    0x01ffff0100010000ull, 0x01ffff0101ffffffull, 0x01ffff0101ffff01ull, 0x01ffff0101ff01ffull,
    0x01ffff0101ff0101ull, 0x01ffff0101000000ull, 0x01ffff010101ffffull, 0x01ffff010101ff01ull,
    0x01ffff01010101ffull, 0x01ffff0101010101ull, 0x01ff00ffff0000ffull, 0x01ff00ffff000100ull,
    0x01ff00ff00ffff00ull, 0x01ff00ff00ff00ffull, 0x01ff00ff0000ff00ull, 0x01ff00ff00000000ull,
    0x01ff00ff00000101ull, 0x01ff00ff0001ff00ull, 0x01ff00ff000100ffull, 0x01ff00ff00010100ull,
    0x01ff00ff010000ffull, 0x01ff00ff01000100ull, 0x01ff0000ffffff00ull, 0x01ff0000ffff0100ull,
    0x01ff0000ff00ff01ull, 0x01ff0000ff000000ull, 0x01ff0000ff000101ull, 0x01ff0000ff010001ull,
    0x01ff0000ff010100ull, 0x01ff000000ffffffull, 0x01ff000000ffff00ull, 0x01ff000000ff0000ull,
    0x01ff000000ff01ffull, 0x01ff00000000ff00ull, 0x01ff0000000000ffull, 0x01ff000000000000ull,
    0x01ff000000000001ull, 0x01ff000000000100ull, 0x01ff000000000101ull, 0x01ff000000010000ull,
    0x01ff000000010001ull, 0x01ff0000000101ffull, 0x01ff000000010101ull, 0x01ff000001ffff00ull,
    0x01ff000001ff00ffull, 0x01ff000001ff0001ull, 0x01ff000001ff0100ull, 0x01ff00000100ffffull,
    0x01ff00000100ff01ull, 0x01ff000001000000ull, 0x01ff0000010001ffull, 0x01ff000001010001ull,
    0x01ff0001ff00ff00ull, 0x01ff0001ff000001ull, 0x01ff0001ff000100ull, 0x01ff0001ff010000ull,
    0x01ff000100ffff00ull, 0x01ff000100ff00ffull, 0x01ff000100ff0100ull, 0x01ff000100ff0101ull,
    0x01ff00010000ffffull, 0x01ff000100000000ull, 0x01ff000100000100ull, 0x01ff000100000101ull,
    0x01ff00010001ff00ull, 0x01ff000100010001ull, 0x01ff000100010101ull, 0x01ff000101ff0000ull,
    0x01ff00010100ff00ull, 0x01ff000101000101ull, 0x01ff0001010100ffull, 0x01ff01ffffffffffull,
    0x01ff01ffffffff01ull, 0x01ff01ffffff01ffull, 0x01ff01ffffff0101ull, 0x01ff01ffff000000ull,
    0x01ff01ffff01ffffull, 0x01ff01ffff01ff01ull, 0x01ff01ffff0101ffull, 0x01ff01ffff010101ull,
    0x01ff01ff00ffff00ull, 0x01ff01ff00ff0000ull, 0x01ff01ff0000ff00ull, 0x01ff01ff000000ffull,
    0x01ff01ff00000100ull, 0x01ff01ff00010000ull, 0x01ff01ff00010100ull, 0x01ff01ff01ffffffull,
    0x01ff01ff01ffff01ull, 0x01ff01ff01ff01ffull, 0x01ff01ff01ff0101ull, 0x01ff01ff01000000ull,
    0x01ff01ff0101ffffull, 0x01ff01ff0101ff01ull, 0x01ff01ff010101ffull, 0x01ff01ff01010101ull,
    0x01ff0100ffff0000ull, 0x01ff0100ffff0001ull, 0x01ff0100ff00ff00ull, 0x01ff0100ff0000ffull,
    0x01ff0100ff000001ull, 0x01ff0100ff010000ull, 0x01ff010000ffff00ull, 0x01ff010000ff00ffull,
    0x01ff010000ff0001ull, 0x01ff010000ff0100ull, 0x01ff01000000ffffull, 0x01ff01000000ff01ull,
    0x01ff010000000000ull, 0x01ff010000000101ull, 0x01ff01000001ff00ull, 0x01ff0100000100ffull,
    0x01ff010001ff0000ull, 0x01ff010001000001ull, 0x01ff010001000100ull, 0x01ff010001010000ull,
    0x01ff0101ffffffffull, 0x01ff0101ffffff01ull, 0x01ff0101ffff01ffull, 0x01ff0101ffff0101ull,
    0x01ff0101ff000000ull, 0x01ff0101ff01ffffull, 0x01ff0101ff01ff01ull, 0x01ff0101ff0101ffull,
    0x01ff0101ff010101ull, 0x01ff010100ff0000ull, 0x01ff01010000ff00ull, 0x01ff0101000000ffull,
    0x01ff010100000001ull, 0x01ff010101ffffffull, 0x01ff010101ffff01ull, 0x01ff010101ff01ffull,
    0x01ff010101ff0101ull, 0x01ff010101000000ull, 0x01ff01010101ffffull, 0x01ff01010101ff01ull,
    0x01ff0101010101ffull, 0x01ff010101010101ull, 0x0100ffffffff0000ull, 0x0100ffffff00ff00ull,
    0x0100ffffff000001ull, 0x0100ffffff0001ffull, 0x0100ffffff000100ull, 0x0100ffffff010000ull,
    0x0100ffff00ffff00ull, 0x0100ffff00ff0001ull, 0x0100ffff00ff0100ull, 0x0100ffff00000000ull,
    0x0100ffff000001ffull, 0x0100ffff00000101ull, 0x0100ffff00010100ull, 0x0100ffff00010101ull,
    0x0100ffff01ff0000ull, 0x0100ffff0100ff00ull, 0x0100ffff010000ffull, 0x0100ffff01000001ull,
    0x0100ffff01000100ull, 0x0100ffff01010000ull, 0x0100ff00ffffff00ull, 0x0100ff00ffff00ffull,
    0x0100ff00ffff0001ull, 0x0100ff00ffff0100ull, 0x0100ff00ff00ffffull, 0x0100ff00ff000000ull,
    0x0100ff00ff0001ffull, 0x0100ff00ff000101ull, 0x0100ff00ff01ff00ull, 0x0100ff00ff0100ffull,
    0x0100ff00ff010001ull, 0x0100ff00ff010100ull, 0x0100ff0000ffffffull, 0x0100ff0000ff0000ull,
    0x0100ff000000ffffull, 0x0100ff000000ff00ull, 0x0100ff00000000ffull, 0x0100ff0000000000ull,
    0x0100ff0000000001ull, 0x0100ff0000000100ull, 0x0100ff000001ff01ull, 0x0100ff0000010000ull,
    0x0100ff0001ff00ffull, 0x0100ff0001ff0001ull, 0x0100ff000100ff01ull, 0x0100ff0001000000ull,
    0x0100ff00010001ffull, 0x0100ff000101ff00ull, 0x0100ff00010100ffull, 0x0100ff0001010001ull,
    0x0100ff0001010100ull, 0x0100ff01ffff0000ull, 0x0100ff01ff00ff00ull, 0x0100ff01ff0000ffull,
    0x0100ff01ff000100ull, 0x0100ff01ff010000ull, 0x0100ff0100ff00ffull, 0x0100ff0100ff0001ull,
    0x0100ff0100ff0100ull, 0x0100ff010000ffffull, 0x0100ff010000ff01ull, 0x0100ff0100000000ull,
    0x0100ff01000001ffull, 0x0100ff0100010001ull, 0x0100ff0100010100ull, 0x0100ff0101ff0000ull,
    0x0100ff01010000ffull, 0x0100ff0101000001ull, 0x0100ff0101010100ull, 0x010000ffffffff00ull,
    0x010000ffffff00ffull, 0x010000ffffff0001ull, 0x010000ffff00ffffull, 0x010000ffff000000ull,
    0x010000ffff0001ffull, 0x010000ffff010001ull, 0x010000ff00ffffffull, 0x010000ff00ff0101ull,
    0x010000ff0000ff00ull, 0x010000ff000000ffull, 0x010000ff00000000ull, 0x010000ff00000001ull,
    0x010000ff000001ffull, 0x010000ff00000100ull, 0x010000ff0001ffffull, 0x010000ff0001ff00ull,
    0x010000ff0001ff01ull, 0x010000ff00010000ull, 0x010000ff01ff00ffull, 0x010000ff01ff0001ull,
    0x010000ff0100ff01ull, 0x010000ff010000ffull, 0x010000ff01000000ull, 0x010000ff010001ffull,
    0x010000ff0101ff00ull, 0x010000ff01010100ull, 0x01000000ffffffffull, 0x01000000ffff0000ull,
    0x01000000ffff01ffull, 0x01000000ffff0101ull, 0x01000000ff00ffffull, 0x01000000ff00ff00ull,
    0x01000000ff0000ffull, 0x01000000ff000000ull, 0x01000000ff000001ull, 0x01000000ff000100ull,
    0x01000000ff01ff00ull, 0x01000000ff010000ull, 0x01000000ff010100ull, 0x01000000ff010101ull,
    0x0100000000ffff00ull, 0x0100000000ff00ffull, 0x0100000000ff0000ull, 0x0100000000ff0001ull,
    0x0100000000ff0100ull, 0x010000000000ffffull, 0x010000000000ff00ull, 0x010000000000ff01ull,
    0x01000000000000ffull, 0x0100000000000000ull, 0x0100000000000001ull, 0x01000000000001ffull,
    0x0100000000000100ull, 0x0100000000000101ull, 0x010000000001ff00ull, 0x01000000000100ffull,
    0x0100000000010000ull, 0x0100000000010001ull, 0x0100000000010100ull, 0x0100000001ffff00ull,
    0x0100000001ff0000ull, 0x0100000001ff01ffull, 0x010000000100ff00ull, 0x010000000100ff01ull,
    0x01000000010000ffull, 0x0100000001000000ull, 0x0100000001000001ull, 0x0100000001000100ull,
    0x0100000001000101ull, 0x010000000101ffffull, 0x010000000101ff01ull, 0x0100000001010000ull,
    0x01000000010101ffull, 0x0100000001010101ull, 0x01000001ffffff00ull, 0x01000001ffff00ffull,
    0x01000001ff00ffffull, 0x01000001ff000000ull, 0x01000001ff000100ull, 0x01000001ff01ffffull,
    0x01000001ff010001ull, 0x01000001ff010100ull, 0x0100000100ff0000ull, 0x0100000100ff01ffull,
    0x0100000100ff0100ull, 0x010000010000ff00ull, 0x010000010000ff01ull, 0x0100000100000000ull,
    0x0100000100000001ull, 0x0100000100000100ull, 0x0100000100010000ull, 0x01000001000101ffull,
    0x0100000101ffff01ull, 0x0100000101ff00ffull, 0x0100000101ff0100ull, 0x0100000101ff0101ull,
    0x010000010100ff01ull, 0x01000001010000ffull, 0x0100000101000000ull, 0x01000001010100ffull,
    0x0100000101010001ull, 0x0100000101010100ull, 0x010001ffffff0000ull, 0x010001ffff000001ull,
    0x010001ffff000100ull, 0x010001ffff010000ull, 0x010001ff00ffff00ull, 0x010001ff00ff0001ull,
    0x010001ff0000ffffull, 0x010001ff0000ff01ull, 0x010001ff00000000ull, 0x010001ff00000001ull,
    0x010001ff00000101ull, 0x010001ff000100ffull, 0x010001ff00010000ull, 0x010001ff01ff0000ull,
    0x010001ff0100ff00ull, 0x010001ff01000001ull, 0x010001ff01000100ull, 0x010001ff01010000ull,
    0x01000100ffff00ffull, 0x01000100ffff0001ull, 0x01000100ffff0100ull, 0x01000100ff00ffffull,
    0x01000100ff00ff01ull, 0x01000100ff000000ull, 0x01000100ff0001ffull, 0x01000100ff000101ull,
    0x01000100ff01ffffull, 0x01000100ff01ff00ull, 0x01000100ff0100ffull, 0x01000100ff010001ull,
    0x0100010000ffffffull, 0x0100010000ffff01ull, 0x0100010000ff0000ull, 0x0100010000ff01ffull,
    0x0100010000ff0101ull, 0x010001000000ff00ull, 0x01000100000000ffull, 0x0100010000000000ull,
    0x0100010000000001ull, 0x0100010000000100ull, 0x010001000001ff01ull, 0x0100010000010000ull,
    0x0100010000010001ull, 0x0100010000010101ull, 0x0100010001ffff00ull, 0x0100010001ff00ffull,
    0x010001000100ffffull, 0x010001000100ff01ull, 0x0100010001000000ull, 0x0100010001000101ull,
    0x010001000101ff00ull, 0x0100010001010001ull, 0x01000101ffff0000ull, 0x01000101ff000000ull,
    0x01000101ff010000ull, 0x0100010100ff00ffull, 0x0100010100ff0001ull, 0x0100010100ff0100ull,
    0x010001010000ffffull, 0x0100010100000000ull, 0x01000101000001ffull, 0x010001010001ff00ull,
    0x0100010101ff0000ull, 0x010001010100ff00ull, 0x01000101010000ffull, 0x0100010101000000ull,
    0x0100010101000001ull, 0x0101ffffffffffffull, 0x0101ffffffffff01ull, 0x0101ffffffff01ffull,
    0x0101ffffffff0101ull, 0x0101ffffff000000ull, 0x0101ffffff01ffffull, 0x0101ffffff01ff01ull,
    0x0101ffffff0101ffull, 0x0101ffffff010101ull, 0x0101ffff00ff0000ull, 0x0101ffff0000ff00ull,
    0x0101ffff000000ffull, 0x0101ffff00000001ull, 0x0101ffff00000100ull, 0x0101ffff01ffffffull,
    0x0101ffff01ffff01ull, 0x0101ffff01ff01ffull, 0x0101ffff01ff0101ull, 0x0101ffff01000000ull,
    0x0101ffff0101ffffull, 0x0101ffff0101ff01ull, 0x0101ffff010101ffull, 0x0101ffff01010101ull,
    0x0101ff00ffff0000ull, 0x0101ff00ffff0100ull, 0x0101ff00ff00ff00ull, 0x0101ff00ff0000ffull,
    0x0101ff00ff000001ull, 0x0101ff00ff000100ull, 0x0101ff00ff000101ull, 0x0101ff0000ff0001ull,
    0x0101ff0000ff0100ull, 0x0101ff000000ff00ull, 0x0101ff0000000000ull, 0x0101ff00000001ffull,
    0x0101ff0000000101ull, 0x0101ff000001ff00ull, 0x0101ff00000100ffull, 0x0101ff0001ff0000ull,
    0x0101ff000100ffffull, 0x0101ff000100ff01ull, 0x0101ff0001000001ull, 0x0101ff0001000100ull,
    0x0101ff01ffffff01ull, 0x0101ff01ffff01ffull, 0x0101ff01ffff0101ull, 0x0101ff01ff00ffffull,
    0x0101ff01ff000100ull, 0x0101ff01ff01ff01ull, 0x0101ff01ff0101ffull, 0x0101ff01ff010101ull,
    0x0101ff0100ff0000ull, 0x0101ff010000ff00ull, 0x0101ff0100000001ull, 0x0101ff0100000100ull,
    0x0101ff0100010000ull, 0x0101ff0101ffffffull, 0x0101ff0101ffff01ull, 0x0101ff0101ff01ffull,
    0x0101ff0101ff0101ull, 0x0101ff0101000000ull, 0x0101ff010101ffffull, 0x0101ff010101ff01ull,
    0x0101ff01010101ffull, 0x0101ff0101010101ull, 0x010100ffff000100ull, 0x010100ffff010000ull,
    0x010100ff00ffff00ull, 0x010100ff00ff00ffull, 0x010100ff0000ffffull, 0x010100ff000000ffull,
    0x010100ff00000000ull, 0x010100ff000001ffull, 0x010100ff00000101ull, 0x010100ff0001ff00ull,
    0x010100ff00010000ull, 0x010100ff00010001ull, 0x010100ff000101ffull, 0x010100ff00010100ull,
    0x010100ff01ff0000ull, 0x01010000ffff0001ull, 0x01010000ffff0100ull, 0x01010000ff00ffffull,
    0x01010000ff00ff01ull, 0x01010000ff000000ull, 0x01010000ff0001ffull, 0x01010000ff010001ull,
    0x01010000ff010100ull, 0x0101000000ffff01ull, 0x0101000000ff0000ull, 0x010100000000ff00ull,
    0x01010000000000ffull, 0x0101000000000000ull, 0x0101000000000001ull, 0x0101000000000100ull,
    0x0101000000010000ull, 0x0101000000010101ull, 0x0101000001ffff00ull, 0x0101000001ff00ffull,
    0x0101000001ff0000ull, 0x0101000001ff0001ull, 0x0101000001ff0100ull, 0x010100000100ff01ull,
    0x0101000001000000ull, 0x01010000010001ffull, 0x01010001ffff0000ull, 0x01010001ff00ff00ull,
    0x01010001ff000001ull, 0x01010001ff000101ull, 0x01010001ff01ff00ull, 0x01010001ff010000ull,
    0x0101000100ff00ffull, 0x0101000100ff0001ull, 0x0101000100ff0101ull, 0x010100010000ff01ull,
    0x0101000100000000ull, 0x0101000100000001ull, 0x01010001000001ffull, 0x010100010001ffffull,
    0x010100010001ff01ull, 0x0101000101ff0001ull, 0x010100010100ffffull, 0x0101000101000000ull,
    0x0101000101000001ull, 0x0101000101000100ull, 0x010100010101ff00ull, 0x01010001010100ffull,
    0x0101000101010001ull, 0x010101ffffffffffull, 0x010101ffffffff01ull, 0x010101ffffff01ffull,
    0x010101ffffff0101ull, 0x010101ffff01ffffull, 0x010101ffff01ff01ull, 0x010101ffff0101ffull,
    0x010101ffff010101ull, 0x010101ff0000ff00ull, 0x010101ff000000ffull, 0x010101ff00000001ull,
    0x010101ff00000100ull, 0x010101ff01ffffffull, 0x010101ff01ffff01ull, 0x010101ff01ff01ffull,
    0x010101ff01ff0101ull, 0x010101ff01000000ull, 0x010101ff0101ffffull, 0x010101ff0101ff01ull,
    0x010101ff010101ffull, 0x010101ff01010101ull, 0x01010100ffff0000ull, 0x01010100ff0000ffull,
    0x01010100ff000100ull, 0x01010100ff01ff00ull, 0x01010100ff010000ull, 0x0101010000ffff00ull,
    0x010101000000ffffull, 0x0101010000000000ull, 0x0101010000000101ull, 0x010101000001ff00ull,
    0x0101010000010001ull, 0x0101010000010100ull, 0x010101000100ffffull, 0x0101010001000001ull,
    0x01010101ffffffffull, 0x01010101ffffff01ull, 0x01010101ffff01ffull, 0x01010101ffff0101ull,
    0x01010101ff01ffffull, 0x01010101ff01ff01ull, 0x01010101ff0101ffull, 0x01010101ff010101ull,
    0x010101010000ff00ull, 0x01010101000000ffull, 0x0101010100000001ull, 0x0101010101ffffffull,
    0x0101010101ffff01ull, 0x0101010101ff01ffull, 0x0101010101ff0101ull, 0x0101010101000000ull,
    0x010101010101ffffull, 0x010101010101ff01ull, 0x01010101010101ffull, 0x0101010101010101ull,
};

/* The 4-bit non-linear codebook, indexed by a nibble.
 * Transcribed from `IQ4NL_VALUES` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) signed char SHISO_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38,
    53, 69, 89, 113,
};

/* Sign patterns for the 2-bit formats, indexed by the sign-pattern word.
 * Transcribed from `KSIGNS_IQ2XS` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) unsigned char SHISO_KSIGNS_IQ2XS[128] = {
    0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139,
    12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23,
    24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163,
    36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175,
    48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187,
    60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71,
    72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83,
    212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95,
    96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235,
    108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119,
    120, 249, 250, 123, 252, 125, 126, 255,
};

/* Sign patterns as full-byte masks: 0xff per negative lane, packed into a word.
 * Transcribed from `KSIGNS64` in `crates/shiso-modelfile/src/quant_tables.rs`;
 * `kernel_iq_table_copy` below is what grades the transcription on the device. */
static const __device__ __align__(8) uint64_t SHISO_KSIGNS64[128] = {
    0x0000000000000000ull, 0xff000000000000ffull, 0xff0000000000ff00ull, 0x000000000000ffffull,
    0xff00000000ff0000ull, 0x0000000000ff00ffull, 0x0000000000ffff00ull, 0xff00000000ffffffull,
    0xff000000ff000000ull, 0x00000000ff0000ffull, 0x00000000ff00ff00ull, 0xff000000ff00ffffull,
    0x00000000ffff0000ull, 0xff000000ffff00ffull, 0xff000000ffffff00ull, 0x00000000ffffffffull,
    0xff0000ff00000000ull, 0x000000ff000000ffull, 0x000000ff0000ff00ull, 0xff0000ff0000ffffull,
    0x000000ff00ff0000ull, 0xff0000ff00ff00ffull, 0xff0000ff00ffff00ull, 0x000000ff00ffffffull,
    0x000000ffff000000ull, 0xff0000ffff0000ffull, 0xff0000ffff00ff00ull, 0x000000ffff00ffffull,
    0xff0000ffffff0000ull, 0x000000ffffff00ffull, 0x000000ffffffff00ull, 0xff0000ffffffffffull,
    0xff00ff0000000000ull, 0x0000ff00000000ffull, 0x0000ff000000ff00ull, 0xff00ff000000ffffull,
    0x0000ff0000ff0000ull, 0xff00ff0000ff00ffull, 0xff00ff0000ffff00ull, 0x0000ff0000ffffffull,
    0x0000ff00ff000000ull, 0xff00ff00ff0000ffull, 0xff00ff00ff00ff00ull, 0x0000ff00ff00ffffull,
    0xff00ff00ffff0000ull, 0x0000ff00ffff00ffull, 0x0000ff00ffffff00ull, 0xff00ff00ffffffffull,
    0x0000ffff00000000ull, 0xff00ffff000000ffull, 0xff00ffff0000ff00ull, 0x0000ffff0000ffffull,
    0xff00ffff00ff0000ull, 0x0000ffff00ff00ffull, 0x0000ffff00ffff00ull, 0xff00ffff00ffffffull,
    0xff00ffffff000000ull, 0x0000ffffff0000ffull, 0x0000ffffff00ff00ull, 0xff00ffffff00ffffull,
    0x0000ffffffff0000ull, 0xff00ffffffff00ffull, 0xff00ffffffffff00ull, 0x0000ffffffffffffull,
    0xffff000000000000ull, 0x00ff0000000000ffull, 0x00ff00000000ff00ull, 0xffff00000000ffffull,
    0x00ff000000ff0000ull, 0xffff000000ff00ffull, 0xffff000000ffff00ull, 0x00ff000000ffffffull,
    0x00ff0000ff000000ull, 0xffff0000ff0000ffull, 0xffff0000ff00ff00ull, 0x00ff0000ff00ffffull,
    0xffff0000ffff0000ull, 0x00ff0000ffff00ffull, 0x00ff0000ffffff00ull, 0xffff0000ffffffffull,
    0x00ff00ff00000000ull, 0xffff00ff000000ffull, 0xffff00ff0000ff00ull, 0x00ff00ff0000ffffull,
    0xffff00ff00ff0000ull, 0x00ff00ff00ff00ffull, 0x00ff00ff00ffff00ull, 0xffff00ff00ffffffull,
    0xffff00ffff000000ull, 0x00ff00ffff0000ffull, 0x00ff00ffff00ff00ull, 0xffff00ffff00ffffull,
    0x00ff00ffffff0000ull, 0xffff00ffffff00ffull, 0xffff00ffffffff00ull, 0x00ff00ffffffffffull,
    0x00ffff0000000000ull, 0xffffff00000000ffull, 0xffffff000000ff00ull, 0x00ffff000000ffffull,
    0xffffff0000ff0000ull, 0x00ffff0000ff00ffull, 0x00ffff0000ffff00ull, 0xffffff0000ffffffull,
    0xffffff00ff000000ull, 0x00ffff00ff0000ffull, 0x00ffff00ff00ff00ull, 0xffffff00ff00ffffull,
    0x00ffff00ffff0000ull, 0xffffff00ffff00ffull, 0xffffff00ffffff00ull, 0x00ffff00ffffffffull,
    0xffffffff00000000ull, 0x00ffffff000000ffull, 0x00ffffff0000ff00ull, 0xffffffff0000ffffull,
    0x00ffffff00ff0000ull, 0xffffffff00ff00ffull, 0xffffffff00ffff00ull, 0x00ffffff00ffffffull,
    0x00ffffffff000000ull, 0xffffffffff0000ffull, 0xffffffffff00ff00ull, 0x00ffffffff00ffffull,
    0xffffffffffff0000ull, 0x00ffffffffff00ffull, 0x00ffffffffffff00ull, 0xffffffffffffffffull,
};


/* The grid byte `j` of a packed entry, as a float (the 2- and 3-bit grids). */
__device__ __forceinline__ float shiso_iq_byte(uint64_t grid, int j) {
    return (float) ((grid >> (8 * j)) & 0xFFu);
}

/* The same for the ternary grid, whose bytes are signed. */
__device__ __forceinline__ float shiso_iq_sbyte(uint64_t grid, int j) {
    return (float) (signed char) ((grid >> (8 * j)) & 0xFFu);
}

/* The sign lane `j` of a sign byte: the 2- and 3-bit formats negate the
 * grid byte where the byte's bit `j` is set. */
__device__ __forceinline__ float shiso_iq_sign(unsigned char signs, int j) {
    return (signs & (unsigned char) (1u << j)) ? -1.0f : 1.0f;
}

/* Sub-block `s` (32 weights) of a q2_K block as dequantized weights.
 *
 * q2_K stores its weights plane by plane, not in subblock order: the 16
 * weights of subblock `16*(8*plane + 2*j + h)` sit at byte `32*plane + 16*h`
 * of `qs` in bit field `2*j`, and take scale byte `8*plane + 2*j + h` as
 * `d * (sc & 0xF) * q - dmin * (sc >> 4)`. A 32-weight lane is one `j` of
 * one plane, so it reads 32 bytes at a fixed shift and two scale bytes. */
__device__ __forceinline__ void shiso_deq32_q2_K(
        const struct shiso_block_q2_K *xb, int s, float4 *qv, float2 *ab) {
    const int plane = s >> 2;
    const int shift = 2 * (s & 3);
    const unsigned char *q = xb->qs + plane * 32;
    const unsigned char *sc = xb->scales + plane * 8 + 2 * (s & 3);
    float *out = (float *) qv;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const float dl = __half2float(xb->d) * (float) (sc[h] & 0xF);
        const float ml = __half2float(xb->dmin) * (float) (sc[h] >> 4);
#pragma unroll
        for (int l = 0; l < 16; ++l) {
            out[16 * h + l] = dl * (float) ((q[h * 16 + l] >> shift) & 3) - ml;
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* Two index words then two sign-and-scale words per 32-weight group, as
 * four u16s; the group's scale is the top nibble of the second word. */
__device__ __forceinline__ void shiso_deq32_iq2_xxs(
        const struct shiso_block_iq2_xxs *xb, int s, float4 *qv, float2 *ab) {
    const unsigned aux0 = (unsigned) xb->qs[4 * s + 0] | ((unsigned) xb->qs[4 * s + 1] << 16);
    const unsigned aux1 = (unsigned) xb->qs[4 * s + 2] | ((unsigned) xb->qs[4 * s + 3] << 16);
    const float db = __half2float(xb->d) * (0.5f + (float) ((aux1 >> 28) & 0xF)) * 0.25f;
    float *out = (float *) qv;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint64_t grid = SHISO_IQ2XXS_GRID[(aux0 >> (8 * l)) & 0xFF];
        const unsigned char signs = SHISO_KSIGNS_IQ2XS[(aux1 >> (7 * l)) & 0x7F];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[8 * l + j] = db * shiso_iq_byte(grid, j) * shiso_iq_sign(signs, j);
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* Nine index bits into the grid, seven sign bits above them, two 4-bit
 * scales a byte for the group's two 16-weight halves. */
__device__ __forceinline__ void shiso_deq32_iq2_xs(
        const struct shiso_block_iq2_xs *xb, int s, float4 *qv, float2 *ab) {
    const float d = __half2float(xb->d);
    const unsigned char sc = xb->scales[s];
    const float db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    float *out = (float *) qv;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned short q = xb->qs[4 * s + l];
        const uint64_t grid = SHISO_IQ2XS_GRID[q & 511];
        const unsigned char signs = SHISO_KSIGNS_IQ2XS[q >> 9];
        const float db = (l < 2) ? db0 : db1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[8 * l + j] = db * shiso_iq_byte(grid, j) * shiso_iq_sign(signs, j);
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* The grid indices occupy the first half of `qs` and the per-weight signs
 * the second; two high index bits per quad come from `qh`. */
__device__ __forceinline__ void shiso_deq32_iq2_s(
        const struct shiso_block_iq2_s *xb, int s, float4 *qv, float2 *ab) {
    const float d = __half2float(xb->d);
    const unsigned char sc = xb->scales[s];
    const float db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
    const float db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
    const unsigned qh = xb->qh[s];
    float *out = (float *) qv;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned idx = (unsigned) xb->qs[4 * s + l] | ((qh << (8 - 2 * l)) & 0x300);
        const uint64_t grid = SHISO_IQ2S_GRID[idx];
        const unsigned char signs = xb->qs[SHISO_QK_K / 8 + 4 * s + l];
        const float db = (l < 2) ? db0 : db1;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[8 * l + j] = db * shiso_iq_byte(grid, j) * shiso_iq_sign(signs, j);
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* Sixty-four grid-index bytes, then one sign-and-scale word per group after
 * them. Each index covers four weights, so a quad reads two grid entries
 * and splits its eight weights between them. */
__device__ __forceinline__ void shiso_deq32_iq3_xxs(
        const struct shiso_block_iq3_xxs *xb, int s, float4 *qv, float2 *ab) {
    const unsigned char *a = xb->qs + SHISO_QK_K / 4 + 4 * s;
    const unsigned aux = (unsigned) a[0] | ((unsigned) a[1] << 8)
                       | ((unsigned) a[2] << 16) | ((unsigned) a[3] << 24);
    const float db = __half2float(xb->d) * (0.5f + (float) ((aux >> 28) & 0xF)) * 0.5f;
    float *out = (float *) qv;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned char signs = SHISO_KSIGNS_IQ2XS[(aux >> (7 * l)) & 0x7F];
        const unsigned g1 = SHISO_IQ3XXS_GRID[xb->qs[8 * s + 2 * l]];
        const unsigned g2 = SHISO_IQ3XXS_GRID[xb->qs[8 * s + 2 * l + 1]];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[8 * l + j] = db * (float) ((g1 >> (8 * j)) & 0xFF) * shiso_iq_sign(signs, j);
            out[8 * l + j + 4] = db * (float) ((g2 >> (8 * j)) & 0xFF)
                               * shiso_iq_sign(signs, j + 4);
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* Odd 4-bit scales (`1 + 2 * nibble`), one scale byte per two groups; the
 * high index bit of each of a group's eight indices is one bit of `qh`, and
 * the signs are one byte per quad. */
__device__ __forceinline__ void shiso_deq32_iq3_s(
        const struct shiso_block_iq3_s *xb, int s, float4 *qv, float2 *ab) {
    const float d = __half2float(xb->d);
    const unsigned char sc = xb->scales[s / 2];
    const float db = d * (1.0f + 2.0f * (float) ((s & 1) ? (sc >> 4) : (sc & 0xF)));
    const unsigned qh = xb->qh[s];
    float *out = (float *) qv;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned g1 = SHISO_IQ3S_GRID[(unsigned) xb->qs[8 * s + 2 * l]
                                            | ((qh << (8 - 2 * l)) & 256)];
        const unsigned g2 = SHISO_IQ3S_GRID[(unsigned) xb->qs[8 * s + 2 * l + 1]
                                            | ((qh << (7 - 2 * l)) & 256)];
        const unsigned char signs = xb->signs[4 * s + l];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            out[8 * l + j] = db * (float) ((g1 >> (8 * j)) & 0xFF) * shiso_iq_sign(signs, j);
            out[8 * l + j + 4] = db * (float) ((g2 >> (8 * j)) & 0xFF)
                               * shiso_iq_sign(signs, j + 4);
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* The group word carries the high three index bits of each quad (three bits
 * a quad from bit 0 up), a 3-bit scale multiplier at bits 12..14, and the
 * sign of the grid shift at bit 15: a weight is `dl * (grid + delta)`, and
 * skipping the delta is wrong, not merely imprecise. */
__device__ __forceinline__ void shiso_deq32_iq1_s(
        const struct shiso_block_iq1_s *xb, int s, float4 *qv, float2 *ab) {
    const unsigned short qh = xb->qh[s];
    const float dl = __half2float(xb->d) * (float) (2 * ((qh >> 12) & 7) + 1);
    const float delta = (qh & 0x8000) ? -0.125f : 0.125f;
    float *out = (float *) qv;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const unsigned idx = (unsigned) xb->qs[4 * s + l]
                           | ((unsigned) ((qh >> (3 * l)) & 7)) << 8;
        const uint64_t grid = SHISO_IQ1S_GRID[idx];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[8 * l + j] = dl * (shiso_iq_sbyte(grid, j) + delta);
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* The block f16 is not stored: it is the four top nibbles of the odd scale
 * bytes, least significant first. Each 16-weight subblock then carries its
 * own 3-bit scale multiplier, and each quad its own grid-shift sign, so the
 * lane assembles two scales and four deltas for its 32 weights. */
__device__ __forceinline__ void shiso_deq32_iq1_m(
        const struct shiso_block_iq1_m *xb, int s, float4 *qv, float2 *ab) {
    const unsigned w0 = (unsigned) xb->scales[0] | ((unsigned) xb->scales[1] << 8);
    const unsigned w1 = (unsigned) xb->scales[2] | ((unsigned) xb->scales[3] << 8);
    const unsigned w2 = (unsigned) xb->scales[4] | ((unsigned) xb->scales[5] << 8);
    const unsigned w3 = (unsigned) xb->scales[6] | ((unsigned) xb->scales[7] << 8);
    const unsigned short dbits = (unsigned short) ((w0 >> 12) | ((w1 >> 8) & 0x00F0)
                                                   | ((w2 >> 4) & 0x0F00) | (w3 & 0xF000));
    const float d = __half2float(__ushort_as_half(dbits));
    const unsigned words[4] = { w0, w1, w2, w3 };
    const unsigned word = words[s / 2];
    const int shift = 6 * (s % 2);
    const float dl0 = d * (float) (2 * ((word >> shift) & 7) + 1);
    const float dl1 = d * (float) (2 * ((word >> (shift + 3)) & 7) + 1);
    float *out = (float *) qv;
#pragma unroll
    for (int m = 0; m < 4; ++m) {
        const unsigned char qhb = xb->qh[2 * s + m / 2];
        const unsigned idx = (unsigned) xb->qs[4 * s + m]
                           | ((((unsigned) qhb) << ((m & 1) ? 4 : 8)) & 0x700);
        const float delta = (qhb & ((m & 1) ? 0x80 : 0x08)) ? -0.125f : 0.125f;
        const float dl = (m < 2) ? dl0 : dl1;
        const uint64_t grid = SHISO_IQ1S_GRID[idx];
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[8 * m + j] = dl * (shiso_iq_sbyte(grid, j) + delta);
        }
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

/* A 6-bit subblock scale split across two planes, offset so 32 means no
 * scaling; the weights are nibble indices into the non-linear codebook, low
 * nibble first for the group's first 16 weights. */
__device__ __forceinline__ void shiso_deq32_iq4_xs(
        const struct shiso_block_iq4_xs *xb, int s, float4 *qv, float2 *ab) {
    const unsigned ls = ((unsigned) (xb->scales_l[s / 2] >> (4 * (s % 2))) & 0xF)
                      | ((((unsigned) xb->scales_h) >> (2 * s)) & 3) << 4;
    const float dl = __half2float(xb->d) * ((float) ls - 32.0f);
    float *out = (float *) qv;
#pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned char q = xb->qs[16 * s + j];
        out[j] = dl * (float) SHISO_IQ4NL[q & 0xF];
        out[16 + j] = dl * (float) SHISO_IQ4NL[q >> 4];
    }
    ab->x = 1.0f;
    ab->y = 0.0f;
}

#endif /* SHISO_GENERIC_CUH */
