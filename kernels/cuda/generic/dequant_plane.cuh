/* Widening a whole quantized weight plane to f32, so that the vendor GEMM can
 * read it.
 *
 * # Why this exists
 *
 * `cublas` has no quantized operand type. Every weight the prefill GEMM is
 * handed is block-quantized - `q4_K` for the bulk of a `Q4_K_M` container,
 * `q4_0` for a Gemma 4 QAT export, `q8_0` for the recurrent gates - so routing
 * a prefill chunk through the vendor library is *dequantize, then multiply*,
 * and this is the first half of that. The measurement that says the round trip
 * is worth its own memory traffic is in `docs/handoff/cuda-decode-perf.md`
 * §17: the hand-written tile reaches 2.4 TFLOP/s on these shapes and the
 * vendor library reaches 25.9.
 *
 * # The output layout is the one the vendor library already wants
 *
 * A weight is stored row-major, `out_dim` rows of `in_dim` elements, and this
 * writes f32 in exactly that order. Read column-major - which is the only way
 * the vendor library reads anything - that same buffer is an `in_dim` by
 * `out_dim` matrix with a leading dimension of `in_dim`, which is the
 * transposed left operand of `dst = weight * src`. So no transpose, no
 * repacking and no second layout: the caller passes `Op::T` and a leading
 * dimension of `in_dim`.
 *
 * # Where it rounds, and why that is the same place as before
 *
 * Every element goes through the same `shiso_dequantize_*` helper the matvec
 * and the tiled GEMM call, so the dequantized *value* is bit-identical to the
 * one those two produce. What differs downstream is only that the tile then
 * narrows it to `half` before multiplying and this does not - which makes the
 * vendor route the more precise of the two, not the less.
 *
 * **That is a claim about the arithmetic and not about the plane, and reading
 * it as the latter is how a defect shipped.** Where an element *lands* is the
 * walk's business, not the dequantizer's, and the `f16` walk put it in the
 * wrong slot: the staged plane was not what the matvec and the tile read,
 * whatever each element's bits were. Both halves are now graded, the value by
 * `shiso_kerneltest::qwen35::gemv` and the placement by
 * `shiso_kerneltest::generic::plane`.
 */
#ifndef SHISO_DEQUANT_PLANE_CUH
#define SHISO_DEQUANT_PLANE_CUH

#include "generic.cuh"

/* `shiso_engine::launch::DequantPlaneArgs`, field for field. */
struct shiso_args_dequant_plane {
    int32_t  ne00;
    int32_t  ne01;
    uint64_t nb01;
};
static_assert(sizeof(struct shiso_args_dequant_plane) == 16,
              "DequantPlaneArgs is 16 bytes on the host and must be here too");

/* Threads one row's block runs. */
#define SHISO_DEQUANT_BLOCK 256

/* Four consecutive elements of one quantized block, `il` indexing the
 * four-element chunks within it. One specialization per codec; the walk over
 * the plane is the same for all of them. */
template <typename codec>
__device__ __forceinline__ void shiso_dequant_plane_t4(
        const codec *xb, int il, float4 *reg);

/* Chunks of four in one block of this codec. A plain `constexpr` per
 * specialization rather than a field of the block struct, because the block
 * structs are the loader's layout and carry no arithmetic. */
template <typename codec>
__device__ __forceinline__ constexpr int shiso_dequant_plane_chunks();

template <>
__device__ __forceinline__ void shiso_dequant_plane_t4<struct shiso_block_q4_K>(
        const struct shiso_block_q4_K *xb, int il, float4 *reg) {
    shiso_dequantize_q4_K_t4(xb, il, reg);
}
template <>
__device__ __forceinline__ constexpr int
shiso_dequant_plane_chunks<struct shiso_block_q4_K>() {
    return SHISO_QK_K / 4;
}

/* q6_K, the widest super-block this tree reads. Its 256 elements are the same
 * chunk count as the other two `_K` formats.
 *
 * Here for the routed expert stages: `unsloth`'s "UD" containers are DYNAMIC -
 * they vary the codec per layer - and `UD-Q4_K_XL` stores three of its
 * forty-one `ffn_down_exps` in q6_K where the other thirty-eight are q5_K. A
 * build with entries for only the common codec loads thirty-four layers and
 * then refuses. */
template <>
__device__ __forceinline__ void shiso_dequant_plane_t4<struct shiso_block_q6_K>(
        const struct shiso_block_q6_K *xb, int il, float4 *reg) {
    shiso_dequantize_q6_K_t4(xb, il, reg);
}
template <>
__device__ __forceinline__ constexpr int
shiso_dequant_plane_chunks<struct shiso_block_q6_K>() {
    return SHISO_QK_K / 4;
}

/* q5_K, whose fifth-bit plane is the only thing that makes it not q4_K. Its
 * super-block is the same 256 elements, so the walk's chunk count is too.
 *
 * Added for the routed expert stages, which are the first caller in this tree
 * to read a `Q5_K` weight through this dispatch: the `UD-Q4_K_XL` container
 * stores every expert's `down` projection in it and the gate and up in q4_K, so
 * a routed block of that file reads two codecs and cannot be served by one. */
template <>
__device__ __forceinline__ void shiso_dequant_plane_t4<struct shiso_block_q5_K>(
        const struct shiso_block_q5_K *xb, int il, float4 *reg) {
    shiso_dequantize_q5_K_t4(xb, il, reg);
}
template <>
__device__ __forceinline__ constexpr int
shiso_dequant_plane_chunks<struct shiso_block_q5_K>() {
    return SHISO_QK_K / 4;
}

template <>
__device__ __forceinline__ void shiso_dequant_plane_t4<struct shiso_block_q4_0>(
        const struct shiso_block_q4_0 *xb, int il, float4 *reg) {
    shiso_dequantize_q4_0_t4(xb, il, reg);
}
template <>
__device__ __forceinline__ constexpr int
shiso_dequant_plane_chunks<struct shiso_block_q4_0>() {
    return SHISO_QK4_0 / 4;
}

/* q8_0 has no `_t4` of its own: its sixteen-element helper is the one the
 * tiled GEMM calls, and four elements of it are the four this reads back. The
 * arithmetic is one multiply per element either way, so building sixteen to
 * keep four would be four times the work rather than four times the loads. */
template <>
__device__ __forceinline__ void shiso_dequant_plane_t4<struct shiso_block_q8_0>(
        const struct shiso_block_q8_0 *xb, int il, float4 *reg) {
    const float d = __half2float(xb->d);
    reg->x = (float) xb->qs[4 * il + 0] * d;
    reg->y = (float) xb->qs[4 * il + 1] * d;
    reg->z = (float) xb->qs[4 * il + 2] * d;
    reg->w = (float) xb->qs[4 * il + 3] * d;
}
template <>
__device__ __forceinline__ constexpr int
shiso_dequant_plane_chunks<struct shiso_block_q8_0>() {
    return SHISO_QK8_0 / 4;
}

/* f16 is not quantized and has no block of its own, so this declares one: four
 * consecutive halves, which is the one chunk the walk below reads per step.
 *
 * **This struct is the fix for a shipped defect, and it is spelled as a struct
 * rather than as an exception in the walk on purpose.** The codec was `__half`
 * itself, with `shiso_dequant_plane_chunks<__half>()` returning 1 - and the two
 * halves of that pair disagreed. The walk's `blocks + c / per_block` steps one
 * `codec` per chunk, which for a bare `__half` is one *element* and not four,
 * while the destination advanced four; so chunk `c` read source elements
 * `c .. c+3` and wrote `4c .. 4c+3`, and the staged plane was correct only at
 * `c == 0`. Gemma 4 greedy parity went from 64/64 to 0/64 the day this route
 * arrived, and no case anywhere in the tree graded a staged plane.
 *
 * The walk was never wrong: it assumed its template parameter was a block
 * struct holding `4 * shiso_dequant_plane_chunks<codec>()` elements, which the
 * other three codecs are and this one was not. Making `f16` a block restores
 * that invariant for every codec at once, which is what makes this a class of
 * bug removed rather than an instance patched - a future codec whose type is a
 * bare element cannot reintroduce it without declaring its block first.
 *
 * It is here at all because `launch::mm_name` answers for `f16`, and an operand
 * type the route can name but not stage would be a prefill that refuses at its
 * first chunk. */
struct shiso_block_f16 {
    __half qs[4];
};
static_assert(sizeof(struct shiso_block_f16) == 8, "an f16 chunk is four halves");

template <>
__device__ __forceinline__ void shiso_dequant_plane_t4<struct shiso_block_f16>(
        const struct shiso_block_f16 *xb, int il, float4 *reg) {
    reg->x = __half2float(xb->qs[4 * il + 0]);
    reg->y = __half2float(xb->qs[4 * il + 1]);
    reg->z = __half2float(xb->qs[4 * il + 2]);
    reg->w = __half2float(xb->qs[4 * il + 3]);
}
template <>
__device__ __forceinline__ constexpr int
shiso_dequant_plane_chunks<struct shiso_block_f16>() {
    return 1;
}

/* One row per block, the row's four-element chunks split across its threads.
 *
 * **`codec` must be a struct holding exactly `4 *
 * shiso_dequant_plane_chunks<codec>()` elements.** That is the whole of what
 * the two lines below assume: `blocks + c / per_block` steps one such struct
 * per `per_block` chunks, and `c % per_block` picks the chunk inside it. A
 * `codec` that is a bare element rather than a block breaks the walk and not
 * the dequantizer, which is a plane made of the right numbers in the wrong
 * order - and `shiso_block_f16` above exists because that shipped once.
 * `shiso_kerneltest::generic::plane` grades all four codecs against the
 * staged plane itself, which is the only comparison that can see it.
 *
 * The grid is sized from `ne01` and the loop is bounded by `ne00 / 4`, so
 * neither axis carries a tail guard - and what makes that safe is two
 * host-side checks rather than one, because the four codecs' blocks do not
 * all constrain the width.
 *
 *  - `q4_0` and `q8_0` pack 32 elements and `q4_K` packs 256, so a row of any
 *    of the three is a whole number of four-element chunks by construction.
 *    `projection::row_bytes` is what refuses a width that is not a whole
 *    number of blocks, and it runs before this launch is built. (It is NOT
 *    `projection::dispatchable`, which this comment used to name: that one
 *    only decides whether a matvec exists at a width.)
 *  - `shiso_block_f16` is a block this file declares rather than one the
 *    format has, so nothing about `f16` constrains `ne00` at all.
 *    `projection::blas_projection` refuses an `in_dim` that is not a multiple
 *    of four for exactly this reason, and takes the tile instead.
 *
 * A width that slipped past both would leave the row's last one to three
 * elements holding whatever the previous tensor left in the slab - which is
 * not cleared on growth - and the multiply would read it. Silent garbage, not
 * a fault, which is the whole reason the host checks. */
template <typename codec>
__device__ __forceinline__ void shiso_dequant_plane_body(
        const struct shiso_args_dequant_plane &args,
        const char *__restrict__ src,
        float *__restrict__ dst) {
    const int row = (int) blockIdx.x;
    if (row >= args.ne01) {
        return;
    }
    const codec *blocks = (const codec *) (src + args.nb01 * (uint64_t) row);
    float *out = dst + (uint64_t) args.ne00 * (uint64_t) row;
    const int chunks = args.ne00 / 4;
    constexpr int per_block = shiso_dequant_plane_chunks<codec>();
    for (int c = (int) threadIdx.x; c < chunks; c += SHISO_DEQUANT_BLOCK) {
        float4 v;
        shiso_dequant_plane_t4<codec>(blocks + c / per_block, c % per_block, &v);
        out[4 * c + 0] = v.x;
        out[4 * c + 1] = v.y;
        out[4 * c + 2] = v.z;
        out[4 * c + 3] = v.w;
    }
}

/* One entry per codec, wrapping the template so the host can name it.
 *
 * `extern "C"` for the reason every entry in this tree is: the engine writes
 * these names down as `&'static str` literals and the driver resolves them
 * verbatim, so a mangled instantiation would be a name nothing could ask for. */
#define SHISO_DEQUANT_PLANE(suffix, codec)                                     \
    extern "C" __global__ void kernel_dequant_plane_##suffix##_f32(            \
            const struct shiso_args_dequant_plane args,                        \
            const char *__restrict__ src,                                      \
            char *__restrict__ dst) {                                          \
        shiso_dequant_plane_body<codec>(args, src, (float *) dst);             \
    }

#endif /* SHISO_DEQUANT_PLANE_CUH */
