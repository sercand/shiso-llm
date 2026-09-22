// Shared geometry, the argument block and the q4-affine matrix tile for the
// qwen35moe kernel set.
//
// This file is concatenated ahead of every other qwen35moe entry (see `SOURCES`
// in `src/compile.rs`), so the files after it may not `#include` it: the library
// is compiled from one source string through `newLibraryWithSource`, there is no
// include path, and a quote-include of a sibling would not resolve. The same
// rule the qwen35, vision and dense headers already live under.
//
// Two things here are not guessable and cost an hour each to rediscover:
//
//   * `uint4b_format` is declared in <metal_packed_numeric>, NOT in
//     <metal_tensor>. Without that include the type is undeclared and every
//     diagnostic points at the tensor constructor instead of at the include.
//   * a tensor's `data_handle_type` for a *format* element type is
//     `device uchar *`. The raw `uchar *` is what the constructor takes; a
//     `reinterpret_cast<device uint4b_format *>(ptr)` fails with "no matching
//     constructor".
//
// No `#pragma METAL fp math_mode` here. Fast math is the library's build
// default and this file is written against it.

#include <metal_stdlib>
#include <metal_tensor>
#include <metal_packed_numeric>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace metal;
using namespace mpp::tensor_ops;

// Codes per quantization group; the affine codec's only group size here.
constant constexpr uint Q35MOE_GROUP = 64;

// Outputs one storage tile of an affine slab holds -- the host packer's
// `storage_n`, and what a caller passes `q35moe_affine_tile` as `StorageN` when
// the slab it reads is a real projection rather than the probe's single tile.
//
// It is a constant rather than a template parameter on the expert entries
// because the *resident* layout is one decision for the whole kernel set: the
// loader packs a projection once and every reader of that slab has to agree
// with it. `pack_q4_affine_tiled` in `crates/shiso-kerneltest/src/qwen35moe/
// q4affine.rs` enforces `out_dim % storage_n == 0` for the same reason the
// entries below assume it: the last tile is a whole tile, and a projection that
// would need a short one is padded by the loader rather than special-cased
// here. The two widths this set runs are 2,048 (hidden) and 512 (the expert
// intermediate); both are whole multiples of this.
constant constexpr uint Q35MOE_STORAGE_N = 256;

// Output rows one expert threadgroup covers, one per simdgroup.
//
// `SHISO_Q35MOE_EXPERT_WARPS` in `kernels/cuda/qwen35moe/experts.cuh` and
// `EXPERT_ROWS` in `crates/shiso-engine/src/moe.rs`, which is what sizes the
// grid: `grid.x` is `ceil(out_dim / 4)` and the block is 128 threads, so a
// threadgroup is exactly four simdgroups and a simdgroup is exactly one output
// row. Change this and all three have to change together.
constant constexpr uint Q35MOE_EXPERT_ROWS = 4;

// The widths one expert stage is launched over. Sixteen bytes, at
// `[[buffer(0)]]`.
struct q35moe_expert_args {
    int in_dim;
    int out_dim;
    int n_expert;
    int n_used;
};

// Staging slots one block-wide reduction needs: 32 simdgroup partials plus the
// broadcast slot the whole threadgroup reads the answer out of. The same shape
// `q35_common.h` uses, and the same the CUDA reference's `__shared__ float
// s[32]` plus its `total` slot uses.
#define Q35MOE_BLOCK_SLOTS 33u

// The three helpers below are the Metal spelling of `q4e_block_sum`,
// `q4e_block_max` and `q4e_block_argmax` in `kernels/cuda/q4e/q4e.cuh`, which
// the router is ported from. Shape, exactly: reduce inside the simdgroup, lane
// 0 publishes to `scratch[simdgroup]`, barrier, simdgroup 0 re-reduces the
// slots below `n_simd`, thread 0 broadcasts through the last slot.
//
// `scratch` is passed in rather than declared here because the shading language
// only allows a threadgroup array to be declared inside a kernel, so an inlined
// helper has none of its own to keep -- the same reason `q35_block_sum` takes
// one.
//
// Bare `simd_sum` over the whole threadgroup is not what these do, and not only
// because it does not exist: `metal/qwen35/KERNELS.md` fixes the reduction
// SHAPE, because a port is graded against the reference's summation order and
// not against a mathematically equal one. The sum's simdgroup stage is
// therefore the explicit `simd_shuffle_down` fold `warp_sum_f32` is, in
// decreasing-offset order, rather than the builtin whose order the language
// leaves unspecified. The max's stage is `simd_max`, because `fmax` is
// associative and exact and no order of it can produce a different float.

/// Inclusive sum over one simdgroup, folded in decreasing-offset order so that
/// lane 0 holds the total. Only lane 0's result is read, which is what makes
/// the out-of-range shuffle in the last steps harmless -- the same property the
/// CUDA `__shfl_down_sync` fold relies on.
static inline float q35moe_simd_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

/// Block-wide sum over at most 32 simdgroups, broadcast to every thread.
///
/// `scratch` needs `Q35MOE_BLOCK_SLOTS` floats.
static inline float q35moe_block_sum(
        float v,
        threadgroup float *scratch,
        uint tid,
        uint nthreads) {
    const uint lane = tid & 31u;
    const uint simd = tid >> 5u;
    v = q35moe_simd_sum(v);
    if (lane == 0u) {
        scratch[simd] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_simd = (nthreads + 31u) >> 5u;
    v = (tid < n_simd) ? scratch[tid] : 0.0f;
    if (simd == 0u) {
        v = q35moe_simd_sum(v);
    }
    if (tid == 0u) {
        scratch[Q35MOE_BLOCK_SLOTS - 1u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[Q35MOE_BLOCK_SLOTS - 1u];
}

/// Block-wide maximum over at most 32 simdgroups, broadcast to every thread.
///
/// `scratch` needs `Q35MOE_BLOCK_SLOTS` floats.
static inline float q35moe_block_max(
        float v,
        threadgroup float *scratch,
        uint tid,
        uint nthreads) {
    const uint lane = tid & 31u;
    const uint simd = tid >> 5u;
    v = simd_max(v);
    if (lane == 0u) {
        scratch[simd] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_simd = (nthreads + 31u) >> 5u;
    v = (tid < n_simd) ? scratch[tid] : -INFINITY;
    if (simd == 0u) {
        v = simd_max(v);
    }
    if (tid == 0u) {
        scratch[Q35MOE_BLOCK_SLOTS - 1u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[Q35MOE_BLOCK_SLOTS - 1u];
}

/// Block-wide argmax, ties resolved toward the LOWER index, broadcast to every
/// thread.
///
/// The tie rule is load bearing, not a detail: it is what a stable descending
/// argsort does, and the router has to agree with the reference on which expert
/// a tie picks. A candidate replaces the incumbent when it is strictly larger,
/// or when it is equal and carries a valid, smaller id -- so a `-1` never wins
/// and an equal pair keeps the smaller expert.
///
/// There is no `simd_argmax`, so both stages are the explicit
/// `simd_shuffle_down` fold the CUDA body writes, step for step. `s_v` and
/// `s_i` each need `Q35MOE_BLOCK_SLOTS` elements; slot 0 doubles as the
/// broadcast slot, which is what the reference does too.
static inline void q35moe_block_argmax(
        float v,
        int i,
        thread float *out_v,
        thread int *out_i,
        threadgroup float *s_v,
        threadgroup int *s_i,
        uint tid,
        uint nthreads) {
    const uint lane = tid & 31u;
    const uint simd = tid >> 5u;
    for (uint d = 16u; d > 0u; d >>= 1u) {
        const float ov = simd_shuffle_down(v, d);
        const int oi = simd_shuffle_down(i, d);
        if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
            v = ov;
            i = oi;
        }
    }
    if (lane == 0u) {
        s_v[simd] = v;
        s_i[simd] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_simd = (nthreads + 31u) >> 5u;
    if (simd == 0u) {
        v = (lane < n_simd) ? s_v[lane] : -INFINITY;
        i = (lane < n_simd) ? s_i[lane] : -1;
        for (uint d = 16u; d > 0u; d >>= 1u) {
            const float ov = simd_shuffle_down(v, d);
            const int oi = simd_shuffle_down(i, d);
            if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
                v = ov;
                i = oi;
            }
        }
        if (lane == 0u) {
            s_v[0] = v;
            s_i[0] = i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *out_v = s_v[0];
    *out_i = s_i[0];
}

// Accumulates `Rows x TileN` of one q4-affine projection over the whole of
// `in_dim` and stores the result to `out`.
//
// The arithmetic is ported from `third_party/splash/runtime/metal/kernels/
// common/q4_mpp_tiles.h`. The matmul runs over the raw 4-bit *codes*, and the
// affine scale and bias are applied per quantization group afterwards:
//
//     sum_k a[m][k] * (code[n][k] * scale[g][n] + bias[g][n])
//       == scale[g][n] * sum_k a[m][k] * code[n][k]
//        + bias[g][n]  * sum_k a[m][k]
//
// which is why `row_sums` exists: the second term needs the plain row sum of
// the input over the same 64 lanes, and the matmul cannot produce it.
//
// Layout, and this is the convention the whole kernel set is built on: the
// packed slab is GROUP-MAJOR, `[quant_group][StorageN][GROUP]` nibbles, so the
// codes for one group of one output are 32 contiguous bytes and the codes for
// one output are NOT contiguous. `scales` and `biases` are one `bfloat` each
// per `(quant_group, output)` in the same group-major order. This is
// `q4_mpp_tiles.h`'s layout; it is not the row-major `[out][in]` a host packer
// reaches for first.
//
// `row_sums` is threadgroup scratch of at least `Rows` floats, owned by the
// caller. `weights` is the packed-code slab for ONE expert; the tensor operand
// takes the raw `device uchar *`.
template <uint Rows, uint TileN, uint Simdgroups, uint StorageN = TileN>
inline void q35moe_affine_tile(device const bfloat *input,
                               device const uchar *weights,
                               device const bfloat *scales,
                               device const bfloat *biases,
                               uint in_dim,
                               device float *out,
                               uint out_row_stride,
                               uint out_origin,
                               threadgroup float *row_sums,
                               uint simd_lane,
                               uint simd_group) {
    auto a = tensor(const_cast<device bfloat *>(input),
                    dextents<int, 2>{int(in_dim), int(Rows)},
                    array<int, 2>{1, int(in_dim)});
    constexpr auto descriptor =
        matmul2d_descriptor(Rows, TileN, Q35MOE_GROUP, false, true, false);
    matmul2d<descriptor, execution_simdgroups<Simdgroups>> operation;

    const uint groups = in_dim / Q35MOE_GROUP;
    const uint tile = out_origin / StorageN;
    const uint tile_offset = out_origin % StorageN;
    device const uchar *tile_weights =
        weights + ulong(tile) * groups * StorageN * Q35MOE_GROUP / 2;

    auto a0 = a.template slice<Q35MOE_GROUP, Rows>(0, 0);
    tensor<device uint4b_format, dextents<int, 2>, tensor_inline> b0(
        const_cast<device uchar *>(tile_weights) + tile_offset * (Q35MOE_GROUP / 2),
        dextents<int, 2>{int(Q35MOE_GROUP), int(TileN)},
        array<int, 2>{1, int(Q35MOE_GROUP)});
    auto bs0 = b0.template slice<Q35MOE_GROUP, TileN>(0, 0);
    auto acc = operation.template get_destination_cooperative_tensor<
        decltype(a0), decltype(bs0), float>();
    // Guarded like the epilogue and the store below, and like the reference's
    // `q4_visit`: an element outside the logical Rows x TileN destination is
    // never read back, so it is never written either.
    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (acc.is_valid_element(i)) {
            acc[i] = 0.0f;
        }
    }

    for (uint g = 0; g < groups; ++g) {
        // The previous group's epilogue read `row_sums`; this rewrites it.
        //
        // NOT A SHAPE TO INHERIT VERBATIM: this recomputes the row sums once
        // per quant group behind two threadgroup barriers, where the reference
        // amortises four groups per pass over a double-buffered scratch region
        // and pays two barriers per four groups. At in_dim 2,048 that is 32x
        // the barrier traffic. It is written this way because this file's job
        // is to settle the layout conventions and the simplest correct sweep
        // is the one to settle them against; a perf task rewrites it into
        // `q4_store_input_sums`'s rolling window and regrades against this
        // same case, which must stay bit-exact across the change.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = simd_group; row < Rows; row += Simdgroups) {
            const uint origin = row * in_dim + g * Q35MOE_GROUP + simd_lane;
            const float rs =
                simd_sum(float(input[origin]) + float(input[origin + 32]));
            if (simd_lane == 0) {
                row_sums[row] = rs;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        auto a_slice = a.template slice<Q35MOE_GROUP, Rows>(g * Q35MOE_GROUP, 0);
        tensor<device uint4b_format, dextents<int, 2>, tensor_inline> bg(
            const_cast<device uchar *>(tile_weights) +
                (ulong(g) * StorageN + tile_offset) * (Q35MOE_GROUP / 2),
            dextents<int, 2>{int(Q35MOE_GROUP), int(TileN)},
            array<int, 2>{1, int(Q35MOE_GROUP)});
        auto b_slice = bg.template slice<Q35MOE_GROUP, TileN>(0, 0);
        auto partial = operation.template get_destination_cooperative_tensor<
            decltype(a_slice), decltype(b_slice), float>();
        // Zeroed rather than left to `run` to define. Measured on this
        // machine: dropping these stores still grades bit-exact, so `run`
        // overwrites its destination rather than accumulating into it, which
        // is what lets `q4_mpp_tiles.h` leave its partials uninitialized. The
        // stores are kept because the guarantee is a measurement and not a
        // documented one, and 32 stores over a whole tile buy nothing back.
        for (ushort i = 0; i < partial.get_capacity(); ++i) {
            if (partial.is_valid_element(i)) {
                partial[i] = 0.0f;
            }
        }
        operation.run(a_slice, b_slice, partial);

        for (ushort i = 0; i < acc.get_capacity(); ++i) {
            if (!acc.is_valid_element(i)) {
                continue;
            }
            const auto index = acc.get_multidimensional_index(i);
            const uint column = uint(index[0]);
            const uint row = uint(index[1]);
            const ulong parameter = (ulong(tile) * groups + g) * StorageN +
                                    tile_offset + column;
            acc[i] += partial[i] * float(scales[parameter]) +
                      row_sums[row] * float(biases[parameter]);
        }
    }

    for (ushort i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i)) {
            continue;
        }
        const auto index = acc.get_multidimensional_index(i);
        out[uint(index[1]) * out_row_stride + out_origin + uint(index[0])] =
            acc[i];
    }
    // A caller that runs the next tile on the same scratch straight away would
    // otherwise rewrite `row_sums` under a straggling simdgroup's epilogue.
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The second reader of the affine slab, and why there are two.
//
// `q35moe_affine_tile` above is the dense reader: it multiplies `Rows`
// CONTIGUOUS activation rows by `TileN` outputs through an MPP `matmul2d`,
// which needs both operands addressable as tensors with a constant stride.
//
// A routed expert stage has neither property. Its activation rows are the ones
// `sorted[bounds[e] .. bounds[e + 1]]` names -- an arbitrary, data-dependent
// gather -- so there is no stride that walks them, and the grid the host fixes
// (`ceil(out_dim / 4)` threadgroups of 128 threads) spends one SIMDGROUP on one
// output row rather than a threadgroup on a `Rows x TileN` tile. So the tile is
// not a building block the expert entries can stand on; they get their own
// reader, below, over the same bytes and the same placement arithmetic.
//
// The two agree on the layout and differ only in the order they fold it:
// the tile factors the epilogue per group (`scale * sum a*code + bias * sum a`)
// because the matmul can only give it the code product, while this one
// dequantizes each code where it reads it, which is the elementwise
// `fmaf(weight, x, acc)` shape of `shiso_q35moe_expert_dot` in
// `kernels/cuda/qwen35moe/experts.cuh` and of the host reference both are
// graded against.

/// The parameter index of quant group `g` of output `row`, within one expert.
///
/// The device half of `placement` in
/// `crates/shiso-kerneltest/src/qwen35moe/q4affine.rs`, and the same expression
/// `q35moe_affine_tile`'s epilogue computes. `row` must be inside a whole
/// storage tile, which `Q35MOE_STORAGE_N`'s note explains.
static inline ulong q35moe_affine_parameter(uint row, uint g, uint groups) {
    const uint tile = row / Q35MOE_STORAGE_N;
    const uint tile_offset = row % Q35MOE_STORAGE_N;
    return (ulong(tile) * groups + g) * Q35MOE_STORAGE_N + tile_offset;
}

/// One output row of one expert's affine slab, dotted against one activation
/// row, folded so that **lane 0 holds the answer**.
///
/// `code_base` is the expert's byte offset into the code slab and
/// `parameter_base` its element offset into the scale and bias slabs; both are
/// the caller's because they do not change across the expert's columns.
///
/// Each lane takes one byte of every quantization group, which is codes
/// `2 * lane` and `2 * lane + 1` of it: a group's 32 code bytes are contiguous
/// (that is what group-major buys) so a simdgroup consumes a whole group in one
/// coalesced 32-byte load, and `in_dim / 64` iterations cover the row. That is
/// a different partition of the reduction from the CUDA reference's
/// `k = lane, lane + 32, ...`, which the layout forces and which nothing grades
/// against: no CUDA entry reads this codec.
///
/// The fold is `q35moe_simd_sum`, the explicit decreasing-offset
/// `simd_shuffle_down` shape this set fixes for sums. Every lane of the
/// simdgroup must reach it, which is why the callers below take their early
/// returns on quantities -- the expert, its span, the output row -- that are
/// uniform across a simdgroup.
template <typename Act>
inline float q35moe_affine_row_dot(device const uchar *codes,
                                   device const bfloat *scales,
                                   device const bfloat *biases,
                                   ulong code_base,
                                   ulong parameter_base,
                                   uint row,
                                   uint in_dim,
                                   device const Act *x,
                                   uint simd_lane) {
    const uint groups = in_dim / Q35MOE_GROUP;
    float acc = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        const ulong parameter = q35moe_affine_parameter(row, g, groups);
        const float scale = float(scales[parameter_base + parameter]);
        const float bias = float(biases[parameter_base + parameter]);
        const uchar byte =
            codes[code_base + parameter * (Q35MOE_GROUP / 2) + simd_lane];
        const uint k = g * Q35MOE_GROUP + simd_lane * 2u;
        // Low nibble first, which is `pack_q4_affine`'s order and MLX's.
        acc += (float(byte & 0x0Fu) * scale + bias) * float(x[k]);
        acc += (float(byte >> 4u) * scale + bias) * float(x[k + 1u]);
    }
    return q35moe_simd_sum(acc);
}
