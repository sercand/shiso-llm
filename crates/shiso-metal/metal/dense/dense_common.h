// Shared reduction shapes, activations and argument blocks for the dense kernel
// set.  This file is concatenated ahead of every other dense entry, so it may
// not include anything and the files after it may not need to declare a helper.
//
// "dense" is the parametric family: attention and its helpers written against
// argument blocks rather than against one checkpoint's widths, so a second
// model of the same shape is a different set of numbers and not a second set of
// kernels.  Nothing here assumes a hidden width, a head count or a head
// dimension.
//
// The reduction helpers below are the only sanctioned way to reduce in a dense
// kernel.  The bare builtin is never used here: its summation order is
// unspecified by the shading language, and a golden that pins a decode step
// pins the order its partial sums were added in.

// 32 simdgroup partials plus the broadcast slot; see dense_block_sum.
#define DENSE_BLOCK_SUM_SLOTS 33u

// The two reduction helpers below -- dense_warp_sum and dense_block_sum -- are
// written as a fixed sequence of additions, and the library is compiled with
// the shading language's default fast math, which lets the optimizer
// reassociate one.  A fixed shape the optimizer is free to re-fold is not a
// fixed shape, so the pair is compiled safe and the default is restored
// immediately after them.
//
// MSL has no push/pop for this pragma: the `fast` below restores the BUILD
// DEFAULT, not a saved state.  See the note at
// shiso/crates/shiso-metal/src/compile.rs (compile_library) before changing the
// library's math options.  Nothing below the restore feeds the bit-exactness
// claim, and the activations are transcendental-heavy code that leans on fast
// math.
#pragma METAL fp math_mode(safe)

/// Inclusive sum over one simdgroup, left-folded in decreasing-offset order so
/// lane 0 ends up holding the total.  Spelled out instead of using the builtin
/// because the shading language leaves that summation order unspecified.
static inline float dense_warp_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

/// Block-wide reduction over at most 32 simdgroups.
///
/// The same shape as q35_block_sum in metal/qwen35/q35_common.h, copied rather
/// than shared because the two sets are separately auditable: warp reduce, lane
/// 0 publishes to scratch[warp], barrier, warp 0 re-sums the lanes below
/// n_warp, thread 0 broadcasts through the last slot.  `scratch` needs
/// DENSE_BLOCK_SUM_SLOTS floats and is passed in because an inlined helper has
/// no threadgroup array of its own to keep.
static inline float dense_block_sum(
        float v,
        threadgroup float *scratch,
        uint tid,
        uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = dense_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = dense_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[DENSE_BLOCK_SUM_SLOTS - 1u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[DENSE_BLOCK_SUM_SLOTS - 1u];
}

/// Fused multiply-add.  MSL spells it `fma`; it is wrapped once here, as
/// q35_fma is in metal/qwen35/q35_common.h, so a kernel's inner product reads
/// as one decision and no call site can pick up a non-fused multiply by
/// accident.  It sits inside the safe bracket for the same reason the sums do:
/// a fold whose shape the optimizer may re-associate is not a fixed shape.
static inline float dense_fma(float a, float b, float c) {
    return fma(a, b, c);
}

// End of the fixed-order block: back to the library's default math mode.
#pragma METAL fp math_mode(fast)

/// The tanh approximation of the Gaussian error linear unit, as the reference
/// writes it: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
///
/// The constant is sqrt(2/pi) to the digits an f32 holds.  The other GELU -- the
/// exact one, through the error function -- differs from this by about 1e-4 per
/// value, so which of the two a layer uses is part of the checkpoint and not a
/// detail of the port.
///
/// `precise::tanh` rather than the library's default, which this file's
/// `fp math_mode(fast)` would otherwise select.  The fast variant is a rational
/// form in `exp(2y)`, and the cubic inside makes `y` grow fast: a gate value of
/// about 10.7 already puts `y` past 44, where `exp(2y)` overflows and the
/// quotient is inf/inf -- NaN, from an input the model produces routinely.  One
/// such element poisons a whole row of the projection below it, so the entire
/// residual stream is NaN from the first feed-forward onward.  The precise
/// entry saturates to 1 instead, which is what the reference does and what the
/// value is.
static inline float dense_gelu_tanh(float x) {
    return 0.5f * x * (1.0f + precise::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
}

// --- argument blocks -------------------------------------------------------
//
// One block per kernel family, passed as `constant T & args [[buffer(0)]]`.
// Field order is the binding contract the host mirrors, so it is spelled out
// here once and never re-derived inside a kernel.

/// x[i] *= s over the first `n` elements.
struct dense_args_scale {
    uint32_t n;
    float s;
};

/// x[i] = cap * tanh(x[i] / cap) over the first `n` elements.
struct dense_args_softcap {
    uint32_t n;
    float cap;
};

/// Per-head RMSNorm plus the parametric rotation of the query projection.
///
/// The query row is `n_head * head_dim` wide: this set has no output gate riding
/// beside the query, which is what makes its rows narrower than qwen35's.
struct dense_args_q_norm_rope {
    uint32_t head_dim;
    uint32_t n_head;
    /// Leading dimensions the rotation touches.
    uint32_t n_rot;
    /// Width the frequency ladder is derived over. See `dense_rope_pairs`: this
    /// is NOT `n_rot` in general, and conflating the two leaves every global
    /// layer plausible and wrong.
    uint32_t freq_dims;
    uint32_t n_tok;
    float rope_base;
    float eps;
};

/// Key norm + rotation and the value store, both into the f16 cache.
struct dense_args_kv_norm_rope {
    uint32_t head_dim;
    uint32_t n_head_kv;
    uint32_t n_rot;
    uint32_t freq_dims;
    uint32_t n_tok;
    float rope_base;
    float eps;
    /// Rows per KV page: a logical position becomes a physical row only
    /// through the session page table. See `dense_kv_row`.
    uint32_t page_tokens;
    /// Non-zero when the value is normalized before it is cached. The
    /// normalization is the same mean square the key uses and carries NO
    /// weight, which is the whole of the difference between the two halves.
    uint32_t v_norm;
    /// Entries between one row's page table and the next, or 0 when every
    /// row of this launch reads the table the pointer names. See
    /// `dense_row_pages`.
    uint32_t page_stride;
};

/// The per-layer embedding mix, one threadgroup per (token, layer slot).
///
/// `row_bytes` is the stride of ONE TOKEN's whole row -- every slot's block laid
/// end to end -- and not one block, because that is what turns a row identifier
/// into an offset.  `resident` says where the identifier comes from: a resident
/// table is indexed by the token id, and a streamed one has already been
/// narrowed to the chunk, so row `t` is token `t`'s.
struct dense_args_ple {
    uint32_t n_tok;
    uint32_t n_layers;
    /// Weights in one slot's row.  Must be DENSE_PLE_DIM: the row is exactly one
    /// q6_K block and the staging below is sized for it.  The engine asserts it
    /// on the host, because a kernel cannot refuse.
    uint32_t dim;
    /// Bytes one token's whole row occupies: one q6_K block per layer slot.
    uint32_t row_bytes;
    /// Non-zero when `rows` is the whole table and `ids` picks the row.
    uint32_t resident;
    float eps;
    /// What the model projection is multiplied by before it is normalized, so
    /// the scale reaches the magnitudes the normalization divides by as well as
    /// the value it then multiplies.
    float proj_scale;
    /// What the dequantized embedding is multiplied by.
    float embed_scale;
    /// What their sum is multiplied by.
    float mix_scale;
};

/// The gate multiply at one layer slot.
struct dense_args_ple_gate {
    uint32_t n_tok;
    uint32_t n_layers;
    uint32_t dim;
    /// Which slot of the prepared block this launch reads.
    uint32_t layer;
};

/// Windowed, bounded and scaled GQA over the paged f16 cache.
struct dense_args_attn {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_tok;
    /// Rows per KV page: a logical position becomes a physical row only
    /// through the session page table. See `dense_kv_row`.
    uint32_t page_tokens;
    /// Positions a query may look back over; 0 is no window at all.
    uint32_t window;
    /// The number every score is multiplied by. A per-layer fact rather than
    /// `1 / sqrt(head_dim)`: this family's layers pass 1.0.
    float scale;
    /// Entries between one row's page table and the next, or 0 when every
    /// row of this launch reads the table the pointer names. See
    /// `dense_row_pages`.
    uint32_t page_stride;
};

/// Per-head RMSNorm of the three attention planes and the two-axis rotation of
/// the query and the key, for the vision tower.
///
/// The rotation is the same neox pairing the language model uses, applied twice
/// over one head: the leading half of the head's channels rotates by the patch's
/// COLUMN and the trailing half by its ROW, each over a frequency ladder built
/// for a width of `head_dim / 2`. The value plane is normalized with no weight,
/// exactly as the key/value cache kernel does, and is not rotated.
struct dense_args_vit_rope2d {
    uint32_t rows;
    uint32_t heads;
    uint32_t head_dim;
    /// Columns in the patch grid: a row's column is `row % grid_w` and its row
    /// is `row / grid_w`, both taken within the item (a still is one plane).
    uint32_t grid_w;
    float eps;
    float rope_base;
};

/// A `3 x 3` average pool over the patch grid, scaled by a constant.
///
/// The pooler that folds a block of patch rows into one language-model token.
/// `grid_h` and `grid_w` are the patch grid before pooling and are both a whole
/// multiple of the pool side, which the resize rule guarantees; the output grid
/// is a third of each.
struct dense_args_vit_pool {
    uint32_t grid_h;
    uint32_t grid_w;
    uint32_t hidden;
    float scale;
};

/// `out[i] = clamp(in[i, min], max)` over the first `count` elements.
///
/// The clamped linear projection the reference wraps every vision weight in: the
/// input is clamped before the multiply and the output after it, and both are
/// this one elementwise kernel run over a different buffer.
struct dense_args_clamp {
    uint32_t count;
    float min;
    float max;
};

// --- launch geometry -------------------------------------------------------
//
// The numbers below are compile-time strides into threadgroup memory, so
// they bound what the argument blocks may carry: `head_dim` may not exceed
// DENSE_ATTN_HD, and the engine's smem constants in
// crates/shiso-engine/src/launch.rs are computed from exactly these.

// Simdgroups per attention threadgroup (256 threads).
#define DENSE_ATTN_WARPS 8u
// Widest head this set stages in threadgroup memory.
#define DENSE_ATTN_HD 512u
// Accumulator slots one lane holds: DENSE_ATTN_HD / 32.
#define DENSE_ATTN_LANE_SLOTS 16u
// Queries per prefill threadgroup: two per simdgroup.
#define DENSE_ATTN_QT 16u
// Keys per prefill tile, one per lane of the low half of a simdgroup.
#define DENSE_ATTN_KT 16u
// Weights in one per-layer embedding row, which is one q6_K block exactly.  The
// prepare kernel stages a decoded row of this width in threadgroup memory, so
// it bounds `dense_args_ple::dim` and the engine asserts the equality on the
// host.
#define DENSE_PLE_DIM 256u
// Elements one q6_K dequantize call produces, which is how the staging loop
// divides the row between threads.
#define DENSE_PLE_GROUP 16u

/// Physical row of a logical KV position, through the session page table.
///
/// The same translation `q35_kv_row` performs, copied rather than shared for the
/// reason `dense_block_sum` is: the two sets are separately auditable.
/// `pages[logical page]` is the physical page holding that span, so a position
/// becomes `pages[p / page_tokens] * page_tokens + p % page_tokens`.
///
/// A null table means the caller's rows are already physical. Production callers
/// always pass a table; a single-page fixture is the case that does not need one.
static inline uint64_t dense_kv_row(device const int32_t *pages, uint64_t position,
                                    uint32_t page_tokens) {
    if (pages == nullptr) {
        return position;
    }
    // The divide is 32-bit: a position fits (a context is far below 2^32
    // tokens), and a 64-bit divide is a software routine on this GPU that
    // cost decode attention ~3000 cycles a key on an M1 Max.
    const uint32_t p = (uint32_t) position;
    return (uint64_t) pages[p / page_tokens] * page_tokens + (p % page_tokens);
}

/// Row `t`'s own page table inside a packed block of them.
///
/// A decode step may carry one row per SESSION, and two sessions share no page:
/// a kernel that read one table for every row would answer row one out of row
/// zero's conversation, fluently and at the right shape. So the table a row
/// reads is selected here, by the row index the grid already carries.
///
/// `stride` is entries between one row's table and the next, and `0` is the
/// shared case rather than a sentinel - `pages + t * 0` is `pages` for every
/// row, which is exactly what a prefill chunk and a single-session step want.
/// A null table stays null, because that is the "no paging, position IS the
/// row" convention `dense_kv_row` answers for.
static inline device const int32_t *dense_row_pages(device const int32_t *pages,
                                                    uint t, uint32_t stride) {
    if (pages == nullptr) {
        return pages;
    }
    return pages + (uint64_t) t * stride;
}


/// Rotary over the leading `n_rot` dimensions of a threadgroup-resident head,
/// with the frequency ladder derived over `freq_dims`.
///
/// Pair `i` with `i + freq_dims / 2` for `i < n_rot / 2`, at the angle
/// `pos * base^(-2i / freq_dims)`. The two widths are separate arguments because
/// a family exists that derives the ladder over the whole head and rotates only
/// its leading slice: at head_dim 512, n_rot 128 and freq_dims 512 the pairs are
/// (i, i + 256) for i < 64, and a kernel that read the pair stride off `n_rot`
/// would pair (i, i + 64) instead -- every head still normalized, every head
/// still rotated, every head wrong. Setting freq_dims = n_rot recovers the neox
/// rotation `q35_rope_neox` performs.
///
/// PRECONDITION: freq_dims >= n_rot.  The loop runs over i < n_rot / 2 and
/// writes i + freq_dims / 2, so a freq_dims below n_rot makes the partner index
/// land back inside the loop's own range: lane A's partner write and lane B's
/// pair read would then race over the same dimension, and the head would come
/// out differently from one dispatch to the next.  Nothing here checks it --
/// a kernel cannot refuse -- so the engine validates the pair on the host
/// before it ever builds the argument block.
///
/// `pow`, `sin` and `cos` are the accurate builtins rather than the `fast::`
/// ones, as `q35_rope_neox` also uses: the angle ladder spans six decades of
/// base and the fast variants lose the far end of it.
static inline void dense_rope_pairs(
        threadgroup float *v,
        uint n_rot,
        uint freq_dims,
        float base,
        float pos,
        uint lane,
        uint nlanes) {
    const uint half_width = freq_dims / 2u;
    const uint n_pair = n_rot / 2u;
    for (uint i = lane; i < n_pair; i += nlanes) {
        const float theta = pos * pow(base, -2.0f * (float) i / (float) freq_dims);
        const float s = sin(theta);
        const float c = cos(theta);
        const float x0 = v[i];
        const float x1 = v[i + half_width];
        v[i] = x0 * c - x1 * s;
        v[i + half_width] = x0 * s + x1 * c;
    }
}

/// Argument block for packed affine matvec and GEMM kernels.
struct dense_args_affine {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_tok;
    uint32_t bits;
};

/// Strided row copy (`dense_block_draft.metal`).
struct dense_args_copy_rows {
    uint32_t n_rows;
    uint32_t width;
    uint32_t dst_stride;
    uint32_t dst_offset;
};

/// Grouped dynamic causal convolution (`dense_block_draft.metal`).
struct dense_args_dyn_conv {
    uint32_t n_rows;
    uint32_t hidden;
    uint32_t group;
    uint32_t taps;
    uint32_t part;
    uint32_t pad0;
    uint32_t pad1;
    uint32_t pad2;
};

/// Row-exact affine matmul (`dense_exact_affine.metal`): the weight's shape and
/// the rows of the call.
struct dense_args_exact_affine {
    uint32_t in_dim;
    uint32_t out_dim;
    uint32_t n_rows;
    uint32_t pad;
};

/// Widens one bf16 value by placing its 16 bits into the high half of an f32.
static inline float dense_bf16(device const ushort *p) {
    return as_type<float>((uint) (*p) << 16);
}

/// Argument block for affine embedding gather.
struct dense_args_embed_affine {
    uint32_t hidden;
    uint32_t n_tok;
};
