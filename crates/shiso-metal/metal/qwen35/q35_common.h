// Shared geometry, reduction shapes and argument blocks for the qwen35 kernel
// set.  This file is concatenated ahead of every other q35 entry, so it may not
// include anything and the files after it may not need to declare a helper.
//
// The reduction helpers below are the only sanctioned way to reduce in a q35
// kernel.  The bare builtin is never used here: its summation order is unspecified
// by the shading language, and these kernels are ported against a fixed
// reduction shape rather than against a mathematically equal one.

#define Q35_GDN_D 128u
#define Q35_GDN_CHUNK 64u
#define Q35_GDN_QTILE 32u
#define Q35_GDN_CSPLIT 8u
#define Q35_ATTN_QT 16u
#define Q35_ATTN_KT 32u
#define Q35_ATTN_WARPS 8u
#define Q35_ATTN_LANE_SLOTS 8u
// Staging stride for the per-simdgroup partials: the attention head_dim bound.
#define Q35_ATTN_HD 256u
// conv_kernel 4 minus the current token.
#define Q35_CONV_HIST 3u
// 32 simdgroup partials plus the broadcast slot; see q35_block_sum.
#define Q35_BLOCK_SUM_SLOTS 33u
// One quarter of a state row: head_dim / 4.  The recurrent sweep is built from
// quarters and not from lanes, so this is a property of the summation order
// rather than of how many lanes a variant happens to launch with.
#define Q35_GDN_QUARTER 32u
// Recurrent scratch: block-sum slots plus the published k and q rows.
#define Q35_GDN_REC_SLOTS (Q35_BLOCK_SUM_SLOTS + 2u * Q35_GDN_D)

// The four reduction helpers below -- q35_warp_sum, q35_block_sum,
// q35_canonical4 and q35_fma -- are written as a fixed sequence of additions,
// and the library is compiled with the shading language's default fast math,
// which lets the optimizer reassociate one. A fixed shape the optimizer is
// free to re-fold is not a fixed shape, so those four are compiled safe and
// the default is restored immediately after them. `math_mode` governs
// reassociation and contraction only; the precision of the math library
// functions is a separate option (MTLMathFloatingPointFunctions) and is left
// alone, which is why this costs no occupancy.
//
// The region stops at q35_fma deliberately. Nothing below it feeds the
// bit-exactness claim, and silu/sigmoid/log1p/softplus and the rope helpers
// are division- and transcendental-heavy code that the other q35 kernels lean
// on fast math for.
//
// MSL has no push/pop for this pragma: the `fast` below restores the BUILD
// DEFAULT, not a saved state. See the note at
// shiso/crates/shiso-metal/src/compile.rs (compile_library) before changing
// the library's math options.
#pragma METAL fp math_mode(safe)

/// Inclusive sum over one simdgroup, left-folded in decreasing-offset order so
/// lane 0 ends up holding the total.  Spelled out instead of using the builtin
/// because the shading language leaves that summation order unspecified.
static inline float q35_warp_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

/// Block-wide reduction over at most 32 simdgroups.
///
/// Shape: warp reduce, lane 0 publishes to scratch[warp], barrier, warp 0
/// re-sums the lanes below n_warp, thread 0 broadcasts through the last slot.
/// `scratch` needs Q35_BLOCK_SUM_SLOTS floats and is passed in because an
/// inlined helper has no threadgroup array of its own to keep.
static inline float q35_block_sum(
        float v,
        threadgroup float *scratch,
        uint tid,
        uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = q35_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = q35_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[Q35_BLOCK_SUM_SLOTS - 1u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[Q35_BLOCK_SUM_SLOTS - 1u];
}

/// The canonical combination of four unit-chain partials.
///
/// This is the association the CUDA reference produces and the one every
/// variant of every GDN kernel must reproduce: the quarters pair up before
/// they cross.  Written once, here, so that "did a variant change the math"
/// is one function to read rather than five kernel bodies to diff.
///
/// Floating-point addition is commutative but not associative, so the
/// pairing is the whole content of this function -- there is nothing else
/// in it, and that is the point.
static inline float q35_canonical4(float q0, float q1, float q2, float q3) {
    return (q0 + q1) + (q2 + q3);
}

/// Fused multiply-add.  MSL spells the CUDA __fmaf_rn equivalent `fma`; it is
/// wrapped once here so a kernel reads as the port it is and no call site can
/// pick up a non-fused multiply by accident.
static inline float q35_fma(float a, float b, float c) {
    return fma(a, b, c);
}

// End of the fixed-order block: back to the library's default math mode.
#pragma METAL fp math_mode(fast)

/// x / (1 + exp(-x)), written as the source writes it rather than as exp2.
static inline float q35_silu(float a) {
    return a / (1.0f + exp(-a));
}

static inline float q35_sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

/// log(1 + x) for x > -1.
///
/// The shading language has no log1p and the naive log(1 + x) loses the
/// argument completely once x falls below the ulp of 1, which is exactly the
/// range the decay gate reaches for a strongly negative alpha.  The correction
/// is the standard one: return x where 1 + x rounds to 1, otherwise rescale
/// log(u) so the result stays within an ulp of the libm function the port is
/// written against.
static inline float q35_log1p(float x) {
    const float u = 1.0f + x;
    if (u == 1.0f) {
        return x;
    }
    return x * log(u) / (u - 1.0f);
}

/// softplus with the overflow guard the decay gate relies on: above 20 the
/// exp() term is far below the ulp of x, and below it the naive form overflows.
static inline float q35_softplus(float x) {
    return (x > 20.0f) ? x : q35_log1p(exp(x));
}

/// NeoX rope over the first n_rot dims of a threadgroup-resident row: pair
/// (i, i + n_rot/2), everything past n_rot passes through untouched.  Accurate
/// pow/sin/cos, because the angle grows with the position and the fast
/// variants drift by more than an ulp of the rotated value.
static inline void q35_rope_neox(
        threadgroup float *v,
        uint n_rot,
        float base,
        float pos,
        uint lane,
        uint nlanes) {
    const uint n_half = n_rot / 2u;
    for (uint i = lane; i < n_half; i += nlanes) {
        const float theta = pos * pow(base, -2.0f * (float) i / (float) n_rot);
        const float s = sin(theta);
        const float c = cos(theta);
        const float x0 = v[i];
        const float x1 = v[i + n_half];
        v[i] = x0 * c - x1 * s;
        v[i + n_half] = x0 * s + x1 * c;
    }
}

/// Interleaved multimodal rope over the first n_rot dims of a
/// threadgroup-resident row.
///
/// Same pairing and same frequency ladder as q35_rope_neox -- pair
/// (i, i + n_rot/2), angle pos * base^(-2i/n_rot) -- and the same untouched
/// tail past n_rot.  The one difference is where the position comes from:
/// pair i belongs to section i % 3, and the three sections are the temporal,
/// height and width streams in that order.  A position stream is a whole grid
/// coordinate, so the three agree on every text token and disagree only over
/// the placeholder run an image or a clip occupies.
///
/// Three agreeing streams therefore reproduce q35_rope_neox exactly, which is
/// a property worth stating and NOT one this file leans on: a chunk with no
/// media is dispatched to the one-dimensional entry points instead, so the
/// text-only path never reaches this function at all.
static inline void q35_rope_mrope_interleaved(
        threadgroup float *v,
        uint n_rot,
        float base,
        float pos_t,
        float pos_h,
        float pos_w,
        uint lane,
        uint nlanes) {
    const uint n_half = n_rot / 2u;
    for (uint i = lane; i < n_half; i += nlanes) {
        const uint section = i % 3u;
        const float pos = (section == 0u) ? pos_t : ((section == 1u) ? pos_h : pos_w);
        const float theta = pos * pow(base, -2.0f * (float) i / (float) n_rot);
        const float s = sin(theta);
        const float c = cos(theta);
        const float x0 = v[i];
        const float x1 = v[i + n_half];
        v[i] = x0 * c - x1 * s;
        v[i + n_half] = x0 * s + x1 * c;
    }
}

// --- argument blocks -------------------------------------------------------
//
// One block per kernel family, passed as `constant T & args [[buffer(0)]]`.
// Field order is the binding contract the host mirrors, so it is spelled out
// here once and never re-derived inside a kernel.

/// Causal depthwise conv over the q|k|v channels plus silu.
struct q35_args_gdn_conv {
    uint32_t channels;
    uint32_t conv_kernel;
    uint32_t n_tok;
};

/// In-place L2 normalization of one family of heads inside a strided row.
struct q35_args_gdn_l2norm {
    uint32_t head_dim;
    uint32_t n_heads;
    uint32_t stride;
    uint32_t offset;
};

/// decay = softplus(alpha_proj + dt_bias) * a, beta = sigmoid(beta_proj).
struct q35_args_gdn_gates {
    uint32_t n_head;
    uint32_t n_tok;
};

/// Sequential gated delta rule, one threadgroup per value head.
struct q35_args_gdn_recurrent {
    uint32_t head_dim;
    uint32_t n_head_k;
    uint32_t n_head_v;
    uint32_t k_offset;
    uint32_t v_offset;
    uint32_t stride;
    uint32_t n_tok;
};

/// Chunked-parallel gated delta rule, one threadgroup per value head.
struct q35_args_gdn_chunk {
    uint32_t head_dim;
    uint32_t n_head_k;
    uint32_t n_head_v;
    uint32_t k_offset;
    uint32_t v_offset;
    uint32_t stride;
    uint32_t n_tok;
};

/// Per-head RMSNorm of the delta-rule output, then the sigmoid output gate.
struct q35_args_gdn_out_gate {
    uint32_t head_dim;
    uint32_t n_head;
    float eps;
};

/// Per-head RMSNorm and partial rope of the q projection, plus the raw
/// output-gate slice that rides beside each query in the same row.
struct q35_args_q_norm_rope {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_rot;
    uint32_t n_tok;
    float rope_base;
    float eps;
};

/// Key norm + rope into the f16 cache, with the value copied beside it.
struct q35_args_k_norm_rope {
    uint32_t head_dim;
    uint32_t n_head_kv;
    uint32_t n_rot;
    uint32_t n_tok;
    float rope_base;
    float eps;
    /// Rows per KV page: a logical position becomes a physical row only
    /// through the session page table. See `q35_kv_row`.
    uint32_t page_tokens;
    /// Entries between one row's page table and the next, or 0 when every
    /// row of this launch reads the table the pointer names. See
    /// `q35_row_pages`.
    uint32_t page_stride;
};

/// Causal GQA over a contiguous f16 cache, one threadgroup per (head, token).
struct q35_args_attn_decode {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_tok;
    /// Rows per KV page: a logical position becomes a physical row only
    /// through the session page table. See `q35_kv_row`.
    uint32_t page_tokens;
    /// Entries between one row's page table and the next, or 0 when every
    /// row of this launch reads the table the pointer names. See
    /// `q35_row_pages`.
    uint32_t page_stride;
};

/// Tiled causal attention over a query tile of Q35_ATTN_QT rows.
struct q35_args_attn_prefill {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_head_kv;
    uint32_t n_tok;
    /// Rows per KV page: a logical position becomes a physical row only
    /// through the session page table. See `q35_kv_row`.
    uint32_t page_tokens;
    /// Entries between one row's page table and the next, or 0 when every
    /// row of this launch reads the table the pointer names. See
    /// `q35_row_pages`.
    uint32_t page_stride;
};

/// out = attn * sigmoid(gate), the gate half of the q projection.
struct q35_args_attn_gate {
    uint32_t head_dim;
    uint32_t n_head;
    uint32_t n_tok;
};

/// Physical row of a logical KV position, through the session page table.
///
/// `pages[logical page]` is the physical page holding that span, so a position
/// becomes `pages[p / page_tokens] * page_tokens + p % page_tokens`. Every
/// KV-touching kernel goes through this helper: a tile that starts inside a page
/// and stays inside it translates its first position and walks the rest of the
/// page contiguously, so the translation costs one load per tile instead of one
/// per row.
///
/// A null table means the caller rows are already physical. Production callers
/// always pass a table; a single-page fixture is the case that does not need one.
static inline uint64_t q35_kv_row(device const int32_t *pages, uint64_t position,
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
/// row" convention `q35_kv_row` answers for.
static inline device const int32_t *q35_row_pages(device const int32_t *pages,
                                                  uint t, uint32_t stride) {
    if (pages == nullptr) {
        return pages;
    }
    return pages + (uint64_t) t * stride;
}

