// Argument blocks and helpers for the audio encoder kernel set.
//
// Concatenated ahead of every audio entry, so it may not include anything.
// The set serves transformer and convolution-augmented encoders over f32
// activations: every kernel accumulates in f32 whatever width it read, and
// every reduction below is spelled out so its summation order is fixed.

#define AUDIO_LINEAR_BLOCK 256u
// Full unrolling of a fixed-count loop over simdgroup fragments. Without it
// the compiler may keep a fragment array in memory and index it: a 4x4 block
// of accumulators per simdgroup measured six times slower that way, and the
// GEMM's own 4x2 block 7% slower on M1 Max.
#define AUDIO_UNROLL _Pragma("clang loop unroll(full)")
#define AUDIO_BLOCK_SUM_SLOTS 33u
#define AUDIO_GEMM_TILE 32u
#define AUDIO_GEMM_TILE_M 32u
#define AUDIO_GEMM_TILE_N 64u
#define AUDIO_GEMM_TILE_K 32u
#define AUDIO_GEMM64_TILE_M 64u
#define AUDIO_GEMV_ROWS 8u
#define AUDIO_GEMV_COLS 4u
// Columns one matrix-vector-by-runs threadgroup produces, and the simdgroups
// that share each column's k range.
#define AUDIO_MATVEC_COLS 4u
#define AUDIO_MATVEC_SG 2u
// Input rows one threadgroup of the multi-row matrix-vector kernels takes:
// a decoder batch's rows share each weight run the threadgroup reads.
#define AUDIO_MATVEC_ROWS 8u
// Values one thread of the fused residual-add RMS norm loads at once.
#define AUDIO_ADD_RMS_RUN 8u
// Head dims up to 32 lanes x 4 values; the attention keeps a lane's slice of
// the query, the accumulator and the output in registers.
#define AUDIO_ATTN_SLOTS 4u
// Queries one attention simdgroup takes at once.
#define AUDIO_ATTN_QB 8u
// Distance rows one relative-score simdgroup takes at once.
#define AUDIO_REL_ROWS 4u
// Threads one row argmax spreads a row over.
#define AUDIO_ARGMAX_BLOCK 256u
// Cached rows one decode-attention simdgroup reads per block, and the query
// heads of a kv head's group it scores against each row it reads. Both are
// small on purpose: at 8 rows and 4 heads the block's registers spill and the
// kernel ran 1.7x slower over 4096 rows on an M1 Max.
#define AUDIO_DECODE_KB 4u
#define AUDIO_DECODE_GT 2u
// Splits one decode-attention threadgroup runs, a simdgroup each; the most
// splits a launch may cut the rows into; the merge's threads per head and the
// simdgroups they make.
#define AUDIO_DECODE_SPLITS_PER_TG 4u
#define AUDIO_DECODE_MAX_SPLITS 256u
#define AUDIO_DECODE_MERGE_THREADS 256u
#define AUDIO_DECODE_MERGE_SG 8u

#define AUDIO_OP_RELU 0u
#define AUDIO_OP_SILU 1u
#define AUDIO_OP_SIGMOID 2u
#define AUDIO_OP_GELU_ERF 3u
#define AUDIO_OP_TANH 4u
#define AUDIO_OP_CLAMP 5u
#define AUDIO_OP_SCALE 6u

#define AUDIO_NORM_HAS_WEIGHT 1u
#define AUDIO_NORM_HAS_BIAS 2u
#define AUDIO_GEMM_HAS_BIAS 1u

typedef struct { uint count; uint op; float a; float b; } audio_args_unary;
typedef struct { uint count; float alpha; } audio_args_axpy;
typedef struct { uint rows; uint cols; } audio_args_bias_rows;
typedef struct { uint rows; uint cols; } audio_args_argmax;
typedef struct { uint rows; uint channels; } audio_args_glu;
typedef struct { uint count; } audio_args_silu_mul;
typedef struct { uint rows; uint dim; float eps; uint flags; } audio_args_norm;
typedef struct { uint m; uint n; uint k; uint flags; uint k_chunk; uint splits; } audio_args_gemm;
typedef struct { uint n0; uint n1; uint n2; uint k; } audio_args_matvec3;
typedef struct {
    uint t_in; uint f_in; uint c_in;
    uint t_out; uint f_out; uint c_out;
    uint pad_t; uint pad_f;
    uint depthwise; uint has_bias; uint relu;
    uint in_valid; uint out_valid;
    uint batch;
} audio_args_conv2d;
typedef struct { uint rows; uint channels; uint taps; } audio_args_dwconv1d;
typedef struct { uint rows; uint channels; uint taps; uint pad; } audio_args_unfold;
typedef struct { uint batch; uint rows; uint cols; } audio_args_transpose;
typedef struct {
    uint n_q; uint n_k; uint heads; uint head_dim;
    uint q_abs0; uint k_abs0; uint k_end;
    uint chunk; uint left_chunks;
    int d_min; uint n_d;
    float scale; float ac_scale; float softcap;
    uint has_bias_uv; uint has_q_dim_scale;
} audio_args_window_attn;
typedef struct { uint rows; uint heads; uint head_dim; uint pos0; float theta; } audio_args_rope;
typedef struct { uint hidden; } audio_args_lstm;
typedef struct { uint rows; uint cols; uint avail; } audio_args_greedy_window;
typedef struct { uint rows; uint avail; uint blank; uint max_symbols; uint slot; } audio_args_greedy_step;
typedef struct { uint width; uint vocab; } audio_args_embed_row;
typedef struct { uint count; } audio_args_keep;
typedef struct {
    uint n_q; uint heads; uint kv_heads; uint head_dim;
    uint pos0; uint cap; float scale; uint splits;
} audio_args_cache_attn;
typedef struct { uint rows; uint kv_heads; uint head_dim; uint cap; } audio_args_kv_append;
typedef struct {
    uint rows; uint heads; uint kv_heads; uint head_dim; uint cap; float eps; float theta;
} audio_args_qk_rope;
typedef struct { uint cols; uint span; uint parts; } audio_args_argmax_split;
typedef struct { uint eos0; uint eos1; uint max_tokens; uint advance; uint slot; } audio_args_decode_step;
typedef struct { uint width; uint vocab; uint group; } audio_args_embed_token;
typedef struct {
    uint seqs; uint heads; uint kv_heads; uint head_dim; float scale; uint splits;
} audio_args_batch_attn;
typedef struct {
    uint rows; uint seqs; uint heads; uint kv_heads; uint head_dim; float eps; float theta;
} audio_args_qk_rope_seq;
typedef struct {
    uint eos0; uint eos1; uint max_tokens; uint advance; uint slot; uint seqs;
} audio_args_decode_step_batch;

// Greedy search's state on the device, one u32 each: the frame (relative to
// the first pending row), the symbols emitted on it, whether the search has
// reached the last pending row, the last token, and whether the last step
// emitted it.
#define AUDIO_GREEDY_T 0u
#define AUDIO_GREEDY_SYM 1u
#define AUDIO_GREEDY_DONE 2u
#define AUDIO_GREEDY_TOKEN 3u
#define AUDIO_GREEDY_EMIT 4u
#define AUDIO_GREEDY_NONE 0xffffffffu

// A causal decoder's state on the device, one u32 each: the cache rows
// filled (the position of the token the next pass feeds), the tokens
// emitted, done, the last token emitted (the one the next pass feeds),
// whether the last step emitted, and why the decoder stopped.
#define AUDIO_DECODE_POS 0u
#define AUDIO_DECODE_COUNT 1u
#define AUDIO_DECODE_DONE 2u
#define AUDIO_DECODE_TOKEN 3u
#define AUDIO_DECODE_EMIT 4u
#define AUDIO_DECODE_STOP 5u
#define AUDIO_DECODE_NONE 0xffffffffu
#define AUDIO_DECODE_STOP_EOS 1u
#define AUDIO_DECODE_STOP_CAP 2u
// Words of one decoder state; a batch keeps its sequences' states back to
// back, sequence s's at s * AUDIO_DECODE_STATE_WORDS.
#define AUDIO_DECODE_STATE_WORDS 8u

// A batch of causal-decoder sequences, eight u32 words per sequence: the
// first cache row of its `[kv_heads][cap][hd]` plane in the batch's shared
// key and value buffers (row r of kv head h is at `(plane + h * cap + r) *
// hd`), the plane's capacity, and for a segmented prefill its first stacked
// row, its row count and the position of that first row. The decode entries
// read only the plane and the capacity.
#define AUDIO_SEQ_WORDS 8u
#define AUDIO_SEQ_PLANE 0u
#define AUDIO_SEQ_CAP 1u
#define AUDIO_SEQ_ROW0 2u
#define AUDIO_SEQ_ROWS 3u
#define AUDIO_SEQ_POS0 4u

/// One brain-float, widened exactly (the 16 bits become the high half).
static inline float audio_bf16(device const ushort *p) {
    return as_type<float>((uint) (*p) << 16);
}

/// Simdgroup sum folded in decreasing-offset order; lane 0 holds the total.
static inline float audio_warp_sum(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

/// Simdgroup sum, butterfly-folded so every lane holds the total.
static inline float audio_warp_sum_all(float v) {
    for (uint off = 16u; off > 0u; off >>= 1u) {
        v += simd_shuffle_xor(v, off);
    }
    return v;
}

/// Block-wide sum over at most 32 simdgroups; `scratch` holds
/// AUDIO_BLOCK_SUM_SLOTS floats.
static inline float audio_block_sum(float v, threadgroup float *scratch, uint tid, uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = audio_warp_sum(v);
    if (lane == 0u) {
        scratch[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? scratch[tid] : 0.0f;
    if (warp == 0u) {
        v = audio_warp_sum(v);
    }
    if (tid == 0u) {
        scratch[AUDIO_BLOCK_SUM_SLOTS - 1u] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[AUDIO_BLOCK_SUM_SLOTS - 1u];
}

/// The error function, rational-times-Gaussian form with error below 1.5e-7.
static inline float audio_erf(float x) {
    const float sign = (x < 0.0f) ? -1.0f : 1.0f;
    const float a = fabs(x);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float poly = t * (0.254829592f + t * (-0.284496736f + t * (1.421413741f
            + t * (-1.453152027f + t * 1.061405429f))));
    return sign * (1.0f - poly * precise::exp(-a * a));
}

/// The logistic function, with the exponential evaluated precisely.
static inline float audio_sigmoid(float x) {
    return 1.0f / (1.0f + precise::exp(-x));
}
