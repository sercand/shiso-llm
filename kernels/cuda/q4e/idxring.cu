/* The indexer's pooled key pass over a per-session ring of raw keys instead of
 * a raw key row for every cached position.
 *
 * Not an upstream unit, for the reason `exact.cu` is not: `idx.cu` is a byte
 * copy of its source and stays one. The body below is `q4e_idx_pool_kernel`'s
 * arithmetic unchanged; what moved is where a raw key is read from.
 *
 * # Which raw keys are ever read
 *
 * Exactly one kernel in the family reads a raw indexer key: the pooled pass,
 * which averages the r = 4 raw keys of a block the moment the block's last
 * position is written. The scorers read pooled rows, the attention reads the
 * K and V planes (the tail of the incomplete block included), and nothing
 * reads a raw key of a block that has been pooled. So the only raw keys worth
 * keeping between two passes are those of the block the session's frontier
 * stands inside: at most r - 1 = 3 of them.
 *
 * A speculative step makes that "at most a few more": a verify pass writes
 * rows past the frontier and the session may then settle anywhere inside
 * them, and the draft head's chain writes ahead and its catch-up pass starts
 * behind. If every pass starts no more than D positions before the highest
 * position written so far, a pass starting at s reads raw keys of positions
 * n in [s - 3, s), and a later write that could have landed on n's slot is a
 * position q with q == n (mod RING), n - D < q <= s - 1 + D. Two positions of
 * one slot are RING apart, so RING >= D + 3 is enough for the slot to still
 * hold n's own key. A verify is at most MAX_DRAFT + 1 = 9 rows and the chain
 * rewinds by at most MAX_DRAFT, so D <= 9 and RING = 16 holds with room. The
 * host pins the inequality (`kernels::IDX_RING`).
 *
 * # One launch: pool, then tap, then store
 *
 * A raw key of a position this pass writes is read straight from the pass's
 * own f32 key rows, rounded to f16 exactly as the cache store rounded it, so
 * the pooled row is bit-identical to the one the raw plane produced. A raw key
 * of an earlier position is read from the ring. Only block b0 - the block
 * holding pos[0] - can reach before pos[0], and it is always block 0 of the
 * grid, so block 0 also does the two writes after its pool, behind a barrier:
 * the tap (a copy of the frontier block's raw keys as they stand before row
 * `snap_row`, into a checkpoint's ring section), then the ring store (the
 * last RING rows of the pass, each at its position's slot). A pass longer than
 * RING writes a slot more than once; storing only its last RING rows keeps one
 * writer per slot, and those are the only ones a later pass can read.
 *
 * The geometry is the ported pool's: `n_tok / r + 1` blocks of 128 threads.
 * There is no batched sibling; a batch launches this once per row, as it
 * launched the pool.
 */

#include "q4e.cuh"

#define Q4E_IR_DIM  128u
#define Q4E_IR_R    4u
#define Q4E_IR_RING 16u
#define Q4E_IR_NO_TAP 0xFFFFFFFFu

static_assert(SHS_Q4E_PAGE_TOKENS % Q4E_IR_R == 0u,
              "a pooled key block must not straddle a KV page");
static_assert(Q4E_IR_RING % Q4E_IR_R == 0u && Q4E_IR_RING >= 3u * Q4E_IR_R,
              "the ring must hold a whole block and a verify's rewind behind it");

/* The f16 raw key of position x, element i: this pass's own row rounded as the
 * cache store rounds it, or the ring's copy of an earlier pass's. */
__device__ __forceinline__ static __half q4e_ir_raw(const __half *ring, const float *k,
                                                    uint32_t p0, uint32_t x, uint32_t i) {
    return x >= p0 ? __float2half(k[(uint64_t)(x - p0) * Q4E_IR_DIM + i])
                   : ring[(uint64_t)(x % Q4E_IR_RING) * Q4E_IR_DIM + i];
}

extern "C" __global__ void q4e_idx_ring_pool_kernel(
        __half *pooled, __half *ring, const float *k, const float *w, const int32_t *pos,
        const int32_t *pages, uint32_t n_tok, uint32_t n_rot, float rope_base,
        const int32_t *mrope_all, uint32_t sec_t, uint32_t sec_h, uint32_t sec_w, float eps,
        __half *snap, uint32_t snap_row) {
    const uint32_t p0 = (uint32_t)pos[0];
    const uint32_t b0 = p0 / Q4E_IR_R;
    const uint32_t b1 = ((uint32_t)pos[n_tok - 1u] + 1u) / Q4E_IR_R;
    const uint32_t b = b0 + blockIdx.x;
    /* Block-uniform, so the barriers inside are reached by every thread or by
     * none; block 0 falls through to the writes either way. */
    if (b < b1) {
        const uint32_t krow = q4e_kv_row(pages, b * Q4E_IR_R);
        const uint32_t brow = krow / Q4E_IR_R;          /* pooled row of block b */
        __shared__ float s_k[Q4E_IR_DIM];
        float sum = 0.0f;
        for (uint32_t i = threadIdx.x; i < Q4E_IR_DIM; i += blockDim.x) {
            float acc = 0.0f;
            for (uint32_t j = 0; j < Q4E_IR_R; j++) {
                acc += __half2float(q4e_ir_raw(ring, k, p0, b * Q4E_IR_R + j, i));
            }
            acc *= 1.0f / (float)Q4E_IR_R;
            s_k[i] = acc;
            sum += acc * acc;
        }
        const float scale = rsqrtf(q4e_block_sum(sum) / (float)Q4E_IR_DIM + eps);
        for (uint32_t i = threadIdx.x; i < Q4E_IR_DIM; i += blockDim.x) s_k[i] *= scale * w[i];
        __syncthreads();
        /* The block's start position rounds pos[0] down, so it can sit up to
         * r - 1 rows before the pass's first token: the rotary table carries
         * SHS_Q4E_MROPE_BACK rows in front of token 0 for exactly this read. */
        const int32_t block_pos = (int32_t)(b * Q4E_IR_R);
        const uint32_t mrow = (uint32_t)((int32_t)SHS_Q4E_MROPE_BACK + block_pos - pos[0]);
        q4e_rope_mrope(s_k, n_rot, rope_base,
                       mrope_all[3u * mrow], mrope_all[3u * mrow + 1u], mrope_all[3u * mrow + 2u],
                       sec_t, sec_h, sec_w, threadIdx.x, blockDim.x);
        __syncthreads();
        for (uint32_t i = threadIdx.x; i < Q4E_IR_DIM; i += blockDim.x) {
            pooled[(uint64_t)brow * Q4E_IR_DIM + i] = __float2half(s_k[i]);
        }
    }
    if (blockIdx.x != 0u) return;
    /* Every read of the ring above happens before any write below. */
    __syncthreads();
    if (snap_row != Q4E_IR_NO_TAP) {
        const uint32_t sp = p0 + snap_row;
        for (uint32_t x = (sp / Q4E_IR_R) * Q4E_IR_R; x < sp; x++) {
            for (uint32_t i = threadIdx.x; i < Q4E_IR_DIM; i += blockDim.x) {
                snap[(uint64_t)(x % Q4E_IR_RING) * Q4E_IR_DIM + i] = q4e_ir_raw(ring, k, p0, x, i);
            }
        }
    }
    __syncthreads();
    const uint32_t first = n_tok > Q4E_IR_RING ? n_tok - Q4E_IR_RING : 0u;
    for (uint32_t t = first; t < n_tok; t++) {
        const uint32_t x = (uint32_t)pos[t];
        for (uint32_t i = threadIdx.x; i < Q4E_IR_DIM; i += blockDim.x) {
            ring[(uint64_t)(x % Q4E_IR_RING) * Q4E_IR_DIM + i] =
                __float2half(k[(uint64_t)t * Q4E_IR_DIM + i]);
        }
    }
}
