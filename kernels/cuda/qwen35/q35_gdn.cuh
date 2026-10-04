/* The gated delta network: the four small entries, the sequential recurrence
 * and the chunked-parallel form.
 *
 * The CUDA twins of `crates/shiso-metal/metal/qwen35/gdn_conv.metal`,
 * `gdn_recurrent.metal`, `gdn_recurrent_hybrid.metal`, `gdn_chunk.metal` and
 * `gdn_chunk_hybrid.metal`, over the laws `kernels/cuda/q4e/gdn.cu` carries as
 * `q4e_gdn_conv_kernel`, `q4e_gdn_l2norm_kernel`, `q4e_gdn_gates_kernel`,
 * `q4e_gdn_out_gate_kernel`, `q4e_gdn_recurrent_kernel` and
 * `q4e_gdn_chunk_kernel`.
 *
 * ==========================================================================
 * WHY THE LANE-WIDTH SIBLINGS ARE HERE AND NOT THE LAW'S TWO SHAPES
 * ==========================================================================
 *
 * The Metal set carries `_s1/_s2/_s4` and `_c1/_c2/_c4/_c8` where the law has
 * one kernel each, because the law's recurrence wants 66048 B of shared memory
 * for its 128x129 state and its chunk form wants 99072 B, against the 32768 B
 * every Apple Silicon GPU is built to. That is a Metal constraint and it does
 * not exist here. Measured on this GB10 with a standalone probe:
 *
 *     cudaDevAttrMaxSharedMemoryPerBlockOptin        101376
 *     cudaFuncSetAttribute(..., 99072) + launch      no error
 *     cudaFuncSetAttribute(..., 101376) + launch     no error
 *     cudaFuncSetAttribute(..., 131072)              invalid argument
 *
 * So the law's shapes WOULD run, and this file still carries the nine. The
 * reason is that the entry names, the lane counts and the threadgroup memory
 * are not this file's to choose: `crates/shiso-models/src/qwen35/kernels/
 * gdn_tune.rs` is the shape table both backends share, `launch::
 * RECURRENT_SMEM_BYTES` (1156) and `launch::CHUNK_SMEM_BYTES` (16896) are what
 * every launch reserves, and `shiso_engine::tune` probes the device at startup
 * and picks among the nine. A CUDA set of two entries at 66048 B and 99072 B
 * is a host redesign - a backend-conditional shape table, a backend-conditional
 * smem, and two of the GDN oracle cases left with a single variant to compare
 * against itself - rather than a kernel change.
 *
 * What the probe DID decide: on this device all nine lane counts (128 to 1024)
 * are launchable, so the tuner measures rather than falls back, and the two
 * hybrids are not a floor anybody needs - they are simply two more candidates.
 * Replacing the nine with the law's two remains available to a perf task, with
 * the probe's number as the evidence that it can be done. It is not available
 * to a correctness one, because the Metal arrangement is what produced the
 * committed goldens and what `crates/shiso-kerneltest/src/qwen35/gdn.rs`
 * grades.
 *
 * A COST THIS PORT PAYS AND THE METAL ONE DOES NOT, recorded because a later
 * measurement will find it: `dreg` in the chunk body is indexed by a runtime
 * token, so on CUDA it is a local array and local arrays with a dynamic index
 * live in local memory whatever their size. The `_c1..c8` variants therefore
 * do not get the register residency their Metal twins get, and are closer to
 * `_chunk_hybrid` than the names suggest. Correctness is unaffected - the
 * arithmetic and the address arithmetic are identical - and the fix is the
 * law's shared-memory `s_D`, which is the same redesign as above.
 *
 * ==========================================================================
 *
 * ONE DIVERGENCE FROM THE LAW, AND IT IS DELIBERATE. `q4e_gdn_out_gate_kernel`
 * applies a SIGMOID output gate; `gdn_conv.metal` applies SILU and says why:
 * "The two differ by a factor of the gate's own value, which is enough to turn
 * the model's output into punctuation: with the plain sigmoid this checkpoint
 * answers a factual prompt in fragments, and with `silu` it answers in
 * English. The C reference this family was read from applies a sigmoid, but it
 * describes a different member of the family - one with a different value-head
 * count - and it is not this checkpoint's gate." The Metal form is what
 * produced the goldens and is what is here.
 *
 * Value head `h` reads query/key head `h % n_head_k`: MODULO, not divide. The
 * attention family next door divides. They are ten lines apart in the source
 * both were extracted from and swapping them produces plausible garbage.
 */
#ifndef SHISO_Q35_GDN_CUH
#define SHISO_Q35_GDN_CUH

#include "q35_common.cuh"

/* Slots in the canonical four-slot unit. Four `float4` slots, sixteen floats,
 * and Q35_GDN_CSPLIT units tile one Q35_GDN_D row. */
#define Q35_GDN_UNIT_SLOTS 4u

/* The deltas of one block: one run of Q35_GDN_CHUNK tokens for each of the
 * Q35_GDN_D value columns. The host sizes the scratch as
 * `n_head_v * Q35_GDN_CHUNK * Q35_GDN_D` floats so two heads never share. */
#define Q35_GDN_CHUNK_SPILL (Q35_GDN_CHUNK * Q35_GDN_D)

/* --- the four small entries ---------------------------------------------- */

/* Causal depthwise conv over the q|k|v channels plus silu, one thread per
 * channel. Handles decode (n_tok 1) and chunked prefill: the window carries the
 * tail of the previous call, so a chunked prefill matches a single-shot one. */
extern "C" __global__ void kernel_q35_gdn_conv(
        const struct q35_args_gdn_conv args,
        float *__restrict__ out,
        float *__restrict__ state,
        const float *__restrict__ x,
        const float *__restrict__ w) {
    const unsigned c = blockIdx.x * 256u + threadIdx.x;
    if (c >= args.channels) {
        return;
    }
    const unsigned hist = args.conv_kernel - 1u;

    /* This channel's history, oldest first. */
    float win[8];
    for (unsigned i = 0u; i < hist; i++) {
        win[i] = state[(uint64_t) i * args.channels + c];
    }

    const float *wc = w + (uint64_t) c * args.conv_kernel;
    for (unsigned t = 0u; t < args.n_tok; t++) {
        const float cur = x[(uint64_t) t * args.channels + c];
        float acc = 0.0f;
        for (unsigned k = 0u; k < hist; k++) {
            acc += wc[k] * win[k];
        }
        acc += wc[hist] * cur;
        out[(uint64_t) t * args.channels + c] = q35_silu(acc);
        for (unsigned k = 0u; k + 1u < hist; k++) {
            win[k] = win[k + 1u];
        }
        if (hist > 0u) {
            win[hist - 1u] = cur;
        }
    }

    for (unsigned i = 0u; i < hist; i++) {
        state[(uint64_t) i * args.channels + c] = win[i];
    }
}

/* The convolution, also writing the window after every row but the last into
 * a trail of `n_tok - 1` records laid out as the window is. The same body as
 * `kernel_q35_gdn_conv` with the record written as the window moves. */
extern "C" __global__ void kernel_q35_gdn_conv_trail(
        const struct q35_args_gdn_conv args,
        float *__restrict__ out,
        float *__restrict__ state,
        const float *__restrict__ x,
        const float *__restrict__ w,
        float *__restrict__ trail) {
    const unsigned c = blockIdx.x * 256u + threadIdx.x;
    if (c >= args.channels) {
        return;
    }
    const unsigned hist = args.conv_kernel - 1u;
    float win[8];
    for (unsigned i = 0u; i < hist; i++) {
        win[i] = state[(uint64_t) i * args.channels + c];
    }
    const float *wc = w + (uint64_t) c * args.conv_kernel;
    for (unsigned t = 0u; t < args.n_tok; t++) {
        const float cur = x[(uint64_t) t * args.channels + c];
        float acc = 0.0f;
        for (unsigned k = 0u; k < hist; k++) {
            acc += wc[k] * win[k];
        }
        acc += wc[hist] * cur;
        out[(uint64_t) t * args.channels + c] = q35_silu(acc);
        for (unsigned k = 0u; k + 1u < hist; k++) {
            win[k] = win[k + 1u];
        }
        if (hist > 0u) {
            win[hist - 1u] = cur;
        }
        if (t + 1u < args.n_tok) {
            float *rec = trail + (uint64_t) t * hist * args.channels;
            for (unsigned i = 0u; i < hist; i++) {
                rec[(uint64_t) i * args.channels + c] = win[i];
            }
        }
    }
    for (unsigned i = 0u; i < hist; i++) {
        state[(uint64_t) i * args.channels + c] = win[i];
    }
}

/* In-place L2 norm of one head of one token. Not an RMS norm: no weight, no
 * mean, and the epsilon is added to the raw sum of squares. */
extern "C" __global__ void kernel_q35_gdn_l2norm(
        const struct q35_args_gdn_l2norm args,
        float *__restrict__ qk) {
    const unsigned t = blockIdx.y;
    const unsigned h = blockIdx.x;
    float *v = qk + (uint64_t) t * args.stride + args.offset
            + (uint64_t) h * args.head_dim;
    float *scratch = (float *) q35_smem;

    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < args.head_dim; i += 128u) {
        sum += v[i] * v[i];
    }
    const float total = q35_block_sum(sum, scratch, threadIdx.x, 128u);
    const float inv = rsqrtf(total + 1e-12f);
    for (unsigned i = threadIdx.x; i < args.head_dim; i += 128u) {
        v[i] *= inv;
    }
}

/* decay = softplus(alpha_proj + dt_bias) * ssm_a, beta = sigmoid(beta_proj).
 * ssm_a already holds -exp(A_log), so decay is negative and both consumers
 * exponentiate it directly. */
extern "C" __global__ void kernel_q35_gdn_gates(
        const struct q35_args_gdn_gates args,
        float *__restrict__ decay,
        float *__restrict__ beta,
        const float *__restrict__ alpha_proj,
        const float *__restrict__ beta_proj,
        const float *__restrict__ dt_bias,
        const float *__restrict__ a) {
    const uint64_t i = (uint64_t) blockIdx.x * 256u + threadIdx.x;
    if (i >= (uint64_t) args.n_head * args.n_tok) {
        return;
    }
    const unsigned h = (unsigned) (i % args.n_head);
    const float x = alpha_proj[i] + dt_bias[h];
    decay[i] = q35_softplus(x) * a[h];
    beta[i] = q35_sigmoid(beta_proj[i]);
}

/* Per-head RMSNorm of the delta-rule output, then the output gate.
 *
 * The norm weight is applied raw: it is the one weight the converter does not
 * offset by one. The gate is SILU, not sigmoid - see this file's header. */
extern "C" __global__ void kernel_q35_gdn_out_gate(
        const struct q35_args_gdn_out_gate args,
        float *__restrict__ out,
        const float *__restrict__ attn,
        const float *__restrict__ z,
        const float *__restrict__ w) {
    const unsigned group = blockIdx.x;
    const uint64_t base = (uint64_t) group * args.head_dim;
    float *scratch = (float *) q35_smem;

    float sum = 0.0f;
    for (unsigned i = threadIdx.x; i < args.head_dim; i += 128u) {
        const float v = attn[base + i];
        sum += v * v;
    }
    const float scale = rsqrtf(
            q35_block_sum(sum, scratch, threadIdx.x, 128u) / (float) args.head_dim + args.eps);
    for (unsigned i = threadIdx.x; i < args.head_dim; i += 128u) {
        out[base + i] = attn[base + i] * scale * w[i] * q35_silu(z[base + i]);
    }
}

/* --- the sequential recurrence ------------------------------------------- */

/* One body, several block widths, one summation order.
 *
 * The law keeps the 128x129 transposed state in shared memory. Here, as on the
 * Metal side, the same state lives in registers: SPLIT lanes share one value
 * column and lane (j, sub) owns SEG = 128 / SPLIT floats of that column's state
 * row, so a block of 128 * SPLIT lanes holds exactly one full state.
 *
 * What makes SPLIT a knob rather than a numerical choice is that the state row
 * is walked as four quarters of 32 and the rotation of each quarter is keyed to
 * the QUARTER index, never to the lane that happens to be walking it:
 *
 *   inner(q) = 32-FMA chain over quarter q from the rotated start (n + 8q) % 32
 *   total    = (inner(0) + inner(1)) + (inner(2) + inner(3))
 *
 * A lane that owns four quarters folds them with `q35_canonical4`; a lane that
 * owns two adds its pair and crosses one xor step; a lane that owns one crosses
 * two. All three are the same tree, so all three produce the same bits. */
/* The first quarter lane `tid` owns at a split of SPLIT lanes per column. */
template <unsigned SPLIT>
__device__ __forceinline__ unsigned q_first_of(unsigned tid) {
    return (tid % SPLIT) * (4u / SPLIT);
}

/* Writes a lane's sweep-ordered row back in natural order, a chunk of eight
 * as two sixteen-byte stores: chunk `c` of quarter `q` is natural chunk
 * `(c + q) & 3`. Used for the state at the end of the pass and for every
 * trail record, whose scalar stores were the other half of what made a
 * recording pass slow. */
template <unsigned SPLIT, unsigned SEG>
__device__ __forceinline__ void q35_gdn_store_row(float *out, const float *row, unsigned q_first) {
#pragma unroll
    for (unsigned qi = 0u; qi < 4u / SPLIT; qi++) {
        const unsigned q = q_first + qi;
#pragma unroll
        for (unsigned c = 0u; c < 4u; c++) {
            const float *v = row + qi * Q35_GDN_QUARTER + c * 8u;
            float4 *dst = (float4 *) (out + qi * Q35_GDN_QUARTER + ((c + q) & 3u) * 8u);
            dst[0] = make_float4(v[0], v[1], v[2], v[3]);
            dst[1] = make_float4(v[4], v[5], v[6], v[7]);
        }
    }
}

template <unsigned SPLIT, unsigned SEG>
__device__ __forceinline__ void q35_gdn_recurrent_impl(
        const struct q35_args_gdn_recurrent &args,
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta,
        float *trail) {
    const unsigned head_dim = args.head_dim;
    /* Both the register layout and the rotated sweep assume one quarter of
     * 128. */
    if (head_dim != Q35_GDN_D) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned h = blockIdx.x;
    /* Value head h reads query/key head h % n_head_k: modulo, not divide. */
    const unsigned hk = h % args.n_head_k;
    const unsigned seg = head_dim / SPLIT;
    const unsigned j = tid / SPLIT;
    const unsigned sub = tid % SPLIT;
    const unsigned i0 = sub * seg;
    /* 1, 2 or 4 lanes per column and nothing else: at SPLIT 8 `quarters` would
     * be 0, both quarter loops would be skipped, `part_k[0]` would be read
     * uninitialised and the combination below would fall into the two-xor arm -
     * a wrong answer with no diagnostic. */
    static_assert(SPLIT == 1u || SPLIT == 2u || SPLIT == 4u,
                  "q35_gdn_recurrent_impl is written for SPLIT in {1, 2, 4}");
    const unsigned quarters = 4u / SPLIT;
    const unsigned q_first = sub * quarters;
    const float scale = rsqrtf((float) head_dim);

    float *scratch = (float *) q35_smem;            /* Q35_BLOCK_SUM_SLOTS */
    float *s_k = scratch + Q35_BLOCK_SUM_SLOTS;
    float *s_q = scratch + Q35_BLOCK_SUM_SLOTS + Q35_GDN_D;

    /* One quarter of the transposed state per lane, in registers, held in
     * SWEEP order: `row[r0 + n]` is element `(n + 8q) & 31` of quarter `q`.
     * The rotation is by whole eight-element chunks, so it is a chunk
     * rotation - register indices stay compile-time constants and only the
     * chunk ADDRESSES of the load and the stores depend on the lane. Indexed
     * the natural way, `row[r0 + rot]` was a lane-dependent register index,
     * which put the whole row in local memory: every FMA of the recurrence
     * was a load from the stack, and a verify step spent 40 us a token a
     * layer here. The values and the order they are combined in are
     * unchanged. */
    float *S = state + (uint64_t) h * head_dim * head_dim
            + (uint64_t) j * head_dim + i0;
    float row[SEG];
#pragma unroll
    for (unsigned qi = 0u; qi < 4u / SPLIT; qi++) {
        const unsigned q = q_first_of<SPLIT>(tid) + qi;
#pragma unroll
        for (unsigned c = 0u; c < 4u; c++) {
            const float4 *src4 = (const float4 *) (S + qi * Q35_GDN_QUARTER
                                                   + ((c + q) & 3u) * 8u);
            const float4 a = src4[0];
            const float4 b = src4[1];
            float *dst = row + qi * Q35_GDN_QUARTER + c * 8u;
            dst[0] = a.x; dst[1] = a.y; dst[2] = a.z; dst[3] = a.w;
            dst[4] = b.x; dst[5] = b.y; dst[6] = b.z; dst[7] = b.w;
        }
    }
    for (unsigned t = 0u; t < args.n_tok; t++) {
        const float *base = qkv + (uint64_t) t * args.stride;
        const float *q_d = base + (uint64_t) hk * head_dim;
        const float *k_d = base + args.k_offset + (uint64_t) hk * head_dim;
        const float *v_d = base + args.v_offset + (uint64_t) h * head_dim;

        /* The sub == 0 lanes own one dimension each and publish k and q. */
        if (sub == 0u && j < head_dim) {
            s_k[j] = k_d[j];
            s_q[j] = q_d[j];
        }
        /* <k,q> is needed by every lane and its summation order has to be the
         * same one at every width. The block sum's tree is fixed by WHICH lane
         * holds which product, so the products are laid down on the first
         * Q35_GDN_D lanes - a dense prefix that does not move when SPLIT does -
         * and the reduction is told that length rather than the block's. */
        const float kq_part = (tid < Q35_GDN_D) ? (k_d[tid] * q_d[tid]) : 0.0f;
        const float s_kq = q35_block_sum(kq_part, scratch, tid, Q35_GDN_D);

        const uint64_t gi = (uint64_t) t * args.n_head_v + h;
        const float dec = __expf(decay[gi]);
        const float bt = beta[gi];

        float part_k[4];
        float part_q[4];
#pragma unroll
        for (unsigned qi = 0u; qi < 4u / SPLIT; qi++) {
            const unsigned q = q_first + qi;          /* quarter, not lane */
            const unsigned qbase = q * Q35_GDN_QUARTER;
            const unsigned r0 = qi * Q35_GDN_QUARTER;
            float sk_q = 0.0f;
            float sq_q = 0.0f;
#pragma unroll
            for (unsigned n = 0u; n < Q35_GDN_QUARTER; n++) {
                const unsigned rot = (n + 8u * q) & (Q35_GDN_QUARTER - 1u);
                const float mv = row[r0 + n];
                sk_q = q35_fma(mv, s_k[qbase + rot], sk_q);
                sq_q = q35_fma(mv, s_q[qbase + rot], sq_q);
            }
            part_k[qi] = sk_q;
            part_q[qi] = sq_q;
        }
        /* Combine what this lane holds, then cross the remaining lanes. Both
         * halves use the same pairing, so the tree is the same tree however the
         * quarters are distributed. SPLIT is a template parameter, so exactly
         * one arm survives compilation. */
        float sk;
        float sq;
        if (SPLIT == 1u) {
            sk = q35_canonical4(part_k[0], part_k[1], part_k[2], part_k[3]);
            sq = q35_canonical4(part_q[0], part_q[1], part_q[2], part_q[3]);
        } else if (SPLIT == 2u) {
            sk = part_k[0] + part_k[1];
            sq = part_q[0] + part_q[1];
            sk += __shfl_xor_sync(Q35_FULL_WARP, sk, 1);
            sq += __shfl_xor_sync(Q35_FULL_WARP, sq, 1);
        } else {
            sk = part_k[0];
            sq = part_q[0];
            sk += __shfl_xor_sync(Q35_FULL_WARP, sk, 1);
            sk += __shfl_xor_sync(Q35_FULL_WARP, sk, 2);
            sq += __shfl_xor_sync(Q35_FULL_WARP, sq, 1);
            sq += __shfl_xor_sync(Q35_FULL_WARP, sq, 2);
        }
        const float delta = (v_d[j] - dec * sk) * bt;

        if (sub == 0u) {
            attn_out[((uint64_t) t * args.n_head_v + h) * head_dim + j] =
                    (dec * sq + delta * s_kq) * scale;
        }
        /* The carry walks the same rotated per-quarter sequence as the read
         * above; anything else would pair a register with the wrong key. */
#pragma unroll
        for (unsigned qi = 0u; qi < 4u / SPLIT; qi++) {
            const unsigned q = q_first + qi;
            const unsigned qbase = q * Q35_GDN_QUARTER;
            const unsigned r0 = qi * Q35_GDN_QUARTER;
#pragma unroll
            for (unsigned n = 0u; n < Q35_GDN_QUARTER; n++) {
                const unsigned rot = (n + 8u * q) & (Q35_GDN_QUARTER - 1u);
                row[r0 + n] = q35_fma(row[r0 + n], dec, delta * s_k[qbase + rot]);
            }
        }
        /* The state after this row, into its record of the trail - every row
         * but the last, whose state is the one the state buffer keeps. */
        if (trail != nullptr && t + 1u < args.n_tok) {
            float *R = trail + (uint64_t) t * args.n_head_v * head_dim * head_dim
                    + (uint64_t) h * head_dim * head_dim + (uint64_t) j * head_dim + i0;
            q35_gdn_store_row<SPLIT, SEG>(R, row, q_first);
        }
        /* k and q are rewritten by the next token. */
        __syncthreads();
    }

    q35_gdn_store_row<SPLIT, SEG>(S, row, q_first);
}

#define Q35_GDN_RECURRENT_BODY(NAME, SPLIT, SEG, LANES)                       \
extern "C" __global__ void __launch_bounds__(LANES) NAME(                     \
        const struct q35_args_gdn_recurrent args,                             \
        float *attn_out, float *state,                                        \
        const float *qkv, const float *decay, const float *beta) {            \
    q35_gdn_recurrent_impl<SPLIT, SEG>(args, attn_out, state, qkv, decay, beta, nullptr); \
}

Q35_GDN_RECURRENT_BODY(kernel_q35_gdn_recurrent_s1, 1u, 128u, 128)
Q35_GDN_RECURRENT_BODY(kernel_q35_gdn_recurrent_s2, 2u, 64u, 256)
Q35_GDN_RECURRENT_BODY(kernel_q35_gdn_recurrent_s4, 4u, 32u, 512)

/* The same recurrence, also recording the state after every row but the last
 * into a trail of `n_tok - 1` records laid out as the state is: what a verify
 * pass needs to wind a session back to any accepted prefix. One body with the
 * plain entries, so a recorded pass and a plain one leave the same bits. */
#define Q35_GDN_RECURRENT_TRAIL_BODY(NAME, SPLIT, SEG, LANES)                 \
extern "C" __global__ void __launch_bounds__(LANES) NAME(                     \
        const struct q35_args_gdn_recurrent args,                             \
        float *attn_out, float *state,                                        \
        const float *qkv, const float *decay, const float *beta,              \
        float *trail) {                                                       \
    q35_gdn_recurrent_impl<SPLIT, SEG>(args, attn_out, state, qkv, decay, beta, trail); \
}

Q35_GDN_RECURRENT_TRAIL_BODY(kernel_q35_gdn_recurrent_trail_s1, 1u, 128u, 128)
Q35_GDN_RECURRENT_TRAIL_BODY(kernel_q35_gdn_recurrent_trail_s2, 2u, 64u, 256)
Q35_GDN_RECURRENT_TRAIL_BODY(kernel_q35_gdn_recurrent_trail_s4, 4u, 32u, 512)

/* The always-fits recurrent variant: quarter 0 of each lane's state row staged
 * in shared memory, quarters 1, 2 and 3 in registers, 128 lanes.
 *
 * It is the SPLIT = 1 body with one thing changed: WHERE one of the four
 * quarters lives. The staged span is per-lane private - lane `tid` owns exactly
 * `staged[tid * 32 .. tid * 32 + 32)` - so no barrier guards it beyond the one
 * the token loop already takes for `s_k`/`s_q`.
 *
 * On this device it is a candidate rather than a floor: every register-resident
 * width launches here. It is kept because the shape table is shared with Metal,
 * where it is the shape a device that cannot afford the 128-float row falls back
 * to, and because the Metal measurements have it winning at n_tok = 32. */
extern "C" __global__ void __launch_bounds__(128) kernel_q35_gdn_recurrent_hybrid(
        const struct q35_args_gdn_recurrent args,
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta) {
    const unsigned head_dim = args.head_dim;
    if (head_dim != Q35_GDN_D) {
        return;
    }
    const unsigned tid = threadIdx.x;
    const unsigned h = blockIdx.x;
    const unsigned hk = h % args.n_head_k;
    const unsigned j = tid;
    const float scale = rsqrtf((float) head_dim);

    float *scratch = (float *) q35_smem;
    float *s_k = scratch + Q35_BLOCK_SUM_SLOTS;
    float *s_q = scratch + Q35_BLOCK_SUM_SLOTS + Q35_GDN_D;
    /* The staged quarter follows the scratch, k and q rows. */
    float *my_staged = scratch + Q35_GDN_REC_SLOTS + tid * Q35_GDN_QUARTER;

    float *S = state + (uint64_t) h * head_dim * head_dim + (uint64_t) j * head_dim;
    for (unsigned n = 0u; n < Q35_GDN_QUARTER; n++) {
        my_staged[n] = S[n];
    }
    float row_hi[96];
    for (unsigned n = 0u; n < 96u; n++) {
        row_hi[n] = S[Q35_GDN_QUARTER + n];
    }
    for (unsigned t = 0u; t < args.n_tok; t++) {
        const float *base = qkv + (uint64_t) t * args.stride;
        const float *q_d = base + (uint64_t) hk * head_dim;
        const float *k_d = base + args.k_offset + (uint64_t) hk * head_dim;
        const float *v_d = base + args.v_offset + (uint64_t) h * head_dim;

        if (j < head_dim) {
            s_k[j] = k_d[j];
            s_q[j] = q_d[j];
        }
        const float kq_part = (tid < Q35_GDN_D) ? (k_d[tid] * q_d[tid]) : 0.0f;
        const float s_kq = q35_block_sum(kq_part, scratch, tid, Q35_GDN_D);

        const uint64_t gi = (uint64_t) t * args.n_head_v + h;
        const float dec = __expf(decay[gi]);
        const float bt = beta[gi];

        float part_k[4];
        float part_q[4];
        for (unsigned qi = 0u; qi < 4u; qi++) {
            const unsigned qbase = qi * Q35_GDN_QUARTER;
            float sk_q = 0.0f;
            float sq_q = 0.0f;
            for (unsigned n = 0u; n < Q35_GDN_QUARTER; n++) {
                const unsigned rot = (n + 8u * qi) & (Q35_GDN_QUARTER - 1u);
                const float mv = (qi == 0u) ? my_staged[rot]
                                            : row_hi[(qi - 1u) * Q35_GDN_QUARTER + rot];
                sk_q = q35_fma(mv, s_k[qbase + rot], sk_q);
                sq_q = q35_fma(mv, s_q[qbase + rot], sq_q);
            }
            part_k[qi] = sk_q;
            part_q[qi] = sq_q;
        }
        const float sk = q35_canonical4(part_k[0], part_k[1], part_k[2], part_k[3]);
        const float sq = q35_canonical4(part_q[0], part_q[1], part_q[2], part_q[3]);
        const float delta = (v_d[j] - dec * sk) * bt;

        attn_out[((uint64_t) t * args.n_head_v + h) * head_dim + j] =
                (dec * sq + delta * s_kq) * scale;

        for (unsigned qi = 0u; qi < 4u; qi++) {
            const unsigned qbase = qi * Q35_GDN_QUARTER;
            for (unsigned n = 0u; n < Q35_GDN_QUARTER; n++) {
                const unsigned rot = (n + 8u * qi) & (Q35_GDN_QUARTER - 1u);
                const float delta_k = delta * s_k[qbase + rot];
                if (qi == 0u) {
                    my_staged[rot] = q35_fma(my_staged[rot], dec, delta_k);
                } else {
                    const unsigned idx = (qi - 1u) * Q35_GDN_QUARTER + rot;
                    row_hi[idx] = q35_fma(row_hi[idx], dec, delta_k);
                }
            }
        }
        __syncthreads();
    }

    for (unsigned n = 0u; n < Q35_GDN_QUARTER; n++) {
        S[n] = my_staged[n];
    }
    for (unsigned n = 0u; n < 96u; n++) {
        S[Q35_GDN_QUARTER + n] = row_hi[n];
    }
}

/* --- the chunked-parallel form ------------------------------------------- */

/* Fold this lane's per-unit partials into the total for its value column.
 *
 * Unit index `u = ui * CSPLIT + sub`, so `sub` occupies the low bits of `u` and
 * `ui` the high ones. The reference combination at CSPLIT = 8 is a
 * recursive-doubling xor butterfly over the eight units, pairing by bit 0, then
 * bit 1, then bit 2; splitting that by where the bits live gives the two loops
 * below, in this order and only in this order. `part` is clobbered.
 *
 * `crates/shiso-kerneltest/src/qwen35/chunk_fold.rs` replays this partition as
 * host index arithmetic and runs on every machine. */
template <unsigned CSPLIT>
__device__ __forceinline__ float q35_gdn_chunk_fold(float *part) {
    const unsigned units = Q35_GDN_CSPLIT / CSPLIT;
    for (unsigned ui = 0u; ui < units; ui++) {
        for (unsigned o = 1u; o < CSPLIT; o <<= 1u) {
            part[ui] += __shfl_xor_sync(Q35_FULL_WARP, part[ui], o);
        }
    }
    for (unsigned step = 1u; step < units; step <<= 1u) {
        for (unsigned i = 0u; i + step < units; i += 2u * step) {
            part[i] += part[i + step];
        }
    }
    return part[0];
}

/* The chunk body, parameterized on the lane split and on where the deltas live.
 *
 * `dreg` is the per-column delta store: a pointer into the caller's local array
 * for the register variants, or into a device scratch for the hybrid, so the
 * body is written once and the hybrid cannot drift from it. `DSTRIDE` is the
 * distance between two of this lane's delta slots - 1 for the local array, and
 * the row pitch for the device scratch, where laying the deltas out as
 * [token][column] is what keeps the 128 lanes of a CSPLIT = 1 block reading one
 * contiguous span instead of 128 lines. */
template <unsigned CSPLIT, unsigned DSTRIDE>
__device__ __forceinline__ void q35_gdn_chunk_impl(
        const struct q35_args_gdn_recurrent &args,
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta,
        float *dreg) {
    /* 1, 2, 4 or 8 lanes per value column, and nothing else. At CSPLIT 16
     * `units` would be 0, every unit loop would be skipped, the fold would read
     * part[0] uninitialised and the answer would be wrong with no diagnostic. */
    static_assert(CSPLIT == 1u || CSPLIT == 2u || CSPLIT == 4u || CSPLIT == 8u,
                  "q35_gdn_chunk_impl is written for CSPLIT in {1, 2, 4, 8}");
    const unsigned tid = threadIdx.x;
    const unsigned D = args.head_dim;
    /* The four-slot unit is sixteen floats, an eighth of a 128-float row. At
     * any other head_dim a lane would own a fractional number of units and
     * there would be no CSPLIT = 8 order to reproduce. */
    if (D != Q35_GDN_D) {
        return;
    }
    const unsigned C = Q35_GDN_CHUNK;
    const unsigned n_tok = args.n_tok;
    const unsigned stride = args.stride;
    const unsigned h = blockIdx.x;
    const unsigned hk = h % args.n_head_k;
    const unsigned n_head_v = args.n_head_v;
    const unsigned j = tid / CSPLIT;
    const unsigned sub = tid % CSPLIT;
    const unsigned lane = tid & 31u;
    const unsigned warp = tid >> 5u;
    const unsigned n_warp = blockDim.x >> 5u;
    const float scale = rsqrtf((float) D);
    const unsigned units = Q35_GDN_CSPLIT / CSPLIT;

    float *smem = (float *) q35_smem;
    float *s_A = smem;              /* [C][C] */
    float *s_G = smem + C * C;      /* inclusive prefix sum of log decay */
    float *s_b = s_G + C;

    const float *k_base = qkv + args.k_offset + (uint64_t) hk * D;
    const float *q_base = qkv + (uint64_t) hk * D;
    const float *v_base = qkv + args.v_offset;

    /* This lane's slice of the state: `units` four-slot units of its value
     * column's row, each walked in the order that unit's rotation gives, so the
     * sequence belongs to the UNIT and not to the lane that happens to hold it.
     * The slots are loaded component-wise rather than as a `float4` because the
     * host may hand out a buffer view at any 4-byte offset. */
    float *m_row = state + ((uint64_t) h * D + j) * D;
    float4 sreg[Q35_GDN_UNIT_SLOTS * (Q35_GDN_CSPLIT / CSPLIT)];
    for (unsigned ui = 0u; ui < units; ui++) {
        const unsigned usub = ui * CSPLIT + sub;
        const unsigned urot = usub / 2u;
        for (unsigned n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
            const unsigned slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
            const float *m = m_row + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
            sreg[ui * Q35_GDN_UNIT_SLOTS + n4] = make_float4(m[0u], m[1u], m[2u], m[3u]);
        }
    }

    for (unsigned t0 = 0u; t0 < n_tok; t0 += C) {
        const unsigned n = (n_tok - t0 < C) ? (n_tok - t0) : C;

        /* Gates in parallel, then the inclusive prefix sum of the log decays by
         * one thread over shared memory: 64 dependent adds are cheaper than a
         * scan's barriers, but 64 strided global loads on that serial path are
         * not. The strided handout is a load, so the width it is strided by does
         * not reach the arithmetic. */
        for (unsigned t = tid; t < n; t += blockDim.x) {
            const uint64_t gi = (uint64_t) (t0 + t) * n_head_v + h;
            s_G[t] = decay[gi];
            s_b[t] = beta[gi];
        }
        __syncthreads();
        if (tid == 0u) {
            float acc = 0.0f;
            for (unsigned t = 0u; t < n; t++) {
                acc += s_G[t];
                s_G[t] = acc;
            }
        }
        __syncthreads();

        /* A[t][u] = b_t e^{G_t - G_u} (k_u . k_t), strictly below the diagonal.
         * One warp per row, rows handed out low/high so a warp that gets both
         * halves of the run costs the same. The reduction is `q35_warp_sum` over
         * one warp, so the value of an entry does not depend on how many warps
         * the block has; only on which warp computes it, and that is a schedule,
         * not a sum. */
        for (unsigned rr = warp; rr < C; rr += n_warp) {
            const unsigned t = ((rr / n_warp) % 2u == 0u) ? rr : (C - 1u - (rr - n_warp));
            if (t >= n) {
                continue;
            }
            const float *k_t = k_base + (uint64_t) (t0 + t) * stride;
            const float bt = s_b[t];
            const float gt = s_G[t];
            for (unsigned u = 0u; u < t; u++) {
                const float *k_u = k_base + (uint64_t) (t0 + u) * stride;
                const unsigned i = 4u * lane;
                float dot = k_t[i] * k_u[i];
                dot += k_t[i + 1u] * k_u[i + 1u];
                dot += k_t[i + 2u] * k_u[i + 2u];
                dot += k_t[i + 3u] * k_u[i + 3u];
                dot = q35_warp_sum(dot);
                if (lane == 0u) {
                    s_A[t * C + u] = bt * __expf(gt - s_G[u]) * dot;
                }
            }
        }
        __syncthreads();

        /* rhs_t = b_t (v_t - e^{G_t} S0^T k_t). One four-slot chain per unit
         * this lane owns, each with its four `float4` components kept as four
         * partials, so the grouping the law gets from `float4` arithmetic is
         * reproduced rather than re-associated; then the units fold in the
         * butterfly's order. */
        for (unsigned t = 0u; t < n; t++) {
            const float *k_row = k_base + (uint64_t) (t0 + t) * stride;
            const float *v_row = v_base + (uint64_t) (t0 + t) * stride + (uint64_t) h * D;
            float part[Q35_GDN_CSPLIT / CSPLIT];
            for (unsigned ui = 0u; ui < units; ui++) {
                const unsigned usub = ui * CSPLIT + sub;
                const unsigned urot = usub / 2u;
                float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                for (unsigned n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
                    const unsigned slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
                    const float *k4 = k_row + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
                    const float4 sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
                    p0 = q35_fma(sv.x, k4[0u], p0);
                    p1 = q35_fma(sv.y, k4[1u], p1);
                    p2 = q35_fma(sv.z, k4[2u], p2);
                    p3 = q35_fma(sv.w, k4[3u], p3);
                }
                part[ui] = q35_canonical4(p0, p1, p2, p3);
            }
            const float p = q35_gdn_chunk_fold<CSPLIT>(part);
            if (sub == (t & (CSPLIT - 1u))) {
                dreg[(t / CSPLIT) * DSTRIDE] = s_b[t] * (v_row[j] - p * __expf(s_G[t]));
            }
        }

        /* Forward substitution: d_t = rhs_t - sum_{u<t} A[t][u] d_u. Lane `sub`
         * owns the deltas for t == sub (mod CSPLIT), which is exactly the set
         * this loop reads back for this lane, so the warp barrier the
         * shared-memory version needs between the steps has nothing left to
         * order and is gone rather than translated.
         *
         * The chains are the CSPLIT = 8 chains: one per unit index modulo
         * Q35_GDN_CSPLIT, never one long chain over the lane's whole residue
         * class. */
        for (unsigned t = 1u; t < n; t++) {
            float part[Q35_GDN_CSPLIT / CSPLIT];
            for (unsigned ui = 0u; ui < units; ui++) {
                const unsigned usub = ui * CSPLIT + sub;
                float acc = 0.0f;
                for (unsigned u = usub; u < t; u += Q35_GDN_CSPLIT) {
                    acc = q35_fma(s_A[t * C + u], dreg[(u / CSPLIT) * DSTRIDE], acc);
                }
                part[ui] = acc;
            }
            const float acc = q35_gdn_chunk_fold<CSPLIT>(part);
            if (sub == (t & (CSPLIT - 1u))) {
                dreg[(t / CSPLIT) * DSTRIDE] -= acc;
            }
        }

        /* The tile below is staged into the same `s_A` the substitution has just
         * finished reading, so a barrier is needed between the two phases. */
        __syncthreads();

        /* Outputs, one query tile at a time. */
        for (unsigned tb = 0u; tb < n; tb += Q35_GDN_QTILE) {
            const unsigned nt = (n - tb < Q35_GDN_QTILE) ? (n - tb) : Q35_GDN_QTILE;
            for (unsigned pair = warp; pair < nt * C; pair += n_warp) {
                const unsigned tt = pair / C;
                const unsigned u = pair - tt * C;
                const unsigned t = tb + tt;
                float v = 0.0f;
                if (u <= t) {
                    const float *q_row = q_base + (uint64_t) (t0 + t) * stride;
                    const float *k_u = k_base + (uint64_t) (t0 + u) * stride;
                    const unsigned i = 4u * lane;
                    float dot = q_row[i] * k_u[i];
                    dot += q_row[i + 1u] * k_u[i + 1u];
                    dot += q_row[i + 2u] * k_u[i + 2u];
                    dot += q_row[i + 3u] * k_u[i + 3u];
                    dot = q35_warp_sum(dot);
                    v = __expf(s_G[t] - s_G[u]) * dot;
                }
                if (lane == 0u) {
                    s_A[tt * C + u] = v;
                }
            }
            __syncthreads();

            /* The state term and the delta term of one unit meet in a single
             * fma BEFORE anything crosses, so both have to belong to the same
             * unit index: the interleaved handout is what puts the unit's
             * four-slot chain and the deltas of `u == usub (mod 8)` in the same
             * lane. */
            for (unsigned tt = 0u; tt < nt; tt++) {
                const unsigned t = tb + tt;
                const float *q_row = q_base + (uint64_t) (t0 + t) * stride;
                const float eg = __expf(s_G[t]);
                float part[Q35_GDN_CSPLIT / CSPLIT];
                for (unsigned ui = 0u; ui < units; ui++) {
                    const unsigned usub = ui * CSPLIT + sub;
                    const unsigned urot = usub / 2u;
                    float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
                    for (unsigned n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
                        const unsigned slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
                        const float *q4 = q_row + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
                        const float4 sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
                        p0 = q35_fma(sv.x, q4[0u], p0);
                        p1 = q35_fma(sv.y, q4[1u], p1);
                        p2 = q35_fma(sv.z, q4[2u], p2);
                        p3 = q35_fma(sv.w, q4[3u], p3);
                    }
                    const float p = q35_canonical4(p0, p1, p2, p3);
                    float acc = 0.0f;
                    for (unsigned u = usub; u <= t; u += Q35_GDN_CSPLIT) {
                        acc = q35_fma(s_A[tt * C + u], dreg[(u / CSPLIT) * DSTRIDE], acc);
                    }
                    part[ui] = q35_fma(p, eg, acc);
                }
                const float acc = q35_gdn_chunk_fold<CSPLIT>(part);
                if (sub == 0u) {
                    attn_out[((uint64_t) (t0 + t) * n_head_v + h) * D + j] = acc * scale;
                }
            }
            __syncthreads();
        }

        /* Carry: S <- e^{G_last} S0 + sum_t e^{G_last - G_t} k_t d_t^T. The delta
         * for token t lives in the lane with sub == t % CSPLIT, so it is read
         * back through a broadcast instead of a shared array; the broadcast stays
         * inside the CSPLIT lanes of one value column, so the shuffle only
         * rewrites the low bits of the lane index. */
        const float g_last = s_G[n - 1u];
        const float dec = __expf(g_last);
        for (unsigned i = 0u; i < Q35_GDN_UNIT_SLOTS * units; i++) {
            sreg[i].x *= dec;
            sreg[i].y *= dec;
            sreg[i].z *= dec;
            sreg[i].w *= dec;
        }
        for (unsigned t = 0u; t < n; t++) {
            const unsigned owner = (lane & ~(CSPLIT - 1u)) | (t & (CSPLIT - 1u));
            const float d = __shfl_sync(Q35_FULL_WARP, dreg[(t / CSPLIT) * DSTRIDE], (int) owner)
                    * __expf(g_last - s_G[t]);
            const float *k_row = k_base + (uint64_t) (t0 + t) * stride;
            for (unsigned ui = 0u; ui < units; ui++) {
                const unsigned usub = ui * CSPLIT + sub;
                const unsigned urot = usub / 2u;
                for (unsigned n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
                    const unsigned slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
                    const float *k4 = k_row + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
                    float4 &sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
                    sv.x = q35_fma(k4[0u], d, sv.x);
                    sv.y = q35_fma(k4[1u], d, sv.y);
                    sv.z = q35_fma(k4[2u], d, sv.z);
                    sv.w = q35_fma(k4[3u], d, sv.w);
                }
            }
        }
        /* The next run rewrites s_G, s_b and s_A while the carry above is still
         * reading s_G and the last tile is still reading s_A. */
        __syncthreads();
    }

    for (unsigned ui = 0u; ui < units; ui++) {
        const unsigned usub = ui * CSPLIT + sub;
        const unsigned urot = usub / 2u;
        for (unsigned n4 = 0u; n4 < Q35_GDN_UNIT_SLOTS; n4++) {
            const unsigned slot = (n4 + urot) & (Q35_GDN_UNIT_SLOTS - 1u);
            float *m = m_row + 4u * (uint64_t) (usub * Q35_GDN_UNIT_SLOTS + slot);
            const float4 sv = sreg[ui * Q35_GDN_UNIT_SLOTS + n4];
            m[0u] = sv.x;
            m[1u] = sv.y;
            m[2u] = sv.z;
            m[3u] = sv.w;
        }
    }
}

#define Q35_GDN_CHUNK_BODY(NAME, CSPLIT, LANES)                               \
extern "C" __global__ void __launch_bounds__(LANES) NAME(                     \
        const struct q35_args_gdn_recurrent args,                             \
        float *attn_out, float *state,                                        \
        const float *qkv, const float *decay, const float *beta) {            \
    float dreg[Q35_GDN_CHUNK / (CSPLIT)];                                     \
    q35_gdn_chunk_impl<CSPLIT, 1u>(args, attn_out, state, qkv, decay, beta, dreg); \
}

Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c1, 1u, 128)
Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c2, 2u, 256)
Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c4, 4u, 512)
Q35_GDN_CHUNK_BODY(kernel_q35_gdn_chunk_c8, 8u, 1024)

/* The CSPLIT = 1 body with its deltas in device memory: the seventh buffer, and
 * the only delta-rule entry point that is not the six-buffer one.
 *
 * The row is laid out [token][column] rather than [column][token]: the 128 lanes
 * read the delta of the same token at the same moment in the substitution and
 * the output phases, so that order turns 128 scattered loads into one contiguous
 * span. It does not change the arithmetic - it is the same `DSTRIDE` template
 * argument the register variants pass as 1. */
extern "C" __global__ void __launch_bounds__(128) kernel_q35_gdn_chunk_hybrid(
        const struct q35_args_gdn_recurrent args,
        float *attn_out, float *state,
        const float *qkv, const float *decay, const float *beta,
        float *dreg_spill) {
    float *dreg = dreg_spill + (uint64_t) blockIdx.x * Q35_GDN_CHUNK_SPILL + threadIdx.x;
    q35_gdn_chunk_impl<1u, Q35_GDN_D>(args, attn_out, state, qkv, decay, beta, dreg);
}

#endif /* SHISO_Q35_GDN_CUH */
