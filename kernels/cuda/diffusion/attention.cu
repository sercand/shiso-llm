/* Attention entries of the `diffusion` module: the per-row joint attention
 * (any head width), the fused flash-style joint attention (heads <= 128 in
 * multiples of 8, plain and causal, generic and fixed-128), and the causal
 * grouped-query attention of the text encoder.
 *
 * Include-only piece of `diffusion.cu`; ported from the Metal twin.
 *
 * Row layout (the host's fused-row contract): head `h` of a row starts at
 * `row * stride + offset + h * head_dim`, each operand with its OWN offset
 * (the transformer binds one fused qkv buffer as q, k and v with offsets 0,
 * heads*head_dim, 2*heads*head_dim; reading k or v at the q offset would
 * silently compute softmax(q q^T) q). Joint key order is [txt, img, cache];
 * softmax is order-invariant. When cache_tokens == 0 the cache bindings are a
 * 4-byte stand-in and are never read.
 */

/* Where head `h` of joint row `t` starts for one operand. */
static __device__ __forceinline__ const float* dit_joint_row(
        const dit_args_sdpa& args, u32 t, u32 h, u32 offset,
        const float* txt, const float* img, const float* cache) {
    const u32 hd = args.head_dim;
    if (t < args.txt_tokens) {
        return txt + (u64)t * args.txt_stride + offset + h * hd;
    }
    if (t < args.txt_tokens + args.img_tokens) {
        return img + (u64)(t - args.txt_tokens) * args.img_stride + offset + h * hd;
    }
    return cache + (u64)(t - args.txt_tokens - args.img_tokens) * args.cache_stride
            + offset + h * hd;
}

/* One dot of a query head against a key head, reduced across the 64-thread
 * block (every thread gets the total). */
static __device__ __forceinline__ float dit_head_dot(
        const float* q, const float* k, u32 head_dim, u32 tid, float* scratch) {
    float part = 0.0f;
    for (u32 d = tid; d < head_dim; d += DIT_BLOCK) {
        part += q[d] * k[d];
    }
    return dit_block_sum(part, scratch, tid);
}

/* Per-row joint attention, grid (rows, heads) x 64: one block per (query row,
 * head). Query rows below txt_tokens come from the txt stream and write
 * out_txt, the rest from img and write out_img. Two passes over the keys: the
 * running maximum, then exponentials, their sum and the value mix (every
 * thread recomputes the same block-broadcast scores; each thread owns head
 * dims tid + 64*slot). Head width is bounded at 64 * DIT_ATTN_SLOTS = 512. */
extern "C" __global__ void kernel_dit_sdpa_f32(
        const dit_args_sdpa args, float* out_txt, float* out_img,
        const float* q_txt, const float* k_txt, const float* v_txt,
        const float* q_img, const float* k_img, const float* v_img,
        const float* cache_k, const float* cache_v) {
    __shared__ float scratch[3];
    const u32 t = blockIdx.x;
    const u32 h = blockIdx.y;
    const u32 tid = threadIdx.x;
    if (t >= args.txt_tokens + args.img_tokens || h >= args.heads) {
        return;
    }
    const u32 hd = args.head_dim;
    const u32 keys = args.txt_tokens + args.img_tokens + args.cache_tokens;
    const float* q = dit_joint_row(args, t, h, args.q_offset, q_txt, q_img, q_txt);
    const float scale = 1.0f / sqrtf((float)hd);

    float max_score = -DIT_INF;
    for (u32 s = 0u; s < keys; s++) {
        const float* k = dit_joint_row(args, s, h, args.k_offset, k_txt, k_img, cache_k);
        const float score = dit_head_dot(q, k, hd, tid, scratch) * scale;
        max_score = fmaxf(max_score, score);
    }

    float acc[DIT_ATTN_SLOTS];
#pragma unroll
    for (u32 slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
        acc[slot] = 0.0f;
    }
    float sum = 0.0f;
    for (u32 s = 0u; s < keys; s++) {
        const float* k = dit_joint_row(args, s, h, args.k_offset, k_txt, k_img, cache_k);
        const float score = dit_head_dot(q, k, hd, tid, scratch) * scale;
        const float p = expf(score - max_score);
        sum += p;
        const float* v = dit_joint_row(args, s, h, args.v_offset, v_txt, v_img, cache_v);
#pragma unroll
        for (u32 slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
            const u32 d = tid + slot * DIT_BLOCK;
            if (d < hd) {
                acc[slot] += p * v[d];
            }
        }
    }

    const float inv = 1.0f / sum;
    const u32 width = args.heads * hd;
    float* dst;
    u64 base;
    if (t < args.txt_tokens) {
        dst = out_txt;
        base = (u64)t * width + h * hd;
    } else {
        dst = out_img;
        base = (u64)(t - args.txt_tokens) * width + h * hd;
    }
#pragma unroll
    for (u32 slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
        const u32 d = tid + slot * DIT_BLOCK;
        if (d < hd) {
            dst[base + d] = acc[slot] * inv;
        }
    }
}

/* Causal grouped-query attention over packed rows (text encoder prefill),
 * grid (tokens, q_heads) x 64: query head h reads kv head h / (q_heads /
 * kv_heads) and attends to keys 0..=t of which only those below valid_keys
 * count (a padded prompt's pad rows attend every real key and no pad key).
 * Rows are (token, head, head_dim) packed; kv rows use kv_heads. */
extern "C" __global__ void kernel_dit_gqa_f32(
        const dit_args_gqa args, float* out, const float* q, const float* k,
        const float* v) {
    __shared__ float scratch[3];
    const u32 t = blockIdx.x;
    const u32 h = blockIdx.y;
    const u32 tid = threadIdx.x;
    if (t >= args.tokens || h >= args.q_heads) {
        return;
    }
    const u32 hd = args.head_dim;
    const u32 group = args.q_heads / args.kv_heads;
    const u32 kvh = h / group;
    const u64 qrow = ((u64)t * args.q_heads + h) * hd;
    const float scale = 1.0f / sqrtf((float)hd);

    const u32 seen = min(t + 1u, args.valid_keys);
    float max_score = -DIT_INF;
    for (u32 s = 0u; s < seen; s++) {
        const u64 krow = ((u64)s * args.kv_heads + kvh) * hd;
        const float score = dit_head_dot(q + qrow, k + krow, hd, tid, scratch) * scale;
        max_score = fmaxf(max_score, score);
    }

    float acc[DIT_ATTN_SLOTS];
#pragma unroll
    for (u32 slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
        acc[slot] = 0.0f;
    }
    float sum = 0.0f;
    for (u32 s = 0u; s < seen; s++) {
        const u64 krow = ((u64)s * args.kv_heads + kvh) * hd;
        const float score = dit_head_dot(q + qrow, k + krow, hd, tid, scratch) * scale;
        const float p = expf(score - max_score);
        sum += p;
#pragma unroll
        for (u32 slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
            const u32 d = tid + slot * DIT_BLOCK;
            if (d < hd) {
                acc[slot] += p * v[krow + d];
            }
        }
    }
    const float inv = 1.0f / sum;
#pragma unroll
    for (u32 slot = 0u; slot < DIT_ATTN_SLOTS; slot++) {
        const u32 d = tid + slot * DIT_BLOCK;
        if (d < hd) {
            out[qrow + d] = acc[slot] * inv;
        }
    }
}

/* Fused joint attention with an online softmax: the same contract, bindings
 * and output as kernel_dit_sdpa_f32, for head_dim % 8 == 0 and head_dim <= 128.
 *
 * Grid (ceil(rows/64), heads) x 256. A block takes 64 query rows of one head,
 * FOUR THREADS PER ROW: thread (row = tid/4, part = tid%4) owns head dims
 * part + 4i (i < head_dim/4), so a row's partial dots are combined with two
 * xor-shuffles (lanes 4r..4r+3 are one aligned group of a warp) and all four
 * threads then hold the same score and the same running max/sum. Keys stream
 * 32 at a time through static shared memory as f32 (K and V tiles, 32 KiB);
 * q and the output accumulator live in registers (32 floats each). The Metal
 * twin stages half and uses simdgroup matrices; this one is f32 throughout, so
 * it is the more accurate of the two and the slower one, and is the first
 * candidate for a tensor-core rewrite.
 *
 * Every thread of the block must reach the barriers and shuffles, so rows past
 * the end carry a zero query and are simply not written. CAUSAL masks every key
 * after the query's own joint row (key > t) and stops streaming at the tile's
 * last row; the host sends it no cached rows. Keys past `keys` are masked, so
 * ragged lengths need no special case. */
template <bool CAUSAL, u32 HD>
static __device__ __forceinline__ void dit_flash_sdpa(
        const dit_args_sdpa& args, float* out_txt, float* out_img,
        const float* q_txt, const float* k_txt, const float* v_txt,
        const float* q_img, const float* k_img, const float* v_img,
        const float* cache_k, const float* cache_v) {
    constexpr u32 QR = DIT_FA_QROWS, KB = DIT_FA_KEYS, NT = DIT_FA_THREADS;
    constexpr u32 MAXD = DIT_FA_MAX_D;
    constexpr u32 MAX4 = MAXD / 4u;
    __shared__ float sk[KB * MAXD];
    __shared__ float sv[KB * MAXD];

    const u32 h = blockIdx.y;
    if (h >= args.heads) {
        return; /* uniform across the block */
    }
    const u32 q0 = blockIdx.x * QR;
    const u32 tid = threadIdx.x;
    const u32 hd = HD == 0u ? args.head_dim : HD;
    const u32 hd4 = hd / 4u;
    const u32 rows = args.txt_tokens + args.img_tokens;
    const u32 keys = CAUSAL ? min(rows, q0 + QR) : rows + args.cache_tokens;
    const float scale = 1.0f / sqrtf((float)hd);

    const u32 part = tid % 4u;
    const u32 qrow = q0 + tid / 4u;
    const bool live = qrow < rows;

    float qv[MAX4];
    float o[MAX4];
    const float* qp = live ? dit_joint_row(args, qrow, h, args.q_offset, q_txt, q_img, q_txt)
                           : q_txt;
#pragma unroll
    for (u32 i = 0; i < MAX4; ++i) {
        qv[i] = (live && i < hd4) ? qp[part + 4u * i] * scale : 0.0f;
        o[i] = 0.0f;
    }
    float run_max = -DIT_INF;
    float run_sum = 0.0f;

    for (u32 s0 = 0u; s0 < keys; s0 += KB) {
        __syncthreads(); /* the previous block's reads of sk/sv are done */
        for (u32 e = tid; e < KB * hd; e += NT) {
            const u32 j = e / hd;
            const u32 d = e - j * hd;
            const u32 s = s0 + j;
            float kval = 0.0f;
            float vval = 0.0f;
            if (s < keys) {
                kval = dit_joint_row(args, s, h, args.k_offset, k_txt, k_img, cache_k)[d];
                vval = dit_joint_row(args, s, h, args.v_offset, v_txt, v_img, cache_v)[d];
            }
            sk[j * MAXD + d] = kval;
            sv[j * MAXD + d] = vval;
        }
        __syncthreads();

        float sc[KB];
        float block_max = -DIT_INF;
#pragma unroll
        for (u32 j = 0; j < KB; ++j) {
            float p = 0.0f;
#pragma unroll
            for (u32 i = 0; i < MAX4; ++i) {
                if (i < hd4) {
                    p += qv[i] * sk[j * MAXD + part + 4u * i];
                }
            }
            p += __shfl_xor_sync(0xffffffffu, p, 1);
            p += __shfl_xor_sync(0xffffffffu, p, 2);
            const u32 key = s0 + j;
            if (key >= keys || (CAUSAL && key > qrow)) {
                p = -DIT_INF;
            }
            sc[j] = p;
            block_max = fmaxf(block_max, p);
        }
        const float new_max = fmaxf(run_max, block_max);
        /* A row with no valid key yet keeps run_max = -inf: its alpha is zero
         * and its exponent base is zero, never exp(-inf - -inf) = NaN. */
        const float alpha = (run_max == -DIT_INF) ? 0.0f : expf(run_max - new_max);
        const float base = (new_max == -DIT_INF) ? 0.0f : new_max;
        float block_sum = 0.0f;
#pragma unroll
        for (u32 j = 0; j < KB; ++j) {
            const float pj = (sc[j] == -DIT_INF) ? 0.0f : expf(sc[j] - base);
            sc[j] = pj;
            block_sum += pj;
        }
        run_sum = run_sum * alpha + block_sum;
        run_max = new_max;
#pragma unroll
        for (u32 i = 0; i < MAX4; ++i) {
            o[i] *= alpha;
        }
#pragma unroll
        for (u32 j = 0; j < KB; ++j) {
#pragma unroll
            for (u32 i = 0; i < MAX4; ++i) {
                if (i < hd4) {
                    o[i] += sc[j] * sv[j * MAXD + part + 4u * i];
                }
            }
        }
    }

    if (live) {
        const float inv = 1.0f / run_sum;
        const u32 width = args.heads * hd;
        float* op = qrow < args.txt_tokens
                ? out_txt + (u64)qrow * width + h * hd
                : out_img + (u64)(qrow - args.txt_tokens) * width + h * hd;
#pragma unroll
        for (u32 i = 0; i < MAX4; ++i) {
            if (i < hd4) {
                op[part + 4u * i] = o[i] * inv;
            }
        }
    }
}

#define DIT_FLASH_SDPA_PARAMS \
    const dit_args_sdpa args, float* out_txt, float* out_img, \
    const float* q_txt, const float* k_txt, const float* v_txt, \
    const float* q_img, const float* k_img, const float* v_img, \
    const float* cache_k, const float* cache_v

#define DIT_FLASH_SDPA_CALL(CAUSAL, HD) \
    dit_flash_sdpa<CAUSAL, HD>(args, out_txt, out_img, q_txt, k_txt, v_txt, q_img, k_img, \
                               v_img, cache_k, cache_v);

extern "C" __global__ void kernel_dit_flash_sdpa_f32(DIT_FLASH_SDPA_PARAMS) {
    DIT_FLASH_SDPA_CALL(false, 0u)
}
extern "C" __global__ void kernel_dit_flash_sdpa_causal_f32(DIT_FLASH_SDPA_PARAMS) {
    DIT_FLASH_SDPA_CALL(true, 0u)
}
/* The 128-wide entries fix the head width at compile time so every head loop
 * unrolls to a constant bound. */
extern "C" __global__ void kernel_dit_flash_sdpa_d128_f32(DIT_FLASH_SDPA_PARAMS) {
    DIT_FLASH_SDPA_CALL(false, 128u)
}
extern "C" __global__ void kernel_dit_flash_sdpa_causal_d128_f32(DIT_FLASH_SDPA_PARAMS) {
    DIT_FLASH_SDPA_CALL(true, 128u)
}
