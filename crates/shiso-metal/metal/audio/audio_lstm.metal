// Transducer and decoder steps: the recurrent cell of a transducer's
// prediction network and its greedy search, and the step and embedding row of
// a causal decoder, all resident on the device.

/// Gates in (i, f, g, o) order, biases already folded in by the projections:
/// c' = sig(f) c + sig(i) tanh(g);  h' = sig(o) tanh(c').  `c` is updated in place.
kernel void kernel_audio_lstm_cell_f32(
        constant audio_args_lstm & args [[buffer(0)]],
        device float *h_out             [[buffer(1)]],
        device float *c                 [[buffer(2)]],
        device const float *gates       [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint u = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    const uint H = args.hidden;
    if (u >= H) {
        return;
    }
    const float gi = audio_sigmoid(gates[u]);
    const float gf = audio_sigmoid(gates[H + u]);
    const float gg = precise::tanh(gates[2u * H + u]);
    const float go = audio_sigmoid(gates[3u * H + u]);
    const float cn = gf * c[u] + gi * gg;
    c[u] = cn;
    h_out[u] = go * precise::tanh(cn);
}

// Greedy transducer search, resident on the device: a step is the window
// below, the joint's head, the row argmax, then `kernel_audio_greedy_step_u32`
// deciding, and the gated prediction network advancing on what it decided. A
// run of steps queues without the host reading anything back, because every
// decision a step makes is read by the next step from the state buffer.

/// `out[r][c] = relu(pending[t + r][c] + dec[c])` for the window's rows, 0 for
/// rows at or past `avail`; `t` is read from the search state.
kernel void kernel_audio_greedy_window_f32(
        constant audio_args_greedy_window & args [[buffer(0)]],
        device float *out                        [[buffer(1)]],
        device const float *pending              [[buffer(2)]],
        device const float *dec                  [[buffer(3)]],
        device const uint *state                 [[buffer(4)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.rows * args.cols) {
        return;
    }
    const uint r = i / args.cols;
    const uint c = i - r * args.cols;
    const uint src = state[AUDIO_GREEDY_T] + r;
    out[i] = src < args.avail ? max(pending[(uint64_t) src * args.cols + c] + dec[c], 0.0f) : 0.0f;
}

/// One decision of greedy search over the window's argmax `ids`, on one
/// thread: the first non-blank row is emitted at its frame (a later frame
/// starts its symbol count afresh), a window of blanks moves past itself, and
/// the frame moves on after `max_symbols` symbols. Writes `(token, frame)` or
/// `(NONE, 0)` to `out[slot]`, and sets the state's emit flag for the gated
/// prediction network that follows.
kernel void kernel_audio_greedy_step_u32(
        constant audio_args_greedy_step & args [[buffer(0)]],
        device uint *state                     [[buffer(1)]],
        device uint *out                       [[buffer(2)]],
        device const uint *ids                 [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }
    uint t = state[AUDIO_GREEDY_T];
    uint sym = state[AUDIO_GREEDY_SYM];
    out[2u * args.slot] = AUDIO_GREEDY_NONE;
    out[2u * args.slot + 1u] = 0u;
    state[AUDIO_GREEDY_EMIT] = 0u;
    if (t >= args.avail) {
        state[AUDIO_GREEDY_DONE] = 1u;
        return;
    }
    const uint m = min(args.rows, args.avail - t);
    uint j = m;
    for (uint r = 0; r < m; ++r) {
        if (ids[r] != args.blank) {
            j = r;
            break;
        }
    }
    if (j == m) {
        t += m;
        sym = 0u;
    } else {
        if (j > 0u) {
            t += j;
            sym = 0u;
        }
        const uint k = ids[j];
        out[2u * args.slot] = k;
        out[2u * args.slot + 1u] = t;
        state[AUDIO_GREEDY_TOKEN] = k;
        state[AUDIO_GREEDY_EMIT] = 1u;
        sym += 1u;
        if (sym >= args.max_symbols) {
            t += 1u;
            sym = 0u;
        }
    }
    state[AUDIO_GREEDY_T] = t;
    state[AUDIO_GREEDY_SYM] = sym;
    state[AUDIO_GREEDY_DONE] = t >= args.avail ? 1u : 0u;
}

/// `out = table[token]`, the state's last token's row; zeros for a token past
/// the table.
kernel void kernel_audio_embed_row_f32(
        constant audio_args_embed_row & args [[buffer(0)]],
        device float *out                    [[buffer(1)]],
        device const float *table            [[buffer(2)]],
        device const uint *state             [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint u = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (u >= args.width) {
        return;
    }
    const uint k = state[AUDIO_GREEDY_TOKEN];
    out[u] = k < args.vocab ? table[(uint64_t) k * args.width + u] : 0.0f;
}

/// The cell above, with the new state kept only when the last step emitted:
/// `x_out` always takes h' (the next layer's input), `h` and `c` take h' and
/// c' only under the state's emit flag.
kernel void kernel_audio_lstm_cell_gated_f32(
        constant audio_args_lstm & args [[buffer(0)]],
        device float *x_out             [[buffer(1)]],
        device float *h                 [[buffer(2)]],
        device float *c                 [[buffer(3)]],
        device const float *gates       [[buffer(4)]],
        device const uint *state        [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint u = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    const uint H = args.hidden;
    if (u >= H) {
        return;
    }
    const float gi = audio_sigmoid(gates[u]);
    const float gf = audio_sigmoid(gates[H + u]);
    const float gg = precise::tanh(gates[2u * H + u]);
    const float go = audio_sigmoid(gates[3u * H + u]);
    const float cn = gf * c[u] + gi * gg;
    const float hn = go * precise::tanh(cn);
    x_out[u] = hn;
    if (state[AUDIO_GREEDY_EMIT] != 0u) {
        c[u] = cn;
        h[u] = hn;
    }
}

/// `dst = src` when the last step emitted, else nothing.
kernel void kernel_audio_keep_f32(
        constant audio_args_keep & args [[buffer(0)]],
        device float *dst               [[buffer(1)]],
        device const float *src         [[buffer(2)]],
        device const uint *state        [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint i = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (i >= args.count || state[AUDIO_GREEDY_EMIT] == 0u) {
        return;
    }
    dst[i] = src[i];
}

// A causal decoder's greedy loop, resident on the device: a step is the
// embedding row of the state's last token, the layers (the rotation at the
// state's position, the cache append there, decode attention through it), the
// head, the row argmax, then `kernel_audio_decode_step_u32` deciding. A batch
// of steps queues with no read between them; once a step stops the decoder,
// every later step in the batch writes nothing and leaves the state alone, so
// its passes only recompute the row the stop left.

/// One decision of a causal decoder, on one thread. The pass just run fed
/// `advance` rows into the cache (0 for the prefill's own argmax, 1 in
/// decode), so the position moves by that much. Then the argmax `ids[0]` is:
/// an end id - done, stop EOS, nothing emitted; otherwise emitted to
/// `out[slot]` and made the last token, and done with stop CAP when it is the
/// `max_tokens`-th. A step on a done state writes `NONE`, lowers the emit
/// flag and changes nothing else; a step on a state that has already emitted
/// `max_tokens` does the same and marks it done.
/// The decision above for one sequence: `state` its state words, `slot` its
/// output slot and `k` its argmax. The single and batched step entries both
/// run this body.
static inline void audio_decode_step_one(
        uint eos0, uint eos1, uint max_tokens, uint advance,
        device uint *state, device uint *slot, uint k) {
    *slot = AUDIO_DECODE_NONE;
    state[AUDIO_DECODE_EMIT] = 0u;
    if (state[AUDIO_DECODE_DONE] != 0u) {
        return;
    }
    const uint count = state[AUDIO_DECODE_COUNT];
    if (count >= max_tokens) {
        state[AUDIO_DECODE_DONE] = 1u;
        state[AUDIO_DECODE_STOP] = AUDIO_DECODE_STOP_CAP;
        return;
    }
    state[AUDIO_DECODE_POS] += advance;
    if (k == eos0 || k == eos1) {
        state[AUDIO_DECODE_DONE] = 1u;
        state[AUDIO_DECODE_STOP] = AUDIO_DECODE_STOP_EOS;
        return;
    }
    *slot = k;
    state[AUDIO_DECODE_TOKEN] = k;
    state[AUDIO_DECODE_EMIT] = 1u;
    state[AUDIO_DECODE_COUNT] = count + 1u;
    if (count + 1u >= max_tokens) {
        state[AUDIO_DECODE_DONE] = 1u;
        state[AUDIO_DECODE_STOP] = AUDIO_DECODE_STOP_CAP;
    }
}

kernel void kernel_audio_decode_step_u32(
        constant audio_args_decode_step & args [[buffer(0)]],
        device uint *state                     [[buffer(1)]],
        device uint *out                       [[buffer(2)]],
        device const uint *ids                 [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid != 0u) {
        return;
    }
    audio_decode_step_one(args.eos0, args.eos1, args.max_tokens, args.advance,
            state, out + args.slot, ids[0]);
}

/// The step over a batch, one thread per sequence: sequence `s` decides on
/// `ids[s]` (the batch's row argmax) over state block `s`, exactly as the
/// single step, and writes its slot to `out[slot * seqs + s]`, so a batch of
/// steps leaves `[steps][seqs]`. A finished sequence goes quiet as a
/// finished single state does.
kernel void kernel_audio_decode_step_batch_u32(
        constant audio_args_decode_step_batch & args [[buffer(0)]],
        device uint *states                          [[buffer(1)]],
        device uint *out                             [[buffer(2)]],
        device const uint *ids                       [[buffer(3)]],
        uint tid [[thread_position_in_grid]]) {
    if (tid >= args.seqs) {
        return;
    }
    audio_decode_step_one(args.eos0, args.eos1, args.max_tokens, args.advance,
            states + (uint64_t) tid * AUDIO_DECODE_STATE_WORDS,
            out + (uint64_t) args.slot * args.seqs + tid, ids[tid]);
}

/// `out = table[token]` dequantised to f32, the decoder state's last token's
/// row of a `[vocab][width]` table in `W`'s storage (the GEMM's weight
/// readers, so a Q8_0 row is `width / 32` blocks); zeros for a token past the
/// table.
template <typename W>
kernel void kernel_audio_embed_token(
        constant audio_args_embed_token & args [[buffer(0)]],
        device float *out                      [[buffer(1)]],
        device const void *table               [[buffer(2)]],
        device const uint *state               [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint u = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (u >= args.width) {
        return;
    }
    const uint k = state[AUDIO_DECODE_TOKEN];
    out[u] = k < args.vocab ? W::load(table, (uint64_t) k * args.width + u) : 0.0f;
}

typedef decltype(kernel_audio_embed_token<audio_w_f32>) audio_embed_token_t;
template [[host_name("kernel_audio_embed_token_f32w")]]  kernel audio_embed_token_t kernel_audio_embed_token<audio_w_f32>;
template [[host_name("kernel_audio_embed_token_bf16w")]] kernel audio_embed_token_t kernel_audio_embed_token<audio_w_bf16>;
template [[host_name("kernel_audio_embed_token_f16w")]]  kernel audio_embed_token_t kernel_audio_embed_token<audio_w_f16>;
template [[host_name("kernel_audio_embed_token_q8_0w")]] kernel audio_embed_token_t kernel_audio_embed_token<audio_w_q8_0>;

/// The row above for each sequence of a batch: row `s` of `[seqs][width]`
/// is the last token of state block `s`. Grid `(ceil(width / 256), seqs)`.
template <typename W>
kernel void kernel_audio_embed_token_batch(
        constant audio_args_embed_token & args [[buffer(0)]],
        device float *out                      [[buffer(1)]],
        device const void *table               [[buffer(2)]],
        device const uint *states              [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint u = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (u >= args.width) {
        return;
    }
    const uint k = states[(uint64_t) tgpig.y * AUDIO_DECODE_STATE_WORDS + AUDIO_DECODE_TOKEN];
    out[(uint64_t) tgpig.y * args.width + u] =
            k < args.vocab ? W::load(table, (uint64_t) k * args.width + u) : 0.0f;
}

typedef decltype(kernel_audio_embed_token_batch<audio_w_f32>) audio_embed_token_batch_t;
template [[host_name("kernel_audio_embed_token_batch_f32w")]]  kernel audio_embed_token_batch_t kernel_audio_embed_token_batch<audio_w_f32>;
template [[host_name("kernel_audio_embed_token_batch_bf16w")]] kernel audio_embed_token_batch_t kernel_audio_embed_token_batch<audio_w_bf16>;
template [[host_name("kernel_audio_embed_token_batch_f16w")]]  kernel audio_embed_token_batch_t kernel_audio_embed_token_batch<audio_w_f16>;
template [[host_name("kernel_audio_embed_token_batch_q8_0w")]] kernel audio_embed_token_batch_t kernel_audio_embed_token_batch<audio_w_q8_0>;

/// The row above from an 8-bit affine table: per `group` elements of a row a
/// bf16 scale and bias, the bytes packed four to a little-endian u32, and
/// each value one fused `q * scale + bias`.
kernel void kernel_audio_embed_token_affine8w(
        constant audio_args_embed_token & args [[buffer(0)]],
        device float *out                      [[buffer(1)]],
        device const uint *words               [[buffer(2)]],
        device const ushort *scales            [[buffer(3)]],
        device const ushort *biases            [[buffer(4)]],
        device const uint *state               [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint u = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (u >= args.width) {
        return;
    }
    const uint k = state[AUDIO_DECODE_TOKEN];
    if (k >= args.vocab) {
        out[u] = 0.0f;
        return;
    }
    const uint64_t i = (uint64_t) k * args.width + u;
    const uint64_t g = (uint64_t) k * (args.width / args.group) + u / args.group;
    const float qv = (float) ((words[i >> 2u] >> (8u * (uint) (i & 3u))) & 0xffu);
    out[u] = fma(qv, audio_bf16(scales + g), audio_bf16(biases + g));
}

/// The affine row above for each sequence of a batch, as the dense batch
/// entries lay it out.
kernel void kernel_audio_embed_token_batch_affine8w(
        constant audio_args_embed_token & args [[buffer(0)]],
        device float *out                      [[buffer(1)]],
        device const uint *words               [[buffer(2)]],
        device const ushort *scales            [[buffer(3)]],
        device const ushort *biases            [[buffer(4)]],
        device const uint *states              [[buffer(5)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]]) {
    const uint u = tgpig.x * AUDIO_LINEAR_BLOCK + tpitg.x;
    if (u >= args.width) {
        return;
    }
    device float *row = out + (uint64_t) tgpig.y * args.width;
    const uint k = states[(uint64_t) tgpig.y * AUDIO_DECODE_STATE_WORDS + AUDIO_DECODE_TOKEN];
    if (k >= args.vocab) {
        row[u] = 0.0f;
        return;
    }
    const uint64_t i = (uint64_t) k * args.width + u;
    const uint64_t g = (uint64_t) k * (args.width / args.group) + u / args.group;
    const float qv = (float) ((words[i >> 2u] >> (8u * (uint) (i & 3u))) & 0xffu);
    row[u] = fma(qv, audio_bf16(scales + g), audio_bf16(biases + g));
}
