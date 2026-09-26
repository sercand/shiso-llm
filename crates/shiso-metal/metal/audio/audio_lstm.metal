// The recurrent cell of a transducer's prediction network.

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
