// The seam between the tower and the language model: the encoder's rows take
// the place of the embedding rows at the placeholder tokens.
//
// The prompt carries one placeholder token per merged patch block, the tower
// produced one vector per merged patch block, and this is where the two meet.
// Everything else about the prefill is unchanged -- the placeholders were
// embedded like any other token first, and their rows are simply overwritten
// here.
//
// A per-token index row rather than one copy per placeholder: a 1280x1280
// image is 1600 placeholders after merging, and one device-to-device copy each
// is 1600 command encodings for what is a single strided read.  The index row
// is built on the host in the same walk that builds the position streams, so
// the two cannot disagree about where a run started.

/// `hidden[r] = src[map[r]]` for every token row whose `map` entry names an
/// encoder row, and nothing at all for the rest.
///
/// A negative entry means a text token, which is the common case and the
/// reason the map is signed. An entry at or past `src_rows` is a bug upstream
/// rather than an input to honour, and it is skipped: reading the row would
/// hand the model whatever followed the tower's output, which is the failure
/// nothing downstream could detect. The row-count contract is checked on the
/// host, where it can still be reported.
///
/// One threadgroup per token row, whatever the block width: the loop strides by
/// `threads_per_threadgroup`.
kernel void kernel_vit_splice_rows(
        constant vit_args_splice & args [[buffer(0)]],
        device float *hidden             [[buffer(1)]],
        device const float *src          [[buffer(2)]],
        device const int32_t *map        [[buffer(3)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3 ntg[[threads_per_threadgroup]]) {
    const uint r = tgpig.x;
    if (r >= args.rows) {
        return;
    }
    const int32_t row = map[r];
    if (row < 0 || (uint32_t) row >= args.src_rows) {
        return;
    }
    const uint hidden_dim = args.hidden;
    const uint64_t from = (uint64_t) (uint32_t) row * hidden_dim;
    const uint64_t to = (uint64_t) r * hidden_dim;
    for (uint i = tpitg.x; i < hidden_dim; i += ntg.x) {
        hidden[to + i] = src[from + i];
    }
}
