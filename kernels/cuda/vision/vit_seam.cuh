/* The seam between the tower and the language model.
 *
 * The CUDA twin of `metal/vision/vit_splice.metal`. The only entry in this
 * module that does not belong to the tower: the language model's prefill embeds
 * every token of a chunk, placeholders included, and this overwrites the
 * placeholder rows with the rows the merger produced. Both buffers are f32
 * `[rows][hidden]` at the LANGUAGE model's hidden width, not the tower's.
 *
 * A per-token index row rather than one copy per placeholder: a 1280x1280 image
 * is 1600 placeholders after merging, and one device-to-device copy each is 1600
 * launches for what is a single strided read. The index row is built on the host
 * in the same walk that builds the position streams, so the two cannot disagree
 * about where a run started.
 */
#ifndef SHISO_VIT_SEAM_CUH
#define SHISO_VIT_SEAM_CUH

#include "vit_common.cuh"

/* `hidden[r] = src[map[r]]` for every token row whose `map` entry names an
 * encoder row, and nothing at all for the rest.
 *
 * A negative entry means a text token, which is the common case and the reason
 * the map is signed. An entry at or past `src_rows` is a bug upstream rather
 * than an input to honour, and it is skipped: reading the row would hand the
 * model whatever followed the tower's output, which is the failure nothing
 * downstream could detect. The row-count contract is checked on the host, where
 * it can still be reported.
 *
 * One block per token row, whatever the block width: the loop strides by
 * `blockDim.x`. */
extern "C" __global__ void kernel_vit_splice_rows(
        const struct vit_args_splice args,
        float *__restrict__ hidden,
        const float *__restrict__ src,
        const int32_t *__restrict__ map) {
    const unsigned r = blockIdx.x;
    if (r >= args.rows) {
        return;
    }
    const int32_t row = map[r];
    if (row < 0 || (uint32_t) row >= args.src_rows) {
        return;
    }
    const unsigned hidden_dim = args.hidden;
    const uint64_t from = (uint64_t) (uint32_t) row * hidden_dim;
    const uint64_t to = (uint64_t) r * hidden_dim;
    for (unsigned i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        hidden[to + i] = src[from + i];
    }
}

#endif /* SHISO_VIT_SEAM_CUH */
