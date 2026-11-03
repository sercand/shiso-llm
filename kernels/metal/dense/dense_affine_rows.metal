// MLX affine token embedding gather and affine per-layer embedding prepare kernels.

#include <metal_stdlib>
using namespace metal;

/// Gathers and dequantizes token rows from a 4-bit affine embedding table.
kernel void kernel_dense_embed_affine4_f32(
        constant dense_args_embed_affine & args,
        device       float    * out,
        device const uint32_t * words,
        device const ushort   * scales,
        device const ushort   * biases,
        device const int32_t  * ids,
        uint2 id [[thread_position_in_grid]]) {
    const uint col = id.x;
    const uint tok = id.y;
    if (col >= args.hidden || tok >= args.n_tok) return;

    const uint64_t row = (uint64_t) ids[tok];
    const uint words_per_row = args.hidden / 8u;
    const uint groups_per_row = args.hidden / 64u;

    const uint g = col / 64u;
    const uint in_group = col % 64u;
    const uint word_idx = in_group / 8u;
    const uint shift = (in_group % 8u) * 4u;

    const uint32_t word = words[row * (uint64_t) words_per_row + g * 8u + word_idx];
    const float q = (float) ((word >> shift) & 0x0fu);
    const float scale = dense_bf16(scales + row * (uint64_t) groups_per_row + g);
    const float bias = dense_bf16(biases + row * (uint64_t) groups_per_row + g);

    out[(uint64_t) tok * args.hidden + col] = q * scale + bias;
}

/// Prepares per-layer embedding rows for a slot from the 4-bit affine sidecar.
kernel void kernel_dense_ple_prepare_affine4(
        constant dense_args_ple & args,
        device       float   * out,
        device const float   * proj,
        device const float   * norm_w,
        device const char    * rows,
        device const int32_t * ids,
        threadgroup float * tbuf [[threadgroup(0)]],
        uint3   tgpig [[threadgroup_position_in_grid]],
        ushort3 tpitg [[thread_position_in_threadgroup]],
        ushort3 ntg_v [[threads_per_threadgroup]]) {
    const uint l = tgpig.x;
    const uint t = tgpig.y;
    if (l >= args.n_layers || t >= args.n_tok) return;
    const uint tpitg_x = (uint) tpitg.x;
    const uint ntg_x = (uint) ntg_v.x;

    const uint dim = args.dim;
    threadgroup float * emb = tbuf + DENSE_BLOCK_SUM_SLOTS;

    const uint64_t row = (args.resident != 0u) ? (uint64_t) ids[t] : (uint64_t) t;
    device const char * row_ptr = rows + row * (uint64_t) args.row_bytes;

    // A row holds every slot's words, then every slot's scales, then every
    // slot's biases - the order `gemma4::ple_repack` writes them in. The three
    // bases are derived from the slot count and the slot width rather than
    // written down, so this decodes whatever row stride the caller passes and
    // not only the one the shipped checkpoint happens to have.
    const uint groups_per_slot = dim / 64u;
    const uint words_per_slot = dim / 2u;           // four bits a weight
    const uint scales_per_slot = groups_per_slot * 2u;   // one bf16 a group
    const uint scales_base = args.n_layers * words_per_slot;
    const uint biases_base = scales_base + args.n_layers * scales_per_slot;

    device const uint32_t * w = (device const uint32_t *)(row_ptr + l * words_per_slot);
    device const ushort * s = (device const ushort *)(row_ptr + scales_base + l * scales_per_slot);
    device const ushort * b = (device const ushort *)(row_ptr + biases_base + l * scales_per_slot);

    for (uint i = tpitg_x; i < dim; i += ntg_x) {
        const uint g = i / 64u;
        const uint in_group = i % 64u;
        const uint word_idx = in_group / 8u;
        const uint shift = (in_group % 8u) * 4u;

        const uint32_t word = w[g * 8u + word_idx];
        const float q = (float) ((word >> shift) & 0x0fu);
        const float scale = dense_bf16(s + g);
        const float bias = dense_bf16(b + g);
        emb[i] = q * scale + bias;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    device const float * p = proj + ((uint64_t) t * args.n_layers + l) * dim;
    device       float * dst = out + ((uint64_t) t * args.n_layers + l) * dim;

    float acc = 0.0f;
    for (uint i = tpitg_x; i < dim; i += ntg_x) {
        const float v = p[i] * args.proj_scale;
        acc = dense_fma(v, v, acc);
    }
    const float total = dense_block_sum(acc, tbuf, tpitg_x, ntg_x);
    const float inv = rsqrt(total / (float) dim + args.eps);

    for (uint i = tpitg_x; i < dim; i += ntg_x) {
        const float pv = p[i] * args.proj_scale * inv * norm_w[i];
        const float ev = emb[i] * args.embed_scale;
        dst[i] = (pv + ev) * args.mix_scale;
    }
}
