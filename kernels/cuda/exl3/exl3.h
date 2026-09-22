/* shiso entry points for the vendored exllamav3 EXL3 GEMM kernels (see VENDOR.md).
 *
 * Every activation is fp32 on both sides, as everywhere in shiso's qwen4exp
 * graph; the kernels convert to fp16 inside their input Hadamard phase.  A
 * trellis tensor is addressed by three device pointers -- the tiles, the
 * fp16 input scale vector suh[k] and the fp16 output vector svh[n] -- which
 * the GGUF stores back to back in one tensor payload (exl3_scale_bytes).
 *
 * The kernels are cooperative launches (grid-wide barriers between the input
 * transform and the matmul, and between 16-row passes), so a grid is at most
 * the device's co-resident block count and needs the per-stream lock buffer
 * this file owns.  Both entries return 1 on success, 0 on failure. */
#ifndef SHISO_EXL3_H
#define SHISO_EXL3_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/* y[m][n] = Had128(x[m][k] * suh) @ B, then Had128 * svh on every 128-wide
 * output block.  Any m; rows are processed 16 at a time. */
int shs_exl3_gemm(const float *x, const void *tiles, const void *suh, const void *svh,
                  float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                  cudaStream_t stream);

/* Expert fan-out over a stacked tensor ([E] tiles, [E] suh, [E] svh): slot j
 * multiplies its input by expert ids[j] into y[j][m][n].  The input is x
 * broadcast to every slot when x_per_slot is 0, else x[j][m][k].  ids may be
 * NULL for the identity (slot j is matrix j). */
int shs_exl3_mgemm(const float *x, int x_per_slot,
                   const void *tiles, const void *suh, const void *svh,
                   const int32_t *ids, uint32_t n_slots,
                   float *y, uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                   cudaStream_t stream);

/* The same fan-out over two stacked tensors of one shape in a single launch:
 * the slots run once over (tiles, suh, svh) into y and once over (tiles2,
 * suh2, svh2) into y2, from the same inputs and ids -- a gate and an up
 * projection that read the same activation. */
int shs_exl3_mgemm_pair(const float *x, int x_per_slot,
                        const void *tiles, const void *suh, const void *svh, float *y,
                        const void *tiles2, const void *suh2, const void *svh2, float *y2,
                        const int32_t *ids, uint32_t n_slots,
                        uint32_t m, uint32_t k, uint32_t n, uint32_t bits,
                        cudaStream_t stream);

/* The routed-expert block of one layer as a single fused launch (exllamav3's
 * exl3_moe): slot_out[slot][hidden] = down(silu(gate(x)) * up(x)) through the
 * expert of routing slot = token * n_used + rank, unweighted -- the caller's
 * combine applies the routing weights in a fixed order, which keeps a chunk
 * reproducible where the kernel's own atomic scatter-add was not.  The three
 * projections are stacked expert tensors ([E] tiles, [E] suh, [E] svh;
 * gate/up are hidden -> inter, down is inter -> hidden, all of one bit
 * width).  The routing is the expert-sorted map cuda/mmq builds: expert e
 * owns slot_sorted[bounds[e] .. bounds[e+1]).  Prefill only: the staging
 * buffers are sized by n_tok and never allocated under graph capture. */
int shs_exl3_moe(const float *x, float *slot_out, uint32_t n_tok, uint32_t hidden, uint32_t inter,
                 const void *gate_tiles, const void *gate_suh, const void *gate_svh,
                 const void *up_tiles, const void *up_suh, const void *up_svh,
                 const void *down_tiles, const void *down_suh, const void *down_svh,
                 uint32_t bits, const int32_t *expert_bounds, const int32_t *slot_sorted,
                 uint32_t n_expert, uint32_t n_used, cudaStream_t stream);

/* The same routed block as shs_exl3_moe, weight-stationary: three ordinary
 * launches (input prep, gate+up with the fused SiLU epilogue, down) where each
 * block owns one (expert, 128-column tile) and decodes every weight once per
 * 64 rows of that expert instead of once per 16.  Same inputs, same per-slot
 * output layout; the staging (fp16 gate/up inputs and down input, sized by
 * n_tok * n_used rows) lives beside the fused kernel's on the stream. */
int shs_exl3_moe_ws(const float *x, float *slot_out, uint32_t n_tok, uint32_t hidden, uint32_t inter,
                    const void *gate_tiles, const void *gate_suh, const void *gate_svh,
                    const void *up_tiles, const void *up_suh, const void *up_svh,
                    const void *down_tiles, const void *down_suh, const void *down_svh,
                    uint32_t bits, const int32_t *expert_bounds, const int32_t *slot_sorted,
                    uint32_t n_expert, uint32_t n_used, cudaStream_t stream);

/* Expand a trellis tensor to fp16 W[k][n] in the original basis
 * (diag(suh) . H128 . W_hat . H128 . diag(svh)), so that y = x @ W on the raw
 * activations equals shs_exl3_gemm up to fp16 rounding.  For prefill: past a
 * few hundred rows one expansion plus a cuBLAS GEMM beats streaming the trellis
 * once per 16 rows.  Needs k % 128 == 0 and n % 128 == 0. */
int shs_exl3_reconstruct(const void *tiles, const void *suh, const void *svh,
                         uint32_t k, uint32_t n, uint32_t bits, void *w_out,
                         cudaStream_t stream);

/* Size the transformed-activation scratch for the largest launch (slots * m *
 * k halfs) ahead of CUDA graph capture, inside which it cannot grow. */
int shs_exl3_reserve(cudaStream_t stream, uint64_t had_halfs);

/* Give `stream` a lock buffer and a scratch as large as any stream has needed
 * so far.  Call before capturing a graph on it; a no-op when no EXL3 launch
 * has happened yet (the model is not EXL3). */
int shs_exl3_prepare_stream(cudaStream_t stream);

#ifdef __cplusplus
}
#endif

#endif
