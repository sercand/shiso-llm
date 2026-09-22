#pragma once

// Vendored from exllamav3 (quant/exl3_moe_kernel.cuh, MIT, Copyright (c) 2025 Turboderp)
// with shiso modifications, see VENDOR.md:
//   - hidden_state is fp32 (shiso keeps every activation in fp32); the two input
//     gathers convert while they Hadamard-transform;
//   - the nine per-expert pointer tables are three stacked tensors addressed
//     as base + expert * stride, like exl3_mgemm_kernel;
//   - the routing arrives as int32 expert bounds plus expert-sorted slots (the
//     map cuda/mmq already builds), and each assignment's down projection is
//     stored to its own row instead of being weighted and atomically added
//     into the token's row: fp32 atomics sum in whichever order the experts
//     finish, and at long context that run-to-run noise was enough to flip
//     router near-ties (26k-token logit L2 vs exllamav3 swung 5.6-16%);
//   - the gate/up outputs, the gated product and the down output are staged
//     in fp32, not fp16, so the fused block computes what shiso's staged path
//     (fp32 GEMM outputs, fp32 SwiGLU) computed and differs from it only in
//     summation order;
//   - one bit width per instance (no runtime K switch), SiLU gating only, and
//     the staging buffers are sized by num_tokens rather than a separate
//     max_tokens_per_expert with an overflow fallback: no expert can hold more
//     rows than there are tokens.

#include <stdint.h>

#include "exl3_moe_common.cuh"
#include "util.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_gemm_inner.cuh"
#include "exl3_devctx.cuh"
#include "ptx.cuh"

// Fused op, fp32 in: o <- in_had(silu(out_had(g)) * out_had(u)), the fp32
// twin of had_hf_r_128_guad_inner with shiso's SiLU (g / (1 + exp(-g))).
inline __device__
void had_ff_r_128_guad_inner
(
    const float* __restrict__ input_ptr_g,
    const float* __restrict__ input_ptr_u,
    half* __restrict__ output_ptr,
    const half* __restrict__ post_scale_g,
    const half* __restrict__ post_scale_u,
    const half* __restrict__ pre_scale_d,
    const float r_scale
)
{
    int t = threadIdx.x & 31;

    auto had = [&](float4& v)
    {
        float s0 = v.x + v.y, d0 = v.x - v.y;
        float s1 = v.z + v.w, d1 = v.z - v.w;
        v.x = s0 + s1;
        v.y = d0 + d1;
        v.z = s0 - s1;
        v.w = d0 - d1;
        shuffle_had_f2x32(v.x, v.y, t);
        shuffle_had_f2x32(v.z, v.w, t);
        v.x *= r_scale;
        v.y *= r_scale;
        v.z *= r_scale;
        v.w *= r_scale;
    };
    auto scale = [&](float4& v, const half* __restrict__ sc)
    {
        half4 s = ((const half4*) sc)[t];
        v.x *= __low2float(s.x);
        v.y *= __high2float(s.x);
        v.z *= __low2float(s.y);
        v.w *= __high2float(s.y);
    };
    auto silu = [](float g) { return g / (1.0f + __expf(-g)); };

    float4 g = ((const float4*) input_ptr_g)[t];
    float4 u = ((const float4*) input_ptr_u)[t];
    had(g);
    had(u);
    scale(g, post_scale_g);
    scale(u, post_scale_u);
    g.x = silu(g.x) * u.x;
    g.y = silu(g.y) * u.y;
    g.z = silu(g.z) * u.z;
    g.w = silu(g.w) * u.w;
    scale(g, pre_scale_d);
    had(g);

    half4 o;
    o.x = __floats2half2_rn(g.x, g.y);
    o.y = __floats2half2_rn(g.z, g.w);
    ((half4*) output_ptr)[t] = o;
}

// Fused routed-expert block: for every expert with tokens, gather its rows,
// gate and up GEMMs, SiLU gating, down GEMM, and the down output stored per
// assignment into slot_out[n_tokens * num_experts_per_tok][hidden] (fp32).
// The output Hadamards run in fp32 on fp32 GEMM outputs (see the header).  The grid is gridDim.z groups of
// gridDim.x co-resident blocks; each group takes one expert at a time from a
// self-resetting ticket counter, so the whole launch is one kernel however
// the tokens spread over the experts.
template<int t_bits, int MOE_TILESIZE_N, int cb>
__global__ __launch_bounds__(EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16)
void exl3_moe_kernel(EXL3_MOE_KERNEL_ARGS)
{
    const int group_idx = blockIdx.z;
    const int block_idx = blockIdx.x;
    const int group_size = gridDim.x;  // SMs per expert, set at launch
    const int num_groups = gridDim.z;
    const int block_threads = EXL3_GEMM_BASE_THREADS * MOE_TILESIZE_K / 16;  // blockDim.x
    const int group_threads = group_size * block_threads;
    const int warp_id = threadIdx.x / 32;
    const int warps_per_group = group_threads / 32;
    const int warps_per_block = block_threads / 32;
    const int warp_idx0 = block_idx * warps_per_block + warp_id;

    // Buffers for group.  The down output reuses the gathered-input bytes:
    // gate and up are done with them by then, and two fp16 rows are one fp32 row.
    temp_state += (size_t) group_idx * 2 * num_tokens * hidden_dim;
    half* temp_state_g = temp_state;
    half* temp_state_u = temp_state + (size_t) num_tokens * hidden_dim;
    float* temp_down = (float*) temp_state;
    temp_intermediate += (size_t) group_idx * 2 * num_tokens * intermediate_dim;
    float* temp_intermediate_g = temp_intermediate;
    float* temp_intermediate_u = temp_intermediate + (size_t) num_tokens * intermediate_dim;
    temp_act += (size_t) group_idx * num_tokens * intermediate_dim;

    // Barriers for group sync
    int* barrier_counters_sense = locks + BARRIER_LOCKS_OFFSET;

    // Expert scheduler state, self-resetting: [0] next ticket, [1] retired groups, [2 + g] ticket for group g
    int* sched = locks + MOE_SCHED_OFFSET;

    // Individual GEMM barriers per group
    locks += group_idx * MAX(hidden_dim, intermediate_dim) / 128;

    // Stacked expert tensors
    const size_t trellis_stride = (size_t) hidden_dim * intermediate_dim * t_bits / 16;

    // Dynamic expert assignment: active experts are numbered in scan order, and each group processes the active
    // expert matching its current ticket. Initial tickets are the group indices; after finishing an expert, a group
    // draws the next unclaimed ticket, so load balances greedily without assuming uniform cost per expert
    int ticket = group_idx;

    // Loop over experts
    int expert_idx_assign = 0;
    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx)
    {
        // Token span for current expert
        const int start = expert_bounds[expert_idx];
        const int end = expert_bounds[expert_idx + 1];
        const int token_count = end - start;
        if (token_count == 0) continue;

        // Skip if expert is claimed by a different group
        if (expert_idx_assign++ != ticket) continue;

        // EXL3 weights for g, u, d
        const uint16_t* exp_gate_trellis = gate_trellis + expert_idx * trellis_stride;
        const half* exp_gate_suh = gate_suh + (size_t) expert_idx * hidden_dim;
        const half* exp_gate_svh = gate_svh + (size_t) expert_idx * intermediate_dim;
        const uint16_t* exp_up_trellis = up_trellis + expert_idx * trellis_stride;
        const half* exp_up_suh = up_suh + (size_t) expert_idx * hidden_dim;
        const half* exp_up_svh = up_svh + (size_t) expert_idx * intermediate_dim;
        const uint16_t* exp_down_trellis = down_trellis + expert_idx * trellis_stride;
        const half* exp_down_suh = down_suh + (size_t) expert_idx * intermediate_dim;
        const half* exp_down_svh = down_svh + (size_t) expert_idx * hidden_dim;

        // Gather + input hadamard for g, u
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int32_t* slots = slot_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_idx = slots[warp_idx / warps_per_token] / num_experts_per_tok;
                int token_off = warp_idx % warps_per_token;
                const float* in_ptr = hidden_state + (size_t) token_idx * hidden_dim + token_off * 128;
                had_fh_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_g + 128 * warp_idx,
                    exp_gate_suh + 128 * token_off,
                    0.088388347648f
                );
                had_fh_r_128_inner<true, false>
                (
                    in_ptr,
                    temp_state_u + 128 * warp_idx,
                    exp_up_suh + 128 * token_off,
                    0.088388347648f
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        }

        // 16 rows of A at a time, as the standalone GEMM does; fp32 outputs
        auto gemm = [&](const half* in_addr, float* out_addr, const uint16_t* trellis, const int size_k, const int size_n)
        {
            int size_m = token_count;
            while (size_m > 0)
            {
                exl3_gemm_kernel_inner<t_bits, true, cb, MOE_TILESIZE_M, MOE_TILESIZE_K, MOE_TILESIZE_N,
                                       MOE_SH_STAGES, MOE_FRAG_STAGES, false>
                    (in_addr, trellis, (void*) out_addr, MIN(size_m, 16), size_k, size_n, locks, nullptr);
                in_addr += 16 * size_k;
                out_addr += 16 * size_n;
                size_m -= 16;
            }
        };

        // g, u GEMM
        gemm(temp_state_g, temp_intermediate_g, exp_gate_trellis, hidden_dim, intermediate_dim);
        gemm(temp_state_u, temp_intermediate_u, exp_up_trellis, hidden_dim, intermediate_dim);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for g, u + silu gate + input hadamard for d
        {
            const int warps_per_token = intermediate_dim / 128;
            const int total_warps = token_count * warps_per_token;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int token_off = warp_idx % warps_per_token;
                had_ff_r_128_guad_inner
                (
                    temp_intermediate_g + 128 * warp_idx,
                    temp_intermediate_u + 128 * warp_idx,
                    temp_act + 128 * warp_idx,
                    exp_gate_svh + 128 * token_off,
                    exp_up_svh + 128 * token_off,
                    exp_down_suh + 128 * token_off,
                    0.088388347648f
                );
            }
            group_barrier(group_idx, group_size, barrier_counters_sense);
        }

        // d GEMM
        gemm(temp_act, temp_down, exp_down_trellis, intermediate_dim, hidden_dim);
        group_barrier(group_idx, group_size, barrier_counters_sense);

        // Output hadamard for d, stored to the assignment's own row
        {
            const int warps_per_token = hidden_dim / 128;
            const int total_warps = token_count * warps_per_token;
            const int32_t* slots = slot_sorted + start;
            for (int warp_idx = warp_idx0; warp_idx < total_warps; warp_idx += warps_per_group)
            {
                int slot = slots[warp_idx / warps_per_token];
                int token_off = warp_idx % warps_per_token;
                float* out_ptr = slot_out + (size_t) slot * hidden_dim + token_off * 128;
                had_ff_r_128_inner<false, true>
                (
                    temp_down + 128 * warp_idx,
                    out_ptr,
                    exp_down_svh + 128 * token_off,
                    0.088388347648f
                );
            }
        }

        // Draw the next ticket and publish it to the group through the end-of-expert barrier, which also protects
        // the temp buffers for reuse. Grabbed tickets continue from num_groups since 0..num_groups-1 are implicit
        if (block_idx == 0 && threadIdx.x == 0)
            sched[2 + group_idx] = num_groups + atomicAdd(&sched[0], 1);
        group_barrier(group_idx, group_size, barrier_counters_sense);
        ticket = sched[2 + group_idx];
    }

    // Retire group; last group out resets the scheduler for the next launch. The acq_rel increment orders each
    // group's earlier ticket grabs before the last group's reset (plain atomics are relaxed, so without this a
    // straggler's in-flight grab could land after the reset and leak into the next launch)
    if (block_idx == 0 && threadIdx.x == 0)
    {
        cuda::atomic_ref<int, cuda::thread_scope_device> next_ticket(sched[0]);
        cuda::atomic_ref<int, cuda::thread_scope_device> retired_groups(sched[1]);
        int retired = retired_groups.fetch_add(1, cuda::memory_order_acq_rel);
        if (retired == num_groups - 1)
        {
            next_ticket.store(0, cuda::memory_order_relaxed);
            retired_groups.store(0, cuda::memory_order_relaxed);
        }
    }
}
