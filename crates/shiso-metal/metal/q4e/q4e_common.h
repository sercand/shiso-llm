// Cross-family helpers for the qwen4exp (q4e) kernel set's Metal port.
//
// The Metal spelling of `kernels/cuda/q4e/q4e.cuh` and the two helpers it
// borrows from `shiso_dev_inc/common.cuh`. Every entry in `metal/q4e/` is a
// port of the CUDA entry of the same name: the host code in
// `crates/shiso-engine/src/sparse/` launches both by that name, with the same
// argument order and the same grid, so a kernel here binds its parameters at
// `[[buffer(i)]]` for the CUDA parameter at position `i` and reads `grid` and
// `block` exactly as CUDA does.
//
// This file is concatenated ahead of every q4e entry by `SOURCES` in
// `src/compile.rs`; the library is one source string, so nothing below may be
// `#include`d by a sibling.
//
// The reductions keep the CUDA reduction SHAPE -- a `shfl_down` fold inside a
// simdgroup, lane 0 publishes, a barrier, simdgroup 0 folds the partials --
// rather than a mathematically equal rewrite, because a port is graded against
// the reference's summation order. A threadgroup array cannot be declared
// inside a helper, so each takes its scratch from the caller: `Q4E_SLOTS`
// floats (32 partials and one broadcast slot).
//
// No `#pragma METAL fp math_mode` here. Fast math is the library default.

#include <metal_stdlib>
using namespace metal;

// `shiso_core::page`: a session's cache is paged in runs of this many
// positions, and a kernel finds a position's physical row through the page
// table. Must stay equal to the Rust constants.
constant constexpr uint Q4E_PAGE_TOKENS = 256u;
constant constexpr uint Q4E_PAGE_SHIFT = 8u;
constant constexpr uint Q4E_PAGE_MASK = Q4E_PAGE_TOKENS - 1u;
// Rows the mRoPE table carries in front of a chunk's first token.
constant constexpr uint Q4E_MROPE_BACK = 4u;

#define Q4E_SLOTS 33u

/// `warp_sum_f32`: a decreasing-offset fold; lane 0 holds the total.
static inline float q4e_warp_sum(float v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        v += simd_shuffle_down(v, off);
    }
    return v;
}

/// `warp_max_f32`, with `fmax` so the result is exact under any order.
static inline float q4e_warp_max(float v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        v = fmax(v, simd_shuffle_down(v, off));
    }
    return v;
}

/// A butterfly sum: every lane holds the total (`__shfl_xor_sync` fold).
static inline float q4e_warp_allsum(float v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        v += simd_shuffle_xor(v, off);
    }
    return v;
}

/// A butterfly max: every lane holds the maximum.
static inline float q4e_warp_allmax(float v) {
    for (ushort off = 16; off > 0; off >>= 1) {
        v = fmax(v, simd_shuffle_xor(v, off));
    }
    return v;
}

/// `q4e_block_sum`: block-wide sum broadcast to every thread.
static inline float q4e_block_sum(float v, threadgroup float *s, uint tid, uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = q4e_warp_sum(v);
    if (lane == 0u) {
        s[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? s[tid] : 0.0f;
    if (warp == 0u) {
        v = q4e_warp_sum(v);
    }
    if (tid == 0u) {
        s[32] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = s[32];
    // The next call may reuse `s`; nobody may overwrite it before every
    // thread has read the broadcast.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/// `q4e_block_max`: block-wide max broadcast to every thread.
static inline float q4e_block_max(float v, threadgroup float *s, uint tid, uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    v = q4e_warp_max(v);
    if (lane == 0u) {
        s[warp] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = (nthreads + 31u) >> 5u;
    v = (tid < n_warp) ? s[tid] : -INFINITY;
    if (warp == 0u) {
        v = q4e_warp_max(v);
    }
    if (tid == 0u) {
        s[32] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float total = s[32];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return total;
}

/// `q4e_block_argmax`: ties resolve toward the lower index, which is what a
/// stable descending argsort does. `sv` and `si` need 32 entries each.
static inline void q4e_block_argmax(float v, int i, thread float *out_v, thread int *out_i,
                                    threadgroup float *sv, threadgroup int *si,
                                    uint tid, uint nthreads) {
    const uint lane = tid & 31u;
    const uint warp = tid >> 5u;
    for (ushort d = 16; d > 0; d >>= 1) {
        const float ov = simd_shuffle_down(v, d);
        const int oi = simd_shuffle_down(i, d);
        if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
            v = ov;
            i = oi;
        }
    }
    if (lane == 0u) {
        sv[warp] = v;
        si[warp] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint n_warp = (nthreads + 31u) >> 5u;
    if (warp == 0u) {
        v = (lane < n_warp) ? sv[lane] : -INFINITY;
        i = (lane < n_warp) ? si[lane] : -1;
        for (ushort d = 16; d > 0; d >>= 1) {
            const float ov = simd_shuffle_down(v, d);
            const int oi = simd_shuffle_down(i, d);
            if (ov > v || (ov == v && oi >= 0 && (i < 0 || oi < i))) {
                v = ov;
                i = oi;
            }
        }
        if (lane == 0u) {
            sv[0] = v;
            si[0] = i;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    *out_v = sv[0];
    *out_i = si[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline void q4e_store(device float *p, float v) { *p = v; }
static inline void q4e_store(device half *p, float v) { *p = half(v); }
static inline float q4e_load(device const float *p) { return *p; }
static inline float q4e_load(device const half *p) { return float(*p); }

/// A brain-float bit pattern widened to single precision, exactly.
static inline float q4e_bf16(ushort bits) { return as_type<float>(uint(bits) << 16); }

/// `q4e_rope_mrope`: interleaved multimodal rotary over `v[0 .. n_rot)`,
/// pair `i` belonging to axis `i % 3` until that axis's section is used up.
///
/// Accurate `pow`/`sincos` (the `precise::` forms), as the CUDA body insists:
/// at base 1e7 the fast forms are a systematic rotation offset.
template <typename P>
static inline void q4e_rope_mrope(P v, uint n_rot, float base,
                                  int pos_t, int pos_h, int pos_w,
                                  uint sec_t, uint sec_h, uint sec_w,
                                  uint lane, uint nlanes) {
    const uint half_rot = n_rot / 2u;
    for (uint i = lane; i < half_rot; i += nlanes) {
        uint axis = i % 3u;
        const uint used = i / 3u;
        const uint budget = (axis == 0u) ? sec_t : ((axis == 1u) ? sec_h : sec_w);
        if (used >= budget) {
            axis = 0u;
        }
        const int p = (axis == 0u) ? pos_t : ((axis == 1u) ? pos_h : pos_w);
        const float theta = float(uint(p)) * precise::pow(base, -float(2u * i) / float(n_rot));
        float c;
        const float s = precise::sincos(theta, c);
        const float x0 = v[i];
        const float x1 = v[i + half_rot];
        v[i] = x0 * c - x1 * s;
        v[i + half_rot] = x0 * s + x1 * c;
    }
}

/// `q4e_kv_row`: the physical cache row of logical position `p`.
static inline uint q4e_kv_row(device const int *pages, uint p) {
    return uint(pages[p >> Q4E_PAGE_SHIFT]) * Q4E_PAGE_TOKENS + (p & Q4E_PAGE_MASK);
}
