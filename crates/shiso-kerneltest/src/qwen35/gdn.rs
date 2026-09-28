//! Device cases for the qwen35 GDN (gated delta network) kernels.
//!
//! Part of task-05 (docs/superpowers/plans/tasks/task-05-qwen35-msl-kernels.md).
//!
//! The delta rule is implemented twice and the two agree only if the state carry
//! is right: `kernel_q35_gdn_recurrent` walks tokens one at a time, while
//! `kernel_q35_gdn_chunk` solves the same recurrence in runs of 64 tokens and
//! carries the state across the run boundaries.  The chunk case therefore compares
//! the two device paths at lengths that straddle the 64-token run boundaries, and
//! the recurrent kernel is separately held to a naive CPU reading of the same
//! update, so the two device paths cannot agree on the same mistake.
//!
//! The conv case checks the other half of the same property: a five-token run and a
//! two-then-three launch over the same window must produce the same values, which
//! is what lets a chunked prefill reuse a decode-sized kernel.
//!
//! The `#[test]` wrappers live in each backend's own driver -
//! `crates/shiso-metal/tests/q35_gdn.rs`, `crates/shiso-cuda/tests/q35_gdn.rs`.
//! `RECURRENT_VARIANTS` and `CHUNK_VARIANTS` are here rather than there
//! because the variant list is a property of the kernel set, not of a device -
//! and `viable` asks the device which of them it will actually launch, so a
//! backend that carries one shape of each rather than four skips the rest
//! rather than failing on them.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "kernel argument structs take u32 dimensions while the fixtures are written in usize, and every fixture dimension here is a small literal"
)]

use crate::oracle;

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

/// GDN `head_dim` (`Q35_GDN_D`).  The recurrent kernel returns early on any other
/// value because both its register layout and its rotated sweep are built from it,
/// and the chunk kernel's `float4` slots are derived from it as well.
const HD: usize = 128;
/// Grouped key/query heads.  Value head `h` reads key head `h % N_HEAD_K`.
const N_HEAD_K: usize = 2;
const N_HEAD_V: usize = 4;
/// The `qkv` row both delta-rule kernels read, and the channel count of the conv
/// that produces it: `[q | k | v]`, one `head_dim` slice per head.
const K_OFFSET: usize = N_HEAD_K * HD;
const V_OFFSET: usize = 2 * K_OFFSET;
const STRIDE: usize = 2 * K_OFFSET + N_HEAD_V * HD;
/// Launch geometry and threadgroup memory, per `KERNELS.md`: the block-sum slots
/// (132 B) wherever a reduction happens, plus the two published rows for the
/// recurrent kernel and the `A` matrix with its two gate arrays for the chunk one.
const CONV_BLOCK: u32 = 256;
const NORM_BLOCK: u32 = 128;
const GATE_BLOCK: u32 = 256;
const NORM_SMEM: u32 = 132;
const RECURRENT_SMEM: u32 = 1_156;
const CHUNK_SMEM: u32 = 16_896;
/// Chunk against sequential: the chunk form re-associates the same updates into one
/// triangular solve plus one carry per run, so it agrees with the token-at-a-time
/// path to the rounding of a 64-token block rather than to `TOL_F32`.
const TOL_CHUNK: f32 = 1e-3;
/// Tokens per chunk run: `Q35_GDN_CHUNK` in `gdn_chunk.metal`, so a length above
/// this is the first one that reaches the carry between two runs.
const CHUNK_RUN: usize = 64;
/// `conv_kernel` 4: three earlier tokens in the window plus the current one.
const CONV_KERNEL: usize = 4;
const CONV_HIST: usize = CONV_KERNEL - 1;
/// `rms_eps` from the model geometry, used by the output norm.
const EPS: f32 = 1e-6;

/// `q35_args_gdn_conv`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConvArgs {
    channels: u32,
    conv_kernel: u32,
    n_tok: u32,
}

/// `q35_args_gdn_l2norm`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct L2NormArgs {
    head_dim: u32,
    n_heads: u32,
    stride: u32,
    offset: u32,
}

/// `q35_args_gdn_gates`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GateArgs {
    n_head: u32,
    n_tok: u32,
}

/// `q35_args_gdn_out_gate`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct OutGateArgs {
    head_dim: u32,
    n_head: u32,
    eps: f32,
}

/// `q35_args_gdn_recurrent` and `q35_args_gdn_chunk`, which share the layout.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GdnArgs {
    head_dim: u32,
    n_head_k: u32,
    n_head_v: u32,
    k_offset: u32,
    v_offset: u32,
    stride: u32,
    n_tok: u32,
}

/// Launches one kernel with the geometry `KERNELS.md` records for it.
fn launch(
    gpu: &Gpu,
    entry: &'static str,
    grid: Extent3,
    block: u32,
    smem_bytes: u32,
    args: &[KernelArg<'_>],
) {
    run(
        gpu,
        KernelRef::new("qwen35", entry),
        &LaunchArgs {
            grid,
            block: Extent3::linear(block),
            smem_bytes,
            args,
        },
    );
}

/// A deterministic fixture in `[-1, 1)`.
fn signal(len: usize, seed: u32) -> Vec<f32> {
    oracle::signal(len, seed)
}

/// The `qkv` rows for `n_tok` tokens, with the query and key slices normalized the
/// way the l2norm kernel normalizes them upstream of the delta rule.
///
/// Normalizing is what keeps the recurrence bounded rather than merely realistic:
/// the update is only stable while `beta * <k,k>` stays near 1, and an unnormalized
/// key row would have both sides agree on an exploding state instead of on the
/// update.
fn qkv_fixture(n_tok: usize, seed: u32) -> Vec<f32> {
    let raw = signal(n_tok * STRIDE, seed);
    let mut row = vec![0.0f32; n_tok * STRIDE];
    for t in 0..n_tok {
        let src = &raw[t * STRIDE..][..STRIDE];
        let dst = &mut row[t * STRIDE..][..STRIDE];
        for h in 0..2 * N_HEAD_K + N_HEAD_V {
            let chunk = &src[h * HD..][..HD];
            if h < 2 * N_HEAD_K {
                let norm = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
                for (slot, v) in dst[h * HD..][..HD].iter_mut().zip(chunk) {
                    *slot = v / norm;
                }
            } else {
                dst[h * HD..][..HD].copy_from_slice(chunk);
            }
        }
    }
    row
}

/// The decay gate the two delta-rule kernels consume.
///
/// `kernel_q35_gdn_gates` produces `softplus(alpha_proj + dt_bias) * a` with `a`
/// holding `-exp(A_log)`, so the result is negative and both consumers exponentiate
/// it directly.  A run of 64 tokens multiplies 64 of those exponentials together, so
/// the fixture stays in `[-0.03, -0.01]`: a decay that a chunk carry can survive.
fn decay_fixture(len: usize, seed: u32) -> Vec<f32> {
    signal(len, seed)
        .into_iter()
        .map(|v| -0.01 - 0.02 * (0.5 + 0.5 * v))
        .collect()
}

/// The beta gate, inside the `(0, 1)` range `sigmoid` leaves it in.
fn beta_fixture(len: usize, seed: u32) -> Vec<f32> {
    signal(len, seed)
        .into_iter()
        .map(|v| 0.2 + 0.6 * (0.5 + 0.5 * v))
        .collect()
}

/// The sequential recurrence the kernel implements, as a naive `f32` loop:
///
/// ```text
///   sk     = S_row . k          sq = S_row . q
///   delta  = (v[j] - dec * sk) * beta
///   out[j] = (dec * sq + delta * <k,q>) * rsqrt(head_dim)
///   S_row  = S_row * dec + delta * k
/// ```
///
/// `state` is `[head][value][key]` and is updated in place, so the caller can
/// compare the carry as well as the output.  The device reaches the same numbers
/// through one quarter of the row per lane plus a four-lane butterfly, which is why
/// the comparison is by tolerance rather than by bits.
fn recurrent_oracle(
    qkv: &[f32],
    decay: &[f32],
    beta: &[f32],
    n_tok: usize,
    state: &mut [f32],
) -> Vec<f32> {
    let scale = 1.0 / (HD as f32).sqrt();
    let mut out = vec![0.0; n_tok * N_HEAD_V * HD];
    for t in 0..n_tok {
        let row = &qkv[t * STRIDE..][..STRIDE];
        for h in 0..N_HEAD_V {
            let hk = h % N_HEAD_K;
            let q = &row[hk * HD..][..HD];
            let k = &row[K_OFFSET + hk * HD..][..HD];
            let v = &row[V_OFFSET + h * HD..][..HD];
            let dec = decay[t * N_HEAD_V + h].exp();
            let bt = beta[t * N_HEAD_V + h];
            let kq = oracle::dot(k, q);
            for j in 0..HD {
                let srow = &mut state[h * HD * HD + j * HD..][..HD];
                let sk = oracle::dot(srow, k);
                let sq = oracle::dot(srow, q);
                let delta = (v[j] - dec * sk) * bt;
                out[(t * N_HEAD_V + h) * HD + j] = (dec * sq + delta * kq) * scale;
                for i in 0..HD {
                    srow[i] = srow[i] * dec + delta * k[i];
                }
            }
        }
    }
    out
}

/// The seven-field argument block both delta-rule kernels take.
fn gdn_args(n_tok: usize) -> GdnArgs {
    GdnArgs {
        head_dim: HD as u32,
        n_head_k: N_HEAD_K as u32,
        n_head_v: N_HEAD_V as u32,
        k_offset: K_OFFSET as u32,
        v_offset: V_OFFSET as u32,
        stride: STRIDE as u32,
        n_tok: n_tok as u32,
    }
}

/// Launches one of the two delta-rule kernels over `n_tok` tokens and returns the
/// output together with the state it left behind.
///
/// `qkv`, `decay` and `beta` are read from token 0 of their buffers, so a caller
/// that wants to continue a run feeds in the tail of the fixture and the state the
/// previous launch wrote.
#[expect(
    clippy::too_many_arguments,
    reason = "a launch helper takes the buffers, the geometry and the length it launches over"
)]
fn delta_rule(
    gpu: &Gpu,
    entry: &'static str,
    block: u32,
    smem_bytes: u32,
    qkv: &DeviceBuffer,
    decay: &DeviceBuffer,
    beta: &DeviceBuffer,
    state: &DeviceBuffer,
    out: &DeviceBuffer,
    n_tok: usize,
) -> (Vec<f32>, Vec<f32>) {
    delta_rule_spill(
        gpu, entry, block, smem_bytes, qkv, decay, beta, state, out, None, n_tok,
    )
}

/// `delta_rule` with the optional scratch buffer the spilling variants take.
///
/// `kernel_q35_gdn_chunk_hybrid` is the one delta-rule entry point whose
/// signature is not the six-buffer one: it keeps its deltas in a device
/// scratch instead of in registers, bound at index 6, because the 32 KB that
/// scratch needs does not fit beside `s_A` in threadgroup memory.  Every other
/// entry point passes `None` and binds nothing there, so the extra buffer is
/// a property of the one variant rather than of the family.
#[expect(
    clippy::too_many_arguments,
    reason = "a launch helper takes the buffers, the geometry and the length it launches over"
)]
fn delta_rule_spill(
    gpu: &Gpu,
    entry: &'static str,
    block: u32,
    smem_bytes: u32,
    qkv: &DeviceBuffer,
    decay: &DeviceBuffer,
    beta: &DeviceBuffer,
    state: &DeviceBuffer,
    out: &DeviceBuffer,
    spill: Option<&DeviceBuffer>,
    n_tok: usize,
) -> (Vec<f32>, Vec<f32>) {
    let args = gdn_args(n_tok);
    let mut params = vec![
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(state.view()),
        KernelArg::Buffer(qkv.view()),
        KernelArg::Buffer(decay.view()),
        KernelArg::Buffer(beta.view()),
    ];
    if let Some(spill) = spill {
        params.push(KernelArg::Buffer(spill.view()));
    }
    launch(
        gpu,
        entry,
        Extent3::linear(N_HEAD_V as u32),
        block,
        smem_bytes,
        &params,
    );
    drain(gpu);
    (gpu.read_f32(out), gpu.read_f32(state))
}

/// Launches the conv over `n_tok` tokens of `x`, returning output and window.
fn conv(
    gpu: &Gpu,
    x: &DeviceBuffer,
    w: &DeviceBuffer,
    state: &DeviceBuffer,
    out: &DeviceBuffer,
    n_tok: usize,
    channels: usize,
) -> (Vec<f32>, Vec<f32>) {
    let args = ConvArgs {
        channels: channels as u32,
        conv_kernel: CONV_KERNEL as u32,
        n_tok: n_tok as u32,
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(state.view()),
        KernelArg::Buffer(x.view()),
        KernelArg::Buffer(w.view()),
    ];
    launch(
        gpu,
        "kernel_q35_gdn_conv",
        Extent3::linear((channels as u32).div_ceil(CONV_BLOCK)),
        CONV_BLOCK,
        0,
        &params,
    );
    drain(gpu);
    (gpu.read_f32(out), gpu.read_f32(state))
}

/// Launches the in-place L2 norm over `n_heads` rows of `head_dim` at `offset` in
/// each token row, and returns the whole buffer so the caller can also check that
/// nothing outside the normalized group moved.
fn l2norm(gpu: &Gpu, qkv: &DeviceBuffer, n_heads: usize, offset: usize, n_tok: usize) -> Vec<f32> {
    let args = L2NormArgs {
        head_dim: HD as u32,
        n_heads: N_HEAD_K as u32,
        stride: STRIDE as u32,
        offset: offset as u32,
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(qkv.view()),
    ];
    launch(
        gpu,
        "kernel_q35_gdn_l2norm",
        Extent3::new(n_heads as u32, n_tok as u32, 1),
        NORM_BLOCK,
        NORM_SMEM,
        &params,
    );
    drain(gpu);
    gpu.read_f32(qkv)
}

/// Launches the decay/beta gate kernel, which writes both halves in place.
#[expect(
    clippy::too_many_arguments,
    reason = "a launch helper takes the buffers, the geometry and the length it launches over"
)]
fn gates(
    gpu: &Gpu,
    alpha_proj: &DeviceBuffer,
    beta_proj: &DeviceBuffer,
    dt_bias: &DeviceBuffer,
    a: &DeviceBuffer,
    decay: &DeviceBuffer,
    beta: &DeviceBuffer,
    n_tok: usize,
) {
    let args = GateArgs {
        n_head: N_HEAD_V as u32,
        n_tok: n_tok as u32,
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(decay.view()),
        KernelArg::Buffer(beta.view()),
        KernelArg::Buffer(alpha_proj.view()),
        KernelArg::Buffer(beta_proj.view()),
        KernelArg::Buffer(dt_bias.view()),
        KernelArg::Buffer(a.view()),
    ];
    launch(
        gpu,
        "kernel_q35_gdn_gates",
        Extent3::linear(((N_HEAD_V * n_tok) as u32).div_ceil(GATE_BLOCK)),
        GATE_BLOCK,
        0,
        &params,
    );
    drain(gpu);
}

/// Launches the output norm-and-gate kernel: one threadgroup per (token, head).
fn out_gate(
    gpu: &Gpu,
    attn: &DeviceBuffer,
    z: &DeviceBuffer,
    w: &DeviceBuffer,
    out: &DeviceBuffer,
    n_tok: usize,
) -> Vec<f32> {
    let args = OutGateArgs {
        head_dim: HD as u32,
        n_head: N_HEAD_V as u32,
        eps: EPS,
    };
    let params = [
        KernelArg::Bytes(bytemuck::bytes_of(&args)),
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(attn.view()),
        KernelArg::Buffer(z.view()),
        KernelArg::Buffer(w.view()),
    ];
    launch(
        gpu,
        "kernel_q35_gdn_out_gate",
        Extent3::linear((n_tok * N_HEAD_V) as u32),
        NORM_BLOCK,
        NORM_SMEM,
        &params,
    );
    drain(gpu);
    gpu.read_f32(out)
}

/// The kernel's own `log1p`: `x * ln(u) / (u - 1)` with the `u == 1` early return.
///
/// Mirrored instead of replaced by `ln_1p` because the decay gate's
/// `log1p(exp(-30))` lands exactly on that early return -- `exp(-30)` is far below
/// one ulp of 1, so `u` rounds to exactly 1 and the kernel returns its argument --
/// and the gate is held to the `f32` tolerance.
#[expect(
    clippy::float_cmp,
    reason = "the early return is the branch the kernel takes, and it is only taken on an exact equality"
)]
fn log1p_like_kernel(x: f32) -> f32 {
    let u = 1.0 + x;
    if u == 1.0 { x } else { x * u.ln() / (u - 1.0) }
}

/// The decay gate's softplus, including the `x > 20` branch the guard takes.
fn softplus_like_kernel(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        log1p_like_kernel(x.exp())
    }
}

fn sigmoid_like_kernel(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu_like_kernel(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The causal depthwise conv plus silu, one channel at a time, with the same
/// separate `+ wc[hist] * cur` term the kernel uses rather than folding the current
/// token into the window loop.
///
/// Returns the outputs and the window the run leaves behind, which is what the next
/// call reads.
fn conv_oracle(x: &[f32], w: &[f32], state: &[f32], n_tok: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0; n_tok * STRIDE];
    let mut next = vec![0.0; CONV_HIST * STRIDE];
    for c in 0..STRIDE {
        let mut win = [0.0f32; CONV_HIST];
        for (i, slot) in win.iter_mut().enumerate() {
            *slot = state[i * STRIDE + c];
        }
        let wc = &w[c * CONV_KERNEL..][..CONV_KERNEL];
        for t in 0..n_tok {
            let cur = x[t * STRIDE + c];
            let mut acc = 0.0f32;
            for i in 0..CONV_HIST {
                acc += wc[i] * win[i];
            }
            acc += wc[CONV_HIST] * cur;
            out[t * STRIDE + c] = silu_like_kernel(acc);
            for i in 0..CONV_HIST - 1 {
                win[i] = win[i + 1];
            }
            win[CONV_HIST - 1] = cur;
        }
        for i in 0..CONV_HIST {
            next[i * STRIDE + c] = win[i];
        }
    }
    (out, next)
}

/// The conv must produce the same values whether it sees five tokens at once or a
/// two-token call followed by a three-token one that resumes from the window the
/// first call left behind.
///
/// This is the property a chunked prefill rests on: the kernel is launched with a
/// token count, so a run that is split across two launches has to produce the same
/// values as one that is not.  The window is seeded non-zero so a kernel that
/// skipped the state load would read zeros instead of the previous tail, and the
/// carried window is compared as well, since that is what the next call consumes.
pub fn conv_resumes_from_the_window_a_previous_call_left(gpu: &Gpu) {
    let n_tok = 5;
    let x = signal(n_tok * STRIDE, 11);
    let w = signal(STRIDE * CONV_KERNEL, 12);
    let seeded = signal(CONV_HIST * STRIDE, 13);

    let x_buf = gpu.upload(&f32_bytes(&x));
    let w_buf = gpu.upload(&f32_bytes(&w));
    let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * STRIDE]));
    let state_buf = gpu.upload(&f32_bytes(&seeded));
    let (got, final_state) = conv(gpu, &x_buf, &w_buf, &state_buf, &out_buf, n_tok, STRIDE);

    let (expected, expected_state) = conv_oracle(&x, &w, &seeded, n_tok);
    for (i, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        crate::assert_close!(
            *got,
            *expected,
            oracle::TOL_F32,
            "token {} channel {}",
            i / STRIDE,
            i % STRIDE,
        );
    }
    for (i, (got, expected)) in final_state.iter().zip(expected_state.iter()).enumerate() {
        crate::assert_close!(*got, *expected, oracle::TOL_F32, "carried window slot {i}");
    }

    // The same five tokens as a two-token call and then a three-token one that
    // resumes from the state the first launch wrote.
    let head_buf = gpu.upload(&f32_bytes(&x[..2 * STRIDE]));
    let head_out = gpu.upload(&f32_bytes(&vec![0.0; 2 * STRIDE]));
    // The five-token run above wrote its tail into `state_buf`, so the head run is
    // pointed at a fresh copy of the seed: pointed at the buffer the five-token run
    // just wrote, it would resume from that run's tail instead of from the window
    // the full run started with.
    let head_state = gpu.upload(&f32_bytes(&seeded));
    let (_, carried) = conv(gpu, &head_buf, &w_buf, &head_state, &head_out, 2, STRIDE);
    let carried_buf = gpu.upload(&f32_bytes(&carried));
    let tail_buf = gpu.upload(&f32_bytes(&x[2 * STRIDE..]));
    let tail_out = gpu.upload(&f32_bytes(&vec![0.0; (n_tok - 2) * STRIDE]));
    let (split_tail, split_state) = conv(
        gpu,
        &tail_buf,
        &w_buf,
        &carried_buf,
        &tail_out,
        n_tok - 2,
        STRIDE,
    );

    let head = gpu.read_f32(&head_out);
    for (i, (got, expected)) in head.iter().zip(got[..2 * STRIDE].iter()).enumerate() {
        crate::assert_close!(
            *got,
            *expected,
            oracle::TOL_F32,
            "first two tokens, flat {i}"
        );
    }
    for (i, (got, expected)) in split_tail.iter().zip(got[2 * STRIDE..].iter()).enumerate() {
        crate::assert_close!(
            *got,
            *expected,
            oracle::TOL_F32,
            "resumed token {} channel {}",
            (i / STRIDE) + 2,
            i % STRIDE,
        );
    }
    for (i, (got, expected)) in split_state.iter().zip(final_state.iter()).enumerate() {
        crate::assert_close!(
            *got,
            *expected,
            oracle::TOL_F32,
            "window after the split run, slot {i}",
        );
    }
}

/// The L2 norm is in place, over one group of head rows of a strided row, so the
/// case checks three things at once: the normalization itself, that every head row
/// is normalized against *its own* sum, and that nothing outside the group the
/// launch was pointed at moved.
///
/// The fixture is deliberately *not* pre-normalized here: normalizing an already
/// unit-length row is a no-op and would pass for the wrong reason.
pub fn l2norm_normalizes_each_head_row_in_place(gpu: &Gpu) {
    let n_tok = 3;
    let raw = signal(n_tok * STRIDE, 21);
    let buf = gpu.upload(&f32_bytes(&raw));

    let after = l2norm(gpu, &buf, N_HEAD_K, K_OFFSET, n_tok);
    let mut touched = vec![false; after.len()];
    for t in 0..n_tok {
        for h in 0..N_HEAD_K {
            let base = t * STRIDE + K_OFFSET + h * HD;
            let src = &raw[base..][..HD];
            let inv = 1.0 / (src.iter().map(|v| v * v).sum::<f32>() + 1e-12).sqrt();
            for d in 0..HD {
                touched[base + d] = true;
                crate::assert_close!(
                    after[base + d],
                    src[d] * inv,
                    oracle::TOL_F32,
                    "token {t} head {h} dim {d}",
                );
            }
        }
    }
    // Everything the launch was not pointed at has to come back byte for byte,
    // which is what pins the launch to the group it was pointed at.
    for (i, (got, given)) in after.iter().zip(raw.iter()).enumerate() {
        if touched[i] {
            continue;
        }
        crate::assert_close!(*got, *given, oracle::TOL_F32, "untouched flat index {i}");
    }
}
/// `decay = softplus(alpha_proj + dt_bias) * a` and `beta = sigmoid(beta_proj)`,
/// with both guards exercised: one input above the softplus cutoff of 20, where the
/// naive form overflows, and one far below it, where `exp` lands under the
/// representable threshold of `1 + x` and the kernel's `log1p` takes its `u == 1`
/// early return.
pub fn gates_apply_the_softplus_guard_and_the_sigmoid(gpu: &Gpu) {
    let n_tok = 3;
    let n = N_HEAD_V * n_tok;
    let mut alpha_proj = signal(n, 31);
    alpha_proj[0] = 25.0;
    alpha_proj[1] = -30.0;
    let beta_proj = signal(n, 32);
    let dt_bias = signal(N_HEAD_V, 33);
    // `a` is -exp(A_log), so the decay it produces is negative.
    let a: Vec<f32> = signal(N_HEAD_V, 34)
        .into_iter()
        .map(|v| -(0.5 + v.abs()))
        .collect();

    let alpha_buf = gpu.upload(&f32_bytes(&alpha_proj));
    let beta_proj_buf = gpu.upload(&f32_bytes(&beta_proj));
    let dt_bias_buf = gpu.upload(&f32_bytes(&dt_bias));
    let a_buf = gpu.upload(&f32_bytes(&a));
    let decay_buf = gpu.upload(&f32_bytes(&vec![0.0; n]));
    let beta_buf = gpu.upload(&f32_bytes(&vec![0.0; n]));

    gates(
        gpu,
        &alpha_buf,
        &beta_proj_buf,
        &dt_bias_buf,
        &a_buf,
        &decay_buf,
        &beta_buf,
        n_tok,
    );

    let decay = gpu.read_f32(&decay_buf);
    let beta = gpu.read_f32(&beta_buf);
    for i in 0..n {
        let h = i % N_HEAD_V;
        let expected = softplus_like_kernel(alpha_proj[i] + dt_bias[h]) * a[h];
        crate::assert_close!(
            decay[i],
            expected,
            oracle::TOL_F32,
            "decay at flat index {i}"
        );
    }
    for (i, (got, given)) in beta.iter().zip(beta_proj.iter()).enumerate() {
        crate::assert_close!(
            *got,
            sigmoid_like_kernel(*given),
            oracle::TOL_F32,
            "beta at flat index {i}"
        );
    }
}

/// The output norm is `attn * rsqrt(mean(attn^2) + eps) * w * silu(z)`, in that
/// multiplication order, over one head row per threadgroup.
///
/// Three properties are pinned beyond the values.  The mean is taken over one head
/// rather than over the whole token row, so a head sitting next to a huge one
/// normalizes exactly as it would alone; the gate reads the RAW companion slice
/// rather than the normalized value, which is the difference between `ssm_norm`
/// being applied once and twice; and the gate is `silu`, not `sigmoid`.
///
/// That last one is why this case is worth its length.  The two activations differ
/// by a factor of the gate's own value, which no shape check and no launch
/// validation can see - it is arithmetic, and the only thing that catches it is a
/// number compared against a number.  An earlier version of this case pinned the
/// sigmoid, and it passed against a kernel that made the whole model answer in
/// punctuation.
pub fn out_gate_normalizes_one_head_and_scales_it_by_the_gate(gpu: &Gpu) {
    let n_tok = 3;
    let n_row = n_tok * N_HEAD_V * HD;
    let attn = signal(n_row, 51);
    let z = signal(n_row, 52);
    // A weight with both signs, so a sign error in the weight cannot hide.
    let w = signal(HD, 53);

    let attn_buf = gpu.upload(&f32_bytes(&attn));
    let z_buf = gpu.upload(&f32_bytes(&z));
    let w_buf = gpu.upload(&f32_bytes(&w));
    let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_row]));
    let got = out_gate(gpu, &attn_buf, &z_buf, &w_buf, &out_buf, n_tok);

    for t in 0..n_tok {
        for h in 0..N_HEAD_V {
            let base = (t * N_HEAD_V + h) * HD;
            let sum = attn[base..][..HD].iter().map(|v| v * v).sum::<f32>();
            let scale = 1.0 / (sum / (HD as f32) + EPS).sqrt();
            for d in 0..HD {
                let expected = attn[base + d] * scale * w[d] * silu_like_kernel(z[base + d]);
                crate::assert_close!(
                    got[base + d],
                    expected,
                    oracle::TOL_F32,
                    "token {t} head {h} dim {d}",
                );
            }
        }
    }
}

/// The recurrent kernel against the recurrence it is a translation of.
///
/// The oracle keeps the whole `f32` state and applies
/// `S <- e^d (I - beta k k^T) S + beta k v^T` per token, which is the same
/// arithmetic the kernel does with one quarter of each state row per lane.  Both
/// the emitted tokens and the state the call leaves behind are compared: a kernel
/// that gets the outputs right and the carry wrong still has to fail here, because
/// a wrong carry is invisible for eight tokens and ruinous by token 4000.
///
/// It runs at whichever width this device will launch, rather than at a fixed
/// 512 lanes.  That is not a convenience: the variants are bit-identical, so
/// any of them answers this question, and a case pinned to one width is a case
/// that reports "the recurrence is wrong" on a part whose only problem is a
/// narrower threadgroup.  `every_viable_recurrent_variant_matches_the_oracle`
/// then says the same of every other width; what is here and not there is the
/// carried state, which that case does not read.
pub fn recurrent_matches_a_naive_delta_rule(gpu: &Gpu) {
    let Some(&(entry, lanes)) = RECURRENT_VARIANTS
        .iter()
        .find(|&&(entry, lanes)| viable(gpu, entry, lanes))
    else {
        // Not a `return`. This case is the only one that reads the carried
        // state, so a device that can launch none of these widths has nothing
        // here to compare against and must say so - a quiet return reports a
        // pass having launched nothing, which is the shape every silent defect
        // on this branch has had.
        panic!("no recurrent variant is viable on this device: {RECURRENT_VARIANTS:?}");
    };
    let n_tok = 8;
    let qkv = qkv_fixture(n_tok, 61);
    let decay = decay_fixture(n_tok * N_HEAD_V, 62);
    let beta = beta_fixture(n_tok * N_HEAD_V, 63);

    let qkv_buf = gpu.upload(&f32_bytes(&qkv));
    let decay_buf = gpu.upload(&f32_bytes(&decay));
    let beta_buf = gpu.upload(&f32_bytes(&beta));
    let state_buf = gpu.upload(&f32_bytes(&vec![0.0; N_HEAD_V * HD * HD]));
    let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD]));

    let mut expected_state = vec![0.0f32; N_HEAD_V * HD * HD];
    let expected = recurrent_oracle(&qkv, &decay, &beta, n_tok, &mut expected_state);
    let (got, state) = delta_rule(
        gpu,
        entry,
        lanes,
        recurrent_smem(entry),
        &qkv_buf,
        &decay_buf,
        &beta_buf,
        &state_buf,
        &out_buf,
        n_tok,
    );

    for (i, (got, expected)) in got.iter().zip(expected.iter()).enumerate() {
        crate::assert_close!(
            *got,
            *expected,
            oracle::TOL_F32,
            "token {} head {}",
            i / (N_HEAD_V * HD),
            (i / HD) % N_HEAD_V,
        );
    }
    for (i, (got, expected)) in state.iter().zip(expected_state.iter()).enumerate() {
        crate::assert_close!(
            *got,
            *expected,
            oracle::TOL_F32,
            "carried state, head {i} value {} key {}",
            (i / HD) % HD,
            i % HD,
        );
    }
}

/// Every recurrent variant, with the lanes it must be launched at.
pub(crate) const RECURRENT_VARIANTS: [(&str, u32); 4] = [
    ("kernel_q35_gdn_recurrent_s1", 128),
    ("kernel_q35_gdn_recurrent_s2", 256),
    ("kernel_q35_gdn_recurrent_s4", 512),
    ("kernel_q35_gdn_recurrent_hybrid", 128),
];

/// Threadgroup bytes one recurrent variant needs: the hybrid stages one
/// quarter of the state (128 lanes * 32 floats * 4 B = 16384 B), the rest
/// keep all of it in registers. 16384 B, not 32768 B: staging two quarters
/// plus the 1156 B of scratch this kernel already uses is 33924 B, over the
/// 32768 B threadgroup memory limit every Apple Silicon GPU (Mac GPU family
/// 2) is built to -- not a device-specific shortfall, a family-wide one.
fn recurrent_smem(entry: &str) -> u32 {
    if entry.ends_with("_hybrid") {
        RECURRENT_SMEM + 16_384
    } else {
        RECURRENT_SMEM
    }
}

/// A variant this device cannot launch is skipped, not failed: which
/// variants are viable is a property of the device under test.
fn viable(gpu: &Gpu, entry: &'static str, lanes: u32) -> bool {
    let limit = shiso_core::Backend::max_block_threads(
        gpu.device().as_ref(),
        KernelRef::new("qwen35", entry),
    );
    matches!(limit, Some(max) if max >= lanes)
}

/// Every variant this device can launch has to agree with the CPU oracle: the
/// register split changes how the state is spread across lanes, not the
/// numbers it produces.  A device that cannot build a variant at all (its
/// lane count is over this device's per-threadgroup limit) skips it instead
/// of failing, which is why the final assertion is `ran > 0` rather than
/// `ran == RECURRENT_VARIANTS.len()`.
pub fn every_viable_recurrent_variant_matches_the_oracle(gpu: &Gpu) {
    let n_tok = 6usize;

    let qkv = qkv_fixture(n_tok, 61);
    let decay = decay_fixture(n_tok * N_HEAD_V, 62);
    let beta = beta_fixture(n_tok * N_HEAD_V, 63);

    let mut expected_state = vec![0.0f32; N_HEAD_V * HD * HD];
    let expected = recurrent_oracle(&qkv, &decay, &beta, n_tok, &mut expected_state);

    let mut ran = 0usize;
    for (entry, lanes) in RECURRENT_VARIANTS {
        if !viable(gpu, entry, lanes) {
            continue;
        }
        ran += 1;
        let qkv_buf = gpu.upload(&f32_bytes(&qkv));
        let decay_buf = gpu.upload(&f32_bytes(&decay));
        let beta_buf = gpu.upload(&f32_bytes(&beta));
        let state_buf = gpu.upload(&f32_bytes(&vec![0.0; N_HEAD_V * HD * HD]));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD]));

        let (got, _state) = delta_rule(
            gpu,
            entry,
            lanes,
            recurrent_smem(entry),
            &qkv_buf,
            &decay_buf,
            &beta_buf,
            &state_buf,
            &out_buf,
            n_tok,
        );
        for (i, (got, want)) in got.iter().zip(expected.iter()).enumerate() {
            crate::assert_close!(*got, *want, oracle::TOL_F32, "{entry} element {i}");
        }
    }
    assert!(ran > 0, "no recurrent variant was viable on this device");
}

/// Variants differ in lane count and in nothing else. Byte equality, not
/// closeness: this is the property that lets the shape be chosen per
/// device, per boot, without the answer moving.
pub fn recurrent_variants_are_bit_identical(gpu: &Gpu) {
    let n_tok = 6usize;

    let qkv = qkv_fixture(n_tok, 61);
    let decay = decay_fixture(n_tok * N_HEAD_V, 62);
    let beta = beta_fixture(n_tok * N_HEAD_V, 63);

    let mut reference: Option<(&str, Vec<f32>, Vec<f32>)> = None;
    for (entry, lanes) in RECURRENT_VARIANTS {
        if !viable(gpu, entry, lanes) {
            continue;
        }
        let qkv_buf = gpu.upload(&f32_bytes(&qkv));
        let decay_buf = gpu.upload(&f32_bytes(&decay));
        let beta_buf = gpu.upload(&f32_bytes(&beta));
        let state_buf = gpu.upload(&f32_bytes(&vec![0.0; N_HEAD_V * HD * HD]));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD]));

        let (got, state) = delta_rule(
            gpu,
            entry,
            lanes,
            recurrent_smem(entry),
            &qkv_buf,
            &decay_buf,
            &beta_buf,
            &state_buf,
            &out_buf,
            n_tok,
        );
        match &reference {
            None => reference = Some((entry, got, state)),
            Some((first, want_out, want_state)) => {
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{entry} output differs from {first}",
                );
                assert_eq!(
                    state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{entry} state differs from {first}",
                );
            }
        }
    }
    assert!(reference.is_some(), "no recurrent variant was viable");
}

/// Every chunk variant, with the lanes it must be launched at.
///
/// `CSPLIT` lanes share one value column, so the lane count is `128 * CSPLIT`.
/// The list stops at 8: the canonical unit is a four-slot chain of the state
/// row, a row holds eight of them, and a lane has to hold a whole number of
/// units -- 16 would split one and there would be no order to reproduce.
pub(crate) const CHUNK_VARIANTS: [(&str, u32); 5] = [
    ("kernel_q35_gdn_chunk_c1", 128),
    ("kernel_q35_gdn_chunk_c2", 256),
    ("kernel_q35_gdn_chunk_c4", 512),
    ("kernel_q35_gdn_chunk_c8", 1024),
    ("kernel_q35_gdn_chunk_hybrid", 128),
];

/// Floats of device scratch the hybrid's spilled deltas need.
///
/// One delta per (value head, token of the run, value dimension): the deltas
/// are per-column, so the 128 lanes of a `CSPLIT = 1` threadgroup each need
/// their own row of `Q35_GDN_CHUNK` and the threadgroup needs 32 KB of them.
/// That is why they cannot be staged in threadgroup memory beside the 16896 B
/// `s_A` already claims of the 32768 B family limit, and so why this one
/// variant takes a seventh buffer.
const CHUNK_SPILL: usize = N_HEAD_V * CHUNK_RUN * HD;

/// Chunk variants differ in lane count and in nothing else.  Byte equality,
/// not closeness: this is the property that lets Task 8 choose a shape per
/// device, per boot, without the answer moving.
///
/// The length crosses a run boundary so the carry between two runs is compared
/// as well as the outputs -- a variant that gets every output right and the
/// carry wrong is exactly the failure a single-run fixture cannot see.
///
/// A variant this device cannot launch is skipped rather than failed: which
/// lane counts are viable is a property of the device under test, which is why
/// the final assertion is `is_some()` rather than a count.
pub fn chunk_variants_are_bit_identical(gpu: &Gpu) {
    let n_tok = CHUNK_RUN + 8;

    let qkv = qkv_fixture(n_tok, 71);
    let decay = decay_fixture(n_tok * N_HEAD_V, 72);
    let beta = beta_fixture(n_tok * N_HEAD_V, 73);

    let mut reference: Option<(&str, Vec<f32>, Vec<f32>)> = None;
    for (entry, lanes) in CHUNK_VARIANTS {
        if !viable(gpu, entry, lanes) {
            continue;
        }
        let qkv_buf = gpu.upload(&f32_bytes(&qkv));
        let decay_buf = gpu.upload(&f32_bytes(&decay));
        let beta_buf = gpu.upload(&f32_bytes(&beta));
        let state_buf = gpu.upload(&f32_bytes(&vec![0.0; N_HEAD_V * HD * HD]));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD]));
        let spill = gpu.upload(&f32_bytes(&vec![0.0; CHUNK_SPILL]));

        let (got, state) = delta_rule_spill(
            gpu,
            entry,
            lanes,
            CHUNK_SMEM,
            &qkv_buf,
            &decay_buf,
            &beta_buf,
            &state_buf,
            &out_buf,
            entry.ends_with("_hybrid").then_some(&spill),
            n_tok,
        );
        match &reference {
            None => reference = Some((entry, got, state)),
            Some((first, want_out, want_state)) => {
                assert_eq!(
                    got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{entry} output differs from {first}",
                );
                assert_eq!(
                    state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    want_state.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "{entry} state differs from {first}",
                );
            }
        }
    }
    assert!(reference.is_some(), "no chunk variant was viable");
}

/// Every chunk variant this device can launch solves the same recurrence the
/// sequential kernel does.
///
/// `chunk_variants_are_bit_identical` only says the variants agree with each
/// other, and agreeing is not being right: a rotation or a unit-to-memory
/// mapping that is wrong the same way at every width would satisfy it.  This
/// case pins the family to the other device path, which is itself pinned to the
/// CPU oracle by `every_viable_recurrent_variant_matches_the_oracle`, so the two
/// cannot agree on the same mistake.  That case and not
/// `recurrent_matches_a_naive_delta_rule`: both are pinned to the same CPU
/// oracle, but the per-variant one runs at whatever width this device will
/// launch, and the chain has to hold on the device in front of you.
///
/// The lengths straddle the run boundary because what a single run cannot see
/// is the carry, and they are the ten the legacy `kernel_q35_gdn_chunk` case
/// used to walk: that case dispatched the 1024-lane alias, which this change
/// deletes, and everything it checked is checked here at every width instead of
/// at one.  The looser tolerance is the re-association a 64-token solve costs.
pub fn every_viable_chunk_variant_matches_the_recurrence(gpu: &Gpu) {
    // Whichever sequential width this device can launch: they are bit-identical
    // to each other, so which one is the reference does not matter, but a
    // device that can launch none of them has nothing to compare against.
    let Some(&(seq_entry, seq_lanes)) = RECURRENT_VARIANTS
        .iter()
        .find(|&&(entry, lanes)| viable(gpu, entry, lanes))
    else {
        // The comment below this block already says a device where none are
        // viable "must not pass silently". A `return` here is exactly that, so
        // it is a panic.
        panic!("no recurrent variant is viable to compare a chunk against: {RECURRENT_VARIANTS:?}");
    };

    // This is the only case left that pins a chunk variant against the
    // oracle path (`chunk_variants_are_bit_identical` only checks the
    // variants against each other), so a device where none of them are
    // viable must not pass silently: every sibling case that loops over a
    // fallible `viable()` filter carries this same guard.
    let mut ran = 0usize;

    for n_tok in [
        1usize,
        2,
        3,
        CHUNK_RUN - 1,
        CHUNK_RUN,
        CHUNK_RUN + 1,
        CHUNK_RUN + 8,
        2 * CHUNK_RUN,
        2 * CHUNK_RUN + 1,
        3 * CHUNK_RUN,
    ] {
        let qkv = qkv_fixture(n_tok, 71);
        let decay = decay_fixture(n_tok * N_HEAD_V, 72);
        let beta = beta_fixture(n_tok * N_HEAD_V, 73);
        let qkv_buf = gpu.upload(&f32_bytes(&qkv));
        let decay_buf = gpu.upload(&f32_bytes(&decay));
        let beta_buf = gpu.upload(&f32_bytes(&beta));
        let seq_state = gpu.upload(&f32_bytes(&vec![0.0; N_HEAD_V * HD * HD]));
        let seq_out = gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD]));

        let (want, want_state) = delta_rule(
            gpu,
            seq_entry,
            seq_lanes,
            recurrent_smem(seq_entry),
            &qkv_buf,
            &decay_buf,
            &beta_buf,
            &seq_state,
            &seq_out,
            n_tok,
        );

        for (entry, lanes) in CHUNK_VARIANTS {
            if !viable(gpu, entry, lanes) {
                continue;
            }
            ran += 1;
            let state_buf = gpu.upload(&f32_bytes(&vec![0.0; N_HEAD_V * HD * HD]));
            let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD]));
            let spill = gpu.upload(&f32_bytes(&vec![0.0; CHUNK_SPILL]));
            let (got, state) = delta_rule_spill(
                gpu,
                entry,
                lanes,
                CHUNK_SMEM,
                &qkv_buf,
                &decay_buf,
                &beta_buf,
                &state_buf,
                &out_buf,
                entry.ends_with("_hybrid").then_some(&spill),
                n_tok,
            );
            for (i, (got, want)) in got.iter().zip(want.iter()).enumerate() {
                crate::assert_close!(
                    *got,
                    *want,
                    TOL_CHUNK,
                    "{entry} n_tok {n_tok} token {} head {} dim {}",
                    i / (N_HEAD_V * HD),
                    (i / HD) % N_HEAD_V,
                    i % HD,
                );
            }
            for (i, (got, want)) in state.iter().zip(want_state.iter()).enumerate() {
                crate::assert_close!(
                    *got,
                    *want,
                    TOL_CHUNK,
                    "{entry} n_tok {n_tok} state, head {} value {} key {}",
                    i / (HD * HD),
                    (i / HD) % HD,
                    i % HD,
                );
            }
        }
    }
    assert!(ran > 0, "no chunk variant was viable on this device");
}

/// Every recording twin of a recurrent variant, with the lanes it launches at.
pub(crate) const RECURRENT_TRAIL_VARIANTS: [(&str, &str, u32); 3] = [
    (
        "kernel_q35_gdn_recurrent_trail_s1",
        "kernel_q35_gdn_recurrent_s1",
        128,
    ),
    (
        "kernel_q35_gdn_recurrent_trail_s2",
        "kernel_q35_gdn_recurrent_s2",
        256,
    ),
    (
        "kernel_q35_gdn_recurrent_trail_s4",
        "kernel_q35_gdn_recurrent_s4",
        512,
    ),
];

/// A recording twin writes, for every row but the last, exactly the state a
/// plain launch over the rows up to it leaves - and otherwise answers exactly
/// what the plain variant answers. Byte equality: a verify pass winds back to
/// these records, and the serial decode it must equal ran the plain kernel.
pub fn recurrent_trail_records_are_the_per_row_walk(gpu: &Gpu) {
    let n_tok = 6usize;
    let qkv = qkv_fixture(n_tok, 71);
    let decay = decay_fixture(n_tok * N_HEAD_V, 72);
    let beta = beta_fixture(n_tok * N_HEAD_V, 73);
    let seed = signal(N_HEAD_V * HD * HD, 74);
    let state_len = N_HEAD_V * HD * HD;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let mut ran = 0usize;
    for (trail_entry, plain, lanes) in RECURRENT_TRAIL_VARIANTS {
        if !viable(gpu, trail_entry, lanes) {
            continue;
        }
        ran += 1;
        let qkv_buf = gpu.upload(&f32_bytes(&qkv));
        let decay_buf = gpu.upload(&f32_bytes(&decay));
        let beta_buf = gpu.upload(&f32_bytes(&beta));
        let state_buf = gpu.upload(&f32_bytes(&seed));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD]));
        let trail_buf = gpu.upload(&f32_bytes(&vec![0.0; (n_tok - 1) * state_len]));
        let args = gdn_args(n_tok);
        launch(
            gpu,
            trail_entry,
            Extent3::linear(N_HEAD_V as u32),
            lanes,
            RECURRENT_SMEM,
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out_buf.view()),
                KernelArg::Buffer(state_buf.view()),
                KernelArg::Buffer(qkv_buf.view()),
                KernelArg::Buffer(decay_buf.view()),
                KernelArg::Buffer(beta_buf.view()),
                KernelArg::Buffer(trail_buf.view()),
            ],
        );
        drain(gpu);
        let (got_out, got_state, trail) = (
            gpu.read_f32(&out_buf),
            gpu.read_f32(&state_buf),
            gpu.read_f32(&trail_buf),
        );
        let (want_out, want_state) = delta_rule(
            gpu,
            plain,
            lanes,
            RECURRENT_SMEM,
            &gpu.upload(&f32_bytes(&qkv)),
            &gpu.upload(&f32_bytes(&decay)),
            &gpu.upload(&f32_bytes(&beta)),
            &gpu.upload(&f32_bytes(&seed)),
            &gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD])),
            n_tok,
        );
        assert_eq!(bits(&got_out), bits(&want_out), "{trail_entry}: output");
        assert_eq!(
            bits(&got_state),
            bits(&want_state),
            "{trail_entry}: final state"
        );
        for t in 0..n_tok - 1 {
            let (_, after) = delta_rule(
                gpu,
                plain,
                lanes,
                RECURRENT_SMEM,
                &gpu.upload(&f32_bytes(&qkv)),
                &gpu.upload(&f32_bytes(&decay)),
                &gpu.upload(&f32_bytes(&beta)),
                &gpu.upload(&f32_bytes(&seed)),
                &gpu.upload(&f32_bytes(&vec![0.0; n_tok * N_HEAD_V * HD])),
                t + 1,
            );
            assert_eq!(
                bits(&trail[t * state_len..(t + 1) * state_len]),
                bits(&after),
                "{trail_entry}: the record after row {t}"
            );
        }
    }
    assert!(
        ran > 0,
        "no recording recurrent variant was viable on this device"
    );
}

/// The convolution's recording twin: the window after every row but the last,
/// each the window a plain launch over the rows up to it leaves, and the plain
/// kernel's outputs and final window. Byte equality, for the reason the
/// recurrent case gives.
pub fn conv_trail_records_are_the_per_row_walk(gpu: &Gpu) {
    let n_tok = 5usize;
    let x = signal(n_tok * STRIDE, 81);
    let w = signal(STRIDE * CONV_KERNEL, 82);
    let seeded = signal(CONV_HIST * STRIDE, 83);
    let window = CONV_HIST * STRIDE;
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    let x_buf = gpu.upload(&f32_bytes(&x));
    let w_buf = gpu.upload(&f32_bytes(&w));
    let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n_tok * STRIDE]));
    let state_buf = gpu.upload(&f32_bytes(&seeded));
    let trail_buf = gpu.upload(&f32_bytes(&vec![0.0; (n_tok - 1) * window]));
    let args = ConvArgs {
        channels: STRIDE as u32,
        conv_kernel: CONV_KERNEL as u32,
        n_tok: n_tok as u32,
    };
    launch(
        gpu,
        "kernel_q35_gdn_conv_trail",
        Extent3::linear((STRIDE as u32).div_ceil(CONV_BLOCK)),
        CONV_BLOCK,
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out_buf.view()),
            KernelArg::Buffer(state_buf.view()),
            KernelArg::Buffer(x_buf.view()),
            KernelArg::Buffer(w_buf.view()),
            KernelArg::Buffer(trail_buf.view()),
        ],
    );
    drain(gpu);
    let (got_out, got_state, trail) = (
        gpu.read_f32(&out_buf),
        gpu.read_f32(&state_buf),
        gpu.read_f32(&trail_buf),
    );
    let fresh = |n: usize| {
        conv(
            gpu,
            &gpu.upload(&f32_bytes(&x)),
            &w_buf,
            &gpu.upload(&f32_bytes(&seeded)),
            &gpu.upload(&f32_bytes(&vec![0.0; n_tok * STRIDE])),
            n,
            STRIDE,
        )
    };
    let (want_out, want_state) = fresh(n_tok);
    assert_eq!(bits(&got_out), bits(&want_out), "conv trail: output");
    assert_eq!(
        bits(&got_state),
        bits(&want_state),
        "conv trail: final window"
    );
    for t in 0..n_tok - 1 {
        let (_, after) = fresh(t + 1);
        assert_eq!(
            bits(&trail[t * window..(t + 1) * window]),
            bits(&after),
            "conv trail: the record after row {t}"
        );
    }
}
