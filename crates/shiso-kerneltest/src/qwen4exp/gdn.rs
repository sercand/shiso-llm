//! The sparse family's gdn kernels: cases graded against a host restatement of the CUDA bodies.
//!
//! The shape is the real model's recurrent block: 48 value heads reading 16
//! query/key heads (value head `h` reads key head `h % 16`), a 128-wide head,
//! a four-tap convolution over `2 * 16 * 128 + 48 * 128 = 10240` channels. Each
//! case launches with the grid and block `shiso_engine::sparse::kernels`
//! derives for that entry.
//!
//! The oracles are plain `f64` loops over the formula each CUDA body states,
//! not a transcription of its summation order: the device folds through a
//! 32-lane shuffle tree and evaluates `exp` in the library's fast form, so the
//! budgets below are relative and sized for `f32` rounding over the case's
//! depth, not for a second implementation of the same tree.
//!
//! The recurrence cases cover a decode row, a 37-row prefill and a verify pass
//! that writes the rollback slab, then replay the slab and check the rebuilt
//! state against the oracle's state after the kept rows. The engine passes an
//! absent slab as a zero 64-bit scalar with a zero count, and so do these cases,
//! so a kernel that trusts the pointer rather than the count writes through
//! eight bytes of argument storage and fails here.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "kernel arguments are u32 while the fixtures index in usize, and the \
              oracles widen small fixture values to f64 and back; every dimension \
              here is a literal far below 2^24"
)]

use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module every entry lives in, mirroring `kernels::MOD_GDN`.
const MODULE: &str = "q4e_gdn";
/// Value heads.
const V_HEADS: usize = 48;
/// Query/key heads.
const QK_HEADS: usize = 16;
/// Width of one head, mirroring `kernels::GDN_STATE_HEAD_DIM`.
const HEAD_DIM: usize = 128;
/// Convolution taps.
const CONV_KERNEL: usize = 4;
/// Width of the query (or key) block, `SparseShape::gdn_qk_dim`.
const QK_DIM: usize = QK_HEADS * HEAD_DIM;
/// Channels of the fused q|k|v row, `SparseShape::gdn_in_dim`.
const IN_DIM: usize = 2 * QK_DIM + V_HEADS * HEAD_DIM;
/// Lanes per state column, mirroring `kernels::GDN_SPLIT`.
const GDN_SPLIT: u32 = 4;
/// Threads of the elementwise and linear launches, `kernels::BLOCK_LINEAR`.
const BLOCK_LINEAR: u32 = 256;
/// Threads of the per-row launches, `kernels::BLOCK_NARROW`.
const BLOCK_NARROW: u32 = 128;
/// Rows of the prefill cases.
const PREFILL: usize = 37;

/// The engine's absent pointer (`Operand::Null` binds as `KernelArg::U64(0)`).
const NULL: KernelArg<'static> = KernelArg::U64(0);

/// Blocks of `per` covering `n`, as `kernels::blocks_for`.
fn blocks_for(n: usize, per: u32) -> u32 {
    n.div_ceil(per as usize) as u32
}

/// Asserts `got` matches `want` within `rel` of `max(|want|, floor)`, naming
/// the first index that does not.
fn agree(got: &[f32], want: &[f64], rel: f64, floor: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let err = (f64::from(g) - w).abs();
        assert!(
            err <= rel * w.abs().max(floor),
            "{what}: element {i} is {g}, expected {w} (error {err:e})"
        );
    }
}

/// Fresh device memory holding `len` NaNs, so an element a kernel forgot to
/// write cannot pass as a zero.
fn sentinel(gpu: &Gpu, len: usize) -> DeviceBuffer {
    gpu.upload(&f32_bytes(&vec![f32::NAN; len]))
}

// --- the convolution --------------------------------------------------------

/// The convolution's host oracle: outputs, the new window, and the
/// checkpoints for `t < min(cap, n_tok - 1)`.
fn conv_oracle(
    state: &[f32],
    x: &[f32],
    w: &[f32],
    n_tok: usize,
    cap: usize,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let hist = CONV_KERNEL - 1;
    let mut out = vec![0.0; n_tok * IN_DIM];
    let mut new_state = vec![0.0; hist * IN_DIM];
    let mut ckpt = vec![0.0; cap * hist * IN_DIM];
    for c in 0..IN_DIM {
        let mut win: Vec<f64> = (0..hist)
            .map(|i| f64::from(state[i * IN_DIM + c]))
            .collect();
        let wc = &w[c * CONV_KERNEL..(c + 1) * CONV_KERNEL];
        for t in 0..n_tok {
            let cur = f64::from(x[t * IN_DIM + c]);
            let mut acc: f64 = (0..hist).map(|k| f64::from(wc[k]) * win[k]).sum();
            acc += f64::from(wc[hist]) * cur;
            out[t * IN_DIM + c] = acc / (1.0 + (-acc).exp());
            win.remove(0);
            win.push(cur);
            if t < cap && t + 1 < n_tok {
                for i in 0..hist {
                    ckpt[(t * hist + i) * IN_DIM + c] = win[i];
                }
            }
        }
        for i in 0..hist {
            new_state[i * IN_DIM + c] = win[i];
        }
    }
    (out, new_state, ckpt)
}

/// Runs `q4e_gdn_conv_kernel` over `n_tok` rows with a rollback cap (0 is the
/// plain pass, whose slab is absent) and grades all three outputs.
fn conv_case(gpu: &Gpu, n_tok: usize, cap: usize) {
    let hist = CONV_KERNEL - 1;
    let state = signal(hist * IN_DIM, 3);
    let x = signal(n_tok * IN_DIM, 5);
    let w: Vec<f32> = signal(IN_DIM * CONV_KERNEL, 7)
        .iter()
        .map(|v| v * 0.5)
        .collect();
    let (want_out, want_state, want_ckpt) = conv_oracle(&state, &x, &w, n_tok, cap);

    let out = sentinel(gpu, n_tok * IN_DIM);
    let state_dev = gpu.upload(&f32_bytes(&state));
    let x_dev = gpu.upload(&f32_bytes(&x));
    let w_dev = gpu.upload(&f32_bytes(&w));
    // Sized to the cap exactly; the records the pass does not reach stay NaN
    // on the device and zero in the oracle, so they are graded separately.
    let ckpt_dev = (cap > 0).then(|| sentinel(gpu, cap * hist * IN_DIM));
    let ckpt_arg = ckpt_dev
        .as_ref()
        .map_or(NULL, |b| KernelArg::Buffer(b.view()));
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(state_dev.view()),
        KernelArg::Buffer(x_dev.view()),
        KernelArg::Buffer(w_dev.view()),
        KernelArg::U32(IN_DIM as u32),
        KernelArg::U32(CONV_KERNEL as u32),
        KernelArg::U32(n_tok as u32),
        ckpt_arg,
        KernelArg::U32(cap as u32),
    ];
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_gdn_conv_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(blocks_for(IN_DIM, BLOCK_LINEAR)),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: 0,
            args: &args,
        },
    );
    drain(gpu);
    let what = format!("conv n_tok {n_tok} cap {cap}");
    // The floor is the size of the four products the tap sum cancels, not of
    // the output: a near-zero pre-activation carries their rounding.
    agree(
        &gpu.read_f32(&out),
        &want_out,
        1e-5,
        0.1,
        &format!("{what}: out"),
    );
    let got_state: Vec<f64> = gpu
        .read_f32(&state_dev)
        .iter()
        .map(|&v| f64::from(v))
        .collect();
    assert_eq!(
        got_state, want_state,
        "{what}: the window is a copy of inputs"
    );
    if let Some(ckpt_dev) = ckpt_dev {
        let got = gpu.read_f32(&ckpt_dev);
        let written = cap.min(n_tok.saturating_sub(1)) * hist * IN_DIM;
        let got_written: Vec<f64> = got[..written].iter().map(|&v| f64::from(v)).collect();
        assert_eq!(got_written, want_ckpt[..written], "{what}: checkpoints");
        assert!(
            got[written..].iter().all(|v| v.is_nan()),
            "{what}: a record past the pass's last row was written"
        );
    }
}

/// The convolution over one decode row, no rollback slab.
pub fn conv_decode(gpu: &Gpu) {
    conv_case(gpu, 1, 0);
}

/// The convolution over a 37-row prefill, no rollback slab.
pub fn conv_prefill(gpu: &Gpu) {
    conv_case(gpu, PREFILL, 0);
}

/// A five-row verify pass records the window after each of its first four
/// rows, and nothing for its last.
pub fn conv_records_rollback_windows(gpu: &Gpu) {
    conv_case(gpu, 5, 4);
    // A cap wider than the pass: records stop at `n_tok - 1`.
    conv_case(gpu, 3, 4);
}

// --- the norm and the gates -------------------------------------------------

/// `q4e_gdn_l2norm_kernel` over the query and then the key block of a
/// 37-row pass, as the engine launches it (two launches, offsets 0 and
/// `QK_DIM`); the value block must come through untouched.
pub fn l2norm_prefill(gpu: &Gpu) {
    let qkv = signal(PREFILL * IN_DIM, 13);
    let dev = gpu.upload(&f32_bytes(&qkv));
    for offset in [0, QK_DIM] {
        let args = [
            KernelArg::Buffer(dev.view()),
            KernelArg::U32(HEAD_DIM as u32),
            KernelArg::U32(QK_HEADS as u32),
            KernelArg::U32(IN_DIM as u32),
            KernelArg::U32(offset as u32),
        ];
        run(
            gpu,
            KernelRef::new(MODULE, "q4e_gdn_l2norm_kernel"),
            &LaunchArgs {
                grid: Extent3::new(QK_HEADS as u32, PREFILL as u32, 1),
                block: Extent3::linear(BLOCK_NARROW),
                smem_bytes: 0,
                args: &args,
            },
        );
    }
    drain(gpu);
    let mut want: Vec<f64> = qkv.iter().map(|&v| f64::from(v)).collect();
    for t in 0..PREFILL {
        for h in 0..2 * QK_HEADS {
            let row = &mut want[t * IN_DIM + h * HEAD_DIM..t * IN_DIM + (h + 1) * HEAD_DIM];
            let sum: f64 = row.iter().map(|v| v * v).sum();
            let inv = 1.0 / (sum + 1e-12).sqrt();
            for v in row.iter_mut() {
                *v *= inv;
            }
        }
    }
    agree(&gpu.read_f32(&dev), &want, 1e-5, 1e-2, "l2norm");
}

/// `q4e_gdn_gates_kernel` over 48 heads and 37 rows. The alpha inputs span
/// the softplus guard at 20 and reach far enough below zero that the plain
/// `log(1 + e^x)` would lose the argument, which is what the port's log1p is
/// for.
pub fn gates_prefill(gpu: &Gpu) {
    let n = V_HEADS * PREFILL;
    // alpha + dt_bias in roughly [-30, 30].
    let alpha: Vec<f32> = signal(n, 17).iter().map(|v| v * 28.0).collect();
    let beta_proj: Vec<f32> = signal(n, 19).iter().map(|v| v * 6.0).collect();
    let dt_bias: Vec<f32> = signal(V_HEADS, 23).iter().map(|v| v * 2.0).collect();
    let a: Vec<f32> = signal(V_HEADS, 29)
        .iter()
        .map(|v| -(v.abs() * 8.0 + 0.1))
        .collect();
    let decay = sentinel(gpu, n);
    let beta = sentinel(gpu, n);
    let alpha_dev = gpu.upload(&f32_bytes(&alpha));
    let beta_proj_dev = gpu.upload(&f32_bytes(&beta_proj));
    let dt_dev = gpu.upload(&f32_bytes(&dt_bias));
    let a_dev = gpu.upload(&f32_bytes(&a));
    let args = [
        KernelArg::Buffer(decay.view()),
        KernelArg::Buffer(beta.view()),
        KernelArg::Buffer(alpha_dev.view()),
        KernelArg::Buffer(beta_proj_dev.view()),
        KernelArg::Buffer(dt_dev.view()),
        KernelArg::Buffer(a_dev.view()),
        KernelArg::U32(V_HEADS as u32),
        KernelArg::U32(PREFILL as u32),
    ];
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_gdn_gates_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(blocks_for(n, BLOCK_LINEAR)),
            block: Extent3::linear(BLOCK_LINEAR),
            smem_bytes: 0,
            args: &args,
        },
    );
    drain(gpu);
    let mut want_decay = vec![0.0; n];
    let mut want_beta = vec![0.0; n];
    for i in 0..n {
        let h = i % V_HEADS;
        // The argument is rounded to f32 exactly as the device forms it.
        let x = f64::from(alpha[i] + dt_bias[h]);
        let sp = if x > 20.0 { x } else { x.exp().ln_1p() };
        want_decay[i] = sp * f64::from(a[h]);
        want_beta[i] = 1.0 / (1.0 + (-f64::from(beta_proj[i])).exp());
    }
    // No floor on decay: a strongly negative input's decay is tiny, and it is
    // the relative error of exactly that value the log1p guard protects.
    agree(
        &gpu.read_f32(&decay),
        &want_decay,
        1e-5,
        0.0,
        "gates: decay",
    );
    agree(&gpu.read_f32(&beta), &want_beta, 1e-5, 1e-3, "gates: beta");
}

/// `q4e_gdn_out_gate_kernel` over 48 heads and 37 rows.
pub fn out_gate_prefill(gpu: &Gpu) {
    let groups = V_HEADS * PREFILL;
    let eps = 1e-6_f32;
    let attn = signal(groups * HEAD_DIM, 31);
    let z: Vec<f32> = signal(groups * HEAD_DIM, 37)
        .iter()
        .map(|v| v * 4.0)
        .collect();
    let w = signal(HEAD_DIM, 41);
    let out = sentinel(gpu, groups * HEAD_DIM);
    let attn_dev = gpu.upload(&f32_bytes(&attn));
    let z_dev = gpu.upload(&f32_bytes(&z));
    let w_dev = gpu.upload(&f32_bytes(&w));
    let args = [
        KernelArg::Buffer(out.view()),
        KernelArg::Buffer(attn_dev.view()),
        KernelArg::Buffer(z_dev.view()),
        KernelArg::Buffer(w_dev.view()),
        KernelArg::U32(HEAD_DIM as u32),
        KernelArg::U32(V_HEADS as u32),
        KernelArg::F32(eps),
    ];
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_gdn_out_gate_kernel"),
        &LaunchArgs {
            grid: Extent3::linear(groups as u32),
            block: Extent3::linear(BLOCK_NARROW),
            smem_bytes: 0,
            args: &args,
        },
    );
    drain(gpu);
    let mut want = vec![0.0; groups * HEAD_DIM];
    for g in 0..groups {
        let src = &attn[g * HEAD_DIM..(g + 1) * HEAD_DIM];
        let ss: f64 = src.iter().map(|&v| f64::from(v) * f64::from(v)).sum();
        let scale = 1.0 / (ss / HEAD_DIM as f64 + f64::from(eps)).sqrt();
        for i in 0..HEAD_DIM {
            let gate = 1.0 / (1.0 + (-f64::from(z[g * HEAD_DIM + i])).exp());
            want[g * HEAD_DIM + i] = f64::from(src[i]) * scale * f64::from(w[i]) * gate;
        }
    }
    agree(&gpu.read_f32(&out), &want, 1e-5, 1e-3, "out gate");
}

// --- the recurrence ---------------------------------------------------------

/// Dynamic threadgroup memory for the recurrence and the replay.
///
/// The engine requests `kernels::gdn_recurrent_smem(128)` = 66 048 bytes, the
/// CUDA body's shared state tile, and so does every case here. The Metal port
/// keeps the state in registers and declares no dynamic threadgroup binding,
/// which is what tells that backend the request describes nothing and must not
/// be applied - so the case asks exactly what the engine asks.
fn recurrent_smem(_gpu: &Gpu) -> u32 {
    (HEAD_DIM * (HEAD_DIM + 1) * 4) as u32
}

/// The recurrence's geometry, `kernels::gdn_recurrent(128, 48)`.
fn recurrent_launch<'a>(gpu: &Gpu, args: &'a [KernelArg<'a>]) -> LaunchArgs<'a> {
    LaunchArgs {
        grid: Extent3::linear(V_HEADS as u32),
        block: Extent3::linear(HEAD_DIM as u32 * GDN_SPLIT),
        smem_bytes: recurrent_smem(gpu),
        args,
    }
}

/// The recurrence's inputs: an L2-normalized q and k per head, as the pass
/// hands them over, and a decay/beta pair per (row, value head).
struct RecFixture {
    n_tok: usize,
    qkv: Vec<f32>,
    decay: Vec<f32>,
    beta: Vec<f32>,
    /// The state `[h][j][i]` = S[i][j] of head h, as it is stored.
    state: Vec<f32>,
}

impl RecFixture {
    fn new(n_tok: usize) -> Self {
        let mut qkv = signal(n_tok * IN_DIM, 43);
        for t in 0..n_tok {
            for h in 0..2 * QK_HEADS {
                let row = &mut qkv[t * IN_DIM + h * HEAD_DIM..t * IN_DIM + (h + 1) * HEAD_DIM];
                let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
                for v in row.iter_mut() {
                    *v /= norm;
                }
            }
        }
        let decay = signal(n_tok * V_HEADS, 47)
            .iter()
            .map(|v| -(v.abs() * 1.5 + 0.01))
            .collect();
        let beta = signal(n_tok * V_HEADS, 53)
            .iter()
            .map(|v| v.abs() * 0.9 + 0.05)
            .collect();
        let state = signal(V_HEADS * HEAD_DIM * HEAD_DIM, 59)
            .iter()
            .map(|v| v * 0.5)
            .collect();
        Self {
            n_tok,
            qkv,
            decay,
            beta,
            state,
        }
    }

    /// The first `rows` rows of the recurrence: every row's outputs and the
    /// state after the last one.
    fn oracle(&self, rows: usize) -> (Vec<f64>, Vec<f64>) {
        let dim = HEAD_DIM;
        let scale = 1.0 / (dim as f64).sqrt();
        let mut state: Vec<f64> = self.state.iter().map(|&v| f64::from(v)).collect();
        let mut out = vec![0.0; rows * V_HEADS * dim];
        for t in 0..rows {
            let base = &self.qkv[t * IN_DIM..(t + 1) * IN_DIM];
            for h in 0..V_HEADS {
                let hk = h % QK_HEADS;
                let qr = &base[hk * dim..(hk + 1) * dim];
                let kr = &base[QK_DIM + hk * dim..QK_DIM + (hk + 1) * dim];
                let vr = &base[2 * QK_DIM + h * dim..2 * QK_DIM + (h + 1) * dim];
                let kq: f64 = kr
                    .iter()
                    .zip(qr)
                    .map(|(&a, &b)| f64::from(a) * f64::from(b))
                    .sum();
                let dec = f64::from(self.decay[t * V_HEADS + h]).exp();
                let bt = f64::from(self.beta[t * V_HEADS + h]);
                for j in 0..dim {
                    let row = &mut state[(h * dim + j) * dim..(h * dim + j + 1) * dim];
                    let sk: f64 = row.iter().zip(kr).map(|(s, &kv)| s * f64::from(kv)).sum();
                    let sq: f64 = row.iter().zip(qr).map(|(s, &qv)| s * f64::from(qv)).sum();
                    let delta = (f64::from(vr[j]) - dec * sk) * bt;
                    out[(t * V_HEADS + h) * dim + j] = (dec * sq + delta * kq) * scale;
                    for (s, &kv) in row.iter_mut().zip(kr) {
                        *s = *s * dec + delta * f64::from(kv);
                    }
                }
            }
        }
        (out, state)
    }
}

/// Floats of one session's rollback slab: the start-of-pass state, then
/// `cap` recorded qkv rows, decays and betas (`paged::state_rollback_bytes`).
const fn slab_len(cap: usize) -> usize {
    V_HEADS * HEAD_DIM * HEAD_DIM + cap * (IN_DIM + 2 * V_HEADS)
}

/// Runs `q4e_gdn_recurrent_kernel` over the fixture with rollback cap `cap`
/// (0: absent slab) and grades the outputs and the final state. Returns the
/// device's final state and the slab's contents when there is one.
fn recurrent_case(gpu: &Gpu, fx: &RecFixture, cap: usize) -> (Vec<f32>, Option<Vec<f32>>) {
    let (want_out, want_state) = fx.oracle(fx.n_tok);
    let attn = sentinel(gpu, fx.n_tok * V_HEADS * HEAD_DIM);
    let state = gpu.upload(&f32_bytes(&fx.state));
    let qkv = gpu.upload(&f32_bytes(&fx.qkv));
    let decay = gpu.upload(&f32_bytes(&fx.decay));
    let beta = gpu.upload(&f32_bytes(&fx.beta));
    // Zero rather than NaN: the replay reads this slab's rows as a qkv block,
    // and the query section it never records must not be what poisons it.
    let slab = (cap > 0).then(|| gpu.upload(&f32_bytes(&vec![0.0; slab_len(cap)])));
    let slab_arg = slab.as_ref().map_or(NULL, |b| KernelArg::Buffer(b.view()));
    let args = [
        KernelArg::Buffer(attn.view()),
        KernelArg::Buffer(state.view()),
        KernelArg::Buffer(qkv.view()),
        KernelArg::Buffer(decay.view()),
        KernelArg::Buffer(beta.view()),
        KernelArg::U32(HEAD_DIM as u32),
        KernelArg::U32(QK_HEADS as u32),
        KernelArg::U32(V_HEADS as u32),
        KernelArg::U32(QK_DIM as u32),
        KernelArg::U32(2 * QK_DIM as u32),
        KernelArg::U32(IN_DIM as u32),
        KernelArg::U32(fx.n_tok as u32),
        slab_arg,
        KernelArg::U32(cap as u32),
    ];
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_gdn_recurrent_kernel"),
        &recurrent_launch(gpu, &args),
    );
    drain(gpu);
    let what = format!("recurrence n_tok {} cap {cap}", fx.n_tok);
    agree(
        &gpu.read_f32(&attn),
        &want_out,
        1e-4,
        1e-2,
        &format!("{what}: out"),
    );
    let got_state = gpu.read_f32(&state);
    agree(
        &got_state,
        &want_state,
        1e-4,
        1e-2,
        &format!("{what}: state"),
    );
    (got_state, slab.map(|s| gpu.read_f32(&s)))
}

/// The recurrence over one decode row, no rollback slab.
pub fn recurrent_decode(gpu: &Gpu) {
    recurrent_case(gpu, &RecFixture::new(1), 0);
}

/// The recurrence over a 37-row prefill, no rollback slab.
pub fn recurrent_prefill(gpu: &Gpu) {
    recurrent_case(gpu, &RecFixture::new(PREFILL), 0);
}

/// A five-row verify pass with a four-row cap records the start-of-pass state
/// and its first four rows' inputs; a replay of the first three rebuilds the
/// state the pass held after them.
///
/// # Panics
///
/// When the device disagrees with the oracle or with its own direct pass.
pub fn recurrent_rollback_and_replay(gpu: &Gpu) {
    let cap = 4;
    let keep = 3;
    let fx = RecFixture::new(5);
    let slab = recurrent_case(gpu, &fx, cap)
        .1
        .expect("a capped pass has a slab");

    // The record: the state as the pass found it, then the v rows of every
    // head and the k rows of every key head, then decay and beta, each
    // bit-for-bit a copy.
    let s = V_HEADS * HEAD_DIM * HEAD_DIM;
    assert!(slab[..s] == fx.state[..], "slab: start-of-pass state");
    let rows = &slab[s..];
    for t in 0..cap {
        let src = &fx.qkv[t * IN_DIM..(t + 1) * IN_DIM];
        let dst = &rows[t * IN_DIM..(t + 1) * IN_DIM];
        assert!(dst[QK_DIM..] == src[QK_DIM..], "slab: row {t} k|v");
        assert!(
            dst[..QK_DIM].iter().all(|&v| v == 0.0),
            "slab: row {t} q is not recorded"
        );
    }
    let rdec = &rows[cap * IN_DIM..cap * IN_DIM + cap * V_HEADS];
    let rbeta = &rows[cap * IN_DIM + cap * V_HEADS..];
    assert!(rdec == &fx.decay[..cap * V_HEADS], "slab: decay");
    assert!(rbeta == &fx.beta[..cap * V_HEADS], "slab: beta");

    // The replay writes the kept state into a buffer that held something
    // else, as a live session's slot does after the rejected rows.
    let (_, want_state) = fx.oracle(keep);
    let state = sentinel(gpu, s);
    let slab_dev = gpu.upload(&f32_bytes(&slab));
    let args = [
        KernelArg::Buffer(state.view()),
        KernelArg::Buffer(slab_dev.view()),
        KernelArg::U32(HEAD_DIM as u32),
        KernelArg::U32(QK_HEADS as u32),
        KernelArg::U32(V_HEADS as u32),
        KernelArg::U32(QK_DIM as u32),
        KernelArg::U32(2 * QK_DIM as u32),
        KernelArg::U32(IN_DIM as u32),
        KernelArg::U32(keep as u32),
        KernelArg::U32(cap as u32),
    ];
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_gdn_replay_kernel"),
        &recurrent_launch(gpu, &args),
    );
    drain(gpu);
    let replayed = gpu.read_f32(&state);
    agree(&replayed, &want_state, 1e-4, 1e-2, "replay: state");
    // And bit for bit what the device's own pass over those rows leaves: the
    // fixture's data is one generator stream per field, so a three-row
    // fixture is the five-row one's prefix.
    let (direct, _) = recurrent_case(gpu, &RecFixture::new(keep), 0);
    assert!(
        replayed == direct,
        "replay: differs from a direct pass over the kept rows"
    );
    assert!(
        gpu.read_f32(&slab_dev) == slab,
        "replay: the slab is read-only"
    );
}
