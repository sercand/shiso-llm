//! The sparse family's ple kernels: cases graded against a host restatement of the CUDA bodies.
//!
//! Two entries of the n-gram embedding block, at the shipped geometry: the
//! gate (hidden 2,560, four streams) and the dilated convolution (10,240
//! channels, kernel 4, dilation 3, so nine history columns).
//!
//! The convolution is launched the two ways the engine launches it: in place
//! (`out` and `x` the same buffer) without a checkpoint slab, which arrives as
//! the host's null - a zero 64-bit scalar - beside a zero capacity; and into a
//! separate output while recording rollback windows.
#![expect(
    clippy::cast_possible_truncation,
    clippy::missing_panics_doc,
    reason = "every case asserts by panicking, which is the whole of what a case does; \
              kernel arguments are u32 while the fixtures index in usize; every dimension \
              here is a small literal"
)]

use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// The module the ple entries live in, mirroring `kernels::MOD_PLE`.
const MODULE: &str = "q4e_ple";
/// Threads per block of both entries, mirroring `kernels::BLOCK_LINEAR`.
const BLOCK: u32 = 256;
/// The shipped hidden width.
const N_EMBD: u32 = 2_560;
/// The shipped residual stream count.
const STREAMS: u32 = 4;
/// The convolution's channels: every stream's hidden width.
const CHANNELS: u32 = N_EMBD * STREAMS;
/// The shipped convolution kernel width.
const KERNEL: u32 = 4;
/// The shipped n-gram size, which is the convolution's dilation.
const DILATION: u32 = 3;
/// History columns the convolution carries.
const HIST: u32 = (KERNEL - 1) * DILATION;
/// A value no kernel writes, for slots that must stay untouched.
const SENTINEL: f32 = -12_345.0;

/// Asserts two `f32` slabs agree, naming the first slot that does not.
fn agree(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol * w.abs().max(1.0),
            "{what}: slot {i} is {g}, expected {w}"
        );
    }
}

/// The gate multiplies the token's value by `sigmoid(sgn(s) * sqrt(|s|))` of
/// the stream's scaled key-query product, for products of both signs.
pub fn the_gated_value_takes_a_signed_square_root_before_the_sigmoid(gpu: &Gpu) {
    let n_tok = 3u32;
    let width = (n_tok * STREAMS * N_EMBD) as usize;
    let key = signal(width, 43);
    let query = signal(width, 47);
    let value = signal((n_tok * N_EMBD) as usize, 53);
    let gv = gpu.upload(&f32_bytes(&vec![SENTINEL; width]));
    let key_dev = gpu.upload(&f32_bytes(&key));
    let query_dev = gpu.upload(&f32_bytes(&query));
    let value_dev = gpu.upload(&f32_bytes(&value));
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_ple_gated_value_kernel"),
        &LaunchArgs {
            grid: Extent3::new(n_tok, STREAMS, 1),
            block: Extent3::new(BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(gv.view()),
                KernelArg::Buffer(key_dev.view()),
                KernelArg::Buffer(query_dev.view()),
                KernelArg::Buffer(value_dev.view()),
                KernelArg::U32(N_EMBD),
                KernelArg::U32(STREAMS),
            ],
        },
    );
    drain(gpu);
    let e = N_EMBD as usize;
    let mut want = vec![0.0f32; width];
    let (mut pos, mut neg) = (0, 0);
    for t in 0..n_tok as usize {
        for h in 0..STREAMS as usize {
            let base = (t * STREAMS as usize + h) * e;
            let dot: f64 = (0..e)
                .map(|i| f64::from(key[base + i]) * f64::from(query[base + i]))
                .sum::<f64>()
                / f64::from(N_EMBD).sqrt();
            if dot > 0.0 {
                pos += 1;
            } else {
                neg += 1;
            }
            let mag = dot.abs().max(1e-6).sqrt();
            let g = 1.0 / (1.0 + (-mag.copysign(dot)).exp());
            for i in 0..e {
                want[base + i] = (f64::from(value[t * e + i]) * g) as f32;
            }
        }
    }
    assert!(
        pos > 0 && neg > 0,
        "the fixture must gate with products of both signs ({pos} positive, {neg} negative)"
    );
    agree(&gpu.read_f32(&gv), &want, 1e-4, "gated value");
}

/// The convolution's host restatement: output rows, the final window, and
/// the window after each token that a checkpoint slab of `cap` slots records.
struct ConvWant {
    out: Vec<f32>,
    state: Vec<f32>,
    ckpt: Vec<f32>,
}

/// `q4e_ple_conv_kernel`'s body per channel, in its own order.
fn conv(state: &[f32], x: &[f32], w: &[f32], n_tok: u32, cap: u32) -> ConvWant {
    let (ch, hist, k) = (CHANNELS as usize, HIST as usize, KERNEL as usize);
    let mut want = ConvWant {
        out: vec![0.0; n_tok as usize * ch],
        state: vec![0.0; hist * ch],
        ckpt: vec![SENTINEL; cap as usize * hist * ch],
    };
    for c in 0..ch {
        let mut win: Vec<f32> = (0..hist).map(|i| state[i * ch + c]).collect();
        let wc = &w[c * k..(c + 1) * k];
        for t in 0..n_tok as usize {
            let cur = x[t * ch + c];
            let mut acc = wc[k - 1] * cur;
            for (kk, &wk) in wc[..k - 1].iter().enumerate() {
                let back = (k - 1 - kk) * DILATION as usize;
                acc += wk * win[hist - back];
            }
            want.out[t * ch + c] = acc / (1.0 + (-acc).exp());
            win.remove(0);
            win.push(cur);
            if t < cap as usize && t + 1 < n_tok as usize {
                for (i, &v) in win.iter().enumerate() {
                    want.ckpt[(t * hist + i) * ch + c] = v;
                }
            }
        }
        for (i, &v) in win.iter().enumerate() {
            want.state[i * ch + c] = v;
        }
    }
    want
}

/// One convolution launch at the engine's geometry.
fn launch_conv(gpu: &Gpu, bufs: [&DeviceBuffer; 4], n_tok: u32, ckpt: KernelArg<'_>, cap: u32) {
    let [out, state, x, w] = bufs;
    run(
        gpu,
        KernelRef::new(MODULE, "q4e_ple_conv_kernel"),
        &LaunchArgs {
            grid: Extent3::new(CHANNELS.div_ceil(BLOCK), 1, 1),
            block: Extent3::new(BLOCK, 1, 1),
            smem_bytes: 0,
            args: &[
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(state.view()),
                KernelArg::Buffer(x.view()),
                KernelArg::Buffer(w.view()),
                KernelArg::U32(CHANNELS),
                KernelArg::U32(KERNEL),
                KernelArg::U32(DILATION),
                KernelArg::U32(n_tok),
                ckpt,
                KernelArg::U32(cap),
            ],
        },
    );
    drain(gpu);
}

/// In place and without a checkpoint slab, as a non-speculative chunk runs it:
/// every tap reads the window the right distance back, including across the
/// carried history, and the window left behind is the chunk's last nine
/// columns.
pub fn the_convolution_runs_in_place_across_its_carried_history(gpu: &Gpu) {
    let n_tok = 7u32;
    let state = signal((HIST * CHANNELS) as usize, 59);
    let x = signal((n_tok * CHANNELS) as usize, 61);
    let w = signal((CHANNELS * KERNEL) as usize, 67);
    let want = conv(&state, &x, &w, n_tok, 0);
    let state_dev = gpu.upload(&f32_bytes(&state));
    let xo = gpu.upload(&f32_bytes(&x));
    let w_dev = gpu.upload(&f32_bytes(&w));
    // The host's null checkpoint pointer, exactly as the engine binds it.
    launch_conv(
        gpu,
        [&xo, &state_dev, &xo, &w_dev],
        n_tok,
        KernelArg::U64(0),
        0,
    );
    agree(&gpu.read_f32(&xo), &want.out, 1e-5, "in-place conv output");
    agree(&gpu.read_f32(&state_dev), &want.state, 0.0, "conv window");
}

/// Into a separate output and recording rollback windows: slot `t` holds the
/// window after token `t` for every token but the last, up to the capacity.
pub fn the_convolution_records_a_window_per_token_up_to_its_capacity(gpu: &Gpu) {
    let n_tok = 5u32;
    let cap = 3u32;
    let state = signal((HIST * CHANNELS) as usize, 71);
    let x = signal((n_tok * CHANNELS) as usize, 73);
    let w = signal((CHANNELS * KERNEL) as usize, 79);
    let want = conv(&state, &x, &w, n_tok, cap);
    let out = gpu.upload(&f32_bytes(&vec![SENTINEL; (n_tok * CHANNELS) as usize]));
    let state_dev = gpu.upload(&f32_bytes(&state));
    let x_dev = gpu.upload(&f32_bytes(&x));
    let w_dev = gpu.upload(&f32_bytes(&w));
    let ckpt = gpu.upload(&f32_bytes(&vec![
        SENTINEL;
        (cap * HIST * CHANNELS) as usize
    ]));
    launch_conv(
        gpu,
        [&out, &state_dev, &x_dev, &w_dev],
        n_tok,
        KernelArg::Buffer(ckpt.view()),
        cap,
    );
    agree(&gpu.read_f32(&out), &want.out, 1e-5, "conv output");
    agree(&gpu.read_f32(&state_dev), &want.state, 0.0, "conv window");
    agree(&gpu.read_f32(&ckpt), &want.ckpt, 0.0, "conv checkpoints");
    agree(&gpu.read_f32(&x_dev), &x, 0.0, "conv input untouched");
}
