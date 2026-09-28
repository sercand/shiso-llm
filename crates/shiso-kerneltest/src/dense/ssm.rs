//! Cases for the state-space block's three kernels (`dense_ssm.metal`).
//!
//! Each is held to a host rule within a float tolerance. The convolution and
//! the scan are run as two chunks, so the session records they rewrite are
//! graded too: the second chunk has to start exactly where the first left off.

#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "fixture dimensions are small positive integers"
)]

use crate::harness::{Gpu, drain, f32_bytes, run};
use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg, KernelRef, LaunchArgs};

/// Every entry point this module's cases launch.
pub const ENTRIES: [&str; 3] = [
    "kernel_dense_ssm_conv_f32",
    "kernel_dense_ssm_scan_f32",
    "kernel_dense_ssm_gate_norm_f32",
];

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConvArgs {
    n_tok: u32,
    channels: u32,
    taps: u32,
    in_stride: u32,
    in_offset: u32,
    pad: [u32; 3],
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ScanArgs {
    n_tok: u32,
    heads: u32,
    head_dim: u32,
    state: u32,
    groups: u32,
    channels: u32,
    in_stride: u32,
    dt_offset: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct NormArgs {
    n_tok: u32,
    inner: u32,
    group_size: u32,
    in_stride: u32,
    eps: f32,
    pad: [u32; 3],
}

fn signal(len: usize, seed: u32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i as u32).wrapping_mul(2_654_435_761) ^ seed) % 1000) as f32 / 500.0 - 1.0)
        .collect()
}

fn close(have: f32, want: f32, tol: f32) -> bool {
    (have - want).abs() <= tol * (1.0 + want.abs())
}

fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// The convolution over two chunks against the host rule, window included.
pub fn ssm_conv_carries_its_window_across_chunks(gpu: &Gpu) {
    let (channels, kernel, in_offset, in_stride) = (96usize, 4usize, 5usize, 110usize);
    let chunks = [3usize, 2];
    let total: usize = chunks.iter().sum();
    let proj = signal(total * in_stride, 21);
    let weight = signal(channels * kernel, 22);
    let bias = signal(channels, 23);
    let window0 = signal((kernel - 1) * channels, 24);
    // Host: the whole sequence at once, with the starting window prepended.
    let mut seq = window0.clone();
    for t in 0..total {
        seq.extend_from_slice(
            &proj[t * in_stride + in_offset..t * in_stride + in_offset + channels],
        );
    }
    let window_buf = gpu.upload(&f32_bytes(&window0));
    let w_buf = gpu.upload(&f32_bytes(&weight));
    let b_buf = gpu.upload(&f32_bytes(&bias));
    let mut row = 0usize;
    for &n in &chunks {
        let p_buf = gpu.upload(&f32_bytes(&proj[row * in_stride..(row + n) * in_stride]));
        let out_buf = gpu.upload(&f32_bytes(&vec![0.0; n * channels]));
        let args = ConvArgs {
            n_tok: n as u32,
            channels: channels as u32,
            taps: kernel as u32,
            in_stride: in_stride as u32,
            in_offset: in_offset as u32,
            pad: [0; 3],
        };
        run(
            gpu,
            KernelRef::new("dense", "kernel_dense_ssm_conv_f32"),
            &LaunchArgs {
                grid: Extent3::linear((channels as u32).div_ceil(256)),
                block: Extent3::linear(256),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(p_buf.view()),
                    KernelArg::Buffer(window_buf.view()),
                    KernelArg::Buffer(w_buf.view()),
                    KernelArg::Buffer(b_buf.view()),
                    KernelArg::Buffer(out_buf.view()),
                ],
            },
        );
        drain(gpu);
        let got = gpu.read_f32(&out_buf);
        for t in 0..n {
            for c in 0..channels {
                let mut acc = 0.0f32;
                for j in 0..kernel {
                    acc += weight[c * kernel + j] * seq[(row + t + j) * channels + c];
                }
                let want = silu(acc + bias[c]);
                let have = got[t * channels + c];
                assert!(
                    close(have, want, 1e-5),
                    "row {} channel {c}: {have} against {want}",
                    row + t
                );
            }
        }
        row += n;
    }
    let window = gpu.read_f32(&window_buf);
    for j in 0..kernel - 1 {
        for c in 0..channels {
            let want = seq[(total + j) * channels + c];
            assert_eq!(
                window[j * channels + c].to_bits(),
                want.to_bits(),
                "window {j} channel {c}"
            );
        }
    }
}

/// Shape of one scan fixture.
#[derive(Debug, Clone, Copy)]
struct ScanShape {
    heads: usize,
    head_dim: usize,
    state: usize,
    groups: usize,
}

impl ScanShape {
    fn inner(self) -> usize {
        self.heads * self.head_dim
    }
    fn channels(self) -> usize {
        self.inner() + 2 * self.groups * self.state
    }
}

/// The recurrence over two chunks against the host rule, state included.
#[expect(
    clippy::too_many_lines,
    reason = "one fixture: the host rule and two chunks"
)]
pub fn ssm_scan_carries_its_state_across_chunks(gpu: &Gpu) {
    let s = ScanShape {
        heads: 4,
        head_dim: 16,
        state: 64,
        groups: 2,
    };
    let chunks = [3usize, 2];
    let total: usize = chunks.iter().sum();
    let in_stride = s.inner() + s.channels() + s.heads;
    let dt_offset = s.inner() + s.channels();
    let xbc = signal(total * s.channels(), 31);
    let proj = signal(total * in_stride, 32);
    let dt_bias = signal(s.heads, 33);
    let a_log: Vec<f32> = signal(s.heads, 34).iter().map(|v| v * 0.5).collect();
    let d = signal(s.heads, 35);
    let state0: Vec<f32> = signal(s.heads * s.head_dim * s.state, 36)
        .iter()
        .map(|v| v * 0.1)
        .collect();

    // Host.
    let mut host = state0.clone();
    let mut want_y = vec![0.0f32; total * s.inner()];
    let per_group = s.heads / s.groups;
    for t in 0..total {
        let row = &xbc[t * s.channels()..(t + 1) * s.channels()];
        for h in 0..s.heads {
            let raw = proj[t * in_stride + dt_offset + h] + dt_bias[h];
            let dt = raw.max(0.0) + (1.0 + (-raw.abs()).exp()).ln();
            let da = (-a_log[h].exp() * dt).exp();
            let g = h / per_group;
            for p in 0..s.head_dim {
                let x = row[h * s.head_dim + p];
                let mut acc = 0.0f32;
                for n in 0..s.state {
                    let at = (h * s.head_dim + p) * s.state + n;
                    host[at] = da * host[at] + x * dt * row[s.inner() + g * s.state + n];
                    acc += host[at] * row[s.inner() + (s.groups + g) * s.state + n];
                }
                want_y[t * s.inner() + h * s.head_dim + p] = acc + x * d[h];
            }
        }
    }

    let st_buf = gpu.upload(&f32_bytes(&state0));
    let bias_buf = gpu.upload(&f32_bytes(&dt_bias));
    let a_buf = gpu.upload(&f32_bytes(&a_log));
    let d_buf = gpu.upload(&f32_bytes(&d));
    let mut row = 0usize;
    for &n in &chunks {
        let x_buf = gpu.upload(&f32_bytes(
            &xbc[row * s.channels()..(row + n) * s.channels()],
        ));
        let p_buf = gpu.upload(&f32_bytes(&proj[row * in_stride..(row + n) * in_stride]));
        let y_buf = gpu.upload(&f32_bytes(&vec![0.0; n * s.inner()]));
        let args = ScanArgs {
            n_tok: n as u32,
            heads: s.heads as u32,
            head_dim: s.head_dim as u32,
            state: s.state as u32,
            groups: s.groups as u32,
            channels: s.channels() as u32,
            in_stride: in_stride as u32,
            dt_offset: dt_offset as u32,
        };
        run(
            gpu,
            KernelRef::new("dense", "kernel_dense_ssm_scan_f32"),
            &LaunchArgs {
                grid: Extent3::new((s.head_dim as u32).div_ceil(8), s.heads as u32, 1),
                block: Extent3::linear(256),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(x_buf.view()),
                    KernelArg::Buffer(p_buf.view()),
                    KernelArg::Buffer(bias_buf.view()),
                    KernelArg::Buffer(a_buf.view()),
                    KernelArg::Buffer(d_buf.view()),
                    KernelArg::Buffer(st_buf.view()),
                    KernelArg::Buffer(y_buf.view()),
                ],
            },
        );
        drain(gpu);
        let got = gpu.read_f32(&y_buf);
        for (i, have) in got.iter().enumerate() {
            let want = want_y[row * s.inner() + i];
            assert!(
                close(*have, want, 1e-4),
                "row {} value {}: {have} against {want}",
                row + i / s.inner(),
                i % s.inner()
            );
        }
        row += n;
    }
    let state = gpu.read_f32(&st_buf);
    for (i, (have, want)) in state.iter().zip(&host).enumerate() {
        assert!(
            close(*have, *want, 1e-4),
            "state {i}: {have} against {want}"
        );
    }
}

/// `w * rms_norm(silu(z) * y)` per group, in place.
pub fn ssm_gate_norm_matches_the_host_rule(gpu: &Gpu) {
    let (n_tok, inner, group, extra) = (3usize, 256usize, 64usize, 40usize);
    let in_stride = inner + extra;
    let eps = 1e-5f32;
    let y = signal(n_tok * inner, 41);
    let proj = signal(n_tok * in_stride, 42);
    let w = signal(inner, 43);
    let y_buf = gpu.upload(&f32_bytes(&y));
    let p_buf = gpu.upload(&f32_bytes(&proj));
    let w_buf = gpu.upload(&f32_bytes(&w));
    let args = NormArgs {
        n_tok: n_tok as u32,
        inner: inner as u32,
        group_size: group as u32,
        in_stride: in_stride as u32,
        eps,
        pad: [0; 3],
    };
    run(
        gpu,
        KernelRef::new("dense", "kernel_dense_ssm_gate_norm_f32"),
        &LaunchArgs {
            grid: Extent3::new((inner / group) as u32, n_tok as u32, 1),
            block: Extent3::linear(256),
            smem_bytes: ((33 + group) * 4) as u32,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(y_buf.view()),
                KernelArg::Buffer(p_buf.view()),
                KernelArg::Buffer(w_buf.view()),
            ],
        },
    );
    drain(gpu);
    let got = gpu.read_f32(&y_buf);
    for t in 0..n_tok {
        for g in 0..inner / group {
            let v: Vec<f32> = (0..group)
                .map(|i| silu(proj[t * in_stride + g * group + i]) * y[t * inner + g * group + i])
                .collect();
            let ms = v.iter().map(|x| x * x).sum::<f32>() / group as f32;
            let inv = 1.0 / (ms + eps).sqrt();
            for (i, x) in v.iter().enumerate() {
                let want = w[g * group + i] * x * inv;
                let have = got[t * inner + g * group + i];
                assert!(
                    close(have, want, 1e-5),
                    "row {t} group {g} value {i}: {have} against {want}"
                );
            }
        }
    }
}
