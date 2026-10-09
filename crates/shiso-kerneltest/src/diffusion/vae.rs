//! The residual VAE decoder's entries: the channel RMS norm's inverse-norm
//! pass, the two convolutions that fuse their input transform into the
//! implicit-GEMM gather, and the dup-up shortcut.
//!
//! Every oracle here materializes what the kernel avoids materializing - the
//! normalized and activated field, the upsampled field, the repeated and
//! reshaped shortcut tensor - and runs a direct `f64` convolution over it,
//! so a gather that indexes the fused transform wrongly cannot agree with
//! its own mistake. Shapes leave ragged 64-wide tiles in both GEMM axes.
#![expect(
    clippy::cast_possible_truncation,
    reason = "fixture dimensions are small literals marshalled as the kernels' u32 argument fields"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the oracles are written in the convolution's own letters (c, h, w, k, x), which is what makes them checkable against it"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "fixture sizes converted to floats are small, exact in f32"
)]

use bytemuck::{Pod, Zeroable};
use shiso_core::{DeviceBuffer, Extent3, KernelArg, KernelRef, LaunchArgs};

use crate::harness::{Gpu, drain, f32_bytes, run};
use crate::oracle::signal;

/// Every entry point this file's cases launch.
pub const ENTRIES: [&str; 4] = [
    "kernel_dit_channel_rms_inv_f32",
    "kernel_dit_conv2d_rms_gemm_f32",
    "kernel_dit_conv2d_up2_gemm_f32",
    "kernel_dit_dup_up_add_f32",
];

const MODULE: &str = "diffusion";
/// Threads per elementwise threadgroup.
const BLOCK: u32 = 64;
/// Output tile edge of the simdgroup convolution.
const G64_TILE: u32 = 64;
/// Threads per simdgroup-tile threadgroup.
const G64_THREADS: u32 = 128;
/// The reference's normalize epsilon.
const EPS: f32 = 1e-12;

// Argument blocks: byte for byte `diffusion_common.h` and the driver's Pods.

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ChannelRmsArgs {
    channels: u32,
    spatial: u32,
    scale: f32,
    eps: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ConvVaeArgs {
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    out_h: u32,
    out_w: u32,
    k: u32,
    pad: u32,
    has_bias: u32,
    accumulate: u32,
    silu: u32,
    reserved: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DupUpArgs {
    out_c: u32,
    h: u32,
    w: u32,
    factor: u32,
    offset: u32,
    repeats: u32,
}

const _: () = assert!(size_of::<ChannelRmsArgs>() == 16);
const _: () = assert!(size_of::<ConvVaeArgs>() == 48);
const _: () = assert!(size_of::<DupUpArgs>() == 24);

fn launch(gpu: &Gpu, entry: &'static str, grid: Extent3, block: Extent3, args: &[KernelArg<'_>]) {
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs {
            grid,
            block,
            smem_bytes: 0,
            args,
        },
    );
}

fn upload(gpu: &Gpu, values: &[f32]) -> DeviceBuffer {
    gpu.upload(&f32_bytes(values))
}

fn silu(v: f64) -> f64 {
    v / (1.0 + (-v).exp())
}

/// Asserts agreement within `tol` relative to the larger of each element and
/// the set's RMS.
fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms =
        (want.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / want.len() as f64).sqrt() as f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = tol * w.abs().max(rms).max(f32::MIN_POSITIVE);
        assert!(
            (g - w).abs() <= bound,
            "{what}: element {i} is {g}, expected {w} (tolerance {bound})"
        );
    }
}

/// The reference norm: `x / max(||x[:, s]||, eps) * sqrt(C) * gamma[c]`,
/// then `SiLU` when asked, over a `(c, s)` field.
fn norm_host(x: &[f32], c: usize, gamma: &[f32], with_silu: bool) -> Vec<f64> {
    let s = x.len() / c;
    let mut out = vec![0f64; x.len()];
    for p in 0..s {
        let norm = (0..c)
            .map(|ch| f64::from(x[ch * s + p]).powi(2))
            .sum::<f64>()
            .sqrt()
            .max(f64::from(EPS));
        for ch in 0..c {
            let v = f64::from(x[ch * s + p]) / norm * (c as f64).sqrt() * f64::from(gamma[ch]);
            out[ch * s + p] = if with_silu { silu(v) } else { v };
        }
    }
    out
}

/// Direct same-padded stride-1 convolution of a materialized `(in_c, h, w)`
/// field, plus bias and an optional prior value of the output.
struct DirectConv<'a> {
    field: &'a [f64],
    in_c: usize,
    h: usize,
    w: usize,
    weight: &'a [f32],
    bias: &'a [f32],
    out_c: usize,
    k: usize,
    prior: Option<&'a [f32]>,
}

fn conv_host(c: &DirectConv<'_>) -> Vec<f32> {
    let pad = c.k / 2;
    let mut out = vec![0f32; c.out_c * c.h * c.w];
    for o in 0..c.out_c {
        for y in 0..c.h {
            for x in 0..c.w {
                let mut acc = f64::from(c.bias[o]);
                for i in 0..c.in_c {
                    for ky in 0..c.k {
                        for kx in 0..c.k {
                            let (Some(iy), Some(ix)) =
                                ((y + ky).checked_sub(pad), (x + kx).checked_sub(pad))
                            else {
                                continue;
                            };
                            if iy >= c.h || ix >= c.w {
                                continue;
                            }
                            let v = c.field[(i * c.h + iy) * c.w + ix];
                            acc +=
                                v * f64::from(c.weight[((o * c.in_c + i) * c.k + ky) * c.k + kx]);
                        }
                    }
                }
                let idx = (o * c.h + y) * c.w + x;
                out[idx] = (acc + c.prior.map_or(0.0, |p| f64::from(p[idx]))) as f32;
            }
        }
    }
    out
}

fn conv_args(
    in_c: usize,
    (h, w): (usize, usize),
    out_c: usize,
    k: usize,
    accumulate: bool,
    silu: bool,
    up: bool,
) -> ConvVaeArgs {
    let f = if up { 2 } else { 1 };
    ConvVaeArgs {
        in_c: in_c as u32,
        in_h: h as u32,
        in_w: w as u32,
        out_c: out_c as u32,
        out_h: (f * h) as u32,
        out_w: (f * w) as u32,
        k: k as u32,
        pad: (k / 2) as u32,
        has_bias: 1,
        accumulate: u32::from(accumulate),
        silu: u32::from(silu),
        reserved: 0,
    }
}

fn conv_grid(out_c: usize, plane: usize) -> Extent3 {
    Extent3::new(
        (out_c as u32).div_ceil(G64_TILE),
        (plane as u32).div_ceil(G64_TILE),
        1,
    )
}

/// `kernel_dit_channel_rms_inv_f32` writes `sqrt(C) / max(||x[:, s]||, eps)`
/// per pixel, including the floor on an all-zero pixel.
///
/// # Panics
///
/// When the device disagrees with the oracle.
pub fn channel_rms_inv_is_the_scaled_inverse_pixel_norm(gpu: &Gpu) {
    let (c, s) = (37usize, 131usize);
    let mut x = signal(c * s, 11);
    for ch in 0..c {
        x[ch * s + 5] = 0.0;
    }
    let d_x = upload(gpu, &x);
    let d_inv = upload(gpu, &vec![0.0; s]);
    let scale = (c as f32).sqrt();
    let args = ChannelRmsArgs {
        channels: c as u32,
        spatial: s as u32,
        scale,
        eps: EPS,
    };
    launch(
        gpu,
        "kernel_dit_channel_rms_inv_f32",
        Extent3::linear((s as u32).div_ceil(BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_inv.view()),
            KernelArg::Buffer(d_x.view()),
        ],
    );
    drain(gpu);
    let got = gpu.read_f32(&d_inv);
    let want: Vec<f32> = (0..s)
        .map(|p| {
            let n = (0..c)
                .map(|ch| f64::from(x[ch * s + p]).powi(2))
                .sum::<f64>()
                .sqrt();
            (f64::from(scale) / n.max(f64::from(EPS))) as f32
        })
        .collect();
    assert!(
        got[5].is_finite() && got[5] > 1e11,
        "zero pixel takes the floor: {}",
        got[5]
    );
    close(&got, &want, 1e-5, "channel_rms_inv");
}

/// One norm-then-convolve case: `x` is `(in_c, h, w)`, `prior` the
/// output's starting value when the convolution accumulates.
struct RmsConv<'a> {
    x: &'a [f32],
    in_c: usize,
    h: usize,
    w: usize,
    gamma: &'a [f32],
    weight: &'a [f32],
    bias: &'a [f32],
    out_c: usize,
    k: usize,
    with_silu: bool,
    prior: Option<&'a [f32]>,
}

/// Runs the norm pass then the fused convolution, as the driver does.
fn rms_conv_device(gpu: &Gpu, case: &RmsConv<'_>) -> Vec<f32> {
    let RmsConv {
        x,
        in_c,
        h,
        w,
        gamma,
        weight,
        bias,
        out_c,
        k,
        with_silu,
        prior,
    } = *case;
    let s = h * w;
    let d_x = upload(gpu, x);
    let d_inv = upload(gpu, &vec![0.0; s]);
    let rms = ChannelRmsArgs {
        channels: in_c as u32,
        spatial: s as u32,
        scale: (in_c as f32).sqrt(),
        eps: EPS,
    };
    launch(
        gpu,
        "kernel_dit_channel_rms_inv_f32",
        Extent3::linear((s as u32).div_ceil(BLOCK)),
        Extent3::linear(BLOCK),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&rms)),
            KernelArg::Buffer(d_inv.view()),
            KernelArg::Buffer(d_x.view()),
        ],
    );
    let d_out = upload(gpu, prior.unwrap_or(&vec![0.0; out_c * s]));
    let (d_w, d_b, d_g) = (upload(gpu, weight), upload(gpu, bias), upload(gpu, gamma));
    let args = conv_args(in_c, (h, w), out_c, k, prior.is_some(), with_silu, false);
    launch(
        gpu,
        "kernel_dit_conv2d_rms_gemm_f32",
        conv_grid(out_c, s),
        Extent3::linear(G64_THREADS),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_out.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_b.view()),
            KernelArg::Buffer(d_inv.view()),
            KernelArg::Buffer(d_g.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&d_out)
}

/// `kernel_dit_conv2d_rms_gemm_f32` is norm -> silu -> zero-padded conv of
/// the materialized field, written or accumulated onto the output; and the
/// norm without `SiLU` for a 1x1 kernel (the attention's qkv projection).
///
/// # Panics
///
/// When the device disagrees with the oracle.
pub fn conv2d_rms_is_the_conv_of_the_normalized_activated_field(gpu: &Gpu) {
    let (in_c, h, w, out_c, k) = (16usize, 5usize, 13usize, 70usize, 3usize);
    // A per-pixel scale spread so the norm is not the same everywhere.
    let x: Vec<f32> = signal(in_c * h * w, 3)
        .iter()
        .enumerate()
        .map(|(i, v)| v * (1.0 + (i % (h * w)) as f32 * 0.1))
        .collect();
    let gamma: Vec<f32> = signal(in_c, 4).iter().map(|v| 1.0 + 0.5 * v).collect();
    let weight: Vec<f32> = signal(out_c * in_c * k * k, 5)
        .iter()
        .map(|v| v * 0.2)
        .collect();
    let bias = signal(out_c, 6);
    let prior = signal(out_c * h * w, 7);
    for (with_silu, acc) in [(true, false), (true, true), (false, false)] {
        let field = norm_host(&x, in_c, &gamma, with_silu);
        let want = conv_host(&DirectConv {
            field: &field,
            in_c,
            h,
            w,
            weight: &weight,
            bias: &bias,
            out_c,
            k,
            prior: acc.then_some(prior.as_slice()),
        });
        let got = rms_conv_device(
            gpu,
            &RmsConv {
                x: &x,
                in_c,
                h,
                w,
                gamma: &gamma,
                weight: &weight,
                bias: &bias,
                out_c,
                k,
                with_silu,
                prior: acc.then_some(prior.as_slice()),
            },
        );
        close(
            &got,
            &want,
            1e-4,
            &format!("conv2d_rms silu={with_silu} accumulate={acc}"),
        );
    }
    // 1x1, no activation.
    let (in_c, k) = (32usize, 1usize);
    let x = signal(in_c * h * w, 8);
    let gamma = signal(in_c, 9);
    let weight = signal(out_c * in_c, 10);
    let field = norm_host(&x, in_c, &gamma, false);
    let want = conv_host(&DirectConv {
        field: &field,
        in_c,
        h,
        w,
        weight: &weight,
        bias: &bias,
        out_c,
        k,
        prior: None,
    });
    let got = rms_conv_device(
        gpu,
        &RmsConv {
            x: &x,
            in_c,
            h,
            w,
            gamma: &gamma,
            weight: &weight,
            bias: &bias,
            out_c,
            k,
            with_silu: false,
            prior: None,
        },
    );
    close(&got, &want, 1e-4, "conv2d_rms 1x1");
}

/// `kernel_dit_conv2d_up2_gemm_f32` is the zero-padded conv of the
/// materialized nearest-2x field.
///
/// # Panics
///
/// When the device disagrees with the oracle.
pub fn conv2d_up2_is_the_conv_of_the_upsampled_field(gpu: &Gpu) {
    let (in_c, h, w, out_c, k) = (16usize, 3usize, 7usize, 67usize, 3usize);
    let x = signal(in_c * h * w, 21);
    let weight: Vec<f32> = signal(out_c * in_c * k * k, 22)
        .iter()
        .map(|v| v * 0.2)
        .collect();
    let bias = signal(out_c, 23);
    let (uh, uw) = (2 * h, 2 * w);
    let mut field = vec![0f64; in_c * uh * uw];
    for c in 0..in_c {
        for y in 0..uh {
            for xx in 0..uw {
                field[(c * uh + y) * uw + xx] = f64::from(x[(c * h + y / 2) * w + xx / 2]);
            }
        }
    }
    let want = conv_host(&DirectConv {
        field: &field,
        in_c,
        h: uh,
        w: uw,
        weight: &weight,
        bias: &bias,
        out_c,
        k,
        prior: None,
    });
    let d_x = upload(gpu, &x);
    let d_out = upload(gpu, &vec![0.0; out_c * uh * uw]);
    let (d_w, d_b) = (upload(gpu, &weight), upload(gpu, &bias));
    let args = conv_args(in_c, (h, w), out_c, k, false, false, true);
    launch(
        gpu,
        "kernel_dit_conv2d_up2_gemm_f32",
        conv_grid(out_c, uh * uw),
        Extent3::linear(G64_THREADS),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(d_out.view()),
            KernelArg::Buffer(d_x.view()),
            KernelArg::Buffer(d_w.view()),
            KernelArg::Buffer(d_b.view()),
        ],
    );
    drain(gpu);
    close(&gpu.read_f32(&d_out), &want, 1e-4, "conv2d_up2");
}

/// The reference shortcut, step by step: repeat every input channel
/// `repeats` times, view the channels as `(out_c, ft, 2, 2)`, move those
/// three axes into time, height and width, keep the last time slot.
fn dup_up_host(
    x: &[f32],
    in_c: usize,
    out_c: usize,
    (h, w): (usize, usize),
    ft: usize,
) -> Vec<f32> {
    let repeats = out_c * ft * 4 / in_c;
    let repeated: Vec<f32> = (0..in_c * repeats)
        .flat_map(|j| x[(j / repeats) * h * w..(j / repeats + 1) * h * w].to_vec())
        .collect();
    // video[c][t][y][x] over (out_c, ft, 2h, 2w)
    let mut video = vec![0f32; out_c * ft * 4 * h * w];
    for c in 0..out_c {
        for t in 0..ft {
            for sy in 0..2 {
                for sx in 0..2 {
                    let j = ((c * ft + t) * 2 + sy) * 2 + sx;
                    for yy in 0..h {
                        for xx in 0..w {
                            let v = repeated[(j * h + yy) * w + xx];
                            video[((c * ft + t) * 2 * h + 2 * yy + sy) * 2 * w + 2 * xx + sx] = v;
                        }
                    }
                }
            }
        }
    }
    let plane = 4 * h * w;
    (0..out_c)
        .flat_map(|c| video[(c * ft + ft - 1) * plane..(c * ft + ft) * plane].to_vec())
        .collect()
}

/// `kernel_dit_dup_up_add_f32` adds the reference's reshaped shortcut, with
/// and without a temporal factor, widening and narrowing the channels.
///
/// # Panics
///
/// When the device disagrees with the oracle.
pub fn dup_up_add_matches_the_reshaped_shortcut(gpu: &Gpu) {
    let (h, w) = (3usize, 5usize);
    for (in_c, out_c, ft) in [(8usize, 8usize, 2usize), (8, 4, 2), (6, 3, 1), (4, 8, 1)] {
        let x = signal(in_c * h * w, 31);
        let prior = signal(out_c * 4 * h * w, 32);
        let shortcut = dup_up_host(&x, in_c, out_c, (h, w), ft);
        let want: Vec<f32> = prior.iter().zip(&shortcut).map(|(a, b)| a + b).collect();
        let factor = ft * 4;
        let args = DupUpArgs {
            out_c: out_c as u32,
            h: h as u32,
            w: w as u32,
            factor: factor as u32,
            offset: ((ft - 1) * 4) as u32,
            repeats: (out_c * factor / in_c) as u32,
        };
        let d_x = upload(gpu, &x);
        let d_out = upload(gpu, &prior);
        launch(
            gpu,
            "kernel_dit_dup_up_add_f32",
            Extent3::new(((4 * h * w) as u32).div_ceil(BLOCK), out_c as u32, 1),
            Extent3::linear(BLOCK),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(d_out.view()),
                KernelArg::Buffer(d_x.view()),
            ],
        );
        drain(gpu);
        close(
            &gpu.read_f32(&d_out),
            &want,
            1e-6,
            &format!("dup_up {in_c}->{out_c} ft {ft}"),
        );
    }
}
