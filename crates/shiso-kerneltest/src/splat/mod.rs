//! Device cases for the `splat` module: the background remover's window
//! attention, deformable sampling, resampling, patch merge and activations.
//!
//! Each oracle is written the way the reference computes, not the way the
//! kernel indexes: the window attention's oracle pads the grid, rolls it,
//! partitions it into windows, attends inside each with the bias table and
//! the region mask, then reverses, unrolls and crops - the kernel does all of
//! that in one index - and the deformable oracle samples each tap with the
//! four-corner rule in `f64`.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    reason = "fixture dimensions are small literals marshalled as the kernels' u32 argument fields"
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
pub const ENTRIES: [&str; 7] = [
    "kernel_splat_relu_f32",
    "kernel_splat_sigmoid_f32",
    "kernel_splat_gelu_erf_f32",
    "kernel_splat_resize_bilinear_f32",
    "kernel_splat_merge2x2_f32",
    "kernel_splat_window_attn_f32",
    "kernel_splat_deform_im2col_f32",
];

const MODULE: &str = "splat";
/// Threads per threadgroup, every entry.
const BLOCK: u32 = 64;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct UnaryArgs {
    count: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ResizeArgs {
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct MergeArgs {
    h: u32,
    w: u32,
    channels: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct WindowArgs {
    h: u32,
    w: u32,
    hp: u32,
    wp: u32,
    window: u32,
    shift: u32,
    heads: u32,
    head_dim: u32,
    scale: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct DeformArgs {
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    k_h: u32,
    k_w: u32,
    stride: u32,
    pad: u32,
    row0: u32,
    rows: u32,
}

const _: () = assert!(size_of::<UnaryArgs>() == 4);
const _: () = assert!(size_of::<ResizeArgs>() == 20);
const _: () = assert!(size_of::<MergeArgs>() == 12);
const _: () = assert!(size_of::<WindowArgs>() == 36);
const _: () = assert!(size_of::<DeformArgs>() == 44);

fn launch(gpu: &Gpu, entry: &'static str, grid: Extent3, args: &[KernelArg<'_>]) {
    run(
        gpu,
        KernelRef::new(MODULE, entry),
        &LaunchArgs {
            grid,
            block: Extent3::linear(BLOCK),
            smem_bytes: 0,
            args,
        },
    );
}

fn upload(gpu: &Gpu, values: &[f32]) -> DeviceBuffer {
    gpu.upload(&f32_bytes(values))
}

/// Asserts `|got - want| <= tol * max(|want|, floor)` element by element.
fn close(got: &[f32], want: &[f64], tol: f64, floor: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = tol * w.abs().max(floor);
        assert!(
            (f64::from(g) - w).abs() <= bound,
            "{what}: element {i} is {g}, expected {w} (tolerance {bound})"
        );
    }
}

/// Runs one elementwise entry over `x`.
fn unary(gpu: &Gpu, entry: &'static str, x: &[f32]) -> Vec<f32> {
    let input = upload(gpu, x);
    let out = upload(gpu, &vec![0.0; x.len()]);
    let args = UnaryArgs {
        count: x.len() as u32,
    };
    launch(
        gpu,
        entry,
        Extent3::linear((x.len() as u32).div_ceil(BLOCK)),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(input.view()),
        ],
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// `erf` by its Maclaurin series in `f64`, exact to ~1e-12 for `|x| <= 3`.
fn erf_series(x: f64) -> f64 {
    let mut term = x;
    let mut sum = x;
    for n in 1..80 {
        term *= -x * x / f64::from(n);
        sum += term / f64::from(2 * n + 1);
    }
    sum * 2.0 / std::f64::consts::PI.sqrt()
}

/// `relu`, the logistic sigmoid and exact GELU against their formulas, over
/// negative, zero and positive inputs, including the sigmoid's saturated
/// tails where the naive form overflows.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn activations_match_their_formulas(gpu: &Gpu) {
    let mut x: Vec<f32> = (0..161).map(|i| (i as f32 - 80.0) / 20.0).collect();
    x.extend([-100.0, 100.0, 0.0]);
    let relu = unary(gpu, "kernel_splat_relu_f32", &x);
    let want: Vec<f64> = x.iter().map(|&v| f64::from(v).max(0.0)).collect();
    close(&relu, &want, 0.0, 0.0, "relu");

    let sigmoid = unary(gpu, "kernel_splat_sigmoid_f32", &x);
    let want: Vec<f64> = x.iter().map(|&v| 1.0 / (1.0 + (-f64::from(v)).exp())).collect();
    close(&sigmoid, &want, 1e-6, 1e-30, "sigmoid");

    let x: Vec<f32> = (0..161).map(|i| (i as f32 - 80.0) / 20.0).collect();
    let gelu = unary(gpu, "kernel_splat_gelu_erf_f32", &x);
    let want: Vec<f64> = x
        .iter()
        .map(|&v| {
            let v = f64::from(v);
            0.5 * v * (1.0 + erf_series(v / std::f64::consts::SQRT_2))
        })
        .collect();
    close(&gelu, &want, 5e-6, 1e-30, "gelu_erf");
    // The tanh form differs from the exact one by ~1e-3 near |x| = 2: a
    // kernel that used it would fail the bound above.
}

/// Aligned-corner bilinear resampling up, down and to a single pixel, against
/// the reference's own arithmetic in `f64`.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn resize_bilinear_aligns_corners(gpu: &Gpu) {
    let (channels, in_h, in_w) = (3usize, 5usize, 7usize);
    let field = signal(channels * in_h * in_w, 11);
    for (out_h, out_w) in [(9usize, 13usize), (3, 4), (1, 1), (5, 7)] {
        let input = upload(gpu, &field);
        let out = upload(gpu, &vec![0.0; channels * out_h * out_w]);
        let args = ResizeArgs {
            channels: channels as u32,
            in_h: in_h as u32,
            in_w: in_w as u32,
            out_h: out_h as u32,
            out_w: out_w as u32,
        };
        launch(
            gpu,
            "kernel_splat_resize_bilinear_f32",
            Extent3::new((out_w as u32).div_ceil(BLOCK), out_h as u32, channels as u32),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(input.view()),
            ],
        );
        drain(gpu);
        let got = gpu.read_f32(&out);
        let scale = |n_in: usize, n_out: usize| {
            if n_out > 1 {
                (n_in - 1) as f64 / (n_out - 1) as f64
            } else {
                0.0
            }
        };
        let (sy, sx) = (scale(in_h, out_h), scale(in_w, out_w));
        let mut want = vec![0.0f64; channels * out_h * out_w];
        for c in 0..channels {
            let at = |y: usize, x: usize| f64::from(field[(c * in_h + y) * in_w + x]);
            for y in 0..out_h {
                let fy = sy * y as f64;
                let y0 = (fy.floor() as usize).min(in_h - 1);
                let y1 = (y0 + 1).min(in_h - 1);
                let ly = fy - y0 as f64;
                for x in 0..out_w {
                    let fx = sx * x as f64;
                    let x0 = (fx.floor() as usize).min(in_w - 1);
                    let x1 = (x0 + 1).min(in_w - 1);
                    let lx = fx - x0 as f64;
                    want[(c * out_h + y) * out_w + x] = (1.0 - ly)
                        * ((1.0 - lx) * at(y0, x0) + lx * at(y0, x1))
                        + ly * ((1.0 - lx) * at(y1, x0) + lx * at(y1, x1));
                }
            }
        }
        close(&got, &want, 1e-5, 1e-3, "resize_bilinear");
        if (out_h, out_w) == (in_h, in_w) {
            close(&got, &want, 0.0, 0.0, "resize to the same size is a copy");
        }
    }
}

/// The 2x2 merge picks `(2i, 2j)`, `(2i + 1, 2j)`, `(2i, 2j + 1)`,
/// `(2i + 1, 2j + 1)` in that order: a row-major `(0,0),(0,1),(1,0),(1,1)`
/// order would swap the middle two blocks.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn merge2x2_takes_the_reference_order(gpu: &Gpu) {
    let (h, w, channels) = (4usize, 6usize, 3usize);
    let tokens: Vec<f32> = (0..h * w * channels).map(|v| v as f32).collect();
    let input = upload(gpu, &tokens);
    let out = upload(gpu, &vec![0.0; tokens.len()]);
    let args = MergeArgs {
        h: h as u32,
        w: w as u32,
        channels: channels as u32,
    };
    launch(
        gpu,
        "kernel_splat_merge2x2_f32",
        Extent3::new(
            ((4 * channels) as u32).div_ceil(BLOCK),
            ((h / 2) * (w / 2)) as u32,
            1,
        ),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(input.view()),
        ],
    );
    drain(gpu);
    let got = gpu.read_f32(&out);
    let mut want = Vec::new();
    for i in 0..h / 2 {
        for j in 0..w / 2 {
            for (dy, dx) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                for c in 0..channels {
                    want.push(f64::from(tokens[((2 * i + dy) * w + 2 * j + dx) * channels + c]));
                }
            }
        }
    }
    close(&got, &want, 0.0, 0.0, "merge2x2");
}

/// The window attention's shape.
#[derive(Debug, Clone, Copy)]
struct Window {
    h: usize,
    w: usize,
    side: usize,
    shift: usize,
    heads: usize,
    head_dim: usize,
}

/// The reference's steps literally: pad (padded keys and values are the
/// pad rows), roll by `-shift`, partition, attend with bias and mask, reverse,
/// roll back, crop.
#[expect(
    clippy::too_many_lines,
    reason = "the reference's seven steps, kept in one place so they read against it"
)]
fn window_oracle(
    s: Window,
    qkv: (&[f32], &[f32], &[f32]),
    pads: (&[f32], &[f32]),
    table: &[f32],
) -> Vec<f64> {
    let Window {
        h,
        w,
        side: ws,
        shift,
        heads,
        head_dim: hd,
    } = s;
    let width = heads * hd;
    let hp = h.div_ceil(ws) * ws;
    let wp = w.div_ceil(ws) * ws;
    // Step 1: the padded grid. A padded token's key and value are the pads;
    // its query is never read back.
    let grid = |src: &[f32], pad: &[f32]| -> Vec<Vec<f64>> {
        (0..hp * wp)
            .map(|t| {
                let (y, x) = (t / wp, t % wp);
                (0..width)
                    .map(|c| {
                        if y < h && x < w {
                            f64::from(src[(y * w + x) * width + c])
                        } else {
                            f64::from(pad[c])
                        }
                    })
                    .collect()
            })
            .collect()
    };
    let zeros = vec![0.0f32; width];
    let (gq, gk, gv) = (grid(qkv.0, &zeros), grid(qkv.1, pads.0), grid(qkv.2, pads.1));
    // Step 2: roll by -shift: rolled[y][x] = grid[(y + shift) % hp][(x + shift) % wp].
    let rolled = |g: &Vec<Vec<f64>>| -> Vec<Vec<f64>> {
        (0..hp * wp)
            .map(|t| {
                let (y, x) = (t / wp, t % wp);
                g[((y + shift) % hp) * wp + (x + shift) % wp].clone()
            })
            .collect()
    };
    let (rq, rk, rv) = (rolled(&gq), rolled(&gk), rolled(&gv));
    // The region labels of the rolled grid, as the reference slices them.
    let region = |v: usize, n: usize| {
        if v < n - ws {
            0
        } else if v < n - shift {
            1
        } else {
            2
        }
    };
    let span = 2 * ws - 1;
    let scale = 1.0 / (hd as f64).sqrt();
    let mut rolled_out = vec![vec![0.0f64; width]; hp * wp];
    // Steps 3-5: partition, attend, reverse.
    for wy in 0..hp / ws {
        for wx in 0..wp / ws {
            let member = |i: usize| (wy * ws + i / ws, wx * ws + i % ws);
            for head in 0..heads {
                for i in 0..ws * ws {
                    let (qy, qx) = member(i);
                    let scores: Vec<f64> = (0..ws * ws)
                        .map(|j| {
                            let (ky, kx) = member(j);
                            let dot: f64 = (0..hd)
                                .map(|d| {
                                    rq[qy * wp + qx][head * hd + d] * scale
                                        * rk[ky * wp + kx][head * hd + d]
                                })
                                .sum();
                            let rel = (i / ws + ws - 1 - j / ws) * span + (i % ws + ws - 1 - j % ws);
                            let mut score = dot + f64::from(table[rel * heads + head]);
                            if shift > 0
                                && (region(qy, hp), region(qx, wp)) != (region(ky, hp), region(kx, wp))
                            {
                                score -= 100.0;
                            }
                            score
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
                    let sum: f64 = exps.iter().sum();
                    for d in 0..hd {
                        rolled_out[qy * wp + qx][head * hd + d] = (0..ws * ws)
                            .map(|j| {
                                let (ky, kx) = member(j);
                                exps[j] * rv[ky * wp + kx][head * hd + d]
                            })
                            .sum::<f64>()
                            / sum;
                    }
                }
            }
        }
    }
    // Steps 6-7: roll back by +shift and crop.
    let mut out = vec![0.0f64; h * w * width];
    for y in 0..h {
        for x in 0..w {
            let ry = (y + hp - shift % hp) % hp;
            let rx = (x + wp - shift % wp) % wp;
            out[(y * w + x) * width..(y * w + x + 1) * width]
                .copy_from_slice(&rolled_out[ry * wp + rx]);
        }
    }
    out
}

fn window_case(gpu: &Gpu, s: Window, seed: u32) -> (Vec<f32>, Vec<f64>) {
    let width = s.heads * s.head_dim;
    let tokens = s.h * s.w;
    let q = signal(tokens * width, seed);
    let k = signal(tokens * width, seed + 1);
    let v = signal(tokens * width, seed + 2);
    let k_pad = signal(width, seed + 3);
    let v_pad = signal(width, seed + 4);
    let span = 2 * s.side - 1;
    let table = signal(span * span * s.heads, seed + 5);
    let hp = s.h.div_ceil(s.side) * s.side;
    let wp = s.w.div_ceil(s.side) * s.side;
    let bufs = [&q, &k, &v, &k_pad, &v_pad, &table].map(|b| upload(gpu, b));
    let out = upload(gpu, &vec![0.0; tokens * width]);
    let args = WindowArgs {
        h: s.h as u32,
        w: s.w as u32,
        hp: hp as u32,
        wp: wp as u32,
        window: s.side as u32,
        shift: s.shift as u32,
        heads: s.heads as u32,
        head_dim: s.head_dim as u32,
        scale: 1.0 / (s.head_dim as f32).sqrt(),
    };
    launch(
        gpu,
        "kernel_splat_window_attn_f32",
        Extent3::new(((hp / s.side) * (wp / s.side)) as u32, s.heads as u32, 1),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(bufs[0].view()),
            KernelArg::Buffer(bufs[1].view()),
            KernelArg::Buffer(bufs[2].view()),
            KernelArg::Buffer(bufs[3].view()),
            KernelArg::Buffer(bufs[4].view()),
            KernelArg::Buffer(bufs[5].view()),
        ],
    );
    drain(gpu);
    let want = window_oracle(s, (&q, &k, &v), (&k_pad, &v_pad), &table);
    (gpu.read_f32(&out), want)
}

/// A padded, unshifted grid: windows that hang over the edge attend to the
/// pad rows as keys, and the bias table is indexed by 2D offsets.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn window_attention_pads_with_the_pad_rows(gpu: &Gpu) {
    let shape = Window {
        h: 7,
        w: 5,
        side: 4,
        shift: 0,
        heads: 2,
        head_dim: 8,
    };
    let (got, want) = window_case(gpu, shape, 31);
    close(&got, &want, 1e-4, 1e-2, "window attention, padded, unshifted");
}

/// A shifted window: the roll moves tokens into windows they share with the
/// opposite edge, and the region mask keeps them from attending across.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn window_attention_shifts_and_masks_the_regions(gpu: &Gpu) {
    let shape = Window {
        h: 8,
        w: 8,
        side: 4,
        shift: 2,
        heads: 3,
        head_dim: 4,
    };
    let (got, want) = window_case(gpu, shape, 47);
    close(&got, &want, 1e-4, 1e-2, "window attention, shifted");
    // Padded and shifted at once, as the remover's 264-wide stage runs.
    let shape = Window {
        h: 7,
        w: 9,
        side: 4,
        shift: 2,
        heads: 2,
        head_dim: 4,
    };
    let (got, want) = window_case(gpu, shape, 59);
    close(&got, &want, 1e-4, 1e-2, "window attention, padded and shifted");
}

/// The deformable columns: fractional offsets interpolate between four
/// pixels, offsets past the edge read zero, the mask scales every sample, and
/// the offsets are `(y, x)` per tap.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn deform_im2col_samples_bilinearly_with_the_mask(gpu: &Gpu) {
    let (channels, in_h, in_w) = (3usize, 6usize, 5usize);
    let (k_h, k_w, stride, pad) = (3usize, 3usize, 1usize, 1usize);
    let (out_h, out_w) = (in_h, in_w);
    let taps = k_h * k_w;
    let plane = out_h * out_w;
    let field = signal(channels * in_h * in_w, 71);
    // Offsets up to +-2.5 pixels, so some taps leave the field entirely.
    let offset: Vec<f32> = signal(2 * taps * plane, 73).iter().map(|v| v * 2.5).collect();
    let mask: Vec<f32> = signal(taps * plane, 79).iter().map(|v| v + 1.0).collect();
    let (first, rows) = (7usize, 19usize);
    let bufs = [&field, &offset, &mask].map(|b| upload(gpu, b));
    let cols = upload(gpu, &vec![0.0; rows * channels * taps]);
    let args = DeformArgs {
        channels: channels as u32,
        in_h: in_h as u32,
        in_w: in_w as u32,
        out_h: out_h as u32,
        out_w: out_w as u32,
        k_h: k_h as u32,
        k_w: k_w as u32,
        stride: stride as u32,
        pad: pad as u32,
        row0: first as u32,
        rows: rows as u32,
    };
    launch(
        gpu,
        "kernel_splat_deform_im2col_f32",
        Extent3::new(((channels * taps) as u32).div_ceil(BLOCK), rows as u32, 1),
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(cols.view()),
            KernelArg::Buffer(bufs[0].view()),
            KernelArg::Buffer(bufs[1].view()),
            KernelArg::Buffer(bufs[2].view()),
        ],
    );
    drain(gpu);
    let got = gpu.read_f32(&cols);
    let sample = |c: usize, y: f64, x: f64| -> f64 {
        if y <= -1.0 || y >= in_h as f64 || x <= -1.0 || x >= in_w as f64 {
            return 0.0;
        }
        let (y0, x0) = (y.floor() as i64, x.floor() as i64);
        let (ly, lx) = (y - y0 as f64, x - x0 as f64);
        let at = |yy: i64, xx: i64| {
            if yy < 0 || xx < 0 || yy >= in_h as i64 || xx >= in_w as i64 {
                0.0
            } else {
                f64::from(field[(c * in_h + yy as usize) * in_w + xx as usize])
            }
        };
        (1.0 - ly) * (1.0 - lx) * at(y0, x0)
            + (1.0 - ly) * lx * at(y0, x0 + 1)
            + ly * (1.0 - lx) * at(y0 + 1, x0)
            + ly * lx * at(y0 + 1, x0 + 1)
    };
    let mut want = Vec::with_capacity(got.len());
    let mut outside = 0;
    for r in 0..rows {
        let p = first + r;
        let (oh, ow) = (p / out_w, p % out_w);
        for c in 0..channels {
            for t in 0..taps {
                let dy = f64::from(offset[2 * t * plane + p]);
                let dx = f64::from(offset[(2 * t + 1) * plane + p]);
                let y = (oh * stride) as f64 - pad as f64 + (t / k_w) as f64 + dy;
                let x = (ow * stride) as f64 - pad as f64 + (t % k_w) as f64 + dx;
                if y <= -1.0 || y >= in_h as f64 || x <= -1.0 || x >= in_w as f64 {
                    outside += 1;
                }
                want.push(f64::from(mask[t * plane + p]) * sample(c, y, x));
            }
        }
    }
    assert!(outside > 0, "the fixture never samples outside the field");
    // The floor is the field's own scale: a sample of four nearly cancelling
    // corners is small, and its error is the corners' rounding, not its own.
    close(&got, &want, 1e-5, 1.0, "deform im2col");
}
