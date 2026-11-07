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
pub const ENTRIES: [&str; 9] = [
    "kernel_splat_relu_f32",
    "kernel_splat_sigmoid_f32",
    "kernel_splat_gelu_erf_f32",
    "kernel_splat_resize_bilinear_f32",
    "kernel_splat_merge2x2_f32",
    "kernel_splat_window_attn_f32",
    "kernel_splat_deform_im2col_f32",
    "kernel_splat_attn_f32",
    "kernel_splat_rotary_tables_f32",
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

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct AttnArgs {
    queries: u32,
    keys: u32,
    heads: u32,
    width: u32,
    scale: f32,
}

/// The fused attention's fixed head width, query rows per threadgroup and
/// threads per threadgroup (`SPLAT_ATTN_*` in the kernel files).
const ATTN_HEAD: usize = 64;
const ATTN_QROWS: usize = 64;
const ATTN_THREADS: u32 = 256;

const _: () = assert!(size_of::<UnaryArgs>() == 4);
const _: () = assert!(size_of::<AttnArgs>() == 20);
const _: () = assert!(size_of::<RotaryArgs>() == 16);

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct RotaryArgs {
    row0: u32,
    rows: u32,
    heads: u32,
    head_dim: u32,
}
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
    let want: Vec<f64> = x
        .iter()
        .map(|&v| 1.0 / (1.0 + (-f64::from(v)).exp()))
        .collect();
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
            Extent3::new(
                (out_w as u32).div_ceil(BLOCK),
                out_h as u32,
                channels as u32,
            ),
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
                    want.push(f64::from(
                        tokens[((2 * i + dy) * w + 2 * j + dx) * channels + c],
                    ));
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
    let (gq, gk, gv) = (
        grid(qkv.0, &zeros),
        grid(qkv.1, pads.0),
        grid(qkv.2, pads.1),
    );
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
                                    rq[qy * wp + qx][head * hd + d]
                                        * scale
                                        * rk[ky * wp + kx][head * hd + d]
                                })
                                .sum();
                            let rel =
                                (i / ws + ws - 1 - j / ws) * span + (i % ws + ws - 1 - j % ws);
                            let mut score = dot + f64::from(table[rel * heads + head]);
                            if shift > 0
                                && (region(qy, hp), region(qx, wp))
                                    != (region(ky, hp), region(kx, wp))
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
    close(
        &got,
        &want,
        1e-4,
        1e-2,
        "window attention, padded, unshifted",
    );
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
    close(
        &got,
        &want,
        1e-4,
        1e-2,
        "window attention, padded and shifted",
    );
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
    let offset: Vec<f32> = signal(2 * taps * plane, 73)
        .iter()
        .map(|v| v * 2.5)
        .collect();
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

/// The remover's own window: 12 wide (144 tokens, more than one
/// threadgroup's 64 threads, so each thread takes several queries), shifted
/// by 6, on a grid that is not a multiple of it, with a 23x23 bias table.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn window_attention_at_the_production_window(gpu: &Gpu) {
    let shape = Window {
        h: 14,
        w: 17,
        side: 12,
        shift: 6,
        heads: 2,
        head_dim: 8,
    };
    let (got, want) = window_case(gpu, shape, 83);
    close(
        &got,
        &want,
        1e-4,
        1e-2,
        "window attention, 12-wide, shifted, padded",
    );
    let shape = Window { shift: 0, ..shape };
    let (got, want) = window_case(gpu, shape, 89);
    close(
        &got,
        &want,
        1e-4,
        1e-2,
        "window attention, 12-wide, unshifted, padded",
    );
}

/// `softmax(scale * q k^T) v` per head in `f64`, max-subtracted, for the
/// query rows in `rows` only: `q` is `(queries, heads * 64)` and `k`, `v` are
/// `(keys, heads * 64)`, row-major. The answer is `(rows.len(), heads * 64)`.
fn attn_oracle(
    qkv: (&[f32], &[f32], &[f32]),
    keys: usize,
    heads: usize,
    rows: std::ops::Range<usize>,
) -> Vec<f64> {
    let (q, k, v) = qkv;
    let width = heads * ATTN_HEAD;
    let scale = 1.0 / (ATTN_HEAD as f64).sqrt();
    let mut out = vec![0.0f64; rows.len() * width];
    let mut scores = vec![0.0f64; keys];
    for (i, r) in rows.enumerate() {
        for h in 0..heads {
            let qrow = &q[r * width + h * ATTN_HEAD..][..ATTN_HEAD];
            for (j, score) in scores.iter_mut().enumerate() {
                let krow = &k[j * width + h * ATTN_HEAD..][..ATTN_HEAD];
                *score = scale
                    * qrow
                        .iter()
                        .zip(krow)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum::<f64>();
            }
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for score in &mut scores {
                *score = (*score - max).exp();
                sum += *score;
            }
            let orow = &mut out[i * width + h * ATTN_HEAD..][..ATTN_HEAD];
            for (j, p) in scores.iter().enumerate() {
                let vrow = &v[j * width + h * ATTN_HEAD..][..ATTN_HEAD];
                for (o, &x) in orow.iter_mut().zip(vrow) {
                    *o += p / sum * f64::from(x);
                }
            }
        }
    }
    out
}

/// Launches the fused attention over device q, k, v and reads the answer.
fn attn_launch(gpu: &Gpu, qkv: (&[f32], &[f32], &[f32]), queries: usize, heads: usize) -> Vec<f32> {
    let width = heads * ATTN_HEAD;
    let keys = qkv.1.len() / width;
    let [q, k, v] = [qkv.0, qkv.1, qkv.2].map(|b| upload(gpu, b));
    let out = upload(gpu, &vec![f32::NAN; queries * width]);
    let args = AttnArgs {
        queries: queries as u32,
        keys: keys as u32,
        heads: heads as u32,
        width: width as u32,
        scale: 1.0 / (ATTN_HEAD as f32).sqrt(),
    };
    run(
        gpu,
        KernelRef::new(MODULE, "kernel_splat_attn_f32"),
        &LaunchArgs {
            grid: Extent3::new(queries.div_ceil(ATTN_QROWS) as u32, heads as u32, 1),
            block: Extent3::linear(ATTN_THREADS),
            smem_bytes: 0,
            args: &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(out.view()),
                KernelArg::Buffer(q.view()),
                KernelArg::Buffer(k.view()),
                KernelArg::Buffer(v.view()),
            ],
        },
    );
    drain(gpu);
    gpu.read_f32(&out)
}

/// Cross attention with ragged counts: 45 queries over 77 keys, so the last
/// key block and the only query tile are both partial.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn attention_matches_the_softmax_oracle(gpu: &Gpu) {
    let (queries, keys, heads) = (45, 77, 3);
    let width = heads * ATTN_HEAD;
    let q = signal(queries * width, 41);
    let k = signal(keys * width, 42);
    let v = signal(keys * width, 43);
    let got = attn_launch(gpu, (&q, &k, &v), queries, heads);
    let want = attn_oracle((&q, &k, &v), keys, heads, 0..queries);
    close(&got, &want, 1e-5, 0.1, "fused attention, ragged");
}

/// Scores in the tens: the online softmax must subtract its running maximum
/// before exponentiating, or the sums overflow.
///
/// # Panics
///
/// When the device's answer is outside the case's bound or not finite.
pub fn attention_survives_large_scores(gpu: &Gpu) {
    let (queries, keys, heads) = (64, 96, 2);
    let width = heads * ATTN_HEAD;
    let q: Vec<f32> = signal(queries * width, 51)
        .into_iter()
        .map(|x| x * 20.0)
        .collect();
    let k = signal(keys * width, 52);
    let v = signal(keys * width, 53);
    let got = attn_launch(gpu, (&q, &k, &v), queries, heads);
    assert!(got.iter().all(|x| x.is_finite()), "a non-finite output");
    let want = attn_oracle((&q, &k, &v), keys, heads, 0..queries);
    // A score near 25 carries about 1e-5 of f32 rounding from its own dot
    // product, which `exp` hands on to every probability: 5e-6 absolute is
    // that rounding, where an overflowed or unshifted softmax is off by far
    // more or not finite.
    close(&got, &want, 5e-5, 0.1, "fused attention, large scores");
}

/// The denoiser's joint shape, 12294 rows of 16 heads: the first and last
/// query tiles of the last head against the oracle, which catches tile and
/// key-block boundaries at the real size.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn attention_at_the_joint_shape(gpu: &Gpu) {
    let (rows, heads) = (12_294, 16);
    let width = heads * ATTN_HEAD;
    let q = signal(rows * width, 61);
    let k = signal(rows * width, 62);
    let v = signal(rows * width, 63);
    let got = attn_launch(gpu, (&q, &k, &v), rows, heads);
    for range in [0..64, rows - 64..rows] {
        let want = attn_oracle((&q, &k, &v), rows, heads, range.clone());
        let got_rows = &got[range.start * width..range.end * width];
        // Only the last head is compared; the oracle computes all of them.
        let head = heads - 1;
        let pick = |all: &[f32]| -> Vec<f32> {
            all.chunks_exact(width)
                .flat_map(|row| row[head * ATTN_HEAD..][..ATTN_HEAD].to_vec())
                .collect()
        };
        let pick64 = |all: &[f64]| -> Vec<f64> {
            all.chunks_exact(width)
                .flat_map(|row| row[head * ATTN_HEAD..][..ATTN_HEAD].to_vec())
                .collect()
        };
        close(
            &pick(got_rows),
            &pick64(&want),
            1e-4,
            0.1,
            "fused attention, joint shape",
        );
    }
}

/// The denoiser's joint attention, 12294 rows of 16 heads of 64, for
/// `kernel_bench`.
pub(crate) fn bench_attention(gpu: &Gpu) -> crate::bench::Fixture {
    let (rows, heads) = (12_294usize, 16usize);
    let width = heads * ATTN_HEAD;
    let [q, k, v] = [61, 62, 63].map(|seed| upload(gpu, &signal(rows * width, seed)));
    let out = upload(gpu, &vec![0.0; rows * width]);
    let args = AttnArgs {
        queries: rows as u32,
        keys: rows as u32,
        heads: heads as u32,
        width: width as u32,
        scale: 1.0 / (ATTN_HEAD as f32).sqrt(),
    };
    let bytes = (4 * rows * width * 4) as u64;
    let flops = (4 * rows * rows * ATTN_HEAD * heads) as u64;
    crate::bench::Fixture::new(bytes, flops, move |gpu| {
        run(
            gpu,
            KernelRef::new(MODULE, "kernel_splat_attn_f32"),
            &LaunchArgs {
                grid: Extent3::new(rows.div_ceil(ATTN_QROWS) as u32, heads as u32, 1),
                block: Extent3::linear(ATTN_THREADS),
                smem_bytes: 0,
                args: &[
                    KernelArg::Bytes(bytemuck::bytes_of(&args)),
                    KernelArg::Buffer(out.view()),
                    KernelArg::Buffer(q.view()),
                    KernelArg::Buffer(k.view()),
                    KernelArg::Buffer(v.view()),
                ],
            },
        );
    })
}

/// The rotary tables across two launches (70000 rows, past one launch's
/// 65535 on the grid's second axis), at offsets up to 64 so the angles reach
/// the hundreds, where a fast sine is off by far more than the bound.
///
/// The oracle forms each angle in `f32` exactly as the kernel's contract
/// does (`offset * rate`, the rate being the frequency times pi rounded once)
/// and takes its sine and cosine in `f64`: the bound is then the kernel's
/// own trigonometry.
///
/// # Panics
///
/// When the device's answer is outside the case's bound.
pub fn rotary_tables_match_the_angle_formula(gpu: &Gpu) {
    let (rows, heads, head_dim) = (70_000usize, 2usize, 8usize);
    let pairs = head_dim / 2;
    let width = heads * head_dim;
    let offsets: Vec<f32> = signal(rows * heads * 3, 71)
        .into_iter()
        .map(|x| x * 64.0)
        .collect();
    let rates = [0.1f32, 0.7, 2.3, 4.0].map(|f| f * std::f32::consts::PI);
    let axis = [0u32, 1, 2, 0];
    let d_offsets = upload(gpu, &offsets);
    let d_rates = upload(gpu, &rates);
    let d_axis = gpu.upload(bytemuck::cast_slice(&axis));
    let cos_m1 = upload(gpu, &vec![f32::NAN; rows * width]);
    let sin = upload(gpu, &vec![f32::NAN; rows * width]);
    let per_launch = 65_535usize;
    for row0 in (0..rows).step_by(per_launch) {
        let count = per_launch.min(rows - row0);
        let args = RotaryArgs {
            row0: row0 as u32,
            rows: rows as u32,
            heads: heads as u32,
            head_dim: head_dim as u32,
        };
        launch(
            gpu,
            "kernel_splat_rotary_tables_f32",
            Extent3::new(((heads * pairs) as u32).div_ceil(BLOCK), count as u32, 1),
            &[
                KernelArg::Bytes(bytemuck::bytes_of(&args)),
                KernelArg::Buffer(cos_m1.view()),
                KernelArg::Buffer(sin.view()),
                KernelArg::Buffer(d_offsets.view()),
                KernelArg::Buffer(d_rates.view()),
                KernelArg::Buffer(d_axis.view()),
            ],
        );
    }
    drain(gpu);
    let mut want_cos = vec![0.0f64; rows * width];
    let mut want_sin = vec![0.0f64; rows * width];
    for r in 0..rows {
        for h in 0..heads {
            for j in 0..pairs {
                let offset = offsets[(r * heads + h) * 3 + axis[j] as usize];
                let angle = f64::from(offset * rates[j]);
                let at = r * width + h * head_dim + 2 * j;
                for c in [at, at + 1] {
                    want_cos[c] = angle.cos() - 1.0;
                    want_sin[c] = angle.sin();
                }
            }
        }
    }
    close(
        &gpu.read_f32(&cos_m1),
        &want_cos,
        2e-6,
        1.0,
        "rotary cos - 1",
    );
    close(&gpu.read_f32(&sin), &want_sin, 2e-6, 1.0, "rotary sin");
}
