//! Bidirectional span attention, the tanh-GELU gated product and the
//! rotary embedding at per-row positions, each against the plainest reading
//! of its formula: an f64 softmax over the visible keys, the tanh form of
//! GELU written out in f64, and the rotation's angle in f64.

use bytemuck::{Pod, Zeroable};
use shiso_core::{Extent3, KernelArg};

use super::{LINEAR_BLOCK, RopeArgs, blocks, close, launch};
use crate::{Gpu, drain, f32_bytes, oracle};

/// `audio_args_span_attn`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct SpanAttnArgs {
    n: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    left: u32,
    right: u32,
    scale: f32,
    row_groups: u32,
    segmented: u32,
}
const _: () = assert!(size_of::<SpanAttnArgs>() == 36);

/// `audio_args_gelu_mul`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct GeluMulArgs {
    rows: u32,
    cols: u32,
    up_stride: u32,
    up_offset: u32,
}
const _: () = assert!(size_of::<GeluMulArgs>() == 16);

/// The span attention's launch, as the host sizes it: head columns a
/// simdgroup takes, keys a block scores, and row groups of eight query items
/// a threadgroup holds.
const SPAN_SLICE: usize = 64;
const SPAN_KEYS: usize = 16;
/// Row groups for a head of `slices` column slices, as the host picks them.
fn span_row_groups(slices: usize) -> usize {
    (8 / slices).clamp(1, 4)
}

/// One span attention geometry; `segments` are the packed sequences' lengths
/// (`None` is one sequence of `n` rows).
#[derive(Debug, Clone)]
struct SpanCase {
    n: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    left: usize,
    right: usize,
    segments: Option<Vec<usize>>,
}

impl SpanCase {
    /// Each row's `[start, end)`.
    fn ranges(&self) -> Vec<(usize, usize)> {
        let Some(lens) = &self.segments else {
            return vec![(0, self.n); self.n];
        };
        assert_eq!(
            lens.iter().sum::<usize>(),
            self.n,
            "segments cover the rows"
        );
        let mut out = Vec::with_capacity(self.n);
        let mut start = 0;
        for &len in lens {
            out.extend(std::iter::repeat_n((start, start + len), len));
            start += len;
        }
        out
    }
}

/// Query `i` of head `h` over keys `j` in
/// `[max(start_i, i - left), min(end_i - 1, i + right)]` of kv head
/// `h / (heads / kv_heads)`, softmax in f64.
fn span_oracle(c: &SpanCase, q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
    let (n, hd) = (c.n, c.hd);
    let w = c.heads * hd;
    let kvw = c.kv_heads * hd;
    let group = c.heads / c.kv_heads;
    let ranges = c.ranges();
    let mut out = vec![0.0f32; n * w];
    for h in 0..c.heads {
        let kvoff = (h / group) * hd;
        for (i, &(start, end)) in ranges.iter().enumerate() {
            let lo = i.saturating_sub(c.left).max(start);
            let hi = i.saturating_add(c.right).min(end - 1);
            let qi = &q[i * w + h * hd..][..hd];
            let scores: Vec<f64> = (lo..=hi)
                .map(|j| {
                    let kj = &k[j * kvw + kvoff..][..hd];
                    let dot: f64 = qi
                        .iter()
                        .zip(kj)
                        .map(|(&a, &b)| f64::from(a) * f64::from(b))
                        .sum();
                    dot * f64::from(scale)
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..hd {
                let acc: f64 = (lo..=hi)
                    .zip(&weights)
                    .map(|(j, &p)| p * f64::from(v[j * kvw + kvoff + d]))
                    .sum();
                out[i * w + h * hd + d] = (acc / total) as f32;
            }
        }
    }
    out
}

fn span_run(gpu: &Gpu, c: &SpanCase, seed: u32) {
    let scale = 1.0 / (c.hd as f32).sqrt();
    // Scaled up so the softmax is far from uniform and a wrongly visible or
    // hidden key moves the answer well past the tolerance.
    let q: Vec<f32> = oracle::signal(c.n * c.heads * c.hd, seed)
        .iter()
        .map(|x| x * 4.0)
        .collect();
    let k: Vec<f32> = oracle::signal(c.n * c.kv_heads * c.hd, seed + 1)
        .iter()
        .map(|x| x * 4.0)
        .collect();
    let v = oracle::signal(c.n * c.kv_heads * c.hd, seed + 2);
    let (qb, kb, vb) = (
        gpu.upload(&f32_bytes(&q)),
        gpu.upload(&f32_bytes(&k)),
        gpu.upload(&f32_bytes(&v)),
    );
    let seg_words: Vec<u8> = c
        .ranges()
        .iter()
        .flat_map(|&(s, e)| [s as u32, e as u32])
        .flat_map(u32::to_le_bytes)
        .collect();
    let segs = gpu.upload(&seg_words);
    let out = gpu.upload(&f32_bytes(&vec![f32::NAN; c.n * c.heads * c.hd]));
    let args = SpanAttnArgs {
        n: c.n as u32,
        heads: c.heads as u32,
        kv_heads: c.kv_heads as u32,
        head_dim: c.hd as u32,
        left: c.left.min(c.n) as u32,
        right: c.right.min(c.n) as u32,
        scale,
        row_groups: span_row_groups(c.hd.div_ceil(SPAN_SLICE)) as u32,
        segmented: u32::from(c.segments.is_some()),
    };
    let slices = c.hd.div_ceil(SPAN_SLICE);
    let row_groups = span_row_groups(slices);
    let items = c.n * (c.heads / c.kv_heads);
    launch(
        gpu,
        "kernel_audio_span_attn_f32",
        Extent3::new(c.kv_heads as u32, blocks(items, 8 * row_groups as u32), 1),
        Extent3::linear((row_groups * slices * 32) as u32),
        (row_groups * slices * 8 * SPAN_KEYS * 4) as u32,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(out.view()),
            KernelArg::Buffer(qb.view()),
            KernelArg::Buffer(kb.view()),
            KernelArg::Buffer(vb.view()),
            KernelArg::Buffer(segs.view()),
        ],
    );
    drain(gpu);
    let want = span_oracle(c, &q, &k, &v, scale);
    close(
        &gpu.read_f32(&out),
        &want,
        1e-4,
        &format!(
            "span attention n {} heads {}/{} hd {} left {} right {} segments {:?}",
            c.n, c.heads, c.kv_heads, c.hd, c.left, c.right, c.segments
        ),
    );
}

/// Span attention over head widths 64 to 512 (one to eight column slices,
/// 96 a last slice narrower than the 64 a simdgroup takes), query groups of one, two
/// and four heads per kv head, row counts that are no whole number of a
/// threadgroup's items or a key block, the fully bidirectional, causal
/// (`right = 0`) and one-sided spans, a single row, the two real geometries
/// at some hundreds of rows (a 16-row sliding window at 256 wide over two kv
/// heads; 512 wide over one, unbounded), and several packed segments of
/// uneven lengths - one of a single row - alone and under a window. The
/// output starts NaN, so a row the kernel skips fails.
///
/// # Panics
///
/// When an output differs from the oracle by more than 1e-4 relative.
pub fn span_attention_matches_windowed_bidirectional_and_causal(gpu: &Gpu) {
    let full = usize::MAX;
    let segs = |lens: &[usize]| Some(lens.to_vec());
    let cases = [
        // (n, heads, kv_heads, hd, left, right, segments)
        (37, 4, 2, 64, 5, 5, None),
        (37, 8, 2, 64, full, full, None),
        (37, 4, 2, 64, full, 0, None),
        (37, 4, 1, 256, 5, 5, None),
        (37, 4, 2, 256, full, full, None),
        (37, 4, 2, 256, 3, 0, None),
        (37, 4, 1, 512, 5, 5, None),
        (37, 4, 1, 512, full, full, None),
        (37, 8, 2, 512, full, 0, None),
        (37, 4, 2, 256, 0, 7, None),
        (70, 4, 2, 128, 40, 2, None),
        (45, 3, 3, 96, 9, 20, None),
        (1, 4, 2, 64, 5, 5, None),
        (1, 4, 2, 256, full, full, None),
        (1, 4, 1, 512, 0, 0, None),
        (300, 4, 2, 256, 16, 16, None),
        (300, 4, 1, 512, full, full, None),
        (61, 4, 2, 256, full, full, segs(&[17, 1, 30, 13])),
        (61, 4, 1, 512, full, full, segs(&[1, 40, 20])),
        (61, 4, 2, 256, 4, 4, segs(&[17, 1, 30, 13])),
        (77, 4, 1, 512, 6, 3, segs(&[33, 1, 1, 42])),
        (50, 8, 2, 64, full, 0, segs(&[25, 25])),
        (300, 4, 2, 256, 16, 16, segs(&[100, 1, 150, 49])),
    ];
    for (i, (n, heads, kv_heads, hd, left, right, segments)) in cases.into_iter().enumerate() {
        span_run(
            gpu,
            &SpanCase {
                n,
                heads,
                kv_heads,
                hd,
                left,
                right,
                segments,
            },
            301 + 3 * i as u32,
        );
    }
}

/// The rotary embedding at per-row positions: with `positions[r] == pos0 +
/// r` it is the plain rotation bit for bit; at arbitrary positions (two
/// packed sequences restarting at 0, and one far on) each row is the plain
/// rotation of that row alone at its position, bit for bit; and the rows
/// near the start are within 2e-5 of the rotation by the f32 angle taken in
/// f64. Only near the start: the device's `pow` may sit an ulp off the
/// host's, and the angle carries that times the position (~3e-5 of a value
/// at 300 on an M1 Max).
///
/// # Panics
///
/// When a value differs from the plain rotation's, or from the oracle by more
/// than 2e-5 relative.
pub fn rope_half_pos_takes_each_rows_position(gpu: &Gpu) {
    let (rows, heads, hd, theta) = (11usize, 3usize, 256usize, 10_000.0f32);
    let width = heads * hd;
    let x = oracle::signal(rows * width, 97);
    let rotate = |x: &[f32], pos0: u32, positions: Option<&[u32]>| {
        let rows = x.len() / width;
        let buf = gpu.upload(&f32_bytes(x));
        let args = RopeArgs {
            rows: rows as u32,
            heads: heads as u32,
            head_dim: hd as u32,
            pos0,
            theta,
        };
        let pos_buf = positions
            .map(|p| gpu.upload(&p.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()));
        let mut bind = vec![
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(buf.view()),
        ];
        if let Some(p) = &pos_buf {
            bind.push(KernelArg::Buffer(p.view()));
        }
        launch(
            gpu,
            if positions.is_some() {
                "kernel_audio_rope_half_pos_f32"
            } else {
                "kernel_audio_rope_half_f32"
            },
            Extent3::linear(blocks(rows * width / 2, LINEAR_BLOCK)),
            Extent3::linear(LINEAR_BLOCK),
            0,
            &bind,
        );
        drain(gpu);
        gpu.read_f32(&buf)
    };

    let counting: Vec<u32> = (0..rows as u32).map(|r| r + 7).collect();
    close(
        &rotate(&x, 0, Some(&counting)),
        &rotate(&x, 7, None),
        0.0,
        "rope at counting positions",
    );

    let positions: [u32; 11] = [0, 1, 2, 3, 0, 1, 2, 3, 4, 5, 8191];
    let got = rotate(&x, 0, Some(&positions));
    for (r, &pos) in positions.iter().enumerate() {
        close(
            &got[r * width..][..width],
            &rotate(&x[r * width..][..width], pos, None),
            0.0,
            &format!("rope row {r} at position {pos}"),
        );
    }

    let near = &positions[..10];
    let got = rotate(&x[..near.len() * width], 0, Some(near));
    let mut want = x[..near.len() * width].to_vec();
    let half = hd / 2;
    for (r, &pos) in near.iter().enumerate() {
        for h in 0..heads {
            for m in 0..half {
                let inv_freq = theta.powf(-2.0 * m as f32 / hd as f32);
                let ang = f64::from(pos as f32 * inv_freq);
                let base = r * width + h * hd;
                let (a, b) = (f64::from(x[base + m]), f64::from(x[base + m + half]));
                want[base + m] = (a * ang.cos() - b * ang.sin()) as f32;
                want[base + m + half] = (b * ang.cos() + a * ang.sin()) as f32;
            }
        }
    }
    close(&got, &want, 2e-5, "rope at per-row positions");
}

/// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`, in f64.
fn gelu_tanh(x: f32) -> f64 {
    let x = f64::from(x);
    let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

fn gelu_mul_run(gpu: &Gpu, rows: usize, cols: usize, up_stride: usize, up_offset: usize) {
    let count = rows * cols;
    let gate: Vec<f32> = oracle::signal(count, 41).iter().map(|x| x * 6.0).collect();
    let up = oracle::signal(rows * up_stride, 42);
    let g = gpu.upload(&f32_bytes(&gate));
    let u = gpu.upload(&f32_bytes(&up));
    let args = GeluMulArgs {
        rows: rows as u32,
        cols: cols as u32,
        up_stride: up_stride as u32,
        up_offset: up_offset as u32,
    };
    launch(
        gpu,
        "kernel_audio_gelu_tanh_mul_f32",
        Extent3::linear(blocks(count, LINEAR_BLOCK)),
        Extent3::linear(LINEAR_BLOCK),
        0,
        &[
            KernelArg::Bytes(bytemuck::bytes_of(&args)),
            KernelArg::Buffer(g.view()),
            KernelArg::Buffer(g.view()),
            KernelArg::Buffer(u.view()),
        ],
    );
    drain(gpu);
    let want: Vec<f32> = (0..count)
        .map(|i| {
            let (r, c) = (i / cols, i % cols);
            (gelu_tanh(gate[i]) * f64::from(up[r * up_stride + up_offset + c])) as f32
        })
        .collect();
    close(
        &gpu.read_f32(&g),
        &want,
        1e-5,
        &format!("gelu_tanh_mul stride {up_stride} offset {up_offset}"),
    );
}

/// The gated-GELU product takes the tanh GELU of the gate, not of the value, in
/// place on the gate, over a count that is no whole number of blocks: once
/// with the value rows packed (`up_stride == cols`) and once reading the
/// middle column block of a three-block-wide matrix.
///
/// # Panics
///
/// When an output differs from the oracle by more than 1e-5 relative.
pub fn gelu_tanh_mul_gates_plain_and_strided_up_rows(gpu: &Gpu) {
    gelu_mul_run(gpu, 13, 77, 77, 0);
    gelu_mul_run(gpu, 13, 77, 3 * 77, 77);
}
