//! The span attention, the rotation at per-row positions and the tanh-GELU
//! product through [`Ops`], on the device: each head width must reach the
//! kernel with the threads and threadgroup bytes its column slices take, and
//! each descriptor's buffer order must be the kernel's. The kernels' own cases
//! in the shared kernel tests grade the arithmetic over many more shapes;
//! these pin the launchers.

use shiso_core::CoreError;

use super::load::{download_f32, upload_f32};
use super::{GeluMulDesc, Ops, RopeDesc, RopePosDesc, SpanAttnDesc};

/// Deterministic values in `[-1, 1)`.
fn signal(len: usize, seed: u32) -> Vec<f32> {
    let mut state = seed | 0x9E37_79B9;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            #[expect(clippy::cast_precision_loss, reason = "a 24-bit value is exact in f32")]
            let unit = (state >> 8) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        })
        .collect()
}

/// One span attention geometry: rows, query heads, kv heads, head width and
/// the window's two sides.
#[derive(Debug, Clone, Copy)]
struct Span {
    n: usize,
    heads: usize,
    kv_heads: usize,
    hd: usize,
    left: usize,
    right: usize,
}

/// Query `i` of head `h` over keys `[i - left, i + right] ∩ [0, n)` of kv head
/// `h / (heads / kv_heads)`, softmax in f64.
fn span_oracle(span: Span, q: &[f32], k: &[f32], v: &[f32], scale: f32) -> Vec<f32> {
    let (qw, kvw) = (span.heads * span.hd, span.kv_heads * span.hd);
    let mut out = vec![0.0f32; span.n * qw];
    for h in 0..span.heads {
        let kvoff = h / (span.heads / span.kv_heads) * span.hd;
        for i in 0..span.n {
            let keys = i.saturating_sub(span.left)..=i.saturating_add(span.right).min(span.n - 1);
            let scores: Vec<f64> = keys
                .clone()
                .map(|j| {
                    (0..span.hd)
                        .map(|d| {
                            f64::from(q[i * qw + h * span.hd + d])
                                * f64::from(k[j * kvw + kvoff + d])
                        })
                        .sum::<f64>()
                        * f64::from(scale)
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|x| (x - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..span.hd {
                let acc: f64 = keys
                    .clone()
                    .zip(&weights)
                    .map(|(j, &pj)| pj * f64::from(v[j * kvw + kvoff + d]))
                    .sum();
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "the oracle's f64 back to the device's f32"
                )]
                let value = (acc / total) as f32;
                out[i * qw + h * span.hd + d] = value;
            }
        }
    }
    out
}

fn span_case(ops: &Ops, s: Span, seed: u32) {
    #[expect(clippy::cast_precision_loss, reason = "a head width is exact in f32")]
    let scale = 1.0 / (s.hd as f32).sqrt();
    let scaled = |len, seed| -> Vec<f32> { signal(len, seed).iter().map(|x| x * 4.0).collect() };
    let q = scaled(s.n * s.heads * s.hd, seed);
    let k = scaled(s.n * s.kv_heads * s.hd, seed + 1);
    let v = signal(s.n * s.kv_heads * s.hd, seed + 2);
    let (qb, kb, vb) = (
        upload_f32(ops, &q).expect("uploads"),
        upload_f32(ops, &k).expect("uploads"),
        upload_f32(ops, &v).expect("uploads"),
    );
    let out = ops.scratch(q.len()).expect("allocates");
    ops.span_attention(&SpanAttnDesc {
        out: out.view(),
        q: qb.view(),
        k: kb.view(),
        v: vb.view(),
        segments: None,
        n: s.n,
        heads: s.heads,
        kv_heads: s.kv_heads,
        head_dim: s.hd,
        left: s.left,
        right: s.right,
        scale,
    })
    .expect("launches");
    let got = download_f32(ops, out.view(), q.len()).expect("downloads");
    let want = span_oracle(s, &q, &k, &v, scale);
    for (i, (&g, &w)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - w).abs() <= 1e-4 * w.abs().max(1.0),
            "{s:?}[{i}]: got {g}, want {w}"
        );
    }
}

#[test]
#[ignore = "opens a device backend"]
fn span_attention_sizes_every_width_and_refuses_what_it_cannot_hold() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    // The two widths a sliding and a full layer take, and a one-slice width.
    for (seed, s) in [
        Span {
            n: 21,
            heads: 4,
            kv_heads: 2,
            hd: 256,
            left: 4,
            right: 4,
        },
        Span {
            n: 21,
            heads: 4,
            kv_heads: 1,
            hd: 512,
            left: usize::MAX,
            right: usize::MAX,
        },
        Span {
            n: 21,
            heads: 4,
            kv_heads: 2,
            hd: 96,
            left: 21,
            right: 0,
        },
    ]
    .into_iter()
    .enumerate()
    {
        span_case(&ops, s, 11 + 3 * u32::try_from(seed).expect("fits"));
    }

    let buf = ops.scratch(4 * 4 * 1024).expect("allocates");
    let desc = |hd, heads, kv_heads| SpanAttnDesc {
        out: buf.view(),
        q: buf.view(),
        k: buf.view(),
        v: buf.view(),
        segments: None,
        n: 2,
        heads,
        kv_heads,
        head_dim: hd,
        left: 1,
        right: 1,
        scale: 1.0,
    };
    for (hd, heads, kv_heads, key) in [
        (0, 4, 2, "span_attention.head_dim"),
        (48, 4, 2, "span_attention.head_dim"),
        (544, 4, 2, "span_attention.head_dim"),
        (64, 4, 3, "span_attention.heads"),
        (64, 4, 0, "span_attention.heads"),
    ] {
        match ops.span_attention(&desc(hd, heads, kv_heads)) {
            Err(CoreError::InvalidValue { key: got, .. }) => assert_eq!(got, key, "hd {hd}"),
            other => panic!("hd {hd}, heads {heads}/{kv_heads}: {other:?}"),
        }
    }
}

#[test]
#[ignore = "opens a device backend"]
fn gelu_tanh_mul_reads_the_strided_column_block() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    let (rows, cols, blocks, block) = (5usize, 40usize, 3usize, 2usize);
    let stride = blocks * cols;
    let gate: Vec<f32> = signal(rows * cols, 7).iter().map(|x| x * 6.0).collect();
    let up = signal(rows * stride, 8);
    let g = upload_f32(&ops, &gate).expect("uploads");
    let u = upload_f32(&ops, &up).expect("uploads");
    let out = ops.scratch(rows * cols).expect("allocates");
    let desc = GeluMulDesc {
        out: out.view(),
        gate: g.view(),
        up: u.view(),
        rows,
        cols,
        up_stride: stride,
        up_offset: block * cols,
    };
    ops.gelu_tanh_mul(&desc).expect("launches");
    let got = download_f32(&ops, out.view(), rows * cols).expect("downloads");
    for (i, (&g, &x)) in got.iter().zip(&gate).enumerate() {
        let x = f64::from(x);
        let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x * x * x);
        let u = f64::from(up[i / cols * stride + block * cols + i % cols]);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the oracle's f64 back to the device's f32"
        )]
        let want = (0.5 * x * (1.0 + inner.tanh()) * u) as f32;
        assert!(
            (g - want).abs() <= 1e-5 * want.abs().max(1.0),
            "[{i}]: got {g}, want {want}"
        );
    }

    // A block past its row would read the next row's columns.
    let past = GeluMulDesc {
        up_offset: stride - cols + 1,
        ..desc
    };
    assert!(matches!(
        ops.gelu_tanh_mul(&past),
        Err(CoreError::InvalidValue {
            key: "gelu_tanh_mul.up_stride",
            ..
        })
    ));
}

/// `values` as a u32 buffer on the device.
fn upload_u32(ops: &Ops, values: &[u32]) -> shiso_core::DeviceBuffer {
    let buf = ops.scratch(values.len()).expect("allocates");
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    ops.backend().copy_h2d(buf.view(), &bytes).expect("uploads");
    buf
}

#[test]
#[ignore = "opens a device backend"]
fn packed_segments_attend_as_each_sequence_alone() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    // A sliding and a full layer's geometry over three sequences, one of a
    // single row.
    for (hd, kv_heads, radius) in [(256usize, 2usize, 3usize), (512, 1, usize::MAX)] {
        let (heads, lens) = (4usize, [9usize, 1, 14]);
        let n: usize = lens.iter().sum();
        let (qw, kvw) = (heads * hd, kv_heads * hd);
        let q = signal(n * qw, 31);
        let k = signal(n * kvw, 32);
        let v = signal(n * kvw, 33);
        let (qb, kb, vb) = (
            upload_f32(&ops, &q).expect("uploads"),
            upload_f32(&ops, &k).expect("uploads"),
            upload_f32(&ops, &v).expect("uploads"),
        );
        let mut ranges = Vec::new();
        let mut start = 0u32;
        for &len in &lens {
            let end = start + u32::try_from(len).expect("fits");
            for _ in 0..len {
                ranges.extend([start, end]);
            }
            start = end;
        }
        let seg = upload_u32(&ops, &ranges);
        let out = ops.scratch(n * qw).expect("allocates");
        ops.span_attention(&SpanAttnDesc {
            out: out.view(),
            q: qb.view(),
            k: kb.view(),
            v: vb.view(),
            segments: Some(seg.view()),
            n,
            heads,
            kv_heads,
            head_dim: hd,
            left: radius,
            right: radius,
            scale: 0.1,
        })
        .expect("launches");
        let packed = download_f32(&ops, out.view(), n * qw).expect("downloads");

        let mut row0 = 0;
        for &len in &lens {
            let alone = |x: &[f32], w: usize| upload_f32(&ops, &x[row0 * w..(row0 + len) * w]);
            let (qa, ka, va) = (
                alone(&q, qw).expect("uploads"),
                alone(&k, kvw).expect("uploads"),
                alone(&v, kvw).expect("uploads"),
            );
            let oa = ops.scratch(len * qw).expect("allocates");
            ops.span_attention(&SpanAttnDesc {
                out: oa.view(),
                q: qa.view(),
                k: ka.view(),
                v: va.view(),
                segments: None,
                n: len,
                heads,
                kv_heads,
                head_dim: hd,
                left: radius,
                right: radius,
                scale: 0.1,
            })
            .expect("launches");
            let want = download_f32(&ops, oa.view(), len * qw).expect("downloads");
            // The packed launch visits a sequence's keys in other blocks
            // than the lone one, so the sums round apart.
            for (i, (&g, &w)) in packed[row0 * qw..][..len * qw]
                .iter()
                .zip(&want)
                .enumerate()
            {
                assert!(
                    (g - w).abs() <= 1e-5 * w.abs().max(1.0),
                    "hd {hd} sequence at {row0}[{i}]: packed {g}, alone {w}"
                );
            }
            row0 += len;
        }
    }
}

#[test]
#[ignore = "opens a device backend"]
fn rope_half_rows_reads_each_rows_position() {
    let Some(backend) = super::testing::device() else {
        return;
    };
    let ops = Ops::new(backend);
    let (rows, heads, hd, theta) = (6usize, 4usize, 256usize, 10_000.0f32);
    let width = heads * hd;
    let x = signal(rows * width, 41);
    let positions = [0u32, 1, 2, 0, 1, 900];
    let buf = upload_f32(&ops, &x).expect("uploads");
    let pos = upload_u32(&ops, &positions);
    ops.rope_half_rows(&RopePosDesc {
        x: buf.view(),
        positions: pos.view(),
        rows,
        heads,
        head_dim: hd,
        theta,
    })
    .expect("launches");
    let got = download_f32(&ops, buf.view(), rows * width).expect("downloads");
    for (r, &p) in positions.iter().enumerate() {
        let one = upload_f32(&ops, &x[r * width..][..width]).expect("uploads");
        ops.rope_half(&RopeDesc {
            x: one.view(),
            rows: 1,
            heads,
            head_dim: hd,
            pos0: usize::try_from(p).expect("fits"),
            theta,
        })
        .expect("launches");
        let want = download_f32(&ops, one.view(), width).expect("downloads");
        assert_eq!(&got[r * width..][..width], &want[..], "row {r} at {p}");
    }

    // One position short of the rows.
    let short = upload_u32(&ops, &positions[..rows - 1]);
    assert!(matches!(
        ops.rope_half_rows(&RopePosDesc {
            x: buf.view(),
            positions: short.view(),
            rows,
            heads,
            head_dim: hd,
            theta,
        }),
        Err(CoreError::ShapeMismatch {
            field: "rope_half_rows.positions",
            ..
        })
    ));
}
