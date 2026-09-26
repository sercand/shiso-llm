//! `Linear` over every weight type an audio checkpoint carries, on Metal,
//! against a host oracle over the dequantised weights.
//!
//! The quantisation is already inside the oracle's weights, so what the
//! comparison measures is accumulation order and nothing else. `rows = 600`
//! crosses the 512-row slice `Linear::forward` cuts projected calls into, and
//! `Q8_0` at 37 and 600 rows is the case that moved that type onto the audio
//! GEMM: the engine's tiled path staged it through half precision.
#![cfg(target_os = "macos")]
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the fixtures pack small literal-sized quantised blocks"
)]

use std::sync::Arc;

use shiso_core::{Backend, DType, DeviceBuffer};
use shiso_engine::projection::{AffineRanges, Weight};
use shiso_models::audio_ops::{GemmType, Linear, LinearWeight, Ops};

const IN: usize = 1024;
const OUT: usize = 640;

fn signal(len: usize, seed: u32) -> Vec<f32> {
    let mut s = u64::from(seed).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 20_001) as f32 / 10_000.0) - 1.0
        })
        .collect()
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bf16(v: f32) -> u16 {
    (v.to_bits() >> 16) as u16
}

fn widen(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

/// Weight bytes as the checkpoint would hold them, and their dequantised values.
enum Fixture {
    Dense(GemmType, Vec<u8>, Vec<f32>),
    Q8(Vec<u8>, Vec<f32>),
    Affine8 {
        words: Vec<u8>,
        scales: Vec<u8>,
        biases: Vec<u8>,
        values: Vec<f32>,
    },
}

fn fixture(kind: &str) -> Fixture {
    let raw = signal(IN * OUT, 7);
    match kind {
        "f32" => Fixture::Dense(GemmType::F32, f32_bytes(&raw), raw),
        "bf16" => {
            let b: Vec<u16> = raw.iter().map(|v| bf16(*v)).collect();
            Fixture::Dense(
                GemmType::Bf16,
                b.iter().flat_map(|x| x.to_le_bytes()).collect(),
                b.iter().map(|x| widen(*x)).collect(),
            )
        }
        "f16" => {
            let h: Vec<half::f16> = raw.iter().map(|v| half::f16::from_f32(*v)).collect();
            Fixture::Dense(
                GemmType::F16,
                h.iter().flat_map(|x| x.to_le_bytes()).collect(),
                h.iter().map(|x| x.to_f32()).collect(),
            )
        }
        "q8_0" => {
            let (mut bytes, mut values) = (Vec::new(), Vec::new());
            for block in raw.chunks(32) {
                let amax = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
                let d = half::f16::from_f32(amax / 127.0);
                bytes.extend_from_slice(&d.to_le_bytes());
                for v in block {
                    let q = if d.to_f32() == 0.0 {
                        0
                    } else {
                        (v / d.to_f32()).round() as i8
                    };
                    bytes.push(q as u8);
                    values.push(f32::from(q) * d.to_f32());
                }
            }
            Fixture::Q8(bytes, values)
        }
        "affine8" => {
            let (mut words, mut scales, mut biases, mut values) =
                (Vec::new(), Vec::new(), Vec::new(), vec![0.0; IN * OUT]);
            for r in 0..OUT {
                let row = &raw[r * IN..(r + 1) * IN];
                let mut q = vec![0u8; IN];
                for g in 0..IN / 64 {
                    let grp = &row[g * 64..(g + 1) * 64];
                    let (lo, hi) = grp
                        .iter()
                        .fold((f32::MAX, f32::MIN), |(a, b), v| (a.min(*v), b.max(*v)));
                    let (s, b) = (bf16((hi - lo) / 255.0), bf16(lo));
                    scales.extend_from_slice(&s.to_le_bytes());
                    biases.extend_from_slice(&b.to_le_bytes());
                    for (i, v) in grp.iter().enumerate() {
                        let qi = ((v - widen(b)) / widen(s)).round().clamp(0.0, 255.0) as u8;
                        q[g * 64 + i] = qi;
                        values[r * IN + g * 64 + i] = f32::from(qi) * widen(s) + widen(b);
                    }
                }
                for w4 in q.chunks(4) {
                    words.extend_from_slice(
                        &u32::from_le_bytes([w4[0], w4[1], w4[2], w4[3]]).to_le_bytes(),
                    );
                }
            }
            Fixture::Affine8 {
                words,
                scales,
                biases,
                values,
            }
        }
        _ => unreachable!("five weight kinds"),
    }
}

fn upload(backend: &Arc<dyn Backend>, bytes: &[u8]) -> DeviceBuffer {
    let buf = shiso_core::alloc(backend, bytes.len() as u64).expect("allocates");
    backend.copy_h2d(buf.view(), bytes).expect("uploads");
    buf
}

fn read(backend: &Arc<dyn Backend>, buf: &DeviceBuffer) -> Vec<f32> {
    let mut bytes = vec![0u8; buf.len() as usize];
    backend
        .copy_d2h(buf.view(), &mut bytes)
        .expect("reads back");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn linear(backend: &Arc<dyn Backend>, fx: &Fixture, bias: &DeviceBuffer) -> (Linear, Vec<f32>) {
    let bias = Some(upload(backend, &read_bytes(backend, bias)));
    match fx {
        Fixture::Dense(dtype, bytes, values) => (
            Linear {
                weight: LinearWeight::Gemm {
                    dtype: *dtype,
                    buffer: upload(backend, bytes),
                    in_dim: IN,
                    out_dim: OUT,
                },
                bias,
            },
            values.clone(),
        ),
        Fixture::Q8(bytes, values) => {
            let weight = LinearWeight::Gemm {
                dtype: GemmType::Q8_0,
                buffer: upload(backend, bytes),
                in_dim: IN,
                out_dim: OUT,
            };
            (Linear { weight, bias }, values.clone())
        }
        Fixture::Affine8 {
            words,
            scales,
            biases,
            values,
        } => {
            let mut all = words.clone();
            let (w_end, s_end) = (words.len(), words.len() + scales.len());
            all.extend_from_slice(scales);
            all.extend_from_slice(biases);
            let len = all.len();
            let host: Arc<dyn shiso_core::HostBytes> = Arc::new(all);
            let ranges = AffineRanges {
                words: 0..w_end,
                scales: w_end..s_end,
                biases: s_end..len,
            };
            let w = Weight::import_affine(
                backend,
                DType::MlxAffine8Gs64,
                IN as u32,
                OUT as u32,
                host,
                ranges,
            )
            .expect("imports");
            (
                Linear {
                    weight: LinearWeight::Projected(w),
                    bias,
                },
                values.clone(),
            )
        }
    }
}

fn read_bytes(backend: &Arc<dyn Backend>, buf: &DeviceBuffer) -> Vec<u8> {
    let mut bytes = vec![0u8; buf.len() as usize];
    backend
        .copy_d2h(buf.view(), &mut bytes)
        .expect("reads back");
    bytes
}

#[expect(
    clippy::print_stdout,
    reason = "the measured error is this test's report"
)]
fn say(line: &str) {
    println!("{line}");
}

#[test]
#[ignore = "requires a Metal device"]
fn linear_matches_the_oracle_for_every_weight_type() {
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let ops = Ops::new(Arc::clone(&backend));
    let bias_values = signal(OUT, 9);
    let bias = upload(&backend, &f32_bytes(&bias_values));
    // Measured 2026-09-25 on M1 Max: worst 3.5e-5 over every type and row count;
    // the bound is three times that.
    for (kind, tol) in [
        ("f32", 1e-4f32),
        ("bf16", 1e-4),
        ("f16", 1e-4),
        ("q8_0", 1e-4),
        ("affine8", 1e-4),
    ] {
        let fx = fixture(kind);
        let (lin, w) = linear(&backend, &fx, &bias);
        assert_eq!((lin.in_dim(), lin.out_dim()), (IN, OUT));
        for rows in [1usize, 4, 37, 600] {
            let x = signal(rows * IN, 11);
            let xb = upload(&backend, &f32_bytes(&x));
            let yb = upload(&backend, &f32_bytes(&vec![f32::NAN; rows * OUT]));
            lin.forward(&ops, xb.view(), yb.view(), rows)
                .expect("the linear runs");
            backend.sync().expect("drains");
            let got = read(&backend, &yb);
            let mut worst = 0.0f32;
            for r in 0..rows {
                for c in 0..OUT {
                    let mut acc = f64::from(bias_values[c]);
                    for i in 0..IN {
                        acc += f64::from(x[r * IN + i]) * f64::from(w[c * IN + i]);
                    }
                    let (g, want) = (got[r * OUT + c], acc as f32);
                    assert!(g.is_finite(), "{kind} rows {rows} [{r}][{c}] is {g}");
                    worst = worst.max((g - want).abs() / want.abs().max(1.0));
                }
            }
            say(&format!(
                "linear {kind} rows {rows}: worst relative error {worst:.2e}"
            ));
            assert!(worst <= tol, "{kind} rows {rows}: worst {worst} over {tol}");
        }
    }
}

#[test]
#[ignore = "requires a Metal device"]
fn norms_refuse_a_zero_width_row() {
    // A zero-width row has no first value to pivot on and no mean to divide by;
    // the kernel would read one element past an empty view.
    let metal = shiso_metal::MetalBackend::new().expect("the Metal library compiles");
    let backend: Arc<dyn Backend> = metal;
    let ops = Ops::new(Arc::clone(&backend));
    let buf = upload(&backend, &f32_bytes(&[1.0; 4]));
    let empty = buf.slice(16, 0).expect("an empty view at the end");
    let desc = shiso_models::audio_ops::NormDesc {
        out: empty,
        input: empty,
        weight: Some(empty),
        bias: None,
        rows: 3,
        dim: 0,
        eps: 1e-5,
    };
    assert!(ops.layer_norm(&desc).is_err(), "layer norm accepted dim 0");
    assert!(ops.rms_norm(&desc).is_err(), "rms norm accepted dim 0");
}
