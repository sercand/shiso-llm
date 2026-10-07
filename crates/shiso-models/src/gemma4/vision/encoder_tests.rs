//! The Gemma 4 tower's graph against a plain-Rust reading of itself.
//!
//! There is no upstream shading-language source for this tower's graph, so
//! the defence here is the one the Qwen suite uses: every case compares the
//! device against the simplest possible reading of the formula, written as a
//! plain Rust `f32` loop. A disagreement points at the kernel or at its launch
//! geometry rather than at a second, equally subtle implementation. The
//! reference below is deliberately naive — left-fold sums, row-major scans —
//! and the device is held to a tolerance for exactly that reason.
//!
//! The synthetic tower runs at widths that are not the shipped checkpoint's
//! (hidden 32, four heads of eight, depth two, patch four): the encoder must
//! work from the dims it was handed, and a fixture at the shipped widths
//! would hide a literal. The real mmproj is opened by its own ignored case,
//! which checks what the loader derived rather than running the graph over it
//! — the parity gate in the server's suite is where the real picture belongs.
#![expect(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    reason = "the reference works in host floats over loop indices, and every index here is bounded by a small fixture dimension"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the reference uses the tower's own names - q, k, v for the attention planes"
)]
#![expect(
    clippy::too_many_lines,
    reason = "the host reference is a single function implementing the full tower"
)]
#![expect(clippy::print_stderr, reason = "skips print a notice")]
#![expect(
    clippy::needless_range_loop,
    reason = "indexing pooled by position formula is clearer than enumerate"
)]

use std::path::Path;
use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer, alloc};
use shiso_metal::MetalBackend;
use shiso_vision::{VisionError, VisionItem};

use super::*;

/// The clamp pair that changes nothing.
///
/// Not what the shipped checkpoints store - both of them carry real bounds on
/// every projection, around plus or minus six on the inputs and eleven on the
/// outputs - but what a projection with no scalars beside it falls back to.
const NO_CLAMPS: Clamps = Clamps {
    input_min: f32::MIN,
    input_max: f32::MAX,
    output_min: f32::MIN,
    output_max: f32::MAX,
};
use crate::gemma4::vision::loader::{
    Gemma4Block, Gemma4Dims, Gemma4Linear, Gemma4VisionWeights, bf16_bits,
};

/// A deterministic filler: a mixed-radix signal, not a random draw, so a
/// transposed stride or an off-by-one row reads visibly different values.
fn signal(len: usize, salt: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let a = (i % 13) as f32 - 6.0;
            let b = ((i / 13 + salt) % 7) as f32 - 3.0;
            (a * 0.25 + b * 0.125) / (1.0 + (i % 5) as f32 * 0.1)
        })
        .collect()
}

/// The synthetic tower's widths.
fn test_dims() -> Gemma4Dims {
    Gemma4Dims {
        depth: 2,
        hidden: 32,
        heads: 4,
        head_dim: 8,
        intermediate: 48,
        out_hidden: 16,
        patch_dim: 3 * 4 * 4,
        pos_rows: 64,
        patch: 4,
        eps: 1e-6,
        rope_base: 100.0,
    }
}

fn dev_f32(backend: &Arc<dyn Backend>, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let buffer = alloc(backend, bytes.len() as u64).expect("alloc");
    backend.copy_h2d(buffer.view(), &bytes).expect("h2d");
    buffer
}

/// Uploads as brain-float: exactly the rounding the loader performs, so the
/// host reference and the device see one number rather than two roundings.
fn dev_bf16(backend: &Arc<dyn Backend>, values: &[f32]) -> DeviceBuffer {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|v| bf16_bits(*v).to_le_bytes())
        .collect();
    let buffer = alloc(backend, bytes.len() as u64).expect("alloc");
    backend.copy_h2d(buffer.view(), &bytes).expect("h2d");
    buffer
}

/// The same weights, read back as the f32 values the kernels widen them to.
fn widen(bits: &[f32]) -> Vec<f32> {
    bits.iter()
        .map(|v| f32::from_bits(u32::from(bf16_bits(*v)) << 16))
        .collect()
}

/// One linear's host weight `[out][in]` and its clamps.
struct RefLinear {
    w: Vec<f32>,
    out_width: usize,
    in_width: usize,
    clamps: Clamps,
}

/// The whole tower, host-resident, in the order the graph reads it.
struct RefTower {
    dims: Gemma4Dims,
    patch: Vec<f32>,
    pos_x: Vec<f32>,
    pos_y: Vec<f32>,
    blocks: Vec<RefBlock>,
    projection: RefLinear,
}

struct RefBlock {
    ln1: Vec<f32>,
    ln2: Vec<f32>,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    attn_post_norm: Vec<f32>,
    ffn_post_norm: Vec<f32>,
    q: RefLinear,
    k: RefLinear,
    v: RefLinear,
    o: RefLinear,
    gate: RefLinear,
    up: RefLinear,
    down: RefLinear,
}

fn project(x: &[f32], rows: usize, l: &RefLinear) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * l.out_width];
    for r in 0..rows {
        for o in 0..l.out_width {
            let mut acc = 0.0f32;
            for i in 0..l.in_width {
                acc += x[r * l.in_width + i] * l.w[o * l.in_width + i];
            }
            out[r * l.out_width + o] = acc;
        }
    }
    out
}

fn rmsnorm(x: &[f32], w: &[f32], width: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    for (chunk, dst) in x.chunks(width).zip(out.chunks_mut(width)) {
        let ms: f32 = chunk.iter().map(|v| v * v).sum::<f32>() / width as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for i in 0..width {
            dst[i] = chunk[i] * inv * w[i];
        }
    }
    out
}

/// The output half of a projection's clamp pair, in place.
fn clamp_pair(x: &mut [f32], c: &Clamps) {
    for v in x.iter_mut() {
        *v = v.max(c.output_min).min(c.output_max);
    }
}

/// The input half, as a copy: the device clamps a projection's source into a
/// scratch stream rather than over the live one.
fn clamp_in(x: &[f32], c: &Clamps) -> Vec<f32> {
    x.iter()
        .map(|v| v.max(c.input_min).min(c.input_max))
        .collect()
}

fn gelu_tanh(x: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044_715 * x * x * x)).tanh())
}

/// The whole graph, host-side, for one item.
fn reference(t: &RefTower, pixels: &[f32], grid_h: usize, grid_w: usize) -> Vec<f32> {
    let d = t.dims;
    let rows = grid_h * grid_w;
    let h = d.hidden;
    let mut x = project(
        pixels,
        rows,
        &RefLinear {
            w: t.patch.clone(),
            out_width: h,
            in_width: d.patch_dim,
            clamps: NO_CLAMPS,
        },
    );
    for r in 0..rows {
        let (gy, gx) = (r / grid_w, r % grid_w);
        for c in 0..h {
            x[r * h + c] += t.pos_x[gx * h + c] + t.pos_y[gy * h + c];
        }
    }
    for block in &t.blocks {
        let hn = rmsnorm(&x, &block.ln1, h, d.eps);
        let mut q = project(&clamp_in(&hn, &block.q.clamps), rows, &block.q);
        let mut k = project(&clamp_in(&hn, &block.k.clamps), rows, &block.k);
        let mut v = project(&clamp_in(&hn, &block.v.clamps), rows, &block.v);
        for plane in [&mut q, &mut k, &mut v] {
            clamp_pair(plane, &block.q.clamps);
        }
        // The fused rotation normalizes each plane per head — q and k
        // weighted, v not — and rotates the two halves of each head by the
        // patch's column and row.
        let half = d.head_dim / 2;
        let quarter = half / 2;
        let dim = h;
        for n in 0..rows {
            let (col, row) = (n % grid_w, n / grid_w);
            for head in 0..d.heads {
                let off = n * dim + head * d.head_dim;
                for (plane, w) in [
                    (&mut q[..], Some(&block.q_norm[..])),
                    (&mut k[..], Some(&block.k_norm[..])),
                    (&mut v[..], None),
                ] {
                    let ms: f32 = (0..d.head_dim)
                        .map(|i| plane[off + i] * plane[off + i])
                        .sum::<f32>()
                        / d.head_dim as f32;
                    let inv = 1.0 / (ms + d.eps).sqrt();
                    for i in 0..d.head_dim {
                        let weight = w.map_or(1.0, |w| w[i]);
                        plane[off + i] *= inv * weight;
                    }
                }
                for (pos, shift) in [(col, 0usize), (row, half)] {
                    for j in 0..quarter {
                        let theta = pos as f32 * d.rope_base.powf(-2.0 * j as f32 / half as f32);
                        let (s, c) = (theta.sin(), theta.cos());
                        for plane in [&mut q, &mut k] {
                            let (x0, x1) =
                                (plane[off + shift + j], plane[off + shift + j + quarter]);
                            plane[off + shift + j] = x0 * c - x1 * s;
                            plane[off + shift + j + quarter] = x0 * s + x1 * c;
                        }
                    }
                }
            }
        }
        // Bidirectional attention, scale one, per temporal group (one here).
        let mut attn = vec![0.0f32; rows * h];
        for head in 0..d.heads {
            let hd = d.head_dim;
            for n in 0..rows {
                let mut scores: Vec<f32> = (0..rows)
                    .map(|p| {
                        (0..hd)
                            .map(|e| q[n * h + head * hd + e] * k[p * h + head * hd + e])
                            .sum::<f32>()
                    })
                    .collect();
                let m = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let mut den = 0.0f32;
                for s in &mut scores {
                    *s = (*s - m).exp();
                    den += *s;
                }
                for p in 0..rows {
                    let w = scores[p] / den;
                    for e in 0..hd {
                        attn[n * h + head * hd + e] += w * v[p * h + head * hd + e];
                    }
                }
            }
        }
        let mut o = project(&clamp_in(&attn, &block.o.clamps), rows, &block.o);
        clamp_pair(&mut o, &block.o.clamps);
        let post = rmsnorm(&o, &block.attn_post_norm, h, d.eps);
        for i in 0..x.len() {
            x[i] += post[i];
        }
        let hn = rmsnorm(&x, &block.ln2, h, d.eps);
        let mut gate = project(&clamp_in(&hn, &block.gate.clamps), rows, &block.gate);
        let mut up = project(&clamp_in(&hn, &block.up.clamps), rows, &block.up);
        clamp_pair(&mut gate, &block.gate.clamps);
        clamp_pair(&mut up, &block.up.clamps);
        let mid: Vec<f32> = gate
            .iter()
            .zip(&up)
            .map(|(g, u)| gelu_tanh(*g) * *u)
            .collect();
        let mut down = project(&clamp_in(&mid, &block.down.clamps), rows, &block.down);
        clamp_pair(&mut down, &block.down.clamps);
        let post = rmsnorm(&down, &block.ffn_post_norm, h, d.eps);
        for i in 0..x.len() {
            x[i] += post[i];
        }
    }
    // The fold, the unweighted norm, the projection.
    let mut pooled = vec![0.0f32; rows / 9 * h];
    let out_w = grid_w / 3;
    let scale = (h as f32).sqrt();
    for i in 0..pooled.len() {
        let orow = i / h;
        let c = i % h;
        let (oy, ox) = (orow / out_w, orow % out_w);
        let mut sum = 0.0f32;
        for dy in 0..3 {
            for dx in 0..3 {
                let sy = oy * 3 + dy;
                let sx = ox * 3 + dx;
                sum += x[(sy * grid_w + sx) * h + c];
            }
        }
        pooled[i] = sum / 9.0 * scale;
    }
    let ones = vec![1.0f32; h];
    let normed = rmsnorm(&pooled, &ones, h, d.eps);
    project(&normed, rows / 9, &t.projection)
}

/// Builds the synthetic tower on the device and its host twin alongside.
///
/// The device copy holds the loader's exact formats: brain-float linears, f32
/// norms. The host twin reads back the SAME rounded values, so the only
/// difference between the two runs is the arithmetic the device performs.
fn build(backend: &Arc<dyn Backend>, clamps: Clamps) -> (Gemma4VisionWeights, RefTower) {
    let d = test_dims();
    let h = d.hidden;
    let patch = signal(h * d.patch_dim, 1);
    let pos_x = signal(d.pos_rows * h, 2);
    let pos_y = signal(d.pos_rows * h, 3);
    let mut blocks = Vec::new();
    let mut ref_blocks = Vec::new();
    let salt = &mut 10usize;
    for _ in 0..d.depth {
        let lin = |out: usize, inp: usize, s: usize| {
            let w = signal(out * inp, s);
            (dev_bf16(backend, &w), widen(&w))
        };
        let (q_d, q_h) = lin(h, h, {
            *salt += 1;
            *salt
        });
        let (k_d, k_h) = lin(h, h, {
            *salt += 1;
            *salt
        });
        let (v_d, v_h) = lin(h, h, {
            *salt += 1;
            *salt
        });
        let (o_d, o_h) = lin(h, h, {
            *salt += 1;
            *salt
        });
        let (g_d, g_h) = lin(d.intermediate, h, {
            *salt += 1;
            *salt
        });
        let (u_d, u_h) = lin(d.intermediate, h, {
            *salt += 1;
            *salt
        });
        let (dn_d, dn_h) = lin(h, d.intermediate, {
            *salt += 1;
            *salt
        });
        let ln1 = signal(h, {
            *salt += 1;
            *salt
        });
        let ln2 = signal(h, {
            *salt += 1;
            *salt
        });
        let qn = signal(d.head_dim, {
            *salt += 1;
            *salt
        });
        let kn = signal(d.head_dim, {
            *salt += 1;
            *salt
        });
        let apn = signal(h, {
            *salt += 1;
            *salt
        });
        let fpn = signal(h, {
            *salt += 1;
            *salt
        });
        let mk = |w: Vec<f32>, out: usize, inp: usize| RefLinear {
            w,
            out_width: out,
            in_width: inp,
            clamps,
        };
        blocks.push(Gemma4Block {
            ln1: dev_f32(backend, &ln1),
            attn_q: Gemma4Linear {
                dtype: crate::audio_ops::GemmType::Bf16,
                weight: q_d,
                clamps,
            },
            attn_k: Gemma4Linear {
                dtype: crate::audio_ops::GemmType::Bf16,
                weight: k_d,
                clamps,
            },
            attn_v: Gemma4Linear {
                dtype: crate::audio_ops::GemmType::Bf16,
                weight: v_d,
                clamps,
            },
            q_norm: dev_f32(backend, &qn),
            k_norm: dev_f32(backend, &kn),
            attn_out: Gemma4Linear {
                dtype: crate::audio_ops::GemmType::Bf16,
                weight: o_d,
                clamps,
            },
            attn_post_norm: dev_f32(backend, &apn),
            ln2: dev_f32(backend, &ln2),
            ffn_gate: Gemma4Linear {
                dtype: crate::audio_ops::GemmType::Bf16,
                weight: g_d,
                clamps,
            },
            ffn_up: Gemma4Linear {
                dtype: crate::audio_ops::GemmType::Bf16,
                weight: u_d,
                clamps,
            },
            ffn_down: Gemma4Linear {
                dtype: crate::audio_ops::GemmType::Bf16,
                weight: dn_d,
                clamps,
            },
            ffn_post_norm: dev_f32(backend, &fpn),
        });
        ref_blocks.push(RefBlock {
            ln1,
            ln2,
            q_norm: qn,
            k_norm: kn,
            attn_post_norm: apn,
            ffn_post_norm: fpn,
            q: mk(q_h, h, h),
            k: mk(k_h, h, h),
            v: mk(v_h, h, h),
            o: mk(o_h, h, h),
            gate: mk(g_h, d.intermediate, h),
            up: mk(u_h, d.intermediate, h),
            down: mk(dn_h, h, d.intermediate),
        });
    }
    let (proj_d, proj_h) = {
        let w = signal(d.out_hidden * h, 99);
        (dev_bf16(backend, &w), widen(&w))
    };
    let weights = Gemma4VisionWeights {
        dims: d,
        patch: dev_bf16(backend, &patch),
        position_x: pos_x.clone(),
        position_y: pos_y.clone(),
        blocks,
        projection: Gemma4Linear {
            dtype: crate::audio_ops::GemmType::Bf16,
            weight: proj_d,
            clamps: NO_CLAMPS,
        },
        ones: dev_f32(backend, &vec![1.0f32; h]),
    };
    let reference = RefTower {
        dims: d,
        patch: widen(&patch),
        pos_x,
        pos_y,
        blocks: ref_blocks,
        projection: RefLinear {
            w: proj_h,
            out_width: d.out_hidden,
            in_width: d.hidden,
            clamps: NO_CLAMPS,
        },
    };
    (weights, reference)
}

fn metal() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the device case");
        return None;
    }
    Some(MetalBackend::new().expect("backend"))
}

fn close(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = (0.0f32, 0usize);
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = 2.0e-3 * w.abs().max(1.0);
        let err = (g - w).abs();
        if err > worst.0 {
            worst = (err, i);
        }
        assert!(err <= bound, "{what}: element {i} is {g}, expected {w}");
    }
    eprintln!(
        "note: {what}: worst error {} at {worst_elem}",
        worst.0,
        worst_elem = worst.1
    );
}

/// The whole graph, device against host reference, at synthetic widths.
#[test]
#[ignore = "requires a Metal device"]
fn tower_matches_the_host_reference() {
    let Some(backend) = metal() else { return };
    let (weights, rtower) = build(&backend, NO_CLAMPS);
    let (grid_h, grid_w) = (6usize, 9usize);
    let item = VisionItem::new(
        signal(grid_h * grid_w * test_dims().patch_dim, 7).into_boxed_slice(),
        test_dims().patch_dim,
        1,
        grid_h,
        grid_w,
    )
    .expect("item");
    let out = encode_item(&backend, &weights, &item).expect("encode");
    backend.sync().expect("sync");
    let bytes = {
        let mut v = vec![0u8; out.len() as usize];
        backend.copy_d2h(out.view(), &mut v).expect("d2h");
        v
    };
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    let want = reference(&rtower, item.as_flat(), grid_h, grid_w);
    close(&got, &want, "tower output");
}

/// A non-identity clamp pair must move the answer, and the device must move
/// it the same way the reference does: the pair is the graph's, not a no-op
/// that happens to compile.
#[test]
#[ignore = "requires a Metal device"]
fn clamped_tower_matches_the_host_reference() {
    let Some(backend) = metal() else { return };
    let biting = Clamps {
        input_min: -0.25,
        input_max: 0.25,
        output_min: -1.0,
        output_max: 1.0,
    };
    let (weights, rtower) = build(&backend, biting);
    let (grid_h, grid_w) = (3usize, 6usize);
    let item = VisionItem::new(
        signal(grid_h * grid_w * test_dims().patch_dim, 8).into_boxed_slice(),
        test_dims().patch_dim,
        1,
        grid_h,
        grid_w,
    )
    .expect("item");
    let out = encode_item(&backend, &weights, &item).expect("encode");
    backend.sync().expect("sync");
    let mut bytes = vec![0u8; out.len() as usize];
    backend.copy_d2h(out.view(), &mut bytes).expect("d2h");
    let got: Vec<f32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|w| f32::from_le_bytes(*w))
        .collect();
    let want = reference(&rtower, item.as_flat(), grid_h, grid_w);
    close(&got, &want, "clamped tower output");
    assert!(
        want.iter().any(|v: &f32| v.abs() > 0.0),
        "the clamped fixture collapsed to zeros",
    );
}

/// A grid the fold cannot cut is refused before anything is allocated.
#[test]
#[ignore = "requires a Metal device"]
fn a_grid_that_is_not_a_multiple_of_the_pool_is_refused() {
    let Some(backend) = metal() else { return };
    let (weights, _) = build(&backend, NO_CLAMPS);
    let d = test_dims();
    let item = VisionItem::new(
        signal(6 * 7 * d.patch_dim, 9).into_boxed_slice(),
        d.patch_dim,
        1,
        6,
        7,
    )
    .expect("item");
    match encode_item(&backend, &weights, &item) {
        Err(VisionError::ItemGridNotMergeable {
            grid_h: 6,
            grid_w: 7,
            merge: 3,
        }) => {}
        other => panic!(
            "expected the merge refusal, got {other:?}",
            other = other.map(|_| ())
        ),
    }
}

/// The shipped mmproj opens, and the loader derived the widths the reference
/// states: sixteen blocks of 768, twelve heads of 64, a 3072-wide feed-forward
/// and a 2560-wide projection, over a 16-pixel patch and a 10240-row table.
#[test]
#[ignore = "opens the real checkpoint"]
fn the_shipped_mmproj_loads() {
    let Some(backend) = metal() else { return };
    let Ok(mmproj) = std::env::var("SHISO_TEST_MODEL_GEMMA4_MMPROJ") else {
        eprintln!("note: SHISO_TEST_MODEL_GEMMA4_MMPROJ is not set, skipping");
        return;
    };
    let path = Path::new(&mmproj);
    if !path.exists() {
        eprintln!("note: the shipped mmproj is not on this machine, skipping");
        return;
    }
    let tower = Gemma4Tower::open(&backend, path).expect("open");
    let d = tower.weights.dims();
    assert_eq!((d.depth, d.hidden, d.heads, d.head_dim), (16, 768, 12, 64));
    assert_eq!(
        (d.intermediate, d.out_hidden, d.patch_dim, d.pos_rows),
        (3072, 2560, 768, 10240)
    );
    assert_eq!(tower.out_hidden(), 2560);
    assert_eq!(tower.placement(), MediaPlacement::GEMMA4);
    assert!(
        tower.device_bytes() > 280 * 1024 * 1024,
        "device bytes: {}",
        tower.device_bytes()
    );
}

/// The two containers' towers, encoding the same picture.
///
/// The same model shipped twice. Every tensor the tower reads is byte-identical
/// between the mmproj GGUF and the checkpoint directory, checked one at a time,
/// with one exception: the directory quantizes the 768-to-2560 projector to
/// four bits and the file carries it at brain-float. So the rows are expected to
/// differ, by what that one quantization costs and by nothing else, and the
/// tolerance below is what that measures.
///
/// It is the only case that compares the two load paths, which is more than a
/// convenience: the loaders read the same numbers through different code, and a
/// name looked up the wrong way in one of them produces a tower that loads
/// cleanly, runs, and answers about a picture it never saw. That is what this
/// case found - the file's path was reading its clip scalars off the weight's
/// name rather than the projection's, finding nothing, and taking the
/// absent-scalar path, which is the identity. Unclamped against clamped
/// measured 3.78 here, which is why the tolerance is not a round number chosen
/// in advance: it is the measurement, and widening it is how that defect
/// survived.
#[test]
#[ignore = "opens the real checkpoints"]
fn the_mlx_directory_and_the_mmproj_agree() {
    let Some(backend) = metal() else { return };
    let (Ok(mmproj), Ok(mlx)) = (
        std::env::var("SHISO_TEST_MODEL_GEMMA4_MMPROJ"),
        std::env::var("SHISO_TEST_MODEL_GEMMA4_MLX"),
    ) else {
        eprintln!(
            "note: SHISO_TEST_MODEL_GEMMA4_MMPROJ and SHISO_TEST_MODEL_GEMMA4_MLX are not both \
             set, skipping"
        );
        return;
    };
    let mmproj_path = Path::new(&mmproj);
    let mlx_path = Path::new(&mlx);
    if !mmproj_path.is_file() || !mlx_path.is_dir() {
        eprintln!("note: missing real checkpoints, skipping");
        return;
    }

    let mmproj_tower = Gemma4Tower::open(&backend, mmproj_path).expect("open mmproj");
    let mlx_tower = Gemma4Tower::open(&backend, mlx_path).expect("open mlx");
    let png_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/prompts/gemma4-vision-quadrants.png");
    let png_bytes = std::fs::read(&png_path).expect("read png");
    let spec = mlx_tower.preprocess();
    let item =
        shiso_vision::preprocess_image(&png_bytes, spec, shiso_vision::ImageLimits::default())
            .expect("preprocess");

    let mmproj_encoded = mmproj_tower
        .encode(&backend, std::slice::from_ref(&item))
        .expect("encode mmproj");
    let mlx_encoded = mlx_tower.encode(&backend, &[item]).expect("encode mlx");

    assert_eq!(mmproj_encoded.rows.len(), mlx_encoded.rows.len());

    let mut mm_bytes = vec![0u8; usize::try_from(mmproj_encoded.rows.len()).expect("valid len")];
    let mut mlx_bytes = vec![0u8; usize::try_from(mlx_encoded.rows.len()).expect("valid len")];
    backend
        .copy_d2h(mmproj_encoded.rows.view(), &mut mm_bytes)
        .expect("copy mm");
    backend
        .copy_d2h(mlx_encoded.rows.view(), &mut mlx_bytes)
        .expect("copy mlx");

    let mm_f32 = bytemuck::cast_slice::<u8, f32>(&mm_bytes);
    let mlx_f32 = bytemuck::cast_slice::<u8, f32>(&mlx_bytes);

    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut worst_m = 0.0f32;
    let mut worst_x = 0.0f32;
    for (&m, &x) in mm_f32.iter().zip(mlx_f32) {
        let diff = (m - x).abs();
        let rel = if m.abs() > 1e-4 { diff / m.abs() } else { diff };
        if diff > max_abs {
            max_abs = diff;
            worst_m = m;
            worst_x = x;
        }
        if rel > max_rel {
            max_rel = rel;
        }
    }
    eprintln!("measured max |Δ|={max_abs:.4} (m={worst_m}, x={worst_x}), max_rel={max_rel:.4}");
    // Measured 0.40 on the shipped pair. Six tenths leaves room for another
    // picture to land harder on the four-bit projector and still fails by an
    // order of magnitude on a tower that is reading something else.
    let tol = 0.6f32;
    assert!(max_abs <= tol, "max |Δ| {max_abs} exceeded tolerance {tol}");
}

/// The host reference alone, on a tower with no blocks: the fold, the scale,
/// the unweighted norm and the projection, against numbers written by hand.
///
/// A blockless tower is not a shape the device ever runs, but it is the only
/// way to point at the reference's tail without also asserting the whole
/// graph twice. If the fold read the grid in the wrong order, or the scale
/// were the merge kernel's rather than the pool's, or the final norm weighted
/// itself, the four numbers below would not be these.
#[test]
fn the_reference_folds_a_blockless_tower_as_designed() {
    let dims = Gemma4Dims {
        depth: 0,
        hidden: 4,
        heads: 1,
        head_dim: 4,
        intermediate: 4,
        out_hidden: 4,
        patch_dim: 12,
        pos_rows: 9,
        patch: 2,
        eps: 1e-6,
        rope_base: 100.0,
    };
    // The patch projection reads channel `c` of each row out of input `c`;
    // the projection is the identity. Everything else is zero.
    let mut patch = vec![0.0f32; 4 * 12];
    for o in 0..4 {
        patch[o * 12 + o] = 1.0;
    }
    let mut projection = vec![0.0f32; 4 * 4];
    for o in 0..4 {
        projection[o * 4 + o] = 1.0;
    }
    let tower = RefTower {
        dims,
        patch,
        pos_x: vec![0.0; 9 * 4],
        pos_y: vec![0.0; 9 * 4],
        blocks: Vec::new(),
        projection: RefLinear {
            w: projection,
            out_width: 4,
            in_width: 4,
            clamps: NO_CLAMPS,
        },
    };
    // Row `r`, channel `c` of the patch stream is `r + 100 c`: the row mean
    // is 4, and the channels stay a hundred apart, so a fold that mixed them
    // cannot produce the expected numbers.
    let mut pixels = vec![0.0f32; 9 * 12];
    for r in 0..9 {
        for d in 0..12 {
            pixels[r * 12 + d] = r as f32 + 100.0 * d as f32;
        }
    }
    let got = reference(&tower, &pixels, 3, 3);
    assert_eq!(got.len(), 4);
    let pooled: Vec<f32> = (0..4).map(|c| (4.0 + 100.0 * c as f32) * 2.0).collect();
    let ms = pooled.iter().map(|v| v * v).sum::<f32>() / 4.0;
    let inv = 1.0 / (ms + dims.eps).sqrt();
    for c in 0..4 {
        assert!(
            (got[c] - pooled[c] * inv).abs() < 1e-4,
            "channel {c}: got {}, want {}",
            got[c],
            pooled[c] * inv
        );
    }
}
