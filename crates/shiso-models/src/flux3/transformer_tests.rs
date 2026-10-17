//! The transformer against the reference implementation's own module.
//!
//! `scripts/reference/flux3_dit_reference.py` runs the reference `DiT` in
//! float32 at a tiny size (hidden 256, two heads, text at its true 20480
//! width) with seeded weights and writes the weights, inputs and velocities
//! under `test-vectors/references/flux3/dit-tiny/`. These cases replay them
//! on a Metal device with and without a reference image, at the full depth
//! and at one block per stack. Without the fixture or a device a case says so
//! on stderr and returns, which looks like a pass: check stderr before
//! reporting one.
//!
//! The per-mode modulation segments are also graded against an `f64` block
//! written here from the block's formulas, on rows split between two
//! different modulations.
#![expect(
    clippy::cast_precision_loss,
    reason = "row and column indices converted for oracle math are small, exact in f64"
)]
#![expect(
    clippy::cast_possible_truncation,
    reason = "the f64 oracle sums are narrowed to the f32 the device computes in"
)]
#![expect(
    clippy::many_single_char_names,
    reason = "the oracle is written in the formula's own letters"
)]

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_core::{Backend, DeviceBuffer};

use super::config::Flux3Params;
use super::transformer::{DiT, DiTInput, ModVec, Segment, Stream, time_embedding};
use crate::diffusion::ops::Ops;
use crate::diffusion::weights::StStore;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/flux3/dit-tiny")
}

#[expect(
    clippy::print_stderr,
    reason = "a skipped reference case must say why on stderr"
)]
fn skip(reason: &str) {
    eprintln!("SKIP flux3 transformer reference case: {reason}");
}

fn read_f32(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("reading {}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[expect(
    clippy::print_stderr,
    reason = "the device cases are opt-in, so the reason they did not run belongs on stderr"
)]
fn device() -> Option<Arc<dyn Backend>> {
    if !shiso_metal::metal_available() {
        eprintln!("note: no Metal device on this machine, skipping the transformer device case");
        return None;
    }
    let metal = shiso_metal::MetalBackend::new().expect("the embedded kernel library compiles");
    let backend: Arc<dyn Backend> = metal;
    Some(backend)
}

/// Asserts `|got - want| <= tol * max(|want|, rms(want))` element by element.
fn close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let rms =
        (want.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>() / want.len() as f64).sqrt() as f32;
    let mut worst = 0f32;
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let bound = tol * w.abs().max(rms);
        let error = (g - w).abs();
        worst = worst.max(error / w.abs().max(rms));
        assert!(
            error <= bound,
            "{what}[{i}]: device {g} vs reference {w} (bound {bound})"
        );
    }
    #[expect(
        clippy::print_stderr,
        reason = "the measured agreement is the case's report"
    )]
    {
        eprintln!("{what}: worst error {worst:.3e} of max(|w|, rms)");
    }
}

/// The tiny model's hidden width.
const H: usize = 256;

fn params(depth_mode: usize, depth_single: usize) -> Flux3Params {
    Flux3Params {
        hidden: 256,
        heads: 2,
        depth_mode,
        depth_single,
        mlp_hidden: 768,
        txt_in: 20_480,
        img_in: 128,
    }
}

/// The fixture's inputs, uploaded.
struct Inputs {
    image: DeviceBuffer,
    cond: DeviceBuffer,
    text: DeviceBuffer,
    image_ids: DeviceBuffer,
    cond_ids: DeviceBuffer,
    text_ids: DeviceBuffer,
    t: f32,
    t_ref: f32,
    rows: (usize, usize, usize),
}

fn load_inputs(ops: &Ops, dir: &Path) -> Inputs {
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("meta.json")).expect("meta.json"))
            .expect("meta parses");
    let rows = |key: &str| usize::try_from(meta[key].as_u64().expect(key)).expect("fits");
    let up = |name: &str| {
        ops.upload_f32("fixture", &read_f32(&dir.join(name)))
            .expect("uploads")
    };
    #[expect(
        clippy::cast_possible_truncation,
        reason = "times are f32 in the model"
    )]
    Inputs {
        image: up("image.f32"),
        cond: up("cond.f32"),
        text: up("text.f32"),
        image_ids: up("image_ids.f32"),
        cond_ids: up("cond_ids.f32"),
        text_ids: up("text_ids.f32"),
        t: meta["t"].as_f64().expect("t") as f32,
        t_ref: meta["t_ref"].as_f64().expect("t_ref") as f32,
        rows: (rows("image_rows"), rows("cond_rows"), rows("text_rows")),
    }
}

fn forward(ops: &Ops, dit: &DiT, inputs: &Inputs, with_cond: bool) -> Vec<f32> {
    let (image_rows, cond_rows, text_rows) = inputs.rows;
    let out = dit
        .forward(
            ops,
            &DiTInput {
                image: inputs.image.view(),
                image_cond: with_cond.then(|| inputs.cond.view()),
                text: inputs.text.view(),
                image_rows,
                cond_rows: if with_cond { cond_rows } else { 0 },
                text_rows,
                t: inputs.t,
                t_ref: inputs.t_ref,
                image_ids: inputs.image_ids.view(),
                cond_ids: with_cond.then(|| inputs.cond_ids.view()),
                text_ids: inputs.text_ids.view(),
            },
        )
        .expect("the forward pass runs");
    assert_eq!(
        out.view().len(),
        (image_rows * 128 * 4) as u64,
        "the output holds the image rows and nothing else"
    );
    ops.download("velocity", out.view(), image_rows * 128)
        .expect("downloads")
}

/// A device, the fixture and the weight store, or a stderr note why not.
fn setup() -> Option<(Arc<dyn Backend>, PathBuf, StStore)> {
    let dir = fixture_dir();
    if !dir.join("meta.json").is_file() {
        skip(&format!(
            "{} is absent; run scripts/reference/flux3_dit_reference.py",
            dir.display()
        ));
        return None;
    }
    let backend = device()?;
    let store = StStore::open(&dir.join("weights.safetensors")).expect("the fixture weights open");
    Some((backend, dir, store))
}

fn replay(depths: (usize, usize), tag: &str) {
    let Some((backend, dir, store)) = setup() else {
        return;
    };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let dit = DiT::load(&store, &ops, &params(depths.0, depths.1)).expect("loads");
    let inputs = load_inputs(&ops, &dir);
    for (variant, with_cond) in [("cond", true), ("nocond", false)] {
        let want = read_f32(&dir.join(format!("out_{tag}_{variant}.f32")));
        let got = forward(&ops, &dit, &inputs, with_cond);
        close(&got, &want, 5e-4, &format!("{tag} {variant}"));
    }
}

#[test]
#[ignore = "requires a Metal device and test-vectors/references/flux3/dit-tiny"]
fn block_matches_reference_on_tiny_weights() {
    replay((1, 1), "small");
}

#[test]
#[ignore = "requires a Metal device and test-vectors/references/flux3/dit-tiny"]
fn forward_matches_reference_through_the_tail() {
    replay((2, 3), "full");
}

#[test]
#[ignore = "requires a Metal device and test-vectors/references/flux3/dit-tiny"]
fn forward_output_is_image_rows_only() {
    let Some((backend, dir, store)) = setup() else {
        return;
    };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let dit = DiT::load(&store, &ops, &params(1, 1)).expect("loads");
    let inputs = load_inputs(&ops, &dir);
    // `forward` asserts the byte length; with references the stream is longer
    // than the image rows, and the answer is still exactly the image rows.
    let with = forward(&ops, &dit, &inputs, true);
    let without = forward(&ops, &dit, &inputs, false);
    assert_eq!(with.len(), inputs.rows.0 * 128);
    assert_eq!(without.len(), inputs.rows.0 * 128);
    assert!(
        with.iter().zip(&without).any(|(a, b)| (a - b).abs() > 1e-4),
        "the references must change the answer, or the cases above prove nothing"
    );

    // Rows and views that disagree are refused, not run.
    let (image_rows, cond_rows, text_rows) = inputs.rows;
    let refused = dit.forward(
        &ops,
        &DiTInput {
            image: inputs.image.view(),
            image_cond: None,
            text: inputs.text.view(),
            image_rows,
            cond_rows,
            text_rows,
            t: inputs.t,
            t_ref: inputs.t_ref,
            image_ids: inputs.image_ids.view(),
            cond_ids: None,
            text_ids: inputs.text_ids.view(),
        },
    );
    assert!(refused.is_err(), "reference rows without reference tokens");
}

/// Overwrites every `image_cond` tensor of `source` with a constant, so a
/// forward pass that read any of them would change its answer.
fn poison_image_cond(source: &Path, target: &Path) {
    let mut bytes = std::fs::read(source).expect("reads the weights");
    let header_len =
        usize::try_from(u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes"))).expect("fits");
    let header: serde_json::Value =
        serde_json::from_slice(&bytes[8..8 + header_len]).expect("header parses");
    let payload = 8 + header_len;
    let mut poisoned = 0usize;
    for (name, tensor) in header.as_object().expect("object") {
        if !name.contains("image_cond") {
            continue;
        }
        let start = usize::try_from(tensor["data_offsets"][0].as_u64().expect("start")).expect("u");
        let end = usize::try_from(tensor["data_offsets"][1].as_u64().expect("end")).expect("u");
        for chunk in bytes[payload + start..payload + end].chunks_exact_mut(4) {
            chunk.copy_from_slice(&0.37f32.to_le_bytes());
        }
        poisoned += 1;
    }
    assert!(poisoned > 0, "no image_cond tensor to poison");
    std::fs::write(target, bytes).expect("writes the poisoned copy");
}

#[test]
#[ignore = "requires a Metal device and test-vectors/references/flux3/dit-tiny"]
fn empty_image_cond_runs_without_phantom_leak() {
    let Some((backend, dir, store)) = setup() else {
        return;
    };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let inputs = load_inputs(&ops, &dir);
    let dit = DiT::load(&store, &ops, &params(2, 3)).expect("loads");
    let clean = forward(&ops, &dit, &inputs, false);

    let scratch = tempfile::tempdir().expect("tempdir");
    let target = scratch.path().join("weights.safetensors");
    poison_image_cond(&dir.join("weights.safetensors"), &target);
    let poisoned_store = StStore::open(&target).expect("opens");
    let poisoned = DiT::load(&poisoned_store, &ops, &params(2, 3)).expect("loads");
    let after = forward(&ops, &poisoned, &inputs, false);
    assert_eq!(
        clean.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "without references no image_cond weight may reach the answer"
    );
    // And with them the poison is felt, so the comparison above could fail.
    let with_clean = forward(&ops, &dit, &inputs, true);
    let with_poison = forward(&ops, &poisoned, &inputs, true);
    assert!(
        with_clean
            .iter()
            .zip(&with_poison)
            .any(|(a, b)| (a - b).abs() > 1e-3),
        "the poisoned weights must matter when references are present"
    );
}

// --- the time embedding ----------------------------------------------------

#[test]
fn time_embedding_matches_the_reference_in_bf16() {
    let path = fixture_dir().join("time_embedding.json");
    if !path.is_file() {
        skip(&format!("{} is absent", path.display()));
        return;
    }
    let reference: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&path).expect("reads")).expect("parses");
    for (key, values) in reference.as_object().expect("object") {
        let t: f32 = key.parse().expect("a time");
        let want: Vec<f32> = values
            .as_array()
            .expect("array")
            .iter()
            .map(|v| v.as_f64().expect("number") as f32)
            .collect();
        let got = time_embedding(t);
        assert_eq!(got.len(), 256);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            // One brain-float step (2^-8 relative) is the most the two cos and
            // sin implementations may disagree by after the shared rounding.
            assert!(
                (g - w).abs() <= 0.004 * w.abs().max(1e-3),
                "t={t} element {i}: {g} vs reference {w}"
            );
        }
    }
}

#[test]
fn time_embedding_at_zero_is_cos_one_sin_zero() {
    let e = time_embedding(0.0);
    assert!(e[..128].iter().all(|&v| v.to_bits() == 1.0f32.to_bits()));
    assert!(e[128..].iter().all(|&v| v.to_bits() == 0.0f32.to_bits()));
}

#[test]
fn time_embedding_rounds_the_time_to_brain_float_first() {
    // 0.8731 is not a brain float; the reference's fill_ rounds it to 0.8750
    // (then 1000 * t to 875), so the two embeddings must agree exactly.
    assert_eq!(
        time_embedding(0.8731)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>(),
        time_embedding(0.875)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    );
    assert_ne!(
        time_embedding(0.8731)[130].to_bits(),
        time_embedding(0.6)[130].to_bits()
    );
}

// --- per-mode modulation segments -------------------------------------------

/// `(out, in)` row-major weight `name` of the store, as f64.
fn weight(store: &StStore, name: &str) -> Vec<f64> {
    store
        .f32_vec(name)
        .expect("reads the tensor")
        .into_iter()
        .map(f64::from)
        .collect()
}

fn matmul(x: &[f64], rows: usize, w: &[f64], out: usize, inner: usize) -> Vec<f64> {
    let mut y = vec![0.0; rows * out];
    for r in 0..rows {
        for o in 0..out {
            y[r * out + o] = (0..inner)
                .map(|i| x[r * inner + i] * w[o * inner + i])
                .sum();
        }
    }
    y
}

#[expect(
    clippy::too_many_lines,
    reason = "the oracle is one straight-line block, kept whole so it reads against the formulas"
)]
/// One block of `content_mode_blocks.image.0` on `x` with row `r` modulated
/// by `mods[segment_of(r)]` (shift, scale, gate), in `f64`.
fn host_block(
    store: &StStore,
    x: &[f64],
    ids: &[[f32; 4]],
    segments: &[(std::ops::Range<usize>, [Vec<f64>; 3])],
) -> Vec<f64> {
    const HEADS: usize = 2;
    const HD: usize = 128;
    const MLP: usize = 768;
    let rows = x.len() / H;
    let name = |part: &str| format!("content_mode_blocks.image.0.{part}");
    let segment_of = |r: usize| {
        &segments
            .iter()
            .find(|(range, _)| range.contains(&r))
            .expect("every row is in a segment")
            .1
    };
    let mut xn = vec![0.0; rows * H];
    for r in 0..rows {
        let row = &x[r * H..(r + 1) * H];
        let mean = row.iter().sum::<f64>() / H as f64;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / H as f64;
        let inv = 1.0 / (var + 1e-6).sqrt();
        let [shift, scale, _] = segment_of(r);
        for c in 0..H {
            xn[r * H + c] = (1.0 + scale[c]) * (row[c] - mean) * inv + shift[c];
        }
    }
    let project = |part: &str| {
        matmul(
            &xn,
            rows,
            &weight(store, &name(&format!("{part}.weight"))),
            H,
            H,
        )
    };
    let (mut q, mut k, v) = (project("q_proj"), project("k_proj"), project("v_proj"));
    for (tensor, scale_name) in [
        (&mut q, "norm.query_norm.scale"),
        (&mut k, "norm.key_norm.scale"),
    ] {
        let scale = weight(store, &name(scale_name));
        for r in 0..rows {
            for h in 0..HEADS {
                let head = &mut tensor[r * H + h * HD..r * H + (h + 1) * HD];
                let inv = 1.0 / (head.iter().map(|v| v * v).sum::<f64>() / HD as f64 + 1e-6).sqrt();
                for (c, value) in head.iter_mut().enumerate() {
                    *value *= inv * scale[c];
                }
                // Interleaved four-axis rotation, 32 dims per axis.
                for (axis, &position) in ids[r].iter().enumerate() {
                    for j in 0..16 {
                        let angle = f64::from(position) * 10_000f64.powf(-(j as f64) / 16.0);
                        let (s, c) = angle.sin_cos();
                        let i0 = axis * 32 + 2 * j;
                        let (x0, x1) = (head[i0], head[i0 + 1]);
                        head[i0] = c * x0 - s * x1;
                        head[i0 + 1] = s * x0 + c * x1;
                    }
                }
            }
        }
    }
    let mut attn = vec![0.0; rows * H];
    for h in 0..HEADS {
        for r in 0..rows {
            let scores: Vec<f64> = (0..rows)
                .map(|s| {
                    (0..HD)
                        .map(|d| q[r * H + h * HD + d] * k[s * H + h * HD + d])
                        .sum::<f64>()
                        / (HD as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for d in 0..HD {
                attn[r * H + h * HD + d] = exps
                    .iter()
                    .enumerate()
                    .map(|(s, p)| p * v[s * H + h * HD + d])
                    .sum::<f64>()
                    / sum;
            }
        }
    }
    let gate_up = matmul(
        &xn,
        rows,
        &weight(store, &name("mlp_in.weight")),
        2 * MLP,
        H,
    );
    let mut act = vec![0.0; rows * MLP];
    for r in 0..rows {
        for i in 0..MLP {
            let g = gate_up[r * 2 * MLP + i];
            act[r * MLP + i] = g / (1.0 + (-g).exp()) * gate_up[r * 2 * MLP + MLP + i];
        }
    }
    let attn_out = matmul(&attn, rows, &weight(store, &name("attn_out.weight")), H, H);
    let mlp_out = matmul(&act, rows, &weight(store, &name("mlp_out.weight")), H, MLP);
    let mut out = x.to_vec();
    for r in 0..rows {
        let [_, _, gate] = segment_of(r);
        for c in 0..H {
            out[r * H + c] += gate[c] * (attn_out[r * H + c] + mlp_out[r * H + c]);
        }
    }
    out
}

#[test]
#[ignore = "requires a Metal device and test-vectors/references/flux3/dit-tiny"]
fn modulation_applies_per_mode_segment() {
    let Some((backend, _dir, store)) = setup() else {
        return;
    };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let dit = DiT::load(&store, &ops, &params(1, 1)).expect("loads");
    let rows = 8usize;
    let split = 3usize;
    let seeded = |n: usize, seed: u32, scale: f32| -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((state >> 9) as f32 / (1u32 << 23) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    };
    let x_host = seeded(rows * H, 3, 1.0);
    let ids_host: Vec<[f32; 4]> = (0..rows)
        .map(|r| [0.0, 0.5 * (r / 3) as f32, 0.5 * (r % 3) as f32, 0.0])
        .collect();
    let ids_flat: Vec<f32> = ids_host.iter().flatten().copied().collect();
    // Two modulations that differ in every vector, so rows modulated by the
    // wrong one cannot agree with the oracle.
    let mod_a = seeded(3 * H, 11, 0.6);
    let mod_b = seeded(3 * H, 12, 0.6);
    let split_vec = |m: &[f32]| -> [Vec<f64>; 3] {
        [0, 1, 2].map(|i| {
            m[i * H..(i + 1) * H]
                .iter()
                .map(|&v| f64::from(v))
                .collect()
        })
    };
    let segments_host = vec![
        (0..split, split_vec(&mod_a)),
        (split..rows, split_vec(&mod_b)),
    ];
    let x_f64: Vec<f64> = x_host.iter().map(|&v| f64::from(v)).collect();
    let want = host_block(&store, &x_f64, &ids_host, &segments_host);
    let want_all_a = host_block(&store, &x_f64, &ids_host, &[(0..rows, split_vec(&mod_a))]);
    assert!(
        want.iter()
            .zip(&want_all_a)
            .any(|(a, b)| (a - b).abs() > 1e-2),
        "the two modulations must tell a per-segment block from a one-modulation one"
    );

    let run = |segments: &[(std::ops::Range<usize>, &[f32])]| -> Vec<f32> {
        let x = ops.upload_f32("x", &x_host).expect("uploads");
        let ids = ops.upload_f32("ids", &ids_flat).expect("uploads");
        let mods: Vec<ModVec<'_>> = segments
            .iter()
            .map(|(_, m)| ModVec {
                buffer: ops.upload_scratch("mod", m).expect("uploads"),
                hidden: H,
            })
            .collect();
        let stream = Stream {
            x: &x,
            rows,
            ids: ids.view(),
            segments: segments
                .iter()
                .zip(&mods)
                .map(|((range, _), modulation)| Segment {
                    rows: range.clone(),
                    modulation,
                })
                .collect(),
        };
        dit.run_block(&ops, &dit.image_blocks[0], &stream)
            .expect("the block runs");
        ops.download("x", x.view(), rows * H).expect("downloads")
    };
    let got = run(&[(0..split, &mod_a), (split..rows, &mod_b)]);
    let want_f32: Vec<f32> = want.iter().map(|&v| v as f32).collect();
    close(&got, &want_f32, 2e-3, "per-segment block vs f64 oracle");
    let all_a = run(&[(0..rows, &mod_a)]);
    assert!(
        got.iter().zip(&all_a).any(|(a, b)| (a - b).abs() > 1e-2),
        "the second segment must be modulated by its own vector"
    );
}

/// The real export, when `SHISO_TEST_MODEL_FLUX3` names it: every linear
/// loads in its stored quant format and a step over a small token set runs
/// through every kernel entry at the real shapes and alignments. This grades
/// the launches, not the values: no reference output exists at this size, and
/// a file whose weights are zero would pass it too.
#[test]
#[ignore = "requires a Metal device and SHISO_TEST_MODEL_FLUX3"]
#[expect(
    clippy::print_stderr,
    reason = "a skipped checkpoint case must say why on stderr"
)]
fn the_real_export_loads_and_runs_a_step() {
    let Some(path) = std::env::var_os("SHISO_TEST_MODEL_FLUX3").map(PathBuf::from) else {
        eprintln!("SKIP the_real_export_loads_and_runs_a_step: SHISO_TEST_MODEL_FLUX3 is unset");
        return;
    };
    let Some(backend) = device() else { return };
    let ops = Ops::new(&backend).expect("the diffusion kernel set");
    let store = StStore::open(&path).expect("the export opens");
    let params = super::config::inspect(&store).expect("the export inspects");
    let dit = DiT::load(&store, &ops, &params).expect("every linear loads in its stored format");
    let (image_rows, cond_rows, text_rows) = (64usize, 16usize, 80usize);
    let up = |rows: usize, width: usize| {
        let values: Vec<f32> = (0..rows * width)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.01)
            .collect();
        ops.upload_f32("input", &values).expect("uploads")
    };
    let ids = |rows: usize| {
        ops.upload_f32("ids", &vec![0.5f32; rows * 4])
            .expect("uploads")
    };
    let (image, cond, text) = (
        up(image_rows, 128),
        up(cond_rows, 128),
        up(text_rows, params.txt_in),
    );
    let (image_ids, cond_ids, text_ids) = (ids(image_rows), ids(cond_rows), ids(text_rows));
    let out = dit
        .forward(
            &ops,
            &DiTInput {
                image: image.view(),
                image_cond: Some(cond.view()),
                text: text.view(),
                image_rows,
                cond_rows,
                text_rows,
                t: 0.9,
                t_ref: 0.8,
                image_ids: image_ids.view(),
                cond_ids: Some(cond_ids.view()),
                text_ids: text_ids.view(),
            },
        )
        .expect("a step runs at the real shapes");
    let velocity = ops
        .download("velocity", out.view(), image_rows * 128)
        .expect("downloads");
    assert!(
        velocity.iter().all(|v| v.is_finite()),
        "the velocity is finite"
    );
}
