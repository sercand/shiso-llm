//! The transformer's host-side arithmetic against the independent
//! reference: the timestep sinusoid, the rotary positions and angles, the
//! rotation they drive, and every sigma schedule.
//!
//! The fixtures are `test-vectors/references/qwen_image21/{embed,rope,sigmas}.*`,
//! captured from the reference pipeline in f32 (see that directory's
//! README). No device and no checkpoint is needed.
#![expect(
    clippy::cast_possible_truncation,
    reason = "positions and fixture sizes narrowed for comparison are small"
)]

use std::path::{Path, PathBuf};

use serde_json::Value;
use shiso_vision::generator::ImageGenError;

use super::config::QwenImageConfig;
use super::embed::sinusoid;
use super::rope::{angle_table, target_positions, text_positions};
use super::scheduler::{ScheduleSource, ShiftConfig, check_nodes, euler_dt, resolve};
use super::vae::LatentGrid;

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/qwen_image21")
}

fn read_f32(name: &str) -> Vec<f32> {
    let path = fixture_dir().join(name);
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .chunks_exact(4)
        .map(|w| f32::from_le_bytes([w[0], w[1], w[2], w[3]]))
        .collect()
}

fn read_json(name: &str) -> Value {
    let path = fixture_dir().join(name);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).expect("the fixture parses")
}

fn max_abs(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length");
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0, f32::max)
}

// --- timestep sinusoid -----------------------------------------------------------

/// Cosines first, then sines, from `t = 1000 * sigma`, at the four
/// captured sigmas (including the text tokens' `t = 0` row). One f32 ulp
/// of the largest argument (1000) is 6.1e-5, and the reference's own f32
/// table sits 4.1e-5 from exact `f64` arithmetic, so the bound is that ulp
/// (measured: 2.9e-5 at sigma 1); a sines-first layout misses by order one.
#[test]
fn the_sinusoid_is_cos_first_and_matches_the_reference() {
    let embed = read_json("embed.json");
    for (_, row) in embed["rows"].as_object().expect("rows") {
        let sigma = row["sigma"].as_f64().expect("sigma") as f32;
        let want = read_f32(row["sinusoid"].as_str().expect("file"));
        let got = sinusoid(sigma);
        let err = max_abs(&got, &want);
        assert!(err <= 6.2e-5, "sigma {sigma}: max |err| {err:e}");
        if sigma > 0.0 {
            let swapped: Vec<f32> = got[128..].iter().chain(&got[..128]).copied().collect();
            assert!(
                max_abs(&swapped, &want) > 0.1,
                "sigma {sigma}: sines-first would also pass"
            );
        }
    }
}

// --- rotary positions -------------------------------------------------------------

fn rope_case() -> (usize, LatentGrid, Value) {
    let rope = read_json("rope.json");
    let layout = &rope["layout"];
    let p = layout["text_tokens"].as_u64().expect("text") as usize;
    let grid = LatentGrid {
        h: layout["latent_h"].as_u64().expect("h") as usize,
        w: layout["latent_w"].as_u64().expect("w") as usize,
    };
    (p, grid, rope)
}

fn positions_of(p: usize, grid: LatentGrid) -> Vec<[i64; 3]> {
    let mut all = text_positions(p);
    all.extend(target_positions(p, grid));
    all
}

/// Text token `i` at `i` on every axis, the image block at frame `P` with
/// centred height and width, row-major - against the reference's
/// per-axis position lists for 10 text tokens and a 4 x 6 grid.
#[test]
fn the_positions_match_the_reference_layout() {
    let (p, grid, rope) = rope_case();
    let got = positions_of(p, grid);
    for (axis, name) in ["frame", "height", "width"].iter().enumerate() {
        let want: Vec<i64> = rope["positions"][name]
            .as_array()
            .expect("axis")
            .iter()
            .map(|v| v.as_i64().expect("int"))
            .collect();
        let have: Vec<i64> = got.iter().map(|pos| pos[axis]).collect();
        assert_eq!(have, want, "{name} positions");
    }
}

/// The cos/sin table (8 frame, 28 height, 28 width pairs) against the
/// reference's, to within its own table-vs-direct-angle gap.
#[test]
fn the_angle_table_matches_the_reference() {
    let (p, grid, _) = rope_case();
    let (cos, sin) = angle_table(&positions_of(p, grid), [16, 56, 56]);
    let (ec, es) = (
        max_abs(&cos, &read_f32("rope_cos.f32")),
        max_abs(&sin, &read_f32("rope_sin.f32")),
    );
    assert!(ec <= 1e-6 && es <= 1e-6, "cos {ec:e}, sin {es:e}");
}

/// Rotating the reference's q `(34, 2, 128)` by the table with ADJACENT
/// pairs `(x[2i], x[2i+1])` reproduces its output; the rotate-half pairing
/// `(x[i], x[i + 64])` on the same table does not.
#[test]
fn the_rotation_turns_adjacent_pairs_not_halves() {
    let (text, grid, _) = rope_case();
    let (cos, sin) = angle_table(&positions_of(text, grid), [16, 56, 56]);
    let q_in = read_f32("rope_q_in.f32");
    let want = read_f32("rope_q_out.f32");
    let (hd, half) = (128usize, 64usize);
    let rotate = |adjacent: bool| -> Vec<f32> {
        let mut out = q_in.clone();
        for (r, row) in out.chunks_exact_mut(hd).enumerate() {
            let t = r / 2;
            for i in 0..half {
                let (i0, i1) = if adjacent {
                    (2 * i, 2 * i + 1)
                } else {
                    (i, i + half)
                };
                let (x0, x1) = (row[i0], row[i1]);
                let (c, s) = (cos[t * half + i], sin[t * half + i]);
                row[i0] = x0 * c - x1 * s;
                row[i1] = x0 * s + x1 * c;
            }
        }
        out
    };
    let err = max_abs(&rotate(true), &want);
    assert!(err <= 1e-5, "adjacent pairs: max |err| {err:e}");
    let wrong = max_abs(&rotate(false), &want);
    assert!(wrong > 0.1, "rotate-half would also pass ({wrong:e})");
}

// --- schedules ----------------------------------------------------------------------

fn shift_config(sigmas: &Value, which: &str) -> ShiftConfig {
    let config: ShiftConfig =
        serde_json::from_value(sigmas[which].clone()).expect("a scheduler config");
    config.validate().expect("the shipped config is supported");
    config
}

fn floats(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("array")
        .iter()
        .map(|x| x.as_f64().expect("number") as f32)
        .collect()
}

/// Turbo's shipped sigmas through its identity shift (verbatim plus a
/// terminal 0, bit for bit) at every size, and the base model's dynamic
/// shift with its terminal stretch (40 steps at 1024 and 2048, 8 at 512)
/// from the linear default nodes, all within 1e-6.
#[test]
fn every_reference_schedule_is_reproduced() {
    let sigmas = read_json("sigmas.json");
    let turbo = shift_config(&sigmas, "turbo_scheduler_config");
    let base = shift_config(&sigmas, "base_scheduler_config");
    let sample = floats(&sigmas["turbo_sample_sigmas"]);
    for (name, case) in sigmas["cases"].as_object().expect("cases") {
        let steps = case["num_steps"].as_u64().expect("steps") as usize;
        let seq = case["image_seq_len"].as_u64().expect("seq") as usize;
        let want = floats(&case["sigmas"]);
        let mu = case["mu"].as_f64().expect("mu");
        let got = match case["kind"].as_str() {
            Some("turbo") => {
                let got =
                    resolve(ScheduleSource::Nodes(&sample), &turbo, steps, seq).expect("turbo");
                let verbatim: Vec<u32> = sample.iter().chain(&[0.0]).map(|v| v.to_bits()).collect();
                let bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
                assert_eq!(bits, verbatim, "{name}: an identity shift is verbatim");
                got
            }
            Some("base") => {
                assert!((base.mu(seq) - mu).abs() <= 1e-12, "{name}: mu");
                resolve(ScheduleSource::Linear, &base, steps, seq).expect("base")
            }
            other => panic!("{name}: unknown kind {other:?}"),
        };
        let err = max_abs(&got, &want);
        assert!(err <= 1e-6, "{name}: max |err| {err:e}");
    }
}

/// The Euler step `x + (sigma[1] - sigma[0]) * v` against the reference
/// scheduler's own step on seeded data.
#[test]
fn the_euler_step_matches_the_reference() {
    let sigmas = read_json("sigmas.json");
    let ladder = floats(&sigmas["step_check"]["sigma"]);
    let dt = euler_dt(&ladder, 0).expect("a step");
    let (x, v, want) = (
        read_f32("sigmas_step_x.f32"),
        read_f32("sigmas_step_v.f32"),
        read_f32("sigmas_step_prev.f32"),
    );
    let got: Vec<f32> = x.iter().zip(&v).map(|(x, v)| x + dt * v).collect();
    let err = max_abs(&got, &want);
    assert!(err <= 1e-6, "max |err| {err:e}");
    assert!(euler_dt(&ladder, 1).is_err(), "no step past the last rung");
}

/// The six-step fine-tune's `scheduler_config.json`, as it ships.
const SIX_STEP_SCHEDULER: &str = r#"{"base_image_seq_len": 256, "base_shift": 0.5, "invert_sigmas": false,
    "max_image_seq_len": 8192, "max_shift": 0.9, "num_train_timesteps": 1000,
    "shift": 1.0, "shift_terminal": null, "stochastic_sampling": false,
    "time_shift_type": "exponential", "use_beta_sigmas": false,
    "use_dynamic_shifting": true, "use_exponential_sigmas": false,
    "use_karras_sigmas": false}"#;

/// The six-step fine-tune's `sample_sigmas`.
const SIX_STEP_NODES: [f32; 6] = [1.0, 0.9375, 0.875, 0.75, 0.5, 0.25];

/// A step-distilled fine-tune's six `sample_sigmas` under its dynamic-shift
/// config: each node shifted by the resolution's `mu`, no terminal stretch.
///
/// `mu = 0.5 + (0.9 - 0.5) / (8192 - 256) * (seq - 256)`:
/// - 512x512, seq 1024: `mu = 0.5 + 0.4 * 768 / 7936 = 0.538709677`,
///   `e^mu = 1.713794088`;
/// - 1024x1024, seq 4096: `mu = 0.5 + 0.4 * 3840 / 7936 = 0.693548387`,
///   `e^mu = 2.000802574`.
///
/// `sigma = e^mu / (e^mu + (1/t - 1))`; e.g. at 512, t = 0.5 gives
/// `1.713794 / 2.713794 = 0.631512`, t = 0.25 gives `1.713794 / 4.713794 =
/// 0.363570`, and t = 1 stays 1.
fn six_step_ladders() -> [(usize, [f64; 6]); 2] {
    [
        (
            1024,
            [
                1.0,
                0.962_556_508_755_553_1,
                0.923_056_554_452_938_4,
                0.837_170_207_427_812_8,
                0.631_512_204_799_272_4,
                0.363_569_993_895_720_4,
            ],
        ),
        (
            4096,
            [
                1.0,
                0.967_754_457_791_510_3,
                0.933_358_292_955_905_9,
                0.857_191_977_434_887_8,
                0.666_755_817_712_324_6,
                0.400_096_293_430_902_64,
            ],
        ),
    ]
}

fn assert_six_step(got: &[f32], seq: usize, want: &[f64; 6]) {
    assert_eq!(got.len(), 7, "six rungs and the terminal 0");
    assert_eq!(got[6].to_bits(), 0, "a terminal +0");
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        assert!(
            (f64::from(g) - w).abs() <= 1e-6,
            "seq {seq} rung {i}: {g} vs {w}"
        );
    }
}

/// The ladders above, from the nodes and the config directly.
#[test]
fn shipped_nodes_are_shifted_by_the_resolutions_mu() {
    let six_step: ShiftConfig =
        serde_json::from_str(SIX_STEP_SCHEDULER).expect("the checkpoint's scheduler config");
    six_step.validate().expect("supported");
    for (seq, want) in six_step_ladders() {
        let got =
            resolve(ScheduleSource::Nodes(&SIX_STEP_NODES), &six_step, 6, seq).expect("nodes");
        assert_six_step(&got, seq, &want);
    }
}

/// The refusals: no steps, a step count other than the node list's, and
/// nodes that are empty, do not start at 1, or do not strictly fall within
/// (0, 1].
#[test]
fn bad_schedules_are_refused() {
    let sigmas = read_json("sigmas.json");
    let turbo = shift_config(&sigmas, "turbo_scheduler_config");
    let sample = floats(&sigmas["turbo_sample_sigmas"]);
    assert!(matches!(
        resolve(ScheduleSource::Linear, &turbo, 0, 1024),
        Err(ImageGenError::InvalidSteps { steps: 0 })
    ));
    assert!(matches!(
        resolve(ScheduleSource::Nodes(&sample), &turbo, 4, 1024),
        Err(ImageGenError::StepsFixed { steps: 4, fixed: 8 })
    ));
    assert!(matches!(
        resolve(ScheduleSource::Nodes(&[1.0, 0.5]), &turbo, 3, 1024),
        Err(ImageGenError::StepsFixed { steps: 3, fixed: 2 })
    ));
    assert!(matches!(
        check_nodes(&[]),
        Err(ImageGenError::Checkpoint(_))
    ));
    for bad in [
        &[1.0f32, 1.0][..],
        &[1.0, 0.0],
        &[0.5, 0.7],
        &[1.5, 0.5],
        &[0.9, 0.5],
        &[1.0, f32::NAN],
    ] {
        assert!(
            matches!(
                resolve(ScheduleSource::Nodes(bad), &turbo, 2, 1024),
                Err(ImageGenError::Checkpoint(_))
            ),
            "{bad:?}"
        );
    }
}

/// A static `shift` other than 1 applies `shift * t / (1 + (shift - 1) *
/// t)` to shipped nodes, whatever the resolution; with a terminal it then
/// stretches. At shift 3: t = 0.5 gives `1.5 / 2 = 0.75`, t = 0.25 gives
/// `0.75 / 1.5 = 0.5`.
#[test]
fn a_static_shift_is_applied_to_shipped_nodes() {
    let sigmas = read_json("sigmas.json");
    let mut config = shift_config(&sigmas, "turbo_scheduler_config");
    config.shift = 3.0;
    config.validate().expect("supported");
    let nodes = [1.0f32, 0.5, 0.25];
    for seq in [1024, 4096] {
        let got = resolve(ScheduleSource::Nodes(&nodes), &config, 3, seq).expect("static");
        assert_eq!(got, [1.0, 0.75, 0.5, 0.0], "seq {seq}");
    }
    // Stretched so the last rung lands on 0.2: factor (1 - 0.5) / 0.8.
    config.shift_terminal = Some(0.2);
    let got = resolve(ScheduleSource::Nodes(&nodes), &config, 3, 1024).expect("stretched");
    let want = [1.0, 1.0 - 0.25 / 0.625, 0.2, 0.0];
    for (g, w) in got.iter().zip(want) {
        assert!((f64::from(*g) - w).abs() <= 1e-7, "{got:?}");
    }
}

/// The linear dynamic shift `mu / (mu + (1/t - 1))`: at seq 1024 under the
/// 0.5..0.9 line `mu = 0.538709677`, so t = 0.5 gives `mu / (mu + 1) =
/// 0.350105`.
#[test]
fn the_linear_time_shift_is_applied() {
    let sigmas = read_json("sigmas.json");
    let mut config = shift_config(&sigmas, "turbo_scheduler_config");
    config.use_dynamic_shifting = true;
    config.time_shift_type = "linear".to_string();
    config.validate().expect("supported");
    let got = resolve(ScheduleSource::Nodes(&[1.0, 0.5]), &config, 2, 1024).expect("linear");
    let mu = 0.5 + 0.4 * 768.0 / 7936.0;
    assert_eq!(got[0].to_bits(), 1.0f32.to_bits());
    assert!(
        (f64::from(got[1]) - mu / (mu + 1.0)).abs() <= 1e-7,
        "{got:?}"
    );
    config.time_shift_type = "cosine".to_string();
    assert!(
        config.validate().is_err(),
        "an unknown shift form is refused"
    );
}

// --- the install's documents ---------------------------------------------------------

/// The three documents are read together, the fixed sigmas travel with
/// them, and a missing `model_index.json` is refused rather than defaulted
/// (a distilled checkpoint without its sigmas is a different model).
#[test]
fn the_install_documents_are_read_and_the_index_is_required() {
    let sigmas = read_json("sigmas.json");
    let dir = std::env::temp_dir().join(format!("shiso-qwen-image-assets-{}", std::process::id()));
    let write = |rel: &str, body: &str| {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().expect("a parent")).expect("mkdir");
        std::fs::write(path, body).expect("write");
    };
    let block = read_json("block.json");
    write(
        "transformer/config.json",
        &block["model_config"].to_string(),
    );
    write(
        "scheduler/scheduler_config.json",
        &sigmas["turbo_scheduler_config"].to_string(),
    );
    let index = serde_json::json!({"sample_sigmas": sigmas["turbo_sample_sigmas"]});
    write("model_index.json", &index.to_string());
    let config = QwenImageConfig::read(&dir).expect("the documents read");
    assert_eq!(config.dit.hidden(), 4096);
    assert_eq!(config.dit.rope_axes().expect("axes"), [16, 56, 56]);
    let sample = config.sample_sigmas.as_deref().expect("shipped sigmas");
    let ladder = resolve(ScheduleSource::Nodes(sample), &config.shift, 8, 1024).expect("ladder");
    assert_eq!(ladder.len(), 9);
    std::fs::remove_file(dir.join("model_index.json")).expect("rm");
    match QwenImageConfig::read(&dir) {
        Err(ImageGenError::Checkpoint(message)) => {
            assert!(message.contains("model_index.json"), "{message}");
        }
        other => panic!("expected a refusal naming model_index.json, got {other:?}"),
    }
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// The ladder a load would run: the install's `sample_sigmas` (checked as
/// the load checks them) or the linear nodes, shifted by its config.
fn ladder_from(assets: &Path, steps: usize, seq: usize) -> Vec<f32> {
    let config = QwenImageConfig::read(assets).expect("the documents read");
    let source = match config.sample_sigmas.as_deref() {
        Some(nodes) => {
            check_nodes(nodes).expect("well-formed nodes");
            ScheduleSource::Nodes(nodes)
        }
        None => ScheduleSource::Linear,
    };
    resolve(source, &config.shift, steps, seq).expect("a ladder")
}

/// A fine-tune described by its files alone: an index carrying the six
/// nodes beside the fine-tune's own dynamic-shift config gives the
/// hand-computed ladders at 512x512 and 1024x1024; the same documents
/// without the nodes fall back to the linear default.
#[test]
fn a_fine_tune_is_described_by_its_documents() {
    let dir = std::env::temp_dir().join(format!("shiso-qwen-image-six-{}", std::process::id()));
    let write = |rel: &str, body: &str| {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().expect("a parent")).expect("mkdir");
        std::fs::write(path, body).expect("write");
    };
    write(
        "transformer/config.json",
        &read_json("block.json")["model_config"].to_string(),
    );
    write("scheduler/scheduler_config.json", SIX_STEP_SCHEDULER);
    let index = serde_json::json!({
        "_class_name": "QwenImage21Pipeline",
        "sample_sigmas": SIX_STEP_NODES,
    });
    write("model_index.json", &index.to_string());
    for (seq, want) in six_step_ladders() {
        assert_six_step(&ladder_from(&dir, 6, seq), seq, &want);
    }
    write(
        "model_index.json",
        r#"{"_class_name": "QwenImage21Pipeline"}"#,
    );
    let config = QwenImageConfig::read(&dir).expect("the documents read");
    assert!(config.sample_sigmas.is_none());
    let linear = resolve(ScheduleSource::Linear, &config.shift, 6, 1024).expect("linear");
    assert_eq!(ladder_from(&dir, 6, 1024), linear);
    write("model_index.json", r#"{"sample_sigmas": [0.9, 0.5]}"#);
    let config = QwenImageConfig::read(&dir).expect("the documents read");
    assert!(
        matches!(
            check_nodes(config.sample_sigmas.as_deref().expect("nodes")),
            Err(ImageGenError::Checkpoint(_))
        ),
        "nodes that do not start at 1 are refused"
    );
    std::fs::remove_dir_all(&dir).expect("cleanup");
}

/// The staged Turbo install's own documents give exactly the reference's
/// Turbo ladders. Reads `SHISO_TEST_MODEL_QWEN_IMAGE21` (the directory
/// holding `image_assets/`); without it, says so on stderr and returns.
#[expect(
    clippy::print_stderr,
    reason = "a machine without the install skips with its reason on stderr"
)]
#[test]
fn the_turbo_install_documents_give_the_reference_ladders() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN_IMAGE21") else {
        eprintln!("SKIP: SHISO_TEST_MODEL_QWEN_IMAGE21 is unset");
        return;
    };
    let assets = PathBuf::from(root).join("image_assets");
    let sigmas = read_json("sigmas.json");
    for (name, case) in sigmas["cases"].as_object().expect("cases") {
        if case["kind"].as_str() != Some("turbo") {
            continue;
        }
        let steps = case["num_steps"].as_u64().expect("steps") as usize;
        let seq = case["image_seq_len"].as_u64().expect("seq") as usize;
        let want: Vec<u32> = floats(&case["sigmas"])
            .iter()
            .map(|v| v.to_bits())
            .collect();
        let got: Vec<u32> = ladder_from(&assets, steps, seq)
            .iter()
            .map(|v| v.to_bits())
            .collect();
        assert_eq!(got, want, "{name}");
    }
}
