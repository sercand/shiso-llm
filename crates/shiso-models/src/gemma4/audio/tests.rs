//! The Gemma 4 audio tower against the reference's own tower and embedder.
//!
//! Needs `SHISO_TEST_MODEL_GEMMA4_AUDIO_MLX` (the MLX directory) and/or
//! `SHISO_TEST_MODEL_GEMMA4_AUDIO_MMPROJ` (the mmproj GGUF), the committed audio
//! fixtures, and a GPU device (Metal or CUDA). The mel rows fed are the reference
//! extractor's own, so what is compared is the tower alone.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use shiso_audio::{AudioClip, AudioTower, MelFrames};
use shiso_core::Backend;

use super::Gemma4AudioTower;

#[expect(
    clippy::print_stdout,
    reason = "a skipped case says why, and the measured error is the report"
)]
fn say(line: &str) {
    println!("{line}");
}

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../test-vectors/references/audio")
}

fn f32s(path: &Path) -> Vec<f32> {
    std::fs::read(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn json(path: &Path) -> serde_json::Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("the fixture exists")).expect("json")
}

/// The reference extractor's valid mel rows of `clip`, as a tower clip.
fn clip(tower: &Gemma4AudioTower, clip: &str) -> AudioClip {
    let meta = json(&fixtures().join(format!("mel-{clip}-left-padded.json")));
    let valid = usize::try_from(meta["frames_valid"].as_u64().expect("valid")).expect("fits");
    let mut rows = f32s(&fixtures().join(format!("mel-{clip}-left-padded.f32")));
    rows.truncate(valid * 128);
    let tokens = tower.tokens_for(valid).expect("within the limit");
    AudioClip::new(MelFrames::from_rows(rows, 128).expect("whole rows"), tokens).expect("a clip")
}

fn read(backend: &Arc<dyn Backend>, rows: &shiso_core::DeviceBuffer, floats: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; floats * 4];
    backend
        .copy_d2h(
            rows.slice(0, u64::try_from(floats * 4).expect("fits"))
                .expect("range"),
            &mut bytes,
        )
        .expect("reads");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Worst difference relative to each reference row's largest magnitude.
fn worst(got: &[f32], want: &[f32], width: usize) -> f32 {
    got.chunks(width)
        .zip(want.chunks(width))
        .map(|(g, w)| {
            let scale = w.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-6);
            g.iter()
                .zip(w)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale
        })
        .fold(0.0f32, f32::max)
}

/// Per container, (tower, embedded), row-relative. Measured 2026-09-25 on M1
/// Max against the reference in f32 over the MLX weights: MLX tower 1.1e-5,
/// embedded 1.6e-5; mmproj tower 5.4e-6 (the same bf16 tower weights). The
/// mmproj's embedded rows differ by 7.6e-2 because that file ships the
/// projection in bf16 while the reference used the MLX export's 4-bit one - a
/// different weight, not a different computation. Budgets are about twice the
/// measured worst.
const BUDGETS: [(&str, f32, f32); 2] = [("mlx", 2.5e-5, 3.5e-5), ("mmproj", 1.2e-5, 0.15)];

#[test]
#[ignore = "requires a GPU device"]
fn the_tower_and_embedder_match_the_reference() {
    let Some(backend) = crate::audio_ops::testing::device() else {
        return;
    };
    for (name, var) in [
        ("mlx", "SHISO_TEST_MODEL_GEMMA4_AUDIO_MLX"),
        ("mmproj", "SHISO_TEST_MODEL_GEMMA4_AUDIO_MMPROJ"),
    ] {
        let Some(path) = std::env::var_os(var).map(PathBuf::from) else {
            say(&format!("SKIP {name}: {var} is unset"));
            continue;
        };
        let tower = Gemma4AudioTower::open(&backend, &path).expect("the tower opens");
        let (_, tower_budget, embed_budget) = BUDGETS
            .iter()
            .copied()
            .find(|b| b.0 == name)
            .expect("a budget");
        for name_clip in ["speech-en", "weather-es", "speech-en-odd"] {
            let c = clip(&tower, name_clip);
            let tokens = c.tokens();
            let meta = json(&fixtures().join(format!("gemma-{name_clip}.json")));
            assert_eq!(
                u64::try_from(tokens).expect("fits"),
                meta["tokens"].as_u64().expect("tokens"),
                "{name_clip}: token count"
            );
            let (hidden, embedded) = tower.encode_stages(&backend, &c).expect("encodes");
            let t = worst(
                &read(&backend, &hidden, tokens * 1536),
                &f32s(&fixtures().join(format!("gemma-{name_clip}-tower.f32"))),
                1536,
            );
            let e = worst(
                &read(&backend, &embedded, tokens * 2560),
                &f32s(&fixtures().join(format!("gemma-{name_clip}-embedded.f32"))),
                2560,
            );
            say(&format!(
                "{name} {name_clip}: tower {t:.2e}, embedded {e:.2e}"
            ));
            assert!(t <= tower_budget, "{name} {name_clip}: tower {t}");
            assert!(e <= embed_budget, "{name} {name_clip}: embedded {e}");
            let rows = tower
                .encode(&backend, std::slice::from_ref(&c))
                .expect("encodes");
            assert_eq!(rows.grids.len(), 1);
            assert_eq!(
                usize::try_from(rows.rows.len()).expect("fits"),
                tokens * 2560 * 4
            );
        }
    }
}

#[test]
fn the_token_rule_is_two_stride_two_stages_capped_at_thirty_seconds() {
    assert_eq!(super::tokens_for(2999), Some(750));
    assert_eq!(super::tokens_for(1103), Some(276));
    assert_eq!(super::tokens_for(1104), Some(276));
    assert_eq!(super::tokens_for(1), Some(1));
    assert_eq!(super::tokens_for(3001), None);
}

#[test]
fn an_mlx_directory_holds_an_audio_tower_only_when_its_config_says_so() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    let config = dir.path().join("config.json");
    std::fs::write(&config, r#"{"text_config": {"hidden_size": 2560}}"#).expect("writes");
    assert!(!super::holds_audio_tower(dir.path()).expect("the probe reads the config"));
    std::fs::write(
        &config,
        r#"{"text_config": {"hidden_size": 2560}, "audio_config": null}"#,
    )
    .expect("writes");
    assert!(!super::holds_audio_tower(dir.path()).expect("a null block is no tower"));
    std::fs::write(
        &config,
        r#"{"text_config": {"hidden_size": 2560}, "audio_config": {"hidden_size": 1024}}"#,
    )
    .expect("writes");
    assert!(super::holds_audio_tower(dir.path()).expect("the probe reads the config"));
}

#[test]
fn a_path_that_is_not_there_is_an_error_not_a_missing_tower() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    assert!(super::holds_audio_tower(&dir.path().join("absent.gguf")).is_err());
    assert!(super::holds_audio_tower(dir.path()).is_err());
}

#[test]
fn the_gemma_mmproj_and_llm_gguf_answer_the_probe() {
    let Some(mmproj) = std::env::var_os("SHISO_TEST_MODEL_GEMMA4_AUDIO_MMPROJ").map(PathBuf::from)
    else {
        say("SKIP: SHISO_TEST_MODEL_GEMMA4_AUDIO_MMPROJ is unset");
        return;
    };
    assert!(super::holds_audio_tower(&mmproj).expect("the mmproj reads"));
    let llm = mmproj.with_file_name("gemma-4-E4B_q4_0-it.gguf");
    if llm.exists() {
        assert!(!super::holds_audio_tower(&llm).expect("the language model reads"));
    } else {
        say(&format!("SKIP llm half: {} is absent", llm.display()));
    }
}
