//! Every export of the diarizer reads to the same numbers, and a path that is
//! another family's is left to it.
//!
//! Needs `SHISO_TEST_MODEL_DIAR`; a missing export is skipped by name. The
//! skip line goes to stderr, as a device-free case that did not run.

use std::path::PathBuf;

use super::{DiarConfig, Flavour, OFFLINE, locate};

#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
fn say(line: &str) {
    eprintln!("{line}");
}

/// The five files of the four exports, by where the checkpoint holds them.
pub(crate) fn exports() -> Option<Vec<(Flavour, PathBuf)>> {
    let root = PathBuf::from(std::env::var_os("SHISO_TEST_MODEL_DIAR")?);
    Some(vec![
        (Flavour::Hf, root.join("nvidia")),
        (Flavour::Mlx, root.join("mlx")),
        (
            Flavour::GgufAudioFramework,
            root.join("gguf/nemotron-3-diarization-bf16.gguf"),
        ),
        (
            Flavour::GgufAudioFramework,
            root.join("gguf/nemotron-3-diarization-q8_0.gguf"),
        ),
        (
            Flavour::GgufSortformer,
            root.join("nvidia/Nemotron-3-Diarization.q8_0.gguf"),
        ),
    ])
}

#[test]
fn every_export_reads_to_the_same_numbers() {
    let Some(exports) = exports() else {
        return say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
    };
    for (flavour, path) in exports {
        if !path.exists() {
            say(&format!("SKIP {flavour:?}: {} is absent", path.display()));
            continue;
        }
        let located = locate(&path).expect("readable").expect("this family");
        assert_eq!(located.flavour, flavour, "{}", path.display());
        let store = located.store().expect("opens");
        let config = DiarConfig::read(&located, &store).expect("reads");
        assert_eq!(
            config,
            DiarConfig {
                d_model: 512,
                layers: 31,
                heads: 8,
                head_dim: 64,
                ff: 2048,
                mels: 128,
                stack: 8,
                speakers: 8,
                head_hidden: 192,
            },
            "{}",
            path.display()
        );
        located
            .check_offline(&OFFLINE)
            .expect("the stated offline constants are the family's");
    }
}

#[test]
fn another_familys_checkpoint_is_not_claimed() {
    let Some(asr) = std::env::var_os("SHISO_TEST_MODEL_ASR").map(PathBuf::from) else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    for path in [asr.join("nvidia"), asr.join("mlx")] {
        if path.exists() {
            assert!(
                locate(&path).expect("readable").is_none(),
                "{}",
                path.display()
            );
        }
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    assert!(
        locate(dir.path())
            .expect("an empty directory reads")
            .is_none()
    );
}

#[test]
fn a_stated_offline_constant_that_disagrees_is_refused() {
    let dir = tempfile::tempdir().expect("a scratch directory");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "nemotron3_diarization", "chunk_length": 300,
            "chunk_right_context": 40, "fifo_length": 40, "speaker_cache_update_period": 300}"#,
    )
    .expect("writes");
    let located = locate(dir.path()).expect("reads").expect("this family");
    let error = located
        .check_offline(&OFFLINE)
        .expect_err("300 is not the family's chunk");
    assert!(error.to_string().contains("chunk_length"), "{error}");
}

#[test]
fn a_config_stating_fewer_layers_than_the_file_holds_is_refused() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_DIAR").map(PathBuf::from) else {
        return say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
    };
    let mlx = root.join("mlx");
    if !mlx.exists() {
        return say("SKIP: the MLX export is absent");
    }
    let dir = tempfile::tempdir().expect("a scratch directory");
    let mut config: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(mlx.join("config.json")).expect("reads"))
            .expect("json");
    config["encoder_config"]["n_layers"] = serde_json::json!(30);
    std::fs::write(dir.path().join("config.json"), config.to_string()).expect("writes");
    std::os::unix::fs::symlink(
        mlx.join("model.safetensors"),
        dir.path().join("model.safetensors"),
    )
    .expect("links");
    let located = locate(dir.path()).expect("reads").expect("this family");
    let store = located.store().expect("opens");
    let error = DiarConfig::read(&located, &store).expect_err("a 31st layer is in the file");
    assert!(error.to_string().contains("30"), "{error}");
}
