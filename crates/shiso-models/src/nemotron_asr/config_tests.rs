//! Every export of the checkpoint reads to the same numbers, and the language
//! conditioning resolves tags the way the dictionary says.
//!
//! Needs `SHISO_TEST_MODEL_ASR`; a missing export is skipped by name.

use std::path::PathBuf;

use shiso_audio::Language;

use super::{AsrConfig, Flavour, locate, read_filterbank, read_vocab};

#[expect(clippy::print_stdout, reason = "a skipped case says why on stdout")]
fn say(line: &str) {
    println!("{line}");
}

/// The four exports, by the paths the checkpoint directory holds them at.
fn exports() -> Option<Vec<(Flavour, PathBuf)>> {
    let root = PathBuf::from(std::env::var_os("SHISO_TEST_MODEL_ASR")?);
    Some(vec![
        (Flavour::Hf, root.join("nvidia")),
        (Flavour::Mlx, root.join("mlx")),
        (
            Flavour::GgufAsr,
            root.join("nvidia/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"),
        ),
        (
            Flavour::GgufParakeet,
            root.join("gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"),
        ),
    ])
}

#[test]
fn every_export_reads_to_the_same_numbers() {
    let Some(exports) = exports() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let mut first: Option<AsrConfig> = None;
    for (flavour, path) in exports {
        if !path.exists() {
            say(&format!("SKIP {flavour:?}: {} is absent", path.display()));
            continue;
        }
        let located = locate(&path).expect("readable").expect("this family");
        assert_eq!(located.flavour, flavour, "{}", path.display());
        let store = located.store().expect("opens");
        let config = AsrConfig::read(&located, &store).expect("reads");
        assert_eq!(
            (
                config.d_model,
                config.layers,
                config.heads,
                config.head_dim,
                config.ff,
                config.conv_kernel
            ),
            (1024, 24, 8, 128, 4096, 9),
            "{flavour:?}"
        );
        assert_eq!(
            (
                config.sub_channels,
                config.mels,
                config.vocab,
                config.blank,
                config.pred_hidden,
                config.pred_layers
            ),
            (256, 128, 13_088, 13_087, 640, 2),
            "{flavour:?}"
        );
        assert_eq!(
            (
                config.joint,
                config.prompts,
                config.prompt_hidden,
                config.att_left,
                config.max_symbols,
                config.auto_prompt
            ),
            (640, 128, 2048, 56, 10, 101),
            "{flavour:?}"
        );
        assert_eq!(config.lookaheads, vec![0, 3, 6, 13], "{flavour:?}");
        assert_eq!(config.prompt_id(&Language::Auto).expect("auto"), 101);
        assert_eq!(
            config
                .prompt_id(&Language::Tag("en-US".into()))
                .expect("en-US"),
            0
        );
        assert_eq!(
            config.prompt_id(&Language::Tag("en".into())).expect("en"),
            0
        );
        assert_eq!(
            config
                .prompt_id(&Language::Tag("es-US".into()))
                .expect("es-US"),
            3
        );
        assert!(config.prompt_id(&Language::Tag("xx-YY".into())).is_err());

        let vocab = read_vocab(&located, &store).expect("the vocabulary reads");
        assert!(vocab.len() >= 13_087, "{flavour:?}: {} pieces", vocab.len());
        let en_us = (0..13_087u32)
            .find(|&id| vocab.piece(id) == Some("<en-US>"))
            .expect("the en-US tag is a piece");
        assert_eq!(vocab.language_of(en_us), Some("en-US"));
        if flavour == Flavour::GgufAsr {
            assert!(
                read_filterbank(&store, 128, 257).expect("reads").is_some(),
                "the asr export ships its filterbank"
            );
        }
        if let Some(prev) = &first {
            assert_eq!(prev, &config, "{flavour:?} disagrees with the first export");
        } else {
            first = Some(config);
        }
    }
}

#[test]
fn another_family_is_refused_as_not_this_one() {
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_DIAR").map(PathBuf::from) else {
        return say("SKIP: SHISO_TEST_MODEL_DIAR is unset");
    };
    for path in [
        root.join("nvidia"),
        root.join("mlx"),
        root.join("gguf/nemotron-3-diarization-q8_0.gguf"),
    ] {
        if path.exists() {
            assert!(
                locate(&path).expect("readable").is_none(),
                "{} was taken for this family",
                path.display()
            );
        }
    }
}

#[test]
fn a_defect_in_this_familys_config_is_reported_not_passed_over() {
    // A real HF export with a broken prompt dictionary is still this family:
    // the error must say what is broken, not that the path is someone else's.
    let Some(exports) = exports() else {
        return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
    };
    let (_, hf) = &exports[0];
    let dir = tempfile::tempdir().expect("a temp dir");
    for name in ["config.json", "tokenizer.json", "model.safetensors"] {
        std::os::unix::fs::symlink(hf.join(name), dir.path().join(name)).expect("links");
    }
    let mut processor: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(hf.join("processor_config.json")).expect("reads"),
    )
    .expect("json");
    processor["prompt_dictionary"] = serde_json::json!(["en-US"]);
    std::fs::write(
        dir.path().join("processor_config.json"),
        processor.to_string(),
    )
    .expect("writes");
    let located = locate(dir.path())
        .expect("readable")
        .expect("still this family");
    let store = located.store().expect("opens");
    let err = AsrConfig::read(&located, &store).expect_err("refused");
    assert!(err.to_string().contains("prompt dictionary"), "{err}");
}
