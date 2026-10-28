//! Every export of both checkpoints is recognised, reads to the same geometry,
//! and pairs its text GGUF with the right mmproj; a GGUF whose mmproj cannot
//! be chosen, or does not fit, is refused naming the files and the flag.
//!
//! The export cases need `SHISO_TEST_MODEL_QWEN3ASR` and
//! `SHISO_TEST_MODEL_R2T2`; the synthetic cases write tiny GGUF headers and
//! run everywhere.

use std::path::{Path, PathBuf};

use shiso_audio::{Framing, LogScale, Window};

use super::{
    AUDIO_PAD, Flavour, IM_END, Located, Qwen3AsrConfig, Stores, language_name, languages, locate,
    mel_spec, read_merges, read_vocab,
};
use crate::audio_ops::store::TensorStore;
use crate::test_gguf::{Kv, write_gguf};

#[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
pub(crate) fn say(line: &str) {
    eprintln!("{line}");
}

/// One export on disk, and what locating it must find.
pub(crate) struct Export {
    /// A short name for messages.
    pub(crate) name: &'static str,
    /// The path handed to `locate`.
    pub(crate) path: PathBuf,
    /// The flavour it must be read as.
    pub(crate) flavour: Flavour,
    /// The file name of the mmproj it must pair with, for a text GGUF.
    pub(crate) mmproj: Option<&'static str>,
    /// Whether it is the fine-tune (a retrained decoder, same geometry).
    pub(crate) fine_tune: bool,
}

/// The six exports, by the paths the two checkpoint directories hold them at.
/// A root whose variable is unset is skipped, saying so.
pub(crate) fn exports() -> Vec<Export> {
    let mut out = Vec::new();
    match std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR").map(PathBuf::from) {
        Some(root) => out.extend([
            Export {
                name: "qwen hf",
                path: root.join("qwen"),
                flavour: Flavour::Hf,
                mmproj: None,
                fine_tune: false,
            },
            Export {
                name: "qwen mlx",
                path: root.join("mlx"),
                flavour: Flavour::Mlx,
                mmproj: None,
                fine_tune: false,
            },
            Export {
                name: "qwen gguf q8_0",
                path: root.join("gguf/Qwen3-ASR-1.7B-Q8_0.gguf"),
                flavour: Flavour::Gguf,
                mmproj: Some("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf"),
                fine_tune: false,
            },
            Export {
                name: "qwen gguf bf16",
                path: root.join("gguf/Qwen3-ASR-1.7B-bf16.gguf"),
                flavour: Flavour::Gguf,
                mmproj: Some("mmproj-Qwen3-ASR-1.7B-bf16.gguf"),
                fine_tune: false,
            },
        ]),
        None => say("SKIP qwen exports: SHISO_TEST_MODEL_QWEN3ASR is unset"),
    }
    match std::env::var_os("SHISO_TEST_MODEL_R2T2").map(PathBuf::from) {
        Some(root) => out.extend([
            Export {
                name: "r2t2 hf",
                path: root.join("bf16"),
                flavour: Flavour::Hf,
                mmproj: None,
                fine_tune: true,
            },
            Export {
                name: "r2t2 gguf q8_0",
                path: root.join("gguf/Confucius4-R2T2-Q8_0.gguf"),
                flavour: Flavour::Gguf,
                mmproj: Some("mmproj-Confucius4-R2T2-Q8_0.gguf"),
                fine_tune: true,
            },
        ]),
        None => say("SKIP r2t2 exports: SHISO_TEST_MODEL_R2T2 is unset"),
    }
    out
}

#[test]
#[ignore = "reads the real checkpoints named by SHISO_TEST_MODEL_QWEN3ASR and SHISO_TEST_MODEL_R2T2"]
fn every_export_is_located() {
    let mut first: Option<Qwen3AsrConfig> = None;
    let mut seen = 0;
    for export in exports() {
        if !export.path.exists() {
            say(&format!(
                "SKIP {}: {} is absent",
                export.name,
                export.path.display()
            ));
            continue;
        }
        let located = locate(&export.path, None)
            .expect("readable")
            .expect("this family");
        assert_eq!(located.flavour, export.flavour, "{}", export.name);
        assert_eq!(
            located
                .mmproj
                .as_ref()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str()),
            export.mmproj,
            "{}",
            export.name
        );
        let stores = located.stores().expect("opens");
        let config = Qwen3AsrConfig::read(&located, &stores).expect("reads");
        let a = &config.audio;
        assert_eq!(
            (a.mels, a.d_model, a.layers, a.heads, a.ffn, a.conv_channels),
            (128, 1024, 24, 16, 4096, 480),
            "{}",
            export.name
        );
        assert_eq!(a.output_dim, 2048, "{}", export.name);
        let t = &config.text;
        assert_eq!(
            (
                t.hidden, t.layers, t.heads, t.kv_heads, t.head_dim, t.ffn, t.vocab
            ),
            (2048, 28, 16, 8, 128, 6144, 151_936),
            "{}",
            export.name
        );
        assert!((a.norm_eps - 1e-5).abs() < 1e-9, "{}", export.name);
        assert!((t.rms_eps - 1e-6).abs() < 1e-10, "{}", export.name);
        assert!((t.rope_theta - 1e6).abs() < 1.0, "{}", export.name);

        let vocab = read_vocab(&located, &stores).expect("the vocabulary reads");
        assert!(vocab.is_byte_level(), "{}", export.name);
        assert_eq!(vocab.piece(11_528), Some("language"), "{}", export.name);
        assert_eq!(vocab.piece(151_704), Some("<asr_text>"), "{}", export.name);
        assert_eq!(vocab.piece(IM_END), Some("<|im_end|>"), "{}", export.name);
        assert_eq!(
            vocab.piece(AUDIO_PAD),
            Some("<|audio_pad|>"),
            "{}",
            export.name
        );

        if export.flavour == Flavour::Hf && !export.fine_tune {
            // The language table is the config's own list, in its order.
            let text = std::fs::read_to_string(export.path.join("config.json")).expect("reads");
            let json: serde_json::Value = serde_json::from_str(&text).expect("json");
            let names: Vec<&str> = json["support_languages"]
                .as_array()
                .expect("a list")
                .iter()
                .map(|v| v.as_str().expect("a name"))
                .collect();
            let ours: Vec<&str> = languages().iter().map(|(_, name)| *name).collect();
            assert_eq!(ours, names);
        }

        if let Some(prev) = &first {
            assert_eq!(
                prev, &config,
                "{} disagrees with the first export",
                export.name
            );
        } else {
            first = Some(config);
        }
        seen += 1;
    }
    say(&format!("located {seen} exports"));
}

#[test]
#[ignore = "reads the real checkpoints named by SHISO_TEST_MODEL_QWEN3ASR and SHISO_TEST_MODEL_R2T2"]
fn a_gguf_directory_is_its_one_text_file_and_an_explicit_mmproj_wins() {
    if let Some(root) = std::env::var_os("SHISO_TEST_MODEL_R2T2").map(PathBuf::from) {
        let located = locate(&root.join("gguf"), None)
            .expect("readable")
            .expect("this family");
        assert_eq!(located.flavour, Flavour::Gguf);
        assert!(located.weights[0].ends_with("Confucius4-R2T2-Q8_0.gguf"));
    } else {
        say("SKIP r2t2 directory: SHISO_TEST_MODEL_R2T2 is unset");
    }
    let Some(root) = std::env::var_os("SHISO_TEST_MODEL_QWEN3ASR").map(PathBuf::from) else {
        return say("SKIP qwen directory: SHISO_TEST_MODEL_QWEN3ASR is unset");
    };
    let dir = root.join("gguf");
    // Two text GGUFs in one directory: which one was meant is not guessed.
    let err = locate(&dir, None).expect_err("two text files");
    let text = err.to_string();
    assert!(
        text.contains("Qwen3-ASR-1.7B-Q8_0.gguf") && text.contains("Qwen3-ASR-1.7B-bf16.gguf"),
        "{text}"
    );
    let explicit = dir.join("mmproj-Qwen3-ASR-1.7B-bf16.gguf");
    let located = locate(&dir.join("Qwen3-ASR-1.7B-Q8_0.gguf"), Some(&explicit))
        .expect("readable")
        .expect("this family");
    assert_eq!(located.mmproj.as_deref(), Some(explicit.as_path()));
}

fn text_gguf(path: &Path) {
    write_gguf(
        path,
        &[
            ("general.architecture", Kv::Str("qwen3vl")),
            ("qwen3vl.block_count", Kv::U32(28)),
            ("qwen3vl.embedding_length", Kv::U32(2048)),
            ("qwen3vl.feed_forward_length", Kv::U32(6144)),
            ("qwen3vl.attention.head_count", Kv::U32(16)),
            ("qwen3vl.attention.head_count_kv", Kv::U32(8)),
            ("qwen3vl.attention.key_length", Kv::U32(128)),
            ("qwen3vl.attention.value_length", Kv::U32(128)),
            ("qwen3vl.rope.freq_base", Kv::F32(1e6)),
            ("qwen3vl.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
            ("qwen3vl.context_length", Kv::U32(65_536)),
            ("tokenizer.ggml.model", Kv::Str("gpt2")),
            ("tokenizer.ggml.tokens", Kv::Strs(vec!["a", "b"])),
        ],
        &[],
    );
}

fn mmproj_gguf(path: &Path, projection: u32) {
    write_gguf(
        path,
        &[
            ("general.architecture", Kv::Str("clip")),
            ("clip.has_audio_encoder", Kv::Bool(true)),
            ("clip.audio.projector_type", Kv::Str("qwen3a")),
            ("clip.audio.projection_dim", Kv::U32(projection)),
            ("clip.audio.embedding_length", Kv::U32(1024)),
            ("clip.audio.feed_forward_length", Kv::U32(4096)),
            ("clip.audio.block_count", Kv::U32(24)),
            ("clip.audio.attention.head_count", Kv::U32(16)),
            ("clip.audio.num_mel_bins", Kv::U32(128)),
            ("clip.audio.attention.layer_norm_epsilon", Kv::F32(1e-5)),
        ],
        &[],
    );
}

#[test]
fn an_ambiguous_or_missing_mmproj_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let text = dir.path().join("Model-Q8_0.gguf");
    text_gguf(&text);

    // None beside it.
    let err = locate(&text, None).expect_err("no mmproj");
    let msg = err.to_string();
    assert!(
        msg.contains("--asr-mmproj") && msg.contains("Model-Q8_0.gguf"),
        "{msg}"
    );

    // One beside it, but for another quantisation: named, not taken.
    let other = dir.path().join("mmproj-Model-bf16.gguf");
    mmproj_gguf(&other, 2048);
    let err = locate(&text, None).expect_err("no quant match");
    let msg = err.to_string();
    assert!(
        msg.contains("mmproj-Model-bf16.gguf") && msg.contains("--asr-mmproj"),
        "{msg}"
    );

    // The one whose suffix matches is taken.
    let matching = dir.path().join("mmproj-Model-Q8_0.gguf");
    mmproj_gguf(&matching, 2048);
    let located = locate(&text, None).expect("readable").expect("this family");
    assert_eq!(located.mmproj.as_deref(), Some(matching.as_path()));

    // Two whose suffix matches: both named, neither taken.
    let second = dir.path().join("mmproj-Other-Q8_0.gguf");
    mmproj_gguf(&second, 2048);
    let err = locate(&text, None).expect_err("two matches");
    let msg = err.to_string();
    assert!(
        msg.contains("mmproj-Model-Q8_0.gguf")
            && msg.contains("mmproj-Other-Q8_0.gguf")
            && msg.contains("--asr-mmproj"),
        "{msg}"
    );

    // The flag settles it.
    let located = locate(&text, Some(&second))
        .expect("readable")
        .expect("this family");
    assert_eq!(located.mmproj.as_deref(), Some(second.as_path()));

    // A flag naming a file that is not an audio mmproj is refused by name.
    let err = locate(&text, Some(&text)).expect_err("not an mmproj");
    assert!(err.to_string().contains("Model-Q8_0.gguf"), "{err}");

    // A flag for an export that carries its own tower is refused.
    let hf = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        hf.path().join("config.json"),
        r#"{"model_type": "qwen3_asr", "thinker_config": {}}"#,
    )
    .expect("writes");
    let err = locate(hf.path(), Some(&matching)).expect_err("a directory has its own tower");
    assert!(err.to_string().contains("--asr-mmproj"), "{err}");
}

#[test]
fn an_mmproj_that_does_not_fit_the_text_model_is_refused_by_name() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let text = dir.path().join("Model-Q8_0.gguf");
    text_gguf(&text);
    let mmproj = dir.path().join("mmproj-Model-Q8_0.gguf");
    mmproj_gguf(&mmproj, 1024);
    let located = locate(&text, None).expect("readable").expect("this family");
    let stores = located.stores().expect("opens");
    let err = Qwen3AsrConfig::read(&located, &stores).expect_err("1024 into 2048");
    let msg = err.to_string();
    assert!(
        msg.contains("mmproj-Model-Q8_0.gguf")
            && msg.contains("Model-Q8_0.gguf")
            && msg.contains("1024")
            && msg.contains("2048")
            && msg.contains("--asr-mmproj"),
        "{msg}"
    );
}

#[test]
fn an_mlx_export_quantised_otherwise_is_refused() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3_asr", "quantization": {"bits": 4, "group_size": 64}}"#,
    )
    .expect("writes");
    let err = locate(dir.path(), None).expect_err("4-bit");
    assert!(err.to_string().contains("bits 4"), "{err}");
}

#[test]
fn other_paths_are_not_this_family() {
    let dir = tempfile::tempdir().expect("a temp dir");
    assert!(locate(dir.path(), None).expect("readable").is_none());
    std::fs::write(dir.path().join("config.json"), r#"{"model_type": "other"}"#).expect("writes");
    assert!(locate(dir.path(), None).expect("readable").is_none());
    let gguf = dir.path().join("other.gguf");
    write_gguf(&gguf, &[("general.architecture", Kv::Str("llama"))], &[]);
    assert!(locate(&gguf, None).expect("readable").is_none());
    let plain = dir.path().join("notes.txt");
    std::fs::write(&plain, "not a checkpoint").expect("writes");
    assert!(locate(&plain, None).expect("readable").is_none());
}

#[test]
fn the_nemotron_exports_are_not_this_family() {
    let mut paths = Vec::new();
    match std::env::var_os("SHISO_TEST_MODEL_ASR").map(PathBuf::from) {
        Some(root) => paths.extend([
            root.join("nvidia"),
            root.join("mlx"),
            root.join("nvidia/nemotron-3.5-asr-streaming-0.6b.q8_0.gguf"),
            root.join("gguf/nemotron-3.5-asr-streaming-0.6b-Q4_K_M.gguf"),
            root.join("gguf"),
        ]),
        None => say("SKIP nemotron asr: SHISO_TEST_MODEL_ASR is unset"),
    }
    match std::env::var_os("SHISO_TEST_MODEL_DIAR").map(PathBuf::from) {
        Some(root) => paths.extend([
            root.join("nvidia"),
            root.join("mlx"),
            root.join("gguf/nemotron-3-diarization-q8_0.gguf"),
        ]),
        None => say("SKIP nemotron diarizer: SHISO_TEST_MODEL_DIAR is unset"),
    }
    for path in paths {
        if !path.exists() {
            say(&format!("SKIP {}: absent", path.display()));
            continue;
        }
        assert!(
            locate(&path, None).expect("readable").is_none(),
            "{} was taken for this family",
            path.display()
        );
    }
}

#[test]
fn the_constants_the_mmproj_lacks_are_stated() {
    let spec = mel_spec();
    assert_eq!(
        (
            spec.sample_rate,
            spec.n_fft,
            spec.win_length,
            spec.hop_length,
            spec.n_mels
        ),
        (16_000, 400, 400, 160, 128)
    );
    assert_eq!(spec.window, Window::HannPeriodic);
    assert_eq!(spec.framing, Framing::CentredReflect);
    assert_eq!(spec.log_scale, LogScale::Log10MaxFloor);
    assert_eq!(languages().len(), 30);
    assert_eq!(language_name("en"), Some("English"));
    assert_eq!(language_name("EN-us"), Some("English"));
    assert_eq!(language_name("zh"), Some("Chinese"));
    assert_eq!(language_name("yue"), Some("Cantonese"));
    assert_eq!(language_name("mk"), Some("Macedonian"));
    assert_eq!(language_name("xx"), None);
}

#[test]
fn a_bare_relative_gguf_is_looked_for_beside_it_in_the_working_directory() {
    // `--asr-model X-Q8_0.gguf`: the file's parent is the empty path, which
    // is the working directory, not a directory that cannot be listed. The
    // tests run in the crate's directory, which holds no mmproj, so the
    // search ends in the refusal that names the flag, not an IO error.
    let err = super::find_mmproj(Path::new("Absent-Q8_0.gguf")).expect_err("no mmproj here");
    assert!(
        matches!(err, shiso_audio::AudioError::Checkpoint { .. }),
        "{err}"
    );
    assert!(err.to_string().contains("--asr-mmproj"), "{err}");
    assert_eq!(super::dir_of(Path::new("Absent-Q8_0.gguf")), Path::new("."));
    assert_eq!(
        super::dir_of(Path::new("gguf/A-Q8_0.gguf")),
        Path::new("gguf")
    );
    assert_eq!(super::dir_of(Path::new("/A-Q8_0.gguf")), Path::new("/"));
}

/// An opened `Stores`, cheap to build and unused by `read_merges` for
/// `Flavour::Hf`/`Flavour::Mlx` (only the GGUF arm touches it): a stand-in so
/// the HF/MLX cases below have a value of the right type to pass.
fn stores_stub() -> Stores {
    let dir = tempfile::tempdir().expect("a temp dir");
    let gguf = dir.path().join("stub.gguf");
    text_gguf(&gguf);
    Stores {
        text: TensorStore::open_gguf(&gguf).expect("opens"),
        mmproj: None,
    }
}

fn hf_located(dir: &Path) -> Located {
    Located {
        flavour: Flavour::Hf,
        dir: dir.to_path_buf(),
        weights: Vec::new(),
        mmproj: None,
        affine_group: None,
        config: None,
    }
}

#[test]
fn an_hf_export_with_no_merges_file_at_all_loads_with_no_merges() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let located = hf_located(dir.path());
    let stores = stores_stub();
    assert_eq!(read_merges(&located, &stores).expect("no merges"), None);
}

#[test]
fn an_hf_export_falls_back_to_tokenizer_json_when_merges_txt_is_absent() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let located = hf_located(dir.path());
    let stores = stores_stub();

    // Both shapes tokenizer.json may spell a merge entry in.
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"model": {"merges": ["a b", ["c", "d"]]}}"#,
    )
    .expect("writes");
    assert_eq!(
        read_merges(&located, &stores).expect("from tokenizer.json"),
        Some(vec!["a b".to_owned(), "c d".to_owned()])
    );

    // merges.txt, once it shows up, wins over tokenizer.json and its
    // #version header and blank lines are dropped.
    std::fs::write(dir.path().join("merges.txt"), "#version: 0.2\ne f\n\ng h\n").expect("writes");
    assert_eq!(
        read_merges(&located, &stores).expect("from merges.txt"),
        Some(vec!["e f".to_owned(), "g h".to_owned()])
    );
}

#[test]
fn a_malformed_merges_txt_is_refused_rather_than_treated_as_absent() {
    let dir = tempfile::tempdir().expect("a temp dir");
    // Not a directory: reading it as merges.txt fails with something other
    // than NotFound, which must not be swallowed into a fallback.
    std::fs::create_dir(dir.path().join("merges.txt")).expect("writes");
    let located = hf_located(dir.path());
    let stores = stores_stub();
    let err = read_merges(&located, &stores).expect_err("a directory is not a merges file");
    assert!(matches!(err, shiso_audio::AudioError::Io { .. }), "{err:?}");
}

#[test]
fn a_malformed_tokenizer_json_merges_entry_is_refused() {
    let dir = tempfile::tempdir().expect("a temp dir");
    std::fs::write(
        dir.path().join("tokenizer.json"),
        r#"{"model": {"merges": [123]}}"#,
    )
    .expect("writes");
    let located = hf_located(dir.path());
    let stores = stores_stub();
    let err = read_merges(&located, &stores).expect_err("neither string nor pair");
    assert!(
        matches!(err, shiso_audio::AudioError::Checkpoint { .. }),
        "{err:?}"
    );
}
