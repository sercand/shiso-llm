//! The recognizer path on the device against the reference's tokens, frames
//! and text. Greedy search's rules are the step kernel's, pinned by the shared
//! kernel cases (`greedy_step_follows_the_search_rules`).

mod device {
    //! The recognizer path on the device against the reference, f32 export.
    //! Needs `SHISO_TEST_MODEL_ASR`.

    use std::path::{Path, PathBuf};
    use std::sync::Arc;

    use shiso_audio::{Detokenizer, Language};

    use crate::audio_ops::Ops;
    use crate::audio_ops::load::upload_f32;
    use crate::nemotron_asr::config::{AsrConfig, locate, read_vocab};
    use crate::nemotron_asr::encoder::{Encoder, EncoderState};
    use crate::nemotron_asr::transducer::{DecoderState, Transducer};
    use crate::nemotron_asr::weights::AsrWeights;

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
            .expect("the fixture exists")
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn json(path: &Path) -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(path).expect("the fixture exists"))
            .expect("json")
    }

    fn mel(clip: &str) -> Vec<f32> {
        let meta = json(&fixtures().join(format!("mel-{clip}-centred.json")));
        let valid = usize::try_from(meta["frames_valid"].as_u64().expect("valid")).expect("fits");
        let mut rows = f32s(&fixtures().join(format!("mel-{clip}-centred.f32")));
        rows.truncate(valid * 128);
        rows
    }

    struct Rig {
        ops: Ops,
        config: AsrConfig,
        weights: AsrWeights,
        vocab: Arc<shiso_audio::PieceVocab>,
    }

    fn rig() -> Option<Rig> {
        let root = PathBuf::from(std::env::var_os("SHISO_TEST_MODEL_ASR")?);
        let ops = Ops::new(crate::audio_ops::testing::device()?);
        let located = locate(&root.join("nvidia"))
            .expect("readable")
            .expect("this family");
        let store = located.store().expect("opens");
        let config = AsrConfig::read(&located, &store).expect("reads");
        let weights = AsrWeights::load(&ops, &located, &config, &store).expect("loads");
        let vocab = Arc::new(read_vocab(&located, &store).expect("vocabulary"));
        Some(Rig {
            ops,
            config,
            weights,
            vocab,
        })
    }

    fn read(ops: &Ops, buf: &shiso_core::DeviceBuffer, floats: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; floats * 4];
        ops.backend()
            .copy_d2h(
                buf.slice(0, u64::try_from(floats * 4).expect("fits"))
                    .expect("range"),
                &mut bytes,
            )
            .expect("reads");
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    fn language(tag: &str) -> Language {
        if tag == "auto" {
            Language::Auto
        } else {
            Language::Tag(tag.into())
        }
    }

    #[test]
    #[ignore = "requires a GPU device"]
    fn the_prompt_fused_projection_matches_the_reference() {
        let Some(rig) = rig() else {
            return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
        };
        let transducer = Transducer::new(&rig.ops, &rig.weights, &rig.config);
        for lang in ["auto", "en-US"] {
            let enc = f32s(&fixtures().join("asr-speech-en-r13-encoder.f32"));
            let want = f32s(&fixtures().join(format!("asr-speech-en-r13-{lang}-projected.f32")));
            let frames = enc.len() / 1024;
            let dev = upload_f32(&rig.ops, &enc).expect("uploads");
            let got = transducer
                .project(&language(lang), dev.view(), frames)
                .expect("projects");
            let got = read(&rig.ops, &got, frames * 640);
            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let worst = got
                .iter()
                .zip(&want)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale;
            say(&format!(
                "projected speech-en {lang}: worst error relative to max {worst:.2e}"
            ));
            assert!(worst <= PROJECTED_BUDGET, "{lang}: {worst}");
        }
    }

    /// Measured 2026-09-25 on M1 Max against the reference in f32: 1.35e-6
    /// (auto) - accumulation order only. The budget is about three times that.
    const PROJECTED_BUDGET: f32 = 5e-6;

    #[test]
    #[ignore = "requires a GPU device"]
    fn tokens_frames_and_text_match_the_reference_exactly() {
        let Some(rig) = rig() else {
            return say("SKIP: SHISO_TEST_MODEL_ASR is unset");
        };
        let mut cases: Vec<PathBuf> = std::fs::read_dir(fixtures())
            .expect("fixtures")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name().and_then(|n| n.to_str()).is_some_and(|n| {
                    n.starts_with("asr-")
                        && Path::new(n)
                            .extension()
                            .is_some_and(|e| e.eq_ignore_ascii_case("json"))
                })
            })
            .collect();
        cases.sort();
        assert!(cases.len() >= 10, "the ASR fixtures are present");
        for path in cases {
            let case = json(&path);
            let clip = case["clip"].as_str().expect("clip");
            let lookahead =
                usize::try_from(case["lookahead"].as_u64().expect("lookahead")).expect("fits");
            let lang = case["language"].as_str().expect("language");
            let encoder = Encoder::new(&rig.ops, &rig.weights, &rig.config);
            let transducer = Transducer::new(&rig.ops, &rig.weights, &rig.config);
            let mut enc_state = EncoderState::new(&rig.ops, &rig.config, lookahead).expect("state");
            let mut dec_state = DecoderState::new(&transducer, &language(lang)).expect("decoder");
            let mel = mel(clip);
            let dev = upload_f32(&rig.ops, &mel).expect("uploads");
            let mut events = Vec::new();
            let first = encoder
                .push(&mut enc_state, dev.view(), mel.len() / 128, false)
                .expect("pushes");
            events.extend(transducer.feed(&mut dec_state, &first).expect("decodes"));
            let tail = encoder.finish(&mut enc_state).expect("finishes");
            events.extend(
                transducer
                    .feed(&mut dec_state, &tail)
                    .expect("decodes the tail"),
            );
            let tokens: Vec<u64> = events.iter().map(|e| u64::from(e.token.get())).collect();
            let frames: Vec<u64> = events.iter().map(|e| u64::from(e.frame)).collect();
            let want_tokens: Vec<u64> = case["tokens"]
                .as_array()
                .expect("tokens")
                .iter()
                .map(|v| v.as_u64().expect("id"))
                .collect();
            let want_frames: Vec<u64> = case["frames"]
                .as_array()
                .expect("frames")
                .iter()
                .map(|v| v.as_u64().expect("frame"))
                .collect();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("?")
                .to_owned();
            assert_eq!(tokens, want_tokens, "{name}: tokens");
            assert_eq!(frames, want_frames, "{name}: frames");
            let mut detok = Detokenizer::new(Arc::clone(&rig.vocab));
            for e in &events {
                detok.push(e.token.get());
            }
            let want_text = case["text"].as_str().expect("text");
            // The model tags every sentence it ends; the transcript drops the
            // tags and reports the last one as the language.
            let mut body = String::new();
            let mut lang = None;
            for word in want_text.split_whitespace() {
                match word.strip_prefix('<').and_then(|w| w.strip_suffix('>')) {
                    Some(tag) if tag.contains('-') => lang = Some(tag.to_owned()),
                    _ => {
                        if !body.is_empty() {
                            body.push(' ');
                        }
                        body.push_str(word);
                    }
                }
            }
            assert_eq!(detok.text().trim(), body, "{name}: text");
            assert_eq!(detok.language(), lang.as_deref(), "{name}: language");
            say(&format!("{name}: {} tokens exact", tokens.len()));
        }
    }
}
