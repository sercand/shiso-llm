//! Chat audio for the speech decoder: the checkpoint's own template puts an
//! audio part last in the user turn, after one space, wherever the caller put
//! it, and the turn it renders is the transcription core's prompt.
//!
//! Needs `SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE` (the decoder `.gguf`, with
//! `tokenizer.json` and `chat_template.jinja` beside it); says on stderr why
//! it skipped otherwise. The tower is a stand-in with the encoder's frontend
//! shape: what is under test is where the run lands, not what fills it.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

use base64::Engine as _;
use shiso_audio::{
    AudioClip, AudioDecodePolicy, AudioError, AudioTower, DecodeLimits, Filterbank, Framing,
    LogScale, MelScale, MelSpec, Preemphasis, Spectrum, Window,
};
use shiso_core::{Backend, TokenId};
use shiso_ir::{Block, MediaSource, Message, Request, Role, ThinkingDirective};
use shiso_models::nemotron_transcribe::PLACEMENT;
use shiso_runtime::api::transcribe::{Preset, prompt_ids};
use shiso_runtime::boot::SERVER_TOOL_JSON_STYLE;
use shiso_runtime::tokenizer::Vocabulary;
use shiso_runtime::tokenizer::render_prompt::PromptRenderer;
use shiso_vision::EncodedRows;

/// One row per eight mel rows, as the encoder stacks them.
#[derive(Debug)]
struct Stand {
    spec: MelSpec,
}

impl AudioTower for Stand {
    fn mel_spec(&self) -> &MelSpec {
        &self.spec
    }
    fn tokens_for(&self, valid_mel: usize) -> Option<usize> {
        (valid_mel > 0).then(|| valid_mel.div_ceil(8))
    }
    fn max_seconds(&self) -> f64 {
        3600.0
    }
    fn out_hidden(&self) -> usize {
        2688
    }
    fn device_bytes(&self) -> u64 {
        0
    }
    fn encode(&self, _: &Arc<dyn Backend>, _: &[AudioClip]) -> Result<EncodedRows, AudioError> {
        Err(AudioError::Empty)
    }
}

fn stand() -> Arc<dyn AudioTower> {
    Arc::new(Stand {
        spec: MelSpec {
            sample_rate: 16_000,
            n_fft: 512,
            win_length: 400,
            hop_length: 160,
            n_mels: 128,
            window: Window::HannPeriodic,
            framing: Framing::LeftPadded {
                pad_to_multiple: 128,
            },
            preemphasis: Preemphasis::None,
            spectrum: Spectrum::Magnitude,
            filterbank: Filterbank::Computed {
                scale: MelScale::Htk,
                slaney_norm: false,
                f_min: 0.0,
                f_max: 8_000.0,
            },
            log_scale: LogScale::Natural { guard: 1e-3 },
        },
    })
}

/// A 16 kHz mono 16-bit WAV of one second of a quiet tone.
fn wav() -> Vec<u8> {
    let samples = 16_000u32;
    let data_len = samples * 2;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&16_000u32.to_le_bytes());
    out.extend_from_slice(&32_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for n in 0..samples {
        let phase = f64::from(n) * 440.0 * std::f64::consts::TAU / 16_000.0;
        #[expect(
            clippy::cast_possible_truncation,
            reason = "the amplitude is 3000, well inside i16"
        )]
        let value = (phase.sin() * 3000.0) as i16;
        out.extend_from_slice(&value.to_le_bytes());
    }
    out
}

/// The renderer the boot path builds for this checkpoint, or `None` (said on
/// stderr) when the files are not here.
#[expect(clippy::print_stderr, reason = "a skipped test says why")]
fn renderer() -> Option<(PromptRenderer, Arc<Vocabulary>)> {
    let Some(decoder) = std::env::var_os("SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE").map(PathBuf::from)
    else {
        eprintln!("SKIP: SHISO_TEST_MODEL_NEMOTRON_TRANSCRIBE is unset");
        return None;
    };
    let dir = decoder.parent().expect("the decoder sits in a directory");
    // As boot loads it for a speech decoder: no opening token.
    let vocabulary = Arc::new(
        Vocabulary::from_file(&dir.join("tokenizer.json"))
            .expect("the vocabulary loads")
            .with_bos(None),
    );
    let template =
        std::fs::read_to_string(dir.join("chat_template.jinja")).expect("the template reads");
    let renderer = PromptRenderer::new(&template, SERVER_TOOL_JSON_STYLE, Arc::clone(&vocabulary))
        .expect("the template compiles")
        .with_audio_tower(
            stand(),
            PLACEMENT,
            DecodeLimits::new(NonZeroU32::new(3600).expect("non-zero")),
            AudioDecodePolicy::default(),
        );
    Some((renderer, vocabulary))
}

fn render(renderer: &PromptRenderer, blocks: Vec<Block>) -> Vec<TokenId> {
    let mut request = Request::new("speech", vec![Message::new(Role::User, blocks)]);
    request.thinking = ThinkingDirective::Off;
    renderer.render(&request).expect("renders").tokens
}

fn audio() -> Block {
    Block::Audio(MediaSource::Inline(
        base64::engine::general_purpose::STANDARD.encode(wav()),
    ))
}

#[test]
fn audio_part_moves_to_turn_end() {
    let Some((renderer, vocabulary)) = renderer() else {
        return;
    };
    let instruction = Preset::Verbatim.instruction();
    let text_first = render(&renderer, vec![Block::Text(instruction.clone()), audio()]);
    let audio_first = render(&renderer, vec![audio(), Block::Text(instruction)]);
    assert_eq!(audio_first, text_first);

    // And the turn is the transcription core's prompt, id for id.
    let marker = vocabulary
        .media_ids(&PLACEMENT)
        .audio()
        .expect("the vocabulary declares the audio marker");
    let rows = text_first.iter().filter(|&&id| id == marker).count();
    assert!(rows > 0, "the audio run was expanded");
    assert_eq!(
        text_first,
        prompt_ids(&vocabulary, &Preset::Verbatim, rows, marker).expect("renders")
    );
}
