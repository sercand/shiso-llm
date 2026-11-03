//! Transcription through a speech decoder: a language model whose audio
//! tower is its paired encoder, asked with one of its card's instructions.
//!
//! The route and an embedder both come through [`transcribe`]; nothing here
//! knows about HTTP. The prompt is rendered here rather than through the chat
//! template because it is one fixed shape - an empty system turn, the
//! instruction, a space, the audio run, the assistant turn opened with an
//! empty thinking block - and the ids the decoder was graded on are that
//! shape's, with no opening token. `tests/transcribe_prompt.rs` pins them
//! against the reference's own.
//!
//! Decoding is greedy and the output passes the loop guard and the tag
//! handling of [`shiso_models::nemotron_transcribe::postprocess`].
//!
//! Word times come from the tower: the clip is registered with its word
//! timer before the turn, so the encode keeps what timing needs, and once the
//! turn is done the timer aligns the transcript, tags and all, against it.

use std::sync::Arc;

use shiso_audio::transcript::Word;
use shiso_audio::{AudioDecodePolicy, AudioError, AudioTower, DecodeLimits, Pcm, WordTiming};
use shiso_core::TokenId;
use shiso_ir::Request;
use shiso_models::nemotron_transcribe::{PLACEMENT, collapse_loops, speaker_turns, strip_tags};
pub use shiso_models::nemotron_transcribe::{Preset, PresetError, parse_preset};

use crate::api::blocking::collect_response;
use crate::api::pipeline::{PromptField, start_with_fresh_media, start_with_prompt};
use crate::error::AppError;
use crate::media::MediaItem;
use crate::state::AppState;
use crate::tokenizer::Vocabulary;

/// The text before the instruction: an empty system turn, then the user turn.
const HEAD: &str = "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n";
/// The text after the audio: the user turn closed, the assistant turn opened
/// with thinking off.
const TAIL: &str = "<|im_end|>\n<|im_start|>assistant\n<think></think>";

/// A clip quieter than this at its loudest is silence...
const SILENT_PEAK: f32 = 1e-3;
/// ...when its RMS is under this as well. Silence is answered with an empty
/// transcript and never encoded: the decoder asked about silence writes
/// something anyway.
const SILENT_RMS: f32 = 3e-5;

/// The loaded speech decoder's audio side: its tower, and how an upload is
/// decoded for it.
#[derive(Debug, Clone)]
pub struct SpeechDecoder {
    tower: Arc<dyn AudioTower>,
    limits: DecodeLimits,
    decode: Arc<AudioDecodePolicy>,
    max_upload_bytes: usize,
}

impl SpeechDecoder {
    /// The decoder's tower, with the operator's decode limits and policy.
    #[must_use]
    pub fn new(
        tower: Arc<dyn AudioTower>,
        limits: DecodeLimits,
        decode: AudioDecodePolicy,
        max_upload_bytes: usize,
    ) -> Self {
        Self {
            tower,
            limits,
            decode: Arc::new(decode),
            max_upload_bytes,
        }
    }

    /// The largest upload the transcription route reads.
    #[must_use]
    pub const fn max_upload_bytes(&self) -> usize {
        self.max_upload_bytes
    }

    /// Decodes an upload to the 16 kHz mono samples the tower reads, off the
    /// async workers.
    ///
    /// # Errors
    ///
    /// The decoder's refusal: a container this build does not read, malformed
    /// bytes, audio past the operator's length cap.
    pub async fn decode(&self, bytes: bytes::Bytes) -> Result<Pcm, AudioError> {
        let limits = self.limits;
        let decode = Arc::clone(&self.decode);
        tokio::task::spawn_blocking(move || shiso_audio::decode_audio(&bytes, &limits, &decode))
            .await
            .map_err(|error| AudioError::Io {
                what: "running the audio decode",
                source: std::io::Error::other(error),
            })?
    }
}

/// One transcription.
#[derive(Debug)]
pub struct TranscribeRequest {
    /// The audio, already decoded.
    pub pcm: Pcm,
    /// What to ask for.
    pub preset: Preset,
    /// Whether to time the transcript's words.
    pub words: bool,
}

/// What a transcription produced.
#[derive(Debug, Clone, PartialEq)]
pub struct TranscribeOutput {
    /// The transcript: loops collapsed, event and style tags removed, and
    /// speaker tags removed unless the preset asked for them.
    pub text: String,
    /// The speaker turns, for the preset that asks for speaker tags; empty
    /// otherwise.
    pub turns: Vec<(Option<u8>, String)>,
    /// Tokens the decoder generated.
    pub completion_tokens: u64,
    /// The words and their times, when they were asked for: empty for
    /// silence, or for a transcript the timer could not align (logged).
    pub words: Option<Vec<Word>>,
}

/// The prompt's ids: the instruction `preset` spells, `rows` copies of
/// `marker` where the encoder's rows go, and the turn markers around them.
///
/// The caller's part of the instruction (context terms, a language, labels)
/// is read as text even where it spells a marker.
///
/// # Errors
///
/// [`AppError::Internal`] when the vocabulary cannot encode the text.
pub fn prompt_ids(
    vocabulary: &Vocabulary,
    preset: &Preset,
    rows: usize,
    marker: TokenId,
) -> Result<Vec<TokenId>, AppError> {
    let instruction = preset.instruction();
    let head = format!("{HEAD}{instruction} ");
    let literal: Vec<std::ops::Range<usize>> = preset
        .caller_text()
        .filter(|text| !text.is_empty())
        .and_then(|text| {
            let at = instruction.rfind(&text)?;
            Some(HEAD.len() + at..HEAD.len() + at + text.len())
        })
        .into_iter()
        .collect();
    let encode = |text: &str, literal: &[std::ops::Range<usize>]| {
        vocabulary
            .encode_piece(text, literal)
            .map_err(|error| AppError::internal("encoding the transcription prompt", error))
    };
    let mut ids = encode(&head, &literal)?;
    ids.extend(std::iter::repeat_n(marker, rows));
    ids.extend(encode(TAIL, &[])?);
    Ok(ids)
}

/// Whether `pcm` is silence by the reference's rule.
#[must_use]
pub fn is_silence(pcm: &Pcm) -> bool {
    let samples = pcm.samples();
    if samples.is_empty() {
        return true;
    }
    let peak = samples.iter().fold(0f32, |m, s| m.max(s.abs()));
    #[expect(
        clippy::cast_precision_loss,
        reason = "a mean over an hour of samples needs no exact count"
    )]
    let mean_square = samples
        .iter()
        .map(|s| f64::from(*s) * f64::from(*s))
        .sum::<f64>()
        / samples.len() as f64;
    #[expect(
        clippy::cast_possible_truncation,
        reason = "an RMS of samples in [-1, 1] fits an f32"
    )]
    let rms = mean_square.sqrt() as f32;
    peak < SILENT_PEAK && rms < SILENT_RMS
}

/// The token cap for `seconds` of audio: six tokens a second and 24 more,
/// the reference recognizer's budget. A decoder caught in a loop stops here.
#[must_use]
pub fn token_budget(seconds: f64) -> u32 {
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "an hour of audio is 21,624 tokens"
    )]
    let tokens = (6.0 * seconds.max(0.0)).ceil() as u32;
    tokens.saturating_add(24).max(24)
}

/// The speech decoder's audio tower, or the refusal for a model that has
/// none.
fn speech_tower(state: &AppState) -> Result<&Arc<dyn AudioTower>, AppError> {
    state
        .speech
        .as_ref()
        .map(|speech| &speech.tower)
        .ok_or(AppError::NotReady {
            what: "a speech decoder (load one paired with its encoder, or start with --asr-model)",
        })
}

/// A clip the tower refuses, as the status its cause deserves.
fn clip_error(error: AudioError) -> AppError {
    match error {
        AudioError::Device(_) | AudioError::Io { .. } => {
            AppError::internal("preparing the audio", error)
        }
        other => AppError::bad_request(other.to_string(), Some("file"), "invalid_audio"),
    }
}

/// Transcribes `request.pcm` with the loaded speech decoder.
///
/// # Errors
///
/// [`AppError::NotReady`] when the loaded model is no speech decoder or has
/// no vocabulary; [`AppError::BadRequest`] for audio its tower refuses (too
/// short for one row, too long); otherwise as [`start_with_prompt`] and
/// [`collect_response`].
pub async fn transcribe(
    state: &AppState,
    request: TranscribeRequest,
) -> Result<TranscribeOutput, AppError> {
    let tower = speech_tower(state)?;
    let vocabulary = state.vocabulary.as_ref().ok_or(AppError::NotReady {
        what: "the speech decoder's vocabulary",
    })?;
    let timer = if request.words {
        Some(tower.word_timer().ok_or_else(|| {
            AppError::bad_request(
                "this speech decoder's encoder does not time words",
                Some("timestamp_granularities"),
                "invalid_request",
            )
        })?)
    } else {
        None
    };
    if is_silence(&request.pcm) {
        return Ok(TranscribeOutput {
            text: String::new(),
            turns: Vec::new(),
            completion_tokens: 0,
            words: timer.map(|_| Vec::new()),
        });
    }
    let marker = vocabulary
        .media_ids(&PLACEMENT)
        .audio()
        .ok_or(AppError::NotReady {
            what: "an audio marker in the speech decoder's vocabulary",
        })?;
    let clip = shiso_audio::clip_for(tower.as_ref(), &request.pcm).map_err(clip_error)?;
    let ticket = timer.map(|timer| timer.want(&clip));
    let item = MediaItem::Audio(clip);
    let ids = prompt_ids(vocabulary, &request.preset, item.pad_count(1), marker)?;
    let mut generate = Request::new(state.model_id.clone(), Vec::new());
    generate.sampling.temperature = Some(0.0);
    let seconds = request.pcm.duration_secs();
    generate.max_tokens = token_budget(seconds);
    let started = if ticket.is_some() {
        // The timer keeps its evidence only from an encode, so rows cached
        // by an earlier request must not stand in for one.
        start_with_fresh_media(state, &generate, ids, vec![item], PromptField::File).await?
    } else {
        start_with_prompt(state, &generate, ids, vec![item], PromptField::File).await?
    };
    let response = collect_response(started.events).await?;
    let raw = collapse_loops(&response.message.text());
    let words = match ticket {
        Some(ticket) => Some(time_words(Arc::clone(tower), ticket, raw.clone(), seconds).await?),
        None => None,
    };
    let keep = request.preset.keeps_speakers();
    Ok(TranscribeOutput {
        text: strip_tags(&raw, keep),
        turns: if keep {
            speaker_turns(&raw)
        } else {
            Vec::new()
        },
        completion_tokens: response.usage.completion_tokens,
        words,
    })
}

/// `transcript`'s words timed by `tower`'s timer against the clip `ticket`
/// registered, off the async workers: a long clip's alignment is seconds of
/// arithmetic.
async fn time_words(
    tower: Arc<dyn AudioTower>,
    ticket: shiso_audio::TimingTicket,
    transcript: String,
    seconds: f64,
) -> Result<Vec<Word>, AppError> {
    let timed = tokio::task::spawn_blocking(move || {
        let timer = tower.word_timer().ok_or(AudioError::BadSpec {
            reason: "the tower stopped timing words".into(),
        })?;
        timer.words(&ticket, &transcript, seconds)
    })
    .await
    .map_err(|error| {
        AppError::internal(
            "timing the words",
            AudioError::Io {
                what: "running the word alignment",
                source: std::io::Error::other(error),
            },
        )
    })?
    .map_err(|error| AppError::internal("timing the words", error))?;
    Ok(match timed {
        WordTiming::Timed(words) => words,
        WordTiming::Untimed { reason } => {
            tracing::warn!(%reason, "the transcript's words could not be timed");
            Vec::new()
        }
    })
}

#[cfg(test)]
mod tests {
    use shiso_audio::Pcm;

    use super::{is_silence, token_budget};

    #[test]
    fn silence_is_both_quiet_and_low_energy() {
        let zeros = Pcm::new(vec![0.0; 32_000], 16_000).expect("pcm");
        assert!(is_silence(&zeros));
        let mut click = vec![0.0; 32_000];
        click[100] = 0.5;
        assert!(!is_silence(&Pcm::new(click, 16_000).expect("pcm")));
    }

    #[test]
    fn the_budget_is_six_tokens_a_second_and_24() {
        assert_eq!(token_budget(0.0), 24);
        assert_eq!(token_budget(11.0), 90);
        assert_eq!(token_budget(600.0), 3624);
    }
}
