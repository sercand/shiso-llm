//! A transcript in each response format, and as a stream of events.
//!
//! Every JSON body and the final stream event carry the audio's duration as
//! the usage, in whole seconds rounded up, because that is what this surface
//! bills in and what a client sums.

use serde_json::{Value, json};
use shiso_audio::transcript::{self, Segment};

use crate::api::sse::Frame;
use crate::audio::{Transcript, TranscriptEvent};
use crate::openai::transcription_wire::{Granularity, ResponseFormat};

const JSON: &str = "application/json";
const TEXT: &str = "text/plain; charset=utf-8";

fn usage(duration: f64) -> Value {
    // Non-negative and far below 2^53 for any clip the decode limit admits.
    #[expect(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a clip's duration, rounded up, is a small non-negative whole number"
    )]
    let seconds = duration.max(0.0).ceil() as u64;
    json!({"type": "duration", "seconds": seconds})
}

fn diarized_segment(segment: &Segment) -> Value {
    json!({
        "type": "transcript.text.segment",
        "id": format!("seg_{}", segment.id),
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "speaker": transcript::speaker_label(segment.speaker.unwrap_or(0)),
    })
}

/// The body for `format`, with its content type.
#[must_use]
pub fn body(
    format: ResponseFormat,
    t: &Transcript,
    granularities: &[Granularity],
) -> (&'static str, String) {
    match format {
        ResponseFormat::Json => (
            JSON,
            json!({"text": t.text, "usage": usage(t.duration)}).to_string(),
        ),
        ResponseFormat::Text => (TEXT, format!("{}\n", t.text)),
        ResponseFormat::Srt => (TEXT, transcript::srt(&t.segments)),
        ResponseFormat::Vtt => (TEXT, transcript::vtt(&t.segments)),
        ResponseFormat::VerboseJson => {
            let mut value = json!({
                "task": "transcribe",
                "language": t.language,
                "duration": t.duration,
                "text": t.text,
                "usage": usage(t.duration),
            });
            // Segments unless the caller asked only for words.
            let segments =
                granularities.is_empty() || granularities.contains(&Granularity::Segment);
            if segments {
                value["segments"] = t
                    .segments
                    .iter()
                    .map(|s| {
                        json!({"id": s.id, "seek": 0, "start": s.start, "end": s.end,
                               "text": s.text, "temperature": 0.0})
                    })
                    .collect();
            }
            if granularities.contains(&Granularity::Word) {
                value["words"] = t
                    .words
                    .iter()
                    .map(|w| json!({"word": w.text, "start": w.start, "end": w.end}))
                    .collect();
            }
            (JSON, value.to_string())
        }
        ResponseFormat::DiarizedJson => (
            JSON,
            json!({
                "task": "transcribe",
                "duration": t.duration,
                "text": t.text,
                "segments": t.segments.iter().map(diarized_segment).collect::<Vec<_>>(),
                "usage": usage(t.duration),
            })
            .to_string(),
        ),
    }
}

/// The stream's frames, one event at a time.
///
/// A plain stream is text deltas; a diarized one is speaker segments, each
/// sent once it can no longer change, with whatever the job had not settled
/// sent at the end. Either way the last frame is the whole text.
#[derive(Debug)]
pub struct TranscriptionFrames {
    diarized: bool,
    segments_sent: usize,
}

impl TranscriptionFrames {
    /// A stream of the plain or the diarized kind.
    #[must_use]
    pub const fn new(diarized: bool) -> Self {
        Self {
            diarized,
            segments_sent: 0,
        }
    }

    /// The frames `event` produces.
    pub fn on_event(&mut self, event: &TranscriptEvent) -> Vec<Frame> {
        match event {
            TranscriptEvent::Delta(delta) if !self.diarized => vec![Frame::data(
                json!({"type": "transcript.text.delta", "delta": delta}).to_string(),
            )],
            TranscriptEvent::Segment(segment) if self.diarized => {
                self.segments_sent += 1;
                vec![Frame::data(diarized_segment(segment).to_string())]
            }
            TranscriptEvent::Delta(_) | TranscriptEvent::Segment(_) => Vec::new(),
            TranscriptEvent::Done(t) => {
                let mut out = Vec::new();
                if self.diarized {
                    out.extend(
                        t.segments
                            .iter()
                            .skip(self.segments_sent)
                            .map(|s| Frame::data(diarized_segment(s).to_string())),
                    );
                }
                out.push(Frame::data(
                    json!({"type": "transcript.text.done", "text": t.text,
                           "usage": usage(t.duration)})
                    .to_string(),
                ));
                out
            }
            TranscriptEvent::Failed(failure) => vec![Frame::data(
                json!({"type": "error", "error": {"message": failure.message}}).to_string(),
            )],
        }
    }
}

#[cfg(test)]
#[path = "transcription_encode_tests.rs"]
mod tests;
