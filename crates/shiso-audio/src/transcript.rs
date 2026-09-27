//! From timed pieces to what a client reads: words, segments, speakers,
//! subtitles.
//!
//! Model-agnostic on purpose. Any recognizer that reports which encoder frame
//! each piece was emitted at, and any diarizer that reports per-frame speaker
//! probabilities, meet here. The alignment is deliberately simple: a word
//! belongs to whoever is most active while it is spoken. Overlapped speech
//! goes to the dominant voice rather than being transcribed twice.

use std::fmt::Write as _;

use crate::error::AudioError;
use crate::num::{f64_of, usize_floor};

/// A piece of text with the encoder frame it was emitted at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TimedPiece {
    /// Text with word boundaries already turned into spaces.
    pub text: String,
    /// Encoder frame index.
    pub frame: u32,
}

/// A word and when it was spoken, in seconds.
#[derive(Debug, Clone, PartialEq)]
pub struct Word {
    /// The word, trimmed.
    pub text: String,
    /// Start, seconds.
    pub start: f64,
    /// End, seconds.
    pub end: f64,
}

/// A run of words shown together.
#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    /// Position in the transcript, from 0.
    pub id: usize,
    /// Start, seconds.
    pub start: f64,
    /// End, seconds.
    pub end: f64,
    /// Words joined by single spaces.
    pub text: String,
    /// The speaker index, when diarized.
    pub speaker: Option<usize>,
}

/// Per-frame speaker probabilities, row-major `[frames][speakers]`.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerActivity {
    frame_secs: f64,
    speakers: usize,
    probs: Vec<f32>,
}

/// How activity becomes turns.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TurnRule {
    /// A frame is active at or above this probability.
    pub threshold: f32,
    /// Runs shorter than this are dropped.
    pub min_secs: f64,
    /// Runs of one speaker closer than this merge.
    pub merge_gap_secs: f64,
}

/// One speaker's continuous stretch.
#[derive(Debug, Clone, PartialEq)]
pub struct Turn {
    /// Speaker index.
    pub speaker: usize,
    /// Start, seconds.
    pub start: f64,
    /// End, seconds.
    pub end: f64,
}

const SENTENCE_END: [char; 6] = ['.', '?', '!', '。', '？', '！'];
const MAX_GAP_SECS: f64 = 1.0;
const MAX_SEGMENT_SECS: f64 = 30.0;
/// Below this probability everywhere under a word, nobody is speaking it.
const SILENT_BELOW: f32 = 0.5;

impl SpeakerActivity {
    /// Binds probabilities to their geometry.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] when the frame duration is not positive, there
    /// are no speakers, or the block is not a whole number of frames.
    pub fn new(frame_secs: f64, speakers: usize, probs: Vec<f32>) -> Result<Self, AudioError> {
        if frame_secs.is_nan()
            || frame_secs <= 0.0
            || speakers == 0
            || !probs.len().is_multiple_of(speakers)
        {
            return Err(AudioError::BadSpec {
                reason: "speaker activity geometry is inconsistent".into(),
            });
        }
        Ok(Self {
            frame_secs,
            speakers,
            probs,
        })
    }

    /// Frames.
    #[must_use]
    pub fn frames(&self) -> usize {
        self.probs.len() / self.speakers
    }

    /// Speakers.
    #[must_use]
    pub fn speakers(&self) -> usize {
        self.speakers
    }

    /// Seconds per frame.
    #[must_use]
    pub fn frame_secs(&self) -> f64 {
        self.frame_secs
    }

    /// The probability of `speaker` at `frame`.
    #[must_use]
    pub fn prob(&self, frame: usize, speaker: usize) -> f32 {
        self.probs[frame * self.speakers + speaker]
    }
}

/// Pieces grouped into words: `(text, first frame, last frame)`, empty
/// words dropped.
fn spans(pieces: &[TimedPiece]) -> Vec<(String, u32, u32)> {
    let mut out: Vec<(String, u32, u32)> = Vec::new();
    for p in pieces {
        if out.is_empty() || p.text.starts_with(' ') {
            out.push((p.text.trim().to_owned(), p.frame, p.frame));
        } else if let Some(last) = out.last_mut() {
            last.0.push_str(p.text.trim_end());
            last.2 = p.frame;
        }
    }
    out.retain(|(text, _, _)| !text.is_empty());
    out
}

/// Groups pieces into words, each ending one frame after its last piece.
#[must_use]
pub fn words(pieces: &[TimedPiece], frame_secs: f64) -> Vec<Word> {
    words_in_chunks(pieces, frame_secs, &[])
}

/// Groups the pieces of a recognizer that transcribes in chunks ending at
/// `chunk_ends` (frames, increasing; see
/// [`crate::RecognizerStream::chunk_ends`]) into words.
///
/// Such a recognizer places each piece somewhere inside its chunk and knows
/// nothing of where a word stops, so a word ends where the next word of its
/// chunk starts, and a chunk's last word at the chunk's end: a one-word turn
/// then spans its turn rather than one frame, and a segment cut at a turn
/// ends at the cut. A word past every chunk end, and every word when
/// `chunk_ends` is empty, ends one frame after its last piece, as [`words`]
/// has it.
#[must_use]
pub fn words_in_chunks(pieces: &[TimedPiece], frame_secs: f64, chunk_ends: &[u32]) -> Vec<Word> {
    let spans = spans(pieces);
    let chunk_of = |frame: u32| chunk_ends.partition_point(|&end| end <= frame);
    (0..spans.len())
        .map(|i| {
            let (text, first, last) = &spans[i];
            let chunk = chunk_of(*first);
            let end = match chunk_ends.get(chunk) {
                None => last + 1,
                Some(&chunk_end) => match spans.get(i + 1) {
                    Some((_, next, _)) if chunk_of(*next) == chunk => *next,
                    _ => chunk_end,
                },
            };
            Word {
                text: text.clone(),
                start: f64::from(*first) * frame_secs,
                end: f64::from(end) * frame_secs,
            }
        })
        .collect()
}

fn split_runs(
    words: &[Word],
    same_group: impl Fn(usize, usize) -> bool,
    speakers: Option<&[usize]>,
) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut begin = 0;
    for i in 0..words.len() {
        let last = i + 1 == words.len();
        let ends_sentence = words[i].text.ends_with(SENTENCE_END);
        let gap = !last && words[i + 1].start - words[i].end > MAX_GAP_SECS;
        let long = words[i].end - words[begin].start > MAX_SEGMENT_SECS;
        let regroup = !last && !same_group(i, i + 1);
        if last || ends_sentence || gap || long || regroup {
            let text = words[begin..=i]
                .iter()
                .map(|w| w.text.as_str())
                .collect::<Vec<_>>()
                .join(" ");
            out.push(Segment {
                id: out.len(),
                start: words[begin].start,
                end: words[i].end,
                text,
                speaker: speakers.map(|s| s[begin]),
            });
            begin = i + 1;
        }
    }
    out
}

/// Sentence- and gap-bounded segments.
#[must_use]
pub fn segments(words: &[Word]) -> Vec<Segment> {
    split_runs(words, |_, _| true, None)
}

/// Speaker turns from activity.
#[must_use]
pub fn turns(activity: &SpeakerActivity, rule: &TurnRule) -> Vec<Turn> {
    let fs = activity.frame_secs();
    let mut out = Vec::new();
    for s in 0..activity.speakers() {
        let mut runs: Vec<(usize, usize)> = Vec::new();
        let mut open = None;
        for t in 0..=activity.frames() {
            let on = t < activity.frames() && activity.prob(t, s) >= rule.threshold;
            match (on, open) {
                (true, None) => open = Some(t),
                (false, Some(b)) => {
                    runs.push((b, t));
                    open = None;
                }
                _ => {}
            }
        }
        let mut merged: Vec<(usize, usize)> = Vec::new();
        for (b, e) in runs {
            match merged.last_mut() {
                Some(last) if f64_of(b - last.1) * fs < rule.merge_gap_secs => last.1 = e,
                _ => merged.push((b, e)),
            }
        }
        for (b, e) in merged {
            if f64_of(e - b) * fs >= rule.min_secs {
                out.push(Turn {
                    speaker: s,
                    start: f64_of(b) * fs,
                    end: f64_of(e) * fs,
                });
            }
        }
    }
    out.sort_by(|a, b| a.start.total_cmp(&b.start));
    out
}

/// The dominant speaker of each word.
#[must_use]
pub fn assign_speakers(words: &[Word], activity: &SpeakerActivity) -> Vec<usize> {
    let fs = activity.frame_secs();
    let frames = activity.frames();
    let mut out = Vec::with_capacity(words.len());
    let mut previous = 0;
    for w in words {
        let b = usize_floor((w.start / fs).max(0.0)).min(frames.saturating_sub(1));
        let e = usize_floor((w.end / fs).ceil().max(0.0)).clamp(b + 1, frames.max(b + 1));
        let mut best = (previous, f32::MIN);
        let mut loudest = 0.0f32;
        for s in 0..activity.speakers() {
            let mut sum = 0.0f32;
            for t in b..e.min(frames) {
                let p = activity.prob(t, s);
                sum += p;
                loudest = loudest.max(p);
            }
            if sum > best.1 {
                best = (s, sum);
            }
        }
        let who = if loudest < SILENT_BELOW {
            previous
        } else {
            best.0
        };
        out.push(who);
        previous = who;
    }
    out
}

/// The speaker of each word of a transcript recognised in chunks cut at
/// `cuts` (seconds, increasing): the speaker who is dominant, as
/// [`crate::chunks::turn_cuts`] reckons it, over most of the chunk the word's
/// middle falls in. A chunking recognizer's word times are interpolated
/// inside their chunk, so a word near a turn's edge can sit in the silence
/// the cut was put in, where [`assign_speakers`] would give it the previous
/// voice; the chunk is the turn, so its words are that turn's. A chunk where
/// nobody is dominant keeps [`assign_speakers`]' answer for its words.
#[must_use]
pub fn assign_speakers_by_chunk(
    words: &[Word],
    activity: &SpeakerActivity,
    cuts: &[f64],
) -> Vec<usize> {
    let fs = activity.frame_secs();
    let chunk_of = |secs: f64| cuts.partition_point(|&cut| cut <= secs);
    let mut held = vec![vec![0usize; activity.speakers()]; cuts.len() + 1];
    for t in 0..activity.frames() {
        let mut who = None;
        let mut loudest = crate::chunks::DOMINANT_ABOVE;
        for s in 0..activity.speakers() {
            let p = activity.prob(t, s);
            if p > loudest {
                loudest = p;
                who = Some(s);
            }
        }
        if let Some(s) = who {
            held[chunk_of((f64_of(t) + 0.5) * fs)][s] += 1;
        }
    }
    // The most frames, the lower index on a tie.
    let owners: Vec<Option<usize>> = held
        .iter()
        .map(|count| {
            let (who, frames) =
                count.iter().enumerate().fold(
                    (0, 0),
                    |best, (s, &n)| if n > best.1 { (s, n) } else { best },
                );
            (frames > 0).then_some(who)
        })
        .collect();
    assign_speakers(words, activity)
        .into_iter()
        .zip(words)
        .map(|(own, w)| owners[chunk_of(f64::midpoint(w.start, w.end))].unwrap_or(own))
        .collect()
}

/// Segments of consecutive same-speaker words.
#[must_use]
pub fn diarized_segments(words: &[Word], speakers: &[usize]) -> Vec<Segment> {
    split_runs(words, |a, b| speakers[a] == speakers[b], Some(speakers))
}

/// `A`, `B`, …, `Z`, `AA`, `AB`, ….
#[must_use]
pub fn speaker_label(index: usize) -> String {
    let mut n = index + 1;
    let mut out = Vec::new();
    while n > 0 {
        n -= 1;
        out.push(char::from(b'A' + u8::try_from(n % 26).unwrap_or(0)));
        n /= 26;
    }
    out.iter().rev().collect()
}

fn stamp(secs: f64, sep: char) -> String {
    let ms = usize_floor((secs.max(0.0) * 1000.0).round());
    format!(
        "{:02}:{:02}:{:02}{sep}{:03}",
        ms / 3_600_000,
        ms / 60_000 % 60,
        ms / 1000 % 60,
        ms % 1000
    )
}

/// `SubRip` subtitles.
#[must_use]
pub fn srt(segments: &[Segment]) -> String {
    let mut out = String::new();
    for (i, s) in segments.iter().enumerate() {
        let who = s
            .speaker
            .map(|k| format!("{}: ", speaker_label(k)))
            .unwrap_or_default();
        write!(
            out,
            "{}\n{} --> {}\n{who}{}\n\n",
            i + 1,
            stamp(s.start, ','),
            stamp(s.end, ','),
            s.text
        )
        .expect("writing to a String cannot fail");
    }
    out
}

/// Escapes cue text: in `WebVTT`, `<` opens a tag and `&` an entity.
fn vtt_escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

/// `WebVTT` subtitles.
#[must_use]
pub fn vtt(segments: &[Segment]) -> String {
    let mut out = String::from("WEBVTT\n\n");
    for s in segments {
        let who = s
            .speaker
            .map(|k| format!("<v {}>", speaker_label(k)))
            .unwrap_or_default();
        write!(
            out,
            "{} --> {}\n{who}{}\n\n",
            stamp(s.start, '.'),
            stamp(s.end, '.'),
            vtt_escape(&s.text)
        )
        .expect("writing to a String cannot fail");
    }
    out
}

#[cfg(test)]
#[path = "transcript_tests.rs"]
mod tests;
