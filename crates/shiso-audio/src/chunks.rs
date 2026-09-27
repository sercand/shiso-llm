//! Where a chunking recognizer ends one chunk and starts the next.
//!
//! A recognizer that transcribes a clip in pieces needs every piece to end
//! where cutting loses nothing: in a pause, not mid-word. Without a diarizer
//! that is the quietest short window near each chunk-length mark. With one,
//! the speaker turns decide instead, so every chunk is one speaker's turn and
//! a speaker change always lands on the diarizer's own boundary, never inside
//! a chunk; only a turn longer than a chunk falls back to the silence rule
//! inside it.
//!
//! A clip recognised offline is planned whole, before recognition starts
//! ([`plan_offline`]): the same cuts, plus long silences the diarizer hears
//! no one in, which are skipped rather than transcribed, so their encode
//! and decode cost is never paid.
//!
//! Everything here is host arithmetic over frames the caller already has. It
//! is deterministic: the same frames always give the same cuts, ties going to
//! the earliest candidate, so a transcript does not change between runs.

use crate::error::AudioError;
use crate::mel::MelFrames;
use crate::num::{f64_of, usize_floor};
use crate::transcript::SpeakerActivity;

/// The longest a chunk runs before a cut is looked for, seconds.
pub const CHUNK_SECS: f64 = 30.0;
/// How far either side of each mark the quiet window is looked for, seconds.
pub const SEARCH_SECS: f64 = 5.0;
/// The window whose mean energy is compared, seconds.
pub const WINDOW_SECS: f64 = 0.1;
/// A diarizer frame belongs to its loudest speaker only above this
/// probability; below it everywhere, the frame is silence.
pub const DOMINANT_ABOVE: f32 = 0.5;
/// A speaker run shorter than this is a cough or a flicker, not a turn, and
/// joins a neighbour, seconds.
pub const MIN_TURN_SECS: f64 = 0.5;

/// Cut frames for the whole clip by the silence rule: for each multiple of
/// `max_frames` past the last cut, the start of the `window_frames` window
/// with the lowest mean log-mel value within `search_frames` either side of
/// the mark. Given `(3000, 500, 10)` on 10 ms frames, that is the quietest
/// 100 ms within ±5 s of every 30 s mark.
///
/// Cuts are strictly increasing and lie strictly inside `0..mel.valid()`. A
/// clip no longer than `max_frames` has none.
///
/// # Panics
///
/// When `max_frames` or `window_frames` is zero: a chunk of no frames is
/// not a plan, and a window of none measures nothing.
#[must_use]
pub fn silence_cuts(
    mel: &MelFrames,
    max_frames: usize,
    search_frames: usize,
    window_frames: usize,
) -> Vec<usize> {
    let energy = frame_energy(mel);
    let mut out = Vec::new();
    cuts_between(
        &energy,
        Stretch {
            lo: 0,
            hi: mel.valid(),
        },
        &SilenceRule {
            max: max_frames,
            search: search_frames,
            window: window_frames,
        },
        &mut out,
    );
    out
}

/// Cut frames for a diarized clip, in mel frames of `mel_frame_secs` each.
///
/// - The dominant speaker of each diarizer frame is the one with the highest
///   activity above [`DOMINANT_ABOVE`]; a frame where nobody reaches it is
///   silence.
/// - A speaker run shorter than [`MIN_TURN_SECS`] joins its longer neighbour
///   (the earlier one on a tie).
/// - Silence between two runs of one speaker joins them into one turn;
///   silence between different speakers is split at its midpoint.
/// - Every turn change is a cut, and inside any stretch longer than
///   [`CHUNK_SECS`] the [`silence_cuts`] rule adds more.
///
/// No speech at all, or one speaker throughout, therefore gives exactly the
/// silence cuts of the whole clip. The two time bases are the diarizer's
/// [`SpeakerActivity::frame_secs`] and `mel_frame_secs`; each boundary is
/// converted through seconds and rounded to the nearest mel frame.
///
/// [`crate::transcript::turns`] is not used here: it keeps overlapping
/// speakers as overlapping turns and drops short runs, where chunks need
/// one owner per frame and short runs absorbed rather than lost.
///
/// # Panics
///
/// When `mel_frame_secs` is not positive.
#[must_use]
pub fn turn_cuts(activity: &SpeakerActivity, mel: &MelFrames, mel_frame_secs: f64) -> Vec<usize> {
    assert!(
        mel_frame_secs > 0.0,
        "a mel frame lasts a positive time, not {mel_frame_secs}"
    );
    let rule = SilenceRule {
        max: frames_in(CHUNK_SECS, mel_frame_secs),
        search: frames_in(SEARCH_SECS, mel_frame_secs),
        // A mel frame longer than the window is itself the window.
        window: frames_in(WINDOW_SECS, mel_frame_secs).max(1),
    };
    let valid = mel.valid();
    let fs = activity.frame_secs();
    // The fewest frames that last MIN_TURN_SECS: at 0.08 s a frame, 7, since
    // six last 0.48 s. The nudge keeps 0.5 / 0.01 from rounding up to 51.
    let min_frames = usize_floor((MIN_TURN_SECS / fs - 1e-9).ceil()).max(1);
    let runs = absorb_short(dominant_runs(activity), min_frames);
    // Turn changes, in mel frames. Silence runs are skipped: between one
    // speaker's runs that joins them, between two speakers the cut falls at
    // the gap's midpoint, and at either end of the clip it belongs to the
    // nearest turn.
    let mut changes = Vec::new();
    let mut previous: Option<&Run> = None;
    for run in runs.iter().filter(|r| r.who.is_some()) {
        if let Some(p) = previous
            && p.who != run.who
        {
            let secs = f64_of(p.end + run.start) * fs / 2.0;
            let cut = usize_floor((secs / mel_frame_secs).round());
            if cut > 0 && cut < valid && changes.last().is_none_or(|&last| cut > last) {
                changes.push(cut);
            }
        }
        previous = Some(run);
    }
    let energy = frame_energy(mel);
    let mut out = Vec::new();
    let mut lo = 0;
    for hi in changes.into_iter().chain(std::iter::once(valid)) {
        cuts_between(&energy, Stretch { lo, hi }, &rule, &mut out);
        if hi < valid {
            out.push(hi);
        }
        lo = hi;
    }
    out
}

/// A stretch of mel frames in an offline plan, `start..end`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    /// First mel frame.
    pub start: usize,
    /// One past the last mel frame.
    pub end: usize,
    /// Whether the recognizer transcribes it; `false` is a long silence the
    /// plan skips.
    pub transcribe: bool,
}

/// How [`plan_offline`] cuts a whole clip. Frame counts are mel frames,
/// times are seconds.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OfflineRule {
    /// The longest transcribed span. The silence rule looks for a cut within
    /// `search_frames` either side of the mark `max_frames - search_frames`,
    /// so no span it leaves runs past this.
    pub max_frames: usize,
    /// How far either side of each mark the quiet window is looked for.
    pub search_frames: usize,
    /// The window whose mean energy is compared.
    pub window_frames: usize,
    /// The shortest silence worth skipping.
    pub skip_silence_secs: f64,
    /// A diarizer frame where no speaker reaches this activity is silent.
    /// It sits below [`DOMINANT_ABOVE`] on purpose: speech too faint to own
    /// a turn is still speech, and is not skipped.
    pub speech_floor: f32,
    /// Silence kept transcribed on each side of a skipped span that borders
    /// speech, so that a word the diarizer ends a little early is not cut.
    pub pad_secs: f64,
    /// Whether long silences are skipped at all.
    pub skip_silence: bool,
}

impl OfflineRule {
    /// The rule for 10 ms mel frames: spans of at most 35 s cut at the
    /// quietest 100 ms within ±5 s of each 30 s mark (the streaming chunks'
    /// own rule), and silences of 3 s or more under activity 0.3 skipped
    /// with 0.2 s of margin, when `skip_silence` says so.
    #[must_use]
    pub const fn for_10ms_mel(skip_silence: bool) -> Self {
        Self {
            max_frames: 3_500,
            search_frames: 500,
            window_frames: 10,
            skip_silence_secs: 3.0,
            speech_floor: 0.3,
            pad_secs: 0.2,
            skip_silence,
        }
    }

    /// Refuses a rule whose fields cannot make a plan.
    fn check(&self, mel_frame_secs: f64) -> Result<(), AudioError> {
        let bad = |reason: String| Err(AudioError::BadSpec { reason });
        if !(mel_frame_secs.is_finite() && mel_frame_secs > 0.0) {
            return bad(format!(
                "a mel frame lasts a positive time, not {mel_frame_secs}"
            ));
        }
        if self.max_frames <= self.search_frames {
            return bad(format!(
                "an offline span of at most {} frames leaves no mark to search {} frames around",
                self.max_frames, self.search_frames
            ));
        }
        if self.window_frames == 0 {
            return bad("a window of zero frames measures nothing".into());
        }
        if !(self.skip_silence_secs.is_finite() && self.skip_silence_secs > 0.0) {
            return bad(format!(
                "a skipped silence lasts a positive time, not {}",
                self.skip_silence_secs
            ));
        }
        if !(self.pad_secs.is_finite() && self.pad_secs >= 0.0) {
            return bad(format!(
                "a silence margin is a time of zero or more, not {}",
                self.pad_secs
            ));
        }
        if !(self.speech_floor > 0.0 && self.speech_floor <= 1.0) {
            return bad(format!(
                "a speech floor is an activity in (0, 1], not {}",
                self.speech_floor
            ));
        }
        Ok(())
    }
}

/// Cuts a whole clip into spans before any of it is recognised, in mel
/// frames of `mel_frame_secs` each.
///
/// - **Without activity** the cuts are [`silence_cuts`] at the mark
///   `max_frames - search_frames` (30 s under
///   [`OfflineRule::for_10ms_mel`]: the streaming path's own chunks), and
///   every span is transcribed.
/// - **With activity** the cuts are [`turn_cuts`]. When
///   `rule.skip_silence`, every run of diarizer frames at least
///   `skip_silence_secs` long in which no speaker reaches `speech_floor`
///   is skipped, less `pad_secs` on each side that borders speech; a side
///   at the clip's start or end has no speech to keep a margin for. The
///   skipped span replaces whatever turn spans it covers: a turn cut inside
///   it (the midpoint of a pause between two speakers, say) goes, and a
///   turn span it overlaps is split at its edge. Mel frames past the end
///   of the activity are transcribed, since nothing says they are silent,
///   except for less than one diarizer frame's worth, which belongs to the
///   last frame.
/// - **Always**, a transcribed span longer than `max_frames` is split by
///   the silence rule inside it, and the spans cover `0..mel.valid()`
///   exactly, in order, none of them empty. A transcribed span may be
///   shorter than a recognizer's shortest input; padding it is the
///   recognizer's business, and dropping it would lose its words.
///
/// # Errors
///
/// [`AudioError::BadSpec`] for a rule that cannot make a plan (see
/// [`OfflineRule`]) or a non-positive `mel_frame_secs`, and for a plan
/// that does not tile the clip, which would be a bug here.
pub fn plan_offline(
    mel: &MelFrames,
    activity: Option<&SpeakerActivity>,
    mel_frame_secs: f64,
    rule: &OfflineRule,
) -> Result<Vec<Span>, AudioError> {
    rule.check(mel_frame_secs)?;
    let valid = mel.valid();
    if valid == 0 {
        return Ok(Vec::new());
    }
    let marks = SilenceRule {
        max: rule.max_frames - rule.search_frames,
        search: rule.search_frames,
        window: rule.window_frames,
    };
    let cuts = match activity {
        Some(a) => turn_cuts(a, mel, mel_frame_secs),
        None => silence_cuts(mel, marks.max, marks.search, marks.window),
    };
    let skipped = match activity {
        Some(a) if rule.skip_silence => skipped_silences(a, valid, mel_frame_secs, rule),
        _ => Vec::new(),
    };
    // Every edge either list makes; each piece between two is skipped when
    // a skipped stretch holds it, and neighbouring skipped pieces merge.
    let mut edges = cuts;
    edges.extend(skipped.iter().flat_map(|s| [s.lo, s.hi]));
    edges.extend([0, valid]);
    edges.sort_unstable();
    edges.dedup();
    let mut pieces: Vec<Span> = Vec::with_capacity(edges.len());
    for w in edges.windows(2) {
        let (start, end) = (w[0], w[1]);
        let transcribe = !skipped.iter().any(|s| s.lo <= start && end <= s.hi);
        match pieces.last_mut() {
            Some(last) if !last.transcribe && !transcribe => last.end = end,
            _ => pieces.push(Span {
                start,
                end,
                transcribe,
            }),
        }
    }
    let energy = frame_energy(mel);
    let mut out = Vec::with_capacity(pieces.len());
    let mut inner = Vec::new();
    for piece in pieces {
        if !piece.transcribe || piece.end - piece.start <= rule.max_frames {
            out.push(piece);
            continue;
        }
        inner.clear();
        let whole = Stretch {
            lo: piece.start,
            hi: piece.end,
        };
        cuts_between(&energy, whole, &marks, &mut inner);
        let mut start = piece.start;
        for end in inner.iter().copied().chain(std::iter::once(piece.end)) {
            out.push(Span {
                start,
                end,
                transcribe: true,
            });
            start = end;
        }
    }
    check_tiling(&out, valid)?;
    Ok(out)
}

/// Refuses spans that do not cover `0..valid` exactly, in order, none empty.
fn check_tiling(spans: &[Span], valid: usize) -> Result<(), AudioError> {
    let mut at = 0;
    for s in spans {
        if s.start != at || s.end <= s.start {
            return Err(AudioError::BadSpec {
                reason: format!(
                    "an offline plan must tile the clip: span {}..{} where one from {at} was due",
                    s.start, s.end
                ),
            });
        }
        at = s.end;
    }
    if at == valid {
        Ok(())
    } else {
        Err(AudioError::BadSpec {
            reason: format!("an offline plan must tile the clip: it ends at {at}, not {valid}"),
        })
    }
}

/// The mel stretches a long silence in `activity` lets the plan skip,
/// increasing and disjoint, margins already taken off.
fn skipped_silences(
    activity: &SpeakerActivity,
    valid: usize,
    mel_frame_secs: f64,
    rule: &OfflineRule,
) -> Vec<Stretch> {
    let fs = activity.frame_secs();
    let frames = activity.frames();
    let pad = frames_in(rule.pad_secs, mel_frame_secs);
    // Mel frames per diarizer frame, at least one.
    let per_frame = frames_in(fs, mel_frame_secs).max(1);
    let to_mel = |t: usize| usize_floor((f64_of(t) * fs / mel_frame_secs).round()).min(valid);
    let silent =
        |t: usize| (0..activity.speakers()).all(|s| activity.prob(t, s) < rule.speech_floor);
    let mut out = Vec::new();
    let mut t = 0;
    while t < frames {
        if !silent(t) {
            t += 1;
            continue;
        }
        let first = t;
        while t < frames && silent(t) {
            t += 1;
        }
        // Measured in the diarizer's own frames: 3 s over 0.08 s is 37.5, so
        // 37 frames fall short and 38 are enough, with no rounding between.
        if f64_of(t - first) * fs + 1e-9 < rule.skip_silence_secs {
            continue;
        }
        let mut hi = to_mel(t);
        if t == frames && hi + per_frame > valid {
            hi = valid;
        }
        let lo = if first == 0 { 0 } else { to_mel(first) + pad };
        let hi = if hi == valid {
            valid
        } else {
            hi.saturating_sub(pad)
        };
        if lo < hi {
            out.push(Stretch { lo, hi });
        }
    }
    out
}

/// The window arithmetic of the silence rule, every field in mel frames.
struct SilenceRule {
    max: usize,
    search: usize,
    window: usize,
}

/// A half-open range of mel frames, `lo..hi`.
#[derive(Clone, Copy)]
struct Stretch {
    lo: usize,
    hi: usize,
}

/// A run of diarizer frames, `start..end`, owned by `who` (`None`: silence).
#[derive(Debug, Clone, Copy)]
struct Run {
    who: Option<usize>,
    start: usize,
    end: usize,
}

impl Run {
    fn len(&self) -> usize {
        self.end - self.start
    }
}

/// `secs` in whole frames of `frame_secs`, rounded.
fn frames_in(secs: f64, frame_secs: f64) -> usize {
    usize_floor((secs / frame_secs).round())
}

/// Each valid mel row's mean value: the lower, the quieter.
fn frame_energy(mel: &MelFrames) -> Vec<f64> {
    let n = f64_of(mel.n_mels());
    (0..mel.valid())
        .map(|t| mel.row(t).iter().map(|&v| f64::from(v)).sum::<f64>() / n)
        .collect()
}

/// Appends the silence rule's cuts strictly inside `span` to `out`.
fn cuts_between(energy: &[f64], span: Stretch, rule: &SilenceRule, out: &mut Vec<usize>) {
    assert!(rule.max > 0, "a chunk of zero frames is not a plan");
    assert!(rule.window > 0, "a window of zero frames measures nothing");
    let window = rule.window;
    let mut last = span.lo;
    loop {
        let mark = last + rule.max;
        if mark >= span.hi {
            return;
        }
        // Candidates start after the last cut and leave a whole window before
        // the end of the span; with none (a span barely past one chunk and a
        // window wider than what is left), the mark itself is the cut.
        let first = mark.saturating_sub(rule.search).max(last + 1);
        let final_start = (mark + rule.search).min(span.hi.saturating_sub(window));
        let mut best = (mark, f64::INFINITY);
        for s in first..=final_start {
            // Each window is summed afresh, not by a running difference, so
            // two windows of equal frames compare exactly equal and the tie
            // goes to the earlier one.
            let sum: f64 = energy[s..s + window].iter().sum();
            if sum < best.1 {
                best = (s, sum);
            }
        }
        out.push(best.0);
        last = best.0;
    }
}

/// Runs of the dominant speaker, silence included, covering every frame.
fn dominant_runs(activity: &SpeakerActivity) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for t in 0..activity.frames() {
        let mut who = None;
        let mut loudest = DOMINANT_ABOVE;
        for s in 0..activity.speakers() {
            let p = activity.prob(t, s);
            if p > loudest {
                loudest = p;
                who = Some(s);
            }
        }
        match runs.last_mut() {
            Some(last) if last.who == who => last.end = t + 1,
            _ => runs.push(Run {
                who,
                start: t,
                end: t + 1,
            }),
        }
    }
    runs
}

/// Merges every speaker run shorter than `min` frames into its longer
/// neighbour, in one pass from the start. A run only ever grows, so a run
/// long enough when it is reached stays long enough, and a short one is
/// judged with whatever it has already absorbed.
fn absorb_short(mut input: Vec<Run>, min: usize) -> Vec<Run> {
    let mut out: Vec<Run> = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        let mut run = input[i];
        if run.who.is_some() && run.len() < min {
            let before = out.last().map(Run::len);
            let after = input.get(i + 1).map(Run::len);
            match (before, after) {
                (Some(b), Some(a)) if a > b => {
                    // Handed forward, the next run is judged with this one
                    // inside it.
                    input[i + 1].start = run.start;
                    continue;
                }
                (None, Some(_)) => {
                    input[i + 1].start = run.start;
                    continue;
                }
                (Some(_), _) => {
                    run.who = out.last().and_then(|r| r.who);
                }
                (None, None) => {}
            }
        }
        match out.last_mut() {
            Some(last) if last.who == run.who => last.end = run.end,
            _ => out.push(run),
        }
    }
    out
}

#[cfg(test)]
#[path = "chunks_tests.rs"]
mod tests;
