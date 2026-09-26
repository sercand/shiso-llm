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
//! Everything here is host arithmetic over frames the caller already has. It
//! is deterministic: the same frames always give the same cuts, ties going to
//! the earliest candidate, so a transcript does not change between runs.

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
        Span {
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
        cuts_between(&energy, Span { lo, hi }, &rule, &mut out);
        if hi < valid {
            out.push(hi);
        }
        lo = hi;
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
struct Span {
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
fn cuts_between(energy: &[f64], span: Span, rule: &SilenceRule, out: &mut Vec<usize>) {
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
