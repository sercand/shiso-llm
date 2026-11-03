//! Word times: the decoder's transcript force-aligned against the timestamp
//! head's log-probabilities.
//!
//! The reference does not read times off the head's own greedy output; it
//! takes the decoder's words, tokenizes each speaker's words with the head's
//! tokenizer, and finds the best CTC path through them. Speaker activity
//! steers it: each speaker tag is matched to one of the speaker branch's
//! columns, the path runs only over the frames where that column is awake,
//! and its log activity is added to every non-blank step. The path is
//! searched in a band around the trigram anchors the head's greedy output
//! shares with the target. Each word then starts at its first frame, ends at
//! its last frame or a second later, whichever is first, and is refined
//! against the 10 ms diarization: an onset a little after its speaker's
//! segment starts moves back to it, and an end is capped by the next word or
//! the segment's end.
//!
//! The times are graded to the millisecond against the reference's, so every
//! float operation here is the reference's own, in its order and precision:
//! its log is a rational fit, its interpolation positions are computed the
//! way its tensor library computes them, and its dot products are blocked
//! the way its linear algebra library blocks them.
//!
//! Only word units are built: a Chinese, Japanese or Thai transcript, which
//! the reference splits into characters, is refused.

#![expect(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::float_cmp,
    clippy::needless_range_loop,
    reason = "frame and state counts are far below 2^24; the names, the exact float comparisons and the index loops are the reference's, kept so its ties break the same way"
)]

use std::collections::BTreeMap;

use shiso_audio::transcript::Word;

use super::spm::{CtcTokenizer, PieceId};

/// Why a transcript has no word times.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AlignError {
    /// The transcript holds no words.
    #[error("the transcript is empty")]
    Empty,
    /// A script the reference times character by character.
    #[error("word times for a {script} transcript are not built")]
    Script {
        /// The script's name.
        script: &'static str,
    },
    /// The tokenizer's word-by-word and whole-stream encodings disagree.
    #[error("per-word and whole-stream tokenization disagree")]
    Tokenization,
    /// A word the tokenizer encodes to nothing.
    #[error("the word {word:?} tokenizes to nothing")]
    EmptyWord {
        /// The word.
        word: String,
    },
    /// A speaker's words need more frames than the clip has.
    #[error("speaker {tag:?}'s transcript is too long for {frames} timestamp frames")]
    TooLong {
        /// The speaker tag, `None` for words before the first tag.
        tag: Option<u32>,
        /// The clip's timestamp frames.
        frames: usize,
    },
    /// Inputs of mismatched sizes.
    #[error("inconsistent alignment inputs: {reason}")]
    Inputs {
        /// What does not fit.
        reason: String,
    },
    /// The best path skipped a word, which a legal path cannot.
    #[error("the path did not visit {word:?}")]
    Unvisited {
        /// The word.
        word: String,
    },
}

// ---- transcript ----

/// A whitespace character as the reference's `str.isspace` reads it.
const fn is_space(c: char) -> bool {
    matches!(
        c as u32,
        0x09..=0x0d
            | 0x1c..=0x20
            | 0x85
            | 0xa0
            | 0x1680
            | 0x2000..=0x200a
            | 0x2028
            | 0x2029
            | 0x202f
            | 0x205f
            | 0x3000
    )
}

/// `text` split on runs of whitespace.
fn split_words(text: &str) -> Vec<&str> {
    text.split(is_space).filter(|w| !w.is_empty()).collect()
}

/// The first code point of each run of ten decimal digits (Unicode 15.0).
const DIGIT_ZEROS: [u32; 68] = [
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
    0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
    0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
    0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650,
    0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x11f50, 0x16a60, 0x16ac0,
    0x16b50, 0x1d7ce, 0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e4f0, 0x1e950,
    0x1fbf0,
];

/// A decimal digit's value, any script.
fn digit_value(c: char) -> Option<u32> {
    let cp = c as u32;
    let at = DIGIT_ZEROS.partition_point(|&z| z <= cp).checked_sub(1)?;
    let d = cp - DIGIT_ZEROS[at];
    (d < 10).then_some(d)
}

/// A speaker tag `<spk:N>` at the start of `text`, case-insensitive the way
/// the reference's pattern folds case (`ſ` matches `s`, the Kelvin sign
/// matches `k`): its byte length and its number, saturated.
fn speaker_tag_at(text: &str) -> Option<(usize, u32)> {
    let rest = text.strip_prefix('<')?;
    let mut chars = rest.char_indices();
    let mut want = |ok: &dyn Fn(char) -> bool| chars.next().filter(|(_, c)| ok(*c)).map(|_| ());
    want(&|c| matches!(c, 's' | 'S' | '\u{17f}'))?;
    want(&|c| matches!(c, 'p' | 'P'))?;
    want(&|c| matches!(c, 'k' | 'K' | '\u{212a}'))?;
    want(&|c| c == ':')?;
    let mut value = 0u64;
    let mut digits = 0;
    let mut end = None;
    for (i, c) in chars {
        if let Some(d) = digit_value(c) {
            value = (value * 10 + u64::from(d)).min(u64::from(i32::MAX.unsigned_abs()));
            digits += 1;
        } else {
            end = (c == '>').then_some(i);
            break;
        }
    }
    let end = end.filter(|_| digits > 0)?;
    Some((1 + end + 1, value as u32))
}

/// A transcript word and the speaker tag in force for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TranscriptWord {
    /// The word as written.
    pub word: String,
    /// The `<spk:N>` in force; `Some(0)` throughout a transcript without
    /// tags, `None` before the first tag of one with them.
    pub tag: Option<u32>,
}

/// The transcript's words: em dashes separate words, and each `<spk:N>`
/// starts a speaker turn.
#[must_use]
pub fn transcript_words(transcript: &str) -> Vec<TranscriptWord> {
    let text = transcript.replace('\u{2014}', " ");
    let mut parts: Vec<(&str, Option<u32>)> = Vec::new();
    let mut cursor = 0;
    let mut i = 0;
    while i < text.len() {
        if let Some((len, tag)) = speaker_tag_at(&text[i..]) {
            parts.push((&text[cursor..i], Some(tag)));
            cursor = i + len;
            i = cursor;
        } else {
            i += text[i..].chars().next().map_or(1, char::len_utf8);
        }
    }
    let tagged = !parts.is_empty();
    parts.push((&text[cursor..], None));
    let mut tag = if tagged { None } else { Some(0) };
    let mut words = Vec::new();
    for (part, next) in parts {
        words.extend(split_words(part).into_iter().map(|w| TranscriptWord {
            word: w.to_owned(),
            tag,
        }));
        tag = next;
    }
    words
}

/// Code point ranges of the scripts the reference times by character.
const HANGUL: [(u32, u32); 7] = [
    (0x1100, 0x11FF),
    (0x302E, 0x302F),
    (0x3131, 0x318E),
    (0xA960, 0xA97C),
    (0xAC00, 0xD7A3),
    (0xD7B0, 0xD7C6),
    (0xD7CB, 0xD7FB),
];
const KANA: [(u32, u32); 12] = [
    (0x3041, 0x3096),
    (0x309B, 0x30FF),
    (0x31F0, 0x31FF),
    (0x1AFF0, 0x1AFF3),
    (0x1AFF5, 0x1AFFB),
    (0x1AFFD, 0x1AFFE),
    (0x1B000, 0x1B001),
    (0x1B11F, 0x1B122),
    (0x1B132, 0x1B132),
    (0x1B150, 0x1B152),
    (0x1B155, 0x1B155),
    (0x1B164, 0x1B167),
];
const HAN: [(u32, u32); 13] = [
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFA6D),
    (0xFA70, 0xFAD9),
    (0x20000, 0x2A6DF),
    (0x2A700, 0x2B739),
    (0x2B740, 0x2B81D),
    (0x2B820, 0x2CEA1),
    (0x2CEB0, 0x2EBE0),
    (0x2EBF0, 0x2EE5D),
    (0x2F800, 0x2FA1D),
    (0x30000, 0x3134A),
    (0x31350, 0x323AF),
];
const THAI: [(u32, u32); 2] = [(0xE01, 0xE3A), (0xE3F, 0xE5B)];

fn in_ranges(ranges: &[(u32, u32)], c: char) -> bool {
    let cp = c as u32;
    ranges.iter().any(|&(a, b)| (a..=b).contains(&cp))
}

/// A word that is an event tag, `<...>` with no other angle bracket.
fn is_event_tag(word: &str) -> bool {
    word.len() >= 3
        && word.starts_with('<')
        && word.ends_with('>')
        && !word[1..word.len() - 1].contains(['<', '>'])
}

/// The script the reference times by character, if the words hold one:
/// Korean keeps its words, so any Hangul settles it first.
fn character_script(words: &[TranscriptWord]) -> Option<&'static str> {
    let (mut ko, mut ja, mut zh, mut th) = (false, false, false, false);
    for w in words.iter().filter(|w| !is_event_tag(&w.word)) {
        for c in w.word.chars() {
            ko |= in_ranges(&HANGUL, c);
            ja |= in_ranges(&KANA, c);
            zh |= in_ranges(&HAN, c);
            th |= in_ranges(&THAI, c);
        }
    }
    if ko {
        None
    } else if ja {
        Some("Japanese")
    } else if zh {
        Some("Chinese")
    } else if th {
        Some("Thai")
    } else {
        None
    }
}

// ---- targets ----

/// One speaker's words and their blank-expanded CTC target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpeakerStream {
    /// The speaker tag.
    pub tag: Option<u32>,
    /// Indices into [`Targets::words`], in transcript order.
    pub words: Vec<usize>,
    /// Columns of [`Targets::columns`]: blank (0), piece, blank, ..., blank.
    pub labels: Vec<usize>,
    /// The position in `words` each state belongs to; `None` for blanks.
    pub state_word: Vec<Option<usize>>,
}

/// A transcript's alignment targets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Targets {
    /// The transcript's words.
    pub words: Vec<TranscriptWord>,
    /// One stream per speaker tag, in order of first appearance.
    pub streams: Vec<SpeakerStream>,
    /// The head classes the alignment reads: the blank, then every piece the
    /// streams use, ascending.
    pub columns: Vec<PieceId>,
}

/// Frames a target of `labels` needs: one per piece, one more between
/// repeated pieces.
fn minimum_frames(labels: &[usize]) -> usize {
    (1..labels.len())
        .step_by(2)
        .map(|s| 1 + usize::from(s >= 3 && labels[s] == labels[s - 2]))
        .sum()
}

/// The targets of `transcript` for a clip of `frames` timestamp frames.
///
/// # Errors
///
/// [`AlignError`] where the reference rejects the transcript: no words, a
/// character-timed script, a tokenization that does not split at the words,
/// or a speaker whose words need more frames than there are.
pub fn targets(tok: &CtcTokenizer, transcript: &str, frames: usize) -> Result<Targets, AlignError> {
    let words = transcript_words(transcript);
    if words.is_empty() {
        return Err(AlignError::Empty);
    }
    if let Some(script) = character_script(&words) {
        return Err(AlignError::Script { script });
    }
    let mut streams: Vec<SpeakerStream> = Vec::new();
    for (w, word) in words.iter().enumerate() {
        if let Some(s) = streams.iter_mut().find(|s| s.tag == word.tag) {
            s.words.push(w);
        } else {
            streams.push(SpeakerStream {
                tag: word.tag,
                words: vec![w],
                labels: Vec::new(),
                state_word: Vec::new(),
            });
        }
    }
    // Every stream is tokenized before any target is built.
    let mut ids: Vec<Vec<Vec<PieceId>>> = Vec::with_capacity(streams.len());
    for s in &streams {
        let texts: Vec<&str> = s.words.iter().map(|&w| words[w].word.as_str()).collect();
        let whole = tok.encode(&texts.join(" "));
        let mut by_word = Vec::with_capacity(texts.len());
        for text in &texts {
            let piece_ids = tok.encode(text);
            if piece_ids.is_empty() {
                return Err(AlignError::EmptyWord {
                    word: (*text).to_owned(),
                });
            }
            by_word.push(piece_ids);
        }
        if whole.is_empty() || by_word.concat() != whole {
            return Err(AlignError::Tokenization);
        }
        ids.push(by_word);
    }
    let mut used: Vec<PieceId> = ids.iter().flatten().flatten().copied().collect();
    used.sort_unstable();
    used.dedup();
    let column = |id: PieceId| 1 + used.partition_point(|&u| u < id);
    for (s, by_word) in streams.iter_mut().zip(&ids) {
        s.labels.push(0);
        s.state_word.push(None);
        for (w, pieces) in by_word.iter().enumerate() {
            for &id in pieces {
                s.labels.extend([column(id), 0]);
                s.state_word.extend([Some(w), None]);
            }
        }
        if minimum_frames(&s.labels) > frames {
            return Err(AlignError::TooLong { tag: s.tag, frames });
        }
    }
    let blank = PieceId(u32::try_from(tok.len()).map_err(|_| AlignError::Inputs {
        reason: "a vocabulary past u32".into(),
    })?);
    let mut columns = vec![blank];
    columns.extend(used);
    Ok(Targets {
        words,
        streams,
        columns,
    })
}

// ---- head windows ----

/// A window of the head over the clip's encoder frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadWindow {
    /// Frames kept from this window.
    pub start: usize,
    /// One past the last frame kept.
    pub end: usize,
    /// The first frame the head reads.
    pub context_start: usize,
    /// One past the last frame the head reads.
    pub context_end: usize,
}

/// The head's windows over `frames` encoder frames: one when they fit in
/// `core`, else cores of `core` frames read with `left` and `right` frames of
/// context, clipped to the clip.
#[must_use]
pub fn head_windows(frames: usize, core: usize, left: usize, right: usize) -> Vec<HeadWindow> {
    if frames == 0 {
        return Vec::new();
    }
    if core == 0 || frames <= core {
        return vec![HeadWindow {
            start: 0,
            end: frames,
            context_start: 0,
            context_end: frames,
        }];
    }
    (0..frames)
        .step_by(core)
        .map(|start| {
            let end = (start + core).min(frames);
            HeadWindow {
                start,
                end,
                context_start: start.saturating_sub(left),
                context_end: (end + right).min(frames),
            }
        })
        .collect()
}

// ---- the reference's numerics ----

/// `log(x)` as the reference's array library computes it in f32: a degree-5
/// rational fit of `log(m)` for `m` in `(√½, √2]`, evaluated with fused
/// multiply-adds, plus the exponent times `ln 2`.
#[expect(
    clippy::excessive_precision,
    clippy::unreadable_literal,
    reason = "the coefficients are the fit's, digit for digit"
)]
#[must_use]
pub fn ref_log(x: f32) -> f32 {
    const P: [f32; 6] = [
        0.0,
        9.999999999999998702521e-01,
        2.112677543073053063722e+00,
        1.480000633576506585156e+00,
        3.808837741388407920751e-01,
        2.589979117907922693523e-02,
    ];
    const Q: [f32; 6] = [
        1.0,
        2.612677543073109236779e+00,
        2.453006071784736363091e+00,
        9.864942958519418960339e-01,
        1.546476374983906719538e-01,
        5.875095259e-03,
    ];
    const LN2: f32 = std::f32::consts::LN_2;
    const SQRT_HALF: f32 = std::f32::consts::FRAC_1_SQRT_2;
    if x.is_nan() {
        return x;
    }
    if x < 0.0 {
        return -f32::NAN;
    }
    if x == 0.0 {
        return f32::NEG_INFINITY;
    }
    if x.is_infinite() {
        return x;
    }
    let denormal = x < f32::MIN_POSITIVE;
    let x = if denormal {
        x * f32::from_bits(0x7180_0000)
    } else {
        x
    };
    let mut exponent = ((x.to_bits() >> 23) as i32 - 0x7e) as f32;
    if denormal {
        exponent -= 100.0;
    }
    let mut m = f32::from_bits((x.to_bits() & 0x7f_ffff) | (126 << 23));
    if m <= SQRT_HALF {
        m += m;
        exponent -= 1.0;
    }
    m -= 1.0;
    let mut num = P[5].mul_add(m, P[4]);
    let mut den = Q[5].mul_add(m, Q[4]);
    for k in (0..=3).rev() {
        num = num.mul_add(m, P[k]);
        den = den.mul_add(m, Q[k]);
    }
    exponent.mul_add(LN2, num / den)
}

/// Element `idx` of an evenly spaced f32 ramp from `start` to `end` in
/// `steps` points, as the reference's tensor library computes it: the lower
/// half stepped up from the start, the upper half fused down from the end.
fn ramp(start: f32, end: f32, steps: usize, idx: usize) -> f32 {
    if steps == 1 {
        return start;
    }
    let step = (end - start) / (steps - 1) as f32;
    if idx < steps / 2 {
        start + step * idx as f32
    } else {
        (-step).mul_add((steps - idx - 1) as f32, end)
    }
}

/// Speaker probabilities `[rows][speakers]` clamped to `[0, 1]` and resampled
/// to `frames` rows by linear interpolation.
fn resample_speakers(probs: &[f32], rows: usize, speakers: usize, frames: usize) -> Vec<f32> {
    let clamped: Vec<f32> = probs.iter().map(|v| v.clamp(0.0, 1.0)).collect();
    if rows == frames {
        return clamped;
    }
    let mut out = vec![0.0f32; frames * speakers];
    for t in 0..frames {
        let position = ramp(0.0, (rows - 1) as f32, frames, t);
        let (lower_f, upper_f) = (position.floor(), position.ceil());
        let fraction = position - lower_f;
        let (lower, upper) = (lower_f as usize, upper_f as usize);
        for c in 0..speakers {
            let a = clamped[lower * speakers + c] * (1.0 - fraction);
            let b = clamped[upper * speakers + c] * fraction;
            out[t * speakers + c] = if rows == 1 { clamped[c] } else { a + b };
        }
    }
    out
}

/// Pairwise summation as the reference's array library reduces a
/// contiguous axis.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0, |s, v| s + v);
    }
    if n <= 128 {
        let mut r = [0.0f64; 8];
        r.copy_from_slice(&a[..8]);
        let mut i = 8;
        while i < n - n % 8 {
            for k in 0..8 {
                r[k] += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for v in &a[i..] {
            res += v;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// A dot product blocked as the reference's linear algebra library blocks
/// it: one fused chain per block of 256, a remainder under 512 split in
/// halves rounded up to 4, the blocks summed in order.
fn blocked_dot(a: &[f64], b: &[f64]) -> f64 {
    const BLOCK: usize = 256;
    const UNROLL: usize = 4;
    let n = a.len();
    let mut c = 0.0f64;
    let mut ls = 0;
    while ls < n {
        let mut block = n - ls;
        if block >= 2 * BLOCK {
            block = BLOCK;
        } else if block > BLOCK {
            block = (block / 2).div_ceil(UNROLL) * UNROLL;
        }
        let acc = (ls..ls + block).fold(0.0f64, |acc, k| a[k].mul_add(b[k], acc));
        c = if ls == 0 { acc } else { c + acc };
        ls += block;
    }
    c
}

/// The minimum-cost assignment of a `rows x cols` cost matrix by shortest
/// augmenting paths: the assigned `(row, column)` pairs, rows ascending, ties
/// broken as the reference breaks them.
fn assignment(cost_in: &[f64], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    if rows == 0 || cols == 0 {
        return Vec::new();
    }
    let transpose = cols < rows;
    let (nr, nc) = if transpose {
        (cols, rows)
    } else {
        (rows, cols)
    };
    let mut cost = cost_in.to_vec();
    if transpose {
        for i in 0..rows {
            for j in 0..cols {
                cost[j * rows + i] = cost_in[i * cols + j];
            }
        }
    }
    let inf = f64::INFINITY;
    let mut u = vec![0.0f64; nr];
    let mut v = vec![0.0f64; nc];
    let mut shortest = vec![inf; nc];
    let mut path = vec![usize::MAX; nc];
    let mut col4row = vec![usize::MAX; nr];
    let mut row4col = vec![usize::MAX; nc];
    let mut remaining = vec![0usize; nc];
    let mut sr = vec![false; nr];
    let mut sc = vec![false; nc];
    for cur in 0..nr {
        let mut min_val = 0.0f64;
        let mut i = cur;
        let mut sink = usize::MAX;
        let mut num_remaining = nc;
        for (it, r) in remaining.iter_mut().enumerate() {
            *r = nc - it - 1;
        }
        sr.fill(false);
        sc.fill(false);
        shortest.fill(inf);
        while sink == usize::MAX {
            let mut index = usize::MAX;
            let mut lowest = inf;
            sr[i] = true;
            for it in 0..num_remaining {
                let j = remaining[it];
                let r = min_val + cost[i * nc + j] - u[i] - v[j];
                if r < shortest[j] {
                    path[j] = i;
                    shortest[j] = r;
                }
                if shortest[j] < lowest || (shortest[j] == lowest && row4col[j] == usize::MAX) {
                    lowest = shortest[j];
                    index = it;
                }
            }
            min_val = lowest;
            if min_val == inf || index == usize::MAX {
                // An infeasible matrix: finite costs never get here.
                return Vec::new();
            }
            let j = remaining[index];
            if row4col[j] == usize::MAX {
                sink = j;
            } else {
                i = row4col[j];
            }
            sc[j] = true;
            num_remaining -= 1;
            remaining[index] = remaining[num_remaining];
        }
        u[cur] += min_val;
        for r in 0..nr {
            if sr[r] && r != cur {
                u[r] += min_val - shortest[col4row[r]];
            }
        }
        for c in 0..nc {
            if sc[c] {
                v[c] -= min_val - shortest[c];
            }
        }
        let mut j = sink;
        loop {
            let r = path[j];
            row4col[j] = r;
            std::mem::swap(&mut col4row[r], &mut j);
            if r == cur {
                break;
            }
        }
    }
    if transpose {
        let mut order: Vec<usize> = (0..nr).collect();
        order.sort_by_key(|&k| col4row[k]);
        order.into_iter().map(|k| (col4row[k], k)).collect()
    } else {
        (0..nr).map(|r| (r, col4row[r])).collect()
    }
}

// ---- alignment ----

/// The speaker column each stream is matched to: the head's greedy onsets
/// counted per column while it is active, compared with each stream's
/// target pieces (an IDF-weighted cosine of square-root counts, less a
/// penalty on the log ratio of the totals), matched by least cost.
fn speaker_mapping(
    greedy: &[usize],
    columns: usize,
    streams: &[SpeakerStream],
    speaker: &[f32],
    speakers: usize,
    threshold: f32,
) -> Vec<Option<usize>> {
    let mut mapping = vec![None; streams.len()];
    if streams.is_empty() || speakers == 0 {
        return mapping;
    }
    let (c_n, s_n, n) = (columns, speakers, streams.len());
    let mut observed = vec![0i64; s_n * c_n];
    let mut expected = vec![0i64; n * c_n];
    for (t, &g) in greedy.iter().enumerate() {
        if g == 0 || (t > 0 && g == greedy[t - 1]) {
            continue;
        }
        for c in 0..s_n {
            if speaker[t * s_n + c] >= threshold {
                observed[c * c_n + g] += 1;
            }
        }
    }
    for (s, stream) in streams.iter().enumerate() {
        for k in (1..stream.labels.len()).step_by(2) {
            expected[s * c_n + stream.labels[k]] += 1;
        }
    }
    let idf: Vec<f64> = (0..c_n)
        .map(|k| {
            let present = (0..s_n).filter(|c| observed[c * c_n + k] > 0).count();
            (s_n as f64 / (1 + present) as f64).ln_1p()
        })
        .collect();
    let a: Vec<f64> = (0..n * c_n)
        .map(|i| (expected[i] as f64 * idf[i % c_n]).sqrt())
        .collect();
    let b: Vec<f64> = (0..s_n * c_n)
        .map(|i| (observed[i] as f64 * idf[i % c_n]).sqrt())
        .collect();
    let norm = |row: &[f64]| {
        let squares: Vec<f64> = row.iter().map(|v| v * v).collect();
        pairwise_sum(&squares).sqrt()
    };
    let norm_b: Vec<f64> = (0..s_n).map(|c| norm(&b[c * c_n..(c + 1) * c_n])).collect();
    let total_b: Vec<i64> = (0..s_n)
        .map(|c| observed[c * c_n..(c + 1) * c_n].iter().sum())
        .collect();
    let mut cost = vec![0.0f64; n * s_n];
    for s in 0..n {
        let row = &a[s * c_n..(s + 1) * c_n];
        let norm_a = norm(row);
        let total_a: i64 = expected[s * c_n..(s + 1) * c_n].iter().sum();
        for c in 0..s_n {
            let mut similarity =
                blocked_dot(row, &b[c * c_n..(c + 1) * c_n]) / 1e-8f64.max(norm_a * norm_b[c]);
            similarity -= 0.1 * ((total_a + 1) as f64 / (total_b[c] + 1) as f64).ln().abs();
            cost[s * s_n + c] = -similarity;
        }
    }
    for (row, column) in assignment(&cost, n, s_n) {
        mapping[row] = Some(column);
    }
    mapping
}

/// First start and count of every piece trigram in `tokens`.
fn trigrams(tokens: &[usize]) -> BTreeMap<[usize; 3], (usize, usize)> {
    let mut table = BTreeMap::new();
    for i in 0..tokens.len().saturating_sub(2) {
        table
            .entry([tokens[i], tokens[i + 1], tokens[i + 2]])
            .or_insert((i, 0))
            .1 += 1;
    }
    table
}

/// The first state of the band at each searched frame: a line through the
/// unique trigram anchors the greedy onsets share with the target, kept
/// reachable from both ends.
fn band_starts(greedy: &[usize], target: &[usize], states: i64, half: i64, width: i64) -> Vec<i64> {
    let frames = greedy.len() as i64;
    let mut observed = Vec::new();
    let mut times = Vec::new();
    for (i, &g) in greedy.iter().enumerate() {
        if g != 0 && (i == 0 || g != greedy[i - 1]) {
            observed.push(g);
            times.push(i as i64);
        }
    }
    let (left, right) = (trigrams(&observed), trigrams(target));
    let mut pairs: Vec<(i64, i64)> = Vec::new();
    for (key, &(first, count)) in &left {
        if count != 1 {
            continue;
        }
        if let Some(&(at, 1)) = right.get(key) {
            pairs.push((times[first + 1], 2 * (at as i64 + 1) + 1));
        }
    }
    pairs.sort_unstable();
    // The longest chain increasing in state, by patience sorting.
    let mut tails: Vec<i64> = Vec::new();
    let mut tail_ids: Vec<usize> = Vec::new();
    let mut parent: Vec<Option<usize>> = Vec::new();
    for (i, &(_, s)) in pairs.iter().enumerate() {
        let j = tails.partition_point(|&t| t < s);
        parent.push(j.checked_sub(1).map(|p| tail_ids[p]));
        if j == tails.len() {
            tails.push(s);
            tail_ids.push(i);
        } else {
            tails[j] = s;
            tail_ids[j] = i;
        }
    }
    let mut chosen = Vec::new();
    let mut k = tail_ids.last().copied();
    while let Some(i) = k {
        chosen.push(pairs[i]);
        k = parent[i];
    }
    chosen.reverse();
    let mut xs = vec![0.0f64];
    let mut ys = vec![0.0f64];
    for &(t, s) in &chosen {
        if 0 < t && t < frames - 1 {
            xs.push(t as f64);
            ys.push(s as f64);
        }
    }
    xs.push((frames - 1) as f64);
    ys.push((states - 1) as f64);
    let n = xs.len();
    let mut starts: Vec<i64> = (0..frames)
        .map(|t| {
            let x = t as f64;
            let j = xs.partition_point(|&v| v <= x) - 1;
            let center = if j == n - 1 || xs[j] == x {
                ys[j]
            } else {
                (ys[j + 1] - ys[j]) / (xs[j + 1] - xs[j]) * (x - xs[j]) + ys[j]
            };
            let start = center.round_ties_even() as i64 - half;
            start.max(0).min(states - width)
        })
        .collect();
    let mut running = i64::MAX;
    for (t, start) in starts.iter_mut().enumerate() {
        let t = t as i64;
        running = running.min(*start - 2 * t);
        *start = (running + 2 * t).max(0.max(states - width - 2 * (frames - 1 - t)));
    }
    starts
}

/// The search's back-pointers, one per band cell: how many states the best
/// path into the cell advanced (0 stay, 1 step, 2 skip), four to a byte.
///
/// The table is every searched frame by the band's width, and an hour of
/// one speaker is some 90k frames by a band of a few thousand states, so a
/// byte a cell would be hundreds of megabytes per stream.
struct Moves {
    packed: Vec<u8>,
}

impl Moves {
    /// A table of `cells` moves, every one 0.
    fn new(cells: usize) -> Self {
        Self {
            packed: vec![0; cells.div_ceil(4)],
        }
    }

    /// Records `step` (at most 2) for `cell`, which holds 0 until then.
    fn set(&mut self, cell: usize, step: u8) {
        self.packed[cell / 4] |= step << (2 * (cell % 4));
    }

    /// The move recorded for `cell`.
    fn get(&self, cell: usize) -> u8 {
        (self.packed[cell / 4] >> (2 * (cell % 4))) & 0b11
    }
}

/// The best CTC path's state at each searched frame, or `None` when no
/// legal path fits the band.
struct Path {
    states: Vec<usize>,
}

/// The max-sum CTC search over the frames `positions` of `log_probs`
/// (`[frames][columns]`, blank in column 0), every state under 1000 searched
/// frames, else in a band of half-width `ceil(0.1 · states + 100)` around the
/// anchors. `prior` (per frame) is added to every non-blank state. Ties
/// prefer staying, then stepping, then skipping; f32 throughout.
fn banded_search(
    log_probs: &[f32],
    columns: usize,
    labels: &[usize],
    positions: &[usize],
    prior: Option<&[f32]>,
    greedy: &[usize],
) -> Option<Path> {
    let neg_inf = f32::NEG_INFINITY;
    let frames = positions.len() as i64;
    let states = labels.len() as i64;
    let half = if frames < 1000 {
        states
    } else {
        states.min((0.10 * states as f64 + 100.0).ceil() as i64)
    };
    let width = states.min(2 * half + 1);
    let target: Vec<usize> = labels.iter().skip(1).step_by(2).copied().collect();
    let needed = target.len() + target.windows(2).filter(|w| w[0] == w[1]).count();
    if frames < needed as i64 || frames == 0 {
        return None;
    }
    let searched: Vec<usize> = positions.iter().map(|&p| greedy[p]).collect();
    let starts = band_starts(&searched, &target, states, half, width);
    let row = |t: usize| &log_probs[positions[t] * columns..(positions[t] + 1) * columns];
    let w = width as usize;
    let mut previous = vec![neg_inf; w];
    let mut padded = vec![neg_inf; w + 4];
    let mut moves = Moves::new(frames as usize * w);
    for i in 0..w.min(2) {
        previous[i] = row(0)[labels[i]];
        if let Some(p) = prior
            && labels[i] != 0
        {
            previous[i] += p[positions[0]];
        }
    }
    let can_skip: Vec<bool> = (0..labels.len())
        .map(|s| s >= 2 && labels[s] != 0 && labels[s] != labels[s - 2])
        .collect();
    let at = |padded: &[f32], i: i64| {
        usize::try_from(i)
            .ok()
            .and_then(|i| padded.get(i))
            .copied()
            .unwrap_or(neg_inf)
    };
    for t in 1..frames as usize {
        let start = starts[t];
        let delta = start - starts[t - 1];
        padded[2..2 + w].copy_from_slice(&previous);
        let emissions = row(t);
        let p = prior.map_or(0.0, |p| p[positions[t]]);
        for i in 0..w {
            let ii = i as i64;
            let mut best = at(&padded, 2 + delta + ii);
            let mut step = 0u8;
            let one = at(&padded, 1 + delta + ii);
            if one > best {
                best = one;
                step = 1;
            }
            let two = at(&padded, delta + ii);
            if two > best && can_skip[(start + ii) as usize] {
                best = two;
                step = 2;
            }
            let label = labels[(start + ii) as usize];
            let mut score = best + emissions[label];
            if prior.is_some() && label != 0 {
                score += p;
            }
            previous[i] = score;
            moves.set(t * w + i, step);
        }
    }
    let last = starts[frames as usize - 1];
    let mut end: Option<(i64, f32)> = None;
    for candidate in [states - 1, states - 2] {
        let local = candidate - last;
        if local < 0 || local >= width {
            continue;
        }
        let score = previous[local as usize];
        if end.is_none_or(|(_, best)| score > best) {
            end = Some((candidate, score));
        }
    }
    let (mut state, score) = end?;
    if !score.is_finite() {
        return None;
    }
    let mut out = vec![0usize; frames as usize];
    for t in (0..frames as usize).rev() {
        out[t] = state as usize;
        if t > 0 {
            state -= i64::from(moves.get(t * w + (state - starts[t]) as usize));
        }
    }
    Some(Path { states: out })
}

/// One speaker's 10 ms activity run.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Segment {
    speaker: usize,
    start: f64,
    end: f64,
}

/// Every speaker's runs of 10 ms activity above one half, ends clipped to
/// the clip, sorted by start, end, speaker.
fn diarization_segments(
    activity10: &[f32],
    frames: usize,
    speakers: usize,
    frame_seconds: f64,
    duration: f64,
) -> Vec<Segment> {
    let mut segments = Vec::new();
    for c in 0..speakers {
        let mut begin = None;
        for t in 0..=frames {
            let active = t < frames && activity10[t * speakers + c] > 0.5;
            if active && begin.is_none() {
                begin = Some(t);
            }
            if !active && let Some(b) = begin.take() {
                let start = b as f64 * frame_seconds;
                let end = (t as f64 * frame_seconds).min(duration);
                if end > start {
                    segments.push(Segment {
                        speaker: c,
                        start,
                        end,
                    });
                }
            }
        }
    }
    segments.sort_by(|a, b| {
        a.start
            .total_cmp(&b.start)
            .then(a.end.total_cmp(&b.end))
            .then(a.speaker.cmp(&b.speaker))
    });
    segments
}

/// A word's place in time.
#[derive(Debug, Clone, Copy, PartialEq)]
struct WordRow {
    word: usize,
    tag: Option<u32>,
    start: f64,
    end: f64,
}

/// Onsets moved back to their speaker segment's start when they lag it by
/// at most `max_frames` frames (of at most 80 ms), and ends capped by the
/// next word in the segment, the segment's end, and that same limit.
fn refine(
    words: &mut [WordRow],
    mapping: &[(Option<u32>, Option<usize>)],
    diarization: &[Segment],
    step: f64,
    max_frames: usize,
) {
    let limit = max_frames as f64 * step.min(0.08);
    let mut by_speaker: BTreeMap<usize, Vec<(f64, f64)>> = BTreeMap::new();
    for s in diarization.iter().filter(|s| s.end > s.start) {
        by_speaker
            .entry(s.speaker)
            .or_default()
            .push((s.start, s.end));
    }
    for intervals in by_speaker.values_mut() {
        intervals.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
    }
    let none = Vec::new();
    let mut first = 0;
    while first < words.len() {
        let tag = words[first].tag;
        let last = first + words[first..].iter().take_while(|w| w.tag == tag).count();
        let group = &mut words[first..last];
        group.sort_by(|a, b| a.start.total_cmp(&b.start));
        let column = mapping
            .iter()
            .rev()
            .find(|(t, _)| *t == tag)
            .and_then(|(_, c)| *c);
        let intervals = column.and_then(|c| by_speaker.get(&c)).unwrap_or(&none);
        let member: Vec<Option<usize>> = group
            .iter()
            .map(|w| {
                let index = intervals
                    .partition_point(|iv| iv.0 <= w.start)
                    .checked_sub(1)?;
                (w.start < intervals[index].1).then_some(index)
            })
            .collect();
        let mut seen = vec![false; intervals.len()];
        for (word, m) in group.iter_mut().zip(&member) {
            if let Some(m) = *m
                && !seen[m]
            {
                let onset = intervals[m].0;
                if 0.0 < word.start - onset && word.start - onset <= limit {
                    word.start = onset;
                }
                seen[m] = true;
            }
        }
        let n = group.len();
        for k in 0..n {
            let cap = match member[k] {
                None => {
                    if k + 1 < n {
                        group[k + 1].start
                    } else {
                        group[k].end
                    }
                }
                Some(m) if k + 1 < n && member[k + 1] == Some(m) => group[k + 1].start,
                Some(m) => intervals[m].1,
            };
            let w = &mut group[k];
            w.end = w.start.max(cap.min(w.start + limit));
        }
        first = last;
    }
}

/// `x` rounded to `decimals` places the way a decimal print and re-read
/// rounds it: the decimal nearest the exact binary value, ties to even.
#[must_use]
pub fn round_decimals(x: f64, decimals: usize) -> f64 {
    if !x.is_finite() {
        return x;
    }
    format!("{x:.decimals$}").parse().unwrap_or(x)
}

/// The speaker branch's activity, as the alignment reads it.
#[derive(Debug, Clone, Copy)]
pub struct Activity<'a> {
    /// `[encoder frames][speakers]`, 80 ms rows.
    pub rows80: &'a [f32],
    /// `[mel frames][speakers]`, 10 ms rows.
    pub rows10: &'a [f32],
    /// Speakers per row.
    pub speakers: usize,
}

/// What one transcript is aligned against.
#[derive(Debug, Clone, Copy)]
pub struct AlignDesc<'a> {
    /// The transcript's targets.
    pub targets: &'a Targets,
    /// `[frames][targets.columns.len()]` log-probabilities.
    pub log_probs: &'a [f32],
    /// Timestamp frames.
    pub frames: usize,
    /// The speaker branch's activity; `None` aligns without speakers.
    pub activity: Option<Activity<'a>>,
    /// The clip's length, seconds.
    pub duration: f64,
}

/// The refinement cap: 12 encoder frames, in timestamp frames.
const MAX_WORD_ENCODER_FRAMES: usize = 12;
/// A word ends at most this long after it starts, seconds.
const MAX_WORD_SECONDS: f64 = 1.0;
/// A speaker column counts as active at or above this.
const ACTIVE: f32 = 0.5;
/// Frames where a stream's column is below this are not searched.
const AWAKE: f32 = 0.01;
/// The floor of the activity prior.
const PRIOR_FLOOR: f32 = 1.0e-6;
/// The 10 ms diarization's frame length.
const LABEL_SECONDS: f64 = 0.01;

/// The column each frame's log-probabilities peak in, the first on a tie
/// (a NaN wins and stops the scan, as the reference's argmax has it).
fn greedy_columns(log_probs: &[f32], columns: usize) -> Vec<usize> {
    log_probs
        .chunks_exact(columns)
        .map(|row| {
            let mut best = 0;
            for (c, v) in row.iter().enumerate() {
                if row[best].is_nan() {
                    break;
                }
                if v.is_nan() || *v > row[best] {
                    best = c;
                }
            }
            best
        })
        .collect()
}

/// Each of `stream`'s words from the first to the last frame the path
/// spends in its pieces, ending at most [`MAX_WORD_SECONDS`] after it starts.
fn word_rows(
    targets: &Targets,
    stream: &SpeakerStream,
    positions: &[usize],
    path: &Path,
    step: f64,
) -> Result<Vec<WordRow>, AlignError> {
    let mut first_last: Vec<Option<(usize, usize)>> = vec![None; stream.words.len()];
    for (&t, &state) in positions.iter().zip(&path.states) {
        if let Some(w) = stream.state_word[state] {
            first_last[w].get_or_insert((t, t)).1 = t;
        }
    }
    first_last
        .iter()
        .zip(&stream.words)
        .map(|(span, &word)| {
            let (f0, f1) = span.ok_or_else(|| AlignError::Unvisited {
                word: targets.words[word].word.clone(),
            })?;
            let start = f0 as f64 * step;
            Ok(WordRow {
                word,
                tag: stream.tag,
                start,
                end: ((f1 + 1) as f64 * step).min(start + MAX_WORD_SECONDS),
            })
        })
        .collect()
}

/// Refuses activity buffers that are not whole rows of every speaker, at
/// least one: a ragged or short buffer would be resampled from row
/// `rows - 1` of nothing.
fn whole_rows(a: &Activity<'_>) -> Result<(), AlignError> {
    if a.speakers == 0 {
        return Ok(());
    }
    for (name, buf) in [("80 ms", a.rows80), ("10 ms", a.rows10)] {
        if buf.is_empty() || !buf.len().is_multiple_of(a.speakers) {
            return Err(AlignError::Inputs {
                reason: format!(
                    "{} {name} activity values are not whole rows of {} speakers",
                    buf.len(),
                    a.speakers
                ),
            });
        }
    }
    Ok(())
}

/// The words of `desc.targets` timed against the head's log-probabilities,
/// times rounded to the millisecond and ordered by start, then end. A
/// speaker whose words have no legal path is left out, as the reference
/// leaves it out.
///
/// # Errors
///
/// [`AlignError::Inputs`] for buffers of the wrong size;
/// [`AlignError::Unvisited`] if a path skips a word.
pub fn align(desc: &AlignDesc<'_>, upsampling: usize) -> Result<Vec<Word>, AlignError> {
    let targets = desc.targets;
    let frames = desc.frames;
    let c_n = targets.columns.len();
    let bad = |reason: &str| AlignError::Inputs {
        reason: reason.to_owned(),
    };
    if frames == 0 || c_n == 0 || desc.log_probs.len() != frames * c_n || desc.duration <= 0.0 {
        return Err(bad("log-probabilities of the wrong size"));
    }
    let step = desc.duration / frames as f64;
    let greedy = greedy_columns(desc.log_probs, c_n);
    if let Some(a) = desc.activity {
        whole_rows(&a)?;
    }
    let (speaker, s_n) = match desc.activity {
        Some(a) if a.speakers > 0 => {
            let rows = a.rows80.len() / a.speakers;
            (
                Some(resample_speakers(a.rows80, rows, a.speakers, frames)),
                a.speakers,
            )
        }
        _ => (None, 0),
    };
    let mapping = speaker.as_ref().map_or_else(
        || vec![None; targets.streams.len()],
        |sp| speaker_mapping(&greedy, c_n, &targets.streams, sp, s_n, ACTIVE),
    );
    let mut words: Vec<WordRow> = Vec::new();
    let mut tag_columns = Vec::new();
    let mut prior = vec![0.0f32; frames];
    for (stream, &column) in targets.streams.iter().zip(&mapping) {
        tag_columns.push((stream.tag, column));
        let mut positions = Vec::new();
        for (t, p) in prior.iter_mut().enumerate() {
            let activity = match (&speaker, column) {
                (None, _) => 1.0,
                (Some(_), None) => 0.0,
                (Some(sp), Some(c)) => sp[t * s_n + c],
            };
            if activity >= AWAKE {
                positions.push(t);
            }
            *p = ref_log(activity.max(PRIOR_FLOOR));
        }
        let Some(path) = banded_search(
            desc.log_probs,
            c_n,
            &stream.labels,
            &positions,
            speaker.as_ref().map(|_| prior.as_slice()),
            &greedy,
        ) else {
            tracing::warn!(
                tag = ?stream.tag,
                words = stream.words.len(),
                "no legal timestamp path in the anchor band; these words go untimed"
            );
            continue;
        };
        words.extend(word_rows(targets, stream, &positions, &path, step)?);
    }
    let diarization = desc.activity.map_or_else(Vec::new, |a| {
        if a.speakers == 0 {
            return Vec::new();
        }
        diarization_segments(
            a.rows10,
            a.rows10.len() / a.speakers,
            a.speakers,
            LABEL_SECONDS,
            desc.duration,
        )
    });
    refine(
        &mut words,
        &tag_columns,
        &diarization,
        step,
        MAX_WORD_ENCODER_FRAMES * upsampling,
    );
    for w in &mut words {
        w.start = round_decimals(w.start, 3);
        w.end = round_decimals(w.end, 3);
    }
    words.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.end.total_cmp(&b.end)));
    Ok(words
        .into_iter()
        .map(|w| Word {
            text: targets.words[w.word].word.clone(),
            start: w.start,
            end: w.end,
        })
        .collect())
}

#[cfg(test)]
#[path = "align_tests.rs"]
pub(crate) mod tests;
