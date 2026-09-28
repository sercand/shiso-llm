//! One clip through the tower and the decoder, a chunk at a time.
//!
//! The model has no streaming encoder and no timestamps of its own, so a clip
//! is cut into chunks and each chunk is one prompt and one greedy decode with
//! no context from the one before, as the reference transcribes long audio.
//!
//! - **Where a chunk ends.** At the cuts the caller gave
//!   ([`StreamOptions::cuts`](shiso_audio::StreamOptions), a diarized job's
//!   speaker turns), or else by the silence rule of
//!   [`shiso_audio::chunks::silence_cuts`]: the quietest 100 ms within 5 s of
//!   each 30 s mark. The rule's first cut past the last one depends only on the
//!   35.1 s after it, so a chunk is transcribed as soon as those rows are in,
//!   and the cuts are exactly the rule's over the whole clip, whatever the push
//!   schedule.
//! - **The frontend.** The recognizer's frames stop at `log10`; each chunk's
//!   rows are floored at their own maximum and rescaled here
//!   ([`shiso_audio::log10_max_floor`]), because the reference extracts every
//!   chunk's features on their own. A chunk shorter than 0.5 s is padded
//!   with silent frames first, as the reference pads its waveform with zeros.
//!   The reference frames each chunk's samples, so its first and last frame
//!   see the chunk edge where these see the audio past it; the rows between
//!   are the same.
//! - **What is emitted.** The chunk's ids are decoded, collapsed and parsed as
//!   [`prompt::parse`] does; the events are the ids that spell the parsed text,
//!   without the `language X<asr_text>` prefix ([`chunk_events`]). Each takes
//!   the mel frame `a + (b - a) * chars_before / chars` over the chunk's frames
//!   `[a, b)`: words land in the right chunk, and only approximately at the
//!   right time inside it. The stream reports each chunk's end
//!   ([`RecognizerStream::chunk_ends`]), so that a word ends where the next
//!   one starts and a chunk's last word at the chunk's end.
//! - **A chunk that echoes the keywords.** Given keywords, the model now and
//!   then writes the list out in place of quiet or unclear speech. A chunk
//!   whose parsed text [`echoes`] them is decoded once more from the same
//!   tower rows with no context, and that text is kept; the no-context
//!   decode is what the chunk would have read without keywords at all.

use std::ops::Range;
use std::sync::Arc;

use shiso_audio::chunks::{CHUNK_SECS, SEARCH_SECS, WINDOW_SECS, silence_cuts};
use shiso_audio::{
    AudioError, Detokenizer, MelFrames, PieceVocab, RecognizerStream, TokenEvent, is_echo,
    log10_max_floor,
};
use shiso_core::TokenId;

use super::Model;
use super::config::{ASR_TEXT, language_code};
use super::decoder::{Decoder, KvCache, MAX_NEW_TOKENS, POOLED_CAPACITY};
use super::encoder::Tower;
use super::prompt::{self, Conditioning};
use crate::audio_ops::Workspace;

/// Seconds per mel frame, the unit of every token's frame.
pub(crate) const MEL_FRAME_SECS: f64 = 0.01;
/// The shortest chunk the tower is given, in mel frames: 0.5 s, the
/// shortest input the reference transcribes without padding it.
const MIN_CHUNK_FRAMES: usize = 50;
/// The first-stage value of a frame of digital silence, `log10(1e-10)`: what
/// the frontend gives the zeros the reference pads a short chunk with.
const SILENT_LOG10: f32 = -10.0;
/// `Ġ`, a lone space: put between two chunks' texts.
const SPACE: u32 = 220;

/// `secs` in whole mel frames.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the chunking constants are small positive seconds"
)]
fn frames_in(secs: f64) -> usize {
    (secs / MEL_FRAME_SECS).round() as usize
}

pub(crate) fn frame_word(frame: usize) -> Result<u32, AudioError> {
    u32::try_from(frame).map_err(|_| AudioError::BadSpec {
        reason: format!("mel frame {frame} does not fit a stream's frame index"),
    })
}

/// Where one stream's chunks end, in mel frames from its start.
#[derive(Debug)]
pub(crate) struct Planner {
    /// The caller's cuts, strictly increasing; `None` for the silence rule.
    given: Option<Vec<usize>>,
    /// The first given cut not yet reached.
    next: usize,
    /// Where the current chunk starts: the last end handed out.
    start: usize,
    /// The silence rule's chunk, search and window lengths.
    max: usize,
    search: usize,
    window: usize,
}

impl Planner {
    /// A plan cut at `cuts`, or by the silence rule without them.
    ///
    /// # Errors
    ///
    /// [`AudioError::BadSpec`] for cuts that are not strictly increasing or
    /// start at frame 0: a chunk of no frames is not a plan.
    pub(crate) fn new(cuts: Option<&[u32]>) -> Result<Self, AudioError> {
        let given = match cuts {
            None => None,
            Some(cuts) => {
                let mut last = 0;
                let mut out = Vec::with_capacity(cuts.len());
                for &cut in cuts {
                    let cut = usize::try_from(cut).unwrap_or(usize::MAX);
                    if cut <= last {
                        return Err(AudioError::BadSpec {
                            reason: format!(
                                "chunk cuts must be strictly increasing past frame 0; \
                                 {cut} follows {last}"
                            ),
                        });
                    }
                    out.push(cut);
                    last = cut;
                }
                Some(out)
            }
        };
        Ok(Self {
            given,
            next: 0,
            start: 0,
            max: frames_in(CHUNK_SECS),
            search: frames_in(SEARCH_SECS),
            window: frames_in(WINDOW_SECS),
        })
    }

    /// Where the current chunk starts.
    pub(crate) fn start(&self) -> usize {
        self.start
    }

    /// The ends of the chunks that `rows` - every mel row from
    /// [`Self::start`] received so far, `n_mels` wide - completes, in order;
    /// with `finishing`, through the last row. A given cut at or past the end
    /// of a finished clip is not a chunk.
    pub(crate) fn ends(&mut self, rows: &[f32], n_mels: usize, finishing: bool) -> Vec<usize> {
        let have = rows.len() / n_mels.max(1);
        let received = self.start + have;
        let mut out = Vec::new();
        let mut at = self.start;
        if let Some(given) = &self.given {
            while let Some(&cut) = given.get(self.next)
                && cut <= received
            {
                self.next += 1;
                if cut > at {
                    out.push(cut);
                    at = cut;
                }
            }
        } else {
            // The rule's next cut, from `at`, needs the chunk, the search past
            // its mark and one window: what the whole-clip rule would read.
            let need = self.max + self.search + self.window;
            while received - at >= need && !finishing {
                let lo = at - self.start;
                let cut = self.first_cut(&rows[lo * n_mels..(lo + need) * n_mels], n_mels);
                at += cut;
                out.push(at);
            }
            if finishing && received > at {
                let lo = at - self.start;
                for cut in self.cuts(&rows[lo * n_mels..], n_mels) {
                    out.push(at + cut);
                }
                at = out.last().copied().unwrap_or(at);
            }
        }
        if finishing && received > at {
            out.push(received);
        }
        if let Some(&last) = out.last() {
            self.start = last;
        }
        out
    }

    /// The silence rule's cuts over `rows`, from their first.
    fn cuts(&self, rows: &[f32], n_mels: usize) -> Vec<usize> {
        MelFrames::from_rows(rows.to_vec(), n_mels)
            .map(|mel| silence_cuts(&mel, self.max, self.search, self.window))
            .unwrap_or_default()
    }

    /// The first of [`Self::cuts`] over rows that run past the first mark's
    /// search, which the rule always finds.
    fn first_cut(&self, rows: &[f32], n_mels: usize) -> usize {
        self.cuts(rows, n_mels).first().copied().unwrap_or(self.max)
    }
}

/// What one chunk's decoded ids come to.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ChunkText {
    /// The ids that spell the chunk's text, each at its interpolated frame.
    pub(crate) events: Vec<TokenEvent>,
    /// The language the chunk was transcribed in, as an ISO code where the
    /// model's name for it has one.
    pub(crate) language: Option<String>,
    /// The text the events spell, the space before it included.
    pub(crate) text: String,
}

impl ChunkText {
    /// The last character the chunk wrote, for the next chunk's join; `None`
    /// when it wrote nothing, so the join looks past it.
    pub(crate) fn last_char(&self) -> Option<char> {
        self.text.trim_end().chars().next_back()
    }
}

/// A chunk's first-stage rows (`n` bins wide) as the tower is given them:
/// padded with silent frames to [`MIN_CHUNK_FRAMES`], then floored at their
/// own maximum and rescaled.
///
/// # Errors
///
/// [`AudioError::BadSpec`] for rows that are not whole frames.
pub(crate) fn chunk_mel(mut data: Vec<f32>, n: usize) -> Result<MelFrames, AudioError> {
    if data.len() < MIN_CHUNK_FRAMES * n {
        data.resize(MIN_CHUNK_FRAMES * n, SILENT_LOG10);
    }
    Ok(log10_max_floor(&MelFrames::from_rows(data, n)?))
}

/// Says so when a chunk at frames `span` decoded to the token cap: its text
/// is kept, but it is most likely a repetition the model could not leave.
pub(crate) fn warn_at_cap(ids: &[u32], span: &Range<u32>) {
    if ids.len() >= MAX_NEW_TOKENS {
        tracing::warn!(
            start = span.start,
            end = span.end,
            "a chunk decoded to the token cap; its text is kept"
        );
    }
}

/// Whether `c` is a letter of a script written without spaces between
/// words - Han, kana, Thai, Lao, Khmer, Myanmar - so that no space goes
/// between it and a neighbouring chunk's text. Hangul is written with spaces
/// and is not one. Nor is CJK punctuation: a Latin word after a `。` is still
/// a word of its own, and a Han character on the other side keeps the join
/// unspaced anyway.
pub(crate) fn unspaced(c: char) -> bool {
    matches!(
        c,
        '\u{0E00}'..='\u{0EFF}' // Thai, Lao
            | '\u{1000}'..='\u{109F}' // Myanmar
            | '\u{1780}'..='\u{17FF}' // Khmer
            | '\u{3040}'..='\u{30FF}' // Hiragana, Katakana
            | '\u{31F0}'..='\u{31FF}' // Katakana phonetic extensions
            | '\u{3400}'..='\u{4DBF}' // Han extension A
            | '\u{4E00}'..='\u{9FFF}' // Han
            | '\u{F900}'..='\u{FAFF}' // Han compatibility
            | '\u{FF66}'..='\u{FF9F}' // half-width Katakana
            | '\u{20000}'..='\u{3FFFF}' // Han extensions B onward
    )
}

/// The events of one chunk of frames `span` whose greedy ids are `ids`
/// (without the end id), decoded through `vocab`; `forced` is the language
/// the prompt forced, by the model's name for it; `after` is the last
/// character earlier chunks wrote, if they wrote any.
///
/// The ids kept are those after the first `<asr_text>` (the whole output when
/// the language was forced, or when the model wrote no tag), up to the first
/// one whose text would take the transcript away from [`prompt::parse`]'s:
///
/// - a leading id that is only whitespace is dropped, since the parsed text
///   is stripped, and so is a trailing one;
/// - where the repetition collapse changed the text, the ids that spell the
///   repeats past the first copy are dropped, and emitting resumes at the
///   first later id from which the ids spell the rest of the collapsed text
///   ([`resume_at`]): the collapse keeps what follows a run, so the
///   transcript does too. Where no later id does (the collapse changed the
///   rest as well), the ids stop there.
///
/// Ids that decode to nothing until a later id completes a character share
/// that id's fate and frame. A chunk after one that wrote text starts with a
/// [`SPACE`] at its first frame unless its own text starts with whitespace
/// or either side of the join is [`unspaced`]: the reference joins chunk
/// texts with nothing between, which would glue the last word of one chunk
/// to the first of the next, and is right only where words take no spaces.
pub(crate) fn chunk_events(
    vocab: &Arc<PieceVocab>,
    ids: &[u32],
    forced: Option<&str>,
    span: Range<u32>,
    after: Option<char>,
) -> ChunkText {
    let mut raw = Detokenizer::new(Arc::clone(vocab));
    for &id in ids {
        raw.push(id);
    }
    raw.flush();
    let parsed = prompt::parse(raw.text(), forced);
    let language = parsed
        .language
        .as_deref()
        .map(|name| language_code(name).map_or_else(|| name.to_lowercase(), str::to_owned));
    let kept = match (forced, ids.iter().position(|&id| id == ASR_TEXT)) {
        (None, Some(tag)) => &ids[tag + 1..],
        _ => ids,
    };
    let target = parsed.text.as_str();
    // (ids, characters of kept text before them)
    let mut groups: Vec<(Vec<u32>, usize)> = Vec::new();
    let mut text = String::new();
    // Bytes of `target` the kept text spells.
    let mut matched = 0;
    let mut accept = |ids: Vec<u32>, delta: &str, text: &mut String, matched: &mut usize| {
        // Leading whitespace is not the stripped target's.
        let spelled = if text.is_empty() {
            delta.trim_start()
        } else {
            delta
        };
        if spelled.is_empty() {
            return true;
        }
        if !target[*matched..].starts_with(spelled) {
            return false;
        }
        groups.push((ids, text.chars().count()));
        text.push_str(delta);
        *matched += spelled.len();
        true
    };
    let mut waiting: Vec<u32> = Vec::new();
    let mut detok = Detokenizer::new(Arc::clone(vocab));
    let mut agreed = true;
    let mut i = 0;
    while let Some(&id) = kept.get(i) {
        i += 1;
        waiting.push(id);
        let Some(delta) = detok.push(id) else {
            continue;
        };
        let from = i - waiting.len();
        if accept(
            std::mem::take(&mut waiting),
            &delta,
            &mut text,
            &mut matched,
        ) {
            continue;
        }
        let Some(skip) = resume_at(vocab, &kept[from..], &target[matched..]) else {
            agreed = false;
            break;
        };
        i = from + skip;
        detok = Detokenizer::new(Arc::clone(vocab));
    }
    if agreed && let Some(delta) = detok.flush() {
        accept(
            std::mem::take(&mut waiting),
            &delta,
            &mut text,
            &mut matched,
        );
    }
    let chars = u64::try_from(text.chars().count().max(1)).unwrap_or(u64::MAX);
    let (a, b) = (span.start, span.end.max(span.start));
    let at = |before: usize| {
        let before = u64::try_from(before).unwrap_or(u64::MAX);
        let offset = u64::from(b - a) * before / chars;
        a + u32::try_from(offset).unwrap_or(0)
    };
    let mut events = Vec::with_capacity(kept.len() + 1);
    let spaced = |c: char| !c.is_whitespace() && !unspaced(c);
    if after.is_some_and(spaced) && text.chars().next().is_some_and(spaced) {
        events.push(TokenEvent {
            token: TokenId::new(SPACE),
            frame: a,
        });
        text.insert(0, ' ');
    }
    for (ids, before) in groups {
        let frame = at(before);
        events.extend(ids.into_iter().map(|id| TokenEvent {
            token: TokenId::new(id),
            frame,
        }));
    }
    ChunkText {
        events,
        language,
        text,
    }
}

/// Whether a chunk whose greedy ids are `ids` (without the end id) echoes
/// `keywords` ([`is_echo`]), judged on its text as [`prompt::parse`] reads
/// it: the `language X<asr_text>` prefix is the model's, not the speech's.
/// `forced` is as for [`chunk_events`]. No keywords never echo, so a
/// prompt-only context never costs a second decode.
pub(crate) fn echoes(
    vocab: &Arc<PieceVocab>,
    ids: &[u32],
    forced: Option<&str>,
    keywords: &[Vec<String>],
) -> bool {
    if keywords.is_empty() {
        return false;
    }
    let mut raw = Detokenizer::new(Arc::clone(vocab));
    for &id in ids {
        raw.push(id);
    }
    raw.flush();
    is_echo(&prompt::parse(raw.text(), forced).text, keywords)
}

/// Where emitting resumes after `ids[0]` began text the collapsed target does
/// not have: the index of the first later id, at a character boundary, from
/// which `ids` spell exactly `rest` (the target's text not yet spelled),
/// trailing whitespace aside. `None` when nothing is left to spell or no
/// suffix spells it.
///
/// One decode of `ids` finds every boundary, and a suffix of the wrong
/// length is refused before its bytes are compared, so a chunk at the token
/// cap costs one pass over its ids, not one per id.
fn resume_at(vocab: &Arc<PieceVocab>, ids: &[u32], rest: &str) -> Option<usize> {
    if rest.is_empty() {
        return None;
    }
    let mut detok = Detokenizer::new(Arc::clone(vocab));
    // (the id after a whole delta, the text's length before that id)
    let mut starts = Vec::new();
    for (i, &id) in ids.iter().enumerate() {
        if detok.push(id).is_some() {
            starts.push((i + 1, detok.text().len()));
        }
    }
    detok.flush();
    let all = detok.text();
    starts
        .into_iter()
        .find(|&(i, at)| i < ids.len() && all[at..].trim_end() == rest)
        .map(|(i, _)| i)
}

/// A stream's device memory: the decoder's cache and activations and the
/// tower's scratch, all written before they are read. A closed stream hands
/// it back to the model for the next one ([`give_back`]), because a fresh
/// buffer's first touch costs some 50 us a megabyte (M1 Max) and a stream's
/// is several hundred megabytes.
#[derive(Debug)]
pub(crate) struct Memory {
    cache: KvCache,
    decoder: Workspace,
    pub(crate) tower: Workspace,
}

impl Memory {
    /// A memory from `model`'s pool, or a fresh one when the pool is empty.
    pub(crate) fn take(model: &Model) -> Self {
        let spare = model
            .spare
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .pop();
        spare.unwrap_or_else(|| Self {
            cache: KvCache::new(&model.config.text),
            decoder: Workspace::new(),
            tower: Workspace::new(),
        })
    }

    /// Returns this memory to `model`'s pool ([`give_back`]).
    pub(crate) fn put_back(self, model: &Model) {
        give_back(
            &mut model
                .spare
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
            self,
        );
    }
}

/// Memories the model keeps for the next streams: the server's default
/// `--audio-max-streams`. The model is opened without the server's limit, so
/// it cannot size the pool to it; a server running more streams than this
/// allocates the rest afresh, and one running fewer never fills it.
pub(crate) const POOL_STREAMS: usize = 4;

/// Returns a closed stream's `memory` to `pool`, or drops it: a cache that
/// grew past [`POOLED_CAPACITY`] held one degenerate chunk's rows - up to
/// some 1.3 GB at the token cap - and would hold them for the life of the
/// process; a pool of [`POOL_STREAMS`] already serves every stream the
/// server runs at once by default.
pub(crate) fn give_back(pool: &mut Vec<Memory>, memory: Memory) {
    if memory.cache.capacity() <= POOLED_CAPACITY && pool.len() < POOL_STREAMS {
        pool.push(memory);
    }
}

/// A stream of this recognizer.
#[derive(Debug)]
pub(crate) struct QwenStream {
    model: Arc<Model>,
    /// The language the prompt forces, by the model's name for it.
    forced: Option<&'static str>,
    /// The system turn's ids, shared by every chunk.
    context: Arc<[u32]>,
    /// The keywords `context` was made from, as word sequences: a chunk
    /// that echoes them is decoded again without it.
    keywords: Arc<[Vec<String>]>,
    /// How many chunks were decoded again for echoing the keywords.
    redecoded: usize,
    planner: Planner,
    /// Always `Some` until the stream is dropped and gives it back.
    memory: Option<Memory>,
    /// Mel rows from the planner's start, row-major, first-stage `log10`.
    pending: Vec<f32>,
    /// The last character the chunks have written.
    last_char: Option<char>,
    /// The first language a chunk reported.
    language: Option<String>,
    /// Where each chunk transcribed so far ends, in mel frames.
    ends: Vec<u32>,
    finished: bool,
}

impl QwenStream {
    pub(crate) fn new(
        model: Arc<Model>,
        forced: Option<&'static str>,
        context: Arc<[u32]>,
        keywords: Arc<[Vec<String>]>,
        cuts: Option<&[u32]>,
    ) -> Result<Self, AudioError> {
        let memory = Memory::take(&model);
        Ok(Self {
            planner: Planner::new(cuts)?,
            model,
            forced,
            context,
            keywords,
            redecoded: 0,
            memory: Some(memory),
            pending: Vec::new(),
            last_char: None,
            language: None,
            ends: Vec::new(),
            finished: false,
        })
    }

    /// Transcribes every chunk the rows in so far complete.
    fn run(&mut self, finishing: bool) -> Result<Vec<TokenEvent>, AudioError> {
        let n = self.model.config.audio.mels;
        let start = self.planner.start();
        let ends = self.planner.ends(&self.pending, n, finishing);
        let mut events = Vec::new();
        let mut from = start;
        for end in ends {
            let rows = self.pending[(from - start) * n..(end - start) * n].to_vec();
            let chunk = self.chunk(rows, frame_word(from)?..frame_word(end)?)?;
            if let Some(last) = chunk.last_char() {
                self.last_char = Some(last);
            }
            if self.language.is_none() {
                self.language = chunk.language;
            }
            events.extend(chunk.events);
            self.ends.push(frame_word(end)?);
            from = end;
        }
        self.pending.drain(..(from - start) * n);
        Ok(events)
    }

    /// One chunk of first-stage rows at frames `span`.
    fn chunk(&mut self, data: Vec<f32>, span: Range<u32>) -> Result<ChunkText, AudioError> {
        let m = &*self.model;
        let mel = chunk_mel(data, m.config.audio.mels)?;
        let memory = self.memory.as_mut().ok_or_else(|| AudioError::Checkpoint {
            what: "a stream without its device memory".into(),
        })?;
        let audio = Tower::new(&m.ops, &m.weights.tower, &m.config.audio)
            .encode_with(&mel, &mut memory.tower)?;
        let decoder = Decoder::new(&m.ops, &m.weights.text, &m.config.text);
        let mut ids = decoder.transcribe_with(
            &mut memory.cache,
            &mut memory.decoder,
            &audio,
            &Conditioning {
                forced: self.forced,
                context: &self.context,
            },
        )?;
        if echoes(&m.vocab, &ids, self.forced, &self.keywords) {
            ids = decoder.transcribe_with(
                &mut memory.cache,
                &mut memory.decoder,
                &audio,
                &Conditioning::forced(self.forced),
            )?;
            self.redecoded += 1;
        }
        warn_at_cap(&ids, &span);
        Ok(chunk_events(
            &m.vocab,
            &ids,
            self.forced,
            span,
            self.last_char,
        ))
    }
}

impl Drop for QwenStream {
    fn drop(&mut self) {
        if let Some(memory) = self.memory.take() {
            memory.put_back(&self.model);
        }
    }
}

impl RecognizerStream for QwenStream {
    fn push(&mut self, mel: &MelFrames) -> Result<Vec<TokenEvent>, AudioError> {
        if self.finished {
            return Err(AudioError::Checkpoint {
                what: "push on a finished stream".into(),
            });
        }
        let n = self.model.config.audio.mels;
        if mel.valid() > 0 && mel.n_mels() != n {
            return Err(AudioError::BadSpec {
                reason: format!("{} mel bins, this model reads {n}", mel.n_mels()),
            });
        }
        self.pending
            .extend_from_slice(&mel.as_slice()[..mel.valid() * mel.n_mels()]);
        self.run(false)
    }

    fn finish(&mut self) -> Result<Vec<TokenEvent>, AudioError> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let events = self.run(true)?;
        if !self.keywords.is_empty() {
            tracing::debug!(
                chunks = self.ends.len(),
                redecoded = self.redecoded,
                "a stream's chunks decoded again for echoing the keywords"
            );
        }
        Ok(events)
    }

    /// Mel frames transcribed so far.
    fn frames(&self) -> u32 {
        u32::try_from(self.planner.start()).unwrap_or(u32::MAX)
    }

    fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    /// Every chunk's end: a token's frame is only interpolated inside its
    /// chunk, so its word ends at the next word or the chunk's end.
    fn chunk_ends(&self) -> &[u32] {
        &self.ends
    }
}

#[cfg(test)]
#[path = "stream_tests.rs"]
pub(crate) mod tests;
