//! One sequence, from the slot it was given to the terminal event it produced.
//!
//! # Why the state is a phase and not a set of flags
//!
//! A sequence is reading its prompt, or standing on a sampled token waiting for
//! a step, or over. Those are the only three, and they are exclusive: a sequence
//! that is "finished and still has prompt left" or "waiting for a decode step it
//! has no token for" is not a situation the scheduler has to handle, it is a
//! situation that must not be representable. So the phase is one field, and
//! every question the loop asks - may it decode, has it prompt left, is it done
//! - is answered by reading it rather than by combining flags that can disagree.
//!
//! # Why nothing here sends anything
//!
//! Accepting a token produces a list of events and the caller sends them. The
//! send is bounded and its outcome is what cancellation is derived from, so a
//! sequence that sent its own events would have to know about the cancellation
//! policy as well - and the two would then be able to disagree about whether a
//! sequence that lost its client is still alive.

use std::sync::Arc;
use std::time::{Duration, Instant};

use shiso_core::{DecodeStep, SamplingParams, Seed, SessionId, StopTokens, TokenId};
use shiso_ir::{BlockDelta, BlockStart, StopReason, StreamEvent, Usage};
use tokio::sync::mpsc;

use crate::scheduler::batching::SlotFacts;
use crate::scheduler::stop::{StopStep, StopWatcher};
use crate::tokenizer::Vocabulary;
use crate::vision::EncodedMedia;
use shiso_parsers::Segment;

/// Which block is open and what index the next one takes.
#[derive(Debug, Default)]
pub(crate) struct BlockCursor {
    pub(crate) next: u32,
    pub(crate) open: Option<(u32, OpenKind)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OpenKind {
    Text,
    Thinking,
    ToolUse,
}

impl BlockCursor {
    /// Events that close whatever is open and open a block of `kind` at the next index.
    pub(crate) fn ensure(
        &mut self,
        kind: OpenKind,
        start: impl FnOnce(u32) -> BlockStart,
    ) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        if let Some((idx, open_kind)) = self.open {
            if open_kind == kind {
                return events;
            }
            events.push(StreamEvent::BlockStop { index: idx });
            self.open = None;
        }
        let index = self.next;
        self.next = self.next.saturating_add(1);
        events.push(StreamEvent::BlockStart {
            index,
            block: start(index),
        });
        self.open = Some((index, kind));
        events
    }

    /// `BlockStop` for the open block.
    pub(crate) fn close(&mut self) -> Option<StreamEvent> {
        let (idx, _) = self.open.take()?;
        Some(StreamEvent::BlockStop { index: idx })
    }

    /// Current open block index and kind, if any.
    pub(crate) fn current(&self) -> Option<(u32, OpenKind)> {
        self.open
    }
}

/// Where a sequence is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// Still reading prompt. `cursor` tokens of it have been read.
    Prefill,
    /// Standing on a sampled token, waiting for its decode step.
    Decode,
    /// Over. The reason is in [`SeqSlot::ending`].
    Done,
}

/// Why a sequence stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Ending {
    /// It reached a terminal condition of its own: an end-of-turn marker, its
    /// budget, a stop sequence, or its deadline.
    Stopped(StopReason),
    /// Its client stopped reading. Nothing more is sent, and the session is
    /// still released by the coordinator rather than by the client.
    Cancelled,
    /// Something below it failed. The message is what the terminal event says.
    Failed(String),
}

/// The text a turn has produced so far, and how much of it has been sent.
///
/// A separate little struct rather than the crate's [`Detokenizer`], which
/// borrows the vocabulary for its whole life: the vocabulary and the sequences
/// live in the same owner here, so a borrow held inside a sequence would make
/// that owner self-referential. The decoding rule is the same one - decode the
/// whole run every time and send what is new - because a piece's rendering
/// depends on its neighbours and separately decoded pieces do not concatenate
/// into the blocking answer.
///
/// [`Detokenizer`]: crate::tokenizer::Detokenizer
#[derive(Debug, Default)]
struct TextCursor {
    emitted: usize,
}

impl TextCursor {
    /// The text `tokens` has gained since the last call.
    fn fresh(&mut self, vocabulary: Option<&Vocabulary>, tokens: &[TokenId]) -> String {
        let Some(vocabulary) = vocabulary else {
            return String::new();
        };
        let whole = match vocabulary.decode(tokens) {
            Ok(text) => text,
            Err(error) => {
                // A run that will not decode is reported and then treated as
                // producing no text. Failing the turn over it would throw away
                // an answer whose ids are perfectly good, and the ids are what
                // a caller can still act on.
                tracing::warn!(%error, "a generated run did not decode; its text is being skipped");
                return String::new();
            }
        };
        // An incomplete multi-byte sequence at the tail decodes to U+FFFD; hold
        // it back until subsequent tokens complete the character.
        if whole.ends_with('\u{FFFD}') {
            return String::new();
        }
        // A shrinking decode is possible in principle - a vocabulary is free to
        // render a pair differently from its parts - and `get` rather than an
        // index is what keeps that a missing fragment rather than a panic.
        let fresh = whole.get(self.emitted..).unwrap_or_default().to_owned();
        self.emitted = whole.len();
        fresh
    }

    /// Delivers whatever text was held back because the tail ended with an
    /// incomplete multi-byte character.
    fn flush(&mut self, vocabulary: Option<&Vocabulary>, tokens: &[TokenId]) -> String {
        let Some(vocabulary) = vocabulary else {
            return String::new();
        };
        let whole = match vocabulary.decode(tokens) {
            Ok(text) => text,
            Err(error) => {
                tracing::warn!(%error, "a generated run did not decode; its text is being skipped");
                return String::new();
            }
        };
        let fresh = whole.get(self.emitted..).unwrap_or_default().to_owned();
        self.emitted = whole.len();
        fresh
    }
}

/// Generated ids a tool-call proposal decodes to find the open call.
const SCAFFOLD_TAIL: usize = 256;

/// Fewest prefilled tokens a prompt needs before its wall time says anything
/// about the prefill rate.
const MIN_TIMED_PREFILL: u32 = 512;

/// Whether a sequence's path carries media, and if so how the cache may file it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PathMedia {
    /// No media: the path is its own key.
    Text,
    /// Media the model files under no content key. Kept off the tier, because
    /// a placeholder run is the same ids for any picture. A resume can cover
    /// every picture so that nothing is encoded, and the path still carries
    /// their runs, which is why this is a fact about the request and not
    /// about what was encoded.
    Unkeyed,
    /// Media the model files under the pictures' content key.
    Keyed,
}

/// How a sequence draws its tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DrawRule {
    /// The stream sampler: each stochastic draw advances the seed.
    Stream,
    /// The keyed rule: a function of the row, the seed and the position,
    /// which a speculating server uses so a sampled draft verifies by equality.
    Keyed,
}

/// One resident sequence.
#[derive(Debug)]
pub(crate) struct SeqSlot {
    /// The identifier its stream opened with.
    pub(crate) id: String,
    /// The session it holds, which exactly one call will close.
    pub(crate) session: SessionId,
    /// The rendered prompt.
    pub(crate) prompt: Arc<[TokenId]>,
    /// The rendered prompt as text, empty when the surface had none.
    prompt_text: Arc<str>,
    /// Where its events go.
    pub(crate) sink: mpsc::Sender<StreamEvent>,
    /// Prompt tokens read so far, resume included.
    cursor: usize,
    /// Prompt tokens a resume covered, which are charged as cached.
    cached: u64,
    /// Whether the whole prompt was read by this process rather than resumed.
    cold: bool,
    /// The token the next decode step feeds in, when one is due.
    next_input: Option<TokenId>,
    /// The ids generated, in order.
    out: Vec<TokenId>,
    /// The declared tools a speculative step proposes a call's fixed text
    /// from, `None` for a request with none or a scheme it does not spell.
    scaffold: Option<Arc<shiso_parsers::ToolScaffold>>,
    /// The end-of-turn marker the turn stopped on, which `out` leaves out.
    end_marker: Option<TokenId>,
    /// How many tokens this turn may still generate.
    budget: u32,
    /// Sampling settings and the stream they draw from.
    sampling: SamplingParams,
    /// The PRNG state, which is per sequence so that two turns in one batch do
    /// not draw from each other's stream.
    seed: Seed,
    /// How its tokens are drawn.
    rule: DrawRule,
    /// The stop-sequence hold-back.
    watcher: StopWatcher,
    /// How much generated text has been sent.
    text: TextCursor,
    /// Ids whose text is being held back, waiting for a delta that carries text.
    trailing: Vec<u32>,
    /// The stream parser for thinking tags and tool calls.
    pub(crate) parser: Box<dyn shiso_parsers::StreamParser>,
    /// Which block is open and the index counter.
    pub(crate) blocks: BlockCursor,
    /// How many tool calls have been emitted in this turn.
    pub(crate) calls_emitted: u32,
    /// Generated tokens spent on the reasoning trace (see [`SeqSlot::accept`]).
    reasoning: u64,
    /// Where it is.
    phase: Phase,
    /// Why it stopped, once it has.
    pub(crate) ending: Option<Ending>,
    /// When it was admitted.
    started_at: Instant,
    /// When its first token was sampled.
    first_token_at: Option<Instant>,
    /// When it must be cut, when the server has a deadline.
    pub(crate) deadline: Option<Instant>,
    /// The frontier length this sequence last put on disk.
    pub(crate) stored_len: u32,
    /// Prompt tokens this process prefilled, and the wall time the calls took.
    prefill_timed: (u32, Duration),
    /// The coordinator's decode-step count when the first prompt chunk ran, and
    /// whether it moved before the last: a prefill with decode steps between its
    /// chunks did not have the device to itself.
    prefill_decodes: Option<u64>,
    prefill_mixed: bool,
    /// The encoder rows its prompt's placeholders stand for.
    ///
    /// Owned by the sequence rather than by the tick that encoded them: the
    /// rows are read by the forward pass and the forward pass happens later,
    /// on a tick this sequence may not be alone on. `None` is a text-only
    /// request and takes the launches it took before any of this existed.
    media: Option<EncodedMedia>,
    /// Whether its path carries media and how the cache may file it.
    path_media: PathMedia,
    /// Audio placeholder tokens in the prompt, counted from the request rather
    /// than from what was encoded, which a resume can make smaller.
    audio_pads: u32,
    /// Whether it runs only when no standard sequence has work.
    flex: bool,
    /// When the planner last started leaving it out, while it still is.
    paused_since: Option<Instant>,
    /// Guesses drafted for it, guesses kept, and speculative steps taken:
    /// the turn's own share of the scheduler's speculation counters.
    spec: (u64, u64, u64),
}

/// Everything a slot needs that is not derived from the request.
#[derive(Debug)]
pub(crate) struct SlotSetup {
    /// The identifier the stream opened with.
    pub(crate) id: String,
    /// The session the coordinator opened for it.
    pub(crate) session: SessionId,
    /// Where its events go.
    pub(crate) sink: mpsc::Sender<StreamEvent>,
    /// When it was admitted.
    pub(crate) started_at: Instant,
    /// When it must be cut.
    pub(crate) deadline: Option<Instant>,
    /// Tokens it may generate, already clamped against the context.
    pub(crate) budget: u32,
    /// Whether its tokens are drawn by the keyed rule, which a speculating
    /// server uses for every sequence so a sampled one can verify drafts.
    pub(crate) keyed: bool,
}

impl SeqSlot {
    /// A sequence over `request`, holding `setup.session`.
    pub(crate) fn new(
        request: crate::scheduler::GenerateRequest,
        setup: SlotSetup,
        media: Option<EncodedMedia>,
        parser: Box<dyn shiso_parsers::StreamParser>,
    ) -> Self {
        let path_media = if request.media.is_empty() && media.is_none() {
            PathMedia::Text
        } else {
            PathMedia::Unkeyed
        };
        let audio_pads = request
            .media
            .iter()
            .filter(|item| item.is_audio())
            .map(|item| u32::try_from(item.pad_count(1)).unwrap_or(u32::MAX))
            .fold(0u32, u32::saturating_add);
        let audio_pads = media
            .as_ref()
            .map_or(audio_pads, |media| audio_pads.max(media.audio_pads()));
        Self {
            flex: request.is_flex(),
            paused_since: None,
            media,
            path_media,
            audio_pads,
            parser,
            blocks: BlockCursor::default(),
            calls_emitted: 0,
            reasoning: 0,
            id: setup.id,
            session: setup.session,
            prompt: request.prompt_tokens,
            prompt_text: request.prompt_text,
            sink: setup.sink,
            cursor: 0,
            cached: 0,
            cold: true,
            next_input: None,
            out: Vec::new(),
            scaffold: None,
            end_marker: None,
            budget: setup.budget,
            sampling: request.sampling,
            seed: request.seed,
            rule: if setup.keyed {
                DrawRule::Keyed
            } else {
                DrawRule::Stream
            },
            watcher: StopWatcher::new(&request.stop),
            text: TextCursor::default(),
            trailing: Vec::new(),
            phase: Phase::Prefill,
            ending: None,
            started_at: setup.started_at,
            first_token_at: None,
            deadline: setup.deadline,
            stored_len: 0,
            prefill_timed: (0, Duration::ZERO),
            prefill_decodes: None,
            prefill_mixed: false,
            spec: (0, 0, 0),
        }
    }

    /// What the planner needs to know about it.
    pub(crate) fn facts(&self) -> SlotFacts {
        SlotFacts {
            ready_to_decode: self.phase == Phase::Decode && self.next_input.is_some(),
            prompt_remaining: if self.phase == Phase::Prefill {
                self.prompt.len().saturating_sub(self.cursor)
            } else {
                0
            },
            prompt_at: self.cursor,
            indivisible_prompt: self.media.is_some(),
            flex: self.flex,
        }
    }

    /// Whether this is a flex sequence.
    pub(crate) const fn is_flex(&self) -> bool {
        self.flex
    }

    /// Records whether this tick left this sequence out for standard work, and
    /// logs the moment that changes: how long a flex sequence waited is the one
    /// number an operator needs to tell "paused" from "stuck".
    pub(crate) fn mark_paused(&mut self, paused: bool, now: Instant) {
        match (paused, self.paused_since) {
            (true, None) => {
                tracing::debug!(slot = self.id, "a flex sequence paused for standard work");
                self.paused_since = Some(now);
            }
            (false, Some(since)) => {
                tracing::debug!(
                    slot = self.id,
                    paused_ms = now.saturating_duration_since(since).as_millis(),
                    "a flex sequence resumed"
                );
                self.paused_since = None;
            }
            _ => {}
        }
    }

    /// The encoder rows this sequence's placeholders stand for, when it has any.
    ///
    /// `None` is a text-only sequence, which is a fact about the request rather
    /// than a failure to produce rows: a request with no media never reached the
    /// tower.
    pub(crate) const fn media(&self) -> Option<&EncodedMedia> {
        self.media.as_ref()
    }

    /// Hands it the encoder rows for the items its prompt still has to read.
    pub(crate) fn attach_media(&mut self, media: EncodedMedia) {
        self.media = Some(media);
    }

    /// Records that the model files this sequence under a content key.
    pub(crate) const fn set_keyed(&mut self) {
        if matches!(self.path_media, PathMedia::Unkeyed) {
            self.path_media = PathMedia::Keyed;
        }
    }

    /// Prompt tokens read so far, a resume included.
    pub(crate) const fn prompt_cursor(&self) -> usize {
        self.cursor
    }

    /// Whether this sequence's prefix may be stored or resumed.
    ///
    /// The tier keys on token ids, and a picture is a run of one placeholder id
    /// whichever picture it is. So a media request may file or stand on a
    /// prefix only when the model keys its path by the pictures' content (see
    /// [`shiso_core::media::MediaKey`]); without that, both directions serve one
    /// picture's activations for another picture's request - fluently, and
    /// with nothing in the response to show for it.
    pub(crate) const fn checkpointable(&self) -> bool {
        !matches!(self.path_media, PathMedia::Unkeyed)
    }

    /// Records one prefill call: `tokens` covered in `elapsed`, with the
    /// coordinator having run `decode_steps` decode steps so far.
    pub(crate) fn time_prefill(&mut self, tokens: usize, elapsed: Duration, decode_steps: u64) {
        match self.prefill_decodes {
            None => self.prefill_decodes = Some(decode_steps),
            Some(at) if at != decode_steps => self.prefill_mixed = true,
            Some(_) => {}
        }
        let tokens = u32::try_from(tokens).unwrap_or(u32::MAX);
        self.prefill_timed = (
            self.prefill_timed.0.saturating_add(tokens),
            self.prefill_timed.1 + elapsed,
        );
    }

    /// The prompt's prefilled tokens and their wall time, when no decode step
    /// ran between its first chunk and `decode_steps` and there were enough of
    /// them to measure: a few dozen tokens cost a pass's fixed overhead, not a
    /// per-token rate.
    pub(crate) fn solo_prefill(&self, decode_steps: u64) -> Option<(u32, Duration)> {
        let (tokens, elapsed) = self.prefill_timed;
        (!self.prefill_mixed
            && self.prefill_decodes == Some(decode_steps)
            && tokens >= MIN_TIMED_PREFILL)
            .then_some((tokens, elapsed))
    }

    /// Its decode step, when it is standing on a token.
    pub(crate) fn decode_step(&self) -> Option<DecodeStep> {
        self.next_input.map(|token| DecodeStep {
            session: self.session,
            token,
        })
    }

    /// The sampling this sequence was opened with.
    pub(crate) const fn sampling(&self) -> &SamplingParams {
        &self.sampling
    }

    /// Whether it has stopped.
    pub(crate) const fn is_done(&self) -> bool {
        self.ending.is_some()
    }

    /// The ids it has generated, which with the prompt is the path its session
    /// stands on.
    pub(crate) fn generated(&self) -> &[TokenId] {
        &self.out
    }

    /// The turn as the next request in its conversation would carry it: the
    /// rendered prompt text and ids, then the generation and the marker it
    /// ended on.
    ///
    /// `None` for a turn that did not end on its marker - a stop sequence
    /// leaves ids past the text the caller received, and a cut budget can end
    /// mid-character - for one with media, whose pad ids are the same for any
    /// picture, and for one whose surface rendered no text.
    pub(crate) fn finished_turn(
        &self,
        vocabulary: &Vocabulary,
    ) -> Option<(String, Arc<[TokenId]>)> {
        let marker = self.end_marker?;
        if !matches!(self.path_media, PathMedia::Text)
            || self.prompt_text.is_empty()
            || !matches!(
                self.ending,
                Some(Ending::Stopped(StopReason::EndTurn | StopReason::ToolUse))
            )
        {
            return None;
        }
        let mut generated = Vec::with_capacity(self.out.len() + 1);
        generated.extend_from_slice(&self.out);
        generated.push(marker);
        let decoded = vocabulary.decode(&generated).ok()?;
        let mut text = String::with_capacity(self.prompt_text.len() + decoded.len());
        text.push_str(&self.prompt_text);
        text.push_str(&decoded);
        let tokens = self.prompt.iter().chain(&generated).copied().collect();
        Some((text, tokens))
    }

    /// The whole path its session stands on: prompt then generated.
    #[cfg(test)]
    pub(crate) fn frontier_path(&self) -> Vec<TokenId> {
        let mut path = Vec::with_capacity(self.prompt.len() + self.out.len());
        path.extend_from_slice(&self.prompt);
        path.extend_from_slice(&self.out);
        path
    }

    /// Positions its session has run: the frontier less a sampled token still
    /// waiting for the decode step that feeds it in.
    pub(crate) fn written_len(&self) -> u32 {
        self.frontier_len()
            .saturating_sub(u32::from(self.next_input.is_some()))
    }

    /// Positions its session has written.
    pub(crate) fn frontier_len(&self) -> u32 {
        u32::try_from(self.prompt.len().saturating_add(self.out.len())).unwrap_or(u32::MAX)
    }

    /// Whether the whole prompt was prefilled by this process.
    #[cfg(test)]
    pub(crate) const fn was_cold(&self) -> bool {
        self.cold
    }

    /// Its sampling settings for the next token and its PRNG stream, for the
    /// one sampler that draws for it.
    ///
    /// Greedy wherever the output parser says a sampled token could only break
    /// the format - inside tool-call tags - and the request's own settings
    /// everywhere else. Greedy requests are unchanged, since every step already
    /// is.
    pub(crate) fn draw(&mut self) -> (SamplingParams, &mut Seed) {
        let mut params = self.sampling;
        if self.parser.wants_greedy() {
            params.temperature = 0.0;
        }
        (params, &mut self.seed)
    }

    /// The token this sequence draws from one row of logits for its next
    /// position.
    ///
    /// Under the keyed rule the draw is a function of the row, the sequence's
    /// seed and the position alone - the same token whether the row came from
    /// a one-token step or from a speculative verify pass - and the seed is
    /// never advanced. Otherwise it is the stream sampler's next draw.
    pub(crate) fn choose(&mut self, scores: &[f32]) -> TokenId {
        let position = self.frontier_len();
        let keyed = self.rule == DrawRule::Keyed;
        let (params, seed) = self.draw();
        if keyed {
            shiso_engine::keyed_sample(scores, &params, seed.get(), position)
        } else {
            shiso_engine::sample_logits(scores, &params, seed)
        }
    }

    /// How this sequence will choose its next tokens, for a drafter to follow:
    /// its key and its current temperature under the keyed rule, greedy
    /// otherwise.
    pub(crate) fn draft_hint(&mut self) -> shiso_core::model::DraftHint {
        if self.rule != DrawRule::Keyed {
            return shiso_core::model::DraftHint::greedy();
        }
        let (params, seed) = self.draw();
        shiso_core::model::DraftHint {
            key: seed.get(),
            temperature: params.temperature,
        }
    }

    /// Attaches the declared tools a speculative step proposes from.
    pub(crate) fn set_scaffold(&mut self, scaffold: Option<Arc<shiso_parsers::ToolScaffold>>) {
        self.scaffold = scaffold;
    }

    /// The token ids of the text a tool call must say next, or empty.
    ///
    /// Asked only inside a call's tags - where the parser wants the draw
    /// greedy - because that is the only place the answer is ever not empty,
    /// and every other step would pay a decode of the tail for nothing. The
    /// tail is the last [`SCAFFOLD_TAIL`] ids, which reach back past a call's
    /// opening tag for any call whose values are short enough to have a
    /// fixed part left to propose.
    pub(crate) fn scaffold_proposal(&self, vocabulary: Option<&Vocabulary>) -> Vec<TokenId> {
        let (Some(scaffold), Some(vocabulary)) = (self.scaffold.as_ref(), vocabulary) else {
            return Vec::new();
        };
        if !self.parser.wants_greedy() {
            return Vec::new();
        }
        let from = self.out.len().saturating_sub(SCAFFOLD_TAIL);
        let Ok(text) = vocabulary.decode(&self.out[from..]) else {
            return Vec::new();
        };
        scaffold
            .next(&text)
            .and_then(|next| vocabulary.encode(&next, false).ok())
            .unwrap_or_default()
    }

    /// Whether this sequence draws by the keyed rule.
    pub(crate) fn is_keyed(&self) -> bool {
        self.rule == DrawRule::Keyed
    }

    /// The half-open range of the prompt this quantum covers.
    ///
    /// A range rather than a slice, so the caller can hold its own clone of the
    /// prompt while the model call runs instead of borrowing the sequence across
    /// it - the sequence has to be reachable again immediately afterwards to be
    /// told how much was read.
    pub(crate) fn prompt_window(&self, tokens: usize) -> (usize, usize) {
        let end = self.cursor.saturating_add(tokens).min(self.prompt.len());
        (self.cursor.min(end), end)
    }

    /// Records that `tokens` more prompt tokens were read.
    pub(crate) fn advance_prefill(&mut self, tokens: usize) {
        self.cursor = self.cursor.saturating_add(tokens).min(self.prompt.len());
    }

    /// Stands the sequence on a prefix a checkpoint resume restored.
    ///
    /// The tokens it covers were never run through a forward pass by this
    /// process, which is what makes them cached rather than prompt work, and it
    /// is also why the turn is no longer a cold one: a cold store would file the
    /// prefix under a key the entry it resumed from already occupies.
    pub(crate) fn resumed(&mut self, tokens_reused: u32) {
        let reused = usize::try_from(tokens_reused)
            .unwrap_or(usize::MAX)
            .min(self.prompt.len());
        self.cursor = reused;
        self.cached = reused as u64;
        self.cold = false;
        self.stored_len = u32::try_from(reused).unwrap_or(u32::MAX);
    }

    /// Whether the prompt has been read to its end.
    pub(crate) fn prompt_done(&self) -> bool {
        self.cursor >= self.prompt.len()
    }

    /// The event that opens the turn's text block, once.
    pub(crate) fn open_block(&mut self) -> Option<StreamEvent> {
        let events = self.blocks.ensure(OpenKind::Text, |_| BlockStart::Text);
        events.into_iter().next()
    }

    /// The id of this sequence's `n`th tool call.
    ///
    /// Scoped by the sequence id, which is unique per request in this process:
    /// a counter alone restarts every turn, and clients match a tool result to
    /// its call by id, so the second turn's `call_0` answered the first turn's.
    pub(crate) fn call_id(&self, n: u32) -> String {
        format!("call_{}_{n}", self.id)
    }

    fn emit_segments(&mut self, segments: Vec<Segment>) -> Vec<StreamEvent> {
        let mut events = Vec::new();
        let mut token_ids = std::mem::take(&mut self.trailing);
        let mut first_delta = true;

        for seg in segments {
            match seg {
                Segment::Text(s) => {
                    events.extend(self.blocks.ensure(OpenKind::Text, |_| BlockStart::Text));
                    if let Some((index, _)) = self.blocks.current() {
                        let ids = if first_delta {
                            first_delta = false;
                            std::mem::take(&mut token_ids)
                        } else {
                            Vec::new()
                        };
                        events.push(StreamEvent::BlockDelta {
                            index,
                            delta: BlockDelta::Text {
                                text: s,
                                token_ids: ids,
                            },
                        });
                    }
                }
                Segment::ThinkingStart => {
                    events.extend(
                        self.blocks
                            .ensure(OpenKind::Thinking, |_| BlockStart::Thinking),
                    );
                }
                Segment::Thinking(s) => {
                    events.extend(
                        self.blocks
                            .ensure(OpenKind::Thinking, |_| BlockStart::Thinking),
                    );
                    if let Some((index, _)) = self.blocks.current() {
                        let ids = if first_delta {
                            first_delta = false;
                            std::mem::take(&mut token_ids)
                        } else {
                            Vec::new()
                        };
                        events.push(StreamEvent::BlockDelta {
                            index,
                            delta: BlockDelta::Thinking {
                                text: s,
                                token_ids: ids,
                            },
                        });
                    }
                }
                Segment::ThinkingEnd => {
                    if let Some(event) = self.blocks.close() {
                        events.push(event);
                    }
                }
                Segment::ToolCall { name, arguments } => {
                    let call_id = self.call_id(self.calls_emitted);
                    self.calls_emitted = self.calls_emitted.saturating_add(1);
                    events.extend(
                        self.blocks
                            .ensure(OpenKind::ToolUse, |_| BlockStart::ToolUse {
                                id: call_id,
                                name,
                            }),
                    );
                    if let Some((index, _)) = self.blocks.current() {
                        let ids = if first_delta {
                            first_delta = false;
                            std::mem::take(&mut token_ids)
                        } else {
                            Vec::new()
                        };
                        events.push(StreamEvent::BlockDelta {
                            index,
                            delta: BlockDelta::ToolInput {
                                fragment: arguments,
                                token_ids: ids,
                            },
                        });
                    }
                    if let Some(event) = self.blocks.close() {
                        events.push(event);
                    }
                }
                _ => {}
            }
        }

        if !token_ids.is_empty() {
            self.trailing.extend(token_ids);
        }
        events
    }

    /// Takes a freshly sampled token and returns the events it produced.
    ///
    /// The precedence is fixed here rather than at the call site: any of the ids
    /// the model declares as end-of-turn ends the turn before its budget does,
    /// and a stop sequence is only reported once its text has actually been
    /// matched. A caller free to check them in another order would be able to
    /// report a turn the model finished as one the server truncated.
    ///
    /// `stop` is the whole declared set rather than one id, because a family can
    /// end a turn several ways and the one a turn ends on is not always the one
    /// a single-EOS caller would have recorded.
    pub(crate) fn accept(
        &mut self,
        token: TokenId,
        vocabulary: Option<&Vocabulary>,
        stop: StopTokens,
        now: Instant,
    ) -> Vec<StreamEvent> {
        if self.first_token_at.is_none() {
            self.first_token_at = Some(now);
        }
        self.next_input = None;

        if stop.contains(token) {
            // A marker produces no text, and its id still has to add up: it
            // rides the terminal event rather than a delta.
            self.trailing.push(token.get());
            self.end_marker = Some(token);
            self.stop(StopReason::EndTurn);
            return Vec::new();
        }

        self.out.push(token);
        let fresh = self.text.fresh(vocabulary, &self.out);
        let (text, matched) = match self.watcher.push(&fresh) {
            StopStep::Continue(text) => (text, None),
            StopStep::Stopped { text, sequence } => (text, Some(sequence)),
        };

        self.trailing.push(token.get());
        // A token is reasoning when the stream was inside the trace as it
        // arrived - the closer's own token included - or when it opened the
        // trace. Read off the parser rather than off the text, because the
        // parser is what decides which text is the trace. A token whose text a
        // stop-sequence hold-back is still keeping is counted by the state the
        // parser was last left in, which is off by at most that hold-back.
        let reasoning_before = self.parser.in_reasoning();
        let segments = if vocabulary.is_none() {
            vec![Segment::Text(String::new())]
        } else {
            self.parser.push(&text)
        };
        if reasoning_before || self.parser.in_reasoning() {
            self.reasoning = self.reasoning.saturating_add(1);
        }

        let events = self.emit_segments(segments);

        if let Some(sequence) = matched {
            self.stop(StopReason::StopSequence(sequence));
            return events;
        }
        if u32::try_from(self.out.len()).unwrap_or(u32::MAX) >= self.budget {
            let tail = self.text.flush(vocabulary, &self.out);
            let mut events = events;
            if !tail.is_empty() {
                let (text, sequence) = match self.watcher.push(&tail) {
                    StopStep::Continue(text) => (text, None),
                    StopStep::Stopped { text, sequence } => (text, Some(sequence)),
                };
                let segments = self.parser.push(&text);
                events.extend(self.emit_segments(segments));
                if let Some(sequence) = sequence {
                    self.stop(StopReason::StopSequence(sequence));
                    return events;
                }
            }
            self.stop(StopReason::MaxTokens);
            return events;
        }
        self.next_input = Some(token);
        self.phase = Phase::Decode;
        events
    }

    /// Delivers whatever the stop-sequence hold-back was still keeping.
    pub(crate) fn flush_tail(&mut self) -> Vec<StreamEvent> {
        let tail = self.watcher.flush();
        if tail.is_empty() {
            return Vec::new();
        }
        let segments = self.parser.push(&tail);
        self.emit_segments(segments)
    }

    /// Marks the sequence as stopped for a reason of its own.
    pub(crate) fn stop(&mut self, reason: StopReason) {
        self.phase = Phase::Done;
        self.next_input = None;
        if self.ending.is_none() {
            self.ending = Some(Ending::Stopped(reason));
        }
    }

    /// Marks the sequence as having lost its client.
    ///
    /// Idempotent, and deliberately weaker than [`SeqSlot::stop`]: a sequence
    /// that already ended for a reason of its own keeps that reason, because the
    /// terminal event it produced is the true account of the turn even if
    /// nobody was there to read it.
    pub(crate) fn cancel(&mut self) {
        self.phase = Phase::Done;
        self.next_input = None;
        if self.ending.is_none() {
            self.ending = Some(Ending::Cancelled);
        }
    }

    /// Marks the sequence as having failed underneath.
    pub(crate) fn fail(&mut self, message: String) {
        self.phase = Phase::Done;
        self.next_input = None;
        self.ending = Some(Ending::Failed(message));
    }

    /// The terminal events this turn ends with, in order.
    ///
    /// A cancelled turn produces none: its client is gone, and the send that
    /// discovered that is the send that would carry them.
    pub(crate) fn terminal_events(&mut self, now: Instant) -> Vec<StreamEvent> {
        let ending = self.ending.clone();
        match ending {
            None | Some(Ending::Cancelled) => Vec::new(),
            Some(Ending::Failed(message)) => vec![StreamEvent::Failed { message }],
            Some(Ending::Stopped(mut stop)) => {
                let mut events = Vec::new();
                events.extend(self.flush_tail());
                let finish_segments = self.parser.finish();
                events.extend(self.emit_segments(finish_segments));

                if self.calls_emitted > 0 && stop == StopReason::EndTurn {
                    stop = StopReason::ToolUse;
                    self.ending = Some(Ending::Stopped(StopReason::ToolUse));
                }
                if self.blocks.next == 0 && self.blocks.open.is_none() {
                    events.extend(self.open_block());
                }
                if let Some(event) = self.blocks.close() {
                    events.push(event);
                }

                events.push(StreamEvent::Stop {
                    stop,
                    usage: self.usage(now),
                    token_ids: std::mem::take(&mut self.trailing),
                });
                events
            }
        }
    }

    /// What the turn cost.
    pub(crate) fn usage(&self, now: Instant) -> Usage {
        let elapsed = now.saturating_duration_since(self.started_at).as_secs_f64();
        let ttft = self
            .first_token_at
            .map(|at| at.saturating_duration_since(self.started_at).as_secs_f64());
        // Tokens after the first one, over the time after the first one: a rate
        // that included the prefill would report a number that gets better the
        // longer the answer is, for a decode speed that never changed.
        let after_first = ttft.map_or(0.0, |at| elapsed - at);
        let completion = self.completion_tokens();
        #[expect(
            clippy::cast_precision_loss,
            reason = "a token count large enough to lose precision as f64 is not reachable in one turn"
        )]
        let tokens_per_second = if after_first > 0.0 && completion > 1 {
            (completion - 1) as f64 / after_first
        } else {
            0.0
        };
        Usage {
            prompt_tokens: u64::try_from(self.prompt.len()).unwrap_or(u64::MAX),
            completion_tokens: completion,
            cached_prompt_tokens: self.cached,
            audio_prompt_tokens: u64::from(self.audio_pads),
            reasoning_tokens: self.reasoning,
            time_to_first_token_ms: ttft.map_or(0.0, |at| at * 1000.0),
            tokens_per_second,
            accepted_prediction_tokens: self.spec.1,
        }
    }

    /// Tokens this turn generated, end-of-turn marker included.
    ///
    /// The marker is generated work and is charged for: a caller comparing its
    /// bill against the ids it received should find the marker in both.
    fn completion_tokens(&self) -> u64 {
        let produced = self.out.len() + usize::from(self.ended_on_marker());
        u64::try_from(produced).unwrap_or(u64::MAX)
    }

    /// Whether the turn ended because the model produced its end-of-turn marker.
    fn ended_on_marker(&self) -> bool {
        matches!(self.ending, Some(Ending::Stopped(StopReason::EndTurn)))
    }

    /// How long this turn took to produce its first token.
    ///
    /// A [`Duration`] rather than a count of milliseconds, because the counter
    /// that consumes it and the accounting the client reads want different
    /// units, and a number whose unit lives in its variable name is a number
    /// that gets divided by a thousand twice.
    pub(crate) fn ttft(&self) -> Duration {
        self.first_token_at.map_or(Duration::ZERO, |at| {
            at.saturating_duration_since(self.started_at)
        })
    }

    /// Prompt tokens a resume covered.
    pub(crate) const fn cached_tokens(&self) -> u64 {
        self.cached
    }

    /// When it was admitted.
    pub(crate) const fn started_at(&self) -> Instant {
        self.started_at
    }

    /// One speculative step for it made `drafted` guesses and kept `accepted`.
    pub(crate) fn speculated(&mut self, accepted: u32, drafted: u32) {
        let (d, a, steps) = self.spec;
        self.spec = (d + u64::from(drafted), a + u64::from(accepted), steps + 1);
    }

    /// Guesses drafted, guesses kept, and speculative steps, in that order.
    pub(crate) const fn speculation(&self) -> (u64, u64, u64) {
        self.spec
    }

    /// Prompt tokens a second over the prefill calls this process made for it;
    /// `None` when a resume covered the whole prompt. Mixed with decodes or
    /// not: this is the rate the turn saw, not the device's solo rate.
    pub(crate) fn prefill_rate(&self) -> Option<f64> {
        let (tokens, elapsed) = self.prefill_timed;
        let secs = elapsed.as_secs_f64();
        (tokens > 0 && secs > 0.0).then(|| f64::from(tokens) / secs)
    }
}

#[cfg(test)]
mod tests {
    use super::{Ending, SeqSlot, SlotSetup};
    use crate::scheduler::GenerateRequest;
    use crate::tokenizer::Vocabulary;
    use shiso_core::{SamplingParams, Seed, SessionId, StopTokens, TokenId};
    use shiso_ir::StopReason;
    use std::num::NonZeroU64;
    use std::str::FromStr as _;
    use std::sync::Arc;
    use std::time::Instant;

    /// The stop set of the fake these cases run against: one id, 63.
    fn stop() -> StopTokens {
        StopTokens::one(TokenId::new(63))
    }

    fn slot_with(
        prompt: &[u32],
        budget: u32,
        parser: Box<dyn shiso_parsers::StreamParser>,
    ) -> SeqSlot {
        slot_with_id("s", prompt, budget, parser)
    }

    fn slot_with_id(
        id: &str,
        prompt: &[u32],
        budget: u32,
        parser: Box<dyn shiso_parsers::StreamParser>,
    ) -> SeqSlot {
        let request = GenerateRequest::new(
            Arc::from(prompt.iter().copied().map(TokenId::new).collect::<Vec<_>>()),
            SamplingParams::neutral(),
            Seed::new(1),
            budget,
        );
        let (sink, _rx) = tokio::sync::mpsc::channel(8);
        let mut ids = NonZeroU64::MIN;
        SeqSlot::new(
            request,
            SlotSetup {
                id: id.to_owned(),
                session: SessionId::next(&mut ids).expect("a session id"),
                sink,
                started_at: Instant::now(),
                deadline: None,
                budget,
                keyed: false,
            },
            None,
            parser,
        )
    }

    fn slot(prompt: &[u32], budget: u32) -> SeqSlot {
        let mut s = slot_with(prompt, budget, shiso_parsers::passthrough());
        s.open_block();
        s
    }

    fn test_vocab(mapping: &[(u32, &'static str)]) -> Arc<Vocabulary> {
        let mut vocab_map = serde_json::Map::new();
        vocab_map.insert("<unk>".to_string(), serde_json::json!(0));
        for &(id, word) in mapping {
            vocab_map.insert(word.to_string(), serde_json::json!(id));
        }
        let doc = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": [],
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": serde_json::Value::Null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": true
            },
            "model": {
                "type": "BPE",
                "vocab": vocab_map,
                "merges": []
            }
        })
        .to_string();
        let inner = tokenizers::Tokenizer::from_str(&doc).expect("the test vocabulary loads");
        Arc::new(Vocabulary::from_tokenizer(inner))
    }

    fn slot_with_text(prompt: &[u32], text: &str, budget: u32) -> SeqSlot {
        let mut slot = slot(prompt, budget);
        slot.prompt_text = Arc::from(text);
        slot
    }

    /// A turn that ended on its marker is kept as the next prompt would render
    /// it: the prompt text, the generation, then the marker's own text, beside
    /// exactly the ids the model ran.
    #[test]
    fn a_turn_ended_on_its_marker_is_kept_with_the_ids_it_ran() {
        let vocab = test_vocab(&[(10, "a"), (11, "b"), (63, "<e>")]);
        let mut slot = slot_with_text(&[1, 2], "<p>", 16);
        let now = Instant::now();
        for id in [10, 11, 63] {
            slot.accept(TokenId::new(id), Some(&vocab), stop(), now);
        }
        let (text, tokens) = slot.finished_turn(&vocab).expect("a marker-ended turn");
        assert_eq!(text, "<p>ab<e>");
        let raw: Vec<u32> = tokens.iter().map(|id| id.get()).collect();
        assert_eq!(raw, [1, 2, 10, 11, 63]);
    }

    /// A cut budget, or a surface that rendered no text, leaves nothing a next
    /// prompt's text could be matched against.
    #[test]
    fn a_turn_without_its_marker_or_its_text_is_not_kept() {
        let vocab = test_vocab(&[(10, "a"), (11, "b"), (63, "<e>")]);
        let now = Instant::now();
        let mut cut = slot_with_text(&[1], "<p>", 2);
        cut.accept(TokenId::new(10), Some(&vocab), stop(), now);
        cut.accept(TokenId::new(11), Some(&vocab), stop(), now);
        assert_eq!(cut.ending, Some(Ending::Stopped(StopReason::MaxTokens)));
        assert!(cut.finished_turn(&vocab).is_none());

        let mut untitled = slot(&[1], 16);
        untitled.accept(TokenId::new(10), Some(&vocab), stop(), now);
        untitled.accept(TokenId::new(63), Some(&vocab), stop(), now);
        assert!(untitled.finished_turn(&vocab).is_none());
    }

    /// The budget counts generated tokens, and the turn stops on the one that
    /// reaches it rather than one step later.
    #[test]
    fn a_turn_stops_on_the_token_that_spends_the_last_of_its_budget() {
        let mut slot = slot(&[1, 2], 2);
        let now = Instant::now();
        slot.accept(TokenId::new(10), None, stop(), now);
        assert!(!slot.is_done(), "one of two is not the end");
        slot.accept(TokenId::new(11), None, stop(), now);
        assert_eq!(slot.ending, Some(Ending::Stopped(StopReason::MaxTokens)));
        assert_eq!(slot.usage(now).completion_tokens, 2);
    }

    /// The marker itself is charged for and carried as an id, but produces no
    /// text delta: a client that rendered it would show the caller a control
    /// token.
    #[test]
    fn an_end_of_turn_marker_ends_the_turn_and_is_still_paid_for() {
        let mut slot = slot(&[1], 16);
        let now = Instant::now();
        let events = slot.accept(TokenId::new(10), None, stop(), now);
        assert_eq!(events.len(), 1);
        let events = slot.accept(TokenId::new(63), None, stop(), now);
        assert!(events.is_empty(), "a marker produces no delta: {events:?}");
        assert_eq!(slot.ending, Some(Ending::Stopped(StopReason::EndTurn)));
        assert_eq!(slot.usage(now).completion_tokens, 2);
    }

    /// A family that declares several ends its turn on any of them, and not only
    /// on the one a single-EOS caller would have recorded.
    #[test]
    fn a_turn_ends_on_any_id_the_model_declared() {
        let stop = StopTokens::new(&[TokenId::new(1), TokenId::new(106), TokenId::new(50)])
            .expect("three ids fit");
        for ending in [TokenId::new(1), TokenId::new(106), TokenId::new(50)] {
            let mut slot = slot(&[1], 16);
            slot.accept(ending, None, stop, Instant::now());
            assert_eq!(
                slot.ending,
                Some(Ending::Stopped(StopReason::EndTurn)),
                "{ending:?} is declared and did not end the turn"
            );
        }
        let mut slot = slot(&[1], 16);
        slot.accept(TokenId::new(2), None, stop, Instant::now());
        assert_eq!(slot.ending, None, "an id nobody declared ended a turn");
    }

    /// A client that hung up does not overwrite the reason a turn had already
    /// ended for, because that reason is the true account of what the model did.
    #[test]
    fn cancelling_a_turn_that_already_ended_keeps_the_reason_it_ended_for() {
        let mut slot = slot(&[1], 1);
        slot.accept(TokenId::new(10), None, stop(), Instant::now());
        slot.cancel();
        assert_eq!(slot.ending, Some(Ending::Stopped(StopReason::MaxTokens)));
    }

    /// A resume is charged as cached rather than as prompt work, and the turn
    /// stops being a cold one - the distinction the store reasons about.
    #[test]
    fn a_resumed_prefix_is_cached_rather_than_prefilled_and_is_not_cold() {
        let mut slot = slot(&[1, 2, 3, 4], 8);
        assert!(slot.was_cold());
        slot.resumed(3);
        assert!(!slot.was_cold());
        assert_eq!(slot.cached_tokens(), 3);
        assert_eq!(slot.facts().prompt_remaining, 1);
        assert_eq!(slot.usage(Instant::now()).cached_prompt_tokens, 3);
    }

    /// The frontier is the prompt plus what has been generated, and the session
    /// has run all of it but the last sampled token.
    #[test]
    fn the_frontier_is_the_prompt_and_everything_generated_after_it() {
        let mut slot = slot(&[1, 2], 8);
        assert_eq!(slot.frontier_len(), 2);
        slot.accept(TokenId::new(10), None, stop(), Instant::now());
        assert_eq!(slot.frontier_len(), 3);
        assert_eq!(
            slot.frontier_path(),
            vec![TokenId::new(1), TokenId::new(2), TokenId::new(10)]
        );
        assert_eq!(slot.written_len(), 2, "the sampled token has not been run");
    }

    /// A sequence that is still reading prompt must never appear in a decode
    /// batch, and one standing on a token must never be asked to read prompt.
    #[test]
    fn a_sequence_is_either_reading_prompt_or_waiting_to_decode_and_never_both() {
        let mut slot = slot(&[1, 2, 3], 8);
        assert_eq!(slot.facts().prompt_remaining, 3);
        assert!(!slot.facts().ready_to_decode);

        slot.advance_prefill(3);
        assert!(slot.prompt_done());
        slot.accept(TokenId::new(10), None, stop(), Instant::now());
        let facts = slot.facts();
        assert!(facts.ready_to_decode);
        assert_eq!(facts.prompt_remaining, 0);
        assert!(slot.decode_step().is_some());
    }

    #[test]
    fn a_gemma_thinking_run_becomes_a_thinking_block() {
        use shiso_ir::{BlockDelta, BlockStart, StreamEvent};
        let vocab = test_vocab(&[
            (10, "<|channel>thought\n"),
            (11, "let me see\n"),
            (12, "<channel|>"),
            (13, "Paris."),
            (63, "<turn|>"),
        ]);
        let mut slot = slot_with(&[1], 16, shiso_parsers::gemma4());
        let now = Instant::now();
        let ev1 = slot.accept(TokenId::new(10), Some(&vocab), stop(), now);
        let ev2 = slot.accept(TokenId::new(11), Some(&vocab), stop(), now);
        let ev3 = slot.accept(TokenId::new(12), Some(&vocab), stop(), now);
        let ev4 = slot.accept(TokenId::new(13), Some(&vocab), stop(), now);
        let ev5 = slot.accept(TokenId::new(63), Some(&vocab), stop(), now);
        let term = slot.terminal_events(now);

        let mut all_events = Vec::new();
        all_events.extend(ev1);
        all_events.extend(ev2);
        all_events.extend(ev3);
        all_events.extend(ev4);
        all_events.extend(ev5);
        all_events.extend(term);

        assert_eq!(
            all_events[0],
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::Thinking,
            }
        );
        assert_eq!(
            all_events[1],
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Thinking {
                    text: "let me see\n".to_string(),
                    token_ids: vec![10, 11],
                },
            }
        );
        assert_eq!(all_events[2], StreamEvent::BlockStop { index: 0 });
        assert_eq!(
            all_events[3],
            StreamEvent::BlockStart {
                index: 1,
                block: BlockStart::Text,
            }
        );
        assert_eq!(
            all_events[4],
            StreamEvent::BlockDelta {
                index: 1,
                delta: BlockDelta::Text {
                    text: "Paris.".to_string(),
                    token_ids: vec![12, 13],
                },
            }
        );
        assert_eq!(all_events[5], StreamEvent::BlockStop { index: 1 });
        match &all_events[6] {
            StreamEvent::Stop {
                stop, token_ids, ..
            } => {
                assert_eq!(*stop, StopReason::EndTurn);
                assert_eq!(*token_ids, vec![63]);
            }
            other => panic!("expected Stop, got {other:?}"),
        }
    }

    /// Two turns of one conversation never hand out the same call id: clients key
    /// tool results by id, and a second `call_0` answers the first turn's call.
    #[test]
    fn call_ids_carry_the_sequence_id() {
        let a = slot_with_id("a", &[1], 8, shiso_parsers::passthrough());
        let b = slot_with_id("b", &[1], 8, shiso_parsers::passthrough());
        assert_ne!(a.call_id(0), b.call_id(0));
        assert!(a.call_id(0).starts_with("call_"));
    }

    #[test]
    fn a_tool_call_ends_the_turn_as_tool_use() {
        use shiso_ir::{BlockDelta, BlockStart, StreamEvent};
        let vocab = test_vocab(&[(10, "<|tool_call>call:f{}<tool_call|>"), (63, "<turn|>")]);
        let mut slot = slot_with(&[1], 16, shiso_parsers::gemma4());
        let now = Instant::now();
        let ev1 = slot.accept(TokenId::new(10), Some(&vocab), stop(), now);
        let ev2 = slot.accept(TokenId::new(63), Some(&vocab), stop(), now);
        let term = slot.terminal_events(now);

        let mut all_events = Vec::new();
        all_events.extend(ev1);
        all_events.extend(ev2);
        all_events.extend(term);

        assert_eq!(slot.ending, Some(Ending::Stopped(StopReason::ToolUse)));
        assert_eq!(
            all_events[0],
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::ToolUse {
                    id: slot.call_id(0),
                    name: "f".to_string(),
                },
            }
        );
        match &all_events[1] {
            StreamEvent::BlockDelta {
                index,
                delta: BlockDelta::ToolInput { fragment, .. },
            } => {
                assert_eq!(*index, 0);
                let json_val: serde_json::Value =
                    serde_json::from_str(fragment).expect("tool input fragment must parse as json");
                assert_eq!(json_val, serde_json::json!({}));
            }
            other => panic!("expected ToolInput delta, got {other:?}"),
        }
        assert_eq!(all_events[2], StreamEvent::BlockStop { index: 0 });
        match &all_events[3] {
            StreamEvent::Stop { stop, .. } => {
                assert_eq!(*stop, StopReason::ToolUse);
            }
            other => panic!("expected Stop, got {other:?}"),
        }
    }

    #[test]
    fn pass_through_is_today() {
        use shiso_ir::{BlockDelta, BlockStart, StreamEvent};
        let vocab = test_vocab(&[(10, "Hello "), (11, "world!"), (63, "<turn|>")]);
        let mut slot = slot_with(&[1], 16, shiso_parsers::passthrough());
        let now = Instant::now();
        let ev1 = slot.accept(TokenId::new(10), Some(&vocab), stop(), now);
        let ev2 = slot.accept(TokenId::new(11), Some(&vocab), stop(), now);
        let ev3 = slot.accept(TokenId::new(63), Some(&vocab), stop(), now);
        let term = slot.terminal_events(now);

        let mut all_events = Vec::new();
        all_events.extend(ev1);
        all_events.extend(ev2);
        all_events.extend(ev3);
        all_events.extend(term);

        assert_eq!(
            all_events[0],
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::Text,
            }
        );
        assert_eq!(
            all_events[1],
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: "Hello ".to_string(),
                    token_ids: vec![10],
                },
            }
        );
        assert_eq!(
            all_events[2],
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: "world!".to_string(),
                    token_ids: vec![11],
                },
            }
        );
        assert_eq!(all_events[3], StreamEvent::BlockStop { index: 0 });
        match &all_events[4] {
            StreamEvent::Stop {
                stop, token_ids, ..
            } => {
                assert_eq!(*stop, StopReason::EndTurn);
                assert_eq!(*token_ids, vec![63]);
            }
            other => panic!("expected Stop, got {other:?}"),
        }
    }

    #[test]
    fn token_ids_add_up() {
        use shiso_ir::{BlockDelta, StreamEvent};
        let vocab = test_vocab(&[
            (10, "<|channel>thought\n"),
            (11, "thinking\n"),
            (12, "<channel|>"),
            (13, "<|tool_call>call:f{}<tool_call|>"),
            (14, "Finished."),
            (63, "<turn|>"),
        ]);
        let mut slot = slot_with(&[1], 16, shiso_parsers::gemma4());
        let now = Instant::now();
        let mut accepted_ids = Vec::new();
        let mut all_events = Vec::new();

        for id in [10, 11, 12, 13, 14, 63] {
            accepted_ids.push(id);
            all_events.extend(slot.accept(TokenId::new(id), Some(&vocab), stop(), now));
        }
        all_events.extend(slot.terminal_events(now));

        let mut reported_ids = Vec::new();
        for event in all_events {
            match event {
                StreamEvent::BlockDelta {
                    delta:
                        BlockDelta::Text { token_ids, .. }
                        | BlockDelta::Thinking { token_ids, .. }
                        | BlockDelta::ToolInput { token_ids, .. },
                    ..
                }
                | StreamEvent::Stop { token_ids, .. } => {
                    reported_ids.extend(token_ids);
                }
                _ => {}
            }
        }

        assert_eq!(reported_ids, accepted_ids);
    }

    /// Inside tool-call structure the draw is greedy whatever the request asked;
    /// outside it the request's own temperature stands.
    #[test]
    fn the_draw_is_greedy_exactly_where_the_parser_asks() {
        let mut slot = slot_with(
            &[1],
            8,
            shiso_parsers::parser(
                shiso_parsers::ReasoningScheme::None,
                shiso_parsers::ToolCallScheme::Qwen35,
            ),
        );
        let asked = SamplingParams {
            top_k: std::num::NonZeroU32::new(20),
            ..SamplingParams::neutral()
        };
        slot.sampling = asked;
        assert!(!asked.is_greedy(), "the case needs a sampling request");
        assert_eq!(slot.draw().0, asked);
        let _ = slot.parser.push("<tool_call>\n<func");
        assert_eq!(
            slot.draw().0,
            SamplingParams {
                temperature: 0.0,
                ..asked
            }
        );
        // The request's own settings are untouched: the override is per draw.
        assert_eq!(slot.sampling, asked);
    }

    /// A qwen4exp turn of the shape the matrix recorded - a trace, a sentence,
    /// two calls - run through the slot, the one fold, and every encoder.
    ///
    /// The framing newlines the template puts around the calls are not content
    /// on any surface, and the trace's tokens are counted as reasoning: the
    /// three up to and including the closer, and none after it.
    #[test]
    fn a_thinking_turn_with_calls_encodes_trimmed_prose_and_counts_its_trace() {
        use shiso_ir::{ResponseBuilder, StreamEvent};
        const CALL_A: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>";
        const CALL_B: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\nb\n</parameter>\n</function>\n</tool_call>";
        let vocab = test_vocab(&[
            (10, "plan one "),
            (11, "plan two\n"),
            (12, "</think>"),
            (13, "\n\n"),
            (14, "I'll read both."),
            (15, CALL_A),
            (16, "\n"),
            (17, CALL_B),
            (63, "<|im_end|>"),
        ]);
        let parser = shiso_parsers::parser_for_prompt(
            shiso_parsers::ReasoningScheme::Qwen3,
            shiso_parsers::ToolCallScheme::Qwen35,
            "<|im_start|>user\nread a and b<|im_end|>\n<|im_start|>assistant\n<think>\n",
        );
        let mut slot = slot_with(&[1], 32, parser);
        let now = Instant::now();
        let mut events = vec![StreamEvent::Start {
            id: "turn".to_owned(),
            model: "m".to_owned(),
            created: 0,
        }];
        for id in [10, 11, 12, 13, 14, 13, 15, 16, 17, 63] {
            events.extend(slot.accept(TokenId::new(id), Some(&vocab), stop(), now));
        }
        events.extend(slot.terminal_events(now));
        let mut builder = ResponseBuilder::new();
        for event in events {
            builder.accept(event).expect("the slot's events fold");
        }
        let response = builder.finish().expect("the turn is complete");
        assert_eq!(response.stop, StopReason::ToolUse);
        assert_eq!(response.usage.reasoning_tokens, 3);

        let chat = crate::openai::chat_encode::encode_response(
            &response,
            &crate::openai::chat_encode::EncodeOptions::default(),
        );
        let message = &chat["choices"][0]["message"];
        assert_eq!(message["content"], "I'll read both.");
        assert_eq!(message["reasoning_content"], "plan one plan two\n");
        assert_eq!(message["tool_calls"].as_array().map(Vec::len), Some(2));
        assert_eq!(
            chat["usage"]["completion_tokens_details"]["reasoning_tokens"],
            3
        );

        let responses = crate::openai::responses_encode::encode_response(
            &response,
            &crate::openai::responses_encode::ResponseOptions::default(),
        );
        assert_eq!(
            responses["usage"]["output_tokens_details"]["reasoning_tokens"],
            3
        );

        let anthropic = crate::anthropic::messages_encode::encode_response(
            &response,
            &crate::anthropic::messages_encode::EncodeOptions::default(),
        );
        let texts: Vec<&str> = anthropic["content"]
            .as_array()
            .expect("content is a list")
            .iter()
            .filter(|block| block["type"] == "text")
            .filter_map(|block| block["text"].as_str())
            .collect();
        assert_eq!(texts, vec!["I'll read both."]);
    }

    /// A turn that is only calls has no prose at all, so its content is `null`
    /// rather than the newlines between the calls.
    #[test]
    fn a_turn_of_only_calls_has_null_content() {
        use shiso_ir::{ResponseBuilder, StreamEvent};
        const CALL: &str = "<tool_call>\n<function=now>\n</function>\n</tool_call>";
        let vocab = test_vocab(&[
            (10, "plan\n"),
            (12, "</think>"),
            (13, "\n\n"),
            (15, CALL),
            (16, "\n"),
            (63, "<|im_end|>"),
        ]);
        let parser = shiso_parsers::parser_for_prompt(
            shiso_parsers::ReasoningScheme::Qwen3,
            shiso_parsers::ToolCallScheme::Qwen35,
            "<|im_start|>assistant\n<think>\n",
        );
        let mut slot = slot_with(&[1], 32, parser);
        let now = Instant::now();
        let mut events = vec![StreamEvent::Start {
            id: "turn".to_owned(),
            model: "m".to_owned(),
            created: 0,
        }];
        for id in [10, 12, 13, 15, 16, 15, 16, 15, 63] {
            events.extend(slot.accept(TokenId::new(id), Some(&vocab), stop(), now));
        }
        events.extend(slot.terminal_events(now));
        let mut builder = ResponseBuilder::new();
        for event in events {
            builder.accept(event).expect("the slot's events fold");
        }
        let response = builder.finish().expect("the turn is complete");
        let chat = crate::openai::chat_encode::encode_response(
            &response,
            &crate::openai::chat_encode::EncodeOptions::default(),
        );
        let message = &chat["choices"][0]["message"];
        assert_eq!(message["content"], serde_json::Value::Null);
        assert_eq!(message["tool_calls"].as_array().map(Vec::len), Some(3));
        assert_eq!(response.usage.reasoning_tokens, 2);
    }

    /// A multi-byte character split across two tokens must be held back until
    /// complete, so emojis and Unicode characters do not decode as replacement
    /// characters and corrupt the slice boundary.
    #[test]
    fn multi_byte_emojis_survive_incremental_streaming() {
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": [],
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": serde_json::Value::Null,
            "post_processor": serde_json::Value::Null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "model": {
                "type": "BPE",
                "vocab": {
                    "ĠðŁĺ": 0,
                    "Ĭ": 1,
                    "hello": 2,
                },
                "merges": [],
            },
        })
        .to_string();
        let tok = tokenizers::Tokenizer::from_str(&document).expect("stub loads");
        let vocab = Vocabulary::from_tokenizer(tok);

        let mut cursor = super::TextCursor::default();
        let tokens = [TokenId::new(2)];
        let first = cursor.fresh(Some(&vocab), &tokens);
        assert_eq!(first, "hello");

        let tokens = [TokenId::new(2), TokenId::new(0)];
        let second = cursor.fresh(Some(&vocab), &tokens);
        assert_eq!(second, "", "the incomplete emoji prefix is held back");

        let tokens = [TokenId::new(2), TokenId::new(0), TokenId::new(1)];
        let third = cursor.fresh(Some(&vocab), &tokens);
        assert_eq!(third, " 😊", "the completed emoji arrives intact");
    }

    #[test]
    fn text_cursor_flushes_incomplete_tail_on_demand() {
        let document = serde_json::json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": [],
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": serde_json::Value::Null,
            "post_processor": serde_json::Value::Null,
            "decoder": {
                "type": "ByteLevel",
                "add_prefix_space": false,
                "trim_offsets": false,
                "use_regex": false
            },
            "model": {
                "type": "BPE",
                "vocab": {
                    "ĠðŁĺ": 0,
                    "hello": 2,
                },
                "merges": [],
            },
        })
        .to_string();
        let tok = tokenizers::Tokenizer::from_str(&document).expect("stub loads");
        let vocab = Vocabulary::from_tokenizer(tok);

        let mut cursor = super::TextCursor::default();
        let tokens = [TokenId::new(2), TokenId::new(0)];
        let _ = cursor.fresh(Some(&vocab), &tokens[..1]);
        let second = cursor.fresh(Some(&vocab), &tokens);
        assert_eq!(second, "");

        let tail = cursor.flush(Some(&vocab), &tokens);
        assert_eq!(tail, " \u{FFFD}", "flushing delivers any uncompleted tail");
    }
}
