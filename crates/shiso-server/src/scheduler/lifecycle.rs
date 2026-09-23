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
        // A shrinking decode is possible in principle - a vocabulary is free to
        // render a pair differently from its parts - and `get` rather than an
        // index is what keeps that a missing fragment rather than a panic.
        let fresh = whole.get(self.emitted..).unwrap_or_default().to_owned();
        self.emitted = whole.len();
        fresh
    }
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
    /// The rendered prompt as text, which is the checkpoint tier's key.
    pub(crate) prompt_text: Arc<str>,
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
    /// How many tokens this turn may still generate.
    budget: u32,
    /// Sampling settings and the stream they draw from.
    sampling: SamplingParams,
    /// The PRNG state, which is per sequence so that two turns in one batch do
    /// not draw from each other's stream.
    seed: Seed,
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
    /// The encoder rows its prompt's placeholders stand for.
    ///
    /// Owned by the sequence rather than by the tick that encoded them: the
    /// rows are read by the forward pass and the forward pass happens later,
    /// on a tick this sequence may not be alone on. `None` is a text-only
    /// request and takes the launches it took before any of this existed.
    media: Option<EncodedMedia>,
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
}

impl SeqSlot {
    /// A sequence over `request`, holding `setup.session`.
    pub(crate) fn new(
        request: crate::scheduler::GenerateRequest,
        setup: SlotSetup,
        media: Option<EncodedMedia>,
        parser: Box<dyn shiso_parsers::StreamParser>,
    ) -> Self {
        Self {
            media,
            parser,
            blocks: BlockCursor::default(),
            calls_emitted: 0,
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
            budget: setup.budget,
            sampling: request.sampling,
            seed: request.seed,
            watcher: StopWatcher::new(&request.stop),
            text: TextCursor::default(),
            trailing: Vec::new(),
            phase: Phase::Prefill,
            ending: None,
            started_at: setup.started_at,
            first_token_at: None,
            deadline: setup.deadline,
            stored_len: 0,
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
            indivisible_prompt: self.media.is_some(),
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

    /// Whether this sequence's prefix may be stored or resumed.
    ///
    /// The checkpoint tier keys on the rendered prompt TEXT, and a prompt that
    /// carries a picture renders to the same text whichever picture it is: the
    /// placeholder marker is one token and the pixels are not in the string. So
    /// a media request must never file a prefix under that key and must never
    /// stand on one somebody else filed, because both directions serve one
    /// picture's activations for another picture's request - fluently, and with
    /// nothing in the response to show for it.
    pub(crate) fn checkpointable(&self) -> bool {
        !self.prompt_text.is_empty() && self.media.is_none()
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

    /// The whole path its session stands on: prompt then generated.
    pub(crate) fn frontier_path(&self) -> Vec<TokenId> {
        let mut path = Vec::with_capacity(self.prompt.len() + self.out.len());
        path.extend_from_slice(&self.prompt);
        path.extend_from_slice(&self.out);
        path
    }

    /// Positions its session has written.
    pub(crate) fn frontier_len(&self) -> u32 {
        u32::try_from(self.prompt.len().saturating_add(self.out.len())).unwrap_or(u32::MAX)
    }

    /// Whether the whole prompt was prefilled by this process.
    pub(crate) const fn was_cold(&self) -> bool {
        self.cold
    }

    /// Its sampling settings and PRNG stream, for the one sampler that draws
    /// for it.
    pub(crate) fn draw(&mut self) -> (&SamplingParams, &mut Seed) {
        (&self.sampling, &mut self.seed)
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
        let segments = if vocabulary.is_none() {
            vec![Segment::Text(String::new())]
        } else {
            self.parser.push(&text)
        };

        let events = self.emit_segments(segments);

        if let Some(sequence) = matched {
            self.stop(StopReason::StopSequence(sequence));
            return events;
        }
        if u32::try_from(self.out.len()).unwrap_or(u32::MAX) >= self.budget {
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
            time_to_first_token_ms: ttft.map_or(0.0, |at| at * 1000.0),
            tokens_per_second,
            ..Usage::default()
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

    /// The frontier is the prompt plus what has been generated, which is the
    /// path the checkpoint tier will refuse to store under any other length.
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
}
