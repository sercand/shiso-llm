//! The Messages stream: named events over the same block list.
//!
//! # The sequence
//!
//! `message_start`, then - per block of the turn, in block order - a
//! `content_block_start`, the deltas that fill it in, and a
//! `content_block_stop`; then `message_delta` carrying why the turn ended and
//! what it cost, then `message_stop`. Every frame has an `event:` line naming
//! the same `type` its data carries, because clients of this format dispatch on
//! the event name and validate against the type.
//!
//! # Why there is no index to translate
//!
//! The neutral stream numbers blocks by their position in the finished message,
//! and so does this format. The two numbering schemes are the same numbering
//! scheme, so this encoder carries no map from one to the other - unlike the
//! chat surface, which has to number tool calls among themselves. What state it
//! does keep is only which indices are open, so that a turn which ends with a
//! block still open is closed rather than left dangling.
//!
//! # Why a failure is an `error` event and not a silent end
//!
//! A stream that has already sent a 200 cannot change its status, so a failure
//! has to arrive as content. Ending the stream with `end_turn` instead - which
//! is what a translator that has no error arm does - tells the client the model
//! finished, and a truncated answer that claims to be complete is the one
//! failure nobody downstream can detect. So a failure is an `error` frame
//! carrying the same body every other failure on this surface uses, and then
//! `message_stop`, because a client waiting for the terminator waits forever
//! either way.
//!
//! # What `message_start` cannot say
//!
//! This format's opening frame carries the prompt's token count, split into
//! what was prefilled and what was read from a cache. The pipeline counted the
//! rendered prompt before the stream opened, so the total is known; the split
//! is not, because whether a stored prefix covers the prompt is decided once
//! the scheduler admits the turn. The opening frame therefore reports the whole
//! prompt as input, and `message_delta`, whose usage is the whole turn's,
//! carries the split. A client that reads only the opening frame over-reports
//! a cache hit; a client that reads the closing one, as every client of this
//! format does, is exact.

use std::collections::BTreeSet;

use serde_json::{Value, json};
use shiso_ir::{BlockDelta, BlockStart, StopReason, StreamEvent, Usage};

use crate::anthropic::messages_encode::{EncodeOptions, encode_usage, stop_reason, stop_sequence};
use crate::api::sse::{Frame, FrameEncoder};

/// Turns the neutral stream into this format's named events.
#[derive(Debug)]
pub struct MessagesEventEncoder {
    options: EncodeOptions,
    /// Whether the opening frame has been sent. Every emitting arm checks it,
    /// so a stream whose first event is a block - which the engine does not
    /// produce, but which a future producer could - still opens a message
    /// before it describes one.
    started: bool,
    /// The blocks that have opened and not closed, in index order.
    open: BTreeSet<u32>,
    /// Whether the closing frames have been sent, so a stream that both fails
    /// and ends does not send two terminators.
    closed: bool,
    /// Tokens the rendered prompt occupies, known before the turn starts.
    ///
    /// This format's opening frame carries the input accounting, and the frame
    /// is sent before a single token has been generated - so the number cannot
    /// come from the turn's own usage, which does not exist yet. It comes from
    /// the pipeline, which rendered the prompt and counted it. A zero here would
    /// be corrected by the terminal frame, and a client that reads the opening
    /// one to decide whether to proceed would have been told the prompt was
    /// free.
    input_tokens: u64,
}

impl MessagesEventEncoder {
    /// An encoder for one request.
    #[must_use]
    pub fn new(options: EncodeOptions) -> Self {
        Self {
            options,
            started: false,
            open: BTreeSet::new(),
            closed: false,
            input_tokens: 0,
        }
    }

    /// The same encoder, told how long the rendered prompt was.
    #[must_use]
    pub const fn with_input_tokens(mut self, tokens: u64) -> Self {
        self.input_tokens = tokens;
        self
    }

    /// One frame, named and carrying its own name as `type`.
    fn emit(name: &str, mut data: Value) -> Frame {
        if let Some(fields) = data.as_object_mut() {
            fields.insert("type".to_owned(), json!(name));
        }
        Frame::named(name, data.to_string())
    }

    /// The opening frame, if it has not been sent yet.
    fn ensure_started(&mut self) -> Vec<Frame> {
        if self.started {
            return Vec::new();
        }
        self.started = true;
        vec![Self::emit(
            "message_start",
            json!({"message": {
                "id": self.options.id,
                "type": "message",
                "role": "assistant",
                "model": self.options.model,
                "content": [],
                "stop_reason": Value::Null,
                "stop_sequence": Value::Null,
                "usage": {"input_tokens": self.input_tokens, "output_tokens": 0},
            }}),
        )]
    }

    /// The frames that announce one block.
    fn open_block(&mut self, index: u32, block: &BlockStart) -> Vec<Frame> {
        let content_block = match block {
            BlockStart::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
            BlockStart::ToolUse { id, name } => {
                json!({"type": "tool_use", "id": id, "name": name, "input": {}})
            }
            // Prose, a refusal, and any block kind added after this was
            // written. A refusal is prose the caller is meant to read and this
            // format's only carrier of prose is a text block, which is also
            // what the blocking body says about the same turn.
            _ => json!({"type": "text", "text": ""}),
        };
        let mut frames = self.ensure_started();
        self.open.insert(index);
        frames.push(Self::emit(
            "content_block_start",
            json!({"index": index, "content_block": content_block}),
        ));
        frames
    }

    /// The frame one fragment produces.
    fn extend(&mut self, index: u32, delta: &BlockDelta) -> Vec<Frame> {
        let body = match delta {
            // A refusal streams as prose, for the same reason it is encoded as
            // a text block in the blocking body: this format has one carrier
            // for text the caller is meant to read.
            BlockDelta::Text { text, .. } | BlockDelta::Refusal { text } => {
                json!({"type": "text_delta", "text": text})
            }
            BlockDelta::Thinking { text, .. } => {
                json!({"type": "thinking_delta", "thinking": text})
            }
            // The signature arrives once, whole, and this format has a delta
            // for exactly that: a trace replayed into a later turn needs it, so
            // a stream that dropped it would produce a turn that cannot be
            // replayed.
            BlockDelta::ThinkingSignature(signature) => {
                json!({"type": "signature_delta", "signature": signature})
            }
            BlockDelta::ToolInput { fragment, .. } => {
                json!({"type": "input_json_delta", "partial_json": fragment})
            }
            ref other => {
                tracing::debug!(?other, "no content-block delta for this fragment");
                return Vec::new();
            }
        };
        let mut frames = self.ensure_started();
        frames.push(Self::emit(
            "content_block_delta",
            json!({"index": index, "delta": body}),
        ));
        frames
    }

    /// The frame that closes one block, if it is open.
    fn close_block(&mut self, index: u32) -> Vec<Frame> {
        if !self.open.remove(&index) {
            return Vec::new();
        }
        vec![Self::emit("content_block_stop", json!({"index": index}))]
    }

    /// The frames that end the turn: whatever is still open, then why it ended
    /// and what it cost, then the terminator.
    fn close(&mut self, stop: &StopReason, usage: &Usage) -> Vec<Frame> {
        let mut frames = self.ensure_started();
        let open: Vec<u32> = self.open.iter().copied().collect();
        for index in open {
            frames.extend(self.close_block(index));
        }
        self.closed = true;
        frames.push(Self::emit(
            "message_delta",
            json!({
                "delta": {
                    "stop_reason": stop_reason(stop),
                    "stop_sequence": stop_sequence(stop),
                },
                "usage": encode_usage(usage),
            }),
        ));
        frames.push(Self::emit("message_stop", json!({})));
        frames
    }

    /// The frames a failure produces: what went wrong, and then the
    /// terminator.
    fn fail(&mut self, message: &str) -> Vec<Frame> {
        let mut frames = self.ensure_started();
        self.closed = true;
        frames.push(Self::emit(
            "error",
            json!({"error": {"type": "api_error", "message": message}}),
        ));
        frames.push(Self::emit("message_stop", json!({})));
        frames
    }
}

impl FrameEncoder for MessagesEventEncoder {
    fn on_event(&mut self, event: StreamEvent) -> Vec<Frame> {
        match event {
            StreamEvent::Start { .. } => self.ensure_started(),
            StreamEvent::BlockStart { index, block } => self.open_block(index, &block),
            StreamEvent::BlockDelta { index, delta } => self.extend(index, &delta),
            StreamEvent::BlockStop { index } => self.close_block(index),
            StreamEvent::Stop { stop, usage, .. } => self.close(&stop, &usage),
            StreamEvent::Failed { message } => self.fail(&message),
            // The stream is open on purpose: an event added after this was
            // written has no frame here, and a client is better served by the
            // frames it does understand than by a stream that stops.
            ref other => {
                tracing::debug!(?other, "no message event for this event");
                Vec::new()
            }
        }
    }

    fn on_end(&mut self) -> Vec<Frame> {
        if self.closed {
            return Vec::new();
        }
        // The producer stopped without a terminal event. Ending with `end_turn`
        // would say the model finished, which is the claim this encoder refuses
        // to make about a turn nobody completed.
        tracing::warn!("a stream ended with no terminal event; failing it");
        self.fail("the engine's stream ended without completing the turn")
    }
}

#[cfg(test)]
mod tests {
    use super::MessagesEventEncoder;
    use crate::anthropic::messages_encode::EncodeOptions;
    use crate::api::sse::{Frame, FrameEncoder as _, drain};
    use serde_json::Value;
    use shiso_ir::{BlockDelta, BlockStart, StopReason, StreamEvent, Usage};

    fn options() -> EncodeOptions {
        EncodeOptions {
            id: "msg_abc".to_owned(),
            model: "a-model".to_owned(),
        }
    }

    fn encoder() -> MessagesEventEncoder {
        MessagesEventEncoder::new(options())
    }

    fn start() -> StreamEvent {
        StreamEvent::Start {
            id: "bare".to_owned(),
            model: "a-model".to_owned(),
            created: 1700,
        }
    }

    fn stop(stop: StopReason) -> StreamEvent {
        StreamEvent::Stop {
            stop,
            usage: Usage {
                prompt_tokens: 11,
                completion_tokens: 2,
                ..Usage::default()
            },
            token_ids: Vec::new(),
        }
    }

    fn names(frames: &[Frame]) -> Vec<String> {
        frames
            .iter()
            .map(|frame| frame.name.clone().unwrap_or_default())
            .collect()
    }

    fn data(frames: &[Frame]) -> Vec<Value> {
        frames
            .iter()
            .map(|frame| serde_json::from_str(&frame.data).expect("every frame is JSON"))
            .collect()
    }

    fn text_turn() -> Vec<StreamEvent> {
        vec![
            start(),
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::Text,
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: "Par".to_owned(),
                    token_ids: vec![1],
                },
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: "is".to_owned(),
                    token_ids: vec![2],
                },
            },
            StreamEvent::BlockStop { index: 0 },
            stop(StopReason::EndTurn),
        ]
    }

    #[test]
    fn a_text_turn_is_framed_exactly_as_a_client_of_this_format_reads_it() {
        let frames = drain(encoder(), text_turn());
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "content_block_delta",
                "content_block_delta",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
        let bodies = data(&frames);
        assert_eq!(bodies[0]["type"], "message_start");
        assert_eq!(bodies[0]["message"]["id"], "msg_abc");
        assert_eq!(bodies[0]["message"]["role"], "assistant");
        assert_eq!(bodies[0]["message"]["model"], "a-model");
        assert_eq!(bodies[1]["content_block"]["type"], "text");
        assert_eq!(bodies[2]["delta"]["type"], "text_delta");
        assert_eq!(bodies[2]["delta"]["text"], "Par");
        assert_eq!(bodies[4]["index"], 0);
        assert_eq!(bodies[5]["delta"]["stop_reason"], "end_turn");
        assert_eq!(bodies[5]["delta"]["stop_sequence"], Value::Null);
        assert_eq!(bodies[5]["usage"]["input_tokens"], 11);
        assert_eq!(bodies[5]["usage"]["output_tokens"], 2);
    }

    /// A prompt a stored prefix covered whole was read, not prefilled: the
    /// closing frame reports it as a cache read and none of it as input.
    #[test]
    fn a_whole_cache_hit_closes_with_no_input_and_the_prompt_as_a_read() {
        let mut turn = text_turn();
        turn.pop();
        turn.push(StreamEvent::Stop {
            stop: StopReason::EndTurn,
            usage: Usage {
                prompt_tokens: 11,
                completion_tokens: 2,
                cached_prompt_tokens: 11,
                ..Usage::default()
            },
            token_ids: Vec::new(),
        });
        let bodies = data(&drain(encoder(), turn));
        let usage = &bodies[5]["usage"];
        assert_eq!(usage["input_tokens"], 0, "{usage}");
        assert_eq!(usage["cache_read_input_tokens"], 11, "{usage}");
    }

    /// The `event:` line and the `type` in the data have to agree, because a
    /// client dispatches on one and validates against the other.
    #[test]
    fn every_frames_name_is_the_type_its_data_carries() {
        let frames = drain(encoder(), text_turn());
        for frame in &frames {
            let body: Value = serde_json::from_str(&frame.data).expect("a frame is JSON");
            assert_eq!(
                frame.name.as_deref(),
                body["type"].as_str(),
                "the name and the type disagree: {frame:?}"
            );
        }
    }

    /// Text and thinking are two blocks, and their indices are the positions
    /// they hold in the finished message - the same numbers the blocking body
    /// lists them at.
    #[test]
    fn a_thinking_block_and_a_text_block_keep_their_own_indices() {
        let frames = drain(
            encoder(),
            vec![
                start(),
                StreamEvent::BlockStart {
                    index: 0,
                    block: BlockStart::Thinking,
                },
                StreamEvent::BlockDelta {
                    index: 0,
                    delta: BlockDelta::Thinking {
                        text: "hmm".to_owned(),
                        token_ids: vec![1],
                    },
                },
                StreamEvent::BlockDelta {
                    index: 0,
                    delta: BlockDelta::ThinkingSignature("sig-abc".to_owned()),
                },
                StreamEvent::BlockStop { index: 0 },
                StreamEvent::BlockStart {
                    index: 1,
                    block: BlockStart::Text,
                },
                StreamEvent::BlockDelta {
                    index: 1,
                    delta: BlockDelta::Text {
                        text: "Paris".to_owned(),
                        token_ids: vec![2],
                    },
                },
                StreamEvent::BlockStop { index: 1 },
                stop(StopReason::EndTurn),
            ],
        );
        let bodies = data(&frames);
        assert_eq!(bodies[1]["content_block"]["type"], "thinking");
        assert_eq!(bodies[2]["delta"]["type"], "thinking_delta");
        assert_eq!(bodies[2]["delta"]["thinking"], "hmm");
        assert_eq!(bodies[3]["delta"]["type"], "signature_delta");
        assert_eq!(bodies[3]["delta"]["signature"], "sig-abc");
        assert_eq!(bodies[5]["index"], 1);
        assert_eq!(bodies[5]["content_block"]["type"], "text");
    }

    /// A call's arguments stream as the JSON text the model is writing, under
    /// the one delta type a client of this format assembles them from.
    #[test]
    fn a_tool_call_streams_its_identity_first_and_its_arguments_after() {
        let frames = drain(
            encoder(),
            vec![
                start(),
                StreamEvent::BlockStart {
                    index: 0,
                    block: BlockStart::ToolUse {
                        id: "call_1".to_owned(),
                        name: "lookup".to_owned(),
                    },
                },
                StreamEvent::BlockDelta {
                    index: 0,
                    delta: BlockDelta::ToolInput {
                        fragment: "{\"city\":".to_owned(),
                        token_ids: vec![9],
                    },
                },
                StreamEvent::BlockDelta {
                    index: 0,
                    delta: BlockDelta::ToolInput {
                        fragment: "\"Paris\"}".to_owned(),
                        token_ids: vec![10],
                    },
                },
                StreamEvent::BlockStop { index: 0 },
                stop(StopReason::ToolUse),
            ],
        );
        let bodies = data(&frames);
        assert_eq!(bodies[1]["content_block"]["type"], "tool_use");
        assert_eq!(bodies[1]["content_block"]["id"], "call_1");
        assert_eq!(bodies[1]["content_block"]["name"], "lookup");
        assert_eq!(bodies[2]["delta"]["type"], "input_json_delta");
        assert_eq!(bodies[2]["delta"]["partial_json"], "{\"city\":");
        let last = bodies.len() - 2;
        assert_eq!(bodies[last]["delta"]["stop_reason"], "tool_use");
    }

    /// The sequence that matched rides in the terminal frame, read from the
    /// same value that decides the reason.
    #[test]
    fn a_stop_sequence_is_echoed_in_the_frame_that_says_why_the_turn_ended() {
        let frames = drain(
            encoder(),
            vec![start(), stop(StopReason::StopSequence("END".to_owned()))],
        );
        let bodies = data(&frames);
        assert_eq!(bodies[1]["delta"]["stop_reason"], "stop_sequence");
        assert_eq!(bodies[1]["delta"]["stop_sequence"], "END");
    }

    /// A block left open by a producer that stopped announcing is closed
    /// before the turn ends, so the stream never leaves a block dangling.
    #[test]
    fn a_block_still_open_when_the_turn_ends_is_closed_first() {
        let frames = drain(
            encoder(),
            vec![
                start(),
                StreamEvent::BlockStart {
                    index: 0,
                    block: BlockStart::Text,
                },
                stop(StopReason::MaxTokens),
            ],
        );
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "content_block_stop",
                "message_delta",
                "message_stop",
            ]
        );
    }

    /// A failed generation must not end as a clean turn: the client is told
    /// what went wrong and then that the stream is over.
    #[test]
    fn a_failure_is_reported_as_an_error_frame_and_not_as_a_finished_turn() {
        let frames = drain(
            encoder(),
            vec![
                start(),
                StreamEvent::BlockStart {
                    index: 0,
                    block: BlockStart::Text,
                },
                StreamEvent::Failed {
                    message: "the device went away".to_owned(),
                },
            ],
        );
        assert_eq!(
            names(&frames),
            vec![
                "message_start",
                "content_block_start",
                "error",
                "message_stop",
            ]
        );
        let bodies = data(&frames);
        assert_eq!(bodies[2]["error"]["type"], "api_error");
        assert_eq!(bodies[2]["error"]["message"], "the device went away");
    }

    /// A stream that just stops has produced half a turn, and a client that
    /// was told `end_turn` would believe it had the whole answer.
    #[test]
    fn a_stream_that_ends_without_a_terminal_event_is_failed_rather_than_finished() {
        let frames = drain(encoder(), vec![start()]);
        assert_eq!(
            names(&frames),
            vec!["message_start", "error", "message_stop"]
        );
    }

    /// One terminator, whatever happened: a stream that fails after it has
    /// already ended must not send a second.
    #[test]
    fn a_stream_that_both_ends_and_fails_sends_one_terminator() {
        let mut encoder = encoder();
        let first = encoder.on_end();
        assert_eq!(
            names(&first),
            vec!["message_start", "error", "message_stop"]
        );
        assert!(
            encoder.on_end().is_empty(),
            "the terminator is sent once, however many times the end is reached"
        );
    }
}
