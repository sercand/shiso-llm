//! The chat-completions stream: one `data:` frame per thing that happened.
//!
//! # What the framing promises
//!
//! A role prologue, then one frame per fragment, then a frame carrying the
//! finish reason, then - only if the caller asked - a frame carrying nothing but
//! the accounting, then `[DONE]`. Concatenating the `content` deltas of that
//! stream gives the `content` of the blocking answer for the same turn, and the
//! finish reason is the same word. Those two equalities are the whole contract,
//! and they hold because both encodings read the same events.
//!
//! # Why the tool index is not the block index
//!
//! The neutral stream numbers blocks by their position in the finished message,
//! so a turn that says something and then calls a tool opens block one for the
//! call. This format numbers tool calls among themselves, from zero, and a
//! client assembling arguments keys on that number. Translating between the two
//! is this encoder's job and is the reason it is a state machine rather than a
//! function.
//!
//! # Why a failure is a JSON frame
//!
//! A stream that has already sent a 200 cannot change its status, so a failure
//! has to arrive as content. It arrives as the same error object every other
//! failure on this surface uses, followed by the terminator, because a client
//! that is parsing every frame as JSON should not have to special-case the one
//! frame that says what went wrong.

use std::collections::BTreeMap;

use serde_json::{Map, Value, json};
use shiso_ir::{BlockDelta, BlockStart, StopReason, StreamEvent, Usage};

use crate::api::sse::{Frame, FrameEncoder};
use crate::openai::SYSTEM_FINGERPRINT;
use crate::openai::chat_encode::{EncodeOptions, encode_usage, finish_reason};

/// Turns the neutral stream into chat-completion chunks.
#[derive(Debug)]
pub struct ChatChunkEncoder {
    options: EncodeOptions,
    include_usage: bool,
    created: u64,
    /// Which tool-call ordinal each open block was given.
    ordinals: BTreeMap<u32, usize>,
    next_ordinal: usize,
    /// Whether the terminator has already been queued, so that a stream which
    /// both fails and ends does not send two.
    closed: bool,
}

impl ChatChunkEncoder {
    /// An encoder for one request.
    #[must_use]
    pub fn new(options: EncodeOptions, include_usage: bool) -> Self {
        Self {
            options,
            include_usage,
            created: 0,
            ordinals: BTreeMap::new(),
            next_ordinal: 0,
            closed: false,
        }
    }

    /// One chunk with one choice in it.
    fn chunk(&self, delta: &Value, finish: Option<&str>) -> Frame {
        let choice = json!({
            "index": 0,
            "delta": delta,
            "finish_reason": finish,
            "logprobs": Value::Null,
        });
        self.frame(Value::Array(vec![choice]), None)
    }

    /// One chunk with whatever choices and accounting it was given.
    fn frame(&self, choices: Value, usage: Option<Value>) -> Frame {
        let mut body = Map::new();
        body.insert("id".to_owned(), json!(self.options.id));
        body.insert("object".to_owned(), json!("chat.completion.chunk"));
        body.insert("created".to_owned(), json!(self.created));
        body.insert("model".to_owned(), json!(self.options.model));
        body.insert("system_fingerprint".to_owned(), json!(SYSTEM_FINGERPRINT));
        body.insert("choices".to_owned(), choices);
        if let Some(usage) = usage {
            body.insert("usage".to_owned(), usage);
        }
        if let Some(tier) = self.options.service_tier {
            body.insert("service_tier".to_owned(), json!(tier.as_str()));
        }
        Frame::data(Value::Object(body).to_string())
    }

    /// The ordinal this format gives the tool call at `index`.
    fn ordinal(&mut self, index: u32) -> usize {
        *self.ordinals.entry(index).or_insert_with(|| {
            let ordinal = self.next_ordinal;
            self.next_ordinal += 1;
            ordinal
        })
    }

    /// The delta for a fragment of text, with the ids when they were asked for.
    fn text_delta(&self, key: &str, text: &str, token_ids: &[u32]) -> Value {
        let mut delta = Map::new();
        delta.insert(key.to_owned(), json!(text));
        if self.options.return_token_ids {
            delta.insert("token_ids".to_owned(), json!(token_ids));
        }
        Value::Object(delta)
    }

    /// The frames that end the stream: the finish reason, the accounting when it
    /// was asked for, and the terminator.
    ///
    /// The finish frame comes before the accounting frame. Every client of this
    /// format documents the order that way - the turn ends, and then a frame
    /// with an empty `choices` list carries the totals - and a client that stops
    /// reading at the finish reason still gets a well-formed stream.
    fn close(&mut self, stop: &StopReason, usage: &Usage, token_ids: &[u32]) -> Vec<Frame> {
        self.closed = true;
        let mut delta = Map::new();
        if self.options.return_token_ids && !token_ids.is_empty() {
            delta.insert("token_ids".to_owned(), json!(token_ids));
        }
        let mut frames = vec![self.chunk(&Value::Object(delta), Some(finish_reason(stop)))];
        if self.include_usage {
            frames.push(self.frame(Value::Array(Vec::new()), Some(encode_usage(usage))));
        }
        frames.push(Frame::done());
        frames
    }
}

impl FrameEncoder for ChatChunkEncoder {
    fn on_event(&mut self, event: StreamEvent) -> Vec<Frame> {
        match event {
            StreamEvent::Start { created, .. } => {
                self.created = created;
                // The prologue, which every client of this format expects before
                // any content: it is what tells a client which speaker the
                // fragments belong to.
                vec![self.chunk(&json!({"role": "assistant", "content": ""}), None)]
            }
            StreamEvent::BlockStart {
                index,
                block: BlockStart::ToolUse { id, name },
            } => {
                let ordinal = self.ordinal(index);
                vec![self.chunk(
                    &json!({"tool_calls": [{
                        "index": ordinal,
                        "id": id,
                        "type": "function",
                        "function": {"name": name, "arguments": ""},
                    }]}),
                    None,
                )]
            }
            // Text, thinking and refusal blocks announce nothing: this format
            // has no frame for "a block opened", and inventing one would put a
            // frame on the wire that no client reads.
            StreamEvent::BlockStart { .. } | StreamEvent::BlockStop { .. } => Vec::new(),
            StreamEvent::BlockDelta { index, delta } => self.delta(index, delta),
            StreamEvent::Stop {
                stop,
                usage,
                token_ids,
            } => self.close(&stop, &usage, &token_ids),
            StreamEvent::Failed { message } => {
                self.closed = true;
                let body = json!({"error": {
                    "message": message,
                    "type": "server_error",
                    "param": Value::Null,
                    "code": "internal_error",
                }});
                vec![Frame::data(body.to_string()), Frame::done()]
            }
            // The stream is open on purpose: an event added after this was
            // written has no chunk here, and a client is better served by the
            // frames it does understand than by a stream that stops.
            ref other => {
                tracing::debug!(?other, "no chat chunk for this event");
                Vec::new()
            }
        }
    }

    fn on_end(&mut self) -> Vec<Frame> {
        if self.closed {
            return Vec::new();
        }
        // The producer stopped without a terminal event. A client waiting for
        // the terminator would wait forever, so it is sent with no finish
        // reason, which is the only honest thing to say about a turn that never
        // ended.
        tracing::warn!("a stream ended with no terminal event; closing it anyway");
        vec![self.chunk(&json!({}), None), Frame::done()]
    }
}

impl ChatChunkEncoder {
    /// The frames one fragment of an open block produces.
    fn delta(&mut self, index: u32, delta: BlockDelta) -> Vec<Frame> {
        match delta {
            BlockDelta::Text { text, token_ids } => {
                vec![self.chunk(&self.text_delta("content", &text, &token_ids), None)]
            }
            // The trace travels in its own key and is never mirrored into
            // `content`: a client that showed both would show the reasoning to
            // the user twice.
            BlockDelta::Thinking { text, token_ids } => {
                vec![self.chunk(
                    &self.text_delta("reasoning_content", &text, &token_ids),
                    None,
                )]
            }
            BlockDelta::Refusal { text } => {
                vec![self.chunk(&json!({"refusal": text}), None)]
            }
            BlockDelta::ToolInput { fragment, .. } => {
                let ordinal = self.ordinal(index);
                vec![self.chunk(
                    &json!({"tool_calls": [{
                        "index": ordinal,
                        "function": {"arguments": fragment},
                    }]}),
                    None,
                )]
            }
            // A signature is a token for replaying a trace on the surface that
            // issued it. This format has no field for one, and a client that
            // received it would have nowhere to put it.
            BlockDelta::ThinkingSignature(_) => Vec::new(),
            ref other => {
                tracing::debug!(?other, "no chat chunk for this fragment");
                Vec::new()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ChatChunkEncoder;
    use crate::api::sse::{Frame, drain};
    use crate::openai::chat_encode::EncodeOptions;
    use serde_json::Value;
    use shiso_ir::{BlockDelta, BlockStart, StopReason, StreamEvent, Usage};

    fn options() -> EncodeOptions {
        EncodeOptions {
            id: "chatcmpl-1".to_owned(),
            model: "a-model".to_owned(),
            service_tier: None,
            return_token_ids: false,
        }
    }

    fn start() -> StreamEvent {
        StreamEvent::Start {
            id: "bare".to_owned(),
            model: "a-model".to_owned(),
            created: 1700,
        }
    }

    fn bodies(frames: &[Frame]) -> Vec<Value> {
        frames
            .iter()
            .filter(|frame| frame.data != "[DONE]")
            .map(|frame| serde_json::from_str(&frame.data).expect("every frame is JSON"))
            .collect()
    }

    fn text_events() -> Vec<StreamEvent> {
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
            StreamEvent::Stop {
                stop: StopReason::MaxTokens,
                usage: Usage {
                    prompt_tokens: 4,
                    completion_tokens: 2,
                    ..Usage::default()
                },
                token_ids: Vec::new(),
            },
        ]
    }

    #[test]
    fn a_text_turn_is_a_role_prologue_then_fragments_then_a_finish_then_done() {
        let frames = drain(ChatChunkEncoder::new(options(), false), text_events());
        assert_eq!(frames.last().expect("a terminator"), &Frame::done());
        let bodies = bodies(&frames);
        assert_eq!(bodies.len(), 4, "prologue, two fragments, finish");
        assert_eq!(bodies[0]["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(bodies[1]["choices"][0]["delta"]["content"], "Par");
        assert_eq!(bodies[2]["choices"][0]["delta"]["content"], "is");
        assert_eq!(bodies[3]["choices"][0]["finish_reason"], "length");
        for body in &bodies {
            assert_eq!(body["object"], "chat.completion.chunk");
            assert_eq!(body["id"], "chatcmpl-1");
            assert_eq!(body["created"], 1700);
            assert_eq!(body["system_fingerprint"], "fp_shiso");
        }
    }

    /// Concatenating the fragments has to give the blocking answer's content,
    /// or the two encodings are describing different turns.
    #[test]
    fn the_fragments_concatenate_to_what_a_blocking_caller_would_have_read() {
        let frames = drain(ChatChunkEncoder::new(options(), false), text_events());
        let joined: String = bodies(&frames)
            .iter()
            .filter_map(|body| body["choices"][0]["delta"]["content"].as_str())
            .collect();
        assert_eq!(joined, "Paris");
    }

    #[test]
    fn asking_for_the_accounting_adds_one_frame_with_no_choices_after_the_finish() {
        let frames = drain(ChatChunkEncoder::new(options(), true), text_events());
        let bodies = bodies(&frames);
        let finish = &bodies[bodies.len() - 2];
        let totals = bodies.last().expect("the accounting frame");
        assert_eq!(finish["choices"][0]["finish_reason"], "length");
        assert!(
            finish.get("usage").is_none(),
            "the finish frame carries none"
        );
        assert_eq!(totals["choices"].as_array().expect("an array").len(), 0);
        assert_eq!(totals["usage"]["prompt_tokens"], 4);
        assert_eq!(totals["usage"]["total_tokens"], 6);
        assert_eq!(frames.last().expect("a terminator"), &Frame::done());
    }

    #[test]
    fn a_silent_caller_never_sees_the_accounting_frame() {
        let frames = drain(ChatChunkEncoder::new(options(), false), text_events());
        assert!(
            bodies(&frames)
                .iter()
                .all(|body| body.get("usage").is_none()),
            "the accounting is opt in"
        );
    }

    /// The neutral stream numbers blocks; this format numbers tool calls. A call
    /// that opens as block one is still tool call zero.
    #[test]
    fn a_tool_call_is_numbered_among_the_calls_and_not_among_the_blocks() {
        let events = vec![
            start(),
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::Text,
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: "looking".to_owned(),
                    token_ids: vec![1],
                },
            },
            StreamEvent::BlockStart {
                index: 1,
                block: BlockStart::ToolUse {
                    id: "call_1".to_owned(),
                    name: "lookup".to_owned(),
                },
            },
            StreamEvent::BlockDelta {
                index: 1,
                delta: BlockDelta::ToolInput {
                    fragment: "{\"city\":".to_owned(),
                    token_ids: vec![2],
                },
            },
            StreamEvent::BlockDelta {
                index: 1,
                delta: BlockDelta::ToolInput {
                    fragment: "\"Paris\"}".to_owned(),
                    token_ids: vec![3],
                },
            },
            StreamEvent::BlockStop { index: 1 },
            StreamEvent::Stop {
                stop: StopReason::ToolUse,
                usage: Usage::default(),
                token_ids: Vec::new(),
            },
        ];
        let bodies = bodies(&drain(ChatChunkEncoder::new(options(), false), events));
        let calls: Vec<&Value> = bodies
            .iter()
            .filter(|body| !body["choices"][0]["delta"]["tool_calls"].is_null())
            .collect();
        assert_eq!(calls.len(), 3, "an opening and two argument fragments");
        assert_eq!(calls[0]["choices"][0]["delta"]["tool_calls"][0]["index"], 0);
        assert_eq!(
            calls[0]["choices"][0]["delta"]["tool_calls"][0]["id"],
            "call_1"
        );
        assert_eq!(
            calls[0]["choices"][0]["delta"]["tool_calls"][0]["type"],
            "function"
        );
        assert_eq!(
            calls[0]["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"],
            ""
        );
        let arguments: String = calls[1..]
            .iter()
            .filter_map(|body| {
                body["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str()
            })
            .collect();
        assert_eq!(arguments, "{\"city\":\"Paris\"}");
        assert_eq!(
            bodies.last().expect("a finish frame")["choices"][0]["finish_reason"],
            "tool_calls"
        );
    }

    /// A trace never mirrors into `content`, or a client that renders both shows
    /// the reasoning to the user twice.
    #[test]
    fn a_trace_fragment_carries_only_the_reasoning_key() {
        let events = vec![
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
            StreamEvent::Stop {
                stop: StopReason::EndTurn,
                usage: Usage::default(),
                token_ids: Vec::new(),
            },
        ];
        let bodies = bodies(&drain(ChatChunkEncoder::new(options(), false), events));
        assert_eq!(bodies[1]["choices"][0]["delta"]["reasoning_content"], "hmm");
        assert!(
            bodies[1]["choices"][0]["delta"]["content"].is_null(),
            "no mirror into content"
        );
    }

    /// A 200 has already been sent, so the failure has to arrive as content -
    /// and as the same error object every other failure uses.
    #[test]
    fn a_failure_mid_stream_arrives_as_the_error_object_and_then_the_terminator() {
        let frames = drain(
            ChatChunkEncoder::new(options(), false),
            vec![
                start(),
                StreamEvent::Failed {
                    message: "the device went away".to_owned(),
                },
            ],
        );
        assert_eq!(frames.last().expect("a terminator"), &Frame::done());
        let bodies = bodies(&frames);
        let failure = bodies.last().expect("a failure frame");
        assert_eq!(failure["error"]["type"], "server_error");
        assert!(
            failure["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("device")),
            "{failure}"
        );
    }

    /// A producer that stops without saying so would leave a client waiting on a
    /// terminator that never comes.
    #[test]
    fn a_stream_that_just_stops_is_still_terminated() {
        let frames = drain(ChatChunkEncoder::new(options(), false), vec![start()]);
        assert_eq!(frames.last().expect("a terminator"), &Frame::done());
    }

    #[test]
    fn the_service_class_is_echoed_on_every_frame_when_the_caller_named_one() {
        let mut options = options();
        options.service_tier = Some(crate::openai::chat_wire::ServiceTier::Flex);
        let frames = drain(ChatChunkEncoder::new(options, true), text_events());
        assert!(
            bodies(&frames)
                .iter()
                .all(|body| body["service_tier"] == "flex"),
            "every frame, including the accounting one"
        );
    }
}
