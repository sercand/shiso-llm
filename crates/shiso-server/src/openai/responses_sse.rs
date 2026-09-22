//! The Responses stream: named events, numbered in order.
//!
//! # The shape of the sequence
//!
//! `response.created`, `response.in_progress`, then - per block of the turn, in
//! block order - an `output_item.added`, the deltas that fill it in, and an
//! `output_item.done`; then `response.completed` carrying the whole finished
//! response. Each frame has an `event:` line naming its `type`, and a
//! `sequence_number` that increases by one every time. A client reconnecting or
//! reordering reads that number, so it is minted in one place and nowhere else.
//!
//! # Why the terminal frame is built by the accumulator
//!
//! The last frame carries the whole response, which has to be the same object a
//! blocking caller would have received. Rather than assembling it from the
//! frames this encoder just emitted - a second assembly that can drift from the
//! first - the events are also fed to [`shiso_ir::ResponseBuilder`], and the
//! terminal frame is that builder's answer encoded by the blocking encoder. The
//! two paths are then the same code, not two encoders that agree today.

use std::collections::BTreeMap;

use serde_json::{Value, json};
use shiso_ir::{BlockDelta, BlockStart, ResponseBuilder, StreamEvent};

use crate::api::sse::{Frame, FrameEncoder};
use crate::openai::responses_encode::{
    ResponseOptions, encode_envelope, encode_failure, encode_response, item_id,
};

/// What an open item is, which decides which events fill it in and close it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ItemKind {
    /// Prose, inside a message item.
    Text,
    /// A reasoning summary.
    Reasoning,
    /// A function call's arguments.
    FunctionCall,
    /// A refusal, inside a message item.
    Refusal,
}

/// One item the stream has announced and not yet closed.
#[derive(Debug)]
struct OpenItem {
    kind: ItemKind,
    id: String,
    output_index: usize,
    /// Everything that has arrived for it, so the `.done` frame can carry the
    /// whole value rather than making a client concatenate the deltas itself.
    text: String,
    /// The call's identifier and name, for the item a `.done` frame repeats.
    call: Option<(String, String)>,
}

/// Turns the neutral stream into this format's named events.
#[derive(Debug)]
pub struct ResponsesEventEncoder {
    options: ResponseOptions,
    sequence: u64,
    created: u64,
    open: BTreeMap<u32, OpenItem>,
    builder: Option<ResponseBuilder>,
    closed: bool,
}

impl ResponsesEventEncoder {
    /// An encoder for one request.
    #[must_use]
    pub fn new(options: ResponseOptions) -> Self {
        Self {
            options,
            sequence: 0,
            created: 0,
            open: BTreeMap::new(),
            builder: Some(ResponseBuilder::new()),
            closed: false,
        }
    }

    /// One frame, named and numbered.
    fn emit(&mut self, name: &str, mut data: Value) -> Frame {
        if let Some(fields) = data.as_object_mut() {
            fields.insert("type".to_owned(), json!(name));
            fields.insert("sequence_number".to_owned(), json!(self.sequence));
        }
        self.sequence += 1;
        Frame::named(name, data.to_string())
    }

    /// The envelope as it looks before any output exists.
    fn skeleton(&self, status: &str) -> Value {
        encode_envelope(
            self.created,
            &self.options,
            Vec::new(),
            status,
            Value::Null,
            None,
        )
    }

    /// Announces one block as an output item.
    fn open_item(&mut self, index: u32, block: BlockStart) -> Vec<Frame> {
        let output_index = usize::try_from(index).unwrap_or(usize::MAX);
        let (kind, prefix) = match block {
            BlockStart::Refusal => (ItemKind::Refusal, "msg"),
            BlockStart::Thinking => (ItemKind::Reasoning, "rs"),
            BlockStart::ToolUse { .. } => (ItemKind::FunctionCall, "fc"),
            // Prose, and any block kind added after this was written: announcing
            // an unknown one as a message keeps every later item's index right,
            // which matters more than a frame nobody can read.
            _ => (ItemKind::Text, "msg"),
        };
        let id = item_id(prefix, &self.options.id, output_index);
        let call = match block {
            BlockStart::ToolUse { id: call_id, name } => Some((call_id, name)),
            _ => None,
        };
        let item = Self::item_value(kind, &id, call.as_ref(), "in_progress", "");
        self.open.insert(
            index,
            OpenItem {
                kind,
                id: id.clone(),
                output_index,
                text: String::new(),
                call,
            },
        );

        let mut frames = vec![self.emit(
            "response.output_item.added",
            json!({"output_index": output_index, "item": item}),
        )];
        match kind {
            ItemKind::Text | ItemKind::Refusal => frames.push(self.emit(
                "response.content_part.added",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "content_index": 0,
                    "part": part_value(kind, ""),
                }),
            )),
            ItemKind::Reasoning => frames.push(self.emit(
                "response.reasoning_summary_part.added",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                }),
            )),
            // A call announces nothing further: its arguments are the item's own
            // field rather than a content part.
            ItemKind::FunctionCall => {}
        }
        frames
    }

    /// The item object a `.added` or `.done` frame repeats.
    fn item_value(
        kind: ItemKind,
        id: &str,
        call: Option<&(String, String)>,
        status: &str,
        text: &str,
    ) -> Value {
        match kind {
            ItemKind::Text | ItemKind::Refusal => json!({
                "type": "message",
                "id": id,
                "status": status,
                "role": "assistant",
                "content": if text.is_empty() && status == "in_progress" {
                    json!([])
                } else {
                    json!([part_value(kind, text)])
                },
            }),
            ItemKind::Reasoning => json!({
                "type": "reasoning",
                "id": id,
                "summary": if text.is_empty() && status == "in_progress" {
                    json!([])
                } else {
                    json!([{"type": "summary_text", "text": text}])
                },
                "content": [],
            }),
            ItemKind::FunctionCall => {
                let (call_id, name) = call.cloned().unwrap_or_default();
                json!({
                    "type": "function_call",
                    "id": id,
                    "call_id": call_id,
                    "name": name,
                    "arguments": text,
                    "status": status,
                })
            }
        }
    }

    /// The frames one fragment produces.
    fn extend(&mut self, index: u32, delta: &BlockDelta) -> Vec<Frame> {
        let Some((kind, id, output_index)) = self
            .open
            .get_mut(&index)
            .map(|item| (item.kind, item.id.clone(), item.output_index))
        else {
            return Vec::new();
        };
        let (name, fragment) = match (kind, delta) {
            (ItemKind::Text, BlockDelta::Text { text, .. }) => ("response.output_text.delta", text),
            (ItemKind::Reasoning, BlockDelta::Thinking { text, .. }) => {
                ("response.reasoning_summary_text.delta", text)
            }
            (ItemKind::FunctionCall, BlockDelta::ToolInput { fragment, .. }) => {
                ("response.function_call_arguments.delta", fragment)
            }
            (ItemKind::Refusal, BlockDelta::Refusal { text }) => ("response.refusal.delta", text),
            // A fragment that does not match the block it names is the
            // accumulator's problem to report; this encoder has no frame for it.
            _ => return Vec::new(),
        };
        if let Some(item) = self.open.get_mut(&index) {
            item.text.push_str(fragment);
        }
        let mut data = json!({
            "item_id": id,
            "output_index": output_index,
            "delta": fragment,
        });
        if matches!(kind, ItemKind::Text | ItemKind::Refusal)
            && let Some(fields) = data.as_object_mut()
        {
            fields.insert("content_index".to_owned(), json!(0));
        }
        if kind == ItemKind::Reasoning
            && let Some(fields) = data.as_object_mut()
        {
            fields.insert("summary_index".to_owned(), json!(0));
        }
        vec![self.emit(name, data)]
    }

    /// The frames that close one item.
    fn close_item(&mut self, index: u32) -> Vec<Frame> {
        let Some(item) = self.open.remove(&index) else {
            return Vec::new();
        };
        let OpenItem {
            kind,
            id,
            output_index,
            text,
            call,
        } = item;
        let mut frames = Vec::new();
        match kind {
            ItemKind::Text | ItemKind::Refusal => {
                let (name, key) = if kind == ItemKind::Text {
                    ("response.output_text.done", "text")
                } else {
                    ("response.refusal.done", "refusal")
                };
                frames.push(self.emit(
                    name,
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "content_index": 0,
                        key: text,
                    }),
                ));
                frames.push(self.emit(
                    "response.content_part.done",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "content_index": 0,
                        "part": part_value(kind, &text),
                    }),
                ));
            }
            ItemKind::Reasoning => {
                frames.push(self.emit(
                    "response.reasoning_summary_text.done",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "summary_index": 0,
                        "text": text,
                    }),
                ));
                frames.push(self.emit(
                    "response.reasoning_summary_part.done",
                    json!({
                        "item_id": id,
                        "output_index": output_index,
                        "summary_index": 0,
                        "part": {"type": "summary_text", "text": text},
                    }),
                ));
            }
            ItemKind::FunctionCall => frames.push(self.emit(
                "response.function_call_arguments.done",
                json!({
                    "item_id": id,
                    "output_index": output_index,
                    "arguments": text,
                }),
            )),
        }
        let done = Self::item_value(kind, &id, call.as_ref(), "completed", &text);
        frames.push(self.emit(
            "response.output_item.done",
            json!({"output_index": output_index, "item": done}),
        ));
        frames
    }

    /// The terminal frame, built by the accumulator so it matches the blocking
    /// body exactly.
    fn complete(&mut self) -> Vec<Frame> {
        self.closed = true;
        let Some(builder) = self.builder.take() else {
            return vec![self.failure_frame("the stream was already closed")];
        };
        match builder.finish() {
            Ok(response) => {
                let body = encode_response(&response, &self.options);
                vec![self.emit("response.completed", json!({"response": body}))]
            }
            Err(error) => {
                let message = error.to_string();
                vec![self.failure_frame(&message)]
            }
        }
    }

    /// The frame a failure is reported in.
    fn failure_frame(&mut self, message: &str) -> Frame {
        let body = encode_failure(self.created, &self.options, message);
        self.emit("response.failed", json!({"response": body}))
    }
}

/// The content part one item kind carries.
fn part_value(kind: ItemKind, text: &str) -> Value {
    if kind == ItemKind::Refusal {
        json!({"type": "refusal", "refusal": text})
    } else {
        json!({"type": "output_text", "text": text, "annotations": []})
    }
}

impl FrameEncoder for ResponsesEventEncoder {
    fn on_event(&mut self, event: StreamEvent) -> Vec<Frame> {
        // Fed to the accumulator as well as to the frames, so the terminal frame
        // is the turn itself rather than a reassembly of what was just sent.
        if let Some(builder) = self.builder.as_mut()
            && let Err(error) = builder.accept(event.clone())
        {
            let message = error.to_string();
            self.builder = None;
            self.closed = true;
            return vec![self.failure_frame(&message)];
        }

        match event {
            StreamEvent::Start { created, .. } => {
                self.created = created;
                let created_frame = self.emit(
                    "response.created",
                    json!({"response": self.skeleton("in_progress")}),
                );
                let in_progress = self.emit(
                    "response.in_progress",
                    json!({"response": self.skeleton("in_progress")}),
                );
                vec![created_frame, in_progress]
            }
            StreamEvent::BlockStart { index, block } => self.open_item(index, block),
            StreamEvent::BlockDelta { index, ref delta } => self.extend(index, delta),
            StreamEvent::BlockStop { index } => self.close_item(index),
            StreamEvent::Stop { .. } => {
                // Anything still open is closed first, in index order, so the
                // stream never announces an item it does not finish.
                let open: Vec<u32> = self.open.keys().copied().collect();
                let mut frames = Vec::new();
                for index in open {
                    frames.extend(self.close_item(index));
                }
                frames.extend(self.complete());
                frames
            }
            StreamEvent::Failed { message } => {
                self.closed = true;
                vec![self.failure_frame(&message)]
            }
            ref other => {
                tracing::debug!(?other, "no response event for this event");
                Vec::new()
            }
        }
    }

    fn on_end(&mut self) -> Vec<Frame> {
        if self.closed {
            return Vec::new();
        }
        tracing::warn!("a stream ended with no terminal event; failing it");
        vec![self.failure_frame("the engine's stream ended without completing the turn")]
    }
}

#[cfg(test)]
mod tests {
    use super::ResponsesEventEncoder;
    use crate::api::sse::{Frame, drain};
    use crate::openai::responses_encode::ResponseOptions;
    use serde_json::Value;
    use shiso_ir::{BlockDelta, BlockStart, StopReason, StreamEvent, Usage};

    fn options() -> ResponseOptions {
        ResponseOptions {
            id: "resp_abc".to_owned(),
            model: "a-model".to_owned(),
            ..ResponseOptions::default()
        }
    }

    fn start() -> StreamEvent {
        StreamEvent::Start {
            id: "bare".to_owned(),
            model: "a-model".to_owned(),
            created: 1700,
        }
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
                stop: StopReason::EndTurn,
                usage: Usage {
                    prompt_tokens: 4,
                    completion_tokens: 2,
                    ..Usage::default()
                },
                token_ids: Vec::new(),
            },
        ]
    }

    fn names(frames: &[Frame]) -> Vec<String> {
        frames
            .iter()
            .filter_map(|frame| frame.name.clone())
            .collect()
    }

    fn bodies(frames: &[Frame]) -> Vec<Value> {
        frames
            .iter()
            .map(|frame| serde_json::from_str(&frame.data).expect("every frame is JSON"))
            .collect()
    }

    #[test]
    fn a_text_turn_is_the_named_sequence_this_format_defines() {
        let frames = drain(ResponsesEventEncoder::new(options()), text_events());
        assert_eq!(
            names(&frames),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.content_part.added",
                "response.output_text.delta",
                "response.output_text.delta",
                "response.output_text.done",
                "response.content_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
    }

    /// The number is what a client reorders on, so it is minted in one place and
    /// increases by one every time.
    #[test]
    fn the_sequence_number_increases_by_one_on_every_frame() {
        let frames = drain(ResponsesEventEncoder::new(options()), text_events());
        let numbers: Vec<u64> = bodies(&frames)
            .iter()
            .filter_map(|body| body["sequence_number"].as_u64())
            .collect();
        assert_eq!(numbers, (0..10).collect::<Vec<u64>>());
    }

    /// The `event:` line and the `type` field have to agree, or a client that
    /// dispatches on one and validates the other rejects every frame.
    #[test]
    fn the_event_line_and_the_type_field_are_the_same_word() {
        let frames = drain(ResponsesEventEncoder::new(options()), text_events());
        for frame in &frames {
            let body: Value = serde_json::from_str(&frame.data).expect("JSON");
            assert_eq!(body["type"].as_str(), frame.name.as_deref());
        }
    }

    /// The terminal frame carries the whole turn, and it is the accumulator's
    /// answer rather than a reassembly of the frames just sent.
    #[test]
    fn the_terminal_frame_carries_the_turn_a_blocking_caller_would_have_read() {
        let frames = drain(ResponsesEventEncoder::new(options()), text_events());
        let last = bodies(&frames).pop().expect("a terminal frame");
        assert_eq!(last["type"], "response.completed");
        let response = &last["response"];
        assert_eq!(response["status"], "completed");
        assert_eq!(response["id"], "resp_abc");
        assert_eq!(response["output"][0]["content"][0]["text"], "Paris");
        assert_eq!(response["usage"]["input_tokens"], 4);
        assert_eq!(response["usage"]["output_tokens"], 2);
    }

    /// The item announced on the stream and the item in the finished response
    /// must be the same object under the same name.
    #[test]
    fn a_streamed_item_and_the_finished_ones_carry_the_same_identifier() {
        let frames = drain(ResponsesEventEncoder::new(options()), text_events());
        let bodies = bodies(&frames);
        let announced = bodies[2]["item"]["id"].clone();
        let finished =
            bodies.last().expect("a terminal frame")["response"]["output"][0]["id"].clone();
        assert_eq!(announced, finished);
        assert_eq!(announced, "msg_abc_0");
    }

    #[test]
    fn a_tool_call_is_announced_then_filled_in_then_closed() {
        let events = vec![
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
                    token_ids: vec![1],
                },
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::ToolInput {
                    fragment: "\"Paris\"}".to_owned(),
                    token_ids: vec![2],
                },
            },
            StreamEvent::BlockStop { index: 0 },
            StreamEvent::Stop {
                stop: StopReason::ToolUse,
                usage: Usage::default(),
                token_ids: Vec::new(),
            },
        ];
        let frames = drain(ResponsesEventEncoder::new(options()), events);
        assert_eq!(
            names(&frames),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.delta",
                "response.function_call_arguments.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
        let bodies = bodies(&frames);
        assert_eq!(bodies[2]["item"]["call_id"], "call_1");
        assert_eq!(bodies[2]["item"]["name"], "lookup");
        assert_eq!(bodies[5]["arguments"], "{\"city\":\"Paris\"}");
    }

    #[test]
    fn a_trace_is_announced_as_a_reasoning_item_with_its_summary_quartet() {
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
            StreamEvent::BlockStop { index: 0 },
            StreamEvent::Stop {
                stop: StopReason::EndTurn,
                usage: Usage::default(),
                token_ids: Vec::new(),
            },
        ];
        let frames = drain(ResponsesEventEncoder::new(options()), events);
        assert_eq!(
            names(&frames),
            vec![
                "response.created",
                "response.in_progress",
                "response.output_item.added",
                "response.reasoning_summary_part.added",
                "response.reasoning_summary_text.delta",
                "response.reasoning_summary_text.done",
                "response.reasoning_summary_part.done",
                "response.output_item.done",
                "response.completed",
            ]
        );
    }

    /// A stream that ends while an item is open would leave a client waiting for
    /// a `.done` that never comes.
    #[test]
    fn an_item_still_open_when_the_turn_ends_is_closed_before_the_terminal_frame() {
        let mut events = text_events();
        events.retain(|event| !matches!(event, StreamEvent::BlockStop { .. }));
        let frames = drain(ResponsesEventEncoder::new(options()), events);
        let names = names(&frames);
        assert!(names.contains(&"response.output_item.done".to_owned()));
        assert_eq!(names.last().map(String::as_str), Some("response.completed"));
    }

    #[test]
    fn a_failure_mid_turn_is_a_failed_frame_carrying_the_reason() {
        let frames = drain(
            ResponsesEventEncoder::new(options()),
            vec![
                start(),
                StreamEvent::Failed {
                    message: "the device went away".to_owned(),
                },
            ],
        );
        let last = bodies(&frames).pop().expect("a terminal frame");
        assert_eq!(last["type"], "response.failed");
        assert_eq!(last["response"]["status"], "failed");
        assert!(
            last["response"]["error"]["message"]
                .as_str()
                .is_some_and(|message| message.contains("device")),
            "{last}"
        );
    }

    #[test]
    fn a_stream_that_just_stops_fails_rather_than_completing() {
        let frames = drain(ResponsesEventEncoder::new(options()), vec![start()]);
        let last = bodies(&frames).pop().expect("a terminal frame");
        assert_eq!(last["type"], "response.failed");
    }

    /// A truncated turn is incomplete, and the terminal frame is where a client
    /// learns which limit stopped it.
    #[test]
    fn a_turn_that_ran_out_of_budget_completes_as_incomplete() {
        let mut events = text_events();
        if let Some(StreamEvent::Stop { stop, .. }) = events.last_mut() {
            *stop = StopReason::MaxTokens;
        }
        let frames = drain(ResponsesEventEncoder::new(options()), events);
        let last = bodies(&frames).pop().expect("a terminal frame");
        assert_eq!(last["response"]["status"], "incomplete");
        assert_eq!(
            last["response"]["incomplete_details"]["reason"],
            "max_output_tokens"
        );
    }
}
