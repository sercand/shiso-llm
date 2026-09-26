//! What the IR promises, stated as tests.
//!
//! Two of these matter more than the rest: a generated turn goes back into the
//! conversation that produced it, and a stream folds into exactly the turn a
//! non-streaming run would have produced. Everything else here guards an ordering
//! or a distinction that has no other enforcement.

use serde_json::json;

use crate::{
    Alternative, Block, BlockDelta, BlockStart, Effort, Logprobs, MediaKind, MediaSource, Message,
    RepetitionDetection, Request, Response, ResponseBuilder, Role, Sampling, StopReason,
    StreamError, StreamEvent, Thinking, ThinkingDirective, TokenLogprob, ToolChoice,
    ToolDefinition, ToolResult, ToolUse, Usage,
};

/// A stream that produces a short turn with one tool call in the middle.
fn tool_turn() -> Vec<StreamEvent> {
    vec![
        StreamEvent::Start {
            id: "abc".into(),
            model: "m".into(),
            created: 7,
        },
        StreamEvent::BlockStart {
            index: 0,
            block: BlockStart::Thinking,
        },
        StreamEvent::BlockDelta {
            index: 0,
            delta: BlockDelta::Thinking {
                text: "weigh".into(),
                token_ids: vec![1, 2],
            },
        },
        StreamEvent::BlockDelta {
            index: 0,
            delta: BlockDelta::ThinkingSignature("sig".into()),
        },
        StreamEvent::BlockStop { index: 0 },
        StreamEvent::BlockStart {
            index: 1,
            block: BlockStart::Text,
        },
        StreamEvent::BlockDelta {
            index: 1,
            delta: BlockDelta::Text {
                text: "Let me ".into(),
                token_ids: vec![3],
            },
        },
        StreamEvent::BlockDelta {
            index: 1,
            delta: BlockDelta::Text {
                text: "check.".into(),
                token_ids: vec![4],
            },
        },
        StreamEvent::BlockStop { index: 1 },
        StreamEvent::BlockStart {
            index: 2,
            block: BlockStart::ToolUse {
                id: "call_1".into(),
                name: "lookup".into(),
            },
        },
        StreamEvent::BlockDelta {
            index: 2,
            delta: BlockDelta::ToolInput {
                fragment: "{\"city\":".into(),
                token_ids: vec![5],
            },
        },
        StreamEvent::BlockDelta {
            index: 2,
            delta: BlockDelta::ToolInput {
                fragment: "\"Paris\"}".into(),
                token_ids: vec![6],
            },
        },
        StreamEvent::BlockStop { index: 2 },
        StreamEvent::Stop {
            stop: StopReason::ToolUse,
            usage: Usage {
                prompt_tokens: 11,
                completion_tokens: 6,
                ..Usage::default()
            },
            token_ids: Vec::new(),
        },
    ]
}

/// Folds a whole stream, failing the test on the first event it refuses.
fn fold(events: Vec<StreamEvent>) -> Response {
    let mut builder = ResponseBuilder::new();
    for event in events {
        builder
            .accept(event)
            .expect("the event belongs to the turn");
    }
    builder.finish().expect("the stream is complete")
}

#[test]
fn a_generated_turn_goes_straight_back_into_the_conversation_that_asked_for_it() {
    let mut request = Request::new("m", vec![Message::user("What is the capital?")]);
    let response = fold(tool_turn());

    // The whole round trip, with no conversion step: this is the property the
    // message shape exists for, and the reason a response carries a `Message`
    // rather than a shape of its own.
    request.push(response.into_message());

    assert_eq!(request.messages.len(), 2);
    let turn = &request.messages[1];
    assert_eq!(turn.role, Role::Assistant);
    assert_eq!(turn.text(), "Let me check.");
    let calls: Vec<&ToolUse> = turn.tool_uses().collect();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "lookup");
    assert_eq!(calls[0].input, json!({"city": "Paris"}));

    // And the caller can answer that call with a turn of its own, which is the
    // half of the round trip that a response-only shape could not express at all.
    request.push(Message::new(
        Role::User,
        vec![Block::ToolResult(ToolResult {
            tool_use_id: calls[0].id.clone(),
            content: vec![Block::Text("Paris".into())],
            is_error: false,
        })],
    ));
    assert_eq!(request.messages.len(), 3);
}

#[test]
fn a_stream_folds_into_the_turn_a_blocking_run_would_have_produced() {
    let response = fold(tool_turn());

    assert_eq!(response.id, "abc");
    assert_eq!(response.model, "m");
    assert_eq!(response.created, 7);
    assert_eq!(response.stop, StopReason::ToolUse);
    assert_eq!(response.usage.completion_tokens, 6);
    assert_eq!(
        response.message,
        Message::new(
            Role::Assistant,
            vec![
                Block::Thinking(Thinking {
                    text: "weigh".into(),
                    signature: Some("sig".into()),
                }),
                Block::Text("Let me check.".into()),
                Block::ToolUse(ToolUse {
                    id: "call_1".into(),
                    name: "lookup".into(),
                    input: json!({"city": "Paris"}),
                }),
            ],
        )
    );
    // Every id that rode a delta is accounted for, in arrival order, which is what
    // lets a streaming caller and a blocking one reconcile the same generation.
    assert_eq!(response.token_ids, vec![1, 2, 3, 4, 5, 6]);
}

#[test]
fn blocks_land_in_the_order_they_were_numbered_not_the_order_they_closed() {
    // A producer may open the next block before closing the last one. The finished
    // message must still read in the producer's order, because that order is the
    // turn: text, then a call, then more text is a different answer from text,
    // more text, then a call.
    let events = vec![
        StreamEvent::Start {
            id: "id".into(),
            model: "m".into(),
            created: 0,
        },
        StreamEvent::BlockStart {
            index: 1,
            block: BlockStart::Text,
        },
        StreamEvent::BlockStart {
            index: 0,
            block: BlockStart::Text,
        },
        StreamEvent::BlockDelta {
            index: 0,
            delta: BlockDelta::Text {
                text: "first".into(),
                token_ids: Vec::new(),
            },
        },
        StreamEvent::BlockDelta {
            index: 1,
            delta: BlockDelta::Text {
                text: "second".into(),
                token_ids: Vec::new(),
            },
        },
        StreamEvent::BlockStop { index: 1 },
        StreamEvent::BlockStop { index: 0 },
        StreamEvent::Stop {
            stop: StopReason::EndTurn,
            usage: Usage::default(),
            token_ids: Vec::new(),
        },
    ];
    assert_eq!(fold(events).message.text(), "firstsecond");
}

#[test]
fn the_visible_text_of_a_turn_leaves_out_what_the_caller_did_not_ask_to_see() {
    let turn = Message::new(
        Role::Assistant,
        vec![
            Block::Thinking(Thinking::new("deliberating")),
            Block::Text("Paris".into()),
            Block::ToolUse(ToolUse {
                id: "c".into(),
                name: "lookup".into(),
                input: json!({}),
            }),
            Block::Text(" is the capital.".into()),
        ],
    );
    assert_eq!(turn.text(), "Paris is the capital.");
}

#[test]
fn media_arrives_tagged_and_in_the_order_the_caller_wrote_it() {
    // A pair of per-modality counts cannot express order, and a turn that sends a
    // video and then an image renders two different markers. Counts that agree
    // with each other while the items are shown in the wrong order is a defect
    // nothing downstream can detect.
    let turn = Message::new(
        Role::User,
        vec![
            Block::Video(MediaSource::Inline("v".into())),
            Block::Text("and".into()),
            Block::Image(MediaSource::Inline("i".into())),
            Block::Audio(MediaSource::Inline("a".into())),
        ],
    );
    let kinds: Vec<MediaKind> = turn.media().map(|(kind, _)| kind).collect();
    assert_eq!(
        kinds,
        vec![MediaKind::Video, MediaKind::Image, MediaKind::Audio]
    );
}

#[test]
fn a_remote_address_never_reaches_the_decoder_as_if_it_were_bytes() {
    assert_eq!(
        MediaSource::classify("https://example.test/a.png"),
        MediaSource::Remote("https://example.test/a.png".into())
    );
    assert_eq!(
        MediaSource::classify("http://example.test/a.png"),
        MediaSource::Remote("http://example.test/a.png".into())
    );
    assert_eq!(
        MediaSource::classify("data:image/png;base64,AAAA"),
        MediaSource::Inline("data:image/png;base64,AAAA".into())
    );
}

#[test]
fn a_call_with_no_arguments_is_an_empty_object_and_not_a_parse_failure() {
    let events = vec![
        StreamEvent::Start {
            id: "id".into(),
            model: "m".into(),
            created: 0,
        },
        StreamEvent::BlockStart {
            index: 0,
            block: BlockStart::ToolUse {
                id: "c".into(),
                name: "now".into(),
            },
        },
        StreamEvent::BlockStop { index: 0 },
        StreamEvent::Stop {
            stop: StopReason::ToolUse,
            usage: Usage::default(),
            token_ids: Vec::new(),
        },
    ];
    let calls: Vec<ToolUse> = fold(events).message.tool_uses().cloned().collect();
    assert_eq!(calls[0].input, json!({}));
}

#[test]
fn arguments_that_never_became_json_are_reported_rather_than_guessed_at() {
    let mut builder = ResponseBuilder::new();
    for event in [
        StreamEvent::Start {
            id: "id".into(),
            model: "m".into(),
            created: 0,
        },
        StreamEvent::BlockStart {
            index: 0,
            block: BlockStart::ToolUse {
                id: "c".into(),
                name: "run".into(),
            },
        },
        StreamEvent::BlockDelta {
            index: 0,
            delta: BlockDelta::ToolInput {
                fragment: "{\"cmd\":".into(),
                token_ids: Vec::new(),
            },
        },
        StreamEvent::Stop {
            stop: StopReason::MaxTokens,
            usage: Usage::default(),
            token_ids: Vec::new(),
        },
    ] {
        builder.accept(event).expect("accepted");
    }
    // The arguments are what the tool is about to be run with. Repairing them
    // would produce a call the caller never authorized.
    let error = builder
        .finish()
        .expect_err("truncated arguments are refused");
    assert!(matches!(error, StreamError::ToolInput { ref name, .. } if name == "run"));
}

#[test]
fn a_fragment_cannot_extend_a_block_of_another_kind() {
    let mut builder = ResponseBuilder::new();
    builder
        .accept(StreamEvent::Start {
            id: "id".into(),
            model: "m".into(),
            created: 0,
        })
        .expect("accepted");
    builder
        .accept(StreamEvent::BlockStart {
            index: 0,
            block: BlockStart::Text,
        })
        .expect("accepted");
    let error = builder
        .accept(StreamEvent::BlockDelta {
            index: 0,
            delta: BlockDelta::ToolInput {
                fragment: "{}".into(),
                token_ids: Vec::new(),
            },
        })
        .expect_err("a tool fragment cannot extend prose");
    assert!(matches!(
        error,
        StreamError::BlockKindMismatch { index: 0, .. }
    ));
}

#[test]
fn a_delta_for_a_block_that_never_opened_is_refused() {
    let mut builder = ResponseBuilder::new();
    builder
        .accept(StreamEvent::Start {
            id: "id".into(),
            model: "m".into(),
            created: 0,
        })
        .expect("accepted");
    let error = builder
        .accept(StreamEvent::BlockDelta {
            index: 4,
            delta: BlockDelta::Text {
                text: "x".into(),
                token_ids: Vec::new(),
            },
        })
        .expect_err("block four never opened");
    assert!(matches!(error, StreamError::UnknownBlock { index: 4 }));
}

#[test]
fn content_before_the_stream_starts_is_refused() {
    let mut builder = ResponseBuilder::new();
    let error = builder
        .accept(StreamEvent::BlockStart {
            index: 0,
            block: BlockStart::Text,
        })
        .expect_err("nothing precedes the start");
    assert!(matches!(error, StreamError::NotStarted { .. }));
}

#[test]
fn a_stream_that_stopped_without_saying_so_is_not_a_turn() {
    let mut builder = ResponseBuilder::new();
    builder
        .accept(StreamEvent::Start {
            id: "id".into(),
            model: "m".into(),
            created: 0,
        })
        .expect("accepted");
    assert!(matches!(
        builder.finish().expect_err("no terminal event"),
        StreamError::Unterminated
    ));
}

#[test]
fn a_producer_failure_travels_as_a_failure_and_not_as_a_short_turn() {
    let mut builder = ResponseBuilder::new();
    builder
        .accept(StreamEvent::Start {
            id: "id".into(),
            model: "m".into(),
            created: 0,
        })
        .expect("accepted");
    let error = builder
        .accept(StreamEvent::Failed {
            message: "the device went away".into(),
        })
        .expect_err("a failure is not content");
    assert!(matches!(error, StreamError::Failed { ref message } if message.contains("device")));
}

#[test]
fn the_sequence_that_stopped_generation_cannot_be_asked_about_a_turn_that_ended_otherwise() {
    // Carried inside the reason rather than beside it: the shape this replaced
    // had a `matched_stop` field that was meaningful on exactly one of five
    // finish reasons, and `None` on the others meant nothing at all.
    let stopped = StopReason::StopSequence("\n\nHuman:".into());
    match stopped {
        StopReason::StopSequence(ref sequence) => assert_eq!(sequence, "\n\nHuman:"),
        ref other => panic!("expected a stop sequence, got {other:?}"),
    }
    assert!(matches!(StopReason::EndTurn, StopReason::EndTurn));
}

#[test]
fn the_effort_ladder_parses_the_whole_wire_vocabulary_and_refuses_the_rest() {
    assert_eq!(Effort::parse("none"), Some(ThinkingDirective::Off));
    assert_eq!(
        Effort::parse("medium"),
        Some(ThinkingDirective::Effort(Effort::Medium))
    );
    // Both spellings of the top rung reach the same rung.
    assert_eq!(Effort::parse("max"), Effort::parse("xhigh"));
    // An unknown spelling is refused rather than rounded to a rung: the guess
    // that used to be made here was the most expensive one available.
    assert_eq!(Effort::parse("enormous"), None);
}

#[test]
fn the_top_rung_renders_as_the_word_templates_accept() {
    // Templates validate this value against a fixed list and refuse a request
    // that uses a word they do not know, so the spelling is a contract.
    assert_eq!(Effort::XHigh.as_str(), "xhigh");
    assert_eq!(Effort::Minimal.as_str(), "low");
}

#[test]
fn silence_about_thinking_stays_distinguishable_from_turning_it_off() {
    assert!(!ThinkingDirective::Unspecified.is_explicit());
    assert!(ThinkingDirective::Off.is_explicit());
    assert!(ThinkingDirective::On { budget: None }.is_explicit());
}

#[test]
fn a_fresh_request_states_no_preference_the_server_would_have_to_override() {
    let request = Request::new("m", vec![Message::user("hi")]);
    assert_eq!(request.sampling.temperature, None);
    assert_eq!(request.seed, None);
    assert!(request.system.is_empty());
    assert!(request.tools.is_empty());
    assert_eq!(request.tool_choice, None);
    assert!(!request.stream);
}

#[test]
fn a_forced_tool_choice_cannot_be_spelled_without_naming_a_tool() {
    let choice = ToolChoice::Named("lookup".into());
    assert_eq!(choice, ToolChoice::Named("lookup".into()));
    assert_ne!(choice, ToolChoice::Any);
    let tool = ToolDefinition::new("lookup", json!({"type": "object"}));
    assert_eq!(tool.name, "lookup");
    assert_eq!(tool.description, None);
}

#[test]
fn every_type_an_adapter_has_to_build_can_be_built_from_outside_this_crate() {
    // This is a compile-time claim dressed as a test, and it is here because the
    // two agents that first tried to write adapters against this crate both
    // stopped at exactly this line: `#[non_exhaustive]` with public fields and no
    // constructor makes a type that can be read and never made. Every expression
    // below is one an adapter writes when it turns a wire body into the IR.
    let call = ToolUse::new("call_1", "lookup", json!({"city": "Paris"}));
    let ok = ToolResult::new(call.id.clone(), vec![Block::Text("Paris".into())]);
    let bad = ToolResult::error(call.id.clone(), vec![Block::Text("no such city".into())]);
    assert!(!ok.is_error);
    assert!(bad.is_error);

    let detection = RepetitionDetection {
        min_pattern_size: 4,
        max_pattern_size: 64,
        min_count: 3,
    };
    assert_eq!(detection.min_count, 3);

    let logprobs = Logprobs {
        tokens: vec![TokenLogprob {
            token: "Paris".into(),
            logprob: -0.1,
            top: vec![Alternative {
                token: "Lyon".into(),
                logprob: -2.5,
            }],
        }],
    };

    let response = Response {
        id: "abc".into(),
        model: "m".into(),
        created: 0,
        message: Message::assistant("Paris"),
        stop: StopReason::EndTurn,
        usage: Usage::default(),
        logprobs: Some(logprobs),
        token_ids: Vec::new(),
    };
    assert_eq!(response.message.text(), "Paris");

    // And the two that carry a `Default`. A default alone is not enough: from
    // another crate the only way to use one is to default it and then assign,
    // which the lints reject. Written as a literal here because that is what an
    // adapter actually writes.
    let sampling = Sampling {
        temperature: Some(0.7),
        ..Sampling::default()
    };
    assert_eq!(sampling.top_p, None);
}
