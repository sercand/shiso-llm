//! Qwen 3.5 parser tests.
//!
//! Two things separate this family from Qwen 3, and both of them are things the
//! chat template does rather than things the weights do. The template closes the
//! generation prompt with `<think>\n`, so a turn *begins* inside the reasoning
//! trace and the model never emits an opener; and a tool call is an XML body
//! rather than a JSON one.

use shiso_parsers::{
    ReasoningScheme, Segment, ToolCallScheme, parser_for_prompt, resumes_thinking,
};

/// The generation prompt Qwen 3.5's template writes when thinking is on: the
/// opener is the last thing in the prompt, so the turn starts inside the trace.
const THINKING_PROMPT: &str = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n";

/// The same template with `enable_thinking: false`: the trace is opened *and*
/// closed in the prompt, so the turn starts in prose.
const CLOSED_PROMPT: &str =
    "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

#[test]
fn a_prompt_that_ends_inside_the_trace_resumes_thinking() {
    assert!(resumes_thinking(ReasoningScheme::Qwen3, THINKING_PROMPT));
    assert!(!resumes_thinking(ReasoningScheme::Qwen3, CLOSED_PROMPT));
    assert!(!resumes_thinking(ReasoningScheme::Qwen3, ""));
    // A scheme with no reasoning tags has nothing to resume, whatever the
    // prompt happens to contain.
    assert!(!resumes_thinking(ReasoningScheme::None, THINKING_PROMPT));
}

#[test]
fn a_prefilled_opener_puts_the_trace_in_its_own_segment() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        THINKING_PROMPT,
    );
    let mut segs = p.push("weighing it up</think>\n\nHello!");
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![
            Segment::Thinking("weighing it up".to_string()),
            Segment::ThinkingEnd,
            Segment::Text("\n\nHello!".to_string()),
        ]
    );
}

#[test]
fn a_closed_trace_in_the_prompt_leaves_the_turn_in_prose() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let mut segs = p.push("Hello!");
    segs.extend(p.finish());

    assert_eq!(segs, vec![Segment::Text("Hello!".to_string())]);
}

#[test]
fn a_resumed_trace_closes_itself_when_the_turn_ends_early() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        THINKING_PROMPT,
    );
    let mut segs = p.push("still thinking");
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![
            Segment::Thinking("still thinking".to_string()),
            Segment::ThinkingEnd,
        ]
    );
}

#[test]
fn the_closer_survives_being_split_across_pushes() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        THINKING_PROMPT,
    );
    let mut segs = p.push("up and ");
    segs.extend(p.push("down</thi"));
    segs.extend(p.push("nk>answer"));
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![
            Segment::Thinking("up and ".to_string()),
            Segment::Thinking("down".to_string()),
            Segment::ThinkingEnd,
            Segment::Text("answer".to_string()),
        ]
    );
}

#[test]
fn an_xml_call_becomes_a_tool_call() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let raw = "<tool_call>\n<function=get_weather>\n<parameter=city>\nParis, FR\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>";
    let mut segs = p.push(raw);
    segs.extend(p.finish());

    let Some(Segment::ToolCall { name, arguments }) = segs.first() else {
        panic!("the call parsed, got {segs:?}");
    };
    assert_eq!(name, "get_weather");
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).expect("the arguments are JSON"),
        serde_json::json!({"city": "Paris, FR", "days": 3}),
    );
}

#[test]
fn a_json_valued_parameter_keeps_its_shape() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let raw = "<tool_call>\n<function=f>\n<parameter=opts>\n{\"fast\": true, \"limit\": null}\n</parameter>\n<parameter=units>\n[\"c\"]\n</parameter>\n<parameter=note>\nnot json\n</parameter>\n</function>\n</tool_call>";
    let mut segs = p.push(raw);
    segs.extend(p.finish());

    let Some(Segment::ToolCall { arguments, .. }) = segs.first() else {
        panic!("the call parsed, got {segs:?}");
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).expect("the arguments are JSON"),
        serde_json::json!({
            "opts": {"fast": true, "limit": null},
            "units": ["c"],
            "note": "not json",
        }),
    );
}

#[test]
fn a_multi_line_parameter_keeps_its_interior_newlines() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let raw = "<tool_call>\n<function=write>\n<parameter=body>\nline one\nline two\n</parameter>\n</function>\n</tool_call>";
    let mut segs = p.push(raw);
    segs.extend(p.finish());

    let Some(Segment::ToolCall { arguments, .. }) = segs.first() else {
        panic!("the call parsed, got {segs:?}");
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).expect("the arguments are JSON"),
        serde_json::json!({"body": "line one\nline two"}),
    );
}

#[test]
fn a_call_with_no_parameters_carries_an_empty_object() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let mut segs = p.push("<tool_call>\n<function=now>\n</function>\n</tool_call>");
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![Segment::ToolCall {
            name: "now".to_string(),
            arguments: "{}".to_string(),
        }]
    );
}

#[test]
fn prose_before_a_call_is_kept() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let mut segs = p.push("Let me look.<tool_call>\n<function=now>\n</function>\n</tool_call>");
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![
            Segment::Text("Let me look.".to_string()),
            Segment::ToolCall {
                name: "now".to_string(),
                arguments: "{}".to_string(),
            },
        ]
    );
}

#[test]
fn a_body_that_is_not_a_call_stays_text() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let raw = "<tool_call>\nnot a function at all\n</tool_call>";
    let mut segs = p.push(raw);
    segs.extend(p.finish());

    assert_eq!(segs, vec![Segment::Text(raw.to_string())]);
}

#[test]
fn a_whole_turn_splits_into_a_trace_and_an_answer() {
    // The shape the reported request actually produced, with the opener that
    // the prompt - not the model - supplied.
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        THINKING_PROMPT,
    );
    let mut segs = p.push("The user says hello.\n</think>\n\nHello! How can I help you today?");
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![
            Segment::Thinking("The user says hello.\n".to_string()),
            Segment::ThinkingEnd,
            Segment::Text("\n\nHello! How can I help you today?".to_string()),
        ]
    );
}
