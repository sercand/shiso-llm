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

fn email_strings() -> std::sync::Arc<shiso_parsers::StringParams> {
    let schema = serde_json::json!({"type": "object", "properties": {
        "path": {"type": "string"}, "new_text": {"type": "string"},
        "count": {"type": "integer"}, "note": {"type": ["string", "null"]}
    }});
    std::sync::Arc::new(shiso_parsers::StringParams::from_schemas([(
        "edit_file",
        &schema,
    )]))
}

fn edit_call(body: &str) -> serde_json::Value {
    let mut p = shiso_parsers::parser_for_prompt_with(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
        email_strings(),
    );
    let mut segs = p.push(body);
    segs.extend(p.finish());
    match segs.as_slice() {
        [Segment::ToolCall { name, arguments }] if name == "edit_file" => {
            serde_json::from_str(arguments).expect("JSON arguments")
        }
        other => panic!("expected one edit_file call, got {other:?}"),
    }
}

/// With the request's schema in hand, a `string` parameter is a string whatever
/// its text looks like, and a declared integer is still read as one.
#[test]
fn a_string_parameter_that_looks_like_a_number_stays_a_string() {
    let v = edit_call(
        "<tool_call>\n<function=edit_file>\n<parameter=path>\n123\n</parameter>\n<parameter=count>\n2\n</parameter>\n</function>\n</tool_call>",
    );
    assert_eq!(v["path"], serde_json::json!("123"));
    assert_eq!(v["count"], serde_json::json!(2));
}

/// File content that happens to be valid JSON is still the file content, and a
/// nullable string is a string.
#[test]
fn a_json_looking_string_argument_stays_a_string() {
    let v = edit_call(
        "<tool_call>\n<function=edit_file>\n<parameter=new_text>\n{\"a\": 1}\n</parameter>\n<parameter=note>\ntrue\n</parameter>\n</function>\n</tool_call>",
    );
    assert_eq!(v["new_text"], serde_json::json!("{\"a\": 1}"));
    assert_eq!(v["note"], serde_json::json!("true"));
}

/// Without a schema the reader still guesses from the text, as before.
#[test]
fn without_a_schema_a_value_is_read_as_the_json_it_parses_as() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let mut segs = p.push(
        "<tool_call>\n<function=f>\n<parameter=n>\n123\n</parameter>\n</function>\n</tool_call>",
    );
    segs.extend(p.finish());
    let [Segment::ToolCall { arguments, .. }] = segs.as_slice() else {
        panic!("{segs:?}")
    };
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(arguments).expect("JSON")["n"],
        serde_json::json!(123)
    );
}

fn one_call(chunks: &[&str]) -> (String, serde_json::Value) {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let mut segs = Vec::new();
    for chunk in chunks {
        segs.extend(p.push(chunk));
    }
    segs.extend(p.finish());
    match segs.as_slice() {
        [Segment::ToolCall { name, arguments }] => {
            (name.clone(), serde_json::from_str(arguments).expect("JSON"))
        }
        other => panic!("expected one call, got {other:?}"),
    }
}

/// HTML or code that contains the format's own closing tags is a value, not
/// the end of one: a close counts only where the call's structure ends.
#[test]
fn close_tags_inside_a_value_are_part_of_the_value() {
    let (name, args) = one_call(&[
        "<tool_call>\n<function=edit_file>\n<parameter=new_text>\na</parameter>b</function>c</tool_call>d\n</parameter>\n</function>\n</tool_call>",
    ]);
    assert_eq!(name, "edit_file");
    assert_eq!(
        args["new_text"],
        serde_json::json!("a</parameter>b</function>c</tool_call>d")
    );
}

#[test]
fn a_close_split_across_pushes_decodes_the_same() {
    let whole =
        "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n</tool_call>";
    let split = [
        "<tool_call>\n<function=f>\n<parameter=x>\nv\n</para",
        "meter>\n</function>\n</tool_",
        "call>",
    ];
    assert_eq!(one_call(&[whole]), one_call(&split));
}

/// A body with no function in it closes at its first closer, so a malformed
/// call comes back as text while the turn is still streaming instead of
/// swallowing everything after it.
#[test]
fn a_body_with_no_function_closes_at_once() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let raw = "<tool_call>\n{\"name\": \"f\"}\n</tool_call>";
    let segs = p.push(&format!("{raw}after"));
    assert_eq!(
        segs,
        vec![
            Segment::Text(raw.to_string()),
            Segment::Text("after".to_string())
        ]
    );
}

/// A model that drops `</function>` still closed its call with the last
/// parameter's close.
#[test]
fn a_call_that_drops_its_function_close_still_decodes() {
    let (_, args) =
        one_call(&["<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</tool_call>"]);
    assert_eq!(args["x"], serde_json::json!("v"));
}

fn greedy_after(text: &str) -> bool {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        CLOSED_PROMPT,
    );
    let _ = p.push(text);
    p.wants_greedy()
}

#[test]
fn prose_and_values_sample_normally() {
    assert!(!greedy_after("Hello, "));
    assert!(!greedy_after("a < b"));
    assert!(!greedy_after(
        "<tool_call>\n<function=f>\n<parameter=x>\n<html><bo"
    ));
}

#[test]
fn tool_call_structure_is_greedy() {
    assert!(greedy_after("<tool_"));
    assert!(greedy_after("<tool_call>\n<func"));
    assert!(greedy_after(
        "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n"
    ));
    assert!(greedy_after(
        "<tool_call>\n<function=f>\n<parameter=x>\nvalue\n</par"
    ));
}

#[test]
fn a_near_miss_close_returns_to_normal_sampling() {
    assert!(!greedy_after(
        "<tool_call>\n<function=f>\n<parameter=x>\n</paragraph> more"
    ));
}

#[test]
fn thinking_is_never_greedy() {
    let mut p = parser_for_prompt(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        THINKING_PROMPT,
    );
    let _ = p.push("maybe <tool_");
    assert!(!p.wants_greedy());
}
