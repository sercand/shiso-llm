//! Qwen 3 parser tests.

use shiso_parsers::{ReasoningScheme, Segment, ToolCallScheme, parser};

#[test]
fn think_split_into_three_pushes() {
    let mut p = parser(ReasoningScheme::Qwen3, ToolCallScheme::Qwen3);
    let mut segs = Vec::new();
    segs.extend(p.push("<thi"));
    segs.extend(p.push("nk>thinking here</thi"));
    segs.extend(p.push("nk>done"));
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![
            Segment::ThinkingStart,
            Segment::Thinking("thinking here".to_string()),
            Segment::ThinkingEnd,
            Segment::Text("done".to_string()),
        ]
    );
}

#[test]
fn tool_call_with_object_arguments() {
    let mut p = parser(ReasoningScheme::Qwen3, ToolCallScheme::Qwen3);
    let raw = r#"<tool_call>{"name":"f","arguments":{"a":1}}</tool_call>"#;
    let mut segs = p.push(raw);
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![Segment::ToolCall {
            name: "f".to_string(),
            arguments: r#"{"a":1}"#.to_string(),
        }]
    );
}

#[test]
fn tool_call_with_string_arguments_unwrapped() {
    let mut p = parser(ReasoningScheme::Qwen3, ToolCallScheme::Qwen3);
    let raw = r#"<tool_call>{"name":"f","arguments":"{\"a\":1}"}</tool_call>"#;
    let mut segs = p.push(raw);
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![Segment::ToolCall {
            name: "f".to_string(),
            arguments: r#"{"a":1}"#.to_string(),
        }]
    );
}

#[test]
fn two_consecutive_calls_yield_two_segments() {
    let mut p = parser(ReasoningScheme::Qwen3, ToolCallScheme::Qwen3);
    let raw = r#"<tool_call>{"name":"f1","arguments":{"x":10}}</tool_call><tool_call>{"name":"f2","arguments":{"y":20}}</tool_call>"#;
    let mut segs = p.push(raw);
    segs.extend(p.finish());

    assert_eq!(
        segs,
        vec![
            Segment::ToolCall {
                name: "f1".to_string(),
                arguments: r#"{"x":10}"#.to_string(),
            },
            Segment::ToolCall {
                name: "f2".to_string(),
                arguments: r#"{"y":20}"#.to_string(),
            },
        ]
    );
}
