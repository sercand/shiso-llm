//! Gemma 4 parser tests.

use serde_json::json;
use shiso_parsers::{ReasoningScheme, Segment, ToolCallScheme, gemma_dict_to_json, parser};

#[test]
fn thinking_then_answer() {
    let mut p = parser(ReasoningScheme::Gemma4, ToolCallScheme::Gemma4);
    let mut segs = p.push("<|channel>thought\nlet me see\n<channel|>Paris.");
    segs.extend(p.finish());
    assert_eq!(
        segs,
        vec![
            Segment::ThinkingStart,
            Segment::Thinking("let me see\n".to_string()),
            Segment::ThinkingEnd,
            Segment::Text("Paris.".to_string()),
        ]
    );
}

#[test]
fn a_tool_call_converts_to_json() {
    let mut p = parser(ReasoningScheme::Gemma4, ToolCallScheme::Gemma4);
    let raw = "<|tool_call>call:get_weather{city:<|\"|>Paris, FR<|\"|>,days:3,units:[<|\"|>c<|\"|>],opts:{fast:true,limit:null}}<tool_call|>";
    let mut segs = p.push(raw);
    segs.extend(p.finish());
    assert_eq!(segs.len(), 1);
    match &segs[0] {
        Segment::ToolCall { name, arguments } => {
            assert_eq!(name, "get_weather");
            let parsed: serde_json::Value =
                serde_json::from_str(arguments).expect("arguments must be valid json");
            assert_eq!(
                parsed,
                json!({
                    "city": "Paris, FR",
                    "days": 3,
                    "units": ["c"],
                    "opts": {
                        "fast": true,
                        "limit": null
                    }
                })
            );
        }
        other => panic!("expected ToolCall, got {other:?}"),
    }
}

#[test]
fn text_around_a_call_is_kept() {
    let mut p = parser(ReasoningScheme::Gemma4, ToolCallScheme::Gemma4);
    let mut segs = p.push("Sure.<|tool_call>call:f{}<tool_call|>");
    segs.extend(p.finish());
    assert_eq!(
        segs,
        vec![
            Segment::Text("Sure.".to_string()),
            Segment::ToolCall {
                name: "f".to_string(),
                arguments: "{}".to_string(),
            },
        ]
    );
}

#[test]
fn a_broken_call_is_text() {
    let mut p = parser(ReasoningScheme::Gemma4, ToolCallScheme::Gemma4);
    let mut segs = p.push("<|tool_call>call:f{unterminated");
    segs.extend(p.finish());
    assert_eq!(
        segs,
        vec![Segment::Text("<|tool_call>call:f{unterminated".to_string())]
    );
}

#[test]
fn gemma_dict_to_json_escapes_quotes() {
    let dict = r#"{msg:<|"|>hello "world"
newline<|"|>}"#;
    let json_str = gemma_dict_to_json(dict).expect("should parse");
    let parsed: serde_json::Value = serde_json::from_str(&json_str).expect("should be valid json");
    assert_eq!(
        parsed,
        json!({
            "msg": "hello \"world\"\nnewline"
        })
    );
}
