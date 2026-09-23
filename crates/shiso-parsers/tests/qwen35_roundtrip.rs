//! Rendering a call the way the template does, then reading it back, returns
//! the call - for values containing everything that has broken the reader
//! before: markup, entities, quotes, newlines and the format's own closing tags.
//!
//! What the format genuinely cannot carry is listed in `UNCARRIABLE` and checked
//! to stay that way, so a change that makes one of them readable is noticed too.

use shiso_parsers::{
    ReasoningScheme, Segment, StringParams, ToolCallScheme, parser_for_prompt_with,
};
use std::fmt::Write as _;
use std::sync::Arc;

const PROMPT: &str = "<|im_start|>assistant\n<think>\n\n</think>\n\n";

/// The template's own call rendering, for string values.
fn render(name: &str, args: &[(&str, &str)]) -> String {
    let mut out = format!("<tool_call>\n<function={name}>\n");
    for (key, value) in args {
        write!(out, "<parameter={key}>\n{value}\n</parameter>\n").expect("a String takes writes");
    }
    out.push_str("</function>\n</tool_call>");
    out
}

fn read(text: &str, strings: Arc<StringParams>) -> Vec<Segment> {
    let mut p = parser_for_prompt_with(
        ReasoningScheme::Qwen3,
        ToolCallScheme::Qwen35,
        PROMPT,
        strings,
    );
    // One character at a time: the harshest chunking there is.
    let mut segs = Vec::new();
    let mut buf = [0u8; 4];
    for ch in text.chars() {
        segs.extend(p.push(ch.encode_utf8(&mut buf)));
    }
    segs.extend(p.finish());
    segs
}

const VALUES: &[&str] = &[
    "<html><body style=\"font-family:'Times New Roman', serif\">Hi &amp; welcome</body></html>",
    "a</parameter>b",
    "x</function>y",
    "p</tool_call>q",
    "line one\n\nline three\n",
    "{\"looks\": [\"like\", \"json\"]}",
    "   leading and trailing spaces   ",
    "unicode: ü 漢字 🙂",
];

/// Values whose own text ends in a sequence the reader must take for a real
/// close. The format has no escape, so these are limits of the format.
const UNCARRIABLE: &[&str] = &["tail\n</parameter>\n</function>\n</tool_call>"];

fn strings() -> Arc<StringParams> {
    let schema = serde_json::json!({"type": "object", "properties": {"v": {"type": "string"}, "w": {"type": "string"}}});
    Arc::new(StringParams::from_schemas([("t", &schema)]))
}

#[test]
fn every_carriable_value_survives_the_round_trip() {
    for value in VALUES {
        let segs = read(&render("t", &[("v", value), ("w", "second")]), strings());
        let [Segment::ToolCall { name, arguments }] = segs.as_slice() else {
            panic!("value {value:?} did not read back as one call: {segs:?}");
        };
        assert_eq!(name, "t");
        let args: serde_json::Value = serde_json::from_str(arguments).expect("JSON");
        assert_eq!(args["v"], serde_json::json!(value), "value {value:?}");
        assert_eq!(args["w"], serde_json::json!("second"), "value {value:?}");
    }
}

#[test]
fn the_uncarriable_values_are_still_uncarriable() {
    for value in UNCARRIABLE {
        let segs = read(&render("t", &[("v", value)]), strings());
        let intact = matches!(segs.as_slice(), [Segment::ToolCall { arguments, .. }]
            if serde_json::from_str::<serde_json::Value>(arguments)
                .is_ok_and(|a| a["v"] == serde_json::json!(value)));
        assert!(!intact, "{value:?} now round-trips; move it to VALUES");
    }
}
