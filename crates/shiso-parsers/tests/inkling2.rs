//! Stream parser tests for Inkling-2 reasoning and tool-call tags.

use shiso_parsers::{ReasoningScheme, Segment, ToolCallScheme, parser};

#[test]
fn parses_inkling2_thinking_and_prose() {
    let mut p = parser(ReasoningScheme::Inkling2, ToolCallScheme::Inkling2);
    let mut segs = p.push(
        "<|content_thinking|>Let me check.<|end_message|><|message_model|><|content_text|>The answer is 42.<|end_message|>",
    );
    segs.extend(p.finish());
    assert!(segs.contains(&Segment::ThinkingStart));
    assert!(
        segs.iter()
            .any(|s| matches!(s, Segment::Thinking(t) if t.contains("Let me check.")))
    );
    assert!(segs.contains(&Segment::ThinkingEnd));
    let prose: String = segs
        .iter()
        .filter_map(|s| match s {
            Segment::Text(t) => Some(t.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(prose, "The answer is 42.");
}
