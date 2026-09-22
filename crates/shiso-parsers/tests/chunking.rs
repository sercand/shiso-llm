//! Chunking invariance tests for stream parsers.

use shiso_parsers::{ReasoningScheme, Segment, ToolCallScheme, parser};

fn normalize_segments(segments: &[Segment]) -> (String, String, Vec<(String, String)>) {
    let mut text = String::new();
    let mut thinking = String::new();
    let mut tools = Vec::new();

    for seg in segments {
        match seg {
            Segment::Text(s) => text.push_str(s),
            Segment::Thinking(s) => thinking.push_str(s),
            Segment::ToolCall { name, arguments } => {
                tools.push((name.clone(), arguments.clone()));
            }
            _ => {}
        }
    }

    (text, thinking, tools)
}

fn assert_no_tags_in_text(segments: &[Segment], tags: &[&str]) {
    for seg in segments {
        if let Segment::Text(s) = seg {
            for &tag in tags {
                assert!(
                    !s.contains(tag),
                    "Text segment {s:?} contained forbidden tag {tag:?}"
                );
            }
        }
    }
}

fn run_chunking_test(
    reasoning: ReasoningScheme,
    tools: ToolCallScheme,
    input: &str,
    tags: &[&str],
) {
    // 1. One-shot parse
    let mut p_oneshot = parser(reasoning, tools);
    let mut oneshot_segs = p_oneshot.push(input);
    oneshot_segs.extend(p_oneshot.finish());
    let baseline = normalize_segments(&oneshot_segs);
    assert_no_tags_in_text(&oneshot_segs, tags);

    // 2. One byte at a time
    let mut p_1byte = parser(reasoning, tools);
    let mut segs_1byte = Vec::new();
    for byte in input.as_bytes() {
        let s = std::str::from_utf8(std::slice::from_ref(byte)).unwrap_or_default();
        if !s.is_empty() {
            segs_1byte.extend(p_1byte.push(s));
        }
    }
    segs_1byte.extend(p_1byte.finish());
    assert_eq!(normalize_segments(&segs_1byte), baseline, "1-byte mismatch");
    assert_no_tags_in_text(&segs_1byte, tags);

    // 3. Two bytes at a time
    let mut p_2byte = parser(reasoning, tools);
    let mut segs_2byte = Vec::new();
    let mut cursor = 0;
    while cursor < input.len() {
        let end = (cursor + 2).min(input.len());
        let mut actual_end = end;
        while actual_end < input.len() && !input.is_char_boundary(actual_end) {
            actual_end += 1;
        }
        segs_2byte.extend(p_2byte.push(&input[cursor..actual_end]));
        cursor = actual_end;
    }
    segs_2byte.extend(p_2byte.finish());
    assert_eq!(normalize_segments(&segs_2byte), baseline, "2-byte mismatch");
    assert_no_tags_in_text(&segs_2byte, tags);

    // 4. Random chunks (fixed-seed LCG)
    let mut state: u64 = 0x1234_5678_9ABC_DEF0;
    let mut lcg = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        usize::try_from((state >> 33) % 7 + 1).unwrap_or(1)
    };

    let mut p_rand = parser(reasoning, tools);
    let mut segs_rand = Vec::new();
    cursor = 0;
    while cursor < input.len() {
        let step = lcg();
        let end = (cursor + step).min(input.len());
        let mut actual_end = end;
        while actual_end < input.len() && !input.is_char_boundary(actual_end) {
            actual_end += 1;
        }
        segs_rand.extend(p_rand.push(&input[cursor..actual_end]));
        cursor = actual_end;
    }
    segs_rand.extend(p_rand.finish());
    assert_eq!(
        normalize_segments(&segs_rand),
        baseline,
        "random-chunk mismatch"
    );
    assert_no_tags_in_text(&segs_rand, tags);
}

#[test]
fn chunking_gemma4_thinking_and_answer() {
    let input = "<|channel>thought\nlet me see\n<channel|>Paris.";
    let tags = ["<|channel>thought\n", "<channel|>"];
    run_chunking_test(
        ReasoningScheme::Gemma4,
        ToolCallScheme::Gemma4,
        input,
        &tags,
    );
}

#[test]
fn chunking_gemma4_tool_call() {
    let input = "Sure.<|tool_call>call:get_weather{city:<|\"|>Paris, FR<|\"|>,days:3,units:[<|\"|>c<|\"|>],opts:{fast:true,limit:null}}<tool_call|>Done.";
    let tags = ["<|tool_call>", "<tool_call|>"];
    run_chunking_test(
        ReasoningScheme::Gemma4,
        ToolCallScheme::Gemma4,
        input,
        &tags,
    );
}

#[test]
fn chunking_qwen3_thinking_and_tool_call() {
    let input = "<think>I need to check the weather</think>Calling tool now: <tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}</tool_call> finished.";
    let tags = ["<think>", "</think>", "<tool_call>", "</tool_call>"];
    run_chunking_test(ReasoningScheme::Qwen3, ToolCallScheme::Qwen3, input, &tags);
}
