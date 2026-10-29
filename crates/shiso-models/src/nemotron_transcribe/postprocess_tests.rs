//! The loop guard and the tag handling on hand-written transcripts.

use super::{collapse_loops, speaker_turns, strip_tags};

#[test]
fn collapse_loops_examples() {
    assert_eq!(collapse_loops(&"a".repeat(30)), "a");
    assert_eq!(collapse_loops(&"ok ".repeat(25)), "ok ");
    assert_eq!(collapse_loops(&"ab".repeat(5)), "ababababab");
}

#[test]
fn collapse_loops_thresholds() {
    // One character: 20 stays, 21 collapses.
    assert_eq!(collapse_loops(&"-".repeat(20)), "-".repeat(20));
    assert_eq!(collapse_loops(&"-".repeat(21)), "-");
    // A longer block: 19 stays, 20 collapses.
    assert_eq!(collapse_loops(&"no ".repeat(19)), "no ".repeat(19));
    assert_eq!(collapse_loops(&"no ".repeat(20)), "no ");
}

#[test]
fn collapse_loops_keeps_the_text_around_a_loop() {
    let looped = format!("Hello. {}Bye.", "thank you. ".repeat(40));
    assert_eq!(collapse_loops(&looped), "Hello. thank you. Bye.");
    assert_eq!(collapse_loops("plain text"), "plain text");
    assert_eq!(collapse_loops(""), "");
}

#[test]
fn strip_tags_removes_events_and_styles_and_optionally_speakers() {
    let text = "<spk:0> Hi <ev:laugh> there. <sty:whisper>ok <spk:1> yo";
    assert_eq!(strip_tags(text, false), "Hi there. ok yo");
    assert_eq!(strip_tags(text, true), "<spk:0> Hi there. ok <spk:1> yo");
}

#[test]
fn speaker_turns_split_on_tags() {
    assert_eq!(
        speaker_turns("<spk:0> hi <spk:1> yo"),
        vec![(Some(0), "hi".to_owned()), (Some(1), "yo".to_owned())]
    );
}

#[test]
fn speaker_turns_keep_untagged_lead_text() {
    assert_eq!(
        speaker_turns("so <spk:2> right <ev:cough> then"),
        vec![(None, "so".to_owned()), (Some(2), "right then".to_owned())]
    );
    assert_eq!(speaker_turns("no tags"), vec![(None, "no tags".to_owned())]);
    assert!(speaker_turns("  ").is_empty());
}
