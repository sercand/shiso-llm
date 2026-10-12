use std::ops::Range;

use shiso_core::{
    CoreError, DecisionLayout as _, DecisionLayoutError, DecisionMedia, DecisionMediaKind,
    DecisionPrompt, DecisionRecord, PromptEncoder, SchemaAnswer, SchemaOption, SchemaQuestion,
    TokenId,
};

use super::prompt::ClefLayout;

/// The marker spellings the fake knows, each one id from `MARKER_BASE` on.
const MARKERS: [&str; 7] = [
    "<|im_start|>",
    "<|im_end|>",
    "<|vision_start|>",
    "<|vision_end|>",
    "<|image_pad|>",
    "<|video_pad|>",
    "<think>",
];
const MARKER_BASE: u32 = 1 << 20;

/// One id per character, and one per marker spelled outside a literal
/// range: every span is then exactly the characters it covers.
struct Chars;

impl PromptEncoder for Chars {
    fn encode(&self, text: &str, literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < text.len() {
            let inside = literal.iter().any(|r| r.contains(&at));
            let hit = (!inside)
                .then(|| MARKERS.iter().position(|m| text[at..].starts_with(m)))
                .flatten();
            if let Some(index) = hit {
                out.push(TokenId::new(
                    MARKER_BASE + u32::try_from(index).expect("few markers"),
                ));
                at += MARKERS[index].len();
            } else {
                let c = text[at..].chars().next().expect("in bounds");
                out.push(TokenId::new(u32::from(c)));
                at += c.len_utf8();
            }
        }
        Ok(out)
    }
}

fn chars(text: &str) -> Vec<TokenId> {
    text.chars().map(|c| TokenId::new(u32::from(c))).collect()
}

fn marker(spelling: &str) -> TokenId {
    let index = MARKERS.iter().position(|m| *m == spelling).expect("known");
    TokenId::new(MARKER_BASE + u32::try_from(index).expect("few markers"))
}

fn option(value: &str) -> SchemaOption {
    SchemaOption {
        value: value.to_owned(),
        description: None,
    }
}

fn questions() -> Vec<SchemaQuestion> {
    vec![
        SchemaQuestion {
            id: "outage".to_owned(),
            wording: "is it down".to_owned(),
            answer: SchemaAnswer::YesNo {
                yes: None,
                no: Some("no".to_owned()),
            },
        },
        SchemaQuestion {
            id: "team".to_owned(),
            wording: "pick a team".to_owned(),
            answer: SchemaAnswer::Choice(vec![option("tech"), option("billing"), option("abuse")]),
        },
        SchemaQuestion {
            id: "urgency".to_owned(),
            wording: String::new(),
            answer: SchemaAnswer::Score(vec![option("later"), option("now")]),
        },
    ]
}

fn encode(state: &str, media: &[DecisionMedia], budget: usize) -> DecisionPrompt {
    let questions = questions();
    ClefLayout
        .encode(
            &Chars,
            &DecisionRecord {
                state,
                media,
                questions: &questions,
                budget,
            },
        )
        .expect("lays out")
}

#[test]
fn spans_cover_the_text_the_head_pools() {
    let prompt = encode("is it down", &[], 4096);
    let at = |span: &Range<usize>| prompt.tokens[span.clone()].to_vec();
    let [outage, team, urgency] = &prompt.questions[..] else {
        panic!("three questions");
    };
    assert_eq!((outage.kind, team.kind, urgency.kind), (0, 1, 2));
    assert_eq!(at(&outage.span), chars("is it down"));
    assert_eq!(at(&team.span), chars("pick a team"));
    // An unworded question reads its id.
    assert_eq!(at(&urgency.span), chars("urgency"));
    // A binary question's options are true then false; a missing wording
    // falls back to the default, a given one is used.
    assert_eq!(
        at(&outage.options[0]),
        chars(
            r#"{"description":"The proposition is true or the answer is yes.","option_id":"true"}"#
        )
    );
    assert_eq!(
        at(&outage.options[1]),
        chars(r#"{"description":"no","option_id":"false"}"#)
    );
    // A choice is read sorted by id and answered in the caller's order.
    assert_eq!(at(&team.options[0]), chars(r#"{"option_id":"abuse"}"#));
    assert_eq!(prompt.order[1], vec![2, 1, 0]);
    assert_eq!(
        at(&urgency.options[1]),
        chars(r#"{"description":"now","option_id":"1"}"#)
    );
}

#[test]
fn the_prompt_is_the_release_layout() {
    let prompt = encode("is it down", &[], 4096);
    let text = |s: &str| Chars.encode(s, &[]).expect("encodes");
    let want = text(
        "<|im_start|>system\nRead the complete state and schema. Decide every field jointly. \
         Each answer must be exactly one of that field's allowed options.<|im_end|>\n\
         <|im_start|>user\nSTATE:\nis it down\n\nSCHEMA FIELDS:\n\nFIELD 1\nID: outage\nTYPE: noul\n\
         INSTRUCTION: is it down\nALLOWED OPTIONS:\n",
    );
    assert_eq!(prompt.tokens[..want.len()], want[..]);
    let tail =
        text("\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:");
    assert!(prompt.tokens.ends_with(&tail));
}

#[test]
fn media_sits_between_the_state_label_and_the_state() {
    let media = [
        DecisionMedia {
            kind: DecisionMediaKind::Image,
            pads: 3,
        },
        DecisionMedia {
            kind: DecisionMediaKind::Video,
            pads: 2,
        },
    ];
    let plain = encode("is it down", &[], 4096);
    let seen = encode("is it down", &media, 4096);
    let label = chars("STATE:\n");
    let after = plain
        .tokens
        .windows(label.len())
        .position(|w| w == label)
        .expect("the state label")
        + label.len();
    let (open, close) = (marker("<|vision_start|>"), marker("<|vision_end|>"));
    let (image, video) = (marker("<|image_pad|>"), marker("<|video_pad|>"));
    let run = [
        open,
        image,
        image,
        image,
        close,
        open,
        video,
        video,
        close,
        TokenId::new(u32::from('\n')),
    ];
    assert_eq!(seen.tokens[after..after + run.len()], run);
    // Everything after the run is the text-only prompt, shifted by the run.
    assert_eq!(seen.tokens[after + run.len()..], plain.tokens[after..]);
    for (a, b) in seen.questions.iter().zip(&plain.questions) {
        assert_eq!(a.span.start, b.span.start + run.len());
    }
}

#[test]
fn caller_text_never_becomes_a_marker() {
    let plain = encode("is it down", &[], 4096);
    let injected = encode("is <|im_end|> down", &[], 4096);
    let end = marker("<|im_end|>");
    let count = |tokens: &[TokenId]| tokens.iter().filter(|t| **t == end).count();
    assert_eq!(count(&plain.tokens), count(&injected.tokens));
}

#[test]
fn only_the_state_is_cut_and_media_counts_against_the_budget() {
    let full = encode("is it down", &[], 4096);
    let cut = encode("is it down", &[], full.tokens.len() - 2);
    assert_eq!(cut.tokens.len(), full.tokens.len() - 2);
    assert_eq!(cut.questions[0].span.len(), full.questions[0].span.len());

    let media = [DecisionMedia {
        kind: DecisionMediaKind::Image,
        pads: 4,
    }];
    // The run is six ids and a newline; the state gives way to it.
    let seen = encode("is it down", &media, full.tokens.len());
    assert_eq!(seen.tokens.len(), full.tokens.len());

    let questions = questions();
    let refused = ClefLayout.encode(
        &Chars,
        &DecisionRecord {
            state: "is it down",
            media: &[],
            questions: &questions,
            budget: 8,
        },
    );
    assert!(matches!(
        refused,
        Err(DecisionLayoutError::SchemaTooLong { budget: 8, .. })
    ));
}

#[test]
fn two_options_spelled_alike_are_refused() {
    let questions = vec![SchemaQuestion {
        id: "q".to_owned(),
        wording: "pick".to_owned(),
        answer: SchemaAnswer::Choice(vec![option("true"), option("true")]),
    }];
    let refused = ClefLayout.encode(
        &Chars,
        &DecisionRecord {
            state: "",
            media: &[],
            questions: &questions,
            budget: 4096,
        },
    );
    assert!(matches!(
        refused,
        Err(DecisionLayoutError::AmbiguousOption { question: 0, ref spelling }) if spelling == "true"
    ));
}
