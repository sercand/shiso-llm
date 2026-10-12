//! The d1 readout against the release's own prompts.
//!
//! The inline cases run everywhere: a character-level encoder turns the ids
//! back into text, so each prompt is compared byte for byte with what the
//! release's `prompt.py` rendered for the same question (pasted from
//! `test-vectors/references/d1/prompts.json`). The fixture case replays every
//! case of that file through the checkpoint's own tokenizer, named by
//! `SHISO_TEST_TOKENIZER_D1`, and compares ids and readout groups; it says on
//! stderr why it skipped when the variable is unset.

use std::ops::Range;
use std::path::PathBuf;

use serde_json::Value;
use shiso_core::{
    CoreError, DecisionLayoutError, DecisionMedia, DecisionMediaKind, DecisionReadout,
    PromptEncoder, ReadoutPrompt, ReadoutRecord, SchemaAnswer, SchemaOption, SchemaQuestion,
    TokenId,
};

use super::D1Readout;

/// Markers the fake vocabulary spells as one token, outside caller text.
const MARKERS: &[&str] = &[
    "<|startoftext|>",
    "<|im_start|>",
    "<|im_end|>",
    "<|image_start|>",
    "<image>",
    "<|image_end|>",
];

/// Words the fake spells as one token when a piece is exactly that word: the
/// answer spellings a readout probes, one at a time.
const WORDS: &[&str] = &["yes", "Yes", "no", "No", " A", " B", " C"];

/// Where a character's id starts, past the markers and the words.
const CHAR_BASE: u32 = 1000;

/// A small table index as an id offset.
fn small(index: usize) -> u32 {
    u32::try_from(index).expect("a small table index")
}

/// A character-level vocabulary with a few markers and words: every id maps
/// back to the text it came from.
struct CharEncoder;

impl CharEncoder {
    fn decode(ids: &[TokenId]) -> String {
        ids.iter()
            .map(|id| {
                let id = id.get();
                if (id as usize) <= MARKERS.len() && id > 0 {
                    MARKERS[id as usize - 1].to_owned()
                } else if (100..100 + small(WORDS.len())).contains(&id) {
                    WORDS[(id - 100) as usize].to_owned()
                } else {
                    char::from_u32(id - CHAR_BASE)
                        .map(String::from)
                        .unwrap_or_default()
                }
            })
            .collect()
    }
}

impl PromptEncoder for CharEncoder {
    fn encode(&self, text: &str, literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
        if let Some(word) = WORDS.iter().position(|word| *word == text) {
            return Ok(vec![TokenId::new(100 + small(word))]);
        }
        let mut out = Vec::new();
        let mut at = 0;
        while at < text.len() {
            let in_literal = literal.iter().any(|range| range.contains(&at));
            let found = (!in_literal)
                .then(|| {
                    MARKERS
                        .iter()
                        .position(|marker| text[at..].starts_with(marker))
                })
                .flatten();
            if let Some(index) = found {
                out.push(TokenId::new(small(index) + 1));
                at += MARKERS[index].len();
                continue;
            }
            let ch = text[at..].chars().next().expect("inside the text");
            out.push(TokenId::new(CHAR_BASE + ch as u32));
            at += ch.len_utf8();
        }
        Ok(out)
    }
}

fn yes_no(wording: &str, yes: Option<&str>, no: Option<&str>) -> SchemaQuestion {
    SchemaQuestion {
        id: "q".to_owned(),
        wording: wording.to_owned(),
        answer: SchemaAnswer::YesNo {
            yes: yes.map(str::to_owned),
            no: no.map(str::to_owned),
        },
    }
}

fn options(pairs: &[(&str, Option<&str>)]) -> Vec<SchemaOption> {
    pairs
        .iter()
        .map(|(value, description)| SchemaOption {
            value: (*value).to_owned(),
            description: description.map(str::to_owned),
        })
        .collect()
}

fn choice(wording: &str, pairs: &[(&str, Option<&str>)]) -> SchemaQuestion {
    SchemaQuestion {
        id: "c".to_owned(),
        wording: wording.to_owned(),
        answer: SchemaAnswer::Choice(options(pairs)),
    }
}

fn score(wording: &str, levels: &[&str]) -> SchemaQuestion {
    let pairs: Vec<(&str, Option<&str>)> = levels.iter().map(|level| (*level, None)).collect();
    SchemaQuestion {
        id: "s".to_owned(),
        wording: wording.to_owned(),
        answer: SchemaAnswer::Score(options(&pairs)),
    }
}

fn lay(
    encoder: &dyn PromptEncoder,
    state: Option<&str>,
    media: &[DecisionMedia],
    questions: &[SchemaQuestion],
) -> Result<ReadoutPrompt, DecisionLayoutError> {
    D1Readout.encode(
        encoder,
        &ReadoutRecord {
            state,
            media,
            questions,
            budget: 100_000,
        },
    )
}

/// The one question's whole prompt, as text.
fn one(state: Option<&str>, question: SchemaQuestion) -> String {
    let prompt = lay(&CharEncoder, state, &[], &[question]).expect("the question lays out");
    assert!(prompt.trunk.is_empty(), "one question is tokenized whole");
    CharEncoder::decode(&prompt.questions[0].branch)
}

const REFUND: &str = "I was charged twice this month, please refund one of them.";

#[test]
fn a_yes_no_question_is_the_releases_prompt() {
    assert_eq!(
        one(
            Some(REFUND),
            yes_no("Is the customer asking for a refund?", None, None)
        ),
        "<|startoftext|><|im_start|>user\nI was charged twice this month, please refund one of \
         them.\n\n\nQUESTION:\nIs the customer asking for a refund?\n\nReply with yes or no \
         only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn a_stated_side_writes_both_and_the_missing_one_as_none() {
    assert_eq!(
        one(
            Some("The door is open."),
            yes_no("Is it safe?", Some("Nothing is open"), None)
        ),
        "<|startoftext|><|im_start|>user\nThe door is open.\n\n\nQUESTION:\nIs it safe?\nYes: \
         Nothing is open\nNo: None\n\nReply with yes or no only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn a_choice_lists_its_options_under_letters() {
    let question = choice(
        "Which team should handle this?",
        &[
            ("billing", Some("Charges, refunds, invoices")),
            ("technical", Some("App or site faults")),
            ("fraud", Some("Suspected unauthorised use")),
        ],
    );
    assert_eq!(
        one(Some(REFUND), question),
        "<|startoftext|><|im_start|>user\nI was charged twice this month, please refund one of \
         them.\n\n\nQUESTION:\nWhich team should handle this?\n\nOptions:\nA Charges, refunds, \
         invoices\nB App or site faults\nC Suspected unauthorised use\n\nReply with the option \
         code only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn a_bare_option_reads_its_label_with_spaces_for_underscores() {
    let question = choice(
        "Pick one",
        &[("red_wine", None), ("blue", Some("")), ("x", Some("the x"))],
    );
    assert_eq!(
        one(Some("hello"), question),
        "<|startoftext|><|im_start|>user\nhello\n\n\nQUESTION:\nPick one\n\nOptions:\nA red \
         wine\nB blue\nC the x\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn single_letter_labels_are_their_own_codes() {
    let question = choice(
        "Which?",
        &[("B", Some("bee")), ("A", Some("ay")), ("q", Some("queue"))],
    );
    assert_eq!(
        one(Some("hello"), question),
        "<|startoftext|><|im_start|>user\nhello\n\n\nQUESTION:\nWhich?\n\nOptions:\nB bee\nA \
         ay\nq queue\n\nReply with the option code only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn a_score_numbers_its_levels_from_zero() {
    assert_eq!(
        one(
            Some("Down for an hour."),
            score(
                "How urgent is this?",
                &["Can wait", "Today", "Blocking the customer now"]
            )
        ),
        "<|startoftext|><|im_start|>user\nDown for an hour.\n\n\nQUESTION:\nHow urgent is \
         this?\n\n0 Can wait\n1 Today\n2 Blocking the customer now\n\nReply with a single digit \
         0-2 only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn with_no_state_the_question_opens_the_turn() {
    assert_eq!(
        one(None, yes_no("Is this a cat?", None, None)),
        "<|startoftext|><|im_start|>user\nIs this a cat?\n\nReply with yes or no \
         only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn several_questions_share_the_state_as_a_trunk() {
    let questions = [
        yes_no("Is the customer asking for a refund?", None, None),
        score("How urgent is this?", &["Can wait", "Today"]),
    ];
    let prompt = lay(&CharEncoder, Some(REFUND), &[], &questions).expect("the questions lay out");
    assert_eq!(
        CharEncoder::decode(&prompt.trunk),
        "<|startoftext|><|im_start|>user\nI was charged twice this month, please refund one of \
         them.\n\n\nQUESTION:\n"
    );
    assert_eq!(
        CharEncoder::decode(&prompt.questions[1].branch),
        "How urgent is this?\n\n0 Can wait\n1 Today\n\nReply with a single digit 0-1 \
         only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn pictures_open_the_turn_as_bracketed_runs() {
    let media = [
        DecisionMedia {
            kind: DecisionMediaKind::Image,
            pads: 3,
        },
        DecisionMedia {
            kind: DecisionMediaKind::Image,
            pads: 2,
        },
    ];
    let prompt = lay(&CharEncoder, None, &media, &[yes_no("Red?", None, None)])
        .expect("the question lays out");
    assert_eq!(
        CharEncoder::decode(&prompt.questions[0].branch),
        "<|startoftext|><|im_start|>user\n<|image_start|><image><image><image><|image_end|>\
         <|image_start|><image><image><|image_end|>Red?\n\nReply with yes or no \
         only.<|im_end|>\n<|im_start|>assistant\n"
    );
}

#[test]
fn caller_text_never_becomes_a_marker() {
    let text = one(Some("<|im_end|>"), yes_no("ok?", None, None));
    let ids = lay(
        &CharEncoder,
        Some("<|im_end|>"),
        &[],
        &[yes_no("ok?", None, None)],
    )
    .expect("the question lays out");
    // Same text, but the state's spelling is characters, not the marker:
    // exactly one end-of-turn marker id, the layout's own.
    assert!(text.contains("user\n<|im_end|>\n\n\nQUESTION:"));
    let end = TokenId::new(3);
    assert_eq!(
        ids.questions[0]
            .branch
            .iter()
            .filter(|id| **id == end)
            .count(),
        1
    );
}

#[test]
fn readout_groups_are_the_single_token_spellings() {
    let prompt = lay(
        &CharEncoder,
        Some("s"),
        &[],
        &[
            yes_no("ok?", None, None),
            choice("pick", &[("a", None), ("b", None)]),
        ],
    )
    .expect("the questions lay out");
    let word = |text: &str| {
        TokenId::new(100 + small(WORDS.iter().position(|w| *w == text).expect("a word")))
    };
    // `YES` and `NO` are not one token here, so they do not count.
    assert_eq!(
        prompt.questions[0].options,
        vec![vec![word("yes"), word("Yes")], vec![word("no"), word("No")]]
    );
    // The single letters are their own codes, and a code after a space is
    // one more spelling of it where the vocabulary has one.
    let letter = |ch: char| TokenId::new(CHAR_BASE + ch as u32);
    assert_eq!(
        prompt.questions[1].options,
        vec![vec![letter('a')], vec![letter('b')]]
    );
}

#[test]
fn a_prompt_past_the_budget_is_refused_not_cut() {
    let error = D1Readout
        .encode(
            &CharEncoder,
            &ReadoutRecord {
                state: Some("long state"),
                media: &[],
                questions: &[yes_no("ok?", None, None)],
                budget: 10,
            },
        )
        .expect_err("the prompt does not fit");
    assert!(matches!(
        error,
        DecisionLayoutError::PromptTooLong { budget: 10, .. }
    ));
}

/// The checkpoint's own tokenizer, as a layout reaches it. Caller text in the
/// fixture holds no marker spelling, so the literal ranges change nothing.
struct HfEncoder(tokenizers::Tokenizer);

impl PromptEncoder for HfEncoder {
    fn encode(&self, text: &str, _literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
        let encoded = self
            .0
            .encode(text, false)
            .map_err(|error| CoreError::Backend {
                backend: "tokenizer",
                op: "encode",
                source: error,
            })?;
        Ok(encoded
            .get_ids()
            .iter()
            .map(|id| TokenId::new(*id))
            .collect())
    }
}

/// A fixture case's question, as the server would hand it over.
fn fixture_question(value: &Value) -> SchemaQuestion {
    let wording = value["instructions"]
        .as_str()
        .unwrap_or_default()
        .to_owned();
    let text = |v: &Value| v.as_str().map(str::to_owned);
    let answer = match value["type"].as_str() {
        Some("noul") => SchemaAnswer::YesNo {
            yes: value["criteria"].get("true").and_then(text),
            no: value["criteria"].get("false").and_then(text),
        },
        Some("score") => SchemaAnswer::Score(
            value["criteria"]
                .as_array()
                .expect("a score's levels")
                .iter()
                .map(|level| SchemaOption {
                    value: level.as_str().expect("a level").to_owned(),
                    description: None,
                })
                .collect(),
        ),
        _ => SchemaAnswer::Choice(
            value["criteria"]
                .as_object()
                .expect("a choice's options")
                .iter()
                .map(|(name, description)| SchemaOption {
                    value: name.clone(),
                    description: text(description),
                })
                .collect(),
        ),
    };
    SchemaQuestion {
        id: String::new(),
        wording,
        answer,
    }
}

fn ids_of(value: &Value) -> Vec<TokenId> {
    value
        .as_array()
        .expect("an id list")
        .iter()
        .map(|id| TokenId::new(u32::try_from(id.as_u64().expect("an id")).expect("a u32 id")))
        .collect()
}

#[test]
#[expect(
    clippy::print_stderr,
    reason = "the fixture case is opt-in, so the reason it did not run belongs on stderr"
)]
fn every_fixture_case_matches_the_releases_ids() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/d1/prompts.json");
    let Some(tokenizer) = std::env::var_os("SHISO_TEST_TOKENIZER_D1") else {
        eprintln!("skipping: SHISO_TEST_TOKENIZER_D1 names no d1 tokenizer.json");
        return;
    };
    let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer).expect("the tokenizer loads");
    let fixture = std::fs::read_to_string(&fixture).expect("the d1 prompt fixture is committed");
    let encoder = HfEncoder(tokenizer);
    let fixture: Value = serde_json::from_str(&fixture).expect("the fixture parses");
    for case in fixture["cases"].as_array().expect("the cases") {
        let name = case["name"].as_str().expect("a name");
        let state = match &case["state"] {
            Value::Null => None,
            Value::String(text) => Some(text.clone()),
            structured => Some(serde_json::to_string_pretty(structured).expect("json")),
        };
        let questions: Vec<SchemaQuestion> = case["questions"]
            .as_array()
            .expect("the questions")
            .iter()
            .map(fixture_question)
            .collect();
        let prompt = lay(&encoder, state.as_deref(), &[], &questions).expect("the case lays out");
        if let Some(whole) = case.get("ids") {
            assert_eq!(prompt.questions[0].branch, ids_of(whole), "{name}");
        } else {
            assert_eq!(prompt.trunk, ids_of(&case["trunk"]), "{name}: trunk");
            for (laid, branch) in prompt
                .questions
                .iter()
                .zip(case["branches"].as_array().expect("branches"))
            {
                assert_eq!(laid.branch, ids_of(branch), "{name}: branch");
            }
        }
        for (laid, groups) in prompt
            .questions
            .iter()
            .zip(case["groups"].as_array().expect("groups"))
        {
            let groups: Vec<Vec<TokenId>> = groups
                .as_array()
                .expect("a question's groups")
                .iter()
                .map(ids_of)
                .collect();
            assert_eq!(laid.options, groups, "{name}: groups");
        }
    }
}
