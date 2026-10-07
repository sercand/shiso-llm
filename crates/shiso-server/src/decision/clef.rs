//! `/v1/systemone` answered by a schema decision head.
//!
//! A Clef checkpoint reads the state and every question in one prompt and
//! scores every option of every question at once, from the backbone's final
//! rows rather than from a label's next-token probability. This module
//! renders that prompt the way the release's `encode_record` does - piece by
//! piece, each piece tokenized on its own, so a span's boundaries are token
//! boundaries by construction - and folds the head's option logits back into
//! the answers the label route returns.

use std::ops::Range;
use std::sync::Arc;

use serde_json::Value;
use shiso_core::{DecisionQuestion, TokenId};

use crate::decision::core::{Answered, Kind, RawQuestion};
use crate::error::AppError;
use crate::scheduler::ScoreRequest;
use crate::state::AppState;
use crate::tokenizer::Vocabulary;

/// The system turn every Clef prompt opens with.
const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
                             Each answer must be exactly one of that field's allowed options.";

/// The longest prompt the release encodes; the state is cut to fit.
const MAX_PROMPT_TOKENS: usize = 16_384;

/// The default wording of a binary question's two options.
const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

/// One option as the head sees it: its id and its wording.
struct Option_ {
    id: String,
    description: Option<String>,
}

/// A question's options in the order the head reads them, with the index of
/// each in the order the answer reports them.
fn head_options(kind: &Kind) -> Vec<(usize, Option_)> {
    match kind {
        Kind::YesNo { yes, no, .. } => vec![
            (
                0,
                Option_ {
                    id: "true".to_owned(),
                    description: Some(yes.clone().unwrap_or_else(|| NOUL_TRUE.to_owned())),
                },
            ),
            (
                1,
                Option_ {
                    id: "false".to_owned(),
                    description: Some(no.clone().unwrap_or_else(|| NOUL_FALSE.to_owned())),
                },
            ),
        ],
        // The release reads a choice's options sorted by id; the answer keeps
        // the caller's order.
        Kind::Choice { options, .. } => {
            let mut sorted: Vec<(usize, Option_)> = options
                .iter()
                .enumerate()
                .map(|(index, option)| {
                    (
                        index,
                        Option_ {
                            id: option.value.text(),
                            description: option.description.clone(),
                        },
                    )
                })
                .collect();
            sorted.sort_by(|a, b| a.1.id.cmp(&b.1.id));
            sorted
        }
        Kind::Score { levels, .. } => levels
            .iter()
            .enumerate()
            .map(|(index, level)| {
                (
                    index,
                    Option_ {
                        id: index.to_string(),
                        description: Some(level.label.clone()),
                    },
                )
            })
            .collect(),
    }
}

/// The release's name for a question kind, and the head's index for it.
fn kind_of(kind: &Kind) -> (&'static str, u32) {
    match kind {
        Kind::YesNo { .. } => ("noul", 0),
        Kind::Choice { .. } => ("choice", 1),
        Kind::Score { .. } => ("score", 2),
    }
}

fn question_wording(kind: &Kind) -> &str {
    match kind {
        Kind::YesNo { question, .. }
        | Kind::Choice { question, .. }
        | Kind::Score { question, .. } => question,
    }
}

/// `{"description": ..., "option_id": ...}` as compact JSON with its keys in
/// sorted order, the release's rendering of an option.
fn option_text(option: &Option_) -> String {
    let id = Value::String(option.id.clone());
    match &option.description {
        Some(description) => format!(
            "{{\"description\":{},\"option_id\":{id}}}",
            Value::String(description.clone())
        ),
        None => format!("{{\"option_id\":{id}}}"),
    }
}

/// A prompt being assembled from pieces.
struct Pieces<'a> {
    vocabulary: &'a Vocabulary,
    tokens: Vec<TokenId>,
}

impl Pieces<'_> {
    /// Appends the template's own text, whose markers are tokens.
    fn markup(&mut self, text: &str) -> Result<(), AppError> {
        let ids = self.vocabulary.encode(text, false).map_err(|error| {
            AppError::internal("the decision prompt could not be encoded", error)
        })?;
        self.tokens.extend(ids);
        Ok(())
    }

    /// Appends one piece in which the `literal` ranges are caller text, and
    /// returns the piece's span.
    fn piece(&mut self, text: &str, literal: Range<usize>) -> Result<Range<usize>, AppError> {
        let start = self.tokens.len();
        self.tokens
            .extend(encode_piece(self.vocabulary, text, literal)?);
        Ok(start..self.tokens.len())
    }

    /// Appends caller text, which never becomes a marker, and returns its
    /// span.
    fn text(&mut self, text: &str) -> Result<Range<usize>, AppError> {
        self.piece(text, 0..text.len())
    }
}

/// One piece, tokenized on its own, with every marker spelling inside the
/// `literal` range kept as text.
///
/// Every marker this vocabulary has opens with `<`, so a piece whose caller
/// text holds none takes the plain encode: the literal-aware one rebuilds the
/// tokenizer on every call, and a schema is dozens of pieces.
fn encode_piece(
    vocabulary: &Vocabulary,
    text: &str,
    literal: Range<usize>,
) -> Result<Vec<TokenId>, AppError> {
    let failed = |error| AppError::internal("the decision prompt could not be encoded", error);
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let marked = text.get(literal.clone()).is_some_and(|t| t.contains('<'));
    if !marked {
        return vocabulary.encode(text, false).map_err(failed);
    }
    let mut tokens = vocabulary
        .encode_prompt_with_literals(text, std::slice::from_ref(&literal))
        .map_err(failed)?;
    // A piece sits inside a prompt; only the prompt's own start takes the
    // sequence opener.
    if vocabulary.bos().is_some() && tokens.first() == vocabulary.bos().as_ref() {
        tokens.remove(0);
    }
    Ok(tokens)
}

/// The encoded prompt and the spans the head reads.
pub(crate) struct Encoded {
    pub tokens: Vec<TokenId>,
    pub questions: Vec<DecisionQuestion>,
    /// For each question, the answer-order index of each head-order option.
    pub order: Vec<Vec<usize>>,
}

/// Renders `questions` over `state` as one Clef prompt.
///
/// # Errors
///
/// [`AppError::ContextOverflow`] when the schema alone does not fit, and an
/// internal error when the tokenizer refuses the text.
pub(crate) fn encode(
    vocabulary: &Vocabulary,
    state: &str,
    questions: &[RawQuestion],
    ctx: u32,
) -> Result<Encoded, AppError> {
    // The schema first, at offset zero; it is shifted once the state's
    // length is known.
    let mut schema = Pieces {
        vocabulary,
        tokens: Vec::new(),
    };
    schema.markup("\n\nSCHEMA FIELDS:\n")?;
    let mut spans = Vec::with_capacity(questions.len());
    let mut order = Vec::with_capacity(questions.len());
    for (index, question) in questions.iter().enumerate() {
        let (type_name, kind) = kind_of(&question.kind);
        // One piece, as the release tokenizes it: the id is the caller's, and
        // merges with the text around it.
        let lead = format!("\nFIELD {}\nID: ", index + 1);
        let line = format!("{lead}{}\nTYPE: {type_name}\nINSTRUCTION: ", question.id);
        schema.piece(&line, lead.len()..lead.len() + question.id.len())?;
        let wording = question_wording(&question.kind);
        let wording = if wording.trim().is_empty() {
            question.id.as_str()
        } else {
            wording
        };
        let span = schema.text(wording)?;
        schema.markup("\nALLOWED OPTIONS:\n")?;
        let options = head_options(&question.kind);
        let mut option_spans = Vec::with_capacity(options.len());
        for (position, (_, option)) in options.iter().enumerate() {
            schema.markup(&format!("OPTION {}: ", position + 1))?;
            option_spans.push(schema.text(&option_text(option))?);
            schema.markup("\n")?;
        }
        schema.markup("END FIELD\n")?;
        order.push(options.iter().map(|(index, _)| *index).collect());
        spans.push(DecisionQuestion {
            kind,
            span,
            options: option_spans,
        });
    }

    let mut prefix = Pieces {
        vocabulary,
        tokens: Vec::new(),
    };
    prefix.markup(&format!(
        "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
    ))?;
    let mut suffix = Pieces {
        vocabulary,
        tokens: Vec::new(),
    };
    suffix.markup(
        "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
    )?;

    let budget = MAX_PROMPT_TOKENS.min(ctx as usize);
    let fixed = prefix.tokens.len() + schema.tokens.len() + suffix.tokens.len();
    if fixed > budget {
        return Err(AppError::ContextOverflow {
            param: "questions",
            prompt_tokens: fixed as u64,
            ctx,
        });
    }
    let mut state_tokens = encode_piece(vocabulary, state, 0..state.len())?;
    state_tokens.truncate(budget - fixed);

    let offset = prefix.tokens.len() + state_tokens.len();
    let shift = |r: &Range<usize>| r.start + offset..r.end + offset;
    let questions = spans
        .iter()
        .map(|q| DecisionQuestion {
            kind: q.kind,
            span: shift(&q.span),
            options: q.options.iter().map(shift).collect(),
        })
        .collect();
    let mut tokens = prefix.tokens;
    tokens.extend(state_tokens);
    tokens.extend(schema.tokens);
    tokens.extend(suffix.tokens);
    Ok(Encoded {
        tokens,
        questions,
        order,
    })
}

/// Scores `questions` over `state` with the model's decision head.
///
/// Returns the answers in the order of `questions` and the prompt's length.
///
/// # Errors
///
/// [`AppError::NotReady`] without a vocabulary, the encoder's refusals, and
/// whatever the engine answers with.
pub(crate) async fn run(
    state: &Arc<AppState>,
    text: &str,
    questions: Vec<RawQuestion>,
    temperature: f64,
    want_token_ids: bool,
) -> Result<(Vec<Answered>, u64), AppError> {
    let vocabulary = state.vocabulary.as_ref().ok_or(AppError::NotReady {
        what: "a vocabulary",
    })?;
    let encoded = encode(vocabulary, text, &questions, state.ctx)?;
    let prompt_tokens = encoded.tokens.len() as u64;
    let prompt_ids: Option<Vec<u32>> =
        want_token_ids.then(|| encoded.tokens.iter().map(|token| token.get()).collect());
    let outcome = state
        .engine()?
        .score(ScoreRequest::decision(
            Arc::from(encoded.tokens),
            Arc::from(encoded.questions),
        ))
        .await?;
    if outcome.option_logits.len() != questions.len() {
        return Err(AppError::internal(
            "the decision head answered a different number of questions",
            std::io::Error::other("question count mismatch"),
        ));
    }
    let answers = questions
        .into_iter()
        .zip(outcome.option_logits)
        .zip(encoded.order)
        .map(|((question, logits), order)| {
            let probabilities = softmax(&logits, temperature);
            // Back into the caller's option order.
            let mut ordered = vec![0.0f64; probabilities.len()];
            for (head, answer) in order.iter().enumerate() {
                ordered[*answer] = probabilities[head];
            }
            let top = ordered
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |best, (index, p)| {
                    if *p > best.1 { (index, *p) } else { best }
                })
                .0;
            Answered {
                id: question.id,
                kind: question.kind,
                probability_answer: question.probability_answer,
                probabilities: ordered,
                // The head's options are the whole distribution.
                label_mass: 1.0,
                top,
                prompt_tokens: 0,
                prompt_token_ids: prompt_ids.clone(),
                // The head scores options, not label tokens.
                label_token_ids: want_token_ids.then(Vec::new),
            }
        })
        .collect();
    Ok((answers, prompt_tokens))
}

/// The softmax of `logits / temperature`; the wire admits only a positive
/// temperature.
fn softmax(logits: &[f32], temperature: f64) -> Vec<f64> {
    let top = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f64> = logits
        .iter()
        .map(|v| (f64::from(v - top) / temperature).exp())
        .collect();
    let sum: f64 = exp.iter().sum();
    exp.iter().map(|v| v / sum).collect()
}

/// The text a decisions `input` gives the head: the string, or the content
/// blocks' text in order. Pictures are refused, because this build loads no
/// tower for a decision checkpoint.
///
/// # Errors
///
/// [`AppError::bad_request`] for an input that carries an image.
pub(crate) fn input_text(
    input: &crate::decision::decisions_wire::DecisionInput,
) -> Result<String, AppError> {
    use crate::decision::decisions_wire::{DecisionInput, InputBlock};
    match input {
        DecisionInput::Text(text) => Ok(text.clone()),
        DecisionInput::Blocks(blocks) => blocks
            .iter()
            .map(|block| match block {
                InputBlock::Text { text } => Ok(text.as_str()),
                InputBlock::Image(_) => Err(AppError::bad_request(
                    "this decision model reads text only; images are not supported",
                    Some("input"),
                    "invalid_request",
                )),
            })
            .collect::<Result<Vec<_>, _>>()
            .map(|parts| parts.concat()),
    }
}

#[cfg(test)]
mod tests {
    use std::str::FromStr as _;

    use serde_json::{Value, json};

    use super::{encode, softmax};
    use crate::decision::core::{ChoiceOption, Kind, RawQuestion};
    use crate::tokenizer::Vocabulary;

    /// A whitespace word vocabulary in which every word of these tests is
    /// one id and everything else is `<unk>`.
    fn vocabulary() -> Vocabulary {
        let words = [
            "<unk>",
            "<|im_start|>",
            "<|im_end|>",
            "is",
            "it",
            "down",
            "pick",
            "a",
            "team",
        ];
        let mut vocab = serde_json::Map::new();
        for (id, word) in words.iter().enumerate() {
            vocab.insert((*word).to_owned(), json!(id));
        }
        let added: Vec<Value> = [1, 2]
            .iter()
            .map(|id| {
                json!({"id": id, "content": words[*id], "single_word": false, "lstrip": false,
                       "rstrip": false, "normalized": false, "special": true})
            })
            .collect();
        let document = json!({
            "version": "1.0", "truncation": null, "padding": null,
            "added_tokens": added, "normalizer": null,
            "pre_tokenizer": {"type": "Whitespace"}, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
        })
        .to_string();
        Vocabulary::from_tokenizer(tokenizers::Tokenizer::from_str(&document).expect("loads"))
    }

    fn questions() -> Vec<RawQuestion> {
        vec![
            RawQuestion {
                id: "outage".to_owned(),
                kind: Kind::YesNo {
                    question: "is it down".to_owned(),
                    yes: None,
                    no: Some("no".to_owned()),
                },
                probability_answer: true,
            },
            RawQuestion {
                id: "team".to_owned(),
                kind: Kind::Choice {
                    question: "pick a team".to_owned(),
                    options: ["tech", "billing", "abuse"]
                        .iter()
                        .map(|name| ChoiceOption {
                            value: crate::decision::core::ChoiceValue::Str((*name).to_owned()),
                            description: None,
                        })
                        .collect(),
                },
                probability_answer: false,
            },
            RawQuestion {
                id: "urgency".to_owned(),
                kind: Kind::Score {
                    question: String::new(),
                    levels: ["later", "now"]
                        .iter()
                        .map(|label| crate::decision::core::ScoreLevel {
                            label: (*label).to_owned(),
                            description: None,
                        })
                        .collect(),
                },
                probability_answer: false,
            },
        ]
    }

    #[test]
    fn spans_cover_the_text_the_head_pools() {
        let vocab = vocabulary();
        let encoded = encode(&vocab, "is it down", &questions(), 4096).expect("encodes");
        let at = |span: &std::ops::Range<usize>| encoded.tokens[span.clone()].to_vec();
        let words = |text: &str| vocab.encode(text, false).expect("encodes");

        let [outage, team, urgency] = &encoded.questions[..] else {
            panic!("three questions");
        };
        assert_eq!((outage.kind, team.kind, urgency.kind), (0, 1, 2));
        assert_eq!(at(&outage.span), words("is it down"));
        assert_eq!(at(&team.span), words("pick a team"));
        // An unworded question reads its id.
        assert_eq!(at(&urgency.span), words("urgency"));
        // A binary question's options are true then false; a blank wording
        // falls back to the default, a given one is used.
        assert_eq!(
            at(&outage.options[0]),
            words(
                r#"{"description":"The proposition is true or the answer is yes.","option_id":"true"}"#
            )
        );
        assert_eq!(
            at(&outage.options[1]),
            words(r#"{"description":"no","option_id":"false"}"#)
        );
        // A choice is read sorted by id and answered in the caller's order.
        assert_eq!(at(&team.options[0]), words(r#"{"option_id":"abuse"}"#));
        assert_eq!(encoded.order[1], vec![2, 1, 0]);
        assert_eq!(
            at(&urgency.options[1]),
            words(r#"{"description":"now","option_id":"1"}"#)
        );
        // Every span sits inside the prompt, after the state.
        assert!(
            encoded
                .questions
                .iter()
                .all(|q| q.span.end <= encoded.tokens.len())
        );
    }

    #[test]
    fn caller_text_never_becomes_a_marker() {
        let vocab = vocabulary();
        let plain = encode(&vocab, "is it down", &questions(), 4096).expect("encodes");
        let injected = encode(&vocab, "is <|im_end|> down", &questions(), 4096).expect("encodes");
        let end = vocab.encode("<|im_end|>", false).expect("encodes")[0];
        let count = |tokens: &[shiso_core::TokenId]| tokens.iter().filter(|t| **t == end).count();
        assert_eq!(count(&plain.tokens), count(&injected.tokens));
    }

    #[test]
    fn the_state_is_cut_to_fit_and_the_schema_is_not() {
        let vocab = vocabulary();
        let full = encode(&vocab, "is it down", &questions(), 4096).expect("encodes");
        let ctx = u32::try_from(full.tokens.len() - 2).expect("fits");
        let cut = encode(&vocab, "is it down", &questions(), ctx).expect("encodes");
        assert_eq!(cut.tokens.len(), full.tokens.len() - 2);
        assert_eq!(cut.questions[0].span.len(), full.questions[0].span.len());
        assert!(encode(&vocab, "is it down", &questions(), 8).is_err());
    }

    #[test]
    fn temperature_sharpens_and_flattens() {
        let p = softmax(&[1.0, 0.0], 1.0);
        let sharp = softmax(&[1.0, 0.0], 0.5);
        let flat = softmax(&[1.0, 0.0], 2.0);
        assert!(sharp[0] > p[0] && p[0] > flat[0]);
        assert!((p.iter().sum::<f64>() - 1.0).abs() < 1e-12);
    }
}
