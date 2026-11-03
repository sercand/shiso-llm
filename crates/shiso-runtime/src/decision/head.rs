//! Questions answered by the model's schema decision head.
//!
//! A decision checkpoint reads the state and every question in one prompt
//! and scores every option of every question at once, from the backbone's
//! final rows rather than from a label's next-token probability. Which prompt
//! is the family's business: its [`DecisionLayout`] lays the schema out, and
//! this module only hands it the tokenizer and the decoded media, sends the
//! prompt to the engine, and folds the head's option logits back into the
//! answers the label route returns. Nothing here names a family.

use std::ops::Range;
use std::sync::Arc;

use shiso_core::{
    CoreError, DecisionLayout, DecisionLayoutError, DecisionMedia, DecisionMediaKind,
    DecisionRecord, PromptEncoder, SchemaAnswer, SchemaOption, SchemaQuestion, TokenId,
};
use shiso_ir::Block;
use shiso_ir::decision::{Input, Part, Question, QuestionKind};

use crate::decision::core::{self, Answered, InputTally};
use crate::decision::score::{Scored, fold_head};
use crate::error::AppError;
use crate::scheduler::ScoreRequest;
use crate::state::AppState;
use crate::tokenizer::Vocabulary;

/// The vocabulary, as a layout reaches it.
pub(crate) struct Encoder<'a>(pub(crate) &'a Vocabulary);

impl PromptEncoder for Encoder<'_> {
    fn encode(&self, text: &str, literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
        self.0
            .encode_piece(text, literal)
            .map_err(|error| CoreError::Backend {
                backend: "tokenizer",
                op: "encode",
                source: Box::new(error),
            })
    }
}

/// A question as the layout reads it.
pub(crate) fn schema_question(question: &Question) -> SchemaQuestion {
    let answer = match &question.kind {
        QuestionKind::YesNo { yes, no } => SchemaAnswer::YesNo {
            yes: yes.clone(),
            no: no.clone(),
        },
        QuestionKind::Choice(options) => SchemaAnswer::Choice(
            options
                .iter()
                .map(|option| SchemaOption {
                    value: option.value.text(),
                    description: option.description.clone(),
                })
                .collect(),
        ),
        QuestionKind::Score(levels) => SchemaAnswer::Score(
            levels
                .iter()
                .map(|level| SchemaOption {
                    value: level.label.clone(),
                    description: level.description.clone(),
                })
                .collect(),
        ),
    };
    // A blank wording is no wording: the layout decides what stands in.
    let wording = if question.instructions.trim().is_empty() {
        String::new()
    } else {
        question.instructions.clone()
    };
    SchemaQuestion {
        id: question.name.clone(),
        wording,
        answer,
    }
}

/// The state text and the media of a decision input: the string, or the
/// blocks' text in order beside their pictures in order. Where the pictures
/// go in the prompt is the layout's call, not the order the caller
/// interleaved them.
pub(crate) fn split_input(input: &Input) -> (String, Vec<Block>) {
    match input {
        Input::Text(text) => (text.clone(), Vec::new()),
        Input::Blocks(parts) => {
            let mut text = String::new();
            let mut media = Vec::new();
            for part in parts {
                match part {
                    Part::Text(part) => text.push_str(part),
                    Part::Image(_) => media.push(core::part_block(part)),
                }
            }
            (text, media)
        }
        Input::Pictured { media, text } => (
            text.clone(),
            media.iter().cloned().map(Block::Image).collect(),
        ),
    }
}

/// What a refusal of the layout tells the caller.
pub(crate) fn layout_refusal(
    error: DecisionLayoutError,
    questions: &[Question],
    ctx: u32,
) -> AppError {
    match error {
        DecisionLayoutError::SchemaTooLong { tokens, .. } => AppError::ContextOverflow {
            param: "questions",
            prompt_tokens: tokens as u64,
            ctx,
        },
        DecisionLayoutError::AmbiguousOption { question, spelling } => AppError::bad_request(
            format!(
                "{}: the decision head cannot score two options spelled {spelling:?}",
                core::question_name(questions.get(question).map_or("", |q| q.name.as_str()))
            ),
            Some("questions"),
            "invalid_request",
        ),
        other => AppError::internal("the decision prompt could not be laid out", other),
    }
}

/// Scores `questions` over the state `text` and the pictures in `media`
/// with the model's decision head.
///
/// Returns the answers in the order of `questions` and the prompt's length,
/// none of it cached: a decision pass reads every row itself.
///
/// # Errors
///
/// [`AppError::NotReady`] without a vocabulary, or without a renderer when
/// the input carries pictures; the media decoder's refusals; the layout's
/// refusals; and whatever the engine answers with.
pub(crate) async fn run(
    state: &Arc<AppState>,
    layout: &dyn DecisionLayout,
    text: &str,
    media: Vec<Block>,
    questions: Vec<Question>,
    temperature: f64,
    want_token_ids: bool,
) -> Result<(Vec<Answered>, InputTally), AppError> {
    let engine = state.engine()?;
    let vocabulary = state.vocabulary.as_ref().ok_or(AppError::NotReady {
        what: "a vocabulary",
    })?;
    let decoded = if media.is_empty() {
        None
    } else {
        let renderer = state.renderer.as_ref().ok_or(AppError::NotReady {
            what: "a prompt renderer",
        })?;
        Some(renderer.decode_media(media)?)
    };
    // Every block handed to the decoder is a picture.
    let runs: Vec<DecisionMedia> = decoded
        .iter()
        .flat_map(|decoded| &decoded.pad_counts)
        .map(|pads| DecisionMedia {
            kind: DecisionMediaKind::Image,
            pads: *pads,
        })
        .collect();

    let schema: Vec<SchemaQuestion> = questions.iter().map(schema_question).collect();
    let prompt = layout
        .encode(
            &Encoder(vocabulary),
            &DecisionRecord {
                state: text,
                media: &runs,
                questions: &schema,
                budget: state.ctx as usize,
            },
        )
        .map_err(|error| layout_refusal(error, &questions, state.ctx))?;
    let prompt_tokens = prompt.tokens.len() as u64;
    let prompt_ids: Option<Vec<u32>> =
        want_token_ids.then(|| prompt.tokens.iter().map(|token| token.get()).collect());
    let mut request = ScoreRequest::decision(Arc::from(prompt.tokens), Arc::from(prompt.questions));
    if let Some(decoded) = decoded {
        request = request.with_media(Arc::from(decoded.media), Arc::from(decoded.digests));
    }
    let outcome = engine.score(request).await?;
    if outcome.option_logits.len() != questions.len() {
        return Err(AppError::internal(
            "the decision head answered a different number of questions",
            std::io::Error::other("question count mismatch"),
        ));
    }
    let folded =
        fold_head(&outcome.option_logits, &prompt.order, temperature).map_err(|error| {
            AppError::internal(
                "the decision head answered something other than what was asked",
                error,
            )
        })?;
    let answers = questions
        .into_iter()
        .zip(folded)
        .map(|(question, scored)| answer(question, scored, prompt_ids.clone(), want_token_ids))
        .collect();
    Ok((
        answers,
        InputTally {
            input: prompt_tokens,
            cached: 0,
        },
    ))
}

/// One question's answer from the head's folded probabilities, already in the
/// caller's option order.
fn answer(
    question: Question,
    scored: Scored,
    prompt_token_ids: Option<Vec<u32>>,
    want_token_ids: bool,
) -> Answered {
    Answered {
        question,
        probabilities: scored.probabilities,
        // The head's options are the whole distribution.
        label_mass: scored.label_mass,
        top: scored.top,
        prompt_tokens: 0,
        // A decision pass reads every row of its prompt itself.
        cached_tokens: 0,
        prompt_token_ids,
        // The head scores options, not label tokens.
        label_token_ids: want_token_ids.then(Vec::new),
    }
}

#[cfg(test)]
mod tests {
    use super::{schema_question, split_input};
    use shiso_core::SchemaAnswer;
    use shiso_ir::decision::{Choice, ChoiceValue, Input, Part, Question, QuestionKind};
    use shiso_ir::{Block, MediaSource};

    fn choice() -> Question {
        Question {
            name: "team".to_owned(),
            instructions: "  ".to_owned(),
            kind: QuestionKind::Choice(
                ["tech", "billing"]
                    .iter()
                    .map(|name| Choice {
                        value: ChoiceValue::Str((*name).to_owned()),
                        description: None,
                    })
                    .collect(),
            ),
            probability_answer: false,
        }
    }

    #[test]
    fn a_blank_wording_reaches_the_layout_as_none() {
        let schema = schema_question(&choice());
        assert!(schema.wording.is_empty());
        assert!(matches!(schema.answer, SchemaAnswer::Choice(ref options) if options.len() == 2));
    }

    #[test]
    fn pictures_are_taken_out_of_the_state_text() {
        let input = Input::Blocks(vec![
            Part::Text("a ".to_owned()),
            Part::Image(MediaSource::Remote("https://example.com/a.png".to_owned())),
            Part::Text("b".to_owned()),
        ]);
        let (text, media) = split_input(&input);
        assert_eq!(text, "a b");
        assert!(matches!(media.as_slice(), [Block::Image(_)]));
    }

    #[test]
    fn pictures_ahead_of_a_state_keep_the_state_as_sent() {
        let input = Input::Pictured {
            media: vec![MediaSource::Remote("https://example.com/a.png".to_owned())],
            text: "  state\n".to_owned(),
        };
        let (text, media) = split_input(&input);
        assert_eq!(text, "  state\n");
        assert!(matches!(media.as_slice(), [Block::Image(_)]));
    }
}
