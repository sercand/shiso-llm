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

use crate::decision::core::{self, Answered, Kind, RawQuestion};
use crate::decision::decisions_wire::{DecisionInput, InputBlock};
use crate::decision::math::softmax;
use crate::error::AppError;
use crate::scheduler::ScoreRequest;
use crate::state::AppState;
use crate::tokenizer::Vocabulary;

/// The vocabulary, as a layout reaches it.
pub(crate) struct Encoder<'a>(pub(crate) &'a Vocabulary);

impl PromptEncoder for Encoder<'_> {
    fn encode(&self, text: &str, literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
        let failed = |error: crate::tokenizer::TokenizerError| CoreError::Backend {
            backend: "tokenizer",
            op: "encode",
            source: Box::new(error),
        };
        // Every marker this vocabulary has opens with `<`, so a piece whose
        // caller text holds none takes the plain encode: the literal-aware
        // one rebuilds the tokenizer on every call, and a schema is dozens of
        // pieces.
        let marked = literal
            .iter()
            .any(|range| text.get(range.clone()).is_some_and(|t| t.contains('<')));
        if !marked {
            return self.0.encode(text, false).map_err(failed);
        }
        let mut tokens = self
            .0
            .encode_prompt_with_literals(text, literal)
            .map_err(failed)?;
        // A piece sits inside a prompt; only the prompt's own start takes the
        // sequence opener.
        if self.0.bos().is_some() && tokens.first() == self.0.bos().as_ref() {
            tokens.remove(0);
        }
        Ok(tokens)
    }
}

/// A question as the layout reads it.
pub(crate) fn schema_question(question: &RawQuestion) -> SchemaQuestion {
    let (wording, answer) = match &question.kind {
        Kind::YesNo { question, yes, no } => (
            question,
            SchemaAnswer::YesNo {
                yes: yes.clone(),
                no: no.clone(),
            },
        ),
        Kind::Choice { question, options } => (
            question,
            SchemaAnswer::Choice(
                options
                    .iter()
                    .map(|option| SchemaOption {
                        value: option.value.text(),
                        description: option.description.clone(),
                    })
                    .collect(),
            ),
        ),
        Kind::Score { question, levels } => (
            question,
            SchemaAnswer::Score(
                levels
                    .iter()
                    .map(|level| SchemaOption {
                        value: level.label.clone(),
                        description: level.description.clone(),
                    })
                    .collect(),
            ),
        ),
    };
    // A blank wording is no wording: the layout decides what stands in.
    let wording = if wording.trim().is_empty() {
        String::new()
    } else {
        wording.clone()
    };
    SchemaQuestion {
        id: question.id.clone(),
        wording,
        answer,
    }
}

/// The state text and the media of a decisions input: the string, or the
/// blocks' text in order beside their pictures in order. Where the pictures
/// go in the prompt is the layout's call, not the order the caller
/// interleaved them.
pub(crate) fn split_input(input: &DecisionInput) -> (String, Vec<Block>) {
    match input {
        DecisionInput::Text(text) => (text.clone(), Vec::new()),
        DecisionInput::Blocks(blocks) => {
            let mut text = String::new();
            let mut media = Vec::new();
            for block in blocks {
                match block {
                    InputBlock::Text { text: part } => text.push_str(part),
                    InputBlock::Image(source) => media.push(core::image_block(source)),
                }
            }
            (text, media)
        }
    }
}

/// What a refusal of the layout tells the caller.
pub(crate) fn layout_refusal(
    error: DecisionLayoutError,
    questions: &[RawQuestion],
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
                core::question_name(questions.get(question).map_or("", |q| q.id.as_str()))
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
/// Returns the answers in the order of `questions` and the prompt's length.
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
    questions: Vec<RawQuestion>,
    temperature: f64,
    want_token_ids: bool,
) -> Result<(Vec<Answered>, u64), AppError> {
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
    questions
        .into_iter()
        .zip(outcome.option_logits)
        .zip(prompt.order)
        .map(|((question, logits), order)| {
            if logits.len() != order.len() {
                return Err(AppError::internal(
                    "the decision head answered a different number of options",
                    std::io::Error::other("option count mismatch"),
                ));
            }
            let probabilities = softmax(
                &logits.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                temperature,
            );
            Ok(answer(
                question,
                &probabilities,
                &order,
                prompt_ids.clone(),
                want_token_ids,
            ))
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|answers| (answers, prompt_tokens))
}

/// One question's answer from the head's probabilities, put back in the
/// caller's option order.
fn answer(
    question: RawQuestion,
    probabilities: &[f64],
    order: &[usize],
    prompt_token_ids: Option<Vec<u32>>,
    want_token_ids: bool,
) -> Answered {
    let mut ordered = vec![0.0f64; probabilities.len()];
    for (head, caller) in order.iter().enumerate() {
        if let Some(slot) = ordered.get_mut(*caller) {
            *slot = probabilities[head];
        }
    }
    // The first argmax, so a tie goes to the earlier option, as on the label
    // route.
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
        prompt_token_ids,
        // The head scores options, not label tokens.
        label_token_ids: want_token_ids.then(Vec::new),
    }
}

#[cfg(test)]
mod tests {
    use super::{answer, schema_question, split_input};
    use crate::decision::core::{ChoiceOption, ChoiceValue, Kind, RawQuestion};
    use crate::decision::decisions_wire::{DecisionInput, InputBlock, InputImageSource};
    use shiso_core::SchemaAnswer;
    use shiso_ir::Block;

    fn choice() -> RawQuestion {
        RawQuestion {
            id: "team".to_owned(),
            kind: Kind::Choice {
                question: "  ".to_owned(),
                options: ["tech", "billing"]
                    .iter()
                    .map(|name| ChoiceOption {
                        value: ChoiceValue::Str((*name).to_owned()),
                        description: None,
                    })
                    .collect(),
            },
            probability_answer: false,
        }
    }

    #[test]
    fn answers_come_back_in_the_callers_order() {
        // The head read `billing` first; the caller listed it second.
        let answered = answer(choice(), &[0.75, 0.25], &[1, 0], None, false);
        assert_eq!(answered.probabilities, vec![0.25, 0.75]);
        assert_eq!(answered.top, 1);
    }

    #[test]
    fn a_blank_wording_reaches_the_layout_as_none() {
        let schema = schema_question(&choice());
        assert!(schema.wording.is_empty());
        assert!(matches!(schema.answer, SchemaAnswer::Choice(ref options) if options.len() == 2));
    }

    #[test]
    fn pictures_are_taken_out_of_the_state_text() {
        let input = DecisionInput::Blocks(vec![
            InputBlock::Text {
                text: "a ".to_owned(),
            },
            InputBlock::Image(InputImageSource::Url {
                url: "https://example.com/a.png".to_owned(),
            }),
            InputBlock::Text {
                text: "b".to_owned(),
            },
        ]);
        let (text, media) = split_input(&input);
        assert_eq!(text, "a b");
        assert!(matches!(media.as_slice(), [Block::Image(_)]));
    }
}
