//! Questions answered from the next-token rows of a decision checkpoint that
//! has no head.
//!
//! Such a checkpoint was trained to answer each question at the end of its
//! own prompt, with one token: the answer is a softmax over the
//! log-probabilities its options' tokens have at that row. Which prompt, which
//! option codes and which spellings of an answer count are the family's
//! business - its [`DecisionReadout`] lays them out - and this module only
//! hands it the tokenizer and the decoded media, sends one scoring pass per
//! question, and folds each option's token group into one score.
//! Nothing here names a family.
//!
//! # Sharing the state between questions
//!
//! Every question's prompt opens with the same trunk - the state and its
//! pictures - so a request of several questions names the trunk as the
//! shared prefix, and the scheduler reads it once and continues it per
//! question from its checkpoint, as it does for the label route's siblings.
//! The checkpoint is the stack's own, recurrent state included, so a hybrid
//! backbone resumes exactly where a whole read would have stood. With media
//! the checkpoint is filed under the pictures' content key; a model that
//! cannot key it keeps the score off the tier, and every question then reads
//! its pictures whole.

use std::sync::Arc;

use serde_json::Value;
use shiso_core::{
    DecisionLayoutError, DecisionMedia, DecisionMediaKind, DecisionReadout, ReadoutPrompt,
    ReadoutRecord, SchemaQuestion, StateSpelling, TokenId,
};
use shiso_ir::Block;

use crate::api::pipeline::remaining_budget;
use crate::decision::core::{self, Answered, InputTally, RawQuestion};
use crate::decision::head::{self, Encoder};
use crate::decision::math::{label_mass, softmax};
use crate::error::AppError;
use crate::scheduler::ScoreRequest;
use crate::state::AppState;

/// The state a readout reads, spelled the way its family wants.
///
/// A missing or `null` state is no state, and so is a blank string: the
/// caller who sent `""` beside a picture meant the picture to be the whole
/// state, and the layout writes no state block for it. Any other string is
/// the caller's text verbatim, untrimmed, because these bytes are scored.
pub(crate) fn spell_state(state: Option<&Value>, spelling: StateSpelling) -> Option<String> {
    match state? {
        Value::Null => None,
        Value::String(text) if text.trim().is_empty() => None,
        Value::String(text) => Some(text.clone()),
        structured => Some(match spelling {
            StateSpelling::CompactSorted => core::render_text(structured),
            // Serialising a parsed value cannot fail; the compact form is the
            // fallback only so the type says so.
            StateSpelling::Indented => {
                serde_json::to_string_pretty(structured).unwrap_or_else(|_| structured.to_string())
            }
        }),
    }
}

/// What a refusal of the readout layout tells the caller.
fn readout_refusal(error: DecisionLayoutError, questions: &[RawQuestion], ctx: u32) -> AppError {
    match error {
        DecisionLayoutError::PromptTooLong { tokens, .. } => AppError::ContextOverflow {
            param: "input",
            prompt_tokens: tokens as u64,
            ctx,
        },
        DecisionLayoutError::Unreadable { question, reason } => AppError::bad_request(
            format!(
                "{}: {reason}",
                core::question_name(questions.get(question).map_or("", |q| q.id.as_str()))
            ),
            Some("questions"),
            "invalid_request",
        ),
        other => head::layout_refusal(other, questions, ctx),
    }
}

/// One question's scoring pass: its whole prompt and the labels read at its
/// last row.
struct Pass {
    tokens: Vec<TokenId>,
    labels: Vec<TokenId>,
}

/// Every question's whole prompt and label list, checked against the context
/// and against the question it answers, and the tokens the passes read: the
/// trunk once and each branch once.
///
/// # Errors
///
/// [`AppError::ContextOverflow`] for a prompt that leaves no room to read an
/// answer, and [`AppError::Internal`] for a layout that answered a question
/// with the wrong number of options or with an option read from nothing.
fn passes(
    ctx: u32,
    questions: &[RawQuestion],
    prompt: &ReadoutPrompt,
) -> Result<(Vec<Pass>, u64), AppError> {
    let mut passes = Vec::with_capacity(questions.len());
    let mut total = prompt.trunk.len() as u64;
    for (question, laid) in questions.iter().zip(&prompt.questions) {
        let mut tokens: Vec<TokenId> = Vec::with_capacity(prompt.trunk.len() + laid.branch.len());
        tokens.extend_from_slice(&prompt.trunk);
        tokens.extend_from_slice(&laid.branch);
        let prompt_tokens = tokens.len() as u64;
        remaining_budget(ctx, prompt_tokens).ok_or(AppError::ContextOverflow {
            param: "input",
            prompt_tokens,
            ctx,
        })?;
        let options = match &question.kind {
            core::Kind::YesNo { .. } => 2,
            core::Kind::Choice { options, .. } => options.len(),
            core::Kind::Score { levels, .. } => levels.len(),
        };
        if laid.options.len() != options || laid.options.iter().any(Vec::is_empty) {
            return Err(AppError::internal(
                "the decision readout laid out a different number of options",
                std::io::Error::other("option count mismatch"),
            ));
        }
        total += laid.branch.len() as u64;
        passes.push(Pass {
            labels: laid.labels(),
            tokens,
        });
    }
    Ok((passes, total))
}

/// Scores `questions` over `text` (`None` for no state) and the pictures in
/// `media` from the model's answer-slot rows.
///
/// Returns the answers in the order of `questions` and the tokens read: the
/// trunk once and every question's own branch, which is what the passes cost
/// when the trunk is shared. The cached count is what a stored prefix covered
/// beyond that sharing - an earlier request's checkpoint - since the
/// siblings' resumes of the trunk are already left out of the input.
///
/// # Errors
///
/// [`AppError::NotReady`] without a vocabulary, or without a renderer when
/// the input carries pictures; the media decoder's refusals; the layout's
/// refusals; and whatever the engine answers with.
pub(crate) async fn run(
    state: &Arc<AppState>,
    readout: &dyn DecisionReadout,
    text: Option<&str>,
    media: Vec<Block>,
    questions: Vec<RawQuestion>,
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

    let schema: Vec<SchemaQuestion> = questions.iter().map(head::schema_question).collect();
    // One row short of the context: the scoring pass reads its answer at the
    // prompt's last row, and the shared boundary wants room after it, as on
    // the label route.
    let budget = (state.ctx as usize).saturating_sub(1);
    let prompt = readout
        .encode(
            &Encoder(vocabulary),
            &ReadoutRecord {
                state: text,
                media: &runs,
                questions: &schema,
                budget,
            },
        )
        .map_err(|error| readout_refusal(error, &questions, state.ctx))?;
    if prompt.questions.len() != questions.len() {
        return Err(AppError::internal(
            "the decision readout laid out a different number of questions",
            std::io::Error::other("question count mismatch"),
        ));
    }

    let (media_items, digests) = match decoded {
        Some(decoded) => (Arc::from(decoded.media), Arc::from(decoded.digests)),
        None => (Arc::from(Vec::new()), Arc::from(Vec::new())),
    };
    let shared = if prompt.questions.len() > 1 {
        u32::try_from(prompt.trunk.len()).unwrap_or(u32::MAX)
    } else {
        0
    };

    let (passes, total) = passes(state.ctx, &questions, &prompt)?;
    let scored = futures::future::join_all(passes.iter().map(|Pass { tokens, labels }| {
        engine.score(
            ScoreRequest::new(Arc::from(tokens.as_slice()), Arc::from(labels.as_slice()))
                .with_media(Arc::clone(&media_items), Arc::clone(&digests))
                .with_shared_prefix(shared),
        )
    }))
    .await;

    let mut answers = Vec::with_capacity(questions.len());
    for (((question, laid), outcome), Pass { tokens, labels }) in questions
        .into_iter()
        .zip(&prompt.questions)
        .zip(scored)
        .zip(passes)
    {
        let outcome = outcome?;
        let pooled = laid.pool(&outcome.label_logprobs);
        let probabilities = softmax(&pooled, temperature);
        // The first argmax, so a tie goes to the earlier option, as on the
        // label route.
        let top = probabilities
            .iter()
            .enumerate()
            .fold((0usize, f64::NEG_INFINITY), |best, (index, p)| {
                if *p > best.1 { (index, *p) } else { best }
            })
            .0;
        answers.push(Answered {
            id: question.id,
            kind: question.kind,
            probability_answer: question.probability_answer,
            probabilities,
            // The best spelling of each option is the mass that option was
            // given; the other spellings of it are not added on top.
            label_mass: label_mass(&pooled),
            top,
            prompt_tokens: tokens.len() as u64,
            cached_tokens: outcome.cached_tokens.min(tokens.len() as u64),
            prompt_token_ids: want_token_ids
                .then(|| tokens.iter().map(|token| token.get()).collect()),
            label_token_ids: want_token_ids
                .then(|| labels.iter().map(|token| token.get()).collect()),
        });
    }
    let tally = shared_tally(&answers, total);
    Ok((answers, tally))
}

/// The input of passes that share a trunk, counted as `total` - the trunk
/// once and every branch - and the part of it a stored prefix covered.
///
/// The siblings' resumes of the trunk are the reads `total` already left
/// out, so they are taken off the passes' own cached counts: what remains came
/// from a checkpoint the input does count, an earlier request's.
fn shared_tally(answers: &[Answered], total: u64) -> InputTally {
    let whole = InputTally::of(answers);
    let deduplicated = whole.input.saturating_sub(total);
    InputTally {
        input: total,
        cached: whole.cached.saturating_sub(deduplicated).min(total),
    }
}

#[cfg(test)]
mod tests {
    use super::spell_state;
    use serde_json::json;
    use shiso_core::StateSpelling;

    #[test]
    fn a_structured_state_is_spelled_the_familys_way() {
        let state = json!({"b": 1, "a": ["é", true]});
        assert_eq!(
            spell_state(Some(&state), StateSpelling::Indented).as_deref(),
            Some("{\n  \"b\": 1,\n  \"a\": [\n    \"é\",\n    true\n  ]\n}")
        );
        assert_eq!(
            spell_state(Some(&state), StateSpelling::CompactSorted).as_deref(),
            Some("{\"a\":[\"é\",true],\"b\":1}")
        );
    }

    #[test]
    fn a_missing_or_blank_state_is_no_state_and_text_is_verbatim() {
        assert_eq!(spell_state(None, StateSpelling::Indented), None);
        assert_eq!(
            spell_state(Some(&json!(null)), StateSpelling::Indented),
            None
        );
        assert_eq!(
            spell_state(Some(&json!("  ")), StateSpelling::Indented),
            None
        );
        assert_eq!(
            spell_state(Some(&json!(" hi\n")), StateSpelling::Indented).as_deref(),
            Some(" hi\n")
        );
    }
}
