//! The pure steps of scoring a decision from a model's readout rows or its
//! decision head.
//!
//! Whoever runs the model is the scheduler, behind an async call, and it must
//! keep submitting every question's pass at once so it sees the siblings
//! together, reuses the shared-prefix checkpoint and queues them as it always
//! has. So this module never touches a device: [`readout_runs`] and
//! [`fold_readout`] wrap the engine call, and [`reorder`] and [`first_argmax`]
//! put a head's answer in the caller's order.

use shiso_core::{ReadoutPrompt, SchemaAnswer, SchemaQuestion, TokenId};
use thiserror::Error;

use crate::decision::math::{label_mass, softmax};

/// Why a scoring step could not be taken.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum ScoreError {
    /// The layout laid out something other than what was asked.
    #[error("the decision prompt could not be laid out: {0}")]
    Layout(String),
    /// Whoever ran the model answered something other than what was asked.
    #[error("the engine failed: {0}")]
    Engine(String),
}

/// One scoring pass: read `tokens` once and report the log-probability of each
/// of `labels` at the last row.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScoreRun {
    /// The whole prompt, trunk and branch.
    pub tokens: Vec<TokenId>,
    /// The tokens to read at the prompt's last row.
    pub labels: Vec<TokenId>,
    /// How many leading tokens this run shares with its sibling runs, `0` for
    /// a single run: the runner may checkpoint there and resume each sibling
    /// from it.
    pub shared_prefix: u32,
}

/// The answer to one [`ScoreRun`].
#[derive(Debug, Clone, PartialEq)]
pub struct RunScore {
    /// The log-probability of each label of the run, in the run's order.
    pub label_logprobs: Vec<f64>,
    /// Prompt tokens a stored prefix covered rather than this pass reading
    /// them.
    pub cached_tokens: u64,
}

/// One question's scored answer before it is named: probabilities in the
/// caller's option order.
#[derive(Debug, Clone, PartialEq)]
pub struct Scored {
    /// The softmax over the options, in the caller's order.
    pub probabilities: Vec<f64>,
    /// The mass the options carry together; `1.0` for a head.
    pub label_mass: f64,
    /// The first argmax of `probabilities`.
    pub top: usize,
    /// Prompt tokens a stored prefix covered, never more than the prompt.
    pub cached_tokens: u64,
}

/// The first argmax, so a tie goes to the earlier option.
#[must_use]
pub fn first_argmax(probabilities: &[f64]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .fold((0usize, f64::NEG_INFINITY), |best, (index, p)| {
            if *p > best.1 { (index, *p) } else { best }
        })
        .0
}

/// Puts a head's probabilities, in its own option order, back in the caller's:
/// `order[h]` is where the head's option `h` sits in the caller's list.
#[must_use]
pub fn reorder(probabilities: &[f64], order: &[usize]) -> Vec<f64> {
    let mut ordered = vec![0.0f64; probabilities.len()];
    for (head, caller) in order.iter().enumerate() {
        if let (Some(slot), Some(p)) = (ordered.get_mut(*caller), probabilities.get(head)) {
            *slot = *p;
        }
    }
    ordered
}

/// How many options a question allows.
fn option_count(question: &SchemaQuestion) -> usize {
    match &question.answer {
        SchemaAnswer::YesNo { .. } => 2,
        SchemaAnswer::Choice(options) | SchemaAnswer::Score(options) => options.len(),
    }
}

/// One scoring run per question of `prompt`: the trunk then the question's
/// branch, reading every spelling of every option.
///
/// With more than one question the trunk is the shared prefix, so the runner
/// reads it once and continues each question from its checkpoint.
///
/// # Errors
///
/// [`ScoreError::Layout`] when the layout laid out a different number of
/// questions than `schema` holds, or a different number of options for one, or
/// an option read from no token.
pub fn readout_runs(
    prompt: &ReadoutPrompt,
    schema: &[SchemaQuestion],
) -> Result<Vec<ScoreRun>, ScoreError> {
    if prompt.questions.len() != schema.len() {
        return Err(ScoreError::Layout(
            "the readout laid out a different number of questions".to_owned(),
        ));
    }
    let shared = if prompt.questions.len() > 1 {
        u32::try_from(prompt.trunk.len()).unwrap_or(u32::MAX)
    } else {
        0
    };
    schema
        .iter()
        .zip(&prompt.questions)
        .map(|(question, laid)| {
            if laid.options.len() != option_count(question)
                || laid.options.iter().any(Vec::is_empty)
            {
                return Err(ScoreError::Layout(
                    "the readout laid out a different number of options".to_owned(),
                ));
            }
            let mut tokens = Vec::with_capacity(prompt.trunk.len() + laid.branch.len());
            tokens.extend_from_slice(&prompt.trunk);
            tokens.extend_from_slice(&laid.branch);
            Ok(ScoreRun {
                tokens,
                labels: laid.labels(),
                shared_prefix: shared,
            })
        })
        .collect()
}

/// Folds each run's label log-probabilities into one [`Scored`] per question:
/// the best spelling of each option, softmaxed at `temperature`, with the mass
/// those best spellings carry.
///
/// # Errors
///
/// [`ScoreError::Engine`] when `scores` does not hold one entry per question.
pub fn fold_readout(
    prompt: &ReadoutPrompt,
    runs: &[ScoreRun],
    scores: &[RunScore],
    temperature: f64,
) -> Result<Vec<Scored>, ScoreError> {
    if scores.len() != prompt.questions.len() || runs.len() != scores.len() {
        return Err(ScoreError::Engine(
            "the scorer answered a different number of runs".to_owned(),
        ));
    }
    Ok(prompt
        .questions
        .iter()
        .zip(runs)
        .zip(scores)
        .map(|((laid, run), score)| {
            let pooled = laid.pool(&score.label_logprobs);
            let probabilities = softmax(&pooled, temperature);
            Scored {
                top: first_argmax(&probabilities),
                // The best spelling of each option is the mass that option was
                // given; the other spellings of it are not added on top.
                label_mass: label_mass(&pooled),
                cached_tokens: score
                    .cached_tokens
                    .min(u64::try_from(run.tokens.len()).unwrap_or(u64::MAX)),
                probabilities,
            }
        })
        .collect())
}

/// Folds a head's option logits into one [`Scored`] per question, probabilities
/// back in the caller's option order.
///
/// # Errors
///
/// [`ScoreError::Engine`] when the head answered a different number of
/// questions or options than the prompt laid out.
pub fn fold_head(
    logits: &[Vec<f32>],
    order: &[Vec<usize>],
    temperature: f64,
) -> Result<Vec<Scored>, ScoreError> {
    if logits.len() != order.len() {
        return Err(ScoreError::Engine(
            "the decision head answered a different number of questions".to_owned(),
        ));
    }
    logits
        .iter()
        .zip(order)
        .map(|(row, order)| {
            if row.len() != order.len() {
                return Err(ScoreError::Engine(
                    "the decision head answered a different number of options".to_owned(),
                ));
            }
            let head = softmax(
                &row.iter().map(|v| f64::from(*v)).collect::<Vec<_>>(),
                temperature,
            );
            let probabilities = reorder(&head, order);
            Ok(Scored {
                top: first_argmax(&probabilities),
                // The head's options are the whole distribution.
                label_mass: 1.0,
                // A decision pass reads every row of its prompt itself.
                cached_tokens: 0,
                probabilities,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use shiso_core::{
        CoreError, DecisionLayoutError, DecisionReadout, PromptEncoder, ReadoutPrompt,
        ReadoutQuestion, ReadoutRecord, SchemaOption, SchemaQuestion, StateSpelling, TokenId,
    };

    use super::{RunScore, ScoreError, fold_head, fold_readout, readout_runs};

    fn token(id: u32) -> TokenId {
        TokenId::new(id)
    }

    /// A readout whose option `j` of every question is the token group
    /// `groups[j]`, behind a one-token trunk and a one-token branch.
    #[derive(Debug)]
    struct FakeReadout {
        groups: Vec<Vec<TokenId>>,
    }

    impl DecisionReadout for FakeReadout {
        fn state_spelling(&self) -> StateSpelling {
            StateSpelling::CompactSorted
        }

        fn encode(
            &self,
            _: &dyn PromptEncoder,
            record: &ReadoutRecord<'_>,
        ) -> Result<ReadoutPrompt, DecisionLayoutError> {
            Ok(ReadoutPrompt {
                trunk: vec![token(1)],
                questions: record
                    .questions
                    .iter()
                    .map(|_| ReadoutQuestion {
                        branch: vec![token(2)],
                        options: self.groups.clone(),
                    })
                    .collect(),
            })
        }
    }

    struct NoEncoder;

    impl PromptEncoder for NoEncoder {
        fn encode(&self, _: &str, _: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
            Ok(Vec::new())
        }
    }

    /// One choice question over `values`, the way a layout is handed it.
    fn schema_question(values: &[&str]) -> SchemaQuestion {
        SchemaQuestion {
            id: "pick".to_owned(),
            wording: "pick one".to_owned(),
            answer: shiso_core::SchemaAnswer::Choice(
                values
                    .iter()
                    .map(|value| SchemaOption {
                        value: (*value).to_owned(),
                        description: None,
                    })
                    .collect(),
            ),
        }
    }

    /// Lays `readout` out for one question over `values` and folds the label
    /// log-probabilities `logprobs` the run would have answered with.
    fn fold_one(readout: &FakeReadout, values: &[&str], logprobs: Vec<f64>) -> super::Scored {
        let schema = vec![schema_question(values)];
        let prompt = readout
            .encode(
                &NoEncoder,
                &ReadoutRecord {
                    state: Some("state"),
                    media: &[],
                    questions: &schema,
                    budget: 100,
                },
            )
            .expect("laid out");
        let runs = readout_runs(&prompt, &schema).expect("runs");
        let scores = [RunScore {
            label_logprobs: logprobs,
            cached_tokens: 0,
        }];
        fold_readout(&prompt, &runs, &scores, 1.0)
            .expect("folded")
            .remove(0)
    }

    #[test]
    fn readout_softmaxes_pooled_option_groups() {
        // Option a reads the best of two spellings (.1 and .3), option b one (.2).
        let readout = FakeReadout {
            groups: vec![vec![token(10), token(11)], vec![token(12)]],
        };
        let scored = fold_one(
            &readout,
            &["a", "b"],
            vec![0.1f64.ln(), 0.3f64.ln(), 0.2f64.ln()],
        );
        assert_eq!(scored.top, 0);
        // softmax([ln .3, ln .2]) = [.6, .4]; the labels carry .3 + .2 = .5 (best spellings).
        assert!((scored.probabilities[0] - 0.6).abs() < 1e-6);
        assert!((scored.probabilities[1] - 0.4).abs() < 1e-6);
        assert!((scored.label_mass - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ties_pick_the_first_option() {
        let readout = FakeReadout {
            groups: vec![vec![token(10)], vec![token(11)]],
        };
        let scored = fold_one(&readout, &["x", "y"], vec![0.25f64.ln(), 0.25f64.ln()]);
        assert_eq!(scored.top, 0);
    }

    fn laid(groups: &[Vec<TokenId>]) -> ReadoutQuestion {
        ReadoutQuestion {
            branch: vec![token(2)],
            options: groups.to_vec(),
        }
    }

    fn schema_of(questions: usize, options: &[&str]) -> Vec<SchemaQuestion> {
        (0..questions).map(|_| schema_question(options)).collect()
    }

    #[test]
    fn several_questions_share_the_trunk_as_their_prefix() {
        let groups = vec![vec![token(10)], vec![token(11)]];
        let prompt = ReadoutPrompt {
            trunk: vec![token(1), token(3), token(4)],
            questions: vec![laid(&groups), laid(&groups)],
        };
        let runs = readout_runs(&prompt, &schema_of(2, &["a", "b"])).expect("runs");
        assert!(runs.iter().all(|run| run.shared_prefix == 3));
        assert_eq!(runs[0].tokens.len(), 4);
        // One question shares nothing.
        let one = ReadoutPrompt {
            trunk: prompt.trunk.clone(),
            questions: vec![laid(&groups)],
        };
        let runs = readout_runs(&one, &schema_of(1, &["a", "b"])).expect("runs");
        assert_eq!(runs[0].shared_prefix, 0);
    }

    #[test]
    fn cached_tokens_never_exceed_the_prompt() {
        let groups = vec![vec![token(10)], vec![token(11)]];
        let prompt = ReadoutPrompt {
            trunk: vec![token(1)],
            questions: vec![laid(&groups)],
        };
        let runs = readout_runs(&prompt, &schema_of(1, &["a", "b"])).expect("runs");
        let scores = [RunScore {
            label_logprobs: vec![-1.0, -1.0],
            cached_tokens: 99,
        }];
        let folded = fold_readout(&prompt, &runs, &scores, 1.0).expect("folded");
        assert_eq!(folded[0].cached_tokens, 2);
    }

    #[test]
    fn count_mismatches_are_typed_errors() {
        let groups = vec![vec![token(10)], vec![token(11)]];
        let prompt = ReadoutPrompt {
            trunk: vec![token(1)],
            questions: vec![laid(&groups)],
        };
        // Schema holds two questions, the layout one.
        let error = readout_runs(&prompt, &schema_of(2, &["a", "b"])).expect_err("questions");
        assert!(matches!(error, ScoreError::Layout(_)), "{error:?}");
        // Schema question has three options, the layout two.
        let error = readout_runs(&prompt, &schema_of(1, &["a", "b", "c"])).expect_err("options");
        assert!(matches!(error, ScoreError::Layout(_)), "{error:?}");
        // An option read from no token.
        let empty = ReadoutPrompt {
            trunk: vec![token(1)],
            questions: vec![laid(&[vec![token(10)], Vec::new()])],
        };
        let error = readout_runs(&empty, &schema_of(1, &["a", "b"])).expect_err("empty option");
        assert!(matches!(error, ScoreError::Layout(_)), "{error:?}");

        let runs = readout_runs(&prompt, &schema_of(1, &["a", "b"])).expect("runs");
        let error = fold_readout(&prompt, &runs, &[], 1.0).expect_err("scores");
        assert!(matches!(error, ScoreError::Engine(_)), "{error:?}");

        let error = fold_head(&[vec![1.0, 2.0]], &[], 1.0).expect_err("questions");
        assert!(matches!(&error, ScoreError::Engine(m) if m.contains("number of questions")));
        let error = fold_head(&[vec![1.0]], &[vec![0, 1]], 1.0).expect_err("options");
        assert!(matches!(&error, ScoreError::Engine(m) if m.contains("number of options")));
    }

    #[test]
    fn fold_head_puts_options_back_in_the_callers_order() {
        // The head read the caller's second option first.
        let folded = fold_head(&[vec![0.0, 3.0f32.ln()]], &[vec![1, 0]], 1.0).expect("folded");
        assert!((folded[0].probabilities[0] - 0.75).abs() < 1e-6);
        assert!((folded[0].probabilities[1] - 0.25).abs() < 1e-6);
        assert_eq!(folded[0].top, 0);
        assert!((folded[0].label_mass - 1.0).abs() < f64::EPSILON);
    }
}
