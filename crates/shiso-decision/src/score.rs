//! Scoring a decision from a model's readout rows or its decision head.
//!
//! # The `Scorer` seam is synchronous
//!
//! Whoever runs the model implements [`Scorer`]; this module never touches a
//! device. The trait is sync because the library has no runtime and its
//! embedders (a worker thread that owns the model) block on the model anyway.
//! The server is the one async caller, and it must keep submitting every
//! question's pass at once so its scheduler sees the siblings together, reuses
//! the shared-prefix checkpoint and queues them as it always has. A sync trait
//! would force it to block a runtime thread to do that, so the server does not
//! go through [`Scorer`]: it uses the same pure steps the `decide_with_*`
//! drivers are built from - [`readout_runs`] and [`fold_readout`] around its
//! own async engine call, [`reorder`] and [`first_argmax`] for a head - and
//! its behaviour is the library's by construction.

use shiso_core::{
    DecisionLayout, DecisionLayoutError, DecisionPrompt, DecisionReadout, DecisionRecord,
    PromptEncoder, ReadoutPrompt, ReadoutRecord, SchemaAnswer, SchemaQuestion, TokenId,
};

use crate::error::DecideError;
use crate::math::{label_mass, softmax};
use crate::schema::{Answer, DecisionRequest, Question};

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

/// What scoring needs from whoever runs the model.
pub trait Scorer {
    /// Scores `runs`, one [`RunScore`] per run in the same order.
    ///
    /// # Errors
    ///
    /// [`DecideError::Engine`] when the model cannot be run.
    fn score(&self, runs: &[ScoreRun]) -> Result<Vec<RunScore>, DecideError>;

    /// Scores a decision head's prompt: one logit per option of each question,
    /// in the head's option order.
    ///
    /// # Errors
    ///
    /// [`DecideError::Engine`] when the model cannot be run or has no head.
    fn score_decision(&self, prompt: &DecisionPrompt) -> Result<Vec<Vec<f32>>, DecideError>;
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
/// [`DecideError::Layout`] when the layout laid out a different number of
/// questions than `schema` holds, or a different number of options for one, or
/// an option read from no token.
pub fn readout_runs(
    prompt: &ReadoutPrompt,
    schema: &[SchemaQuestion],
) -> Result<Vec<ScoreRun>, DecideError> {
    if prompt.questions.len() != schema.len() {
        return Err(DecideError::Layout(
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
                return Err(DecideError::Layout(
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
/// [`DecideError::Engine`] when `scores` does not hold one entry per question.
pub fn fold_readout(
    prompt: &ReadoutPrompt,
    runs: &[ScoreRun],
    scores: &[RunScore],
    temperature: f64,
) -> Result<Vec<Scored>, DecideError> {
    if scores.len() != prompt.questions.len() || runs.len() != scores.len() {
        return Err(DecideError::Engine(
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
/// [`DecideError::Engine`] when the head answered a different number of
/// questions or options than the prompt laid out.
pub fn fold_head(
    logits: &[Vec<f32>],
    order: &[Vec<usize>],
    temperature: f64,
) -> Result<Vec<Scored>, DecideError> {
    if logits.len() != order.len() {
        return Err(DecideError::Engine(
            "the decision head answered a different number of questions".to_owned(),
        ));
    }
    logits
        .iter()
        .zip(order)
        .map(|(row, order)| {
            if row.len() != order.len() {
                return Err(DecideError::Engine(
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

/// Checks `req` and returns its temperature.
fn validate(req: &DecisionRequest) -> Result<f64, DecideError> {
    if req.questions.is_empty() {
        return Err(DecideError::Request("no questions".to_owned()));
    }
    for question in &req.questions {
        let options = question.option_values().len();
        if options == 0 {
            return Err(DecideError::Request(format!(
                "question {:?} has no options",
                question.name()
            )));
        }
    }
    let temperature = f64::from(req.temperature.unwrap_or(1.0));
    if !(temperature.is_finite() && temperature > 0.0) {
        return Err(DecideError::Request(
            "temperature must be positive".to_owned(),
        ));
    }
    Ok(temperature)
}

/// A probability or mass as the `f32` an answer reports.
#[expect(
    clippy::cast_possible_truncation,
    reason = "an answer reports f32; the f64 arithmetic is done"
)]
fn narrow(value: f64) -> f32 {
    value as f32
}

/// Names each question's scored answer.
fn name_answers(
    questions: &[Question],
    scored: Vec<Scored>,
    confidence: impl Fn(&Scored) -> f64,
) -> Vec<Answer> {
    questions
        .iter()
        .zip(scored)
        .map(|(question, scored)| {
            let values = question.option_values();
            Answer {
                name: question.name().to_owned(),
                choice: values.get(scored.top).cloned().unwrap_or_default(),
                confidence: narrow(confidence(&scored)),
                probabilities: values
                    .into_iter()
                    .zip(&scored.probabilities)
                    .map(|(value, p)| (value, narrow(*p)))
                    .collect(),
            }
        })
        .collect()
}

fn layout_error(error: &DecisionLayoutError) -> DecideError {
    DecideError::Layout(error.to_string())
}

/// Answers `req` with a model that has no head, from the next-token rows at
/// each question's answer slot.
///
/// `budget` is the most tokens one question's prompt may hold. The answer's
/// confidence is the mass the options' tokens carry together.
///
/// # Errors
///
/// [`DecideError::Request`] for a request that cannot be answered,
/// [`DecideError::Layout`] when the readout refuses the schema, and whatever
/// the scorer answers with.
pub fn decide_with_readout(
    readout: &dyn DecisionReadout,
    vocab: &dyn PromptEncoder,
    scorer: &dyn Scorer,
    req: &DecisionRequest,
    budget: usize,
) -> Result<Vec<Answer>, DecideError> {
    let temperature = validate(req)?;
    let schema: Vec<SchemaQuestion> = req.questions.iter().map(Question::to_schema).collect();
    // A blank state is no state, as on the server: the layout writes no state
    // block for it.
    let state = (!req.input.trim().is_empty()).then_some(req.input.as_str());
    let prompt = readout
        .encode(
            vocab,
            &ReadoutRecord {
                state,
                media: &[],
                questions: &schema,
                budget,
            },
        )
        .map_err(|error| layout_error(&error))?;
    let runs = readout_runs(&prompt, &schema)?;
    let outcomes = scorer.score(&runs)?;
    let folded = fold_readout(&prompt, &runs, &outcomes, temperature)?;
    Ok(name_answers(&req.questions, folded, |s| s.label_mass))
}

/// Answers `req` with a model that carries a decision head, which scores every
/// option of every question from one prompt.
///
/// `budget` is the most tokens the prompt may hold. The answer's confidence is
/// `1.0`: the head's options are the whole distribution.
///
/// # Errors
///
/// [`DecideError::Request`] for a request that cannot be answered,
/// [`DecideError::Layout`] when the layout refuses the schema, and whatever
/// the scorer answers with.
pub fn decide_with_head(
    layout: &dyn DecisionLayout,
    vocab: &dyn PromptEncoder,
    scorer: &dyn Scorer,
    req: &DecisionRequest,
    budget: usize,
) -> Result<Vec<Answer>, DecideError> {
    let temperature = validate(req)?;
    let schema: Vec<SchemaQuestion> = req.questions.iter().map(Question::to_schema).collect();
    let prompt = layout
        .encode(
            vocab,
            &DecisionRecord {
                state: &req.input,
                media: &[],
                questions: &schema,
                budget,
            },
        )
        .map_err(|error| layout_error(&error))?;
    let logits = scorer.score_decision(&prompt)?;
    let folded = fold_head(&logits, &prompt.order, temperature)?;
    Ok(name_answers(&req.questions, folded, |s| s.label_mass))
}

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use shiso_core::{
        CoreError, DecisionLayout, DecisionLayoutError, DecisionPrompt, DecisionQuestion,
        DecisionReadout, PromptEncoder, ReadoutPrompt, ReadoutQuestion, ReadoutRecord,
        StateSpelling, TokenId,
    };

    use super::{
        RunScore, ScoreRun, Scorer, decide_with_head, decide_with_readout, fold_head, fold_readout,
        readout_runs,
    };
    use crate::error::DecideError;
    use crate::schema::{Choice, DecisionRequest, Question};

    struct NoEncoder;

    impl PromptEncoder for NoEncoder {
        fn encode(&self, _: &str, _: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError> {
            Ok(Vec::new())
        }
    }

    fn token(id: u32) -> TokenId {
        TokenId::new(id)
    }

    /// A readout whose option `j` of every question is the token group
    /// `groups[j]`, behind a one-token trunk and a one-token branch.
    #[derive(Debug)]
    struct FakeReadout {
        groups: Vec<Vec<TokenId>>,
        refuse: bool,
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
            if self.refuse {
                return Err(DecisionLayoutError::PromptTooLong {
                    tokens: 9,
                    budget: record.budget,
                });
            }
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

    /// Answers every run with the same label log-probabilities.
    struct FakeScorer {
        logprobs: Vec<f64>,
        logits: Vec<Vec<f32>>,
    }

    impl Scorer for FakeScorer {
        fn score(&self, runs: &[ScoreRun]) -> Result<Vec<RunScore>, DecideError> {
            Ok(runs
                .iter()
                .map(|_| RunScore {
                    label_logprobs: self.logprobs.clone(),
                    cached_tokens: 0,
                })
                .collect())
        }

        fn score_decision(&self, _: &DecisionPrompt) -> Result<Vec<Vec<f32>>, DecideError> {
            Ok(self.logits.clone())
        }
    }

    fn options(values: &[&str]) -> Vec<Choice> {
        values
            .iter()
            .map(|value| Choice {
                value: (*value).to_owned(),
                description: None,
            })
            .collect()
    }

    fn request(values: &[&str]) -> DecisionRequest {
        DecisionRequest {
            input: "state".to_owned(),
            questions: vec![Question::Choice {
                name: "pick".to_owned(),
                instructions: String::new(),
                choices: options(values),
            }],
            temperature: None,
        }
    }

    fn quiet_scorer() -> FakeScorer {
        FakeScorer {
            logprobs: Vec::new(),
            logits: Vec::new(),
        }
    }

    #[test]
    fn readout_softmaxes_pooled_option_groups() {
        // Option a reads the best of two spellings (.1 and .3), option b one (.2).
        let readout = FakeReadout {
            groups: vec![vec![token(10), token(11)], vec![token(12)]],
            refuse: false,
        };
        let scorer = FakeScorer {
            logprobs: vec![0.1f64.ln(), 0.3f64.ln(), 0.2f64.ln()],
            logits: Vec::new(),
        };
        let answers =
            decide_with_readout(&readout, &NoEncoder, &scorer, &request(&["a", "b"]), 100)
                .expect("scored");
        let [answer] = answers.as_slice() else {
            panic!("one answer expected");
        };
        assert_eq!(answer.name, "pick");
        assert_eq!(answer.choice, "a");
        // softmax([ln .3, ln .2]) = [.6, .4]; the labels carry .3 + .2 = .5 (best spellings).
        assert_eq!(answer.probabilities[0].0, "a");
        assert!((answer.probabilities[0].1 - 0.6).abs() < 1e-6);
        assert!((answer.probabilities[1].1 - 0.4).abs() < 1e-6);
        assert!((answer.confidence - 0.5).abs() < 1e-6);
    }

    #[test]
    fn ties_pick_the_first_option() {
        let readout = FakeReadout {
            groups: vec![vec![token(10)], vec![token(11)]],
            refuse: false,
        };
        let scorer = FakeScorer {
            logprobs: vec![0.25f64.ln(), 0.25f64.ln()],
            logits: Vec::new(),
        };
        let answers =
            decide_with_readout(&readout, &NoEncoder, &scorer, &request(&["x", "y"]), 100)
                .expect("scored");
        assert_eq!(answers[0].choice, "x");
    }

    #[test]
    fn head_confidence_is_one() {
        #[derive(Debug)]
        struct FakeLayout;

        impl DecisionLayout for FakeLayout {
            fn encode(
                &self,
                _: &dyn PromptEncoder,
                record: &shiso_core::DecisionRecord<'_>,
            ) -> Result<DecisionPrompt, DecisionLayoutError> {
                Ok(DecisionPrompt {
                    tokens: vec![token(1), token(2)],
                    questions: record
                        .questions
                        .iter()
                        .map(|_| DecisionQuestion {
                            kind: 1,
                            span: 0..1,
                            options: vec![1..2, 1..2],
                        })
                        .collect(),
                    // The head scores `b` first and `a` second.
                    order: record.questions.iter().map(|_| vec![1, 0]).collect(),
                })
            }
        }

        let scorer = FakeScorer {
            logprobs: Vec::new(),
            logits: vec![vec![3.0, 0.0]],
        };
        let answers =
            decide_with_head(&FakeLayout, &NoEncoder, &scorer, &request(&["a", "b"]), 100)
                .expect("scored");
        let [answer] = answers.as_slice() else {
            panic!("one answer expected");
        };
        // The head's first option is the caller's second.
        assert_eq!(answer.choice, "b");
        assert!((answer.confidence - 1.0).abs() < f32::EPSILON);
        assert_eq!(answer.probabilities[0].0, "a");
        assert!(answer.probabilities[1].1 > answer.probabilities[0].1);
    }

    #[test]
    fn layout_error_is_typed() {
        let readout = FakeReadout {
            groups: vec![vec![token(10)], vec![token(11)]],
            refuse: true,
        };
        let error = decide_with_readout(
            &readout,
            &NoEncoder,
            &quiet_scorer(),
            &request(&["a", "b"]),
            4,
        )
        .expect_err("the layout refused");
        assert!(matches!(error, DecideError::Layout(_)), "{error:?}");
    }

    #[test]
    fn a_non_positive_temperature_is_refused() {
        let readout = FakeReadout {
            groups: vec![vec![token(10)], vec![token(11)]],
            refuse: false,
        };
        let mut req = request(&["a", "b"]);
        req.temperature = Some(0.0);
        let error = decide_with_readout(&readout, &NoEncoder, &quiet_scorer(), &req, 100)
            .expect_err("zero temperature");
        assert!(matches!(error, DecideError::Request(_)), "{error:?}");
    }

    fn laid(groups: &[Vec<TokenId>]) -> ReadoutQuestion {
        ReadoutQuestion {
            branch: vec![token(2)],
            options: groups.to_vec(),
        }
    }

    fn schema_of(questions: usize, options: &[&str]) -> Vec<shiso_core::SchemaQuestion> {
        (0..questions)
            .map(|_| request(options).questions[0].to_schema())
            .collect()
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
        assert!(matches!(error, DecideError::Layout(_)), "{error:?}");
        // Schema question has three options, the layout two.
        let error = readout_runs(&prompt, &schema_of(1, &["a", "b", "c"])).expect_err("options");
        assert!(matches!(error, DecideError::Layout(_)), "{error:?}");
        // An option read from no token.
        let empty = ReadoutPrompt {
            trunk: vec![token(1)],
            questions: vec![laid(&[vec![token(10)], Vec::new()])],
        };
        assert!(readout_runs(&empty, &schema_of(1, &["a", "b"])).is_err());

        let runs = readout_runs(&prompt, &schema_of(1, &["a", "b"])).expect("runs");
        let error = fold_readout(&prompt, &runs, &[], 1.0).expect_err("scores");
        assert!(matches!(error, DecideError::Engine(_)), "{error:?}");

        let error = fold_head(&[vec![1.0, 2.0]], &[], 1.0).expect_err("questions");
        assert!(matches!(&error, DecideError::Engine(m) if m.contains("number of questions")));
        let error = fold_head(&[vec![1.0]], &[vec![0, 1]], 1.0).expect_err("options");
        assert!(matches!(&error, DecideError::Engine(m) if m.contains("number of options")));
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
