//! How a schema decision becomes the prompt a decision head reads.
//!
//! A decision head answers every question of a schema from one prompt, and
//! which prompt is the checkpoint's own: the wording around the state, where
//! pictures go, how an option is spelled, how long the prompt may grow. The
//! server holds the schema and the tokenizer; the model family holds that
//! layout. [`DecisionLayout`] is the seam between them, so a surface serves
//! every decision family through the one call and a new family touches no
//! file above the model crate.
//!
//! The layout is a value of its own rather than a method on the model,
//! because only the scheduler thread calls the model and a prompt is built on
//! the request's thread, before anything is queued.

use std::fmt;
use std::ops::Range;

use thiserror::Error;

use crate::error::CoreError;
use crate::ids::TokenId;
use crate::model::DecisionQuestion;

/// One allowed answer of a question, as the caller wrote it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaOption {
    /// The option's value, spelled as text.
    pub value: String,
    /// What the option means, when the caller said.
    pub description: Option<String>,
}

/// What a question may be answered with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchemaAnswer {
    /// Yes or no, each with the caller's wording when it gave one.
    YesNo {
        /// What a yes means.
        yes: Option<String>,
        /// What a no means.
        no: Option<String>,
    },
    /// One of the listed options, in the caller's order.
    Choice(Vec<SchemaOption>),
    /// A level of the listed scale, lowest first.
    Score(Vec<SchemaOption>),
}

/// One question of a schema, before anything is tokenized.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SchemaQuestion {
    /// The caller's id for the question.
    pub id: String,
    /// The question's wording; blank when the caller gave none.
    pub wording: String,
    /// What it may be answered with.
    pub answer: SchemaAnswer,
}

/// The kind of one media item a decision prompt carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecisionMediaKind {
    /// A still image.
    Image,
    /// A video clip.
    Video,
}

/// One decoded media item, as the prompt has to make room for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecisionMedia {
    /// What the item is.
    pub kind: DecisionMediaKind,
    /// How many placeholder rows the encoder writes for it.
    pub pads: usize,
}

/// Everything a layout reads to build one decision prompt.
#[derive(Debug, Clone, Copy)]
pub struct DecisionRecord<'a> {
    /// The state the questions are asked about.
    pub state: &'a str,
    /// The decoded media, in the order its rows will be spliced.
    pub media: &'a [DecisionMedia],
    /// The schema, in the caller's order.
    pub questions: &'a [SchemaQuestion],
    /// The most tokens the prompt may hold: the server's context. A layout
    /// may hold itself to less.
    pub budget: usize,
}

/// A prompt a decision head can read.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecisionPrompt {
    /// The prompt, with every media item's placeholder run already at its
    /// full length.
    pub tokens: Vec<TokenId>,
    /// One entry per question, in the caller's order, with its spans in
    /// `tokens` and its options in the order the head scores them.
    pub questions: Vec<DecisionQuestion>,
    /// For each question, the caller's index of each option the head scores,
    /// in the head's order: `order[q][h]` is where the head's option `h` sits
    /// in the caller's list.
    pub order: Vec<Vec<usize>>,
}

/// Why a schema could not be laid out.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecisionLayoutError {
    /// Everything but the state - the schema and the media - does not fit.
    #[error("the schema needs {tokens} tokens before any state, and the prompt holds {budget}")]
    SchemaTooLong {
        /// Tokens the prompt needs before the state.
        tokens: usize,
        /// Tokens the prompt may hold.
        budget: usize,
    },
    /// Two options of one question read the same to the head, which would
    /// score a coin-flip between them.
    #[error("question {question} has two options spelled {spelling:?}")]
    AmbiguousOption {
        /// The question's index in the schema.
        question: usize,
        /// The spelling the two share.
        spelling: String,
    },
    /// The tokenizer refused the text.
    #[error("the decision prompt could not be encoded")]
    Encode(#[source] CoreError),
    /// A layout that cuts nothing laid out a prompt longer than the budget.
    #[error("the decision prompt needs {tokens} tokens, and the prompt holds {budget}")]
    PromptTooLong {
        /// Tokens the longest question's prompt needs.
        tokens: usize,
        /// Tokens the prompt may hold.
        budget: usize,
    },
    /// The vocabulary cannot spell a question's answers as single tokens, so
    /// no row could be read for them.
    #[error("question {question}: {reason}")]
    Unreadable {
        /// The question's index in the schema.
        question: usize,
        /// What the vocabulary lacks.
        reason: String,
    },
}

/// The tokenizer, as a layout reaches it.
///
/// Every call encodes one piece on its own, with no opening token: a layout
/// that builds its prompt piece by piece gets token boundaries at its piece
/// boundaries, which is what a head trained on such prompts pools over.
pub trait PromptEncoder {
    /// Encodes `text`. Inside each `literal` byte range the text is the
    /// caller's, and a special-token spelling there stays text; everywhere
    /// else it becomes its token.
    ///
    /// # Errors
    ///
    /// [`CoreError`] when the tokenizer refuses the text.
    fn encode(&self, text: &str, literal: &[Range<usize>]) -> Result<Vec<TokenId>, CoreError>;
}

/// A decision family's prompt layout.
pub trait DecisionLayout: Send + Sync + fmt::Debug {
    /// Lays `record` out as the one prompt the head reads.
    ///
    /// # Errors
    ///
    /// [`DecisionLayoutError::SchemaTooLong`] when nothing but the state can
    /// be cut and the rest does not fit, [`DecisionLayoutError::AmbiguousOption`]
    /// for a question the head cannot tell apart, and
    /// [`DecisionLayoutError::Encode`] when the tokenizer refuses.
    fn encode(
        &self,
        encoder: &dyn PromptEncoder,
        record: &DecisionRecord<'_>,
    ) -> Result<DecisionPrompt, DecisionLayoutError>;
}

/// How a structured state - a JSON object or array, rather than a string -
/// is spelled before a readout layout reads it.
///
/// A checkpoint was trained on one spelling, and the bytes are scored, so the
/// spelling is the family's call and not the wire's.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StateSpelling {
    /// Compact JSON with every object's keys sorted, the wire's own default.
    CompactSorted,
    /// JSON indented two spaces, keys in the order the caller sent them, and
    /// non-ASCII text left unescaped.
    Indented,
}

/// Everything a readout layout reads to build one decision's prompts.
#[derive(Debug, Clone, Copy)]
pub struct ReadoutRecord<'a> {
    /// The state the questions are asked about, already spelled as text;
    /// `None` when the caller sent none and the media are the whole state.
    pub state: Option<&'a str>,
    /// The decoded media, in the order their rows will be spliced.
    pub media: &'a [DecisionMedia],
    /// The schema, in the caller's order.
    pub questions: &'a [SchemaQuestion],
    /// The most tokens one question's prompt may hold: the server's context.
    pub budget: usize,
}

/// One question of a readout decision: its continuation of the shared trunk,
/// and the tokens each of its options is read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadoutQuestion {
    /// The tokens after the trunk, ending at the answer slot.
    pub branch: Vec<TokenId>,
    /// One group per option, in the caller's order. An option scores the best
    /// of its group's log-probabilities, so a group is every spelling of that
    /// answer the model may put its mass on - `yes` and `Yes`, a code with
    /// and without a leading space.
    pub options: Vec<Vec<TokenId>>,
}

impl ReadoutQuestion {
    /// Every token the answer slot is read at, the groups back to back: the
    /// label list a scoring pass is asked for.
    #[must_use]
    pub fn labels(&self) -> Vec<TokenId> {
        self.options.iter().flatten().copied().collect()
    }

    /// Folds the log-probabilities of [`ReadoutQuestion::labels`], in that
    /// order, into one score per option: the maximum over its group.
    ///
    /// A missing entry reads as impossible rather than panicking: the caller
    /// asked for exactly these labels, so a short vector is a bug a test sees
    /// as a zero probability.
    #[must_use]
    pub fn pool(&self, label_logprobs: &[f64]) -> Vec<f64> {
        let mut at = 0usize;
        self.options
            .iter()
            .map(|group| {
                let best = label_logprobs
                    .get(at..at + group.len())
                    .unwrap_or(&[])
                    .iter()
                    .fold(f64::NEG_INFINITY, |best, value| best.max(*value));
                at += group.len();
                best
            })
            .collect()
    }
}

/// The prompts a readout layout lays a decision out as.
///
/// Every question's prompt is `trunk` followed by its own branch: the trunk
/// holds what all questions share - the state and its pictures - so a server
/// can read it once and continue it per question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadoutPrompt {
    /// The tokens every question's prompt opens with, media runs at full
    /// length.
    pub trunk: Vec<TokenId>,
    /// One entry per question, in the caller's order.
    pub questions: Vec<ReadoutQuestion>,
}

/// A decision family that answers from the next-token distribution at an
/// answer slot rather than through a head.
///
/// Each question gets its own prompt, and the answer is a softmax over the
/// best log-probability each option's token group has at the prompt's last
/// row. The wording, the option codes and which spellings count are the
/// checkpoint's own, which is why the layout lives with the family and the
/// server only reads rows.
pub trait DecisionReadout: Send + Sync + fmt::Debug {
    /// How a structured state is spelled before [`DecisionReadout::encode`]
    /// reads it.
    fn state_spelling(&self) -> StateSpelling;

    /// Lays `record` out as a shared trunk and one branch per question.
    ///
    /// # Errors
    ///
    /// [`DecisionLayoutError::PromptTooLong`] when a question's prompt does
    /// not fit the budget, [`DecisionLayoutError::Unreadable`] when the
    /// vocabulary cannot spell a question's answers as single tokens, and
    /// [`DecisionLayoutError::Encode`] when the tokenizer refuses.
    fn encode(
        &self,
        encoder: &dyn PromptEncoder,
        record: &ReadoutRecord<'_>,
    ) -> Result<ReadoutPrompt, DecisionLayoutError>;
}

#[cfg(test)]
mod tests {
    use super::ReadoutQuestion;
    use crate::ids::TokenId;

    #[test]
    fn each_option_scores_the_best_of_its_group() {
        let question = ReadoutQuestion {
            branch: Vec::new(),
            options: vec![
                vec![TokenId::new(1), TokenId::new(2)],
                vec![TokenId::new(3)],
            ],
        };
        assert_eq!(question.labels().len(), 3);
        assert_eq!(question.pool(&[-3.0, -1.0, -2.0]), vec![-1.0, -2.0]);
        // A short vector leaves the option impossible, never a panic.
        assert_eq!(question.pool(&[-3.0]), vec![f64::NEG_INFINITY; 2]);
    }
}
