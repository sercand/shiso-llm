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
