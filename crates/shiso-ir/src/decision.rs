//! Decision scoring in the IR: questions in, probabilities out.
//!
//! A decision question is answered from the next-token distribution at one
//! answer position (or from a decision head's option logits): nothing is
//! generated. These types are what `/v1/decisions`, `/v1/systemone` and an
//! embedding application all hand to, and read back from, the one typed core
//! in `shiso-runtime`. Like the rest of this crate they carry no serde
//! derives: each wire format parses its own body into a [`Request`] and spells
//! a [`Response`] out in its own shape.
//!
//! # Probabilities are `f64`
//!
//! The scoring arithmetic is `f64` end to end and the wires print it as it is,
//! so narrowing an answer here would change the bytes a caller reads back.
//!
//! # `/v1/systemone` has no types of its own
//!
//! System One is a second reading of the same engine: a request is the same
//! [`Request`] with the knobs it refuses left at their defaults, and its
//! `noul` criterion is a [`QuestionKind::YesNo`] with
//! [`Question::probability_answer`] set. The two wires differ only in how they
//! parse and spell.

use crate::message::MediaSource;

/// A set of questions about one input.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    /// What is being judged.
    pub input: Input,
    /// What the readout-style layouts read as the state, when it is not the
    /// input's own text.
    pub readout_state: ReadoutState,
    /// The questions, answered in this order.
    pub questions: Vec<Question>,
    /// The softmax temperature; `None` is `1.0`. Must be greater than zero.
    pub temperature: Option<f64>,
    /// Whether the chat template keeps earlier thinking, when the caller said.
    pub preserve_thinking: Option<bool>,
    /// Whether each [`Answer`] reports the token ids the model read.
    pub token_ids: bool,
}

/// The thing being judged.
#[derive(Debug, Clone, PartialEq)]
pub enum Input {
    /// One string.
    Text(String),
    /// The caller's own text and pictures, in the order they were sent.
    Blocks(Vec<Part>),
    /// A state with pictures scored before it, wherever the model's layout
    /// puts them.
    Pictured {
        /// The pictures.
        media: Vec<MediaSource>,
        /// The state's text, as sent.
        text: String,
    },
}

/// One part of [`Input::Blocks`].
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    /// Text, as sent.
    Text(String),
    /// A picture.
    Image(MediaSource),
}

/// How a layout that reads a plain state spells it, when that is not simply
/// the input's text.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ReadoutState {
    /// The input's own text; a blank one is no state at all.
    #[default]
    Input,
    /// No state: the caller sent nothing, or a null.
    Absent,
    /// A structured state, spelled each way a family might want it.
    Structured {
        /// Keys sorted, no whitespace.
        compact: String,
        /// Indented, in the order the caller sent.
        indented: String,
    },
}

/// One question about the input.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    /// The caller's name for it; empty when the caller sent none.
    pub name: String,
    /// What is being asked; a blank one drops the line from the prompt.
    pub instructions: String,
    /// How it is answered.
    pub kind: QuestionKind,
    /// Whether the answer is reported as the probability of `yes` rather than
    /// as a label word. Only meaningful on [`QuestionKind::YesNo`].
    pub probability_answer: bool,
}

/// How a question is answered.
#[derive(Debug, Clone, PartialEq)]
pub enum QuestionKind {
    /// One of several options.
    Choice(Vec<Choice>),
    /// A binary judgement, `yes` or `no`.
    YesNo {
        /// Wording for the `yes` label; its line is dropped when blank.
        yes: Option<String>,
        /// Wording for the `no` label; its line is dropped when blank.
        no: Option<String>,
    },
    /// One of several numbered levels, counted from zero.
    Score(Vec<Level>),
}

/// The value of a choice, typed as the caller sent it: a string and a boolean
/// with the same text are distinct, so an answer echoes back the type it was
/// given.
#[derive(Debug, Clone, PartialEq)]
pub enum ChoiceValue {
    /// A string, spelled verbatim.
    Str(String),
    /// A boolean, spelled `true` or `false`.
    Bool(bool),
}

impl ChoiceValue {
    /// The text the option's line spells for this value.
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Self::Str(text) => text.clone(),
            Self::Bool(value) => value.to_string(),
        }
    }
}

/// One option of a [`QuestionKind::Choice`].
#[derive(Debug, Clone, PartialEq)]
pub struct Choice {
    /// What the option's line spells, and what an answer reads back as.
    pub value: ChoiceValue,
    /// Extra wording shown after `" - "` when non-blank.
    pub description: Option<String>,
}

/// One level of a [`QuestionKind::Score`].
#[derive(Debug, Clone, PartialEq)]
pub struct Level {
    /// What follows the level's digit on its line.
    pub label: String,
    /// Extra wording shown after `" - "` when non-blank.
    pub description: Option<String>,
}

/// Which kind of question an [`Answer`] answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnswerKind {
    /// A choice.
    Choice,
    /// A yes/no judgement.
    YesNo,
    /// A score.
    Score,
}

/// One scored question.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The question's name, as sent; empty when it had none.
    pub name: String,
    /// What kind of question it was.
    pub kind: AnswerKind,
    /// Whether the question asked for the probability of `yes`.
    pub probability_answer: bool,
    /// The most probable option's value; the level number for a score, `yes`
    /// or `no` for a yes/no.
    pub choice: ChoiceValue,
    /// Each option and its probability, in the question's own order: a
    /// choice's values, a score's level labels, `yes` then `no`.
    pub probabilities: Vec<(ChoiceValue, f64)>,
    /// How sure the answer is: how far the top option stands above a uniform
    /// draw, or for a score how concentrated it is around its top level.
    pub confidence: f64,
    /// The mass the options carry together before the softmax, independent of
    /// the temperature; `1.0` from a decision head.
    pub label_mass: f64,
    /// The expectation over the levels; `Some` for a score only.
    pub expected_score: Option<f64>,
    /// The ids the model read, when asked for.
    pub prompt_token_ids: Option<Vec<u32>>,
    /// The ids of the labels that were scored, when asked for; empty for a
    /// head, which scores options rather than label tokens.
    pub label_token_ids: Option<Vec<u32>>,
}

/// What a decision read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Prompt tokens, counting a prefix shared between questions once.
    pub input_tokens: u64,
    /// The part of that a stored prefix covered, never more than the input.
    pub cached_tokens: u64,
}

/// The answers to a [`Request`].
#[derive(Debug, Clone, PartialEq)]
pub struct Response {
    /// One answer per question, in the request's order.
    pub answers: Vec<Answer>,
    /// What was read to produce them.
    pub usage: Usage,
}
