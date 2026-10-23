//! The questions and answers of an in-process decision.
//!
//! These are the library's own neutral shapes: the server's wire structs and
//! an embedding application's types both convert to them, so neither shapes
//! the other.

use shiso_core::{SchemaAnswer, SchemaOption, SchemaQuestion};

/// One allowed answer of a question.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Choice {
    /// The answer's value, as it is reported back.
    pub value: String,
    /// What the answer means, when the caller says.
    pub description: Option<String>,
}

/// One question of a decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Question {
    /// Pick one of `choices`.
    Choice {
        /// The question's id; the answer carries it back.
        name: String,
        /// The question's wording; blank is refused, as on the server's decisions route.
        instructions: String,
        /// The allowed answers, in the caller's order.
        choices: Vec<Choice>,
    },
    /// Yes or no; the options are reported as `yes` and `no`.
    YesNo {
        /// The question's id; the answer carries it back.
        name: String,
        /// The question's wording; blank is refused, as on the server's decisions route.
        instructions: String,
        /// What a yes means, when the caller says.
        yes: Option<String>,
        /// What a no means, when the caller says.
        no: Option<String>,
    },
    /// A level of a scale, lowest first.
    Score {
        /// The question's id; the answer carries it back.
        name: String,
        /// The question's wording; blank is refused, as on the server's decisions route.
        instructions: String,
        /// The levels, lowest first; each value is the level's label.
        levels: Vec<Choice>,
    },
}

impl Question {
    /// The question's id.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Choice { name, .. } | Self::YesNo { name, .. } | Self::Score { name, .. } => name,
        }
    }

    /// The question's wording as the caller wrote it.
    #[must_use]
    pub fn instructions(&self) -> &str {
        match self {
            Self::Choice { instructions, .. }
            | Self::YesNo { instructions, .. }
            | Self::Score { instructions, .. } => instructions,
        }
    }

    /// The values of the allowed answers, in the caller's order.
    #[must_use]
    pub fn option_values(&self) -> Vec<String> {
        match self {
            Self::YesNo { .. } => vec!["yes".to_owned(), "no".to_owned()],
            Self::Choice { choices, .. } => choices.iter().map(|c| c.value.clone()).collect(),
            Self::Score { levels, .. } => levels.iter().map(|c| c.value.clone()).collect(),
        }
    }

    /// The question as a model's layout reads it.
    #[must_use]
    pub fn to_schema(&self) -> SchemaQuestion {
        let options = |choices: &[Choice]| {
            choices
                .iter()
                .map(|choice| SchemaOption {
                    value: choice.value.clone(),
                    description: choice.description.clone(),
                })
                .collect()
        };
        let (name, wording, answer) = match self {
            Self::Choice {
                name,
                instructions,
                choices,
            } => (name, instructions, SchemaAnswer::Choice(options(choices))),
            Self::YesNo {
                name,
                instructions,
                yes,
                no,
            } => (
                name,
                instructions,
                SchemaAnswer::YesNo {
                    yes: yes.clone(),
                    no: no.clone(),
                },
            ),
            Self::Score {
                name,
                instructions,
                levels,
            } => (name, instructions, SchemaAnswer::Score(options(levels))),
        };
        // A blank wording is no wording: the layout decides what stands in.
        let wording = if wording.trim().is_empty() {
            String::new()
        } else {
            wording.clone()
        };
        SchemaQuestion {
            id: name.clone(),
            wording,
            answer,
        }
    }
}

/// Everything one decision reads.
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRequest {
    /// The state the questions are asked about, verbatim; blank is no state.
    pub input: String,
    /// The questions, in the order their answers come back.
    pub questions: Vec<Question>,
    /// Softmax temperature over the options; `None` is `1.0`, and it must be
    /// positive.
    pub temperature: Option<f32>,
}

/// One question's answer.
#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    /// The question's id.
    pub name: String,
    /// The value of the top option; a tie goes to the earlier option.
    pub choice: String,
    /// Every option's value with its probability, in the caller's order.
    pub probabilities: Vec<(String, f32)>,
    /// How much of the model's distribution the options carry: the label mass
    /// for a readout model, `1.0` for a head, whose options are the whole
    /// distribution.
    pub confidence: f32,
}

#[cfg(test)]
mod tests {
    use shiso_core::SchemaAnswer;

    use super::{Choice, Question};

    #[test]
    fn a_blank_wording_reaches_the_layout_as_empty() {
        let question = Question::Choice {
            name: "q".to_owned(),
            instructions: "  \n".to_owned(),
            choices: vec![Choice {
                value: "a".to_owned(),
                description: Some("first".to_owned()),
            }],
        };
        let schema = question.to_schema();
        assert!(schema.wording.is_empty());
        assert_eq!(schema.id, "q");
        assert!(
            matches!(&schema.answer, SchemaAnswer::Choice(o) if o[0].description.as_deref() == Some("first"))
        );
        let spoken = Question::Score {
            name: "s".to_owned(),
            instructions: " how bad ".to_owned(),
            levels: Vec::new(),
        };
        assert_eq!(spoken.to_schema().wording, " how bad ");
    }

    #[test]
    fn a_yes_no_question_has_the_options_yes_then_no() {
        let question = Question::YesNo {
            name: "q".to_owned(),
            instructions: String::new(),
            yes: None,
            no: None,
        };
        assert_eq!(question.option_values(), vec!["yes", "no"]);
    }
}
