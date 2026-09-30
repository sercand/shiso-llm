//! The question model, the prompt-format version-1 wording, and the shared
//! validators of the decision API.
//!
//! Both wires render a question with the same wording because the answer is
//! scored from the next-token row after exactly these bytes: a wording change
//! is a scoring change. The golden tests in this module pin the version-1
//! bytes, so the wording moves only with new goldens, never with a quiet edit.
//!
//! Validation lives here too so the two surfaces refuse the same bad request
//! for the same reason; each refusal is an [`AppError::bad_request`] naming
//! the field at fault and, wherever one is known, the question by id.

// The two wires and the handler glue that call these names land in later
// tasks, so until then nothing in a non-test build reaches them. The
// expectation says exactly that and is scoped to the non-test build on
// purpose: the tests here already call every one of them.
#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the decision routes and handler glue are these items' callers, in later tasks"
    )
)]

use crate::error::AppError;

/// One option of a choice question.
#[derive(Debug)]
pub(crate) struct ChoiceOption {
    /// The option's name: what follows the letter on its line, and what the
    /// caller reads the answer back as.
    pub name: String,
    /// Extra wording shown after `" - "` when non-blank.
    pub description: Option<String>,
}

/// How a question is answered, which is which block [`prompt_body`] renders.
#[derive(Debug)]
pub(crate) enum Kind {
    /// A lettered list; the answer is one letter.
    Choice {
        /// The line above the options; the line is dropped when blank.
        question: String,
        /// The options, in the order the caller listed them.
        options: Vec<ChoiceOption>,
    },
    /// A list of levels numbered from zero; the answer is one number.
    Score {
        /// The line above the levels; the line is dropped when blank.
        question: String,
        /// The level names, indexed from zero.
        levels: Vec<String>,
    },
    /// A binary judgement; the answer is `yes` or `no`.
    YesNo {
        /// What is being asserted, named on the lead line when non-blank.
        question: String,
        /// Wording for the `yes` label; its line is dropped when blank.
        yes: Option<String>,
        /// Wording for the `no` label; its line is dropped when blank.
        no: Option<String>,
    },
}

/// A question as the wire delivered it, before scoring.
#[derive(Debug)]
pub(crate) struct RawQuestion {
    /// The caller's id for the question; blank when the caller sent none.
    pub id: String,
    /// How the question is answered.
    // Nothing in this module reads it - validation and wording see a `Kind`
    // directly, and the module-level expectation covers the non-test build;
    // drop this one when the glue that dispatches on it arrives.
    #[cfg_attr(
        test,
        expect(
            dead_code,
            reason = "the tests exercise ids, and the decision handler glue that dispatches on a question's kind lands in a later task"
        )
    )]
    pub kind: Kind,
}

/// The text a JSON input value contributes to a prompt.
///
/// A string is itself; objects and arrays render as compact JSON and other
/// scalars as their JSON form. Object keys are sorted because these bytes are
/// scored: the same logical object must render the same bytes however the
/// client happened to order its keys, and the shared parser preserves the
/// order it was sent in, so sorting is done here rather than assumed.
pub(crate) fn render_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(text) => text.clone(),
        other => serde_json::to_string(&canonical(other)).unwrap_or_else(|_| other.to_string()),
    }
}

/// A copy of `value` with every object's keys in sorted order, so serialising
/// it is byte-deterministic.
///
/// Recursion depth is bounded by the parser's own nesting limit: a value this
/// walks was parsed, and parsing is where runaway nesting is refused.
fn canonical(value: &serde_json::Value) -> serde_json::Value {
    use serde_json::Value;
    match value {
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            let mut sorted = serde_json::Map::with_capacity(keys.len());
            for key in keys {
                sorted.insert(key.clone(), canonical(&map[key]));
            }
            Value::Object(sorted)
        }
        Value::Array(items) => Value::Array(items.iter().map(canonical).collect()),
        other => other.clone(),
    }
}

/// The prompt-format version-1 body for `input` and one question: the input's
/// text, one blank line, then the kind's lines, with no trailing newline.
///
/// The exact bytes are pinned by the golden tests below, and `input` is
/// trimmed so the seam between it and the block is always exactly one blank
/// line whatever whitespace the caller padded it with.
pub(crate) fn prompt_body(input: &str, kind: &Kind) -> String {
    let block = match kind {
        Kind::Choice { question, options } => {
            let mut lines = Vec::with_capacity(options.len() + 2);
            if !question.trim().is_empty() {
                lines.push(format!("Question: {question}"));
            }
            for (option, label) in options.iter().zip(choice_labels(options.len())) {
                let mut line = format!("{label}: {}", option.name);
                if let Some(description) = option
                    .description
                    .as_deref()
                    .filter(|text| !text.trim().is_empty())
                {
                    line.push_str(" - ");
                    line.push_str(description);
                }
                lines.push(line);
            }
            lines.push("Answer with the letter of one option only.".to_string());
            lines.join("\n")
        }
        Kind::Score { question, levels } => {
            let mut lines = Vec::with_capacity(levels.len() + 2);
            if !question.trim().is_empty() {
                lines.push(format!("Question: {question}"));
            }
            for (level, label) in levels.iter().zip(score_labels(levels.len())) {
                lines.push(format!("{label}: {level}"));
            }
            lines.push("Answer with the number of one level only.".to_string());
            lines.join("\n")
        }
        Kind::YesNo { question, yes, no } => {
            let mut lines = Vec::with_capacity(4);
            if question.trim().is_empty() {
                lines.push("Is the following true?".to_string());
            } else {
                lines.push(format!("Is the following true? {question}"));
            }
            if let Some(text) = yes.as_deref().filter(|text| !text.trim().is_empty()) {
                lines.push(format!("yes: {text}"));
            }
            if let Some(text) = no.as_deref().filter(|text| !text.trim().is_empty()) {
                lines.push(format!("no: {text}"));
            }
            lines.push("Answer with yes or no only.".to_string());
            lines.join("\n")
        }
    };
    let mut body = String::with_capacity(input.len() + block.len() + 2);
    body.push_str(input.trim());
    body.push_str("\n\n");
    body.push_str(&block);
    body
}

/// The letters `A`.. for `n` options.
///
/// Past 26 the vector stops at `Z` rather than inventing labels:
/// [`check_choice_options`] refuses longer lists, so a request that passed
/// validation never sees a short label vector.
pub(crate) fn choice_labels(n: usize) -> Vec<&'static str> {
    const LETTERS: [&str; 26] = [
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R",
        "S", "T", "U", "V", "W", "X", "Y", "Z",
    ];
    LETTERS.iter().copied().take(n.min(26)).collect()
}

/// The numbers `0`.. for `n` levels, as label strings.
pub(crate) fn score_labels(n: usize) -> Vec<String> {
    (0..n).map(|index| index.to_string()).collect()
}

/// The two labels of a yes/no question.
pub(crate) const YES_NO_LABELS: [&str; 2] = ["yes", "no"];

/// How a refusal names the offending question in its message.
///
/// Not one of the shared names - it only shapes the sentence. A blank id has
/// no name to give; the validator that can refuse a blank id,
/// [`check_ids`], is also the one that knows the position and names it there.
fn question_name(id: &str) -> String {
    if id.trim().is_empty() {
        "this question".to_owned()
    } else {
        format!("question \"{id}\"")
    }
}

/// Refuse a `model` that names an adapter with the `base:adapter` form.
///
/// The server scores against the model it loaded, so the colon form asks for
/// something it cannot load; refusing it beats answering under a model the
/// caller did not ask for.
pub(crate) fn check_model(model: Option<&str>) -> Result<(), AppError> {
    if let Some(model) = model.filter(|name| name.contains(':')) {
        return Err(AppError::bad_request(
            format!(
                "model \"{model}\" uses the adapter form \"base:adapter\", which this server does not support"
            ),
            Some("model"),
            "invalid_request",
        ));
    }
    Ok(())
}

/// Refuse an input that carries nothing.
///
/// The input is the thing being judged; a blank one asks the model to score
/// silence, and every label wins equally.
pub(crate) fn check_input(text: &str) -> Result<(), AppError> {
    if text.trim().is_empty() {
        return Err(AppError::bad_request(
            "input must not be blank",
            Some("input"),
            "invalid_request",
        ));
    }
    Ok(())
}

/// Refuse blank question ids when the surface needs them, and duplicate ids
/// always.
///
/// Whether a blank id is a refusal differs per surface - one addresses
/// answers by id, the other reads them back by position - so `blanks_allowed`
/// says which wire is checking. A duplicate is a lie on either: two answers
/// would return under one name. Blank ids never collide with each other,
/// because neither names anything.
pub(crate) fn check_ids(questions: &[RawQuestion], blanks_allowed: bool) -> Result<(), AppError> {
    for (index, question) in questions.iter().enumerate() {
        if question.id.trim().is_empty() {
            if !blanks_allowed {
                return Err(AppError::bad_request(
                    format!("questions[{index}].id must not be blank"),
                    Some("questions"),
                    "invalid_request",
                ));
            }
        } else if let Some(first) = questions[..index]
            .iter()
            .position(|earlier| earlier.id == question.id)
        {
            return Err(AppError::bad_request(
                format!(
                    "question id \"{}\" is used by questions[{first}] and questions[{index}]",
                    question.id
                ),
                Some("questions"),
                "invalid_request",
            ));
        }
    }
    Ok(())
}

/// Refuse a question line that carries nothing, unless the surface renders
/// without one.
///
/// A blank question drops its line from the version-1 block, so it is only a
/// refusal on a surface that requires every question to state something;
/// `blanks_allowed` says which that is.
pub(crate) fn check_question_text(
    text: &str,
    id: &str,
    blanks_allowed: bool,
) -> Result<(), AppError> {
    if !blanks_allowed && text.trim().is_empty() {
        return Err(AppError::bad_request(
            format!("{}: the question text must not be blank", question_name(id)),
            Some("questions"),
            "invalid_request",
        ));
    }
    Ok(())
}

/// Refuse a choice option list the answer position cannot carry.
///
/// The upper bound is the alphabet - labels run `A..Z`, so a longer list
/// could not be answered by letter - and the lower bound is the surface's
/// `min`, because one wire scores a binary choice and the other a real list.
/// Names must be non-blank (a blank one renders a label pointing at nothing),
/// free of control characters (one inside an option's name breaks the
/// one-option-per-line shape of the block), and unique after trimming and
/// case folding (a collision would score two letters against the same answer
/// text, so one of them could never win).
pub(crate) fn check_choice_options(
    options: &[ChoiceOption],
    id: &str,
    min: usize,
) -> Result<(), AppError> {
    let name = question_name(id);
    if options.len() < min {
        return Err(AppError::bad_request(
            format!(
                "{name}: needs at least {min} options, got {}",
                options.len()
            ),
            Some("questions"),
            "invalid_request",
        ));
    }
    if options.len() > 26 {
        return Err(AppError::bad_request(
            format!(
                "{}: supports at most 26 options, got {}",
                name,
                options.len()
            ),
            Some("questions"),
            "invalid_request",
        ));
    }
    for (index, option) in options.iter().enumerate() {
        let trimmed = option.name.trim();
        if trimmed.is_empty() {
            return Err(AppError::bad_request(
                format!("{name}: option {} must have a non-blank name", index + 1),
                Some("questions"),
                "invalid_request",
            ));
        }
        if option.name.chars().any(char::is_control) {
            return Err(AppError::bad_request(
                format!(
                    "{name}: option name {:?} contains a control character",
                    option.name
                ),
                Some("questions"),
                "invalid_request",
            ));
        }
        let folded = trimmed.to_lowercase();
        if options[..index]
            .iter()
            .any(|earlier| earlier.name.trim().to_lowercase() == folded)
        {
            return Err(AppError::bad_request(
                format!(
                    "{name}: option name {:?} is used more than once",
                    option.name
                ),
                Some("questions"),
                "invalid_request",
            ));
        }
    }
    Ok(())
}

/// Refuse a score level list the answer position cannot carry.
///
/// The upper bound is ten because the labels are single digits, so an
/// eleventh level could not be answered with one number, and the lower bound
/// is the surface's `min`. A blank level is refused outright: it still takes
/// a digit, and a digit pointing at nothing is a level no answer maps to.
pub(crate) fn check_score_levels(levels: &[String], id: &str, min: usize) -> Result<(), AppError> {
    let name = question_name(id);
    if levels.len() < min {
        return Err(AppError::bad_request(
            format!("{name}: needs at least {min} levels, got {}", levels.len()),
            Some("questions"),
            "invalid_request",
        ));
    }
    if levels.len() > 10 {
        return Err(AppError::bad_request(
            format!("{}: supports at most 10 levels, got {}", name, levels.len()),
            Some("questions"),
            "invalid_request",
        ));
    }
    for (index, level) in levels.iter().enumerate() {
        if level.trim().is_empty() {
            return Err(AppError::bad_request(
                format!("{name}: level {} must not be blank", index + 1),
                Some("questions"),
                "invalid_request",
            ));
        }
    }
    Ok(())
}

/// The temperature to score with: the caller's, or `1.0` when none was sent.
///
/// Zero and negatives are refused rather than clamped: the softmax divides
/// by the temperature, so `<= 0` either divides by zero or inverts the
/// ranking - a request that asks for that is a caller bug, not a setting to
/// honour. The guard rejects NaN with them, since NaN is not greater than
/// zero either.
pub(crate) fn check_temperature(t: Option<f64>) -> Result<f64, AppError> {
    match t {
        None => Ok(1.0),
        Some(value) if value > 0.0 => Ok(value),
        Some(value) => Err(AppError::bad_request(
            format!("temperature must be greater than 0, got {value}"),
            Some("temperature"),
            "invalid_request",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn choice(kind_input: &str, q: &str, opts: &[(&str, Option<&str>)]) -> String {
        let options = opts
            .iter()
            .map(|(name, description)| ChoiceOption {
                name: (*name).to_string(),
                description: description.map(ToString::to_string),
            })
            .collect();
        prompt_body(
            kind_input,
            &Kind::Choice {
                question: q.to_string(),
                options,
            },
        )
    }

    fn yes_no(id: &str) -> RawQuestion {
        RawQuestion {
            id: id.to_string(),
            kind: Kind::YesNo {
                question: String::new(),
                yes: None,
                no: None,
            },
        }
    }

    fn option(name: &str) -> ChoiceOption {
        ChoiceOption {
            name: name.to_string(),
            description: None,
        }
    }

    fn options(n: usize) -> Vec<ChoiceOption> {
        (0..n)
            .map(|index| option(&format!("opt {index}")))
            .collect()
    }

    fn levels(n: usize) -> Vec<String> {
        (0..n).map(|index| format!("level {index}")).collect()
    }

    fn refusal(result: Result<(), AppError>) -> String {
        result
            .expect_err("the validator must refuse this input")
            .to_string()
    }

    #[test]
    fn choice_body_has_question_options_and_instruction_in_order() {
        assert_eq!(
            choice(
                "the input",
                "pick one",
                &[("red", None), ("blue", Some("the sky one"))]
            ),
            "the input\n\nQuestion: pick one\nA: red\nB: blue - the sky one\nAnswer with the letter of one option only."
        );
    }

    #[test]
    fn choice_body_drops_the_question_line_when_the_question_is_blank() {
        assert_eq!(
            choice("input", "   ", &[("x", None)]),
            "input\n\nA: x\nAnswer with the letter of one option only."
        );
    }

    #[test]
    fn choice_body_drops_a_blank_description() {
        assert_eq!(
            choice("in", "pick one", &[("x", Some("  "))]),
            "in\n\nQuestion: pick one\nA: x\nAnswer with the letter of one option only."
        );
    }

    #[test]
    fn score_body_numbers_its_levels_from_zero() {
        let kind = Kind::Score {
            question: "how bad".to_string(),
            levels: ["none", "some", "lots"]
                .iter()
                .map(|level| (*level).to_string())
                .collect(),
        };
        assert_eq!(
            prompt_body("in", &kind),
            "in\n\nQuestion: how bad\n0: none\n1: some\n2: lots\nAnswer with the number of one level only."
        );
    }

    #[test]
    fn yes_no_body_leads_with_the_question() {
        let kind = Kind::YesNo {
            question: "it is ready".to_string(),
            yes: Some("confirmed".to_string()),
            no: Some("denied".to_string()),
        };
        assert_eq!(
            prompt_body("in", &kind),
            "in\n\nIs the following true? it is ready\nyes: confirmed\nno: denied\nAnswer with yes or no only."
        );
    }

    #[test]
    fn yes_no_body_is_bare_when_the_question_is_blank() {
        let kind = Kind::YesNo {
            question: String::new(),
            yes: None,
            no: None,
        };
        assert_eq!(
            prompt_body("in", &kind),
            "in\n\nIs the following true?\nAnswer with yes or no only."
        );
    }

    #[test]
    fn body_separates_input_and_block_with_one_blank_line() {
        // An input ending in whitespace keeps its text, and the seam between
        // that text and the block stays exactly one blank line.
        let kind = Kind::YesNo {
            question: String::new(),
            yes: None,
            no: None,
        };
        assert_eq!(
            prompt_body("the input  \n", &kind),
            "the input\n\nIs the following true?\nAnswer with yes or no only."
        );
    }

    #[test]
    fn render_text_passes_strings_and_compacts_containers() {
        assert_eq!(render_text(&json!("plain")), "plain");
        assert_eq!(
            render_text(&json!({"b": 1, "a": [1, 2]})),
            "{\"a\":[1,2],\"b\":1}"
        );
        assert_eq!(render_text(&json!(42)), "42");
    }

    #[test]
    fn labels_are_letters_then_digits_then_yes_no() {
        assert_eq!(choice_labels(3), ["A", "B", "C"]);
        assert_eq!(score_labels(3), ["0", "1", "2"]);
        assert_eq!(YES_NO_LABELS, ["yes", "no"]);
    }

    #[test]
    fn check_model_refuses_the_adapter_colon() {
        assert!(check_model(None).is_ok());
        assert!(check_model(Some("a-model")).is_ok());
        assert!(check_model(Some("base:lora")).is_err());
    }

    #[test]
    fn check_input_refuses_blank() {
        assert!(check_input("hello").is_ok());
        assert!(check_input("").is_err());
        assert!(check_input("   \n").is_err());
    }

    #[test]
    fn check_ids_refuse_blank_unless_allowed_and_duplicates_always() {
        assert!(check_ids(&[yes_no("a"), yes_no("b")], false).is_ok());
        assert!(check_ids(&[yes_no("a"), yes_no("")], false).is_err());
        assert!(check_ids(&[yes_no("a"), yes_no("")], true).is_ok());
        assert!(check_ids(&[yes_no("a"), yes_no("a")], true).is_err());
        // Blank ids do not collide with each other: neither names anything.
        assert!(check_ids(&[yes_no(""), yes_no("")], true).is_ok());
    }

    #[test]
    fn check_choice_options_bounds_two_to_twenty_six() {
        assert!(check_choice_options(&options(26), "q", 2).is_ok());
        assert!(check_choice_options(&options(27), "q", 2).is_err());
        assert!(check_choice_options(&options(1), "q", 1).is_ok());
        assert!(check_choice_options(&options(1), "q", 2).is_err());
    }

    #[test]
    fn check_choice_options_refuse_blank_control_and_duplicate_names() {
        assert!(check_choice_options(&[option(""), option("b")], "q", 2).is_err());
        assert!(check_choice_options(&[option("a\u{7}"), option("b")], "q", 2).is_err());
        // Duplicates compare after trim and case folding.
        assert!(check_choice_options(&[option("Red"), option(" red ")], "q", 2).is_err());
        assert!(check_choice_options(&[option("Red"), option("blue")], "q", 2).is_ok());
    }

    #[test]
    fn check_score_levels_bounds_and_blank() {
        assert!(check_score_levels(&levels(10), "q", 2).is_ok());
        assert!(check_score_levels(&levels(11), "q", 2).is_err());
        assert!(check_score_levels(&levels(1), "q", 1).is_ok());
        assert!(check_score_levels(&levels(1), "q", 2).is_err());
        assert!(check_score_levels(&["ok".to_string(), "  ".to_string()], "q", 2).is_err());
    }

    #[test]
    // `check_temperature` passes the caller's value through unchanged and the
    // default is a constant, so exact equality is the contract here, not an
    // accident of arithmetic.
    #[expect(
        clippy::float_cmp,
        reason = "the temperature is returned as given and the default is a literal, so the exact values are the point"
    )]
    fn check_temperature_defaults_and_refuses_non_positive() {
        assert_eq!(check_temperature(None).expect("the default is valid"), 1.0);
        assert_eq!(
            check_temperature(Some(0.5)).expect("a positive temperature is valid"),
            0.5
        );
        assert!(check_temperature(Some(0.0)).is_err());
        assert!(check_temperature(Some(-2.0)).is_err());
    }

    #[test]
    fn errors_name_the_question_id() {
        // A question that was given an id is named by it.
        let message = refusal(check_question_text("", "q3", false));
        assert!(message.contains("q3"), "{message}");
        // A blank id has nothing to name, so the position in the request
        // identifies the question instead.
        let message = refusal(check_ids(&[yes_no("a"), yes_no("")], false));
        assert!(message.contains("questions[1]"), "{message}");
    }
}
