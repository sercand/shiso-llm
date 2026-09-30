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
//!
//! Below the validators sit the shared preparation: a question rendered to a
//! prompt, its labels proven to be one fresh token each at the prompt's end,
//! and the scored fan-out that turns label log-probabilities into answers.

use std::sync::Arc;

use futures::future::join_all;
use shiso_core::TokenId;
use shiso_ir::{Message, Request, ThinkingDirective};

use crate::api::pipeline::remaining_budget;
use crate::decision::math::{label_mass, softmax};
use crate::error::AppError;
use crate::scheduler::ScoreRequest;
use crate::state::AppState;

/// One option of a choice question.
#[derive(Debug, Clone)]
pub(crate) struct ChoiceOption {
    /// The option's name: what follows the letter on its line, and what the
    /// caller reads the answer back as.
    pub name: String,
    /// Extra wording shown after `" - "` when non-blank.
    pub description: Option<String>,
}

/// How a question is answered, which is which block [`prompt_body`] renders.
#[derive(Debug, Clone)]
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

// --- wire helpers shared by both surfaces -----------------------------------

/// The question line as text: absent is blank.
///
/// Both surfaces spell the question key the same way and render it with the
/// shared rule, so the reading lives here; whether a blank is a refusal or a
/// dropped line is each surface's [`check_question_text`] call, not this
/// function's business.
pub(crate) fn question_text(value: Option<&serde_json::Value>) -> String {
    value.map_or_else(String::new, render_text)
}

/// The `yes`/`no` wording: absent or blank is no wording at all, which drops
/// that line from the block.
pub(crate) fn wording(value: Option<&serde_json::Value>) -> Option<String> {
    let text = value.map(render_text)?;
    (!text.trim().is_empty()).then_some(text)
}

/// Refuses a field the question's type has no place for - `options` on a
/// score question is a caller expecting a different question than the one its
/// `type` says will be scored.
///
/// Each wire calls this with its own struct in hand - the question's id, the
/// type word and the field-presence table are arguments, because the fields
/// belong to the wire - while the refusal itself is shared: the same sentence
/// naming the same field, the same `questions` param, a 400 on both routes.
/// A schema difference between the surfaces (400-versus-422) is about what
/// the *body* says, not about what a misplaced field means to the scored
/// bytes, so this refusal does not split.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the misplaced field and the question.
pub(crate) fn refuse_fields_other_than(
    id: &str,
    kind: &str,
    present: &[(&str, bool)],
    allowed: &[&str],
) -> Result<(), AppError> {
    for (field, present) in present {
        if *present && !allowed.contains(field) {
            return Err(AppError::bad_request(
                format!("question \"{id}\": {field} does not belong on a \"{kind}\" question"),
                Some("questions"),
                "invalid_request",
            ));
        }
    }
    Ok(())
}

// --- preparation and scoring fan-out ---------------------------------------

/// The wire word for a kind, used in answers and in refusals alike.
pub(crate) fn type_name(kind: &Kind) -> &'static str {
    match kind {
        Kind::Choice { .. } => "choice",
        Kind::Score { .. } => "score",
        Kind::YesNo { .. } => "yes_no",
    }
}

/// The label words of a kind, in answer order: the letters, the digits, or
/// `yes` and `no`.
fn labels_for(kind: &Kind) -> Vec<String> {
    match kind {
        Kind::Choice { options, .. } => choice_labels(options.len())
            .into_iter()
            .map(ToString::to_string)
            .collect(),
        Kind::Score { levels, .. } => score_labels(levels.len()),
        Kind::YesNo { .. } => YES_NO_LABELS
            .iter()
            .map(|label| (*label).to_string())
            .collect(),
    }
}

/// The answer text of a choice question: the option name the top label names.
/// A score kind spells its answer as the level number and a yes/no kind as
/// its label, so every kind has a name for its `top` - which indexes that
/// kind's own labels by construction, being the argmax of the probabilities
/// built from exactly those.
pub(crate) fn choice_name(kind: &Kind, top: usize) -> String {
    match kind {
        Kind::Choice { options, .. } => options
            .get(top)
            .map(|option| option.name.clone())
            .unwrap_or_default(),
        Kind::Score { .. } => top.to_string(),
        Kind::YesNo { .. } => answer_label(kind, top),
    }
}

/// The answer label of a yes/no question: `"yes"` for the first label,
/// `"no"` for the second.
pub(crate) fn answer_label(kind: &Kind, top: usize) -> String {
    match kind {
        Kind::YesNo { .. } => YES_NO_LABELS
            .get(top)
            .copied()
            .unwrap_or_default()
            .to_owned(),
        // Not this kind's answer shape. The encoders dispatch on the kind, so
        // this arm means a wrong caller rather than a request outcome, and an
        // empty name says so without a panic path on a scored response.
        _ => String::new(),
    }
}

/// Chat-template knobs a decision request sent, validated and kept raw.
///
/// Every key rule fires once, in [`Self::from_wire`], so the refusal names
/// the offending key while the request is still being read; [`Self::apply`]
/// only lowers what survived.
#[derive(Debug)]
pub(crate) struct TemplateKwargs(serde_json::Map<String, serde_json::Value>);

impl TemplateKwargs {
    /// Accepts exactly three keys: `enable_thinking` (which must say false or
    /// nothing, because scoring never opens a thinking block),
    /// `reasoning_effort` (a spelling this server knows, or nothing), and
    /// `preserve_thinking` (a boolean, or nothing). Every other key is a 400.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the key or the spelling at fault.
    pub(crate) fn from_wire(
        map: Option<serde_json::Map<String, serde_json::Value>>,
    ) -> Result<Self, AppError> {
        let mut checked = serde_json::Map::new();
        for (key, value) in map.unwrap_or_default() {
            match key.as_str() {
                // The answer is the token right after the question block;
                // a thinking block would open at that position and score a
                // reasoning token instead of the label. `false` is accepted
                // because callers that send the key everywhere should not
                // have to special-case this surface.
                "enable_thinking"
                    if matches!(
                        value,
                        serde_json::Value::Bool(false) | serde_json::Value::Null
                    ) => {}
                "enable_thinking" => {
                    return Err(AppError::bad_request(
                        "a decision is scored from one prefill with thinking off; \
                         enable_thinking must be false",
                        Some("chat_template_kwargs"),
                        "invalid_request",
                    ));
                }
                "reasoning_effort" => match &value {
                    serde_json::Value::Null => {}
                    serde_json::Value::String(spelling) => {
                        if shiso_ir::Effort::parse(spelling).is_none() {
                            return Err(AppError::bad_request(
                                format!(
                                    "reasoning_effort {spelling:?} is not a level this server knows"
                                ),
                                Some("chat_template_kwargs"),
                                "invalid_request",
                            ));
                        }
                    }
                    other => {
                        return Err(AppError::bad_request(
                            format!("reasoning_effort must be a string or null, got {other}"),
                            Some("chat_template_kwargs"),
                            "invalid_request",
                        ));
                    }
                },
                "preserve_thinking"
                    if matches!(value, serde_json::Value::Bool(_) | serde_json::Value::Null) => {}
                "preserve_thinking" => {
                    return Err(AppError::bad_request(
                        "preserve_thinking must be a boolean or null",
                        Some("chat_template_kwargs"),
                        "invalid_request",
                    ));
                }
                other => {
                    return Err(AppError::bad_request(
                        format!(
                            "chat_template_kwargs key {other:?} is not accepted on this surface"
                        ),
                        Some("chat_template_kwargs"),
                        "invalid_request",
                    ));
                }
            }
            checked.insert(key, value);
        }
        Ok(Self(checked))
    }

    /// Lowers the surviving knobs onto the request a question is rendered
    /// with.
    pub(crate) fn apply(&self, request: &mut Request) {
        if let Some(serde_json::Value::Bool(keep)) = self.0.get("preserve_thinking") {
            request.preserve_thinking = Some(*keep);
        }
        // `enable_thinking` can only have said false or nothing, which is
        // what every scoring render already does, so it lowers to no change.
        // `reasoning_effort` was parsed and checked and is deliberately not
        // lowered: scoring renders with `ThinkingDirective::Off`, so no rung
        // would ever reach a template. A caller that named one gets the same
        // answer as one that did not, and the check exists to refuse a typo,
        // not to promise a thinking budget this surface cannot spend.
    }
}

/// A question rendered, tokenized and label-checked: everything a scoring
/// pass needs and nothing else.
#[derive(Debug)]
pub(crate) struct PreparedQuestion {
    /// The caller's id for the question.
    pub id: String,
    /// How the question is answered.
    pub kind: Kind,
    /// The rendered prompt ids, in the order the model will read them.
    pub prompt_tokens: Vec<TokenId>,
    /// The label ids, in the question's own label order.
    pub labels: Vec<TokenId>,
}

/// Renders one question over the shared input and proves the labels are
/// scorable: one token each, no two the same token, and each one a fresh
/// token at the very end of the rendered prompt.
///
/// The three label proofs are what make the answer mean what it says: a label
/// that split into two tokens could not be read from one row, two labels that
/// reached the same token could never be told apart, and a label that only
/// re-tokenized the prompt's last chunk would score the wrong position.
///
/// `want_token_ids` is not needed to prepare - the id lists are assembled
/// with the answer - but the signature keeps it so both surfaces of the
/// decision family drive preparation with the one flag they read from their
/// wires.
///
/// # Errors
///
/// [`AppError::NoModel`] / [`AppError::NotReady`] for a server that cannot
/// render, [`AppError::BadRequest`] for a conversation the template refuses
/// and for every failed proof above (naming the question), and
/// [`AppError::ContextOverflow`] with `param` `"input"` when the rendered
/// prompt leaves no room to read an answer.
pub(crate) fn prepare_question(
    state: &AppState,
    input: &str,
    question: &RawQuestion,
    kwargs: &TemplateKwargs,
    _want_token_ids: bool,
) -> Result<PreparedQuestion, AppError> {
    // The same three refusals `pipeline::start` makes, in the same shapes:
    // no engine is "no model is loaded", and no renderer is "not ready".
    state.engine()?;
    let renderer = state.renderer.as_ref().ok_or(AppError::NotReady {
        what: "a prompt renderer",
    })?;
    let vocabulary = renderer.vocabulary();

    let mut request = Request::new(
        state.model_id.clone(),
        vec![Message::user(prompt_body(input, &question.kind))],
    );
    // Scoring reads the first row after the question block. A thinking block
    // would open *at* the answer position and the row would score a reasoning
    // token, so Off is not a preference here - it is what the answer is.
    request.thinking = ThinkingDirective::Off;
    kwargs.apply(&mut request);
    let prompt = renderer.render(&request)?;

    let name = question_name(&question.id);
    let plain = vocabulary
        .encode(&prompt.text, false)
        .map_err(|error| AppError::internal("re-encoding the rendered prompt", error))?;
    // The renderer's ids are this re-encode plus, when the checkpoint has an
    // opening token and the template did not write it, that token in front.
    // Anything else means the text the engine would score is not the text
    // that was checked, and no answer from it means what the labels say.
    let expected: &[TokenId] = match (vocabulary.bos(), prompt.tokens.split_first()) {
        (Some(bos), Some((first, rest))) if *first == bos && plain.first() != Some(&bos) => rest,
        _ => &prompt.tokens,
    };
    if plain != expected {
        return Err(AppError::bad_request(
            format!("{name}: the rendered prompt did not survive re-encoding"),
            Some("questions"),
            "invalid_request",
        ));
    }

    let mut labels: Vec<TokenId> = Vec::new();
    let mut words: Vec<String> = Vec::new();
    for label in labels_for(&question.kind) {
        let encoded = vocabulary
            .encode(&label, false)
            .map_err(|error| AppError::internal("encoding a label", error))?;
        let [only] = encoded.as_slice() else {
            return Err(AppError::bad_request(
                format!(
                    "{name}: the label \"{label}\" is not one token in this server's vocabulary"
                ),
                Some("questions"),
                "invalid_request",
            ));
        };
        if let Some(at) = labels.iter().position(|earlier| earlier == only) {
            return Err(AppError::bad_request(
                format!(
                    "{name}: the labels {:?} and {:?} are the same token in this server's vocabulary",
                    words[at], label
                ),
                Some("questions"),
                "invalid_request",
            ));
        }
        // The answer position reads one token: the label must follow the
        // rendered prompt as exactly that token, not re-tokenize together
        // with whatever the template ended with.
        let mut following = plain.clone();
        following.push(*only);
        let joined = vocabulary
            .encode(&format!("{}{}", prompt.text, label), false)
            .map_err(|error| AppError::internal("encoding a label after the prompt", error))?;
        if joined != following {
            return Err(AppError::bad_request(
                format!(
                    "{name}: the label \"{label}\" does not follow the rendered prompt as one fresh token"
                ),
                Some("questions"),
                "invalid_request",
            ));
        }
        labels.push(*only);
        words.push(label);
    }

    let prompt_tokens = u64::try_from(prompt.tokens.len()).unwrap_or(u64::MAX);
    // The scoring pass answers at the prompt's last row, so it needs the
    // prompt to leave room the same way a generation does - the shared
    // boundary, blamed on the field this surface carries.
    remaining_budget(state.ctx, prompt_tokens).ok_or(AppError::ContextOverflow {
        param: "input",
        prompt_tokens,
        ctx: state.ctx,
    })?;
    Ok(PreparedQuestion {
        id: question.id.clone(),
        kind: question.kind.clone(),
        prompt_tokens: prompt.tokens,
        labels,
    })
}

/// One question's scored answer, before any wire shape.
#[derive(Debug)]
pub(crate) struct Answered {
    /// The caller's id for the question.
    pub id: String,
    /// How the question was answered.
    pub kind: Kind,
    /// The label probabilities at the answer position, in label order.
    pub probabilities: Vec<f64>,
    /// The probability mass the labels carry together, independent of the
    /// temperature.
    pub label_mass: f64,
    /// The top label: the first argmax of `probabilities`, so a tie goes to
    /// the earlier label.
    pub top: usize,
    /// Tokens this question's prompt occupied.
    pub prompt_tokens: u64,
    /// The prompt's ids, when the caller asked for them.
    pub prompt_token_ids: Option<Vec<u32>>,
    /// The label ids, when the caller asked for them.
    pub label_token_ids: Option<Vec<u32>>,
}

/// Scores every prepared question at once - one prefill each - and folds each
/// answer's label log-probabilities into probabilities, mass and a top.
///
/// All questions ride concurrently because each waits on the engine, not on
/// the caller's CPU; and one failing question fails the whole request, because
/// a caller that asked five questions and got four answers and an error would
/// have to guess whether the four are complete.
///
/// # Errors
///
/// Whatever [`crate::scheduler::EngineHandle::score`] answers with for any
/// question: [`AppError::Overloaded`] when the queue stayed full, and the
/// engine's refusals.
pub(crate) async fn run_questions(
    state: &Arc<AppState>,
    prepared: Vec<PreparedQuestion>,
    temperature: f64,
    want_token_ids: bool,
) -> Result<Vec<Answered>, AppError> {
    let engine = state.engine()?;
    let scored = join_all(prepared.into_iter().map(|question| {
        let scored = engine.score(ScoreRequest::new(
            Arc::from(question.prompt_tokens.clone()),
            Arc::from(question.labels.clone()),
        ));
        async move {
            let outcome = scored.await?;
            let probabilities = softmax(&outcome.label_logprobs, temperature);
            // `>` and never `>=`: the first argmax wins, so on a dead heat
            // the first option and `yes` win by position rather than by coin.
            let top = probabilities
                .iter()
                .enumerate()
                .fold((0usize, f64::NEG_INFINITY), |best, (index, p)| {
                    if *p > best.1 { (index, *p) } else { best }
                })
                .0;
            let prompt_tokens = u64::try_from(question.prompt_tokens.len()).unwrap_or(u64::MAX);
            let ids = want_token_ids.then(|| {
                (
                    question
                        .prompt_tokens
                        .iter()
                        .map(|token| token.get())
                        .collect::<Vec<u32>>(),
                    question
                        .labels
                        .iter()
                        .map(|token| token.get())
                        .collect::<Vec<u32>>(),
                )
            });
            Ok(Answered {
                id: question.id,
                kind: question.kind,
                probabilities,
                label_mass: label_mass(&outcome.label_logprobs),
                top,
                prompt_tokens,
                prompt_token_ids: ids.as_ref().map(|(prompt, _)| prompt.clone()),
                label_token_ids: ids.map(|(_, labels)| labels),
            })
        }
    }))
    .await;
    scored.into_iter().collect()
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
