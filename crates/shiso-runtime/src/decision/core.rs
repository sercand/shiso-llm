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
use shiso_core::media::MediaDigest;
use shiso_core::{StateSpelling, TokenId};
use shiso_ir::{Block, MediaSource, Message, Request, Role, ThinkingDirective};

use shiso_ir::decision::{
    Answer, AnswerKind, Choice, ChoiceValue, Input, Level, Part, Question, QuestionKind,
    ReadoutState, Request as DecisionRequest, Response as DecisionResponse, Usage,
};

use crate::api::pipeline::remaining_budget;
use crate::decision::head;
use crate::decision::math::{
    choice_confidence, expected_score, label_mass, score_confidence, softmax,
};
use crate::decision::readout;
use crate::error::AppError;
use crate::media::MediaItem;
use crate::scheduler::ScoreRequest;
use crate::state::AppState;
use crate::tokenizer::{AddedSeam, Vocabulary};

/// The question's block: the lines that answer-position the prompt, without
/// the input or the blank line that precedes them.
///
/// The exact bytes are pinned by the golden tests below, through
/// [`prompt_body`] and, on the blocks path, through the whitespace goldens in
/// `tests/decisions.rs`.
///
/// `labels` are the question's label words from [`labels_for`]; a choice block
/// letters its options with them, the other kinds spell their own.
#[must_use]
pub fn question_block(question: &Question, labels: &[String]) -> String {
    let wording = question.instructions.as_str();
    match &question.kind {
        QuestionKind::Choice(options) => {
            let mut lines = Vec::with_capacity(options.len() + 2);
            if !wording.trim().is_empty() {
                lines.push(format!("Question: {wording}"));
            }
            for (option, label) in options.iter().zip(labels) {
                let mut line = format!("{label}: {}", option.value.text());
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
            // A list within the alphabet keeps the version-1 wording byte for
            // byte; only a wide list, whose labels are two letters, says so.
            if options.len() <= LETTERS.len() {
                lines.push("Answer with the letter of one option only.".to_string());
            } else {
                lines.push("Answer with the two-letter label of one option only.".to_string());
            }
            lines.join("\n")
        }
        QuestionKind::Score(levels) => {
            let mut lines = Vec::with_capacity(levels.len() + 2);
            if !wording.trim().is_empty() {
                lines.push(format!("Question: {wording}"));
            }
            for (level, label) in levels.iter().zip(score_labels(levels.len())) {
                let mut line = format!("{label}: {}", level.label);
                if let Some(description) = level
                    .description
                    .as_deref()
                    .filter(|text| !text.trim().is_empty())
                {
                    line.push_str(" - ");
                    line.push_str(description);
                }
                lines.push(line);
            }
            lines.push("Answer with the number of one level only.".to_string());
            lines.join("\n")
        }
        QuestionKind::YesNo { yes, no } => {
            let mut lines = Vec::with_capacity(4);
            if wording.trim().is_empty() {
                lines.push("Is the following true?".to_string());
            } else {
                lines.push(format!("Is the following true? {wording}"));
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
    }
}

/// The prompt-format version-1 body for `input` and one question: the input's
/// text, one blank line, then the question's lines, with no trailing newline.
///
/// The exact bytes are pinned by the golden tests below, and `input` is
/// trimmed so the seam between it and the block is always exactly one blank
/// line whatever whitespace the caller padded it with.
#[must_use]
pub fn prompt_body(input: &str, question: &Question, labels: &[String]) -> String {
    let block = question_block(question, labels);
    let mut body = String::with_capacity(input.len() + block.len() + 2);
    body.push_str(input.trim());
    body.push_str("\n\n");
    body.push_str(&block);
    body
}

/// The single letters a list within the alphabet is labelled with.
const LETTERS: [&str; 26] = [
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S",
    "T", "U", "V", "W", "X", "Y", "Z",
];

/// The most options one choice question takes: the System One wire's own
/// bound. Any vocabulary that spells most two-letter pairs as one token
/// labels a list this long; one that cannot is refused at preparation.
pub const MAX_CHOICE_OPTIONS: usize = 255;

/// The labels for `n` options: the letters `A`.. when `n` fits the alphabet,
/// otherwise two letters each - `AA`, `AB`, .. - keeping only the pairs
/// `one_token` accepts, in that order.
///
/// A wide list is two letters throughout rather than `A`..`Z` then `AA`..:
/// the answer is read from one row, and a model spelling `AB` as `A`, `B`
/// would put that answer's mass on `A`. With no single-letter label in the
/// list, such a spelling is lost mass, never a different answer. Pairs the
/// vocabulary splits are skipped because a split label cannot be read from
/// one row at all.
///
/// The vector is shorter than `n` when the vocabulary runs out of one-token
/// pairs; `prepare_question` refuses that list by name.
pub fn choice_labels(n: usize, one_token: impl Fn(&str) -> bool) -> Vec<String> {
    if n <= LETTERS.len() {
        return LETTERS.iter().take(n).map(ToString::to_string).collect();
    }
    LETTERS
        .iter()
        .flat_map(|first| LETTERS.iter().map(move |second| format!("{first}{second}")))
        .filter(|pair| one_token(pair))
        .take(n)
        .collect()
}

/// The numbers `0`.. for `n` levels, as label strings.
#[must_use]
pub fn score_labels(n: usize) -> Vec<String> {
    (0..n).map(|index| index.to_string()).collect()
}

/// The two labels of a yes/no question.
pub const YES_NO_LABELS: [&str; 2] = ["yes", "no"];

/// How a refusal names the offending question in its message.
///
/// Not one of the shared names - it only shapes the sentence. A blank id has
/// no name to give; the validator that can refuse a blank id,
/// [`check_ids`], is also the one that knows the position and names it there.
#[must_use]
pub fn question_name(id: &str) -> String {
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
///
/// # Errors
///
/// [`AppError::BadRequest`] naming `model` when it uses the adapter form.
pub fn check_model(model: Option<&str>) -> Result<(), AppError> {
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
///
/// # Errors
///
/// [`AppError::BadRequest`] naming `input` when it is blank.
pub fn check_input(text: &str) -> Result<(), AppError> {
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
///
/// # Errors
///
/// [`AppError::BadRequest`] naming `questions` for a blank id where blanks
/// are not allowed, and for any id used twice.
pub fn check_ids(questions: &[Question], blanks_allowed: bool) -> Result<(), AppError> {
    for (index, question) in questions.iter().enumerate() {
        if question.name.trim().is_empty() {
            if !blanks_allowed {
                return Err(AppError::bad_request(
                    format!("questions[{index}].id must not be blank"),
                    Some("questions"),
                    "invalid_request",
                ));
            }
        } else if let Some(first) = questions[..index]
            .iter()
            .position(|earlier| earlier.name == question.name)
        {
            return Err(AppError::bad_request(
                format!(
                    "question \"{}\" is used by questions[{first}] and questions[{index}]",
                    question.name
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
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the question when its text is blank
/// where blanks are not allowed.
pub fn check_question_text(text: &str, id: &str, blanks_allowed: bool) -> Result<(), AppError> {
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
/// The upper bound is [`MAX_CHOICE_OPTIONS`] - whether this server's
/// vocabulary can label that many is proven later, at preparation, where the
/// vocabulary is in hand - and the lower bound is the surface's
/// `min`, because one wire scores a binary choice and the other a real list.
/// String values must be non-blank (a blank one renders a label pointing at
/// nothing) and free of control characters (one inside an option's value
/// breaks the one-option-per-line shape of the block); values must be unique
/// within their type after trimming and case folding (a collision would score
/// two letters against the same answer text, so one of them could never win),
/// while a string and a boolean with the same text stay distinct.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the question for a list outside its
/// bounds, a blank or control-character value, or a repeated value.
pub fn check_choice_options(options: &[Choice], id: &str, min: usize) -> Result<(), AppError> {
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
    if options.len() > MAX_CHOICE_OPTIONS {
        return Err(AppError::bad_request(
            format!(
                "{name}: supports at most {MAX_CHOICE_OPTIONS} options per choice, got {}",
                options.len()
            ),
            Some("questions"),
            "invalid_request",
        ));
    }
    for (index, option) in options.iter().enumerate() {
        if let ChoiceValue::Str(text) = &option.value {
            if text.trim().is_empty() {
                return Err(AppError::bad_request(
                    format!("{name}: option {} must have a non-blank value", index + 1),
                    Some("questions"),
                    "invalid_request",
                ));
            }
            if text.chars().any(char::is_control) {
                return Err(AppError::bad_request(
                    format!("{name}: option value {text:?} contains a control character"),
                    Some("questions"),
                    "invalid_request",
                ));
            }
        }
        // Duplicates are typed: a string and a boolean with the same text
        // are distinct options, and only same-type values compare (strings
        // after trim and case folding).
        let duplicate =
            options[..index]
                .iter()
                .any(|earlier| match (&earlier.value, &option.value) {
                    (ChoiceValue::Str(a), ChoiceValue::Str(b)) => {
                        a.trim().to_lowercase() == b.trim().to_lowercase()
                    }
                    (ChoiceValue::Bool(a), ChoiceValue::Bool(b)) => a == b,
                    _ => false,
                });
        if duplicate {
            return Err(AppError::bad_request(
                format!(
                    "{name}: option value {:?} is used more than once",
                    option.value.text()
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
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the question for a list outside its
/// bounds or a blank level.
pub fn check_score_levels(levels: &[Level], id: &str, min: usize) -> Result<(), AppError> {
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
        if level.label.trim().is_empty() {
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
///
/// # Errors
///
/// [`AppError::BadRequest`] naming `temperature` when it is not greater
/// than zero.
pub fn check_temperature(t: Option<f64>) -> Result<f64, AppError> {
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

// --- preparation and scoring fan-out ---------------------------------------

/// The label words of a kind, in answer order: the letters (or, for a wide
/// list, the one-token letter pairs `one_token` accepts), the digits, or
/// `yes` and `no`.
pub fn labels_for(kind: &QuestionKind, one_token: impl Fn(&str) -> bool) -> Vec<String> {
    match kind {
        QuestionKind::Choice(options) => choice_labels(options.len(), one_token),
        QuestionKind::Score(levels) => score_labels(levels.len()),
        QuestionKind::YesNo { .. } => YES_NO_LABELS
            .iter()
            .map(|label| (*label).to_string())
            .collect(),
    }
}

/// A question rendered, tokenized and label-checked: everything a scoring
/// pass needs and nothing else.
#[derive(Debug)]
pub(crate) struct PreparedQuestion {
    /// The question that was prepared: its name, kind and whether it answers
    /// as the yes probability, all carried to the answer.
    pub question: Question,
    /// The rendered prompt ids, in the order the model will read them.
    pub prompt_tokens: Vec<TokenId>,
    /// The label ids, in the question's own label order.
    pub labels: Vec<TokenId>,
    /// The decoded media the prompt refers to, in prompt order. Empty for a
    /// text prompt - the same fact the scheduler reads to decide whether the
    /// encode step runs.
    pub media: Arc<[MediaItem]>,
    /// One digest per item of `media`, in the same order.
    pub media_digests: Arc<[MediaDigest]>,
}

/// The IR block an input picture is decoded from.
fn image_block(source: &MediaSource) -> Block {
    Block::Image(source.clone())
}

/// The IR block of one part of the caller's input.
pub(crate) fn part_block(part: &Part) -> Block {
    match part {
        Part::Text(text) => Block::Text(text.clone()),
        Part::Image(source) => image_block(source),
    }
}

/// The IR message a content-block `input` becomes: the caller's blocks in
/// order, then one final text part carrying the question block.
///
/// Text parts are verbatim - the template concatenates parts with no
/// delimiter, so the caller's spacing is the spacing. The final part's
/// whitespace is pinned (spec §2): the blank line the string path puts
/// between input and block survives only when the caller's last block is
/// text; after a picture the pad run already ends the text side, and the
/// block follows plain.
fn blocks_message(blocks: &[Part], question: &Question, labels: &[String]) -> Message {
    let mut parts: Vec<Block> = blocks.iter().map(part_block).collect();
    let wording = question_block(question, labels);
    let question_text = if matches!(blocks.last(), Some(Part::Text(_))) {
        format!("\n\n{wording}")
    } else {
        wording
    };
    parts.push(Block::Text(question_text));
    Message::new(Role::User, parts)
}

/// The kind's labels under `vocabulary`, refusing a choice list it cannot
/// label: a wide list keeps only the pairs this vocabulary spells as one
/// token, so it can come up short of the options it must label.
fn vocabulary_labels(
    vocabulary: &Vocabulary,
    kind: &QuestionKind,
    name: &str,
) -> Result<Vec<String>, AppError> {
    let one_token = |label: &str| {
        vocabulary
            .encode(label, false)
            .is_ok_and(|ids| ids.len() == 1)
    };
    let labels = labels_for(kind, one_token);
    if let QuestionKind::Choice(options) = kind
        && labels.len() < options.len()
    {
        return Err(AppError::bad_request(
            format!(
                "{name}: this server's vocabulary labels at most {} options per choice, got {}",
                labels.len(),
                options.len()
            ),
            Some("questions"),
            "invalid_request",
        ));
    }
    Ok(labels)
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
/// `input` is the decisions wire's split: the text arm renders today's
/// version-1 body byte for byte, the blocks arm renders the caller's parts
/// with the question block appended per the spec's whitespace rule. A media
/// prompt skips the full-text re-encode (its text spells one marker where its
/// ids hold a pad run) and proves the seam against the tail after the last
/// span instead.
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
    input: &Input,
    question: &Question,
    preserve_thinking: Option<bool>,
) -> Result<PreparedQuestion, AppError> {
    // The same three refusals `pipeline::start` makes, in the same shapes:
    // no engine is "no model is loaded", and no renderer is "not ready".
    state.engine()?;
    let renderer = state.renderer.as_ref().ok_or(AppError::NotReady {
        what: "a prompt renderer",
    })?;
    let vocabulary = renderer.vocabulary();
    let name = question_name(&question.name);

    // The labels are chosen before the prompt is rendered because the prompt
    // spells them; the proofs below still check each one against the
    // rendered tail.
    let label_words = vocabulary_labels(vocabulary, &question.kind, &name)?;

    let message = match input {
        Input::Text(text) => Message::user(prompt_body(text, question, &label_words)),
        Input::Blocks(blocks) => blocks_message(blocks, question, &label_words),
        Input::Pictured { media, text } if media.is_empty() => {
            Message::user(prompt_body(text, question, &label_words))
        }
        // Composition (spec §3): images before the state, clef's rule. The
        // state enters as a *trimmed* text block, which is what keeps the
        // question-wording seam (`"\n\n" + wording` after a text block)
        // byte-identical to the string path's `trim() + "\n\n" + wording`.
        // A blank state sends no text block at all: image-only is valid.
        Input::Pictured { media, text } => {
            let mut blocks: Vec<Part> = media.iter().cloned().map(Part::Image).collect();
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                blocks.push(Part::Text(trimmed.to_owned()));
            }
            blocks_message(&blocks, question, &label_words)
        }
    };
    let mut request = Request::new(state.model_id.clone(), vec![message]);
    // Scoring reads the first row after the question block. A thinking block
    // would open *at* the answer position and the row would score a reasoning
    // token, so Off is not a preference here - it is what the answer is.
    request.thinking = ThinkingDirective::Off;
    if preserve_thinking.is_some() {
        request.preserve_thinking = preserve_thinking;
    }
    let prompt = renderer.render(&request)?;

    // The seam the answer is read at: the prompt's tail text and the ids
    // beside it. Without media the tail is the whole prompt, and the whole
    // prompt must first survive re-encoding - the text the caller is told
    // about and the ids the engine reads must be one object. With media the
    // text spells one marker where the ids hold a whole pad run, so the
    // full-text re-encode cannot match by construction and is skipped; the
    // tail after the last span is where text and ids agree again, and the
    // proof needs only the spans - `core` never sees a marker string.
    let (tail_text, tail_tokens, seam): (&str, Vec<TokenId>, Option<AddedSeam>) = if prompt
        .media
        .is_empty()
    {
        let (plain, seam) = vocabulary
            .encode_with_seam(&prompt.text)
            .map_err(|error| AppError::internal("re-encoding the rendered prompt", error))?;
        // The renderer's ids are this re-encode plus, when the checkpoint has
        // an opening token and the template did not write it, that token in
        // front. Anything else means the text the engine would score is not
        // the text that was checked, and no answer from it means what the
        // labels say.
        let expected: &[TokenId] = match (vocabulary.bos(), prompt.tokens.split_first()) {
            (Some(bos), Some((first, rest))) if *first == bos && plain.first() != Some(&bos) => {
                rest
            }
            _ => &prompt.tokens,
        };
        if plain != expected {
            return Err(AppError::bad_request(
                format!("{name}: the rendered prompt did not survive re-encoding"),
                Some("questions"),
                "invalid_request",
            ));
        }
        (prompt.text.as_str(), plain, seam)
    } else {
        let last = prompt
            .media_spans
            .last()
            .expect("a prompt that rendered media reports its spans");
        let tail_text = &prompt.text[last.text.end..];
        let tail_tokens = prompt.tokens[last.tokens.end..].to_vec();
        // The tail's ids were never re-encoded here, so its encoding is
        // taken once to find the seam, and the seam is used only when the
        // ids in front of it are the tail's own: otherwise every label
        // takes the whole-tail check below, which refuses it as before.
        let (encoded, seam) = vocabulary
            .encode_with_seam(tail_text)
            .map_err(|error| AppError::internal("encoding the prompt's tail", error))?;
        let seam = seam.filter(|seam| encoded.get(..seam.tokens) == tail_tokens.get(..seam.tokens));
        (tail_text, tail_tokens, seam)
    };

    let labels = prove_labels(
        vocabulary,
        &name,
        label_words,
        &LabelSeam {
            text: tail_text,
            tokens: &tail_tokens,
            seam: seam.as_ref(),
        },
    )?;

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
        question: question.clone(),
        prompt_tokens: prompt.tokens,
        labels,
        media: Arc::from(prompt.media),
        media_digests: Arc::from(prompt.digests),
    })
}

/// A prompt's tail, as [`prove_labels`] checks labels against it.
struct LabelSeam<'a> {
    /// The tail's text: the whole prompt without media, the text after the
    /// last item's marker with it.
    text: &'a str,
    /// The tail's ids.
    tokens: &'a [TokenId],
    /// Where the tail's encoding last restarted, when it did and the ids in
    /// front of that point are the tail's own.
    seam: Option<&'a AddedSeam>,
}

/// Encodes every label, and proves each one is one token of its own that
/// follows the prompt's tail as exactly that token.
///
/// The proof per label is that the tail's text with the label appended
/// encodes as the tail's ids with the label's id appended. Done literally,
/// that re-encodes the whole tail once per label - a 255-option question
/// over a long input is hundreds of whole-prompt encodes. With a seam it
/// encodes only the stretch after the tail's last added token, plus the
/// label: the encoder splits added tokens out first and encodes every stretch
/// between them alone, so the ids in front of the seam are the same whatever
/// follows it (and are the tail's own, which the caller checked), and the ids
/// after it are what this stretch encodes to. The one way an appended label
/// could reach behind the seam, completing an added token spelled across it,
/// is ruled out per label by [`AddedSeam::holds_after`], and a label it does
/// not clear takes the whole-tail check instead. Either way the answer
/// is the literal check's: same labels accepted, same ids, same refusals.
fn prove_labels(
    vocabulary: &Vocabulary,
    name: &str,
    label_words: Vec<String>,
    tail: &LabelSeam<'_>,
) -> Result<Vec<TokenId>, AppError> {
    let mut labels: Vec<TokenId> = Vec::new();
    let mut words: Vec<String> = Vec::new();
    for label in label_words {
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
        // prompt's tail as exactly that token, not re-tokenize together with
        // whatever the tail ended with.
        let (text, ids) = match tail.seam.filter(|seam| seam.holds_after(tail.text, &label)) {
            Some(seam) => (&tail.text[seam.text..], &tail.tokens[seam.tokens..]),
            None => (tail.text, tail.tokens),
        };
        let mut following = ids.to_vec();
        following.push(*only);
        let joined = vocabulary
            .encode(&format!("{text}{label}"), false)
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
    Ok(labels)
}

/// One question's scored answer, before any wire shape.
#[derive(Debug)]
pub(crate) struct Answered {
    /// The question this answers: its name, kind and whether it reports the
    /// yes probability rather than a label word.
    pub question: Question,
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
    /// Of `prompt_tokens`, those a stored prefix covered: a sibling
    /// question's checkpoint or an earlier request's.
    pub cached_tokens: u64,
    /// The prompt's ids, when the caller asked for them.
    pub prompt_token_ids: Option<Vec<u32>>,
    /// The label ids, when the caller asked for them.
    pub label_token_ids: Option<Vec<u32>>,
}

/// What a request's prompts cost: the tokens `usage` counts as input, and how
/// many of those a stored prefix covered rather than a pass reading them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct InputTally {
    /// The input side of `usage`.
    pub input: u64,
    /// Of `input`, the tokens resumed from a checkpoint. Never more than
    /// `input`.
    pub cached: u64,
}

impl InputTally {
    /// Every question's whole prompt counted, as the label route reads them:
    /// one prefill each, so a sibling standing on the first question's
    /// checkpoint is cached tokens of its own prompt.
    pub(crate) fn of(answers: &[Answered]) -> Self {
        let input: u64 = answers.iter().map(|answer| answer.prompt_tokens).sum();
        let cached: u64 = answers.iter().map(|answer| answer.cached_tokens).sum();
        Self {
            input,
            cached: cached.min(input),
        }
    }
}

/// How many leading prompt tokens every question of a request shares, zero
/// for a request of one question.
///
/// The questions render the same input ahead of their own blocks, so this is
/// roughly the input's length - and the point the scheduler checkpoints the
/// first question it reads, so every later one reads only its own block.
///
/// Questions over pictures share only when every one carries the same
/// pictures, by digest: a pad run is one id whichever picture it is, so equal
/// ids say something about equal rows only when the pictures are the same.
/// The scheduler then files the shared state under the pictures' content key,
/// and a model that cannot key it keeps every question's read whole.
fn shared_prefix(prepared: &[PreparedQuestion]) -> u32 {
    let [first, rest @ ..] = prepared else {
        return 0;
    };
    if rest.is_empty()
        || rest
            .iter()
            .any(|question| question.media_digests != first.media_digests)
    {
        return 0;
    }
    let common = rest
        .iter()
        .fold(first.prompt_tokens.len(), |common, question| {
            first.prompt_tokens[..common]
                .iter()
                .zip(&question.prompt_tokens)
                .take_while(|(a, b)| a == b)
                .count()
        });
    u32::try_from(common).unwrap_or(u32::MAX)
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
    let shared = shared_prefix(&prepared);
    let scored = join_all(prepared.into_iter().map(|question| {
        // Always through `with_media`: empty arcs are a refcount pair and
        // the scheduler's encode step skips on an empty list, so a text
        // question's request is the same request it was before.
        let scored = engine.score(
            ScoreRequest::new(
                Arc::from(question.prompt_tokens.clone()),
                Arc::from(question.labels.clone()),
            )
            .with_media(question.media.clone(), question.media_digests.clone())
            .with_shared_prefix(shared),
        );
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
                question: question.question,
                probabilities,
                label_mass: label_mass(&outcome.label_logprobs),
                top,
                prompt_tokens,
                cached_tokens: outcome.cached_tokens.min(prompt_tokens),
                prompt_token_ids: ids.as_ref().map(|(prompt, _)| prompt.clone()),
                label_token_ids: ids.map(|(_, labels)| labels),
            })
        }
    }))
    .await;
    scored.into_iter().collect()
}

impl Answered {
    /// The IR answer, with the token ids that were asked for.
    pub(crate) fn into_answer(self) -> Answer {
        let Self {
            question,
            probabilities,
            label_mass,
            top,
            prompt_token_ids,
            label_token_ids,
            ..
        } = self;
        Answer {
            prompt_token_ids,
            label_token_ids,
            ..scored_answer(question, probabilities, label_mass, top)
        }
    }
}

/// The IR answer to `question` from its scored label `probabilities`: the
/// question's options paired with them, and the numbers a wire spells out
/// derived once, here, so the two surfaces cannot disagree about them.
///
/// `top` indexes `probabilities` (the first argmax) and `label_mass` is the
/// mass the labels carried before the softmax. Token ids are left unset.
#[must_use]
pub fn scored_answer(
    question: Question,
    probabilities: Vec<f64>,
    label_mass: f64,
    top: usize,
) -> Answer {
    let (kind, values, choice) = match question.kind {
        QuestionKind::Choice(options) => {
            let values: Vec<ChoiceValue> = options.into_iter().map(|option| option.value).collect();
            let choice = values
                .get(top)
                .cloned()
                .unwrap_or_else(|| ChoiceValue::Str(String::new()));
            (AnswerKind::Choice, values, choice)
        }
        QuestionKind::Score(levels) => {
            let values: Vec<ChoiceValue> = levels
                .into_iter()
                .map(|level| ChoiceValue::Str(level.label))
                .collect();
            let choice = values
                .get(top)
                .cloned()
                .unwrap_or_else(|| ChoiceValue::Str(String::new()));
            (AnswerKind::Score, values, choice)
        }
        QuestionKind::YesNo { .. } => (
            AnswerKind::YesNo,
            YES_NO_LABELS
                .iter()
                .map(|label| ChoiceValue::Str((*label).to_owned()))
                .collect(),
            ChoiceValue::Str(
                YES_NO_LABELS
                    .get(top)
                    .copied()
                    .unwrap_or_default()
                    .to_owned(),
            ),
        ),
    };
    let confidence = match kind {
        AnswerKind::Score => score_confidence(&probabilities),
        AnswerKind::Choice | AnswerKind::YesNo => choice_confidence(&probabilities),
    };
    Answer {
        name: question.name,
        kind,
        probability_answer: question.probability_answer,
        choice,
        confidence,
        label_mass,
        level: (kind == AnswerKind::Score).then(|| u32::try_from(top).unwrap_or(u32::MAX)),
        expected_score: (kind == AnswerKind::Score).then(|| expected_score(&probabilities)),
        probabilities: values.into_iter().zip(probabilities).collect(),
        prompt_token_ids: None,
        label_token_ids: None,
    }
}

/// How a readout layout is handed the state: the input's own text, or the
/// structured spelling the layout's family asked for.
///
/// A blank input is no state, and so is a state the caller sent as null: the
/// caller who sent `""` beside a picture meant the picture to be the whole
/// state, and the layout writes no state block for it.
pub(crate) fn readout_text(
    state: &ReadoutState,
    text: &str,
    spelling: StateSpelling,
) -> Option<String> {
    match state {
        ReadoutState::Input => (!text.trim().is_empty()).then(|| text.to_owned()),
        ReadoutState::Absent => None,
        ReadoutState::Structured { compact, indented } => Some(match spelling {
            StateSpelling::CompactSorted => compact.clone(),
            StateSpelling::Indented => indented.clone(),
        }),
    }
}

/// The shape checks every request must pass, wherever it came from.
///
/// This is the floor under every surface's own, stricter validation: ids may
/// be blank (but never repeat), options and levels must be at least one and
/// within bounds, and the temperature must be positive. Whether the input or a
/// question's wording may be blank is each surface's call, so it is not here.
///
/// # Errors
///
/// The [`AppError::BadRequest`] of the first check that fails, in the order
/// ids, each question's options or levels, temperature.
pub fn validate(request: &DecisionRequest) -> Result<(), AppError> {
    check_ids(&request.questions, true)?;
    for question in &request.questions {
        match &question.kind {
            QuestionKind::Choice(options) => check_choice_options(options, &question.name, 1)?,
            QuestionKind::Score(levels) => check_score_levels(levels, &question.name, 1)?,
            QuestionKind::YesNo { .. } => {}
        }
    }
    check_temperature(request.temperature)?;
    Ok(())
}

/// Answers a decision request from the model this state serves.
///
/// A decision checkpoint answers every question from one prompt through its
/// head, at the caller's temperature, laid out the way its family lays it
/// out; one with no head answers each question from its own prompt's
/// next-token row; any other model answers each question from the label
/// route's own prompt. Every question is prepared before any is scored: a
/// refusal to score the third question reaches the caller as a refusal, not as
/// a partial answer with the first two already paid for.
///
/// The request is checked with [`validate`] first, the floor every surface
/// already exceeds, so hand-built IR is refused rather than scored wrongly.
///
/// # Errors
///
/// [`AppError::NotReady`] and the model errors of [`AppState::engine`] for a
/// server that cannot score; [`AppError::BadRequest`] for a question the
/// model cannot be asked, naming it; [`AppError::ContextOverflow`] for a
/// prompt that leaves no room to read an answer; and whatever the engine
/// answers with.
pub async fn decide(
    state: &Arc<AppState>,
    request: DecisionRequest,
) -> Result<DecisionResponse, AppError> {
    validate(&request)?;
    let temperature = check_temperature(request.temperature)?;
    let DecisionRequest {
        input,
        readout_state,
        questions,
        preserve_thinking,
        token_ids,
        ..
    } = request;

    if let Some(layout) = state.decision_layout.as_deref() {
        let (text, media) = head::split_input(&input);
        let (answers, tally) = head::run(
            state,
            layout,
            &text,
            media,
            questions,
            temperature,
            token_ids,
        )
        .await?;
        return Ok(response(answers, tally));
    }

    if let Some(layout) = state.decision_readout.as_deref() {
        let (text, media) = head::split_input(&input);
        let text = readout_text(&readout_state, &text, layout.state_spelling());
        let (answers, tally) = readout::run(
            state,
            layout,
            text.as_deref(),
            media,
            questions,
            temperature,
            token_ids,
        )
        .await?;
        return Ok(response(answers, tally));
    }

    let mut prepared = Vec::with_capacity(questions.len());
    for question in &questions {
        prepared.push(prepare_question(
            state,
            &input,
            question,
            preserve_thinking,
        )?);
    }
    let answers = run_questions(state, prepared, temperature, token_ids).await?;
    let tally = InputTally::of(&answers);
    Ok(response(answers, tally))
}

/// The IR response for scored answers and what they read.
fn response(answers: Vec<Answered>, tally: InputTally) -> DecisionResponse {
    DecisionResponse {
        answers: answers.into_iter().map(Answered::into_answer).collect(),
        usage: Usage {
            input_tokens: tally.input,
            cached_tokens: tally.cached,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A nameless question with the given wording and kind.
    fn kinded(instructions: impl Into<String>, kind: QuestionKind) -> Question {
        Question {
            name: String::new(),
            instructions: instructions.into(),
            kind,
            probability_answer: false,
        }
    }

    fn choice(kind_input: &str, q: &str, opts: &[(&str, Option<&str>)]) -> String {
        let options = opts
            .iter()
            .map(|(value, description)| Choice {
                value: ChoiceValue::Str((*value).to_string()),
                description: description.map(ToString::to_string),
            })
            .collect();
        body(
            kind_input,
            &kinded(q.to_string(), QuestionKind::Choice(options)),
        )
    }

    /// The version-1 body under a vocabulary that spells every label as one
    /// token, which is what the wording goldens pin.
    fn body(input: &str, question: &Question) -> String {
        prompt_body(input, question, &labels_for(&question.kind, |_| true))
    }

    fn yes_no(id: &str) -> Question {
        Question {
            name: id.to_string(),
            ..kinded(
                String::new(),
                QuestionKind::YesNo {
                    yes: None,
                    no: None,
                },
            )
        }
    }

    fn option(name: &str) -> Choice {
        Choice {
            value: ChoiceValue::Str(name.to_string()),
            description: None,
        }
    }

    fn options(n: usize) -> Vec<Choice> {
        (0..n)
            .map(|index| option(&format!("opt {index}")))
            .collect()
    }

    fn levels(n: usize) -> Vec<Level> {
        (0..n)
            .map(|index| Level {
                label: format!("level {index}"),
                description: None,
            })
            .collect()
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
        let kind = kinded(
            "how bad".to_string(),
            QuestionKind::Score(
                ["none", "some", "lots"]
                    .iter()
                    .map(|level| Level {
                        label: (*level).to_string(),
                        description: None,
                    })
                    .collect(),
            ),
        );
        assert_eq!(
            body("in", &kind),
            "in\n\nQuestion: how bad\n0: none\n1: some\n2: lots\nAnswer with the number of one level only."
        );
    }

    #[test]
    fn yes_no_body_leads_with_the_question() {
        let kind = kinded(
            "it is ready".to_string(),
            QuestionKind::YesNo {
                yes: Some("confirmed".to_string()),
                no: Some("denied".to_string()),
            },
        );
        assert_eq!(
            body("in", &kind),
            "in\n\nIs the following true? it is ready\nyes: confirmed\nno: denied\nAnswer with yes or no only."
        );
    }

    #[test]
    fn yes_no_body_is_bare_when_the_question_is_blank() {
        let kind = kinded(
            String::new(),
            QuestionKind::YesNo {
                yes: None,
                no: None,
            },
        );
        assert_eq!(
            body("in", &kind),
            "in\n\nIs the following true?\nAnswer with yes or no only."
        );
    }

    #[test]
    fn body_separates_input_and_block_with_one_blank_line() {
        // An input ending in whitespace keeps its text, and the seam between
        // that text and the block stays exactly one blank line.
        let kind = kinded(
            String::new(),
            QuestionKind::YesNo {
                yes: None,
                no: None,
            },
        );
        assert_eq!(
            body("the input  \n", &kind),
            "the input\n\nIs the following true?\nAnswer with yes or no only."
        );
    }

    fn structured() -> ReadoutState {
        ReadoutState::Structured {
            compact: "{\"a\":[\"é\",true],\"b\":1}".to_owned(),
            indented: "{\n  \"b\": 1,\n  \"a\": [\n    \"é\",\n    true\n  ]\n}".to_owned(),
        }
    }

    #[test]
    fn a_structured_state_is_spelled_the_familys_way() {
        assert_eq!(
            readout_text(&structured(), "ignored", StateSpelling::Indented).as_deref(),
            Some("{\n  \"b\": 1,\n  \"a\": [\n    \"é\",\n    true\n  ]\n}")
        );
        assert_eq!(
            readout_text(&structured(), "ignored", StateSpelling::CompactSorted).as_deref(),
            Some("{\"a\":[\"é\",true],\"b\":1}")
        );
    }

    #[test]
    fn a_missing_or_blank_state_is_no_state_and_text_is_verbatim() {
        let spelling = StateSpelling::Indented;
        assert_eq!(readout_text(&ReadoutState::Absent, "x", spelling), None);
        assert_eq!(readout_text(&ReadoutState::Input, "  ", spelling), None);
        assert_eq!(readout_text(&ReadoutState::Input, "", spelling), None);
        assert_eq!(
            readout_text(&ReadoutState::Input, " hi\n", spelling).as_deref(),
            Some(" hi\n")
        );
    }

    #[test]
    fn labels_are_letters_then_digits_then_yes_no() {
        assert_eq!(choice_labels(3, |_| true), ["A", "B", "C"]);
        assert_eq!(score_labels(3), ["0", "1", "2"]);
        assert_eq!(YES_NO_LABELS, ["yes", "no"]);
    }

    fn prepared(ids: &[u32]) -> PreparedQuestion {
        PreparedQuestion {
            question: kinded(
                String::new(),
                QuestionKind::YesNo {
                    yes: None,
                    no: None,
                },
            ),
            prompt_tokens: ids.iter().copied().map(TokenId::new).collect(),
            labels: Vec::new(),
            media: Arc::from(Vec::new()),
            media_digests: Arc::from(Vec::new()),
        }
    }

    #[test]
    fn shared_prefix_is_what_every_question_starts_with() {
        // One question shares with nobody.
        assert_eq!(shared_prefix(&[prepared(&[1, 2, 3])]), 0);
        assert_eq!(shared_prefix(&[]), 0);
        // The shortest agreement wins, wherever in the list it is.
        let questions = [
            prepared(&[1, 2, 3, 4, 5]),
            prepared(&[1, 2, 3, 4, 9]),
            prepared(&[1, 2, 7]),
        ];
        assert_eq!(shared_prefix(&questions), 2);
        // A prompt that is a prefix of another shares all of itself.
        assert_eq!(shared_prefix(&[prepared(&[1, 2]), prepared(&[1, 2, 3])]), 2);
        assert_eq!(shared_prefix(&[prepared(&[1, 2, 3]), prepared(&[4])]), 0);
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
    fn a_wide_list_is_two_letters_throughout_skipping_split_pairs() {
        // Within the alphabet the vocabulary is never asked: the letters are
        // the version-1 labels.
        assert_eq!(choice_labels(26, |_| false).len(), 26);
        let all = choice_labels(27, |_| true);
        assert_eq!(all.len(), 27);
        assert_eq!(all[..3], ["AA", "AB", "AC"]);
        assert_eq!(all[26], "BA");
        // A pair the vocabulary splits is skipped, not renamed.
        let skipping = choice_labels(30, |pair| pair != "AB");
        assert_eq!(skipping[..3], ["AA", "AC", "AD"]);
        // A vocabulary that runs out comes up short; preparation refuses it.
        assert_eq!(choice_labels(30, |pair| pair.starts_with('A')).len(), 26);
        assert_eq!(
            choice_labels(MAX_CHOICE_OPTIONS, |_| true).len(),
            MAX_CHOICE_OPTIONS
        );
    }

    #[test]
    fn choice_body_words_a_wide_list_by_its_two_letter_labels() {
        let opts: Vec<(String, Option<&str>)> =
            (0..27).map(|index| (format!("o{index}"), None)).collect();
        let opts: Vec<(&str, Option<&str>)> = opts
            .iter()
            .map(|(name, description)| (name.as_str(), *description))
            .collect();
        let rendered = choice("in", "pick one", &opts);
        assert!(
            rendered.starts_with("in\n\nQuestion: pick one\nAA: o0\nAB: o1\n"),
            "{rendered}"
        );
        assert!(
            rendered.ends_with("\nBA: o26\nAnswer with the two-letter label of one option only."),
            "{rendered}"
        );
    }

    fn str_option(name: &str) -> Choice {
        Choice {
            value: ChoiceValue::Str(name.to_string()),
            description: None,
        }
    }

    fn bool_option(value: bool) -> Choice {
        Choice {
            value: ChoiceValue::Bool(value),
            description: None,
        }
    }

    #[test]
    fn a_boolean_choice_value_renders_as_the_word() {
        // The official wire's choice values are typed; a boolean spells
        // itself `true`/`false` on the option's line.
        let kind = kinded(
            "q".to_string(),
            QuestionKind::Choice(vec![bool_option(true), bool_option(false)]),
        );
        assert_eq!(
            body("in", &kind),
            "in\n\nQuestion: q\nA: true\nB: false\nAnswer with the letter of one option only."
        );
    }

    #[test]
    fn a_score_level_description_shares_its_digit_line() {
        let kind = kinded(
            "how bad".to_string(),
            QuestionKind::Score(vec![
                Level {
                    label: "Cosmetic".to_string(),
                    description: None,
                },
                Level {
                    label: "Blocked".to_string(),
                    description: Some("no workaround".to_string()),
                },
            ]),
        );
        assert_eq!(
            body("in", &kind),
            "in\n\nQuestion: how bad\n0: Cosmetic\n1: Blocked - no workaround\nAnswer with the number of one level only."
        );
    }

    #[test]
    fn a_blank_score_level_description_keeps_the_bare_line() {
        let kind = kinded(
            "how bad".to_string(),
            QuestionKind::Score(vec![
                Level {
                    label: "a".to_string(),
                    description: Some("   ".to_string()),
                },
                Level {
                    label: "b".to_string(),
                    description: None,
                },
            ]),
        );
        assert_eq!(
            body("in", &kind),
            "in\n\nQuestion: how bad\n0: a\n1: b\nAnswer with the number of one level only."
        );
    }

    #[test]
    fn typed_choice_values_collide_only_within_their_type() {
        // The official wire says a string and a boolean with the same text
        // are distinct options: `"true"` beside `true` is a legal list.
        assert!(check_choice_options(&[str_option("true"), bool_option(true)], "q", 2).is_ok());
        // Two booleans spelling the same word are the same option.
        assert!(check_choice_options(&[bool_option(true), bool_option(true)], "q", 2).is_err());
        // Strings keep today's trim-and-fold comparison.
        assert!(check_choice_options(&[str_option("Red"), str_option(" red ")], "q", 2).is_err());
    }

    #[test]
    fn check_choice_options_bounds_two_to_the_wire_maximum() {
        assert!(check_choice_options(&options(26), "q", 2).is_ok());
        assert!(check_choice_options(&options(27), "q", 2).is_ok());
        assert!(check_choice_options(&options(MAX_CHOICE_OPTIONS), "q", 2).is_ok());
        let message = refusal(check_choice_options(
            &options(MAX_CHOICE_OPTIONS + 1),
            "q",
            2,
        ));
        assert!(message.contains("options per choice"), "{message}");
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
        assert!(
            check_score_levels(
                &[
                    Level {
                        label: "ok".to_string(),
                        description: None
                    },
                    Level {
                        label: "  ".to_string(),
                        description: None
                    },
                ],
                "q",
                2
            )
            .is_err()
        );
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

    /// The label proof as it was first written: every label re-encodes the
    /// whole tail. The reference the seam proof must answer exactly as.
    fn literal_proof(
        vocabulary: &Vocabulary,
        text: &str,
        tokens: &[TokenId],
        label_words: &[String],
    ) -> Result<Vec<TokenId>, String> {
        let mut labels = Vec::new();
        for label in label_words {
            let encoded = vocabulary.encode(label, false).map_err(|e| e.to_string())?;
            let [only] = encoded.as_slice() else {
                return Err(format!("{label} is not one token"));
            };
            if labels.contains(only) {
                return Err(format!("{label} repeats a label"));
            }
            let mut following = tokens.to_vec();
            following.push(*only);
            let joined = vocabulary
                .encode(&format!("{text}{label}"), false)
                .map_err(|e| e.to_string())?;
            if joined != following {
                return Err(format!("{label} does not follow"));
            }
            labels.push(*only);
        }
        Ok(labels)
    }

    /// The seam proof over `text`, with refusals reduced to whether they
    /// happened.
    fn seam_proof(vocabulary: &Vocabulary, text: &str, label_words: &[String]) -> Option<Vec<u32>> {
        let (tokens, seam) = vocabulary.encode_with_seam(text).expect("encodes");
        prove_labels(
            vocabulary,
            "question \"q\"",
            label_words.to_vec(),
            &LabelSeam {
                text,
                tokens: &tokens,
                seam: seam.as_ref(),
            },
        )
        .ok()
        .map(|ids| ids.iter().map(|id| id.get()).collect())
    }

    /// A word-level vocabulary with three added tokens, one of which is
    /// spelled across a seam: `<|a|>` then a newline then `B`.
    fn seam_vocabulary() -> Vocabulary {
        use std::str::FromStr as _;
        let words = [
            "<unk>", "A", "B", "C", "AB", "yes", "no", "the", "input", "Answer", "<|u|>", "<|a|>",
            "<|a|>\nB",
        ];
        let vocab: serde_json::Map<String, serde_json::Value> = words
            .iter()
            .enumerate()
            .map(|(id, word)| ((*word).to_owned(), json!(id)))
            .collect();
        let added = |id: usize| {
            json!({
                "id": id, "content": words[id], "single_word": false, "lstrip": false,
                "rstrip": false, "normalized": false, "special": true,
            })
        };
        let document = json!({
            "version": "1.0",
            "truncation": serde_json::Value::Null,
            "padding": serde_json::Value::Null,
            "added_tokens": [added(10), added(11), added(12)],
            "normalizer": serde_json::Value::Null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": serde_json::Value::Null,
            "decoder": serde_json::Value::Null,
            "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "<unk>"},
        })
        .to_string();
        Vocabulary::from_tokenizer(
            tokenizers::Tokenizer::from_str(&document).expect("the vocabulary loads"),
        )
    }

    #[test]
    fn the_seam_proof_answers_exactly_what_the_whole_tail_proof_answers() {
        let vocabulary = seam_vocabulary();
        let labels = |words: &[&str]| words.iter().map(|w| (*w).to_owned()).collect::<Vec<_>>();
        for (text, words) in [
            // A fresh token after the opener: accepted either way.
            ("<|u|> the input <|a|>\n", labels(&["A", "C", "yes", "no"])),
            // A tail that ends on a word swallows the label: refused.
            ("<|u|> the input <|a|> Answer", labels(&["A"])),
            // No added token at all: the whole tail is the stretch.
            ("the input ", labels(&["A", "C"])),
            // The label completes an added token spelled across the seam
            // (`<|a|>` + newline + `B`): the literal proof refuses it, and
            // the seam must not be used for it.
            ("<|u|> the input <|a|>\n", labels(&["A", "B"])),
            // A two-letter label that is one token, then one that is not.
            ("<|u|> the input <|a|>\n", labels(&["AB", "C"])),
            ("<|u|> the input <|a|>\n", labels(&["A", "BC"])),
            // Two labels that are one token.
            ("<|u|> the input <|a|>\n", labels(&["A", "A"])),
        ] {
            let (tokens, _seam) = vocabulary.encode_with_seam(text).expect("encodes");
            let want = literal_proof(&vocabulary, text, &tokens, &words)
                .ok()
                .map(|ids| ids.iter().map(|id| id.get()).collect::<Vec<_>>());
            assert_eq!(
                seam_proof(&vocabulary, text, &words),
                want,
                "{text:?} {words:?}"
            );
        }
    }

    #[test]
    fn the_seam_is_just_past_the_last_added_token_and_held_only_where_nothing_spans_it() {
        let vocabulary = seam_vocabulary();
        let text = "<|u|> the input <|a|>\n";
        let (tokens, seam) = vocabulary.encode_with_seam(text).expect("encodes");
        let seam = seam.expect("the text holds an added token");
        assert_eq!(&text[seam.text..], "\n");
        assert_eq!(tokens[seam.tokens - 1].get(), 11, "the seam follows <|a|>");
        assert!(seam.holds_after(text, "A"));
        assert!(
            !seam.holds_after(text, "B"),
            "`<|a|>\\nB` would match across the seam"
        );
        assert!(
            !seam.holds_after(text, "\u{e9}"),
            "a non-ASCII label is not cleared"
        );
        let (_tokens, none) = vocabulary.encode_with_seam("the input").expect("encodes");
        assert!(none.is_none(), "no added token, no seam");
    }

    /// The label proof's cost on the served vocabulary: one 151-option
    /// question over about four thousand words of input, proved the literal
    /// way (a whole-prompt encode per label) and the seam way. Both must
    /// accept the same labels with the same ids. Needs the served export's
    /// directory in `SHISO_TEST_MODEL_QWEN4EXP_EXL3` for its `tokenizer.json`.
    #[test]
    #[ignore = "needs the served vocabulary named by SHISO_TEST_MODEL_QWEN4EXP_EXL3"]
    #[expect(clippy::print_stdout, reason = "the timings are the test's output")]
    fn a_wide_question_over_a_long_input_proves_its_labels_in_label_time() {
        let Some(dir) = std::env::var_os("SHISO_TEST_MODEL_QWEN4EXP_EXL3") else {
            println!("SKIP: no export named by SHISO_TEST_MODEL_QWEN4EXP_EXL3");
            return;
        };
        let vocabulary = Vocabulary::from_file(&std::path::Path::new(&dir).join("tokenizer.json"))
            .expect("the vocabulary loads");
        let words = [
            "the",
            "review",
            "says",
            "battery",
            "lasts",
            "two",
            "days",
            "and",
            "screen",
            "is",
            "bright",
            "but",
            "camera",
            "struggles",
            "at",
            "night",
            "while",
            "price",
            "seems",
            "fair",
        ];
        let input = (0..4_000)
            .map(|at| words[(at * 7 + at / 3) % words.len()])
            .collect::<Vec<_>>()
            .join(" ");
        let kind = kinded(
            "Which aspect is discussed?".to_owned(),
            QuestionKind::Choice(options(151)),
        );
        let started = std::time::Instant::now();
        let labels = vocabulary_labels(&vocabulary, &kind.kind, "q").expect("151 labels");
        let labelled = started.elapsed();
        let text = format!(
            "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
            prompt_body(&input, &kind, &labels)
        );

        let before = std::time::Instant::now();
        let tokens = vocabulary.encode(&text, false).expect("encodes");
        let want = literal_proof(&vocabulary, &text, &tokens, &labels).expect("the labels hold");
        let before = before.elapsed();

        let after = std::time::Instant::now();
        let (seamed, seam) = vocabulary.encode_with_seam(&text).expect("encodes");
        let got = prove_labels(
            &vocabulary,
            "q",
            labels.clone(),
            &LabelSeam {
                text: &text,
                tokens: &seamed,
                seam: seam.as_ref(),
            },
        )
        .expect("the labels hold");
        let after = after.elapsed();

        assert_eq!(seamed, tokens);
        assert_eq!(got, want);
        assert!(
            seam.is_some(),
            "the served template ends past an added token"
        );
        println!(
            "prompt {} tokens, {} labels: label choice {labelled:?}; proof before {before:?}, \
             after {after:?}",
            tokens.len(),
            labels.len()
        );
    }

    fn request_of(questions: Vec<Question>, temperature: Option<f64>) -> DecisionRequest {
        DecisionRequest {
            input: Input::Text("x".to_owned()),
            readout_state: ReadoutState::Input,
            questions,
            temperature,
            preserve_thinking: None,
            token_ids: false,
        }
    }

    fn choice_question(name: &str, values: &[&str]) -> Question {
        kinded(
            "pick",
            QuestionKind::Choice(values.iter().map(|value| option(value)).collect()),
        )
        .named(name)
    }

    trait Named {
        fn named(self, name: &str) -> Self;
    }

    impl Named for Question {
        fn named(mut self, name: &str) -> Self {
            self.name = name.to_owned();
            self
        }
    }

    #[test]
    fn validate_accepts_what_the_surfaces_accept() {
        let request = request_of(
            vec![
                choice_question("", &["a"]),
                choice_question("", &["b", "c"]),
                yes_no("n"),
                kinded(
                    "",
                    QuestionKind::Score(vec![Level {
                        label: "only".to_owned(),
                        description: None,
                    }]),
                ),
            ],
            Some(0.5),
        );
        assert!(validate(&request).is_ok());
    }

    #[test]
    fn validate_refuses_each_malformed_part() {
        let refused = |request: DecisionRequest| {
            validate(&request)
                .expect_err("the floor must refuse this request")
                .to_string()
        };
        let duplicate = refused(request_of(vec![yes_no("a"), yes_no("a")], None));
        assert!(duplicate.contains("used by"), "{duplicate}");
        let control = refused(request_of(vec![choice_question("q", &["a\nb"])], None));
        assert!(control.contains("control character"), "{control}");
        let blank = refused(request_of(vec![choice_question("q", &["  "])], None));
        assert!(blank.contains("non-blank"), "{blank}");
        let repeated = refused(request_of(vec![choice_question("q", &["a", "A"])], None));
        assert!(repeated.contains("more than once"), "{repeated}");
        let none = refused(request_of(vec![choice_question("q", &[])], None));
        assert!(none.contains("at least 1"), "{none}");
        let levels = refused(request_of(
            vec![kinded("q", QuestionKind::Score(Vec::new()))],
            None,
        ));
        assert!(levels.contains("at least 1 levels"), "{levels}");
        let blank_level = refused(request_of(
            vec![kinded(
                "q",
                QuestionKind::Score(vec![Level {
                    label: " ".to_owned(),
                    description: None,
                }]),
            )],
            None,
        ));
        assert!(blank_level.contains("must not be blank"), "{blank_level}");
        let temperature = refused(request_of(vec![yes_no("a")], Some(0.0)));
        assert!(temperature.contains("temperature"), "{temperature}");
    }

    #[test]
    fn a_score_answer_names_its_level_by_label_and_number() {
        let answer = scored_answer(
            kinded(
                "q",
                QuestionKind::Score(
                    ["low", "high"]
                        .iter()
                        .map(|label| Level {
                            label: (*label).to_owned(),
                            description: None,
                        })
                        .collect(),
                ),
            ),
            vec![0.25, 0.75],
            1.0,
            1,
        );
        assert_eq!(answer.choice, ChoiceValue::Str("high".to_owned()));
        assert!(
            answer
                .probabilities
                .iter()
                .any(|(value, _)| *value == answer.choice)
        );
        assert_eq!(answer.level, Some(1));
    }
}
