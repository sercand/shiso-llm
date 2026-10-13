//! The answer-slot readout of a d1 decision checkpoint.
//!
//! A d1 checkpoint is the plain backbone with no head beside it: it was
//! trained to answer a question with one token at the start of the assistant
//! turn, so each question gets its own prompt and its answer is a softmax over
//! the best log-probability each option's spellings have at that row. The
//! prompt is the release's own, byte for byte, and the checkpoint ships it as
//! its `systemone` chat template:
//!
//! ```text
//! <|startoftext|><|im_start|>user\n{pictures}{state}\n\n\nQUESTION:\n{question}<|im_end|>\n<|im_start|>assistant\n
//! ```
//!
//! With no state the question follows the pictures directly. A string state is
//! written as it is and a structured one as two-space indented JSON in the
//! caller's key order, which is why this readout asks for
//! [`StateSpelling::Indented`].
//!
//! # Options are read by code, not by name
//!
//! A choice's options are listed under one-token codes - the labels
//! themselves when every label is one letter, `A`.. up to 26 options, `00`..
//! past that - and a code the vocabulary does not spell as one fresh token is
//! swapped for the next one from a fixed pool, so every option can be read
//! from the one row. An option scores the best of its code and the code after
//! a space; a yes/no question the best of `yes`/`Yes`/`YES` against the
//! same of `no`; a score its level's digit. Only forms the vocabulary spells
//! as one token count, as in the release.
//!
//! # How the prompt is tokenized
//!
//! One question is tokenized whole, several as the shared state and then each
//! question on its own, as the release does: the second is what lets the
//! server read the state once, and its seam - right after `QUESTION:\n` - is
//! where the release cuts too, so either way the ids are the release's. Text
//! the caller wrote stays text: a marker spelled inside a state or a question
//! is never read as the marker.

use std::ops::Range;

use shiso_core::{
    CoreError, DecisionLayoutError, DecisionMediaKind, DecisionReadout, PromptEncoder,
    ReadoutPrompt, ReadoutQuestion, ReadoutRecord, SchemaAnswer, SchemaOption, SchemaQuestion,
    StateSpelling, TokenId,
};

/// The token every prompt opens with.
const BOS: &str = "<|startoftext|>";
/// The user turn's opening, after the leading token.
const USER: &str = "<|im_start|>user\n";
/// What closes the question and opens the answer slot.
const ANSWER_SLOT: &str = "<|im_end|>\n<|im_start|>assistant\n";
/// The run each picture occupies, around its repeated placeholder.
const IMAGE_OPEN: &str = "<|image_start|>";
const IMAGE_PAD: &str = "<image>";
const IMAGE_CLOSE: &str = "<|image_end|>";

/// The spellings a yes and a no are read from.
const YES_FORMS: [&str; 3] = ["yes", "Yes", "YES"];
const NO_FORMS: [&str; 3] = ["no", "No", "NO"];

/// The d1 readout.
#[derive(Debug, Clone, Copy, Default)]
pub struct D1Readout;

/// A piece of text being assembled, with the byte ranges that are the
/// caller's.
#[derive(Default)]
struct Text {
    text: String,
    literal: Vec<Range<usize>>,
}

impl Text {
    /// Appends the layout's own wording.
    fn markup(&mut self, text: &str) {
        self.text.push_str(text);
    }

    /// Appends caller text, which stays text.
    fn caller(&mut self, text: &str) {
        let start = self.text.len();
        self.text.push_str(text);
        if !text.is_empty() {
            self.literal.push(start..self.text.len());
        }
    }

    /// Appends another piece, keeping its caller ranges.
    fn append(&mut self, other: &Self) {
        let offset = self.text.len();
        self.text.push_str(&other.text);
        self.literal.extend(
            other
                .literal
                .iter()
                .map(|range| range.start + offset..range.end + offset),
        );
    }

    fn encode(&self, encoder: &dyn PromptEncoder) -> Result<Vec<TokenId>, DecisionLayoutError> {
        if self.text.is_empty() {
            return Ok(Vec::new());
        }
        encoder
            .encode(&self.text, &self.literal)
            .map_err(DecisionLayoutError::Encode)
    }
}

/// The ids `text` encodes to, alone.
fn ids(encoder: &dyn PromptEncoder, text: &str) -> Result<Vec<TokenId>, DecisionLayoutError> {
    encoder
        .encode(text, &[])
        .map_err(DecisionLayoutError::Encode)
}

/// The one id a marker encodes to.
fn marker(encoder: &dyn PromptEncoder, spelling: &str) -> Result<TokenId, DecisionLayoutError> {
    match ids(encoder, spelling)?.as_slice() {
        [one] => Ok(*one),
        _ => Err(DecisionLayoutError::Encode(CoreError::InvalidValue {
            key: "d1.marker",
            value: spelling.to_owned(),
        })),
    }
}

/// The one id `text` encodes to, when it encodes to exactly one.
///
/// `None` is not a refusal: it means the vocabulary splits `text`, so that
/// spelling cannot be read from one row and the caller tries another.
fn one_token(
    encoder: &dyn PromptEncoder,
    text: &str,
) -> Result<Option<TokenId>, DecisionLayoutError> {
    Ok(match ids(encoder, text)?.as_slice() {
        [one] => Some(*one),
        _ => None,
    })
}

/// The single-token ids of `forms`, each once, in order: the spellings of one
/// answer that can be read from a row.
fn readable(
    encoder: &dyn PromptEncoder,
    forms: impl IntoIterator<Item = String>,
) -> Result<Vec<TokenId>, DecisionLayoutError> {
    let mut out = Vec::new();
    for form in forms {
        if let Some(id) = one_token(encoder, &form)?
            && !out.contains(&id)
        {
            out.push(id);
        }
    }
    Ok(out)
}

/// The codes `labels` are listed under before any is checked against the
/// vocabulary: the labels themselves when every one is a single letter, the
/// alphabet up to 26 options, two digits past that.
fn option_codes(labels: &[&str]) -> Vec<String> {
    let trimmed: Vec<&str> = labels.iter().map(|label| label.trim()).collect();
    let letters = !trimmed.is_empty()
        && trimmed.iter().all(|label| {
            let mut chars = label.chars();
            chars.next().is_some_and(char::is_alphabetic) && chars.next().is_none()
        });
    if letters {
        return trimmed.iter().map(|label| (*label).to_owned()).collect();
    }
    if trimmed.len() <= 26 {
        return ('A'..='Z').take(trimmed.len()).map(String::from).collect();
    }
    (0..trimmed.len())
        .map(|index| format!("{index:02}"))
        .collect()
}

/// The pool a code is swapped from when the vocabulary does not spell it as
/// one fresh token, in the order it is searched. Two capital letters come
/// last: where a vocabulary splits digits they are often one token, and where
/// it has digit pairs the search never reaches them.
fn fallback_pool() -> impl Iterator<Item = String> {
    let upper = ('A'..='Z').map(String::from);
    let pairs = (0..100).map(|index| format!("{index:02}"));
    let lower = ('a'..='z').map(String::from);
    let hashes = (0..200).map(|index| format!("#{index}"));
    let doubles = ('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}")));
    upper.chain(pairs).chain(lower).chain(hashes).chain(doubles)
}

/// Every option's code and its id: the planned code when it is one token no
/// earlier option took, otherwise the first such entry of the pool.
fn aliases(
    encoder: &dyn PromptEncoder,
    question: usize,
    labels: &[&str],
) -> Result<Vec<(String, TokenId)>, DecisionLayoutError> {
    let mut out: Vec<(String, TokenId)> = Vec::with_capacity(labels.len());
    for code in option_codes(labels) {
        let fresh = |id: &TokenId, out: &[(String, TokenId)]| !out.iter().any(|(_, t)| t == id);
        if let Some(id) = one_token(encoder, &code)?
            && fresh(&id, &out)
        {
            out.push((code, id));
            continue;
        }
        let mut found = None;
        for raw in fallback_pool() {
            if let Some(id) = one_token(encoder, &raw)?
                && fresh(&id, &out)
            {
                found = Some((raw, id));
                break;
            }
        }
        out.push(found.ok_or_else(|| DecisionLayoutError::Unreadable {
            question,
            reason: format!("no single-token code is left for {} options", labels.len()),
        })?);
    }
    Ok(out)
}

/// A choice's lines and its readout groups.
fn choice(
    encoder: &dyn PromptEncoder,
    question: usize,
    options: &[SchemaOption],
    text: &mut Text,
) -> Result<Vec<Vec<TokenId>>, DecisionLayoutError> {
    let labels: Vec<&str> = options.iter().map(|option| option.value.as_str()).collect();
    let codes = aliases(encoder, question, &labels)?;
    text.markup("\n\nOptions:\n");
    let mut groups = Vec::with_capacity(options.len());
    for (index, (option, (code, id))) in options.iter().zip(&codes).enumerate() {
        if index > 0 {
            text.markup("\n");
        }
        text.markup(code);
        text.markup(" ");
        // An empty description is no description, as the release reads it:
        // the label stands in, its underscores read as spaces.
        match option.description.as_deref().filter(|d| !d.is_empty()) {
            Some(description) => text.caller(description),
            None => text.caller(&option.value.replace('_', " ")),
        }
        let mut group = vec![*id];
        group.extend(
            readable(encoder, [format!(" {code}")])?
                .into_iter()
                .filter(|other| other != id),
        );
        groups.push(group);
    }
    text.markup("\n\nReply with the option code only.");
    Ok(groups)
}

/// A yes/no question's lines and its two readout groups, yes first.
fn yes_no(
    encoder: &dyn PromptEncoder,
    question: usize,
    yes: Option<&str>,
    no: Option<&str>,
    text: &mut Text,
) -> Result<Vec<Vec<TokenId>>, DecisionLayoutError> {
    // Either side stated means both lines are written, the missing one as the
    // release spells an absent value.
    if yes.is_some() || no.is_some() {
        text.markup("\nYes: ");
        text.caller(yes.unwrap_or("None"));
        text.markup("\nNo: ");
        text.caller(no.unwrap_or("None"));
    }
    text.markup("\n\nReply with yes or no only.");
    let yes = readable(encoder, YES_FORMS.map(String::from))?;
    let no = readable(encoder, NO_FORMS.map(String::from))?;
    if yes.is_empty() || no.is_empty() {
        return Err(DecisionLayoutError::Unreadable {
            question,
            reason: "the vocabulary has no single-token yes or no".to_owned(),
        });
    }
    Ok(vec![yes, no])
}

/// A score's levels and their digit groups.
fn score(
    encoder: &dyn PromptEncoder,
    question: usize,
    levels: &[SchemaOption],
    text: &mut Text,
) -> Result<Vec<Vec<TokenId>>, DecisionLayoutError> {
    text.markup("\n\n");
    let mut groups = Vec::with_capacity(levels.len());
    for (index, level) in levels.iter().enumerate() {
        text.markup(&format!("{index} "));
        text.caller(&level.value);
        // The release's levels are one string each; a level the caller also
        // described keeps the description after the server's own separator.
        if let Some(description) = level
            .description
            .as_deref()
            .filter(|d| !d.trim().is_empty())
        {
            text.markup(" - ");
            text.caller(description);
        }
        text.markup("\n");
        let group = readable(encoder, [index.to_string()])?;
        if group.is_empty() {
            return Err(DecisionLayoutError::Unreadable {
                question,
                reason: format!(
                    "a score of {} levels needs a single-token digit for each, \
                     and this vocabulary has none for {index}",
                    levels.len()
                ),
            });
        }
        groups.push(group);
    }
    text.markup(&format!(
        "\nReply with a single digit 0-{} only.",
        levels.len().saturating_sub(1)
    ));
    Ok(groups)
}

/// One question's text up to the answer slot, and its readout groups.
fn question_text(
    encoder: &dyn PromptEncoder,
    index: usize,
    question: &SchemaQuestion,
) -> Result<(Text, Vec<Vec<TokenId>>), DecisionLayoutError> {
    let mut text = Text::default();
    text.caller(&question.wording);
    let groups = match &question.answer {
        SchemaAnswer::YesNo { yes, no } => {
            yes_no(encoder, index, yes.as_deref(), no.as_deref(), &mut text)?
        }
        SchemaAnswer::Choice(options) => choice(encoder, index, options, &mut text)?,
        SchemaAnswer::Score(levels) => score(encoder, index, levels, &mut text)?,
    };
    text.markup(ANSWER_SLOT);
    Ok((text, groups))
}

/// The pictures' runs, each its placeholder bracketed and repeated to its row
/// count, back to back.
fn media_tokens(
    encoder: &dyn PromptEncoder,
    record: &ReadoutRecord<'_>,
) -> Result<Vec<TokenId>, DecisionLayoutError> {
    if record.media.is_empty() {
        return Ok(Vec::new());
    }
    let open = marker(encoder, IMAGE_OPEN)?;
    let pad = marker(encoder, IMAGE_PAD)?;
    let close = marker(encoder, IMAGE_CLOSE)?;
    let mut tokens = Vec::new();
    for item in record.media {
        if item.kind != DecisionMediaKind::Image {
            return Err(DecisionLayoutError::Encode(CoreError::Unsupported {
                what: "a video clip in a d1 decision",
                backend: "core",
            }));
        }
        tokens.push(open);
        tokens.extend(std::iter::repeat_n(pad, item.pads.max(1)));
        tokens.push(close);
    }
    Ok(tokens)
}

impl DecisionReadout for D1Readout {
    fn state_spelling(&self) -> StateSpelling {
        StateSpelling::Indented
    }

    fn encode(
        &self,
        encoder: &dyn PromptEncoder,
        record: &ReadoutRecord<'_>,
    ) -> Result<ReadoutPrompt, DecisionLayoutError> {
        // The leading token is encoded alone, so no encoder can mistake it
        // for one it would add itself and drop.
        let bos = marker(encoder, BOS)?;
        let media = media_tokens(encoder, record)?;
        let mut body = Text::default();
        if let Some(state) = record.state {
            body.caller(state);
            body.markup("\n\n\nQUESTION:\n");
        }

        let laid: Vec<(Text, Vec<Vec<TokenId>>)> = record
            .questions
            .iter()
            .enumerate()
            .map(|(index, question)| question_text(encoder, index, question))
            .collect::<Result<_, _>>()?;

        // Without pictures the turn's opening and the state are one stretch
        // of text, tokenized together as the release tokenizes them; with
        // them the pictures' markers cut the text there anyway.
        let (lead, mut body) = if media.is_empty() {
            let mut whole = Text::default();
            whole.markup(USER);
            whole.append(&body);
            (vec![bos], whole)
        } else {
            let mut lead = vec![bos];
            lead.extend(ids(encoder, USER)?);
            lead.extend(media);
            (lead, body)
        };

        let prompt = if let [(question, groups)] = laid.as_slice() {
            body.append(question);
            let mut branch = lead;
            branch.extend(body.encode(encoder)?);
            ReadoutPrompt {
                trunk: Vec::new(),
                questions: vec![ReadoutQuestion {
                    branch,
                    options: groups.clone(),
                }],
            }
        } else {
            let mut trunk = lead;
            trunk.extend(body.encode(encoder)?);
            let questions = laid
                .into_iter()
                .map(|(question, groups)| {
                    Ok(ReadoutQuestion {
                        branch: question.encode(encoder)?,
                        options: groups,
                    })
                })
                .collect::<Result<_, DecisionLayoutError>>()?;
            ReadoutPrompt { trunk, questions }
        };

        let longest = prompt
            .questions
            .iter()
            .map(|question| prompt.trunk.len() + question.branch.len())
            .max()
            .unwrap_or(prompt.trunk.len());
        if longest > record.budget {
            return Err(DecisionLayoutError::PromptTooLong {
                tokens: longest,
                budget: record.budget,
            });
        }
        Ok(prompt)
    }
}

#[cfg(test)]
#[path = "decision_tests.rs"]
mod tests;
