//! The prompt a Clef head reads, laid out the way the release's
//! `encode_record` lays it out.
//!
//! One prompt carries the state and the whole schema. It is built piece by
//! piece, each piece tokenized on its own, so every span the head pools
//! begins and ends on a token boundary by construction - which is also how
//! the head was trained, so a prompt tokenized whole would read differently
//! at every seam.
//!
//! Media sits between `STATE:` and the state, one bracketed placeholder run
//! per item at its full length, and counts against the length budget the way
//! the release's processor-expanded ids do: only the state is ever cut.

use std::ops::Range;

use serde_json::Value;
use shiso_core::{
    DecisionLayout, DecisionLayoutError, DecisionMediaKind, DecisionPrompt, DecisionQuestion,
    DecisionRecord, PromptEncoder, SchemaAnswer, SchemaQuestion, TokenId,
};

/// The system turn every Clef prompt opens with.
const SYSTEM_PROMPT: &str = "Read the complete state and schema. Decide every field jointly. \
                             Each answer must be exactly one of that field's allowed options.";

/// The longest prompt the release encodes; the state is cut to fit.
const MAX_PROMPT_TOKENS: usize = 16_384;

/// The default wording of a binary question's two options.
const NOUL_TRUE: &str = "The proposition is true or the answer is yes.";
const NOUL_FALSE: &str = "The proposition is false or the answer is no.";

/// The run each media item occupies, before its placeholder is repeated.
const IMAGE_OPEN: &str = "<|vision_start|>";
const IMAGE_PAD: &str = "<|image_pad|>";
const VIDEO_PAD: &str = "<|video_pad|>";
const MEDIA_CLOSE: &str = "<|vision_end|>";

/// Clef's prompt layout.
#[derive(Debug, Clone, Copy, Default)]
pub struct ClefLayout;

/// One option as the head sees it: its id and its wording.
struct HeadOption {
    id: String,
    description: Option<String>,
}

/// A question's options in the order the head reads them, with the caller's
/// index of each.
fn head_options(answer: &SchemaAnswer) -> Vec<(usize, HeadOption)> {
    match answer {
        SchemaAnswer::YesNo { yes, no } => vec![
            (
                0,
                HeadOption {
                    id: "true".to_owned(),
                    description: Some(yes.clone().unwrap_or_else(|| NOUL_TRUE.to_owned())),
                },
            ),
            (
                1,
                HeadOption {
                    id: "false".to_owned(),
                    description: Some(no.clone().unwrap_or_else(|| NOUL_FALSE.to_owned())),
                },
            ),
        ],
        // The release reads a choice's options sorted by id; the answer keeps
        // the caller's order.
        SchemaAnswer::Choice(options) => {
            let mut sorted: Vec<(usize, HeadOption)> = options
                .iter()
                .enumerate()
                .map(|(index, option)| {
                    (
                        index,
                        HeadOption {
                            id: option.value.clone(),
                            description: option.description.clone(),
                        },
                    )
                })
                .collect();
            sorted.sort_by(|a, b| a.1.id.cmp(&b.1.id));
            sorted
        }
        // A level is named by its index and described by its label.
        SchemaAnswer::Score(levels) => levels
            .iter()
            .enumerate()
            .map(|(index, level)| {
                (
                    index,
                    HeadOption {
                        id: index.to_string(),
                        description: Some(level.value.clone()),
                    },
                )
            })
            .collect(),
    }
}

/// The release's name for a question kind, and the head's index for it.
const fn kind_of(answer: &SchemaAnswer) -> (&'static str, u32) {
    match answer {
        SchemaAnswer::YesNo { .. } => ("noul", 0),
        SchemaAnswer::Choice(_) => ("choice", 1),
        SchemaAnswer::Score(_) => ("score", 2),
    }
}

/// `{"description": ..., "option_id": ...}` as compact JSON with its keys in
/// sorted order, the release's rendering of an option.
fn option_text(option: &HeadOption) -> String {
    let id = Value::String(option.id.clone());
    match &option.description {
        Some(description) => format!(
            "{{\"description\":{},\"option_id\":{id}}}",
            Value::String(description.clone())
        ),
        None => format!("{{\"option_id\":{id}}}"),
    }
}

/// Refuses a choice the head cannot tell apart.
///
/// The schema names each option by the text of its value, so two values that
/// spell the same word render two identical option lines and score a
/// coin-flip between them.
fn check_spellings(questions: &[SchemaQuestion]) -> Result<(), DecisionLayoutError> {
    for (question, schema) in questions.iter().enumerate() {
        if let SchemaAnswer::Choice(options) = &schema.answer {
            for (index, option) in options.iter().enumerate() {
                if options[..index]
                    .iter()
                    .any(|earlier| earlier.value == option.value)
                {
                    return Err(DecisionLayoutError::AmbiguousOption {
                        question,
                        spelling: option.value.clone(),
                    });
                }
            }
        }
    }
    Ok(())
}

/// A prompt being assembled from pieces.
struct Pieces<'a> {
    encoder: &'a dyn PromptEncoder,
    tokens: Vec<TokenId>,
}

impl<'a> Pieces<'a> {
    const fn new(encoder: &'a dyn PromptEncoder) -> Self {
        Self {
            encoder,
            tokens: Vec::new(),
        }
    }

    /// Appends the layout's own text, whose markers are tokens.
    fn markup(&mut self, text: &str) -> Result<(), DecisionLayoutError> {
        self.piece(text, &[]).map(drop)
    }

    /// Appends one piece in which the `literal` ranges are caller text, and
    /// returns the piece's span.
    fn piece(
        &mut self,
        text: &str,
        literal: &[Range<usize>],
    ) -> Result<Range<usize>, DecisionLayoutError> {
        let start = self.tokens.len();
        if !text.is_empty() {
            let ids = self
                .encoder
                .encode(text, literal)
                .map_err(DecisionLayoutError::Encode)?;
            self.tokens.extend(ids);
        }
        Ok(start..self.tokens.len())
    }

    /// Appends caller text, which never becomes a marker, and returns its
    /// span.
    fn text(&mut self, text: &str) -> Result<Range<usize>, DecisionLayoutError> {
        self.piece(text, std::slice::from_ref(&(0..text.len())))
    }
}

/// The one id a marker spelling encodes to.
fn marker(encoder: &dyn PromptEncoder, spelling: &str) -> Result<TokenId, DecisionLayoutError> {
    let ids = encoder
        .encode(spelling, &[])
        .map_err(DecisionLayoutError::Encode)?;
    match ids.as_slice() {
        [one] => Ok(*one),
        _ => Err(DecisionLayoutError::Encode(
            shiso_core::CoreError::InvalidValue {
                key: "clef.marker",
                value: spelling.to_owned(),
            },
        )),
    }
}

/// The media runs, as the release's processor writes them: each item's
/// bracketed placeholder, repeated to its row count, then one newline.
fn media_tokens(
    encoder: &dyn PromptEncoder,
    record: &DecisionRecord<'_>,
) -> Result<Vec<TokenId>, DecisionLayoutError> {
    if record.media.is_empty() {
        return Ok(Vec::new());
    }
    let open = marker(encoder, IMAGE_OPEN)?;
    let close = marker(encoder, MEDIA_CLOSE)?;
    let mut tokens = Vec::new();
    for item in record.media {
        let pad = match item.kind {
            DecisionMediaKind::Image => marker(encoder, IMAGE_PAD)?,
            DecisionMediaKind::Video => marker(encoder, VIDEO_PAD)?,
        };
        tokens.push(open);
        tokens.extend(std::iter::repeat_n(pad, item.pads.max(1)));
        tokens.push(close);
    }
    tokens.extend(
        encoder
            .encode("\n", &[])
            .map_err(DecisionLayoutError::Encode)?,
    );
    Ok(tokens)
}

impl DecisionLayout for ClefLayout {
    fn encode(
        &self,
        encoder: &dyn PromptEncoder,
        record: &DecisionRecord<'_>,
    ) -> Result<DecisionPrompt, DecisionLayoutError> {
        check_spellings(record.questions)?;

        // The schema first, at offset zero; it is shifted once the state's
        // length is known.
        let mut schema = Pieces::new(encoder);
        schema.markup("\n\nSCHEMA FIELDS:\n")?;
        let mut spans = Vec::with_capacity(record.questions.len());
        let mut order = Vec::with_capacity(record.questions.len());
        for (index, question) in record.questions.iter().enumerate() {
            let (type_name, kind) = kind_of(&question.answer);
            // One piece, as the release tokenizes it: the id is the caller's,
            // and merges with the text around it.
            let lead = format!("\nFIELD {}\nID: ", index + 1);
            let line = format!("{lead}{}\nTYPE: {type_name}\nINSTRUCTION: ", question.id);
            let id = lead.len()..lead.len() + question.id.len();
            schema.piece(&line, std::slice::from_ref(&id))?;
            let wording = if question.wording.is_empty() {
                question.id.as_str()
            } else {
                question.wording.as_str()
            };
            let span = schema.text(wording)?;
            schema.markup("\nALLOWED OPTIONS:\n")?;
            let options = head_options(&question.answer);
            let mut option_spans = Vec::with_capacity(options.len());
            for (position, (_, option)) in options.iter().enumerate() {
                schema.markup(&format!("OPTION {}: ", position + 1))?;
                option_spans.push(schema.text(&option_text(option))?);
                schema.markup("\n")?;
            }
            schema.markup("END FIELD\n")?;
            order.push(options.iter().map(|(index, _)| *index).collect());
            spans.push(DecisionQuestion {
                kind,
                span,
                options: option_spans,
            });
        }

        let mut prefix = Pieces::new(encoder);
        prefix.markup(&format!(
            "<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"
        ))?;
        prefix.tokens.extend(media_tokens(encoder, record)?);
        let mut suffix = Pieces::new(encoder);
        suffix.markup(
            "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:",
        )?;

        let budget = MAX_PROMPT_TOKENS.min(record.budget);
        let fixed = prefix.tokens.len() + schema.tokens.len() + suffix.tokens.len();
        if fixed > budget {
            return Err(DecisionLayoutError::SchemaTooLong {
                tokens: fixed,
                budget,
            });
        }
        let mut state = Pieces::new(encoder);
        state.text(record.state)?;
        let mut state_tokens = state.tokens;
        state_tokens.truncate(budget - fixed);

        let offset = prefix.tokens.len() + state_tokens.len();
        let shift = |r: &Range<usize>| r.start + offset..r.end + offset;
        let questions = spans
            .iter()
            .map(|q| DecisionQuestion {
                kind: q.kind,
                span: shift(&q.span),
                options: q.options.iter().map(shift).collect(),
            })
            .collect();
        let mut tokens = prefix.tokens;
        tokens.extend(state_tokens);
        tokens.extend(schema.tokens);
        tokens.extend(suffix.tokens);
        Ok(DecisionPrompt {
            tokens,
            questions,
            order,
        })
    }
}
