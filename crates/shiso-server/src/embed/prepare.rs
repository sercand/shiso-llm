//! A request's inputs, turned into the parts the model reads.
//!
//! # The task prompt
//!
//! A model trained with task prompts embeds `task: search result | query:
//! <text>`, not `<text>`, and the two land in different places. The request
//! names the task (`input_type`) and the prompt is the checkpoint's own, by
//! that name: exactly, then ignoring case and punctuation (`search_query`
//! finds `SearchQuery`), then through the two aliases the wire's own
//! vocabulary implies (`search_document` and `passage` are the document side).
//! An unknown task is refused with the names that exist, rather than embedded
//! without a prompt and compared against vectors that had one.
//!
//! The prompt opens the input, whatever follows it - the reference prepends
//! it to a picture-only input too - and is tokenized together with the text
//! that follows it, because the reference tokenizes them as one string and a
//! seam between them would change the ids.

use shiso_core::TokenId;
use shiso_ir::{Block, EmbedInput};
use shiso_models::embedder::{EmbedPart, EmbedderInfo};

use super::Shared;
use crate::error::AppError;
use crate::tokenizer::message_preprocess::media_bytes;

/// The two document-side task names the wire uses that no checkpoint spells.
const DOCUMENT_ALIASES: [&str; 3] = ["searchdocument", "passage", "searchpassage"];

/// Lowercase letters and digits only: how two spellings of one task name are
/// compared.
fn folded(name: &str) -> String {
    name.chars()
        .filter(char::is_ascii_alphanumeric)
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The prompt `task` selects, `None` when the request named no task.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the tasks the model knows, for one it
/// does not.
pub(super) fn prompt_for(
    info: &EmbedderInfo,
    task: Option<&str>,
) -> Result<Option<String>, AppError> {
    let Some(task) = task else {
        return Ok(None);
    };
    let exact = info.prompts.iter().find(|(name, _)| name == task);
    let want = folded(task);
    let want = if DOCUMENT_ALIASES.contains(&want.as_str()) {
        "document".to_owned()
    } else {
        want
    };
    let loose = || info.prompts.iter().find(|(name, _)| folded(name) == want);
    if let Some((_, prompt)) = exact.or_else(loose) {
        return Ok(Some(prompt.clone()));
    }
    let names: Vec<&str> = info.prompts.iter().map(|(n, _)| n.as_str()).collect();
    Err(AppError::bad_request(
        if names.is_empty() {
            format!("input_type {task:?}: this model was trained with no task prompts")
        } else {
            format!("input_type {task:?} is not a task this model knows; it takes {names:?}")
        },
        Some("input_type"),
        "invalid_request",
    ))
}

/// A media failure, as the caller's mistake.
fn bad_media(error: impl std::fmt::Display) -> AppError {
    AppError::bad_request(error.to_string(), Some("input"), "invalid_media")
}

/// The failure for media the model was loaded without a tower for.
fn no_tower(kind: &str) -> AppError {
    AppError::bad_request(
        format!(
            "this embedding model was loaded without a {kind} encoder; \
             start the server with --vision naming the projector"
        ),
        Some("input"),
        "invalid_request",
    )
}

/// Collects text and media into parts, tokenizing each run of text once.
struct Builder<'a> {
    shared: &'a Shared,
    text: String,
    parts: Vec<EmbedPart>,
}

impl Builder<'_> {
    fn flush(&mut self) -> Result<(), AppError> {
        if self.text.is_empty() {
            return Ok(());
        }
        let ids = self
            .shared
            .vocabulary
            .encode(&self.text, false)
            .map_err(|e| AppError::internal("tokenizing embedding input", e))?;
        self.text.clear();
        if !ids.is_empty() {
            self.parts.push(EmbedPart::Text(ids));
        }
        Ok(())
    }

    fn media(&mut self, part: EmbedPart) -> Result<(), AppError> {
        self.flush()?;
        self.parts.push(part);
        Ok(())
    }
}

/// One input's parts.
///
/// # Errors
///
/// [`AppError::BadRequest`] for an input with nothing in it, media this
/// model has no encoder for or that does not decode, and a content block an
/// embedding has no meaning for.
pub(super) fn parts(
    shared: &Shared,
    prompt: Option<&str>,
    input: &EmbedInput,
) -> Result<Vec<EmbedPart>, AppError> {
    let blocks = match input {
        EmbedInput::Tokens(ids) => {
            if ids.is_empty() {
                return Err(AppError::bad_request(
                    "an input of no tokens",
                    Some("input"),
                    "invalid_request",
                ));
            }
            let mut all = match prompt {
                Some(p) => shared
                    .vocabulary
                    .encode(p, false)
                    .map_err(|e| AppError::internal("tokenizing the task prompt", e))?,
                None => Vec::new(),
            };
            all.extend(ids.iter().copied().map(TokenId::new));
            return Ok(vec![EmbedPart::Text(all)]);
        }
        EmbedInput::Content(blocks) => blocks,
    };
    let carries_something = blocks.iter().any(|b| match b {
        Block::Text(t) => !t.is_empty(),
        _ => true,
    });
    if !carries_something {
        return Err(AppError::bad_request(
            "an input with no content",
            Some("input"),
            "invalid_request",
        ));
    }
    let rules = &shared.media;
    let front = &shared.frontend;
    let mut b = Builder {
        shared,
        text: prompt.unwrap_or_default().to_owned(),
        parts: Vec::new(),
    };
    for block in blocks {
        match block {
            Block::Text(text) => b.text.push_str(text),
            Block::Image(source) => {
                let spec = front.image.as_ref().ok_or_else(|| no_tower("vision"))?;
                let bytes = media_bytes(source, rules.allow_remote).map_err(bad_media)?;
                let item = shiso_vision::preprocess_image(&bytes, spec, rules.image_limits)
                    .map_err(bad_media)?;
                b.media(EmbedPart::Image(item))?;
            }
            Block::Video(source) => {
                let video = front.video.as_ref().ok_or_else(|| no_tower("vision"))?;
                let bytes = media_bytes(source, rules.allow_remote).map_err(bad_media)?;
                let clip =
                    shiso_vision::decode_clip(&bytes, video.fps, rules.video_ffmpeg.as_ref())
                        .map_err(bad_media)?;
                let frames = video
                    .sample(clip.frames().len(), clip.fps())
                    .into_iter()
                    .map(|i| {
                        shiso_vision::preprocess_video(
                            std::slice::from_ref(&clip.frames()[i]),
                            &video.frames,
                            rules.image_limits,
                        )
                        .map_err(bad_media)
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                b.media(EmbedPart::Video(frames))?;
            }
            Block::Audio(source) => {
                let tower = front.audio.as_ref().ok_or_else(|| no_tower("audio"))?;
                let bytes = media_bytes(source, rules.allow_remote).map_err(bad_media)?;
                let pcm = shiso_audio::decode_audio(
                    &bytes,
                    &rules.audio_limits,
                    rules.audio_ffmpeg.as_ref(),
                )
                .map_err(bad_media)?;
                let clip = shiso_audio::clip_for(tower.as_ref(), &pcm).map_err(bad_media)?;
                b.media(EmbedPart::Audio(clip))?;
            }
            other => {
                return Err(AppError::bad_request(
                    format!(
                        "a {} block has no meaning in an embedding input",
                        kind(other)
                    ),
                    Some("input"),
                    "invalid_request",
                ));
            }
        }
    }
    b.flush()?;
    Ok(b.parts)
}

/// A block's kind, for a refusal.
fn kind(block: &Block) -> &'static str {
    match block {
        Block::Thinking(_) => "thinking",
        Block::ToolUse(_) => "tool-use",
        Block::ToolResult(_) => "tool-result",
        Block::Refusal(_) => "refusal",
        _ => "content",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info() -> EmbedderInfo {
        EmbedderInfo {
            model_id: "m".to_owned(),
            dim: 4,
            truncations: vec![4],
            max_tokens: 16,
            prompts: vec![
                (
                    "SearchQuery".to_owned(),
                    "task: search result | query: ".to_owned(),
                ),
                ("Document".to_owned(), "title: none | text: ".to_owned()),
            ],
        }
    }

    #[test]
    fn a_task_finds_its_prompt_by_any_spelling() {
        let info = info();
        let get = |t| {
            prompt_for(&info, Some(t))
                .expect("known")
                .expect("a prompt")
        };
        assert_eq!(get("SearchQuery"), "task: search result | query: ");
        assert_eq!(get("search_query"), "task: search result | query: ");
        assert_eq!(get("search_document"), "title: none | text: ");
        assert_eq!(get("passage"), "title: none | text: ");
        assert_eq!(prompt_for(&info, None).expect("no task"), None);
    }

    #[test]
    fn an_unknown_task_is_refused_with_the_known_ones() {
        let error = prompt_for(&info(), Some("summarize")).expect_err("unknown");
        let message = error.to_string();
        assert!(
            message.contains("SearchQuery") && message.contains("Document"),
            "{message}"
        );
    }
}
