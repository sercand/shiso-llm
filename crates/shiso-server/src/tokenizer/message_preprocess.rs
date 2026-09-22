//! Turning the neutral conversation into the value shape a chat template reads,
//! and pulling the media out of it on the way.
//!
//! # The template's shape is not the IR's shape, and that is deliberate
//!
//! The IR has two roles and puts every kind of content in one ordered block
//! list, because that is the shape that survives being appended to the next
//! request. Chat templates were written against a different shape: a tool
//! result is a `tool` role, a tool call is a `tool_calls` array beside the
//! content, and a reasoning trace is its own key. Those templates ship inside
//! checkpoints and cannot be changed, so this module is where the two shapes
//! meet.
//!
//! The translation only ever goes this way. Nothing here parses a template's
//! output back into the IR, and nothing here is a round trip - which is why the
//! IR is free to be the better shape rather than the one a template happens to
//! want.
//!
//! # Why the tool results become their own turns
//!
//! An IR user turn may carry tool results and prose in one block list. A
//! template expects the results as separate `tool` entries in the order they
//! answer in. Splitting one turn into several is therefore not a loss of
//! information: the block order is preserved exactly, and what changes is only
//! which of them the template sees as a turn boundary.
//!
//! # Media leaves as items and pad counts, together
//!
//! Every media block contributes one [`VisionItem`] and one pad count, in block
//! order, and the two vectors are built in the same pass. The count is the
//! item's own [`VisionItem::pad_count`] rather than a number computed here,
//! because the pad run in the prompt and the embedding rows the encoder returns
//! have to be the same number and there is exactly one function that says what
//! it is.

use serde_json::{Value, json};
use shiso_ir::{Block, MediaSource, Message, Role};
use shiso_vision::{FfmpegPolicy, ImageLimits, PreprocessSpec, VisionError, VisionItem};

/// What the media flags allow, and what the checkpoint's tower needs.
///
/// Borrowed rather than owned because one of these is built per request from
/// state that outlives it, and the pixel limits are `Copy` while the tower
/// config is not.
#[derive(Debug, Clone, Copy)]
pub struct MediaPolicy<'a> {
    /// How this checkpoint's tower wants a frame preprocessed: its geometry,
    /// its resize rule and its normalization. `None` means this checkpoint has
    /// no vision tower, so a media block is a request error rather than a
    /// decode.
    pub vision: Option<&'a PreprocessSpec>,
    /// Decode limits for one still.
    pub limits: ImageLimits,
    /// Whether an http or https address may be fetched.
    pub allow_remote: bool,
    /// The subprocess decode policy, or `None` when video decoding is off.
    pub ffmpeg: Option<&'a FfmpegPolicy>,
    /// The rate a clip is sampled at.
    pub video_fps: f64,
}

impl Default for MediaPolicy<'_> {
    fn default() -> Self {
        Self {
            vision: None,
            limits: ImageLimits::default(),
            allow_remote: false,
            ffmpeg: None,
            video_fps: shiso_vision::DEFAULT_FPS,
        }
    }
}

/// Why a conversation could not be prepared.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PreprocessError {
    /// The request carries media and this checkpoint has no vision tower.
    #[error("this model has no vision tower, so it cannot be sent {kind}")]
    NoVisionTower {
        /// What was sent: an image or a video.
        kind: &'static str,
    },
    /// The request names an address the server would have to fetch.
    ///
    /// Refused before any network call is made, and refused by default: a
    /// server that fetches whatever a request names is a request-forgery
    /// primitive that happens to also caption pictures.
    #[error(
        "remote media is not allowed: {url}; send the bytes inline, \
         or start the server with --vision-allow-remote-images"
    )]
    RemoteRefused {
        /// The address as the request spelled it.
        url: String,
    },
    /// Remote media is allowed by policy and this build cannot fetch it.
    ///
    /// Distinct from the refusal above so that an operator who turned the flag
    /// on is told the truth - nothing here talks outbound - rather than being
    /// told to turn on a flag that is already on.
    #[error("remote media is allowed but this build has no outbound fetch: {url}")]
    RemoteUnsupported {
        /// The address as the request spelled it.
        url: String,
    },
    /// The inline payload is not decodable base64.
    #[error("the inline media payload is not valid base64")]
    NotBase64,
    /// The media source is spelled in a way this build does not understand.
    #[error("this build does not know how to reach the media source {what}")]
    UnknownSource {
        /// The source, as it was described.
        what: String,
    },
    /// The media layer refused the payload.
    #[error(transparent)]
    Media(#[from] VisionError),
}

/// A conversation, ready for a template and for the engine.
#[derive(Debug, Default)]
pub struct Prepared {
    /// The turns, in the shape a template reads.
    pub messages: Vec<Value>,
    /// The media, in the order the prompt refers to it.
    pub media: Vec<VisionItem>,
    /// How many pad tokens each item occupies, index for index with `media`.
    pub pad_counts: Vec<usize>,
}

/// Prepares a conversation.
///
/// `system` is rendered as a leading `system` turn, because that is what a
/// template expects; it is a field on the request rather than a turn in the IR
/// because instructions are not something a speaker said.
///
/// # Errors
///
/// [`PreprocessError`] for media this server will not or cannot decode. Text
/// alone cannot fail.
pub fn preprocess(
    system: &[Block],
    messages: &[Message],
    policy: &MediaPolicy<'_>,
) -> Result<Prepared, PreprocessError> {
    let mut out = Prepared::default();
    if !system.is_empty() {
        let parts = render_blocks(system, policy, &mut out)?;
        out.messages
            .push(json!({"role": "system", "content": parts}));
    }
    for message in messages {
        push_turn(message, policy, &mut out)?;
    }
    Ok(out)
}

/// Renders one IR turn as one or more template turns.
fn push_turn(
    message: &Message,
    policy: &MediaPolicy<'_>,
    out: &mut Prepared,
) -> Result<(), PreprocessError> {
    let mut pending: Vec<&Block> = Vec::new();
    for block in &message.content {
        if let Block::ToolResult(result) = block {
            flush(message.role, &mut pending, policy, out)?;
            let content = render_blocks(&result.content, policy, out)?;
            out.messages.push(json!({
                "role": "tool",
                "tool_call_id": result.tool_use_id,
                "is_error": result.is_error,
                "content": content,
            }));
        } else {
            pending.push(block);
        }
    }
    flush(message.role, &mut pending, policy, out)
}

/// Emits whatever has accumulated as one turn of `role`.
fn flush(
    role: Role,
    pending: &mut Vec<&Block>,
    policy: &MediaPolicy<'_>,
    out: &mut Prepared,
) -> Result<(), PreprocessError> {
    if pending.is_empty() {
        return Ok(());
    }
    let blocks: Vec<Block> = pending.drain(..).cloned().collect();
    let content = render_blocks(&blocks, policy, out)?;
    let mut turn = serde_json::Map::new();
    turn.insert("role".to_owned(), Value::String(role.as_str().to_owned()));
    turn.insert("content".to_owned(), Value::Array(content));
    // The calls also ride beside the content, because that is where a template
    // looks for them. They stay inside the content list too, so the block order
    // a template that reads the list sees is still the order the model produced.
    let calls: Vec<Value> = blocks
        .iter()
        .filter_map(|block| match block {
            Block::ToolUse(use_) => Some(json!({
                "id": use_.id,
                "type": "function",
                "function": {"name": use_.name, "arguments": use_.input},
            })),
            _ => None,
        })
        .collect();
    if !calls.is_empty() {
        turn.insert("tool_calls".to_owned(), Value::Array(calls));
    }
    if let Some(trace) = blocks.iter().find_map(|block| match block {
        Block::Thinking(thinking) => Some(thinking.text.clone()),
        _ => None,
    }) {
        turn.insert("reasoning_content".to_owned(), Value::String(trace));
    }
    out.messages.push(Value::Object(turn));
    Ok(())
}

/// Renders a block list as the template's content parts, decoding media.
fn render_blocks(
    blocks: &[Block],
    policy: &MediaPolicy<'_>,
    out: &mut Prepared,
) -> Result<Vec<Value>, PreprocessError> {
    let mut parts = Vec::with_capacity(blocks.len());
    for block in blocks {
        match block {
            // A refusal is rendered as the text it is: a template has no
            // refusal concept, and the words shown in place of the withheld
            // output are what the next turn has to be able to refer to.
            Block::Text(text) | Block::Refusal(text) => {
                parts.push(json!({"type": "text", "text": text}));
            }
            Block::Thinking(thinking) => {
                parts.push(json!({"type": "thinking", "thinking": thinking.text}));
            }
            Block::ToolUse(use_) => parts.push(json!({
                "type": "tool_use",
                "id": use_.id,
                "name": use_.name,
                "input": use_.input,
            })),
            // A nested tool result is only reachable inside another tool
            // result's content, which no wire format produces; rendering it as
            // its text keeps the function total without inventing a shape.
            Block::ToolResult(result) => {
                let inner = render_blocks(&result.content, policy, out)?;
                parts.extend(inner);
            }
            Block::Image(source) => {
                let bytes = payload(source, policy, "an image")?;
                let item = shiso_vision::preprocess_image(
                    &bytes,
                    tower(policy, "an image")?,
                    policy.limits,
                )?;
                push_item(item, policy, out);
                parts.push(json!({"type": "image"}));
            }
            Block::Video(source) => {
                let bytes = payload(source, policy, "a video")?;
                let spec = tower(policy, "a video")?;
                let clip = shiso_vision::decode_clip(&bytes, policy.video_fps, policy.ffmpeg)?;
                let frames =
                    clip.into_sampled(policy.video_fps, spec.geometry.temporal_patch_size());
                let item = shiso_vision::preprocess_video(&frames, spec, policy.limits)?;
                push_item(item, policy, out);
                parts.push(json!({"type": "video"}));
            }
            // The block list is open: a modality added to the IR after this was
            // written has no rendering here, and dropping it silently would send
            // the model a turn that is missing content the caller sent. Saying
            // so in the prompt is the least bad of the available answers, and
            // the log line is what an operator sees.
            other => {
                tracing::warn!(?other, "a content block has no rendering in this build");
                parts.push(json!({"type": "text", "text": ""}));
            }
        }
    }
    Ok(parts)
}

/// Records one decoded item and the pad run it will occupy.
fn push_item(item: VisionItem, policy: &MediaPolicy<'_>, out: &mut Prepared) {
    let merge = policy
        .vision
        .map_or(1, |spec| spec.geometry.spatial_merge_size());
    out.pad_counts.push(item.pad_count(merge));
    out.media.push(item);
}

/// The tower's preprocessing spec, or the error a text-only checkpoint owes a
/// media request.
fn tower<'a>(
    policy: &MediaPolicy<'a>,
    kind: &'static str,
) -> Result<&'a PreprocessSpec, PreprocessError> {
    policy.vision.ok_or(PreprocessError::NoVisionTower { kind })
}

/// The bytes behind a media source, under the remote policy.
fn payload(
    source: &MediaSource,
    policy: &MediaPolicy<'_>,
    _kind: &'static str,
) -> Result<Vec<u8>, PreprocessError> {
    match source {
        // Either way the answer is no, and either way it is reached before any
        // socket is opened. What differs is which of the two facts the operator
        // is told: that the policy forbids it, or that the policy allows
        // something this build cannot do.
        MediaSource::Remote(url) if policy.allow_remote => {
            Err(PreprocessError::RemoteUnsupported { url: url.clone() })
        }
        MediaSource::Remote(url) => Err(PreprocessError::RemoteRefused { url: url.clone() }),
        MediaSource::Inline(value) => decode_inline(value),
        // A source spelling added after this was written is not one this server
        // knows how to reach, and guessing would be reaching for bytes nobody
        // authorised.
        other => Err(PreprocessError::UnknownSource {
            what: format!("{other:?}"),
        }),
    }
}

/// Decodes an inline payload, with or without a data-URI wrapper.
fn decode_inline(value: &str) -> Result<Vec<u8>, PreprocessError> {
    use base64::Engine as _;
    // A data URI carries its own base64 after a comma; raw base64 has no comma
    // at all. Splitting on the marker rather than parsing the URI keeps the two
    // spellings on one path.
    let encoded = value
        .strip_prefix("data:")
        .and_then(|rest| rest.split_once(',').map(|(_meta, data)| data))
        .unwrap_or(value);
    base64::engine::general_purpose::STANDARD
        .decode(encoded.trim())
        .map_err(|_bad| PreprocessError::NotBase64)
}

/// Expands each pad marker into the run of pads its item needs.
///
/// A chat template writes exactly one pad marker per media item, because that
/// is how many the item is in the conversation. The encoder produces one
/// embedding row per merged patch, and the prompt has to have one position for
/// each of those rows, or every embedding after the first item lands on the
/// wrong token. This is the fan-out.
///
/// A count list that does not match the markers found is left alone rather than
/// half-expanded: a mismatch means the template and the preprocessor disagree
/// about the conversation, and guessing which is right produces a prompt that
/// is silently about the wrong picture.
///
/// # The bracket, and why it is ordinary tokens
///
/// A family whose template brackets the run - Gemma writes `<|image>` before it
/// and `<image|>` after - passes `wrap`, and each marker becomes
/// `open + marker * count + close`. The two bracket tokens are ORDINARY tokens
/// to everything downstream: the engine's media walk finds a picture by looking
/// for a run of the placeholder id, the splice map counts only that run, and a
/// bracket token that shared the placeholder's id would lengthen every item's
/// run by two and put every embedding after the first on the wrong token.
///
/// The bracket goes around the image marker only. The families that bracket
/// have no clip marker at all, and a bracket invented for one would be a prompt
/// shape no checkpoint was trained on.
///
/// # Errors
///
/// [`PadMismatch`] when the marker count and the item count disagree.
pub fn expand_vision_pads(
    tokens: Vec<shiso_core::TokenId>,
    pad_counts: &[usize],
    pads: shiso_core::PadTokens,
    wrap: Option<(shiso_core::TokenId, shiso_core::TokenId)>,
) -> Result<Vec<shiso_core::TokenId>, PadMismatch> {
    let markers = tokens.iter().filter(|token| pads.matches(**token)).count();
    if markers != pad_counts.len() {
        return Err(PadMismatch {
            markers,
            items: pad_counts.len(),
        });
    }
    if pad_counts.is_empty() {
        return Ok(tokens);
    }
    let extra: usize = pad_counts
        .iter()
        .map(|count| count.saturating_sub(1).saturating_add(2))
        .sum();
    let mut out = Vec::with_capacity(tokens.len() + extra);
    let mut item = 0usize;
    for token in tokens {
        if pads.matches(token) {
            let count = pad_counts.get(item).copied().unwrap_or(1).max(1);
            let bracket = wrap.filter(|_| Some(token) == pads.image());
            if let Some((open, _close)) = bracket {
                out.push(open);
            }
            out.extend(std::iter::repeat_n(token, count));
            if let Some((_open, close)) = bracket {
                out.push(close);
            }
            item += 1;
        } else {
            out.push(token);
        }
    }
    Ok(out)
}

/// The rendered prompt has a different number of pad markers than there are
/// media items.
#[derive(Debug, thiserror::Error)]
#[error("the rendered prompt has {markers} media marker(s) for {items} media item(s)")]
pub struct PadMismatch {
    /// Markers the template wrote.
    pub markers: usize,
    /// Items the preprocessor decoded.
    pub items: usize,
}
