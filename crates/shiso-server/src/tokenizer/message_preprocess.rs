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
//! Every media block contributes one item and one pad count, in block order,
//! and the two vectors are built in the same pass. The count is the item's own
//! [`shiso_vision::VisionItem::pad_count`] rather than a number computed here,
//! because the pad run in the prompt and the embedding rows the encoder returns
//! have to be the same number and there is exactly one function that says what
//! it is.

use serde_json::{Value, json};
use shiso_audio::{AudioError, AudioTower, DecodeLimits};
use shiso_ir::{Block, MediaSource, Message, Role};
use shiso_vision::{FfmpegPolicy, ImageLimits, PreprocessSpec, VisionError};

use crate::media::{EncodedItem, MediaItem};
use crate::media_cache::{MediaCache, payload_key};
use shiso_core::media::MediaDigest;

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
    /// This checkpoint's audio tower. `None` means an audio part is a request
    /// error rather than a decode.
    pub audio: Option<&'a dyn AudioTower>,
    /// Decode limits for one audio clip.
    pub audio_limits: DecodeLimits,
    /// The subprocess decoder for audio containers other than WAV, or `None`
    /// when only WAV is accepted. Separate from the video policy because the
    /// two are switched on by separate flags, and a refusal names its own.
    pub audio_ffmpeg: Option<&'a FfmpegPolicy>,
    /// Where an item seen before is found without decoding it. `None` decodes
    /// every item, as every request did before there was one.
    pub cache: Option<&'a MediaCache>,
}

/// The audio decode cap when nothing else was configured: the longest clip the
/// one shipped audio tower accepts.
pub const DEFAULT_AUDIO_LIMITS: DecodeLimits =
    DecodeLimits::new(std::num::NonZeroU32::MIN.saturating_add(29));

impl Default for MediaPolicy<'_> {
    fn default() -> Self {
        Self {
            vision: None,
            limits: ImageLimits::default(),
            allow_remote: false,
            ffmpeg: None,
            video_fps: shiso_vision::DEFAULT_FPS,
            audio: None,
            audio_limits: DEFAULT_AUDIO_LIMITS,
            audio_ffmpeg: None,
            cache: None,
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
    /// The request carries audio and this checkpoint has no audio tower.
    #[error("this model has no audio tower, so it cannot be sent audio")]
    NoAudioTower,
    /// The audio layer refused the payload: a container this build cannot
    /// decode, a clip past the limit, or one too short to make a token.
    #[error(transparent)]
    Audio(#[from] AudioError),
}

/// A conversation, ready for a template and for the engine.
#[derive(Debug, Default)]
pub struct Prepared {
    /// The turns, in the shape a template reads.
    pub messages: Vec<Value>,
    /// The media, in the order the prompt refers to it.
    pub media: Vec<MediaItem>,
    /// How many pad tokens each item occupies, index for index with `media`.
    pub pad_counts: Vec<usize>,
    /// Each item's content digest, index for index with `media`: what the
    /// prefix cache files its placeholder run under.
    pub digests: Vec<MediaDigest>,
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
    // The calls ride beside the content and only there, because that is where
    // a template looks for them. A copy inside the content list is a part type
    // the shipped templates do not accept: they raise on it, and every replayed
    // assistant turn that had called a tool became a refused request.
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
            // A trace and a call are not content: `flush` puts them where every
            // template looks for them, `reasoning_content` and `tool_calls`. The
            // templates checkpoints ship accept only text and media parts in
            // `content` and raise on anything else, so a copy here turned every
            // replayed assistant turn into a refused request.
            Block::Thinking(_) | Block::ToolUse(_) => {}
            // A nested tool result is only reachable inside another tool
            // result's content, which no wire format produces; rendering it as
            // its text keeps the function total without inventing a shape.
            Block::ToolResult(result) => {
                let inner = render_blocks(&result.content, policy, out)?;
                parts.extend(inner);
            }
            Block::Image(source) => {
                let spec = tower(policy, "an image")?;
                push_media("an image", source, policy, out, |bytes| {
                    let item = shiso_vision::preprocess_image(bytes, spec, policy.limits)?;
                    Ok(MediaItem::Vision(item))
                })?;
                parts.push(json!({"type": "image"}));
            }
            Block::Video(source) => {
                let spec = tower(policy, "a video")?;
                push_media("a video", source, policy, out, |bytes| {
                    let clip = shiso_vision::decode_clip(bytes, policy.video_fps, policy.ffmpeg)?;
                    let frames =
                        clip.into_sampled(policy.video_fps, spec.geometry.temporal_patch_size());
                    let item = shiso_vision::preprocess_video(&frames, spec, policy.limits)?;
                    Ok(MediaItem::Vision(item))
                })?;
                parts.push(json!({"type": "video"}));
            }
            Block::Audio(source) => {
                let tower = policy.audio.ok_or(PreprocessError::NoAudioTower)?;
                push_media("audio", source, policy, out, |bytes| {
                    let pcm = shiso_audio::decode_audio(
                        bytes,
                        &policy.audio_limits,
                        policy.audio_ffmpeg,
                    )?;
                    Ok(MediaItem::Audio(shiso_audio::clip_for(tower, &pcm)?))
                })?;
                parts.push(json!({"type": "audio"}));
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

/// Records one media item, the pad run it will occupy and its digest,
/// decoding it only when the cache cannot stand in for it.
///
/// A payload the cache recognises, whose rows are still resident, becomes a
/// [`MediaItem::Encoded`] holding those rows: nothing is base64-decoded,
/// decoded, resized or hashed. Anything else is decoded by `decode`, digested
/// once, and the digest remembered against the payload for the next request
/// that carries it.
fn push_media(
    kind: &'static str,
    source: &MediaSource,
    policy: &MediaPolicy<'_>,
    out: &mut Prepared,
    decode: impl FnOnce(&[u8]) -> Result<MediaItem, PreprocessError>,
) -> Result<(), PreprocessError> {
    let merge = policy
        .vision
        .map_or(1, |spec| spec.geometry.spatial_merge_size());
    let key = match (policy.cache, source) {
        (Some(_), MediaSource::Inline(value)) => Some(payload_key(kind, value)),
        _ => None,
    };
    let known = policy
        .cache
        .zip(key.as_ref())
        .and_then(|(cache, key)| cache.rows_for_payload(key));
    let (item, digest) = if let Some((digest, rows)) = known {
        (MediaItem::Encoded(EncodedItem { digest, rows }), digest)
    } else {
        let bytes = payload(source, policy, kind)?;
        let item = decode(&bytes)?;
        let digest = item.digest();
        if let (Some(cache), Some(key)) = (policy.cache, key) {
            cache.remember_payload(key, digest);
        }
        (item, digest)
    };
    out.pad_counts.push(item.pad_count(merge));
    out.digests.push(digest);
    out.media.push(item);
    Ok(())
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

/// The bytes behind a media source under a remote policy alone: the
/// embeddings route decodes the same sources as a chat turn and must refuse
/// an address for the same reason, in the same words.
///
/// # Errors
///
/// The refusals of the chat path: a remote address, bytes that are not
/// base64, a source spelling this build does not know.
pub(crate) fn media_bytes(
    source: &MediaSource,
    allow_remote: bool,
) -> Result<Vec<u8>, PreprocessError> {
    let policy = MediaPolicy {
        allow_remote,
        ..MediaPolicy::default()
    };
    payload(source, &policy, "media")
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

/// The brackets a family writes around each kind of media run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MediaWraps {
    /// Around an image's run.
    pub image: Option<(shiso_core::TokenId, shiso_core::TokenId)>,
    /// Around an audio clip's run.
    pub audio: Option<(shiso_core::TokenId, shiso_core::TokenId)>,
}

/// Repeats each media marker to its item's pad count, bracketing image runs
/// only; the audio-free spelling of [`expand_media_pads`] that keeps only the
/// tokens.
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
    expand_media_pads(
        tokens,
        pad_counts,
        pads,
        MediaWraps {
            image: wrap,
            audio: None,
        },
    )
    .map(|(tokens, _ranges)| tokens)
}

/// Repeats each media marker `pad_counts[i]` times, in prompt order, with the
/// bracket its kind carries, and reports where each item landed.
///
/// A marker is bracketed by its own kind's pair only: a clip has no bracket in
/// families that have no clip marker at all, and a bracket invented for one
/// would be a prompt shape no checkpoint was trained on.
///
/// The second half of the returned pair is each item's expanded token range
/// `[start, end)` - open bracket through close bracket inclusive, for a
/// bracketed family - index for index with `pad_counts`. The renderer pairs
/// those with the marker's byte range in the text so a caller downstream can
/// find the tail after an item without ever seeing a marker string.
///
/// # Errors
///
/// [`PadMismatch`] when the marker count and the item count disagree.
pub fn expand_media_pads(
    tokens: Vec<shiso_core::TokenId>,
    pad_counts: &[usize],
    pads: shiso_core::PadTokens,
    wraps: MediaWraps,
) -> Result<(Vec<shiso_core::TokenId>, Vec<std::ops::Range<usize>>), PadMismatch> {
    let markers = tokens.iter().filter(|token| pads.matches(**token)).count();
    if markers != pad_counts.len() {
        return Err(PadMismatch {
            markers,
            items: pad_counts.len(),
        });
    }
    if pad_counts.is_empty() {
        return Ok((tokens, Vec::new()));
    }
    let extra: usize = pad_counts
        .iter()
        .map(|count| count.saturating_sub(1).saturating_add(2))
        .sum();
    let mut out = Vec::with_capacity(tokens.len() + extra);
    let mut ranges = Vec::with_capacity(pad_counts.len());
    let mut item = 0usize;
    for token in tokens {
        if pads.matches(token) {
            let start = out.len();
            let count = pad_counts.get(item).copied().unwrap_or(1).max(1);
            let bracket = if Some(token) == pads.image() {
                wraps.image
            } else if Some(token) == pads.audio() {
                wraps.audio
            } else {
                None
            };
            if let Some((open, _close)) = bracket {
                out.push(open);
            }
            out.extend(std::iter::repeat_n(token, count));
            if let Some((_open, close)) = bracket {
                out.push(close);
            }
            ranges.push(start..out.len());
            item += 1;
        } else {
            out.push(token);
        }
    }
    Ok((out, ranges))
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

#[cfg(test)]
mod tests {
    use super::{MediaPolicy, preprocess};
    use shiso_ir::{Block, Message, Role, Thinking, ToolUse};

    /// A replayed assistant turn reaches the template with its trace in
    /// `reasoning_content`, its calls in `tool_calls`, and neither in `content`:
    /// the templates checkpoints ship accept only text and media parts there and
    /// raise on anything else.
    #[test]
    fn an_assistant_turn_keeps_traces_and_calls_out_of_its_content() {
        let turn = Message::new(
            Role::Assistant,
            vec![
                Block::Thinking(Thinking::new("weighing it")),
                Block::Text("done".to_owned()),
                Block::ToolUse(ToolUse::new(
                    "call_1",
                    "read_file",
                    serde_json::json!({"path": "a"}),
                )),
            ],
        );
        let prepared = preprocess(&[], &[turn], &MediaPolicy::default()).expect("text cannot fail");
        let rendered = &prepared.messages[0];
        assert_eq!(
            rendered["content"],
            serde_json::json!([{"type": "text", "text": "done"}])
        );
        assert_eq!(rendered["reasoning_content"], "weighing it");
        assert_eq!(rendered["tool_calls"][0]["function"]["name"], "read_file");
    }
}
