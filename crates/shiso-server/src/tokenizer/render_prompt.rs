//! The one entry every surface renders a prompt through.
//!
//! # Why this is a type and not a function
//!
//! Rendering needs three things that are chosen when the process starts: the
//! checkpoint's chat template, its vocabulary, and whatever the operator allowed
//! the media layer to do. A free function could only reach those through a
//! process-wide cell, which makes the renderer untestable with a second template
//! and turns initialisation order into a runtime failure. So one of these is
//! built at boot and lives in the application state, and every surface borrows
//! the same one.
//!
//! # Why the rendered text is returned beside the tokens
//!
//! The text is what a prefix cache keys on and what a golden test pins. Deriving
//! it a second time - by decoding the tokens, or by rendering again somewhere
//! else - produces a string that is *nearly* the prompt, and a cache keyed on
//! nearly the prompt is a cache that occasionally answers with somebody else's
//! conversation.
//!
//! # What this does not decide
//!
//! How many tokens may be generated, what the sampler does, and whether the
//! prompt fits: those need the server's context and its preset, and they belong
//! to the pipeline that owns both. This turns a conversation into tokens and
//! stops.

use std::sync::Arc;

use shiso_core::{PadTokens, TokenId, VisionConfig};
use shiso_ir::{Block, Effort, Message, Request, Role, ThinkingDirective};
use shiso_vision::{FfmpegPolicy, ImageLimits, MediaPlacement, PreprocessSpec};

use shiso_audio::{AudioTower, DecodeLimits};

use crate::media::MediaItem;
use crate::media_cache::MediaCache;
use shiso_core::media::MediaDigest;

use crate::error::AppError;
use crate::tokenizer::Vocabulary;
use crate::tokenizer::chat_render::{ChatTemplate, RenderFlags};
use crate::tokenizer::jinja_helpers::ToolJsonStyle;
use crate::tokenizer::literal_text;
use crate::tokenizer::message_preprocess::{
    DEFAULT_AUDIO_LIMITS, MediaPolicy, MediaWraps, expand_media_pads, preprocess,
};
use crate::tokenizer::turn_replay::TurnReplay;

/// Where one media item sits in a rendered prompt: the token range it
/// occupies in [`Rendered::tokens`] (open bracket + pad run + close bracket,
/// as expanded) and the byte range of its marker string in [`Rendered::text`].
///
/// The two ranges end where the item ends on each side, which is what lets a
/// caller compare the text after the marker with the tokens after the run
/// without ever seeing a marker string: the brackets are added token-side
/// only, and the marker is one special token, so both tails begin at the
/// same piece of the caller's prompt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MediaSpan {
    /// The item's run in the expanded token stream, `[start, end)`.
    pub tokens: std::ops::Range<usize>,
    /// The item's marker string in the rendered text, as byte offsets.
    pub text: std::ops::Range<usize>,
}

/// Media decoded for a prompt the template does not write.
#[derive(Debug)]
pub struct DecodedMedia {
    /// The decoded media, in the order the blocks gave it.
    pub media: Vec<MediaItem>,
    /// Each item's content digest, index for index with `media`.
    pub digests: Vec<MediaDigest>,
    /// How many placeholder rows each item occupies, index for index with
    /// `media`.
    pub pad_counts: Vec<usize>,
}

/// A rendered prompt, in all three of the forms something downstream needs.
#[derive(Debug, Clone)]
pub struct Rendered {
    /// The prompt, tokenized, with every media marker already fanned out to the
    /// run of pads its item occupies.
    pub tokens: Vec<TokenId>,
    /// The prompt text, byte for byte as the template produced it.
    pub text: String,
    /// The decoded media, in the order the prompt refers to it.
    pub media: Vec<MediaItem>,
    /// Each item's content digest, index for index with `media`.
    pub digests: Vec<MediaDigest>,
    /// Where each item sits, index for index with `media`. Empty when the
    /// prompt carries no media.
    pub media_spans: Vec<MediaSpan>,
}

impl Rendered {
    /// How many tokens the prompt occupies.
    #[must_use]
    pub fn token_count(&self) -> u64 {
        u64::try_from(self.tokens.len()).unwrap_or(u64::MAX)
    }
}

/// How one family's chat template spells each effort rung.
///
/// Templates validate the word against a fixed list and raise on any other, so
/// the spelling is part of a family's contract with its template rather than a
/// label. Every rung has a word: a table with a hole would turn a valid request
/// into a template exception.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EffortSpelling {
    /// The word for [`Effort::Minimal`].
    pub minimal: &'static str,
    /// The word for [`Effort::Low`].
    pub low: &'static str,
    /// The word for [`Effort::Medium`].
    pub medium: &'static str,
    /// The word for [`Effort::High`].
    pub high: &'static str,
    /// The word for [`Effort::XHigh`].
    pub xhigh: &'static str,
}

impl EffortSpelling {
    /// The words most templates accept: every rung by its own name, with the
    /// rung below low folded into low.
    pub const STANDARD: Self = Self {
        minimal: "low",
        low: "low",
        medium: "medium",
        high: "high",
        xhigh: "xhigh",
    };

    /// For templates that accept only low, medium and xhigh: the top two rungs
    /// share xhigh.
    pub const XHIGH_TOP: Self = Self {
        minimal: "low",
        low: "low",
        medium: "medium",
        high: "xhigh",
        xhigh: "xhigh",
    };

    /// The word this family's template accepts for `effort`.
    #[must_use]
    pub const fn spell(self, effort: Effort) -> &'static str {
        match effort {
            Effort::Minimal => self.minimal,
            Effort::Low => self.low,
            Effort::Medium => self.medium,
            Effort::High => self.high,
            Effort::XHigh => self.xhigh,
        }
    }
}

/// Everything needed to turn a conversation into a prompt.
#[derive(Debug)]
pub struct PromptRenderer {
    template: ChatTemplate,
    vocabulary: Arc<Vocabulary>,
    vision: Option<PreprocessSpec>,
    /// The placeholder ids this checkpoint's vocabulary gives the family's
    /// markers, resolved once when the tower was attached.
    pads: PadTokens,
    /// Every special-token spelling this vocabulary declares: what caller
    /// text is kept from turning into a token.
    protected: literal_text::Protected,
    /// The bracket ids, for a family that brackets an image's run.
    wrap: Option<(TokenId, TokenId)>,
    /// The bracket around an audio clip's run.
    audio_wrap: Option<(TokenId, TokenId)>,
    /// The family's placement as the tower attached it: the marker *strings*
    /// the template writes, kept so `render` can locate each item's marker in
    /// the final text and report [`MediaSpan`]s. `Copy`, and the last tower
    /// attached wins exactly as it does for `pads`.
    placement: Option<MediaPlacement>,
    limits: ImageLimits,
    allow_remote: bool,
    ffmpeg: Option<FfmpegPolicy>,
    video_fps: f64,
    thinking_default: bool,
    effort: EffortSpelling,
    /// Finished turns whose ids a prompt extending them reuses, when kept.
    replay: Option<Arc<TurnReplay>>,
    /// The audio tower whose frontend and token rule size a clip's pad run.
    audio: Option<Arc<dyn AudioTower>>,
    audio_limits: DecodeLimits,
    audio_ffmpeg: Option<FfmpegPolicy>,
    /// Where a picture seen before is found without decoding it again.
    media_cache: Option<Arc<MediaCache>>,
}

impl PromptRenderer {
    /// Compiles `template` and binds it to `vocabulary`.
    ///
    /// # Errors
    ///
    /// [`AppError::Internal`] when the template does not compile. A template
    /// that does not parse is a defect in the checkpoint or in the path the
    /// operator pointed at, not something a request can be blamed for, so it is
    /// reported as a server failure at the one place that can still name the
    /// file - the boot path that calls this.
    pub fn new(
        template: &str,
        style: ToolJsonStyle,
        vocabulary: Arc<Vocabulary>,
    ) -> Result<Self, AppError> {
        let template = ChatTemplate::new(template, style)
            .map_err(|error| AppError::internal("compiling the chat template", error))?;
        let protected = literal_text::Protected::new(&vocabulary.special_spellings())
            .map_err(|error| AppError::internal("compiling the special-token spellings", error))?;
        Ok(Self {
            template,
            vocabulary,
            vision: None,
            pads: PadTokens::default(),
            protected,
            wrap: None,
            audio_wrap: None,
            placement: None,
            limits: ImageLimits::default(),
            allow_remote: false,
            ffmpeg: None,
            video_fps: shiso_vision::DEFAULT_FPS,
            thinking_default: true,
            effort: EffortSpelling::STANDARD,
            replay: None,
            audio: None,
            audio_limits: DEFAULT_AUDIO_LIMITS,
            audio_ffmpeg: None,
            media_cache: None,
        })
    }

    /// The same renderer, finding pictures it has seen before in `cache`
    /// rather than decoding them. The engine thread fills the same cache.
    #[must_use]
    pub fn with_media_cache(mut self, cache: Arc<MediaCache>) -> Self {
        self.media_cache = Some(cache);
        self
    }

    /// The same renderer with the checkpoint's own vision geometry attached.
    ///
    /// This is the door a server with no `--vision` comes through: the
    /// checkpoint declares a geometry, so a prompt can still be rendered with
    /// the right number of pads even though nothing will encode them. The
    /// config is narrowed to a [`PreprocessSpec`] here rather than per request,
    /// because the geometry is a property of the checkpoint and a
    /// `vision_config` that cannot drive a preprocessor is a boot-time fact
    /// that belongs in the boot-time log. A config that fails that narrowing
    /// leaves the renderer without vision: refuse media rather than guess a
    /// geometry.
    #[must_use]
    pub fn with_vision(
        mut self,
        vision: &VisionConfig,
        placement: MediaPlacement,
        limits: ImageLimits,
    ) -> Self {
        match PreprocessSpec::from_vision_config(vision) {
            Ok(spec) => self.with_tower(&spec, placement, limits),
            Err(error) => {
                tracing::error!(
                    %error,
                    "the checkpoint's vision geometry cannot drive a preprocessor; \
                     this server will refuse media rather than guess it"
                );
                // The limits are the operator's bound and the failure is the
                // geometry's, so the degraded renderer carries them exactly as
                // the tower-carrying one does; `with_tower` sets them on the
                // other arm and this sets the same value here.
                self.limits = limits;
                self
            }
        }
    }

    /// The same renderer with a loaded tower's spec and placement attached.
    ///
    /// The markers are resolved to ids HERE, once, rather than per render: a
    /// marker this checkpoint's vocabulary does not declare is a mismatch
    /// between the family's template and the checkpoint, and the only place it
    /// can still be reported to an operator is the boot log. Per render it
    /// would be an empty expansion nobody sees.
    #[must_use]
    pub fn with_tower(
        mut self,
        spec: &PreprocessSpec,
        placement: MediaPlacement,
        limits: ImageLimits,
    ) -> Self {
        self.pads = self.vocabulary.media_ids(&placement);
        if !self.pads.any() {
            tracing::warn!(
                marker = placement.image_marker,
                "this checkpoint's vocabulary declares none of the family's media markers; \
                 a prompt carrying a picture will have nowhere to put its rows"
            );
        }
        self.wrap = self.vocabulary.wrap_ids(&placement);
        self.audio_wrap = self.vocabulary.audio_wrap_ids(&placement);
        self.placement = Some(placement);
        if self.wrap.is_none() && placement.wrap_image.is_some() {
            tracing::warn!(
                "this family brackets an image's pad run but this vocabulary declares no \
                 bracket; the run will be rendered bare"
            );
        }
        self.vision = Some(spec.clone());
        self.limits = limits;
        self
    }

    /// The same renderer with an audio tower attached.
    ///
    /// The placement is resolved again here, as [`Self::with_tower`] does,
    /// because the audio marker and its bracket are ids in the same
    /// vocabulary and a clip rendered with no pad id resolved is a pad run the
    /// expansion cannot find.
    #[must_use]
    pub fn with_audio_tower(
        mut self,
        tower: Arc<dyn AudioTower>,
        placement: MediaPlacement,
        limits: DecodeLimits,
        ffmpeg: Option<FfmpegPolicy>,
    ) -> Self {
        self.pads = self.vocabulary.media_ids(&placement);
        self.audio_wrap = self.vocabulary.audio_wrap_ids(&placement);
        if self.pads.audio().is_none() {
            tracing::warn!(
                marker = placement.audio_marker,
                "this checkpoint's vocabulary declares no audio marker; a prompt carrying \
                 a clip will have nowhere to put its rows"
            );
        }
        self.audio = Some(tower);
        self.audio_limits = limits;
        self.audio_ffmpeg = ffmpeg;
        self
    }

    /// The same renderer with the remote-media policy set.
    #[must_use]
    pub const fn with_remote_media(mut self, allow: bool) -> Self {
        self.allow_remote = allow;
        self
    }

    /// The same renderer with video decoding switched on.
    #[must_use]
    pub fn with_video(mut self, ffmpeg: Option<FfmpegPolicy>, fps: f64) -> Self {
        self.ffmpeg = ffmpeg;
        self.video_fps = fps;
        self
    }

    /// The same renderer with a different answer for a caller that said nothing
    /// about thinking.
    ///
    /// The default is on, because these templates open a thinking block unless
    /// told otherwise and a server that silently turned that off would answer
    /// every effort-silent request with a different model than the one the
    /// checkpoint describes.
    #[must_use]
    pub const fn with_thinking_default(mut self, on: bool) -> Self {
        self.thinking_default = on;
        self
    }

    /// The same renderer, spelling effort rungs the way this family's template
    /// accepts them.
    #[must_use]
    pub const fn with_effort_spelling(mut self, effort: EffortSpelling) -> Self {
        self.effort = effort;
        self
    }

    /// The same renderer, reusing the ids of finished turns in `replay` for the
    /// span of a prompt that extends one. `None` encodes every prompt cold.
    #[must_use]
    pub fn with_turn_replay(mut self, replay: Option<Arc<TurnReplay>>) -> Self {
        self.replay = replay;
        self
    }

    /// The vocabulary this renderer encodes with.
    #[must_use]
    pub fn vocabulary(&self) -> &Arc<Vocabulary> {
        &self.vocabulary
    }

    /// Renders one request into a prompt.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for media this server will not decode - a
    /// remote address under the default policy, a picture sent to a text-only
    /// checkpoint, a payload that is not base64 - and for a conversation the
    /// template itself refuses. [`AppError::Internal`] when the vocabulary
    /// cannot encode the text the template produced, which is a checkpoint
    /// mismatch rather than a request defect.
    pub fn render(&self, request: &Request) -> Result<Rendered, AppError> {
        let policy = self.media_policy();
        let mut prepared = preprocess(&request.system, &request.messages, &policy)
            .map_err(|error| AppError::bad_request(error.to_string(), None, "invalid_media"))?;
        // Special-token text a caller wrote is escaped here and read as plain
        // text when encoded, so the only control tokens in the ids are the
        // ones the template wrote: its turn markers, and a media marker per
        // attached item.
        let mut escaped = literal_text::escape(&mut prepared.messages, &self.protected);

        let mut tools = Self::tool_documents(request);
        if let Some(tools) = tools.as_mut() {
            literal_text::escape_more(tools, &self.protected, &mut escaped);
        }
        let text = self
            .template
            .render(&prepared.messages, tools.as_deref(), self.flags(request))
            .map_err(|error| {
                // A template raises for conversations it refuses to render - an
                // assistant turn with no content, a tool result with no call.
                // That is the request being wrong, and the raise message is
                // written for the person who sent it.
                AppError::bad_request(error.to_string(), Some("messages"), "invalid_request")
            })?;
        let (text, literal) = literal_text::unescape(text, &escaped)
            .map_err(|error| AppError::internal("restoring escaped message text", error))?;

        // `encode_prompt` rather than `encode`: a template that writes the
        // opening token itself has already put it in the text and is left
        // alone, and one that does not gets it here. Either way the prompt
        // starts with exactly one, which is what the model was trained on and
        // what upstream llama.cpp produces for the same conversation.
        let tokens = self
            .encode_with_replay(&text, &literal, prepared.media.is_empty())
            .map_err(|error| AppError::internal("encoding the rendered prompt", error))?;
        let wraps = MediaWraps {
            image: self.wrap,
            audio: self.audio_wrap,
        };
        let (tokens, token_ranges) =
            expand_media_pads(tokens, &prepared.pad_counts, self.pads, wraps).map_err(|error| {
                // The template writes one marker per item and marker text in a
                // message is read as text, so a count that disagrees is media
                // in a turn the template does not render.
                AppError::bad_request(
                    format!(
                        "the prompt holds {} media marker(s) for {} attached item(s); media is \
                         accepted only where this model's template renders it",
                        error.markers, error.items
                    ),
                    Some("messages"),
                    "invalid_media",
                )
            })?;
        let media_spans = self.locate_media_spans(&text, &literal, &token_ranges)?;

        Ok(Rendered {
            tokens,
            text,
            media: prepared.media,
            digests: prepared.digests,
            media_spans,
        })
    }

    /// Decodes the media in `blocks` for a prompt the template does not
    /// write, under the same policy and cache [`PromptRenderer::render`]
    /// uses: a decision head's layout places the runs itself, and needs only
    /// how long each one is.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for media this server will not decode, the
    /// same refusals `render` makes: a picture sent to a server with no tower
    /// among them.
    pub fn decode_media(&self, blocks: Vec<Block>) -> Result<DecodedMedia, AppError> {
        let policy = self.media_policy();
        let message = Message::new(Role::User, blocks);
        let prepared = preprocess(&[], std::slice::from_ref(&message), &policy)
            .map_err(|error| AppError::bad_request(error.to_string(), None, "invalid_media"))?;
        Ok(DecodedMedia {
            media: prepared.media,
            digests: prepared.digests,
            pad_counts: prepared.pad_counts,
        })
    }

    /// What the media layer may do, as the operator configured it.
    fn media_policy(&self) -> MediaPolicy<'_> {
        MediaPolicy {
            vision: self.vision.as_ref(),
            limits: self.limits,
            allow_remote: self.allow_remote,
            ffmpeg: self.ffmpeg.as_ref(),
            video_fps: self.video_fps,
            audio: self.audio.as_deref(),
            audio_limits: self.audio_limits,
            audio_ffmpeg: self.audio_ffmpeg.as_ref(),
            cache: self.media_cache.as_deref(),
        }
    }

    /// The byte range of each template-written marker in the final `text`,
    /// zipped with the expanded token ranges `expand_media_pads` reported
    /// into one [`MediaSpan`] per item.
    ///
    /// Marker text a caller wrote was escaped before the template ran and is
    /// restored by `unescape`, so the final text can spell a marker as
    /// ordinary text: an occurrence inside a `literal` range is text, not a
    /// placeholder, and is skipped. At one position the longest spelling
    /// wins - a family whose bracket spells a prefix of its marker must not
    /// locate half a marker. A count that disagrees with the token side is
    /// the same refusal as a [`super::message_preprocess::PadMismatch`]: a
    /// prompt whose text and ids disagree is refused, not guessed at.
    fn locate_media_spans(
        &self,
        text: &str,
        literal: &[std::ops::Range<usize>],
        token_ranges: &[std::ops::Range<usize>],
    ) -> Result<Vec<MediaSpan>, AppError> {
        if token_ranges.is_empty() {
            return Ok(Vec::new());
        }
        let placement = self
            .placement
            .expect("a prompt with media only renders through a tower-attached renderer");
        let spellings = [
            Some(placement.image_marker),
            placement.video_marker,
            placement.audio_marker,
        ];
        let bytes = text.as_bytes();
        let mut text_ranges = Vec::with_capacity(token_ranges.len());
        let mut at = 0usize;
        while at < bytes.len() {
            let hit = spellings
                .iter()
                .flatten()
                .filter(|marker| bytes[at..].starts_with(marker.as_bytes()))
                .max_by_key(|marker| marker.len());
            match hit {
                Some(marker) => {
                    let span = at..at + marker.len();
                    if !literal
                        .iter()
                        .any(|lit| span.start >= lit.start && span.end <= lit.end)
                    {
                        text_ranges.push(span.clone());
                    }
                    at = span.end;
                }
                None => at += 1,
            }
        }
        if text_ranges.len() != token_ranges.len() {
            return Err(AppError::bad_request(
                format!(
                    "the prompt holds {} media marker(s) for {} attached item(s); media is \
                     accepted only where this model's template renders it",
                    text_ranges.len(),
                    token_ranges.len()
                ),
                Some("messages"),
                "invalid_media",
            ));
        }
        Ok(text_ranges
            .into_iter()
            .zip(token_ranges)
            .map(|(text, tokens)| MediaSpan {
                tokens: tokens.clone(),
                text,
            })
            .collect())
    }

    /// Encodes `text`, taking the ids of the longest finished turn it extends
    /// for that span and encoding only the rest.
    ///
    /// The rest is encoded without an opening token: the replayed ids came out
    /// of this function once already and carry it. A prompt with media is
    /// always encoded cold, because its pad markers are one id for any picture,
    /// and so is one with `literal` marker text, whose ids differ from the ids
    /// the same text would get from the template.
    fn encode_with_replay(
        &self,
        text: &str,
        literal: &[std::ops::Range<usize>],
        text_only: bool,
    ) -> Result<Vec<TokenId>, super::TokenizerError> {
        if !literal.is_empty() {
            return self.vocabulary.encode_prompt_with_literals(text, literal);
        }
        let replayed = self
            .replay
            .as_ref()
            .filter(|_| text_only)
            .and_then(|replay| replay.longest_prefix(text));
        let Some(replayed) = replayed else {
            return self.vocabulary.encode_prompt(text);
        };
        let rest = text.get(replayed.text_len..).unwrap_or_default();
        let mut tokens = Vec::with_capacity(replayed.tokens.len() + rest.len() / 3);
        tokens.extend_from_slice(&replayed.tokens);
        tokens.extend(self.vocabulary.encode(rest, false)?);
        tracing::debug!(
            replayed = replayed.tokens.len(),
            total = tokens.len(),
            "a prompt extends a finished turn; its ids are reused"
        );
        Ok(tokens)
    }

    /// The tool list in the shape a chat template writes into the prompt.
    ///
    /// The nested `function` wrapper is not decoration: these templates dump a
    /// tool with `tojson` and read a call back as `tool_call.function.name`, so
    /// the wrapper is the shape the model was trained to see. The IR is flat
    /// because a wrapper is a fact about one wire format; this is the one place
    /// the wrapper is put back on.
    fn tool_documents(request: &Request) -> Option<Vec<serde_json::Value>> {
        if request.tools.is_empty() {
            return None;
        }
        Some(
            request
                .tools
                .iter()
                .map(|tool| {
                    let mut function = serde_json::Map::new();
                    function.insert(
                        "name".to_owned(),
                        serde_json::Value::String(tool.name.clone()),
                    );
                    if let Some(description) = tool.description.as_ref() {
                        function.insert(
                            "description".to_owned(),
                            serde_json::Value::String(description.clone()),
                        );
                    }
                    function.insert("parameters".to_owned(), tool.input_schema.clone());
                    serde_json::json!({"type": "function", "function": function})
                })
                .collect(),
        )
    }

    /// What the template is told about thinking.
    ///
    /// A named rung is spelled the way this family's template accepts it; no
    /// rung leaves the template's own default in charge.
    ///
    /// A budget does not appear here and is not enforced anywhere: it is
    /// accepted and ignored, by decision. Clients steer with effort, and a hard
    /// token cap cuts reasoning off mid-thought, which is worse for the answer
    /// than the thinking it saves.
    fn flags(&self, request: &Request) -> RenderFlags<'static> {
        let (enable_thinking, effort) = match request.thinking {
            ThinkingDirective::Unspecified => (self.thinking_default, None),
            ThinkingDirective::Off => (false, None),
            ThinkingDirective::Effort(level) => (true, Some(self.effort.spell(level))),
            // An explicit budget, and anything added after this was written, is
            // thinking on with no rung named: a budget is not enforced, and a
            // directive this build does not recognise is still the caller saying
            // something rather than nothing.
            _ => (true, None),
        };
        RenderFlags {
            enable_thinking,
            reasoning_effort: effort,
            preserve_thinking: request.preserve_thinking,
            continue_final: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;
    use std::str::FromStr;
    use std::sync::Arc;

    use serde_json::{Value, json};
    use shiso_core::{HfConfig, VisionConfig};
    use shiso_vision::{ImageLimits, MediaPlacement};

    use super::PromptRenderer;
    use crate::tokenizer::Vocabulary;
    use crate::tokenizer::jinja_helpers::ToolJsonStyle;

    /// A template with no media handling at all: the degraded path never
    /// renders, so the stub only has to compile.
    const TEMPLATE: &str = "{% for m in messages %}{{ m.content }}{% endfor %}";

    /// The smallest vocabulary `PromptRenderer::new` accepts. The degraded
    /// arm resolves no markers, so a vocabulary that declares nothing beyond
    /// a word is enough.
    fn stub_vocabulary() -> Arc<Vocabulary> {
        let document = json!({
            "version": "1.0",
            "truncation": Value::Null,
            "padding": Value::Null,
            "added_tokens": [],
            "normalizer": Value::Null,
            "pre_tokenizer": {"type": "Whitespace"},
            "post_processor": Value::Null,
            "decoder": Value::Null,
            "model": {
                "type": "WordLevel",
                "vocab": {"<unk>": 0, "word": 1},
                "unk_token": "<unk>",
            },
        })
        .to_string();
        let inner = tokenizers::Tokenizer::from_str(&document).expect("the stub vocabulary loads");
        Arc::new(Vocabulary::from_tokenizer(inner))
    }

    /// A vision block whose patch edge is nonzero - so the config layer
    /// accepts it - and past the geometry bound the preprocessor refuses,
    /// which is the only thing that sends `with_vision` down its degraded
    /// arm: a `vision_config` that parses and still cannot drive a tower.
    fn impossible_vision() -> VisionConfig {
        let document = json!({
            "model_type": "qwen3_5",
            "image_token_id": 10,
            "video_token_id": 11,
            "vision_start_token_id": 12,
            "vision_end_token_id": 13,
            "vision_config": {
                "depth": 1,
                "hidden_size": 8,
                "num_heads": 1,
                "intermediate_size": 8,
                "out_hidden_size": 8,
                "patch_size": 300,
                "temporal_patch_size": 1,
                "spatial_merge_size": 2,
                "num_position_embeddings": 4,
            },
        });
        HfConfig::from_json_str(&document.to_string())
            .expect("the head parses")
            .vision_config()
            .expect("the vision block parses")
            .expect("the document carries a vision block")
    }

    /// What the degradation guarantees: the geometry is refused and vision
    /// stays off, but the operator's limits - a fact about the deployment,
    /// not about the failed geometry - are applied exactly as they would be
    /// on the arm that attached a tower.
    /// The sparse family's template accepts only low, medium and xhigh, so the
    /// top two rungs both spell xhigh; the standard spelling keeps high.
    #[test]
    fn a_family_spelling_maps_every_rung_to_a_word_its_template_accepts() {
        use super::EffortSpelling;
        use shiso_ir::Effort;
        assert_eq!(EffortSpelling::XHIGH_TOP.spell(Effort::High), "xhigh");
        assert_eq!(EffortSpelling::XHIGH_TOP.spell(Effort::Minimal), "low");
        assert_eq!(EffortSpelling::STANDARD.spell(Effort::High), "high");
    }

    /// A prompt extending a finished turn carries that turn's ids even where
    /// encoding its text would split differently, and encodes only the rest;
    /// a prompt with media never does.
    #[test]
    fn a_prompt_extending_a_finished_turn_reuses_its_ids() {
        use crate::tokenizer::turn_replay::TurnReplay;
        use shiso_core::TokenId;
        let replay = Arc::new(TurnReplay::new(64));
        // Not what encoding "word word" gives, which is [1, 1]: the model's own
        // split, which is the one the cache holds.
        replay.record("word word".to_owned(), [7, 8].map(TokenId::new).into());
        let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, stub_vocabulary())
            .expect("the stub template compiles")
            .with_turn_replay(Some(replay));
        let raw = |tokens: Vec<TokenId>| tokens.iter().map(|id| id.get()).collect::<Vec<_>>();
        let spliced = renderer
            .encode_with_replay("word word word", &[], true)
            .expect("encodes");
        assert_eq!(raw(spliced), [7, 8, 1]);
        let with_media = renderer
            .encode_with_replay("word word word", &[], false)
            .expect("encodes");
        assert_eq!(raw(with_media), [1, 1, 1]);
    }

    #[test]
    fn a_degraded_vision_attachment_keeps_the_limits_and_stays_without_a_tower() {
        let limits = ImageLimits::default()
            .with_max_pixels(NonZeroU64::new(48 * 48).expect("a nonzero bound"))
            .with_max_decoded_bytes(NonZeroU64::new(1024 * 1024).expect("a nonzero budget"));
        let renderer = PromptRenderer::new(TEMPLATE, ToolJsonStyle::Compact, stub_vocabulary())
            .expect("the stub template compiles")
            .with_vision(&impossible_vision(), MediaPlacement::GEMMA4, limits);
        assert!(
            renderer.vision.is_none(),
            "a geometry that cannot drive a preprocessor must leave the renderer without a tower"
        );
        assert_eq!(
            renderer.limits, limits,
            "the degraded arm applies the same limits the tower-carrying arm gets"
        );
    }
}
