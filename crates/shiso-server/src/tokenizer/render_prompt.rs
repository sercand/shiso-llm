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
use shiso_ir::{Request, ThinkingDirective};
use shiso_vision::{FfmpegPolicy, ImageLimits, MediaPlacement, PreprocessSpec, VisionItem};

use crate::error::AppError;
use crate::tokenizer::Vocabulary;
use crate::tokenizer::chat_render::{ChatTemplate, RenderFlags};
use crate::tokenizer::jinja_helpers::ToolJsonStyle;
use crate::tokenizer::message_preprocess::{MediaPolicy, expand_vision_pads, preprocess};

/// A rendered prompt, in all three of the forms something downstream needs.
#[derive(Debug, Clone)]
pub struct Rendered {
    /// The prompt, tokenized, with every media marker already fanned out to the
    /// run of pads its item occupies.
    pub tokens: Vec<TokenId>,
    /// The prompt text, byte for byte as the template produced it.
    pub text: String,
    /// The decoded media, in the order the prompt refers to it.
    pub media: Vec<VisionItem>,
}

impl Rendered {
    /// How many tokens the prompt occupies.
    #[must_use]
    pub fn token_count(&self) -> u64 {
        u64::try_from(self.tokens.len()).unwrap_or(u64::MAX)
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
    /// The bracket ids, for a family that brackets an image's run.
    wrap: Option<(TokenId, TokenId)>,
    limits: ImageLimits,
    allow_remote: bool,
    ffmpeg: Option<FfmpegPolicy>,
    video_fps: f64,
    thinking_default: bool,
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
        Ok(Self {
            template,
            vocabulary,
            vision: None,
            pads: PadTokens::default(),
            wrap: None,
            limits: ImageLimits::default(),
            allow_remote: false,
            ffmpeg: None,
            video_fps: shiso_vision::DEFAULT_FPS,
            thinking_default: true,
        })
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
        let policy = MediaPolicy {
            vision: self.vision.as_ref(),
            limits: self.limits,
            allow_remote: self.allow_remote,
            ffmpeg: self.ffmpeg.as_ref(),
            video_fps: self.video_fps,
        };
        let prepared = preprocess(&request.system, &request.messages, &policy)
            .map_err(|error| AppError::bad_request(error.to_string(), None, "invalid_media"))?;

        let tools = Self::tool_documents(request);
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

        // `encode_prompt` rather than `encode`: a template that writes the
        // opening token itself has already put it in the text and is left
        // alone, and one that does not gets it here. Either way the prompt
        // starts with exactly one, which is what the model was trained on and
        // what upstream llama.cpp produces for the same conversation.
        let tokens = self
            .vocabulary
            .encode_prompt(&text)
            .map_err(|error| AppError::internal("encoding the rendered prompt", error))?;
        let tokens = expand_vision_pads(tokens, &prepared.pad_counts, self.pads, self.wrap)
            .map_err(|error| AppError::internal("expanding the media pad markers", error))?;

        Ok(Rendered {
            tokens,
            text,
            media: prepared.media,
        })
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
    /// A budget does not appear here: a template has no way to express one, and
    /// the number only means something to the sampler that enforces it. It rides
    /// on the request and is honoured by the scheduler.
    fn flags(&self, request: &Request) -> RenderFlags<'static> {
        let (enable_thinking, effort) = match request.thinking {
            ThinkingDirective::Unspecified => (self.thinking_default, None),
            ThinkingDirective::Off => (false, None),
            ThinkingDirective::Effort(level) => (true, Some(level.as_str())),
            // An explicit budget, and anything added after this was written, is
            // thinking on with no rung named: a budget is a number only the
            // sampler can enforce, and a directive this build does not recognise
            // is still the caller saying something rather than nothing.
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
