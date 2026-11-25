//! Serialization formats for `/images` and `/v1/images/generations`.
//!
//! The shape is the one the two published image surfaces speak: `OpenRouter`'s
//! `/images` and `OpenAI`'s `/v1/images/generations` agree on the field names
//! this server answers to, so one wire type serves both spellings. Like every
//! other wire module here, this one owns its serde types and hands them to
//! [`crate::openai::images`], which converts toward the domain job; nothing
//! between the parse and the encode knows these names exist.
//!
//! Unknown fields are accepted, as on every other surface here: image clients
//! carry a long tail of provider-specific extras, and refusing one would make
//! this server answer a working request with a parse error.

use serde::{Deserialize, Serialize};
use shiso_ir::image::{Background, Generate, Quality};

use crate::error::AppError;

/// Request payload matching `OpenRouter`'s `/images` and `OpenAI`'s
/// `/v1/images/generations`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageGenerationRequest {
    /// Text description of the desired image.
    pub prompt: String,
    /// Requested model name. Accepted and answered by whichever generator is
    /// loaded: a client that names the model it believes is behind this server
    /// is not refused for being right about nothing.
    #[serde(default)]
    pub model: Option<String>,
    /// Number of images to generate (1-10).
    #[serde(default)]
    pub n: Option<usize>,
    /// Output dimensions, e.g. "1024x1024", "512x512", a named aspect ratio,
    /// or a named resolution ("2K"). `"auto"` asks for the server default.
    #[serde(default)]
    pub size: Option<String>,
    /// Aspect ratio, e.g. "1:1", "16:9", "9:16".
    #[serde(default)]
    pub aspect_ratio: Option<String>,
    /// Rendering quality: `low`, `medium`, `high`, `xhigh`, `max`, `auto`,
    /// or the older `standard` and `hd`. Validated and echoed in the answer;
    /// it changes nothing, because the models served here run one schedule.
    #[serde(default)]
    pub quality: Option<String>,
    /// The pictures' background: `transparent`, `opaque` or `auto`.
    /// `transparent` needs png output and a background remover.
    #[serde(default)]
    pub background: Option<String>,
    /// Content-filter strictness, `low` or `auto`. Validated and otherwise
    /// unused: this server runs no content filter to tune.
    #[serde(default)]
    pub moderation: Option<String>,
    /// Denoising steps per image (1-100). Not part of either published shape;
    /// omitted, the image gets the operator's `--image-default-steps`.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Deterministic seed. Image `i` of a multi-image request gets `seed + i`.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Output encoding format (`png`, `jpeg`).
    #[serde(default)]
    pub output_format: Option<String>,
    /// `OpenAI` format name (`b64_json`); `url` is refused, as this server
    /// hosts nothing to point a URL at.
    #[serde(default)]
    pub response_format: Option<String>,
    /// Compression level (1-100). Zero is refused: it names no encoding this
    /// server can produce.
    #[serde(default)]
    pub output_compression: Option<u32>,
    /// Whether to stream partial images via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
    /// User identifier. Accepted and unused: this server keeps no per-user
    /// state to group.
    #[serde(default)]
    pub user: Option<String>,
    /// Session grouping ID. Accepted and unused, for the same reason.
    #[serde(default)]
    pub session_id: Option<String>,
}

impl ImageGenerationRequest {
    /// The refusal this format can make on the body alone: a response format
    /// this server cannot honor. The ranges of `n`, `steps` and
    /// `output_compression` are the IR's own floor
    /// ([`shiso_runtime::image::validate`]), which the core applies first
    /// thing.
    ///
    /// The size grammar and the output format are checked by the core, which
    /// is where the generator's bounds are known; a blank prompt is refused by
    /// the job's own validation, which is the one prompt rule this server
    /// enforces.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] naming the field at fault.
    pub fn validate(&self) -> Result<(), AppError> {
        // Named here rather than left to the generator's parser: webp is a
        // format the published shape lists, so its refusal says what to use.
        if self
            .output_format
            .as_deref()
            .is_some_and(|format| format.eq_ignore_ascii_case("webp"))
        {
            return Err(AppError::bad_request(
                "webp output is not supported; use png or jpeg",
                Some("output_format"),
                "unsupported_parameter",
            ));
        }
        if let Some(moderation) = self.moderation.as_deref()
            && !matches!(moderation, "low" | "auto")
        {
            return Err(AppError::bad_request(
                format!("moderation must be 'low' or 'auto', got '{moderation}'"),
                Some("moderation"),
                "invalid_request",
            ));
        }
        self.background()?;
        self.quality()?;
        if let Some(format) = self.response_format.as_deref()
            && !format.eq_ignore_ascii_case("b64_json")
        {
            return Err(AppError::bad_request(
                format!(
                    "response_format '{format}' is not supported; only 'b64_json' is supported"
                ),
                Some("response_format"),
                "unsupported_parameter",
            ));
        }
        Ok(())
    }

    /// The background the body names; absent is `auto`.
    fn background(&self) -> Result<Background, AppError> {
        match self.background.as_deref() {
            None | Some("auto") => Ok(Background::Auto),
            Some("opaque") => Ok(Background::Opaque),
            Some("transparent") => Ok(Background::Transparent),
            Some(other) => Err(AppError::bad_request(
                format!("background must be 'transparent', 'opaque' or 'auto', got '{other}'"),
                Some("background"),
                "invalid_request",
            )),
        }
    }

    /// The quality the body names.
    // `Ok(None)` is a body that names no quality, which is not a refusal:
    // the answer then echoes none.
    fn quality(&self) -> Result<Option<Quality>, AppError> {
        self.quality
            .as_deref()
            .map(|quality| match quality {
                "low" => Ok(Quality::Low),
                "medium" => Ok(Quality::Medium),
                "high" => Ok(Quality::High),
                "xhigh" => Ok(Quality::Xhigh),
                "max" => Ok(Quality::Max),
                "auto" => Ok(Quality::Auto),
                "standard" => Ok(Quality::Standard),
                "hd" => Ok(Quality::Hd),
                other => Err(AppError::bad_request(
                    format!(
                        "quality must be one of low, medium, high, xhigh, max, auto, standard, \
                         hd; got '{other}'"
                    ),
                    Some("quality"),
                    "invalid_request",
                )),
            })
            .transpose()
    }

    /// The IR request: the fields the core reads. `stream` picks which core
    /// answers it; `model`, `moderation`, `user` and `session_id` are
    /// accepted and unused, for the reasons spelled out on the fields.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a `background` or `quality` outside its
    /// values; [`Self::validate`] refuses the same, earlier.
    pub fn to_ir(&self) -> Result<Generate, AppError> {
        Ok(Generate {
            prompt: self.prompt.clone(),
            n: self.n,
            size: self.size.clone(),
            aspect_ratio: self.aspect_ratio.clone(),
            steps: self.steps,
            seed: self.seed,
            output_format: self.output_format.clone(),
            output_compression: self.output_compression,
            background: self.background()?,
            quality: self.quality()?,
        })
    }
}

/// The wire spelling of a quality, for the echo.
#[must_use]
pub const fn quality_name(quality: Quality) -> &'static str {
    match quality {
        Quality::Low => "low",
        Quality::Medium => "medium",
        Quality::High => "high",
        Quality::Xhigh => "xhigh",
        Quality::Max => "max",
        Quality::Auto => "auto",
        Quality::Standard => "standard",
        Quality::Hd => "hd",
    }
}

/// Base64-encoded image entry in response data.
#[derive(Debug, Clone, Serialize)]
pub struct ImageData {
    /// Base64 encoded image string.
    pub b64_json: String,
    /// Media MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
}

/// Input tokens by class.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct InputTokensDetails {
    /// Text tokens of the prompt.
    pub text_tokens: u32,
    /// Tokens of the reference pictures.
    pub image_tokens: u32,
}

/// Output tokens by class.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct OutputTokensDetails {
    /// Tokens of picture produced.
    pub image_tokens: u32,
    /// Text produced: always zero here.
    pub text_tokens: u32,
}

/// Token accounting, in the image models' shape: input and output, each
/// split by class, because the classes are priced apart.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ImageGenerationUsage {
    /// Input tokens of both classes.
    pub input_tokens: u32,
    /// Input tokens by class.
    pub input_tokens_details: InputTokensDetails,
    /// Output tokens.
    pub output_tokens: u32,
    /// Output tokens by class.
    pub output_tokens_details: OutputTokensDetails,
    /// Every token.
    pub total_tokens: u32,
}

/// Response payload for completed generation.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationResponse {
    /// Unix creation timestamp.
    pub created: u64,
    /// The background produced: `transparent` when any picture is.
    pub background: &'static str,
    /// Generated images list.
    pub data: Vec<ImageData>,
    /// The encoding of every picture: `png` or `jpeg`.
    pub output_format: String,
    /// The quality the caller named, echoed; absent for `auto` or none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<&'static str>,
    /// The pictures' dimensions, `WIDTHxHEIGHT`.
    pub size: String,
    /// Optional token accounting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImageGenerationUsage>,
}

/// Streaming SSE partial image event.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenPartialImageEvent {
    /// Event discriminant: `image_generation.partial_image`, or
    /// `image_edit.partial_image` on the edit route.
    pub r#type: &'static str,
    /// Partial image index.
    pub partial_image_index: u32,
    /// Base64 partial image data.
    pub b64_json: String,
}

/// Streaming SSE completed event.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenCompletedEvent {
    /// Event discriminant: `image_generation.completed`, or
    /// `image_edit.completed` on the edit route.
    pub r#type: &'static str,
    /// Unix creation timestamp.
    pub created: u64,
    /// Base64 final image data.
    pub b64_json: String,
    /// Media MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
    /// This picture's background: `transparent` or `opaque`.
    pub background: &'static str,
    /// Its encoding: `png` or `jpeg`.
    pub output_format: String,
    /// The quality the caller named, echoed; absent for `auto` or none.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<&'static str>,
    /// Its dimensions, `WIDTHxHEIGHT`.
    pub size: String,
    /// Optional token accounting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImageGenerationUsage>,
}

#[cfg(test)]
mod tests {
    use shiso_ir::image::Generate;

    use super::ImageGenerationRequest;

    /// Every field a route reads reaches the IR as sent; the ones it ignores
    /// do not.
    #[test]
    fn a_body_becomes_the_ir_request_it_names() {
        let body = serde_json::json!({
            "prompt": "a cat",
            "model": "anything",
            "n": 2,
            "size": "1024x768",
            "aspect_ratio": "16:9",
            "quality": "hd",
            "steps": 12,
            "seed": 9,
            "output_format": "jpeg",
            "response_format": "b64_json",
            "output_compression": 80,
            "stream": true,
            "user": "u",
            "session_id": "s",
            "an_extra_nobody_knows": 1,
        });
        let parsed: ImageGenerationRequest =
            serde_json::from_value(body).expect("unknown fields are accepted");
        parsed.validate().expect("b64_json is served");
        assert_eq!(
            parsed.to_ir().expect("converts"),
            Generate {
                prompt: "a cat".to_owned(),
                n: Some(2),
                size: Some("1024x768".to_owned()),
                aspect_ratio: Some("16:9".to_owned()),
                steps: Some(12),
                seed: Some(9),
                output_format: Some("jpeg".to_owned()),
                output_compression: Some(80),
                background: shiso_ir::image::Background::Auto,
                quality: Some(shiso_ir::image::Quality::Hd),
            }
        );
        assert_eq!(parsed.stream, Some(true));
    }

    #[test]
    fn a_body_that_names_only_a_prompt_names_nothing_else() {
        let parsed: ImageGenerationRequest =
            serde_json::from_str(r#"{"prompt":"x"}"#).expect("parses");
        let ir = parsed.to_ir().expect("converts");
        assert_eq!(ir.n, None);
        assert_eq!(ir.size, None);
        assert_eq!(ir.seed, None);
    }

    #[test]
    fn a_url_response_format_is_refused_by_the_wire() {
        let parsed: ImageGenerationRequest =
            serde_json::from_str(r#"{"prompt":"x","response_format":"url"}"#).expect("parses");
        let error = parsed.validate().expect_err("refused");
        assert!(error.to_string().contains("only 'b64_json'"), "{error}");
    }
}
