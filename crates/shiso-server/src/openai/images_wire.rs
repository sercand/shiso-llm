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
use shiso_ir::image::Generate;

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
    /// Rendering quality ("auto", "high", etc.). Accepted and ignored: this
    /// server has one rendering path, and a value it cannot honor is not a
    /// reason to refuse a picture the caller can still use.
    #[serde(default)]
    pub quality: Option<String>,
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

    /// The IR request: the fields the core reads. `stream` picks which core
    /// answers it, and `model`, `quality`, `user` and `session_id` are
    /// accepted and unused, for the reasons spelled out on the fields.
    #[must_use]
    pub fn to_ir(&self) -> Generate {
        Generate {
            prompt: self.prompt.clone(),
            n: self.n,
            size: self.size.clone(),
            aspect_ratio: self.aspect_ratio.clone(),
            steps: self.steps,
            seed: self.seed,
            output_format: self.output_format.clone(),
            output_compression: self.output_compression,
        }
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

/// Token and cost usage.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationUsage {
    /// Prompt tokens.
    pub prompt_tokens: u32,
    /// Generated completion tokens.
    pub completion_tokens: u32,
    /// Total tokens.
    pub total_tokens: u32,
}

/// Response payload for completed generation.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenerationResponse {
    /// Unix creation timestamp.
    pub created: u64,
    /// Generated images list.
    pub data: Vec<ImageData>,
    /// Optional token accounting.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<ImageGenerationUsage>,
}

/// Streaming SSE partial image event.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenPartialImageEvent {
    /// Event discriminant: `image_generation.partial_image`.
    pub r#type: &'static str,
    /// Partial image index.
    pub partial_image_index: u32,
    /// Base64 partial image data.
    pub b64_json: String,
}

/// Streaming SSE completed event.
#[derive(Debug, Clone, Serialize)]
pub struct ImageGenCompletedEvent {
    /// Event discriminant: `image_generation.completed`.
    pub r#type: &'static str,
    /// Unix creation timestamp.
    pub created: u64,
    /// Base64 final image data.
    pub b64_json: String,
    /// Media MIME type.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub media_type: Option<String>,
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
            parsed.to_ir(),
            Generate {
                prompt: "a cat".to_owned(),
                n: Some(2),
                size: Some("1024x768".to_owned()),
                aspect_ratio: Some("16:9".to_owned()),
                steps: Some(12),
                seed: Some(9),
                output_format: Some("jpeg".to_owned()),
                output_compression: Some(80),
            }
        );
        assert_eq!(parsed.stream, Some(true));
    }

    #[test]
    fn a_body_that_names_only_a_prompt_names_nothing_else() {
        let parsed: ImageGenerationRequest =
            serde_json::from_str(r#"{"prompt":"x"}"#).expect("parses");
        let ir = parsed.to_ir();
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
