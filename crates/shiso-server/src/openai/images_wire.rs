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

use crate::error::AppError;

/// The most images one request may ask for.
pub const MAX_IMAGES_PER_REQUEST: usize = 10;

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
    /// Output dimensions, e.g. "1024x1024", "512x512", or a named aspect ratio.
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
    /// Compression level (0-100).
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
    /// The refusals this format can make on the body alone: a response format
    /// this server cannot honor, an image count outside `1..=10`, a
    /// compression level outside `0..=100`.
    ///
    /// The size grammar and the output format are checked by the handler,
    /// which is where the server's defaults and the generator's bounds are
    /// known; a blank prompt is refused by the job's own validation, which is
    /// the one prompt rule this server enforces.
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
        if let Some(n) = self.n
            && (n == 0 || n > MAX_IMAGES_PER_REQUEST)
        {
            return Err(AppError::bad_request(
                format!("n must be between 1 and {MAX_IMAGES_PER_REQUEST}"),
                Some("n"),
                "invalid_request",
            ));
        }
        if let Some(compression) = self.output_compression
            && compression > 100
        {
            return Err(AppError::bad_request(
                "output_compression must be between 0 and 100",
                Some("output_compression"),
                "invalid_request",
            ));
        }
        Ok(())
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
