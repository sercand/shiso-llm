//! Serialization formats for `POST /v1/3d/generations`.
//!
//! One picture in, one encoded `.ply` out. Like every other wire module here,
//! this one owns its serde types and hands them to
//! [`crate::splat::generate`], which converts toward the IR; nothing between the
//! parse and the encode knows these names exist.
//!
//! Unknown fields are accepted, as on every other surface here: a client
//! carries a long tail of provider-specific extras, and refusing one would
//! make this server answer a working request with a parse error.

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use shiso_ir::splat::{Generate, Mode};

/// A refusal this wire can make on the body alone, naming the field at fault.
///
/// A typed refusal rather than a bare message because the handler has to fill
/// in the `OpenAI` error's `param`: recovering the field by matching on the
/// message's first word would silently mislabel the next refusal added here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WireRefusal {
    /// The request field at fault.
    pub field: &'static str,
    /// What is wrong, in words a caller can act on.
    pub message: String,
}

impl WireRefusal {
    fn new(field: &'static str, message: impl Into<String>) -> Self {
        Self {
            field,
            message: message.into(),
        }
    }
}

/// Request payload for `POST /v1/3d/generations`.
#[derive(Debug, Clone, Deserialize)]
pub struct SplatGenerationRequest {
    /// The input picture: raw base64 or a `data:` URL.
    pub image: String,
    /// What the picture is: `synthetic` (a render or a clean cut-out) or
    /// `real` (a photograph). Omitted, `synthetic` as in the reference. It
    /// only chooses the default crop margin.
    #[serde(default)]
    pub mode: Option<String>,
    /// Deterministic seed. Omitted, one is drawn, so a retry gets a different
    /// object; naming it makes the request reproducible.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Denoising steps (1-100). Omitted, the operator's default.
    #[serde(default)]
    pub steps: Option<u32>,
    /// Classifier-free guidance strength. Omitted, the model default.
    #[serde(default)]
    pub cfg_strength: Option<f32>,
    /// Time-grid shift towards high noise. Omitted, the model default.
    #[serde(default)]
    pub rescale_t: Option<f32>,
    /// How many Gaussians to place: a multiple of 32 in 32768..=1048576.
    /// Omitted, the operator's default.
    #[serde(default)]
    pub num_gaussians: Option<u32>,
    /// Overrides the mode's crop margin when present.
    #[serde(default)]
    pub crop_margin: Option<f32>,
    /// Whether to stream progress via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
}

impl SplatGenerationRequest {
    /// The IR request this body describes.
    ///
    /// The picture's bytes are decoded here because a body that is not base64
    /// is the wire's refusal, not the model's. An empty picture is left to the
    /// runtime so both paths answer with the same message.
    ///
    /// # Errors
    ///
    /// A [`WireRefusal`] naming `image` for bytes that are not base64, or
    /// `mode` for a value that is neither `synthetic` nor `real`.
    pub fn into_ir(self) -> Result<Generate, WireRefusal> {
        let image = decode_image(&self.image)?;
        let mode = match self.mode.as_deref() {
            None => Mode::Synthetic,
            Some(text) if text.eq_ignore_ascii_case("synthetic") => Mode::Synthetic,
            Some(text) if text.eq_ignore_ascii_case("real") => Mode::Real,
            Some(_) => {
                return Err(WireRefusal::new(
                    "mode",
                    "mode must be 'synthetic' or 'real'",
                ));
            }
        };
        Ok(Generate {
            image,
            mode,
            seed: self.seed,
            steps: self.steps,
            cfg_strength: self.cfg_strength,
            rescale_t: self.rescale_t,
            num_gaussians: self.num_gaussians,
            crop_margin: self.crop_margin,
        })
    }
}

/// Decodes a body's `image`, with or without a `data:` prefix.
///
/// A `data:` URL's payload starts after the first comma; anything else is
/// taken as plain base64. The media type in the prefix is not checked: the
/// decoder that runs next reads the bytes and refuses what it cannot decode,
/// so a mislabelled prefix costs nothing and a correct one is not a promise
/// this server has to keep.
fn decode_image(text: &str) -> Result<Vec<u8>, WireRefusal> {
    let payload = match text.strip_prefix("data:") {
        Some(rest) => rest
            .split_once(',')
            .map_or(rest, |(_meta, payload)| payload),
        None => text,
    };
    base64::engine::general_purpose::STANDARD
        .decode(payload.trim())
        .map_err(|_err| WireRefusal::new("image", "image must be base64 or a data: URL"))
}

/// One finished splat in a blocking answer.
#[derive(Debug, Clone, Serialize)]
pub struct SplatData {
    /// The encoded `.ply` file, base64.
    pub b64_ply: String,
}

/// Token accounting for one request.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct SplatGenerationUsage {
    /// Always zero today; kept so the shape matches the image wire.
    pub prompt_tokens: u32,
    /// The denoiser sequence length times the step count.
    pub completion_tokens: u32,
    /// The sum of both counters.
    pub total_tokens: u32,
}

/// A finished blocking request.
#[derive(Debug, Clone, Serialize)]
pub struct SplatGenerationResponse {
    /// Unix time the request finished.
    pub created: u64,
    /// The request's splats, in request order.
    pub data: Vec<SplatData>,
    /// The request's token accounting.
    pub usage: SplatGenerationUsage,
}

/// The first streamed frame: the job was admitted to the queue.
#[derive(Debug, Clone, Serialize)]
pub struct SplatAdmittedEvent {
    /// Always `splat_generation.admitted`.
    pub r#type: &'static str,
}

/// One Euler step finished.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SplatProgressEvent {
    /// Always `splat_generation.progress`.
    pub r#type: &'static str,
    /// Zero-based step index.
    pub step: u32,
    /// Total steps.
    pub total_steps: u32,
}

/// The last streamed frame: the splat finished.
#[derive(Debug, Clone, Serialize)]
pub struct SplatCompletedEvent {
    /// Always `splat_generation.completed`.
    pub r#type: &'static str,
    /// Unix time the request started.
    pub created: u64,
    /// The encoded `.ply` file, base64.
    pub b64_ply: String,
    /// The request's token accounting.
    pub usage: SplatGenerationUsage,
}
