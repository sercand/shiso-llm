//! Serialization formats for `POST /v1/3d/generations`.
//!
//! One picture in, one encoded splat file out: a `.ply`, or a `.glb` when
//! the body's `output_format` or the `Accept` header asks for one. Like every
//! other wire module here,
//! this one owns its serde types and hands them to
//! [`crate::splat::generate`], which converts toward the IR; nothing between the
//! parse and the encode knows these names exist.
//!
//! Unknown fields are accepted, as on every other surface here: a client
//! carries a long tail of provider-specific extras, and refusing one would
//! make this server answer a working request with a parse error.

use base64::Engine as _;
use serde::{Deserialize, Serialize};

use shiso_ir::splat::{Format, Generate, Mode};

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
    /// The file the splat is written as: `ply` (the default) or `glb`, a
    /// binary glTF carrying the Gaussians through `KHR_gaussian_splatting`.
    /// Omitted, the `Accept` header decides, and `ply` when it names neither.
    #[serde(default)]
    pub output_format: Option<String>,
    /// Every `refresh_every`-th step runs in full (1-4), and so do the first
    /// two; the steps between hold the last full step's guidance direction
    /// and conditioning rows.
    /// One is the reference's sampler, more is faster and approximate.
    /// Omitted, the operator's default.
    #[serde(default)]
    pub refresh_every: Option<u32>,
    /// Whether to stream progress via SSE.
    #[serde(default)]
    pub stream: Option<bool>,
}

/// The format an `output_format` value names.
///
/// # Errors
///
/// A [`WireRefusal`] naming `output_format` for any other value.
pub fn parse_format(text: &str) -> Result<Format, WireRefusal> {
    if text.eq_ignore_ascii_case("ply") {
        Ok(Format::Ply)
    } else if text.eq_ignore_ascii_case("glb") {
        Ok(Format::Glb)
    } else {
        Err(WireRefusal::new(
            "output_format",
            "output_format must be 'ply' or 'glb'",
        ))
    }
}

/// A format's MIME type, as the `Accept` header names it and as a raw answer
/// is labelled.
#[must_use]
pub const fn media_type(format: Format) -> &'static str {
    match format {
        Format::Ply => "model/ply",
        Format::Glb => "model/gltf-binary",
    }
}

/// The format an `Accept` header asks for by its MIME type, when it names
/// one: the first media range, in the header's order, that is a splat
/// file's type. `*/*`, `application/json` and anything else name none.
///
/// A header that names one also asks for the file itself as the answer's
/// body, not for JSON around its base64.
#[must_use]
pub fn accepted_format(accept: &str) -> Option<Format> {
    accept
        .split(',')
        .filter_map(|range| range.split(';').next())
        .map(str::trim)
        .find_map(|range| {
            [Format::Ply, Format::Glb]
                .into_iter()
                .find(|format| range.eq_ignore_ascii_case(media_type(*format)))
        })
}

impl SplatGenerationRequest {
    /// The IR request this body describes.
    ///
    /// The picture's bytes are decoded here because a body that is not base64
    /// is the wire's refusal, not the model's. An empty picture is left to the
    /// runtime so both paths answer with the same message.
    ///
    /// `accepted` is the format the `Accept` header named, if it named one
    /// ([`accepted_format`]). The body's `output_format` and the header may
    /// each be silent, but they may not disagree: answering one of two
    /// contradictory asks would hand a client a file it then misreads.
    ///
    /// # Errors
    ///
    /// A [`WireRefusal`] naming `image` for bytes that are not base64,
    /// `mode` for a value that is neither `synthetic` nor `real`, or
    /// `output_format` for an unknown format or one the header contradicts.
    pub fn into_ir(self, accepted: Option<Format>) -> Result<Generate, WireRefusal> {
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
        let named = self
            .output_format
            .as_deref()
            .map(parse_format)
            .transpose()?;
        let format = match (named, accepted) {
            (Some(body), Some(header)) if body != header => {
                return Err(WireRefusal::new(
                    "output_format",
                    format!(
                        "output_format asks for {} and the Accept header for {}",
                        media_type(body),
                        media_type(header)
                    ),
                ));
            }
            (Some(format), _) | (None, Some(format)) => format,
            (None, None) => Format::Ply,
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
            format,
            refresh_every: self.refresh_every,
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

/// One finished splat in a blocking answer: exactly one of the two fields,
/// named after the file it carries.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SplatData {
    /// The encoded `.ply` file, base64.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_ply: Option<String>,
    /// The encoded `.glb` file, base64.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b64_glb: Option<String>,
}

impl SplatData {
    /// `bytes`, a file of `media_type`, under the field that names its
    /// format. Any type but the `.glb`'s is the `.ply`.
    #[must_use]
    pub fn encode(bytes: &[u8], media_type: &str) -> Self {
        let text = base64::engine::general_purpose::STANDARD.encode(bytes);
        if media_type == self::media_type(Format::Glb) {
            Self {
                b64_glb: Some(text),
                ..Self::default()
            }
        } else {
            Self {
                b64_ply: Some(text),
                ..Self::default()
            }
        }
    }
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
    /// The encoded file, base64, as `b64_ply` or `b64_glb`.
    #[serde(flatten)]
    pub file: SplatData,
    /// The request's token accounting.
    pub usage: SplatGenerationUsage,
}
