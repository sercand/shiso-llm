//! The FLUX.2-klein pipeline: a prompt in, encoded image bytes out.
//!
//! One [`Flux2KleinModel`] holds the three components - the Qwen3-style text
//! encoder, the diffusion transformer, the VAE decoder - loaded onto one
//! backend, plus the tokenizer and chat template that turn text into ids.
//! [`ImageGenerator::generate`] runs the rectified-flow loop: encode the
//! prompt, draw Gaussian noise from the job's seed, take Euler steps along
//! the transformer's velocity field, and decode the final latent to RGB.
//!
//! # Control tokens and caller text
//!
//! The chat template writes turn markers (`<|im_start|>`) as text and the
//! vocabulary turns that text into control ids. A prompt that *spells* a
//! marker must not gain a turn boundary, so the template is rendered with a
//! private-use sentinel in the content slot and the rendered string is
//! encoded in three pieces: the template's own segments encode normally,
//! the caller's text encodes with added tokens read as plain text. The
//! template's markers become tokens; the caller's copies stay words.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use image::ImageEncoder as _;
use shiso_core::Backend;
use shiso_vision::generator::{
    GeneratedImage, ImageGenError, ImageGenerationJob, ImageGenerator, ImageOutputFormat,
    ImageStepEvent,
};

use super::config::{Flux2Located, Flux2Params};
use super::kv_cache::Flux2KvCache;
use super::scheduler::new_scheduler;
use super::text_encoder::TextEncoder;
use super::transformer::{DiT, DiTInput, KvMode};
use super::vae::Vae;
use crate::diffusion::ops::Ops;
use crate::diffusion::weights::StStore;

/// The text encoder reads at most this many prompt tokens.
const MAX_TEXT_TOKENS: usize = 512;

/// Private-use character written into the template's content slot instead of
/// the prompt, so the rendered string can be split back around it. No chat
/// template emits a private-use codepoint of its own.
const CONTENT_MARK: char = '\u{F000}';

/// Answers the Python-shaped string methods a checkpoint's chat template
/// calls (`startswith`, `split`, `strip('\n')`, ...) that the Rust engine
/// exposes only as filters or not at all. Mirrors the server's tokenizer
/// bridge; it is duplicated rather than shared because `shiso-models` must
/// not depend on `shiso-server`, and the templates — not our code — decide
/// which methods exist. An argument is treated as a Python character set,
/// which is what the Qwen3-family templates pass (`'\n'`).
fn bridge_python_methods(
    _state: &minijinja::State<'_, '_>,
    value: &minijinja::Value,
    method: &str,
    args: &[minijinja::Value],
) -> Result<minijinja::Value, minijinja::Error> {
    use minijinja::value::{ValueKind, from_args};
    if value.kind() != ValueKind::String {
        return Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod));
    }
    let text = value.as_str().unwrap_or_default();
    match method {
        "startswith" => {
            let (prefix,): (String,) = from_args(args)?;
            Ok(minijinja::Value::from(text.starts_with(&prefix)))
        }
        "endswith" => {
            let (suffix,): (String,) = from_args(args)?;
            Ok(minijinja::Value::from(text.ends_with(&suffix)))
        }
        "replace" => {
            let (from, to): (String, String) = from_args(args)?;
            Ok(minijinja::Value::from(text.replace(&from, &to)))
        }
        "strip" | "lstrip" | "rstrip" => {
            let (chars,): (Option<String>,) = from_args(args)?;
            let stripped = match chars.as_deref() {
                Some(set) if !set.is_empty() => match method {
                    "strip" => text.trim_matches(|c| set.contains(c)),
                    "lstrip" => text.trim_start_matches(|c| set.contains(c)),
                    _ => text.trim_end_matches(|c| set.contains(c)),
                },
                // Python's no-argument form strips whitespace. `strip('')`
                // is a no-op there, but no template in the wild passes an
                // empty set, so the empty set falls in with whitespace.
                _ => match method {
                    "strip" => text.trim(),
                    "lstrip" => text.trim_start(),
                    _ => text.trim_end(),
                },
            };
            Ok(minijinja::Value::from(stripped))
        }
        "split" => {
            let (sep,): (Option<String>,) = from_args(args)?;
            let parts: Vec<String> = match sep.as_deref() {
                Some(s) if !s.is_empty() => text.split(s).map(str::to_owned).collect(),
                // No separator (or the empty string): Python splits on
                // runs of whitespace, which is how the thinking-block
                // templates never hit this path anyway.
                _ => text.split_whitespace().map(str::to_owned).collect(),
            };
            Ok(minijinja::Value::from(parts))
        }
        _ => Err(minijinja::Error::from(minijinja::ErrorKind::UnknownMethod)),
    }
}

/// The FLUX.2-klein 4B generator: text encoder, transformer and VAE on one
/// device, driven one job at a time.
#[derive(Debug)]
pub(crate) struct Flux2KleinModel {
    model_id: String,
    params: Flux2Params,
    ops: Ops,
    text: TextEncoder,
    dit: DiT,
    vae: Vae,
    tokenizer: tokenizers::Tokenizer,
    /// The spellings the vocabulary reads as control tokens; caller text
    /// containing one is encoded as plain text so it cannot forge a marker.
    special_spellings: Vec<String>,
    chat_template: String,
}

impl Flux2KleinModel {
    /// Locates, inspects and uploads every component described by `located`
    /// onto `backend`.
    pub(crate) fn load(
        located: &Flux2Located,
        backend: &Arc<dyn Backend>,
    ) -> Result<Self, ImageGenError> {
        let ops = Ops::new(backend)?;
        let text = TextEncoder::load(&located.text_encoder_path, &ops)?;
        if text.context_width() != located.params.context_in_dim {
            return Err(ImageGenError::Checkpoint(format!(
                "the text encoder's taps are {} wide but the transformer reads {}",
                text.context_width(),
                located.params.context_in_dim
            )));
        }

        // Open exactly the files discovery selected - never the parent
        // directory. Indexing the directory would fold a second export
        // beside the selected weights (an FP8 copy of the same tensors)
        // into the store, where each shared name resolves to whichever
        // shard was indexed last: the wrong weights, silently.
        let store = StStore::open_files(&located.weight_files)?;
        let dit = DiT::load(&store, &ops, &located.params)?;
        if dit.out_width() != located.params.in_channels {
            return Err(ImageGenError::Checkpoint(format!(
                "the transformer projects to {} channels but its input takes {}",
                dit.out_width(),
                located.params.in_channels
            )));
        }
        let vae = Vae::load(&located.vae_path, &ops)?;

        let tokenizer_file = if located.tokenizer_path.is_dir() {
            located.tokenizer_path.join("tokenizer.json")
        } else {
            located.tokenizer_path.clone()
        };
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_file).map_err(|e| {
            ImageGenError::Checkpoint(format!(
                "opening tokenizer {}: {e}",
                tokenizer_file.display()
            ))
        })?;
        let special_spellings: Vec<String> = tokenizer
            .get_added_tokens_decoder()
            .into_values()
            .filter(|token| token.special)
            .map(|token| token.content)
            .collect();
        let chat_template = load_chat_template(tokenizer_file.parent())?;

        Ok(Self {
            model_id: if located.params.is_fp8 {
                "flux-2-klein-4b-fp8".to_string()
            } else {
                "flux-2-klein-4b".to_string()
            },
            params: located.params.clone(),
            ops,
            text,
            dit,
            vae,
            tokenizer,
            special_spellings,
            chat_template,
        })
    }

    /// Renders the template with the user turn only - this encoder's
    /// reference conditions on the user turn alone, and a system turn would
    /// shift the tapped hidden states off the distribution the transformer
    /// was trained to read. The content slot is filled by [`CONTENT_MARK`]
    /// and the result is split around the mark, giving the text before and
    /// after the caller's words. The generation prompt stays on: the
    /// assistant turn opener (and, for the Qwen3 template with thinking
    /// disabled, its empty `think` block) belongs to the tail that encodes
    /// after the prompt.
    fn render(&self) -> Result<(String, String), ImageGenError> {
        let mut env = minijinja::Environment::new();
        env.set_trim_blocks(true);
        env.set_lstrip_blocks(true);
        env.set_unknown_method_callback(bridge_python_methods);
        let template = env
            .template_from_str(&self.chat_template)
            .map_err(|e| ImageGenError::Checkpoint(format!("chat template: {e}")))?;
        let rendered = template
            .render(serde_json::json!({
                "messages": [
                    { "role": "user", "content": CONTENT_MARK.to_string() },
                ],
                "add_generation_prompt": true,
                "enable_thinking": false,
            }))
            .map_err(|e| ImageGenError::Checkpoint(format!("chat template: {e}")))?;
        rendered
            .split_once(CONTENT_MARK)
            .map(|(head, tail)| (head.to_string(), tail.to_string()))
            .ok_or_else(|| {
                ImageGenError::Checkpoint(
                    "the chat template did not emit the prompt content".to_string(),
                )
            })
    }

    /// Turns the prompt into token ids: the template's markers encode as
    /// control tokens, the caller's text never does.
    fn tokenize(&self, prompt: &str) -> Result<Vec<u32>, ImageGenError> {
        let (head, tail) = self.render()?;
        let encode = |tokenizer: &tokenizers::Tokenizer, text: &str| {
            tokenizer
                .encode(text, false)
                .map(|encoding| encoding.get_ids().to_vec())
                .map_err(|e| ImageGenError::Checkpoint(format!("tokenizing: {e}")))
        };
        let mut ids = encode(&self.tokenizer, &head)?;
        if self
            .special_spellings
            .iter()
            .any(|spelling| prompt.contains(spelling.as_str()))
        {
            // The caller spelled a control token in their prompt. Cloning
            // the vocabulary to read it as text is paid only by prompts that
            // actually do, as on the language path.
            let mut as_text = self.tokenizer.clone();
            as_text.set_encode_special_tokens(true);
            ids.extend(encode(&as_text, prompt)?);
        } else {
            ids.extend(encode(&self.tokenizer, prompt)?);
        }
        let tail_start = ids.len();
        ids.extend(encode(&self.tokenizer, &tail)?);
        // Truncating to the encoder's window instead of padding it is safe:
        // the reference pads every sequence to its 512-token maximum *with an
        // attention mask*, and for a single sequence a masked pad contributes
        // nothing a shorter unmasked sequence does not already omit.
        if ids.len() > MAX_TEXT_TOKENS && tail_start < ids.len() {
            // The tail is the assistant-opener segment, and it sits at the
            // end, so any cut reaches it: a half-cut opener conditions the
            // transformer on a turn structure no training sample carries.
            // Refuse loudly and name the overflow rather than silently
            // cutting the opener.
            return Err(ImageGenError::Backend(format!(
                "the rendered prompt is {} tokens, over the text encoder's \
                 {MAX_TEXT_TOKENS}-token window; truncating would cut the \
                 assistant turn opener",
                ids.len()
            )));
        }
        ids.truncate(MAX_TEXT_TOKENS);
        if ids.is_empty() {
            return Err(ImageGenError::EmptyPrompt);
        }
        Ok(ids)
    }
}

impl ImageGenerator for Flux2KleinModel {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    #[expect(
        clippy::too_many_lines,
        reason = "one full generation: tokenize, noise, positions, the Euler loop, decode and encode, straight-line by construction"
    )]
    #[expect(
        clippy::cast_precision_loss,
        reason = "token and grid indices are small sequence positions, far below f32's exact-integer range"
    )]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "token counts, step indices and image dimensions are bounded by the generation contract; they fit u32"
    )]
    fn generate(
        &self,
        job: &ImageGenerationJob,
        progress: Option<&dyn Fn(ImageStepEvent)>,
    ) -> Result<GeneratedImage, ImageGenError> {
        job.validate()?;
        if job.steps == 0 {
            return Err(ImageGenError::Backend(
                "steps must be at least one".to_string(),
            ));
        }
        if !job.input_references.is_empty() {
            return Err(ImageGenError::ReferenceImage(
                "reference images are not accepted: the generation endpoints carry text input only"
                    .to_string(),
            ));
        }

        // Declared first so it drops last, after every working buffer of this
        // generation is back in the pool: an idle server should not hold a
        // 1024x1024 image's activations.
        let _trim = TrimPool(&self.ops);
        let steps = job.steps as usize;
        let began = Instant::now();
        let ids = self.tokenize(&job.prompt)?;
        let t_txt = ids.len();
        let (w, h) = (job.width as usize, job.height as usize);
        let (wl, hl) = (w / 16, h / 16);
        let t_img = wl * hl;
        tracing::info!(
            width = w,
            height = h,
            steps,
            txt_tokens = t_txt,
            img_tokens = t_img,
            streaming = progress.is_some(),
            "image generation started"
        );

        // The prefill ends in a readback, so the clock covers the device work.
        let phase = Instant::now();
        let context = self.text.prefill(&self.ops, &ids)?;
        let ctx_buf = self.ops.upload_f32("pipeline context", &context)?;
        tracing::info!(
            seconds = phase.elapsed().as_secs_f64(),
            txt_tokens = t_txt,
            "image prompt encoded"
        );

        let latent_count = t_img * self.params.in_channels;

        // Standard-normal noise by Box-Muller, drawn from the job's seed:
        // the same seed gives the same image on any machine, because the
        // stream and the transform are both fixed here, not by the RNG's
        // platform.
        let mut rng = job.seed;
        let mut noise = Vec::with_capacity(latent_count + 2);
        while noise.len() < latent_count {
            let u1 = rng.next_f32().max(f32::MIN_POSITIVE);
            let u2 = rng.next_f32();
            let radius = (-2.0f32 * u1.ln()).sqrt();
            let angle = std::f32::consts::TAU * u2;
            noise.push(radius * angle.cos());
            noise.push(radius * angle.sin());
        }
        noise.truncate(latent_count);
        let mut latents = self.ops.upload_scratch("pipeline noise", &noise)?;

        // 4-axis positions, matching the reference pipeline's id prep: image
        // tokens carry (frame 0, row, column, 0) and text tokens (0, 0, 0,
        // index) - the text-id prep enumerates every combination of t=[0],
        // h=[0], w=[0] with l=0..L, so the sequence index rides the fourth
        // axis, not the second. Single blocks run the joint sequence
        // text-first.
        let mut img_positions = Vec::with_capacity(t_img * 4);
        for token in 0..t_img {
            let (li, lj) = (token / wl, token % wl);
            img_positions.extend_from_slice(&[0.0, li as f32, lj as f32, 0.0]);
        }
        let mut txt_positions = Vec::with_capacity(t_txt * 4);
        for token in 0..t_txt {
            txt_positions.extend_from_slice(&[0.0, 0.0, 0.0, token as f32]);
        }
        let mut joint_positions = txt_positions.clone();
        joint_positions.extend_from_slice(&img_positions);
        let img_pos = self.ops.upload_f32("pipeline img pos", &img_positions)?;
        let txt_pos = self.ops.upload_f32("pipeline txt pos", &txt_positions)?;
        let joint_pos = self
            .ops
            .upload_f32("pipeline joint pos", &joint_positions)?;

        let mut kv = Flux2KvCache::new(self.params.depth, self.params.depth_single_blocks);
        let scheduler = new_scheduler(t_img, steps);

        // One step per scheduled timestep. The transformer's embedding
        // applies the reference's 1000x time factor itself, so it receives
        // the raw sigma, not the scheduler's 1000-scaled timestep.
        let denoise = Instant::now();
        for (i, _) in scheduler.timesteps.iter().enumerate() {
            let phase = Instant::now();
            let sigma = scheduler.sigmas[i];
            let velocity = self.dit.forward(
                &self.ops,
                &DiTInput {
                    latents: latents.view(),
                    txt: ctx_buf.view(),
                    timestep: sigma,
                    img_positions: img_pos.view(),
                    txt_positions: txt_pos.view(),
                    joint_positions: joint_pos.view(),
                    txt_tokens: t_txt,
                    img_tokens: t_img,
                    ref_tokens: 0,
                    kv_mode: KvMode::None,
                },
                &mut kv,
            )?;
            let dt = scheduler.sigmas[i + 1] - scheduler.sigmas[i];
            let next = self.ops.alloc_f32("pipeline latents", latent_count)?;
            self.ops.axpy(
                "pipeline euler",
                next.view(),
                latents.view(),
                velocity.view(),
                latent_count,
                dt,
            )?;
            latents = next;
            // Launches only queue; draining here charges the step's device
            // work to the step that issued it. One drain per step is noise
            // beside a transformer pass.
            self.ops
                .backend()
                .sync()
                .map_err(|e| ImageGenError::Backend(format!("pipeline step sync: {e}")))?;
            tracing::info!(
                step = i + 1,
                steps,
                sigma,
                seconds = phase.elapsed().as_secs_f64(),
                "image denoising step"
            );

            if let Some(report) = progress {
                // The last step's preview would be the finished image byte
                // for byte, which the consumer drops in favour of the
                // completed frame; decoding it would cost a whole extra VAE
                // pass. Its tick still goes out.
                let partial_image = if i + 1 == steps {
                    None
                } else {
                    let phase = Instant::now();
                    let host =
                        self.ops
                            .download("pipeline progress", latents.view(), latent_count)?;
                    let rgb = self.vae.decode(&self.ops, &host, w, h)?;
                    let bytes = encode_image(&rgb, w, h, ImageOutputFormat::Png, None)?;
                    tracing::info!(
                        step = i + 1,
                        seconds = phase.elapsed().as_secs_f64(),
                        "image partial preview decoded"
                    );
                    Some(bytes)
                };
                report(ImageStepEvent {
                    step: i as u32,
                    total_steps: job.steps,
                    partial_image,
                });
            }
        }
        tracing::info!(
            seconds = denoise.elapsed().as_secs_f64(),
            steps,
            "image denoising finished"
        );

        let phase = Instant::now();
        let host = self
            .ops
            .download("pipeline final", latents.view(), latent_count)?;
        let rgb = self.vae.decode(&self.ops, &host, w, h)?;
        tracing::info!(
            seconds = phase.elapsed().as_secs_f64(),
            "image latents decoded"
        );
        let phase = Instant::now();
        let bytes = encode_image(&rgb, w, h, job.output_format, job.output_compression)?;
        tracing::info!(
            seconds = phase.elapsed().as_secs_f64(),
            total_seconds = began.elapsed().as_secs_f64(),
            bytes = bytes.len(),
            "image encoded"
        );
        Ok(GeneratedImage {
            bytes,
            media_type: job.output_format.mime_type(),
            width: job.width,
            height: job.height,
            prompt_tokens: t_txt as u32,
            completion_tokens: (t_img * steps) as u32,
        })
    }
}

/// Empties the working-buffer pool when a generation ends, however it ends.
struct TrimPool<'o>(&'o Ops);

impl Drop for TrimPool<'_> {
    fn drop(&mut self) {
        self.0.trim_pool();
    }
}

/// Reads the checkpoint's chat template: `chat_template.jinja` beside the
/// tokenizer, else `tokenizer_config.json`'s `chat_template` (a string or an
/// object with a `default`).
///
/// A checkpoint that carries neither is refused: the template decides the
/// turn structure the text encoder was trained to read, and a built-in
/// generic one would silently condition the transformer on markers the
/// checkpoint's own vocabulary may not even carry. Every real checkpoint
/// ships its template beside the tokenizer.
fn load_chat_template(tokenizer_dir: Option<&Path>) -> Result<String, ImageGenError> {
    let dir = tokenizer_dir.ok_or_else(|| {
        ImageGenError::Checkpoint("the tokenizer path names no directory to read".to_string())
    })?;
    let jinja = dir.join("chat_template.jinja");
    if jinja.exists() {
        return std::fs::read_to_string(&jinja)
            .map_err(|e| ImageGenError::Checkpoint(format!("reading {}: {e}", jinja.display())));
    }
    let config = dir.join("tokenizer_config.json");
    if config.exists() {
        let text = std::fs::read_to_string(&config)
            .map_err(|e| ImageGenError::Checkpoint(format!("reading {}: {e}", config.display())))?;
        let document: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| ImageGenError::Checkpoint(format!("parsing {}: {e}", config.display())))?;
        match &document["chat_template"] {
            serde_json::Value::String(template) => return Ok(template.clone()),
            serde_json::Value::Object(map) => {
                if let Some(serde_json::Value::String(template)) = map.get("default") {
                    return Ok(template.clone());
                }
            }
            _ => {}
        }
    }
    Err(ImageGenError::Checkpoint(format!(
        "no chat template found beside the tokenizer (searched {} and its \
         tokenizer_config.json's chat_template); refusing to condition on a \
         built-in template the checkpoint was not trained with",
        jinja.display()
    )))
}

/// Clamps a planar (3, height, width) f32 field in [0, 255] and encodes it
/// as PNG or JPEG. WebP and every other codec are out of contract.
#[expect(
    clippy::cast_possible_truncation,
    reason = "colour samples are clamped to [0, 255] before the u8 cast, and image dimensions fit u32"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "colour samples are clamped to [0, 255] before the u8 cast, so they are non-negative"
)]
fn encode_image(
    rgb: &[f32],
    width: usize,
    height: usize,
    format: ImageOutputFormat,
    compression: Option<u32>,
) -> Result<Vec<u8>, ImageGenError> {
    let plane = width * height;
    if rgb.len() != 3 * plane {
        return Err(ImageGenError::Encoding(format!(
            "expected {} colour samples, got {}",
            3 * plane,
            rgb.len()
        )));
    }
    let clamp8 = |v: &f32| (v.clamp(0.0, 255.0)).round() as u8;
    let mut pixels = Vec::with_capacity(plane * 3);
    for i in 0..plane {
        pixels.extend_from_slice(&[
            clamp8(&rgb[i]),
            clamp8(&rgb[plane + i]),
            clamp8(&rgb[2 * plane + i]),
        ]);
    }
    let buffer = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::from_raw(
        width as u32,
        height as u32,
        pixels,
    )
    .ok_or_else(|| ImageGenError::Encoding("pixel buffer does not match dimensions".to_string()))?;
    let mut out = Vec::new();
    match format {
        ImageOutputFormat::Png => {
            image::codecs::png::PngEncoder::new(&mut out)
                .write_image(
                    buffer.as_raw(),
                    width as u32,
                    height as u32,
                    image::ExtendedColorType::Rgb8,
                )
                .map_err(|e| ImageGenError::Encoding(format!("png: {e}")))?;
        }
        ImageOutputFormat::Jpeg => {
            let quality = compression.unwrap_or(90).clamp(1, 100) as u8;
            image::codecs::jpeg::JpegEncoder::new_with_quality(&mut out, quality)
                .write_image(
                    buffer.as_raw(),
                    width as u32,
                    height as u32,
                    image::ExtendedColorType::Rgb8,
                )
                .map_err(|e| ImageGenError::Encoding(format!("jpeg: {e}")))?;
        }
    }
    Ok(out)
}
