//! The typed image cores: [`generate`], [`generate_stream`], [`edit`] and
//! [`edit_stream`] answer a `shiso_ir::image` request from the generator an
//! [`AppState`] serves.
//!
//! A generation and an edit are one pipeline that differs in the references an
//! edit carries, so each pair shares its implementation: [`edit`] is
//! [`generate`] with references, and the same for the streamed pair.
//!
//! # One request, two readings
//!
//! The blocking cores wait for each finished picture and return them all; the
//! streaming ones hand each [`Step`] to a sink as it happens. Both build the
//! same [`ImageGenerationJob`]s, so the bytes of a streamed picture and of a
//! blocking one are the same for the same request.
//!
//! # Why a sink and not a `Stream`
//!
//! A returned `Stream` would have to be driven by a task the core spawns, which
//! ties an embedder to a tokio context and leaves the caller no way to know
//! whether it has been admitted before it commits to anything. A sink runs in
//! the caller's own task: the core returns its refusal as an `Err` while the
//! sink has been called zero times (so a caller that has sent nothing yet can
//! still answer it as a status code), and the caller decides how the steps
//! travel - a channel, a vector, a callback.
//!
//! # What the wire values become
//!
//! The size grammar lives in [`size`]: `WxH` is rounded to the generator's
//! 16-grid and clamped to its bounds, the named aspect ratios become the
//! table's concrete pixels, and a named resolution (`2K`) becomes its square.
//! A size of `auto` is an omitted size.

use shiso_core::ids::Seed;
use shiso_ir::image::{Edit, Generate, Image, Images, Step, Usage};
use shiso_vision::generator::{
    GeneratedImage, ImageGenerationJob, ImageOutputFormat, ImageStepEvent, InputReferenceImage,
};
use tokio::sync::mpsc;

use super::{ImageService, UNLOADED_LIMITS, size};
use crate::error::AppError;
use crate::state::AppState;

/// The most pictures one request may ask for.
pub const MAX_IMAGES_PER_REQUEST: usize = 10;

/// The most denoising steps one picture may ask for. The generator runs one
/// job at a time, so an unbounded count would let a single request hold the
/// queue for as long as it liked.
pub const MAX_STEPS_PER_IMAGE: u32 = 100;

/// The refusals the IR alone can make: a picture count outside `1..=10`, a
/// step count outside `1..=100`, a compression level outside `1..=100`. Zero
/// is refused rather than clamped: a quality of zero has no encoding to mean
/// it, and silently encoding at quality 1 would answer a request nobody made.
///
/// The size grammar and the output format are judged against the loaded
/// model, so they are checked when the request is run; a blank prompt is
/// refused by the job's own validation. [`generate`] and the other cores call
/// this first, the floor every surface already exceeds.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the field at fault.
pub fn validate(request: &Generate) -> Result<(), AppError> {
    if let Some(n) = request.n
        && (n == 0 || n > MAX_IMAGES_PER_REQUEST)
    {
        return Err(AppError::bad_request(
            format!("n must be between 1 and {MAX_IMAGES_PER_REQUEST}"),
            Some("n"),
            "invalid_request",
        ));
    }
    if let Some(steps) = request.steps
        && (steps == 0 || steps > MAX_STEPS_PER_IMAGE)
    {
        return Err(AppError::bad_request(
            format!("steps must be between 1 and {MAX_STEPS_PER_IMAGE}"),
            Some("steps"),
            "invalid_request",
        ));
    }
    if let Some(compression) = request.output_compression
        && !(1..=100).contains(&compression)
    {
        return Err(AppError::bad_request(
            "output_compression must be between 1 and 100",
            Some("output_compression"),
            "invalid_request",
        ));
    }
    Ok(())
}

/// The refusals an edit makes on its shape alone, before anything else about
/// the request is judged: no reference at all, and `n` other than 1.
///
/// Public so the server's handler can run it where these refusals have always
/// come, ahead of the wire's own checks; [`validate_edit`] runs the same
/// function, so the messages have one copy.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the field at fault.
pub fn validate_edit_shape(references: &[Vec<u8>], n: Option<usize>) -> Result<(), AppError> {
    if references.is_empty() {
        return Err(AppError::bad_request(
            "an image edit needs at least one reference image in 'image'; \
             text-to-image is /v1/images/generations",
            Some("image"),
            "invalid_request",
        ));
    }
    if let Some(n) = n
        && n != 1
    {
        return Err(AppError::bad_request(
            format!("n must be 1 for an edit, got {n}"),
            Some("n"),
            "invalid_request",
        ));
    }
    Ok(())
}

/// [`validate`] for an edit: [`validate_edit_shape`], then everything a
/// generation is held to.
///
/// # Errors
///
/// [`AppError::BadRequest`] naming the field at fault.
pub fn validate_edit(request: &Edit) -> Result<(), AppError> {
    validate_edit_shape(&request.references, request.generate.n)?;
    validate(&request.generate)
}

/// Judges a request exactly as [`generate`] and [`edit`] would before they
/// queue anything, without running it: the IR floor, the size and output
/// format, the missing-model refusal, and what the loaded model says of the
/// references (pass an edit's, an empty slice for a generation).
///
/// For a caller that has to know a request will be accepted before it commits
/// to anything, as the server does before it spawns a stream. The checks are
/// the cores' own, so the answer cannot drift; the cost is that an edit's
/// references are checked twice, which is a copy of their bytes and, when the
/// request names no size, one decode of the first.
///
/// # Errors
///
/// The refusals of [`generate`] that precede the first queue slot.
pub fn preflight(
    state: &AppState,
    request: &Generate,
    references: &[Vec<u8>],
) -> Result<(), AppError> {
    validate(request)?;
    let plan = Plan::from_request(request, state.image.as_ref())?;
    let service = state.image()?;
    plan.with_references(references_of(references.to_vec()), service)?;
    Ok(())
}

/// Makes the pictures of a text-to-image request, in turn.
///
/// In turn, not at once: the service runs one generation at a time anyway, and
/// a caller that asked for three gets them in request order.
///
/// All-or-nothing: an overload partway through an `n > 1` request fails the
/// whole request and returns no picture. A result carrying fewer than the `n`
/// pictures the caller asked for would be a silent contract violation - the
/// caller could not tell a partial success from a truncated list - so the
/// refusal is the whole answer and a retry redoes the lot.
///
/// # Errors
///
/// [`AppError::BadRequest`] for a request [`validate`] or the loaded model
/// refuses (a malformed size, an unknown output format, a job the generator
/// would reject); [`AppError::NoImageModel`] when no generator is loaded;
/// [`AppError::Overloaded`] when the queue is full; [`AppError::Internal`]
/// when the worker is gone or the generator failed on the device.
pub async fn generate(state: &AppState, request: Generate) -> Result<Images, AppError> {
    validate(&request)?;
    run(state, request, Vec::new()).await
}

/// Makes a text-to-image request's pictures, handing each [`Step`] to `sink`
/// as it happens.
///
/// `sink` returns `false` when nobody is reading any more: the core stops
/// asking for further pictures, and returns `Ok`. A reader that stops does not
/// stop the picture already running; the service finishes it.
///
/// A refusal that arrives before the first call to `sink` is the request's
/// refusal; one that arrives after is a failure of a stream already begun.
///
/// # Errors
///
/// The errors of [`generate`].
pub async fn generate_stream(
    state: &AppState,
    request: Generate,
    sink: impl FnMut(Step) -> bool + Send,
) -> Result<(), AppError> {
    validate(&request)?;
    run_stream(state, request, Vec::new(), sink).await
}

/// Edits the references into one picture.
///
/// # Errors
///
/// The errors of [`generate`], and [`AppError::BadRequest`] from
/// [`validate_edit`] or when the loaded model takes no references, takes fewer
/// than were given, or cannot read them.
pub async fn edit(state: &AppState, request: Edit) -> Result<Images, AppError> {
    validate_edit(&request)?;
    run(state, request.generate, references_of(request.references)).await
}

/// [`generate_stream`] for an edit.
///
/// # Errors
///
/// The errors of [`edit`].
pub async fn edit_stream(
    state: &AppState,
    request: Edit,
    sink: impl FnMut(Step) -> bool + Send,
) -> Result<(), AppError> {
    validate_edit(&request)?;
    run_stream(
        state,
        request.generate,
        references_of(request.references),
        sink,
    )
    .await
}

fn references_of(bytes: Vec<Vec<u8>>) -> Vec<InputReferenceImage> {
    bytes
        .into_iter()
        .map(|bytes| InputReferenceImage { bytes })
        .collect()
}

/// The blocking answer, once the request is checked.
async fn run(
    state: &AppState,
    request: Generate,
    references: Vec<InputReferenceImage>,
) -> Result<Images, AppError> {
    let plan = Plan::from_request(&request, state.image.as_ref())?;
    let service = state.image()?;
    let plan = plan.with_references(references, service)?;
    let mut images = Vec::with_capacity(plan.n);
    let mut usage = Usage::default();
    for index in 0..plan.n {
        let generated = service.generate(plan.job(index)).await?;
        add_usage(&mut usage, &generated);
        images.push(image_of(&generated));
    }
    Ok(Images {
        created: crate::ids::unix_timestamp(),
        images,
        usage,
    })
}

/// The streamed answer, once the request is checked.
async fn run_stream(
    state: &AppState,
    request: Generate,
    references: Vec<InputReferenceImage>,
    mut sink: impl FnMut(Step) -> bool + Send,
) -> Result<(), AppError> {
    let plan = Plan::from_request(&request, state.image.as_ref())?;
    let service = state.image()?;
    let plan = plan.with_references(references, service)?;
    let created = crate::ids::unix_timestamp();
    let mut usage = Usage::default();
    // The wire's `partial_image_index` counts across ALL images of one
    // request: with n>1 the counter keeps incrementing into the second image
    // rather than restarting, which is why it lives outside the loop.
    let mut partial_index = 0u32;
    for index in 0..plan.n {
        let (progress_tx, mut progress_rx) = mpsc::unbounded_channel::<ImageStepEvent>();
        // Admission is decided synchronously - validate and the queue slot -
        // so a refusal is still answerable as a status code while nothing
        // has gone out, and an accepted job can flush the head before the
        // first step runs.
        let mut running = service.admit(plan.job(index), Some(progress_tx))?;
        if !sink(Step::Admitted) {
            return Ok(());
        }
        // The worker drops its progress sender before it sends the reply,
        // so `recv` starts answering `None` while the answer is still
        // pending; a bare `recv` branch would then hand back `None` forever
        // and spin beside the running generation, so the branch is disabled
        // once the channel says it is done.
        let mut progress_open = true;
        let answer = loop {
            tokio::select! {
                step = progress_rx.recv(), if progress_open => {
                    match step {
                        Some(event) => {
                            if !forward_partial(event, &mut partial_index, &mut sink) {
                                return Ok(());
                            }
                        }
                        None => progress_open = false,
                    }
                }
                answer = &mut running => break answer,
            }
        };
        // The worker sends every step event before the reply, so anything
        // still buffered belongs to this image and goes out before its
        // completed step.
        while let Ok(event) = progress_rx.try_recv() {
            if !forward_partial(event, &mut partial_index, &mut sink) {
                return Ok(());
            }
        }
        let generated = answer?;
        add_usage(&mut usage, &generated);
        let completed = Step::Completed {
            created,
            image: image_of(&generated),
            usage,
        };
        if !sink(completed) {
            return Ok(());
        }
    }
    Ok(())
}

/// Hands one step event to the sink as a [`Step::Partial`], when it has a
/// preview to show; `false` when the sink is gone.
///
/// A step that carries no bytes produces nothing: its payload is the partial
/// image, and a tick has nothing to put in it. The last step is skipped too:
/// its partial decode is the finished image byte for byte, so forwarding it
/// would make every client decode the same picture twice for no new
/// information; the completed step carries it once.
fn forward_partial(
    event: ImageStepEvent,
    index: &mut u32,
    sink: &mut impl FnMut(Step) -> bool,
) -> bool {
    if event.step.saturating_add(1) >= event.total_steps {
        return true;
    }
    let Some(bytes) = event.partial_image else {
        return true;
    };
    let step = Step::Partial {
        index: *index,
        bytes,
    };
    *index += 1;
    sink(step)
}

fn add_usage(usage: &mut Usage, image: &GeneratedImage) {
    usage.prompt_tokens = usage.prompt_tokens.saturating_add(image.prompt_tokens);
    usage.completion_tokens = usage
        .completion_tokens
        .saturating_add(image.completion_tokens);
}

fn image_of(generated: &GeneratedImage) -> Image {
    Image {
        bytes: generated.bytes.clone(),
        media_type: generated.media_type.to_owned(),
    }
}

/// One validated request as the service will run it: the job template and how
/// many of it to make.
#[derive(Debug)]
struct Plan {
    /// Prompt every image of the request carries.
    prompt: String,
    /// Output width, normalized to the grid.
    width: u32,
    /// Output height, normalized to the grid.
    height: u32,
    /// Denoising steps: the named count, or the service default.
    steps: u32,
    /// The seed of the first image: the named one, or entropy so the images
    /// of one request differ from each other.
    seed: u64,
    /// Encoding every image is returned in.
    output_format: ImageOutputFormat,
    /// Compression, when the caller named one.
    output_compression: Option<u32>,
    /// How many images to generate, `1..=10`.
    n: usize,
    /// Whether the caller named a size (or an aspect ratio); an edit that
    /// names none takes the size its references imply.
    size_named: bool,
    /// The reference pictures every image of the request is edited from;
    /// empty for text-to-image.
    references: Vec<InputReferenceImage>,
}

impl Plan {
    /// Checks the values the IR alone cannot judge - the size grammar, the
    /// output format - and folds in the service defaults.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] for a size string that is neither shape, and
    /// for an output format the generator does not encode.
    fn from_request(request: &Generate, service: Option<&ImageService>) -> Result<Self, AppError> {
        // With no model loaded the request is still judged for shape, against
        // the published limits, so a malformed one stays a 400 ahead of the 503.
        let limits = service.map_or(UNLOADED_LIMITS, |s| *s.limits());
        let defaults = service.map_or_else(|| super::ImageDefaults::of(&limits), |s| *s.defaults());
        // `auto` is an omitted size: the caller is asking for the default,
        // which is exactly what an omitted size gets. It is read out here
        // rather than in the grammar because the grammar has no business
        // knowing what the default is.
        let size = request
            .size
            .as_deref()
            .filter(|value| !size::is_auto(value));
        let (width, height) = match (size, request.aspect_ratio.as_deref()) {
            // `size` wins when both are named: it is the more specific of the
            // two, and a client that sends both wants the pixels it spelled.
            (Some(value), _) => {
                size::normalize_size(value, &limits).map_err(|error| size_error(&error, "size"))?
            }
            (None, Some(value)) => size::normalize_size(value, &limits)
                .map_err(|error| size_error(&error, "aspect_ratio"))?,
            (None, None) => (defaults.width, defaults.height),
        };
        // The generator's own format parser is the one authority on what it
        // encodes; its refusal is the request's fault and names the field.
        let output_format = request
            .output_format
            .as_deref()
            .map(str::parse::<ImageOutputFormat>)
            .transpose()
            .map_err(super::refuse)?;
        Ok(Self {
            prompt: request.prompt.clone(),
            width,
            height,
            steps: request.steps.unwrap_or(defaults.steps),
            // A named seed makes the request reproducible and image `i` of it
            // gets `seed + i`; an unnamed one draws a base from the process's
            // one entropy source so the `n` images of one request differ.
            seed: request.seed.unwrap_or_else(crate::ids::entropy_u64),
            output_format: output_format.unwrap_or_default(),
            output_compression: request.output_compression,
            n: request.n.unwrap_or(1),
            size_named: size.is_some() || request.aspect_ratio.is_some(),
            references: Vec::new(),
        })
    }

    /// Attaches the references, after the checks that only the loaded model
    /// can make, all before any queue slot or device work is spent.
    ///
    /// # Errors
    ///
    /// [`AppError::BadRequest`] when the model takes no references, when there
    /// are more than it takes, or when they imply no size.
    fn with_references(
        mut self,
        references: Vec<InputReferenceImage>,
        service: &ImageService,
    ) -> Result<Self, AppError> {
        if references.is_empty() {
            return Ok(self);
        }
        let max = service.max_references();
        if max == 0 {
            return Err(AppError::bad_request(
                format!(
                    "the loaded model '{}' does not take reference images",
                    service.model_id()
                ),
                Some("image"),
                "unsupported_parameter",
            ));
        }
        if references.len() > max {
            return Err(AppError::bad_request(
                format!(
                    "{} reference images were given; the loaded model '{}' accepts at most {max}",
                    references.len(),
                    service.model_id()
                ),
                Some("image"),
                "invalid_request",
            ));
        }
        // The model's own size for these references, not the text-to-image
        // default: a single reference is edited at its own aspect ratio.
        if !self.size_named {
            (self.width, self.height) = service.default_size_for_references(&references)?;
        }
        self.references = references;
        Ok(self)
    }

    /// The job for image `index` of the request.
    fn job(&self, index: usize) -> ImageGenerationJob {
        let offset = u64::try_from(index).unwrap_or(u64::MAX);
        ImageGenerationJob {
            prompt: self.prompt.clone(),
            width: self.width,
            height: self.height,
            steps: self.steps,
            seed: Seed::new(self.seed.wrapping_add(offset)),
            output_format: self.output_format,
            output_compression: self.output_compression,
            input_references: self.references.clone(),
        }
    }
}

/// A size refusal, naming the field that carried the value.
fn size_error(error: &size::SizeError, param: &str) -> AppError {
    AppError::bad_request(error.to_string(), Some(param), "invalid_request")
}
