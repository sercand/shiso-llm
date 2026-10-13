//! Video to patch rows: the temporal sibling of [`crate::preprocess_image`].
//!
//! # What a video is, to this tower
//!
//! The vision transformer has NO temporal attention. Frames fuse INSIDE a
//! patch: the flattened patch dimension is `3 * temporal_patch_size * patch^2`,
//! so one patch already spans `temporal_patch_size` frames' worth of pixels. A
//! still image fills that axis by replicating itself - the axis was always
//! there, and a still is the degenerate case.
//!
//! So a clip of `n` frames becomes `n / temporal_patch_size` TEMPORAL GROUPS,
//! each a full `grid_h x grid_w` patch plane built from consecutive frames.
//! Every group is shaped exactly like a preprocessed still, which is why the
//! encoder needs no notion of time at all and why both paths produce the same
//! [`VisionItem`] type. What does differ is POSITION: a still holds the
//! multimodal rope's temporal coordinate constant across its whole pad run
//! while a video advances it once per group, and that is why the group count
//! is carried rather than the groups being flattened into separate images.
//!
//! # Container support
//!
//! Two backends, chosen by MAGIC BYTES rather than by the declared MIME type:
//! GIF decodes in process, pure Rust, always available; everything else goes to
//! a subprocess decoder that the operator has to enable. Sniffing the bytes
//! rather than trusting the label means a client that sends an mp4 labelled
//! `video/gif`, or labelled nothing at all, still reaches the right decoder -
//! and a GIF mislabelled as mp4 does not needlessly spawn a process.

use image::RgbImage;

use crate::chain::{StepInput, VideoDecodePolicy, VideoStep, run_chain};
use crate::config::{ImageLimits, PreprocessSpec};
use crate::error::{Container, VisionError};
use crate::image::{filter, pack_group, round_u32, side_as_usize, target_size_for, widen};
use crate::item::VisionItem;

/// Frames per second to sample at when the caller has nothing better to go on.
/// Matches the rate every shipped video processor block for this family
/// declares.
pub const DEFAULT_FPS: f64 = 2.0;

/// Sampling floor, from the checkpoints' own video processor.
///
/// The floor matters more than it looks: with a temporal patch size of two,
/// fewer than two frames cannot fill a single group, and a one-frame "video"
/// would otherwise silently become a still.
pub const DEFAULT_MIN_FRAMES: usize = 4;

/// Sampling ceiling, from the checkpoints' own video processor. A long clip is
/// truncated in token cost rather than refused.
pub const DEFAULT_MAX_FRAMES: usize = 768;

/// One decoded frame, in RGB.
///
/// A newtype rather than the decoder library's own image type: this is a public
/// signature, and exposing the library type there would make a point release of
/// an image decoder a breaking change for every caller of this crate.
#[derive(Clone, PartialEq, Eq)]
pub struct DecodedFrame {
    frame: RgbImage,
}

impl DecodedFrame {
    /// Wraps a packed RGB byte buffer, three bytes per pixel, row-major.
    ///
    /// # Errors
    ///
    /// [`VisionError::FrameShape`] when the buffer cannot hold the declared
    /// size. The size and the bytes are separate inputs, so this check is the
    /// only thing between a caller's stride mistake and a garbled frame that
    /// decodes, encodes and answers.
    pub fn from_rgb8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, VisionError> {
        let expected = side_as_usize(width)
            .saturating_mul(side_as_usize(height))
            .saturating_mul(3);
        let got = pixels.len();
        RgbImage::from_raw(width, height, pixels)
            .map(|frame| Self { frame })
            .ok_or(VisionError::FrameShape {
                width,
                height,
                expected,
                got,
            })
    }

    /// Frame width in pixels.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.frame.width()
    }

    /// Frame height in pixels.
    #[must_use]
    pub fn height(&self) -> u32 {
        self.frame.height()
    }

    /// The packed RGB bytes, three per pixel, row-major.
    #[must_use]
    pub fn as_rgb8(&self) -> &[u8] {
        self.frame.as_raw()
    }

    /// Borrowed as the decoder library's own type, for the resize step.
    pub(crate) fn inner(&self) -> &RgbImage {
        &self.frame
    }

    /// Adopts a frame the decoders produced directly.
    pub(crate) fn from_image(frame: RgbImage) -> Self {
        Self { frame }
    }
}

/// Summarised rather than derived; a frame is megabytes of pixel data.
impl std::fmt::Debug for DecodedFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodedFrame")
            .field("width", &self.frame.width())
            .field("height", &self.frame.height())
            .finish()
    }
}

/// Decoded frames together with the rate they represent.
///
/// The two travel as one value because sampling needs both and they are
/// meaningless apart: a frame list whose rate is guessed somewhere else samples
/// to the wrong count, and the resulting token count is plausible rather than
/// obviously wrong.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DecodedClip {
    frames: Vec<DecodedFrame>,
    fps: f64,
}

impl DecodedClip {
    /// Binds frames to the rate they play at.
    ///
    /// A non-finite or non-positive rate is normalized to [`DEFAULT_FPS`]
    /// rather than refused: a container that declares nothing usable is common,
    /// and the sampler needs a defined divisor, not a failed request.
    ///
    /// # Errors
    ///
    /// [`VisionError::NoFrames`] when the frame list is empty.
    pub fn new(frames: Vec<DecodedFrame>, fps: f64) -> Result<Self, VisionError> {
        if frames.is_empty() {
            return Err(VisionError::NoFrames);
        }
        Ok(Self {
            frames,
            fps: usable_fps(fps),
        })
    }

    /// The decoded frames, in order.
    #[must_use]
    pub fn frames(&self) -> &[DecodedFrame] {
        &self.frames
    }

    /// The rate the frames represent, always finite and positive.
    #[must_use]
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Keeps the frames that make the clip play at `target_fps`, dropping the
    /// rest.
    ///
    /// Consuming rather than borrowing: the kept frames are moved out, so
    /// sampling a 768-frame clip costs no pixel copies at all. Frames the
    /// sampler did not choose are dropped here, which is the point - holding
    /// the whole decode alongside the sample doubles the peak for no reason.
    #[must_use]
    pub fn into_sampled(self, target_fps: f64, temporal_patch_size: usize) -> Vec<DecodedFrame> {
        let keep = sample_indices(
            self.frames.len(),
            self.fps,
            target_fps,
            DEFAULT_MIN_FRAMES,
            DEFAULT_MAX_FRAMES,
            temporal_patch_size,
        );
        let mut pool: Vec<Option<DecodedFrame>> = self.frames.into_iter().map(Some).collect();
        keep.iter()
            .filter_map(|&index| pool.get_mut(index).and_then(Option::take))
            .collect()
    }
}

/// Picks which frame indices to keep so the clip plays at `target_fps`.
///
/// Sampling is by NEAREST INDEX over a uniform grid rather than by keeping
/// every Nth frame. The latter quantises badly whenever the ratio is not a
/// whole number - a 30fps clip at 2fps is fine as "every 15th", at 2.5fps it is
/// not - and the error shows up as a clip that drifts out of time with itself.
///
/// The count is clamped into `[min_frames, max_frames]`, then to a whole
/// multiple of `temporal_patch_size`, because a partial group has no
/// representation in the patch tensor. It is never raised above the frames that
/// actually exist: upsampling by repeating frames inflates the token count with
/// no new information, so a short clip is reported as short.
///
/// The one case that escapes the whole-group rule is a clip with fewer frames
/// than a single group. There, this reports honestly what it has and
/// [`preprocess_video`] is what refuses it - rounding one frame down to zero
/// here would turn a describable error into an empty selection the caller has
/// to reverse-engineer.
#[must_use]
pub fn sample_indices(
    n_frames: usize,
    native_fps: f64,
    target_fps: f64,
    min_frames: usize,
    max_frames: usize,
    temporal_patch_size: usize,
) -> Vec<usize> {
    if n_frames == 0 {
        return Vec::new();
    }
    let tp = temporal_patch_size.max(1);
    let native_fps = usable_fps(native_fps);
    let target_fps = usable_fps(target_fps);

    let duration = widen_usize(n_frames) / native_fps;
    let wanted = round_frames(duration * target_fps).max(1);

    let max_frames = max_frames.max(1);
    let min_frames = min_frames.max(1).min(max_frames);
    let wanted = wanted.clamp(min_frames, max_frames).min(n_frames);

    // Round DOWN to whole groups, but never below one: there has to be
    // something to encode.
    let wanted = (wanted / tp).max(1) * tp;
    let wanted = wanted.min((n_frames / tp).max(1) * tp).min(n_frames);

    if wanted >= n_frames {
        return (0..n_frames).collect();
    }
    let mut out = Vec::with_capacity(wanted);
    let span = widen_usize(n_frames - 1);
    for i in 0..wanted {
        let position = if wanted == 1 {
            0.0
        } else {
            widen_usize(i) * span / widen_usize(wanted - 1)
        };
        out.push(round_frames(position).min(n_frames - 1));
    }
    out
}

/// Decodes a container, choosing a backend by what the bytes actually are.
///
/// GIF is decoded in process and reports the container's own average rate, so
/// the caller still has to sample it. Everything else goes down the decode chain
/// (see [`crate::chain`]), whose steps sample DURING decode, so their frames are
/// already at `target_fps` and the clip reports that rate - which makes the
/// caller's sampling step a no-op rather than a second, lossy resample.
///
/// `policy` being [`None`] IS the disabled state. There is no policy value that
/// means "configured but off", so a deployment that never passes a policy can
/// never reach a decoder, and the error a caller gets names both the container
/// and the flag that would allow it.
///
/// # Errors
///
/// [`VisionError::EmptyPayload`] for no bytes;
/// [`VisionError::VideoDecodeDisabled`] when the container needs the chain and
/// no policy was supplied; [`VisionError::NoVideoDecoder`] when no step in this
/// build handled it; [`VisionError::NoFrames`] when the container held no
/// video; and the decoder's own failures otherwise.
pub fn decode_clip(
    bytes: &[u8],
    target_fps: f64,
    policy: Option<&VideoDecodePolicy>,
) -> Result<DecodedClip, VisionError> {
    if bytes.is_empty() {
        return Err(VisionError::EmptyPayload);
    }
    let container = sniff_container(bytes);
    if container == Container::Gif {
        return decode_gif(bytes);
    }
    let Some(policy) = policy else {
        return Err(VisionError::VideoDecodeDisabled { container });
    };
    #[cfg(feature = "decode-ffmpeg")]
    let ffmpeg_step = policy.ffmpeg().map(crate::chain::FfmpegStep);
    // The steps this build has, in chain order.
    #[cfg_attr(
        not(feature = "decode-ffmpeg"),
        expect(
            unused_mut,
            reason = "the steps are pushed only by compiled-in decoders"
        )
    )]
    let mut steps: Vec<&dyn VideoStep> = Vec::new();
    #[cfg(feature = "decode-ffmpeg")]
    if let Some(step) = &ffmpeg_step {
        steps.push(step);
    }
    let frames = run_chain(
        &steps,
        &StepInput::new(bytes, container, target_fps, policy),
    )?;
    DecodedClip::new(frames, target_fps)
}

/// Identifies a container from its leading bytes.
#[must_use]
pub fn sniff_container(bytes: &[u8]) -> Container {
    if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Container::Gif
    } else if bytes.len() > 12 && bytes.get(4..8) == Some(b"ftyp") {
        Container::Mp4
    } else if bytes.starts_with(&[0x1a, 0x45, 0xdf, 0xa3]) {
        Container::Matroska
    } else if bytes.starts_with(b"RIFF") {
        Container::Avi
    } else {
        Container::Unknown
    }
}

/// In-process GIF decode.
///
/// The rate comes from the per-frame delays the format stores. A GIF may
/// declare a zero delay, meaning "as fast as the viewer can", which is treated
/// as the default rate rather than divided by.
fn decode_gif(bytes: &[u8]) -> Result<DecodedClip, VisionError> {
    use image::AnimationDecoder;
    use image::codecs::gif::GifDecoder;

    let decoder = GifDecoder::new(std::io::Cursor::new(bytes))
        .map_err(|source| VisionError::ImageDecode { source })?;
    let frames = decoder
        .into_frames()
        .collect_frames()
        .map_err(|source| VisionError::ImageDecode { source })?;
    if frames.is_empty() {
        return Err(VisionError::NoFrames);
    }

    let total_ms: f64 = frames
        .iter()
        .map(|frame| {
            let (numer, denom) = frame.delay().numer_denom_ms();
            if denom == 0 {
                0.0
            } else {
                f64::from(numer) / f64::from(denom)
            }
        })
        .sum();
    let fps = if total_ms > 0.0 {
        widen_usize(frames.len()) * 1000.0 / total_ms
    } else {
        DEFAULT_FPS
    };

    let rgb = frames
        .into_iter()
        .map(|frame| {
            DecodedFrame::from_image(image::DynamicImage::ImageRgba8(frame.into_buffer()).to_rgb8())
        })
        .collect();
    DecodedClip::new(rgb, fps)
}

/// Groups already-sampled frames into the patch rows the tower consumes.
///
/// Produces the same [`VisionItem`] type a still does, with a temporal extent
/// above one. A separate video type carrying identical fields would duplicate
/// the pad-count arithmetic and let the two drift, and that arithmetic is
/// exactly what the prompt's pad run and the encoder's row count both read.
///
/// Sizing is decided ONCE, from the first frame, and applied to all of them.
/// Per-frame sizing would be a correctness bug rather than a refinement: the
/// groups are concatenated into one pad run whose token count assumes a single
/// grid.
///
/// # Errors
///
/// [`VisionError::NoFrames`] for an empty sequence; [`VisionError::ShortClip`]
/// when the frames do not fill one temporal group; and
/// [`VisionError::FrameSizeMismatch`] when the frames disagree about their
/// size, since no single grid then describes them. A geometry that cannot
/// drive a preprocessor is refused when the spec is built, not here.
pub fn preprocess_video(
    frames: &[DecodedFrame],
    spec: &PreprocessSpec,
    limits: ImageLimits,
) -> Result<VisionItem, VisionError> {
    let geom = spec.geometry;
    let tp = geom.temporal_patch_size();
    let Some(first) = frames.first() else {
        return Err(VisionError::NoFrames);
    };
    if frames.len() < tp {
        return Err(VisionError::ShortClip {
            frames: frames.len(),
            temporal_patch_size: tp,
        });
    }
    for (index, frame) in frames.iter().enumerate() {
        if frame.width() != first.width() || frame.height() != first.height() {
            return Err(VisionError::FrameSizeMismatch {
                index,
                width: frame.width(),
                height: frame.height(),
                first_width: first.width(),
                first_height: first.height(),
            });
        }
    }

    let (th, tw) = target_size_for(first.height(), first.width(), spec, limits.max_pixels());
    let ps = geom.patch_size();
    let grid_h = side_as_usize(th) / ps;
    let grid_w = side_as_usize(tw) / ps;
    // A trailing partial group is dropped rather than padded. Sampling already
    // rounds to whole groups, so this only fires for a caller that assembled
    // the sequence itself, and half a group of motion has no representation.
    let grid_t = frames.len() / tp;
    let patch_dim = geom.patch_dim();

    let mut pixels = Vec::with_capacity(grid_t * grid_h * grid_w * patch_dim);
    for group in 0..grid_t {
        // Resize this group's frames once each, up front: the packing loop
        // reads every pixel of every frame, so resizing inside it would redo
        // the work `patch^2` times.
        let resized: Vec<RgbImage> = frames
            .iter()
            .skip(group * tp)
            .take(tp)
            .map(|frame| image::imageops::resize(frame.inner(), tw, th, filter(spec.resample)))
            .collect();
        let borrowed: Vec<&RgbImage> = resized.iter().collect();
        pixels.extend_from_slice(&pack_group(&borrowed, spec, grid_h, grid_w));
    }

    VisionItem::new(pixels.into_boxed_slice(), patch_dim, grid_t, grid_h, grid_w)
}

/// Replaces a degenerate rate with the checkpoint default.
///
/// A zero, negative or non-finite rate must not divide, and must not fail the
/// request either: containers declare nonsense rates routinely.
fn usable_fps(fps: f64) -> f64 {
    if fps.is_finite() && fps > 0.0 {
        fps
    } else {
        DEFAULT_FPS
    }
}

/// Widens a frame count for the sampling arithmetic.
fn widen_usize(count: usize) -> f64 {
    widen(u64::try_from(count).unwrap_or(u64::MAX))
}

/// Rounds a frame position to the nearest whole frame, saturating.
fn round_frames(value: f64) -> usize {
    side_as_usize(round_u32(value))
}

#[cfg(test)]
#[path = "video_tests.rs"]
mod tests;
