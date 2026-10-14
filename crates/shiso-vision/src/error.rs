//! Typed failures of the host vision pipeline.
//!
//! Every variant here is something a caller can act on, and the thing they
//! would act on stays a field rather than being formatted into a message:
//! the server turns a rejected decode into a 400 and a missing decoder binary
//! into a 500, and it should not have to read English to tell them apart.
//!
//! Decoder failures keep the underlying `image` or `io` error as a real
//! [`std::error::Error::source`], so an error-chain walker reaches the
//! decoder's own complaint — "Image size exceeds limit", say — without this
//! crate having to paraphrase it.

use std::path::PathBuf;

use shiso_core::DType;
use thiserror::Error;

/// Everything the host vision pipeline can refuse.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum VisionError {
    /// A geometry field the preprocessor divides by is zero.
    ///
    /// Config parsing reports a missing key as `0`, so an absent `patch_size`
    /// would otherwise arrive here as a divisor and panic inside a request
    /// task. There is deliberately no fallback default: silently assuming the
    /// usual `16` would let a mismatched checkpoint produce a wrongly-shaped
    /// pixel buffer, which is a wrong answer rather than an error.
    #[error("vision geometry field {field} is zero (missing or invalid in the checkpoint config)")]
    GeometryZero {
        /// Name of the offending field, as it is spelled in the config.
        field: &'static str,
    },

    /// A geometry field is far above anything a real tower declares.
    ///
    /// The bound exists so that `patch_size x spatial_merge_size` and
    /// `3 x temporal_patch_size x patch_size^2` are provably in range on every
    /// supported target, which is what lets the rest of the crate multiply them
    /// without a checked-arithmetic dance on every use.
    #[error("vision geometry field {field} is {value}, above the supported maximum of {max}")]
    GeometryTooLarge {
        /// Name of the offending field.
        field: &'static str,
        /// Value read from the config.
        value: usize,
        /// Largest value this build accepts.
        max: usize,
    },

    /// A geometry field does not fit this target's pointer width.
    #[error("vision geometry field {field} does not fit a usize on this target")]
    GeometryWidth {
        /// Name of the offending field.
        field: &'static str,
    },

    /// The container could not be read: its index or sample tables are broken.
    /// Refused rather than retried on another decoder, which would hide a
    /// corrupt upload behind whichever decoder happens to tolerate it.
    #[error("the video container is malformed: {source}")]
    Demux {
        /// What the demuxer found wrong.
        #[from]
        source: crate::demux::DemuxError,
    },

    /// The media payload carried no bytes at all.
    #[error("the media payload is empty")]
    EmptyPayload,

    /// The bytes are not a picture this build can decode, or they declare a
    /// size past the decoder limits. The decoder's own complaint is the source.
    #[error("the image could not be decoded")]
    ImageDecode {
        /// The decoder's own error, including its limit messages.
        #[source]
        source: image::ImageError,
    },

    /// The container is one the in-process GIF decoder does not handle and the
    /// operator did not enable video decoding.
    #[error(
        "decoding {container} needs video decoding, which is disabled; \
         pass --video-allow-decode to enable it, or send an animated GIF"
    )]
    VideoDecodeDisabled {
        /// What the payload's magic bytes say it is.
        container: Container,
    },

    /// Video decoding is enabled but no decoder in this build handled the
    /// container. Every step declined (an `Unsupported` outcome, never a
    /// failure), so the request is refused rather than guessed at.
    #[error(
        "no video decoder in this build handled {container}{}; tried: {}. Enable the \
         `decode-ffmpeg` feature and pass --ffmpeg-path, or use a platform with an OS decoder",
        codec.as_deref().map(|c| format!(" ({c})")).unwrap_or_default(),
        if tried.is_empty() { "nothing".to_owned() } else { tried.join(", ") }
    )]
    NoVideoDecoder {
        /// What the payload's magic bytes say it is.
        container: Container,
        /// The video codec, when the container was read far enough to know it.
        codec: Option<String>,
        /// The steps that were offered the container, in order.
        tried: Vec<&'static str>,
    },

    /// The container parsed but held no frames.
    #[error("the container decoded to zero frames (is there a video stream?)")]
    NoFrames,

    /// The clip is shorter than a single temporal group, so it has no
    /// representation in the patch tensor.
    ///
    /// Refused rather than padded into a still: a still would report a
    /// plausible token count for something the model never saw as motion.
    #[error(
        "the clip has {frames} usable frame(s) but temporal_patch_size is \
         {temporal_patch_size}; a clip must carry at least one full temporal group"
    )]
    ShortClip {
        /// Frames that survived sampling.
        frames: usize,
        /// Frames one temporal group needs.
        temporal_patch_size: usize,
    },

    /// Frames of one clip disagree about their size, so no single patch grid
    /// describes them. Sizing is decided once from the first frame and applied
    /// to the rest, and that is only sound when they match.
    #[error(
        "frame {index} is {width}x{height} but the clip started at {first_width}x{first_height}"
    )]
    FrameSizeMismatch {
        /// Position of the odd frame in the sampled sequence.
        index: usize,
        /// Width of the odd frame.
        width: u32,
        /// Height of the odd frame.
        height: u32,
        /// Width the first frame established.
        first_width: u32,
        /// Height the first frame established.
        first_height: u32,
    },

    /// The configured decoder binary could not be started.
    ///
    /// Distinct from a decode failure on purpose: a misconfigured deployment
    /// and a corrupt upload have completely different fixes, and an error that
    /// reads like the latter sends the operator looking at the wrong thing.
    #[error(
        "could not run {binary:?} - is ffmpeg installed and on PATH? (set --ffmpeg-path to point at it)"
    )]
    DecoderSpawn {
        /// The binary as configured, resolved on PATH or used as given.
        binary: PathBuf,
        /// Why the spawn failed.
        #[source]
        source: std::io::Error,
    },

    /// The decoder ran past its wall-clock budget and was killed.
    #[error("decoding exceeded {secs}s and was stopped")]
    DecoderTimeout {
        /// The budget that was exceeded.
        secs: u64,
    },

    /// The decoder produced more bytes than the output cap allows. The child
    /// is killed at the moment the cap is passed, not after it finishes.
    #[error("decoded output exceeded the {cap}-byte cap")]
    DecoderOutputCap {
        /// The cap that was passed.
        cap: usize,
    },

    /// The decoder exited non-zero. `detail` is the last line it wrote to
    /// stderr, which is where it says what it actually disliked.
    #[error("decoder failed: {detail}")]
    DecoderFailed {
        /// Last line of the decoder's stderr, or a placeholder when it was
        /// silent.
        detail: String,
    },

    /// One frame of the decoder's output stream did not parse as a PNG.
    #[error("frame {index} of the decoded stream did not decode as PNG")]
    FramePng {
        /// Position of the frame in the concatenated stream.
        index: usize,
        /// The decoder's own error.
        #[source]
        source: image::ImageError,
    },

    /// Talking to the decoder subprocess failed.
    #[error("{what} failed")]
    Io {
        /// The step that failed, so the `io::Error` has a subject.
        what: &'static str,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },

    /// An RGB frame was offered with a byte count its declared size cannot
    /// hold. The two are separate inputs, so nothing but this check stops a
    /// caller's stride mistake from being read as image content.
    #[error("an RGB frame of {width}x{height} needs {expected} byte(s), got {got}")]
    FrameShape {
        /// Declared width in pixels.
        width: u32,
        /// Declared height in pixels.
        height: u32,
        /// Bytes the declared size needs.
        expected: usize,
        /// Bytes that were offered.
        got: usize,
    },

    /// The pixel buffer handed to [`crate::VisionItem::new`] does not match the
    /// grid it claims to describe.
    ///
    /// A silent mismatch here is the worst failure in this crate: the pad run
    /// in the prompt, the rows the encoder returns and the position stream all
    /// derive from this grid, and a desync between them produces fluent output
    /// about the wrong picture.
    #[error(
        "a {grid_t}x{grid_h}x{grid_w} grid of {row_len}-value rows needs {expected} values, got {got}"
    )]
    ItemShape {
        /// Temporal groups the buffer claims.
        grid_t: usize,
        /// Patch rows per group, vertically.
        grid_h: usize,
        /// Patch rows per group, horizontally.
        grid_w: usize,
        /// Values in one patch row.
        row_len: usize,
        /// Values the grid implies.
        expected: usize,
        /// Values the buffer actually holds.
        got: usize,
    },

    /// A vision item was asked for with a zero extent. There is no such item:
    /// an encoder run over it would return no rows, and a pad run of length
    /// zero is not a media reference at all.
    #[error(
        "a vision item needs a non-zero extent; got {grid_t}x{grid_h}x{grid_w} rows of {row_len}"
    )]
    ItemEmpty {
        /// Temporal groups asked for.
        grid_t: usize,
        /// Patch rows asked for, vertically.
        grid_h: usize,
        /// Patch rows asked for, horizontally.
        grid_w: usize,
        /// Values per patch row asked for.
        row_len: usize,
    },

    // -- the tower loader ---------------------------------------------------
    /// A tower field the encoder's shapes are derived from is zero.
    #[error("vision tower field {field} is zero (missing or invalid in the checkpoint config)")]
    TowerFieldZero {
        /// Name of the offending field, as the config spells it.
        field: &'static str,
    },

    /// A width the tower splits evenly is not divisible by what splits it.
    ///
    /// Left unchecked this truncates every head or every merge block by a
    /// fraction, and nothing downstream can see that it happened - the shapes
    /// still multiply out, the kernels still run, and the answers are wrong.
    #[error("vision tower {field} is {value}, which {divisor_field} ({divisor}) does not divide")]
    TowerNotDivisible {
        /// The width being split.
        field: &'static str,
        /// Its value.
        value: usize,
        /// What was meant to split it.
        divisor_field: &'static str,
        /// The divisor's value.
        divisor: usize,
    },

    /// The learned position table is not a square grid, so there is nothing to
    /// resample it over.
    #[error("the tower declares {count} position embeddings, which is not a square grid")]
    PositionGridNotSquare {
        /// Rows the table declares.
        count: usize,
    },

    /// Reading the tower's file failed.
    #[error("reading the vision tower at {path} failed")]
    TowerIo {
        /// The file that could not be read.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },

    /// The tower shard's header cannot be trusted.
    #[error("the vision tower shard at {path} has {fault}")]
    ShardHeader {
        /// The shard that could not be parsed.
        path: PathBuf,
        /// What is wrong with it.
        fault: ShardFault,
    },

    /// The standalone tower container could not be opened or read.
    #[error("the vision tower container at {path} could not be read")]
    TowerContainer {
        /// The file that could not be opened.
        path: PathBuf,
        /// The container reader's own error.
        #[source]
        source: shiso_modelfile::GgufError,
    },

    /// The checkpoint config beside a tower could not be read.
    #[error("the checkpoint config at {path} could not be read")]
    TowerConfig {
        /// The config that could not be parsed.
        path: PathBuf,
        /// The parser's own error.
        #[source]
        source: shiso_core::CoreError,
    },

    /// The checkpoint declares no vision tower, so there is nothing to load.
    ///
    /// Distinct from "the tower failed to load": a text-only checkpoint is a
    /// correct thing to have, and the caller's answer to it is to serve text
    /// rather than to report a broken model.
    #[error("the checkpoint at {path} declares no vision tower")]
    NoVisionConfig {
        /// The checkpoint directory that was asked.
        path: PathBuf,
    },

    /// The checkpoint's config names no tower shard, and the source said the
    /// tower is in one.
    ///
    /// There is no conventional path to try instead: the shard is wherever the
    /// config's `optiq_vision.sidecar` says, or the operator names the file.
    #[error("the config of the checkpoint at {path} names no vision tower shard")]
    NoTowerShard {
        /// The checkpoint directory that was asked.
        path: PathBuf,
    },

    /// The standalone tower carries neither spelling of the tower prefix.
    #[error("no vision tower found at {path} under {short:?} or {canonical:?}")]
    NoTowerPrefix {
        /// The container that was searched.
        path: PathBuf,
        /// The short prefix that was probed first.
        short: &'static str,
        /// The longer spelling that was probed second.
        canonical: &'static str,
    },

    /// Metadata the tower's dimensions come from is absent.
    #[error("the vision tower is missing metadata key {key}")]
    TowerMetadata {
        /// The key, under the file's own architecture prefix.
        key: String,
    },

    /// The tower does not hold a tensor its dimensions require.
    #[error("the vision tower is missing tensor {name}")]
    MissingTensor {
        /// Fully qualified name the manifest asked for.
        name: String,
    },

    /// A tower tensor is not the extent the checkpoint's own geometry implies.
    ///
    /// Separate from [`VisionError::MissingTensor`] because the two say
    /// different things about the checkpoint: a name that is absent is a file
    /// this build cannot read, while a name that is present at another extent
    /// is a file whose config and whose shards disagree. Loading it anyway
    /// would produce rows of the right shape holding something else.
    #[error("the vision tower's {name} holds {got} values where its geometry implies {want}")]
    TowerTensorExtent {
        /// Fully qualified name the loader asked for.
        name: String,
        /// Values the geometry implies.
        want: usize,
        /// Values the tensor holds.
        got: usize,
    },

    /// A tensor is present but not the shape the tower's dimensions require.
    ///
    /// This is the check the manifest exists for. Every tensor in a mismatched
    /// tower is individually well-formed, so without it a tower of the wrong
    /// width loads without complaint and is wrong everywhere.
    #[error("tensor {name} is {declared:?} but the tower's dimensions require {expected:?}")]
    TensorShape {
        /// The tensor that disagreed.
        name: String,
        /// Shape the container declares, as the container writes it.
        declared: Vec<u64>,
        /// Shape the dimensions require, slowest-varying axis first.
        expected: Vec<u64>,
    },

    /// A tensor is stored in a format this loader does not upload.
    #[error("tensor {name} is stored as {} which the tower loader cannot upload", dtype.type_name())]
    TensorDType {
        /// The tensor that could not be uploaded.
        name: String,
        /// The format it is stored in.
        dtype: DType,
    },

    /// The tower holds a different number of tensors than its dimensions imply.
    #[error("the vision tower holds {found} tensor(s) but its dimensions require {expected}")]
    TensorCount {
        /// Tensors the container declares.
        found: usize,
        /// Tensors the dimensions require.
        expected: usize,
    },

    /// The standalone tower disagrees with the published contract for its
    /// architecture.
    ///
    /// Separate from a shape mismatch on purpose: a file that declares its real
    /// shape and disagrees with the published one is a different artefact,
    /// while a file that declares one shape and stores another is corrupt.
    #[error(
        "the {architecture} tower contract requires {field} = {required}, \
         but the file declares {got}"
    )]
    SidecarContract {
        /// Architecture the contract belongs to.
        architecture: &'static str,
        /// The field that disagreed.
        field: &'static str,
        /// What the contract requires.
        required: usize,
        /// What the file declares.
        got: usize,
    },

    /// Moving a tensor to the device failed.
    #[error("uploading vision tensor {name} to the device failed")]
    Upload {
        /// The tensor that was in flight.
        name: String,
        /// The device's own error.
        #[source]
        source: shiso_core::CoreError,
    },

    /// The item's patch grid is not a whole number of merge blocks.
    ///
    /// The merger folds a square block of patches into one token, so a grid with
    /// a leftover row or column has patches that belong to no token. The
    /// preprocessor snaps every grid to the merge unit, which means this can
    /// only be an item built by hand or an item paired with the wrong tower -
    /// and the second is the one worth naming, because a silently truncated grid
    /// would crop the picture rather than fail.
    #[error("a {grid_h}x{grid_w} patch grid is not a whole number of {merge}x{merge} merge blocks")]
    ItemGridNotMergeable {
        /// Patch rows in the grid.
        grid_h: usize,
        /// Patch columns in the grid.
        grid_w: usize,
        /// Side of the block the merger folds.
        merge: usize,
    },

    /// The item's rows are not the width this tower's patch projection consumes.
    ///
    /// Both numbers are products of a patch geometry, so a disagreement means
    /// the item was preprocessed for a different tower. Every row would still
    /// multiply cleanly against the projection - it would simply be reading the
    /// wrong pixels - which is why the widths are compared rather than trusted.
    #[error("the tower's patch projection takes rows of {expected} values, but the item has {got}")]
    ItemRowWidth {
        /// Width the tower's patch projection consumes.
        expected: usize,
        /// Width the item's rows carry.
        got: usize,
    },

    /// The tower's heads are wider than the attention kernel can hold.
    ///
    /// The kernel keeps a lane's share of the head in registers, so the width is
    /// bounded. A head past the bound would be truncated, and a truncated head
    /// produces a plausible vector rather than a failure, so the bound is checked
    /// on the host before anything launches.
    #[error("the tower's head width of {head_dim} is above the supported maximum of {max}")]
    HeadTooWide {
        /// Width of one attention head.
        head_dim: usize,
        /// Widest head the kernels hold.
        max: usize,
    },

    /// A weight is stored in a width the encoder's kernels do not read.
    ///
    /// Distinct from [`VisionError::TensorDType`], which is the loader refusing
    /// to upload a format at all: this tensor is on the device and readable, but
    /// the kernels are built per storage width - a brain float widened by a
    /// shift, a half by its conversion - and read one width for the whole
    /// tower. Handed anything else they would read wrong numbers and say
    /// nothing.
    #[error(
        "vision weight {name} is stored as {}, but the encoder's kernels read the whole tower in one width, bf16 or f16",
        dtype.type_name()
    )]
    WeightDType {
        /// The weight the encoder was about to bind.
        name: String,
        /// The width it is stored in.
        dtype: DType,
    },

    /// The family this checkpoint belongs to ships no vision tower in this
    /// build.
    ///
    /// Reported by the family's own tower opener rather than by the boot path,
    /// so that "this build serves no tower for this family" and "this tower
    /// would not load" stay two different answers. An operator who pointed
    /// `--vision` at a perfectly good file and a family that cannot use one
    /// should not be told their file is broken.
    #[error("{what}")]
    Unsupported {
        /// What this build cannot do, written for the operator.
        what: &'static str,
    },

    /// A step of the encode failed on the device.
    #[error("the vision encoder failed during {step}")]
    Encode {
        /// The step that failed, as the encoder names it.
        step: &'static str,
        /// The device's own error.
        #[source]
        source: shiso_core::CoreError,
    },
}

/// What a media payload's leading bytes say it is.
///
/// Dispatch is on magic bytes rather than the declared MIME type, so a client
/// that labels an MP4 `video/gif`, or sends `application/octet-stream`, still
/// reaches the right decoder. The value is carried in the error so the operator
/// is told which container they sent as well as which flag turns it on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Container {
    /// ISO base media: `.mp4`, `.m4v`, `.mov`.
    Mp4,
    /// Matroska and `WebM`.
    Matroska,
    /// RIFF/AVI.
    Avi,
    /// Animated GIF, the one container decoded in process.
    Gif,
    /// Leading bytes match nothing this build recognizes.
    Unknown,
}

impl std::fmt::Display for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Container::Mp4 => "an mp4/mov container",
            Container::Matroska => "a matroska/webm container",
            Container::Avi => "an avi container",
            Container::Gif => "a gif",
            Container::Unknown => "an unrecognized container",
        };
        f.write_str(text)
    }
}

/// What is wrong with a tower shard's header.
///
/// An enum rather than a message: the caller that has to decide between "the
/// download is truncated" and "this build cannot read this format" should not
/// be matching on English to do it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ShardFault {
    /// The file ends before its header does.
    Truncated,
    /// The declared header length is zero or implausibly large.
    HeaderLength,
    /// The header is not the JSON table the format requires.
    HeaderJson,
    /// A tensor declares a storage format this loader does not read.
    UnsupportedDType,
    /// A tensor declares no usable shape.
    MissingShape,
    /// A tensor's payload range runs outside the file.
    OffsetsOutOfRange,
}

impl std::fmt::Display for ShardFault {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            ShardFault::Truncated => "a truncated header",
            ShardFault::HeaderLength => "an implausible header length",
            ShardFault::HeaderJson => "a malformed header table",
            ShardFault::UnsupportedDType => "a tensor format this build cannot read",
            ShardFault::MissingShape => "a tensor with no usable shape",
            ShardFault::OffsetsOutOfRange => "a tensor payload outside the file",
        };
        f.write_str(text)
    }
}
