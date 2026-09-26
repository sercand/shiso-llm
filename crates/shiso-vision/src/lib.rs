//! The host side of the vision tower: decode, resize, normalize, group.
//!
//! This crate turns bytes off the wire into [`VisionItem`]: a flat `f32` block
//! of patch rows plus the grid they came from. It stops there on purpose. The
//! encoder that consumes those rows is a device concern and lives with the
//! kernels; the HTTP layer that turns a request into bytes is a different
//! crate; and nothing here fetches a URL, because remote media is a policy
//! decision made at the edge, not a decoder feature.
//!
//! # What is here
//!
//! - [`preprocess_image`] turns one encoded image into a [`VisionItem`] whose
//!   temporal extent is one. What it does is not fixed: a [`PreprocessSpec`]
//!   carries the tower's geometry, its [`ResizeRule`], its normalization
//!   statistics and its [`Resample`] filter, because two families that agree on
//!   the patch size still want different target sizes for the same picture.
//! - [`decode_clip`] turns an animated container into frames plus the rate they
//!   represent, [`sample_indices`] chooses which of those frames to keep, and
//!   [`preprocess_video`] groups the survivors into the SAME [`VisionItem`]
//!   type with a temporal extent above one. One type for both, because a still
//!   is genuinely the degenerate video: the tower's patch dimension spans
//!   several frames whether or not the input moves.
//! - [`FfmpegPolicy`] and [`probe`] are the subprocess decode backend. It is
//!   off unless a caller hands a policy over, and [`Option::None`] IS the off
//!   state, so no value exists that means "configured but disabled".
//! - [`VisionTower`] is the contract a family's tower meets and the server
//!   consumes, and [`MediaPlacement`] is what that family's template writes for
//!   a picture. Both are here rather than in the server because a server that
//!   knew either one would know which family it was serving.
//! - [`open_checkpoint`] and [`open_gguf`] are the two doors a tower's
//!   weights arrive through, and [`VisionSource`] names which one the operator
//!   pointed at. What the tensors behind the door *mean* — the manifest they
//!   must match, the axis order they are stored in — is a family decision, so
//!   it lives with the family, not here.
//!
//! # What is not here
//!
//! The encoder and the weight loader. Both read device memory and both are
//! written against one family's graph, so each family ships them beside its
//! own model: `shiso-models`' `qwen35::vision` and `gemma4::vision`. What
//! stays shared is everything that is the same whatever tower reads the
//! pixels: the decode, the resize, the patch packing, the container doors,
//! and the [`VisionTower`] contract the server drives. A loader that read two
//! families' manifests from one place would be a place where a build serving
//! one family silently uploads the other's weights.
//!
//! # The invariant everything else leans on
//!
//! [`VisionItem::pad_count`] is at once the number of placeholder tokens the
//! item occupies in the prompt and the number of embedding rows the encoder
//! will hand back for it. They are the same number by construction. A drift
//! between them does not raise an error anywhere - the request succeeds, the
//! output is fluent, and it describes the wrong picture - which is why the grid
//! and the pixels are bound together in one checked constructor rather than
//! travelling as a buffer and a shape that must be kept in step.
#![forbid(unsafe_code)]

mod config;
mod container;
mod error;
mod ffmpeg;
mod image;
mod item;
pub mod shard;
mod tower;
mod video;

#[cfg(test)]
mod testing;

pub use config::{ImageLimits, PatchGeometry, PreprocessSpec, Resample, ResizeRule};
pub use container::{VisionSource, open_checkpoint, open_gguf};
pub use error::{Container, ShardFault, VisionError};
pub use ffmpeg::{Availability, FfmpegPolicy, decode_frames, probe, run_decoder};
pub use image::preprocess_image;
pub use item::VisionItem;
pub use shard::VisionShard;
pub use tower::{EncodedRows, MediaPlacement, VisionStages, VisionTower};
pub use video::{
    DEFAULT_FPS, DEFAULT_MAX_FRAMES, DEFAULT_MIN_FRAMES, DecodedClip, DecodedFrame, decode_clip,
    preprocess_video, sample_indices, sniff_container,
};
