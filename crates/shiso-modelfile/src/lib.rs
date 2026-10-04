//! Model-file loader: reads a model's on-disk directory, describes every
//! tensor, and maps tensor names onto the layer roles a forward pass needs.
//!
//! This crate is a *describer*, not a converter. Opening a model never copies
//! a weight and never dequantizes one: [`TensorView`] hands out a borrow into
//! the file mapping, and the kernels in the device crates read the quantized
//! blocks in place. That is also why [`GgufFile::mapping`] exists - the same
//! mapping the directory was parsed from is what a device backend wraps, so
//! weights reach the GPU with no copy and no lifetime hazard.
//!
//! # Module map
//!
//! | Module | Responsibility |
//! |---|---|
//! | [`reader`] | header, metadata table, tensor directory, the mapping |
//! | [`metadata`] | key/value metadata, lazily decoded arrays, compat getters |
//! | [`quant`] | one `#[repr(C)]` struct per quant block format, block sizing |
//! | [`quant_tables`] | the fixed lookup tables the 2-bit and 4-bit formats use |
//! | [`source`] | the one trait a role mapping needs from a container |
//! | [`st_header`] | the safetensors container itself: header, spans, formats |
//! | [`safetensors`] | the MLX/OptiQ export: affine triples, role translation |
//! | [`modelopt`] | the modelopt mixed-precision export: plain, FP8 and NVFP4 |
//! | [`mlx_quant`] | the affine group-quantization format, decoded to `f16` |
//! | [`nvfp4`] | four-bit floats under two levels of scale, decoded to `f32` |
//! | [`fp8`] | e4m3, decoded for weights and encoded for activations |
//! | [`modelfile`] | one handle over both containers, chosen by the path |
//!
//! The module tree is format-neutral at the top on purpose: [`source`] reads
//! [`source::TensorSource`] rather than a file format, so the two containers
//! bind through exactly the same role names, layout checks and errors.
//!
//! # The two containers are not the same shape
//!
//! A single-file checkpoint stores weights in formats the kernels read, so its
//! tensors are windows onto a mapping and reading one costs nothing. A
//! checkpoint directory stores them packed four or eight bits at a time, so its
//! tensors have to be decoded, and a decoded model does not fit where the
//! packed one did. That difference is not hidden: [`ModelFile::tensor_bytes`]
//! returns a [`std::borrow::Cow`], and [`ModelFile::is_mmap_backed`] answers
//! the one question a device-side caller actually has.
//!
//! # The mmap boundary
//!
//! [`memmap2::Mmap::map`] is `unsafe` for a real reason: the returned bytes are
//! a window onto a file, so if another process truncates or overwrites that
//! file while it is mapped, every `&[u8]` this crate has handed out becomes a
//! view of changing or unmapped memory. No amount of type-checking catches
//! that, so the risk is written down rather than assumed away:
//!
//! - exactly two `unsafe` blocks exist in this crate, one per container's
//!   private mapping constructor, and each carries the invariant in full;
//! - the caller-facing half of that invariant is documented on
//!   [`GgufFile::open`], [`GgufFile::open_shared`] and
//!   [`SafetensorsFile::open`], which stay safe functions because the rule is a
//!   filesystem-level one that no signature can express;
//! - everything else, including [`TensorView::blocks`] and the whole of
//!   [`mlx_quant`], is safe code built on checked casts.
//!
//! # Errors
//!
//! Every failure is a [`GgufError`] variant naming the tensor or key involved.
//! Nothing in this crate logs-and-continues past a malformed file, and nothing
//! calls `panic!` on untrusted input: a corrupt model is a normal runtime
//! condition, not a bug in the caller.
#![allow(
    unsafe_code,
    reason = "memmap2::Mmap::map is unsafe; see the mmap boundary in the crate docs"
)]

pub mod fp8;
pub mod metadata;
pub mod mlx_quant;
pub mod modelfile;
pub mod modelopt;
pub mod name_map;
pub mod nvfp4;
pub mod quant;
pub mod quant_tables;
pub mod reader;
pub mod safetensors;
pub mod source;
pub mod st_header;
pub mod ternary;

pub use fp8::{Fp8Layout, dequant_fp8, e4m3, quantize_activations, to_e4m3};
pub use metadata::{GgufMetadata, MetaArray, MetaType, MetaValue};
pub use mlx_quant::{MlxAffineLayout, dequant_affine};
pub use modelfile::ModelFile;
pub use modelopt::{Census, ModeloptFile, ModeloptTensor, Nvfp4Parts, Stored};
pub use name_map::{GdnHeads, HeadAxis, HeadPermute, IdentityNames, NameMap, Role, ValueOp};
pub use nvfp4::{Nvfp4Layout, dequant_nvfp4};
pub use quant::QuantBlock;
pub use reader::{GgufError, GgufFile, GgufResult, MAX_DIMS, TensorDesc, TensorShape, TensorView};
pub use safetensors::{AffineTriple, PackedAffine, SafetensorsFile, ScaleFormat};
pub use source::TensorSource;
