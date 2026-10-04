//! Shared vocabulary for the inference stack: identifiers, device memory, the
//! device and model traits, and tensor formats.
//!
//! Every other crate is built on the types defined here, so the rules that
//! matter most here are the ones that turn mistakes into type errors: newtypes
//! instead of bare integers, device buffers that free themselves, borrowed
//! device ranges that cannot outlive the buffer they name, and constructors
//! instead of two-phase initialization.
//!
//! Nothing here depends on a device. The crate builds and tests identically
//! with no GPU present, which is where the config and ownership rules are
//! exercised before any device is involved.
//!
//! # Nothing here names a model
//!
//! No checkpoint family appears in this crate: not in a type, not in a module
//! name, not in a variant. What the bottom of the stack needs to know about a
//! loaded model is [`ModelInfo`] - five numbers a scheduler reads. Geometry
//! belongs to whatever runs the forward pass, and the identity of a checkpoint
//! belongs to the crate that can load it. Adding a third family should touch no
//! file here, and the way to keep that true is for there to be nothing here to
//! touch.
#![forbid(unsafe_code)]

pub mod backend;
pub mod buffer;
pub mod config;
pub mod dtype;
pub mod error;
pub mod ids;
pub mod ledger;
pub mod media;
pub mod model;
pub mod page;

pub use backend::{
    Backend, BlasBackend, CoopLaunch, DEVICE_POINTER_BYTES, Extent3, FnConst, FnConstValue,
    GemmDType, GemmDesc, GemmOp, GemmReduction, GraphBackend, GraphCapture, GraphKey, KernelArg,
    KernelRef, LaunchArgs, MemoryInfo, PointerPack,
};
pub use buffer::{
    DeviceBuffer, DeviceView, HostBytes, RawBuffer, alloc, alloc_tagged, import_host,
};
pub use config::{HfConfig, HiddenAct, VisionConfig};
pub use dtype::DType;
pub use error::{Chain, CoreError, CoreResult, chain_of};
pub use ids::{LayerIdx, Seed, SessionId, TokenId, TokenOutOfRange};
pub use ledger::{LedgerReport, MemLedger, MemTag, Residence};
pub use media::{
    KEY_ID_FLOOR, MediaDigest, MediaKey, MediaModel, MediaSpan, PadTokens, VisionGrid,
    VisionPrefill,
};
pub use model::{
    DecisionModel, DecisionQuestion, DecodeStep, DraftHint, Logits, Model, ModelInfo,
    SamplingParams, Speculated, Speculating, StopTokens, VerifiedBlock,
};
