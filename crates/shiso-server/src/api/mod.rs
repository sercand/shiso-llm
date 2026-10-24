//! The HTTP handlers and the wire shapes they answer with.
//!
//! [`pipeline`], [`blocking`] and [`inference_types`] live in `shiso-runtime`
//! and are re-exported here.
//!
//! [`pipeline`] is the one path from a neutral request to a stream of events,
//! [`blocking`] folds such a stream into the single response a non-streaming
//! caller gets, and [`sse`] is where a streaming one is framed. The wire types
//! and handlers of each surface live beside them, one module per family:
//! [`crate::openai`] and [`crate::anthropic`].

pub mod misc_handlers;
pub use shiso_runtime::api::{blocking, inference_types, pipeline};
pub mod service_tier;
pub mod sse;
