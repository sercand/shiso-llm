//! The path from a neutral request to a stream of events.
//!
//! [`pipeline`] is the one path from a neutral request to a stream of events,
//! and [`blocking`] folds such a stream into the single response a
//! non-streaming caller gets. [`transcribe`] is transcription through a speech
//! decoder, on top of [`pipeline`].

pub mod blocking;
pub mod inference_types;
pub mod pipeline;
pub mod transcribe;
