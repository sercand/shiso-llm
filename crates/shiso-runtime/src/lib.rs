//! shiso-runtime: the engine behind the HTTP surface, as a library with no HTTP
//! in it.
//!
//! A request reaches the engine as a message on the queue in
//! [`scheduler::queue`] and comes back as a stream of [`shiso_ir::StreamEvent`].
//! One OS thread on the other side of that queue owns the model and is the only
//! thing that ever calls a method on it.
//!
//! # What runs behind the queue
//!
//! A batching coordinator: it holds a bounded set of resident sessions, runs one
//! batched decode step over everything that is ready, reads prompt in bounded
//! quanta so a long prompt cannot stall an answer already streaming, cuts what
//! outlived its deadline or lost its client, and stores what is worth storing to
//! the checkpoint tier. It is the only thing in this process that touches the
//! model.
//!
//! [`boot`] opens a device and a checkpoint and starts that machinery;
//! `shiso-server` puts the HTTP routes and the command line on top.
#![forbid(unsafe_code)]

pub mod api;
pub mod audio;
pub mod boot;
pub mod boot_error;
pub mod decision;
pub mod embed;
pub mod error;
pub mod hub;
pub mod ids;
pub mod image;
pub mod media;
pub mod media_cache;
pub mod model_factory;
pub mod options;
pub mod profile;
pub mod recipe_defaults;
pub mod response_store;
pub mod scheduler;
pub mod state;
pub mod tokenizer;
pub mod vision;
