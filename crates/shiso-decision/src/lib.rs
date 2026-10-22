//! Decision models run in-process: the arithmetic that turns scored label
//! rows into answers, shared by the server's decision routes and by embedders
//! that have no HTTP in between.
//!
//! This crate names no model family and depends on no server, router or HTTP
//! crate; whoever runs the model hands it rows through a narrow seam.

pub mod error;
pub mod math;
pub mod schema;
pub mod score;
