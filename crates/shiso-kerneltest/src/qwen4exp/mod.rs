//! The sparse family's kernel set: cases for the entries whose failure a whole
//! pass hides.
//!
//! A model-level replay runs every one of these entries, but it runs them over
//! a page table the pool handed out in order, so a kernel that reads the pool
//! as if a session's pages were contiguous answers correctly there and only
//! goes wrong once the pool has been through enough sessions to hand out pages
//! out of order. The cases here build that table on purpose.
//!
//! The `#[test]` wrappers live in the backend's own driver -
//! `crates/shiso-cuda/tests/q4e_idx.rs`.
//!
//! Every case asserts by panicking, because that is the whole of what a case
//! does.

pub mod affine;
pub mod bf16;
pub mod exl3;
pub mod ffn;
pub mod fp8;
pub mod gdn;
pub mod gdn_fused;
pub mod hc;
pub mod hc_mix;
pub mod idx;
pub mod idxring;
pub mod indexer;
pub mod kv8;
pub mod lowp;
pub mod matvec;
pub mod misc;
pub mod mxfp8;
pub mod nvfp4;
pub mod p12;
pub mod ple;
pub mod ple_nvfp4;
pub mod qsa;
pub mod route;
