//! The `qwen35moe` kernel set's device cases: the two codecs a
//! `Qwen3_5MoeForConditionalGeneration` checkpoint stores its weights in.
//!
//! Every case here grades a kernel against `crates/shiso-modelfile`'s own host
//! decoder over the same bytes, and those decoders are themselves graded bit
//! for bit against torch 2.14. So the chain is device against host against an
//! independent implementation, and a failure names one link of it rather than
//! "the numbers disagree".
//!
//! **The tolerances in this module are measured and printed, not chosen.** A
//! host decode is a table lookup and a multiply and is exact; a device dot
//! product over 2,048 terms is not, and how far apart they land is a number
//! this suite reports rather than a dial it turns. Every case prints the worst
//! divergence it saw beside the budget it was held to.
//!
//! The `#[test]` wrappers live in the backend's own driver -
//! `crates/shiso-cuda/tests/q35moe_fp8.rs`.

pub mod experts;
pub mod fp8;
pub mod nvfp4;
pub mod q4affine;
