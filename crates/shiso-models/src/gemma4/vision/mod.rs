//! The Gemma 4 family's vision tower.
//!
//! The mmproj container is read by [`loader`], the graph is run by
//! [`encoder`], and [`Gemma4Tower`] is the object the server holds behind the
//! [`shiso_vision::VisionTower`] trait. The family's preprocessing contract —
//! patch side, pooling, normalization — is `shiso_vision`'s
//! [`PreprocessSpec::gemma4`]; the tower hands it out rather than restating
//! it, so the preprocessor and the encoder cannot disagree about a row.

mod encoder;
mod loader;

pub use encoder::Gemma4Tower;
