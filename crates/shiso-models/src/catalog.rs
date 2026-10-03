//! Family catalog: facts about every checkpoint family this build can serve.
//!
//! Used by `shiso-registry` and `shiso-hub` to validate recipes against
//! the models actually implemented in this commit.

use serde::Serialize;

/// Facts about a model family supported by `shiso-models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FamilyFacts {
    /// Family name as `shiso-models` spells it (e.g. "qwen35", "`nemotron_asr`").
    pub name: &'static str,
}
