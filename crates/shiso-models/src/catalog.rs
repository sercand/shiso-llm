//! Family catalog: facts about every checkpoint family this build can serve.
//!
//! Used by `shiso-registry-editor` and `shiso-hub` to validate recipes against
//! the models actually implemented in this commit.

use serde::Serialize;

/// Facts about a model family supported by `shiso-models`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FamilyFacts {
    /// Family name as `shiso-models` spells it (e.g. "qwen35", "`nemotron_asr`").
    pub name: &'static str,
    /// Roles this family accepts (e.g. "base", "draft", "vision", "asr").
    pub accepted_roles: &'static [&'static str],
    /// API tasks this family serves (e.g. "chat", "transcription", "diarization", "`image_generation`").
    pub tasks: &'static [&'static str],
    /// Implemented feature facts for this family.
    pub features: FamilyFeatures,
}

/// Feature capabilities actually implemented for a family in `shiso`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct FamilyFeatures {
    /// Whether tool calling is supported.
    pub tools: bool,
    /// Supported reasoning effort labels. Empty if reasoning is not supported.
    pub reasoning_efforts: &'static [&'static str],
    /// Default reasoning effort, if reasoning is supported.
    pub default_reasoning_effort: Option<&'static str>,
    /// Whether structured output is supported.
    pub structured_output: bool,
}

impl FamilyFeatures {
    /// Returns whether this family supports reasoning traces.
    #[must_use]
    pub const fn supports_reasoning(&self) -> bool {
        !self.reasoning_efforts.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use crate::family_catalog;

    #[test]
    fn catalog_contains_all_nine_families() {
        let catalog = family_catalog();
        assert_eq!(catalog.len(), 9);
        let names: Vec<&str> = catalog.iter().map(|f| f.name).collect();
        assert!(names.contains(&"qwen35"));
        assert!(names.contains(&"qwen35moe"));
        assert!(names.contains(&"qwen4exp"));
        assert!(names.contains(&"gemma4"));
        assert!(names.contains(&"nemotron_h"));
        assert!(names.contains(&"nemotron_asr"));
        assert!(names.contains(&"nemotron_diar"));
        assert!(names.contains(&"qwen3_asr"));
        assert!(names.contains(&"flux2"));
    }

    #[test]
    fn catalog_serializes_to_json() {
        let catalog = family_catalog();
        let json = serde_json::to_string(catalog).expect("catalog should serialize");
        assert!(json.contains("qwen35"));
    }
}
