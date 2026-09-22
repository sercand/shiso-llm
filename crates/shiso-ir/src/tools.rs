//! Tools: what the caller offers, and how it wants them used.
//!
//! Both shapes are the neutral ones. Neither carries a discriminator field, a
//! nested wrapper object, or a serde derive, because none of those are facts
//! about a tool - they are facts about one wire format's envelope, and the
//! adapter that parses that envelope is where they belong.

use serde_json::Value;

/// One tool the caller is offering.
#[derive(Debug, Clone, PartialEq)]
///
/// Not `#[non_exhaustive]`: this is a type the surfaces BUILD, field by field,
/// from what a caller asked for. Per the rule the workspace settled on, the
/// attribute belongs on what a crate hands out, not on what another crate has to
/// construct.
pub struct ToolDefinition {
    /// The name the model emits to call it.
    pub name: String,
    /// What it does, shown to the model. Absent means undocumented.
    pub description: Option<String>,
    /// JSON Schema for the arguments.
    ///
    /// Kept as a raw value: the chat template renders it verbatim, and the JSON
    /// library is built to preserve key order, so the caller's schema reaches the
    /// model spelled the way the caller wrote it.
    pub input_schema: Value,
}

impl ToolDefinition {
    /// A tool with a name and a schema.
    #[must_use]
    pub fn new(name: impl Into<String>, input_schema: Value) -> Self {
        Self {
            name: name.into(),
            description: None,
            input_schema,
        }
    }
}

/// How the caller wants the offered tools used.
///
/// A closed enum rather than a mode string beside an optional name, which is the
/// shape both wire formats use and which admits two states that mean nothing: a
/// mode of "none" with a forced name, and a forced choice with no name.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum ToolChoice {
    /// The model decides whether to call anything.
    Auto,
    /// The model must not call a tool.
    None,
    /// The model must call at least one, its pick.
    Any,
    /// The model must call this one.
    Named(String),
}

/// Per-request thresholds for the repeated-output detector.
///
/// All three fields together or none at all: a partial override would mix the
/// caller's span length with the server's repeat count, and the result is a
/// detector neither of them configured.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
///
/// Not `#[non_exhaustive]`, and the reason is the same everywhere it is missing
/// in this workspace: that attribute exists so a library can add a field without
/// breaking callers it cannot recompile. Every caller here is in this repository
/// and recompiles with it, and this is a type other crates have to BUILD - so all
/// the attribute would buy is that they cannot. A new field should break every
/// site that now has to supply it.
pub struct RepetitionDetection {
    /// Shortest repeated span worth flagging, in tokens.
    pub min_pattern_size: u32,
    /// Longest repeated span worth flagging, in tokens.
    pub max_pattern_size: u32,
    /// How often a span must repeat before it counts as a loop.
    pub min_count: u32,
}
