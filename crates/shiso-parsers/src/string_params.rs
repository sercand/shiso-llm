//! Which tool parameters are declared as strings.
//!
//! The Qwen 3.5 XML call body carries no types: `<parameter=path>\n123\n` is
//! the same bytes whether the tool wanted the string "123" or the number 123.
//! Without a schema the reader guesses from the text. With one it must not:
//! a `path` of `"123"`, or file content that happens to be valid JSON, becomes a
//! number or an object, and the tool receives something it never declared.

use std::collections::{HashMap, HashSet};

/// The string-typed parameters of each tool a request declared.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct StringParams {
    by_tool: HashMap<String, HashSet<String>>,
}

impl StringParams {
    /// Reads `(tool name, JSON schema)` pairs. A parameter counts as a string
    /// when its `type` is `"string"`, or a list whose only non-null member is
    /// `"string"`. Anything the schema does not say is left to the reader's
    /// guess, as before.
    #[must_use]
    pub fn from_schemas<'a>(
        tools: impl IntoIterator<Item = (&'a str, &'a serde_json::Value)>,
    ) -> Self {
        let mut by_tool = HashMap::new();
        for (tool, schema) in tools {
            let Some(properties) = schema
                .get("properties")
                .and_then(serde_json::Value::as_object)
            else {
                continue;
            };
            let strings: HashSet<String> = properties
                .iter()
                .filter(|(_, property)| is_string_type(property.get("type")))
                .map(|(name, _)| name.clone())
                .collect();
            if !strings.is_empty() {
                by_tool.insert(tool.to_owned(), strings);
            }
        }
        Self { by_tool }
    }

    /// Whether `param` of `tool` was declared as a string.
    #[must_use]
    pub fn is_string(&self, tool: &str, param: &str) -> bool {
        self.by_tool
            .get(tool)
            .is_some_and(|params| params.contains(param))
    }
}

/// Whether a schema `type` names exactly the string type, allowing a nullable
/// spelling (`["string", "null"]`). A union with any other type is not a
/// string: the value could legitimately be that other type, so the reader's
/// guess stands.
fn is_string_type(ty: Option<&serde_json::Value>) -> bool {
    match ty {
        Some(serde_json::Value::String(t)) => t == "string",
        Some(serde_json::Value::Array(types)) => {
            let mut named = types
                .iter()
                .filter_map(serde_json::Value::as_str)
                .filter(|t| *t != "null");
            named.next() == Some("string") && named.next().is_none()
        }
        _ => false,
    }
}
