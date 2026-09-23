//! The OpenAI-shaped surfaces: chat completions, the legacy completions route,
//! and the Responses API.
//!
//! # Three wire formats, one pipeline
//!
//! Each of the three parses its own body into [`shiso_ir::Request`] and encodes
//! what comes back in its own shape. Nothing between the parse and the encode
//! knows which of them asked: the Responses surface in particular is mapped
//! straight to the neutral request rather than being lowered into a chat
//! completion first, so a chat wire type is never constructed on that path.
//! Routing one wire format through another would make every Responses field a
//! question about what the chat format does with it.
//!
//! # Where the prefixes live
//!
//! An identifier's prefix is a fact about one wire format, so the minting
//! happens here rather than in [`crate::ids`], which knows only how to produce
//! random bytes.

pub mod chat;
pub mod chat_encode;
pub mod chat_sse;
pub mod chat_to_ir;
pub mod chat_wire;
pub mod completions;
pub mod responses;
pub mod responses_encode;
pub mod responses_sse;
pub mod responses_to_ir;
pub mod responses_wire;

/// The fingerprint every chat body carries.
///
/// A constant rather than a build hash: clients use it to notice that the
/// backend changed under them, and a value that changed on every rebuild would
/// cry wolf on every deploy of identical weights.
pub const SYSTEM_FINGERPRINT: &str = "fp_shiso";

/// An identifier for a legacy completion.
#[must_use]
pub fn new_completion_id() -> String {
    format!("cmpl-{}", crate::ids::uuid_v4())
}

/// An identifier for a chat completion, blocking or streamed.
///
/// The same prefix for both, because a streamed chat completion and a blocking
/// one are the same object delivered differently, and a client that stored the
/// identifier from a stream must be able to fetch it back.
#[must_use]
pub fn new_chunk_id() -> String {
    crate::response_store::StoredKind::ChatCompletion.mint()
}

/// An identifier for a stored response.
#[must_use]
pub fn new_response_id() -> String {
    crate::response_store::StoredKind::Response.mint()
}

/// A call's arguments as the text this format carries.
///
/// The neutral shape keeps arguments parsed so that exactly one parse exists,
/// and keeps them as a string value when they were not JSON. That string is
/// already the text: serialising it again would hand the client a JSON string
/// of its own arguments, and a replay would send that back doubly quoted.
#[must_use]
pub(crate) fn arguments_text(input: &serde_json::Value) -> String {
    match input {
        serde_json::Value::String(raw) => raw.clone(),
        other => other.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{arguments_text, new_chunk_id, new_completion_id, new_response_id};

    /// Arguments that arrived as text the server could not parse go back out
    /// as that same text, not as a JSON string of it.
    #[test]
    fn unparsed_arguments_are_not_encoded_twice() {
        assert_eq!(
            arguments_text(&serde_json::json!("not json {")),
            "not json {"
        );
        assert_eq!(arguments_text(&serde_json::json!({"a": 1})), "{\"a\":1}");
    }

    /// The prefix is how a client tells which surface minted an identifier, and
    /// how the store refuses to answer one surface's question with the other's
    /// entry.
    #[test]
    fn each_surface_mints_an_identifier_a_client_can_tell_apart() {
        assert!(new_completion_id().starts_with("cmpl-"));
        assert!(new_chunk_id().starts_with("chatcmpl-"));
        assert!(new_response_id().starts_with("resp_"));
    }
}
