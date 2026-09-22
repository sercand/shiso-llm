//! The Anthropic Messages surface.
//!
//! # The adapter the IR was shaped for
//!
//! This is the wire format the neutral request and response were rebuilt
//! toward, so the five stages here - wire types, request to IR, IR to a body,
//! IR to frames, and the handler that sequences them - are thinner than the
//! `OpenAI` family's. A turn is an ordered list of content blocks on both sides;
//! a tool result is a block with a failure flag on both sides; a thinking trace
//! is a block with a signature on both sides; instructions sit beside the
//! conversation on both sides; and the stream numbers blocks the same way. What
//! is left to translate is the envelope around each of those, which is what an
//! adapter should be.
//!
//! # What is deliberately not served
//!
//! `POST /v1/messages/count_tokens` has no route. A count is only useful if it
//! is a count of the *exact* prompt a real request would run, which means
//! running the same render-and-tokenize the serving path runs; anything else is
//! a second lowering of the same conversation, and a second lowering is the one
//! thing every module in this crate is arranged to avoid. A 501 stub would not
//! be free either - a route, a test, a doc comment and an acceptance item, all
//! carried for a feature nobody has asked for - so the path falls through to
//! the router's 404, which is honest about what this server does.

pub mod messages;
pub mod messages_encode;
pub mod messages_sse;
pub mod messages_to_ir;
pub mod messages_wire;

/// An identifier for one message, blocking or streamed.
///
/// The prefix is minted here rather than in [`crate::ids`], which knows only how
/// to produce random bytes: which prefix a value carries is a fact about one
/// wire format, and the neutral response carries the identifier bare precisely
/// so that each surface can stamp its own.
#[must_use]
pub fn new_message_id() -> String {
    format!("msg_{}", crate::ids::uuid_v4())
}

#[cfg(test)]
mod tests {
    use super::new_message_id;

    /// Clients of this format recognise a message by its prefix, and the
    /// identifier a stream announced has to be the one a blocking answer would
    /// have carried.
    #[test]
    fn a_message_identifier_is_one_a_client_of_this_format_recognises() {
        let id = new_message_id();
        assert!(id.starts_with("msg_"), "{id}");
        assert_ne!(id, new_message_id());
    }
}
