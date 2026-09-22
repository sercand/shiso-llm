//! Turning a stream of events into the one response a non-streaming caller
//! gets.
//!
//! # There is one fold, and it is not here
//!
//! [`shiso_ir::ResponseBuilder`] already folds a stream into exactly the
//! response a blocking run produces, and that equivalence is the property the
//! whole event shape exists to guarantee. So this module drives the builder and
//! translates its failures into HTTP; it does not re-derive the fold. A second
//! accumulator written for the server would be a second thing that can disagree
//! with the first, and the disagreement would show up as a field that only one
//! of the two encodings ever fills in.
//!
//! # Why a stream failure is a 500 and not a 400
//!
//! Every failure a stream can report is about what happened after the request
//! was accepted: a device error, a block the producer never opened, arguments
//! the model wrote that are not JSON. None of them are things the caller can fix
//! by sending a different body, and reporting them as a client error sends a
//! client into a retry loop it can never leave.

use futures::StreamExt as _;
use shiso_ir::{EventStream, Response, ResponseBuilder, StreamError};

use crate::error::AppError;

/// Reads a stream to its end and folds it into one response.
///
/// This is the single collector for every surface. A blocking chat completion,
/// a blocking Responses call and - later - a blocking message all reach it, so
/// the three of them cannot drift apart in what they consider a finished turn.
///
/// # Errors
///
/// [`AppError::Internal`] for anything that stopped the turn: a producer
/// failure, a stream that ended without saying it was complete, or a tool call
/// whose arguments never became valid JSON. The cause travels as a `#[source]`,
/// so the log line an operator reads names which of those it was.
pub async fn collect_response(mut events: EventStream) -> Result<Response, AppError> {
    let mut builder = ResponseBuilder::new();
    while let Some(event) = events.next().await {
        builder.accept(event).map_err(from_stream_error)?;
    }
    builder.finish().map_err(from_stream_error)
}

/// The HTTP failure one stream failure becomes.
///
/// A function rather than a `From` impl, because the mapping is this crate's
/// opinion about a type from another crate and a `From` would have to live in
/// one of the two - putting HTTP statuses in the IR, or the IR's error in the
/// HTTP layer's inherent impl. A named function in the module that cares is
/// neither.
fn from_stream_error(error: StreamError) -> AppError {
    let context = match error {
        StreamError::Failed { .. } => "the engine reported a failure mid-turn",
        StreamError::Unterminated => "the engine's stream ended without completing the turn",
        StreamError::ToolInput { .. } => "the model wrote tool arguments that are not JSON",
        _ => "the engine produced a stream that does not describe a turn",
    };
    AppError::internal(context, error)
}

#[cfg(test)]
mod tests {
    use super::collect_response;
    use shiso_ir::{Block, BlockDelta, BlockStart, EventStream, StopReason, StreamEvent, Usage};

    /// A stream over a fixed list of events, which is what the engine's channel
    /// looks like from this side.
    fn stream(events: Vec<StreamEvent>) -> EventStream {
        Box::pin(futures::stream::iter(events))
    }

    fn start() -> StreamEvent {
        StreamEvent::Start {
            id: "abc".to_owned(),
            model: "a-model".to_owned(),
            created: 11,
        }
    }

    #[tokio::test]
    async fn a_text_turn_folds_into_the_response_its_events_described() {
        let response = collect_response(stream(vec![
            start(),
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::Text,
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: "Par".to_owned(),
                    token_ids: vec![1],
                },
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::Text {
                    text: "is".to_owned(),
                    token_ids: vec![2],
                },
            },
            StreamEvent::BlockStop { index: 0 },
            StreamEvent::Stop {
                stop: StopReason::EndTurn,
                usage: Usage {
                    prompt_tokens: 4,
                    completion_tokens: 2,
                    ..Usage::default()
                },
                token_ids: Vec::new(),
            },
        ]))
        .await
        .expect("the turn is complete");

        assert_eq!(response.id, "abc");
        assert_eq!(response.model, "a-model");
        assert_eq!(response.message.text(), "Paris");
        assert_eq!(response.stop, StopReason::EndTurn);
        assert_eq!(response.token_ids, vec![1, 2]);
    }

    /// A tool call's arguments arrive as JSON text and are parsed once, by the
    /// accumulator. A surface that parsed them again would be a second place the
    /// parse can fail.
    #[tokio::test]
    async fn a_tool_call_arrives_with_its_arguments_already_parsed() {
        let response = collect_response(stream(vec![
            start(),
            StreamEvent::BlockStart {
                index: 0,
                block: BlockStart::ToolUse {
                    id: "call_1".to_owned(),
                    name: "lookup".to_owned(),
                },
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::ToolInput {
                    fragment: "{\"city\":".to_owned(),
                    token_ids: vec![9],
                },
            },
            StreamEvent::BlockDelta {
                index: 0,
                delta: BlockDelta::ToolInput {
                    fragment: "\"Paris\"}".to_owned(),
                    token_ids: vec![10],
                },
            },
            StreamEvent::BlockStop { index: 0 },
            StreamEvent::Stop {
                stop: StopReason::ToolUse,
                usage: Usage::default(),
                token_ids: Vec::new(),
            },
        ]))
        .await
        .expect("the turn is complete");

        match response.message.content.first() {
            Some(Block::ToolUse(call)) => {
                assert_eq!(call.name, "lookup");
                assert_eq!(call.input["city"], "Paris");
            }
            other => panic!("the turn is one tool call, not {other:?}"),
        }
    }

    /// A producer failure is a server failure, because nothing the caller could
    /// send differently would have avoided it.
    #[tokio::test]
    async fn a_producer_failure_becomes_a_server_error_and_not_a_client_one() {
        let error = collect_response(stream(vec![
            start(),
            StreamEvent::Failed {
                message: "the device went away".to_owned(),
            },
        ]))
        .await
        .expect_err("the turn failed");
        assert_eq!(
            error.status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
    }

    /// A stream that stops without a terminal event has produced half a turn,
    /// and half a turn must not be answered with a 200.
    #[tokio::test]
    async fn a_stream_that_just_stops_is_reported_rather_than_answered() {
        let error = collect_response(stream(vec![start()]))
            .await
            .expect_err("the turn never completed");
        assert_eq!(
            error.status(),
            axum::http::StatusCode::INTERNAL_SERVER_ERROR
        );
    }
}
