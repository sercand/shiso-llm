//! The server-sent-events seam: one frame type, one adapter, one place the
//! response is built.
//!
//! # Why an encoder trait and not a function per surface
//!
//! Every streaming surface is a state machine over the same event stream - it
//! remembers which blocks are open, how many tool calls it has announced, what
//! sequence number it is on - and every one of them has to be testable without a
//! socket. So a surface implements [`FrameEncoder`], which is a pure function of
//! its own state, and [`encode`] is the single place that turns one of those
//! into an HTTP response. A test drives the encoder directly with a list of
//! events and asserts on the exact frames, which is the assertion that actually
//! matters: a client sees bytes, not a type.
//!
//! # Why the frames are strings and not serialized values
//!
//! The event type the HTTP stack wants is write-only - you can build one and
//! hand it over, and you cannot ask it what it would send. A golden test for
//! wire framing that cannot read the frames is not a golden test, so the frame
//! is built here as the text it will become and converted at the last moment.
//!
//! # How a streamed turn gets stored
//!
//! A surface whose caller said `store: true` needs the finished turn *after*
//! the handler has already returned its response, which is why storing a
//! streamed turn used to be impossible. The seam is [`StoreHook`]: the adapter
//! folds the same events through [`ShisoResponseBuilder`] as they pass, and
//! hands the resulting turn to the hook when the stream drains.
//!
//! It deliberately does not ask the encoder for the body it built. An encoder
//! assembles frames, and re-deriving a whole response out of them would be a
//! second accumulator beside the one the blocking path uses - which is exactly
//! the drift the one-collector rule exists to prevent. Folding the events
//! instead means a stored streamed turn and a stored blocking turn come from
//! the same fold and the same body encoder, and cannot disagree.
//!
//! [`ShisoResponseBuilder`]: shiso_ir::ResponseBuilder

use std::collections::VecDeque;

use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse as _, Response};
use futures::{Stream, StreamExt as _};
use shiso_ir::{EventStream, ResponseBuilder, StreamEvent};

/// One frame, as a client reads it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Frame {
    /// The `event:` line, when the surface names its frames. `None` sends only
    /// a `data:` line, which is what the chat surfaces do.
    pub name: Option<String>,
    /// The `data:` payload, already serialized.
    pub data: String,
}

impl Frame {
    /// An unnamed frame carrying `data`.
    #[must_use]
    pub fn data(data: impl Into<String>) -> Self {
        Self {
            name: None,
            data: data.into(),
        }
    }

    /// A frame with an `event:` line.
    #[must_use]
    pub fn named(name: impl Into<String>, data: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            data: data.into(),
        }
    }

    /// The terminator every OpenAI-family stream ends with.
    #[must_use]
    pub fn done() -> Self {
        Self::data("[DONE]")
    }

    /// The frame as the HTTP stack will send it.
    ///
    /// `event:` is written before `data:`, which is the order both published
    /// APIs this mirrors print and the order the builder's call sequence
    /// decides. The event-stream format dispatches on the blank line rather
    /// than on field order, so either way is legal - but a client that reads a
    /// name/data pair instead of buffering to the blank line sees every named
    /// event as nameless when `data:` comes first, and enough clients are
    /// written that way to make matching the published order worth the nothing
    /// it costs.
    pub fn into_event(self) -> Event {
        match self.name {
            Some(name) => Event::default().event(name).data(self.data),
            None => Event::default().data(self.data),
        }
    }
}

/// A surface's streaming state machine.
///
/// Both methods return the frames the event produced, which may be none: a
/// block opening is an event every surface sees and only some of them say
/// anything about.
pub trait FrameEncoder: Send + 'static {
    /// Takes one event from the engine.
    fn on_event(&mut self, event: StreamEvent) -> Vec<Frame>;

    /// The frames that close the stream, after the last event.
    ///
    /// Reached whether the turn completed or the producer gave up, because a
    /// client waiting on a terminator waits forever either way.
    fn on_end(&mut self) -> Vec<Frame>;
}

/// What a surface does with a streamed turn once its stream has drained.
///
/// Boxed and `FnOnce` because it runs exactly once, at the end, and it owns
/// whatever the surface needs to store the turn - its identifier, its options,
/// the conversation it continues. A trait with one method would say the same
/// thing and force every surface that does not store to implement it.
pub type StoreHook = Box<dyn FnOnce(shiso_ir::Response) + Send>;

/// Turns an event stream into the frames one encoder produces.
///
/// A plain stream adapter rather than a task with a channel: nothing here needs
/// to run ahead of the client, and a buffered task would add a second place
/// events can be dropped when a client stops reading.
pub fn frames<E: FrameEncoder>(events: EventStream, encoder: E) -> impl Stream<Item = Frame> {
    frames_storing(events, encoder, None)
}

/// The same adapter, folding the turn as it passes for a surface that stores it.
///
/// With no hook nothing is folded and nothing is cloned, so a surface that does
/// not store pays for none of this.
pub fn frames_storing<E: FrameEncoder>(
    events: EventStream,
    encoder: E,
    store: Option<StoreHook>,
) -> impl Stream<Item = Frame> {
    /// Events already encoded and not yet handed out, plus whether the source is
    /// exhausted.
    struct State<E> {
        events: EventStream,
        encoder: E,
        queue: VecDeque<Frame>,
        ended: bool,
        builder: Option<ResponseBuilder>,
        store: Option<StoreHook>,
    }

    futures::stream::unfold(
        State {
            events,
            encoder,
            queue: VecDeque::new(),
            ended: false,
            builder: store.is_some().then(ResponseBuilder::new),
            store,
        },
        |mut state: State<E>| async move {
            loop {
                if let Some(frame) = state.queue.pop_front() {
                    return Some((frame, state));
                }
                if state.ended {
                    return None;
                }
                if let Some(event) = state.events.next().await {
                    if let Some(builder) = state.builder.as_mut()
                        && let Err(error) = builder.accept(event.clone())
                    {
                        // A turn that does not fold is a turn that cannot be
                        // stored, and it is still perfectly streamable: the
                        // frames go out and the re-fetch is the thing that will
                        // not be there.
                        tracing::warn!(
                            %error,
                            "a streamed turn did not fold into a response; it will not be stored"
                        );
                        state.builder = None;
                    }
                    state.queue.extend(state.encoder.on_event(event));
                } else {
                    state.ended = true;
                    state.queue.extend(state.encoder.on_end());
                    finish_store(state.builder.take(), state.store.take());
                }
            }
        },
    )
}

/// Folds what was accumulated and hands it to the surface's hook.
fn finish_store(builder: Option<ResponseBuilder>, store: Option<StoreHook>) {
    let (Some(builder), Some(store)) = (builder, store) else {
        return;
    };
    match builder.finish() {
        Ok(response) => store(response),
        Err(error) => tracing::warn!(
            %error,
            "a streamed turn ended without completing; it will not be stored"
        ),
    }
}

/// The streaming HTTP response for one encoder.
///
/// The keep-alive is not decoration: a turn whose first token takes tens of
/// seconds looks like a dead connection to every proxy in between, and a comment
/// frame is what keeps it open without adding anything a client has to parse.
#[must_use]
pub fn encode<E: FrameEncoder>(events: EventStream, encoder: E) -> Response {
    encode_storing(events, encoder, None)
}

/// The streaming HTTP response for one encoder, storing the turn as it drains.
#[must_use]
pub fn encode_storing<E: FrameEncoder>(
    events: EventStream,
    encoder: E,
    store: Option<StoreHook>,
) -> Response {
    let stream = frames_storing(events, encoder, store)
        .map(|frame| Ok::<Event, std::convert::Infallible>(frame.into_event()));
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// Runs an encoder over a fixed list of events and collects the frames.
///
/// The test seam, and the reason the frames are strings: this is how a golden
/// test for wire framing reads what a client would have received.
#[must_use]
pub fn drain<E: FrameEncoder>(mut encoder: E, events: Vec<StreamEvent>) -> Vec<Frame> {
    let mut out = Vec::new();
    for event in events {
        out.extend(encoder.on_event(event));
    }
    out.extend(encoder.on_end());
    out
}

#[cfg(test)]
mod tests {
    use super::{Frame, FrameEncoder, drain, frames};
    use futures::StreamExt as _;
    use shiso_ir::{EventStream, StopReason, StreamEvent, Usage};

    /// An encoder that says one thing per event, so the adapter is what is under
    /// test rather than any real surface's state machine.
    struct Counting(u32);

    impl FrameEncoder for Counting {
        fn on_event(&mut self, _event: StreamEvent) -> Vec<Frame> {
            self.0 += 1;
            vec![Frame::data(format!("event {}", self.0))]
        }

        fn on_end(&mut self) -> Vec<Frame> {
            vec![Frame::done()]
        }
    }

    fn events() -> Vec<StreamEvent> {
        vec![
            StreamEvent::Start {
                id: "a".to_owned(),
                model: "m".to_owned(),
                created: 0,
            },
            StreamEvent::Stop {
                stop: StopReason::EndTurn,
                usage: Usage::default(),
                token_ids: Vec::new(),
            },
        ]
    }

    #[tokio::test]
    async fn the_adapter_hands_out_every_frame_in_order_and_then_the_terminator() {
        let stream: EventStream = Box::pin(futures::stream::iter(events()));
        let collected: Vec<Frame> = frames(stream, Counting(0)).collect().await;
        assert_eq!(
            collected,
            vec![
                Frame::data("event 1"),
                Frame::data("event 2"),
                Frame::done()
            ]
        );
    }

    /// The test seam and the live adapter have to agree, or a golden test pins
    /// frames no client ever receives.
    #[tokio::test]
    async fn draining_an_encoder_produces_what_the_adapter_would_have_sent() {
        let stream: EventStream = Box::pin(futures::stream::iter(events()));
        let live: Vec<Frame> = frames(stream, Counting(0)).collect().await;
        assert_eq!(live, drain(Counting(0), events()));
    }

    #[test]
    fn a_named_frame_keeps_its_name_and_an_unnamed_one_has_none() {
        assert_eq!(
            Frame::named("response.created", "{}").name.as_deref(),
            Some("response.created")
        );
        assert_eq!(Frame::data("{}").name, None);
        assert_eq!(Frame::done().data, "[DONE]");
    }
}
