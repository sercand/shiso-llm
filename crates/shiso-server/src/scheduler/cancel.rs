//! Getting an event to a client, and what it means when that fails.
//!
//! # A hang-up is discovered, not announced
//!
//! There is no cancel message on the queue. A client that gives up drops its
//! stream, which closes the receiving end of the sequence's channel, and the
//! next send discovers it. That is deliberate: a cancel message would be a
//! second way of saying something the drop already says unambiguously, and the
//! two could disagree - a request could be cancelled and still be read from, or
//! dropped without a message ever arriving.
//!
//! # The worker frees the session, never the client
//!
//! Discovering a hang-up marks the sequence dead. It does *not* close the
//! session: that happens at the end of the tick, on the one thread that owns the
//! model, through the same retirement path every other ending takes. A client
//! that could cause a session to be released would be a client that can make the
//! engine thread free device state in the middle of a batched forward pass.
//!
//! # Why a slow client is abandoned rather than waited for
//!
//! This is the one thread every other request is waiting on. A blocking send
//! hands one slow reader the power to stop the whole server; a `try_send` that
//! gave up instantly would drop events during an ordinary scheduling hiccup. So
//! a send retries for a bounded time and then abandons the event - and an
//! abandoned event is treated exactly as a hang-up, because a client that has
//! not read for five seconds is a client that is not going to.

use std::time::{Duration, Instant};

use shiso_ir::{StopReason, StreamEvent};
use tokio::sync::mpsc;

use crate::scheduler::lifecycle::SeqSlot;
use crate::scheduler::stats::SchedulerStats;

/// How long a blocked send waits before the client is abandoned, in
/// milliseconds.
const DEFAULT_SEND_TIMEOUT_MS: u64 = 5_000;

/// How long the loop sleeps between attempts at a full event queue.
const SEND_POLL_MS: u64 = 1;

/// The environment knob for the send deadline.
const SEND_TIMEOUT_ENV: &str = "SHISO_STREAM_SEND_TIMEOUT_MS";

/// What became of one event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Delivery {
    /// The client has it.
    Delivered,
    /// The client hung up; nothing more will be read.
    Closed,
    /// The client is alive but has not read for longer than the deadline, so
    /// the event was dropped rather than blocking the engine thread.
    Abandoned,
}

impl Delivery {
    /// Whether this outcome means the client is gone for scheduling purposes.
    pub(crate) const fn is_lost(self) -> bool {
        matches!(self, Self::Closed | Self::Abandoned)
    }
}

/// Sends one event, bounded.
pub(crate) fn send_event(tx: &mpsc::Sender<StreamEvent>, event: StreamEvent) -> Delivery {
    let deadline = Duration::from_millis(send_timeout_ms());
    let started = Instant::now();
    let mut pending = event;
    loop {
        match tx.try_send(pending) {
            Ok(()) => return Delivery::Delivered,
            Err(mpsc::error::TrySendError::Closed(_)) => return Delivery::Closed,
            Err(mpsc::error::TrySendError::Full(event)) => {
                if started.elapsed() >= deadline {
                    tracing::warn!(
                        waited_ms = started.elapsed().as_millis(),
                        "a client did not read its stream within the deadline; \
                         abandoning an event"
                    );
                    return Delivery::Abandoned;
                }
                pending = event;
                std::thread::sleep(Duration::from_millis(SEND_POLL_MS));
            }
        }
    }
}

/// Sends every event of a batch, stopping at the first outcome that says the
/// client is gone.
///
/// Stopping matters: once a client is gone every remaining send would take the
/// whole abandonment deadline to discover the same thing, which would turn one
/// hung-up client into seconds of the one engine thread.
pub(crate) fn send_all(
    stats: &SchedulerStats,
    slot: &mut SeqSlot,
    events: Vec<StreamEvent>,
) -> Delivery {
    let mut outcome = Delivery::Delivered;
    for event in events {
        outcome = send_event(&slot.sink, event);
        if outcome.is_lost() {
            break;
        }
    }
    if outcome.is_lost() {
        note_lost_client(stats, slot, outcome);
    }
    outcome
}

/// Marks a sequence whose client is gone.
///
/// Counted once per sequence, which is what makes the counter a count of
/// abandoned turns rather than of abandoned events.
pub(crate) fn note_lost_client(stats: &SchedulerStats, slot: &mut SeqSlot, outcome: Delivery) {
    if slot.is_done() {
        return;
    }
    tracing::info!(
        slot = slot.id,
        emitted = slot.generated().len(),
        abandoned = matches!(outcome, Delivery::Abandoned),
        "the client stopped reading; the sequence will be retired at the step boundary"
    );
    slot.cancel();
    stats.cancelled();
}

/// Cuts every sequence that has outlived its deadline.
///
/// Costs nothing when no sequence carries one, which is the ordinary case: a
/// server with no `--request-timeout` never pays for the check.
pub(crate) fn enforce_deadlines(stats: &SchedulerStats, slots: &mut [SeqSlot], now: Instant) {
    if !slots.iter().any(|slot| slot.deadline.is_some()) {
        return;
    }
    for slot in slots {
        if slot.is_done() {
            continue;
        }
        let Some(deadline) = slot.deadline else {
            continue;
        };
        if now < deadline {
            continue;
        }
        tracing::warn!(
            slot = slot.id,
            emitted = slot.generated().len(),
            "a request outlived its deadline and is being cut"
        );
        // A deadline is reported as a deadline and never as "length". The two
        // are indistinguishable to a client that only reads a wire word, and
        // they call for opposite responses: a truncated answer can be continued,
        // a timed-out one should be retried or asked for more cheaply.
        slot.stop(StopReason::Timeout);
        stats.timed_out();
    }
}

/// The send deadline, read once.
///
/// Read once rather than per send: a value that could change between two events
/// of the same stream would make the abandonment threshold a property of when an
/// event happened rather than of how the server was configured.
fn send_timeout_ms() -> u64 {
    static VALUE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *VALUE.get_or_init(|| {
        std::env::var(SEND_TIMEOUT_ENV)
            .ok()
            .and_then(|raw| raw.parse().ok())
            .unwrap_or(DEFAULT_SEND_TIMEOUT_MS)
    })
}

#[cfg(test)]
mod tests {
    use super::{Delivery, enforce_deadlines, send_event};
    use crate::scheduler::GenerateRequest;
    use crate::scheduler::lifecycle::{Ending, SeqSlot, SlotSetup};
    use crate::scheduler::stats::SchedulerStats;
    use shiso_core::{SamplingParams, Seed, SessionId, TokenId};
    use shiso_ir::{StopReason, StreamEvent};
    use std::num::NonZeroU64;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    /// A sequence and the receiver that keeps its stream alive.
    ///
    /// The receiver is returned rather than dropped: a dropped one makes every
    /// send report a hang-up, which is the opposite of what the deadline case is
    /// about and would let that test pass for the wrong reason.
    fn slot(deadline: Option<Instant>) -> (SeqSlot, tokio::sync::mpsc::Receiver<StreamEvent>) {
        let request = GenerateRequest::new(
            Arc::from(vec![TokenId::new(1)]),
            SamplingParams::neutral(),
            Seed::new(1),
            8,
        );
        let (sink, rx) = tokio::sync::mpsc::channel(8);
        let mut ids = NonZeroU64::MIN;
        let slot = SeqSlot::new(
            request,
            SlotSetup {
                id: "s".to_owned(),
                session: SessionId::next(&mut ids).expect("a session id"),
                sink,
                started_at: Instant::now(),
                deadline,
                budget: 8,
                keyed: false,
                suppressed: Arc::from([]),
                score_reply: None,
            },
            None,
            shiso_parsers::passthrough(),
        );
        (slot, rx)
    }

    #[tokio::test]
    async fn a_dropped_receiver_is_reported_as_a_hang_up_rather_than_as_a_send() {
        let (tx, rx) = tokio::sync::mpsc::channel(4);
        drop(rx);
        assert_eq!(
            send_event(&tx, StreamEvent::BlockStop { index: 0 }),
            Delivery::Closed
        );
    }

    /// A server with no deadline never pays for the check, and never cuts
    /// anything.
    #[tokio::test]
    async fn nothing_is_cut_when_no_sequence_carries_a_deadline() {
        let stats = SchedulerStats::new();
        let (held, _alive) = slot(None);
        let mut slots = vec![held];
        enforce_deadlines(
            &stats,
            &mut slots,
            Instant::now() + Duration::from_secs(3600),
        );
        assert!(!slots[0].is_done());
        assert_eq!(stats.timeouts(), 0);
    }

    /// The reason is a deadline, never a budget: the two call for opposite
    /// responses from a client that reads only the wire word.
    #[tokio::test]
    async fn a_sequence_past_its_deadline_stops_for_that_reason_and_not_for_length() {
        let stats = SchedulerStats::new();
        let now = Instant::now();
        let (held, _alive) = slot(Some(now));
        let mut slots = vec![held];
        enforce_deadlines(&stats, &mut slots, now + Duration::from_millis(1));
        assert_eq!(slots[0].ending, Some(Ending::Stopped(StopReason::Timeout)));
        assert_eq!(stats.timeouts(), 1);
    }
}
