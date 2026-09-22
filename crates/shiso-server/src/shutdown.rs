//! Graceful shutdown: one notification, delivered to everyone waiting.
//!
//! # The order of the two operations in [`wait`] is the whole point
//!
//! A waiter that checks the flag first and then registers interest can miss the
//! signal entirely: if the signal arrives between the check and the
//! registration, the check said "not yet" and the notification went to nobody.
//! [`tokio::sync::Notify::notified`] is documented to register interest at the
//! moment the future is created, so creating the future *before* re-reading the
//! flag closes that window - the waiter is either already registered when the
//! notification fires, or it sees the flag it set beforehand.
//!
//! # Why the state is process-global
//!
//! There is one process, one listener and one set of signals, and a handler
//! installed twice would deliver twice. A `OnceLock` is what makes the second
//! installation a no-op rather than a second handler.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::sync::Notify;

/// Set once the first signal arrives, and never cleared.
static REQUESTED: AtomicBool = AtomicBool::new(false);

/// The rendezvous every waiter parks on.
fn notify() -> &'static Notify {
    static NOTIFY: OnceLock<Notify> = OnceLock::new();
    NOTIFY.get_or_init(Notify::new)
}

/// Whether shutdown has been asked for.
#[must_use]
pub fn is_requested() -> bool {
    REQUESTED.load(Ordering::SeqCst)
}

/// Asks for shutdown and wakes every waiter.
///
/// Public because a test - and, later, an administrative route - needs to drive
/// the drain path without sending itself a signal.
pub fn request() {
    REQUESTED.store(true, Ordering::SeqCst);
    notify().notify_waiters();
}

/// Starts listening for interrupt and termination.
///
/// Installs once; later calls do nothing. Both signals mean the same thing
/// here, which is to stop accepting and finish what is in flight, and both are
/// handled identically: a server that drains on one and dies on the other has
/// two shutdown behaviours and an operator has to remember which is which.
pub fn install() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    if INSTALLED.set(()).is_err() {
        return;
    }
    tokio::spawn(async {
        listen().await;
        tracing::info!("shutdown requested");
        request();
    });
}

/// Resolves on the first interrupt or termination signal.
#[cfg(unix)]
async fn listen() {
    use tokio::signal::unix::{SignalKind, signal};
    let mut terminate = match signal(SignalKind::terminate()) {
        Ok(stream) => stream,
        Err(error) => {
            tracing::warn!(%error, "SIGTERM could not be listened for; interrupt still works");
            let _ = tokio::signal::ctrl_c().await;
            return;
        }
    };
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {}
        _ = terminate.recv() => {}
    }
}

/// Resolves on the first interrupt.
#[cfg(not(unix))]
async fn listen() {
    let _ = tokio::signal::ctrl_c().await;
}

/// Resolves once shutdown has been asked for.
///
/// Safe to call any number of times and from any number of tasks; a caller that
/// arrives after the signal returns immediately.
pub async fn wait() {
    // Registration first, then the re-read. See the module docs for what the
    // other order loses.
    let notified = notify().notified();
    if is_requested() {
        return;
    }
    notified.await;
}
