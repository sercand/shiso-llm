//! Where the coordinator reads the time.
//!
//! # Why the loop does not call `Instant::now`
//!
//! Two of the scheduler's decisions are decisions about elapsed time: a request
//! that outlived its deadline is cut, and the accounting a turn reports is the
//! difference between two readings. A loop that took those readings from the
//! clock on the wall can only be tested by sleeping, and a test that sleeps is
//! a test that is either slow or flaky depending on how long it sleeps.
//!
//! So every reading goes through this trait. In a server it is [`SystemClock`]
//! and the readings are real; in a test it is [`FakeClock`] and the test moves
//! time itself, which is what lets a deadline case run in microseconds and
//! never disagree with itself on a loaded machine.
//!
//! # Why the fake is here and not in the test directory
//!
//! Three integration tests need it and none of them can see into another. A
//! fake duplicated three times is three fakes that can drift apart, and the one
//! that drifts is the one whose test then passes for the wrong reason.

use std::fmt;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// The reading the coordinator makes decisions from.
///
/// `Send + Sync` because the handle that holds one is cloned into the engine
/// thread, and `Debug` because it is a field of [`super::SchedulerConfig`],
/// which is logged at boot.
pub trait Clock: Send + Sync + fmt::Debug {
    /// The current instant, as this clock understands the word.
    fn now(&self) -> Instant;
}

/// The clock on the wall.
#[derive(Debug, Clone, Copy, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

/// A clock that only moves when a test moves it.
///
/// Readings are a fixed base plus an offset, so two readings taken without an
/// intervening [`FakeClock::advance`] are equal - which is the property that
/// makes a deadline test assert on the policy rather than on how fast the
/// machine ran the loop.
#[derive(Debug)]
pub struct FakeClock {
    base: Instant,
    offset: Mutex<Duration>,
}

impl FakeClock {
    /// A clock stopped at the moment it was built.
    #[must_use]
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            base: Instant::now(),
            offset: Mutex::new(Duration::ZERO),
        })
    }

    /// Moves every future reading forward by `by`.
    pub fn advance(&self, by: Duration) {
        let mut offset = self.offset.lock();
        *offset = offset.saturating_add(by);
    }
}

impl Clock for FakeClock {
    fn now(&self) -> Instant {
        self.base + *self.offset.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::{Clock as _, FakeClock, SystemClock};
    use std::time::Duration;

    #[test]
    fn a_stopped_clock_reads_the_same_instant_until_it_is_moved() {
        let clock = FakeClock::new();
        let first = clock.now();
        assert_eq!(clock.now(), first, "nothing moved, so nothing elapsed");
        clock.advance(Duration::from_secs(5));
        assert_eq!(clock.now() - first, Duration::from_secs(5));
    }

    /// The real clock is monotonic, which is the one property the deadline
    /// arithmetic below it assumes.
    #[test]
    fn the_system_clock_never_goes_backwards() {
        let clock = SystemClock;
        let first = clock.now();
        assert!(clock.now() >= first);
    }
}
