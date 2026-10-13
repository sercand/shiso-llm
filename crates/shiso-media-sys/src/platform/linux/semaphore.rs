//! A counting semaphore that caps concurrent decoder sessions.
//!
//! A GB10 has one NVDEC engine and sessions time-share it; past a handful of
//! sessions the extra ones only add surface memory and context switches without
//! adding throughput, and each holds device memory for its decode surfaces. So
//! the process opens at most a fixed number and an extra `open` waits for one to
//! close rather than failing: a request that waits is better than one refused
//! for a reason the client cannot act on.

use std::sync::{Condvar, Mutex, PoisonError};

/// A fixed number of permits, handed out by [`Semaphore::acquire`].
#[derive(Debug)]
pub(super) struct Semaphore {
    capacity: usize,
    in_use: Mutex<usize>,
    released: Condvar,
}

impl Semaphore {
    /// A semaphore with `capacity` permits. `const` so the process-wide one can
    /// be a plain `static`.
    pub(super) const fn new(capacity: usize) -> Self {
        Self {
            capacity,
            in_use: Mutex::new(0),
            released: Condvar::new(),
        }
    }

    /// Waits until a permit is free and takes it. The permit is returned when
    /// the guard drops.
    pub(super) fn acquire(&self) -> Permit<'_> {
        // A poisoned lock means a thread panicked while holding it; the count
        // itself is only ever changed by one statement, so it is still right.
        let mut in_use = self.in_use.lock().unwrap_or_else(PoisonError::into_inner);
        while *in_use >= self.capacity {
            in_use = self
                .released
                .wait(in_use)
                .unwrap_or_else(PoisonError::into_inner);
        }
        *in_use += 1;
        Permit { semaphore: self }
    }

    /// Permits currently held.
    #[cfg(test)]
    fn in_use(&self) -> usize {
        *self.in_use.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One held permit; dropping it frees the slot and wakes one waiter.
#[derive(Debug)]
pub(super) struct Permit<'a> {
    semaphore: &'a Semaphore,
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        let mut in_use = self
            .semaphore
            .in_use
            .lock()
            .unwrap_or_else(PoisonError::into_inner);
        *in_use = in_use.saturating_sub(1);
        drop(in_use);
        self.semaphore.released.notify_one();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn permits_up_to_capacity_are_granted_at_once() {
        let sem = Semaphore::new(3);
        let held: Vec<_> = (0..3).map(|_| sem.acquire()).collect();
        assert_eq!(sem.in_use(), 3);
        drop(held);
        assert_eq!(sem.in_use(), 0);
    }

    #[test]
    fn an_extra_acquire_waits_for_a_release() {
        let sem = Arc::new(Semaphore::new(2));
        let a = sem.acquire();
        let _b = sem.acquire();
        let (tx, rx) = mpsc::channel();
        let waiter = {
            let sem = Arc::clone(&sem);
            thread::spawn(move || {
                let _c = sem.acquire();
                tx.send(()).expect("the test is listening");
            })
        };
        // The third acquire must not get through while both permits are held.
        assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());
        drop(a);
        rx.recv_timeout(Duration::from_secs(10))
            .expect("the waiter gets the released permit");
        waiter.join().expect("the waiter does not panic");
        // The waiter's permit was dropped when its thread ended.
        assert_eq!(sem.in_use(), 1);
    }

    #[test]
    fn concurrency_never_exceeds_capacity() {
        let sem = Arc::new(Semaphore::new(4));
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let threads: Vec<_> = (0..16)
            .map(|_| {
                let (sem, live, peak) = (Arc::clone(&sem), Arc::clone(&live), Arc::clone(&peak));
                thread::spawn(move || {
                    for _ in 0..20 {
                        let _permit = sem.acquire();
                        let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                        peak.fetch_max(now, Ordering::SeqCst);
                        thread::yield_now();
                        live.fetch_sub(1, Ordering::SeqCst);
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().expect("no worker panics");
        }
        assert!(peak.load(Ordering::SeqCst) <= 4);
        assert_eq!(sem.in_use(), 0);
    }
}
