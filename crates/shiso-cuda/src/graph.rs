//! Graph capture, replay, and the cache of captured graphs.
//!
//! # What a captured graph is worth, and what it costs
//!
//! A decode step is hundreds of small launches whose per-launch host cost is a
//! large fraction of the step. Capturing that sequence once and replaying it as
//! a single graph removes almost all of it. The price is that a graph bakes in
//! the addresses it was captured with: nothing updates them afterwards, so
//! every buffer the captured region touches must still be allocated at the same
//! address when the graph replays. That is the whole reason
//! [`shiso_core::GraphBackend::invalidate_all`] exists, and the reason allocation
//! is refused while a capture is running.
//!
//! # The guard, not a begin/end pair
//!
//! Capture is expressed as a guard the core crate owns, and that is
//! load-bearing rather than stylistic. Between "begin capture" and "end
//! capture" the caller runs arbitrary work, and that work can fail. With a
//! begin/end pair, one `?` in the middle leaves the stream in capture mode
//! forever: every later launch on it disappears into a capture nobody will ever
//! end, and the failure surfaces somewhere else entirely as a stream that
//! stopped executing. The guard's `Drop` aborts instead, so the `?` that would
//! have stranded the device now merely fails the request.
//!
//! # What this cache decides, and what it leaves to its caller
//!
//! It owns the *resource* rules: one capture at a time, a bounded number of
//! live graphs, and a slot that failed to capture is not retried. It does not
//! own the *policy* of when to run eagerly, when to capture and when to replay
//! — that depends on what a caller knows about its own workload, and it is
//! expressed by which of these methods the caller calls.

use std::collections::{HashMap, HashSet};

use shiso_core::{CoreResult, GraphCapture, GraphKey};

use crate::CudaBackend;
use crate::device::DeviceCtx;
use crate::sys::nvidia;
use crate::sys::{CuGraphExec, CuStream, host_error};

/// Largest number of captured graphs kept at once.
///
/// A caller keys its graphs by whatever identity its own work has; this bound
/// exists so that a caller whose identity turns out to be unbounded — a key
/// that accidentally includes a token count, say — runs out of graph slots
/// loudly instead of consuming device memory until something else fails.
pub(crate) const GRAPH_CACHE_CAP: usize = 4096;

/// Captured graphs, and the one capture that may be in flight.
#[derive(Debug, Default)]
pub(crate) struct GraphCache {
    /// The slot being captured into, when a capture is running.
    capturing: Option<GraphKey>,
    /// Instantiated graphs, ready to replay.
    execs: HashMap<GraphKey, CuGraphExec>,
    /// Slots whose capture failed. A failed capture is not a transient
    /// condition — it means the work in that region cannot be expressed as a
    /// graph — so retrying it every step would spend the whole saving on
    /// failures. The caller learns this as a refusal and falls back.
    dead: HashSet<GraphKey>,
    /// Whether the "no free graph slots" warning has already been emitted.
    warned_full: bool,
    /// Whether launches go to the branch stream a fork opened.
    branch: bool,
}

impl GraphCache {
    /// Whether a capture is in flight.
    pub(crate) const fn is_capturing(&self) -> bool {
        self.capturing.is_some()
    }

    /// The stream work should be submitted to right now.
    ///
    /// This is the entire mechanism by which a launch ends up inside a capture:
    /// ordinary work goes to the work stream, and while a capture is running it
    /// goes to the capture stream instead. Nothing else in the launch path
    /// knows that graphs exist.
    ///
    /// Between a fork and the resume that ends its branch, work goes to the
    /// branch stream beside whichever of the two is current.
    pub(crate) const fn stream(&self, device: &DeviceCtx) -> CuStream {
        match (self.capturing.is_some(), self.branch) {
            (true, false) => device.capture_stream(),
            (true, true) => device.side_capture_stream(),
            (false, false) => device.work_stream(),
            (false, true) => device.side_work_stream(),
        }
    }

    /// The main stream, whatever branch is open.
    const fn main_stream(&self, device: &DeviceCtx) -> CuStream {
        if self.capturing.is_some() {
            device.capture_stream()
        } else {
            device.work_stream()
        }
    }

    /// The branch stream beside the main one.
    const fn side_stream(&self, device: &DeviceCtx) -> CuStream {
        if self.capturing.is_some() {
            device.side_capture_stream()
        } else {
            device.side_work_stream()
        }
    }

    /// Number of captured graphs held.
    pub(crate) fn len(&self) -> usize {
        self.execs.len()
    }

    /// Decides whether a capture into `key` may start.
    ///
    /// Pure, and separate from starting one, so that the driver is never asked
    /// to enter capture mode for a request this cache was going to refuse — and
    /// so that the refusals can be tested without a device.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when a capture is already running,
    /// when this slot has already failed once, or when every slot is taken.
    fn admit(&mut self, key: GraphKey) -> CoreResult<()> {
        if let Some(running) = self.capturing {
            return Err(host_error(
                "graph_capture",
                format!(
                    "a capture into slot {} is already running; captures do not nest",
                    running.get()
                ),
            ));
        }
        if self.dead.contains(&key) {
            return Err(host_error(
                "graph_capture",
                format!(
                    "slot {} failed to capture once and is not retried",
                    key.get()
                ),
            ));
        }
        if !self.execs.contains_key(&key) && self.execs.len() >= GRAPH_CACHE_CAP {
            if !self.warned_full {
                self.warned_full = true;
                tracing::warn!(
                    capacity = GRAPH_CACHE_CAP,
                    "every graph slot is taken; further captures are refused",
                );
            }
            return Err(host_error(
                "graph_capture",
                format!("all {GRAPH_CACHE_CAP} graph slots are in use"),
            ));
        }
        Ok(())
    }

    /// Records that a capture into `key` has started.
    fn arm(&mut self, key: GraphKey) {
        self.capturing = Some(key);
    }

    /// Records that no capture is running, whatever the outcome was.
    fn disarm(&mut self) {
        self.capturing = None;
    }

    /// Checks that the running capture is the one `key` names.
    ///
    /// # Errors
    ///
    /// [`shiso_core::CoreError::Backend`] when no capture is running, or when a
    /// different slot is being captured.
    fn expect_capturing(&self, op: &'static str, key: GraphKey) -> CoreResult<()> {
        match self.capturing {
            Some(running) if running == key => Ok(()),
            Some(running) => Err(host_error(
                op,
                format!(
                    "slot {} is being captured, not slot {}",
                    running.get(),
                    key.get()
                ),
            )),
            None => Err(host_error(
                op,
                format!(
                    "no capture is running, so slot {} cannot end one",
                    key.get()
                ),
            )),
        }
    }

    /// Installs a freshly instantiated graph, replacing any earlier one.
    ///
    /// Returns the replaced graph so the caller can destroy it outside the
    /// lock's critical section of driver work.
    fn install(&mut self, key: GraphKey, exec: CuGraphExec) -> Option<CuGraphExec> {
        self.execs.insert(key, exec)
    }

    /// Marks a slot as one that will not be captured again.
    fn mark_dead(&mut self, key: GraphKey) {
        self.dead.insert(key);
    }

    /// The graph registered for `key`, if any.
    fn get(&self, key: GraphKey) -> Option<CuGraphExec> {
        self.execs.get(&key).copied()
    }

    /// Empties the cache, yielding every graph that has to be destroyed.
    fn take_all(&mut self) -> Vec<CuGraphExec> {
        self.dead.clear();
        self.warned_full = false;
        self.execs.drain().map(|(_, exec)| exec).collect()
    }
}

impl Drop for GraphCache {
    /// Destroys every graph the cache still holds.
    ///
    /// The cache is dropped before the context it was captured against, which
    /// field order in the backend guarantees, so the handles are still valid
    /// here.
    fn drop(&mut self) {
        for exec in self.take_all() {
            // SAFETY: each handle came out of the map, which held it exactly
            // once and has now given up its only copy.
            if let Err(err) = unsafe { nvidia::graph_exec_destroy(exec) }
                && !err.is_teardown_noop()
            {
                tracing::error!(error = %err, "graph destroy failed during teardown");
            }
        }
    }
}

/// Puts the capture stream into capture mode and hands back the guard.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when a capture is already running, when
/// this slot has already failed, when every slot is taken, or when the driver
/// refuses to enter capture mode.
pub(crate) fn capture(backend: &CudaBackend, key: GraphKey) -> CoreResult<GraphCapture<'_>> {
    backend.device.bind()?;
    let stream = backend.device.capture_stream();
    let mut state = backend.state.lock();
    state.graphs.admit(key)?;
    // The matrix-multiply library records its work on whichever stream it is
    // bound to. Unbound, its calls would land on the work stream while
    // everything around them landed in the capture, which the strict capture
    // mode reports as an invalidated capture rather than silently dropping
    // them — but only after the fact. Binding it first is what keeps a captured
    // region that contains a matrix multiply capturable at all.
    backend.blas.bind_stream(stream)?;
    // SAFETY: the capture stream is owned by `backend.device` and `admit`
    // proved no capture is already running on it.
    if let Err(err) = unsafe { nvidia::stream_begin_capture(stream) } {
        // Put the library back on the work stream: the capture never started,
        // so nothing else will.
        if let Err(rebind) = backend.blas.bind_stream(backend.device.work_stream()) {
            tracing::error!(error = %rebind, "could not rebind the matrix library after a failed capture");
        }
        return Err(err.into());
    }
    state.graphs.arm(key);
    drop(state);
    tracing::debug!(slot = key.get(), "graph capture started");
    Ok(GraphCapture::started(backend, key))
}

/// Ends the running capture and installs what it produced.
///
/// Any failure between here and an installed graph marks the slot dead: the
/// work in that region could not be expressed as a graph, and that is a
/// property of the work rather than a transient condition, so retrying it every
/// step would turn a saving into a cost.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when no capture is running or a different
/// slot is, and when the driver reports an invalidated capture or refuses to
/// instantiate it.
pub(crate) fn finish_capture(backend: &CudaBackend, key: GraphKey) -> CoreResult<()> {
    backend.device.bind()?;
    let stream = backend.device.capture_stream();
    let mut state = backend.state.lock();
    state.graphs.expect_capturing("graph_finish_capture", key)?;

    // SAFETY: `expect_capturing` proved this stream is the capturing one.
    let ended = unsafe { nvidia::stream_end_capture(stream) };
    state.graphs.disarm();
    let graph = match ended {
        Ok(graph) => graph,
        Err(err) => {
            state.graphs.mark_dead(key);
            drop(state);
            rebind_work_stream(backend);
            tracing::error!(slot = key.get(), error = %err, "graph capture was invalidated");
            return Err(err.into());
        }
    };

    // SAFETY: the graph came from the capture that just ended and is owned here.
    let instantiated = unsafe { nvidia::graph_instantiate(graph) };
    // The instantiated copy does not borrow from the captured graph, so the
    // captured one is released either way.
    // SAFETY: as above; this is the only owner and the only destroy.
    if let Err(err) = unsafe { nvidia::graph_destroy(graph) } {
        tracing::error!(error = %err, "captured graph destroy failed");
    }
    let exec = match instantiated {
        Ok(exec) => exec,
        Err(err) => {
            state.graphs.mark_dead(key);
            drop(state);
            rebind_work_stream(backend);
            return Err(err.into());
        }
    };

    let replaced = state.graphs.install(key, exec);
    let held = state.graphs.len();
    drop(state);
    if let Some(old) = replaced {
        // SAFETY: the map has given up its only copy of the replaced handle.
        if let Err(err) = unsafe { nvidia::graph_exec_destroy(old) } {
            tracing::error!(error = %err, "replaced graph destroy failed");
        }
    }
    rebind_work_stream(backend);
    tracing::debug!(slot = key.get(), held, "graph captured and installed");
    Ok(())
}

/// Abandons the running capture, leaving the stream usable.
///
/// This runs from the capture guard's `Drop`, so it has to succeed at the one
/// thing that matters — taking the stream out of capture mode — even when the
/// capture itself was already invalidated. The driver reports that invalidation
/// through the same call that ends the capture, so the error is logged and the
/// stream is still left clean.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when no capture is running or a different
/// slot is. A driver failure while ending an already-doomed capture is logged
/// rather than returned: there is nothing a caller could do with it, and the
/// stream is clean either way.
pub(crate) fn abort_capture(backend: &CudaBackend, key: GraphKey) -> CoreResult<()> {
    backend.device.bind()?;
    let stream = backend.device.capture_stream();
    let mut state = backend.state.lock();
    state.graphs.expect_capturing("graph_abort_capture", key)?;
    // SAFETY: `expect_capturing` proved this stream is the capturing one.
    let ended = unsafe { nvidia::stream_end_capture(stream) };
    state.graphs.disarm();
    drop(state);
    match ended {
        Ok(graph) => {
            // SAFETY: the graph came from the capture that just ended and has
            // no other owner.
            if let Err(err) = unsafe { nvidia::graph_destroy(graph) } {
                tracing::error!(error = %err, "abandoned graph destroy failed");
            }
        }
        Err(err) => {
            tracing::debug!(slot = key.get(), error = %err, "capture abandoned");
        }
    }
    rebind_work_stream(backend);
    Ok(())
}

/// Replays the graph registered for `key`.
///
/// `Ok(false)` means nothing is registered, which is a question a caller is
/// expected to ask rather than an error: "is there a graph for this yet" is how
/// a caller decides whether to capture one.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when a capture is running — a graph is
/// not a nestable unit here — or when the replay cannot be queued.
pub(crate) fn replay(backend: &CudaBackend, key: GraphKey) -> CoreResult<bool> {
    backend.device.bind()?;
    let state = backend.state.lock();
    if state.graphs.is_capturing() {
        return Err(host_error(
            "graph_replay",
            "a graph cannot be replayed while a capture is running",
        ));
    }
    let Some(exec) = state.graphs.get(key) else {
        return Ok(false);
    };
    // SAFETY: the handle is owned by the cache, which is still locked, so it
    // cannot be destroyed under the launch. Every buffer the graph names is
    // still allocated, because allocation is refused during capture and this
    // backend's caller invalidates the cache when it moves a buffer.
    unsafe { nvidia::graph_launch(exec, backend.device.work_stream()) }?;
    Ok(true)
}

/// Destroys every captured graph.
///
/// A caller runs this whenever a buffer a graph might name has moved, because a
/// graph replays into the addresses it was captured with and nothing updates
/// them. It also clears the record of which slots failed, so a slot that could
/// not be captured against one memory layout gets a fresh chance against the
/// next one.
pub(crate) fn invalidate_all(backend: &CudaBackend) {
    if let Err(err) = backend.device.bind() {
        tracing::error!(error = %err, "could not bind the context to invalidate graphs");
        return;
    }
    let mut state = backend.state.lock();
    if state.graphs.is_capturing() {
        tracing::warn!("graphs invalidated while a capture is running; the capture will abort");
    }
    let execs = state.graphs.take_all();
    drop(state);
    let count = execs.len();
    for exec in execs {
        // SAFETY: each handle came out of the map, which held it exactly once
        // and has now given up its only copy.
        if let Err(err) = unsafe { nvidia::graph_exec_destroy(exec) } {
            tracing::error!(error = %err, "graph destroy failed");
        }
    }
    if count > 0 {
        tracing::debug!(count, "captured graphs invalidated");
    }
}

/// Puts the matrix library back on the work stream after a capture.
fn rebind_work_stream(backend: &CudaBackend) {
    if let Err(err) = backend.blas.bind_stream(backend.device.work_stream()) {
        tracing::error!(error = %err, "could not rebind the matrix library to the work stream");
    }
}

/// Opens a branch: the branch stream waits for everything queued on the main
/// stream so far, and launches go to it until [`resume`].
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when a branch is already open, and when
/// the driver refuses the record or the wait.
pub(crate) fn fork(backend: &CudaBackend) -> CoreResult<()> {
    backend.device.bind()?;
    let mut state = backend.state.lock();
    if state.graphs.branch {
        return Err(host_error("fork", "a fork inside an open branch"));
    }
    let main = state.graphs.main_stream(&backend.device);
    let side = state.graphs.side_stream(&backend.device);
    let event = backend.device.fork_event();
    // SAFETY: both streams and the event are owned by `backend.device` and
    // live for as long as it does.
    unsafe { nvidia::event_record(event, main) }?;
    // SAFETY: as above.
    unsafe { nvidia::stream_wait_event(side, event) }?;
    state.graphs.branch = true;
    Ok(())
}

/// Ends a branch's launches: the branch stream marks where it got to, and
/// launches go back to the main stream, which runs on beside it until
/// [`join`].
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when no branch is open, and when the
/// driver refuses the record.
pub(crate) fn resume(backend: &CudaBackend) -> CoreResult<()> {
    backend.device.bind()?;
    let mut state = backend.state.lock();
    if !state.graphs.branch {
        return Err(host_error("resume", "a resume with no branch open"));
    }
    let side = state.graphs.side_stream(&backend.device);
    // SAFETY: the stream and the event are owned by `backend.device`.
    unsafe { nvidia::event_record(backend.device.join_event(), side) }?;
    state.graphs.branch = false;
    Ok(())
}

/// Makes the main stream wait for the branch that last resumed.
///
/// # Errors
///
/// [`shiso_core::CoreError::Backend`] when a branch is still open, and when
/// the driver refuses the wait.
pub(crate) fn join(backend: &CudaBackend) -> CoreResult<()> {
    backend.device.bind()?;
    let state = backend.state.lock();
    if state.graphs.branch {
        return Err(host_error("join", "a join inside an open branch"));
    }
    let main = state.graphs.main_stream(&backend.device);
    // SAFETY: the stream and the event are owned by `backend.device`.
    unsafe { nvidia::stream_wait_event(main, backend.device.join_event()) }?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use shiso_core::{CoreError, GraphKey};

    use super::{GRAPH_CACHE_CAP, GraphCache};
    use crate::sys::CuGraphExec;

    /// A graph handle that no driver ever produced.
    ///
    /// Only the cache's own bookkeeping is under test here, and it never
    /// dereferences a handle — it only counts them and hands them back. The
    /// cache is leaked at the end of each test rather than dropped, because
    /// dropping it would try to destroy these through a driver that is not
    /// there.
    fn handle(n: usize) -> CuGraphExec {
        CuGraphExec::from_raw(n + 1)
    }

    /// Builds a cache and prevents its `Drop` from reaching the driver.
    fn cache() -> std::mem::ManuallyDrop<GraphCache> {
        std::mem::ManuallyDrop::new(GraphCache::default())
    }

    /// The slot `n` names, as a key.
    fn slot(n: usize) -> GraphKey {
        GraphKey::new(u64::try_from(n).expect("a test slot number fits a key"))
    }

    #[test]
    fn a_fresh_cache_holds_nothing_and_is_not_capturing() {
        let cache = cache();
        assert!(!cache.is_capturing());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn captures_do_not_nest() {
        let mut cache = cache();
        let first = GraphKey::new(1);
        cache.admit(first).expect("the first capture is admitted");
        cache.arm(first);
        match cache.admit(GraphKey::new(2)) {
            Err(CoreError::Backend { source, .. }) => {
                let text = source.to_string();
                assert!(text.contains("do not nest"), "{text}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        cache.disarm();
        cache
            .admit(GraphKey::new(2))
            .expect("once the first capture ends another may start");
    }

    #[test]
    fn a_slot_whose_capture_failed_is_never_retried() {
        let mut cache = cache();
        let key = GraphKey::new(7);
        cache.admit(key).expect("the first attempt is admitted");
        cache.arm(key);
        cache.disarm();
        cache.mark_dead(key);
        match cache.admit(key) {
            Err(CoreError::Backend { source, .. }) => {
                let text = source.to_string();
                assert!(text.contains("not retried"), "{text}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        cache
            .admit(GraphKey::new(8))
            .expect("one dead slot must not poison the others");
    }

    #[test]
    fn a_full_cache_refuses_new_slots_but_still_accepts_a_recapture() {
        let mut cache = cache();
        for n in 0..GRAPH_CACHE_CAP {
            cache.install(slot(n), handle(n));
        }
        assert_eq!(cache.len(), GRAPH_CACHE_CAP);
        assert!(
            cache.admit(slot(GRAPH_CACHE_CAP)).is_err(),
            "a full cache has no room for a slot it does not already hold",
        );
        cache
            .admit(GraphKey::new(0))
            .expect("re-capturing a slot the cache already holds needs no new room");
        let _ = cache.take_all();
    }

    #[test]
    fn a_full_cache_warns_exactly_once() {
        let mut cache = cache();
        for n in 0..GRAPH_CACHE_CAP {
            cache.install(slot(n), handle(n));
        }
        let overflow = slot(GRAPH_CACHE_CAP + 1);
        assert!(cache.admit(overflow).is_err());
        assert!(cache.warned_full, "the first refusal arms the warning");
        assert!(cache.admit(overflow).is_err());
        assert!(
            cache.warned_full,
            "the flag stays set, so the warning is not repeated every step",
        );
        let _ = cache.take_all();
    }

    #[test]
    fn installing_over_a_slot_hands_back_the_graph_it_replaced() {
        let mut cache = cache();
        let key = GraphKey::new(3);
        assert!(
            cache.install(key, handle(1)).is_none(),
            "the first install replaces nothing",
        );
        assert_eq!(
            cache.install(key, handle(2)),
            Some(handle(1)),
            "a replaced graph has to be handed back so it can be destroyed",
        );
        assert_eq!(cache.len(), 1);
        let _ = cache.take_all();
    }

    #[test]
    fn ending_a_capture_that_is_not_running_names_what_is() {
        let mut cache = cache();
        let key = GraphKey::new(5);
        match cache.expect_capturing("graph_finish_capture", key) {
            Err(CoreError::Backend { op, source, .. }) => {
                assert_eq!(op, "graph_finish_capture");
                assert!(source.to_string().contains("no capture is running"));
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        cache.arm(GraphKey::new(6));
        match cache.expect_capturing("graph_finish_capture", key) {
            Err(CoreError::Backend { source, .. }) => {
                let text = source.to_string();
                assert!(text.contains("slot 6"), "{text}");
                assert!(text.contains("slot 5"), "{text}");
            }
            other => panic!("expected a refusal, got {other:?}"),
        }
        cache
            .expect_capturing("graph_finish_capture", GraphKey::new(6))
            .expect("the running slot may end its own capture");
        cache.disarm();
    }

    #[test]
    fn invalidating_clears_the_graphs_and_the_record_of_failed_slots() {
        let mut cache = cache();
        let dead = GraphKey::new(9);
        cache.mark_dead(dead);
        cache.install(GraphKey::new(1), handle(1));
        cache.install(GraphKey::new(2), handle(2));

        let taken = cache.take_all();
        assert_eq!(taken.len(), 2, "every graph is handed back to be destroyed");
        assert_eq!(cache.len(), 0);
        cache.admit(dead).expect(
            "a new memory layout deserves a fresh attempt at a slot that failed under the old one",
        );
    }

    #[test]
    fn a_missing_slot_is_absent_rather_than_an_error() {
        let cache = cache();
        assert_eq!(cache.get(GraphKey::new(42)), None);
    }
}
