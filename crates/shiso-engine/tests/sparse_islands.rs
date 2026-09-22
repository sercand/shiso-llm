//! The three-state protocol an island runs under, and what happens when the
//! third state is reached the hard way.
//!
//! Capturing a device graph is an optimization; losing an island is a wrong
//! answer. So the property worth proving is not that a capture succeeds - it is
//! that every way a capture can fail still leaves the island encoded exactly
//! once. This binary drives a device that fails on command and counts.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use parking_lot::Mutex;

use shiso_core::backend::{GraphBackend, GraphCapture, GraphKey};
use shiso_core::error::{CoreError, CoreResult};
use shiso_engine::sparse::island::{IslandRun, run};

/// A device whose graph tier fails wherever a test tells it to.
#[derive(Debug, Default)]
struct FakeGraphs {
    /// Keys a replay would find.
    registered: Mutex<Vec<GraphKey>>,
    /// Whether entering capture mode fails.
    refuse_capture: AtomicBool,
    /// Whether committing a capture fails.
    refuse_finish: AtomicBool,
    captures: AtomicUsize,
    aborts: AtomicUsize,
    /// Replays that actually found a graph and ran it.
    replays: AtomicUsize,
}

impl FakeGraphs {
    fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    fn register(&self, key: GraphKey) {
        self.registered.lock().push(key);
    }

    fn fail(op: &'static str) -> CoreError {
        CoreError::Backend {
            backend: "fake",
            op,
            source: Box::new(std::io::Error::other("the test asked for this")),
        }
    }
}

impl GraphBackend for FakeGraphs {
    fn capture(&self, key: GraphKey) -> CoreResult<GraphCapture<'_>> {
        if self.refuse_capture.load(Ordering::Relaxed) {
            return Err(Self::fail("capture"));
        }
        self.captures.fetch_add(1, Ordering::Relaxed);
        Ok(GraphCapture::started(self, key))
    }

    fn finish_capture(&self, key: GraphKey) -> CoreResult<()> {
        if self.refuse_finish.load(Ordering::Relaxed) {
            return Err(Self::fail("finish_capture"));
        }
        self.register(key);
        Ok(())
    }

    fn abort_capture(&self, _key: GraphKey) -> CoreResult<()> {
        self.aborts.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn replay(&self, key: GraphKey) -> CoreResult<bool> {
        let found = self.registered.lock().contains(&key);
        if found {
            self.replays.fetch_add(1, Ordering::Relaxed);
        }
        Ok(found)
    }

    fn invalidate_all(&self) {
        self.registered.lock().clear();
    }
}

const KEY: GraphKey = GraphKey::new(7);

#[test]
fn a_device_with_no_graph_tier_encodes_the_island_and_says_so() {
    let mut encodes = 0;
    let outcome = run(None, Some(KEY), || {
        encodes += 1;
        Ok(())
    })
    .expect("an eager island runs");
    assert_eq!(outcome, IslandRun::Eager);
    assert_eq!(encodes, 1);
}

#[test]
fn an_island_with_no_slot_is_encoded_every_time_on_a_device_that_could_capture() {
    // A batched island has no slot: the pointers a capture would bake in belong
    // to the exact ordered set of sessions it was encoded for, and no field of
    // the key can name a set. So the device's graph tier is present, willing and
    // never asked - and "never asked" is the part worth a test, because a `run`
    // that filled the missing key in with a default would capture the first
    // batch and replay it for every batch after it.
    let graphs = FakeGraphs::new();
    let tier: &dyn GraphBackend = graphs.as_ref();
    let mut encodes = 0;
    for _ in 0..3 {
        let outcome = run(Some(tier), None, || {
            encodes += 1;
            Ok(())
        })
        .expect("an island with no slot runs");
        assert_eq!(outcome, IslandRun::Eager);
    }
    assert_eq!(encodes, 3, "an island with no slot is encoded every time");
    assert_eq!(
        graphs.captures.load(Ordering::Relaxed),
        0,
        "the device was asked to capture an island nothing can replay"
    );
    assert!(graphs.registered.lock().is_empty());
}

#[test]
fn the_first_pass_captures_and_the_second_replays_without_encoding() {
    let graphs = FakeGraphs::new();
    let tier: &dyn GraphBackend = graphs.as_ref();
    let mut encodes = 0;

    let first = run(Some(tier), Some(KEY), || {
        encodes += 1;
        Ok(())
    })
    .expect("the first pass captures");
    assert_eq!(first, IslandRun::Captured);
    assert_eq!(encodes, 1);
    // A capture records launches; it does not run them. So the pass that
    // captured an island has to launch what it just recorded, or that pass did
    // none of the island's work - which is not a failure anywhere, only a wrong
    // answer at the first row count each session reaches.
    assert_eq!(
        graphs.replays.load(Ordering::Relaxed),
        1,
        "the capturing pass launched the graph it had just committed"
    );

    let second = run(Some(tier), Some(KEY), || {
        encodes += 1;
        Ok(())
    })
    .expect("the second pass replays");
    assert_eq!(second, IslandRun::Replayed);
    assert_eq!(encodes, 1, "a replay encodes nothing");
    assert_eq!(graphs.replays.load(Ordering::Relaxed), 2);
}

#[test]
fn a_device_that_refuses_capture_mode_still_runs_the_island() {
    let graphs = FakeGraphs::new();
    graphs.refuse_capture.store(true, Ordering::Relaxed);
    let tier: &dyn GraphBackend = graphs.as_ref();
    let mut encodes = 0;

    let outcome = run(Some(tier), Some(KEY), || {
        encodes += 1;
        Ok(())
    })
    .expect("a refused capture is not a refused island");
    assert_eq!(outcome, IslandRun::Eager);
    assert_eq!(encodes, 1);
}

#[test]
fn an_encoding_that_fails_inside_a_capture_aborts_and_is_encoded_again() {
    let graphs = FakeGraphs::new();
    let tier: &dyn GraphBackend = graphs.as_ref();
    let mut encodes = 0;

    let outcome = run(Some(tier), Some(KEY), || {
        encodes += 1;
        // The first attempt fails inside the capture; the fallback succeeds.
        if encodes == 1 {
            return Err(CoreError::Unsupported {
                what: "a launch the test refused",
                backend: "fake",
            });
        }
        Ok(())
    })
    .expect("the island still runs");
    assert_eq!(outcome, IslandRun::Eager);
    assert_eq!(
        encodes, 2,
        "the island is encoded again outside the capture"
    );
    assert_eq!(
        graphs.aborts.load(Ordering::Relaxed),
        1,
        "the device is taken out of capture mode before the retry"
    );
}

#[test]
fn a_capture_that_cannot_be_committed_is_encoded_again_rather_than_lost() {
    let graphs = FakeGraphs::new();
    graphs.refuse_finish.store(true, Ordering::Relaxed);
    let tier: &dyn GraphBackend = graphs.as_ref();
    let mut encodes = 0;

    let outcome = run(Some(tier), Some(KEY), || {
        encodes += 1;
        Ok(())
    })
    .expect("the island still runs");
    assert_eq!(outcome, IslandRun::Eager);
    // Everything encoded into a capture that was not committed went with it, so
    // the island has to be encoded a second time.
    assert_eq!(encodes, 2);
}

#[test]
fn an_island_that_cannot_run_at_all_reports_rather_than_looping() {
    let graphs = FakeGraphs::new();
    let tier: &dyn GraphBackend = graphs.as_ref();
    let mut encodes = 0;

    let err = run(Some(tier), Some(KEY), || {
        encodes += 1;
        Err(CoreError::Unsupported {
            what: "a launch the test always refuses",
            backend: "fake",
        })
    })
    .expect_err("a launch that never succeeds is a failure");
    assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
    assert_eq!(encodes, 2, "one attempt inside the capture and one outside");
}

#[test]
fn invalidating_the_cache_sends_the_next_pass_back_through_a_capture() {
    // This is what a grown page table does: a captured launch baked the table's
    // address in, so every graph standing on the old one has to go.
    let graphs = FakeGraphs::new();
    let tier: &dyn GraphBackend = graphs.as_ref();
    assert_eq!(
        run(Some(tier), Some(KEY), || Ok(())).expect("captures"),
        IslandRun::Captured
    );
    assert_eq!(
        run(Some(tier), Some(KEY), || Ok(())).expect("replays"),
        IslandRun::Replayed
    );
    graphs.invalidate_all();
    assert_eq!(
        run(Some(tier), Some(KEY), || Ok(())).expect("captures again"),
        IslandRun::Captured
    );
}
