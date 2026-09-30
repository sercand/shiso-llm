//! Device-aware admission control: per-backend concurrency slots, and (from
//! stage 3) the exclusive-group lease that keeps two backends on one
//! accelerator from generating at the same time.
//!
//! Scheduler state deliberately lives here rather than in `state::Routing`.
//! `Routing` is rebuilt wholesale by `reload_routing` after *every* catalog
//! write, so leases and in-flight counters kept there would be dropped the
//! moment an admin edited an unrelated backend — wedging the group or
//! double-admitting. This is the same shape as `TokenCache` and
//! `UsageTracker`: long-lived runtime state keyed by name, *reconciled*
//! against a new routing snapshot instead of rebuilt from it.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, RwLock};
use std::time::Duration;

use tokio::sync::{OwnedSemaphorePermit, Semaphore, mpsc, oneshot};
use tokio::time::Instant;

use crate::balancer::Backend;
use crate::catalog::GroupDef;
use crate::state::Routing;

/// Throughput assumed for a backend that configures no `tok_s_estimate`
/// and has not been observed yet. Deliberately slow.
const FALLBACK_TOK_S: f64 = 10.0;

/// Observed generation rate for one backend, as a small ring of recent
/// per-request tokens/second.
///
/// The estimator reads a **low quantile**, not the mean. With speculative
/// decoding the acceptance rate is prompt-dependent, so throughput is
/// genuinely variable; planning from the mean systematically under-predicts
/// wait and admits requests that then miss the deadline they were promised.
/// Erring slow costs a shed that might have fit — recoverable — while
/// erring fast breaks the guarantee outright.
#[derive(Debug)]
pub struct Throughput {
    /// `tok_s_estimate` from the catalog, used until enough is observed.
    seed: RwLock<Option<f64>>,
    samples: std::sync::Mutex<VecDeque<f64>>,
}

impl Throughput {
    /// Enough recent requests to say something about the spread, small
    /// enough to follow a real change in load or prompt mix.
    const CAPACITY: usize = 32;
    /// Below this the ring says nothing useful and the seed is better.
    const MIN_SAMPLES: usize = 4;
    /// A pessimistic percentile, not the mean — see the type docs.
    const QUANTILE: f64 = 0.2;
    /// Requests too short or too fast to time meaningfully.
    const MIN_TOKENS: i64 = 16;
    const MIN_MS: i64 = 50;

    fn new(seed: Option<f64>) -> Self {
        Throughput {
            seed: RwLock::new(seed),
            samples: std::sync::Mutex::new(VecDeque::with_capacity(Self::CAPACITY)),
        }
    }

    /// Folds in one finished request. Ignores samples too small to time.
    pub fn observe(&self, completion_tokens: i64, elapsed_ms: i64) {
        if completion_tokens < Self::MIN_TOKENS || elapsed_ms < Self::MIN_MS {
            return;
        }
        let rate = completion_tokens as f64 / (elapsed_ms as f64 / 1000.0);
        if !rate.is_finite() || rate <= 0.0 {
            return;
        }
        let mut samples = self.samples.lock().unwrap();
        if samples.len() == Self::CAPACITY {
            samples.pop_front();
        }
        samples.push_back(rate);
    }

    /// Tokens/second to plan with.
    pub fn pessimistic(&self) -> f64 {
        let samples = self.samples.lock().unwrap();
        if samples.len() < Self::MIN_SAMPLES {
            return self.seed.read().unwrap().unwrap_or(FALLBACK_TOK_S);
        }
        let mut sorted: Vec<f64> = samples.iter().copied().collect();
        drop(samples);
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("rates are finite and positive"));
        let idx = ((sorted.len() - 1) as f64 * Self::QUANTILE).floor() as usize;
        sorted[idx]
    }
}

/// Per-backend admission state. `limit` is remembered so `reconcile` can
/// tell a tunable change from a no-op.
#[derive(Debug)]
pub struct BackendSlots {
    pub name: String,
    limit: Option<u32>,
    /// `None` for an unlimited backend — no semaphore at all, rather than a
    /// very large one, so the uncontended path stays a single atomic add.
    sem: Option<Arc<Semaphore>>,
    in_flight: AtomicU32,
    /// Shared across re-creations of this struct, so retuning a concurrency
    /// limit does not throw away what has been learned about the backend.
    throughput: Arc<Throughput>,
}

impl BackendSlots {
    fn new(name: String, limit: Option<u32>, throughput: Arc<Throughput>) -> Self {
        BackendSlots {
            name,
            limit,
            sem: limit.map(|n| Arc::new(Semaphore::new(n as usize))),
            in_flight: AtomicU32::new(0),
            throughput,
        }
    }

    pub fn throughput(&self) -> &Arc<Throughput> {
        &self.throughput
    }

    /// Requests currently holding a slot on this backend.
    pub fn in_flight(&self) -> u32 {
        self.in_flight.load(Ordering::Acquire)
    }

    pub fn limit(&self) -> Option<u32> {
        self.limit
    }

    /// Waits for a free slot and takes it. The returned guard releases on
    /// drop, so every exit path — early return, `?`, panic, task
    /// cancellation, a client vanishing mid-stream — gives the slot back.
    async fn acquire(self: &Arc<Self>) -> SlotGuard {
        let permit = match &self.sem {
            // The semaphore is never closed, so this cannot fail.
            Some(sem) => Some(
                Arc::clone(sem)
                    .acquire_owned()
                    .await
                    .expect("slot semaphore is never closed"),
            ),
            None => None,
        };
        self.in_flight.fetch_add(1, Ordering::AcqRel);
        SlotGuard {
            throughput: Arc::clone(&self.throughput),
            slots: Arc::clone(self),
            permit,
            lease: None,
            granted: std::time::Instant::now(),
        }
    }
}

/// Proof that a request holds a slot on a backend. Held for the whole life
/// of the response: the proxy moves it into `LogContext`, so `TeeStream`
/// owns it until the stream ends, errors, or the client disconnects. There
/// is deliberately no explicit release — `Drop` is the only path.
#[derive(Debug)]
pub struct SlotGuard {
    slots: Arc<BackendSlots>,
    /// `None` on an unlimited backend; the in-flight gauge is still kept.
    permit: Option<OwnedSemaphorePermit>,
    /// Present only for a `grouped` backend. Dropping it tells the group
    /// actor this request has left the device, which is what lets a drain
    /// finish — so it must not be released before the response is.
    lease: Option<GroupLease>,
    throughput: Arc<Throughput>,
    /// When the slot was taken — i.e. when the request actually reached the
    /// device, after any queueing. Timing from here rather than from the
    /// request's start keeps admission wait out of the throughput number.
    granted: std::time::Instant,
}

impl SlotGuard {
    pub fn backend(&self) -> &str {
        &self.slots.name
    }

    /// Folds a finished response into the backend's observed throughput.
    /// Called once from the tee, where the completion-token count and the
    /// elapsed time already exist.
    pub fn observe_completion(&self, completion_tokens: Option<i64>) {
        let Some(tokens) = completion_tokens else {
            return;
        };
        self.throughput
            .observe(tokens, self.granted.elapsed().as_millis() as i64);
    }
}

impl Drop for SlotGuard {
    fn drop(&mut self) {
        let prev = self.slots.in_flight.fetch_sub(1, Ordering::AcqRel);
        // A leaked or double-counted slot silently wedges an exclusive
        // group later, where it looks like a scheduler bug rather than the
        // bookkeeping bug it is. Catch it at the source in debug builds.
        debug_assert!(
            prev >= 1,
            "backend {:?}: in-flight counter underflowed",
            self.slots.name
        );
        drop(self.lease.take());
        drop(self.permit.take());
    }
}

// ---- exclusive groups ---------------------------------------------------

/// One `tokio` task owns all of a group's state; commands arrive over an
/// mpsc channel and lease grants go back over oneshots. Nothing about the
/// group sits behind a shared lock: the two-level structure (group lease,
/// then backend slot) invites lock-ordering bugs, and an actor makes an
/// epoch transition trivially atomic — it is just a field assignment
/// between two message handlings.
enum Cmd {
    Request(Waiter),
    Release {
        backend: String,
        ticket: u64,
    },
    /// Catalog edit: new timings and/or membership.
    Retune {
        def: GroupDef,
        members: HashMap<String, Option<u32>>,
    },
    Probe(oneshot::Sender<GroupSnapshot>),
}

struct Waiter {
    backend: String,
    enqueued: Instant,
    /// Upper bound on how long this request will occupy the device once
    /// granted. Pre-divided by the caller so the actor needs to know
    /// nothing about tokens or throughput.
    est_drain_ms: i64,
    reply: oneshot::Sender<GroupLease>,
}

struct InFlight {
    ticket: u64,
    granted: Instant,
    est_drain_ms: i64,
}

/// Which member currently owns the device.
enum Phase {
    /// Nobody wants it.
    Idle,
    /// `backend` may admit requests, and has held the device since `since`.
    Active { backend: String, since: Instant },
    /// `from` is finishing its in-flight work; nothing new is admitted for
    /// it, and `to` takes over the moment it reaches zero.
    Draining {
        from: String,
        to: String,
        started: Instant,
    },
}

/// Proof that a request holds a place in its group's current epoch.
/// Released on drop — over an unbounded channel, whose `send` neither
/// blocks nor needs a runtime handle, so it is safe from a non-async
/// `Drop`. A failed send just means the group is gone.
#[derive(Debug)]
pub(crate) struct GroupLease {
    tx: mpsc::UnboundedSender<Cmd>,
    backend: String,
    ticket: u64,
}

impl Drop for GroupLease {
    fn drop(&mut self) {
        let _ = self.tx.send(Cmd::Release {
            backend: std::mem::take(&mut self.backend),
            ticket: self.ticket,
        });
    }
}

/// A read-only view of a group, for the wait estimator. Taken by asking the
/// actor, so it is always internally consistent.
#[derive(Debug, Clone)]
pub struct GroupSnapshot {
    pub def: GroupDef,
    /// The member holding the device (the drain source, while draining).
    pub holder: Option<String>,
    /// How long the holder has had it.
    pub held_ms: i64,
    /// Set while draining: who takes over next.
    pub draining_to: Option<String>,
    pub in_flight: HashMap<String, InFlightSummary>,
    /// Backends with waiters, ordered by their oldest waiter — the pending
    /// epoch queue, which is also the order the device will move in.
    pub pending_epochs: Vec<PendingEpoch>,
    pub max_concurrency: HashMap<String, Option<u32>>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct InFlightSummary {
    pub count: usize,
    /// The *longest* remaining drain among these requests, not their sum:
    /// they run concurrently inside the engine's batch, so the device is
    /// free when the slowest one finishes.
    pub max_remaining_ms: i64,
    /// Mean estimated total drain of one request here, for projecting how
    /// long a *future* epoch of this backend will take.
    pub est_request_ms: i64,
}

#[derive(Debug, Clone)]
pub struct PendingEpoch {
    pub backend: String,
    pub waiters: usize,
    pub oldest_wait_ms: i64,
    /// Estimated drain of the waiting requests, for projecting the epochs
    /// queued ahead of the caller.
    pub est_request_ms: i64,
}

struct GroupState {
    id: String,
    def: GroupDef,
    /// Member name -> its concurrency limit (`None` = unlimited).
    members: HashMap<String, Option<u32>>,
    phase: Phase,
    in_flight: HashMap<String, Vec<InFlight>>,
    /// FIFO by arrival, which is exactly the aging order the pending epoch
    /// queue needs: the backend whose oldest waiter is oldest goes next.
    waiters: VecDeque<Waiter>,
    next_ticket: u64,
    /// Weak on purpose. The actor needs a sender to hand to each
    /// `GroupLease`, but holding a strong one would keep its own channel
    /// open forever — the task would outlive a deleted group and never
    /// exit. Once the handle and every outstanding lease are gone there is
    /// nobody left to grant to anyway.
    tx: mpsc::WeakUnboundedSender<Cmd>,
}

impl GroupState {
    fn count(&self, backend: &str) -> usize {
        self.in_flight.get(backend).map_or(0, Vec::len)
    }

    fn limit(&self, backend: &str) -> Option<u32> {
        self.members.get(backend).copied().flatten()
    }

    /// A waiter whose client vanished drops its receiver; skip those rather
    /// than switching the whole device over for a request nobody wants.
    fn prune_dead_waiters(&mut self) {
        self.waiters.retain(|w| !w.reply.is_closed());
    }

    /// Head of the pending epoch queue, excluding `current`.
    fn next_epoch(&self, current: Option<&str>) -> Option<String> {
        self.waiters
            .iter()
            .map(|w| w.backend.as_str())
            .find(|b| Some(*b) != current)
            .map(str::to_string)
    }

    fn begin_epoch(&mut self, backend: String, now: Instant) {
        tracing::debug!(group = %self.id, backend = %backend, "exclusive group epoch started");
        self.phase = Phase::Active {
            backend,
            since: now,
        };
    }

    /// Hands the device to every waiter the current epoch can take.
    /// Returns whether anything was granted.
    fn grant_all(&mut self, backend: &str, now: Instant) -> bool {
        // Every sender is gone: the group was deleted and the loop below is
        // about to exit anyway.
        let Some(tx) = self.tx.upgrade() else {
            return false;
        };
        let mut granted = false;
        loop {
            let room = match self.limit(backend) {
                Some(n) => self.count(backend) < n as usize,
                None => true,
            };
            if !room {
                return granted;
            }
            let Some(pos) = self.waiters.iter().position(|w| w.backend == backend) else {
                return granted;
            };
            let waiter = self.waiters.remove(pos).expect("position just found");
            let ticket = self.next_ticket;
            self.next_ticket += 1;
            let lease = GroupLease {
                tx: tx.clone(),
                backend: waiter.backend.clone(),
                ticket,
            };
            // A client that gave up between the prune and here: the lease
            // drops right back into a Release, so nothing is stranded.
            if waiter.reply.send(lease).is_err() {
                continue;
            }
            self.in_flight
                .entry(waiter.backend)
                .or_default()
                .push(InFlight {
                    ticket,
                    granted: now,
                    est_drain_ms: waiter.est_drain_ms,
                });
            granted = true;
        }
    }

    /// Advances the state machine one step; `true` means something changed
    /// and the caller should look again.
    fn step(&mut self, now: Instant) -> bool {
        match &self.phase {
            Phase::Draining { from, to, started } => {
                let (from, to, started) = (from.clone(), to.clone(), *started);
                if self.count(&from) == 0 {
                    self.begin_epoch(to, now);
                    return true;
                }
                if now.duration_since(started) >= ms(self.def.drain_timeout_ms) {
                    // Drain time is bounded in theory by max_tokens_cap /
                    // tok_s, but not in practice: an upstream hang or a
                    // stuck stream would otherwise wedge this group — and
                    // with it every model on the device — permanently.
                    // Flipping while `from` genuinely still runs briefly
                    // gives us the co-execution we were avoiding, which is
                    // strictly better than a deadlocked accelerator.
                    tracing::error!(
                        group = %self.id,
                        stuck_backend = %from,
                        elapsed_ms = now.duration_since(started).as_millis() as i64,
                        in_flight = self.count(&from),
                        next_backend = %to,
                        "exclusive group drain timed out; force-switching the lease"
                    );
                    self.begin_epoch(to, now);
                    return true;
                }
                false
            }
            Phase::Idle => match self.next_epoch(None) {
                Some(backend) => {
                    self.begin_epoch(backend, now);
                    true
                }
                None => false,
            },
            Phase::Active { backend, since } => {
                let (backend, since) = (backend.clone(), *since);
                if self.grant_all(&backend, now) {
                    return true;
                }
                let Some(next) = self.next_epoch(Some(&backend)) else {
                    return false;
                };
                if !self.should_switch(&backend, since, now) {
                    return false;
                }
                tracing::debug!(group = %self.id, from = %backend, to = %next, "draining");
                self.phase = Phase::Draining {
                    from: backend,
                    to: next,
                    started: now,
                };
                true
            }
        }
    }

    /// Whether the device should move off `holder`, given that someone else
    /// is waiting.
    ///
    /// `min_hold_ms` is the anti-thrash floor and `max_hold_ms` the
    /// anti-starvation ceiling; between them sits a deliberate preference
    /// for the *cheap* handover. Switching while the holder is still busy
    /// means draining — refusing new work on a device that is not yet free
    /// — so past `min_hold_ms` we take the switch only when the holder has
    /// gone idle and it costs nothing. `max_hold_ms` is what forces the
    /// drain anyway, so a backend with unbroken traffic cannot hold the
    /// device forever. (With `min_hold_ms == max_hold_ms` this collapses to
    /// "switch as soon as the floor has elapsed".)
    fn should_switch(&self, holder: &str, since: Instant, now: Instant) -> bool {
        let held = now.duration_since(since);
        held >= ms(self.def.max_hold_ms)
            || (held >= ms(self.def.min_hold_ms) && self.count(holder) == 0)
    }

    /// When the actor must wake even if no command arrives.
    fn next_deadline(&self, now: Instant) -> Option<Instant> {
        match &self.phase {
            Phase::Draining { started, .. } => Some(*started + ms(self.def.drain_timeout_ms)),
            Phase::Idle => None,
            Phase::Active { backend, since } => {
                self.next_epoch(Some(backend))?;
                let held = now.duration_since(*since);
                Some(if held < ms(self.def.min_hold_ms) {
                    *since + ms(self.def.min_hold_ms)
                } else {
                    *since + ms(self.def.max_hold_ms)
                })
            }
        }
    }

    fn snapshot(&self, now: Instant) -> GroupSnapshot {
        let (holder, held_ms, draining_to) = match &self.phase {
            Phase::Idle => (None, 0, None),
            Phase::Active { backend, since } => (
                Some(backend.clone()),
                now.duration_since(*since).as_millis() as i64,
                None,
            ),
            Phase::Draining { from, to, started } => (
                Some(from.clone()),
                now.duration_since(*started).as_millis() as i64,
                Some(to.clone()),
            ),
        };

        let in_flight = self
            .in_flight
            .iter()
            .filter(|(_, v)| !v.is_empty())
            .map(|(backend, reqs)| {
                let remaining = |r: &InFlight| {
                    (r.est_drain_ms - now.duration_since(r.granted).as_millis() as i64).max(0)
                };
                let summary = InFlightSummary {
                    count: reqs.len(),
                    max_remaining_ms: reqs.iter().map(remaining).max().unwrap_or(0),
                    est_request_ms: reqs.iter().map(|r| r.est_drain_ms).sum::<i64>()
                        / reqs.len() as i64,
                };
                (backend.clone(), summary)
            })
            .collect();

        // Ordered by oldest waiter, which the FIFO queue already gives us.
        let mut pending: Vec<PendingEpoch> = Vec::new();
        for w in &self.waiters {
            match pending.iter_mut().find(|p| p.backend == w.backend) {
                Some(p) => {
                    p.waiters += 1;
                    p.est_request_ms = p.est_request_ms.max(w.est_drain_ms);
                }
                None => pending.push(PendingEpoch {
                    backend: w.backend.clone(),
                    waiters: 1,
                    oldest_wait_ms: now.duration_since(w.enqueued).as_millis() as i64,
                    est_request_ms: w.est_drain_ms,
                }),
            }
        }

        GroupSnapshot {
            def: self.def.clone(),
            holder,
            held_ms,
            draining_to,
            in_flight,
            pending_epochs: pending,
            max_concurrency: self.members.clone(),
        }
    }
}

fn ms(v: i64) -> Duration {
    Duration::from_millis(v.max(0) as u64)
}

async fn run_group(mut state: GroupState, mut rx: mpsc::UnboundedReceiver<Cmd>) {
    loop {
        state.prune_dead_waiters();
        // Each pass either grants a waiter, ends an epoch, or settles. The
        // bound is a backstop against a logic error turning into a spin.
        let mut steps = 0;
        while state.step(Instant::now()) {
            steps += 1;
            if steps > 1024 {
                tracing::error!(group = %state.id, "exclusive group failed to settle");
                break;
            }
        }

        let now = Instant::now();
        let cmd = match state.next_deadline(now) {
            Some(at) => tokio::select! {
                cmd = rx.recv() => cmd,
                _ = tokio::time::sleep_until(at) => continue,
            },
            None => rx.recv().await,
        };
        // Every sender is gone: the group was deleted from the catalog.
        let Some(cmd) = cmd else { return };

        match cmd {
            Cmd::Request(w) => state.waiters.push_back(w),
            Cmd::Release { backend, ticket } => {
                if let Some(reqs) = state.in_flight.get_mut(&backend) {
                    reqs.retain(|r| r.ticket != ticket);
                }
            }
            Cmd::Retune { def, members } => {
                state.def = def;
                state.members = members;
            }
            Cmd::Probe(reply) => {
                let _ = reply.send(state.snapshot(Instant::now()));
            }
        }
    }
}

/// Client side of one group actor.
#[derive(Debug)]
pub struct GroupHandle {
    pub id: String,
    def: GroupDef,
    members: HashMap<String, Option<u32>>,
    tx: mpsc::UnboundedSender<Cmd>,
}

impl GroupHandle {
    fn spawn(def: GroupDef, members: HashMap<String, Option<u32>>) -> Self {
        let (tx, rx) = mpsc::unbounded_channel();
        let state = GroupState {
            id: def.id.clone(),
            def: def.clone(),
            members: members.clone(),
            phase: Phase::Idle,
            in_flight: HashMap::new(),
            waiters: VecDeque::new(),
            next_ticket: 0,
            tx: tx.downgrade(),
        };
        tokio::spawn(run_group(state, rx));
        GroupHandle {
            id: def.id.clone(),
            def,
            members,
            tx,
        }
    }

    fn retune(&mut self, def: GroupDef, members: HashMap<String, Option<u32>>) {
        self.def = def.clone();
        self.members = members.clone();
        let _ = self.tx.send(Cmd::Retune { def, members });
    }

    /// Queues for the device and waits for the lease. `None` means the
    /// group has gone away (deleted mid-flight), in which case the request
    /// proceeds unscheduled rather than failing — there is no longer an
    /// exclusivity invariant to protect.
    async fn request(&self, backend: &str, est_drain_ms: i64) -> Option<GroupLease> {
        let (reply, rx) = oneshot::channel();
        let cmd = Cmd::Request(Waiter {
            backend: backend.to_string(),
            enqueued: Instant::now(),
            est_drain_ms,
            reply,
        });
        if self.tx.send(cmd).is_err() {
            return None;
        }
        rx.await.ok()
    }

    /// Current state, for the wait estimator.
    pub async fn snapshot(&self) -> Option<GroupSnapshot> {
        let (reply, rx) = oneshot::channel();
        self.tx.send(Cmd::Probe(reply)).ok()?;
        rx.await.ok()
    }
}

/// Admission control for every backend in the catalog.
#[derive(Default)]
pub struct Scheduler {
    backends: RwLock<HashMap<String, Arc<BackendSlots>>>,
    groups: RwLock<HashMap<String, Arc<GroupHandle>>>,
}

impl std::fmt::Debug for Scheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Scheduler").finish_non_exhaustive()
    }
}

impl Scheduler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Built from a routing snapshot at startup.
    pub fn from_routing(routing: &Routing) -> Self {
        let sched = Self::new();
        sched.reconcile(routing);
        sched
    }

    /// Brings the scheduler in line with a new routing snapshot: adds new
    /// backends, forgets removed ones, and re-creates any whose concurrency
    /// limit changed.
    ///
    /// Re-creating replaces the semaphore, so requests already in flight
    /// under the old limit hold the *old* one and release into it. For a
    /// moment the backend can therefore carry up to `old_in_flight + new
    /// limit` requests. That is bounded, self-healing, and the alternative
    /// (resizing a live semaphore) trades it for a much subtler class of
    /// bug — retuning a limit is a rare admin action.
    pub fn reconcile(&self, routing: &Routing) {
        {
            let mut map = self.backends.write().unwrap();
            map.retain(|name, _| routing.backends.iter().any(|b| &b.name == name));
            for b in &routing.backends {
                match map.get(&b.name) {
                    Some(slots) if slots.limit == b.max_concurrency => {
                        // The seed only matters until real samples arrive,
                        // but an operator retuning it should see the effect.
                        *slots.throughput.seed.write().unwrap() = b.tok_s_estimate;
                    }
                    // Carry the observed throughput across the swap: what
                    // the backend can do has not changed just because its
                    // concurrency limit did.
                    Some(slots) => {
                        let throughput = Arc::clone(&slots.throughput);
                        *throughput.seed.write().unwrap() = b.tok_s_estimate;
                        map.insert(
                            b.name.clone(),
                            Arc::new(BackendSlots::new(
                                b.name.clone(),
                                b.max_concurrency,
                                throughput,
                            )),
                        );
                    }
                    None => {
                        map.insert(
                            b.name.clone(),
                            Arc::new(BackendSlots::new(
                                b.name.clone(),
                                b.max_concurrency,
                                Arc::new(Throughput::new(b.tok_s_estimate)),
                            )),
                        );
                    }
                }
            }
        }

        // Groups are reconciled, never rebuilt: an existing actor keeps its
        // lease, its in-flight bookkeeping and its queue across the edit,
        // and only learns the new timings and membership. Dropping the
        // handle of a deleted group closes its channel, which is how its
        // task exits.
        let mut groups = self.groups.write().unwrap();
        groups.retain(|id, _| routing.groups.iter().any(|g| &g.id == id));
        for def in &routing.groups {
            // Only the members subject to the lease are gang-scheduled;
            // `exempt` backends sit on the same device and are deliberately
            // invisible here.
            let members: HashMap<String, Option<u32>> = routing
                .backends
                .iter()
                .filter(|b| {
                    b.scheduling_class == crate::catalog::SchedulingClass::Grouped
                        && b.exclusive_group_id.as_deref() == Some(def.id.as_str())
                })
                .map(|b| (b.name.clone(), b.max_concurrency))
                .collect();
            match groups.get(&def.id) {
                Some(h) if h.def == *def && h.members == members => {}
                Some(h) => {
                    // Retune in place: cloning out of the Arc would lose
                    // the running actor's channel.
                    let mut updated = GroupHandle {
                        id: h.id.clone(),
                        def: h.def.clone(),
                        members: h.members.clone(),
                        tx: h.tx.clone(),
                    };
                    updated.retune(def.clone(), members);
                    groups.insert(def.id.clone(), Arc::new(updated));
                }
                None => {
                    groups.insert(
                        def.id.clone(),
                        Arc::new(GroupHandle::spawn(def.clone(), members)),
                    );
                }
            }
        }
    }

    pub fn group(&self, id: &str) -> Option<Arc<GroupHandle>> {
        self.groups.read().unwrap().get(id).cloned()
    }

    /// Output tokens per second to plan with for `backend`: the observed
    /// low quantile once there are samples, its configured seed before
    /// that. Deliberately pessimistic — see `Throughput`.
    pub fn tok_s(&self, backend: &Backend) -> f64 {
        let observed = self
            .backends
            .read()
            .unwrap()
            .get(&backend.name)
            .map(|s| s.throughput.pessimistic());
        observed.unwrap_or_else(|| backend.tok_s_estimate.unwrap_or(FALLBACK_TOK_S))
    }

    /// Upper bound, in milliseconds, on how long one request will occupy
    /// `backend`. `requested` is the client's own generated-token limit
    /// when it sent one — the effective ceiling is the smaller of that and
    /// the backend's cap, and the cap alone when the client sent none.
    /// Returns 0 for a backend with no cap, where occupancy is unbounded
    /// and there is nothing to schedule against anyway.
    pub fn est_drain_ms(&self, backend: &Backend, requested: Option<i64>) -> i64 {
        let Some(cap) = backend.max_tokens_cap else {
            return 0;
        };
        let tokens = requested.map_or(cap, |r| r.min(cap)).max(0);
        (tokens as f64 / self.tok_s(backend) * 1000.0) as i64
    }

    /// The slot state for a backend, creating it if the scheduler has not
    /// seen it yet. Missing entries are only possible in the window between
    /// a catalog write and its `reconcile`, so admitting rather than
    /// failing is the right call — the request already resolved a live
    /// target from a consistent routing snapshot.
    pub fn slots(&self, backend: &str, limit: Option<u32>) -> Arc<BackendSlots> {
        if let Some(slots) = self.backends.read().unwrap().get(backend) {
            return Arc::clone(slots);
        }
        let mut map = self.backends.write().unwrap();
        Arc::clone(map.entry(backend.to_string()).or_insert_with(|| {
            Arc::new(BackendSlots::new(
                backend.to_string(),
                limit,
                Arc::new(Throughput::new(None)),
            ))
        }))
    }

    /// How long a request for `backend` is expected to sit in the group's
    /// queue before it reaches the device. 0 for anything not gang
    /// scheduled — an unrestricted or exempt backend has no lease to wait
    /// for.
    ///
    /// This is evaluated *before* enqueuing, which is the whole point: a
    /// deadline checked after joining the queue is a deadline already spent.
    ///
    /// ```text
    /// predicted_wait(b) = time_to_reach_epoch(b) + queue_ahead_delay(b)
    /// ```
    ///
    /// where reaching b's epoch means waiting out whatever is on the device
    /// now, plus `min_hold_ms` and one batch for each epoch queued ahead of
    /// b; and `queue_ahead_delay` is the batches of b's own waiters that
    /// have to clear first. Every term rounds *up* — over-predicting sheds
    /// a request that might have fit, which is recoverable, while
    /// under-predicting quietly breaks the promise the deadline made.
    pub async fn predicted_wait_ms(&self, backend: &Backend, est_drain_ms: i64) -> i64 {
        let Some(group) = backend.lease_group() else {
            return 0;
        };
        let Some(handle) = self.group(group) else {
            return 0;
        };
        let Some(snap) = handle.snapshot().await else {
            return 0;
        };
        let name = backend.name.as_str();

        let concurrency = |b: &str| -> i64 {
            snap.max_concurrency
                .get(b)
                .copied()
                .flatten()
                .map_or(i64::MAX, i64::from)
                .max(1)
        };
        let in_flight = |b: &str| snap.in_flight.get(b).copied().unwrap_or_default();
        // Draining means the device is already committed to leaving its
        // holder, so the holder is not "active" for our purposes.
        let holds_now = snap.holder.as_deref() == Some(name) && snap.draining_to.is_none();

        let mut wait: i64 = 0;
        if !holds_now {
            if let Some(holder) = &snap.holder {
                // Whatever is running has to finish before the lease moves.
                wait += in_flight(holder).max_remaining_ms;
                // …and an epoch that has not reached its floor cannot hand
                // over yet, however idle it looks. This term is why an
                // interactive deadline below `min_hold_ms` sheds nearly
                // every request that arrives mid-epoch.
                if snap.draining_to.is_none() {
                    wait = wait.max(snap.def.min_hold_ms - snap.held_ms);
                }
            }
            // Epochs queued ahead of ours, oldest waiter first.
            for pending in &snap.pending_epochs {
                if pending.backend == name {
                    break;
                }
                wait += snap.def.min_hold_ms + pending.est_request_ms;
            }
        }

        // Our own queue. When we already hold the device but every slot is
        // taken, the first thing to wait for is a slot freeing, not a whole
        // request duration.
        let mut ahead = snap
            .pending_epochs
            .iter()
            .find(|p| p.backend == name)
            .map_or(0, |p| p.waiters as i64);
        let running = in_flight(name);
        if holds_now && running.count as i64 >= concurrency(name) {
            wait += running.max_remaining_ms;
        } else if holds_now {
            ahead = (ahead - (concurrency(name) - running.count as i64)).max(0);
        }
        // Whole batches of waiters ahead of us, each costing one request.
        wait += (ahead / concurrency(name)) * est_drain_ms.max(1);

        wait.max(0)
    }

    /// Admits one request onto `backend`, waiting as long as it takes.
    ///
    /// For a grouped backend this is two gates in order: first the group's
    /// lease (which is what serializes the device), then the backend's own
    /// slot. Taking the lease first is what makes the wait bounded — the
    /// actor only grants up to `max_concurrency` per epoch, so the slot is
    /// already free by construction.
    ///
    /// `est_drain_ms` is this request's upper-bound occupancy of the
    /// device, used by the actor only to answer wait estimates.
    pub async fn admit(&self, backend: &Backend, est_drain_ms: i64) -> SlotGuard {
        let lease = match backend.lease_group() {
            Some(group) => match self.group(group) {
                Some(handle) => handle.request(&backend.name, est_drain_ms).await,
                // Between a catalog write and its reconcile, or on a
                // hand-edited database. Nothing to serialize against.
                None => None,
            },
            None => None,
        };
        let mut guard = self
            .slots(&backend.name, backend.max_concurrency)
            .acquire()
            .await;
        guard.lease = lease;
        guard
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{BackendDef, Catalog, ModelDef, TargetDef};
    use crate::config::BackendKind;

    fn backend_def(name: &str, max_concurrency: Option<u32>) -> BackendDef {
        BackendDef {
            name: name.into(),
            kinds: vec![BackendKind::Openai],
            base_url: "http://127.0.0.1:1".into(),
            api_key: None,
            api_key_env: None,
            anthropic_version: None,
            inject_usage: true,
            scheduling_class: Default::default(),
            exclusive_group_id: None,
            max_concurrency,
            max_tokens_cap: None,
            tok_s_estimate: None,
        }
    }

    fn grouped_def(name: &str, group: &str, max_concurrency: u32) -> BackendDef {
        BackendDef {
            scheduling_class: crate::catalog::SchedulingClass::Grouped,
            exclusive_group_id: Some(group.into()),
            max_concurrency: Some(max_concurrency),
            max_tokens_cap: Some(1000),
            tok_s_estimate: Some(100.0),
            ..backend_def(name, Some(max_concurrency))
        }
    }

    fn group_def(min: i64, max: i64, drain: i64) -> GroupDef {
        GroupDef {
            id: "dgx".into(),
            min_hold_ms: min,
            max_hold_ms: max,
            drain_timeout_ms: drain,
        }
    }

    fn routing_with(groups: Vec<GroupDef>, backends: Vec<BackendDef>) -> Routing {
        let models = backends
            .iter()
            .map(|b| ModelDef {
                name: format!("m-{}", b.name),
                aliases: Vec::new(),
                pricing: None,
                qos_max_wait_ms: None,
                qos_background_max_wait_ms: None,
                on_shed_without_fallback: Default::default(),
                targets: vec![TargetDef {
                    backend: b.name.clone(),
                    model: None,
                    weight: 1,
                    pricing: None,
                    fallback_only: false,
                }],
            })
            .collect();
        Routing::build(Catalog {
            groups,
            backends,
            models,
        })
        .unwrap()
    }

    /// The compiled `Backend` a route points at, which is what `admit` takes.
    fn backend_of(routing: &Routing, name: &str) -> Arc<Backend> {
        Arc::clone(&routing.routes.get(&format!("m-{name}")).unwrap().targets[0].backend)
    }

    fn routing(backends: Vec<BackendDef>) -> Routing {
        let models = backends
            .iter()
            .map(|b| ModelDef {
                name: format!("m-{}", b.name),
                aliases: Vec::new(),
                pricing: None,
                qos_max_wait_ms: None,
                qos_background_max_wait_ms: None,
                on_shed_without_fallback: Default::default(),
                targets: vec![TargetDef {
                    backend: b.name.clone(),
                    model: None,
                    weight: 1,
                    pricing: None,
                    fallback_only: false,
                }],
            })
            .collect();
        Routing::build(Catalog {
            groups: Vec::new(),
            backends,
            models,
        })
        .unwrap()
    }

    #[tokio::test]
    async fn a_limit_of_one_serializes_requests() {
        let sched = Scheduler::from_routing(&routing(vec![backend_def("b", Some(1))]));
        let slots = sched.slots("b", Some(1));

        let first = slots.acquire().await;
        assert_eq!(slots.in_flight(), 1);

        let waiting = {
            let slots = Arc::clone(&slots);
            tokio::spawn(async move { slots.acquire().await })
        };
        // The second acquire cannot complete while the first holds the slot.
        tokio::time::sleep(Duration::from_millis(30)).await;
        assert!(!waiting.is_finished(), "second request must wait");

        drop(first);
        let second = waiting.await.unwrap();
        assert_eq!(slots.in_flight(), 1);
        drop(second);
        assert_eq!(slots.in_flight(), 0);
    }

    #[tokio::test]
    async fn an_unlimited_backend_never_blocks() {
        let sched = Scheduler::from_routing(&routing(vec![backend_def("b", None)]));
        let slots = sched.slots("b", None);
        assert!(slots.limit().is_none());
        let guards: Vec<_> = futures_util::future::join_all((0..50).map(|_| slots.acquire())).await;
        assert_eq!(slots.in_flight(), 50);
        assert_eq!(guards[0].backend(), "b");
        drop(guards);
        assert_eq!(slots.in_flight(), 0);
    }

    /// A dropped future — the shape a client disconnect takes — must give
    /// the slot back, not leak it.
    #[tokio::test]
    async fn dropping_a_pending_acquire_leaves_the_slot_free() {
        let sched = Scheduler::new();
        let slots = sched.slots("b", Some(1));
        let held = slots.acquire().await;

        {
            let pending = slots.acquire();
            tokio::pin!(pending);
            let polled = tokio::time::timeout(Duration::from_millis(20), &mut pending).await;
            assert!(polled.is_err(), "must still be waiting");
        }
        drop(held);
        // The abandoned waiter took nothing with it.
        let next = tokio::time::timeout(Duration::from_millis(50), slots.acquire())
            .await
            .expect("the slot is free again");
        assert_eq!(slots.in_flight(), 1);
        drop(next);
        assert_eq!(slots.in_flight(), 0);
    }

    #[tokio::test]
    async fn reconcile_adds_removes_and_retunes() {
        let sched = Scheduler::new();
        sched.reconcile(&routing(vec![
            backend_def("a", Some(2)),
            backend_def("b", None),
        ]));
        let a = sched.slots("a", Some(2));
        assert_eq!(a.limit(), Some(2));

        // An unrelated edit keeps the same instance, so a held slot stays
        // meaningful across catalog writes.
        let held = a.acquire().await;
        sched.reconcile(&routing(vec![
            backend_def("a", Some(2)),
            backend_def("c", Some(1)),
        ]));
        assert!(
            Arc::ptr_eq(&a, &sched.slots("a", Some(2))),
            "unchanged backend is reused"
        );
        assert_eq!(a.in_flight(), 1);
        assert!(sched.backends.read().unwrap().contains_key("c"));
        assert!(
            !sched.backends.read().unwrap().contains_key("b"),
            "removed backend is forgotten"
        );

        // Retuning the limit swaps the instance; the in-flight request keeps
        // its own and releases into it.
        sched.reconcile(&routing(vec![
            backend_def("a", Some(8)),
            backend_def("c", Some(1)),
        ]));
        let retuned = sched.slots("a", Some(8));
        assert!(!Arc::ptr_eq(&a, &retuned));
        assert_eq!(retuned.limit(), Some(8));
        assert_eq!(retuned.in_flight(), 0);
        drop(held);
        assert_eq!(a.in_flight(), 0);
    }

    // ---- throughput ------------------------------------------------------

    #[test]
    fn throughput_starts_from_the_seed_and_then_follows_observation() {
        let t = Throughput::new(Some(90.0));
        assert_eq!(t.pessimistic(), 90.0);

        // Too few samples to say anything: the seed still wins.
        t.observe(1000, 1000);
        assert_eq!(t.pessimistic(), 90.0);

        // Once the ring has enough, the observed low quantile takes over —
        // even though it is far above the (conservative) seed.
        for _ in 0..4 {
            t.observe(1000, 1000); // 1000 tok/s
        }
        assert!(t.pessimistic() > 90.0, "{}", t.pessimistic());
    }

    #[test]
    fn throughput_plans_from_a_low_quantile_not_the_mean() {
        let t = Throughput::new(Some(50.0));
        // Mostly fast, occasionally slow — the shape speculative decoding
        // actually produces. Planning from the mean here would promise a
        // deadline the slow tail cannot keep.
        for _ in 0..24 {
            t.observe(1000, 1000); // 1000 tok/s
        }
        for _ in 0..8 {
            t.observe(1000, 10_000); // 100 tok/s
        }
        let mean = (24.0 * 1000.0 + 8.0 * 100.0) / 32.0;
        assert!(
            t.pessimistic() < mean,
            "expected a pessimistic estimate, got {} against a mean of {mean}",
            t.pessimistic()
        );
        assert!(t.pessimistic() <= 1000.0);
    }

    #[test]
    fn throughput_ignores_samples_too_small_to_time() {
        let t = Throughput::new(Some(20.0));
        for _ in 0..10 {
            t.observe(2, 1000); // too few tokens
            t.observe(1000, 5); // too short to measure
            t.observe(0, 0);
        }
        assert_eq!(t.pessimistic(), 20.0, "noise must not displace the seed");
    }

    #[test]
    fn throughput_ring_forgets_the_distant_past() {
        let t = Throughput::new(None);
        for _ in 0..Throughput::CAPACITY {
            t.observe(1000, 100_000); // 10 tok/s, an old slow regime
        }
        assert!(t.pessimistic() < 20.0);
        for _ in 0..Throughput::CAPACITY {
            t.observe(1000, 1_000); // 1000 tok/s, the new one
        }
        assert!(t.pessimistic() > 500.0, "{}", t.pessimistic());
    }

    #[tokio::test]
    async fn est_drain_takes_the_smaller_of_the_request_and_the_cap() {
        let sched = Scheduler::new();
        let routing = routing_with(
            vec![group_def(50, 500, 5_000)],
            vec![grouped_def("moe", "dgx", 4)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe"); // cap 1000 tokens at 100 tok/s

        // No client limit: plan for the cap, which is what the clamp will
        // actually enforce.
        assert_eq!(sched.est_drain_ms(&moe, None), 10_000);
        // A smaller client limit binds instead.
        assert_eq!(sched.est_drain_ms(&moe, Some(200)), 2_000);
        // A larger one does not — the clamp brings it back to the cap.
        assert_eq!(sched.est_drain_ms(&moe, Some(50_000)), 10_000);

        // An uncapped backend has unbounded occupancy and nothing to plan
        // against, so it reports no estimate at all.
        let plain = routing_with(vec![], vec![backend_def("mac", None)]);
        assert_eq!(sched.est_drain_ms(&backend_of(&plain, "mac"), Some(500)), 0);
    }

    #[tokio::test]
    async fn observed_throughput_survives_a_concurrency_retune() {
        let sched = Scheduler::new();
        sched.reconcile(&routing(vec![backend_def("b", Some(2))]));
        let slots = sched.slots("b", Some(2));
        for _ in 0..8 {
            slots.throughput().observe(1000, 1000);
        }
        let before = slots.throughput().pessimistic();
        assert!(before > 500.0);

        // Retuning the limit swaps the slot state but must not throw away
        // what the backend has been observed to do.
        sched.reconcile(&routing(vec![backend_def("b", Some(9))]));
        let retuned = sched.slots("b", Some(9));
        assert!(!Arc::ptr_eq(&slots, &retuned));
        assert_eq!(retuned.throughput().pessimistic(), before);
    }

    // ---- exclusive groups ------------------------------------------------

    /// Records who is on the device and screams the moment two different
    /// grouped backends are in flight at once.
    #[derive(Default)]
    struct Exclusivity {
        live: std::sync::Mutex<HashMap<String, usize>>,
        peak_backends: AtomicU32,
    }

    impl Exclusivity {
        fn enter(&self, backend: &str) {
            let mut live = self.live.lock().unwrap();
            *live.entry(backend.to_string()).or_default() += 1;
            let distinct = live.values().filter(|n| **n > 0).count();
            self.peak_backends
                .fetch_max(distinct as u32, Ordering::AcqRel);
            assert_eq!(
                distinct, 1,
                "two grouped backends generating at once: {live:?}"
            );
        }

        fn leave(&self, backend: &str) {
            let mut live = self.live.lock().unwrap();
            *live.get_mut(backend).unwrap() -= 1;
        }
    }

    #[tokio::test(start_paused = true)]
    async fn only_one_grouped_backend_generates_at_a_time() {
        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(50, 500, 5_000)],
            vec![grouped_def("moe", "dgx", 4), grouped_def("dense", "dgx", 4)],
        );
        sched.reconcile(&routing);
        let backends = [backend_of(&routing, "moe"), backend_of(&routing, "dense")];
        let watch = Arc::new(Exclusivity::default());

        // 40 requests alternating between the two models, each occupying
        // the device for a little while.
        let mut tasks = Vec::new();
        for i in 0..40 {
            let (sched, watch) = (Arc::clone(&sched), Arc::clone(&watch));
            let backend = Arc::clone(&backends[i % 2]);
            tasks.push(tokio::spawn(async move {
                let guard = sched.admit(&backend, 100).await;
                watch.enter(&backend.name);
                tokio::time::sleep(Duration::from_millis(20)).await;
                watch.leave(&backend.name);
                drop(guard);
            }));
        }
        for t in tasks {
            t.await.unwrap();
        }

        assert_eq!(watch.peak_backends.load(Ordering::Acquire), 1);
        assert!(watch.live.lock().unwrap().values().all(|n| *n == 0));

        // The group settles back to grantable, with nothing left in flight.
        let snap = sched.group("dgx").unwrap().snapshot().await.unwrap();
        assert!(snap.pending_epochs.is_empty());
        assert!(snap.in_flight.values().all(|s| s.count == 0));
    }

    /// The exclusivity invariant under sustained, irregular load across
    /// three members — arrival gaps, occupancy and burst sizes all vary, so
    /// the actor meets grant/drain/switch/timeout interleavings a fixed
    /// script would not. Preferred over `loom` here: the actor's ordering
    /// is a single-threaded message loop, and what is actually worth
    /// hammering is the *timing*, which paused time makes free.
    #[tokio::test(start_paused = true)]
    async fn exclusivity_holds_under_randomized_load() {
        use rand::Rng;

        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(40, 250, 2_000)],
            vec![
                grouped_def("a", "dgx", 2),
                grouped_def("b", "dgx", 4),
                grouped_def("c", "dgx", 1),
            ],
        );
        sched.reconcile(&routing);
        let backends: Vec<_> = ["a", "b", "c"]
            .iter()
            .map(|n| backend_of(&routing, n))
            .collect();
        let watch = Arc::new(Exclusivity::default());

        let mut rng = rand::rng();
        let mut tasks = Vec::new();
        for _ in 0..400 {
            let (sched, watch) = (Arc::clone(&sched), Arc::clone(&watch));
            let backend = Arc::clone(&backends[rng.random_range(0..3)]);
            let hold = rng.random_range(1..80);
            tasks.push(tokio::spawn(async move {
                let guard = sched.admit(&backend, hold as i64).await;
                watch.enter(&backend.name);
                tokio::time::sleep(Duration::from_millis(hold)).await;
                watch.leave(&backend.name);
                drop(guard);
            }));
            if rng.random_bool(0.4) {
                tokio::time::sleep(Duration::from_millis(rng.random_range(1..15))).await;
            }
        }
        for t in tasks {
            // Every request completes: no waiter is starved and no drain
            // wedges the group.
            tokio::time::timeout(Duration::from_secs(120), t)
                .await
                .expect("every request must eventually be served")
                .unwrap();
        }
        assert_eq!(watch.peak_backends.load(Ordering::Acquire), 1);

        let snap = sched.group("dgx").unwrap().snapshot().await.unwrap();
        assert!(snap.pending_epochs.is_empty());
        assert!(
            snap.in_flight.values().all(|s| s.count == 0),
            "in-flight bookkeeping leaked: {snap:?}"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_lease_moves_only_after_the_minimum_hold() {
        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(1_000, 10_000, 30_000)],
            vec![grouped_def("moe", "dgx", 1), grouped_def("dense", "dgx", 1)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe");
        let dense = backend_of(&routing, "dense");

        // moe takes the device and finishes immediately, so it is idle —
        // but it still owns the device until min_hold has elapsed.
        drop(sched.admit(&moe, 0).await);
        let group = sched.group("dgx").unwrap();
        assert_eq!(
            group.snapshot().await.unwrap().holder.as_deref(),
            Some("moe")
        );

        let waiting = {
            let (sched, dense) = (Arc::clone(&sched), Arc::clone(&dense));
            tokio::spawn(async move { sched.admit(&dense, 0).await })
        };
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert!(
            !waiting.is_finished(),
            "min_hold_ms must keep the lease put"
        );

        // Past the floor, an idle holder hands over without a drain.
        tokio::time::sleep(Duration::from_millis(700)).await;
        let guard = tokio::time::timeout(Duration::from_secs(2), waiting)
            .await
            .expect("the lease should have moved")
            .unwrap();
        assert_eq!(
            group.snapshot().await.unwrap().holder.as_deref(),
            Some("dense")
        );
        drop(guard);
    }

    #[tokio::test(start_paused = true)]
    async fn a_busy_holder_is_forced_off_at_the_maximum_hold() {
        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(200, 1_000, 30_000)],
            vec![grouped_def("moe", "dgx", 4), grouped_def("dense", "dgx", 4)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe");
        let dense = backend_of(&routing, "dense");

        // 100 moe requests arriving 10ms apart and occupying the device
        // 100ms each: ~2.5 s of work behind a 1 s ceiling, and moe is never
        // idle, so the cheap handover at min_hold_ms can never fire. Only
        // max_hold_ms can move the lease here.
        let mut load = Vec::new();
        for _ in 0..100 {
            let (sched, moe) = (Arc::clone(&sched), Arc::clone(&moe));
            load.push(tokio::spawn(async move {
                let guard = sched.admit(&moe, 100).await;
                tokio::time::sleep(Duration::from_millis(100)).await;
                drop(guard);
            }));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let started = Instant::now();
        let waiting = {
            let (sched, dense) = (Arc::clone(&sched), Arc::clone(&dense));
            tokio::spawn(async move { sched.admit(&dense, 0).await })
        };
        let guard = tokio::time::timeout(Duration::from_millis(2_000), waiting)
            .await
            .expect("max_hold_ms must force the switch off a busy holder")
            .unwrap();
        assert_eq!(guard.backend(), "dense");
        // It cut in while moe still had work outstanding — the lease moved
        // mid-stream rather than after moe finally ran dry.
        let waited = Instant::now().duration_since(started);
        assert!(waited < Duration::from_millis(2_000), "waited {waited:?}");
        let snap = sched.group("dgx").unwrap().snapshot().await.unwrap();
        assert!(
            snap.pending_epochs.iter().any(|p| p.backend == "moe")
                || snap.in_flight.get("moe").is_some_and(|s| s.count > 0),
            "moe should still have been busy: {snap:?}"
        );

        drop(guard);
        for t in load {
            tokio::time::timeout(Duration::from_secs(30), t)
                .await
                .unwrap()
                .unwrap();
        }
    }

    #[tokio::test(start_paused = true)]
    async fn a_third_member_is_not_starved_by_two_busy_ones() {
        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(50, 200, 5_000)],
            vec![
                grouped_def("a", "dgx", 1),
                grouped_def("b", "dgx", 1),
                grouped_def("c", "dgx", 1),
            ],
        );
        sched.reconcile(&routing);
        let backends: Vec<_> = ["a", "b", "c"]
            .iter()
            .map(|n| backend_of(&routing, n))
            .collect();

        // `c` queues once, behind heavy a/b traffic. Aging by oldest waiter
        // means it goes next, not last.
        let served = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let mut tasks = Vec::new();
        for i in 0..18 {
            let (sched, served) = (Arc::clone(&sched), Arc::clone(&served));
            let backend = Arc::clone(&backends[if i == 5 { 2 } else { i % 2 }]);
            tasks.push(tokio::spawn(async move {
                let guard = sched.admit(&backend, 50).await;
                served.lock().unwrap().push(backend.name.clone());
                tokio::time::sleep(Duration::from_millis(30)).await;
                drop(guard);
            }));
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        for t in tasks {
            tokio::time::timeout(Duration::from_secs(10), t)
                .await
                .expect("no member may be starved")
                .unwrap();
        }
        let served = served.lock().unwrap();
        assert_eq!(served.len(), 18);
        assert!(served.iter().any(|s| s == "c"), "c never ran: {served:?}");
    }

    #[tokio::test(start_paused = true)]
    async fn a_stuck_drain_is_force_flipped_by_the_watchdog() {
        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(50, 100, 1_000)],
            vec![grouped_def("moe", "dgx", 4), grouped_def("dense", "dgx", 4)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe");
        let dense = backend_of(&routing, "dense");

        // A request that never completes — an upstream hang. Without the
        // watchdog this wedges the group, and with it every model on the
        // device, forever.
        let stuck = sched.admit(&moe, 100).await;

        let waiting = {
            let (sched, dense) = (Arc::clone(&sched), Arc::clone(&dense));
            tokio::spawn(async move { sched.admit(&dense, 0).await })
        };
        let guard = tokio::time::timeout(Duration::from_secs(10), waiting)
            .await
            .expect("the drain watchdog must break the deadlock")
            .unwrap();
        assert_eq!(guard.backend(), "dense");

        let group = sched.group("dgx").unwrap();
        let snap = group.snapshot().await.unwrap();
        assert_eq!(snap.holder.as_deref(), Some("dense"));
        // The straggler is still counted — it is genuinely still running,
        // which is the compromise the force-flip makes.
        assert_eq!(snap.in_flight["moe"].count, 1);

        // And the group keeps working afterwards.
        drop(stuck);
        drop(guard);
        let next = tokio::time::timeout(Duration::from_secs(10), sched.admit(&moe, 0))
            .await
            .expect("the group must still grant leases after a force-flip");
        drop(next);
    }

    #[tokio::test(start_paused = true)]
    async fn a_queued_waiter_that_gives_up_is_skipped() {
        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(50, 200, 5_000)],
            vec![grouped_def("moe", "dgx", 1), grouped_def("dense", "dgx", 1)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe");
        let dense = backend_of(&routing, "dense");

        let held = sched.admit(&moe, 100).await;
        // A client that queues for `dense` and then disconnects.
        let abandoned = {
            let (sched, dense) = (Arc::clone(&sched), Arc::clone(&dense));
            tokio::spawn(async move { sched.admit(&dense, 100).await })
        };
        tokio::time::sleep(Duration::from_millis(20)).await;
        abandoned.abort();
        tokio::time::sleep(Duration::from_millis(20)).await;

        drop(held);
        // The group is grantable again and holds no ghost bookkeeping.
        let next = tokio::time::timeout(Duration::from_secs(5), sched.admit(&moe, 100))
            .await
            .expect("an abandoned waiter must not wedge the group");
        let snap = sched.group("dgx").unwrap().snapshot().await.unwrap();
        assert!(snap.pending_epochs.is_empty(), "{:?}", snap.pending_epochs);
        assert_eq!(snap.in_flight["moe"].count, 1);
        drop(next);
    }

    #[tokio::test(start_paused = true)]
    async fn a_catalog_edit_does_not_drop_a_held_lease() {
        let sched = Arc::new(Scheduler::new());
        let routing = routing_with(
            vec![group_def(1_000, 10_000, 30_000)],
            vec![grouped_def("moe", "dgx", 1), grouped_def("dense", "dgx", 1)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe");
        let held = sched.admit(&moe, 100).await;

        // Retune the group's timings, as an admin edit would. This is the
        // whole reason scheduler state lives outside `Routing`: a rebuild
        // here would drop the lease and zero the in-flight count.
        let edited = routing_with(
            vec![group_def(2_000, 20_000, 40_000)],
            vec![grouped_def("moe", "dgx", 1), grouped_def("dense", "dgx", 1)],
        );
        sched.reconcile(&edited);

        let group = sched.group("dgx").unwrap();
        let snap = group.snapshot().await.unwrap();
        assert_eq!(snap.holder.as_deref(), Some("moe"));
        assert_eq!(snap.in_flight["moe"].count, 1);
        assert_eq!(snap.def.min_hold_ms, 2_000, "new timings took effect");
        drop(held);

        // Removing the group entirely stops scheduling rather than blocking.
        sched.reconcile(&routing_with(vec![], vec![backend_def("moe", None)]));
        assert!(sched.group("dgx").is_none());
        let unscheduled = backend_of(&routing_with(vec![], vec![backend_def("moe", None)]), "moe");
        let guard = tokio::time::timeout(Duration::from_secs(2), sched.admit(&unscheduled, 0))
            .await
            .expect("a removed group must not block its former members");
        drop(guard);
    }

    // ---- predictive admission --------------------------------------------

    #[tokio::test(start_paused = true)]
    async fn predicted_wait_is_zero_only_when_the_device_is_actually_free() {
        let sched = Scheduler::new();
        let routing = routing_with(
            vec![group_def(5_000, 30_000, 45_000)],
            vec![grouped_def("moe", "dgx", 2), grouped_def("dense", "dgx", 2)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe");
        let dense = backend_of(&routing, "dense");
        // 1000-token cap at 100 tok/s => 10 s of worst-case occupancy.
        let est = sched.est_drain_ms(&moe, None);
        assert_eq!(est, 10_000);

        // Idle group: nothing to wait for.
        assert_eq!(sched.predicted_wait_ms(&moe, est).await, 0);

        // moe takes the device with a slot to spare — still free for moe…
        let held = sched.admit(&moe, est).await;
        assert_eq!(sched.predicted_wait_ms(&moe, est).await, 0);

        // …but dense now has to wait out moe's in-flight request *and* the
        // rest of the minimum hold. This is exactly the interaction that
        // makes a 2 s interactive deadline shed against a 5 s floor.
        let wait = sched.predicted_wait_ms(&dense, est).await;
        assert!(
            wait >= 5_000,
            "expected at least the remaining hold, got {wait}"
        );
        drop(held);
    }

    #[tokio::test(start_paused = true)]
    async fn predicted_wait_counts_the_queue_ahead() {
        let sched = Scheduler::new();
        let routing = routing_with(
            vec![group_def(10, 100, 45_000)],
            vec![grouped_def("moe", "dgx", 1), grouped_def("dense", "dgx", 1)],
        );
        sched.reconcile(&routing);
        let moe = backend_of(&routing, "moe");
        let dense = backend_of(&routing, "dense");
        let est = sched.est_drain_ms(&moe, None);

        // Fill moe's single slot, then queue three more behind it.
        let held = sched.admit(&moe, est).await;
        let shared = Arc::new(sched);
        let mut waiters = Vec::new();
        for _ in 0..3 {
            let (sched, moe) = (Arc::clone(&shared), Arc::clone(&moe));
            waiters.push(tokio::spawn(async move { sched.admit(&moe, est).await }));
        }
        tokio::time::sleep(Duration::from_millis(5)).await;

        // A fourth moe request would sit behind all three, each of which
        // occupies the single slot for a full request.
        let queued_wait = shared.predicted_wait_ms(&moe, est).await;
        assert!(
            queued_wait >= 3 * est,
            "three waiters ahead at concurrency 1 should cost three requests, got {queued_wait}"
        );

        // dense waits out the request on the device plus one moe epoch —
        // not moe's whole backlog. That gap is the anti-starvation property
        // the epoch rotation exists for: a queue on one member does not
        // push another member to the back of it.
        let dense_wait = shared.predicted_wait_ms(&dense, est).await;
        assert!(dense_wait > 0);
        assert!(
            dense_wait < queued_wait,
            "dense ({dense_wait}) should reach the device before moe's backlog clears ({queued_wait})"
        );

        drop(held);
        for w in waiters {
            drop(
                tokio::time::timeout(Duration::from_secs(60), w)
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }
    }

    #[tokio::test]
    async fn unscheduled_backends_predict_no_wait_at_all() {
        let sched = Scheduler::new();
        let mut embed = backend_def("embed", None);
        embed.scheduling_class = crate::catalog::SchedulingClass::Exempt;
        embed.exclusive_group_id = Some("dgx".into());
        let routing = routing_with(
            vec![group_def(5_000, 30_000, 45_000)],
            vec![
                grouped_def("moe", "dgx", 1),
                embed,
                backend_def("mac", None),
            ],
        );
        sched.reconcile(&routing);

        // Hold the device, then confirm neither the exempt member nor an
        // off-device backend reports any queueing.
        let held = sched.admit(&backend_of(&routing, "moe"), 10_000).await;
        for name in ["embed", "mac"] {
            let backend = backend_of(&routing, name);
            assert_eq!(sched.predicted_wait_ms(&backend, 1_000).await, 0, "{name}");
        }
        drop(held);
    }

    /// The actor holds only a *weak* sender, so a group deleted from the
    /// catalog stops the task rather than leaking it for the life of the
    /// process. Driven directly here because task exit is not observable
    /// through `Scheduler`.
    #[tokio::test(start_paused = true)]
    async fn a_group_actor_exits_once_its_last_sender_is_gone() {
        let (tx, rx) = mpsc::unbounded_channel();
        let state = GroupState {
            id: "dgx".into(),
            def: group_def(50, 200, 1_000),
            members: HashMap::from([("moe".to_string(), Some(1))]),
            phase: Phase::Idle,
            in_flight: HashMap::new(),
            waiters: VecDeque::new(),
            next_ticket: 0,
            tx: tx.downgrade(),
        };
        let task = tokio::spawn(run_group(state, rx));

        // A lease keeps the channel — and so the actor — alive.
        let (reply, granted) = oneshot::channel();
        tx.send(Cmd::Request(Waiter {
            backend: "moe".into(),
            enqueued: Instant::now(),
            est_drain_ms: 10,
            reply,
        }))
        .unwrap();
        let lease = granted.await.unwrap();
        drop(tx);
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(
            !task.is_finished(),
            "an outstanding lease must keep the actor alive"
        );

        drop(lease);
        tokio::time::timeout(Duration::from_secs(5), task)
            .await
            .expect("the actor must exit once nothing can reach it")
            .unwrap();
    }

    #[tokio::test(start_paused = true)]
    async fn exempt_and_unrestricted_backends_never_take_the_lease() {
        let sched = Arc::new(Scheduler::new());
        let mut embed = backend_def("embed", None);
        embed.scheduling_class = crate::catalog::SchedulingClass::Exempt;
        embed.exclusive_group_id = Some("dgx".into());
        let routing = routing_with(
            vec![group_def(60_000, 60_000, 90_000)],
            vec![
                grouped_def("moe", "dgx", 1),
                embed,
                backend_def("mac", None),
            ],
        );
        sched.reconcile(&routing);

        // moe holds the device for a full minute…
        let held = sched.admit(&backend_of(&routing, "moe"), 100).await;
        // …while the embedder on the same device, and an off-device
        // backend, both go straight through.
        for name in ["embed", "mac"] {
            let backend = backend_of(&routing, name);
            let guard = tokio::time::timeout(Duration::from_millis(100), sched.admit(&backend, 0))
                .await
                .unwrap_or_else(|_| panic!("{name} must not wait for the lease"));
            drop(guard);
        }
        // The lease bookkeeping only ever mentions grouped members.
        let snap = sched.group("dgx").unwrap().snapshot().await.unwrap();
        assert_eq!(snap.max_concurrency.keys().collect::<Vec<_>>(), vec!["moe"]);
        drop(held);
    }
}
