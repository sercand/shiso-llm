//! Routed expert banks streamed from disk into a fixed pool of device slots.
//!
//! A sparse checkpoint's routed experts are most of its bytes - 68 GB of this
//! family's 105 - and a token reads a handful of them per layer. When the
//! banks do not fit beside everything else, they are not made resident at
//! all: a pool of slots, each big enough for one expert's matrices, holds the
//! experts that were routed to recently, and a layer's expert stages reach an
//! expert through that layer's expert-to-slot map instead of through the
//! bank's own index.
//!
//! # When an expert is fetched
//!
//! Only after the router has chosen it. A pass that streams its banks carries a
//! [`crate::sparse::plan::Step::Fetch`] between each layer's sort and its expert
//! stages; issuing that step drains the device, reads the layer's routed ids
//! back, brings every chosen expert that is not in a slot in from disk -
//! evicting the least recently used slot that this layer does not need - and
//! writes the new slots into the layer's map before the stages that read it are
//! encoded. Nothing about which expert a column is multiplied by changes: the
//! map sends expert `e` to wherever its bytes are, and the bytes are the
//! checkpoint's own.
//!
//! # Why the source is a trait
//!
//! Where an expert's bytes sit in a checkpoint is a fact about one container,
//! and this crate names no container: [`ExpertSource`] is what a model crate
//! implements over its own files, and everything here is the bookkeeping that
//! would be the same for any of them.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use std::collections::HashMap;
use std::sync::mpsc;

use parking_lot::{Condvar, Mutex};
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc_tagged};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ledger::MemTag;

use crate::sparse::bind::Role;
use crate::sparse::weights::WeightPart;

/// Projections one expert holds: gate, up and down, in that order.
pub const EXPERT_MATRICES: usize = 3;

/// A slot that holds no expert, and a map entry that names no slot.
const EMPTY: u32 = u32::MAX;

/// Where a streamed bank's experts are read from.
///
/// Implemented by the crate that knows the container; called from whichever
/// thread issues a fetch, and from several at once while one fetch reads
/// several experts.
pub trait ExpertSource: Send + Sync + fmt::Debug {
    /// The parts one projection is stored as, in array order: codes, scales
    /// and biases for an affine bank, codes and block scales for a
    /// block-scaled one.
    fn parts(&self) -> &'static [WeightPart];

    /// Projections one expert holds. Three - gate, up and down - unless the
    /// container's experts are ungated, which hold only up and down.
    fn matrices(&self) -> usize {
        EXPERT_MATRICES
    }

    /// Bytes one expert occupies in each array, the same for every layer:
    /// [`ExpertSource::parts`] for each projection in turn, gate (when there
    /// is one), up, then down.
    fn layout(&self) -> Vec<u64>;

    /// Reads array `array` (an index into [`ExpertSource::layout`]) of expert
    /// `expert` of layer `layer` into `dst`, which is exactly that array's
    /// width for one expert.
    ///
    /// # Errors
    ///
    /// [`CoreError::Io`] or [`CoreError::Backend`] for a read that fails or
    /// comes up short.
    fn read(&self, layer: u32, expert: u32, array: usize, dst: &mut [u8]) -> CoreResult<()>;
}

/// What a stream has done since it opened.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StreamStats {
    /// Routed experts a fetch asked for, counting repeats within one fetch once.
    pub wanted: u64,
    /// Of those, experts that were already in a slot.
    pub hits: u64,
    /// Experts read from disk.
    pub reads: u64,
    /// Bytes read from disk.
    pub read_bytes: u64,
    /// Seconds spent reading, wall clock.
    pub read_seconds: f64,
    /// Fetches issued.
    pub fetches: u64,
    /// Seconds spent draining the device before reading the routed ids, which
    /// is the device's own work up to the fetch plus one round trip.
    pub drain_seconds: f64,
    /// Guessed experts a [`ExpertStream::prefetch`] started reading ahead of
    /// their layer's fetch.
    pub prefetch_reads: u64,
    /// Of the experts a fetch found missing, those a prefetch had already
    /// read, so the fetch only copied them in.
    pub prefetched: u64,
}

/// Where a read-ahead's outcome lands, once its worker has written the slot.
#[derive(Debug, Default)]
struct Landing {
    /// `Some` once the read is over.
    result: Mutex<Option<CoreResult<()>>>,
    /// Signalled when `result` is set.
    ready: Condvar,
}

impl Landing {
    /// Waits for the read and takes its outcome.
    fn wait(&self) -> CoreResult<()> {
        let mut result = self.result.lock();
        while result.is_none() {
            self.ready.wait(&mut result);
        }
        result.take().unwrap_or(Ok(()))
    }

    /// Whether the read is over.
    fn done(&self) -> bool {
        self.result.lock().is_some()
    }
}

/// A read-ahead in flight: the slot it was given and where it lands.
#[derive(Debug, Clone)]
struct Reading {
    slot: u32,
    landing: Arc<Landing>,
}

/// A guess handed to the read-ahead workers: layer, expert, slot, landing.
type PrefetchJob = (u32, u32, u32, Arc<Landing>);

/// What [`ExpertStream::plan`] found for one layer's routed experts.
///
/// `ready` experts are in their slots and the layer's map names them now;
/// `missing` ones have slots and map entries assigned but no bytes until
/// [`ExpertStream::complete`] runs, so a pass can multiply the ready ones
/// while the missing ones are read.
#[derive(Debug, Clone, Default)]
pub struct FetchPlan {
    /// The layer.
    pub layer: u32,
    /// Experts a stage may read now.
    pub ready: Vec<u32>,
    /// Experts a stage may read only after [`ExpertStream::complete`].
    pub missing: Vec<u32>,
    /// The slot each of `missing` was given.
    victims: Vec<u32>,
}

/// The mutable half of a stream.
#[derive(Debug)]
struct State {
    /// `layer * experts + expert` to slot, or [`EMPTY`].
    host_map: Vec<u32>,
    /// Slot to `(layer, expert)`, or `None` for a slot never filled.
    owner: Vec<Option<(u32, u32)>>,
    /// Slot to the fetch that last wanted it.
    last_use: Vec<u64>,
    /// Slots a read-ahead is writing, which no fetch may hand out.
    reserved: Vec<bool>,
    /// Fetches issued so far, which is the recency clock.
    clock: u64,
    /// Running totals.
    stats: StreamStats,
}

/// A pool of expert slots and the per-layer maps into it.
pub struct ExpertStream {
    /// One pool per array, each `slots` experts long; shared with the
    /// read-ahead workers, which write slots of it.
    pools: Arc<Vec<DeviceBuffer>>,
    backend: Arc<dyn Backend>,
    /// The device copy of every layer's map, `layers * experts` words.
    map: DeviceBuffer,
    source: Arc<dyn ExpertSource>,
    layout: Vec<u64>,
    parts: &'static [WeightPart],
    layers: u32,
    experts: u32,
    slots: u32,
    state: Mutex<State>,
    /// Set once [`ExpertStream::preload`] has put every expert of every layer
    /// in a slot of its own. From then on no fetch can miss, so a fetch does
    /// nothing at all - in particular it neither drains the device nor reads
    /// the routed ids back, which a captured step cannot do and which cost a
    /// host round trip per layer per step.
    complete: AtomicBool,
    /// Read-aheads in flight, by `(layer, expert)`. Locked before `state`.
    inflight: Mutex<HashMap<(u32, u32), Reading>>,
    /// Read-aheads their layer's fetch did not want, still writing their
    /// slots; a slot is free again once its read is over.
    orphans: Mutex<Vec<Reading>>,
    /// Where guesses go to be read, once the first prefetch started the
    /// workers.
    jobs: Mutex<Option<mpsc::Sender<PrefetchJob>>>,
}

impl fmt::Debug for ExpertStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExpertStream")
            .field("layers", &self.layers)
            .field("experts", &self.experts)
            .field("slots", &self.slots)
            .field("slot_bytes", &self.slot_bytes())
            .field("source", &self.source)
            .finish_non_exhaustive()
    }
}

/// Concurrent reads one fetch may run: enough to keep a fast solid-state
/// device's queue busy with reads of a few hundred kilobytes each.
const READERS: usize = 16;

/// Threads that read guessed experts while the device computes.
const PREFETCHERS: usize = 4;

/// Read-aheads that may be in flight at once: each holds a slot out of the
/// pool until its layer's fetch takes or releases it.
const PREFETCH_CAP: usize = 48;

impl ExpertStream {
    /// Opens a pool of `slots` expert slots over `source`, for `layers`
    /// layers of `experts` experts each.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a pool smaller than one layer's worth
    /// of routed experts at the widest pass - `min_slots` - which a fetch could
    /// not satisfy without evicting an expert the same layer needs; and
    /// whatever the device reports for the allocations.
    pub fn open(
        backend: &Arc<dyn Backend>,
        source: Arc<dyn ExpertSource>,
        layers: u32,
        experts: u32,
        slots: u32,
        min_slots: u32,
    ) -> CoreResult<Self> {
        if slots < min_slots || slots == 0 {
            return Err(CoreError::ShapeMismatch {
                field: "expert stream slots",
                expected: u64::from(min_slots.max(1)),
                got: u64::from(slots),
            });
        }
        let layout = source.layout();
        let parts = source.parts();
        let matrices = source.matrices();
        if layout.len() != parts.len() * matrices {
            return Err(CoreError::ShapeMismatch {
                field: "expert source arrays",
                expected: (parts.len() * matrices) as u64,
                got: layout.len() as u64,
            });
        }
        let pools = layout
            .iter()
            .map(|bytes| {
                alloc_tagged(
                    backend,
                    MemTag::WeightsExperts,
                    bytes.saturating_mul(u64::from(slots)),
                )
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let entries = u64::from(layers) * u64::from(experts);
        let map = alloc_tagged(backend, MemTag::WeightsExperts, entries * 4)?;
        let slots_total = usize::try_from(entries).map_err(|_| CoreError::ShapeMismatch {
            field: "expert map entries",
            expected: usize::MAX as u64,
            got: entries,
        })?;
        let empty: Vec<u8> = std::iter::repeat_n(EMPTY.to_le_bytes(), slots_total)
            .flatten()
            .collect();
        backend.copy_h2d(map.view(), &empty)?;
        let slot_bytes: u64 = layout.iter().sum();
        tracing::info!(
            slots,
            slot_mib = slot_bytes >> 20,
            pool_gib = (slot_bytes * u64::from(slots)) >> 30,
            "opened the routed expert stream"
        );
        Ok(Self {
            pools: Arc::new(pools),
            backend: Arc::clone(backend),
            map,
            source,
            layout,
            parts,
            layers,
            experts,
            slots,
            state: Mutex::new(State {
                host_map: vec![EMPTY; slots_total],
                owner: vec![None; slots as usize],
                last_use: vec![0; slots as usize],
                reserved: vec![false; slots as usize],
                // One past every slot's `last_use`, so before the first fetch no slot
                // reads as wanted by the current one.
                clock: 1,
                stats: StreamStats::default(),
            }),
            complete: AtomicBool::new(false),
            inflight: Mutex::new(HashMap::new()),
            orphans: Mutex::new(Vec::new()),
            jobs: Mutex::new(None),
        })
    }

    /// Bytes one slot occupies across the arrays.
    #[must_use]
    pub fn slot_bytes(&self) -> u64 {
        self.layout.iter().sum()
    }

    /// Slots in the pool.
    #[must_use]
    pub const fn slots(&self) -> u32 {
        self.slots
    }

    /// What the stream has done so far.
    #[must_use]
    pub fn stats(&self) -> StreamStats {
        self.state.lock().stats
    }

    /// The pool array an expert role and part resolve to.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for a role that is not one of the three
    /// routed banks or a part an affine bank does not have.
    pub fn bank(&self, role: Role, part: WeightPart) -> CoreResult<DeviceView<'_>> {
        let matrix = match role {
            Role::ExpertGate => 0,
            Role::ExpertUp => 1,
            Role::ExpertDown => 2,
            _ => {
                return Err(CoreError::Unsupported {
                    what: "a streamed tensor that is not one of the routed banks",
                    backend: "engine",
                });
            }
        };
        let array =
            self.parts
                .iter()
                .position(|have| *have == part)
                .ok_or(CoreError::Unsupported {
                    what: "a part the streamed bank's format does not have",
                    backend: "engine",
                })?;
        Ok(self.pools[matrix * self.parts.len() + array].view())
    }

    /// The pool array of projection `matrix` (in the source's order) and
    /// part `part`: what a stage over a streamed bank reads through the map.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] for a projection past the source's last or a
    /// part its format does not have.
    pub fn pool(&self, matrix: usize, part: WeightPart) -> CoreResult<DeviceView<'_>> {
        let array = self
            .parts
            .iter()
            .position(|have| *have == part)
            .filter(|_| matrix < self.layout.len() / self.parts.len())
            .ok_or(CoreError::Unsupported {
                what: "a projection or part the streamed bank does not have",
                backend: "engine",
            })?;
        Ok(self.pools[matrix * self.parts.len() + array].view())
    }

    /// Bytes one slot occupies in each array, in the source's order.
    #[must_use]
    pub fn slot_layout(&self) -> &[u64] {
        &self.layout
    }

    /// One layer's slice of the device map.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] for a layer past the last.
    pub fn map(&self, layer: u32) -> CoreResult<DeviceView<'_>> {
        let width = u64::from(self.experts) * 4;
        self.map.slice(u64::from(layer) * width, width)
    }

    /// Reads `missing[i]` into slot `victims[i]` and points the layer's map at
    /// it, every read side by side and straight into the pool.
    fn fill(
        &self,
        backend: &dyn Backend,
        layer: u32,
        missing: &[u32],
        victims: &[u32],
        live: bool,
    ) -> CoreResult<()> {
        // Every array of every missing expert, and its map entry, in one
        // batch of writable ranges: the reads land where the stages read
        // them, and they run side by side because a fast solid-state device
        // only reaches its bandwidth with many reads in flight.
        let mut views = Vec::with_capacity(missing.len() * (self.layout.len() + 1));
        let mut jobs: Vec<(usize, usize)> = Vec::with_capacity(missing.len() * self.layout.len());
        for (index, &slot) in victims.iter().enumerate().take(missing.len()) {
            for (array, (pool, &bytes)) in self.pools.iter().zip(&self.layout).enumerate() {
                views.push(pool.slice(u64::from(slot) * bytes, bytes)?);
                jobs.push((index, array));
            }
        }

        let arrays = views.len();
        for (&expert, _) in missing.iter().zip(victims) {
            views.push(self.map.slice(
                (u64::from(layer) * u64::from(self.experts) + u64::from(expert)) * 4,
                4,
            )?);
        }
        let source = &self.source;
        let write = if live {
            Backend::write_host_live
        } else {
            Backend::write_host
        };
        write(backend, &views, &mut |slices| {
            let (data, entries) = slices.split_at_mut(arrays);
            for (entry, &slot) in entries.iter_mut().zip(victims) {
                entry.copy_from_slice(&slot.to_le_bytes());
            }
            let per = data.len().div_ceil(READERS).max(1);
            std::thread::scope(|scope| -> CoreResult<()> {
                let handles: Vec<_> = data
                    .chunks_mut(per)
                    .zip(jobs.chunks(per))
                    .map(|(bufs, work)| {
                        scope.spawn(move || -> CoreResult<()> {
                            for (buf, &(index, array)) in bufs.iter_mut().zip(work) {
                                source.read(layer, missing[index], array, buf)?;
                            }
                            Ok(())
                        })
                    })
                    .collect();
                for handle in handles {
                    handle.join().map_err(|_| CoreError::Backend {
                        backend: "engine",
                        op: "reading a routed expert",
                        source: "the reader thread panicked".into(),
                    })??;
                }
                Ok(())
            })
        })
    }

    /// Slots a batch of `n` new experts may take: never-filled ones first,
    /// then the least recently used, none that the fetch at `clock` wants and
    /// none a read-ahead is writing.
    fn victims(&self, state: &State, n: usize, clock: u64) -> Vec<u32> {
        let mut order: Vec<u32> = (0..self.slots)
            .filter(|&slot| {
                !state.reserved[slot as usize] && state.last_use[slot as usize] != clock
            })
            .collect();
        order.sort_by_key(|&slot| {
            (
                state.owner[slot as usize].is_some(),
                state.last_use[slot as usize],
            )
        });
        order.truncate(n);
        order
    }

    /// Starts reading `guess`'s experts of `layer` that are in no slot
    /// straight into slots reserved for them, off the fetch path, so the
    /// layer's fetch finds them already there.
    ///
    /// A reserved slot is one no queued work reads - it is none of the slots
    /// the fetch that just ran stamped - so the read writes it while the
    /// device computes. A guess the layer's fetch does not want is released
    /// once its read is over, never mapped; a read that fails is retried by
    /// the fetch itself. Returns at once; at most [`PREFETCH_CAP`] reads are
    /// in flight.
    pub fn prefetch(&self, layer: u32, guess: &[u32]) {
        if self.complete.load(Ordering::Acquire) || layer >= self.layers {
            return;
        }
        let mut inflight = self.inflight.lock();
        let mut state = self.state.lock();
        let base = layer as usize * self.experts as usize;
        let room = PREFETCH_CAP.saturating_sub(inflight.len());
        let wanted: Vec<u32> = guess
            .iter()
            .copied()
            .filter(|&e| {
                e < self.experts
                    && state.host_map[base + e as usize] == EMPTY
                    && !inflight.contains_key(&(layer, e))
            })
            .take(room)
            .collect();
        if wanted.is_empty() {
            return;
        }
        let clock = state.clock;
        let slots = self.victims(&state, wanted.len(), clock);
        let mut jobs = self.jobs.lock();
        let sender = jobs.get_or_insert_with(|| self.start_prefetchers());
        let mut issued = 0u64;
        for (&expert, &slot) in wanted.iter().zip(&slots) {
            let landing = Arc::new(Landing::default());
            if sender
                .send((layer, expert, slot, Arc::clone(&landing)))
                .is_err()
            {
                break;
            }
            if let Some((old_layer, old_expert)) = state.owner[slot as usize].take() {
                let at = old_layer as usize * self.experts as usize + old_expert as usize;
                state.host_map[at] = EMPTY;
            }
            state.reserved[slot as usize] = true;
            inflight.insert((layer, expert), Reading { slot, landing });
            issued += 1;
        }
        state.stats.prefetch_reads += issued;
    }

    /// The read-ahead workers, sharing one queue; they stop when the stream,
    /// and with it the queue's sender, is dropped. Each writes its expert's
    /// arrays straight into the slot it was given.
    fn start_prefetchers(&self) -> mpsc::Sender<PrefetchJob> {
        let (sender, receiver) = mpsc::channel::<PrefetchJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        for _ in 0..PREFETCHERS {
            let receiver = Arc::clone(&receiver);
            let source = Arc::clone(&self.source);
            let pools = Arc::clone(&self.pools);
            let backend = Arc::clone(&self.backend);
            let layout = self.layout.clone();
            std::thread::spawn(move || {
                loop {
                    let job = receiver.lock().recv();
                    let Ok((layer, expert, slot, landing)) = job else {
                        return;
                    };
                    let result = (|| -> CoreResult<()> {
                        let views = pools
                            .iter()
                            .zip(&layout)
                            .map(|(pool, &bytes)| pool.slice(u64::from(slot) * bytes, bytes))
                            .collect::<CoreResult<Vec<_>>>()?;
                        backend.write_host_live(&views, &mut |slices| {
                            for (array, buf) in slices.iter_mut().enumerate() {
                                source.read(layer, expert, array, buf)?;
                            }
                            Ok(())
                        })
                    })();
                    *landing.result.lock() = Some(result);
                    landing.ready.notify_all();
                }
            });
        }
        sender
    }

    /// The distinct experts of the `count` routed ids at `ids`, in order.
    fn routed_ids(
        &self,
        backend: &dyn Backend,
        ids: DeviceView<'_>,
        count: u32,
    ) -> CoreResult<Vec<u32>> {
        let mut raw = vec![0u8; count as usize * 4];
        backend.copy_d2h(ids.slice(0, raw.len() as u64)?, &mut raw)?;
        let mut wanted: Vec<u32> = Vec::with_capacity(count as usize);
        for word in raw.as_chunks::<4>().0 {
            let id = i32::from_le_bytes(*word);
            let expert = u32::try_from(id).ok().filter(|e| *e < self.experts).ok_or(
                CoreError::OutOfBounds {
                    offset: u64::try_from(id).unwrap_or(u64::MAX),
                    len: 1,
                    buffer_bytes: u64::from(self.experts),
                },
            )?;
            if !wanted.contains(&expert) {
                wanted.push(expert);
            }
        }
        Ok(wanted)
    }

    /// Points `layer`'s map at the slots read-aheads `landed` in, before any
    /// stage reads them; the device is idle, so a plain write is as cheap as
    /// any.
    fn write_landed(
        &self,
        backend: &dyn Backend,
        layer: u32,
        landed: &[(u32, u32)],
    ) -> CoreResult<()> {
        if landed.is_empty() {
            return Ok(());
        }
        let entries = landed
            .iter()
            .map(|&(expert, _)| {
                self.map.slice(
                    (u64::from(layer) * u64::from(self.experts) + u64::from(expert)) * 4,
                    4,
                )
            })
            .collect::<CoreResult<Vec<_>>>()?;
        backend.write_host(&entries, &mut |slices| {
            for (entry, &(_, slot)) in slices.iter_mut().zip(landed) {
                entry.copy_from_slice(&slot.to_le_bytes());
            }
            Ok(())
        })
    }

    /// Frees the slots of orphaned read-aheads whose reads are over.
    fn release_orphans(&self) {
        let mut orphans = self.orphans.lock();
        if orphans.is_empty() {
            return;
        }
        let mut state = self.state.lock();
        orphans.retain(|reading| {
            if reading.landing.done() {
                state.reserved[reading.slot as usize] = false;
                false
            } else {
                true
            }
        });
    }

    /// Unmaps `missing` from `layer` and frees `victims`, on both maps.
    fn forget(
        &self,
        backend: &dyn Backend,
        layer: u32,
        missing: &[u32],
        victims: &[u32],
    ) -> CoreResult<()> {
        let mut state = self.state.lock();
        let base = layer as usize * self.experts as usize;
        for (&expert, &slot) in missing.iter().zip(victims) {
            state.host_map[base + expert as usize] = EMPTY;
            state.owner[slot as usize] = None;
            state.last_use[slot as usize] = 0;
        }
        drop(state);
        for &expert in missing {
            backend.copy_h2d(
                self.map.slice(
                    (u64::from(layer) * u64::from(self.experts) + u64::from(expert)) * 4,
                    4,
                )?,
                &EMPTY.to_le_bytes(),
            )?;
        }
        Ok(())
    }

    /// Fills a pool that holds every expert of every layer, once: expert `e`
    /// of layer `l` in slot `l * experts + e`, so no fetch ever reads again.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for a pool smaller than every expert, and
    /// whatever the reads and the device report.
    pub fn preload(&self, backend: &dyn Backend) -> CoreResult<()> {
        let all = u64::from(self.layers) * u64::from(self.experts);
        if u64::from(self.slots) < all {
            return Err(CoreError::ShapeMismatch {
                field: "expert slots for a full preload",
                expected: all,
                got: u64::from(self.slots),
            });
        }
        let started = std::time::Instant::now();
        let experts: Vec<u32> = (0..self.experts).collect();
        for layer in 0..self.layers {
            let slots: Vec<u32> = experts.iter().map(|e| layer * self.experts + e).collect();
            self.fill(backend, layer, &experts, &slots, false)?;
            let mut state = self.state.lock();
            for (&e, &slot) in experts.iter().zip(&slots) {
                state.host_map[(layer * self.experts + e) as usize] = slot;
                state.owner[slot as usize] = Some((layer, e));
            }
        }
        self.complete.store(true, Ordering::Release);
        tracing::info!(
            seconds = started.elapsed().as_secs_f64(),
            "every routed expert is resident"
        );
        Ok(())
    }

    /// Makes every expert `ids` names resident in a slot and points `layer`'s
    /// map at it: [`ExpertStream::plan`], then [`ExpertStream::complete`].
    ///
    /// `ids` is the layer's routed id list on the device, `count` entries of
    /// `i32`; the device is drained first so the list is the one this pass
    /// wrote.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] for a layer or an id outside the bank,
    /// whatever the device reports for the transfers, and whatever the source
    /// reports for a read.
    pub fn fetch(
        &self,
        backend: &dyn Backend,
        layer: u32,
        ids: DeviceView<'_>,
        count: u32,
    ) -> CoreResult<()> {
        let plan = self.plan(backend, layer, ids, count)?;
        self.complete(backend, &plan)
    }

    /// The first half of a fetch: drains the device, reads the routed ids
    /// back, takes what read-aheads landed, and gives every other missing
    /// expert a slot and a map entry - without reading it yet.
    ///
    /// The plan's `ready` experts may be multiplied as soon as this returns;
    /// its `missing` ones only after [`ExpertStream::complete`]. Their slots
    /// are none that the ready experts use, so the two can overlap.
    ///
    /// # Errors
    ///
    /// As [`ExpertStream::fetch`].
    pub fn plan(
        &self,
        backend: &dyn Backend,
        layer: u32,
        ids: DeviceView<'_>,
        count: u32,
    ) -> CoreResult<FetchPlan> {
        if layer >= self.layers {
            return Err(CoreError::OutOfBounds {
                offset: u64::from(layer),
                len: 1,
                buffer_bytes: u64::from(self.layers),
            });
        }
        let mut plan = FetchPlan {
            layer,
            ..FetchPlan::default()
        };
        if self.complete.load(Ordering::Acquire) {
            return Ok(plan);
        }
        let drained = std::time::Instant::now();
        backend.sync()?;
        let drain = drained.elapsed().as_secs_f64();
        let wanted = self.routed_ids(backend, ids, count)?;
        self.release_orphans();
        // Every read-ahead of this layer comes off the books now: the wanted
        // ones are waited for, the others are orphaned.
        let mut taken: HashMap<u32, Reading> = {
            let mut inflight = self.inflight.lock();
            let keys: Vec<(u32, u32)> = inflight
                .keys()
                .copied()
                .filter(|(l, _)| *l == layer)
                .collect();
            keys.into_iter()
                .filter_map(|key| inflight.remove(&key).map(|r| (key.1, r)))
                .collect()
        };
        let mut state = self.state.lock();
        state.clock += 1;
        state.stats.fetches += 1;
        state.stats.drain_seconds += drain;
        let clock = state.clock;
        let base = layer as usize * self.experts as usize;
        state.stats.wanted += wanted.len() as u64;
        let mut landing: Vec<(u32, Reading)> = Vec::new();
        for &expert in &wanted {
            let slot = state.host_map[base + expert as usize];
            if slot != EMPTY {
                state.stats.hits += 1;
                state.last_use[slot as usize] = clock;
                plan.ready.push(expert);
            } else if let Some(reading) = taken.remove(&expert) {
                // Stamped now, so no victim choice below can take it.
                state.last_use[reading.slot as usize] = clock;
                landing.push((expert, reading));
            } else {
                plan.missing.push(expert);
            }
        }
        drop(state);
        let mut landed: Vec<(u32, u32)> = Vec::new();
        let mut failed: Vec<u32> = Vec::new();
        for (expert, reading) in landing {
            if reading.landing.wait().is_ok() {
                landed.push((expert, reading.slot));
            } else {
                failed.push(reading.slot);
                plan.missing.push(expert);
            }
        }
        {
            let mut orphans = self.orphans.lock();
            orphans.extend(taken.into_values());
        }
        let mut state = self.state.lock();
        for &slot in &failed {
            state.reserved[slot as usize] = false;
        }
        for &(expert, slot) in &landed {
            state.reserved[slot as usize] = false;
            state.owner[slot as usize] = Some((layer, expert));
            state.host_map[base + expert as usize] = slot;
            state.stats.prefetched += 1;
            plan.ready.push(expert);
        }
        let victims = self.victims(&state, plan.missing.len(), clock);
        if victims.len() < plan.missing.len() {
            return Err(CoreError::ShapeMismatch {
                field: "expert stream slots free for one layer",
                expected: plan.missing.len() as u64,
                got: victims.len() as u64,
            });
        }
        for (&expert, &slot) in plan.missing.iter().zip(&victims) {
            if let Some((old_layer, old_expert)) = state.owner[slot as usize] {
                let at = old_layer as usize * self.experts as usize + old_expert as usize;
                state.host_map[at] = EMPTY;
            }
            state.owner[slot as usize] = Some((layer, expert));
            state.last_use[slot as usize] = clock;
            state.host_map[base + expert as usize] = slot;
        }
        drop(state);
        plan.victims = victims;
        self.write_landed(backend, layer, &landed)?;
        Ok(plan)
    }

    /// The second half of a fetch: reads `plan`'s missing experts into their
    /// slots and points the map at them, without draining the device - the
    /// slots and map entries are ones no queued work reads.
    ///
    /// # Errors
    ///
    /// Whatever the source reports for a read, after which the plan's
    /// missing experts are forgotten on both maps.
    pub fn complete(&self, backend: &dyn Backend, plan: &FetchPlan) -> CoreResult<()> {
        if plan.missing.is_empty() {
            return Ok(());
        }
        let started = std::time::Instant::now();
        if let Err(err) = self.fill(backend, plan.layer, &plan.missing, &plan.victims, true) {
            // The maps were pointed at the victims before the reads ran. A
            // read that failed leaves its slot holding neither the old expert
            // nor the new one, so every expert of this batch is forgotten on
            // both maps: a later fetch reads it again instead of counting a
            // hit on bytes that never arrived.
            self.forget(backend, plan.layer, &plan.missing, &plan.victims)?;
            return Err(err);
        }
        let mut state = self.state.lock();
        state.stats.reads += plan.missing.len() as u64;
        state.stats.read_bytes += plan.missing.len() as u64 * self.slot_bytes();
        state.stats.read_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }
}
