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

/// One guessed expert's bytes, read off the fetch path by a prefetch worker.
#[derive(Debug, Default)]
struct Staged {
    /// Every array of the expert, in layout order, once the read is done.
    bytes: Mutex<Option<CoreResult<Vec<u8>>>>,
    /// Signalled when `bytes` is filled.
    ready: Condvar,
}

/// A guess handed to the prefetch workers.
type PrefetchJob = (u32, u32, Arc<Staged>);

/// The mutable half of a stream.
#[derive(Debug)]
struct State {
    /// `layer * experts + expert` to slot, or [`EMPTY`].
    host_map: Vec<u32>,
    /// Slot to `(layer, expert)`, or `None` for a slot never filled.
    owner: Vec<Option<(u32, u32)>>,
    /// Slot to the fetch that last wanted it.
    last_use: Vec<u64>,
    /// Fetches issued so far, which is the recency clock.
    clock: u64,
    /// Running totals.
    stats: StreamStats,
}

/// A pool of expert slots and the per-layer maps into it.
pub struct ExpertStream {
    /// One pool per array, each `slots` experts long.
    pools: Vec<DeviceBuffer>,
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
    /// Guesses read or being read ahead of their layer's fetch, by
    /// `(layer, expert)`.
    staged: Mutex<HashMap<(u32, u32), Arc<Staged>>>,
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

/// Guesses that may be staged at once: a bound on the host memory the
/// prefetch holds, a few layers' worth of a decode step's misses.
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
            pools,
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
                clock: 0,
                stats: StreamStats::default(),
            }),
            complete: AtomicBool::new(false),
            staged: Mutex::new(HashMap::new()),
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
        staged: &[Option<Vec<u8>>],
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
        let offsets: Vec<usize> = self
            .layout
            .iter()
            .scan(0usize, |at, &bytes| {
                let here = *at;
                *at += usize::try_from(bytes).unwrap_or(usize::MAX);
                Some(here)
            })
            .collect();
        let arrays = views.len();
        for (&expert, _) in missing.iter().zip(victims) {
            views.push(self.map.slice(
                (u64::from(layer) * u64::from(self.experts) + u64::from(expert)) * 4,
                4,
            )?);
        }
        let (source, offsets) = (&self.source, &offsets);
        backend.write_host(&views, &mut |slices| {
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
                                match staged.get(index).and_then(Option::as_ref) {
                                    // A guess already read it: copy, do not read.
                                    Some(bytes) => {
                                        let at = offsets[array];
                                        buf.copy_from_slice(&bytes[at..at + buf.len()]);
                                    }
                                    None => source.read(layer, missing[index], array, buf)?,
                                }
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

    /// Starts reading `guess`'s experts of `layer` that are in no slot, off
    /// the fetch path, so the layer's fetch only copies the ones it then
    /// wants.
    ///
    /// A guess is advice: one the fetch does not want is dropped when that
    /// fetch runs, never mapped, and a read that fails is retried by the
    /// fetch itself. Returns at once; at most [`PREFETCH_CAP`] guesses are
    /// staged at a time, the rest are not read ahead.
    pub fn prefetch(&self, layer: u32, guess: &[u32]) {
        if self.complete.load(Ordering::Acquire) || layer >= self.layers {
            return;
        }
        let wanted: Vec<u32> = {
            let state = self.state.lock();
            let base = layer as usize * self.experts as usize;
            guess
                .iter()
                .copied()
                .filter(|&e| e < self.experts && state.host_map[base + e as usize] == EMPTY)
                .collect()
        };
        if wanted.is_empty() {
            return;
        }
        let mut issued = 0u64;
        {
            let mut staged = self.staged.lock();
            let mut jobs = self.jobs.lock();
            let sender = jobs.get_or_insert_with(|| self.start_prefetchers());
            for expert in wanted {
                if staged.len() >= PREFETCH_CAP || staged.contains_key(&(layer, expert)) {
                    continue;
                }
                let entry = Arc::new(Staged::default());
                if sender.send((layer, expert, Arc::clone(&entry))).is_ok() {
                    staged.insert((layer, expert), entry);
                    issued += 1;
                }
            }
        }
        self.state.lock().stats.prefetch_reads += issued;
    }

    /// The prefetch workers, sharing one queue; they stop when the stream,
    /// and with it the queue's sender, is dropped.
    fn start_prefetchers(&self) -> mpsc::Sender<PrefetchJob> {
        let (sender, receiver) = mpsc::channel::<PrefetchJob>();
        let receiver = Arc::new(Mutex::new(receiver));
        for _ in 0..PREFETCHERS {
            let receiver = Arc::clone(&receiver);
            let source = Arc::clone(&self.source);
            let layout = self.layout.clone();
            std::thread::spawn(move || {
                loop {
                    let job = receiver.lock().recv();
                    let Ok((layer, expert, staged)) = job else {
                        return;
                    };
                    let total = layout
                        .iter()
                        .map(|&b| usize::try_from(b).unwrap_or(0))
                        .sum();
                    let mut bytes = vec![0u8; total];
                    let mut at = 0usize;
                    let mut result = Ok(());
                    for (array, &len) in layout.iter().enumerate() {
                        let len = usize::try_from(len).unwrap_or(0);
                        result = source.read(layer, expert, array, &mut bytes[at..at + len]);
                        if result.is_err() {
                            break;
                        }
                        at += len;
                    }
                    *staged.bytes.lock() = Some(result.map(|()| bytes));
                    staged.ready.notify_all();
                }
            });
        }
        sender
    }

    /// The staged bytes of each of `missing`, waiting for reads still in
    /// flight, and drops every other guess for `layer`: its fetch is now, so
    /// a guess it did not want is a guess that was wrong.
    fn take_staged(&self, layer: u32, missing: &[u32]) -> Vec<Option<Vec<u8>>> {
        let taken: Vec<Option<Arc<Staged>>> = {
            let mut staged = self.staged.lock();
            let taken = missing
                .iter()
                .map(|e| staged.remove(&(layer, *e)))
                .collect();
            staged.retain(|(l, _), _| *l != layer);
            taken
        };
        taken
            .into_iter()
            .map(|entry| {
                let entry = entry?;
                let mut bytes = entry.bytes.lock();
                while bytes.is_none() {
                    entry.ready.wait(&mut bytes);
                }
                // A failed guess is read again by the fetch.
                bytes.take().and_then(Result::ok)
            })
            .collect()
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
            self.fill(backend, layer, &experts, &slots, &[])?;
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
    /// map at it.
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
        if layer >= self.layers {
            return Err(CoreError::OutOfBounds {
                offset: u64::from(layer),
                len: 1,
                buffer_bytes: u64::from(self.layers),
            });
        }
        if self.complete.load(Ordering::Acquire) {
            return Ok(());
        }
        let drained = std::time::Instant::now();
        backend.sync()?;
        let drain = drained.elapsed().as_secs_f64();
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

        let mut state = self.state.lock();
        state.clock += 1;
        state.stats.fetches += 1;
        state.stats.drain_seconds += drain;
        let clock = state.clock;
        let base = layer as usize * self.experts as usize;
        let mut missing: Vec<u32> = Vec::new();
        state.stats.wanted += wanted.len() as u64;
        for &expert in &wanted {
            let slot = state.host_map[base + expert as usize];
            if slot == EMPTY {
                missing.push(expert);
            } else {
                state.stats.hits += 1;
                state.last_use[slot as usize] = clock;
            }
        }
        if missing.is_empty() {
            return Ok(());
        }

        // Victims: never-filled slots first, then the least recently used,
        // none of which this fetch wants (theirs were stamped with `clock`).
        let mut victims: Vec<u32> = Vec::with_capacity(missing.len());
        let mut order: Vec<u32> = (0..self.slots).collect();
        order.sort_by_key(|&slot| {
            (
                state.owner[slot as usize].is_some(),
                state.last_use[slot as usize],
            )
        });
        for slot in order {
            if victims.len() == missing.len() {
                break;
            }
            if state.last_use[slot as usize] != clock {
                victims.push(slot);
            }
        }
        if victims.len() < missing.len() {
            return Err(CoreError::ShapeMismatch {
                field: "expert stream slots free for one layer",
                expected: missing.len() as u64,
                got: victims.len() as u64,
            });
        }
        for (&expert, &slot) in missing.iter().zip(&victims) {
            if let Some((old_layer, old_expert)) = state.owner[slot as usize] {
                let at = old_layer as usize * self.experts as usize + old_expert as usize;
                state.host_map[at] = EMPTY;
            }
            state.owner[slot as usize] = Some((layer, expert));
            state.last_use[slot as usize] = clock;
            state.host_map[base + expert as usize] = slot;
        }

        drop(state);
        let staged = self.take_staged(layer, &missing);
        let from_disk = staged.iter().filter(|s| s.is_none()).count() as u64;
        let started = std::time::Instant::now();
        if let Err(err) = self.fill(backend, layer, &missing, &victims, &staged) {
            // The maps were pointed at the victims before the reads ran. A
            // read that failed leaves its slot holding neither the old expert
            // nor the new one, so every expert of this batch is forgotten on
            // both maps: a later fetch reads it again instead of counting a
            // hit on bytes that never arrived.
            self.forget(backend, layer, &missing, &victims)?;
            return Err(err);
        }
        let mut state = self.state.lock();
        state.stats.reads += from_disk;
        state.stats.prefetched += missing.len() as u64 - from_disk;
        state.stats.read_bytes += from_disk * self.slot_bytes();
        state.stats.read_seconds += started.elapsed().as_secs_f64();
        Ok(())
    }
}
