//! Where the memory went: every major consumer, counted where it allocates.
//!
//! A capacity question - how much context fits, why the pool is the size it
//! is, what a flag would buy - is only answerable from numbers, and the one
//! number a driver offers, free memory, says nothing about who took the rest.
//! So every allocation that matters carries a [`MemTag`] from the call that
//! made it, and a [`MemLedger`] keeps a running total per tag. Reading those
//! totals back out of free memory instead would attribute a page-cache refill
//! or another process's allocation to whichever component happened to
//! allocate next.
//!
//! # Three ways a byte gets here
//!
//! - **Device allocations** made through [`crate::buffer::alloc_tagged`],
//!   [`crate::buffer::alloc`] or [`crate::buffer::import_host`] are counted by
//!   the buffer itself: it adds its size when it is made and subtracts it when
//!   it drops, so the total is live bytes, not bytes ever allocated. A caller
//!   that has no tag of its own inherits the innermost [`scope`], which is
//!   how a loader labels every tensor it imports without threading a tag
//!   through each helper.
//! - **A backend's own slabs** - scratch it grows on demand, staging it keeps
//!   between calls - never become a buffer, so the backend calls
//!   [`MemLedger::record`] and [`MemLedger::release`] itself.
//! - **Host and disk consumers** whose size changes under a lock the ledger
//!   cannot see - a checkpoint store, a spill directory - register a
//!   [`Probe`], which is asked only when someone reads the ledger. That keeps
//!   the cost off every step: nothing here runs per token.
//!
//! # What "unaccounted" means
//!
//! [`LedgerReport::unaccounted`] is what the device says is in use now, less
//! what it said at the first [`MemLedger::mark`], less everything tagged. It
//! is the driver's own context, its module images, captured graphs and
//! whatever else nobody tagged. On a device that shares memory with its host
//! it also absorbs the page cache moving under the process, so it is a number
//! to watch rather than one to trust to the megabyte: a small figure says the
//! tags cover the process, and a large one is a finding.
//!
//! Nothing in this module names a model family. A tag names a *kind* of
//! consumer - routed-expert weights, the key/value pool - that any family with
//! that kind of block has.

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, PoisonError};

use crate::backend::MemoryInfo;

/// Where a tagged consumer's bytes live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Residence {
    /// Device memory.
    Device,
    /// Host memory. On a device that shares memory with its host this comes
    /// out of the same pool as [`Residence::Device`].
    Host,
    /// Files on disk.
    Disk,
    /// Bytes another entry already counts, shown for occupancy: how much of
    /// an allocation is in use, not a second allocation.
    Included,
}

impl Residence {
    /// The label a log line and a metric carry.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Device => "device",
            Self::Host => "host",
            Self::Disk => "disk",
            Self::Included => "included",
        }
    }
}

/// One kind of memory consumer.
///
/// Fieldless and dense, so a tag is an index into the ledger's counters and
/// counting costs one atomic add. [`MemTag::ALL`] lists every variant in the
/// order a report prints them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MemTag {
    /// A device allocation nobody labelled.
    ///
    /// Not an error: most small buffers never need a label. A large figure
    /// here is the cue to label whatever it is.
    Untagged,
    /// Routed and shared expert weights.
    WeightsExperts,
    /// Full-attention blocks: projections, norms, and the block indexer's own
    /// weights.
    WeightsAttention,
    /// Recurrent (linear-attention) blocks.
    WeightsRecurrent,
    /// The weights that mix a hyper-connected residual's streams.
    WeightsStreamMix,
    /// The token embedding and any n-gram embedding kept resident.
    WeightsEmbedding,
    /// The output norm and the vocabulary projection.
    WeightsHead,
    /// Weights no narrower class covers.
    WeightsOther,
    /// A draft head's weights, and anything derived from them.
    DraftWeights,
    /// A vision tower's weights.
    VisionWeights,
    /// The key/value pool's attention planes.
    KvAttention,
    /// The key/value pool's block-indexer planes: the raw key per position
    /// and the pooled key per block.
    KvIndexer,
    /// The key/value pool's planes for a draft head's own attention layer.
    KvDraft,
    /// Pages of the pool that sessions currently hold.
    KvInUse,
    /// Per-session page tables.
    PageTables,
    /// Per-session recurrent state: convolution windows and state matrices.
    RecurrentState,
    /// The per-drafted-row copies of that state that a speculative step rolls
    /// back to.
    RecurrentRollback,
    /// Any other per-session device state.
    SessionOther,
    /// Per-pass activation scratch.
    Activations,
    /// The block indexer's per-pass scores, selections and queries.
    IndexerScratch,
    /// The expert router's and expert launches' per-pass workspace.
    MoeScratch,
    /// The slab cooperative launches share; it grows on demand, so its peak is
    /// the figure to read.
    CoopScratch,
    /// The slab a matrix multiply widens a quantized operand into.
    MatrixStaging,
    /// A matrix library's own workspace.
    BlasWorkspace,
    /// Device staging a streamed embedding table's rows land in.
    NgramStaging,
    /// The host row cache in front of a streamed embedding table.
    NgramRowCache,
    /// Recurrent-state checkpoints held in host memory.
    CheckpointSlots,
    /// Page-locked host staging a weight load copies through.
    UploadStaging,
    /// The on-disk checkpoint tier.
    DiskTier,
}

impl MemTag {
    /// Every tag, in report order.
    pub const ALL: [Self; 29] = [
        Self::WeightsExperts,
        Self::WeightsAttention,
        Self::WeightsRecurrent,
        Self::WeightsStreamMix,
        Self::WeightsEmbedding,
        Self::WeightsHead,
        Self::WeightsOther,
        Self::DraftWeights,
        Self::VisionWeights,
        Self::KvAttention,
        Self::KvIndexer,
        Self::KvDraft,
        Self::KvInUse,
        Self::PageTables,
        Self::RecurrentState,
        Self::RecurrentRollback,
        Self::SessionOther,
        Self::Activations,
        Self::IndexerScratch,
        Self::MoeScratch,
        Self::CoopScratch,
        Self::MatrixStaging,
        Self::BlasWorkspace,
        Self::NgramStaging,
        Self::NgramRowCache,
        Self::CheckpointSlots,
        Self::UploadStaging,
        Self::DiskTier,
        Self::Untagged,
    ];

    /// The dotted name a log line and a metric label carry.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Untagged => "untagged",
            Self::WeightsExperts => "weights.experts",
            Self::WeightsAttention => "weights.attention",
            Self::WeightsRecurrent => "weights.recurrent",
            Self::WeightsStreamMix => "weights.stream_mix",
            Self::WeightsEmbedding => "weights.embedding",
            Self::WeightsHead => "weights.head",
            Self::WeightsOther => "weights.other",
            Self::DraftWeights => "weights.draft",
            Self::VisionWeights => "weights.vision",
            Self::KvAttention => "kv.attention",
            Self::KvIndexer => "kv.indexer",
            Self::KvDraft => "kv.draft",
            Self::KvInUse => "kv.in_use",
            Self::PageTables => "session.page_tables",
            Self::RecurrentState => "session.recurrent",
            Self::RecurrentRollback => "session.rollback",
            Self::SessionOther => "session.other",
            Self::Activations => "scratch.activations",
            Self::IndexerScratch => "scratch.indexer",
            Self::MoeScratch => "scratch.moe",
            Self::CoopScratch => "scratch.cooperative",
            Self::MatrixStaging => "scratch.matrix_staging",
            Self::BlasWorkspace => "scratch.blas",
            Self::NgramStaging => "ngram.staging",
            Self::NgramRowCache => "ngram.row_cache",
            Self::CheckpointSlots => "ckpt.slots",
            Self::UploadStaging => "load.staging",
            Self::DiskTier => "disk.tier",
        }
    }

    /// Where this consumer's bytes live.
    #[must_use]
    pub const fn residence(self) -> Residence {
        match self {
            Self::NgramRowCache | Self::CheckpointSlots | Self::UploadStaging => Residence::Host,
            Self::DiskTier => Residence::Disk,
            Self::KvInUse => Residence::Included,
            _ => Residence::Device,
        }
    }

    /// This tag's counter slot.
    fn index(self) -> usize {
        Self::ALL
            .iter()
            .position(|tag| *tag == self)
            .unwrap_or(Self::ALL.len() - 1)
    }
}

impl fmt::Display for MemTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// What a [`Probe`] reports: bytes in use now, and the ceiling they may grow
/// to when there is one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    /// Bytes in use.
    pub bytes: u64,
    /// Bytes the consumer is allowed, when it has a ceiling.
    pub limit: Option<u64>,
}

/// A consumer the ledger asks for its size instead of counting it.
///
/// Asked only when the ledger is read, so its cost is a scrape's, not a
/// step's. Any `Fn() -> Usage` closure is one.
pub trait Probe: Send + Sync {
    /// The consumer's size now.
    fn usage(&self) -> Usage;
}

impl<F: Fn() -> Usage + Send + Sync> Probe for F {
    fn usage(&self) -> Usage {
        self()
    }
}

/// What the device and the host said at one named moment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mark {
    /// The moment: `before_load`, `after_weights`, `ready`.
    pub label: &'static str,
    /// The backend's own memory figure, when it can give one.
    pub device: Option<MemoryInfo>,
    /// The host kernel's `MemAvailable`, where there is one to read.
    pub host_available: Option<u64>,
}

/// One tag's figures in a [`LedgerReport`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    /// The consumer.
    pub tag: MemTag,
    /// Bytes it holds now.
    pub bytes: u64,
    /// The most it has held at once.
    pub peak: u64,
    /// Live allocations behind `bytes`; zero for a probed consumer.
    pub count: u64,
    /// Its ceiling, when it has one.
    pub limit: Option<u64>,
}

/// Everything the ledger knows at one moment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LedgerReport {
    /// Every consumer that holds or has held anything, in [`MemTag::ALL`]
    /// order.
    pub entries: Vec<Entry>,
    /// Every mark, oldest first.
    pub marks: Vec<Mark>,
    /// The device figure now, when the caller could read one.
    pub now: Option<Mark>,
    /// Whether device and host memory are one pool.
    pub unified: bool,
}

impl LedgerReport {
    /// Bytes tagged in one residence.
    #[must_use]
    pub fn total(&self, residence: Residence) -> u64 {
        self.entries
            .iter()
            .filter(|entry| entry.tag.residence() == residence)
            .map(|entry| entry.bytes)
            .fold(0, u64::saturating_add)
    }

    /// The bytes a tag holds, zero when it holds none.
    #[must_use]
    pub fn bytes(&self, tag: MemTag) -> u64 {
        self.entries
            .iter()
            .find(|entry| entry.tag == tag)
            .map_or(0, |entry| entry.bytes)
    }

    /// Device memory in use now beyond what the first mark saw, less every
    /// tagged byte that comes out of it: the driver's own share and anything
    /// untagged. Negative when the tags overcount, or when memory was
    /// released behind the process's back since the first mark - on a device
    /// that shares memory with its host, a page cache that shrank.
    ///
    /// `None` without a first mark or a figure now.
    #[must_use]
    pub fn unaccounted(&self) -> Option<i64> {
        let base = self.marks.first()?.device?;
        let now = self.now?.device?;
        let grown = i128::from(in_use(now)) - i128::from(in_use(base));
        let mut tagged = i128::from(self.total(Residence::Device));
        if self.unified {
            tagged += i128::from(self.total(Residence::Host));
        }
        i64::try_from(grown - tagged).ok()
    }

    /// Emits one `info` line per consumer, one per mark, and the totals.
    pub fn log(&self) {
        for mark in &self.marks {
            log_mark("memory mark", mark);
        }
        for entry in &self.entries {
            tracing::info!(
                component = entry.tag.name(),
                residence = entry.tag.residence().name(),
                bytes = entry.bytes,
                mib = mib(entry.bytes),
                peak_mib = mib(entry.peak),
                allocations = entry.count,
                limit_mib = entry.limit.map(mib),
                "memory"
            );
        }
        tracing::info!(
            device_mib = mib(self.total(Residence::Device)),
            host_mib = mib(self.total(Residence::Host)),
            disk_mib = mib(self.total(Residence::Disk)),
            unaccounted_mib = self.unaccounted().map(signed_mib),
            unified = self.unified,
            "memory total"
        );
        if let Some(now) = &self.now {
            log_mark("memory now", now);
        }
    }
}

impl fmt::Display for LedgerReport {
    /// A table for a human: one row per consumer, then the totals.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "{:<24} {:>9} {:>12} {:>12} {:>7} {:>12}",
            "component", "where", "MiB", "peak MiB", "allocs", "limit MiB"
        )?;
        for entry in &self.entries {
            let limit = entry
                .limit
                .map_or_else(String::new, |limit| format!("{:.1}", mib(limit)));
            writeln!(
                f,
                "{:<24} {:>9} {:>12.1} {:>12.1} {:>7} {:>12}",
                entry.tag.name(),
                entry.tag.residence().name(),
                mib(entry.bytes),
                mib(entry.peak),
                entry.count,
                limit
            )?;
        }
        writeln!(
            f,
            "total: device {:.1} MiB, host {:.1} MiB, disk {:.1} MiB",
            mib(self.total(Residence::Device)),
            mib(self.total(Residence::Host)),
            mib(self.total(Residence::Disk)),
        )?;
        if let Some(unaccounted) = self.unaccounted() {
            writeln!(f, "unaccounted: {:.1} MiB", signed_mib(unaccounted))?;
        }
        Ok(())
    }
}

fn log_mark(message: &'static str, mark: &Mark) {
    tracing::info!(
        label = mark.label,
        device_total_mib = mark.device.map(|info| mib(info.total)),
        device_free_mib = mark.device.map(|info| mib(info.free)),
        host_available_mib = mark.host_available.map(mib),
        "{message}"
    );
}

const fn in_use(info: MemoryInfo) -> u64 {
    info.total.saturating_sub(info.free)
}

/// Mebibytes, for a log line.
#[expect(
    clippy::cast_precision_loss,
    reason = "a log line in mebibytes; a mantissa's worth of bytes is not the point"
)]
fn mib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 20)
}

/// Signed mebibytes, for the unaccounted figure.
#[expect(
    clippy::cast_precision_loss,
    reason = "a log line in mebibytes; a mantissa's worth of bytes is not the point"
)]
fn signed_mib(bytes: i64) -> f64 {
    bytes as f64 / f64::from(1u32 << 20)
}

/// The host kernel's estimate of memory a new allocation can still get
/// without swapping, in bytes; `None` where there is no `/proc/meminfo`.
#[must_use]
pub fn host_available() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = meminfo
        .lines()
        .find(|line| line.starts_with("MemAvailable:"))?;
    let kib: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    kib.checked_mul(1024)
}

/// A registered probe and the tag it reports under.
struct Probed {
    tag: MemTag,
    probe: Box<dyn Probe>,
}

/// Running totals per [`MemTag`], shared by a backend and whoever reads them.
///
/// Counting is lock-free: an allocation or a free is one atomic add and one
/// atomic max. Only probes and marks sit behind a lock, and both are touched
/// at boot and when the ledger is read.
pub struct MemLedger {
    live: [AtomicU64; MemTag::ALL.len()],
    peak: [AtomicU64; MemTag::ALL.len()],
    count: [AtomicU64; MemTag::ALL.len()],
    probes: Mutex<Vec<Probed>>,
    marks: Mutex<Vec<Mark>>,
    unified: bool,
}

impl fmt::Debug for MemLedger {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemLedger")
            .field("unified", &self.unified)
            .field("device_bytes", &self.counted(Residence::Device))
            .finish_non_exhaustive()
    }
}

impl MemLedger {
    /// An empty ledger. `unified` says device and host memory are one pool,
    /// which decides whether host consumers count against the device.
    #[must_use]
    pub fn new(unified: bool) -> Self {
        Self {
            live: std::array::from_fn(|_| AtomicU64::new(0)),
            peak: std::array::from_fn(|_| AtomicU64::new(0)),
            count: std::array::from_fn(|_| AtomicU64::new(0)),
            probes: Mutex::new(Vec::new()),
            marks: Mutex::new(Vec::new()),
            unified,
        }
    }

    /// Whether device and host memory are one pool.
    #[must_use]
    pub const fn unified(&self) -> bool {
        self.unified
    }

    /// Counts `bytes` newly held under `tag`.
    pub fn record(&self, tag: MemTag, bytes: u64) {
        let slot = tag.index();
        let now = self.live[slot]
            .fetch_add(bytes, Ordering::Relaxed)
            .saturating_add(bytes);
        self.peak[slot].fetch_max(now, Ordering::Relaxed);
        self.count[slot].fetch_add(1, Ordering::Relaxed);
    }

    /// Counts `bytes` under `tag` released.
    ///
    /// Saturates rather than wrapping: a release the ledger never saw
    /// recorded, such as a buffer made before the ledger existed, must not
    /// turn a total into eighteen exabytes.
    pub fn release(&self, tag: MemTag, bytes: u64) {
        let slot = tag.index();
        let _ = self.live[slot].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |live| {
            Some(live.saturating_sub(bytes))
        });
        let _ = self.count[slot].fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
            Some(count.saturating_sub(1))
        });
    }

    /// Bytes counted under `tag` now, probes aside.
    #[must_use]
    pub fn live(&self, tag: MemTag) -> u64 {
        self.live[tag.index()].load(Ordering::Relaxed)
    }

    /// The most ever counted under `tag` at once.
    #[must_use]
    pub fn peak(&self, tag: MemTag) -> u64 {
        self.peak[tag.index()].load(Ordering::Relaxed)
    }

    /// Asks `probe` for `tag`'s size whenever the ledger is read, replacing
    /// any probe already registered for that tag.
    ///
    /// Replacing is what a consumer that is rebuilt - a cache reopened with a
    /// disk tier - needs: the old probe would keep reporting the object that
    /// was dropped.
    pub fn probe(&self, tag: MemTag, probe: impl Probe + 'static) {
        let mut probes = self.probes.lock().unwrap_or_else(PoisonError::into_inner);
        probes.retain(|probed| probed.tag != tag);
        probes.push(Probed {
            tag,
            probe: Box::new(probe),
        });
    }

    /// Records what the device and the host say now, under `label`.
    ///
    /// `device` is the backend's [`crate::Backend::memory_info`]; the caller
    /// reads it because the ledger holds no backend.
    pub fn mark(&self, label: &'static str, device: Option<MemoryInfo>) {
        let mark = Mark {
            label,
            device,
            host_available: host_available(),
        };
        self.marks
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(mark);
    }

    /// Everything the ledger knows, with `device` as the figure now.
    #[must_use]
    pub fn report(&self, device: Option<MemoryInfo>) -> LedgerReport {
        let registered = self.probes.lock().unwrap_or_else(PoisonError::into_inner);
        let entries = MemTag::ALL
            .iter()
            .filter_map(|&tag| {
                let slot = tag.index();
                let probed = registered
                    .iter()
                    .find(|probed| probed.tag == tag)
                    .map(|probed| probed.probe.usage());
                let entry = match probed {
                    Some(usage) => Entry {
                        tag,
                        bytes: usage.bytes,
                        peak: usage.bytes,
                        count: 0,
                        limit: usage.limit,
                    },
                    None => Entry {
                        tag,
                        bytes: self.live[slot].load(Ordering::Relaxed),
                        peak: self.peak[slot].load(Ordering::Relaxed),
                        count: self.count[slot].load(Ordering::Relaxed),
                        limit: None,
                    },
                };
                (probed.is_some() || entry.peak > 0).then_some(entry)
            })
            .collect();
        drop(registered);
        LedgerReport {
            entries,
            marks: self
                .marks
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .clone(),
            now: Some(Mark {
                label: "now",
                device,
                host_available: host_available(),
            }),
            unified: self.unified,
        }
    }

    fn counted(&self, residence: Residence) -> u64 {
        MemTag::ALL
            .iter()
            .filter(|tag| tag.residence() == residence)
            .map(|tag| self.live(*tag))
            .fold(0, u64::saturating_add)
    }
}

impl Default for MemLedger {
    /// An empty ledger for a device with memory of its own.
    fn default() -> Self {
        Self::new(false)
    }
}

thread_local! {
    /// The tag an allocation on this thread gets when its caller names none.
    static SCOPE: Cell<Option<MemTag>> = const { Cell::new(None) };
}

/// While alive, labels every untagged allocation this thread makes.
///
/// Scopes nest: the innermost wins, and dropping one restores the one outside
/// it. Tied to the thread that opened it, so it is neither `Send` nor `Sync`.
#[derive(Debug)]
#[must_use = "a scope labels allocations only while it is alive"]
pub struct TagScope {
    previous: Option<MemTag>,
    _thread: std::marker::PhantomData<*const ()>,
}

impl Drop for TagScope {
    fn drop(&mut self) {
        SCOPE.with(|scope| scope.set(self.previous));
    }
}

/// Labels every untagged allocation this thread makes with `tag` until the
/// returned guard drops.
pub fn scope(tag: MemTag) -> TagScope {
    let previous = SCOPE.with(|scope| scope.replace(Some(tag)));
    TagScope {
        previous,
        _thread: std::marker::PhantomData,
    }
}

/// The tag an untagged allocation on this thread gets now.
#[must_use]
pub fn current() -> MemTag {
    SCOPE.with(Cell::get).unwrap_or(MemTag::Untagged)
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIB: u64 = 1 << 20;

    #[test]
    fn every_tag_has_its_own_slot() {
        for (index, tag) in MemTag::ALL.iter().enumerate() {
            assert_eq!(tag.index(), index, "{tag}");
        }
        let names: std::collections::HashSet<_> = MemTag::ALL.iter().map(|t| t.name()).collect();
        assert_eq!(names.len(), MemTag::ALL.len());
    }

    #[test]
    fn a_release_undoes_a_record_and_the_peak_stays() {
        let ledger = MemLedger::new(false);
        ledger.record(MemTag::KvAttention, 3 * MIB);
        ledger.record(MemTag::KvAttention, MIB);
        ledger.release(MemTag::KvAttention, 3 * MIB);
        assert_eq!(ledger.live(MemTag::KvAttention), MIB);
        assert_eq!(ledger.peak(MemTag::KvAttention), 4 * MIB);
        let report = ledger.report(None);
        let entry = report.entries[0];
        assert_eq!((entry.tag, entry.count), (MemTag::KvAttention, 1));
    }

    #[test]
    fn a_release_never_seen_recorded_saturates_at_zero() {
        let ledger = MemLedger::new(false);
        ledger.release(MemTag::Activations, MIB);
        assert_eq!(ledger.live(MemTag::Activations), 0);
    }

    #[test]
    fn scopes_nest_and_restore() {
        assert_eq!(current(), MemTag::Untagged);
        let outer = scope(MemTag::WeightsOther);
        {
            let _inner = scope(MemTag::WeightsExperts);
            assert_eq!(current(), MemTag::WeightsExperts);
        }
        assert_eq!(current(), MemTag::WeightsOther);
        drop(outer);
        assert_eq!(current(), MemTag::Untagged);
    }

    #[test]
    fn a_probe_is_read_when_the_ledger_is_and_a_second_one_replaces_it() {
        let ledger = MemLedger::new(false);
        ledger.probe(MemTag::DiskTier, || Usage {
            bytes: 7,
            limit: Some(9),
        });
        ledger.probe(MemTag::DiskTier, || Usage {
            bytes: 8,
            limit: Some(9),
        });
        let report = ledger.report(None);
        assert_eq!(report.entries.len(), 1);
        assert_eq!(report.bytes(MemTag::DiskTier), 8);
        assert_eq!(report.total(Residence::Device), 0);
        assert_eq!(report.total(Residence::Disk), 8);
    }

    #[test]
    fn unaccounted_is_growth_since_the_first_mark_less_the_tags() {
        let total = 100 * MIB;
        let ledger = MemLedger::new(true);
        ledger.mark(
            "before_load",
            Some(MemoryInfo {
                free: 90 * MIB,
                total,
            }),
        );
        ledger.record(MemTag::WeightsExperts, 40 * MIB);
        ledger.probe(MemTag::CheckpointSlots, || Usage {
            bytes: 5 * MIB,
            limit: None,
        });
        // 50 MiB more in use; 45 tagged across device and host.
        let report = ledger.report(Some(MemoryInfo {
            free: 40 * MIB,
            total,
        }));
        assert_eq!(report.unaccounted(), Some(5 * 1024 * 1024));
        // On a discrete device the host entry does not come out of the device.
        let discrete = LedgerReport {
            unified: false,
            ..report
        };
        assert_eq!(discrete.unaccounted(), Some(10 * 1024 * 1024));
    }

    #[test]
    fn an_occupancy_entry_is_not_counted_twice() {
        let ledger = MemLedger::new(false);
        ledger.record(MemTag::KvAttention, 10 * MIB);
        ledger.probe(MemTag::KvInUse, || Usage {
            bytes: 4 * MIB,
            limit: Some(10 * MIB),
        });
        assert_eq!(ledger.report(None).total(Residence::Device), 10 * MIB);
    }
}
