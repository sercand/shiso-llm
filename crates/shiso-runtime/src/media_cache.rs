//! A picture is encoded once, however many requests carry it.
//!
//! A client re-sends every picture of a conversation on every turn. An agent
//! that screenshots once a step carries dozens by the tenth. Without this
//! module each turn paid for all of them again: the request thread
//! base64-decoded, decoded and resized every one and hashed its rows, and the
//! engine thread ran the tower over each one the stored prefix did not cover -
//! with every other sequence's decode stalled behind the tower.
//!
//! # Two maps, because the two halves forget at different rates
//!
//! [`MediaCache`] holds the tower's output rows by content digest, on the
//! device, under a byte budget. They are the expensive half, and they are
//! small: one row per placeholder, a few megabytes a picture. It also holds a
//! memo from the request's own payload to that digest and the item's grid.
//! That is a few dozen bytes an entry, so it is kept far longer than the rows
//! are. A request whose picture is in both is never decoded at all. One whose
//! rows were evicted is decoded and encoded again, which is exactly the cost
//! it had before this existed.
//!
//! # Why a queued request holds its rows rather than a digest
//!
//! The request thread decides not to decode a picture because its rows are
//! resident, and the engine thread uses them some time later. An entry
//! evicted in between would leave a request with neither pixels nor rows. So
//! the request thread takes an [`Arc`] of the rows ([`CachedRows`]) instead of
//! remembering that they existed. Eviction drops the cache's reference and
//! never the request's, so the device memory outlives the map entry by
//! exactly as long as a request still needs it.
//!
//! # Why the digest is over the rows the tower reads
//!
//! The payload key only finds the digest again. The digest itself is taken
//! over the decoded values (see [`crate::media::MediaItem::digest`]), because
//! it is also what the prefix cache files a picture's placeholder run under,
//! and that cache outlives this process on disk. A server restarted with a
//! different resize limit decodes the same file to different rows. A digest
//! of the file would then hand the stored prefix of one picture to another.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use shiso_core::DeviceBuffer;
use shiso_core::media::{MediaDigest, VisionGrid};

/// Payload memo entries kept, whatever the rows' budget. At a few dozen bytes
/// each this is a few megabytes of host memory, and it is the part that saves
/// the decode, so it is sized to outlast every row the device can keep.
const MEMO_ENTRIES: usize = 1 << 16;

/// One item's encoder output, resident on the device.
#[derive(Debug)]
pub struct CachedRows {
    rows: DeviceBuffer,
    grid: VisionGrid,
    audio: bool,
}

impl CachedRows {
    /// Wraps one item's rows: `[grid.pad_count(), hidden]` single precision.
    #[must_use]
    pub const fn new(rows: DeviceBuffer, grid: VisionGrid, audio: bool) -> Self {
        Self { rows, grid, audio }
    }

    /// The rows.
    #[must_use]
    pub const fn rows(&self) -> &DeviceBuffer {
        &self.rows
    }

    /// The item's post-merge grid.
    #[must_use]
    pub const fn grid(&self) -> VisionGrid {
        self.grid
    }

    /// Whether the rows came from the audio tower.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        self.audio
    }
}

/// What the payload memo remembers about one payload.
#[derive(Debug, Clone, Copy)]
struct Seen {
    digest: MediaDigest,
    last: u64,
}

/// One resident entry and when it was last used.
#[derive(Debug)]
struct Resident {
    rows: Arc<CachedRows>,
    last: u64,
}

#[derive(Debug, Default)]
struct Inner {
    rows: HashMap<MediaDigest, Resident>,
    bytes: u64,
    memo: HashMap<[u8; 32], Seen>,
    tick: u64,
    hits: u64,
    misses: u64,
}

impl Inner {
    const fn next_tick(&mut self) -> u64 {
        self.tick = self.tick.wrapping_add(1);
        self.tick
    }
}

/// Encoder rows by content digest, and payloads by the digest they decode to.
/// See the module docs.
#[derive(Debug)]
pub struct MediaCache {
    budget: u64,
    inner: Mutex<Inner>,
}

/// A payload's key in the memo: the item kind and the payload as the request
/// spelled it.
///
/// Hashing the request's own text rather than its decoded bytes is what makes
/// a memo hit free: a base64 string is hashed at memory speed, and decoding it
/// is the cost being avoided.
#[must_use]
pub fn payload_key(kind: &str, payload: &str) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(kind.as_bytes());
    hash.update([0u8]);
    hash.update(payload.as_bytes());
    hash.finalize().into()
}

impl MediaCache {
    /// A cache that keeps at most `budget_bytes` of rows on the device.
    #[must_use]
    pub fn new(budget_bytes: u64) -> Self {
        Self {
            budget: budget_bytes,
            inner: Mutex::new(Inner::default()),
        }
    }

    /// The rows a payload decodes to, when both the payload and its rows are
    /// still known. `None` means decode it.
    #[must_use]
    pub fn rows_for_payload(&self, key: &[u8; 32]) -> Option<(MediaDigest, Arc<CachedRows>)> {
        let mut inner = self.inner.lock();
        let tick = inner.next_tick();
        let seen = inner.memo.get_mut(key)?;
        seen.last = tick;
        let digest = seen.digest;
        let found = inner.rows.get_mut(&digest).map(|resident| {
            resident.last = tick;
            Arc::clone(&resident.rows)
        });
        if found.is_some() {
            inner.hits = inner.hits.saturating_add(1);
        } else {
            inner.misses = inner.misses.saturating_add(1);
        }
        found.map(|rows| (digest, rows))
    }

    /// Remembers that the payload under `key` decodes to `digest`.
    pub fn remember_payload(&self, key: [u8; 32], digest: MediaDigest) {
        let mut inner = self.inner.lock();
        let last = inner.next_tick();
        if inner.memo.len() >= MEMO_ENTRIES && !inner.memo.contains_key(&key) {
            // An eighth at a time, oldest first, so the scan that finds them
            // is paid once per several thousand inserts rather than per insert.
            let mut ages: Vec<u64> = inner.memo.values().map(|seen| seen.last).collect();
            let cut = ages.len() / 8;
            let (_, oldest, _) = ages.select_nth_unstable(cut);
            let oldest = *oldest;
            inner.memo.retain(|_, seen| seen.last > oldest);
        }
        inner.memo.insert(key, Seen { digest, last });
    }

    /// The rows for `digest`, when they are resident.
    #[must_use]
    pub fn rows(&self, digest: &MediaDigest) -> Option<Arc<CachedRows>> {
        let mut inner = self.inner.lock();
        let tick = inner.next_tick();
        let found = inner.rows.get_mut(digest).map(|resident| {
            resident.last = tick;
            Arc::clone(&resident.rows)
        });
        if found.is_some() {
            inner.hits = inner.hits.saturating_add(1);
        } else {
            inner.misses = inner.misses.saturating_add(1);
        }
        found
    }

    /// Keeps `rows` under `digest`, evicting the least recently used entries
    /// until they fit, and hands back the shared handle.
    ///
    /// An item larger than the whole budget is handed back without being kept:
    /// holding it would evict everything else for one picture that the next
    /// turn most likely reads from the stored prefix anyway.
    pub fn insert(&self, digest: MediaDigest, rows: CachedRows) -> Arc<CachedRows> {
        let rows = Arc::new(rows);
        let bytes = rows.rows().len();
        if bytes > self.budget {
            return rows;
        }
        let mut inner = self.inner.lock();
        let last = inner.next_tick();
        if let Some(old) = inner.rows.remove(&digest) {
            inner.bytes = inner.bytes.saturating_sub(old.rows.rows().len());
        }
        while inner.bytes.saturating_add(bytes) > self.budget {
            let Some(victim) = inner
                .rows
                .iter()
                .min_by_key(|(_, resident)| resident.last)
                .map(|(digest, _)| *digest)
            else {
                break;
            };
            if let Some(gone) = inner.rows.remove(&victim) {
                inner.bytes = inner.bytes.saturating_sub(gone.rows.rows().len());
            }
        }
        inner.bytes = inner.bytes.saturating_add(bytes);
        inner.rows.insert(
            digest,
            Resident {
                rows: Arc::clone(&rows),
                last,
            },
        );
        rows
    }

    /// Rows resident, their bytes, and the lookups that found or missed them.
    #[must_use]
    pub fn stats(&self) -> MediaCacheStats {
        let inner = self.inner.lock();
        MediaCacheStats {
            entries: inner.rows.len(),
            bytes: inner.bytes,
            budget: self.budget,
            hits: inner.hits,
            misses: inner.misses,
        }
    }
}

/// A snapshot of [`MediaCache`]'s numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MediaCacheStats {
    /// Items whose rows are resident.
    pub entries: usize,
    /// Bytes those rows occupy.
    pub bytes: u64,
    /// The most they may occupy.
    pub budget: u64,
    /// Lookups that found rows.
    pub hits: u64,
    /// Lookups that did not.
    pub misses: u64,
}
