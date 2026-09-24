//! What the store holds, in memory: every page and entry, their states, and the
//! budget arithmetic over them.
//!
//! The directory is the truth and this is a cache of it, rebuilt by a scan at
//! open. Nothing here is persisted separately - not the reference counts, not a
//! manifest - because a counter file is one more thing a crash can leave
//! disagreeing with the bytes, and recounting from the entries costs one header
//! read per entry.
//!
//! # States
//!
//! A page is `Queued` from the moment an entry that needs it is accepted until
//! some job copies it off the device, `Sent` while its bytes are with the
//! writer, and `Durable` once its file has been renamed into place. An entry is
//! `Pending` until its own file lands and `Durable` after. Only durable entries
//! are found by a lookup: a pending one is still in the tree the job came from,
//! and the tree answers faster.
//!
//! # References
//!
//! A page's count is the number of entries, pending or durable, that list it.
//! The last entry to go deletes the page, which is the whole garbage collector:
//! there is no sweep that could race a write.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use crate::store::format::{
    self, ENTRIES_DIR, ENTRY_SUFFIX, PAGE_SUFFIX, PAGES_DIR, TMP_MARKER, Unusable,
};
use crate::store::key::{EntryKey, Identity, PageKey, PathKeys};

/// Where a page is on its way to disk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageState {
    /// Accepted; nobody has copied its rows off the device yet.
    Queued,
    /// Copied; the writer has the bytes.
    Sent,
    /// On disk under its name.
    Durable,
}

/// One page as the index sees it.
#[derive(Debug, Clone)]
pub(crate) struct PageMeta {
    /// Entries listing this page.
    pub(crate) refs: u32,
    /// File bytes: exact once durable, the reservation before.
    pub(crate) bytes: u64,
    pub(crate) state: PageState,
}

/// One entry as the index sees it.
#[derive(Debug, Clone)]
pub(crate) struct EntryMeta {
    pub(crate) pos: u32,
    pub(crate) pages: Vec<PageKey>,
    pub(crate) has_logits: bool,
    /// File bytes: exact once durable, the reservation before.
    pub(crate) bytes: u64,
    pub(crate) created_at: u64,
    pub(crate) last_used: u64,
    pub(crate) hits: u32,
    pub(crate) durable: bool,
    /// Written by another build: kept only until its bytes are wanted.
    pub(crate) foreign: bool,
}

/// A durable entry a lookup found.
#[derive(Debug, Clone)]
pub(crate) struct Hit {
    pub(crate) key: EntryKey,
    pub(crate) pos: u32,
    pub(crate) pages: Vec<PageKey>,
    /// Bytes a restore reads: the entry file and every page file.
    pub(crate) bytes: u64,
}

/// What accepting an entry did.
#[derive(Debug, Default)]
pub(crate) struct Accepted {
    /// Files the budget pass released, for the writer to unlink.
    pub(crate) deletions: Vec<PathBuf>,
    /// Entries the budget pass released.
    pub(crate) evicted: u32,
}

/// Why an entry was not accepted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum NotAccepted {
    /// The same entry is already pending or on disk.
    AlreadyStored,
    /// It would not fit even after every releasable entry went.
    OverBudget { required: u64, budget: u64 },
}

/// What an entry costs to accept, in the index's own terms.
#[derive(Debug, Clone)]
pub(crate) struct Offer<'a> {
    pub(crate) key: EntryKey,
    pub(crate) pos: u32,
    pub(crate) pages: &'a [PageKey],
    pub(crate) has_logits: bool,
    pub(crate) entry_bytes: u64,
    /// The reservation for one page file not already known to the index.
    pub(crate) page_file_bytes: u64,
    pub(crate) now: u64,
}

/// How the budget pass prices an entry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Pricing {
    pub(crate) page_tokens: u32,
    pub(crate) prefill_s_per_token: f64,
    /// Seconds of idleness that halve an entry's standing.
    pub(crate) decay_s: f64,
}

/// The index.
#[derive(Debug)]
pub(crate) struct Index {
    root: PathBuf,
    pub(crate) pages: HashMap<PageKey, PageMeta>,
    pub(crate) entries: HashMap<EntryKey, EntryMeta>,
    /// Distinct positions of durable entries of this build, with how many
    /// entries sit at each: the candidates a lookup hashes.
    by_pos: BTreeMap<u32, u32>,
    /// Bytes of every file on disk that this index knows.
    pub(crate) durable_bytes: u64,
    /// Bytes promised to writes that have not landed.
    pub(crate) reserved_bytes: u64,
    pub(crate) budget: u64,
    pricing: Pricing,
    /// Files a refused acceptance released on its way to refusing.
    orphaned: Vec<PathBuf>,
}

impl Index {
    /// An empty index over `root`.
    #[must_use]
    pub(crate) fn empty(root: &Path, budget: u64, pricing: Pricing) -> Self {
        Self {
            root: root.to_path_buf(),
            pages: HashMap::new(),
            entries: HashMap::new(),
            by_pos: BTreeMap::new(),
            durable_bytes: 0,
            reserved_bytes: 0,
            budget,
            pricing,
            orphaned: Vec::new(),
        }
    }

    /// The file a page lives in.
    #[must_use]
    pub(crate) fn page_path(&self, key: &PageKey) -> PathBuf {
        format::object_path(&self.root, PAGES_DIR, &key.to_string(), PAGE_SUFFIX)
    }

    /// The file an entry lives in.
    #[must_use]
    pub(crate) fn entry_path(&self, key: &EntryKey) -> PathBuf {
        format::object_path(&self.root, ENTRIES_DIR, &key.to_string(), ENTRY_SUFFIX)
    }

    /// Durable entries of this build.
    #[must_use]
    pub(crate) fn usable_entries(&self) -> u32 {
        self.by_pos.values().copied().sum()
    }

    /// Entries accepted and not yet on disk.
    #[must_use]
    pub(crate) fn pending_entries(&self) -> u32 {
        u32::try_from(self.entries.values().filter(|entry| !entry.durable).count())
            .unwrap_or(u32::MAX)
    }

    /// Whether an entry is accepted, pending or durable.
    #[must_use]
    pub(crate) fn knows(&self, key: &EntryKey) -> bool {
        self.entries.contains_key(key)
    }

    /// The deepest durable entry of this build on `keys`' path, at or before
    /// `len` and at or above `floor`.
    ///
    /// An entry exactly at `len` is only usable with its logits row, because a
    /// resume there answers the next token from that row; one short of `len` is
    /// usable only if `continuable` says a longer prompt may stand on it (see
    /// [`crate::CacheGeometry::continuable`]).
    #[must_use]
    pub(crate) fn lookup(
        &self,
        keys: &PathKeys<'_>,
        len: u32,
        floor: u32,
        continuable: impl Fn(u32, bool) -> bool,
    ) -> Option<Hit> {
        if floor.max(1) > len {
            return None;
        }
        for (&pos, _) in self.by_pos.range(floor.max(1)..=len).rev() {
            let Some(key) = keys.entry_at(pos) else {
                continue;
            };
            let Some(entry) = self.entries.get(&key) else {
                continue;
            };
            if !entry.durable
                || entry.foreign
                || (pos == len && !entry.has_logits)
                || (pos < len && !continuable(pos, entry.has_logits))
            {
                continue;
            }
            let page_bytes: u64 = entry
                .pages
                .iter()
                .filter_map(|page| self.pages.get(page))
                .map(|page| page.bytes)
                .sum();
            return Some(Hit {
                key,
                pos,
                pages: entry.pages.clone(),
                bytes: entry.bytes.saturating_add(page_bytes),
            });
        }
        None
    }

    /// Accepts an entry: reserves its bytes, releasing lower-valued entries to
    /// make room, and registers it and its new pages as pending.
    ///
    /// # Errors
    ///
    /// [`NotAccepted`] when the entry is already known or cannot fit.
    pub(crate) fn accept(&mut self, offer: &Offer<'_>) -> Result<Accepted, NotAccepted> {
        if self.entries.contains_key(&offer.key) {
            return Err(NotAccepted::AlreadyStored);
        }
        let unique: HashSet<PageKey> = offer.pages.iter().copied().collect();
        // The references on pages the index already holds are taken before the
        // budget pass runs, so releasing some other entry cannot delete a page
        // this one is about to list.
        let known: Vec<PageKey> = unique
            .iter()
            .filter(|page| self.pages.contains_key(*page))
            .copied()
            .collect();
        for page in &known {
            if let Some(meta) = self.pages.get_mut(page) {
                meta.refs += 1;
            }
        }
        let new_pages = unique.len() - known.len();
        let required = offer.entry_bytes.saturating_add(
            offer
                .page_file_bytes
                .saturating_mul(u64::try_from(new_pages).unwrap_or(u64::MAX)),
        );
        let mut accepted = Accepted::default();
        let refusal = NotAccepted::OverBudget {
            required,
            budget: self.budget,
        };
        let fits = required <= self.budget && {
            loop {
                if self.durable_bytes + self.reserved_bytes + required <= self.budget {
                    break true;
                }
                let Some(victim) = self.victim(offer.now) else {
                    break false;
                };
                accepted.deletions.extend(self.release(&victim));
                accepted.evicted += 1;
            }
        };
        if !fits {
            for page in &known {
                if let Some(meta) = self.pages.get_mut(page) {
                    meta.refs = meta.refs.saturating_sub(1);
                }
            }
            // Whatever the pass released stays released; the caller still has
            // to unlink it.
            self.orphaned.append(&mut accepted.deletions);
            return Err(refusal);
        }
        for page in unique {
            if known.contains(&page) {
                continue;
            }
            self.pages.insert(
                page,
                PageMeta {
                    refs: 1,
                    bytes: offer.page_file_bytes,
                    state: PageState::Queued,
                },
            );
            self.reserved_bytes += offer.page_file_bytes;
        }
        self.reserved_bytes += offer.entry_bytes;
        self.entries.insert(
            offer.key,
            EntryMeta {
                pos: offer.pos,
                pages: offer.pages.to_vec(),
                has_logits: offer.has_logits,
                bytes: offer.entry_bytes,
                created_at: offer.now,
                last_used: 0,
                hits: 0,
                durable: false,
                foreign: false,
            },
        );
        Ok(accepted)
    }

    /// Files released by a budget pass whose entry was then refused anyway,
    /// for the caller to unlink.
    pub(crate) fn take_orphaned(&mut self) -> Vec<PathBuf> {
        std::mem::take(&mut self.orphaned)
    }

    /// Marks a queued page as copied. `false` when some other job already
    /// copied it, it is already durable, or it was released meanwhile.
    pub(crate) fn claim_page(&mut self, key: &PageKey) -> bool {
        match self.pages.get_mut(key) {
            Some(meta) if meta.state == PageState::Queued => {
                meta.state = PageState::Sent;
                true
            }
            _ => false,
        }
    }

    /// Puts back a claim whose copy did not happen.
    pub(crate) fn unclaim_page(&mut self, key: &PageKey) {
        if let Some(meta) = self.pages.get_mut(key)
            && meta.state == PageState::Sent
        {
            meta.state = PageState::Queued;
        }
    }

    /// Whether a page is copied or durable, which is what an entry's own write
    /// waits for.
    #[must_use]
    pub(crate) fn page_ready(&self, key: &PageKey) -> bool {
        self.pages
            .get(key)
            .is_some_and(|meta| meta.state != PageState::Queued)
    }

    /// Whether a page's file is on disk.
    #[must_use]
    pub(crate) fn page_durable(&self, key: &PageKey) -> bool {
        self.pages
            .get(key)
            .is_some_and(|meta| meta.state == PageState::Durable)
    }

    /// Whether the writer should still write a page it was handed.
    #[must_use]
    pub(crate) fn page_wanted(&self, key: &PageKey) -> bool {
        self.pages
            .get(key)
            .is_some_and(|meta| meta.state == PageState::Sent && meta.refs > 0)
    }

    /// Records that a page's file landed. Returns its path when nothing lists
    /// it any more, so the caller unlinks it.
    pub(crate) fn page_written(&mut self, key: &PageKey, bytes: u64) -> Option<PathBuf> {
        let meta = self.pages.get_mut(key)?;
        if meta.state == PageState::Durable {
            return None;
        }
        self.reserved_bytes = self.reserved_bytes.saturating_sub(meta.bytes);
        meta.bytes = bytes;
        meta.state = PageState::Durable;
        self.durable_bytes += bytes;
        if meta.refs == 0 {
            self.durable_bytes = self.durable_bytes.saturating_sub(bytes);
            self.pages.remove(key);
            return Some(self.page_path(key));
        }
        None
    }

    /// Forgets a page whose write failed or that nobody wants.
    pub(crate) fn page_dropped(&mut self, key: &PageKey) {
        if let Some(meta) = self.pages.get(key)
            && meta.state != PageState::Durable
        {
            self.reserved_bytes = self.reserved_bytes.saturating_sub(meta.bytes);
            self.pages.remove(key);
        }
    }

    /// Records that an entry's file landed.
    pub(crate) fn entry_written(&mut self, key: &EntryKey, bytes: u64) {
        let Some(entry) = self.entries.get_mut(key) else {
            return;
        };
        if entry.durable {
            return;
        }
        self.reserved_bytes = self.reserved_bytes.saturating_sub(entry.bytes);
        entry.bytes = bytes;
        entry.durable = true;
        self.durable_bytes += bytes;
        *self.by_pos.entry(entry.pos).or_insert(0) += 1;
    }

    /// Releases an entry, pending or durable, and every page only it listed.
    /// Returns the files to unlink.
    pub(crate) fn release(&mut self, key: &EntryKey) -> Vec<PathBuf> {
        let mut deletions = Vec::new();
        let Some(entry) = self.entries.remove(key) else {
            return deletions;
        };
        if entry.durable {
            self.durable_bytes = self.durable_bytes.saturating_sub(entry.bytes);
            if !entry.foreign
                && let Some(count) = self.by_pos.get_mut(&entry.pos)
            {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    self.by_pos.remove(&entry.pos);
                }
            }
            deletions.push(self.entry_path(key));
        } else {
            self.reserved_bytes = self.reserved_bytes.saturating_sub(entry.bytes);
        }
        let unique: HashSet<PageKey> = entry.pages.iter().copied().collect();
        for page in unique {
            let Some(meta) = self.pages.get_mut(&page) else {
                continue;
            };
            meta.refs = meta.refs.saturating_sub(1);
            if meta.refs > 0 {
                continue;
            }
            match meta.state {
                PageState::Durable => {
                    self.durable_bytes = self.durable_bytes.saturating_sub(meta.bytes);
                    self.pages.remove(&page);
                    deletions.push(self.page_path(&page));
                }
                PageState::Queued => {
                    self.reserved_bytes = self.reserved_bytes.saturating_sub(meta.bytes);
                    self.pages.remove(&page);
                }
                // With the writer: it writes it, sees no reference, and unlinks.
                PageState::Sent => {}
            }
        }
        deletions
    }

    /// Records a read: one more hit, and `last_used` is now. Returns the two
    /// fields for the writer to put in the file.
    pub(crate) fn touch(&mut self, key: &EntryKey, now: u64) -> Option<(u64, u32)> {
        let entry = self.entries.get_mut(key)?;
        entry.hits = entry.hits.saturating_add(1);
        entry.last_used = now;
        Some((entry.last_used, entry.hits))
    }

    /// The entry the budget pass releases first: another build's before any of
    /// this one's, then the lowest utility, the key breaking a tie so the pass
    /// is deterministic.
    fn victim(&self, now: u64) -> Option<EntryKey> {
        // Every entry of this build, pending ones included, is context for the
        // pricing: a deeper entry still on its way to disk already subsumes a
        // shallower one on the same path.
        let context: Vec<&EntryMeta> = self
            .entries
            .values()
            .filter(|entry| !entry.foreign)
            .collect();
        let mut best: Option<(EntryKey, f64)> = None;
        for (key, entry) in self.entries.iter().filter(|(_, entry)| entry.durable) {
            let utility = if entry.foreign {
                -1.0
            } else {
                self.utility(entry, &context, now)
            };
            let better = best.is_none_or(|(best_key, best_u)| {
                utility < best_u || (utility.total_cmp(&best_u).is_eq() && *key < best_key)
            });
            if better {
                best = Some((*key, utility));
            }
        }
        best.map(|(key, _)| key)
    }

    /// What keeping `entry` is worth: the prefill between it and the nearest
    /// lower entry on its path, per mebibyte it alone holds, decayed by
    /// idleness, raised by reads, and cut when a deeper entry on the same path
    /// subsumes it and it does not sit on a branch.
    fn utility(&self, entry: &EntryMeta, all: &[&EntryMeta], now: u64) -> f64 {
        let pt = self.pricing.page_tokens.max(1);
        let mut lower = 0u32;
        let mut deeper = 0usize;
        let mut branches: HashSet<Option<PageKey>> = HashSet::new();
        for other in all {
            if other.pos < entry.pos && on_path(other, entry, pt) {
                lower = lower.max(other.pos);
            }
            if other.pos > entry.pos && on_path(entry, other, pt) {
                deeper += 1;
                let at = usize::try_from(entry.pos / pt).unwrap_or(usize::MAX);
                branches.insert(other.pages.get(at).copied());
            }
        }
        let saved = f64::from(entry.pos.saturating_sub(lower)) * self.pricing.prefill_s_per_token;
        let own_pages: u64 = entry
            .pages
            .iter()
            .filter_map(|page| self.pages.get(page))
            .filter(|page| page.refs <= 1)
            .map(|page| page.bytes)
            .sum();
        let mib = (as_f64(entry.bytes.saturating_add(own_pages)) / (1024.0 * 1024.0)).max(1.0);
        let used = if entry.last_used == 0 {
            entry.created_at
        } else {
            entry.last_used
        };
        let age = as_f64(now.saturating_sub(used));
        let decay = 1.0 / (1.0 + age / self.pricing.decay_s.max(1.0));
        let reads = 1.0 + f64::from(entry.hits).ln_1p();
        let subsumed = if deeper > 0 && branches.len() < 2 {
            SUBSUMED_FACTOR
        } else {
            1.0
        };
        saved / mib * decay * reads * subsumed
    }
}

/// What a subsumed entry keeps of its value: an entry with a deeper one on the
/// same path and no branch below it is only worth reading when the deeper one
/// is gone, so it goes early but not first.
const SUBSUMED_FACTOR: f64 = 0.1;

/// A byte or second count as a float for pricing; exact below 2^53.
#[expect(
    clippy::cast_precision_loss,
    reason = "exact below 2^53, which no byte count or second span here reaches"
)]
const fn as_f64(value: u64) -> f64 {
    value as f64
}

/// Whether `lower` sits on `upper`'s path, judged by their full pages: the pages
/// a lower entry fills completely must be the upper one's first pages. A lower
/// entry's partial page is not compared - its rows are a prefix of the upper
/// one's page, which the keys cannot show - so this can only err towards
/// "on the path", which prices an entry as less unique than it is.
fn on_path(lower: &EntryMeta, upper: &EntryMeta, page_tokens: u32) -> bool {
    let full = usize::try_from(lower.pos / page_tokens).unwrap_or(usize::MAX);
    full <= upper.pages.len()
        && full <= lower.pages.len()
        && lower.pages[..full] == upper.pages[..full]
}

/// What a startup scan found and did.
#[derive(Debug, Default)]
pub(crate) struct Scanned {
    /// Entries of this build that are usable.
    pub(crate) entries: u32,
    /// Entries of another build, kept until their bytes are wanted.
    pub(crate) foreign: u32,
    /// Files removed: debris, orphans, entries with a missing page.
    pub(crate) removed: u32,
}

impl Index {
    /// Rebuilds the index from the directory, removing what cannot be used.
    ///
    /// Temporary files are debris from a write that never renamed; an entry
    /// whose page is missing can never be restored; a page nothing lists is an
    /// orphan. All three are unlinked here, and nothing else is: an entry of
    /// another build stays, priced below every entry of this one.
    pub(crate) fn scan(&mut self, identity: Identity) -> Scanned {
        let mut scanned = Scanned::default();
        let remove = |path: &Path, scanned: &mut Scanned| {
            if std::fs::remove_file(path).is_ok() {
                scanned.removed += 1;
            }
        };
        for path in list(&self.root.join(PAGES_DIR)) {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let key = name.strip_suffix(PAGE_SUFFIX).and_then(PageKey::parse);
            match (key, std::fs::metadata(&path)) {
                (Some(key), Ok(meta)) if !name.contains(TMP_MARKER) && meta.is_file() => {
                    self.pages.insert(
                        key,
                        PageMeta {
                            refs: 0,
                            bytes: meta.len(),
                            state: PageState::Durable,
                        },
                    );
                    self.durable_bytes += meta.len();
                }
                _ => remove(&path, &mut scanned),
            }
        }
        for path in list(&self.root.join(ENTRIES_DIR)) {
            let name = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .into_owned();
            let Some(key) = name.strip_suffix(ENTRY_SUFFIX).and_then(EntryKey::parse) else {
                remove(&path, &mut scanned);
                continue;
            };
            let (header, pages, bytes) = match format::read_entry_index(&path) {
                Ok(found) => found,
                Err(Unusable::Missing) => continue,
                Err(_) => {
                    remove(&path, &mut scanned);
                    continue;
                }
            };
            if !pages.iter().all(|page| self.pages.contains_key(page)) {
                tracing::debug!(entry = %key, "a stored entry is missing a page; removing it");
                remove(&path, &mut scanned);
                continue;
            }
            let foreign = header.identity != identity;
            let unique: HashSet<&PageKey> = pages.iter().collect();
            for page in unique {
                if let Some(meta) = self.pages.get_mut(page) {
                    meta.refs += 1;
                }
            }
            self.durable_bytes += bytes;
            if foreign {
                scanned.foreign += 1;
            } else {
                scanned.entries += 1;
                *self.by_pos.entry(header.pos).or_insert(0) += 1;
            }
            self.entries.insert(
                key,
                EntryMeta {
                    pos: header.pos,
                    pages,
                    has_logits: header.has_logits,
                    bytes,
                    created_at: header.created_at,
                    last_used: header.last_used,
                    hits: header.hits,
                    durable: true,
                    foreign,
                },
            );
        }
        let orphans: Vec<PageKey> = self
            .pages
            .iter()
            .filter(|(_, meta)| meta.refs == 0)
            .map(|(key, _)| *key)
            .collect();
        for key in orphans {
            if let Some(meta) = self.pages.remove(&key) {
                self.durable_bytes = self.durable_bytes.saturating_sub(meta.bytes);
            }
            remove(&self.page_path(&key), &mut scanned);
        }
        scanned
    }

    /// Releases entries until the durable bytes fit the budget. Returns the
    /// files to unlink.
    pub(crate) fn fit_budget(&mut self, now: u64) -> Vec<PathBuf> {
        let mut deletions = Vec::new();
        while self.durable_bytes + self.reserved_bytes > self.budget {
            let Some(victim) = self.victim(now) else {
                break;
            };
            deletions.extend(self.release(&victim));
        }
        deletions
    }
}

/// Every file two levels below `dir` (the fan-out directories), and any stray
/// file directly inside it.
fn list(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(fans) = std::fs::read_dir(dir) else {
        return out;
    };
    for fan in fans.flatten() {
        let path = fan.path();
        if path.is_dir() {
            if let Ok(files) = std::fs::read_dir(&path) {
                out.extend(files.flatten().map(|file| file.path()));
            }
        } else {
            out.push(path);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{Index, NotAccepted, Offer, PageState, Pricing};
    use crate::store::key::{Identity, PathKeys};
    use shiso_core::TokenId;

    const PAGE_FILE: u64 = 100;
    const ENTRY_FILE: u64 = 50;

    fn pricing() -> Pricing {
        Pricing {
            page_tokens: 4,
            prefill_s_per_token: 0.001,
            decay_s: 3600.0,
        }
    }

    fn path(first: u32, n: u32) -> Vec<TokenId> {
        (first..first + n).map(TokenId::new).collect()
    }

    fn offer_for<'a>(
        keys: &PathKeys<'_>,
        pos: u32,
        pages: &'a [crate::store::key::PageKey],
    ) -> Offer<'a> {
        Offer {
            key: keys.entry_at(pos).expect("inside"),
            pos,
            pages,
            has_logits: true,
            entry_bytes: ENTRY_FILE,
            page_file_bytes: PAGE_FILE,
            now: 1000,
        }
    }

    /// Accepts and lands one entry and its pages, as the writer would.
    fn land(index: &mut Index, keys: &PathKeys<'_>, pos: u32) {
        let pages = keys.pages_to(pos);
        index.accept(&offer_for(keys, pos, &pages)).expect("fits");
        for page in &pages {
            if index.claim_page(page) {
                assert!(index.page_written(page, PAGE_FILE).is_none());
            }
        }
        index.entry_written(&keys.entry_at(pos).expect("inside"), ENTRY_FILE);
    }

    #[test]
    fn two_entries_on_one_path_share_their_common_pages() {
        let dir = tempfile::tempdir().expect("dir");
        let mut index = Index::empty(dir.path(), 10_000, pricing());
        let tokens = path(0, 16);
        let keys = PathKeys::new(Identity::from_bytes([1; 20]), &tokens, 4);
        land(&mut index, &keys, 8);
        land(&mut index, &keys, 16);
        assert_eq!(
            index.pages.len(),
            4,
            "the first two pages were written once"
        );
        assert_eq!(index.durable_bytes, 4 * PAGE_FILE + 2 * ENTRY_FILE);
        let hit = index
            .lookup(&keys, 16, 0, |_, _| true)
            .expect("the deeper one");
        assert_eq!(hit.pos, 16);
        let hit = index
            .lookup(&keys, 12, 0, |_, _| true)
            .expect("the lower one");
        assert_eq!(hit.pos, 8);
    }

    #[test]
    fn releasing_the_last_entry_that_lists_a_page_deletes_it() {
        let dir = tempfile::tempdir().expect("dir");
        let mut index = Index::empty(dir.path(), 10_000, pricing());
        let tokens = path(0, 16);
        let keys = PathKeys::new(Identity::from_bytes([1; 20]), &tokens, 4);
        land(&mut index, &keys, 8);
        land(&mut index, &keys, 16);
        let gone = index.release(&keys.entry_at(16).expect("inside"));
        // Its own entry file and its two unshared pages.
        assert_eq!(gone.len(), 3);
        assert_eq!(index.pages.len(), 2);
        assert!(
            index
                .lookup(&keys, 16, 0, |_, _| true)
                .is_some_and(|hit| hit.pos == 8)
        );
    }

    #[test]
    fn an_entry_at_the_prompt_end_needs_its_logits_row() {
        let dir = tempfile::tempdir().expect("dir");
        let mut index = Index::empty(dir.path(), 10_000, pricing());
        let tokens = path(0, 8);
        let keys = PathKeys::new(Identity::from_bytes([1; 20]), &tokens, 4);
        let pages = keys.pages_to(8);
        let mut offer = offer_for(&keys, 8, &pages);
        offer.has_logits = false;
        index.accept(&offer).expect("fits");
        for page in &pages {
            assert!(index.claim_page(page));
            index.page_written(page, PAGE_FILE);
        }
        index.entry_written(&offer.key, ENTRY_FILE);
        assert!(
            index.lookup(&keys, 8, 0, |_, _| true).is_none(),
            "no row at the very end"
        );
        let longer = path(0, 12);
        let longer_keys = PathKeys::new(Identity::from_bytes([1; 20]), &longer, 4);
        assert_eq!(
            index
                .lookup(&longer_keys, 12, 0, |_, _| true)
                .map(|hit| hit.pos),
            Some(8)
        );
    }

    #[test]
    fn a_prompt_final_entry_off_the_grid_answers_its_own_prompt_and_nothing_longer() {
        // An entry with a logits row at 6, on a path whose grid is 4: it is the
        // end of the prompt that wrote it, and a longer prompt that stood on it
        // would cut where a cold prefill never does. The lower entry at 4 is on
        // the grid and is what the longer prompt gets.
        let dir = tempfile::tempdir().expect("dir");
        let mut index = Index::empty(dir.path(), 10_000, pricing());
        let tokens = path(0, 12);
        let keys = PathKeys::new(Identity::from_bytes([1; 20]), &tokens, 4);
        land(&mut index, &keys, 4);
        land(&mut index, &keys, 6);
        let geometry = crate::CacheGeometry::new(4, 1, 1, 0.001, 600.0).with_resume_grid(4);
        let continuable = |pos, logits| geometry.continuable(pos, logits);
        let exact = index
            .lookup(&keys, 6, 0, continuable)
            .expect("its own prompt");
        assert_eq!(exact.pos, 6);
        let longer = index
            .lookup(&keys, 12, 0, continuable)
            .expect("the grid entry");
        assert_eq!(longer.pos, 4);
    }

    #[test]
    fn the_budget_releases_the_least_valuable_entry_first() {
        let dir = tempfile::tempdir().expect("dir");
        // Room for two short paths and a little more.
        let mut index = Index::empty(dir.path(), 2 * (2 * PAGE_FILE + ENTRY_FILE) + 10, pricing());
        let a = path(0, 8);
        let b = path(100, 8);
        let ka = PathKeys::new(Identity::from_bytes([1; 20]), &a, 4);
        let kb = PathKeys::new(Identity::from_bytes([1; 20]), &b, 4);
        land(&mut index, &ka, 8);
        land(&mut index, &kb, 8);
        // `a` was read since; `b` was not.
        index.touch(&ka.entry_at(8).expect("inside"), 1000);
        let c = path(200, 8);
        let kc = PathKeys::new(Identity::from_bytes([1; 20]), &c, 4);
        let pages = kc.pages_to(8);
        let accepted = index
            .accept(&offer_for(&kc, 8, &pages))
            .expect("room was made");
        assert_eq!(accepted.evicted, 1);
        assert!(
            index.knows(&ka.entry_at(8).expect("inside")),
            "the read one stayed"
        );
        assert!(!index.knows(&kb.entry_at(8).expect("inside")));
    }

    #[test]
    fn an_entry_larger_than_the_budget_is_refused_outright() {
        let dir = tempfile::tempdir().expect("dir");
        let mut index = Index::empty(dir.path(), 100, pricing());
        let tokens = path(0, 8);
        let keys = PathKeys::new(Identity::from_bytes([1; 20]), &tokens, 4);
        let pages = keys.pages_to(8);
        assert!(matches!(
            index.accept(&offer_for(&keys, 8, &pages)),
            Err(NotAccepted::OverBudget { .. })
        ));
        assert!(index.pages.is_empty(), "a refusal registers nothing");
    }

    #[test]
    fn a_page_two_jobs_need_is_copied_once() {
        let dir = tempfile::tempdir().expect("dir");
        let mut index = Index::empty(dir.path(), 10_000, pricing());
        let tokens = path(0, 12);
        let keys = PathKeys::new(Identity::from_bytes([1; 20]), &tokens, 4);
        let short = keys.pages_to(8);
        let long = keys.pages_to(12);
        index.accept(&offer_for(&keys, 8, &short)).expect("fits");
        index.accept(&offer_for(&keys, 12, &long)).expect("fits");
        assert!(index.claim_page(&short[0]));
        assert!(!index.claim_page(&long[0]), "the second job finds it taken");
        // The first job is dropped before its entry lands: its pages stay for
        // the second.
        let gone = index.release(&keys.entry_at(8).expect("inside"));
        assert!(gone.is_empty());
        assert_eq!(index.pages[&long[0]].state, PageState::Sent);
        assert_eq!(index.pages[&long[1]].state, PageState::Queued);
    }
}
