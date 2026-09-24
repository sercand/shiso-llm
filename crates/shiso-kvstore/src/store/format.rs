//! The two file formats, and the one write every file goes through.
//!
//! A page file is one page's rows plus the token ids and parent key that name
//! them. An entry file is one checkpoint: the page keys of its path, the logits
//! row when there is one, and the recurrent state. Both start with a fixed
//! header, both carry a checksum over everything the header does not change in
//! place, and both are written to a temporary name, synced and renamed, so a
//! reader sees either no file or a whole one.
//!
//! # Every file names itself
//!
//! A page file's name is recomputable from its own header and token ids, and an
//! entry file's name from its own header and last page key. A read recomputes
//! it, so a file that was renamed, copied between directories or truncated is a
//! miss rather than rows handed to the device under the wrong name.
//!
//! # Byte table
//!
//! All fields little-endian.
//!
//! | file | offset | bytes | what |
//! |---|---|---|---|
//! | page | 0 | 4 | magic `KVP2` |
//! | page | 4 | 4 | format version, 2 |
//! | page | 8 | 20 | identity digest |
//! | page | 28 | 20 | parent page key, zero for a path's first page |
//! | page | 48 | 4 | token count, `1..=page_tokens` |
//! | page | 56 | 8 | payload bytes |
//! | page | 64 | 8 | checksum over the token ids and the payload |
//! | page | 80 | `4 * n` | token ids |
//! | page | then | payload | the page's runs, as `Backing::page_to_host` lays them out |
//! | entry | 0 | 4 | magic `KVE2` |
//! | entry | 4 | 4 | format version, 2 |
//! | entry | 8 | 20 | identity digest |
//! | entry | 28 | 4 | position the state is valid at |
//! | entry | 32 | 4 | page count |
//! | entry | 36 | 1 | reason |
//! | entry | 37 | 1 | flags; bit 0 is "carries the logits row" |
//! | entry | 40 | 8 | state bytes |
//! | entry | 48 | 8 | logits bytes |
//! | entry | 56 | 8 | checksum over the page keys, the logits and the state |
//! | entry | 64 | 8 | created, seconds since the epoch |
//! | entry | 72 | 8 | last read, seconds since the epoch (rewritten in place) |
//! | entry | 80 | 4 | reads (rewritten in place) |
//! | entry | 96 | `20 * pages` | page keys, first page first |
//! | entry | then | logits | the frontier logits row |
//! | entry | then | state | the recurrent state blob |

use std::fs::{self, File, OpenOptions};
use std::io::{self, Read as _, Seek as _, Write as _};
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use shiso_core::TokenId;

use crate::store::key::{EntryKey, Identity, KEY_BYTES, PageKey, entry_key, page_key};
use crate::store::reason::Reason;

/// The version both formats carry. A file whose version differs is not ours.
pub(crate) const FORMAT_VERSION: u32 = 2;
/// Magic of a page file.
const PAGE_MAGIC: [u8; 4] = *b"KVP2";
/// Magic of an entry file.
const ENTRY_MAGIC: [u8; 4] = *b"KVE2";
/// Bytes in a page file's header.
pub(crate) const PAGE_HEADER_BYTES: usize = 80;
/// Bytes in an entry file's header.
pub(crate) const ENTRY_HEADER_BYTES: usize = 96;
/// Flag bit: the entry carries a logits row.
const FLAG_LOGITS: u8 = 1;
/// Mode of every directory: owner-only, because the files hold model state
/// derived from user prompts.
const DIR_MODE: u32 = 0o700;
/// Mode of every file, for the same reason.
const FILE_MODE: u32 = 0o600;
/// Suffix of a page file.
pub(crate) const PAGE_SUFFIX: &str = ".pg";
/// Suffix of an entry file.
pub(crate) const ENTRY_SUFFIX: &str = ".ck";
/// Marker every temporary name carries, so a startup sweep can find debris.
pub(crate) const TMP_MARKER: &str = ".tmp.";

/// The page directory under the store root.
pub(crate) const PAGES_DIR: &str = "pages";
/// The entry directory under the store root.
pub(crate) const ENTRIES_DIR: &str = "ckpt";

/// A 64-bit checksum over a stream of bytes.
///
/// Not a cryptographic hash: the names are sha1 and already say *what* a file
/// is. This catches a torn or bit-rotted file on the read path, where a sha1 of
/// every page would cost a large fraction of the read itself. Four independent
/// multiply-rotate lanes over little-endian words keep it at memory speed.
#[derive(Debug, Clone)]
pub(crate) struct Checksum {
    lanes: [u64; 4],
    len: u64,
    tail: [u8; 32],
    tail_len: usize,
}

/// The multiplier every lane folds with (a 64-bit odd constant with good bit
/// dispersion; its only job is to spread each word's bits across the lane).
const MIX: u64 = 0x9E37_79B9_7F4A_7C15;
/// Lane seeds, distinct so that swapping two words between lanes changes the sum.
const SEEDS: [u64; 4] = [
    0x243F_6A88_85A3_08D3,
    0x1319_8A2E_0370_7344,
    0xA409_3822_299F_31D0,
    0x082E_FA98_EC4E_6C89,
];

impl Checksum {
    /// An empty checksum.
    #[must_use]
    pub(crate) const fn new() -> Self {
        Self {
            lanes: SEEDS,
            len: 0,
            tail: [0; 32],
            tail_len: 0,
        }
    }

    /// Folds 32 bytes into the four lanes.
    fn block(lanes: &mut [u64; 4], block: &[u8; 32]) {
        for (lane, word) in lanes.iter_mut().zip(block.as_chunks::<8>().0) {
            *lane = (*lane ^ u64::from_le_bytes(*word))
                .wrapping_mul(MIX)
                .rotate_left(29);
        }
    }

    /// Adds `bytes` to the stream.
    pub(crate) fn update(&mut self, mut bytes: &[u8]) {
        self.len = self
            .len
            .wrapping_add(u64::try_from(bytes.len()).unwrap_or(u64::MAX));
        if self.tail_len > 0 {
            let take = (32 - self.tail_len).min(bytes.len());
            self.tail[self.tail_len..self.tail_len + take].copy_from_slice(&bytes[..take]);
            self.tail_len += take;
            bytes = &bytes[take..];
            if self.tail_len < 32 {
                return;
            }
            let block = self.tail;
            Self::block(&mut self.lanes, &block);
            self.tail_len = 0;
        }
        let (blocks, rest) = bytes.as_chunks::<32>();
        for block in blocks {
            Self::block(&mut self.lanes, block);
        }
        self.tail[..rest.len()].copy_from_slice(rest);
        self.tail_len = rest.len();
    }

    /// The final value.
    #[must_use]
    pub(crate) fn finish(&self) -> u64 {
        let mut lanes = self.lanes;
        if self.tail_len > 0 {
            let mut block = [0u8; 32];
            block[..self.tail_len].copy_from_slice(&self.tail[..self.tail_len]);
            Self::block(&mut lanes, &block);
        }
        let mut acc = self.len;
        for lane in lanes {
            acc = fmix(acc ^ lane).wrapping_mul(MIX);
        }
        fmix(acc)
    }
}

/// The final avalanche: every input bit flips each output bit with even odds.
const fn fmix(mut value: u64) -> u64 {
    value ^= value >> 33;
    value = value.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    value ^= value >> 33;
    value = value.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    value ^ (value >> 33)
}

/// The checksum of a sequence of byte slices, as one stream.
#[must_use]
pub(crate) fn checksum_of(parts: &[&[u8]]) -> u64 {
    let mut sum = Checksum::new();
    for part in parts {
        sum.update(part);
    }
    sum.finish()
}

/// Token ids as the bytes a page file stores.
#[must_use]
pub(crate) fn token_bytes(tokens: &[TokenId]) -> Vec<u8> {
    let mut out = Vec::with_capacity(tokens.len() * 4);
    for token in tokens {
        out.extend_from_slice(&token.get().to_le_bytes());
    }
    out
}

/// Where a file with this key and suffix lives: `<root>/<dir>/<2 hex>/<40 hex><suffix>`.
///
/// The fan-out by the first byte keeps any one directory to a few hundred
/// entries on a full disk, which is what keeps a lookup by name a cheap one.
#[must_use]
pub(crate) fn object_path(root: &Path, dir: &str, hex: &str, suffix: &str) -> PathBuf {
    root.join(dir)
        .join(&hex[..2])
        .join(format!("{hex}{suffix}"))
}

/// Seconds since the epoch, or zero on a clock that cannot be read.
///
/// Zero makes every entry look ancient, which prices by size rather than by
/// recency: visibly wrong rather than silently keeping everything.
#[must_use]
pub(crate) fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// A page file's header, as it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PageHeader {
    pub(crate) identity: Identity,
    pub(crate) parent: Option<PageKey>,
    pub(crate) n_tokens: u32,
    pub(crate) payload_bytes: u64,
    pub(crate) checksum: u64,
}

impl PageHeader {
    /// The header record.
    #[must_use]
    pub(crate) fn to_bytes(self) -> [u8; PAGE_HEADER_BYTES] {
        let mut h = [0u8; PAGE_HEADER_BYTES];
        h[0..4].copy_from_slice(&PAGE_MAGIC);
        h[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        h[8..28].copy_from_slice(self.identity.bytes());
        h[28..48].copy_from_slice(&self.parent.map_or([0u8; KEY_BYTES], |key| *key.bytes()));
        h[48..52].copy_from_slice(&self.n_tokens.to_le_bytes());
        h[56..64].copy_from_slice(&self.payload_bytes.to_le_bytes());
        h[64..72].copy_from_slice(&self.checksum.to_le_bytes());
        h
    }

    /// Parses a header, or `None` when the magic or the version is not ours.
    #[must_use]
    pub(crate) fn from_bytes(h: &[u8]) -> Option<Self> {
        if h.len() < PAGE_HEADER_BYTES || h[0..4] != PAGE_MAGIC || word32(h, 4) != FORMAT_VERSION {
            return None;
        }
        let parent = digest(h, 28);
        Some(Self {
            identity: Identity::from_bytes(digest(h, 8)),
            parent: (parent != [0u8; KEY_BYTES]).then(|| PageKey::from_bytes(parent)),
            n_tokens: word32(h, 48),
            payload_bytes: word64(h, 56),
            checksum: word64(h, 64),
        })
    }

    /// Bytes of the whole file this header describes.
    #[must_use]
    pub(crate) fn file_bytes(&self) -> u64 {
        u64::try_from(PAGE_HEADER_BYTES)
            .unwrap_or(u64::MAX)
            .saturating_add(u64::from(self.n_tokens).saturating_mul(4))
            .saturating_add(self.payload_bytes)
    }
}

/// An entry file's header, as it reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct EntryHeader {
    pub(crate) identity: Identity,
    pub(crate) pos: u32,
    pub(crate) n_pages: u32,
    pub(crate) reason: Reason,
    pub(crate) has_logits: bool,
    pub(crate) state_bytes: u64,
    pub(crate) logits_bytes: u64,
    pub(crate) checksum: u64,
    pub(crate) created_at: u64,
    pub(crate) last_used: u64,
    pub(crate) hits: u32,
}

impl EntryHeader {
    /// The header record.
    #[must_use]
    pub(crate) fn to_bytes(self) -> [u8; ENTRY_HEADER_BYTES] {
        let mut h = [0u8; ENTRY_HEADER_BYTES];
        h[0..4].copy_from_slice(&ENTRY_MAGIC);
        h[4..8].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        h[8..28].copy_from_slice(self.identity.bytes());
        h[28..32].copy_from_slice(&self.pos.to_le_bytes());
        h[32..36].copy_from_slice(&self.n_pages.to_le_bytes());
        h[36] = self.reason.as_u8();
        h[37] = if self.has_logits { FLAG_LOGITS } else { 0 };
        h[40..48].copy_from_slice(&self.state_bytes.to_le_bytes());
        h[48..56].copy_from_slice(&self.logits_bytes.to_le_bytes());
        h[56..64].copy_from_slice(&self.checksum.to_le_bytes());
        h[64..72].copy_from_slice(&self.created_at.to_le_bytes());
        h[72..80].copy_from_slice(&self.last_used.to_le_bytes());
        h[80..84].copy_from_slice(&self.hits.to_le_bytes());
        h
    }

    /// Parses a header, or `None` when the magic or the version is not ours.
    #[must_use]
    pub(crate) fn from_bytes(h: &[u8]) -> Option<Self> {
        if h.len() < ENTRY_HEADER_BYTES || h[0..4] != ENTRY_MAGIC || word32(h, 4) != FORMAT_VERSION
        {
            return None;
        }
        Some(Self {
            identity: Identity::from_bytes(digest(h, 8)),
            pos: word32(h, 28),
            n_pages: word32(h, 32),
            reason: Reason::from_byte(h[36]),
            has_logits: h[37] & FLAG_LOGITS != 0,
            state_bytes: word64(h, 40),
            logits_bytes: word64(h, 48),
            checksum: word64(h, 56),
            created_at: word64(h, 64),
            last_used: word64(h, 72),
            hits: word32(h, 80),
        })
    }

    /// Bytes of the page-key table behind the header.
    #[must_use]
    pub(crate) fn keys_bytes(&self) -> u64 {
        u64::from(self.n_pages).saturating_mul(u64::try_from(KEY_BYTES).unwrap_or(u64::MAX))
    }

    /// Bytes of the whole file this header describes.
    #[must_use]
    pub(crate) fn file_bytes(&self) -> u64 {
        u64::try_from(ENTRY_HEADER_BYTES)
            .unwrap_or(u64::MAX)
            .saturating_add(self.keys_bytes())
            .saturating_add(self.logits_bytes)
            .saturating_add(self.state_bytes)
    }
}

/// The little-endian `u32` at `at`.
fn word32(bytes: &[u8], at: usize) -> u32 {
    let mut out = [0u8; 4];
    out.copy_from_slice(&bytes[at..at + 4]);
    u32::from_le_bytes(out)
}

/// The little-endian `u64` at `at`.
fn word64(bytes: &[u8], at: usize) -> u64 {
    let mut out = [0u8; 8];
    out.copy_from_slice(&bytes[at..at + 8]);
    u64::from_le_bytes(out)
}

/// The 20-byte digest at `at`.
fn digest(bytes: &[u8], at: usize) -> [u8; KEY_BYTES] {
    let mut out = [0u8; KEY_BYTES];
    out.copy_from_slice(&bytes[at..at + KEY_BYTES]);
    out
}

/// A counter that makes two temporary names in one process distinct.
static TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// `<path>.tmp.<pid>.<n>`, the name a write lands under before the rename.
fn temporary_name(path: &Path) -> PathBuf {
    let n = TMP_SEQ.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(
        "{}{TMP_MARKER}{}.{n}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ))
}

/// Creates `dir` and its parents at owner-only mode.
///
/// # Errors
///
/// The OS error when the directory cannot be created; one that already exists
/// is not an error.
pub(crate) fn ensure_dir(dir: &Path) -> io::Result<()> {
    fs::DirBuilder::new()
        .recursive(true)
        .mode(DIR_MODE)
        .create(dir)
}

/// Writes `parts` to `path` atomically: a temporary file in the same directory,
/// synced, then renamed over. Returns the bytes written.
///
/// The sync comes before the rename so a name that survives a crash never
/// names a file whose bytes did not: the rename is the commit point.
///
/// # Errors
///
/// The OS error of whichever step failed; the temporary file is removed on
/// every failure path.
pub(crate) fn write_atomic(path: &Path, parts: &[&[u8]]) -> io::Result<u64> {
    if let Some(parent) = path.parent() {
        ensure_dir(parent)?;
    }
    let tmp = temporary_name(path);
    let result = (|| -> io::Result<u64> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(FILE_MODE)
            .open(&tmp)?;
        let mut written = 0u64;
        for part in parts {
            file.write_all(part)?;
            written = written.saturating_add(u64::try_from(part.len()).unwrap_or(u64::MAX));
        }
        file.sync_data()?;
        drop(file);
        fs::rename(&tmp, path)?;
        Ok(written)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

/// What a page write needs.
#[derive(Debug)]
pub(crate) struct PageWrite<'a> {
    pub(crate) identity: Identity,
    pub(crate) parent: Option<PageKey>,
    pub(crate) tokens: &'a [TokenId],
    pub(crate) payload: &'a [u8],
}

/// Writes one page file. Returns its size.
///
/// # Errors
///
/// As [`write_atomic`].
pub(crate) fn write_page(path: &Path, page: &PageWrite<'_>) -> io::Result<u64> {
    let tokens = token_bytes(page.tokens);
    let header = PageHeader {
        identity: page.identity,
        parent: page.parent,
        n_tokens: u32::try_from(page.tokens.len()).unwrap_or(u32::MAX),
        payload_bytes: u64::try_from(page.payload.len()).unwrap_or(u64::MAX),
        checksum: checksum_of(&[&tokens, page.payload]),
    };
    write_atomic(path, &[&header.to_bytes(), &tokens, page.payload])
}

/// What an entry write needs.
#[derive(Debug)]
pub(crate) struct EntryWrite<'a> {
    pub(crate) identity: Identity,
    pub(crate) pos: u32,
    pub(crate) reason: Reason,
    pub(crate) pages: &'a [PageKey],
    pub(crate) logits: Option<&'a [u8]>,
    pub(crate) state: &'a [u8],
    pub(crate) created_at: u64,
}

/// Writes one entry file. Returns its size.
///
/// # Errors
///
/// As [`write_atomic`].
pub(crate) fn write_entry(path: &Path, entry: &EntryWrite<'_>) -> io::Result<u64> {
    let mut keys = Vec::with_capacity(entry.pages.len() * KEY_BYTES);
    for key in entry.pages {
        keys.extend_from_slice(key.bytes());
    }
    let logits = entry.logits.unwrap_or(&[]);
    let header = EntryHeader {
        identity: entry.identity,
        pos: entry.pos,
        n_pages: u32::try_from(entry.pages.len()).unwrap_or(u32::MAX),
        reason: entry.reason,
        has_logits: entry.logits.is_some(),
        state_bytes: u64::try_from(entry.state.len()).unwrap_or(u64::MAX),
        logits_bytes: u64::try_from(logits.len()).unwrap_or(u64::MAX),
        checksum: checksum_of(&[&keys, logits, entry.state]),
        created_at: entry.created_at,
        last_used: 0,
        hits: 0,
    };
    write_atomic(path, &[&header.to_bytes(), &keys, logits, entry.state])
}

/// Why a file read back is not usable. Every one of these is a miss to the
/// caller; the variant is for the log line and for deciding whether the file
/// goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Unusable {
    /// Not there.
    Missing,
    /// The header is not ours, or the file is shorter than it says.
    Malformed,
    /// Written by another build.
    Foreign,
    /// The bytes do not add up to the name or the checksum.
    Corrupt,
}

/// Reads an entry file's header and page-key table, without the state behind
/// them: what a startup scan needs.
///
/// # Errors
///
/// [`Unusable`] naming why the file cannot be indexed.
pub(crate) fn read_entry_index(path: &Path) -> Result<(EntryHeader, Vec<PageKey>, u64), Unusable> {
    let mut file = File::open(path).map_err(|_| Unusable::Missing)?;
    let len = file.metadata().map_err(|_| Unusable::Missing)?.len();
    let mut head = [0u8; ENTRY_HEADER_BYTES];
    file.read_exact(&mut head)
        .map_err(|_| Unusable::Malformed)?;
    let header = EntryHeader::from_bytes(&head).ok_or(Unusable::Malformed)?;
    if header.file_bytes() != len || header.n_pages == 0 {
        return Err(Unusable::Malformed);
    }
    let mut keys =
        vec![0u8; usize::try_from(header.keys_bytes()).map_err(|_| Unusable::Malformed)?];
    file.read_exact(&mut keys)
        .map_err(|_| Unusable::Malformed)?;
    let pages = keys
        .as_chunks::<KEY_BYTES>()
        .0
        .iter()
        .map(|raw| PageKey::from_bytes(*raw))
        .collect();
    Ok((header, pages, len))
}

/// An entry's header and page list, checked against its name and this build,
/// with the body - the logits row and the state - still on disk.
///
/// Split from the body because a restore needs the page list at once and the
/// state only at its end: the body, a hundred mebibytes, is read on a thread
/// of its own while the pages stream in.
#[derive(Debug, Clone)]
pub(crate) struct EntryHead {
    pub(crate) header: EntryHeader,
    pub(crate) pages: Vec<PageKey>,
    path: PathBuf,
}

/// An entry's body, read and checked.
#[derive(Debug)]
pub(crate) struct EntryBody {
    pub(crate) logits: Option<Vec<u8>>,
    pub(crate) state: Vec<u8>,
}

/// Reads an entry's header and page list and checks them against its name and
/// this build.
///
/// # Errors
///
/// [`Unusable`] naming why the file cannot be restored from.
pub(crate) fn read_entry_head(
    path: &Path,
    key: EntryKey,
    identity: Identity,
) -> Result<EntryHead, Unusable> {
    let (header, pages, _bytes) = read_entry_index(path)?;
    if header.identity != identity {
        return Err(Unusable::Foreign);
    }
    let Some(last) = pages.last() else {
        return Err(Unusable::Malformed);
    };
    if entry_key(&identity, last, header.pos) != key {
        return Err(Unusable::Corrupt);
    }
    Ok(EntryHead {
        header,
        pages,
        path: path.to_path_buf(),
    })
}

impl EntryHead {
    /// Reads the body behind this head and checks the whole entry's checksum.
    ///
    /// # Errors
    ///
    /// [`Unusable`] naming why the body cannot be restored from.
    pub(crate) fn read_body(&self) -> Result<EntryBody, Unusable> {
        let mut file = File::open(&self.path).map_err(|_| Unusable::Missing)?;
        let keys_bytes = self.header.keys_bytes();
        let at = u64::try_from(ENTRY_HEADER_BYTES)
            .unwrap_or(u64::MAX)
            .saturating_add(keys_bytes);
        file.seek(io::SeekFrom::Start(at))
            .map_err(|_| Unusable::Malformed)?;
        let logits_len =
            usize::try_from(self.header.logits_bytes).map_err(|_| Unusable::Malformed)?;
        let state_len =
            usize::try_from(self.header.state_bytes).map_err(|_| Unusable::Malformed)?;
        let mut logits = vec![0u8; logits_len];
        let mut state = vec![0u8; state_len];
        file.read_exact(&mut logits)
            .map_err(|_| Unusable::Malformed)?;
        file.read_exact(&mut state)
            .map_err(|_| Unusable::Malformed)?;
        let mut keys = Vec::with_capacity(self.pages.len() * KEY_BYTES);
        for page in &self.pages {
            keys.extend_from_slice(page.bytes());
        }
        if checksum_of(&[&keys, &logits, &state]) != self.header.checksum {
            return Err(Unusable::Corrupt);
        }
        Ok(EntryBody {
            logits: self.header.has_logits.then_some(logits),
            state,
        })
    }
}

/// What a page read is checked against.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PageExpect<'a> {
    pub(crate) key: PageKey,
    pub(crate) identity: Identity,
    pub(crate) parent: Option<PageKey>,
    pub(crate) tokens: &'a [TokenId],
    pub(crate) payload_bytes: u64,
}

/// Checks a whole page file read into `bytes` and returns where its payload
/// starts.
///
/// # Errors
///
/// [`Unusable`] naming why the page cannot be uploaded.
pub(crate) fn verify_page(bytes: &[u8], expect: &PageExpect<'_>) -> Result<usize, Unusable> {
    let header = PageHeader::from_bytes(bytes).ok_or(Unusable::Malformed)?;
    if header.identity != expect.identity {
        return Err(Unusable::Foreign);
    }
    if header.file_bytes() != u64::try_from(bytes.len()).unwrap_or(u64::MAX)
        || header.payload_bytes != expect.payload_bytes
        || usize::try_from(header.n_tokens).ok() != Some(expect.tokens.len())
        || header.parent != expect.parent
    {
        return Err(Unusable::Malformed);
    }
    let tokens_end = PAGE_HEADER_BYTES + expect.tokens.len() * 4;
    let stored = &bytes[PAGE_HEADER_BYTES..tokens_end];
    if stored != token_bytes(expect.tokens).as_slice() {
        return Err(Unusable::Corrupt);
    }
    if page_key(&expect.identity, expect.parent.as_ref(), expect.tokens) != expect.key {
        return Err(Unusable::Corrupt);
    }
    if checksum_of(&[stored, &bytes[tokens_end..]]) != header.checksum {
        return Err(Unusable::Corrupt);
    }
    Ok(tokens_end)
}

/// Rewrites an entry's read count and last-read stamp in place.
///
/// In place rather than through a temporary file, because the two fields are
/// outside the checksum and only the budget pass reads them: a torn write costs
/// one entry's score, never its contents.
///
/// # Errors
///
/// The OS error when the file cannot be opened or rewritten.
pub(crate) fn touch_entry(path: &Path, last_used: u64, hits: u32) -> io::Result<()> {
    let mut file = OpenOptions::new().write(true).open(path)?;
    file.seek(io::SeekFrom::Start(72))?;
    let mut fields = [0u8; 12];
    fields[0..8].copy_from_slice(&last_used.to_le_bytes());
    fields[8..12].copy_from_slice(&hits.to_le_bytes());
    file.write_all(&fields)
}

#[cfg(test)]
mod tests {
    use super::{Checksum, checksum_of};

    #[test]
    fn a_checksum_does_not_depend_on_how_the_stream_was_split() {
        let bytes: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        let whole = checksum_of(&[&bytes]);
        for split in [0, 1, 31, 32, 33, 500, 999, 1000] {
            let (a, b) = bytes.split_at(split);
            assert_eq!(checksum_of(&[a, b]), whole, "split at {split}");
        }
        let mut sum = Checksum::new();
        for byte in &bytes {
            sum.update(std::slice::from_ref(byte));
        }
        assert_eq!(sum.finish(), whole);
    }

    #[test]
    fn a_single_flipped_bit_changes_the_checksum() {
        let mut bytes = vec![0u8; 4096];
        let clean = checksum_of(&[&bytes]);
        bytes[2049] ^= 0x10;
        assert_ne!(checksum_of(&[&bytes]), clean);
        // Length is part of the sum, so trailing zeros are not free.
        assert_ne!(checksum_of(&[&[0u8; 4097][..]]), clean);
    }
}
