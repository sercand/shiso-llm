//! The disk tier: serialized paths, named by the prefix they cache.
//!
//! The tree keeps a span alive as long as memory allows it. This module is what is
//! left when memory does not: one path - its tokens, the page bytes behind them and
//! the recurrent state at its end - written to a file whose name is the sha1 of the
//! text prefix the path stands for, so that a later prompt can ask "have I already
//! computed the front of this?" without a manifest, a database, or anything else
//! that can disagree with the bytes it describes.
//!
//! # The name is the lookup
//!
//! An entry answers "does this checkpoint represent the bytes at the front of the
//! incoming prompt?". Because the answer is a file name, the directory itself is
//! the index: no sidecar manifest to keep in sync, and no entry that can disagree
//! with its own key. A file whose stored text no longer hashes to its name is
//! refused rather than trusted, which is also what makes a renamed or copied file
//! inert instead of silently wrong.
//!
//! # What is fixed, and what is not
//!
//! The 48-byte header and the scoring constants are an on-disk contract: a
//! directory written by an older build has to be readable by a newer one, so the
//! header layout and the scoring constants are fixed. The layout is declared once,
//! in [`SpillHeader`], and pinned by a test that compares all 48 bytes against a
//! hand-built fixture. Nothing else here is interchangeable.
//!
//! # Why the budget pass scores rather than rotates
//!
//! A checkpoint that was read yesterday is evidence the entry earns its bytes; one
//! unread for a month is not, and that evidence decays with a half-life rather than
//! expiring on a date. Density is the other half: an entry that caches a great deal
//! of prefill per byte of disk earns more room than one that caches little per
//! byte. Rotating the oldest file out instead would evict the anchors a session
//! keeps coming back to, which is the opposite of what a cache is for.

use std::fs::{self, File, OpenOptions};
use std::io::{Read as _, Seek as _, Write as _};
use std::os::unix::fs::{DirBuilderExt as _, OpenOptionsExt as _};
use std::path::{Path, PathBuf};

use crate::error::KvError;
use crate::ids::{SpillId, to_u32};

/// Why an entry was written, which is half of what it is worth.
///
/// The reason is stored in the header rather than inferred from the directory,
/// because the reason is a fact about the moment the entry was written: "memory had
/// to give this up" is evidence the entry was doing work, while "this is a system
/// prompt every session reads" is evidence it always will be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Reason {
    /// Read from a file whose reason byte was outside the known range.
    Unknown = 0,
    /// Written because the prefix was cold - it had left memory entirely.
    Cold = 1,
    /// Written as a conversation grew, so the older prefix could be resumed.
    Continued = 2,
    /// Written because page pressure was about to drop the span.
    Evict = 3,
    /// Written because the process is shutting down.
    Shutdown = 4,
    /// Written for a system prompt shared by every session of a deployment.
    AgentSystem = 5,
    /// Written for a session's own prefix, which continues a system prompt.
    AgentSession = 6,
}

impl Reason {
    /// Maps a header byte back to a reason, folding anything unknown to
    /// [`Reason::Unknown`] the way the format requires.
    #[must_use]
    pub const fn from_byte(byte: u8) -> Self {
        match byte {
            1 => Self::Cold,
            2 => Self::Continued,
            3 => Self::Evict,
            4 => Self::Shutdown,
            5 => Self::AgentSystem,
            6 => Self::AgentSession,
            _ => Self::Unknown,
        }
    }

    /// The header byte for this reason.
    #[must_use]
    pub const fn as_u8(self) -> u8 {
        match self {
            Self::Unknown => 0,
            Self::Cold => 1,
            Self::Continued => 2,
            Self::Evict => 3,
            Self::Shutdown => 4,
            Self::AgentSystem => 5,
            Self::AgentSession => 6,
        }
    }

    /// Whether this reason marks an entry as worth keeping under pressure.
    ///
    /// Cold, evict and shutdown are all "memory had to give this up", which is the
    /// evidence the budget pass pays double for: an entry that was holding a live
    /// sequence's place is the expensive thing to lose. The agent reasons are not
    /// anchors - they were written on a prediction, not under pressure - and are
    /// priced by the continued-prefix factor instead.
    #[must_use]
    pub const fn is_anchor(self) -> bool {
        matches!(self, Self::Cold | Self::Evict | Self::Shutdown)
    }
}

/// The 48-byte header every entry starts with.
///
/// The field order is the byte order, and the byte order is a contract with
/// directories this build did not write: a field may be added at the end of the
/// struct, but nothing here may be reordered or resized without a version bump.
/// Bytes 21..24 are reserved and are written zero, which is what lets a later build
/// claim them without moving a later field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillHeader {
    /// Byte 4: quantization the KV bytes were written with. Only 2 and 4 are
    /// restorable, which is the gate that refuses a file from another build.
    pub quant_bits: u8,
    /// Byte 5: why it was written. See [`Reason`].
    pub reason: u8,
    /// Presence bits for optional side tables the engine may hang off an entry.
    pub ext_flags: u8,
    /// Which model's geometry the bytes assume. Byte 7; zero is the historical
    /// default written when this byte was still reserved.
    pub model_id: u8,
    /// Tokens the path covers. Zero is never valid: an empty prefix is not a
    /// checkpoint of anything.
    pub tokens: u32,
    /// Times this entry was read back, which is the evidence the budget pass
    /// decays.
    pub hits: u32,
    /// Context size the path was written for. A smaller-context entry loads into a
    /// larger one; the reverse is refused.
    pub ctx_size: u32,
    /// Version of the serialized path behind the header.
    pub payload_abi: u8,
    /// Seconds since the epoch when the entry was written.
    pub created_at: u64,
    /// Seconds since the epoch of the last read, which is what the budget pass
    /// decays against and what breaks a scoring tie.
    pub last_used: u64,
    /// Bytes of serialized path behind the header and the text prefix.
    pub payload_bytes: u64,
}

impl SpillHeader {
    /// Serializes to the on-disk record. Magic, version and payload ABI are written
    /// from the constants rather than taken from the struct, so a header built by
    /// this build always claims this build's layout.
    #[must_use]
    pub fn to_bytes(self) -> [u8; HEADER_BYTES] {
        let mut h = [0u8; HEADER_BYTES];
        h[0..3].copy_from_slice(&MAGIC);
        h[3] = VERSION;
        h[4] = self.quant_bits;
        h[5] = self.reason;
        h[6] = self.ext_flags;
        h[7] = self.model_id;
        h[8..12].copy_from_slice(&self.tokens.to_le_bytes());
        h[12..16].copy_from_slice(&self.hits.to_le_bytes());
        h[16..20].copy_from_slice(&self.ctx_size.to_le_bytes());
        h[20] = PAYLOAD_ABI;
        h[24..32].copy_from_slice(&self.created_at.to_le_bytes());
        h[32..40].copy_from_slice(&self.last_used.to_le_bytes());
        h[40..48].copy_from_slice(&self.payload_bytes.to_le_bytes());
        h
    }

    /// Parses a record without judging it: every field is taken as it reads, so the
    /// caller can report *which* field made a file unusable instead of only that it
    /// is unusable. [`Self::is_restorable`] is the judgement.
    #[must_use]
    pub fn from_bytes(h: &[u8; HEADER_BYTES]) -> Self {
        Self {
            quant_bits: h[4],
            reason: h[5],
            ext_flags: h[6],
            model_id: h[7],
            tokens: u32::from_le_bytes(slice4(&h[8..12])),
            hits: u32::from_le_bytes(slice4(&h[12..16])),
            ctx_size: u32::from_le_bytes(slice4(&h[16..20])),
            payload_abi: h[20],
            created_at: u64::from_le_bytes(slice8(&h[24..32])),
            last_used: u64::from_le_bytes(slice8(&h[32..40])),
            payload_bytes: u64::from_le_bytes(slice8(&h[40..48])),
        }
    }

    /// Whether this record describes a path this build can hand to the engine.
    ///
    /// Two independent gates, both from the C reader: the envelope has to be one
    /// this build speaks (magic, version, payload ABI), and the entry has to be
    /// meaningful on its own (a non-empty prefix, and a quantization whose bytes
    /// this build can lay back into pages). A file failing either is skipped by the
    /// scan and refused by a read; it is never partially trusted.
    #[must_use]
    pub fn is_restorable(&self) -> bool {
        self.payload_abi == PAYLOAD_ABI
            && self.tokens != 0
            && QUANT_BITS_VALID.contains(&self.quant_bits)
    }

    /// Whether the three magic bytes are present, which is what lets a scan tell "a
    /// file we do not manage" from "an entry of ours with a bad field".
    #[must_use]
    pub fn has_magic(h: &[u8; HEADER_BYTES]) -> bool {
        h[0..3] == MAGIC
    }
}

/// Bytes in the fixed header, which is also the offset of the text length.
const HEADER_BYTES: usize = 48;
/// The three bytes that say a file is an entry of this tier.
const MAGIC: [u8; 3] = *b"KVC";
/// The envelope version. A file whose version byte differs is not an entry.
const VERSION: u8 = 1;
/// The payload ABI byte, which is separate from the envelope version because the
/// envelope can stay stable while the serialized path behind it becomes unsafe to
/// restore.
const PAYLOAD_ABI: u8 = 2;
/// The quantizations whose bytes this build can read back.
const QUANT_BITS_VALID: [u8; 2] = [2, 4];
/// Bytes of the text length that sits between the header and the text.
const TEXT_LEN_BYTES: usize = 4;
/// The budget a zero `budget_mb` means.
const DEFAULT_BUDGET_MB: u64 = 4096;
/// The one environment variable this crate reads, and the default it falls back to.
const HALF_LIFE_ENV: &str = "SHISO_KVSTORE_HIT_HALF_LIFE_SECONDS";
/// Six hours, which is the default half-life: long enough that a session's own
/// prefix outlives a working day, long enough that a weekend does not erase a
/// deployment's system prompt.
const DEFAULT_HALF_LIFE_S: u64 = 6 * 60 * 60;
/// Bytes in a mebibyte, which is what a `budget_mb` is multiplied by.
const MIB: u64 = 1024 * 1024;
/// Below this, a decayed hit count is treated as no evidence at all.
///
/// The floor is what stops a file read once four months ago from beating a file
/// that has never been read: the decay is continuous, and without a cutoff every
/// old entry keeps a nonzero score, so the pass would have no worthless entries to
/// pick first and would rank noise against noise.
const MIN_EFFECTIVE_HITS: f64 = 0.01;
/// The score multiplier for an anchor reason. See [`Reason::is_anchor`].
const ANCHOR_REASON_SCORE_FACTOR: f64 = 2.0;
/// The floor of the continued-prefix multiplier, for a superseded entry that has
/// never been read back.
const CONTINUED_PREFIX_MIN_FACTOR: f64 = 0.05;
/// How much of the continued-prefix multiplier is earned by hits.
const CONTINUED_PREFIX_HIT_FACTOR: f64 = 0.45;
/// The percent of headroom a write reserves above its own size, so a file cannot be
/// written to the last byte of the budget and then be deleted by the next pass.
const BUDGET_SLACK_PCT: u64 = 100;
/// The suffix of an entry's file name, which is also what makes a stray scratch
/// file in the directory obvious in a listing.
const ENTRY_SUFFIX: &str = ".kv";
/// Hex characters in a sha1 digest, which is the whole of a file name's stem.
const SHA_HEX_LEN: usize = 40;
/// Mode of the spill directory: owner-only, because the entries contain model state
/// derived from user prompts.
const DIR_MODE: u32 = 0o700;

/// A byte count or a span of seconds as a `f64`, for the scoring arithmetic.
///
/// `f64` represents every integer up to 2^53 exactly, and no filesystem byte count
/// or epoch-second span reaches that, so the conversion loses nothing.
#[expect(
    clippy::cast_precision_loss,
    reason = "exact below 2^53, which is above any byte count or epoch span here"
)]
fn as_f64(value: u64) -> f64 {
    value as f64
}

/// The four little-endian bytes at `bytes[..4]`.
///
/// The slice comes from a fixed-size header, so the length is a property of the
/// caller rather than of the file.
fn slice4(bytes: &[u8]) -> [u8; 4] {
    let mut out = [0u8; 4];
    out.copy_from_slice(bytes);
    out
}

/// The eight little-endian bytes at `bytes[..8]`.
fn slice8(bytes: &[u8]) -> [u8; 8] {
    let mut out = [0u8; 8];
    out.copy_from_slice(bytes);
    out
}

/// What the disk tier holds, as an operator sees it.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct SpillStats {
    /// The directory being measured.
    pub dir: PathBuf,
    /// Bytes the directory may hold.
    pub budget_bytes: u64,
    /// Bytes its entries occupy now.
    pub total_bytes: u64,
    /// Entries a read would accept, which is not the number of files in the
    /// directory: a file that fails its own gate is invisible here.
    pub entries: u32,
    /// Entries this process has deleted to stay inside the budget.
    pub evictions: u64,
}

/// How the disk tier is opened.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpillConfig {
    /// The directory the entries live in; created if absent.
    pub dir: PathBuf,
    /// The ceiling in mebibytes. Zero means [`DEFAULT_BUDGET_MB`].
    pub budget_mb: u64,
    /// Whether an entry written by a build with a different KV quantization is
    /// refused. True is the safe default: the bytes are not interchangeable.
    pub reject_different_quant: bool,
}

impl SpillConfig {
    /// A store rooted at `dir` with a `budget_mb` ceiling.
    #[must_use]
    pub fn new(dir: impl Into<PathBuf>, budget_mb: u64) -> Self {
        Self {
            dir: dir.into(),
            budget_mb,
            reject_different_quant: true,
        }
    }
}

/// One path, offered to the disk tier.
///
/// The header fields the store owns - hits, both timestamps, the payload length -
/// are not fields here, so a caller cannot write an entry that claims a hit count it
/// has not earned or a payload length its bytes do not have.
#[derive(Debug, Clone, Copy)]
pub struct Incoming<'a> {
    /// The rendered text prefix the entry is keyed by. The file name is the sha1 of
    /// exactly these bytes.
    pub text: &'a [u8],
    /// Tokens the path covers. Zero is refused: an empty prefix checkpoints nothing.
    pub tokens: u32,
    /// Quantization the payload's KV bytes were written with.
    pub quant_bits: u8,
    /// Which model's geometry the payload assumes.
    pub model_id: u8,
    /// Context size the path was written for.
    pub ctx_size: u32,
    /// Why it is being written, which is what the budget pass prices.
    pub reason: Reason,
    /// Presence bits for side tables the caller appended after the payload.
    pub ext_flags: u8,
    /// The serialized path, header included.
    pub payload: &'a [u8],
}

/// What a write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpillWrite {
    /// The bytes reached the disk.
    Written {
        /// The key they were stored under.
        id: SpillId,
        /// Bytes in the file, header and text included.
        bytes: u64,
    },
    /// A compatible entry was already there, so nothing was written.
    AlreadyStored {
        /// The existing entry.
        id: SpillId,
    },
}

/// One entry as the directory index sees it: its header, and nothing behind it.
///
/// Every field is a *claim the file makes about itself*, taken as it reads. Deciding
/// whether the claim is one this build can act on is the reader's, not this crate's:
/// the store knows how big a page is, and not which architecture wrote it.
///
/// The field set is the envelope's own, so like [`SpillHeader`] and
/// [`crate::PathHeader`] it is deliberately not `#[non_exhaustive]`: a field can only
/// appear here once it appears on disk, and the reader that has to decide what to do
/// with one of these has to be able to state a whole record in a test.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpillEntry {
    /// The key it is stored under, which is the sha1 of the text prefix it caches.
    pub id: SpillId,
    /// Tokens the stored path covers.
    pub tokens: u32,
    /// Bytes of the text prefix the key hashes. A reader looking for a prefix of its
    /// own request compares this against its own length before it hashes anything.
    pub text_bytes: u64,
    /// Which model's geometry the payload assumes.
    pub model_id: u8,
    /// Quantization the page bytes were written with.
    pub quant_bits: u8,
    /// Context size the path was prefilled under.
    pub ctx_size: u32,
    /// Why it was written.
    pub reason: Reason,
}

/// One entry read back.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StoredEntry {
    /// The key it was stored under.
    pub id: SpillId,
    /// Its header, hits and timestamps included.
    pub header: SpillHeader,
    /// The text prefix the key hashes to.
    pub text: Vec<u8>,
    /// The serialized path behind it.
    pub payload: Vec<u8>,
}

/// One directory of entries.
#[derive(Debug)]
pub struct SpillStore {
    dir: PathBuf,
    budget_bytes: u64,
    reject_different_quant: bool,
    half_life_s: u64,
    evictions: u64,
}

impl SpillStore {
    /// Opens `config.dir`, creating it at owner-only mode if it is absent, and runs
    /// one budget pass before returning.
    ///
    /// The pass at open is not a courtesy: a directory left over from a run with a
    /// larger budget has to come back inside this run's budget before anything is
    /// added to it, or the first write would be refused by entries that were already
    /// over the line.
    ///
    /// # Errors
    ///
    /// [`KvError::Io`] if the directory cannot be created or an entry inside it
    /// cannot be read or unlinked. A directory that already exists is not an error,
    /// and neither is a directory full of files this build cannot read: those are
    /// simply not entries.
    pub fn open(config: &SpillConfig) -> Result<Self, KvError> {
        fs::DirBuilder::new()
            .recursive(true)
            .mode(DIR_MODE)
            .create(&config.dir)
            .map_err(|source| KvError::io(&config.dir, source))?;
        let budget_mb = if config.budget_mb == 0 {
            DEFAULT_BUDGET_MB
        } else {
            config.budget_mb
        };
        let mut store = Self {
            dir: config.dir.clone(),
            budget_bytes: budget_mb.saturating_mul(MIB),
            reject_different_quant: config.reject_different_quant,
            half_life_s: half_life_seconds(),
            evictions: 0,
        };
        store.evict_to_fit(0, None)?;
        Ok(store)
    }

    /// The directory this store manages.
    #[must_use]
    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Bytes the directory may hold.
    #[must_use]
    pub fn budget_bytes(&self) -> u64 {
        self.budget_bytes
    }

    /// The file an id is stored under: `<dir>/<40 hex>.kv`.
    #[must_use]
    pub fn path_of(&self, id: SpillId) -> PathBuf {
        self.dir.join(format!("{}{ENTRY_SUFFIX}", id.to_hex()))
    }

    /// Whether a read of `id` would find something.
    #[must_use]
    pub fn contains(&self, id: SpillId) -> bool {
        self.path_of(id).is_file()
    }

    /// Writes one path, or reports that an equivalent one is already stored.
    ///
    /// The order is the C writer's order, and each step is there because a
    /// concurrent server may have done the same write: check for a compatible
    /// existing file first (so a duplicate write costs nothing), then price the
    /// write against the budget, then evict, then write to a temporary name and
    /// rename. The rename is what makes a reader see either the old directory state
    /// or the new entry, never a half-written file, and the temporary file is
    /// unlinked on every failure path so a failed write leaves no debris for the
    /// next scan to trip over.
    ///
    /// # Errors
    ///
    /// [`KvError::OverBudget`] if the entry could not fit even an empty directory,
    /// which is a caller sizing error rather than a full disk and should not be
    /// retried. [`KvError::Io`] if the temporary file, the rename, or the eviction
    /// pass fails; a failed write leaves no temporary file behind.
    pub fn write(&mut self, incoming: &Incoming<'_>) -> Result<SpillWrite, KvError> {
        let id = SpillId::of(incoming.text);
        let path = self.path_of(id);
        if self.existing_is_compatible(&path, id, incoming) {
            return Ok(SpillWrite::AlreadyStored { id });
        }
        let bytes =
            entry_bytes(incoming.text.len(), incoming.payload.len()).ok_or(KvError::BadHeader {
                reason: "entry size overflowed 64 bits",
            })?;
        let required = budget_required(bytes).ok_or(KvError::BadHeader {
            reason: "entry budget reservation overflowed",
        })?;
        if self.budget_bytes != 0 && required > self.budget_bytes {
            return Err(KvError::OverBudget {
                required,
                budget: self.budget_bytes,
            });
        }
        self.evict_to_fit(bytes, Some(incoming))?;

        let header = SpillHeader {
            quant_bits: incoming.quant_bits,
            reason: incoming.reason.as_u8(),
            ext_flags: incoming.ext_flags,
            model_id: incoming.model_id,
            tokens: incoming.tokens,
            hits: 0,
            ctx_size: incoming.ctx_size,
            payload_abi: PAYLOAD_ABI,
            created_at: now_secs(),
            last_used: 0,
            payload_bytes: u64::try_from(incoming.payload.len()).unwrap_or(u64::MAX),
        };
        if let Err(error) = write_entry(&path, &header, incoming.text, incoming.payload) {
            let _ = fs::remove_file(temporary_name(&path));
            return Err(error);
        }
        tracing::debug!(
            path = %path.display(),
            tokens = incoming.tokens,
            bytes,
            reason = ?incoming.reason,
            "spill entry written"
        );
        Ok(SpillWrite::Written { id, bytes })
    }

    /// Reads one entry back, or `None` when nothing is stored under `id`.
    ///
    /// A stored file is believed only after it has been checked against its own
    /// name: the text it carries is hashed and compared to the file name, so a file
    /// that was copied, renamed or truncated is refused with [`KvError::ShaMismatch`]
    /// rather than handed to the engine as if it were the prefix it claims to be.
    ///
    /// # Errors
    ///
    /// [`KvError::ShaMismatch`] if the file's stored text does not hash to `id`,
    /// which means the file is not the entry its name claims. Any other failure to
    /// read the file is reported as [`KvError::Io`]; a file that is simply absent,
    /// too short, or not restorable is `Ok(None)` rather than an error, because a
    /// miss is the normal answer from a cache.
    pub fn read(&self, id: SpillId) -> Result<Option<StoredEntry>, KvError> {
        let path = self.path_of(id);
        let mut file = match File::open(&path) {
            Ok(file) => file,
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(source) => return Err(KvError::io(path, source)),
        };
        let Some(header) = read_header(&mut file).filter(SpillHeader::is_restorable) else {
            return Ok(None);
        };
        let mut text_len = [0u8; TEXT_LEN_BYTES];
        if file.read_exact(&mut text_len).is_err() {
            return Ok(None);
        }
        let text_len = usize::try_from(u32::from_le_bytes(text_len)).unwrap_or(usize::MAX);
        let mut text = vec![0u8; text_len];
        if file.read_exact(&mut text).is_err() {
            return Ok(None);
        }
        if SpillId::of(&text) != id {
            return Err(KvError::ShaMismatch { id });
        }
        let payload_len = usize::try_from(header.payload_bytes).unwrap_or(usize::MAX);
        let mut payload = vec![0u8; payload_len];
        if file.read_exact(&mut payload).is_err() {
            return Ok(None);
        }
        Ok(Some(StoredEntry {
            id,
            header,
            text,
            payload,
        }))
    }

    /// Records that an entry was read: one more hit, and `last_used` is now.
    ///
    /// Rewritten in place rather than rewritten through a temp file, because the two
    /// fields it changes are only ever read by the budget pass. A torn write here
    /// costs one entry's score, not an entry's contents.
    ///
    /// # Errors
    ///
    /// [`KvError::Io`] if the entry cannot be opened or rewritten, and
    /// [`KvError::BadHeader`] if it is not an entry this build recognizes.
    pub fn touch(&mut self, id: SpillId) -> Result<(), KvError> {
        let path = self.path_of(id);
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|source| KvError::io(&path, source))?;
        let mut header = read_header(&mut file).ok_or(KvError::BadHeader {
            reason: "not a spill entry",
        })?;
        header.hits = header.hits.saturating_add(1);
        header.last_used = now_secs();
        file.seek(std::io::SeekFrom::Start(0))
            .map_err(|source| KvError::io(&path, source))?;
        file.write_all(&header.to_bytes())
            .map_err(|source| KvError::io(&path, source))
    }

    /// The directory's own numbers, counted from the files a read would accept.
    #[must_use]
    pub fn stats(&self) -> SpillStats {
        let scanned = self.scan();
        SpillStats {
            dir: self.dir.clone(),
            budget_bytes: self.budget_bytes,
            total_bytes: scanned.iter().map(|entry| entry.bytes).sum(),
            entries: to_u32(scanned.len()),
            evictions: self.evictions,
        }
    }

    /// What the directory holds, one record per entry a read would accept.
    ///
    /// The header of every acceptable file, and nothing behind it: an entry is tens
    /// or hundreds of mebibytes of pages, and a caller deciding *which* entry it
    /// wants must not have to read them all to find out. Whoever is looking for a
    /// prefix filters these and then reads exactly one.
    #[must_use]
    pub fn entries(&self) -> Vec<SpillEntry> {
        self.scan()
            .into_iter()
            .map(|found| SpillEntry {
                id: found.id,
                tokens: found.header.tokens,
                text_bytes: found.text_len,
                model_id: found.header.model_id,
                quant_bits: found.header.quant_bits,
                ctx_size: found.header.ctx_size,
                reason: Reason::from_byte(found.header.reason),
            })
            .collect()
    }

    /// Unlinks one entry, reporting whether there was one to unlink.
    ///
    /// The store's own budget pass deletes what it cannot afford; this is the other
    /// direction -- a reader that has decided a stored entry is one this build must
    /// never load again says so, and the file goes. `Ok(false)` when nothing was
    /// stored under `id`, because "it is not there" is the outcome the caller wanted
    /// rather than a failure.
    ///
    /// # Errors
    ///
    /// [`KvError::Io`] when the file exists and cannot be removed.
    pub fn forget(&mut self, id: SpillId) -> Result<bool, KvError> {
        let path = self.path_of(id);
        match fs::remove_file(&path) {
            Ok(()) => {
                tracing::debug!(path = %path.display(), "spill entry forgotten");
                Ok(true)
            }
            Err(source) if source.kind() == std::io::ErrorKind::NotFound => Ok(false),
            Err(source) => Err(KvError::io(path, source)),
        }
    }

    /// Lists the directory's acceptable entries.
    fn scan(&self) -> Vec<Scanned> {
        let mut found = Vec::new();
        let Ok(read_dir) = fs::read_dir(&self.dir) else {
            return found;
        };
        for dirent in read_dir.flatten() {
            let name = dirent.file_name();
            let name = name.to_string_lossy();
            let Some(stem) = entry_stem(&name) else {
                continue;
            };
            let Ok(id) = stem.parse::<SpillId>() else {
                continue;
            };
            let path = dirent.path();
            let Ok(metadata) = dirent.metadata() else {
                continue;
            };
            if !metadata.is_file() {
                continue;
            }
            let bytes = metadata.len();
            let Ok(mut file) = File::open(&path) else {
                continue;
            };
            let Some(header) = read_header(&mut file) else {
                continue;
            };
            if !header.is_restorable() {
                continue;
            }
            let mut text_len = [0u8; TEXT_LEN_BYTES];
            if file.read_exact(&mut text_len).is_err() {
                continue;
            }
            found.push(Scanned {
                id,
                path,
                bytes,
                header,
                text_len: u64::from(u32::from_le_bytes(text_len)),
            });
        }
        found
    }

    /// Deletes entries until `extra_bytes` fits, and reports how many went.
    ///
    /// The victim is the lowest-scoring entry, with the older `last_used` breaking a
    /// tie, and the score is the ported one: decayed hits times tokens-per-byte,
    /// doubled for an anchor reason and cut down for a `Continued` entry the incoming
    /// write supersedes. Deleting the oldest file instead would evict exactly the
    /// anchors the scoring exists to protect.
    fn evict_to_fit(
        &mut self,
        extra_bytes: u64,
        incoming: Option<&Incoming<'_>>,
    ) -> Result<u64, KvError> {
        if self.budget_bytes == 0 || extra_bytes > self.budget_bytes {
            return Ok(0);
        }
        let mut scanned = self.scan();
        let now = now_secs();
        let mut scores: Vec<f64> = scanned
            .iter()
            .map(|entry| self.score(entry, now, incoming))
            .collect();
        let mut total: u64 = scanned.iter().map(|entry| entry.bytes).sum();
        let target = self.budget_bytes - extra_bytes;
        let mut removed = 0u64;
        while total > target {
            let Some(victim) = lowest_score(&scanned, &scores) else {
                break;
            };
            let entry = scanned.remove(victim);
            scores.remove(victim);
            match fs::remove_file(&entry.path) {
                Ok(()) => {
                    tracing::debug!(
                        path = %entry.path.display(),
                        tokens = entry.header.tokens,
                        hits = entry.header.hits,
                        "spill entry evicted: directory over budget"
                    );
                    total = total.saturating_sub(entry.bytes);
                }
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => return Err(KvError::io(entry.path, source)),
            }
            removed += 1;
        }
        self.evictions = self.evictions.saturating_add(removed);
        Ok(removed)
    }

    /// The ported eviction score for one entry.
    fn score(&self, entry: &Scanned, now: u64, incoming: Option<&Incoming<'_>>) -> f64 {
        if entry.bytes == 0 {
            return 0.0;
        }
        let mut effective_hits = f64::from(entry.header.hits);
        let used_at = if entry.header.last_used != 0 {
            entry.header.last_used
        } else {
            entry.header.created_at
        };
        if used_at == 0 {
            effective_hits = 0.0;
        } else if now > used_at {
            let elapsed = as_f64(now - used_at);
            effective_hits *= (-elapsed / as_f64(self.half_life_s)).exp2();
            if effective_hits < MIN_EFFECTIVE_HITS {
                effective_hits = 0.0;
            }
        }
        let mut score =
            (effective_hits + 1.0) * f64::from(entry.header.tokens) / as_f64(entry.bytes);
        if Reason::from_byte(entry.header.reason).is_anchor() {
            score *= ANCHOR_REASON_SCORE_FACTOR;
        }
        if self.supersedes(entry, incoming) {
            let hits = if effective_hits > 0.0 {
                effective_hits / (effective_hits + 1.0)
            } else {
                0.0
            };
            score *= CONTINUED_PREFIX_MIN_FACTOR + CONTINUED_PREFIX_HIT_FACTOR * hits;
        }
        score
    }

    /// Whether the write in progress makes a stored `Continued` entry redundant.
    ///
    /// A continued entry is a prefix of a conversation; once a longer prefix that
    /// starts with it is being written, the shorter one is worth only what its own
    /// read history says, because resuming from the longer entry subsumes it. The
    /// check is on the bytes: the incoming text, truncated to the stored entry's
    /// text length, has to hash to the stored entry's name.
    fn supersedes(&self, entry: &Scanned, incoming: Option<&Incoming<'_>>) -> bool {
        let Some(incoming) = incoming else {
            return false;
        };
        if Reason::from_byte(entry.header.reason) != Reason::Continued {
            return false;
        }
        let stored_text_len = usize::try_from(entry.text_len).unwrap_or(usize::MAX);
        if stored_text_len == 0 || stored_text_len >= incoming.text.len() {
            return false;
        }
        if entry.header.model_id != incoming.model_id {
            return false;
        }
        if self.reject_different_quant && entry.header.quant_bits != incoming.quant_bits {
            return false;
        }
        if incoming.ctx_size > entry.header.ctx_size {
            return false;
        }
        SpillId::of(&incoming.text[..stored_text_len]) == entry.id
    }

    /// Whether the file at `path` already describes this exact path.
    ///
    /// Compatibility is model, quantization, context room, and then the text itself:
    /// the stored text has to equal the incoming text and hash to the name it is
    /// filed under. A file that fails is deleted rather than overwritten in place,
    /// because its name is a claim about its contents and a false claim is not
    /// something a later read should be able to make.
    fn existing_is_compatible(&self, path: &Path, id: SpillId, incoming: &Incoming<'_>) -> bool {
        let Ok(mut file) = File::open(path) else {
            return false;
        };
        let Some(header) = read_header(&mut file) else {
            return false;
        };
        let compatible = header.model_id == incoming.model_id
            && (!self.reject_different_quant || header.quant_bits == incoming.quant_bits)
            && header.ctx_size <= incoming.ctx_size
            && text_matches(&mut file, id, incoming.text);
        if !compatible {
            match fs::remove_file(path) {
                Ok(()) => {
                    tracing::debug!(path = %path.display(), "replaced incompatible spill file");
                }
                Err(source) if source.kind() == std::io::ErrorKind::NotFound => {}
                Err(source) => {
                    tracing::warn!(path = %path.display(), error = %source, "could not remove incompatible spill file");
                }
            }
        }
        compatible
    }
}

/// One directory entry as the scan saw it.
#[derive(Debug)]
struct Scanned {
    id: SpillId,
    path: PathBuf,
    bytes: u64,
    header: SpillHeader,
    text_len: u64,
}

/// The index of the entry the budget pass deletes first: lowest score, and the older
/// `last_used` breaks a tie so the pass is deterministic rather than directory order.
#[expect(
    clippy::float_cmp,
    reason = "the tie-break is an exact equality from the C pass: two entries scored from identical inputs must be ordered by last_used, not by an epsilon"
)]
fn lowest_score(scanned: &[Scanned], scores: &[f64]) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for (index, score) in scores.iter().copied().enumerate() {
        match best {
            None => best = Some((index, score)),
            Some((pick, pick_score)) => {
                if score < pick_score
                    || (score == pick_score
                        && scanned[index].header.last_used < scanned[pick].header.last_used)
                {
                    best = Some((index, score));
                }
            }
        }
    }
    best.map(|(index, _)| index)
}

/// Bytes a file holding `text_len` text and `payload_len` payload occupies.
fn entry_bytes(text_len: usize, payload_len: usize) -> Option<u64> {
    let fixed = u64::try_from(HEADER_BYTES + TEXT_LEN_BYTES).ok()?;
    fixed
        .checked_add(u64::try_from(text_len).ok()?)?
        .checked_add(u64::try_from(payload_len).ok()?)
}

/// The size a write of `bytes` reserves against the budget: itself plus 1% of
/// headroom, so accounting differences cannot leave a file that the next pass
/// immediately deletes.
fn budget_required(bytes: u64) -> Option<u64> {
    let slack = bytes / BUDGET_SLACK_PCT + u64::from(!bytes.is_multiple_of(BUDGET_SLACK_PCT));
    bytes.checked_add(slack)
}

/// The stem a directory entry must have to be an entry at all: 40 hex characters and
/// the entry suffix. Anything else in the directory is somebody else's file.
fn entry_stem(name: &str) -> Option<&str> {
    let stem = name.strip_suffix(ENTRY_SUFFIX)?;
    (stem.len() == SHA_HEX_LEN && stem.bytes().all(|byte| byte.is_ascii_hexdigit())).then_some(stem)
}

/// `<path>.tmp.<pid>`, the name a write lands under before the rename.
fn temporary_name(path: &Path) -> PathBuf {
    path.with_file_name(format!(
        "{}.tmp.{}",
        path.file_name().unwrap_or_default().to_string_lossy(),
        std::process::id()
    ))
}

/// Reads and parses the header of an open entry, without judging the rest.
fn read_header(file: &mut File) -> Option<SpillHeader> {
    let mut raw = [0u8; HEADER_BYTES];
    if file.read_exact(&mut raw).is_err() {
        return None;
    }
    if !SpillHeader::has_magic(&raw) || raw[3] != VERSION {
        return None;
    }
    Some(SpillHeader::from_bytes(&raw))
}

/// Whether the text stored behind `header` is exactly `text` and hashes to `id`.
fn text_matches(file: &mut File, id: SpillId, text: &[u8]) -> bool {
    let mut len_bytes = [0u8; TEXT_LEN_BYTES];
    if file.read_exact(&mut len_bytes).is_err() {
        return false;
    }
    let Ok(stored_len) = usize::try_from(u32::from_le_bytes(len_bytes)) else {
        return false;
    };
    if stored_len != text.len() {
        return false;
    }
    let mut stored = vec![0u8; stored_len];
    if file.read_exact(&mut stored).is_err() {
        return false;
    }
    stored == text && SpillId::of(&stored) == id
}

/// Writes `path` atomically: a temporary file in the same directory, then a rename.
fn write_entry(
    path: &Path,
    header: &SpillHeader,
    text: &[u8],
    payload: &[u8],
) -> Result<(), KvError> {
    let tmp = temporary_name(path);
    let result = (|| -> Result<(), std::io::Error> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .mode(0o600)
            .open(&tmp)?;
        file.write_all(&header.to_bytes())?;
        file.write_all(&to_u32(text.len()).to_le_bytes())?;
        file.write_all(text)?;
        file.write_all(payload)?;
        file.flush()?;
        file.sync_all()?;
        drop(file);
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result.map_err(|source| {
        KvError::io(
            if fs::exists(path).unwrap_or(false) {
                path
            } else {
                &tmp
            },
            source,
        )
    })
}

/// Seconds since the epoch, or zero on a clock that cannot be read.
///
/// Zero is the safe answer, because every use of it is a comparison against a stored
/// timestamp: a clock reading zero makes every entry look old, so the pass evicts by
/// density rather than by recency, which is wrong in a visible way rather than
/// silently keeping everything forever.
fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.as_secs())
}

/// The hit half-life, read once from [`HALF_LIFE_ENV`].
///
/// Cached because it is read on every scoring pass and an environment variable that
/// changes mid-run would make two entries written a second apart be scored against
/// different clocks.
#[must_use]
fn half_life_seconds() -> u64 {
    static CACHED: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| half_life_from(std::env::var_os(HALF_LIFE_ENV).as_deref()))
}

/// Reads the half-life knob, falling back to the default when it is absent or
/// unreadable.
///
/// A value that does not parse is not a reason to refuse to start: the default is a
/// defensible six hours, and a typo in a deployment's environment should cost a
/// scoring curve rather than the server. Zero is included in "unreadable" because a
/// zero half-life would make every decayed hit count collapse to the floor.
fn half_life_from(raw: Option<&std::ffi::OsStr>) -> u64 {
    let Some(raw) = raw else {
        return DEFAULT_HALF_LIFE_S;
    };
    let raw = raw.to_string_lossy();
    match raw.trim().parse::<u64>() {
        Ok(seconds) if seconds > 0 => seconds,
        _ => {
            tracing::warn!(
                value = %raw,
                default = DEFAULT_HALF_LIFE_S,
                "ignoring invalid {HALF_LIFE_ENV}"
            );
            DEFAULT_HALF_LIFE_S
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DEFAULT_HALF_LIFE_S, half_life_from};
    use std::ffi::OsStr;

    #[test]
    fn half_life_knob_falls_back_to_six_hours() {
        assert_eq!(half_life_from(None), DEFAULT_HALF_LIFE_S);
        assert_eq!(DEFAULT_HALF_LIFE_S, 21_600);
    }

    #[test]
    fn half_life_knob_accepts_a_positive_count_of_seconds() {
        assert_eq!(half_life_from(Some(OsStr::new("60"))), 60);
        assert_eq!(half_life_from(Some(OsStr::new("  90 "))), 90);
    }

    #[test]
    fn half_life_knob_refuses_values_that_would_break_the_decay() {
        // Zero would divide the elapsed span by zero, and a negative or
        // non-numeric value is a typo rather than a policy.
        assert_eq!(half_life_from(Some(OsStr::new("0"))), DEFAULT_HALF_LIFE_S);
        assert_eq!(half_life_from(Some(OsStr::new("-5"))), DEFAULT_HALF_LIFE_S);
        assert_eq!(
            half_life_from(Some(OsStr::new("six hours"))),
            DEFAULT_HALF_LIFE_S
        );
    }
}
