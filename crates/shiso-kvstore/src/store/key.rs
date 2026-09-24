//! Content keys: what a page and a checkpoint are named by on disk.
//!
//! A page is named by everything that decides its rows: the build that computed
//! them, the page before it, and its own token ids. Chaining the parent's key in
//! is what makes one name stand for a whole prefix rather than for 256 tokens that
//! might sit anywhere, so two conversations that share an opening share every page
//! of it, and a page is written once however many entries list it.
//!
//! A checkpoint is named by the key of the last page its path touches and its
//! exact position. The last page may be partial, and a partial page's key hashes
//! its own token count, so a checkpoint at 300 and one at 301 name different
//! partial pages and can never be confused for each other.
//!
//! # Why the build is in every key
//!
//! [`Identity`] folds the layout words and the kernel set in. Two builds whose
//! kernels compute different bits produce different rows for the same tokens, and
//! a resume from the other build's rows would answer fluently and differently from
//! a cold run. Putting the identity in the name makes that a miss by construction:
//! the other build's files are simply never looked up.

use std::fmt;

use sha1::{Digest as _, Sha1};
use shiso_core::TokenId;

use crate::geom::{CacheGeometry, PathLayout};

/// Bytes in every key: a sha1 digest.
pub(crate) const KEY_BYTES: usize = 20;

/// Domain tags, so a page key and a checkpoint key over the same bytes differ.
const PAGE_TAG: &[u8] = b"shiso-kv2/page";
const ENTRY_TAG: &[u8] = b"shiso-kv2/ckpt";
const IDENTITY_TAG: &[u8] = b"shiso-kv2/identity";

/// Writes the 40 lowercase hex characters of a digest.
fn write_hex(f: &mut fmt::Formatter<'_>, bytes: &[u8; KEY_BYTES]) -> fmt::Result {
    for byte in bytes {
        write!(f, "{byte:02x}")?;
    }
    Ok(())
}

/// Parses 40 hex characters, either case, into a digest.
fn parse_hex(text: &str) -> Option<[u8; KEY_BYTES]> {
    let digits = text.as_bytes();
    if digits.len() != KEY_BYTES * 2 {
        return None;
    }
    let mut raw = [0u8; KEY_BYTES];
    for (index, byte) in raw.iter_mut().enumerate() {
        let high = hex_value(digits[index * 2])?;
        let low = hex_value(digits[index * 2 + 1])?;
        *byte = high << 4 | low;
    }
    Some(raw)
}

/// The value of one ASCII hex digit.
const fn hex_value(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}

/// The name of one stored checkpoint: a path's recurrent state at one position.
///
/// Public because an operator reading a log line, and a caller told what a
/// persist did, both need to name the entry; nothing outside this crate can mint
/// one, so a key always names a digest this crate computed.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntryKey([u8; KEY_BYTES]);

/// The name of one stored page: its rows, and through its parent every row
/// before it.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct PageKey([u8; KEY_BYTES]);

/// What build wrote a file: the layout words, the page geometry and the kernel
/// set, folded into one digest.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Identity([u8; KEY_BYTES]);

macro_rules! digest_display {
    ($ty:ident) => {
        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write_hex(f, &self.0)
            }
        }

        impl fmt::Debug for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}(", stringify!($ty))?;
                write_hex(f, &self.0)?;
                write!(f, ")")
            }
        }
    };
}

digest_display!(EntryKey);
digest_display!(PageKey);
digest_display!(Identity);

impl EntryKey {
    /// Parses the 40 hex characters an entry file is named by.
    #[must_use]
    pub(crate) fn parse(text: &str) -> Option<Self> {
        parse_hex(text).map(Self)
    }
}

impl PageKey {
    /// The raw digest.
    #[must_use]
    pub(crate) const fn bytes(&self) -> &[u8; KEY_BYTES] {
        &self.0
    }

    /// A key read back off disk, taken as it reads.
    #[must_use]
    pub(crate) const fn from_bytes(raw: [u8; KEY_BYTES]) -> Self {
        Self(raw)
    }

    /// Parses the 40 hex characters a page file is named by.
    #[must_use]
    pub(crate) fn parse(text: &str) -> Option<Self> {
        parse_hex(text).map(Self)
    }
}

impl Identity {
    /// The raw digest.
    #[must_use]
    pub(crate) const fn bytes(&self) -> &[u8; KEY_BYTES] {
        &self.0
    }

    /// An identity read back off disk, taken as it reads.
    #[must_use]
    pub(crate) const fn from_bytes(raw: [u8; KEY_BYTES]) -> Self {
        Self(raw)
    }
}

impl Identity {
    /// The identity of a build: every layout word, the page and checkpoint sizes,
    /// and the kernel set's fingerprint.
    ///
    /// The layout words are hashed in declaration order with fixed widths, so a
    /// field whose value changes changes the digest; the geometry sizes are
    /// hashed because a page of another size is another file format; and
    /// `numerics` is whatever string the device says names the arithmetic it
    /// runs.
    #[must_use]
    pub(crate) fn of(layout: &PathLayout, geometry: &CacheGeometry, numerics: &str) -> Self {
        let mut digest = Sha1::new();
        digest.update(IDENTITY_TAG);
        digest.update([layout.model_id, layout.quant_bits]);
        for word in [
            layout.ctx_size,
            layout.kv_width,
            layout.n_layers,
            layout.n_attn_layers,
            layout.n_gdn_layers,
            layout.vocab,
            layout.gdn_conv_elements,
            layout.gdn_state_elements,
            layout.ple_window_elements,
            layout.hc_width,
        ] {
            digest.update(word.to_le_bytes());
        }
        digest.update([u8::from(layout.indexer), u8::from(layout.draft_head)]);
        digest.update(geometry.page_tokens.to_le_bytes());
        digest.update(geometry.page_bytes.to_le_bytes());
        digest.update(geometry.ckpt_bytes.to_le_bytes());
        let numerics = numerics.as_bytes();
        digest.update(
            u64::try_from(numerics.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        digest.update(numerics);
        Self(digest.finalize().into())
    }
}

/// The key of a page holding `tokens`, whose predecessor is `parent`.
///
/// `parent` is `None` for the first page of a path. The token count is hashed,
/// which is what separates a partial page from the full page that later grows
/// out of the same opening.
#[must_use]
pub(crate) fn page_key(
    identity: &Identity,
    parent: Option<&PageKey>,
    tokens: &[TokenId],
) -> PageKey {
    let mut digest = Sha1::new();
    digest.update(PAGE_TAG);
    digest.update(identity.bytes());
    digest.update(parent.map_or([0u8; KEY_BYTES], |key| *key.bytes()));
    digest.update(
        u32::try_from(tokens.len())
            .unwrap_or(u32::MAX)
            .to_le_bytes(),
    );
    for token in tokens {
        digest.update(token.get().to_le_bytes());
    }
    PageKey(digest.finalize().into())
}

/// The key of a checkpoint at `pos`, whose path's last page is `last_page`.
#[must_use]
pub(crate) fn entry_key(identity: &Identity, last_page: &PageKey, pos: u32) -> EntryKey {
    let mut digest = Sha1::new();
    digest.update(ENTRY_TAG);
    digest.update(identity.bytes());
    digest.update(last_page.bytes());
    digest.update(pos.to_le_bytes());
    EntryKey(digest.finalize().into())
}

/// The keys of one token path: one per full page, computed once so that every
/// candidate position a lookup tries costs at most one partial page's hash.
#[derive(Debug, Clone)]
pub(crate) struct PathKeys<'a> {
    identity: Identity,
    tokens: &'a [TokenId],
    page_tokens: usize,
    /// The key of every full page of `tokens`, first page first.
    full: Vec<PageKey>,
}

impl<'a> PathKeys<'a> {
    /// Hashes every full page of `tokens`.
    #[must_use]
    pub(crate) fn new(identity: Identity, tokens: &'a [TokenId], page_tokens: u32) -> Self {
        let page_tokens = usize::try_from(page_tokens).unwrap_or(usize::MAX).max(1);
        let mut full: Vec<PageKey> = Vec::with_capacity(tokens.len() / page_tokens);
        for chunk in tokens.chunks_exact(page_tokens) {
            let key = page_key(&identity, full.last(), chunk);
            full.push(key);
        }
        Self {
            identity,
            tokens,
            page_tokens,
            full,
        }
    }

    /// The keys of the pages covering positions `[0, pos)`, the last one partial
    /// when `pos` is not a page multiple. Empty for `pos == 0` or a `pos` past the
    /// path.
    #[must_use]
    pub(crate) fn pages_to(&self, pos: u32) -> Vec<PageKey> {
        let pos = usize::try_from(pos).unwrap_or(usize::MAX);
        if pos == 0 || pos > self.tokens.len() {
            return Vec::new();
        }
        let whole = pos / self.page_tokens;
        let mut out = self.full[..whole].to_vec();
        if pos % self.page_tokens != 0 {
            let start = whole * self.page_tokens;
            out.push(page_key(
                &self.identity,
                out.last(),
                &self.tokens[start..pos],
            ));
        }
        out
    }

    /// The key of the checkpoint at `pos` on this path, or `None` when `pos` is
    /// zero or past the path.
    #[must_use]
    pub(crate) fn entry_at(&self, pos: u32) -> Option<EntryKey> {
        let at = usize::try_from(pos).unwrap_or(usize::MAX);
        if pos == 0 || at > self.tokens.len() {
            return None;
        }
        let whole = at / self.page_tokens;
        let last = if at % self.page_tokens == 0 {
            self.full[whole - 1]
        } else {
            let start = whole * self.page_tokens;
            let parent = whole.checked_sub(1).map(|index| &self.full[index]);
            page_key(&self.identity, parent, &self.tokens[start..at])
        };
        Some(entry_key(&self.identity, &last, pos))
    }
}

#[cfg(test)]
mod tests {
    use super::{EntryKey, Identity, PageKey, PathKeys, page_key};
    use shiso_core::TokenId;

    fn identity() -> Identity {
        Identity::from_bytes([7; 20])
    }

    fn tokens(n: u32) -> Vec<TokenId> {
        (0..n).map(TokenId::new).collect()
    }

    #[test]
    fn a_partial_page_is_named_apart_from_the_full_page_it_grows_into() {
        let path = tokens(512);
        let keys = PathKeys::new(identity(), &path, 256);
        let at_300 = keys.pages_to(300);
        let at_512 = keys.pages_to(512);
        assert_eq!(at_300.len(), 2);
        assert_eq!(at_512.len(), 2);
        assert_eq!(at_300[0], at_512[0], "the shared full page has one name");
        assert_ne!(at_300[1], at_512[1], "a partial page is not the full one");
        assert_ne!(keys.entry_at(300), keys.entry_at(301));
    }

    #[test]
    fn a_page_key_names_the_prefix_before_it() {
        let a = tokens(512);
        let mut b = tokens(512);
        b[3] = TokenId::new(99);
        let ka = PathKeys::new(identity(), &a, 256);
        let kb = PathKeys::new(identity(), &b, 256);
        // The second pages hold the same tokens, and still differ, because the
        // first pages did.
        assert_eq!(&a[256..], &b[256..]);
        assert_ne!(ka.pages_to(512)[1], kb.pages_to(512)[1]);
    }

    #[test]
    fn a_different_build_names_every_page_differently() {
        let path = tokens(256);
        let one = page_key(&identity(), None, &path);
        let other = page_key(&Identity::from_bytes([8; 20]), None, &path);
        assert_ne!(one, other);
    }

    #[test]
    fn keys_round_trip_through_their_file_names() {
        let key = PathKeys::new(identity(), &tokens(10), 256)
            .entry_at(10)
            .expect("inside the path");
        assert_eq!(EntryKey::parse(&key.to_string()), Some(key));
        assert!(PageKey::parse("not hex").is_none());
        assert!(
            PathKeys::new(identity(), &tokens(10), 256)
                .entry_at(11)
                .is_none()
        );
        assert!(
            PathKeys::new(identity(), &tokens(10), 256)
                .entry_at(0)
                .is_none()
        );
    }
}
