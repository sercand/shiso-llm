//! The ids this crate hands out.
//!
//! Each is a distinct index newtype: a [`PageId`] passed where a [`NodeId`]
//! belongs is a compile error, and no caller can fabricate one that points
//! outside the arena or the pool. Ids are minted by the owner of the table they
//! index into (the pool mints pages, the tree mints nodes, the checkpoint store
//! mints slots), which is what makes them safe to hold on to.
//!
//! There is no "no node" value. An absent parent, child or checkpoint is
//! `Option<NodeId>`, never a magic maximum.

use std::fmt;
use std::str::FromStr;

use sha1::Digest as _;
use sha1::Sha1;

use crate::error::KvError;

/// Generates the three index newtypes. They differ only in which table they
/// index, which is exactly the distinction Rust can enforce and an integer
/// cannot.
macro_rules! id_newtype {
    ($ty:ident, $doc:expr) => {
        #[doc = $doc]
        #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub struct $ty(u32);

        impl $ty {
            /// Mints the id of an existing table slot.
            ///
            /// Crate-private: the pool mints pages, the tree mints nodes and the
            /// checkpoint store mints slots, so no caller can fabricate an id
            /// that points outside the table it indexes.
            #[must_use]
            pub(crate) const fn new(index: u32) -> Self {
                Self(index)
            }

            /// The raw index, for a kernel's index array and for a caller that
            /// keys its own side table by it.
            #[must_use]
            pub const fn get(self) -> u32 {
                self.0
            }

            /// The index as a `usize`, for slicing the table it belongs to.
            #[must_use]
            pub(crate) const fn index(self) -> usize {
                self.0 as usize
            }
        }

        impl From<$ty> for u32 {
            fn from(id: $ty) -> Self {
                id.0
            }
        }

        impl fmt::Display for $ty {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, "{}({})", stringify!($ty), self.0)
            }
        }
    };
}

id_newtype!(
    PageId,
    "A page in the pool: one immutable `page_bytes` extent of key/value storage.\n\nThe id doubles as the `u32` a kernel indexes its page table through, so the tree, the page table and the device-side index array all name a page with the same value. The bytes themselves live in a `Backing` the store never sees."
);

id_newtype!(
    NodeId,
    "A node of the span tree: one contiguous span of tokens, the pages holding their key/value rows, and the checkpoint taken at the end of the span."
);

id_newtype!(
    SlotId,
    "A checkpoint slot: one snapshot of the recurrent state stack, sized by `CacheGeometry::ckpt_bytes`.\n\nSlots are recycled, so a [`crate::CkptRef`] is checked against the store on every use rather than trusted."
);

/// `usize` to `u32` for counts bounded by the pool, arena and slot counts.
///
/// Every call site is a length of something this crate allocated, so the value
/// fits; saturating keeps a hypothetical 16-gigapixel pool from panicking in a
/// stats getter.
#[must_use]
pub(crate) fn to_u32(value: usize) -> u32 {
    // The one saturating cast in the crate, and the one place `u32::MAX` appears in
    // code rather than prose. It is not a sentinel: nothing reads it back as "no such
    // thing" -- it is a count too large for the field that has to hold it.
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// The same saturation for a byte total, which arrives as a `u64`.
#[must_use]
pub(crate) fn to_u32_wide(value: u64) -> u32 {
    to_u32(usize::try_from(value).unwrap_or(usize::MAX))
}

/// The name of a spilled checkpoint: the sha1 of the rendered byte prefix it
/// caches.
///
/// The digest is kept as bytes, not as a `String`, because the 40 hex characters
/// are a file name and nothing else: comparing, hashing and storing 20 bytes is
/// cheaper, and a `SpillId` can never be a name that sha1 could not have
/// produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpillId([u8; 20]);

impl SpillId {
    /// The digest of a byte prefix, which is how the disk tier keys its index.
    #[must_use]
    pub fn of(text: &[u8]) -> Self {
        let mut digest = Sha1::new();
        digest.update(text);
        Self(digest.finalize().into())
    }

    /// The raw digest bytes, for callers that key their own index by them.
    #[must_use]
    pub const fn as_bytes(&self) -> &[u8; 20] {
        &self.0
    }

    /// The 40 lowercase hex characters this id is named by on disk.
    #[must_use]
    pub fn to_hex(self) -> String {
        hex::encode(self.0)
    }
}

impl From<[u8; 20]> for SpillId {
    fn from(raw: [u8; 20]) -> Self {
        Self(raw)
    }
}

impl fmt::Display for SpillId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for byte in self.0 {
            write!(f, "{byte:02x}")?;
        }
        Ok(())
    }
}

impl FromStr for SpillId {
    type Err = KvError;

    /// Parses the 40 hex characters of a spill name. Anything else — wrong
    /// length, a stray newline in a name read back from a directory listing —
    /// is refused rather than folded into a valid-looking id.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let digits = text.as_bytes();
        if digits.len() != 40 {
            return Err(KvError::BadSpillId {
                found: text.to_string(),
            });
        }
        let mut raw = [0u8; 20];
        for (index, byte_slot) in raw.iter_mut().enumerate() {
            let Some(high) = hex_value(digits[index * 2]) else {
                return Err(KvError::BadSpillId {
                    found: text.to_string(),
                });
            };
            let Some(low) = hex_value(digits[index * 2 + 1]) else {
                return Err(KvError::BadSpillId {
                    found: text.to_string(),
                });
            };
            *byte_slot = high << 4 | low;
        }
        Ok(Self(raw))
    }
}

/// The value of one ASCII hex digit, either case.
fn hex_value(digit: u8) -> Option<u8> {
    match digit {
        b'0'..=b'9' => Some(digit - b'0'),
        b'a'..=b'f' => Some(digit - b'a' + 10),
        b'A'..=b'F' => Some(digit - b'A' + 10),
        _ => None,
    }
}
