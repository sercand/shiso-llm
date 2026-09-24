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
