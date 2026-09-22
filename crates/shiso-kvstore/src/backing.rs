//! The byte mover: the only way page bytes reach this crate.
//!
//! The cache is bookkeeping. It decides *which* page a position lives in, who is
//! holding it, and what it is worth; it never holds a page's bytes and never sees a
//! device pointer. [`Backing`] is the seam: the engine implements it over whatever
//! actually stores key/value rows (a device buffer, a host arena, a memory-mapped
//! file), and the cache calls it at exactly three moments.
//!
//! Keeping the seam this narrow is what lets one cache serve a Metal-resident
//! tensor on one platform and a host-side `Vec<u8>` in a test, with no backend
//! dependency in this crate at all.
//!
//! # The three operations, and when each is called
//!
//! [`Backing::copy_page`] is called by `lookup` and `restore`, once, for the single
//! page a resume point lands inside. A page past the frontier is immutable, so a
//! sequence that resumes *inside* a shared page cannot write its new rows there
//! without corrupting every other path through it; it takes a private copy of that
//! one page instead. That is the whole copy-on-write cost of a resume.
//!
//! [`Backing::page_to_host`] is called by `spill`, once per page of the path being
//! written, and [`Backing::page_from_host`] by `restore`, once per page it has to
//! re-create. Both are the engine's chance to do one large transfer rather than
//! many small ones, which is why the buffer is a whole page.
//!
//! # What an implementation owes the cache
//!
//! A page id is an index the *engine* translates, so the cache's only promise is
//! that it never hands back an id it did not obtain from the pool, and never reuses
//! one while a holder remains. In exchange:
//!
//! - `copy_page` must leave `src` readable and `dst` fully written, including the
//!   unfilled tail of a partly-used page: a page is copied whole, because the cache
//!   does not know how wide a row is.
//! - A failure must leave both pages usable. The cache does not roll a failed copy
//!   back, it releases the destination and reports the error, so a half-written
//!   destination must at least be a page the pool can hand out again.
//! - None of the three may call back into the cache. They run while the cache's
//!   inner lock is held, and the lock order is inner-then-pool with no third party
//!   allowed in the middle.

use crate::error::KvError;
use crate::ids::PageId;

/// Moves the bytes the cache only counts.
///
/// `Send` because the cache is shared between request threads; `Debug` because a
/// cache's `Debug` output is the first thing read when a resume misbehaves, and a
/// page table whose backing cannot be named is only half a diagnosis.
pub trait Backing: Send + std::fmt::Debug {
    /// Copies every row of page `src` into page `dst`.
    ///
    /// Called once per resume that lands inside a shared page. `dst` is a page the
    /// cache has just claimed from the pool and holds one reference to, so an
    /// implementation may assume nothing about its previous contents.
    ///
    /// # Errors
    ///
    /// Any [`KvError`] describing the transfer failure. The cache releases `dst` and
    /// abandons the resume, so a failure must not leave the pool inconsistent.
    fn copy_page(&mut self, dst: PageId, src: PageId) -> Result<(), KvError>;

    /// Copies one page's bytes into a caller-owned buffer.
    ///
    /// `buf.len()` is the geometry's `page_bytes`; a shorter buffer is a caller bug
    /// and is reported as [`KvError::BufferLen`] rather than partially filled. Used
    /// by the spill writer, one page at a time.
    ///
    /// # Errors
    ///
    /// [`KvError::BufferLen`] if `buf` is not a whole page, or a transfer error.
    fn page_to_host(&self, page: PageId, buf: &mut [u8]) -> Result<(), KvError>;

    /// Writes a whole page's bytes from a caller-owned buffer.
    ///
    /// The counterpart of [`Self::page_to_host`], used when a spilled path is
    /// adopted: the pages above the match are freshly claimed, and this is what puts
    /// their rows back. A page is written whole for the same reason it is copied
    /// whole -- the cache does not know where one row ends.
    ///
    /// # Errors
    ///
    /// [`KvError::BufferLen`] if `buf` is not a whole page, or a transfer error.
    fn page_from_host(&mut self, page: PageId, buf: &[u8]) -> Result<(), KvError>;
}
