//! Per-row page tables for a decode step that carries several sessions.
//!
//! A prefill chunk and a single-session decode step are one session's own
//! consecutive positions, so every KV-touching launch of one reads one page
//! table. A batched decode step is not: row `i` belongs to session `i`, and two
//! sessions share no page. The kernels select row `t`'s table at
//! `table + t * stride` ([`crate::attn::Pages`]); this is what writes that block
//! and what says how wide a row of it is.
//!
//! # A packed block, not an array of pointers
//!
//! The obvious shape is one device pointer per row. This workspace has no way
//! to write one: a [`shiso_core::DeviceView`] is a buffer id, an offset and a
//! length, and the backend turns it into an address inside `pack_args`, one
//! pointer per declared kernel parameter. An array of device addresses would be
//! a new backend capability rather than a new argument. One block of
//! fixed-width rows, copied into on the device, is the same per-row table
//! through one pointer and one `uint32_t`.
//!
//! The copies are cheap for the reason the tables are small: a row is
//! `pages_for(ctx)` `int32` slots - 32 of them at the server's 8192-token
//! default - so a batch of eight costs eight device-to-device copies of 128
//! bytes each, against a step that reads gigabytes of weights.
//!
//! # When it is packed, and why that is not a detail
//!
//! `sparse/run.rs`'s ordering rule applies here for the same reason it applies
//! there: **the block is packed after every member has been advanced and before
//! anything is issued.** Advancing a session claims the pages its next token
//! writes and re-uploads that session's own table, so a block packed before the
//! advance holds the table as it was one token ago - which is the right table
//! until the step crosses a page boundary, and then silently is not. Packing it
//! after the pass would be worse still: every launch would dereference whatever
//! the slot last held, which is a wrong answer and not a crash.

use std::sync::Arc;

use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc};
use shiso_core::error::{CoreError, CoreResult};

use crate::attn::Pages;
use crate::kv::PAGE_TABLE_ENTRY_BYTES;

/// The block a batched step's page tables are packed into.
///
/// Allocated once with the stack, at the width one session's table has, so a
/// step never allocates and a launch never races a reallocation.
#[derive(Debug)]
pub(crate) struct RowPages {
    block: DeviceBuffer,
    /// `int32` slots one row of the block holds, which is the entry count of
    /// one session's own table.
    entries: u32,
    /// Rows the block holds, which is the widest batch the stack accepts.
    rows: u32,
}

impl RowPages {
    /// Allocates room for `rows` tables of a session whose context is `ctx`.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when the allocation fails.
    pub(crate) fn new(backend: &Arc<dyn Backend>, ctx: u32, rows: u32) -> CoreResult<Self> {
        let entries = shiso_core::page::pages_for(ctx);
        let bytes = u64::from(entries)
            .saturating_mul(u64::from(rows))
            .saturating_mul(PAGE_TABLE_ENTRY_BYTES);
        Ok(Self {
            block: alloc(backend, bytes)?,
            entries,
            rows,
        })
    }

    /// Copies each member's own table into its row and answers what the
    /// launches read.
    ///
    /// `tables` is one view per row, in the batch's own order, and each is a
    /// whole session page table as [`crate::kv::SessionKv::view`] hands it out.
    /// A view narrower than a row is copied into the front of that row and the
    /// slots behind it keep whatever a previous batch left - which is safe for
    /// the same reason the unused tail of a session's own table is safe: no
    /// position the causal bound admits reaches them.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] when the batch is wider than the block,
    /// [`CoreError::OutOfBounds`] when a member's table is wider than a row,
    /// and whatever the device reports for a copy.
    pub(crate) fn pack<'a>(
        &self,
        backend: &dyn Backend,
        tables: impl ExactSizeIterator<Item = DeviceView<'a>>,
    ) -> CoreResult<Pages<'_>> {
        let count = u32::try_from(tables.len()).unwrap_or(u32::MAX);
        if count > self.rows {
            return Err(CoreError::Unsupported {
                what: "a decode batch wider than the packed page-table block",
                backend: "engine",
            });
        }
        let row_bytes = u64::from(self.entries).saturating_mul(PAGE_TABLE_ENTRY_BYTES);
        for (row, table) in tables.enumerate() {
            let at = row_bytes.saturating_mul(u64::try_from(row).unwrap_or(u64::MAX));
            let len = table.len().min(row_bytes);
            backend.copy_d2d(self.block.slice(at, len)?, table.slice(0, len)?)?;
        }
        Ok(Pages::per_row(self.block.view(), self.entries))
    }
}
