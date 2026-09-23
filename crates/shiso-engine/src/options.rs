//! The options every model in this crate needs, defined once.
//!
//! A struct with defaults rather than a builder chain: every field has a working
//! default, the set is small, and a chain of setters cannot express "these three
//! are the whole set". The same struct is what a second architecture in this crate
//! would reuse, which is the only reason it is a separate type at all - three
//! fields duplicated across two option structs would be three chances for the
//! defaults to drift apart.
//!
//! # Why the KV numbers are here and the geometry is not
//!
//! `kv_pool_pages` and `kv_ckpt_slots` are counts of *cache units*, so a caller can
//! set them without knowing a model's page size. The byte sizes those counts are
//! multiplied by are derived from the shape profile in [`crate::kv`], because a
//! page that is the wrong size for the layer pattern is not a cache unit at all.
//!
//! # Why a fraction of the device is resolved by the stack
//!
//! [`KvFit`] asks for "whatever is left of this share of the device", and only
//! the stack can answer that: it is the party that knows a page's bytes and a
//! session's state, and it answers after its weights and scratch are resident,
//! from what the driver reports then. It writes the answer back into
//! `kv_pool_pages`, so everything that reads the pool size afterwards reads the
//! size that was allocated.

use shiso_core::error::{CoreError, CoreResult};
use shiso_core::page::{PAGE_TOKENS, pages_for};
use shiso_core::{Backend, MemoryInfo};

use crate::mtp::SpecDepth;

/// Everything a model in this crate needs to be built, in one place.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct EngineOptions {
    /// Tokens one session may hold, prompt included.
    ///
    /// This is the cache's ceiling, not the model's: these checkpoints have no
    /// positional limit of their own, so a longer context costs pool pages and
    /// nothing else.
    pub ctx: u32,
    /// Pages in the KV pool, which is the cache's whole memory budget expressed
    /// in [`crate::kv::PAGE_TOKENS`]-token units.
    ///
    /// Pages are claimed lazily, so this is a ceiling on what the cache may hold
    /// rather than an allocation: an over-generous number costs a vector of
    /// refcounts.
    pub kv_pool_pages: u32,
    /// Checkpoint slots, which bounds how many resumable positions the cache can
    /// carry at once.
    ///
    /// A slot's blob is allocated when the slot is first filled, so this is the
    /// one number that bounds the recurrent-state tier directly.
    pub kv_ckpt_slots: u32,
    /// How many tokens a step may draft, which is also what sizes the
    /// per-row recurrent records every session allocates.
    ///
    /// [`SpecDepth::NONE`] is the default and is not a placeholder: a
    /// speculative session carries one state matrix per drafted row per
    /// recurrent slot, which on this family is 2 MiB times the depth times
    /// twenty-four, and a build with no draft head would be paying all of it
    /// for records nothing writes.
    pub spec: SpecDepth,
    /// Size the pool to a share of the device instead of to `kv_pool_pages`.
    ///
    /// `None`, the default, is the page count as given.
    pub kv_fit: Option<KvFit>,
}

impl EngineOptions {
    /// The defaults a server starts with: 4096 tokens, 512 pool pages, 16
    /// checkpoint slots, no spill tier, no speculation.
    ///
    /// The pool default is 512 pages, which at this page size is 4 GiB of address
    /// space that is only touched as sessions fill it.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ctx: 4096,
            kv_pool_pages: 512,
            kv_ckpt_slots: 16,
            spec: SpecDepth::NONE,
            kv_fit: None,
        }
    }

    /// Resolves [`EngineOptions::kv_fit`] into `kv_pool_pages`, when set.
    ///
    /// Called by a stack once everything it allocates up front, other than
    /// the pool, is resident - so the driver's free figure already has its
    /// weights and scratch taken out of it.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] on a device that cannot report its memory,
    /// and whatever [`KvFit::pool_pages`] reports.
    pub fn fit_pool(&mut self, backend: &dyn Backend, cost: PoolCost) -> CoreResult<()> {
        let Some(fit) = self.kv_fit else {
            return Ok(());
        };
        let info = backend.memory_info().ok_or(CoreError::Unsupported {
            what: "sizing the cache to a share of device memory on a device that cannot report \
                   its memory",
            backend: "engine",
        })?;
        self.kv_pool_pages = fit.pool_pages(info, cost, self.ctx)?;
        tracing::info!(
            fraction = fit.fraction(),
            free_gib = gib(info.free),
            total_gib = gib(info.total),
            page_bytes = cost.page_bytes,
            session_bytes = cost.session_bytes,
            fixed_bytes = cost.fixed_bytes,
            sessions = fit.sessions(),
            reserve_gib = gib(fit.reserve_bytes()),
            pool_pages = self.kv_pool_pages,
            pool_tokens = u64::from(self.kv_pool_pages) * u64::from(PAGE_TOKENS),
            "the KV pool is sized to the device"
        );
        Ok(())
    }
}

/// Gibibytes, for a log line.
#[expect(
    clippy::cast_precision_loss,
    reason = "a log line in gibibytes; a mantissa's worth of bytes is not the point"
)]
fn gib(bytes: u64) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

/// What one pool page and one session cost, as the stack that allocates them
/// counts it.
///
/// Deliberately NOT `#[non_exhaustive]`: each stack builds one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolCost {
    /// Device bytes one page of the pool occupies, every plane included.
    pub page_bytes: u64,
    /// Device bytes one session allocates when it opens: its recurrent state
    /// and its rollback records.
    pub session_bytes: u64,
    /// Bytes the stack allocates once, later, whatever the session count: the
    /// checkpoint slots' blobs, which on a device that shares memory with its
    /// host come out of the same memory as the pool.
    pub fixed_bytes: u64,
}

/// A KV pool sized to a share of the device's memory.
///
/// The share is of the device's *total* memory and covers everything on it,
/// this process's weights and scratch and every other tenant's allocations
/// alike: the pool gets what is left of `fraction * total` once all of that,
/// `sessions` sessions' own state and `reserve_bytes` are taken out. On a
/// device that shares memory with its host, "every other tenant" is the rest
/// of the machine.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KvFit {
    fraction: f64,
    sessions: u32,
    reserve_bytes: u64,
}

impl KvFit {
    /// A share of the device, the sessions that will be resident at once, and
    /// the bytes that will be allocated after the stack is built - a vision
    /// tower, the driver's own workspaces.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for a fraction outside `(0, 1]` or zero
    /// sessions: a share of nothing, or of more than the device, is a typo,
    /// and a pool sized for no sessions is sized for nothing.
    pub fn new(fraction: f64, sessions: u32, reserve_bytes: u64) -> CoreResult<Self> {
        if !(fraction > 0.0 && fraction <= 1.0) {
            return Err(CoreError::InvalidValue {
                key: "gpu_memory_utilization",
                value: format!("{fraction} is not a share in (0, 1]"),
            });
        }
        if sessions == 0 {
            return Err(CoreError::InvalidValue {
                key: "max_num_seqs",
                value: "0 sessions".to_owned(),
            });
        }
        Ok(Self {
            fraction,
            sessions,
            reserve_bytes,
        })
    }

    /// The share of the device.
    #[must_use]
    pub const fn fraction(&self) -> f64 {
        self.fraction
    }

    /// The sessions whose state is set aside.
    #[must_use]
    pub const fn sessions(&self) -> u32 {
        self.sessions
    }

    /// The bytes set aside for what is allocated later.
    #[must_use]
    pub const fn reserve_bytes(&self) -> u64 {
        self.reserve_bytes
    }

    /// Pages the pool gets on a device reporting `info`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] when what is left cannot hold one session
    /// at `ctx` tokens. A pool too small for one full context evicts the span
    /// the session is standing on, so this refuses to start rather than serve
    /// that; the message carries the numbers the operator needs to pick a
    /// larger share.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "a share of a byte count: the product is non-negative, at most the \
                  total, and a mantissa's worth of bytes is far below a page"
    )]
    pub fn pool_pages(&self, info: MemoryInfo, cost: PoolCost, ctx: u32) -> CoreResult<u32> {
        let share = (self.fraction * info.total as f64) as u64;
        let in_use = info.total.saturating_sub(info.free);
        let sessions = u64::from(self.sessions).saturating_mul(cost.session_bytes);
        let left = share
            .saturating_sub(in_use)
            .saturating_sub(sessions)
            .saturating_sub(cost.fixed_bytes)
            .saturating_sub(self.reserve_bytes);
        let pages = u32::try_from(left / cost.page_bytes.max(1)).unwrap_or(u32::MAX);
        let needed = pages_for(ctx);
        if pages < needed {
            return Err(CoreError::InvalidValue {
                key: "gpu_memory_utilization",
                value: format!(
                    "{} of {:.2} GiB leaves {:.2} GiB for the KV cache once {:.2} GiB in \
                     use, {:.2} GiB for {} sessions, {:.2} GiB of checkpoint slots and \
                     {:.2} GiB reserved are taken out: {pages} pages, and one {ctx}-token \
                     context needs {needed}",
                    self.fraction,
                    gib(info.total),
                    gib(left),
                    gib(in_use),
                    gib(sessions),
                    self.sessions,
                    gib(cost.fixed_bytes),
                    gib(self.reserve_bytes),
                ),
            });
        }
        Ok(pages)
    }
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GIB: u64 = 1 << 30;
    const COST: PoolCost = PoolCost {
        page_bytes: 16 << 20,
        session_bytes: GIB / 4,
        fixed_bytes: 0,
    };

    #[test]
    fn the_pool_gets_what_is_left_of_the_share_after_everything_else() {
        // 100 GiB device, 60 GiB already in use; 0.9 of it is 90 GiB, so 30
        // GiB are left, less four sessions' 1 GiB and a 1 GiB reserve.
        let fit = KvFit::new(0.9, 4, GIB).expect("a valid share");
        let info = MemoryInfo {
            free: 40 * GIB,
            total: 100 * GIB,
        };
        let pages = fit
            .pool_pages(info, COST, 4096)
            .expect("room for a context");
        assert_eq!(u64::from(pages), (28 * GIB) / COST.page_bytes);
    }

    #[test]
    fn a_share_that_leaves_no_room_for_one_context_is_refused_with_the_numbers() {
        let fit = KvFit::new(0.5, 4, GIB).expect("a valid share");
        let info = MemoryInfo {
            free: 40 * GIB,
            total: 100 * GIB,
        };
        let error = fit
            .pool_pages(info, COST, 4096)
            .expect_err("half the device is already in use");
        assert!(format!("{error}").contains("GiB"), "{error}");
    }

    #[test]
    fn a_share_outside_zero_to_one_or_no_sessions_is_refused() {
        assert!(KvFit::new(0.0, 4, 0).is_err());
        assert!(KvFit::new(1.5, 4, 0).is_err());
        assert!(KvFit::new(f64::NAN, 4, 0).is_err());
        assert!(KvFit::new(0.9, 0, 0).is_err());
        assert!(KvFit::new(1.0, 1, 0).is_ok());
    }
}
