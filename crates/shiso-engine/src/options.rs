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
use shiso_core::page::{KvElem, PAGE_TOKENS, pages_for};
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
    /// Refuse a pool given as a page count that the device cannot hold once
    /// everything else is resident. Ignored when [`EngineOptions::kv_fit`]
    /// sizes the pool, which fits by construction.
    ///
    /// `None`, the default, trusts the count.
    pub kv_check: Option<KvCheck>,
    /// The element the cache stores attention keys and values as.
    ///
    /// [`KvElem::F16`], the default, is the only one every stack supports. A
    /// stack that cannot store another refuses it when it is built rather than
    /// storing f16 under a different name: the pool it sized and the pages it
    /// files on disk would both be the wrong size.
    pub kv_elem: KvElem,
    /// Rows one prefill pass of a hybrid stack carries, which sizes every
    /// activation buffer of its scratch.
    ///
    /// [`crate::MAX_PREFILL_CHUNK`], the default, is the narrow width whose
    /// scratch every family can afford. A stack that streams its routed
    /// experts from disk reads every expert once per pass, so a wide pass is
    /// fewer reads of the same bytes; it costs scratch linear in the width.
    /// A whole multiple of [`crate::MAX_PREFILL_CHUNK`] up to
    /// [`crate::MAX_PREFILL_ROWS`], refused otherwise when the stack is built.
    ///
    /// A wider pass is not always the same arithmetic as the narrow one. A
    /// chunk wider than [`crate::MAX_PREFILL_CHUNK`] skips the banded
    /// matrix-library route of [`crate::projection::projection`], whose bands
    /// are sized for that width, and runs the tile instead; so on a backend
    /// with a matrix tier, for a weight that route takes, a prompt prefilled
    /// wide is not bit-identical to the same prompt prefilled 512 rows at a
    /// time. A checkpoint whose weights do not take that route answers the
    /// same at either width: one with 4-bit nonlinear and 8-bit block weights
    /// was measured bit-identical at 512 and 4096 rows on device.
    pub prefill_rows: u32,
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
            kv_check: None,
            kv_elem: KvElem::F16,
            prefill_rows: crate::MAX_PREFILL_CHUNK,
        }
    }

    /// Resolves [`EngineOptions::kv_fit`] into `kv_pool_pages`, when set, and
    /// otherwise holds `kv_pool_pages` to [`EngineOptions::kv_check`].
    ///
    /// Called by a stack once everything it allocates up front, other than
    /// the pool, is resident. The free figure is the backend's, lowered by its
    /// memory ledger to what is left once every tagged allocation is actually
    /// resident ([`shiso_core::MemLedger::committed`]): on a device that
    /// commits on first touch, the driver alone still counts untouched scratch
    /// as free.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] on a device that cannot report its memory
    /// when a fit was asked for, and whatever [`KvFit::pool_pages`] and
    /// [`KvCheck::check`] report.
    pub fn fit_pool(&mut self, backend: &dyn Backend, cost: PoolCost) -> CoreResult<()> {
        let committed = || {
            backend
                .memory_info()
                .map(|now| backend.ledger().map_or(now, |ledger| ledger.committed(now)))
        };
        let Some(fit) = self.kv_fit else {
            if let (Some(check), Some(info)) = (self.kv_check, committed()) {
                check.check(info, cost, self.kv_pool_pages)?;
            }
            return Ok(());
        };
        let info = committed().ok_or(CoreError::Unsupported {
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

/// A limit on a pool given as a page count: it, `sessions` sessions' own
/// state, the checkpoint slots and `reserve_bytes` must all fit in what the
/// device has free once the stack is resident.
///
/// The reason this is a refusal and not a warning is the device it was
/// written for: on one that commits memory on first touch, a pool that does
/// not fit still allocates, and the failure comes hours later when sessions
/// fill it - as the whole machine running out of memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvCheck {
    sessions: u32,
    reserve_bytes: u64,
}

impl KvCheck {
    /// The sessions that will be resident at once, and the bytes that will be
    /// allocated after the stack is built.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] for zero sessions.
    pub fn new(sessions: u32, reserve_bytes: u64) -> CoreResult<Self> {
        if sessions == 0 {
            return Err(CoreError::InvalidValue {
                key: "max_num_seqs",
                value: "0 sessions".to_owned(),
            });
        }
        Ok(Self {
            sessions,
            reserve_bytes,
        })
    }

    /// Refuses `pool_pages` when they do not fit on a device reporting `info`.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming the numbers and the largest pool
    /// that would fit.
    #[expect(
        clippy::cast_precision_loss,
        reason = "gibibytes in an error message; a mantissa's worth of bytes is not the point"
    )]
    pub fn check(&self, info: MemoryInfo, cost: PoolCost, pool_pages: u32) -> CoreResult<()> {
        let sessions = u64::from(self.sessions).saturating_mul(cost.session_bytes);
        let left = info
            .free
            .saturating_sub(sessions)
            .saturating_sub(cost.fixed_bytes)
            .saturating_sub(self.reserve_bytes);
        let pool = u64::from(pool_pages).saturating_mul(cost.page_bytes);
        let fits = left / cost.page_bytes.max(1);
        tracing::info!(
            free_gib = gib(info.free),
            pool_gib = gib(pool),
            sessions = self.sessions,
            sessions_gib = gib(sessions),
            fixed_gib = gib(cost.fixed_bytes),
            reserve_gib = gib(self.reserve_bytes),
            margin_gib = (left as f64 - pool as f64) / f64::from(1u32 << 30),
            "the KV pool against committed memory"
        );
        if pool > left {
            return Err(CoreError::InvalidValue {
                key: "kv_cache_tokens",
                value: format!(
                    "{} tokens ({:.2} GiB) do not fit: {:.2} GiB is free once everything \
                     allocated so far is resident, less {:.2} GiB for {} sessions, {:.2} GiB \
                     of checkpoint slots and {:.2} GiB reserved leaves {:.2} GiB, room for \
                     {} tokens; lower --kv-cache-tokens, --max-num-seqs, --kv-checkpoints or the \
                     prefill width",
                    u64::from(pool_pages) * u64::from(PAGE_TOKENS),
                    gib(pool),
                    gib(info.free),
                    gib(sessions),
                    self.sessions,
                    gib(cost.fixed_bytes),
                    gib(self.reserve_bytes),
                    gib(left),
                    fits.saturating_mul(u64::from(PAGE_TOKENS)),
                ),
            });
        }
        Ok(())
    }
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
                     context needs {needed}; raise the share or lower --max-num-seqs or \
                     --kv-checkpoints",
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
    fn a_counted_pool_is_held_to_what_is_free_after_sessions_and_reserve() {
        let check = KvCheck::new(4, GIB).expect("sessions");
        let info = MemoryInfo {
            free: 40 * GIB,
            total: 100 * GIB,
        };
        // 40 - 1 (sessions) - 1 (reserve) = 38 GiB: 2432 pages of 16 MiB.
        assert!(check.check(info, COST, 2432).is_ok());
        let error = check
            .check(info, COST, 2433)
            .expect_err("one page too many");
        assert!(format!("{error}").contains("room for"), "{error}");
        assert!(KvCheck::new(0, 0).is_err());
    }

    #[test]
    fn checkpoint_slots_come_out_of_the_pool_that_fits_and_the_refusal_names_them() {
        // The counted pool above, with 2 GiB of checkpoint slots on top: 36
        // GiB are left, so the 2432 pages that fitted before are now too many.
        let check = KvCheck::new(4, GIB).expect("sessions");
        let info = MemoryInfo {
            free: 40 * GIB,
            total: 100 * GIB,
        };
        let cost = PoolCost {
            fixed_bytes: 2 * GIB,
            ..COST
        };
        assert!(check.check(info, cost, 2304).is_ok());
        let error = check
            .check(info, cost, 2305)
            .expect_err("the slots took two gibibytes of it");
        assert!(format!("{error}").contains("--kv-checkpoints"), "{error}");

        let fit = KvFit::new(0.9, 4, GIB).expect("a valid share");
        let pages = fit
            .pool_pages(info, cost, 4096)
            .expect("room for a context");
        assert_eq!(u64::from(pages), (26 * GIB) / COST.page_bytes);
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
