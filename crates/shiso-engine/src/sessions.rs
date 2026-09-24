//! The state one conversation cannot be resumed without.
//!
//! A session is three things: a frontier into the shared cache, which says which
//! cache rows its prefix owns; the recurrent stack, which no cache row can
//! reconstruct; and the count of positions it has written, which is what turns a
//! chunk of tokens into absolute positions. All three move together, so they are
//! owned together: there is no way to advance one and forget another.
//!
//! # Why the recurrent state is per session and the cache is not
//!
//! Cache rows are shareable, which is the whole point of the pooled cache, but a
//! recurrent state belongs to exactly one path through it. Two sessions that share
//! a prefix still diverge inside their recurrent blocks, so the stack is allocated
//! per session and never handed to the cache. The checkpoint path is the one place
//! it leaves: the caller copies the stack out as bytes and back in on a resume.
//!
//! # Ordinals, not parallel vectors
//!
//! The recurrent states are held in the order the layer list names them, and a
//! layer slot carries the ordinal of the state it reads. There is no second vector
//! that has to stay in step with the layers, so an ordinal that names the wrong
//! state is a bug in one place rather than a disagreement between two.

use std::sync::Arc;

use crate::shape::HybridShape;
use shiso_core::backend::Backend;
use shiso_core::buffer::{DeviceBuffer, DeviceView, alloc};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_kvstore::Frontier;

use crate::blocks::StateTrail;
use crate::kv::{Room, SessionKv, conv_state_bytes, state_bytes};
use crate::layers::ResidualIndex;
use crate::mtp::{Rollback, SpecDepth};

/// Bytes cleared per pass when a state buffer is wiped.
const ZERO_RUN: usize = 1 << 16;

/// One recurrent layer's persistent state.
///
/// Both halves are f32 in the layout the kernels read: the convolution window is
/// `conv_kernel - 1` rows of channels, and the recurrent state is one transposed
/// `head_dim` by `head_dim` matrix per value head. The widths come from
/// [`crate::kv`], which is where the checkpoint layout is derived, so the bytes a
/// session holds and the bytes a checkpoint carries are the same bytes.
#[derive(Debug)]
pub(crate) struct RecurrentState {
    conv: DeviceBuffer,
    state: DeviceBuffer,
    /// Where a speculative pass leaves the state it stood on after each row.
    ///
    /// `None` on a session that was not opened for speculation, which is every
    /// session of every build that carries no draft head. The slabs are the
    /// expensive half of a speculative session - one state matrix per drafted
    /// row, per recurrent slot, which on this family is 2 MiB times the depth
    /// times twenty-four - so they are allocated by the depth the session was
    /// opened at and never by a default.
    trail: Option<RecurrentTrail>,
}

/// One recurrent slot's per-row records, for a pass a rejection may undo.
///
/// The two halves are one record between them: the convolution window and the
/// state matrix after the same row. They are allocated and restored together
/// for the reason they are read together, and a rewind that took one from row
/// two and the other from row three would stand the block on a position that
/// never existed.
#[derive(Debug)]
struct RecurrentTrail {
    /// `rows` convolution windows, in row order.
    conv: DeviceBuffer,
    /// `rows` state matrices, in row order.
    state: DeviceBuffer,
    /// Records the two slabs hold, which is the depth the session was opened
    /// at.
    rows: u32,
}

impl RecurrentState {
    /// Allocates one layer's state and clears it.
    ///
    /// A session that has seen no tokens has an empty window and a zero state, and
    /// neither half is a convention the kernels tolerate by accident: the
    /// convolution shifts its window under a running sum and the recurrence updates
    /// the state in place, so both read what is already there before they write it.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when an allocation or a clearing pass fails.
    pub(crate) fn new(
        backend: &Arc<dyn Backend>,
        shape: &HybridShape,
        spec: SpecDepth,
    ) -> CoreResult<Self> {
        let conv_bytes = conv_state_bytes(shape);
        let state_bytes = state_bytes(shape);
        let conv = alloc(backend, conv_bytes)?;
        let state = alloc(backend, state_bytes)?;
        // One record per DRAFTED row, not per verified one: row zero of a
        // verify pass is the token the caller already committed, and a rewind
        // never has to put the session back in front of it.
        let rows = spec.rollback_slots();
        let trail = if rows == 0 {
            None
        } else {
            let span = u64::from(rows);
            Some(RecurrentTrail {
                conv: alloc(backend, conv_bytes.saturating_mul(span))?,
                state: alloc(backend, state_bytes.saturating_mul(span))?,
                rows,
            })
        };
        let mut fresh = Self { conv, state, trail };
        fresh.clear(backend)?;
        Ok(fresh)
    }

    /// Where a pass over `rows` rows should leave its per-row records.
    ///
    /// `None` when this session keeps no trail, which is the answer a plain
    /// decode and every prefill get.
    pub(crate) fn trail(&self) -> Option<StateTrail<'_>> {
        self.trail.as_ref().map(|trail| StateTrail {
            conv: trail.conv.view(),
            state: trail.state.view(),
            rows: trail.rows,
        })
    }

    /// Copies the record taken after row `rollback.slot()` back over the live
    /// state.
    ///
    /// # Errors
    ///
    /// [`CoreError::Unsupported`] on a session that keeps no trail,
    /// [`CoreError::OutOfBounds`] when the slot is past what the trail holds,
    /// and whatever the device reports for the two copies.
    pub(crate) fn rewind_state(
        &self,
        backend: &Arc<dyn Backend>,
        rollback: Rollback,
    ) -> CoreResult<()> {
        let trail = self.trail.as_ref().ok_or(CoreError::Unsupported {
            what: "a speculative rollback on a session opened without a trail",
            backend: "engine",
        })?;
        let conv_bytes = self.conv.len();
        let state_bytes = self.state.len();
        backend.copy_d2d(
            self.conv.view(),
            trail
                .conv
                .view()
                .slice(rollback.offset(conv_bytes), conv_bytes)?,
        )?;
        backend.copy_d2d(
            self.state.view(),
            trail
                .state
                .view()
                .slice(rollback.offset(state_bytes), state_bytes)?,
        )
    }

    /// The convolution window.
    pub(crate) fn conv(&self) -> DeviceView<'_> {
        self.conv.view()
    }

    /// The recurrent state.
    pub(crate) fn state(&self) -> DeviceView<'_> {
        self.state.view()
    }

    /// Overwrites both halves with zero.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when a clearing pass fails.
    pub(crate) fn clear(&mut self, backend: &Arc<dyn Backend>) -> CoreResult<()> {
        clear_range(backend, &self.conv)?;
        clear_range(backend, &self.state)
    }
}

/// Where the last pass left the residual row this session can be drafted from.
///
/// Two numbers, and both of them are the pass's rather than the session's own
/// arithmetic: which of the two residual buffers that pass's sum landed in, and
/// which row of it is this session's last. A draft head starts from exactly
/// that row.
///
/// **It sits on the session and not on the stack, and that is the whole point
/// of the type.** It used to be two fields of `HybridStack`, written by
/// whichever pass ran last. A batched step runs several sessions at once, so
/// after one the stack-wide pair named the last row of whichever session
/// happened to be last in the batch - and a speculative step for session A
/// following a batch containing session B would draft from B's hidden state.
/// Nothing fails: the target verifies every guess, so a draft from the wrong
/// conversation is simply a guess the target rejects, and byte-identity
/// survives untouched while the acceptance rate quietly goes to nothing.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PassRow {
    /// Which of the two residual buffers the pass's sum landed in.
    residual: ResidualIndex,
    /// The row of it that is this session's last.
    row: u32,
}

impl PassRow {
    /// The buffer the pass's residual landed in.
    pub(crate) const fn residual(self) -> ResidualIndex {
        self.residual
    }

    /// The row of that buffer this session's last token wrote.
    pub(crate) const fn row(self) -> u32 {
        self.row
    }
}

/// One live session.
#[derive(Debug)]
pub(crate) struct Session {
    id: SessionId,
    kv: SessionKv,
    states: Vec<RecurrentState>,
    written: u32,
    /// The residual row the last pass left for this session, once one has run.
    ///
    /// `None` on a session no pass has touched, which is the only honest answer
    /// there: a draft head asked to start from "the last pass" of a session
    /// that has had none would otherwise read a row the allocator last wrote.
    /// `None` again after [`Session::adopt`], because a resume is a change of
    /// conversation and the row a pre-resume record names belongs to the one it
    /// replaced.
    last: Option<PassRow>,
    /// How deep this session may speculate.
    ///
    /// A field rather than a stack-wide setting because it is what sized this
    /// session's trails: a session opened for plain decode carries no slabs,
    /// and a request that asked one of those to speculate has to be refused
    /// rather than answered out of memory nothing allocated.
    spec: SpecDepth,
    /// Every token the session has been advanced by, oldest first.
    ///
    /// The cache commits a span by walking the token sequence from position zero,
    /// so a commit needs the whole path rather than the chunk that produced it.
    /// The engine hands tokens over a chunk at a time, which makes the session the
    /// only place that path can live.
    history: Vec<TokenId>,
    /// Whether any chunk this session read spliced media rows in. Such a path
    /// is never checkpointed: a placeholder run is one id repeated, so its
    /// tokens cannot tell two pictures apart.
    media: bool,
    /// The position a resume stood this session on, zero for one prefilled
    /// from nothing: what files a prompt's end as cold or as a continuation.
    resumed_at: u32,
}

impl Session {
    /// Opens `id` over `frontier` with one recurrent state per recurrent slot.
    ///
    /// # Errors
    ///
    /// Whatever [`RecurrentState::new`] or [`SessionKv::open`] reports.
    pub(crate) fn open(
        backend: &Arc<dyn Backend>,
        id: SessionId,
        frontier: Frontier,
        capacity: u32,
        shape: &HybridShape,
        recurrent: usize,
        spec: SpecDepth,
    ) -> CoreResult<Self> {
        let states = (0..recurrent)
            .map(|_| RecurrentState::new(backend, shape, spec))
            .collect::<CoreResult<Vec<_>>>()?;
        let kv = SessionKv::open(backend, frontier, capacity)?;
        Ok(Self {
            id,
            kv,
            states,
            written: 0,
            last: None,
            history: Vec::new(),
            spec,
            media: false,
            resumed_at: 0,
        })
    }

    /// How deep this session may speculate, which is what sized its trails.
    #[must_use]
    pub(crate) const fn spec_depth(&self) -> SpecDepth {
        self.spec
    }

    /// Winds the session back to `position`, discarding the tokens past it.
    ///
    /// The recurrent state is *not* touched here: it is restored from the
    /// trail, by the caller that holds the backend. This is the host half - the
    /// token path and the position count - and the two are separate because a
    /// caller that did one without the other is the failure this whole
    /// mechanism exists to prevent, so neither is allowed to look like the
    /// whole job.
    ///
    /// The cache rows past `position` stay claimed. They hold what the rejected
    /// rows wrote, nothing reads them, and the next pass writes over them
    /// before anything can - re-claiming them would be a page-table churn for
    /// rows this session is about to ask for again.
    pub(crate) fn rewind_to(&mut self, position: u32) {
        if position >= self.written {
            return;
        }
        self.history
            .truncate(usize::try_from(position).unwrap_or(usize::MAX));
        self.written = position;
    }

    /// Claims page-table entries up to `positions`, without moving the count
    /// of positions the session has written.
    ///
    /// [`Session::advance`] extends a session by a row count, because a chunk's
    /// rows are the ones after its frontier. A draft's are not: its chain runs
    /// ahead of the frontier and commits nothing, so what it can state is the
    /// position it reaches rather than how far past the frontier that is.
    ///
    /// # Errors
    ///
    /// Whatever [`SessionKv::ensure_positions`] reports.
    pub(crate) fn reserve_to(&mut self, positions: u32, room: Room<'_>) -> CoreResult<()> {
        self.kv.ensure_positions(positions, room).map(|_moved| ())
    }

    /// The `ordinal`-th recurrent slot's trail, when this session keeps one.
    ///
    /// `None` is not a refusal: a trail is only kept while speculation is on,
    /// and a slot with none is a slot no rejection has anything to put back.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when no slot carries that ordinal.
    pub(crate) fn gdn_trail(&self, ordinal: u32) -> CoreResult<Option<StateTrail<'_>>> {
        Ok(self.gdn(ordinal)?.trail())
    }

    /// Copies every recurrent slot's record for `rollback` back over its live
    /// state.
    ///
    /// # Errors
    ///
    /// Whatever [`RecurrentState::rewind_state`] reports, for the first slot
    /// that reports one.
    pub(crate) fn rewind_states(
        &self,
        backend: &Arc<dyn Backend>,
        rollback: Rollback,
    ) -> CoreResult<()> {
        for state in &self.states {
            state.rewind_state(backend, rollback)?;
        }
        Ok(())
    }

    /// Records where a pass left this session's last residual row.
    ///
    /// Called by the pass itself rather than by its caller, so that a caller
    /// cannot forget: the pass is the only party that knows both which buffer
    /// its sum landed in and which row of it is this session's.
    pub(crate) const fn record_pass(&mut self, residual: ResidualIndex, row: u32) {
        self.last = Some(PassRow { residual, row });
    }

    /// Where the last pass left this session's residual row, if one has run.
    pub(crate) const fn last_pass(&self) -> Option<PassRow> {
        self.last
    }

    /// The tokens the session has consumed, oldest first.
    ///
    /// [`shiso_kvstore::KvCache::commit`] walks the sequence from zero, so the
    /// caller that commits has to be able to name every position behind it.
    #[must_use]
    pub(crate) fn history(&self) -> &[TokenId] {
        &self.history
    }

    /// Appends the chunk the session was just advanced by.
    pub(crate) fn push_history(&mut self, tokens: &[TokenId]) {
        self.history.extend_from_slice(tokens);
    }

    /// Absolute position of the next token to write, which is also the length of
    /// the prefix the cache holds for this session.
    #[must_use]
    pub(crate) const fn written(&self) -> u32 {
        self.written
    }

    /// The session's page table and frontier.
    #[must_use]
    pub(crate) fn kv(&self) -> &SessionKv {
        &self.kv
    }

    /// The recurrent state of the `ordinal`-th recurrent slot.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when no slot carries that ordinal.
    pub(crate) fn gdn(&self, ordinal: u32) -> CoreResult<&RecurrentState> {
        let index = usize::try_from(ordinal).unwrap_or(usize::MAX);
        let count = u64::try_from(self.states.len()).unwrap_or(u64::MAX);
        self.states.get(index).ok_or(CoreError::OutOfBounds {
            offset: u64::from(ordinal),
            len: 1,
            buffer_bytes: count,
        })
    }

    /// Claims the rows the next `n_tok` tokens write, then moves the position
    /// count past them.
    ///
    /// The claim happens first and the count moves only if it succeeded, so a
    /// session that could not get its rows never believes it owns them.
    ///
    /// # Errors
    ///
    /// [`CoreError::ContextOverflow`] when the step reaches past `capacity`, and
    /// whatever [`SessionKv::ensure_positions`] reports.
    pub(crate) fn advance(
        &mut self,
        tokens: &[TokenId],
        capacity: u32,
        room: Room<'_>,
    ) -> CoreResult<()> {
        let n_tok = u32::try_from(tokens.len()).map_err(|_| CoreError::ContextOverflow {
            session: self.id,
            cap: capacity,
        })?;
        let next = self
            .written
            .checked_add(n_tok)
            .ok_or(CoreError::ContextOverflow {
                session: self.id,
                cap: capacity,
            })?;
        if next > capacity {
            return Err(CoreError::ContextOverflow {
                session: self.id,
                cap: capacity,
            });
        }
        self.kv.ensure_positions(next, room)?;
        self.written = next;
        Ok(())
    }

    /// Stands the session on a prefix somebody else computed.
    ///
    /// All three halves move together, which is the point of doing it here rather
    /// than field by field at the call site: the frontier says which cache rows the
    /// prefix owns, the path says which tokens they hold, and the count says where
    /// the next token goes. A session that adopted two of the three would write its
    /// next row over one it believes it already owns.
    ///
    /// The recurrent stack is *not* touched. It is restored by the caller, which is
    /// the only party holding the blob, and this method deliberately cannot be used
    /// without it: a session standing on a resumed prefix with its old recurrent
    /// state would fold two conversations together.
    ///
    /// # The pass record is dropped, and that is a fourth half
    ///
    /// [`Session::last`] is cleared here, because a resume is a change of
    /// conversation and the row it names is a row of the previous one. Nothing
    /// in the scratch is restored by a resume: the residual buffers hold
    /// whatever the last pass of whatever session left in them, and the row
    /// index a pre-resume [`PassRow`] carries is an index into a prefix this
    /// session no longer stands on. A draft head reading it would open its
    /// chain from a hidden state belonging to the conversation the resume
    /// replaced - fluently, at the right shape, through launches that all
    /// succeed, and with only the acceptance rate to show for it.
    ///
    /// The consequence for a caller is that **a resumed session re-stages
    /// rather than reusing**: the first speculative step after a resume has no
    /// pass to start from and is refused by name
    /// ([`CoreError::Unsupported`] from `HybridStack::last_pass`) until a
    /// forward pass has run on the resumed prefix. That is the safe direction
    /// and it is the whole reason the field is an `Option`: the alternative to
    /// refusing is drafting from a row nothing wrote.
    ///
    /// # Errors
    ///
    /// [`CoreError::ContextOverflow`] when the prefix is longer than `capacity`, and
    /// whatever [`SessionKv::adopt`] reports.
    pub(crate) fn adopt(
        &mut self,
        frontier: Frontier,
        tokens: Vec<TokenId>,
        capacity: u32,
    ) -> CoreResult<()> {
        let position = u32::try_from(tokens.len()).map_err(|_| CoreError::ContextOverflow {
            session: self.id,
            cap: capacity,
        })?;
        if position > capacity {
            return Err(CoreError::ContextOverflow {
                session: self.id,
                cap: capacity,
            });
        }
        self.kv.adopt(frontier)?;
        self.history = tokens;
        self.written = position;
        // After the two fallible steps, so a refused adopt leaves the session
        // exactly as it found it - including its pass record, which is still
        // the right one for the prefix it is still standing on.
        self.last = None;
        Ok(())
    }

    /// The `ordinal`-th recurrent slot's convolution window and state matrix, in the
    /// order a checkpoint blob lays them out.
    ///
    /// # Errors
    ///
    /// [`CoreError::OutOfBounds`] when no slot carries that ordinal.
    pub(crate) fn gdn_halves(&self, ordinal: u32) -> CoreResult<(DeviceView<'_>, DeviceView<'_>)> {
        let state = self.gdn(ordinal)?;
        Ok((state.conv(), state.state()))
    }
}

/// Clears `buffer` in [`ZERO_RUN`]-byte passes.
///
/// The run is a constant rather than the buffer's length because a session's
/// recurrent stack is tens of mebibytes, and a copy of that size is not something
/// to hand a backend as one request.
///
/// # Errors
///
/// [`CoreError::Backend`] when a pass fails to copy.
pub(crate) fn clear_range(backend: &Arc<dyn Backend>, buffer: &DeviceBuffer) -> CoreResult<()> {
    // A `static`, not a `const`: a 64 KiB `const` array is duplicated into every
    // function that names it, and this one runs inside a loop over buffers.
    static ZEROS: [u8; ZERO_RUN] = [0; ZERO_RUN];
    let run = u64::try_from(ZERO_RUN).unwrap_or(u64::MAX);
    let mut offset = 0;
    while offset < buffer.len() {
        let len = buffer.len().saturating_sub(offset).min(run);
        let width = usize::try_from(len).unwrap_or(ZERO_RUN);
        backend.copy_h2d(buffer.slice(offset, len)?, &ZEROS[..width])?;
        offset += len;
    }
    Ok(())
}

impl Session {
    /// Whether this session read media, which keeps it off disk.
    #[must_use]
    pub(crate) const fn carries_media(&self) -> bool {
        self.media
    }

    /// Records that a chunk spliced media rows in.
    pub(crate) fn mark_media(&mut self) {
        self.media = true;
    }

    /// The position a resume stood this session on.
    #[must_use]
    pub(crate) const fn resumed_at(&self) -> u32 {
        self.resumed_at
    }

    /// Records where a resume stood this session.
    pub(crate) fn set_resumed_at(&mut self, position: u32) {
        self.resumed_at = position;
    }

    /// The path and the frontier together, as a commit needs them.
    ///
    /// The cache walks the token sequence from position zero while it extends the
    /// frontier, so a commit needs both halves at the same moment. A `&self`
    /// accessor for one and a `&mut self` accessor for the other cannot be held
    /// together, and the alternative to splitting the borrow here is copying a
    /// context-length token vector on every committed step.
    pub(crate) fn commit_parts(&mut self) -> (&[TokenId], &mut SessionKv) {
        (&self.history, &mut self.kv)
    }

    /// How many recurrent slots this session carries state for.
    #[must_use]
    pub(crate) fn recurrent_slots(&self) -> usize {
        self.states.len()
    }
}
