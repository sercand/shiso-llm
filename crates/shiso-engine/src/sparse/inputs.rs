//! The three rows a pass reads before any weight does: the token ids, the
//! positions, and the rotary table.
//!
//! Everything else a launch in this module touches is a weight the load put on
//! the device, a cache page the session owns, or an activation some earlier
//! launch in the same pass wrote. These three are the pass's only *arguments*,
//! and they come from the host: the ids the row gather indexes the vocabulary
//! with, the absolute positions every cache write and every rotation is
//! addressed by, and the per-axis rotary table the attention and indexer blocks
//! rotate against.
//!
//! # Why an unwritten input here is the dangerous kind
//!
//! Nothing fails when they are missing. A zeroed id row gathers vocabulary row
//! zero for every token; a zeroed position row writes every token of every
//! chunk to cache row zero and rotates every query by nothing; a zeroed rotary
//! table agrees with it. Every launch succeeds, every number is finite, and what
//! comes out reads as an answer. So the three are written from one place, and
//! the rows themselves are built by pure functions a test can read back without
//! a device.
//!
//! # The rotary table's prefix, and why it is part of the table
//!
//! The table is `MROPE_BACK + n_tok` rows of three interleaved `int32` (time,
//! height, width), one row per position, in that order. Every rotation but one
//! reads it starting at the chunk's own first row, which is
//! [`MROPE_PREFIX_BYTES`] into the buffer. The exception is the pooled-key pass:
//! it derives a pooled row's position by rounding a position down to the
//! compression boundary, so it can name a position a few slots before the chunk
//! starts, and it therefore reads the table whole and indexes it relative to the
//! prefix.
//!
//! Both bindings exist in the plan, and getting them the wrong way round does
//! not fail either: the whole table bound where a chunk-relative one belongs
//! rotates every row by the lookback, and there is no launch that would notice.
//! One writer fills both regions and [`MROPE_PREFIX_BYTES`] is the single place
//! the boundary between them is stated, so the readers and the writer cannot
//! drift apart.
//!
//! # Three axes, one position
//!
//! A text token's three axes are the same number, which makes the multi-axis
//! rotation identical to the single-axis one. A prompt carrying media gives a
//! placeholder token three different numbers - which frame, which row, which
//! column - and the walk that produces them is [`crate::vision::Walk`], which
//! is family-agnostic. [`InputRows::fill_media`] is the media branch that
//! predecessor documents said would replace the body of [`InputRows::fill`]:
//! it turned out to be a sibling rather than a replacement, because the two
//! differ in what they build and agree on everything below.
//!
//! # The fourth row a media chunk writes, and the prefix it cannot walk
//!
//! A media chunk writes one more row than a text chunk: the splice map, one
//! `int32` per token naming the encoder row that token's embedding is replaced
//! by, or a negative number for a token that keeps its own. It comes out of the
//! same walk as the three axis streams, so the positions and the row the splice
//! copies cannot disagree about where a placeholder run began.
//!
//! The rotary table's prefix is the one thing the walk cannot produce, because
//! the prefix names positions BEFORE the chunk and the walk has not been there.
//! Within one request that is answered by carrying the previous chunk's last
//! [`shiso_core::page::MROPE_BACK`] rows forward - a prompt split across passes
//! sees the rows it would have seen unsplit. At the head of a request the text
//! rule is used, stepping back from the chunk's first position on all three
//! axes, which is exactly right whenever the tokens before the chunk were text
//! and is what the dense family's walk does with the same information. The only
//! reader of those rows is the pooled-key pass, which rounds a position down to
//! the compression boundary and therefore never reaches below zero at the head
//! of a sequence.

use shiso_core::backend::Backend;
use shiso_core::buffer::DeviceView;
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::ids::{SessionId, TokenId};
use shiso_core::model::DecodeStep;
use shiso_core::page::MROPE_BACK;

use crate::sparse::kernels::I32_BYTES;
use crate::sparse::scratch::Buf;
use crate::sparse::stack::{MAX_PREFILL_CHUNK, SparseStack};
use crate::vision::{Streams, Walk};

/// Position axes one rotary table row carries: time, height and width, in that
/// order.
pub const MROPE_AXES: u32 = 3;

/// Bytes of rotary table in front of the chunk's own first row.
///
/// The one statement of where the prefix ends. The writer fills the rows below
/// it, every rotation that reads the chunk's own rows is bound past it, and the
/// pooled-key pass is bound at zero and reaches back into it.
pub const MROPE_PREFIX_BYTES: u64 = MROPE_BACK as u64 * MROPE_AXES as u64 * I32_BYTES;

/// Bytes one rotary table row occupies.
pub const MROPE_ROW_BYTES: u64 = MROPE_AXES as u64 * I32_BYTES;

/// Rows one batched row's lookback group holds: the lookback and the row's own
/// position.
pub const MROPE_GROUP_ROWS: u32 = MROPE_BACK + 1;

/// Byte offset of row `row`'s lookback group inside the rotary table of a
/// batched pass.
///
/// # Two regions, because two readers index the table differently
///
/// Every rotation in a pass but one reads the table at `3 * t` for row `t`, so
/// a batch's own position rows have to sit one row apart, in the same place a
/// chunk's do: [`MROPE_PREFIX_BYTES`] in. The pooled-key pass is the exception.
/// It derives a block's position arithmetically and indexes the table by
/// `MROPE_BACK + block_pos - pos[0]`, which reaches up to a compression ratio
/// *behind* the row it is launched for - and in a batch the row behind row one
/// is not row zero, it is a position in row one's own sequence. So the batch
/// writes a second region past the first: one group of [`MROPE_GROUP_ROWS`] per
/// row, that row's lookback and then that row's own position, and the pooled
/// pass for row `r` is bound at group `r`.
///
/// A single shared prefix would not do and would not fail either: row one's
/// pooled key would be rotated at a position out of row zero's sequence, and
/// only past the selection budget where the indexer is consulted at all.
#[must_use]
pub const fn batch_lookback_offset(rows: u32, row: u32) -> u64 {
    let ahead = MROPE_BACK
        .saturating_add(rows)
        .saturating_add(row.saturating_mul(MROPE_GROUP_ROWS));
    (ahead as u64).saturating_mul(MROPE_ROW_BYTES)
}

/// Rows the rotary table of a batched pass of `rows` rows occupies.
#[must_use]
pub const fn batch_mrope_rows(rows: u32) -> u32 {
    MROPE_BACK
        .saturating_add(rows)
        .saturating_add(rows.saturating_mul(MROPE_GROUP_ROWS))
}

/// One row of a decode batch, as the inputs see it.
///
/// A named pair rather than two parallel slices: a batch whose tokens and
/// positions came from two lists could be one entry out of step and say so
/// nowhere, and every row of the pass would then decode at its neighbour's
/// position.
///
/// Deliberately NOT `#[non_exhaustive]`: a caller builds one per row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DecodeRow {
    /// The token this row decodes from.
    pub token: TokenId,
    /// The absolute position it occupies in its own sequence, which is that
    /// session's own frontier.
    pub position: u32,
}

/// One token as the vocabulary row the gather indexes with.
///
/// Checked here because it cannot be checked anywhere later: two of the three
/// gather entries clamp an id outside the table to row zero, which is an answer
/// nobody can attribute to a token, and the block-quantized entry does not clamp
/// at all, which is a read past the table.
fn vocabulary_row(token: TokenId, vocab: u32) -> CoreResult<i32> {
    let id = token.get();
    i32::try_from(token)
        .ok()
        .filter(|_| id < vocab)
        .ok_or_else(|| CoreError::InvalidValue {
            key: "token id",
            value: format!("{id} is not a row of a {vocab}-entry vocabulary table"),
        })
}

/// A position as the `int32` the kernels read, saturating rather than wrapping.
///
/// A position this large is a context far past anything the cache can hold, so
/// the clamp is unreachable rather than lossy - but a wrap here would be a
/// negative position, and a negative position is a cache row nobody owns.
fn position(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

/// The three host rows a pass's inputs are staged in.
///
/// One value rather than three, because they are always built together from one
/// chunk: a pass whose id row and position row came from different calls would
/// describe two different chunks and say so nowhere. Kept across passes so the
/// allocations happen once rather than per chunk.
///
/// Deliberately NOT `#[non_exhaustive]`: it has no public fields to add to, and
/// a stack builds one.
#[derive(Debug, Default)]
pub struct InputRows {
    ids: Vec<i32>,
    positions: Vec<i32>,
    mrope: Vec<i32>,
    /// The encoder row each token takes, written only by a media chunk.
    splice: Vec<i32>,
    /// The three axis streams and the splice map of the chunk in flight, as the
    /// walk hands them over. Empty, and never read, for a text chunk.
    streams: Streams,
    /// The little-endian bytes one row is uploaded as, reused by all three.
    bytes: Vec<u8>,
}

impl InputRows {
    /// Empty rows.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ids: Vec::new(),
            positions: Vec::new(),
            mrope: Vec::new(),
            splice: Vec::new(),
            streams: Streams::new(),
            bytes: Vec::new(),
        }
    }

    /// Builds all three rows for `tokens`, whose first token sits at absolute
    /// position `pos0`.
    ///
    /// `vocab` is the height of the table the gather will index. It is checked
    /// here because it cannot be checked anywhere later: two of the three gather
    /// entries clamp an id outside the table to row zero, which is an answer
    /// nobody can attribute to a token, and the block-quantized entry does not
    /// clamp at all, which is a read past the table. This is the last place that
    /// sees the ids as ids.
    ///
    /// # Errors
    ///
    /// [`CoreError::InvalidValue`] naming the first token id that is not a row
    /// of a `vocab`-entry table, and [`CoreError::ShapeMismatch`] for a chunk
    /// with more tokens than a position can count.
    pub fn fill(&mut self, tokens: &[TokenId], pos0: u32, vocab: u32) -> CoreResult<()> {
        let n_tok = u32::try_from(tokens.len()).map_err(|_| CoreError::ShapeMismatch {
            field: "chunk rows",
            expected: u64::from(u32::MAX),
            got: u64::try_from(tokens.len()).unwrap_or(u64::MAX),
        })?;

        self.ids.clear();
        for token in tokens {
            self.ids.push(vocabulary_row(*token, vocab)?);
        }

        self.positions.clear();
        self.positions
            .extend((0..n_tok).map(|i| position(pos0.saturating_add(i))));

        // The lookback rows first, then the chunk's own. A prefix row names a
        // position before the chunk started; at the head of a sequence there is
        // no such position, and the rows that would hold one are also the rows
        // the pooled-key pass never reaches - its first block starts at
        // `pos0 - pos0 % compress`, which is never below zero - so clamping
        // them at zero fills a region rather than inventing a position.
        self.mrope.clear();
        for back in (1..=MROPE_BACK).rev() {
            push_row(&mut self.mrope, position(pos0.saturating_sub(back)));
        }
        for i in 0..n_tok {
            push_row(&mut self.mrope, position(pos0.saturating_add(i)));
        }
        // A text chunk splices nothing, and a stale map left from an earlier
        // media chunk of the same stack would name encoder rows for tokens that
        // are not placeholders - which the splice would copy and nothing would
        // report.
        self.splice.clear();
        Ok(())
    }

    /// Copies the three rows this holds onto the device.
    ///
    /// Shared by the chunk path and the draft head's, because what differs
    /// between them is how the rows were *built* and not where they go.
    ///
    /// # Errors
    ///
    /// [`CoreError::Backend`] when a transfer fails, and
    /// [`CoreError::OutOfBounds`] when a row is wider than the buffer it is
    /// written into.
    pub fn upload_to(
        &mut self,
        backend: &dyn Backend,
        scratch: &crate::sparse::scratch::Scratch,
    ) -> CoreResult<()> {
        upload(backend, &mut self.bytes, scratch.view(Buf::Ids)?, &self.ids)?;
        upload(
            backend,
            &mut self.bytes,
            scratch.view(Buf::Positions)?,
            &self.positions,
        )?;
        // From zero, because the prefix rows are part of the table and the one
        // launch that reads them indexes from its front.
        upload(
            backend,
            &mut self.bytes,
            scratch.view(Buf::Mrope)?,
            &self.mrope,
        )
    }

    /// Builds the four rows a media-carrying chunk reads, walking `tokens`
    /// through `walk`.
    ///
    /// The id and position rows are a text chunk's: an id is still the
    /// vocabulary row the gather indexes with - placeholders included, because
    /// the splice overwrites their embeddings rather than skipping their
    /// gather - and a position is still the absolute sequence position the
    /// cache is addressed by. What changes is the rotary table, whose three
    /// axes now differ at a placeholder token, and the splice map, which is new.
    ///
    /// `prefix` is the previous chunk's last [`MROPE_BACK`] rotary rows, or
    /// `None` at the head of a request, where the text rule is used. The module
    /// doc says why the walk cannot produce them.
    ///
    /// # Errors
    ///
    /// As [`InputRows::fill`], plus [`CoreError::ShapeMismatch`] when `prefix`
    /// is the wrong length - which would slide the whole table by a row and
    /// rotate every token of the chunk at its neighbour's position - or when
    /// the walk returned a different number of rows than the chunk has tokens.
    ///
    /// Crate-visible because [`Walk`] is: the walk is the engine's own, and a
    /// caller outside it reaches the media path through the model contract.
    pub(crate) fn fill_media(
        &mut self,
        tokens: &[TokenId],
        pos0: u32,
        vocab: u32,
        walk: &mut Walk<'_>,
        prefix: Option<&[i32]>,
    ) -> CoreResult<()> {
        let n_tok = u32::try_from(tokens.len()).map_err(|_| CoreError::ShapeMismatch {
            field: "chunk rows",
            expected: u64::from(u32::MAX),
            got: u64::try_from(tokens.len()).unwrap_or(u64::MAX),
        })?;
        let prefix_len = usize::try_from(MROPE_PREFIX_BYTES / I32_BYTES).unwrap_or(usize::MAX);
        if let Some(rows) = prefix
            && rows.len() != prefix_len
        {
            return Err(CoreError::ShapeMismatch {
                field: "carried rotary prefix",
                expected: u64::try_from(prefix_len).unwrap_or(u64::MAX),
                got: u64::try_from(rows.len()).unwrap_or(u64::MAX),
            });
        }

        self.ids.clear();
        for token in tokens {
            self.ids.push(vocabulary_row(*token, vocab)?);
        }

        self.positions.clear();
        self.positions
            .extend((0..n_tok).map(|i| position(pos0.saturating_add(i))));

        self.streams.clear();
        walk.chunk(tokens, &mut self.streams);
        let walked = self.streams.t.len();
        if walked != tokens.len()
            || self.streams.h.len() != tokens.len()
            || self.streams.w.len() != tokens.len()
            || self.streams.map.len() != tokens.len()
        {
            return Err(CoreError::ShapeMismatch {
                field: "walked media rows",
                expected: u64::try_from(tokens.len()).unwrap_or(u64::MAX),
                got: u64::try_from(walked).unwrap_or(u64::MAX),
            });
        }

        self.mrope.clear();
        match prefix {
            Some(rows) => self.mrope.extend_from_slice(rows),
            // The text rule, which is what the tokens before a request's first
            // chunk were unless a previous request ended inside a picture. A
            // prompt never ends inside a placeholder run - the run is closed by
            // its own end marker - so the rows this names are text tokens'.
            None => {
                for back in (1..=MROPE_BACK).rev() {
                    push_row(&mut self.mrope, position(pos0.saturating_sub(back)));
                }
            }
        }
        // Interleaved, one row of three, because that is what this family's
        // rotation indexes. The walk hands over three planar streams, which is
        // what the dense family's rotation reads; the transposition is here
        // rather than in the walk so the walk stays the one both families use.
        for i in 0..walked {
            self.mrope.extend_from_slice(&[
                self.streams.t[i],
                self.streams.h[i],
                self.streams.w[i],
            ]);
        }

        self.splice.clear();
        self.splice.extend_from_slice(&self.streams.map);
        Ok(())
    }

    /// The last [`MROPE_BACK`] rotary rows of the chunk just filled, which the
    /// next chunk of the same request carries forward as its prefix.
    ///
    /// Empty for a chunk shorter than the prefix, which cannot happen for a
    /// media chunk of a real prompt and would otherwise hand the next chunk a
    /// short slice that [`InputRows::fill_media`] refuses by length.
    #[must_use]
    pub fn mrope_tail(&self) -> &[i32] {
        let want = usize::try_from(MROPE_PREFIX_BYTES / I32_BYTES).unwrap_or(usize::MAX);
        self.mrope
            .len()
            .checked_sub(want)
            .map_or(&[], |at| &self.mrope[at..])
    }

    /// The encoder row each token of the chunk takes, negative for a token that
    /// keeps the embedding the gather gave it.
    ///
    /// Empty after [`InputRows::fill`], because a text chunk splices nothing.
    #[must_use]
    pub fn splice_map(&self) -> &[i32] {
        &self.splice
    }

    /// The same three rows at positions the caller names.
    ///
    /// A chunk's positions are its own frontier and the ones after it, which is
    /// why [`InputRows::fill`] derives them. A draft head's are not: its chain
    /// runs *ahead* of the frontier, and the pass that catches it up after a
    /// verify starts from inside the accepted prefix. So the positions travel
    /// with the tokens, and the rotary prefix is taken from the first of them
    /// rather than from any session's state.
    ///
    /// # Errors
    ///
    /// As [`InputRows::fill`], plus [`CoreError::ShapeMismatch`] when the two
    /// slices are different lengths - which would rope a token at another
    /// token's position, with every launch succeeding.
    pub fn fill_at(&mut self, tokens: &[TokenId], positions: &[u32], vocab: u32) -> CoreResult<()> {
        if tokens.len() != positions.len() {
            return Err(CoreError::ShapeMismatch {
                field: "draft tokens and positions",
                expected: u64::try_from(tokens.len()).unwrap_or(u64::MAX),
                got: u64::try_from(positions.len()).unwrap_or(u64::MAX),
            });
        }
        self.ids.clear();
        for token in tokens {
            self.ids.push(vocabulary_row(*token, vocab)?);
        }
        self.positions.clear();
        self.positions
            .extend(positions.iter().copied().map(position));

        let first = positions.first().copied().unwrap_or(0);
        self.mrope.clear();
        for back in (1..=MROPE_BACK).rev() {
            push_row(&mut self.mrope, position(first.saturating_sub(back)));
        }
        for at in positions {
            push_row(&mut self.mrope, position(*at));
        }
        self.splice.clear();
        Ok(())
    }

    /// Builds all three rows for a decode batch: one row per sequence, each at
    /// its own absolute position.
    ///
    /// The id and position rows are one entry per batch row, exactly as a
    /// chunk's are one entry per token - what changes is that the positions are
    /// unrelated rather than consecutive. The rotary table gains a second
    /// region, and [`batch_lookback_offset`] says why.
    ///
    /// # Errors
    ///
    /// As [`InputRows::fill`], plus [`CoreError::ShapeMismatch`] for an empty
    /// batch: a pass over no rows writes no inputs, and issuing one would run
    /// every launch on whatever the last pass left behind.
    pub fn fill_batch(&mut self, rows: &[DecodeRow], vocab: u32) -> CoreResult<()> {
        let count = u32::try_from(rows.len()).unwrap_or(u32::MAX);
        if count == 0 {
            return Err(CoreError::ShapeMismatch {
                field: "decode batch rows",
                expected: 1,
                got: 0,
            });
        }

        self.ids.clear();
        for row in rows {
            self.ids.push(vocabulary_row(row.token, vocab)?);
        }

        self.positions.clear();
        self.positions
            .extend(rows.iter().map(|row| position(row.position)));

        // Region one: the lookback in front, then one row per batch row, so
        // every rotation bound past the prefix reads row `t` at `3 * t` exactly
        // as it does for a chunk. The prefix itself is the first row's lookback
        // - no launch of a batched pass reads it, and filling it is cheaper than
        // leaving a hole a later reader would have to be trusted not to touch.
        self.mrope.clear();
        let first = rows[0].position;
        for back in (1..=MROPE_BACK).rev() {
            push_row(&mut self.mrope, position(first.saturating_sub(back)));
        }
        for row in rows {
            push_row(&mut self.mrope, position(row.position));
        }
        // Region two: one group per row, that row's own lookback and then its
        // own position, which is what the pooled-key pass for that row is bound
        // at.
        for row in rows {
            for back in (1..=MROPE_BACK).rev() {
                push_row(&mut self.mrope, position(row.position.saturating_sub(back)));
            }
            push_row(&mut self.mrope, position(row.position));
        }
        self.splice.clear();
        Ok(())
    }

    /// The vocabulary row each token of the chunk gathers, in chunk order.
    #[must_use]
    pub fn ids(&self) -> &[i32] {
        &self.ids
    }

    /// The absolute position of each token of the chunk, in chunk order.
    #[must_use]
    pub fn positions(&self) -> &[i32] {
        &self.positions
    }

    /// The rotary table, prefix rows included: `MROPE_BACK + n_tok` rows of
    /// [`MROPE_AXES`] positions each.
    #[must_use]
    pub fn mrope(&self) -> &[i32] {
        &self.mrope
    }
}

/// The table is written a row at a time and a row is spelled out below, so a
/// change to the axis count has to be made in both places or not at all.
const _: () = assert!(
    MROPE_AXES == 3,
    "a rotary table row is time, height and width"
);

/// Appends one rotary table row: the same position on all three axes, which is
/// what a token that is not part of a picture has.
///
/// The three are written out rather than repeated `MROPE_AXES` times because
/// they are three different axes that happen to agree here, and the media walk
/// that makes them disagree replaces this line and nothing above it.
fn push_row(out: &mut Vec<i32>, pos: i32) {
    out.extend_from_slice(&[pos, pos, pos]);
}

/// Copies one `int32` row onto the front of a device buffer.
///
/// The dense family has a private function of the same shape, and the two are
/// deliberately not shared. What each family *builds* is the part that matters
/// and the two differ completely: three planar position planes plus a splice map
/// there, one interleaved table with a prefix here. What is left over once those
/// differ is the little-endian encode below, and hoisting three lines into a
/// third module would put them an indirection away from both row builders while
/// leaving the halves that can actually be got wrong exactly where they are.
fn upload(
    backend: &dyn Backend,
    bytes: &mut Vec<u8>,
    dst: DeviceView<'_>,
    row: &[i32],
) -> CoreResult<()> {
    bytes.clear();
    for value in row {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let len = u64::try_from(bytes.len()).unwrap_or(u64::MAX);
    if len > dst.len() {
        return Err(CoreError::OutOfBounds {
            offset: 0,
            len,
            buffer_bytes: dst.len(),
        });
    }
    backend.copy_h2d(dst.slice(0, len)?, bytes)
}

impl SparseStack {
    /// Writes the three inputs one pass reads, for one session's next chunk.
    ///
    /// This runs after [`SparseStack::reserve`] and before
    /// [`SparseStack::execute`], and before [`SparseStack::advance`]: the
    /// positions are the ones the chunk is *about to* occupy, so a caller that
    /// recorded the tokens first would write the chunk after next.
    ///
    /// The chunk's first position is the session's own frontier rather than an
    /// argument. A caller able to name the position could name one whose cache
    /// rows were never claimed, and the launches would write them anyway.
    ///
    /// # Errors
    ///
    /// [`CoreError::ShapeMismatch`] for an empty or over-wide chunk,
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] when the
    /// stack's device state has already been released, whatever
    /// [`InputRows::fill`] reports for a token id this vocabulary has no row
    /// for, and [`CoreError::Backend`] when a transfer fails.
    pub fn write_inputs(&mut self, session: SessionId, tokens: &[TokenId]) -> CoreResult<()> {
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if n_tok == 0 || n_tok > MAX_PREFILL_CHUNK {
            return Err(CoreError::ShapeMismatch {
                field: "chunk rows",
                expected: u64::from(MAX_PREFILL_CHUNK),
                got: u64::from(n_tok),
            });
        }
        let vocab = self.shape.vocab;
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let live = sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let rows = &mut staging.inputs;
        rows.fill(tokens, live.written(), vocab)?;

        let scratch = &resident.scratch;
        let device = backend.as_ref();
        upload(device, &mut rows.bytes, scratch.view(Buf::Ids)?, &rows.ids)?;
        upload(
            device,
            &mut rows.bytes,
            scratch.view(Buf::Positions)?,
            &rows.positions,
        )?;
        // From zero, because the prefix rows are part of the table and the one
        // launch that reads them indexes from its front.
        upload(
            device,
            &mut rows.bytes,
            scratch.view(Buf::Mrope)?,
            &rows.mrope,
        )
    }

    /// Writes the four inputs a chunk carrying a picture reads.
    ///
    /// The same three as [`SparseStack::write_inputs`] plus the splice map,
    /// and the rotary table is the walk's rather than the position row
    /// repeated. The run is advanced: the walk consumes this chunk's tokens and
    /// the prefix it carries becomes this chunk's tail, so a caller that wrote
    /// the same chunk twice would walk it twice.
    ///
    /// # Errors
    ///
    /// As [`SparseStack::write_inputs`], plus whatever
    /// [`InputRows::fill_media`] reports.
    pub(crate) fn write_inputs_media(
        &mut self,
        session: SessionId,
        tokens: &[TokenId],
        run: &mut crate::sparse::media::MediaRun<'_>,
    ) -> CoreResult<()> {
        let n_tok = u32::try_from(tokens.len()).unwrap_or(u32::MAX);
        if n_tok == 0 || n_tok > MAX_PREFILL_CHUNK {
            return Err(CoreError::ShapeMismatch {
                field: "chunk rows",
                expected: u64::from(MAX_PREFILL_CHUNK),
                got: u64::from(n_tok),
            });
        }
        let vocab = self.shape.vocab;
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let live = sessions
            .get(&session)
            .ok_or(CoreError::UnknownSession(session))?;
        let rows = &mut staging.inputs;
        // Split rather than borrowed whole: the walk is advanced while the
        // prefix is read, and the prefix is replaced once both are done.
        let crate::sparse::media::MediaRun { walk, prefix, .. } = run;
        let carried = (!prefix.is_empty()).then_some(prefix.as_slice());
        rows.fill_media(tokens, live.written(), vocab, walk, carried)?;
        prefix.clear();
        prefix.extend_from_slice(rows.mrope_tail());

        let scratch = &resident.scratch;
        let device = backend.as_ref();
        upload(device, &mut rows.bytes, scratch.view(Buf::Ids)?, &rows.ids)?;
        upload(
            device,
            &mut rows.bytes,
            scratch.view(Buf::Positions)?,
            &rows.positions,
        )?;
        // From zero, because the prefix rows are part of the table and the one
        // launch that reads them indexes from its front.
        upload(
            device,
            &mut rows.bytes,
            scratch.view(Buf::Mrope)?,
            &rows.mrope,
        )?;
        upload(
            device,
            &mut rows.bytes,
            scratch.view(Buf::SpliceMap)?,
            &rows.splice,
        )
    }

    /// Writes the three inputs a decode batch reads, one row per session.
    ///
    /// Each row's position is its own session's frontier, read here rather than
    /// taken as an argument, for the reason a chunk's first position is: a
    /// caller able to name it could name a position whose cache rows were never
    /// claimed, and the launches would write them anyway.
    ///
    /// # Errors
    ///
    /// [`CoreError::UnknownSession`], [`CoreError::Unsupported`] when the
    /// stack's device state has already been released, whatever
    /// [`InputRows::fill_batch`] reports for an empty batch or a token id this
    /// vocabulary has no row for, and [`CoreError::OutOfBounds`] when the rotary
    /// table's two regions do not fit the buffer - which is a batch wider than
    /// the scratch was sized for.
    pub fn write_batch_inputs(&mut self, steps: &[DecodeStep]) -> CoreResult<()> {
        let vocab = self.shape.vocab;
        let Self {
            backend,
            resident,
            sessions,
            staging,
            ..
        } = self;
        let resident = resident.as_ref().ok_or(CoreError::Unsupported {
            what: "a model whose device state teardown already released",
            backend: "engine",
        })?;
        let batch = steps
            .iter()
            .map(|step| {
                let live = sessions
                    .get(&step.session)
                    .ok_or(CoreError::UnknownSession(step.session))?;
                Ok(DecodeRow {
                    token: step.token,
                    position: live.written(),
                })
            })
            .collect::<CoreResult<Vec<_>>>()?;
        let rows = &mut staging.inputs;
        rows.fill_batch(&batch, vocab)?;

        let scratch = &resident.scratch;
        let device = backend.as_ref();
        upload(device, &mut rows.bytes, scratch.view(Buf::Ids)?, &rows.ids)?;
        upload(
            device,
            &mut rows.bytes,
            scratch.view(Buf::Positions)?,
            &rows.positions,
        )?;
        upload(
            device,
            &mut rows.bytes,
            scratch.view(Buf::Mrope)?,
            &rows.mrope,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shape::MediaPositions;
    use shiso_core::media::{PadTokens, VisionGrid};

    /// The chunk every test below writes, as ids inside a small vocabulary.
    fn chunk() -> Vec<TokenId> {
        [7_u32, 3, 3, 9]
            .into_iter()
            .map(TokenId::new)
            .collect::<Vec<_>>()
    }

    /// Positions one rotary table row carries.
    fn axes() -> usize {
        usize::try_from(MROPE_AXES).expect("a row is three positions wide")
    }

    /// Where the chunk's own first row starts, as an index into the table.
    fn chunk_row() -> usize {
        usize::try_from(MROPE_PREFIX_BYTES / I32_BYTES).expect("the prefix is a handful of rows")
    }

    #[test]
    fn the_id_row_is_the_tokens_the_caller_passed_in_the_order_they_were_passed() {
        let mut rows = InputRows::new();
        rows.fill(&chunk(), 0, 64).expect("every id is a row");
        assert_eq!(rows.ids(), [7, 3, 3, 9]);
    }

    #[test]
    fn a_token_id_the_table_has_no_row_for_is_refused_rather_than_gathered() {
        // Two of the three gather entries would clamp this to row zero and one
        // would read past the table. Neither is an answer about this token.
        let mut rows = InputRows::new();
        let err = rows
            .fill(&[TokenId::new(64)], 0, 64)
            .expect_err("a 64-entry table has no row 64");
        assert!(matches!(err, CoreError::InvalidValue { .. }), "{err:?}");
    }

    #[test]
    fn the_position_row_is_the_chunks_own_absolute_positions() {
        let mut rows = InputRows::new();
        rows.fill(&chunk(), 17, 64).expect("every id is a row");
        assert_eq!(rows.positions(), [17, 18, 19, 20]);
    }

    #[test]
    fn the_rotary_table_carries_the_lookback_in_front_of_the_chunks_first_row() {
        let mut rows = InputRows::new();
        rows.fill(&chunk(), 17, 64).expect("every id is a row");
        let table = rows.mrope();
        assert_eq!(
            table.len(),
            chunk_row() + 4 * axes(),
            "one row per lookback position and one per token"
        );
        // Four rows reaching back from the chunk, then the chunk's own four.
        let want: Vec<i32> = [13, 14, 15, 16, 17, 18, 19, 20]
            .into_iter()
            .flat_map(|p| [p, p, p])
            .collect();
        assert_eq!(table, want);
    }

    #[test]
    fn the_offset_the_query_rotation_binds_lands_on_the_chunks_first_row() {
        // The whole point of the prefix: bound at zero the table starts at the
        // lookback, and bound past the prefix it starts at the chunk. A writer
        // that laid the rows out the other way round would still fill the
        // buffer, and the query rotation would rotate every row by the lookback.
        let mut rows = InputRows::new();
        rows.fill(&chunk(), 17, 64).expect("every id is a row");
        let table = rows.mrope();
        assert_eq!(&table[..axes()], [13, 13, 13]);
        assert_eq!(
            &table[chunk_row()..chunk_row() + axes()],
            [17, 17, 17],
            "past the prefix the table starts at the chunk's own first position"
        );
        assert_eq!(
            i64::from(table[chunk_row()]),
            i64::from(rows.positions()[0]),
            "and that position is the one the position row names"
        );
    }

    #[test]
    fn a_text_tokens_three_axes_are_one_position_repeated() {
        let mut rows = InputRows::new();
        rows.fill(&chunk(), 5, 64).expect("every id is a row");
        for row in rows.mrope().chunks(axes()) {
            assert!(
                row.windows(2).all(|pair| pair[0] == pair[1]),
                "a token that is not part of a picture has one position: {row:?}"
            );
        }
    }

    #[test]
    fn the_lookback_rows_clamp_at_the_head_of_a_sequence() {
        // The pooled-key pass rounds a position down to the compression
        // boundary, which at the head of a sequence is zero, so it never reads
        // these rows. They are filled rather than left alone because a buffer
        // with a hole in it is a buffer whose reader has to be trusted.
        let mut rows = InputRows::new();
        rows.fill(&chunk(), 0, 64).expect("every id is a row");
        assert_eq!(&rows.mrope()[..chunk_row()], [0; 12]);

        // Two positions in, the two rows that would name a negative position
        // clamp and the two that name a real one do not.
        rows.fill(&chunk(), 2, 64).expect("every id is a row");
        let prefix: Vec<i32> = rows.mrope()[..chunk_row()]
            .chunks(axes())
            .map(|row| row[0])
            .collect();
        assert_eq!(prefix, [0, 0, 0, 1]);
    }

    #[test]
    fn the_three_rows_describe_one_chunk_and_the_same_one() {
        let mut rows = InputRows::new();
        rows.fill(&chunk(), 9, 64).expect("every id is a row");
        assert_eq!(rows.ids().len(), 4);
        assert_eq!(rows.positions().len(), 4);
        assert_eq!(
            rows.mrope().len(),
            rows.positions().len() * axes() + chunk_row()
        );
        // A refill replaces all three rather than appending to any of them.
        rows.fill(&[TokenId::new(1)], 40, 64)
            .expect("every id is a row");
        assert_eq!(rows.ids(), [1]);
        assert_eq!(rows.positions(), [40]);
        assert_eq!(rows.mrope().len(), axes() + chunk_row());
    }

    // -- the media branch ----------------------------------------------------

    /// The still-image placeholder this family's tower declares.
    const IMG: TokenId = TokenId::new(248_056);

    /// A prompt of `before` text tokens, one image of `h` by `w` merged
    /// positions, then `after` text tokens.
    fn picture(before: usize, h: u32, w: u32, after: usize) -> (Vec<TokenId>, Vec<VisionGrid>) {
        let pads = usize::try_from(h * w).expect("a fixture grid is small");
        let mut tokens = vec![TokenId::new(7); before];
        tokens.extend(std::iter::repeat_n(IMG, pads));
        tokens.extend(std::iter::repeat_n(TokenId::new(9), after));
        (tokens, vec![VisionGrid::new(1, h, w)])
    }

    fn walk_over(grids: &[VisionGrid], start: u32) -> Walk<'_> {
        Walk::new(
            grids,
            PadTokens::new(Some(IMG), None),
            start,
            MediaPositions::Mrope3D,
        )
    }

    #[test]
    fn a_media_chunk_writes_a_splice_map_and_a_text_chunk_clears_it() {
        // A stale map is the dangerous half: it names encoder rows for tokens
        // that are not placeholders, the splice copies them, and nothing below
        // reports it.
        let (tokens, grids) = picture(2, 2, 2, 1);
        let mut rows = InputRows::new();
        let mut walk = walk_over(&grids, 0);
        rows.fill_media(&tokens, 0, 248_320, &mut walk, None)
            .expect("every id is a row");
        assert_eq!(rows.splice_map(), [-1, -1, 0, 1, 2, 3, -1]);

        rows.fill(&chunk(), 0, 248_320).expect("every id is a row");
        assert!(rows.splice_map().is_empty(), "a text chunk splices nothing");
    }

    #[test]
    fn a_placeholder_token_still_gathers_and_still_occupies_its_own_cache_row() {
        // The splice overwrites the placeholder's embedding rather than
        // skipping its gather, and the position row is the sequence position
        // the cache is addressed by - not the grid coordinate.
        let (tokens, grids) = picture(1, 2, 2, 1);
        let mut rows = InputRows::new();
        let mut walk = walk_over(&grids, 30);
        rows.fill_media(&tokens, 30, 248_320, &mut walk, None)
            .expect("every id is a row");
        assert_eq!(rows.ids().len(), tokens.len());
        assert_eq!(rows.ids()[1], 248_056, "the placeholder id is gathered");
        assert_eq!(rows.positions(), [30, 31, 32, 33, 34, 35]);
    }

    #[test]
    fn the_rotary_table_is_the_walks_three_axes_interleaved_past_the_prefix() {
        // The walk hands over three planar streams and this family's rotation
        // reads one interleaved table. A transposition dropped here would put
        // every token's time axis where its height belongs, with every launch
        // succeeding.
        let (tokens, grids) = picture(1, 2, 2, 0);
        let mut streams = Streams::new();
        walk_over(&grids, 0).chunk(&tokens, &mut streams);

        let mut rows = InputRows::new();
        let mut walk = walk_over(&grids, 0);
        rows.fill_media(&tokens, 0, 248_320, &mut walk, None)
            .expect("every id is a row");
        let table = &rows.mrope()[chunk_row()..];
        for (i, row) in table.chunks(axes()).enumerate() {
            assert_eq!(
                row,
                [streams.t[i], streams.h[i], streams.w[i]],
                "row {i} is the walk's three axes, in the walk's own order"
            );
        }
        assert!(
            table
                .chunks(axes())
                .any(|row| row.windows(2).any(|pair| pair[0] != pair[1])),
            "a picture makes the three axes disagree, or this case proves nothing"
        );
    }

    #[test]
    fn a_split_prompt_writes_the_rows_it_would_have_written_unsplit() {
        // The whole point of carrying the walk and the rotary prefix across a
        // chunk boundary. The prefix is what this family adds over the dense
        // one, and getting it wrong rotates the pooled keys of every chunk but
        // the first at the wrong positions - only past the indexer's budget.
        let (tokens, grids) = picture(3, 2, 3, 2);
        let mut whole = InputRows::new();
        let mut walk = walk_over(&grids, 0);
        whole
            .fill_media(&tokens, 0, 248_320, &mut walk, None)
            .expect("every id is a row");
        let want_table = whole.mrope().to_vec();
        let want_map = whole.splice_map().to_vec();

        // The same prompt in two pieces, cut where a walk would allow it.
        let at = 3usize;
        let mut split = InputRows::new();
        let mut walk = walk_over(&grids, 0);
        split
            .fill_media(&tokens[..at], 0, 248_320, &mut walk, None)
            .expect("every id is a row");
        let head_table = split.mrope().to_vec();
        let head_map = split.splice_map().to_vec();
        let prefix = split.mrope_tail().to_vec();
        split
            .fill_media(
                &tokens[at..],
                u32::try_from(at).expect("a fixture prompt is short"),
                248_320,
                &mut walk,
                Some(&prefix),
            )
            .expect("every id is a row");

        let mut joined = head_table[..chunk_row()].to_vec();
        joined.extend_from_slice(&head_table[chunk_row()..]);
        joined.extend_from_slice(&split.mrope()[chunk_row()..]);
        assert_eq!(joined, want_table, "the split is invisible in the table");

        let mut joined_map = head_map;
        joined_map.extend_from_slice(split.splice_map());
        assert_eq!(joined_map, want_map, "and in the splice map");
    }

    #[test]
    fn the_second_chunks_prefix_is_the_first_chunks_last_rows_and_not_its_positions() {
        // Under the three-axis rule a text token's position is the running
        // MULTIMODAL position, which after one picture is already a different
        // number from the linear one. A prefix rebuilt from `pos0` would name
        // the linear number, and the pooled-key pass reads it.
        let (tokens, grids) = picture(1, 2, 3, 4);
        let at = 7usize;
        let mut rows = InputRows::new();
        let mut walk = walk_over(&grids, 0);
        rows.fill_media(&tokens[..at], 0, 248_320, &mut walk, None)
            .expect("every id is a row");
        let prefix = rows.mrope_tail().to_vec();
        rows.fill_media(
            &tokens[at..],
            u32::try_from(at).expect("a fixture prompt is short"),
            248_320,
            &mut walk,
            Some(&prefix),
        )
        .expect("every id is a row");
        assert_eq!(&rows.mrope()[..chunk_row()], prefix.as_slice());
        let first = u32::try_from(at).expect("a fixture prompt is short");
        let linear: Vec<i32> = (1..=MROPE_BACK)
            .rev()
            .flat_map(|back| {
                let p = position(first.saturating_sub(back));
                [p, p, p]
            })
            .collect();
        assert_ne!(
            rows.mrope()[..chunk_row()],
            linear[..],
            "the prefix is the walk's positions, not the sequence's"
        );
    }

    #[test]
    fn a_carried_prefix_of_the_wrong_length_is_refused_rather_than_slid() {
        // One row short and the whole table slides by a row: every token of the
        // chunk rotates at its neighbour's position and no launch notices.
        let (tokens, grids) = picture(1, 2, 2, 0);
        let mut rows = InputRows::new();
        let mut walk = walk_over(&grids, 0);
        let err = rows
            .fill_media(&tokens, 8, 248_320, &mut walk, Some(&[0, 0, 0]))
            .expect_err("a three-value prefix is not four rows");
        assert!(matches!(err, CoreError::ShapeMismatch { .. }), "{err:?}");
    }
}
