//! Where a token sits when some of the tokens are pieces of a picture.
//!
//! A text-only prompt gives every token one position and the rotary embedding
//! one number to rotate by. A prompt with media gives a placeholder token three:
//! which frame it came from, which row of that frame, and which column. This
//! module is the rule that turns a token run into those three streams, plus the
//! fourth stream the splice needs — which encoder row, if any, each token takes.
//!
//! # Four streams from one walk, not two walks
//!
//! The positions and the splice map are produced together. They are answers to
//! the same question — where does this placeholder run start and how long is it
//! — and computing them in two passes means two scanners that can disagree about
//! a run boundary. When they disagree nothing fails: the request succeeds, the
//! model is fluent, and it describes a rearranged picture. One walk removes the
//! possibility.
//!
//! # The rule
//!
//! - A text token takes `t = h = w = pos` and advances `pos` by one.
//! - A media item of `t_len` temporal groups over a post-merge `gh x gw` grid
//!   occupies `t_len * gh * gw` consecutive placeholder tokens. Token `k` of
//!   group `g` takes `t = base + g`, `h = base + row`, `w = base + col`, and
//!   afterwards `pos` advances by `max(t_len, gh, gw)`.
//!
//! A still image is the `t_len == 1` case and reduces exactly to the image-only
//! rule: `base + g` collapses to `base` and the advance to `max(gh, gw)`.
//!
//! The advance is the item's extent on EVERY axis, so the text that follows
//! starts clear of all three streams. A long clip can exceed its own spatial
//! extent, which is why the temporal length joins the maximum rather than the
//! spatial pair being assumed to dominate.
//!
//! # The other rule, and why there are two
//!
//! That is [`MediaPositions::Mrope3D`]. A second family positions a picture the
//! way it positions text: every token, placeholder or not, takes the next
//! sequence position and the three streams are that one number repeated. What
//! makes a picture a picture there is not the position but the SPAN — a
//! placeholder may read every key of its own item, including the ones that come
//! after it, so a media span is attended both ways while text stays causal.
//!
//! That is the fifth stream. [`Streams::bound`] is the last key position each
//! token may read: its own position for a text token, and the last position of
//! its own item for a placeholder. A slot with no such rule reads a bound equal
//! to the position and is exactly causal, which is why the row is produced for
//! every request rather than only for the family that widens it.
//!
//! # Why a chunk may not be cut inside an item
//!
//! A bound can point PAST the token that carries it, and a key at a position
//! this chunk has not reached yet is a cache row nothing has written. So a
//! prefill split has to fall on an item boundary, which is what
//! [`Walk::split_point`] answers. The one case it cannot answer is an item
//! wider than the whole chunk; there the width wins, because taking nothing
//! would be a prefill that never advances.
//!
//! # Both placeholders, always
//!
//! An item's own temporal length already says whether it is a still or a clip,
//! so the two placeholder ids are consumed identically. The scan still has to
//! recognize both. Recognizing only the image one walks a clip's run a text
//! token at a time: every position after it shifts, no encoder row is spliced
//! over any of its tokens, and the model answers about the placeholder
//! embedding it was left holding.

use shiso_core::ids::TokenId;
use shiso_core::media::{PadTokens, VisionGrid};

use crate::shape::MediaPositions;

/// Positions the running stream advances by once `grid` is consumed.
///
/// The item's extent on EVERY axis, so the text that follows starts clear of
/// all three streams. A long clip can exceed its own spatial extent, which is
/// why the temporal length joins the maximum rather than the spatial pair being
/// assumed to dominate.
const fn advance(grid: VisionGrid) -> u32 {
    let spatial = if grid.grid_h() > grid.grid_w() {
        grid.grid_h()
    } else {
        grid.grid_w()
    };
    if grid.t_len() > spatial {
        grid.t_len()
    } else {
        spatial
    }
}

/// The five `int32` rows one chunk uploads.
///
/// Held together because they are always produced together and always have the
/// same length as the chunk. Three of them are positions the rotation reads, one
/// is the encoder row each token takes (or a negative number for a token that
/// takes none), and the last is how far along the sequence each token may look.
#[derive(Debug, Default)]
pub(crate) struct Streams {
    /// Temporal position per token.
    pub(crate) t: Vec<i32>,
    /// Row position per token.
    pub(crate) h: Vec<i32>,
    /// Column position per token.
    pub(crate) w: Vec<i32>,
    /// Encoder row per token, negative for a text token.
    pub(crate) map: Vec<i32>,
    /// Last key position each token may read.
    ///
    /// The token's own position wherever attention is causal, and the last
    /// position of its own item for a placeholder token of a family whose media
    /// spans are attended both ways.
    pub(crate) bound: Vec<i32>,
}

impl Streams {
    /// Five empty rows.
    ///
    /// A `const` constructor rather than `Default::default`, because the value
    /// that holds one is built in a `const fn`.
    pub(crate) const fn new() -> Self {
        Self {
            t: Vec::new(),
            h: Vec::new(),
            w: Vec::new(),
            map: Vec::new(),
            bound: Vec::new(),
        }
    }

    /// Empties every row, keeping their allocations.
    pub(crate) fn clear(&mut self) {
        self.t.clear();
        self.h.clear();
        self.w.clear();
        self.map.clear();
        self.bound.clear();
    }
}

/// A placeholder run that is part way through.
///
/// A run is the unit the rule is stated in, and a chunk boundary can fall inside
/// one. This is what lets it: the base position and the first encoder row are
/// fixed when the run starts and are still the same numbers when it resumes.
#[derive(Debug, Clone, Copy)]
struct Run {
    grid: VisionGrid,
    /// Position the item's three streams are measured from.
    base: u32,
    /// First encoder row of the item.
    row_base: u32,
    /// Tokens of the run already emitted.
    done: u32,
}

/// A walk over one request's token stream, carried across prefill chunks.
///
/// Stateful on purpose. A prompt wider than the scratch is split, and the split
/// is supposed to be invisible: an item that begins in one sub-chunk and ends in
/// the next has to keep its base position and keep handing out consecutive
/// encoder rows across the boundary. Rebuilding the running position from the
/// sub-chunk's own start would restart the stream at the linear token index,
/// which after a single image is already a different number.
#[derive(Debug)]
pub(crate) struct Walk<'a> {
    grids: &'a [VisionGrid],
    pads: PadTokens,
    /// Which of the two position rules this request's family states.
    positions: MediaPositions,
    /// Next item to consume.
    item: usize,
    /// Next encoder row to hand out.
    row: u32,
    /// Running position, shared by all three streams at a text token.
    ///
    /// Under [`MediaPositions::Linear`] an item advances it by its own token
    /// count, so this and [`Walk::linear`] are the same number at every token;
    /// under [`MediaPositions::Mrope3D`] an item advances it by its grid extent
    /// instead, and the two part company at the first picture.
    pos: u32,
    /// Running SEQUENCE position: one per token, whatever the token is.
    ///
    /// This is what the cache is addressed by, so it is what a bound is measured
    /// in. Keeping it beside `pos` rather than deriving it is what lets a walk
    /// resume mid-run: both are state a chunk boundary has to preserve.
    linear: u32,
    /// The run a previous chunk stopped in the middle of.
    run: Option<Run>,
}

impl<'a> Walk<'a> {
    /// Starts a walk over `grids`, with the running position at `start`.
    ///
    /// `grids` is the request's own window of items. A request that shares a
    /// packed encoder output with others is handed its slice of the grids and
    /// its slice of the rows, so there is no base index to get wrong here.
    ///
    /// `positions` is the family's rule, not the request's: a stack states one
    /// of the two and every request it serves is walked by it.
    pub(crate) const fn new(
        grids: &'a [VisionGrid],
        pads: PadTokens,
        start: u32,
        positions: MediaPositions,
    ) -> Self {
        Self {
            grids,
            pads,
            positions,
            item: 0,
            row: 0,
            pos: start,
            linear: start,
            run: None,
        }
    }

    /// Encoder rows handed out so far.
    pub(crate) const fn rows_consumed(&self) -> u32 {
        self.row
    }

    /// The largest `n <= width` such that `tokens[..n]` does not end inside a
    /// placeholder run, or `width` when no such cut exists.
    ///
    /// A chunk that ends inside an item would hand its first half a bound
    /// naming a key the chunk never reaches, and the cache row behind that key
    /// is one nothing has written yet. So the prefill's own split is pulled back
    /// to the last token before the item starts.
    ///
    /// The fallback is the item that is wider than the whole chunk: there is no
    /// cut that clears it, and returning zero would be a prefill that never
    /// advances. Such an item is cut. Both halves attend causally, so neither
    /// carries a bound naming a cache row its own chunk has not written, and
    /// the second half runs with the first already in the cache.
    ///
    /// This does not consume the walk. It reads the same state
    /// [`Walk::chunk`] would and leaves it alone, so a caller asks and then
    /// walks.
    pub(crate) fn split_point(&self, tokens: &[TokenId], width: usize) -> usize {
        if tokens.len() <= width {
            return tokens.len();
        }
        let mut at = 0usize;
        // What a run a previous chunk stopped inside still owes, and which item
        // the NEXT run is. The two are one decision: a run in flight is the item
        // at `self.item`, because [`Walk::chunk`] advances that index only when a
        // run completes. So the next placeholder run this scan meets is the item
        // after it, and seeding the index with `self.item` would measure that
        // run - and every item after it - by the in-flight item's count. When
        // the item in flight is the shorter of the two that lands the cut inside
        // the next item's run, which is the one thing this function exists to
        // prevent.
        //
        // The run itself is no more cuttable than a fresh one: the bound its
        // tokens carry names the item's own last position either way.
        let (mut owed, mut item) = match self.run {
            Some(run) => (
                count(run.grid.pad_count().saturating_sub(run.done)),
                self.item.saturating_add(1),
            ),
            None => (0, self.item),
        };
        let mut last = 0usize;
        while at < width {
            if owed > 0 {
                at = at.saturating_add(owed);
                owed = 0;
                if at <= width {
                    last = at;
                }
                continue;
            }
            let grid = self
                .pads
                .matches(tokens[at])
                .then(|| self.grids.get(item))
                .flatten();
            if let Some(grid) = grid {
                owed = count(grid.pad_count());
                item += 1;
            } else {
                at += 1;
                last = at;
            }
        }
        if last == 0 { width } else { last }
    }

    /// Appends the four streams for `tokens`, continuing where the last call
    /// left off. Exactly `tokens.len()` rows are appended to each stream.
    ///
    /// A placeholder token with no item left to describe it falls through to the
    /// text rule and takes no encoder row: that is a malformed request, it is
    /// caught by the row-count check before any of this runs, and this is the
    /// behaviour that keeps a walk from indexing past the end if it ever is not.
    pub(crate) fn chunk(&mut self, tokens: &[TokenId], out: &mut Streams) {
        let mut i = 0usize;
        while i < tokens.len() {
            if self.run.is_none() {
                let token = tokens[i];
                let grid = self
                    .pads
                    .matches(token)
                    .then(|| self.grids.get(self.item))
                    .flatten()
                    .copied();
                let Some(grid) = grid else {
                    out.t.push(clamp(self.pos));
                    out.h.push(clamp(self.pos));
                    out.w.push(clamp(self.pos));
                    out.map.push(-1);
                    // A text token reads every key up to its own and no
                    // further, which is what its own sequence position says.
                    out.bound.push(clamp(self.linear));
                    self.pos = self.pos.saturating_add(1);
                    self.linear = self.linear.saturating_add(1);
                    i += 1;
                    continue;
                };
                self.run = Some(Run {
                    grid,
                    base: self.pos,
                    row_base: self.row,
                    done: 0,
                });
            }
            // Unreachable: the block above either continued the loop or
            // installed a run. A break rather than a panic, so an edit that
            // ever broke that invariant would surface as a short stream at the
            // caller's length check instead of taking the process down.
            //
            // A copy rather than a mutable borrow, because the emission below
            // advances the walk's own sequence position at every token; it is
            // written back before this iteration ends, including on the path
            // that stops mid-run.
            let Some(mut run) = self.run else {
                break;
            };
            let grid = run.grid;
            let plane = grid.grid_h().saturating_mul(grid.grid_w());
            let total = grid.pad_count();
            // Under both position rules attention in the language model is causal:
            // every token reads up to its own sequence position and no further.
            while run.done < total && i < tokens.len() {
                // Group, then row, then column: the order the encoder emitted
                // the merged rows in, and therefore the order they are spliced.
                let k = run.done;
                match self.positions {
                    MediaPositions::Mrope3D => {
                        let group = k / plane;
                        let within = k % plane;
                        out.t.push(clamp(run.base.saturating_add(group)));
                        out.h
                            .push(clamp(run.base.saturating_add(within / grid.grid_w())));
                        out.w
                            .push(clamp(run.base.saturating_add(within % grid.grid_w())));
                    }
                    MediaPositions::Linear => {
                        // The next sequence position, exactly as a text token
                        // takes it: what makes this a picture is the span
                        // below, not the position.
                        let at = clamp(run.base.saturating_add(k));
                        out.t.push(at);
                        out.h.push(at);
                        out.w.push(at);
                    }
                }
                out.map.push(clamp(run.row_base.saturating_add(k)));
                out.bound.push(clamp(self.linear));
                self.linear = self.linear.saturating_add(1);
                run.done = run.done.saturating_add(1);
                i += 1;
            }
            let (done, base, row_base) = (run.done, run.base, run.row_base);
            self.run = Some(run);
            if done < total {
                // The chunk ended inside the run. The run keeps its base and its
                // first row, and the next chunk resumes at `done`.
                break;
            }
            self.pos = base.saturating_add(match self.positions {
                MediaPositions::Mrope3D => advance(grid),
                // One position per token, which keeps `pos` and `linear` the
                // same number.
                MediaPositions::Linear => total,
            });
            self.row = row_base.saturating_add(total);
            self.item += 1;
            self.run = None;
        }
    }
}

/// Pulls every bound back to `last`, the last position the chunk itself writes.
///
/// A position as the `int32` rows the kernels read, saturating rather than
/// wrapping. A position this large is a context far past anything the cache can
/// hold, so the clamp is unreachable rather than lossy.
fn clamp(value: u32) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

/// A token count as an index into a chunk, saturating rather than wrapping.
///
/// A count this large is a prompt no context holds, so the saturation is
/// unreachable; what it removes is a cast that could silently narrow on a host
/// whose pointers are smaller than a position.
fn count(value: u32) -> usize {
    usize::try_from(value).unwrap_or(usize::MAX)
}

#[cfg(test)]
#[path = "mrope_tests.rs"]
mod tests;
