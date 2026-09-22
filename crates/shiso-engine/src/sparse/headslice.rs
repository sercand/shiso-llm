//! A vocabulary head narrowed to the rows a draft is likely to want.
//!
//! # Why a draft pays for this and a target does not
//!
//! The vocabulary projection is the single widest matmul in the model: at this
//! checkpoint's widths it streams 477 MB of trellis for one row, which is most
//! of what a *whole target step* moves. A target step pays it once. A
//! speculative step pays it once per guess, and at a chain of three that is
//! two-thirds of what the guessing costs.
//!
//! A draft, though, does not need the right logits. It needs a token the target
//! is likely to agree with, and a token outside the slice is one the target
//! rejects - which costs a verify row that was already paid for. So a narrower
//! head is a speed-for-acceptance trade with **no correctness risk at all**: the
//! target's own pass is what commits, and it reads the whole vocabulary.
//!
//! # Why whole blocks of 128
//!
//! The trellis codec's output transform mixes every 128 consecutive output
//! columns together, so a logit depends on all eight of its block's tiles.
//! Single rows cannot be lifted out; blocks can, intact, and the gathered
//! payload then has the layout of an ordinary `(k, n_sel)` tensor that the
//! ordinary dense launch runs.

use std::sync::Arc;

use shiso_core::backend::{Backend, KernelArg, KernelRef};
use shiso_core::error::{CoreError, CoreResult};
use shiso_core::{Extent3, alloc};

use shiso_modelfile::quant;

use crate::launch;
use crate::sparse::kernels::MOD_MATVEC;
use crate::sparse::weights::{Weight, WeightPart};

/// Output columns one liftable block holds.
pub const BLOCK: u32 = 128;

/// Ids the slice covers when a caller states no preference.
///
/// The reference this port was built against uses the same number, and the
/// corpus counts beside this checkpoint put about 98.7% of the mass inside it.
pub const DEFAULT_IDS: u32 = 65_536;

/// A vocabulary head narrowed to some of its blocks, and the ids it kept.
#[derive(Debug)]
pub struct HeadSlice {
    weight: Weight,
    ids: Vec<u32>,
}

impl HeadSlice {
    /// The narrowed matrix, to project through.
    #[must_use]
    pub const fn weight(&self) -> &Weight {
        &self.weight
    }

    /// Rows the narrowed matrix has.
    #[must_use]
    pub fn rows(&self) -> u32 {
        u32::try_from(self.ids.len()).unwrap_or(u32::MAX)
    }

    /// The vocabulary id at `row` of the narrowed matrix.
    #[must_use]
    pub fn id(&self, row: u32) -> Option<u32> {
        self.ids.get(usize::try_from(row).ok()?).copied()
    }
}

/// Picks the blocks worth keeping, most-used first, and returns them in
/// ascending order.
///
/// `counts` is one `id count` pair per line, `#` comments ignored - the shape
/// the corpus tool beside this checkpoint writes. Mass is summed per block
/// rather than per id because a block is what can be lifted.
///
/// Ascending, because the gather reads the source once in order and the id
/// table has to agree with the row order the gather produced.
///
/// Empty when the counts name no id this head has, or when the target covers
/// the whole head: there is nothing to slice then, and a slice of everything
/// would cost a copy of the widest tensor in the model for no gain.
#[must_use]
pub fn blocks_from_counts(counts: &str, vocab: u32, target_ids: u32) -> Vec<u32> {
    let total = vocab / BLOCK;
    let want = target_ids.div_ceil(BLOCK);
    if total == 0 || want == 0 || want >= total {
        return Vec::new();
    }
    let mut mass = vec![0.0f64; total as usize];
    for line in counts.lines() {
        if line.starts_with('#') {
            continue;
        }
        let mut fields = line.split_whitespace();
        let Some(id) = fields.next().and_then(|f| f.parse::<u32>().ok()) else {
            continue;
        };
        if id >= vocab {
            continue;
        }
        let count = fields
            .next()
            .and_then(|f| f.parse::<f64>().ok())
            .unwrap_or(1.0);
        if let Some(slot) = mass.get_mut((id / BLOCK) as usize) {
            *slot += count;
        }
    }
    if mass.iter().sum::<f64>() <= 0.0 {
        return Vec::new();
    }
    let mut order: Vec<u32> = (0..total).collect();
    // Descending by mass, and by index where two blocks tie, so the choice is a
    // function of the counts rather than of a sort's stability.
    order.sort_by(|a, b| {
        let (x, y) = (mass[*a as usize], mass[*b as usize]);
        y.partial_cmp(&x)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(b))
    });
    let mut kept: Vec<u32> = order.into_iter().take(want as usize).collect();
    kept.sort_unstable();
    kept
}

/// Gathers `blocks` out of `source` into a head of its own.
///
/// # Errors
///
/// [`CoreError::Unsupported`] when the head is not a trellis tensor or the
/// blocks name nothing, and whatever the device reports for the allocation and
/// the two launches.
pub fn build(backend: &Arc<dyn Backend>, source: &Weight, blocks: &[u32]) -> CoreResult<HeadSlice> {
    let bits = quant::exl3::k_bits(source.dtype).ok_or(CoreError::Unsupported {
        what: "a sliced vocabulary head of a projection that is not a trellis tensor",
        backend: "engine",
    })?;
    let kept = u32::try_from(blocks.len()).unwrap_or(0);
    if kept == 0 || !source.out_dim.is_multiple_of(BLOCK) {
        return Err(CoreError::Unsupported {
            what: "a vocabulary head whose rows are not whole liftable blocks",
            backend: "engine",
        });
    }
    let rows = kept.saturating_mul(BLOCK);
    let bytes = Weight::payload_bytes(source.dtype, source.in_dim, rows).ok_or(
        CoreError::ShapeMismatch {
            field: "sliced head bytes",
            expected: 1,
            got: 0,
        },
    )?;
    let buffer = alloc(backend, bytes)?;

    // The block list, on the device the gather runs on.
    let list: Vec<u8> = blocks
        .iter()
        .flat_map(|b| i32::try_from(*b).unwrap_or(0).to_le_bytes())
        .collect();
    let list_bytes = alloc(backend, u64::try_from(list.len()).unwrap_or(0))?;
    backend.copy_h2d(list_bytes.view(), &list)?;

    // Sixteen bytes to a tile piece, eight n16 tiles to a block: the gather
    // moves whole pieces so that it never has to know the codec's bit layout.
    let tile_bytes = u64::from(BLOCK) * 2 * u64::from(bits) / 8;
    let tiles = u64::from(source.in_dim) * u64::from(rows) * u64::from(bits) / 8;
    let pieces = tiles / 16;
    let dst_tiles = buffer.slice(0, tiles)?;
    launch(
        backend.as_ref(),
        "q4e_exl3_slice_tiles_kernel",
        pieces,
        &[
            KernelArg::Buffer(dst_tiles),
            KernelArg::Buffer(source.part(WeightPart::Tiles)?),
            KernelArg::Buffer(list_bytes.view()),
            KernelArg::U32(kept),
            KernelArg::U32(source.out_dim / 16),
            KernelArg::U32(u32::try_from(tile_bytes).unwrap_or(0)),
            KernelArg::U64(pieces),
        ],
    )?;

    // The input scales are per contraction element and copy whole; the output
    // ones are per output row and follow the blocks.
    let input = u64::from(source.in_dim) * 2;
    backend.copy_d2d(
        buffer.slice(tiles, input)?,
        source.part(WeightPart::InputScales)?,
    )?;
    launch(
        backend.as_ref(),
        "q4e_exl3_slice_svh_kernel",
        u64::from(rows),
        &[
            KernelArg::Buffer(buffer.slice(tiles.saturating_add(input), u64::from(rows) * 2)?),
            KernelArg::Buffer(source.part(WeightPart::OutputScales)?),
            KernelArg::Buffer(list_bytes.view()),
            KernelArg::U32(kept),
        ],
    )?;
    backend.sync()?;

    let ids = blocks
        .iter()
        .flat_map(|b| (0..BLOCK).map(move |i| b.saturating_mul(BLOCK).saturating_add(i)))
        .collect();
    Ok(HeadSlice {
        weight: Weight::over(source.dtype, source.in_dim, rows, buffer),
        ids,
    })
}

/// One of the two gather launches, over `items` of work.
fn launch(
    backend: &dyn Backend,
    entry: &'static str,
    items: u64,
    args: &[KernelArg<'_>],
) -> CoreResult<()> {
    /// Threads the two gathers are written for.
    const BLOCK_DIM: u32 = 256;
    let grid = u32::try_from(items.div_ceil(u64::from(BLOCK_DIM))).unwrap_or(u32::MAX);
    launch::run(
        backend,
        KernelRef::new(MOD_MATVEC, entry),
        Extent3::new(grid, 1, 1),
        Extent3::new(BLOCK_DIM, 1, 1),
        0,
        args,
    )
}

#[cfg(test)]
mod tests {
    use super::{BLOCK, blocks_from_counts};

    /// One `id count` line per pair.
    fn counts(pairs: &[(u32, u32)]) -> String {
        use std::fmt::Write as _;

        let mut text = String::from("# a corpus count file\n");
        for (id, count) in pairs {
            let _ = writeln!(text, "{id} {count}");
        }
        text
    }

    /// The blocks with the most mass, and in ascending order whatever order the
    /// counts arrived in.
    ///
    /// Ascending is not cosmetic: the gather reads the source in the order it
    /// is given and the id table is built from the same list, so a descending
    /// list would put row `i` of the slice at a different vocabulary id than
    /// the one the gather copied there. Every launch would still succeed.
    #[test]
    fn the_heaviest_blocks_are_kept_and_handed_back_in_ascending_order() {
        // Four blocks of 128; the counts make block 3 heaviest, then 1.
        let text = counts(&[(400, 50), (130, 40), (10, 1), (300, 2)]);
        let kept = blocks_from_counts(&text, 4 * BLOCK, 2 * BLOCK);
        assert_eq!(kept, vec![1, 3], "the two heaviest, in ascending order");
    }

    /// Mass is summed per block, not per id: a block full of small counts can
    /// beat one with a single large one.
    #[test]
    fn mass_is_summed_over_a_whole_block() {
        let mut pairs = vec![(0u32, 10u32)];
        pairs.extend((BLOCK..BLOCK + 8).map(|id| (id, 2)));
        let kept = blocks_from_counts(&counts(&pairs), 2 * BLOCK, BLOCK);
        assert_eq!(kept, vec![1], "eight twos beat one ten");
    }

    /// A target that covers the head, counts that name nothing, and a head of
    /// no blocks all answer "nothing to slice" rather than a slice of
    /// everything - which would copy the widest tensor in the model for no gain.
    #[test]
    fn a_slice_that_would_keep_everything_is_no_slice_at_all() {
        let text = counts(&[(0, 1), (200, 1)]);
        assert!(blocks_from_counts(&text, 2 * BLOCK, 2 * BLOCK).is_empty());
        assert!(blocks_from_counts(&text, 2 * BLOCK, 4 * BLOCK).is_empty());
        assert!(blocks_from_counts("# nothing\n", 2 * BLOCK, BLOCK).is_empty());
        assert!(blocks_from_counts(&text, 0, BLOCK).is_empty());
    }

    /// An id past the head is dropped rather than folded into a block that does
    /// not exist, and a malformed line is skipped rather than taken as zero.
    #[test]
    fn ids_past_the_head_and_lines_that_are_not_pairs_are_dropped() {
        let text = "# header\n999999 100\nnot a line\n5 7\n";
        assert_eq!(blocks_from_counts(text, 2 * BLOCK, BLOCK), vec![0]);
    }
}
