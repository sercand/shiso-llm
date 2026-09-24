//! The host rules for the trellis-codec projections.
//!
//! These weights are not blocks of quantized numbers but a trellis the kernel
//! walks, and the kernel that walks it is cooperative: every block of the grid
//! has to be resident at once, because they hand partial sums down a lock
//! chain. That makes the *grid* a correctness input rather than a tuning knob,
//! and it is why all of the arithmetic below is here rather than inside a
//! backend: the decisions are the same on every device, and only the two device
//! facts they close over - how many blocks fit at once, and how many
//! multiprocessors there are - come from outside.
//!
//! Nothing in this module launches anything. Every function is a decision, so
//! every one of them is checkable on a machine with no device at all, which is
//! the only way these rules get checked before hardware exists to run them on.

use shiso_core::error::{CoreError, CoreResult};

/// Narrowest trellis width the kernel set was built for.
pub const MIN_BITS: u32 = 4;
/// Widest trellis width the kernel set was built for.
pub const MAX_BITS: u32 = 6;

/// Tiles a shape divides the contraction into, indexed by shape number.
const TILE_K: [u32; 5] = [0, 16, 32, 32, 16];
/// Tiles a shape divides the output into, indexed by shape number.
const TILE_N: [u32; 5] = [0, 128, 128, 256, 512];
/// Threads a shape's block carries, indexed by shape number.
const BLOCK_DIM: [u32; 5] = [0, 256, 512, 512, 256];
/// Shapes the kernel set was built for.
const NUM_SHAPES: u8 = 4;

/// Barriers the device context reserves, which bounds how many slot groups a
/// grouped launch may run at once.
const MAX_BARRIERS: u32 = 1_024;

/// Blocks one expert group gets by default, which is also the group count a
/// device's multiprocessors are divided into.
const MOE_SMS_PER_EXPERT: u32 = 8;
/// Widest an expert group may grow when few experts are active.
const MOE_MAX_SMS_PER_EXPERT: u32 = 32;
/// Groups a routed launch may run at once.
const MOE_MAX_GROUPS: u32 = 64;

/// One of the kernel set's tiling shapes.
///
/// A newtype over the shape number rather than the number itself: the tile
/// widths and the block width are all looked up by it, and a raw integer that
/// happened to be zero or five would index the tables into the padding entry
/// and produce a grid of zero blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GemmShape(u8);

impl GemmShape {
    /// Wraps a shape number, or `None` when it is not one of the built shapes.
    #[must_use]
    pub const fn new(shape: u8) -> Option<Self> {
        if shape == 0 || shape > NUM_SHAPES {
            None
        } else {
            Some(Self(shape))
        }
    }

    /// The shape number, for a trace or a table lookup on the device side.
    #[must_use]
    pub const fn get(self) -> u8 {
        self.0
    }

    /// Every shape the kernel set was built for, narrowest tile first.
    #[must_use]
    pub const fn all() -> [Self; NUM_SHAPES as usize] {
        [Self(1), Self(2), Self(3), Self(4)]
    }

    /// The shape whose tiles are exactly these, or `None` when no built shape
    /// has them.
    ///
    /// This is how a shape number is recovered from a compiled entry point: the
    /// instantiation carries the tile widths, not the number, and the pair is
    /// unique across the built shapes. Recovering it this way rather than by
    /// position means a reordered table cannot quietly renumber a kernel.
    #[must_use]
    pub fn from_tiles(tile_k: u32, tile_n: u32) -> Option<Self> {
        Self::all()
            .into_iter()
            .find(|shape| shape.tile_k() == tile_k && shape.tile_n() == tile_n)
    }

    /// Elements of the contraction one tile covers.
    #[must_use]
    pub const fn tile_k(self) -> u32 {
        TILE_K[self.0 as usize]
    }

    /// Elements of the output one tile covers.
    #[must_use]
    pub const fn tile_n(self) -> u32 {
        TILE_N[self.0 as usize]
    }

    /// Threads one block of this shape carries.
    #[must_use]
    pub const fn block_dim(self) -> u32 {
        BLOCK_DIM[self.0 as usize]
    }

    /// Whether a weight of these dimensions tiles exactly under this shape.
    ///
    /// A shape whose tiles do not divide the weight leaves a partial tile no
    /// block owns, so a shape that does not divide is not a slower choice - it
    /// is a wrong one.
    #[must_use]
    pub const fn divides(self, k: u32, n: u32) -> bool {
        k.is_multiple_of(self.tile_k()) && n.is_multiple_of(self.tile_n())
    }
}

/// The two ways a projection is launched, which changes how the heuristic reads
/// the effective dimensions.
///
/// An enum rather than a `bool`, because at a call site the two look the same
/// and mean different grids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fanout {
    /// One weight, one output: a dense projection.
    Single,
    /// Several weights behind one launch, each with its own slot: the routed
    /// experts and the shared-expert pair.
    Grouped {
        /// Slots the launch covers, counting both weights of a pair.
        slots: u32,
        /// Whether each slot reads its own activation rows rather than sharing
        /// one input with the others.
        input_per_slot: bool,
    },
}

impl Fanout {
    /// The multiplier the heuristic applies to the contraction width.
    const fn input_scale(self) -> u32 {
        match self {
            Self::Single => 1,
            Self::Grouped {
                slots,
                input_per_slot,
            } => {
                if input_per_slot {
                    slots
                } else {
                    1
                }
            }
        }
    }

    /// The multiplier the heuristic applies to the output width.
    const fn output_scale(self) -> u32 {
        match self {
            Self::Single => 1,
            Self::Grouped { slots, .. } => slots,
        }
    }

    /// Whether the heuristic's narrow-weight shortcut applies.
    const fn is_grouped(self) -> bool {
        matches!(self, Self::Grouped { .. })
    }
}

/// Refuses a weight the kernels cannot tile at all.
///
/// Two of the three conditions are the trellis codec's own: the contraction has
/// to be a whole number of sixteen-element groups, and the output a whole
/// number of the 128 columns the output transform mixes together. The third is
/// the built bit widths.
///
/// # Errors
///
/// [`CoreError::ShapeMismatch`] naming the dimension that does not tile, and
/// [`CoreError::Unsupported`] for a trellis width this build has no kernel for.
pub fn check_shape(k: u32, n: u32, bits: u32) -> CoreResult<()> {
    if !(MIN_BITS..=MAX_BITS).contains(&bits) {
        return Err(CoreError::Unsupported {
            what: "a trellis width outside the range this kernel set was built for",
            backend: "engine",
        });
    }
    if k == 0 || !k.is_multiple_of(16) {
        return Err(CoreError::ShapeMismatch {
            field: "trellis contraction width",
            expected: 16,
            got: u64::from(k),
        });
    }
    if n == 0 || !n.is_multiple_of(128) {
        return Err(CoreError::ShapeMismatch {
            field: "trellis output width",
            expected: 128,
            got: u64::from(n),
        });
    }
    Ok(())
}

/// Picks the tiling shape for one projection.
///
/// The heuristic proposes a shape from the effective dimensions, and then the
/// proposal is walked *down* to the first shape that actually tiles the weight.
/// Only if none of those does is the whole table scanned upwards, and only if
/// that fails too is the weight unsupported - which is what the `None` means.
///
/// The row count is not an input. It is deliberately absent from the source
/// heuristic as well, and keeping it absent is what makes a captured graph and
/// an eager launch of the same projection pick the same grid.
#[must_use]
pub fn select_shape(k: u32, n: u32, bits: u32, fanout: Fanout) -> Option<GemmShape> {
    let mod_256 = n.is_multiple_of(256);
    let mod_512 = n.is_multiple_of(512);
    let k_eff = u64::from(k) * u64::from(fanout.input_scale());
    let n_eff = u64::from(n) * u64::from(fanout.output_scale());

    // The proposal. The `bits >= 7` arm is unreachable in this build - a wider
    // trellis is refused by `check_shape` before anything gets here - and it is
    // transcribed anyway so that this table and the one it mirrors can be read
    // against each other line for line.
    let proposed: u8 = if (bits == 4 || bits == 2) && !fanout.is_grouped() && k_eff <= 2_048 {
        1
    } else if bits >= 7 {
        if mod_256 && n_eff <= 8_192 {
            if k_eff > 32_768 { 3 } else { 2 }
        } else if mod_512 && n_eff > 32_768 {
            4
        } else {
            2
        }
    } else if mod_256 && n_eff <= 4_096 {
        if k_eff > 8_192 && bits >= 3 { 3 } else { 2 }
    } else if mod_512 && n_eff > 16_384 {
        4
    } else if mod_256 {
        3
    } else {
        2
    };

    let mut shape = proposed;
    while shape >= 1 {
        if let Some(candidate) = GemmShape::new(shape)
            && candidate.divides(k, n)
        {
            return Some(candidate);
        }
        shape -= 1;
    }
    for shape in 1..=NUM_SHAPES {
        if let Some(candidate) = GemmShape::new(shape)
            && candidate.divides(k, n)
        {
            return Some(candidate);
        }
    }
    None
}

/// Blocks a dense trellis projection launches.
///
/// Blocks sharing an output tile column hand their partial sums down a lock
/// chain, so past about four blocks per column the chain costs more than the
/// extra streams earn. The floor of eight keeps a narrow weight from launching
/// a grid too small to hide its own latency, and the two ceilings are the
/// device's co-resident budget and the work that actually exists.
#[must_use]
pub fn gemm_blocks(k: u32, n: u32, shape: GemmShape, max_blocks: u32) -> u32 {
    let tiles_k = k / shape.tile_k();
    let tiles_n = n / shape.tile_n();
    let tiles = u64::from(tiles_k) * u64::from(tiles_n);
    let mut blocks = u64::from(tiles_n) * 4;
    blocks = blocks.max(8);
    blocks = blocks.min(u64::from(max_blocks));
    blocks = blocks.min(tiles);
    u32::try_from(blocks).unwrap_or(max_blocks)
}

/// The grid a grouped trellis projection launches.
///
/// Every slot gets its own block group where the co-resident budget allows, so
/// a decode launch streams all of a token's experts at once instead of in
/// rounds. The group width is capped by the tiles a single slot has, because
/// blocks past that would have nothing to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct GroupedGrid {
    /// Blocks in one slot's group.
    pub per_group: u32,
    /// Slot groups running at once.
    pub concurrency: u32,
}

impl GroupedGrid {
    /// Total blocks the launch starts.
    #[must_use]
    pub const fn blocks(&self) -> u32 {
        self.per_group.saturating_mul(self.concurrency)
    }
}

/// The grid for a grouped launch over `slots` weights.
#[must_use]
pub fn mgemm_grid(k: u32, n: u32, shape: GemmShape, slots: u32, max_blocks: u32) -> GroupedGrid {
    let tiles = u64::from(k / shape.tile_k()) * u64::from(n / shape.tile_n());
    let mut concurrency = u64::from(slots);
    concurrency = concurrency.min(u64::from(max_blocks));
    concurrency = concurrency.min(u64::from(MAX_BARRIERS));
    let concurrency = concurrency.max(1);
    let mut per_group = u64::from(max_blocks) / concurrency;
    per_group = per_group.min(tiles);
    GroupedGrid {
        per_group: u32::try_from(per_group.max(1)).unwrap_or(1),
        concurrency: u32::try_from(concurrency).unwrap_or(1),
    }
}

/// The grid for a grouped launch over `slots` weights whose blocks per slot are
/// the ones a launch over `basis` slots would give.
///
/// A slot's result depends on how many blocks share its tiles - they split its
/// contraction - so a grid sized by the slot count makes a row's answer depend
/// on how many other rows the pass carries. Pinning the group width to one
/// row's launch and running the extra slots in rounds makes every row of a
/// verify pass or a decode batch the arithmetic a single-row step does.
#[must_use]
pub fn mgemm_grid_rounds(
    k: u32,
    n: u32,
    shape: GemmShape,
    slots: u32,
    basis: u32,
    max_blocks: u32,
) -> GroupedGrid {
    let one = mgemm_grid(k, n, shape, basis, max_blocks);
    let fit = (max_blocks / one.per_group.max(1)).max(1);
    GroupedGrid {
        per_group: one.per_group,
        concurrency: slots.min(fit).clamp(1, MAX_BARRIERS),
    }
}

/// The grid a fused routed launch runs: groups of blocks, one expert at a time
/// each, every block of the grid co-resident for the group barriers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RoutedGrid {
    /// Expert groups running at once.
    pub groups: u32,
    /// Blocks in one group.
    pub group_size: u32,
}

impl RoutedGrid {
    /// Total blocks the launch starts.
    #[must_use]
    pub const fn blocks(&self) -> u32 {
        self.groups.saturating_mul(self.group_size)
    }
}

/// The grid for a fused routed launch on a device with `sms` multiprocessors.
#[must_use]
pub fn routed_grid(sms: u32, max_blocks: u32, experts: u32) -> RoutedGrid {
    let mut groups = sms / MOE_SMS_PER_EXPERT;
    groups = groups.min(MOE_MAX_GROUPS);
    groups = groups.min(experts);
    let groups = groups.max(1);
    let mut group_size = sms / groups;
    group_size = group_size.min(MOE_MAX_SMS_PER_EXPERT);
    if group_size.saturating_mul(groups) > max_blocks {
        group_size = max_blocks / groups;
    }
    RoutedGrid {
        groups,
        group_size: group_size.max(1),
    }
}

/// Bytes of the staging a fused routed launch needs.
///
/// Per group and token: the gathered gate and up inputs in half precision - the
/// same rows later hold the single-precision down output - the two
/// single-precision intermediates, and the half-precision down input. An
/// undersized buffer here is silent out-of-bounds corruption rather than a
/// failed launch, which is why the size is derived rather than guessed at.
#[must_use]
pub fn routed_staging_bytes(grid: RoutedGrid, n_tok: u32, hidden: u32, intermediate: u32) -> u64 {
    routed_staging(grid, n_tok, hidden, intermediate).bytes
}

// ---------------------------------------------------------------------------
// the per-stream scratch a cooperative launch reads
// ---------------------------------------------------------------------------

/// Integers the lock buffer holds for output tiles.
///
/// One per tile of the widest output the kernels were built to address. Blocks
/// sharing an output tile column hand their partial sums down this chain, and a
/// block that found a stale value here would add one twice.
const LOCK_TILE_INTS: u64 = 1 << 20;
/// Integers the lock buffer holds for the grid-wide barriers: a counter and a
/// sense flag for each.
const LOCK_BARRIER_INTS: u64 = 2 * MAX_BARRIERS as u64;
/// Integers the routed scheduler holds: the next ticket, the retired-group
/// count, and one published ticket per group.
const LOCK_SCHED_INTS: u64 = 2 + MOE_MAX_GROUPS as u64;
/// Bytes of one `int32`, which every slot of the lock buffer is.
const LOCK_INT_BYTES: u64 = 4;

/// Bytes the lock buffer occupies, at the head of the per-stream scratch.
///
/// Fixed, and first, for the reason the whole carve-up is arranged around: the
/// kernels leave this buffer zeroed and every launch on a stream shares it, so
/// its address may not move when the staging behind it grows. A reservation
/// that moved it would hand the next launch a chain of stale locks.
pub const LOCK_BYTES: u64 = (LOCK_TILE_INTS + LOCK_BARRIER_INTS + LOCK_SCHED_INTS) * LOCK_INT_BYTES;

/// Alignment the staging region starts at.
///
/// The transformed-activation slab is read and written as 128-bit vectors - a
/// `half4` store in the input transform, a sixteen-byte `cp.async` in the
/// matmul - so an address that is not sixteen-byte aligned is a *misaligned
/// address* fault, not a slow path. The reference never has to think about it
/// because its staging is its own `cudaMalloc`; here it follows the lock
/// buffer, whose size is 4,202,760 bytes - eight past a sixteen-byte boundary.
/// That off-by-eight faulted the first trellis projection of the first layer.
///
/// 256 rather than 16 because that is what a device allocator hands back, so
/// every offset carved inside the region keeps the alignment it would have had
/// as an allocation of its own.
pub const STAGING_ALIGN: u64 = 256;

/// Where the staging region begins inside the per-stream scratch.
///
/// The lock buffer's size rounded up to [`STAGING_ALIGN`]. Every
/// [`crate::sparse::bind::Operand::CoopStaging`] offset is measured from here.
pub const STAGING_BASE: u64 = LOCK_BYTES.next_multiple_of(STAGING_ALIGN);

/// Half-precision elements the transformed-activation slab never drops below.
///
/// A floor rather than an exact size: the slab is grown and never shrunk, so a
/// first launch that asked for exactly its own rows would make every wider
/// launch after it a growth, and a growth cannot happen mid-capture.
const HADAMARD_FLOOR_HALVES: u64 = 8 << 20;
/// Bytes of one half-precision element.
const HALF_BYTES: u64 = 2;

/// Where one cooperative launch's staging lives inside the per-stream scratch.
///
/// The reference this is transcribed from keeps three separate per-stream
/// allocations - the locks, the transformed activations, and the routed
/// block's staging - because growing one there must not move the others. One
/// reservation cannot do that, so the layout here is the lock buffer at a fixed
/// offset of zero followed by a single staging region that every launch carves
/// its own way. That is sound because the staging is transient in exactly the
/// sense the locks are not: it is written and read inside one launch (or, for
/// an expanded weight, by the launch immediately after), and no launch expects
/// to find what a previous one left there. The locks do, which is why they and
/// only they are pinned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub struct Staging {
    /// Bytes of staging this launch carves, past the lock prefix.
    pub bytes: u64,
}

impl Staging {
    /// No staging: a launch that reads only the locks.
    #[must_use]
    pub const fn none() -> Self {
        Self { bytes: 0 }
    }

    /// The staging a launch needs for `bytes` of its own.
    #[must_use]
    pub const fn of(bytes: u64) -> Self {
        Self { bytes }
    }

    /// The transformed-activation slab a dense or grouped trellis launch reads.
    ///
    /// One transformed slab per launch slot and row: an undersized one here is
    /// not a failed launch but a silent write past the end, which is why the
    /// size is derived from the slot count rather than from the dense shape.
    #[must_use]
    pub fn hadamard(slots: u32, rows: u32, k: u32) -> Self {
        let halves = u64::from(slots.max(1))
            .saturating_mul(u64::from(rows))
            .saturating_mul(u64::from(k));
        Self {
            bytes: halves.max(HADAMARD_FLOOR_HALVES) * HALF_BYTES,
        }
    }

    /// The staging a decode-shaped dense projection carves: the transformed
    /// activation, then at [`Staging::decode_partials`] the slices' partial
    /// sums, floored as the transformed slab is so a wider pass is never a
    /// growth.
    #[must_use]
    pub fn decode(rows: u32, k: u32, n: u32, ksplit: u32) -> Self {
        let partials = u64::from(ksplit)
            .saturating_mul(u64::from(rows))
            .saturating_mul(u64::from(n))
            .saturating_mul(4);
        Self {
            bytes: Self::decode_partials(rows, k)
                .saturating_add(partials)
                .max(HADAMARD_FLOOR_HALVES * HALF_BYTES),
        }
    }

    /// Where a decode-shaped dense projection's partial sums start inside its
    /// staging: past the transformed activation, at the staging alignment.
    ///
    /// The activation is stored in MMA fragment order, whole sixteen-row
    /// tiles of it with the rows past the pass zeroed, and in that order one
    /// row's halves are spread through the whole tile. Sized by the rows alone,
    /// the partial sums overlapped the padding - that is, every row's data -
    /// and the slices' stores raced the other blocks' reads of the activation.
    #[must_use]
    pub fn decode_partials(rows: u32, k: u32) -> u64 {
        (u64::from(rows.next_multiple_of(16)) * u64::from(k) * HALF_BYTES)
            .next_multiple_of(STAGING_ALIGN)
    }

    /// The slab an expanded trellis weight is written into.
    #[must_use]
    pub fn expanded(k: u32, n: u32) -> Self {
        Self {
            bytes: u64::from(k).saturating_mul(u64::from(n)) * HALF_BYTES,
        }
    }

    /// The larger of two staging requirements.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self {
            bytes: if self.bytes > other.bytes {
                self.bytes
            } else {
                other.bytes
            },
        }
    }

    /// Bytes a stream must have reserved for a launch that carves this.
    #[must_use]
    pub const fn reservation(self) -> u64 {
        if self.bytes == 0 {
            0
        } else {
            STAGING_BASE.saturating_add(self.bytes)
        }
    }
}

/// Where the routed block's three staging buffers start, as offsets into the
/// staging region.
///
/// The three are one allocation in the reference and are carved in this order:
/// the gathered half-precision gate and up input - which the same bytes later
/// hold the single-precision down output in - then the two single-precision
/// intermediates, then the half-precision down input. A reader who needs to
/// know why the first is sized for halves and read back as floats will find it
/// in [`routed_staging_bytes`], which sizes all three together.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct RoutedStaging {
    /// Offset of the gathered input, which later holds the down output.
    pub state: u64,
    /// Offset of the gate and up outputs.
    pub intermediate: u64,
    /// Offset of the down projection's input.
    pub act: u64,
    /// Bytes the three occupy together.
    pub bytes: u64,
}

/// The routed block's staging offsets for one launch.
#[must_use]
pub fn routed_staging(
    grid: RoutedGrid,
    n_tok: u32,
    hidden: u32,
    intermediate: u32,
) -> RoutedStaging {
    let rows = u64::from(grid.groups).saturating_mul(u64::from(n_tok));
    let state_bytes = rows.saturating_mul(2 * u64::from(hidden) * HALF_BYTES);
    let inter_bytes = rows.saturating_mul(2 * u64::from(intermediate) * 4);
    let act_bytes = rows.saturating_mul(u64::from(intermediate) * HALF_BYTES);
    // Each of the three is reached by vector loads of its own, so each starts
    // where an allocation would have: the widths this family uses make the
    // rounding a no-op, and a width that did not would fault rather than
    // degrade.
    let intermediate_at = state_bytes.next_multiple_of(STAGING_ALIGN);
    let act_at = intermediate_at
        .saturating_add(inter_bytes)
        .next_multiple_of(STAGING_ALIGN);
    RoutedStaging {
        state: 0,
        intermediate: intermediate_at,
        act: act_at,
        bytes: act_at.saturating_add(act_bytes),
    }
}

/// Elements between one expert's tiles and the next, in the units the trellis
/// kernels index a stacked tensor by.
///
/// The stacked tensor's stride is counted in 16-bit words rather than bytes,
/// because that is the element type the kernel walks the tiles as.
#[must_use]
pub fn expert_tile_stride(k: u32, n: u32, bits: u32) -> u64 {
    u64::from(k)
        .saturating_mul(u64::from(n))
        .saturating_mul(u64::from(bits))
        / 8
        / HALF_BYTES
}

/// Threads one expansion block carries.
pub const RECONSTRUCT_THREADS: u32 = 256;
/// Output columns one expansion block covers.
pub const RECONSTRUCT_TILE_N: u32 = 128;
/// Contraction rows one expansion block covers.
pub const RECONSTRUCT_TILE_K: u32 = 128;

/// Whether a projection is wide enough to be worth expanding once.
///
/// The trellis kernel streams the whole weight once per sixteen rows, so a
/// prefill chunk re-reads it over a hundred times; past the threshold it is
/// cheaper to expand the weight to half precision once and hand the result to
/// the BLAS tier. Decode never reaches this: it runs a handful of rows, inside
/// a captured graph where the scratch could not be grown anyway - which is what
/// the `capturing` term says.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct ReconstructGate {
    /// Rows the projection is being asked for.
    pub rows: u32,
    /// Whether a BLAS tier is available at all.
    pub blas: bool,
    /// Whether the stream is mid-capture.
    pub capturing: bool,
}

/// Whether the expand-then-multiply path is taken.
#[must_use]
pub fn reconstruct(gate: ReconstructGate, k: u32, n: u32) -> bool {
    gate.rows > super::kernels::EXL3_RECONSTRUCT_ROWS
        && gate.blas
        && !gate.capturing
        && k.is_multiple_of(128)
        && n.is_multiple_of(128)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_narrow_four_bit_dense_weight_takes_the_smallest_shape() {
        // The arm exists for a short contraction into a narrow output at four
        // bits, dense. The bound is on the contraction, so a 2048-wide one
        // takes it and the model's own 2560-wide indexer key projection - one
        // step past the bound - does not.
        let shape = select_shape(2_048, 128, 4, Fanout::Single).expect("a built shape tiles it");
        assert_eq!(shape.get(), 1);
        assert_eq!(shape.tile_k(), 16);
        assert_eq!(shape.tile_n(), 128);
        let wider = select_shape(2_560, 128, 4, Fanout::Single).expect("a built shape tiles it");
        assert_eq!(wider.get(), 2);
    }

    #[test]
    fn the_same_weight_behind_a_grouped_launch_does_not_take_that_shape() {
        // The narrow arm is explicitly not for grouped launches: the slots
        // multiply the effective widths, so the proposal starts elsewhere.
        let grouped = Fanout::Grouped {
            slots: 10,
            input_per_slot: false,
        };
        let shape = select_shape(2_048, 128, 4, grouped).expect("a built shape tiles it");
        assert_ne!(shape.get(), 1);
    }

    #[test]
    fn a_proposal_that_does_not_tile_walks_down_before_it_scans_up() {
        // 2560 x 640: the 256- and 512-wide output tiles do not divide 640, so
        // whatever is proposed has to come down to a 128-wide one.
        let shape = select_shape(2_560, 640, 4, Fanout::Single).expect("a built shape tiles it");
        assert!(shape.divides(2_560, 640), "the chosen shape has to tile");
        assert!(shape.tile_n() == 128, "only the 128-wide tiles divide 640");
    }

    #[test]
    fn a_weight_no_shape_tiles_is_refused_rather_than_rounded() {
        // A contraction of 24 is not a whole number of tiles under any shape.
        assert_eq!(select_shape(24, 128, 4, Fanout::Single), None);
    }

    #[test]
    fn a_shape_number_outside_the_built_table_cannot_be_constructed() {
        assert_eq!(GemmShape::new(0), None);
        assert_eq!(GemmShape::new(5), None);
        assert!(GemmShape::new(1).is_some());
        assert!(GemmShape::new(4).is_some());
    }

    #[test]
    fn the_dense_block_count_is_clamped_from_below_by_eight_and_from_above_by_the_work() {
        let shape = GemmShape::new(1).expect("shape one is built");
        // 128 outputs is one output tile, so four blocks per column is four -
        // under the floor, which raises it to eight.
        assert_eq!(gemm_blocks(2_560, 128, shape, 48), 8);
        // A wide output asks for more than the device can hold at once.
        let wide = GemmShape::new(2).expect("shape two is built");
        assert_eq!(gemm_blocks(2_560, 2_560, wide, 48), 48);
        // And a tiny weight is bounded by the tiles that exist, not by the
        // floor: 16 x 128 under shape one is one tile.
        assert_eq!(gemm_blocks(16, 128, shape, 48), 1);
    }

    #[test]
    fn a_grouped_launch_gives_every_slot_its_own_group_while_the_budget_allows() {
        let shape = GemmShape::new(1).expect("shape one is built");
        // Ten routed experts on a device holding 48 blocks: ten groups of four.
        let grid = mgemm_grid(2_560, 128, shape, 10, 48);
        assert_eq!(grid.concurrency, 10);
        assert_eq!(grid.per_group, 4);
        assert_eq!(grid.blocks(), 40);
        // The shared expert's pair is two slots, so the groups are wide.
        let pair = mgemm_grid(2_560, 640, shape, 2, 48);
        assert_eq!(pair.concurrency, 2);
        assert_eq!(pair.per_group, 24);
    }

    #[test]
    fn a_grouped_launch_never_starts_more_groups_than_the_device_holds_blocks() {
        let shape = GemmShape::new(1).expect("shape one is built");
        let grid = mgemm_grid(2_560, 128, shape, 512, 48);
        assert_eq!(grid.concurrency, 48);
        assert_eq!(grid.per_group, 1);
    }

    #[test]
    fn the_routed_grid_divides_the_multiprocessors_and_never_exceeds_them() {
        let grid = routed_grid(48, 48, 512);
        assert_eq!(grid.groups, 6);
        assert_eq!(grid.group_size, 8);
        assert_eq!(grid.blocks(), 48);
        // Few experts bound the group count, and the groups widen to the cap.
        let narrow = routed_grid(128, 128, 2);
        assert_eq!(narrow.groups, 2);
        assert_eq!(narrow.group_size, 32);
    }

    #[test]
    fn the_reconstruct_gate_flips_between_one_hundred_forty_four_and_one_more() {
        let at = |rows| ReconstructGate {
            rows,
            blas: true,
            capturing: false,
        };
        assert!(!reconstruct(at(144), 2_560, 2_560));
        assert!(reconstruct(at(145), 2_560, 2_560));
        // A dimension that is not a whole number of 128 has no expanded form.
        // The stream mixes' low-rank width is the one in the model: 320.
        assert!(!reconstruct(at(512), 2_560, 320));
        assert!(!reconstruct(at(512), 320, 2_560));
        // And a stream mid-capture cannot grow the scratch the expansion needs.
        assert!(!reconstruct(
            ReconstructGate {
                rows: 512,
                blas: true,
                capturing: true
            },
            2_560,
            2_560
        ));
        // Nor is there anywhere to send the expanded weight without a BLAS tier.
        assert!(!reconstruct(
            ReconstructGate {
                rows: 512,
                blas: false,
                capturing: false
            },
            2_560,
            2_560
        ));
    }

    #[test]
    fn a_weight_whose_dimensions_do_not_tile_is_named_by_the_dimension_that_fails() {
        let err = check_shape(24, 128, 4).expect_err("24 is not a whole number of groups");
        assert!(
            matches!(err, CoreError::ShapeMismatch { field, .. } if field.contains("contraction")),
            "{err:?}"
        );
        let err = check_shape(2_560, 130, 4).expect_err("130 is not a whole number of columns");
        assert!(
            matches!(err, CoreError::ShapeMismatch { field, .. } if field.contains("output")),
            "{err:?}"
        );
        let err = check_shape(2_560, 128, 3).expect_err("three bits has no kernel here");
        assert!(matches!(err, CoreError::Unsupported { .. }), "{err:?}");
    }

    #[test]
    fn the_lock_prefix_is_the_three_counts_the_kernels_index_it_by() {
        // A tile lock per output tile the kernels can address, a counter and a
        // sense flag per grid-wide barrier, and the routed scheduler's ticket,
        // retired count and one published ticket per group - all as int32.
        let ints = (1 << 20) + 2 * 1_024 + (2 + 64);
        assert_eq!(LOCK_BYTES, ints * 4);
        // And it is first, so that growing the staging behind it cannot move
        // the buffer every launch on a stream shares and leaves zeroed.
        assert_eq!(Staging::none().reservation(), 0);
        assert_eq!(Staging::of(64).reservation(), STAGING_BASE + 64);
    }

    #[test]
    fn the_staging_starts_where_an_allocation_would_and_not_where_the_locks_end() {
        // The locks end eight bytes past a sixteen-byte boundary, and the
        // transform that writes the first staging slab stores 128-bit vectors:
        // handing it that address is a misaligned-address fault on the first
        // trellis projection of the first layer, which is what it did.
        assert_eq!(LOCK_BYTES % 16, 8, "the off-by-eight this guards against");
        assert_eq!(STAGING_BASE % STAGING_ALIGN, 0);
        const {
            assert!(STAGING_BASE >= LOCK_BYTES, "the locks stay whole");
            assert!(
                STAGING_BASE - LOCK_BYTES < STAGING_ALIGN,
                "and nothing more"
            );
        }
    }

    #[test]
    fn the_transformed_activation_slab_has_a_floor_and_counts_every_slot() {
        // One slab per slot and row: sizing a grouped launch for one slot is a
        // write past the end rather than a failed launch. Ten slots over a
        // 512-row chunk is well past the floor, so the count is what decides.
        let ten = Staging::hadamard(10, 512, 2_560);
        assert_eq!(ten.bytes, 10 * 512 * 2_560 * 2);
        // And every launch below the floor asks for the floor, so that the
        // launches after it are not each a growth - which is what a captured
        // stream cannot do. A single slot over the same chunk is one of them.
        let floor = (8_u64 << 20) * 2;
        let one = Staging::hadamard(1, 512, 2_560);
        assert_eq!(one.bytes, floor);
        assert_eq!(Staging::hadamard(1, 1, 128).bytes, floor);
        assert_eq!(one.union(ten).bytes, ten.bytes);
    }

    #[test]
    fn the_expert_tile_stride_counts_sixteen_bit_words_and_not_bytes() {
        // The stacked tensor is walked as 16-bit words, so a stride in bytes
        // would address every other expert.
        assert_eq!(expert_tile_stride(2_560, 640, 4), 2_560 * 640 * 4 / 8 / 2);
    }

    #[test]
    fn the_routed_staging_is_sized_per_group_and_token_rather_than_per_expert() {
        let grid = routed_grid(48, 48, 512);
        let bytes = routed_staging_bytes(grid, 512, 2_560, 640);
        // Six groups, 512 tokens, and one row is 2*2560*2 + 2*640*4 + 640*2.
        let row: u64 = 2 * 2_560 * 2 + 2 * 640 * 4 + 640 * 2;
        assert_eq!(bytes, 6 * 512 * row);
        // The three buffers are carved in the kernel's own order and none of
        // them overlaps the next: the gathered input, which the same bytes
        // later hold the single-precision output in, then the two
        // single-precision intermediates, then the half-precision input the
        // contraction reads.
        let carve = routed_staging(grid, 512, 2_560, 640);
        assert_eq!(carve.state, 0);
        assert_eq!(carve.intermediate, 6 * 512 * 2 * 2_560 * 2);
        assert_eq!(carve.act, carve.intermediate + 6 * 512 * 2 * 640 * 4);
        assert_eq!(carve.bytes, bytes);
    }
}
