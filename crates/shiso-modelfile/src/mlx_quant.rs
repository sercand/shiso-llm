//! Affine group quantization, decoded to `f16`.
//!
//! This is the weight format the array-framework checkpoints in this project
//! use. A matrix is stored as three arrays that must be read together:
//!
//! - `weight`, a row-major `u32` array whose words each pack `32 / bits`
//!   consecutive weights of one row, lowest-numbered weight in the *low* bits;
//! - `scales` and `biases`, row-major `bf16` arrays with one entry per group of
//!   `group_size` consecutive weights along the same row.
//!
//! Decoding one weight is therefore
//!
//! ```text
//! q      = (word >> (bits * (j % (32 / bits)))) & ((1 << bits) - 1)
//! out_ij = f16( f32(q) * f32(scale[i][j / group_size]) + f32(bias[i][j / group_size]) )
//! ```
//!
//! Three properties of that expression are load-bearing and are the reason this
//! module exists at all rather than being three lines inside the loader:
//!
//! 1. **Fields are packed low bits first.** Weight `j` of a word lives at bit
//!    offset `bits * (j % (32 / bits))`, so the first weight of a word is in
//!    the least significant nibble or byte. Reading them the other way round
//!    still produces a plausible-looking matrix, which is exactly why the unit
//!    test pins a hand-written word.
//! 2. **The arithmetic happens in `f32` and rounds to `f16` exactly once.**
//!    `scale` and `bias` are widened from `bf16` before the multiply, the
//!    multiply and the add are separate operations rather than a fused
//!    multiply-add, and the single narrowing is round-to-nearest-even. Any
//!    other order changes the low bit of a large fraction of the weights.
//! 3. **Nothing here touches a file.** Decoding is a pure function over three
//!    byte slices, so it can be tested against an external oracle without a
//!    model on disk and reused by any container that stores this format.
//!
//! The module never widens beyond the two field widths the format defines: a
//! `bits` value outside `{4, 8}` is a load error, not a slow path.

use half::{bf16, f16};

use crate::reader::{GgufError, GgufResult};

/// Bits in one packed storage word.
const WORD_BITS: u32 = 32;
/// Bytes in one packed storage word.
const WORD_BYTES: usize = 4;
/// Bytes in one `bf16` scale or bias.
const SCALE_BYTES: usize = 2;

/// Subject named by the layout errors raised from this module.
///
/// The loader validates every layout against the tensor's own name before it
/// ever calls in here, so these errors are a second line of defence for a
/// direct caller rather than the message a user normally sees.
const SUBJECT: &str = "mlx affine block";

/// How a packed affine matrix is laid out.
///
/// The four numbers are what separate a correct decode from a plausible one,
/// and three of them are the same type, so they travel as one named struct
/// instead of as positional arguments that a caller can silently transpose.
///
/// `out_rows` and `in_dim` describe the *logical* matrix, not the packed one:
/// `in_dim` counts weights, so the packed row holds `in_dim * bits / 32` words.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct MlxAffineLayout {
    /// Rows of the logical matrix, the slowest-varying axis.
    pub out_rows: usize,
    /// Weights per row of the logical matrix, the fastest-varying axis.
    pub in_dim: usize,
    /// Weights sharing one scale and one bias.
    pub group_size: usize,
    /// Bits per packed weight; only 4 and 8 exist.
    pub bits: u32,
}

impl MlxAffineLayout {
    /// Describes a packed matrix.
    ///
    /// The argument order matches the way the numbers are written everywhere
    /// else in this crate - outer axis, inner axis, then the quantization
    /// parameters - so `[out_rows, in_dim]` reads the same here as in the
    /// checkpoint's own shape field.
    #[must_use]
    pub fn new(out_rows: usize, in_dim: usize, group_size: usize, bits: u32) -> Self {
        Self {
            out_rows,
            in_dim,
            group_size,
            bits,
        }
    }

    /// Number of logical weights, `None` when the product overflows.
    #[must_use]
    pub fn elements(&self) -> Option<usize> {
        self.out_rows.checked_mul(self.in_dim)
    }
}

/// The layout with every divisibility question already answered.
///
/// Computing these once turns the inner loop into shifts and adds, and - more
/// to the point - it puts every rejection in one place, so the decode loop has
/// no error paths at all.
#[derive(Debug, Clone, Copy)]
struct Packing {
    /// Weights packed into one 32-bit word.
    values_per_word: u32,
    /// Words spanned by one group.
    words_per_group: usize,
    /// Groups along one row.
    groups_per_row: usize,
    /// Packed words in one row.
    words_per_row: usize,
    /// Field mask, `(1 << bits) - 1`.
    mask: u32,
}

impl Packing {
    /// Validates `layout` and precomputes the strides the decode loop needs.
    fn new(layout: MlxAffineLayout) -> GgufResult<Self> {
        if layout.bits != 4 && layout.bits != 8 {
            return Err(layout_error(
                "4 or 8 bits per weight",
                format!("{} bits", layout.bits),
            ));
        }
        if layout.group_size == 0 || !layout.in_dim.is_multiple_of(layout.group_size) {
            return Err(layout_error(
                "a group size that divides the inner dimension",
                format!(
                    "inner dimension {} in groups of {}",
                    layout.in_dim, layout.group_size
                ),
            ));
        }
        // `bits` is 4 or 8, so this is 8 or 4 and the conversion cannot fail.
        let values_per_word = WORD_BITS / layout.bits;
        let per_word = usize::try_from(values_per_word).unwrap_or(0);
        if per_word == 0 || !layout.group_size.is_multiple_of(per_word) {
            return Err(layout_error(
                "a group size that is a whole number of packed words",
                format!(
                    "groups of {} at {} weights per word",
                    layout.group_size, values_per_word
                ),
            ));
        }
        let groups_per_row = layout.in_dim / layout.group_size;
        let words_per_group = layout.group_size / per_word;
        let words_per_row = groups_per_row.checked_mul(words_per_group).ok_or_else(|| {
            layout_error(
                "a row that fits in memory",
                format!("{} weights per row", layout.in_dim),
            )
        })?;
        Ok(Self {
            values_per_word,
            words_per_group,
            groups_per_row,
            words_per_row,
            mask: (1_u32 << layout.bits) - 1,
        })
    }
}

/// Decodes a packed affine matrix into `out`, row-major `[out_rows, in_dim]`.
///
/// `out` is cleared first and holds exactly `out_rows * in_dim` values when this
/// returns, so a caller can hand the same buffer to every tensor in a model
/// without reallocating per tensor.
///
/// `scales` and `biases` are `bf16`, which is what this format stores; passing
/// `f16` arrays here would decode to numbers that look right to three digits
/// and are wrong in the fourth.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when `bits` is not 4 or 8, when `group_size`
/// does not divide `in_dim`, when a group is not a whole number of packed
/// words, or when any of the three arrays is not exactly the length the layout
/// implies. The lengths are checked before a single byte is read, so a short
/// array is a load error rather than a truncated matrix.
pub fn dequant_affine(
    w: &[u8],
    scales: &[u8],
    biases: &[u8],
    layout: MlxAffineLayout,
    out: &mut Vec<f16>,
) -> GgufResult<()> {
    let packing = Packing::new(layout)?;
    let want_words = layout
        .out_rows
        .checked_mul(packing.words_per_row)
        .ok_or_else(|| layout_error("a matrix that fits in memory", "an overflowing one".into()))?;
    let want_groups = layout
        .out_rows
        .checked_mul(packing.groups_per_row)
        .ok_or_else(|| layout_error("a matrix that fits in memory", "an overflowing one".into()))?;

    // Viewing the three arrays as fixed-size byte groups rather than indexing
    // them by hand keeps the loop free of both offset arithmetic and any way to
    // fail: the casts below are alignment-free, so the only thing that can go
    // wrong is a length, and the lengths are checked here, once.
    let words = byte_groups::<WORD_BYTES>(w, "packed weights", want_words)?;
    let scales = byte_groups::<SCALE_BYTES>(scales, "scales", want_groups)?;
    let biases = byte_groups::<SCALE_BYTES>(biases, "biases", want_groups)?;

    let elements = layout
        .elements()
        .ok_or_else(|| layout_error("a matrix that fits in memory", "an overflowing one".into()))?;
    out.clear();
    out.reserve(elements);

    for row in 0..layout.out_rows {
        let word_base = row * packing.words_per_row;
        let group_base = row * packing.groups_per_row;
        for group in 0..packing.groups_per_row {
            let scale = bf16::from_bits(u16::from_le_bytes(scales[group_base + group])).to_f32();
            let bias = bf16::from_bits(u16::from_le_bytes(biases[group_base + group])).to_f32();
            let start = word_base + group * packing.words_per_group;
            for word in &words[start..start + packing.words_per_group] {
                let word = u32::from_le_bytes(*word);
                for field in 0..packing.values_per_word {
                    let q = u8::try_from((word >> (layout.bits * field)) & packing.mask)
                        .unwrap_or(u8::MAX);
                    // f32 throughout, one narrowing at the end: see the module
                    // docs for why this is not written as a fused multiply-add.
                    out.push(f16::from_f32(f32::from(q) * scale + bias));
                }
            }
        }
    }
    Ok(())
}

/// Views a byte slice as `want` groups of `N` bytes, or reports its length.
///
/// `[u8; N]` has alignment 1, so this cast never fails for a reason other than
/// length - which is exactly the failure worth reporting.
fn byte_groups<'a, const N: usize>(
    src: &'a [u8],
    what: &str,
    want: usize,
) -> GgufResult<&'a [[u8; N]]> {
    check_len(what, src.len(), want.saturating_mul(N))?;
    bytemuck::try_cast_slice(src).map_err(|_| layout_error(what, "an unreadable array".into()))
}

/// Rejects an array whose length disagrees with the layout.
fn check_len(what: &str, got: usize, want: usize) -> GgufResult<()> {
    if got == want {
        return Ok(());
    }
    Err(layout_error(
        &format!("{want} bytes of {what}"),
        format!("{got} bytes"),
    ))
}

/// Builds the one error shape this module reports.
fn layout_error(want: &str, got: String) -> GgufError {
    GgufError::LayoutMismatch {
        tensor: Box::from(SUBJECT),
        want: Box::from(want),
        got: got.into_boxed_str(),
    }
}
