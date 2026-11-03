//! CSF: the compressed storage `modelopt` uses for NVFP4 group scales.
//!
//! An NVFP4 matrix of `rows x cols` weights carries one `e4m3` scale byte per
//! group of [`GROUP`] weights along a row, `rows x cols/16` bytes in all. A
//! checkpoint whose quantization entry says `weight_scale_encoding: "csf"`
//! does not store that matrix. It stores two arrays that rebuild it exactly:
//!
//! - `weight_scale.nvfp4_csf_fixed`, `u8 [rows/16, 16 * (1 + C/2)]` where
//!   `C = cols/16` is the number of scale columns. Each 16-row group is one
//!   record: sixteen **base** bytes (one per row of the group, in row order),
//!   then the rows' four-bit **codes**, row-major, two to a byte, the even
//!   column in the low nibble. A scale is `base + code`.
//! - `weight_scale.nvfp4_csf_exceptions`, `u32 [n]`, little-endian, strictly
//!   ascending: each word is a position `row * C + col` in its low 24 bits and
//!   the **whole** `e4m3` byte in its high 8, which replaces whatever the fixed
//!   record produced there. Scales more than fifteen steps above their row's
//!   base are the ones that need it; about one in three hundred on this
//!   family's experts.
//!
//! The decode is lossless: what comes out is byte for byte the plain
//! `weight_scale` array an uncompressed export would have written, so every
//! NVFP4 kernel downstream reads it unchanged. The reference implementation
//! keeps both arrays on the device and expands them into a scratch buffer on
//! every call; this tree expands them once, at load, because the compressed
//! form saves 47% of a 68 GB checkpoint's *scale* bytes (about 2 GB) and the
//! kernels it would have to be expanded for are this tree's own.
//!
//! # What is checked
//!
//! Every geometric claim the two arrays make about each other: the fixed
//! array's shape against the matrix's, every base at most 240 (so `base + 15`
//! cannot wrap a byte - a wrapped scale is a finite number and the wrong
//! one), every exception inside the matrix, and the exception list strictly
//! ascending, which is what the writer promises and what makes a duplicated
//! or truncated list visible. Golden matrices decoded by the reference's own
//! kernels are replayed in `tests/csf_reference.rs`.

use crate::nvfp4::GROUP;
use crate::reader::{GgufError, GgufResult};

/// Rows that share one fixed record.
pub const ROWS_PER_RECORD: usize = 16;

/// Largest base the writer emits, so that `base + 15` stays inside a byte.
pub const MAX_BASE: u8 = 240;

/// Bits of an exception word that hold the position.
const POSITION_BITS: u32 = 24;

/// Mask selecting the position of an exception word.
const POSITION_MASK: u32 = (1 << POSITION_BITS) - 1;

/// Bytes of one exception word.
const EXCEPTION_BYTES: usize = 4;

/// The geometry of one CSF-compressed scale matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CsfLayout {
    /// Rows of the weight matrix, and of the scale matrix.
    pub rows: usize,
    /// Weights per row; the scale matrix has `cols / 16` columns.
    pub cols: usize,
}

impl CsfLayout {
    /// Scale bytes per row of the decoded matrix.
    #[must_use]
    pub const fn scale_cols(&self) -> usize {
        self.cols / GROUP
    }

    /// Bytes of the decoded scale matrix.
    #[must_use]
    pub const fn scale_bytes(&self) -> usize {
        self.rows * self.scale_cols()
    }

    /// Bytes of one 16-row fixed record.
    #[must_use]
    pub const fn record_bytes(&self) -> usize {
        ROWS_PER_RECORD * (1 + self.scale_cols() / 2)
    }

    /// Bytes of the whole fixed array.
    #[must_use]
    pub const fn fixed_bytes(&self) -> usize {
        (self.rows / ROWS_PER_RECORD) * self.record_bytes()
    }

    /// Checks that a matrix of this shape can be stored as CSF at all.
    fn check(&self, tensor: &str) -> GgufResult<()> {
        let c = self.scale_cols();
        if self.rows == 0
            || !self.rows.is_multiple_of(ROWS_PER_RECORD)
            || !self.cols.is_multiple_of(GROUP)
            || !c.is_multiple_of(2)
            || self.scale_bytes() > (1 << POSITION_BITS)
        {
            return Err(GgufError::LayoutMismatch {
                tensor: tensor.into(),
                want: Box::from(
                    "rows a multiple of 16, an even number of 16-wide groups, at most 2^24 scales",
                ),
                got: Box::from(format!("[{}, {}]", self.rows, self.cols)),
            });
        }
        Ok(())
    }
}

/// Rebuilds the plain `e4m3` scale matrix of one NVFP4 tensor from its CSF
/// arrays, into `out` (`rows x cols/16` bytes, row-major).
///
/// `exceptions` is the raw little-endian bytes of the `u32` array, exactly as
/// the shard stores them.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when the shape cannot be CSF, a base exceeds
/// [`MAX_BASE`], an exception points outside the matrix or the list is not
/// strictly ascending; [`GgufError::SizeMismatch`] when an array or `out` is
/// not the length the layout implies.
pub fn decode_csf(
    tensor: &str,
    layout: CsfLayout,
    fixed: &[u8],
    exceptions: &[u8],
    out: &mut [u8],
) -> GgufResult<()> {
    layout.check(tensor)?;
    let size = |want: usize, got: usize| GgufError::SizeMismatch {
        tensor: tensor.into(),
        want,
        got,
    };
    if fixed.len() != layout.fixed_bytes() {
        return Err(size(layout.fixed_bytes(), fixed.len()));
    }
    if out.len() != layout.scale_bytes() {
        return Err(size(layout.scale_bytes(), out.len()));
    }
    if !exceptions.len().is_multiple_of(EXCEPTION_BYTES) {
        return Err(size(EXCEPTION_BYTES, exceptions.len()));
    }

    let c = layout.scale_cols();
    let half = c / 2;
    for (record, rows_out) in fixed
        .chunks_exact(layout.record_bytes())
        .zip(out.chunks_exact_mut(ROWS_PER_RECORD * c))
    {
        let (bases, codes) = record.split_at(ROWS_PER_RECORD);
        for ((&base, row_codes), row_out) in bases
            .iter()
            .zip(codes.chunks_exact(half))
            .zip(rows_out.chunks_exact_mut(c))
        {
            if base > MAX_BASE {
                return Err(GgufError::LayoutMismatch {
                    tensor: tensor.into(),
                    want: Box::from("CSF row bases of at most 240"),
                    got: Box::from(format!("a base of {base}")),
                });
            }
            for (&pair, scales) in row_codes.iter().zip(row_out.chunks_exact_mut(2)) {
                scales[0] = base + (pair & 0x0F);
                scales[1] = base + (pair >> 4);
            }
        }
    }

    let mut previous: Option<usize> = None;
    for word in exceptions.chunks_exact(EXCEPTION_BYTES) {
        let word = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        let position = (word & POSITION_MASK) as usize;
        if previous.is_some_and(|p| position <= p) {
            return Err(GgufError::LayoutMismatch {
                tensor: tensor.into(),
                want: Box::from("CSF exceptions strictly ascending by position"),
                got: Box::from(format!("{position} after {}", previous.unwrap_or(0))),
            });
        }
        let slot = out
            .get_mut(position)
            .ok_or_else(|| GgufError::LayoutMismatch {
                tensor: tensor.into(),
                want: Box::from(format!(
                    "CSF exception positions below {}",
                    layout.scale_bytes()
                )),
                got: Box::from(format!("{position}")),
            })?;
        *slot = (word >> POSITION_BITS) as u8;
        previous = Some(position);
    }
    Ok(())
}

#[cfg(test)]
#[path = "csf_tests.rs"]
mod tests;
