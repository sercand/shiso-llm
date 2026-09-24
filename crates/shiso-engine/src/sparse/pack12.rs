//! Half-precision weights repacked to twelve bits a weight, losslessly.
//!
//! The checkpoint stores its unquantized projections - the hyper-connection
//! mixes above all, 1.3 GB of every decode step - as halves whose values are
//! brain floats: the low three of the ten mantissa bits are zero, and the
//! exponents a row uses span at most sixteen values. So a weight is carried as
//! a byte of sign and seven mantissa bits and a four-bit code for its exponent
//! against a per-row base, and the ~0.1% that fit neither - subnormals, an
//! exponent more than fourteen below its row's largest, a set low mantissa
//! bit, an infinity - are kept exactly in a per-row list beside the planes,
//! with code zero (value zero) left in their place. Every weight decodes to
//! the half the checkpoint stored; the projection streams three quarters of
//! the bytes.
//!
//! The layout is [`DType::F16Pack12`]'s, and the kernels that read it
//! (`q4e_matmul_p12_rows_kernel`, `q4e_p12_expand_kernel`) restate it:
//!
//! ```text
//! mant [out][in]    u8   sign << 7 | mantissa >> 3
//! expo [out][in/2]  u8   code of column 2j in the low nibble, 2j + 1 high
//! base [out]        i8   exponent = base + code, for a code of 1 to 15
//! ptr  [out + 1]    u32  row r's exceptions are ptr[r]..ptr[r + 1]
//! col  [n]          u32  each exception's column
//! val  [n]          u16  its half bits
//! ```
//!
//! with `ptr` and `col` starting on 16-byte boundaries.
//!
//! [`DType::F16Pack12`]: shiso_core::DType::F16Pack12

/// Columns a packed row has to be a multiple of: one lane's sixteen weights.
pub const PACK12_COLUMN_QUANTUM: u32 = 16;

/// The exponent field of a half.
const fn exponent(bits: u16) -> i32 {
    ((bits >> 10) & 0x1f) as i32
}

/// Rounds a byte offset up to the 16-byte boundary a region starts on.
const fn align16(at: usize) -> usize {
    (at + 15) & !15
}

/// Packs a row-major `out_dim x in_dim` half-precision matrix, given as its
/// bit patterns, or `None` when the contraction is not a whole number of
/// [`PACK12_COLUMN_QUANTUM`] columns (or the lengths disagree).
#[must_use]
pub fn pack(bits: &[u16], in_dim: u32, out_dim: u32) -> Option<Vec<u8>> {
    let (cols, rows) = (in_dim as usize, out_dim as usize);
    if cols == 0 || rows == 0 || !cols.is_multiple_of(PACK12_COLUMN_QUANTUM as usize) {
        return None;
    }
    if bits.len() != cols.checked_mul(rows)? {
        return None;
    }
    let cells = cols * rows;
    let mut mant = vec![0u8; cells];
    let mut expo = vec![0u8; cells / 2];
    let mut base = vec![0u8; rows];
    let mut ptr = Vec::with_capacity(rows + 1);
    let mut exc_col = Vec::new();
    let mut exc_val = Vec::new();
    ptr.push(0u32);
    for r in 0..rows {
        let row = &bits[r * cols..(r + 1) * cols];
        // The row's largest finite exponent sets the window; a row of zeros
        // has none and any base will do.
        let top = row
            .iter()
            .filter(|&&h| h & 0x7fff != 0)
            .map(|&h| exponent(h))
            .filter(|&e| e < 31)
            .max()
            .unwrap_or(15);
        let b = top - 15;
        // `top` is a half's exponent field, so the base is in -15..=15.
        base[r] = i8::try_from(b).ok()?.cast_unsigned();
        for (c, &h) in row.iter().enumerate() {
            let at = r * cols + c;
            if h == 0 {
                continue;
            }
            let e = exponent(h);
            let code = e - b;
            let packable = h.trailing_zeros() >= 3 && (1..=30).contains(&e);
            if let (true, Ok(code @ 1..=15)) = (packable, u8::try_from(code)) {
                let [low, _] = (((h >> 8) & 0x80) | ((h >> 3) & 0x7f)).to_le_bytes();
                mant[at] = low;
                expo[at / 2] |= code << ((at & 1) * 4);
            } else {
                exc_col.push(u32::try_from(c).ok()?);
                exc_val.push(h);
            }
        }
        ptr.push(u32::try_from(exc_col.len()).ok()?);
    }
    let mut out =
        Vec::with_capacity(cells + cells / 2 + rows + 32 + 4 * (rows + 1) + 6 * exc_col.len());
    out.extend_from_slice(&mant);
    out.extend_from_slice(&expo);
    out.extend_from_slice(&base);
    out.resize(align16(out.len()), 0);
    for p in &ptr {
        out.extend_from_slice(&p.to_le_bytes());
    }
    out.resize(align16(out.len()), 0);
    for c in &exc_col {
        out.extend_from_slice(&c.to_le_bytes());
    }
    for v in &exc_val {
        out.extend_from_slice(&v.to_le_bytes());
    }
    Some(out)
}

/// Decodes a packed matrix back to its half bit patterns: the kernels'
/// arithmetic, for a test to hold the packer to.
#[must_use]
pub fn unpack(packed: &[u8], in_dim: u32, out_dim: u32) -> Option<Vec<u16>> {
    let (cols, rows) = (in_dim as usize, out_dim as usize);
    let cells = cols.checked_mul(rows)?;
    let at_base = cells + cells / 2;
    let at_ptr = align16(at_base + rows);
    let at_col = align16(at_ptr + 4 * (rows + 1));
    let word = |at: usize| -> Option<u32> {
        Some(u32::from_le_bytes(packed.get(at..at + 4)?.try_into().ok()?))
    };
    let n = word(at_ptr + 4 * rows)? as usize;
    let at_val = at_col + 4 * n;
    let mut out = vec![0u16; cells];
    for r in 0..rows {
        let b = i32::from(packed.get(at_base + r)?.cast_signed());
        for c in 0..cols {
            let at = r * cols + c;
            let byte = u16::from(*packed.get(at)?);
            let code = i32::from((packed.get(cells + at / 2)? >> ((at & 1) * 4)) & 0xf);
            if code != 0 {
                let e = u16::try_from(b + code).ok()?;
                out[at] = ((byte & 0x80) << 8) | (e << 10) | ((byte & 0x7f) << 3);
            }
        }
        for k in word(at_ptr + 4 * r)? as usize..word(at_ptr + 4 * (r + 1))? as usize {
            let c = word(at_col + 4 * k)? as usize;
            let v = u16::from_le_bytes(
                packed
                    .get(at_val + 2 * k..at_val + 2 * k + 2)?
                    .try_into()
                    .ok()?,
            );
            *out.get_mut(r * cols + c)? = v;
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brain-float-valued halves spanning a row's exponent window, and - at
    /// about the checkpoint's rate - every kind of value that has to fall
    /// through to the exception list.
    fn sample(rows: usize, cols: usize) -> Vec<u16> {
        let mut state = 0x2545_f491_u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        (0..rows * cols)
            .map(|i| match i % 997 {
                0 => 0,      // zero
                1 => 0x8000, // negative zero
                2 => 0x0003, // subnormal
                3 => 0x3c01, // a low mantissa bit
                4 => 0x7c00, // infinity
                5 => 0x0400, // far below the row
                _ => {
                    let e = 1 + (next() % 15) as u16; // a row's window
                    let sign = (next() & 1) as u16;
                    let m = (next() & 0x7f) as u16;
                    (sign << 15) | ((e + 8) << 10) | (m << 3)
                }
            })
            .collect()
    }

    #[test]
    fn every_weight_decodes_to_the_half_it_was() {
        let (rows, cols) = (37_u32, 320_u32);
        let bits = sample(rows as usize, cols as usize);
        let packed = pack(&bits, cols, rows).expect("the shape packs");
        let back = unpack(&packed, cols, rows).expect("the layout reads back");
        assert_eq!(back, bits);
    }

    #[test]
    fn the_planes_are_three_quarters_of_the_halves() {
        let (rows, cols) = (64_u32, 1024_u32);
        let bits = sample(rows as usize, cols as usize);
        let packed = pack(&bits, cols, rows).expect("the shape packs");
        let halves = bits.len() * 2;
        assert!(
            packed.len() < halves * 4 / 5,
            "{} of {halves}",
            packed.len()
        );
    }

    #[test]
    fn a_contraction_off_the_lane_quantum_is_left_unpacked() {
        assert!(pack(&[0; 24], 24, 1).is_none());
        assert!(pack(&[0; 32], 16, 1).is_none(), "lengths that disagree");
    }
}
