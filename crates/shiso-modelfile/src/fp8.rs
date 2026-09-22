//! FP8 E4M3, both ways: the weight decode, and the activation encode.
//!
//! The format `modelopt` writes for an `FP8` target. One byte holds a sign, a
//! four-bit exponent biased by seven and three mantissa bits, with no
//! infinities and `0x7f`/`0xff` reserved for NaN - so it reaches `448` at the
//! top and `2^-9` at the bottom through subnormals.
//!
//! # This is the first W8A8 in this engine, and that is the whole difficulty
//!
//! Every other codec here is W8 or W4: the weights are quantized, the
//! activations are not, and a kernel decodes a weight and multiplies it by a
//! float. This checkpoint quantizes **both sides** of 130 projections, and
//! stores two scalars per target to say so - `weight_scale` and `input_scale`.
//! A product is then
//!
//! ```text
//! y = (w_q . x_q) * weight_scale * input_scale
//! ```
//!
//! and the two scales come out of the accumulation rather than going into it,
//! which is what makes an FP8 tensor-core path possible at all.
//!
//! **A host that decodes the weights and leaves the activations alone gets the
//! right answer and learns nothing.** That path is [`dequant_fp8`] and it is
//! what the oracles grade a device kernel against; it is not what the device
//! kernel does. The activation half is [`to_e4m3`], and it is here rather than
//! in a kernel because rounding a float to a three-bit mantissa is a decision
//! with a tie-break in it, and a tie-break that differs between the host oracle
//! and the device is a disagreement nobody can attribute.
//!
//! # `input_scale` absent is not `input_scale = 1`
//!
//! A target whose activations were not calibrated has no `input_scale` at all,
//! and that is a different kernel rather than a different number:
//! [`Fp8Layout::input_scale`] is an `Option` so a caller has to decide, and
//! [`crate::modelopt`] carries the same distinction up from the file.
//!
//! # The decode is exact, so its unit cases compare on bits
//!
//! Every value is a table lookup and one IEEE multiply, with nothing fused, so
//! there is no tolerance in this module and the cases below compare bit for
//! bit against `torch` 2.14's `float8_e4m3fn`. **E4M3 is one byte, so that
//! comparison is exhaustive**: all 256 codes, not a sample. The tolerance
//! question belongs where the arithmetic stops being exact, which is a device
//! kernel that accumulates in a tensor core.

use crate::reader::{GgufError, GgufResult};

/// Largest finite magnitude an `e4m3` byte can carry.
pub const E4M3_MAX: f32 = 448.0;

/// Subject named by the layout errors raised from this module.
const SUBJECT: &str = "fp8 tensor";

/// Decodes one `e4m3` byte.
///
/// One sign bit, a four-bit exponent biased by seven, three mantissa bits, no
/// infinities, and `0xff`/`0x7f` reserved for NaN. Subnormals are the
/// zero-exponent case and reach down to `2^-9`, which real scale arrays in
/// this checkpoint do use - the smallest byte seen in `experts.0.gate_proj`'s
/// scales is `0x01`, and treating subnormals as zero would silence whole
/// groups of a weight matrix rather than fail.
///
/// A NaN encoding decodes to NaN rather than being refused: an array is read
/// in bulk and one poisoned byte should reach the caller as a NaN it can see,
/// not as an error that names no row.
#[must_use]
pub fn e4m3(byte: u8) -> f32 {
    let sign = if byte & 0x80 == 0 { 1.0_f32 } else { -1.0_f32 };
    let exponent = i32::from((byte >> 3) & 0x0F);
    let mantissa = f32::from(byte & 0x07);
    if exponent == 0x0F {
        // The all-ones exponent is NaN in this format; there is no infinity.
        if byte & 0x07 == 0x07 {
            return f32::NAN;
        }
        return sign * (256.0 * (1.0 + mantissa / 8.0));
    }
    if exponent == 0 {
        // Subnormal: no implicit leading one, and the exponent is the
        // smallest normal one rather than the bias.
        return sign * (mantissa / 8.0) * exp2(-6);
    }
    sign * (1.0 + mantissa / 8.0) * exp2(exponent - 7)
}

/// Encodes one `f32` as an `e4m3` byte, round-to-nearest-even.
///
/// The inverse of [`e4m3`] and not its mirror image: a decode is a lookup and
/// an encode is a rounding, and the rounding has a tie-break. Ties go to even,
/// which is what IEEE says, what `torch`'s `.to(torch.float8_e4m3fn)` does and
/// what a GB10's own conversion instruction does - three implementations that
/// have to agree or a kernel and its oracle diverge on a bit nobody can
/// attribute to either.
///
/// Saturating rather than overflowing to NaN: a magnitude past [`E4M3_MAX`]
/// comes back as [`E4M3_MAX`], which is what `modelopt` calibrates its
/// `input_scale` to make unnecessary and what every reference does when the
/// calibration was slightly optimistic. An input that is already NaN stays NaN.
#[must_use]
pub fn to_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80_u8 } else { 0 };
    let magnitude = value.abs();
    if magnitude >= E4M3_MAX {
        // Saturate. The next representable step above 448 is the NaN encoding,
        // so "round up" at the top of the range would produce a NaN from a
        // finite input.
        return sign | 0x7e;
    }
    // The search is over 127 finite magnitudes, which is small enough that the
    // obvious thing is also the fast thing, and - unlike a bit-twiddling
    // encoder - it cannot be off by one at a subnormal boundary. The two
    // candidates either side of the value are found, and the tie between them
    // goes to the even code.
    let mut code = 0_u8;
    for candidate in 1..=0x7e_u8 {
        if e4m3(candidate) <= magnitude {
            code = candidate;
        } else {
            break;
        }
    }
    let lower = e4m3(code);
    let upper = e4m3(code + 1);
    let below = magnitude - lower;
    let above = upper - magnitude;
    // Ties to even. The equality is exact on purpose and a tolerance here
    // would be a bug rather than a safeguard: `above` and `below` are
    // differences between an `f32` and two of 127 exactly-representable
    // magnitudes, so "these are equal" is the literal question - it is what an
    // exact midpoint IS - and answering it approximately would move the
    // tie-break off whichever side of the margin an input happened to land.
    #[expect(
        clippy::float_cmp,
        reason = "an exact midpoint is exactly what this compares for"
    )]
    let tie = above == below;
    let rounded = if above < below || (tie && !code.is_multiple_of(2)) {
        code + 1
    } else {
        code
    };
    sign | rounded
}

/// How an FP8 matrix is laid out and what its two scales are.
///
/// The scales travel with the geometry because neither is derivable from the
/// bytes: a decode that forgot `weight_scale` produces a matrix of plausible
/// magnitude, and one that forgot `input_scale` produces a product that is
/// wrong by a factor the next normalization very nearly removes.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct Fp8Layout {
    /// Rows of the matrix, the slowest-varying axis.
    pub out_rows: usize,
    /// Values per row, the fastest-varying axis.
    pub in_dim: usize,
    /// The one `f32` every weight of this tensor is multiplied by.
    pub weight_scale: f32,
    /// The scale the activations were calibrated at, when the export states
    /// one. `None` is a target whose activations were NOT quantized, which is
    /// a different launch and not a scale of one.
    pub input_scale: Option<f32>,
}

impl Fp8Layout {
    /// Describes a matrix whose weights are quantized and whose activations
    /// are not.
    #[must_use]
    pub const fn weights_only(out_rows: usize, in_dim: usize, weight_scale: f32) -> Self {
        Self {
            out_rows,
            in_dim,
            weight_scale,
            input_scale: None,
        }
    }

    /// The same, for a target whose activations were calibrated too.
    #[must_use]
    pub const fn with_activations(self, input_scale: f32) -> Self {
        Self {
            input_scale: Some(input_scale),
            ..self
        }
    }

    /// Values the matrix holds, `None` when the product overflows.
    #[must_use]
    pub const fn elements(&self) -> Option<usize> {
        self.out_rows.checked_mul(self.in_dim)
    }

    /// The factor a product of quantized weights and quantized activations is
    /// multiplied by on the way out.
    ///
    /// `None` when the activations were not quantized, because then there is no
    /// such product: the weights are decoded and multiplied by floats, and the
    /// weight scale has already been folded in by [`dequant_fp8`].
    #[must_use]
    pub fn product_scale(&self) -> Option<f32> {
        self.input_scale.map(|input| self.weight_scale * input)
    }
}

/// Decodes an FP8 matrix into `out`, row-major `[out_rows, in_dim]`.
///
/// `out` is cleared first and holds exactly `out_rows * in_dim` values when
/// this returns, so a caller can hand the same buffer to every tensor of a
/// checkpoint without reallocating per tensor.
///
/// `input_scale` is **not** applied: this decodes the weights, and an
/// activation scale belongs to the other operand. A caller that folded it in
/// here would be scaling the weights by a number calibrated for the
/// activations.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when `bytes` is not exactly one byte per
/// value, checked before a byte is read, so a short array is a load error
/// rather than a matrix that is correct down to the row it ran out at.
pub fn dequant_fp8(layout: Fp8Layout, bytes: &[u8], out: &mut Vec<f32>) -> GgufResult<()> {
    let elements = layout.elements().ok_or_else(|| {
        layout_error(
            "a matrix that fits in memory",
            "an overflowing element count",
        )
    })?;
    if bytes.len() != elements {
        return Err(layout_error(
            format!("{elements} bytes"),
            format!("{} bytes", bytes.len()),
        ));
    }
    out.clear();
    out.reserve(elements);
    out.extend(bytes.iter().map(|byte| e4m3(*byte) * layout.weight_scale));
    Ok(())
}

/// Quantizes activations into `out`, one byte per value.
///
/// The other half of W8A8, and the half a host oracle has to do the same way a
/// kernel does or the two disagree on the last bit of every product. `out` is
/// cleared first.
///
/// # Errors
///
/// [`GgufError::LayoutMismatch`] when `scale` is not a positive finite number.
/// A zero or negative scale is not a rounding problem: it maps every activation
/// onto one code, which decodes to a plausible row of zeros or a sign-flipped
/// one, and neither fails anywhere downstream.
pub fn quantize_activations(values: &[f32], scale: f32, out: &mut Vec<u8>) -> GgufResult<()> {
    if !(scale.is_finite() && scale > 0.0) {
        return Err(layout_error(
            "a positive finite activation scale",
            format!("{scale}"),
        ));
    }
    out.clear();
    out.reserve(values.len());
    out.extend(values.iter().map(|value| to_e4m3(value / scale)));
    Ok(())
}

/// Two to a small signed power, exactly.
///
/// `f32::powi` would do it and is not exact for every input on every target;
/// the exponents here span `-9..=8`, so building the bit pattern is both exact
/// and obviously so.
fn exp2(power: i32) -> f32 {
    // The `e4m3` exponents this is called with are inside `-6..=8`, which is
    // far inside `f32`'s normal range, so the bias cannot leave it.
    let biased = u32::try_from(power + 127).unwrap_or(127);
    f32::from_bits(biased << 23)
}

/// One layout rejection, named for this format.
fn layout_error(want: impl Into<String>, got: impl Into<String>) -> GgufError {
    GgufError::LayoutMismatch {
        tensor: Box::from(SUBJECT),
        want: want.into().into_boxed_str(),
        got: got.into().into_boxed_str(),
    }
}

#[cfg(test)]
#[path = "fp8_tests.rs"]
mod tests;
