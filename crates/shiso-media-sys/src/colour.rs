//! YUV to RGB, in safe code.
//!
//! Every decoder here hands back planar or semi-planar YUV (`VideoToolbox` is asked
//! for BGRA and converts itself; NVDEC and the AV1 decoder are not), and the
//! conversion has to honour the stream's matrix and range or the colours are
//! quietly wrong: BT.709 content read as BT.601 is a plausible picture with the
//! wrong greens. One implementation serves every decoder so there is one place
//! to get that right, and it is plain integer arithmetic so it is deterministic.
//!
//! Chroma is upsampled by replication (nearest), which is what a 4:2:0 picture's
//! own sample siting asks for at the pixel level and costs nothing; the
//! difference from a filtered upsample is below the PSNR floor the device tests
//! hold the decoders to.

use crate::HwError;

/// The colour matrix a stream's luma and chroma were derived with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum YuvMatrix {
    /// BT.601, standard definition.
    Bt601,
    /// BT.709, high definition.
    Bt709,
    /// BT.2020 non-constant luminance.
    Bt2020,
}

impl YuvMatrix {
    /// The `(kr, kb)` luma coefficients of the matrix.
    fn coefficients(self) -> (f64, f64) {
        match self {
            Self::Bt601 => (0.299, 0.114),
            Self::Bt709 => (0.2126, 0.0722),
            Self::Bt2020 => (0.2627, 0.0593),
        }
    }

    /// Picks a matrix from the stream's declared matrix-coefficients code
    /// (ITU-T H.273), treating "unspecified" the way players do: by picture
    /// height, since high-definition content is 709 and the rest is 601.
    #[must_use]
    pub fn from_code(code: u8, height: u32) -> Self {
        match code {
            1 => Self::Bt709,
            5 | 6 => Self::Bt601,
            9 | 10 => Self::Bt2020,
            _ if height >= 720 => Self::Bt709,
            _ => Self::Bt601,
        }
    }
}

/// Whether luma spans the full byte range or the broadcast "limited" range.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum YuvRange {
    /// Luma 16..=235 and chroma 16..=240 at 8 bits.
    Limited,
    /// Luma and chroma 0..=255 at 8 bits.
    Full,
}

/// How a frame's samples are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum YuvLayout {
    /// Three planes: Y, then U, then V, the chroma planes half size each way.
    I420,
    /// Two planes: Y, then interleaved U and V at half size each way.
    Nv12,
}

/// A borrowed 4:2:0 frame with 8-bit samples.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Yuv420<'a> {
    /// How the planes are laid out.
    pub layout: YuvLayout,
    /// The luma plane.
    pub y: &'a [u8],
    /// Bytes from one luma row to the next.
    pub y_stride: usize,
    /// For [`YuvLayout::I420`] the U plane; for [`YuvLayout::Nv12`] the
    /// interleaved UV plane.
    pub u: &'a [u8],
    /// The V plane for [`YuvLayout::I420`]; empty for [`YuvLayout::Nv12`].
    pub v: &'a [u8],
    /// Bytes from one chroma row to the next (of the U plane, or of the
    /// interleaved plane).
    pub uv_stride: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

/// Fixed-point scale of the conversion coefficients.
const SHIFT: u32 = 16;

/// The conversion constants for one matrix and range, as fixed-point integers.
#[derive(Debug, Clone, Copy)]
struct Coeffs {
    y_scale: i32,
    y_offset: i32,
    r_v: i32,
    g_u: i32,
    g_v: i32,
    b_u: i32,
    chroma_offset: i32,
}

impl Coeffs {
    fn new(matrix: YuvMatrix, range: YuvRange, bit_depth: u32) -> Self {
        let (kr, kb) = matrix.coefficients();
        let kg = 1.0 - kr - kb;
        let one = f64::from(1u32 << SHIFT);
        let top = f64::from((1u32 << bit_depth) - 1);
        let unit = f64::from(1u32 << (bit_depth - 8));
        // Full range maps 0..=top straight; limited range stretches the nominal
        // luma (219 steps) and chroma (224 steps) back onto 0..=top.
        let (y_scale, c_scale, y_offset) = match range {
            YuvRange::Full => (1.0, 1.0, 0.0),
            YuvRange::Limited => (top / (219.0 * unit), top / (224.0 * unit), 16.0 * unit),
        };
        let chroma_offset = 128.0 * unit;
        let to_fixed = |v: f64| whole(v * one);
        Self {
            y_scale: to_fixed(y_scale),
            y_offset: whole(y_offset),
            r_v: to_fixed(2.0 * (1.0 - kr) * c_scale),
            g_u: to_fixed(2.0 * kb * (1.0 - kb) / kg * c_scale),
            g_v: to_fixed(2.0 * kr * (1.0 - kr) / kg * c_scale),
            b_u: to_fixed(2.0 * (1.0 - kb) * c_scale),
            chroma_offset: whole(chroma_offset),
        }
    }

    /// One pixel's `[red, green, blue]` from its `[y, u, v]` samples at this
    /// coefficient set's depth, whose largest sample value is `max`.
    fn rgb(&self, samples: [i32; 3], max: i32) -> [u8; 3] {
        let [luma_sample, blue_diff, red_diff] = samples;
        let luma = (luma_sample - self.y_offset) * self.y_scale;
        let cb = blue_diff - self.chroma_offset;
        let cr = red_diff - self.chroma_offset;
        let red = luma + self.r_v * cr;
        let green = luma - self.g_u * cb - self.g_v * cr;
        let blue = luma + self.b_u * cb;
        [to_byte(red, max), to_byte(green, max), to_byte(blue, max)]
    }
}

/// Rounds to the nearest integer. The values passed are coefficients (at most a
/// few times `2^SHIFT`) and sample offsets (at most `16 * 2^8`), so they fit an
/// `i32` with room to spare.
fn whole(v: f64) -> i32 {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "the inputs are bounded by the coefficient and offset ranges above"
    )]
    let rounded = v.round() as i32;
    rounded
}

/// Rounds a fixed-point value at `max` sample range down to a byte, clamped.
fn to_byte(fixed: i32, max: i32) -> u8 {
    let rounded = (i64::from(fixed) + (1 << (SHIFT - 1))) >> SHIFT;
    // Rescale a deeper sample to 8 bits by the range ratio, rounding.
    let scaled = (rounded * 255 + i64::from(max) / 2) / i64::from(max);
    u8::try_from(scaled.clamp(0, 255)).unwrap_or(u8::MAX)
}

/// Converts an 8-bit 4:2:0 frame to packed RGB8.
///
/// # Errors
///
/// [`HwError::Failed`] when a plane is too short for the declared size and
/// stride, which is a decoder handing back a buffer it did not fill. Nothing
/// here indexes past a plane.
pub fn yuv420_to_rgb8(
    frame: &Yuv420<'_>,
    matrix: YuvMatrix,
    range: YuvRange,
) -> Result<Vec<u8>, HwError> {
    if frame.width == 0 || frame.height == 0 {
        return Err(HwError::Failed {
            detail: "a frame with no pixels".into(),
        });
    }
    let coeffs = Coeffs::new(matrix, range, 8);
    let mut out = Vec::with_capacity(frame.width * frame.height * 3);
    for row in 0..frame.height {
        let y_row = frame
            .y
            .get(row.saturating_mul(frame.y_stride)..)
            .ok_or_else(|| short("luma"))?;
        let chroma_row = row / 2;
        for col in 0..frame.width {
            let y = i32::from(*y_row.get(col).ok_or_else(|| short("luma"))?);
            let (u, v) = match frame.layout {
                YuvLayout::I420 => {
                    let at = chroma_row
                        .saturating_mul(frame.uv_stride)
                        .saturating_add(col / 2);
                    (
                        *frame.u.get(at).ok_or_else(|| short("U"))?,
                        *frame.v.get(at).ok_or_else(|| short("V"))?,
                    )
                }
                YuvLayout::Nv12 => {
                    let at = chroma_row
                        .saturating_mul(frame.uv_stride)
                        .saturating_add((col / 2) * 2);
                    (
                        *frame.u.get(at).ok_or_else(|| short("UV"))?,
                        *frame.u.get(at + 1).ok_or_else(|| short("UV"))?,
                    )
                }
            };
            out.extend_from_slice(&coeffs.rgb([y, i32::from(u), i32::from(v)], 255));
        }
    }
    Ok(out)
}

/// A borrowed 4:2:0 I420 frame with 16-bit storage and `bit_depth` significant
/// bits (10 for the common deep-colour streams).
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct Yuv420Deep<'a> {
    /// The luma plane, one `u16` per sample.
    pub y: &'a [u16],
    /// Samples (not bytes) from one luma row to the next.
    pub y_stride: usize,
    /// The U plane.
    pub u: &'a [u16],
    /// The V plane.
    pub v: &'a [u16],
    /// Samples from one chroma row to the next.
    pub uv_stride: usize,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Significant bits per sample, 9 to 16.
    pub bit_depth: u32,
}

/// Converts a deep-colour I420 frame to packed RGB8.
///
/// # Errors
///
/// [`HwError::Failed`] for a bit depth outside 9..=16 or a plane too short for
/// its declared size.
pub fn yuv420_deep_to_rgb8(
    frame: &Yuv420Deep<'_>,
    matrix: YuvMatrix,
    range: YuvRange,
) -> Result<Vec<u8>, HwError> {
    if !(9..=16).contains(&frame.bit_depth) {
        return Err(HwError::Failed {
            detail: format!("unsupported sample depth {}", frame.bit_depth),
        });
    }
    let coeffs = Coeffs::new(matrix, range, frame.bit_depth);
    let max = i32::try_from((1u32 << frame.bit_depth) - 1).unwrap_or(i32::MAX);
    let mut out = Vec::with_capacity(frame.width * frame.height * 3);
    for row in 0..frame.height {
        let y_row = frame
            .y
            .get(row.saturating_mul(frame.y_stride)..)
            .ok_or_else(|| short("luma"))?;
        let chroma_row = row / 2;
        for col in 0..frame.width {
            let y = i32::from(*y_row.get(col).ok_or_else(|| short("luma"))?);
            let at = chroma_row
                .saturating_mul(frame.uv_stride)
                .saturating_add(col / 2);
            let u = i32::from(*frame.u.get(at).ok_or_else(|| short("U"))?);
            let v = i32::from(*frame.v.get(at).ok_or_else(|| short("V"))?);
            out.extend_from_slice(&coeffs.rgb([y, u, v], max));
        }
    }
    Ok(out)
}

fn short(plane: &str) -> HwError {
    HwError::Failed {
        detail: format!("the {plane} plane is shorter than the frame's size and stride say"),
    }
}

#[cfg(test)]
#[path = "colour_tests.rs"]
mod tests;
