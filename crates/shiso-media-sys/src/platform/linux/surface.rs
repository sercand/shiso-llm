//! The decisions around the driver calls that need no driver: what a `CUresult`
//! means to the decode chain, whether a capability report admits a stream, how
//! big the output surface is, and how a surface copied to host memory becomes
//! RGB.

use std::ffi::c_int;

use super::config::StreamFormat;
use super::ffi::{
    CHROMA_420, CuResult, CuVideoFormat, CuvidDecodeCaps, SURFACE_NV12, SURFACE_P016,
};
use crate::colour::{
    Yuv420, Yuv420Deep, YuvLayout, YuvMatrix, YuvRange, yuv420_deep_to_rgb8, yuv420_to_rgb8,
};
use crate::{HwCodec, HwError};

/// `CUresult` codes that mean "there is no usable driver or device here" rather
/// than "this call failed": the decode chain moves on instead of refusing the
/// request. Values from the CUDA driver header.
const UNAVAILABLE_CODES: [(c_int, &str); 9] = [
    (3, "CUDA_ERROR_NOT_INITIALIZED"),
    (4, "CUDA_ERROR_DEINITIALIZED"),
    (34, "CUDA_ERROR_STUB_LIBRARY"),
    (46, "CUDA_ERROR_DEVICE_UNAVAILABLE"),
    (100, "CUDA_ERROR_NO_DEVICE"),
    (101, "CUDA_ERROR_INVALID_DEVICE"),
    (102, "CUDA_ERROR_DEVICE_NOT_LICENSED"),
    (803, "CUDA_ERROR_SYSTEM_DRIVER_MISMATCH"),
    (804, "CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE"),
];

/// `CUDA_ERROR_NOT_SUPPORTED`.
const NOT_SUPPORTED: c_int = 801;

/// Other codes worth a name in a message.
const NAMED_CODES: [(c_int, &str); 5] = [
    (1, "CUDA_ERROR_INVALID_VALUE"),
    (2, "CUDA_ERROR_OUT_OF_MEMORY"),
    (201, "CUDA_ERROR_INVALID_CONTEXT"),
    (205, "CUDA_ERROR_MAP_FAILED"),
    (999, "CUDA_ERROR_UNKNOWN"),
];

/// `code` as `NAME (n)`, or just the number for a code without a name here.
fn describe(code: CuResult) -> String {
    UNAVAILABLE_CODES
        .iter()
        .chain(NAMED_CODES.iter())
        .chain(std::iter::once(&(
            NOT_SUPPORTED,
            "CUDA_ERROR_NOT_SUPPORTED",
        )))
        .find(|(n, _)| *n == code.0)
        .map_or_else(
            || format!("error {}", code.0),
            |(n, name)| format!("{name} ({n})"),
        )
}

/// Turns a failed driver call into the decode chain's error.
///
/// A missing or unusable device is [`HwError::Unavailable`]; `NOT_SUPPORTED`
/// from a call made for `codec` is [`HwError::Unsupported`]; everything else is
/// [`HwError::Failed`].
pub(super) fn cu_error(call: &str, code: CuResult, codec: Option<HwCodec>) -> HwError {
    let what = describe(code);
    if UNAVAILABLE_CODES.iter().any(|(n, _)| *n == code.0) {
        return HwError::Unavailable {
            reason: format!("{call} returned {what}"),
        };
    }
    match codec {
        Some(codec) if code.0 == NOT_SUPPORTED => HwError::Unsupported {
            codec,
            reason: format!("{call} returned {what}"),
        },
        _ => HwError::Failed {
            detail: format!("{call} returned {what}"),
        },
    }
}

/// `Ok` for `CUDA_SUCCESS`, the mapped error otherwise.
pub(super) fn check(call: &str, code: CuResult, codec: Option<HwCodec>) -> Result<(), HwError> {
    if code == CuResult::SUCCESS {
        Ok(())
    } else {
        Err(cu_error(call, code, codec))
    }
}

/// The output surface format for a bit depth.
pub(super) fn surface_format(bit_depth: u32) -> c_int {
    if bit_depth > 8 {
        SURFACE_P016
    } else {
        SURFACE_NV12
    }
}

/// Whether `caps` (queried for `format`) admits a `width` x `height` stream, and
/// why not.
///
/// # Errors
///
/// [`HwError::Unsupported`] naming the first limit the stream breaks.
pub(super) fn check_caps(
    codec: HwCodec,
    caps: &CuvidDecodeCaps,
    format: StreamFormat,
    width: u32,
    height: u32,
) -> Result<(), HwError> {
    require_420(codec, format.chroma)?;
    let unsupported = |reason: String| HwError::Unsupported { codec, reason };
    if caps.is_supported == 0 {
        return Err(unsupported(format!(
            "this GPU's NVDEC does not decode {}-bit 4:2:0",
            format.bit_depth
        )));
    }
    if width > caps.max_width || height > caps.max_height {
        return Err(unsupported(format!(
            "{width}x{height} is larger than this GPU's {}x{} limit",
            caps.max_width, caps.max_height
        )));
    }
    if width < u32::from(caps.min_width) || height < u32::from(caps.min_height) {
        return Err(unsupported(format!(
            "{width}x{height} is smaller than this GPU's {}x{} minimum",
            caps.min_width, caps.min_height
        )));
    }
    let macroblocks = u64::from(width.div_ceil(16)) * u64::from(height.div_ceil(16));
    if macroblocks > u64::from(caps.max_mb_count) {
        return Err(unsupported(format!(
            "{width}x{height} is {macroblocks} macroblocks, over this GPU's {}",
            caps.max_mb_count
        )));
    }
    let surface = surface_format(format.bit_depth);
    if (u32::from(caps.output_format_mask) >> surface) & 1 == 0 {
        return Err(unsupported(format!(
            "this GPU cannot output {}-bit 4:2:0 as {}",
            format.bit_depth,
            if surface == SURFACE_NV12 {
                "NV12"
            } else {
                "P016"
            }
        )));
    }
    Ok(())
}

/// Refuses any chroma format but 4:2:0: the colour module converts only that,
/// and the decoder would otherwise hand back a surface layout nothing here reads.
///
/// # Errors
///
/// [`HwError::Unsupported`] for monochrome, 4:2:2 and 4:4:4.
pub(super) fn require_420(codec: HwCodec, chroma: c_int) -> Result<(), HwError> {
    if chroma == CHROMA_420 {
        return Ok(());
    }
    Err(HwError::Unsupported {
        codec,
        reason: format!(
            "chroma format {} (only 4:2:0 is converted)",
            chroma_name(chroma)
        ),
    })
}

fn chroma_name(chroma: c_int) -> String {
    match chroma {
        0 => "monochrome".into(),
        1 => "4:2:0".into(),
        2 => "4:2:2".into(),
        3 => "4:4:4".into(),
        other => format!("code {other}"),
    }
}

/// The geometry and colour of one decoder configuration, derived from the
/// parser's sequence report.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct SurfaceGeometry {
    /// Coded size, what the decoder is created with.
    pub(super) coded_width: u32,
    /// Coded height.
    pub(super) coded_height: u32,
    /// The display rectangle's origin within the coded picture.
    pub(super) left: u32,
    /// The display rectangle's top.
    pub(super) top: u32,
    /// Displayed size: what a returned frame measures.
    pub(super) width: u32,
    /// Displayed height.
    pub(super) height: u32,
    /// Output surface size: the display size rounded up to even, as the decoder
    /// wants its post-processed target size aligned to 2.
    pub(super) target_width: u32,
    /// Output surface height, even.
    pub(super) target_height: u32,
    /// Bits per sample.
    pub(super) bit_depth: u32,
    /// The stream's colour matrix.
    pub(super) matrix: YuvMatrix,
    /// The stream's sample range.
    pub(super) range: YuvRange,
}

impl SurfaceGeometry {
    /// Reads and validates the parser's sequence report.
    ///
    /// # Errors
    ///
    /// [`HwError::Unsupported`] for a chroma format other than 4:2:0 or luma and
    /// chroma of different depths; [`HwError::Failed`] for a display rectangle
    /// outside the coded picture, which is a corrupt sequence header.
    pub(super) fn from_format(codec: HwCodec, f: &CuVideoFormat) -> Result<Self, HwError> {
        require_420(codec, f.chroma_format)?;
        if f.bit_depth_luma_minus8 != f.bit_depth_chroma_minus8 {
            return Err(HwError::Unsupported {
                codec,
                reason: format!(
                    "luma at {} bits and chroma at {}",
                    u32::from(f.bit_depth_luma_minus8) + 8,
                    u32::from(f.bit_depth_chroma_minus8) + 8
                ),
            });
        }
        let area = f.display_area;
        let bad = || HwError::Failed {
            detail: format!(
                "the sequence's display area {},{} to {},{} is not inside its {}x{} coded size",
                area.left, area.top, area.right, area.bottom, f.coded_width, f.coded_height
            ),
        };
        let coord = |v: c_int| u32::try_from(v).map_err(|_| bad());
        let (left, top, right, bottom) = (
            coord(area.left)?,
            coord(area.top)?,
            coord(area.right)?,
            coord(area.bottom)?,
        );
        if left >= right || top >= bottom || right > f.coded_width || bottom > f.coded_height {
            return Err(bad());
        }
        let (width, height) = (right - left, bottom - top);
        // Picture sizes NVDEC decodes are at most 8192 on a side, so rounding up
        // cannot overflow a u32 that passed the check above.
        let even = |v: u32| v.saturating_add(v & 1);
        let signal = f.video_signal_description;
        Ok(Self {
            coded_width: f.coded_width,
            coded_height: f.coded_height,
            left,
            top,
            width,
            height,
            target_width: even(width),
            target_height: even(height),
            bit_depth: u32::from(f.bit_depth_luma_minus8) + 8,
            matrix: YuvMatrix::from_code(signal.matrix_coefficients, height),
            range: if signal.full_range() {
                YuvRange::Full
            } else {
                YuvRange::Limited
            },
        })
    }

    /// Bytes per sample on the output surface.
    pub(super) fn sample_bytes(&self) -> usize {
        if self.bit_depth > 8 { 2 } else { 1 }
    }

    /// Bytes in one packed row of the host copy: the target width in samples.
    pub(super) fn row_bytes(&self) -> usize {
        to_usize(self.target_width) * self.sample_bytes()
    }

    /// Rows in the host copy: the luma rows, then half as many chroma rows,
    /// contiguous because the surface's chroma plane starts right after its
    /// `target_height` luma rows.
    pub(super) fn rows(&self) -> usize {
        to_usize(self.target_height) + to_usize(self.target_height) / 2
    }
}

/// A `u32` picture dimension as a `usize`; every target this crate builds for
/// has a pointer at least 32 bits wide.
fn to_usize(v: u32) -> usize {
    usize::try_from(v).unwrap_or(usize::MAX)
}

/// One picture copied off the device: `geometry.rows()` packed rows of
/// `geometry.row_bytes()` bytes.
#[derive(Clone)]
pub(super) struct HostSurface {
    /// The picture's presentation timestamp.
    pub(super) pts: i64,
    /// The surface's layout and colour.
    pub(super) geometry: SurfaceGeometry,
    /// Luma rows then interleaved chroma rows.
    pub(super) bytes: Vec<u8>,
}

impl std::fmt::Debug for HostSurface {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HostSurface")
            .field("pts", &self.pts)
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl HostSurface {
    /// Converts the picture's display area to packed RGB8.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when the buffer is shorter than its geometry says.
    pub(super) fn to_rgb(&self) -> Result<Vec<u8>, HwError> {
        let g = &self.geometry;
        let row = g.row_bytes();
        let luma_len = row * to_usize(g.target_height);
        let needed = row * g.rows();
        if self.bytes.len() < needed {
            return Err(HwError::Failed {
                detail: format!(
                    "a copied surface holds {} bytes, its geometry needs {needed}",
                    self.bytes.len()
                ),
            });
        }
        let (luma, chroma) = self.bytes[..needed].split_at(luma_len);
        let (width, height) = (to_usize(g.width), to_usize(g.height));
        if g.bit_depth <= 8 {
            return yuv420_to_rgb8(
                &Yuv420 {
                    layout: YuvLayout::Nv12,
                    y: luma,
                    y_stride: row,
                    u: chroma,
                    v: &[],
                    uv_stride: row,
                    width,
                    height,
                },
                g.matrix,
                g.range,
            );
        }
        // P016 keeps the significant bits at the top of each little-endian
        // 16-bit sample, with zeros below; shift them down to the stream's
        // depth and split the interleaved chroma into planes.
        let shift = 16u32.saturating_sub(g.bit_depth);
        let samples = |bytes: &[u8]| -> Vec<u16> {
            bytes
                .chunks_exact(2)
                .map(|p| u16::from_le_bytes([p[0], p[1]]) >> shift)
                .collect()
        };
        let y = samples(luma);
        let uv = samples(chroma);
        let u: Vec<u16> = uv.iter().step_by(2).copied().collect();
        let v: Vec<u16> = uv.iter().skip(1).step_by(2).copied().collect();
        let samples_per_row = to_usize(g.target_width);
        yuv420_deep_to_rgb8(
            &Yuv420Deep {
                y: &y,
                y_stride: samples_per_row,
                u: &u,
                v: &v,
                uv_stride: samples_per_row / 2,
                width,
                height,
                bit_depth: g.bit_depth,
            },
            g.matrix,
            g.range,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::super::ffi::{CHROMA_444, RectInt, VideoSignalDescription};
    use super::*;

    fn caps() -> CuvidDecodeCaps {
        CuvidDecodeCaps {
            is_supported: 1,
            output_format_mask: 0b11,
            max_width: 4096,
            max_height: 4096,
            max_mb_count: 65536,
            min_width: 48,
            min_height: 16,
            ..CuvidDecodeCaps::default()
        }
    }

    #[test]
    fn availability_codes_are_unavailable() {
        for code in [3, 4, 34, 46, 100, 101, 102, 803, 804] {
            assert!(
                matches!(
                    cu_error("cuInit", CuResult(code), Some(HwCodec::H264)),
                    HwError::Unavailable { .. }
                ),
                "code {code}"
            );
        }
    }

    #[test]
    fn not_supported_is_unsupported_only_for_a_codec_call() {
        assert!(matches!(
            cu_error("cuvidCreateDecoder", CuResult(801), Some(HwCodec::Av1)),
            HwError::Unsupported {
                codec: HwCodec::Av1,
                ..
            }
        ));
        assert!(matches!(
            cu_error("cuMemcpy2D_v2", CuResult(801), None),
            HwError::Failed { .. }
        ));
    }

    #[test]
    fn other_codes_fail_and_name_themselves() {
        let err = cu_error("cuvidParseVideoData", CuResult(2), Some(HwCodec::Hevc));
        let HwError::Failed { detail } = err else {
            panic!("expected Failed, got {err:?}");
        };
        assert!(detail.contains("CUDA_ERROR_OUT_OF_MEMORY (2)"), "{detail}");
        assert!(detail.contains("cuvidParseVideoData"), "{detail}");
        let unknown = cu_error("x", CuResult(12345), None);
        assert!(unknown.to_string().contains("error 12345"), "{unknown}");
        assert!(check("x", CuResult::SUCCESS, None).is_ok());
    }

    #[test]
    fn caps_admit_a_stream_inside_every_limit() {
        let ok = check_caps(HwCodec::H264, &caps(), StreamFormat::BASELINE, 1920, 1080);
        assert_eq!(ok, Ok(()));
    }

    #[test]
    fn caps_refuse_each_limit_as_unsupported() {
        let refuse = |c: &CuvidDecodeCaps, f: StreamFormat, w: u32, h: u32| {
            matches!(
                check_caps(HwCodec::Hevc, c, f, w, h),
                Err(HwError::Unsupported { .. })
            )
        };
        let base = StreamFormat::BASELINE;
        let mut unsupported = caps();
        unsupported.is_supported = 0;
        assert!(refuse(&unsupported, base, 640, 480));
        assert!(refuse(&caps(), base, 8192, 480));
        assert!(refuse(&caps(), base, 32, 480));
        let mut few_mbs = caps();
        // 1920x1080 is 120 x 68 = 8160 macroblocks with the height rounded up.
        few_mbs.max_mb_count = 8159;
        assert!(refuse(&few_mbs, base, 1920, 1080));
        few_mbs.max_mb_count = 8160;
        assert!(!refuse(&few_mbs, base, 1920, 1080));
        let mut nv12_only = caps();
        nv12_only.output_format_mask = 0b01;
        let ten = StreamFormat {
            chroma: CHROMA_420,
            bit_depth: 10,
        };
        assert!(refuse(&nv12_only, ten, 640, 480));
        assert!(!refuse(&caps(), ten, 640, 480));
        let full = StreamFormat {
            chroma: CHROMA_444,
            bit_depth: 8,
        };
        assert!(refuse(&caps(), full, 640, 480));
    }

    fn format(display: RectInt, coded: (u32, u32), depth_minus8: u8) -> CuVideoFormat {
        CuVideoFormat {
            codec: 4,
            coded_width: coded.0,
            coded_height: coded.1,
            display_area: display,
            chroma_format: CHROMA_420,
            bit_depth_luma_minus8: depth_minus8,
            bit_depth_chroma_minus8: depth_minus8,
            min_num_decode_surfaces: 8,
            progressive_sequence: 1,
            video_signal_description: VideoSignalDescription {
                bits: 0b0000_0101,
                color_primaries: 1,
                transfer_characteristics: 1,
                matrix_coefficients: 1,
            },
            ..CuVideoFormat::default()
        }
    }

    #[test]
    fn geometry_follows_the_display_area() {
        let f = format(
            RectInt {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1080,
            },
            (1920, 1088),
            0,
        );
        let g = SurfaceGeometry::from_format(HwCodec::H264, &f).expect("valid");
        assert_eq!((g.width, g.height), (1920, 1080));
        assert_eq!((g.target_width, g.target_height), (1920, 1080));
        assert_eq!(g.matrix, YuvMatrix::Bt709);
        assert_eq!(g.range, YuvRange::Limited);
        assert_eq!(g.row_bytes(), 1920);
        assert_eq!(g.rows(), 1620);
    }

    #[test]
    fn odd_display_sizes_round_the_target_up() {
        let f = format(
            RectInt {
                left: 0,
                top: 0,
                right: 641,
                bottom: 361,
            },
            (656, 368),
            2,
        );
        let g = SurfaceGeometry::from_format(HwCodec::Hevc, &f).expect("valid");
        assert_eq!((g.width, g.height), (641, 361));
        assert_eq!((g.target_width, g.target_height), (642, 362));
        assert_eq!(g.bit_depth, 10);
        assert_eq!(g.row_bytes(), 642 * 2);
        assert_eq!(g.rows(), 362 + 181);
    }

    #[test]
    fn a_display_area_outside_the_picture_is_refused() {
        for area in [
            RectInt {
                left: 0,
                top: 0,
                right: 2000,
                bottom: 1080,
            },
            RectInt {
                left: -1,
                top: 0,
                right: 100,
                bottom: 100,
            },
            RectInt {
                left: 10,
                top: 0,
                right: 10,
                bottom: 100,
            },
        ] {
            let f = format(area, (1920, 1088), 0);
            assert!(matches!(
                SurfaceGeometry::from_format(HwCodec::H264, &f),
                Err(HwError::Failed { .. })
            ));
        }
    }

    #[test]
    fn non_420_and_mixed_depths_are_unsupported() {
        let area = RectInt {
            left: 0,
            top: 0,
            right: 64,
            bottom: 64,
        };
        let mut f = format(area, (64, 64), 0);
        f.chroma_format = CHROMA_444;
        assert!(matches!(
            SurfaceGeometry::from_format(HwCodec::H264, &f),
            Err(HwError::Unsupported { .. })
        ));
        let mut f = format(area, (64, 64), 2);
        f.bit_depth_chroma_minus8 = 0;
        assert!(matches!(
            SurfaceGeometry::from_format(HwCodec::H264, &f),
            Err(HwError::Unsupported { .. })
        ));
    }

    fn geometry(width: u32, height: u32, bit_depth: u32) -> SurfaceGeometry {
        let even = |v: u32| v + (v & 1);
        SurfaceGeometry {
            coded_width: even(width),
            coded_height: even(height),
            left: 0,
            top: 0,
            width,
            height,
            target_width: even(width),
            target_height: even(height),
            bit_depth,
            matrix: YuvMatrix::Bt601,
            range: YuvRange::Full,
        }
    }

    #[test]
    fn nv12_surface_converts_through_the_shared_colour_module() {
        // A 3x3 picture on a 4x4 surface: Y rows, then two interleaved UV rows.
        let g = geometry(3, 3, 8);
        let mut bytes = Vec::new();
        for r in 0..4u8 {
            bytes.extend_from_slice(&[10 * r, 10 * r + 1, 10 * r + 2, 10 * r + 3]);
        }
        bytes.extend_from_slice(&[100, 150, 110, 160, 120, 170, 130, 180]);
        let surface = HostSurface {
            pts: 0,
            geometry: g,
            bytes: bytes.clone(),
        };
        let rgb = surface.to_rgb().expect("converts");
        assert_eq!(rgb.len(), 3 * 3 * 3);
        let direct = yuv420_to_rgb8(
            &Yuv420 {
                layout: YuvLayout::Nv12,
                y: &bytes[..16],
                y_stride: 4,
                u: &bytes[16..],
                v: &[],
                uv_stride: 4,
                width: 3,
                height: 3,
            },
            YuvMatrix::Bt601,
            YuvRange::Full,
        )
        .expect("converts");
        assert_eq!(rgb, direct);
    }

    #[test]
    fn p016_surface_matches_the_same_picture_at_eight_bits() {
        // The same 2x2 picture as NV12 and as 10-bit P016 (each 8-bit value v
        // stored as the 10-bit value v << 2, MSB-aligned as v << 8).
        let nv12 = [16u8, 80, 160, 235, 90, 200];
        let mut p016 = Vec::new();
        for v in nv12 {
            p016.extend_from_slice(&(u16::from(v) << 8).to_le_bytes());
        }
        let eight = HostSurface {
            pts: 1,
            geometry: geometry(2, 2, 8),
            bytes: nv12.to_vec(),
        }
        .to_rgb()
        .expect("converts");
        let ten = HostSurface {
            pts: 1,
            geometry: geometry(2, 2, 10),
            bytes: p016,
        }
        .to_rgb()
        .expect("converts");
        assert_eq!(eight.len(), ten.len());
        for (a, b) in eight.iter().zip(&ten) {
            assert!(a.abs_diff(*b) <= 1, "8-bit {eight:?} vs 10-bit {ten:?}");
        }
    }

    #[test]
    fn a_short_surface_is_refused() {
        let surface = HostSurface {
            pts: 0,
            geometry: geometry(4, 4, 8),
            bytes: vec![0; 23],
        };
        assert!(matches!(surface.to_rgb(), Err(HwError::Failed { .. })));
    }
}
