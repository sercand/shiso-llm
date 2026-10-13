//! The AV1 decoder, for machines whose OS has none.
//!
//! A thin wrapper over the AV1 library's C ABI (the `dav1d_*` functions). The
//! library's Rust API is not public, so the calls go through `extern "C"`
//! functions and raw structs, and this module is where that ends: the
//! [`Av1Decoder`] it exports is safe, and nothing it hands back borrows from the
//! decoder.
//!
//! Ownership, because every handle here has exactly one owner (rule 9):
//!
//! - The decoder context is owned by [`Av1Decoder`] and closed in its `Drop`.
//! - Every input buffer is a [`DataGuard`]: the sample is copied into a buffer
//!   the library allocates and refcounts, because the library keeps a reference
//!   to unconsumed input after `dav1d_send_data` returns, and a borrowed request
//!   buffer would not outlive the call. The copy is a few kilobytes against a
//!   decode that costs far more. The guard unrefs whatever was not consumed.
//! - Every picture is a [`PictureGuard`] from the moment the library writes it,
//!   before its return code is looked at, so it is unref'd on every path,
//!   errors included.
//!
//! The decoder runs single-threaded (`n_threads = 1`, `max_frame_delay = 1`).
//! The server already decodes on the request's own blocking thread, and several
//! requests decode at once; a pool of worker threads per session would multiply
//! by the core count under load, for clips a few seconds long. With one frame
//! context, every picture the input completes
//! is out by the time [`Av1Decoder::decode`] returns, already in display order
//! (AV1 shows frames in the order the temporal units carry them), so
//! [`Av1Decoder::flush`] only drains what a hostile stream might have left.
//!
//! What a picture is and whether it is kept travel with the sample through the
//! library's data properties - the presentation timestamp in `timestamp`, the
//! `keep` flag in `offset` (1 keep, 0 drop) - rather than being inferred from
//! call order, so a temporal unit that shows zero frames or two, or a frame
//! that only surfaces at a flush, never takes another sample's timestamp.
//!
//! The library is built without its hand-written assembly (`asm` feature off).
//! The 1.1.0 package does not build with it on aarch64 macOS (its assembly
//! includes `src/arm/asm-offsets.h`, which the published crate leaves out), and
//! on `x86_64` it would add `nasm` to the build. Without it, a 60-frame
//! 1280x720 clip decodes in 225 ms (8-bit) and 240 ms (10-bit) on an Apple M4
//! in `--release`, 3.8 and 4.0 ms a frame; converting a kept frame to RGB adds
//! about 3 ms. Every frame of a clip is decoded, kept or not, so a minute of
//! 30 fps 720p is about 7 s of decoding: if that ever matters, the assembly is
//! the first lever, once the package builds with it.
//!
//! **Known defect, not yet safe for untrusted input.** The 1.1.0 library
//! aborts the process (a panic inside an `extern "C"` function, which cannot
//! unwind) when a frame's tile data fails to decode with one frame context:
//! its frame-submit error path unwraps a frame header its own frame-exit step
//! has already released. One flipped bit in a sample is enough, and the
//! damaged sample is the first to fail, so stopping at the first error does
//! not help. Settings cannot route around it: every single-frame-context
//! configuration takes that path, and in a trial with frame threading
//! (`n_threads = 2`, `max_frame_delay = 2`) the reproducer no longer came back
//! as an error and the mutation loop hung. `tests/av1.rs` holds the reproducer
//! (`a_damaged_tile_fails`) and the mutation loop, both ignored until the
//! library is fixed.

use std::ffi::{c_int, c_void};
use std::mem::MaybeUninit;
use std::ptr::{self, NonNull};

use rav1d::include::dav1d::data::Dav1dData;
use rav1d::include::dav1d::dav1d::{Dav1dContext, Dav1dLogger, Dav1dSettings};
use rav1d::include::dav1d::headers::{
    DAV1D_PIXEL_LAYOUT_I400, DAV1D_PIXEL_LAYOUT_I420, DAV1D_PIXEL_LAYOUT_I422,
    DAV1D_PIXEL_LAYOUT_I444, Dav1dPixelLayout, Dav1dSequenceHeader,
};
use rav1d::include::dav1d::picture::Dav1dPicture;
use rav1d::src::lib::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_get_picture,
    dav1d_open, dav1d_parse_sequence_header, dav1d_picture_unref, dav1d_send_data,
};

use crate::colour::{self, Yuv420, Yuv420Deep, YuvLayout, YuvMatrix, YuvRange};
use crate::{HwCodec, HwDecoderDesc, HwError, HwFrame, HwTimestamp};

/// The largest frame, in pixels, the decoder will allocate for: AV1 level 6's
/// 8192x4352. A sample is a few bytes that can declare a 65536x65536 frame, and
/// without a cap that is a 16 GiB allocation per request. A larger frame is
/// [`HwError::Unsupported`]: real content that size is beyond what the server
/// samples anyway, and the next decoder in the chain has its own caps.
const FRAME_SIZE_LIMIT: u32 = 8192 * 4352;

/// The av1C record's first byte: marker bit set, version 1.
const AV1C_MARKER_VERSION: u8 = 0x81;

/// How many times one sample may be refused with `EAGAIN` before the decoder is
/// declared stuck. Each refusal is followed by a drain that consumes the
/// library's pending input, so the second send always succeeds on a working
/// decoder; this bound only turns a library bug into an error instead of a hang.
const MAX_SEND_RETRIES: usize = 16;

/// An AV1 decode session in software.
///
/// Opened from the container's `av1C` record (or an empty config, when the
/// stream carries its sequence header in band), fed one temporal unit per
/// [`Av1Decoder::decode`] in decode order. 8- and 10-bit 4:2:0 are decoded;
/// monochrome, 4:2:2, 4:4:4 and 12-bit are [`HwError::Unsupported`].
pub struct Av1Decoder {
    /// The library context. Owned: closed exactly once, in `Drop`.
    ctx: Dav1dContext,
}

/// Summarised rather than derived; the context is an opaque handle.
impl std::fmt::Debug for Av1Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Av1Decoder").finish_non_exhaustive()
    }
}

impl Av1Decoder {
    /// Opens a decoder for an AV1 stream.
    ///
    /// `desc.config` is the container's `av1C` record (a Matroska
    /// `CodecPrivate` is the same record), or empty. Its four header bytes are
    /// checked for a chroma layout and bit depth this decoder handles, and the
    /// sequence header in its trailing OBUs, if any, is checked and fed to the
    /// decoder ahead of the first sample.
    ///
    /// # Errors
    ///
    /// [`HwError::Unsupported`] for a codec other than AV1, or for anything but
    /// 8- and 10-bit 4:2:0; [`HwError::Failed`] for a malformed `av1C` record or
    /// when the library cannot open a context.
    pub fn open(desc: &HwDecoderDesc<'_>) -> Result<Self, HwError> {
        if desc.codec != HwCodec::Av1 {
            return Err(HwError::Unsupported {
                codec: desc.codec,
                reason: "the native video decoder reads AV1 only".into(),
            });
        }
        let config_obus = av1c_config_obus(desc.config)?;
        if !config_obus.is_empty() {
            match parse_sequence_header(config_obus) {
                SequenceHeader::Found(header) => check_sequence(&header)?,
                SequenceHeader::Absent => {}
                SequenceHeader::Malformed(code) => {
                    return Err(HwError::Failed {
                        detail: format!(
                            "the av1C record's sequence header does not parse (error {code})"
                        ),
                    });
                }
            }
        }
        let mut decoder = Self {
            ctx: open_context()?,
        };
        if !config_obus.is_empty() {
            // The sequence header goes in first, so the stream's parameters are
            // known before the first sample; it shows no frame, and with
            // `keep = false` nothing would be converted if it did.
            let mut sink = Vec::new();
            decoder.feed(config_obus, HwTimestamp::new(0), false, &mut sink)?;
        }
        Ok(decoder)
    }

    /// Decodes one container sample (one temporal unit of low-overhead OBUs,
    /// as MP4 and Matroska store them), in decode order.
    ///
    /// Returns the frames it completed, in display order; with `keep = false`
    /// the sample's frame is decoded (later frames may reference it) but not
    /// converted, so it is not returned.
    ///
    /// # Errors
    ///
    /// [`HwError::Unsupported`] when the stream turns out to be a layout or
    /// depth this decoder does not convert, or a frame over the size cap;
    /// [`HwError::Failed`] when the sample is empty or the decoder rejects it.
    pub fn decode(
        &mut self,
        sample: &[u8],
        pts: HwTimestamp,
        keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        if sample.is_empty() {
            return Err(HwError::Failed {
                detail: "an empty AV1 sample".into(),
            });
        }
        // A stream with no av1C sequence header declares its layout here, and
        // a header alone shows no picture for the per-picture check to see. A
        // header that does not parse is left to the decoder to refuse.
        if let SequenceHeader::Found(header) = parse_sequence_header(sample) {
            check_sequence(&header)?;
        }
        let mut frames = Vec::new();
        self.feed(sample, pts, keep, &mut frames)?;
        Ok(frames)
    }

    /// Emits every frame the decoder still holds, in display order.
    ///
    /// This drains; it does not reset. (`dav1d_flush` is a seek: it discards
    /// pending pictures, so it is never called here.) With one frame context
    /// the decoder holds nothing between samples, so this normally returns no
    /// frames.
    ///
    /// # Errors
    ///
    /// [`HwError::Failed`] when the decoder fails while draining.
    pub fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        let mut frames = Vec::new();
        self.drain(&mut frames)?;
        Ok(frames)
    }

    /// Sends `bytes` as one input buffer and collects every picture it
    /// completes.
    fn feed(
        &mut self,
        bytes: &[u8],
        pts: HwTimestamp,
        keep: bool,
        frames: &mut Vec<HwFrame>,
    ) -> Result<(), HwError> {
        let mut data = DataGuard::copy_of(bytes)?;
        data.0.m.timestamp = pts.ticks();
        data.0.m.offset = libc::off_t::from(keep);
        for _ in 0..MAX_SEND_RETRIES {
            // SAFETY: `self.ctx` came from `dav1d_open` and is closed only in
            // `Drop`, so it is live; `data.0` is an initialised `Dav1dData` the
            // guard owns, valid to read and write for the call.
            let sent = unsafe { dav1d_send_data(Some(self.ctx), Some(NonNull::from(&mut data.0))) };
            match sent.0 {
                0 => return self.drain(frames),
                code if code == -libc::EAGAIN => self.drain(frames)?,
                code => return Err(library_error(code, "decoding a sample")),
            }
        }
        Err(HwError::Failed {
            detail: "the AV1 decoder kept refusing input it should have consumed".into(),
        })
    }

    /// Takes pictures until the library has none ready.
    fn drain(&mut self, frames: &mut Vec<HwFrame>) -> Result<(), HwError> {
        loop {
            let mut picture = PictureGuard(Dav1dPicture::default());
            let out = NonNull::from(&mut picture.0);
            // SAFETY: `self.ctx` is live (see `feed`); `out` points at
            // `picture.0`, valid to write. The library overwrites it without
            // dropping the default it held, which owns nothing. The guard
            // unrefs whatever it wrote, whatever the return code.
            let got = unsafe { dav1d_get_picture(Some(self.ctx), Some(out)) };
            match got.0 {
                0 => {
                    if let Some(frame) = convert(&picture)? {
                        frames.push(frame);
                    }
                }
                code if code == -libc::EAGAIN => return Ok(()),
                code => return Err(library_error(code, "decoding a picture")),
            }
        }
    }
}

impl Drop for Av1Decoder {
    fn drop(&mut self) {
        let mut slot = Some(self.ctx);
        // SAFETY: `self.ctx` came from `dav1d_open` and this is the one place it
        // is closed; nothing uses it after `drop`. Every picture taken from it
        // was unref'd by its guard before the call that took it returned.
        unsafe { dav1d_close(Some(NonNull::from(&mut slot))) };
    }
}

/// Opens a library context with this module's settings.
fn open_context() -> Result<Dav1dContext, HwError> {
    let mut settings = MaybeUninit::<Dav1dSettings>::uninit();
    // SAFETY: the pointer is to a live, writable `MaybeUninit<Dav1dSettings>`;
    // `dav1d_default_settings` writes every field of it.
    unsafe { dav1d_default_settings(NonNull::from(&mut settings).cast()) };
    // SAFETY: `dav1d_default_settings` initialised it just above.
    let mut settings = unsafe { settings.assume_init() };
    settings.n_threads = 1;
    settings.max_frame_delay = 1;
    // One frame per temporal unit: the highest spatial layer, not every layer.
    settings.all_layers = 0;
    settings.frame_size_limit = FRAME_SIZE_LIMIT;
    // SAFETY: a logger with no callback is never called; the library turns it
    // into "no logging". Without this it writes to stderr on every malformed
    // sample a client sends.
    settings.logger = unsafe { Dav1dLogger::new(None, None) };

    let mut ctx: Option<Dav1dContext> = None;
    // SAFETY: both pointers are to live locals: `ctx` is valid to write and
    // `settings` is an initialised `Dav1dSettings` valid to read.
    let opened = unsafe {
        dav1d_open(
            Some(NonNull::from(&mut ctx)),
            Some(NonNull::from(&mut settings)),
        )
    };
    match (opened.0, ctx) {
        (0, Some(ctx)) => Ok(ctx),
        (0, None) => Err(HwError::Failed {
            detail: "the AV1 decoder opened without a context".into(),
        }),
        (code, _) => Err(library_error(code, "opening the AV1 decoder")),
    }
}

/// An input buffer the library allocated, holding a copy of one sample.
///
/// Dropping it unrefs whatever the library has not consumed; after a successful
/// `dav1d_send_data` the library has taken the buffer and left an empty
/// `Dav1dData`, which unrefs to nothing.
struct DataGuard(Dav1dData);

impl DataGuard {
    /// Copies `bytes` (non-empty) into a library-owned buffer.
    fn copy_of(bytes: &[u8]) -> Result<Self, HwError> {
        let mut guard = Self(Dav1dData::default());
        // SAFETY: `guard.0` is a live `Dav1dData`, valid to write; on success
        // the library points it at a fresh zeroed buffer of `bytes.len()` bytes
        // and returns that buffer's address.
        let buf = unsafe { dav1d_data_create(Some(NonNull::from(&mut guard.0)), bytes.len()) };
        if buf.is_null() || guard.0.sz != bytes.len() {
            return Err(HwError::Failed {
                detail: format!(
                    "the AV1 decoder could not allocate a {} byte input buffer",
                    bytes.len()
                ),
            });
        }
        // SAFETY: `buf` is the start of the `bytes.len()`-byte buffer just
        // created and owned by `guard`; it cannot overlap the caller's `bytes`.
        unsafe { ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len()) };
        Ok(guard)
    }
}

impl Drop for DataGuard {
    fn drop(&mut self) {
        // SAFETY: `self.0` is a `Dav1dData` that is either default, created by
        // `dav1d_data_create`, or left by `dav1d_send_data`; all three are valid
        // to unref, and it is unref'd once, here.
        unsafe { dav1d_data_unref(Some(NonNull::from(&mut self.0))) };
    }
}

/// A picture the library wrote; unref'd when dropped.
struct PictureGuard(Dav1dPicture);

impl Drop for PictureGuard {
    fn drop(&mut self) {
        // SAFETY: `self.0` is either the default picture (which owns nothing)
        // or one `dav1d_get_picture` wrote; both are valid to unref, and it is
        // unref'd once, here.
        unsafe { dav1d_picture_unref(Some(NonNull::from(&mut self.0))) };
    }
}

/// Converts one decoded picture to RGB, or `None` when its sample was not to be
/// kept.
fn convert(picture: &PictureGuard) -> Result<Option<HwFrame>, HwError> {
    let pic = &picture.0;
    // Checked even for dropped frames: a stream that switches to 4:4:4
    // mid-way is refused whether or not that frame was sampled.
    check_layout(pic.p.layout, pic.p.bpc)?;
    if pic.m.offset != 1 {
        return Ok(None);
    }
    let width = positive(pic.p.w, "width")?;
    let height = positive(pic.p.h, "height")?;
    let Some(header) = pic.seq_hdr else {
        return Err(HwError::Failed {
            detail: "a decoded AV1 picture without a sequence header".into(),
        });
    };
    // SAFETY: the picture holds a reference on its sequence header
    // (`seq_hdr_ref`) until it is unref'd, which happens when the guard drops,
    // after this borrow of it ends.
    let header = unsafe { header.as_ref() };
    let matrix = YuvMatrix::from_code(
        u8::try_from(header.mtrx).unwrap_or(2),
        u32::try_from(height).unwrap_or(u32::MAX),
    );
    let range = if header.color_range == 0 {
        YuvRange::Limited
    } else {
        YuvRange::Full
    };
    let chroma_width = width.div_ceil(2);
    let chroma_height = height.div_ceil(2);
    let luma_stride = stride(pic.stride[0], "luma")?;
    let chroma_stride = stride(pic.stride[1], "chroma")?;
    let rgb = if pic.p.bpc == 8 {
        let luma_len = plane_len(height, width, luma_stride)?;
        let chroma_len = plane_len(chroma_height, chroma_width, chroma_stride)?;
        // SAFETY: the library allocates each plane with `p.h` (luma) or
        // `ceil(p.h / 2)` (chroma) rows of `stride` bytes each, and `plane_len`
        // checked the stride covers a row; the picture outlives the slices
        // (they are dropped before `picture`).
        let (y, u, v) = unsafe {
            (
                plane::<u8>(pic.data[0], luma_len)?,
                plane::<u8>(pic.data[1], chroma_len)?,
                plane::<u8>(pic.data[2], chroma_len)?,
            )
        };
        colour::yuv420_to_rgb8(
            &Yuv420 {
                layout: YuvLayout::I420,
                y,
                y_stride: luma_stride,
                u,
                v,
                uv_stride: chroma_stride,
                width,
                height,
            },
            matrix,
            range,
        )?
    } else {
        // 16-bit storage: strides are in bytes, the slices in samples.
        let luma_stride = samples(luma_stride)?;
        let chroma_stride = samples(chroma_stride)?;
        let luma_len = plane_len(height, width, luma_stride)?;
        let chroma_len = plane_len(chroma_height, chroma_width, chroma_stride)?;
        // SAFETY: as for the 8-bit planes, in `u16` samples: `samples` checked
        // each byte stride is a whole number of samples, and `plane` checks
        // the pointers are aligned for `u16`.
        let (y, u, v) = unsafe {
            (
                plane::<u16>(pic.data[0], luma_len)?,
                plane::<u16>(pic.data[1], chroma_len)?,
                plane::<u16>(pic.data[2], chroma_len)?,
            )
        };
        colour::yuv420_deep_to_rgb8(
            &Yuv420Deep {
                y,
                y_stride: luma_stride,
                u,
                v,
                uv_stride: chroma_stride,
                width,
                height,
                bit_depth: u32::try_from(pic.p.bpc).map_err(|_| too_large())?,
            },
            matrix,
            range,
        )?
    };
    let frame = HwFrame::new(
        u32::try_from(width).map_err(|_| too_large())?,
        u32::try_from(height).map_err(|_| too_large())?,
        rgb,
        HwTimestamp::new(pic.m.timestamp),
    )?;
    Ok(Some(frame))
}

/// Refuses every picture format but 8- and 10-bit 4:2:0.
fn check_layout(layout: Dav1dPixelLayout, bits: c_int) -> Result<(), HwError> {
    let chroma = match layout {
        DAV1D_PIXEL_LAYOUT_I420 => None,
        DAV1D_PIXEL_LAYOUT_I400 => Some("monochrome (4:0:0)"),
        DAV1D_PIXEL_LAYOUT_I422 => Some("4:2:2 chroma"),
        DAV1D_PIXEL_LAYOUT_I444 => Some("4:4:4 chroma"),
        _ => Some("an unknown chroma layout"),
    };
    if let Some(what) = chroma {
        return Err(unsupported(format!("{what}; only 4:2:0 is converted")));
    }
    if bits != 8 && bits != 10 {
        return Err(unsupported(format!(
            "{bits}-bit samples; only 8- and 10-bit are converted"
        )));
    }
    Ok(())
}

/// Refuses a sequence header this decoder would not convert.
fn check_sequence(header: &Dav1dSequenceHeader) -> Result<(), HwError> {
    // `hbd` is 0, 1 or 2 for 8, 10 or 12 bits.
    let bits = match header.hbd {
        0 => 8,
        1 => 10,
        _ => 12,
    };
    check_layout(header.layout, bits)
}

/// What [`parse_sequence_header`] found.
enum SequenceHeader {
    /// A sequence header OBU, parsed.
    Found(Box<Dav1dSequenceHeader>),
    /// The bytes parse as OBUs but none is a sequence header.
    Absent,
    /// The bytes do not parse; the library's error code.
    Malformed(c_int),
}

/// Finds and parses the sequence header OBU in a run of low-overhead OBUs.
fn parse_sequence_header(obus: &[u8]) -> SequenceHeader {
    if obus.is_empty() {
        return SequenceHeader::Absent;
    }
    let mut header = Box::new(MaybeUninit::<Dav1dSequenceHeader>::uninit());
    // SAFETY: `header` is valid to write a `Dav1dSequenceHeader` to, and the
    // data pointer and length are those of the live, non-empty `obus` slice,
    // which the library only reads.
    let parsed = unsafe {
        dav1d_parse_sequence_header(
            Some(NonNull::from(&mut *header).cast()),
            Some(NonNull::from(obus).cast()),
            obus.len(),
        )
    };
    match parsed.0 {
        // SAFETY: on success the library wrote the whole header.
        0 => SequenceHeader::Found(unsafe { header.assume_init() }),
        code if code == -libc::ENOENT => SequenceHeader::Absent,
        code => SequenceHeader::Malformed(code),
    }
}

/// The configOBUs of an `av1C` record, after its layout checks; empty for an
/// empty record.
///
/// The record is `marker(1) version(7) | seq_profile(3) seq_level_idx_0(5) |
/// seq_tier_0(1) high_bitdepth(1) twelve_bit(1) monochrome(1)
/// chroma_subsampling_x(1) chroma_subsampling_y(1) chroma_sample_position(2) |
/// reserved(3) initial_presentation_delay(5)`, then the configOBUs.
fn av1c_config_obus(config: &[u8]) -> Result<&[u8], HwError> {
    if config.is_empty() {
        return Ok(config);
    }
    let Some((&[marker, _profile_level, flags, _delay], obus)) = config.split_first_chunk::<4>()
    else {
        return Err(HwError::Failed {
            detail: format!(
                "an av1C record of {} bytes, shorter than its header",
                config.len()
            ),
        });
    };
    if marker != AV1C_MARKER_VERSION {
        return Err(HwError::Failed {
            detail: format!("an av1C record starting {marker:#04x}, not marker and version 1"),
        });
    }
    if flags & 0x20 != 0 {
        return Err(unsupported(
            "12-bit samples; only 8- and 10-bit are converted".into(),
        ));
    }
    if flags & 0x10 != 0 {
        return Err(unsupported(
            "monochrome (4:0:0); only 4:2:0 is converted".into(),
        ));
    }
    if flags & 0x0c != 0x0c {
        return Err(unsupported(
            "chroma that is not subsampled both ways; only 4:2:0 is converted".into(),
        ));
    }
    Ok(obus)
}

/// Borrows a plane of `len` samples from a picture's data pointer.
///
/// # Safety
///
/// When `data` is non-null it must point to at least `len` initialised `T`
/// that stay valid and unwritten for `'a`.
unsafe fn plane<'a, T>(data: Option<NonNull<c_void>>, len: usize) -> Result<&'a [T], HwError> {
    let Some(data) = data else {
        return Err(HwError::Failed {
            detail: "a decoded AV1 picture with a missing plane".into(),
        });
    };
    let data = data.cast::<T>();
    if !data.is_aligned() {
        return Err(HwError::Failed {
            detail: "a decoded AV1 plane that is not aligned for its samples".into(),
        });
    }
    // SAFETY: non-null and aligned (checked above), and the caller guarantees
    // `len` initialised samples that live and stay unwritten for `'a`.
    Ok(unsafe { std::slice::from_raw_parts(data.as_ptr(), len) })
}

/// The span of a plane `rows` high and `row` samples wide at `stride` samples
/// per row: every row but the last in full, and the last row's own samples.
fn plane_len(rows: usize, row: usize, stride: usize) -> Result<usize, HwError> {
    if stride < row {
        return Err(HwError::Failed {
            detail: format!("a decoded AV1 plane with stride {stride} narrower than its {row}"),
        });
    }
    (rows - 1)
        .checked_mul(stride)
        .and_then(|span| span.checked_add(row))
        .ok_or_else(too_large)
}

/// A picture stride in bytes, refusing a negative one (a bottom-up picture
/// this module does not read).
fn stride(bytes: isize, plane: &str) -> Result<usize, HwError> {
    usize::try_from(bytes).map_err(|_| HwError::Failed {
        detail: format!("a decoded AV1 picture with a negative {plane} stride"),
    })
}

/// A byte stride as a count of 16-bit samples.
fn samples(bytes: usize) -> Result<usize, HwError> {
    if bytes.is_multiple_of(2) {
        Ok(bytes / 2)
    } else {
        Err(HwError::Failed {
            detail: format!("a 16-bit AV1 plane with an odd stride of {bytes} bytes"),
        })
    }
}

/// A picture dimension, which must be at least one pixel.
fn positive(value: c_int, what: &str) -> Result<usize, HwError> {
    usize::try_from(value)
        .ok()
        .filter(|&value| value > 0)
        .ok_or_else(|| HwError::Failed {
            detail: format!("a decoded AV1 picture with {what} {value}"),
        })
}

fn too_large() -> HwError {
    HwError::Failed {
        detail: "a decoded AV1 picture too large to address".into(),
    }
}

fn unsupported(reason: String) -> HwError {
    HwError::Unsupported {
        codec: HwCodec::Av1,
        reason,
    }
}

/// Maps a negative errno from the library.
///
/// `ENOPROTOOPT` (a bit depth this build lacks) and `ERANGE` (a frame over
/// [`FRAME_SIZE_LIMIT`]) are the decoder declining a stream it cannot handle,
/// so another decoder may; everything else is this input failing.
fn library_error(code: c_int, during: &str) -> HwError {
    match code.checked_neg() {
        Some(libc::ENOPROTOOPT) => unsupported(format!("{during}: a feature this build lacks")),
        Some(libc::ERANGE) => unsupported(format!(
            "{during}: a frame larger than {FRAME_SIZE_LIMIT} pixels"
        )),
        Some(libc::ENOMEM) => HwError::Failed {
            detail: format!("{during}: out of memory"),
        },
        Some(libc::EINVAL) => HwError::Failed {
            detail: format!("{during}: the bitstream is invalid"),
        },
        _ => HwError::Failed {
            detail: format!("{during}: error {code}"),
        },
    }
}
