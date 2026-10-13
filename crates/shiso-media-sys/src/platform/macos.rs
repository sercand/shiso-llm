//! `VideoToolbox`: H.264, HEVC, VP9 and AV1 through the OS decoder.
//!
//! A [`Session`] is a `VTDecompressionSession` opened from a format
//! description built from the container's configuration record, and asked
//! for 32-bit BGRA so the OS applies the colour matrix and range and this side
//! only drops the alpha byte. The description is built two ways:
//!
//! - **H.264, HEVC:** `CoreMedia`'s parameter-set constructors, fed the SPS/PPS
//!   (and VPS) out of the `avcC`/`hvcC`. They read the picture size and the
//!   VUI colour description themselves.
//! - **VP9, AV1:** `CMVideoFormatDescriptionCreate` with the record attached
//!   under `SampleDescriptionExtensionAtoms`, plus the colour description read
//!   out of the record here ([`config`]) and, for AV1, the sequence header's
//!   picture size.
//!
//! One generic path for all four was tried first and decodes all four, but
//! the decoder converts to BGRA with the matrix the *description* names and
//! never the bitstream's: with none named it used BT.601 for the BT.709
//! fixtures (about 29 dB against the reference instead of 50), and it sized
//! its buffers from the description, not the SPS. Hence the split.
//!
//! Each sample is copied into a `CMBlockBuffer` `CoreMedia` allocates (the
//! caller's slice is borrowed only for the call, and the decoder may read the
//! block after `DecodeFrame` returns), decoded asynchronously, and waited for.
//! The output handler is a block created per sample: it captures the sample's
//! `keep` flag and a reference-counted handle on the session's shared state,
//! so a callback that fires late - during `Invalidate` in [`Drop`] - still finds
//! live state. It converts a kept frame while the decoder's image buffer is
//! valid (only during the callback), records a decoder error for the next
//! `decode` or `flush` to return, and never unwinds into the framework.
//!
//! Frames come back in the order the decoder finishes them, which for H.264
//! and HEVC with B-frames is decode order - measured on this machine with and
//! without `kVTDecodeFrame_EnableTemporalProcessing`, which changes nothing
//! here - so [`Reorder`] puts them in display order (see [`reorder_depth`]).
//!
//! Errors: a configuration record that does not parse, here or in `CoreMedia`,
//! or a picture size no picture can have, is [`HwError::Failed`] - malformed
//! input, not a reason to try another decoder. A well-formed description the
//! OS will not open a session for (`kVTCouldNotFindVideoDecoderErr` for a codec
//! it has no decoder for, a profile or size its decoder refuses) is
//! [`HwError::Unsupported`], carrying the `OSStatus`; so is an `avcC`/`hvcC`
//! that legally carries no parameter sets (they travel in-band), since the
//! description cannot be built without them. Any failure while
//! decoding, including one the decoder reports through the handler, is
//! [`HwError::Failed`].
//!
//! # Where the decoding happens
//!
//! The spec asks whether `VideoToolbox` decodes out of process, so that a
//! hostile bitstream cannot corrupt the server's memory. Checked on an Apple M4,
//! macOS 26.5, with sessions for all four codecs open and mid-stream (`vmmap`
//! of the test process and of the decoder service, 2026-10-14):
//!
//! - **Verified:** the decoders run in a separate `VTDecoderXPCService`
//!   process that `VideoToolbox` launches for the client. With hardware
//!   decoding (the default) it had `AppleVideoDecoder.bundle` loaded, and with
//!   hardware turned off in the decoder specification the software H.264 and
//!   HEVC decoders (`AppleH264SW.bundle`, `VCPHEVC.bundle`) loaded there too.
//!   No decoder plug-in was mapped into the test process in either case.
//! - **Not verified:** that every decoder on every Mac is hosted that way -
//!   in particular the software AV1 decoder a Mac without AV1 hardware would
//!   use (with hardware off, this M4 offered no software VP9 or AV1 decoder at
//!   all, so it could not be observed); and how a crash of the service surfaces
//!   (expected: an `OSStatus` from the session, which becomes `Failed`).
//!
//! Some parsing still happens in this process: `CoreMedia` parses the parameter
//! sets into the format description, and this module parses the records'
//! headers ([`config`]) and every sample's length prefixes
//! ([`framing::check_sample`]) in safe code before anything reaches the
//! framework.

mod config;
mod framing;
mod reorder;

use std::ffi::{c_int, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::{self, NonNull};
use std::sync::{Arc, Mutex, Once, PoisonError};

use block2::RcBlock;
use objc2_core_foundation::{
    CFBoolean, CFData, CFDictionary, CFNumber, CFRetained, CFString, CFType,
};
use objc2_core_media::{
    CMBlockBuffer, CMFormatDescription, CMSampleBuffer, CMSampleTimingInfo, CMTime, CMTimeFlags,
    CMVideoCodecType, CMVideoFormatDescriptionCreate,
    CMVideoFormatDescriptionCreateFromH264ParameterSets,
    CMVideoFormatDescriptionCreateFromHEVCParameterSets, kCMBlockBufferAssureMemoryNowFlag,
    kCMFormatDescriptionColorPrimaries_DCI_P3, kCMFormatDescriptionColorPrimaries_EBU_3213,
    kCMFormatDescriptionColorPrimaries_ITU_R_709_2, kCMFormatDescriptionColorPrimaries_ITU_R_2020,
    kCMFormatDescriptionColorPrimaries_P3_D65, kCMFormatDescriptionColorPrimaries_SMPTE_C,
    kCMFormatDescriptionExtension_ColorPrimaries, kCMFormatDescriptionExtension_FullRangeVideo,
    kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
    kCMFormatDescriptionExtension_TransferFunction, kCMFormatDescriptionExtension_YCbCrMatrix,
    kCMFormatDescriptionTransferFunction_ITU_R_709_2,
    kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG,
    kCMFormatDescriptionTransferFunction_Linear,
    kCMFormatDescriptionTransferFunction_SMPTE_240M_1995,
    kCMFormatDescriptionTransferFunction_SMPTE_ST_428_1,
    kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ,
    kCMFormatDescriptionTransferFunction_sRGB, kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4,
    kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2, kCMFormatDescriptionYCbCrMatrix_ITU_R_2020,
    kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995, kCMVideoCodecType_AV1, kCMVideoCodecType_H264,
    kCMVideoCodecType_HEVC, kCMVideoCodecType_VP9,
};
use objc2_core_video::{
    CVImageBuffer, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow,
    CVPixelBufferGetDataSize, CVPixelBufferGetHeight, CVPixelBufferGetPixelFormatType,
    CVPixelBufferGetWidth, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags,
    CVPixelBufferUnlockBaseAddress, kCVPixelBufferPixelFormatTypeKey, kCVPixelFormatType_32BGRA,
};
use objc2_video_toolbox::{
    VTDecodeFrameFlags, VTDecodeInfoFlags, VTDecompressionSession, VTIsHardwareDecodeSupported,
    VTRegisterSupplementalVideoDecoderIfAvailable,
};

use self::config::ColourCodes;
use self::framing::{Framing, bgra_to_rgb, check_sample, framing};
use self::reorder::Reorder;
use crate::{HwCodec, HwDecoderDesc, HwError, HwFrame, HwProbe, HwTimestamp};

/// How many later frames a decoded frame waits for before it is released.
///
/// H.264 and HEVC reorder up to the size of the decoded picture buffer, at
/// most 16 frames in either standard, so 16 is the depth no conforming stream
/// exceeds without reading `max_num_reorder_frames` out of the SPS. The cost is
/// latency, not correctness: up to 16 *kept* frames are held as RGB before the
/// first is returned (frames decoded with `keep = false` hold no pixels), and
/// [`flush`](Session::flush) returns the rest. VP9 and AV1 have no output
/// reordering - a hidden frame is shown later by a sample of its own - so their
/// frames are released at once, and one out of order fails loudly in
/// [`Reorder`] rather than being returned out of order.
fn reorder_depth(codec: HwCodec) -> usize {
    match codec {
        HwCodec::H264 | HwCodec::Hevc => 16,
        HwCodec::Vp9 | HwCodec::Av1 => 0,
    }
}

/// A `VideoToolbox` decode session.
#[derive(Debug)]
pub(crate) struct Session {
    codec: HwCodec,
    framing: Framing,
    format: CFRetained<CMFormatDescription>,
    decompression: CFRetained<VTDecompressionSession>,
    shared: Arc<Mutex<Shared>>,
    reorder: Reorder,
}

// SAFETY: a `VTDecompressionSession` and a `CMFormatDescription` are CoreFoundation
// objects whose retain and release are thread-safe, and `VideoToolbox` allows a
// session to be driven from any thread as long as calls do not overlap. `Session`
// is only reached through `&mut self` (it is not `Sync`), so calls never overlap;
// the output handler touches only `shared`, which is behind a mutex.
unsafe impl Send for Session {}

/// What the output handler hands back to the session.
#[derive(Debug, Default)]
struct Shared {
    /// Decoded frames not yet passed to the reorder buffer, with their pixels
    /// when the sample was kept.
    frames: Vec<(HwTimestamp, Option<HwFrame>)>,
    /// The first error the handler saw; it fails the next `decode` or `flush`.
    error: Option<String>,
}

impl Session {
    pub(crate) fn open(desc: &HwDecoderDesc<'_>) -> Result<Self, HwError> {
        register_supplemental_decoders();
        let codec = desc.codec;
        let framing = framing(codec, desc.config)?;
        if desc.config.is_empty() {
            return Err(HwError::Failed {
                detail: format!("{codec} stream has an empty configuration record"),
            });
        }
        // A size no picture can have is malformed input whichever decoder
        // reads it; H.264 and HEVC then take their size from the SPS.
        let (width, height) = dimensions(codec, (desc.width, desc.height))?;
        let format = format_description(codec, desc.config, framing, width, height)?;

        // Ask for BGRA so the OS does the colour matrix and range.
        // SAFETY: an immutable CoreVideo constant, initialised when the framework
        // loads.
        let pixel_format_key: &CFString = unsafe { kCVPixelBufferPixelFormatTypeKey };
        let bgra = CFNumber::new_i32(i32::try_from(kCVPixelFormatType_32BGRA).map_err(|_| {
            HwError::Failed {
                detail: "the BGRA pixel format code does not fit a CFNumber".into(),
            }
        })?);
        let attributes =
            CFDictionary::<CFString, CFType>::from_slices(&[pixel_format_key], &[&bgra]);

        let mut raw: *mut VTDecompressionSession = ptr::null_mut();
        // SAFETY: `format` is a live video format description; `attributes` is a
        // CFDictionary of CFString keys to CFType values, as the call requires;
        // a null callback record is allowed when every frame is decoded with an
        // output handler, which is the only way this session decodes; `raw` is a
        // valid out-pointer.
        let status = unsafe {
            VTDecompressionSession::create(
                None,
                &format,
                None,
                Some(attributes.as_opaque()),
                ptr::null(),
                NonNull::from(&mut raw),
            )
        };
        if status != 0 {
            return Err(HwError::Unsupported {
                codec,
                reason: format!(
                    "VTDecompressionSessionCreate refused the stream: {}",
                    describe(status)
                ),
            });
        }
        let raw = NonNull::new(raw).ok_or_else(|| HwError::Unsupported {
            codec,
            reason: "VTDecompressionSessionCreate succeeded without a session".into(),
        })?;
        // SAFETY: `create` returned a +1 reference in `raw` (Create rule), which
        // `CFRetained` takes over and releases on drop.
        let decompression = unsafe { CFRetained::from_raw(raw) };
        Ok(Self {
            codec,
            framing,
            format,
            decompression,
            shared: Arc::new(Mutex::new(Shared::default())),
            reorder: Reorder::new(reorder_depth(codec)),
        })
    }

    pub(crate) fn decode(
        &mut self,
        sample: &[u8],
        pts: HwTimestamp,
        keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        check_sample(self.framing, sample)?;
        let buffer = self.sample_buffer(sample, pts)?;

        let shared = Arc::clone(&self.shared);
        let handler = RcBlock::new(
            move |status: i32,
                  info: VTDecodeInfoFlags,
                  image: *mut CVImageBuffer,
                  out_pts: CMTime,
                  _duration: CMTime| {
                on_output(&shared, keep, status, info, image, out_pts);
            },
        );
        // A frame that will not be kept is still decoded (later frames may
        // reference it), but the decoder is told not to produce an image for
        // it, which spares the BGRA conversion as well as the copy.
        let mut flags = VTDecodeFrameFlags::Frame_EnableAsynchronousDecompression;
        if !keep {
            flags |= VTDecodeFrameFlags::Frame_DoNotOutputFrame;
        }
        let mut info = VTDecodeInfoFlags::empty();
        // SAFETY: the session and the sample buffer are live; `info` is a valid
        // out-pointer; the handler block is retained by the session for as long
        // as it may call it (it is copied on submission), and it owns everything
        // it touches.
        let status = unsafe {
            self.decompression.decode_frame_with_output_handler(
                &buffer,
                flags,
                &raw mut info,
                RcBlock::as_ptr(&handler),
            )
        };
        if status != 0 {
            return Err(HwError::Failed {
                detail: format!(
                    "VTDecompressionSessionDecodeFrame failed on the {} sample at pts {}: {}",
                    self.codec,
                    pts.ticks(),
                    describe(status)
                ),
            });
        }
        self.wait()?;
        self.collect()?;
        self.reorder.ready()
    }

    pub(crate) fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        // SAFETY: the session is live.
        let status = unsafe { self.decompression.finish_delayed_frames() };
        if status != 0 {
            return Err(HwError::Failed {
                detail: format!(
                    "VTDecompressionSessionFinishDelayedFrames failed: {}",
                    describe(status)
                ),
            });
        }
        self.wait()?;
        self.collect()?;
        self.reorder.drain()
    }

    /// Blocks until every frame submitted so far has reached the handler.
    fn wait(&self) -> Result<(), HwError> {
        // SAFETY: the session is live. The handler may run on this thread
        // inside this call; no lock is held here.
        let status = unsafe { self.decompression.wait_for_asynchronous_frames() };
        if status == 0 {
            Ok(())
        } else {
            Err(HwError::Failed {
                detail: format!(
                    "VTDecompressionSessionWaitForAsynchronousFrames failed: {}",
                    describe(status)
                ),
            })
        }
    }

    /// Moves what the handler produced into the reorder buffer, failing on
    /// the first error it recorded.
    fn collect(&mut self) -> Result<(), HwError> {
        let (frames, error) = {
            let mut shared = self.shared.lock().unwrap_or_else(PoisonError::into_inner);
            (std::mem::take(&mut shared.frames), shared.error.take())
        };
        if let Some(detail) = error {
            return Err(HwError::Failed { detail });
        }
        for (pts, frame) in frames {
            self.reorder.push(pts, frame);
        }
        Ok(())
    }

    /// Wraps a copy of `sample` in a ready `CMSampleBuffer` carrying `pts`.
    fn sample_buffer(
        &self,
        sample: &[u8],
        pts: HwTimestamp,
    ) -> Result<CFRetained<CMSampleBuffer>, HwError> {
        let failed = |what: &str, status: i32| HwError::Failed {
            detail: format!("{what} failed: {}", describe(status)),
        };
        let mut block: *mut CMBlockBuffer = ptr::null_mut();
        // SAFETY: a null memory block with the default allocator asks `CoreMedia`
        // to allocate `sample.len()` bytes itself (now, with AssureMemoryNow);
        // no custom block source; `block` is a valid out-pointer.
        let status = unsafe {
            CMBlockBuffer::create_with_memory_block(
                None,
                ptr::null_mut(),
                sample.len(),
                None,
                ptr::null(),
                0,
                sample.len(),
                kCMBlockBufferAssureMemoryNowFlag,
                NonNull::from(&mut block),
            )
        };
        if status != 0 {
            return Err(failed("CMBlockBufferCreateWithMemoryBlock", status));
        }
        let block = NonNull::new(block).ok_or_else(|| HwError::Failed {
            detail: "CMBlockBufferCreateWithMemoryBlock returned no buffer".into(),
        })?;
        // SAFETY: a +1 reference from a Create function.
        let block = unsafe { CFRetained::from_raw(block) };
        let source = NonNull::from(sample).cast::<c_void>();
        // SAFETY: `block` owns exactly `sample.len()` bytes, allocated above;
        // `source` points at `sample.len()` readable bytes for the call.
        let status = unsafe { CMBlockBuffer::replace_data_bytes(source, &block, 0, sample.len()) };
        if status != 0 {
            return Err(failed("CMBlockBufferReplaceDataBytes", status));
        }

        let timing = CMSampleTimingInfo {
            duration: invalid_time(),
            presentationTimeStamp: CMTime {
                value: pts.ticks(),
                timescale: 1,
                flags: CMTimeFlags::Valid,
                epoch: 0,
            },
            decodeTimeStamp: invalid_time(),
        };
        let size = sample.len();
        let mut out: *mut CMSampleBuffer = ptr::null_mut();
        // SAFETY: the block buffer and format description are live; one timing
        // entry and one size entry, each pointing at a local that outlives the
        // call (`CoreMedia` copies them); `out` is a valid out-pointer.
        let status = unsafe {
            CMSampleBuffer::create_ready(
                None,
                Some(&block),
                Some(&self.format),
                1,
                1,
                &raw const timing,
                1,
                &raw const size,
                NonNull::from(&mut out),
            )
        };
        if status != 0 {
            return Err(failed("CMSampleBufferCreateReady", status));
        }
        let out = NonNull::new(out).ok_or_else(|| HwError::Failed {
            detail: "CMSampleBufferCreateReady returned no buffer".into(),
        })?;
        // SAFETY: a +1 reference from a Create function.
        Ok(unsafe { CFRetained::from_raw(out) })
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        // Invalidate before the last release: it stops the decoder and
        // completes or discards outstanding frames, so no handler runs after
        // this. A handler that does run in here only touches the shared state
        // its own block keeps alive.
        // SAFETY: the session is live; invalidating twice is harmless and the
        // session is not used afterwards.
        unsafe { self.decompression.invalidate() };
    }
}

/// The output handler's body: converts a kept frame, records an error.
fn on_output(
    shared: &Mutex<Shared>,
    keep: bool,
    status: i32,
    info: VTDecodeInfoFlags,
    image: *mut CVImageBuffer,
    pts: CMTime,
) {
    // A panic must not unwind into the framework; it becomes a decode error.
    let result = catch_unwind(AssertUnwindSafe(|| output(keep, status, info, image, pts)))
        .unwrap_or_else(|_| Err("the output handler panicked".into()));
    let mut shared = shared.lock().unwrap_or_else(PoisonError::into_inner);
    match result {
        Ok(entry) => shared.frames.push(entry),
        Err(detail) => {
            shared.error.get_or_insert(detail);
        }
    }
}

fn output(
    keep: bool,
    status: i32,
    info: VTDecodeInfoFlags,
    image: *mut CVImageBuffer,
    pts: CMTime,
) -> Result<(HwTimestamp, Option<HwFrame>), String> {
    if status != 0 {
        return Err(format!(
            "the decoder rejected a frame: {}",
            describe(status)
        ));
    }
    if !pts.flags.contains(CMTimeFlags::Valid) || pts.timescale != 1 {
        return Err("the decoder returned a frame without the timestamp it was given".into());
    }
    // A by-value copy: `CMTime` is packed, so its fields cannot be borrowed.
    let value = pts.value;
    let ticks = HwTimestamp::new(value);
    if !keep {
        // Asked for no image; whether one came anyway does not matter.
        return Ok((ticks, None));
    }
    let Some(image) = NonNull::new(image) else {
        return Err(if info.contains(VTDecodeInfoFlags::FrameDropped) {
            format!("the decoder dropped the frame at pts {value}")
        } else {
            format!("the decoder returned no image for pts {value}")
        });
    };
    // SAFETY: the image buffer is valid for the duration of the callback, which
    // is the whole of this function.
    let image = unsafe { image.as_ref() };
    let frame = copy_frame(image, ticks).map_err(|e| e.to_string())?;
    Ok((ticks, Some(frame)))
}

/// Copies a BGRA pixel buffer into an RGB8 frame.
fn copy_frame(image: &CVImageBuffer, pts: HwTimestamp) -> Result<HwFrame, HwError> {
    let format = CVPixelBufferGetPixelFormatType(image);
    if format != kCVPixelFormatType_32BGRA {
        return Err(HwError::Failed {
            detail: format!("the decoder returned pixel format {format:#010x}, not BGRA"),
        });
    }
    // SAFETY: `image` is a live pixel buffer; read-only lock, paired with the
    // unlock below on every path.
    let status = unsafe { CVPixelBufferLockBaseAddress(image, CVPixelBufferLockFlags::ReadOnly) };
    if status != 0 {
        return Err(HwError::Failed {
            detail: format!("CVPixelBufferLockBaseAddress failed: {status}"),
        });
    }
    let copied = copy_locked(image, pts);
    // SAFETY: balances the lock above with the same flags.
    unsafe { CVPixelBufferUnlockBaseAddress(image, CVPixelBufferLockFlags::ReadOnly) };
    copied
}

fn copy_locked(image: &CVImageBuffer, pts: HwTimestamp) -> Result<HwFrame, HwError> {
    let width = CVPixelBufferGetWidth(image);
    let height = CVPixelBufferGetHeight(image);
    let stride = CVPixelBufferGetBytesPerRow(image);
    let size = CVPixelBufferGetDataSize(image);
    let base = CVPixelBufferGetBaseAddress(image);
    let Some(base) = NonNull::new(base.cast::<u8>()) else {
        return Err(HwError::Failed {
            detail: "a locked pixel buffer has no base address".into(),
        });
    };
    // The buffer must hold every row it claims; `size` is CoreVideo's own
    // count of the bytes behind `base`.
    let needed = stride
        .checked_mul(height)
        .filter(|n| *n <= size)
        .ok_or_else(|| HwError::Failed {
            detail: format!(
                "a {width}x{height} pixel buffer with stride {stride} claims only {size} bytes"
            ),
        })?;
    // SAFETY: the buffer is locked for reading, `base` is its non-null base
    // address, and `needed <= size` bytes lie behind it for as long as the lock
    // is held, which outlives this borrow.
    let pixels = unsafe { std::slice::from_raw_parts(base.as_ptr(), needed) };
    let rgb = bgra_to_rgb(pixels, width, height, stride)?;
    let too_big = |v: usize| HwError::Failed {
        detail: format!("a pixel buffer dimension of {v} does not fit a frame"),
    };
    let width = u32::try_from(width).map_err(|_| too_big(width))?;
    let height = u32::try_from(height).map_err(|_| too_big(height))?;
    HwFrame::new(width, height, rgb, pts)
}

/// Builds the format description from the container's configuration record.
///
/// H.264 and HEVC go through `CoreMedia`'s parameter-set constructors, which
/// read the size and the colour description (VUI) from the SPS; VP9 and AV1
/// go through the generic constructor with the record attached as a
/// sample-description atom and the colour read here ([`config`]), because
/// the decoder converts with the matrix the description names and never the
/// bitstream's.
fn format_description(
    codec: HwCodec,
    config: &[u8],
    framing: Framing,
    width: i32,
    height: i32,
) -> Result<CFRetained<CMFormatDescription>, HwError> {
    let mut out: *const CMFormatDescription = ptr::null();
    let status = if let (HwCodec::H264 | HwCodec::Hevc, Framing::LengthPrefixed(length)) =
        (codec, framing)
    {
        let sets = config::parameter_sets(codec, config)?;
        let mut pointers: Vec<NonNull<u8>> = sets
            .iter()
            .map(|s| NonNull::from(*s).cast::<u8>())
            .collect();
        let mut sizes: Vec<usize> = sets.iter().map(|s| s.len()).collect();
        let length = c_int::try_from(length).map_err(|_| HwError::Failed {
            detail: format!("a NAL length size of {length}"),
        })?;
        let pointers_ptr = NonNull::new(pointers.as_mut_ptr()).ok_or_else(no_sets)?;
        let sizes_ptr = NonNull::new(sizes.as_mut_ptr()).ok_or_else(no_sets)?;
        if codec == HwCodec::H264 {
            // SAFETY: `pointers` and `sizes` hold `sets.len()` entries, each
            // pointer to that many readable bytes borrowed from `config`,
            // all alive for the call (`CoreMedia` copies them); `out` is a
            // valid out-pointer.
            unsafe {
                CMVideoFormatDescriptionCreateFromH264ParameterSets(
                    None,
                    sets.len(),
                    pointers_ptr,
                    sizes_ptr,
                    length,
                    NonNull::from(&mut out),
                )
            }
        } else {
            // SAFETY: as above; no extensions.
            unsafe {
                CMVideoFormatDescriptionCreateFromHEVCParameterSets(
                    None,
                    sets.len(),
                    pointers_ptr,
                    sizes_ptr,
                    length,
                    None,
                    NonNull::from(&mut out),
                )
            }
        }
    } else {
        // AV1 frames come out at the description's size when it is
        // larger than the picture (measured: a 320x200 description over a
        // 160x96 stream gives 320x200 buffers, the picture in one corner), so
        // the sequence header's own maximum wins over the container's
        // claim. A VP9 description of the wrong size is refused by the
        // decoder instead (kVTVideoDecoderBadDataErr), so the container's
        // size stands there.
        let (colour, (width, height)) = match codec {
            HwCodec::Vp9 => (config::vpcc_colour(config)?, (width, height)),
            _ => match config::av1c_sequence(config)? {
                Some(seq) => (seq.colour, dimensions(codec, seq.max_size)?),
                None => (ColourCodes::UNSPECIFIED, (width, height)),
            },
        };
        let extensions = generic_extensions(codec, config, colour);
        // SAFETY: `extensions` is a CFDictionary of CFString keys to
        // property-list values, as the call requires, and is deep-copied by
        // it; `out` is a valid out-pointer.
        unsafe {
            CMVideoFormatDescriptionCreate(
                None,
                codec_type(codec),
                width,
                height,
                Some(extensions.as_opaque()),
                NonNull::from(&mut out),
            )
        }
    };
    if status != 0 {
        return Err(HwError::Failed {
            detail: format!(
                "CoreMedia refused the {codec} configuration record: {}",
                describe(status)
            ),
        });
    }
    let out = NonNull::new(out.cast_mut()).ok_or_else(|| HwError::Failed {
        detail: "CoreMedia returned no format description".into(),
    })?;
    // SAFETY: a +1 reference from a Create function.
    Ok(unsafe { CFRetained::from_raw(out) })
}

/// A picture size as `CoreMedia` takes it, refusing zero and anything past
/// `i32::MAX`.
fn dimensions(codec: HwCodec, (width, height): (u32, u32)) -> Result<(i32, i32), HwError> {
    let check = |v: u32| i32::try_from(v).ok().filter(|v| *v > 0);
    check(width)
        .zip(check(height))
        .ok_or_else(|| HwError::Failed {
            detail: format!("{codec} stream declares a picture size of {width}x{height}"),
        })
}

fn no_sets() -> HwError {
    HwError::Failed {
        detail: "a configuration record with no parameter sets".into(),
    }
}

/// The extensions of a generic (VP9, AV1) format description: the record as
/// a sample-description atom, and the colour description it carries.
fn generic_extensions(
    codec: HwCodec,
    config: &[u8],
    colour: ColourCodes,
) -> CFRetained<CFDictionary<CFString, CFType>> {
    let atom = CFString::from_static_str(match codec {
        HwCodec::Vp9 => "vpcC",
        _ => "av1C",
    });
    let record = CFData::from_bytes(config);
    let atoms = CFDictionary::<CFString, CFType>::from_slices(&[&atom], &[&record]);
    let full_range = CFBoolean::new(colour.full_range);

    // SAFETY: immutable `CoreMedia` constants, initialised when the framework
    // loads.
    let (atoms_key, matrix_key, primaries_key, transfer_key, range_key) = unsafe {
        (
            kCMFormatDescriptionExtension_SampleDescriptionExtensionAtoms,
            kCMFormatDescriptionExtension_YCbCrMatrix,
            kCMFormatDescriptionExtension_ColorPrimaries,
            kCMFormatDescriptionExtension_TransferFunction,
            kCMFormatDescriptionExtension_FullRangeVideo,
        )
    };
    let mut keys: Vec<&CFString> = vec![atoms_key, range_key];
    let mut values: Vec<&CFType> = vec![&atoms, full_range];
    if let Some(matrix) = matrix_name(colour.matrix) {
        keys.push(matrix_key);
        values.push(matrix);
    }
    if let Some(primaries) = primaries_name(colour.primaries) {
        keys.push(primaries_key);
        values.push(primaries);
    }
    if let Some(transfer) = transfer_name(colour.transfer) {
        keys.push(transfer_key);
        values.push(transfer);
    }
    CFDictionary::from_slices(&keys, &values)
}

/// The `CoreMedia` name of an H.273 `matrix_coefficients` code, when it has
/// one; an unnamed code ("unspecified") leaves the choice to the decoder, which
/// picked BT.601 at 160x96.
fn matrix_name(code: u8) -> Option<&'static CFString> {
    // SAFETY: immutable `CoreMedia` constants, initialised when the framework
    // loads.
    unsafe {
        match code {
            1 => Some(kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2),
            5 | 6 => Some(kCMFormatDescriptionYCbCrMatrix_ITU_R_601_4),
            7 => Some(kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995),
            9 | 10 => Some(kCMFormatDescriptionYCbCrMatrix_ITU_R_2020),
            _ => None,
        }
    }
}

/// The `CoreMedia` name of an H.273 `colour_primaries` code.
fn primaries_name(code: u8) -> Option<&'static CFString> {
    // SAFETY: as in `matrix_name`.
    unsafe {
        match code {
            1 => Some(kCMFormatDescriptionColorPrimaries_ITU_R_709_2),
            5 => Some(kCMFormatDescriptionColorPrimaries_EBU_3213),
            6 | 7 => Some(kCMFormatDescriptionColorPrimaries_SMPTE_C),
            9 => Some(kCMFormatDescriptionColorPrimaries_ITU_R_2020),
            11 => Some(kCMFormatDescriptionColorPrimaries_DCI_P3),
            12 => Some(kCMFormatDescriptionColorPrimaries_P3_D65),
            _ => None,
        }
    }
}

/// The `CoreMedia` name of an H.273 `transfer_characteristics` code.
fn transfer_name(code: u8) -> Option<&'static CFString> {
    // SAFETY: as in `matrix_name`.
    unsafe {
        match code {
            1 | 6 | 14 | 15 => Some(kCMFormatDescriptionTransferFunction_ITU_R_709_2),
            7 => Some(kCMFormatDescriptionTransferFunction_SMPTE_240M_1995),
            8 => Some(kCMFormatDescriptionTransferFunction_Linear),
            13 => Some(kCMFormatDescriptionTransferFunction_sRGB),
            16 => Some(kCMFormatDescriptionTransferFunction_SMPTE_ST_2084_PQ),
            17 => Some(kCMFormatDescriptionTransferFunction_SMPTE_ST_428_1),
            18 => Some(kCMFormatDescriptionTransferFunction_ITU_R_2100_HLG),
            _ => None,
        }
    }
}

fn codec_type(codec: HwCodec) -> CMVideoCodecType {
    match codec {
        HwCodec::H264 => kCMVideoCodecType_H264,
        HwCodec::Hevc => kCMVideoCodecType_HEVC,
        HwCodec::Vp9 => kCMVideoCodecType_VP9,
        HwCodec::Av1 => kCMVideoCodecType_AV1,
    }
}

/// VP9 and AV1 decoders ship with the OS but are not registered until asked.
fn register_supplemental_decoders() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        for codec in [HwCodec::Vp9, HwCodec::Av1] {
            // SAFETY: takes a codec type by value; a codec with no supplemental
            // decoder is a no-op.
            unsafe { VTRegisterSupplementalVideoDecoderIfAvailable(codec_type(codec)) };
        }
    });
}

fn invalid_time() -> CMTime {
    CMTime {
        value: 0,
        timescale: 0,
        flags: CMTimeFlags::empty(),
        epoch: 0,
    }
}

/// An `OSStatus` with its `VideoToolbox` name where it has one.
fn describe(status: i32) -> String {
    let name = match status {
        -12710 => "kCMFormatDescriptionError_InvalidParameter",
        -12902 => "kVTParameterErr",
        -12903 => "kVTInvalidSessionErr",
        -12904 => "kVTAllocationFailedErr",
        -12906 => "kVTCouldNotFindVideoDecoderErr",
        -12907 => "kVTCouldNotCreateInstanceErr",
        -12909 => "kVTVideoDecoderBadDataErr",
        -12910 => "kVTVideoDecoderUnsupportedDataFormatErr",
        -12911 => "kVTVideoDecoderMalfunctionErr",
        -12913 => "kVTVideoDecoderNotAvailableNowErr",
        -17694 => "kVTVideoDecoderReferenceMissingErr",
        -17696 => "kVTVideoDecoderUnknownErr",
        _ => return format!("OSStatus {status}"),
    };
    format!("{name} ({status})")
}

#[expect(
    clippy::unnecessary_wraps,
    reason = "the platform contract: a platform without an OS decoder answers Unavailable"
)]
pub(crate) fn probe() -> Result<HwProbe, HwError> {
    register_supplemental_decoders();
    let codecs = [HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1]
        .into_iter()
        // SAFETY: takes a codec type by value and only reports.
        .filter(|codec| unsafe { VTIsHardwareDecodeSupported(codec_type(*codec)) })
        .collect();
    Ok(HwProbe {
        decoder: "VideoToolbox",
        codecs,
    })
}
