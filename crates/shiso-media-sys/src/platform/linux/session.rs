//! One NVDEC decode: a parser that turns the elementary stream into pictures,
//! the decoder it creates, and the frames copied out in display order.
//!
//! The parser drives everything through three callbacks, called synchronously
//! from inside `cuvidParseVideoData` on the thread that called it:
//!
//! - **sequence**: the stream's format is known (or changed). The format is
//!   checked against the decoder's capabilities and the decoder is created for it.
//! - **decode**: a picture's slices are ready; they go to `cuvidDecodePicture`.
//! - **display**: a picture is next in display order. A kept picture is mapped,
//!   copied to host memory and unmapped right there, before the parser can reuse
//!   its surface for a later picture; colour conversion waits until the parse
//!   call returns.
//!
//! The callbacks reach the session's state through the parser's user-data
//! pointer, so that state lives in its own heap allocation that is only ever
//! touched through a raw pointer, and no Rust reference to it is held across a
//! parse call. A callback never unwinds into the C parser: a panic is caught and
//! recorded, and the callback returns 0, which makes the parse call fail.

use std::collections::HashMap;
use std::ffi::{c_int, c_ulong, c_void};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::ptr::{self, NonNull};

use super::annexb::{avcc_param_sets, hvcc_param_sets, length_prefixed_to_annexb};
use super::config::{StreamFormat, av1_config_obus, stream_format};
use super::driver::{Driver, PrimaryContext, driver};
use super::ffi::{
    CHROMA_420, CODEC_AV1, CODEC_H264, CODEC_HEVC, CODEC_VP9, CREATE_PREFER_CUVID, CuResult,
    CuVideoFormat, CuvidDecodeCaps, CuvidDecodeCreateInfo, CuvidParserDispInfo, CuvidParserParams,
    CuvidPicParams, CuvidProcParams, CuvidSourceDataPacket, DEINTERLACE_ADAPTIVE,
    DEINTERLACE_WEAVE, PKT_ENDOFSTREAM, PKT_TIMESTAMP, RectShort,
};
use super::semaphore::{Permit, Semaphore};
use super::surface::{
    HostSurface, SurfaceGeometry, check, check_caps, require_420, surface_format,
};
use crate::{HwCodec, HwDecoderDesc, HwError, HwFrame, HwProbe, HwTimestamp};

/// Concurrent decoder sessions per process. A GB10 has one NVDEC engine that
/// sessions time-share; more sessions than this add surface memory without
/// adding throughput, so an extra `open` waits for one to close.
pub(crate) const MAX_NVDEC_SESSIONS: usize = 4;

/// The process-wide session cap.
static SESSIONS: Semaphore = Semaphore::new(MAX_NVDEC_SESSIONS);

/// Decode surfaces when the parser does not report a minimum: enough for the
/// largest H.264/HEVC reference set (16) plus the picture being decoded and the
/// ones queued for display.
const FALLBACK_DECODE_SURFACES: u32 = 20;

/// Output surfaces that can be mapped at once. One is mapped at a time here; the
/// second lets the decoder post-process the next picture meanwhile.
const OUTPUT_SURFACES: c_ulong = 2;

/// The `cudaVideoCodec` value for a codec.
fn codec_id(codec: HwCodec) -> c_int {
    match codec {
        HwCodec::H264 => CODEC_H264,
        HwCodec::Hevc => CODEC_HEVC,
        HwCodec::Vp9 => CODEC_VP9,
        HwCodec::Av1 => CODEC_AV1,
    }
}

/// A capability query for `codec` at `format`.
fn caps_query(codec: HwCodec, format: StreamFormat) -> CuvidDecodeCaps {
    CuvidDecodeCaps {
        codec_type: codec_id(codec),
        chroma_format: format.chroma,
        bit_depth_minus8: format.bit_depth.saturating_sub(8),
        ..CuvidDecodeCaps::default()
    }
}

/// A created decoder and the configuration it was created for.
#[derive(Debug)]
struct Decoder {
    handle: NonNull<c_void>,
    geometry: SurfaceGeometry,
    surfaces: u32,
    progressive: bool,
}

/// Everything the parser callbacks read and write.
#[derive(Debug)]
struct ParserState {
    driver: &'static Driver,
    codec: HwCodec,
    decoder: Option<Decoder>,
    /// Timestamps of samples the caller keeps, counted, filled by `decode` in
    /// decode order and consumed by the display callback in display order.
    keep: HashMap<i64, u32>,
    /// Kept pictures copied to host memory, in display order.
    ready: Vec<HostSurface>,
    /// The first error a callback hit. Sticky: after it the session only
    /// reports it.
    error: Option<HwError>,
}

impl ParserState {
    /// Whether the picture stamped `pts` is kept, consuming one keep.
    fn take_keep(&mut self, pts: i64) -> bool {
        match self.keep.get_mut(&pts) {
            Some(n) if *n > 1 => {
                *n -= 1;
                true
            }
            Some(_) => {
                self.keep.remove(&pts);
                true
            }
            None => false,
        }
    }

    /// The sequence callback: (re)creates the decoder for `format` and returns
    /// the decode-surface count, which the parser takes as its picture buffer
    /// size when it is above 1.
    fn on_sequence(&mut self, format: &CuVideoFormat) -> Result<c_int, HwError> {
        let geometry = SurfaceGeometry::from_format(self.codec, format)?;
        let stream = StreamFormat {
            chroma: CHROMA_420,
            bit_depth: geometry.bit_depth,
        };
        let mut caps = caps_query(self.codec, stream);
        self.driver.decoder_caps(self.codec, &mut caps)?;
        check_caps(
            self.codec,
            &caps,
            stream,
            geometry.coded_width,
            geometry.coded_height,
        )?;
        let surfaces = match format.min_num_decode_surfaces {
            0 => FALLBACK_DECODE_SURFACES,
            n => u32::from(n),
        };
        let progressive = format.progressive_sequence != 0;
        let unchanged = self.decoder.as_ref().is_some_and(|d| {
            d.geometry == geometry && d.surfaces == surfaces && d.progressive == progressive
        });
        if !unchanged {
            self.destroy_decoder();
            self.decoder = Some(self.create_decoder(geometry, surfaces, progressive)?);
        }
        Ok(c_int::try_from(surfaces).unwrap_or(c_int::MAX))
    }

    fn create_decoder(
        &self,
        geometry: SurfaceGeometry,
        surfaces: u32,
        progressive: bool,
    ) -> Result<Decoder, HwError> {
        let short = |v: u32, what: &str| {
            i16::try_from(v).map_err(|_| HwError::Unsupported {
                codec: self.codec,
                reason: format!("a display {what} of {v} does not fit the decoder's 16-bit field"),
            })
        };
        let g = &geometry;
        let mut info = CuvidDecodeCreateInfo {
            width: c_ulong::from(g.coded_width),
            height: c_ulong::from(g.coded_height),
            num_decode_surfaces: c_ulong::from(surfaces),
            codec_type: codec_id(self.codec),
            chroma_format: CHROMA_420,
            creation_flags: CREATE_PREFER_CUVID,
            bit_depth_minus8: c_ulong::from(g.bit_depth - 8),
            intra_decode_only: 0,
            max_width: c_ulong::from(g.coded_width),
            max_height: c_ulong::from(g.coded_height),
            reserved1: 0,
            display_area: RectShort {
                left: short(g.left, "left edge")?,
                top: short(g.top, "top edge")?,
                right: short(g.left + g.width, "right edge")?,
                bottom: short(g.top + g.height, "bottom edge")?,
            },
            output_format: surface_format(g.bit_depth),
            deinterlace_mode: if progressive {
                DEINTERLACE_WEAVE
            } else {
                DEINTERLACE_ADAPTIVE
            },
            target_width: c_ulong::from(g.target_width),
            target_height: c_ulong::from(g.target_height),
            num_output_surfaces: OUTPUT_SURFACES,
            vid_lock: ptr::null_mut(),
            // An empty rectangle means the whole target.
            target_rect: RectShort::default(),
            enable_histogram: 0,
            enable_decode_features: 0,
            reserved2: [0; 3],
        };
        let mut handle: *mut c_void = ptr::null_mut();
        // SAFETY: both pointers are valid for the call; the primary context is
        // current, because this runs inside a parse call made with it pushed.
        let rc = unsafe { (self.driver.create_decoder)(&raw mut handle, &raw mut info) };
        check("cuvidCreateDecoder", rc, Some(self.codec))?;
        let handle = NonNull::new(handle).ok_or_else(|| HwError::Failed {
            detail: "cuvidCreateDecoder returned a null decoder".into(),
        })?;
        Ok(Decoder {
            handle,
            geometry,
            surfaces,
            progressive,
        })
    }

    /// Destroys the decoder, if there is one. The context must be current.
    fn destroy_decoder(&mut self) {
        if let Some(decoder) = self.decoder.take() {
            // SAFETY: the handle came from cuvidCreateDecoder and was taken out
            // of the state, so this is its only destroy.
            let rc = unsafe { (self.driver.destroy_decoder)(decoder.handle.as_ptr()) };
            if rc != CuResult::SUCCESS {
                tracing::warn!(code = rc.0, "cuvidDestroyDecoder failed");
            }
        }
    }

    /// The decode callback: hands the parser's picture to the decoder.
    fn on_decode(&mut self, params: NonNull<CuvidPicParams>) -> Result<c_int, HwError> {
        let decoder = self.decoder.as_ref().ok_or_else(|| HwError::Failed {
            detail: "the stream has a picture before its sequence header".into(),
        })?;
        // SAFETY: the decoder is live, `params` is the parser's own picture
        // parameter block for this callback, forwarded untouched, and the
        // context is current inside the parse call.
        let rc = unsafe { (self.driver.decode_picture)(decoder.handle.as_ptr(), params.as_ptr()) };
        check("cuvidDecodePicture", rc, Some(self.codec))?;
        Ok(1)
    }

    /// The display callback: copies a kept picture off the device.
    fn on_display(&mut self, info: &CuvidParserDispInfo) -> Result<c_int, HwError> {
        if !self.take_keep(info.timestamp) {
            return Ok(1);
        }
        let decoder = self.decoder.as_ref().ok_or_else(|| HwError::Failed {
            detail: "a picture to display before the decoder exists".into(),
        })?;
        if info.picture_index < 0 {
            return Err(HwError::Failed {
                detail: format!("the parser displayed picture index {}", info.picture_index),
            });
        }
        let geometry = decoder.geometry;
        let mut params = CuvidProcParams {
            progressive_frame: info.progressive_frame,
            second_field: 0,
            top_field_first: info.top_field_first,
            unpaired_field: c_int::from(info.repeat_first_field < 0),
            reserved_flags: 0,
            reserved_zero: 0,
            raw_input_dptr: 0,
            raw_input_pitch: 0,
            raw_input_format: 0,
            raw_output_dptr: 0,
            raw_output_pitch: 0,
            reserved1: 0,
            output_stream: ptr::null_mut(),
            reserved: [0; 46],
            histogram_dptr: ptr::null_mut(),
            cuvid_proc_ext: ptr::null_mut(),
        };
        let mut surface = 0u64;
        let mut pitch = 0u32;
        // SAFETY: the decoder is live, the out-pointers and params are valid for
        // the call, the picture index came from the parser, and the context is
        // current inside the parse call.
        let rc = unsafe {
            (self.driver.map_frame)(
                decoder.handle.as_ptr(),
                info.picture_index,
                &raw mut surface,
                &raw mut pitch,
                &raw mut params,
            )
        };
        check("cuvidMapVideoFrame64", rc, Some(self.codec))?;
        let row_bytes = geometry.row_bytes();
        let mut bytes = vec![0u8; row_bytes * geometry.rows()];
        // The chroma plane starts `target_height` rows (an even count) below the
        // luma plane at the same pitch, so one copy of luma-plus-chroma rows
        // reads both planes.
        let copied = usize::try_from(pitch)
            .map_err(|_| HwError::Failed {
                detail: format!("a surface pitch of {pitch}"),
            })
            .and_then(|pitch| {
                self.driver
                    .copy_to_host(surface, pitch, row_bytes, geometry.rows(), &mut bytes)
            });
        // SAFETY: `surface` is the pointer this callback just mapped on this
        // decoder; it is unmapped exactly once, whether or not the copy worked.
        let rc = unsafe { (self.driver.unmap_frame)(decoder.handle.as_ptr(), surface) };
        copied?;
        check("cuvidUnmapVideoFrame64", rc, Some(self.codec))?;
        self.ready.push(HostSurface {
            pts: info.timestamp,
            geometry,
            bytes,
        });
        Ok(1)
    }
}

/// Runs a callback body against the state behind `user`, turning an error or a
/// panic into a recorded error and a 0 return, the parser's failure code.
///
/// # Safety
///
/// `user` must be null or the session's state pointer, with no other reference
/// to the state live for the duration of the call.
unsafe fn with_state(
    user: *mut c_void,
    body: impl FnOnce(&mut ParserState) -> Result<c_int, HwError>,
) -> c_int {
    // SAFETY: per the contract, `user` is the session's state allocation and
    // nothing else references it while the parse call runs.
    let Some(state) = (unsafe { user.cast::<ParserState>().as_mut() }) else {
        return 0;
    };
    if state.error.is_some() {
        return 0;
    }
    let outcome = catch_unwind(AssertUnwindSafe(|| body(&mut *state)));
    let err = match outcome {
        Ok(Ok(n)) => return n,
        Ok(Err(err)) => err,
        Err(_) => HwError::Failed {
            detail: "an NVDEC parser callback panicked".into(),
        },
    };
    state.error = Some(err);
    0
}

/// `PFNVIDSEQUENCECALLBACK`.
///
/// # Safety
///
/// Called by the parser with the user data it was created with and a format
/// valid for the call.
unsafe extern "C" fn sequence_trampoline(user: *mut c_void, format: *mut CuVideoFormat) -> c_int {
    let body = |state: &mut ParserState| {
        // SAFETY: the parser's format block is valid for the callback.
        let format = unsafe { format.as_ref() }.ok_or_else(|| HwError::Failed {
            detail: "the parser reported a null sequence format".into(),
        })?;
        state.on_sequence(format)
    };
    // SAFETY: the parser passes back the state pointer it was created with,
    // during a parse call, when the session holds no reference to the state.
    unsafe { with_state(user, body) }
}

/// `PFNVIDDECODECALLBACK`.
///
/// # Safety
///
/// As [`sequence_trampoline`], with a picture parameter block valid for the call.
unsafe extern "C" fn decode_trampoline(user: *mut c_void, params: *mut CuvidPicParams) -> c_int {
    // SAFETY: as in `sequence_trampoline`.
    unsafe {
        with_state(user, |state| {
            let params = NonNull::new(params).ok_or_else(|| HwError::Failed {
                detail: "the parser handed over a null picture".into(),
            })?;
            state.on_decode(params)
        })
    }
}

/// `PFNVIDDISPLAYCALLBACK`.
///
/// # Safety
///
/// As [`sequence_trampoline`], with a display record valid for the call or null
/// (an end-of-stream marker, which needs nothing).
unsafe extern "C" fn display_trampoline(
    user: *mut c_void,
    info: *mut CuvidParserDispInfo,
) -> c_int {
    let body = |state: &mut ParserState| {
        // SAFETY: the parser's display record is valid for the callback.
        match unsafe { info.as_ref() } {
            Some(info) => state.on_display(info),
            None => Ok(1),
        }
    };
    // SAFETY: as in `sequence_trampoline`.
    unsafe { with_state(user, body) }
}

/// An NVDEC session.
#[derive(Debug)]
pub(crate) struct Session {
    codec: HwCodec,
    /// The callbacks' state, owned here and freed in `Drop`; reached only
    /// through this pointer.
    state: NonNull<ParserState>,
    /// Null until the parser is created.
    parser: *mut c_void,
    /// Bytes in each sample's NAL length prefix, for H.264 and HEVC.
    nal_length_size: Option<usize>,
    /// What goes in front of the first sample: the configuration record's
    /// parameter sets (H.264, HEVC) or OBUs (AV1). Emptied once fed.
    prefix: Vec<u8>,
    /// Reused for each packet's bytes.
    packet: Vec<u8>,
    ctx: PrimaryContext,
    _permit: Permit<'static>,
}

// SAFETY: the session's handles (parser, decoder, context) are not tied to the
// thread that created them: the driver only requires the context to be current
// on whichever thread makes a call, and every call here pushes it first. The
// state is reached only through `&mut self` or `Drop`, so moving the session to
// another thread moves sole access with it. It is not `Sync`.
unsafe impl Send for Session {}

impl Session {
    pub(crate) fn open(desc: &HwDecoderDesc<'_>) -> Result<Self, HwError> {
        let codec = desc.codec;
        let driver = driver()?;
        if desc.width == 0 || desc.height == 0 {
            return Err(HwError::Failed {
                detail: format!("a {}x{} {codec} stream", desc.width, desc.height),
            });
        }
        // The record is parsed in full first, so a malformed one is refused
        // rather than read as a format hint and passed on as Unsupported.
        let (nal_length_size, prefix) = match codec {
            HwCodec::H264 => {
                let sets = avcc_param_sets(desc.config)?;
                (Some(sets.nal_length_size), sets.prefix)
            }
            HwCodec::Hevc => {
                let sets = hvcc_param_sets(desc.config)?;
                (Some(sets.nal_length_size), sets.prefix)
            }
            HwCodec::Vp9 => (None, Vec::new()),
            HwCodec::Av1 => (None, av1_config_obus(desc.config).to_vec()),
        };
        let format = stream_format(codec, desc.config).unwrap_or(StreamFormat::BASELINE);
        require_420(codec, format.chroma)?;
        // Taken only once the driver is known to be there, so a machine without
        // NVDEC never waits on it.
        let permit = SESSIONS.acquire();
        let ctx = PrimaryContext::retain(driver)?;
        {
            let _current = ctx.push()?;
            let mut caps = caps_query(codec, format);
            driver.decoder_caps(codec, &mut caps)?;
            check_caps(codec, &caps, format, desc.width, desc.height)?;
        }
        let state = Box::new(ParserState {
            driver,
            codec,
            decoder: None,
            keep: HashMap::new(),
            ready: Vec::new(),
            error: None,
        });
        // From here `Drop` frees the state (and the parser, once it exists) on
        // every path.
        let mut session = Self {
            codec,
            state: NonNull::from(Box::leak(state)),
            parser: ptr::null_mut(),
            nal_length_size,
            prefix,
            packet: Vec::new(),
            ctx,
            _permit: permit,
        };
        session.parser = session.create_parser()?;
        Ok(session)
    }

    fn create_parser(&self) -> Result<*mut c_void, HwError> {
        let mut params = CuvidParserParams {
            codec_type: codec_id(self.codec),
            // The sequence callback's return value replaces this once the
            // stream's own requirement is known.
            max_num_decode_surfaces: 1,
            // 0 is the default 10 MHz clock; the timestamps are passed through
            // untouched, so the unit does not matter here.
            clock_rate: 0,
            // Do not decode a picture whose bitstream is corrupt at all.
            error_threshold: 0,
            max_display_delay: 0,
            flags: 0,
            reserved1: [0; 4],
            user_data: self.state.as_ptr().cast::<c_void>(),
            sequence_callback: Some(sequence_trampoline),
            decode_picture: Some(decode_trampoline),
            display_picture: Some(display_trampoline),
            get_operating_point: ptr::null_mut(),
            get_sei_msg: ptr::null_mut(),
            reserved2: [ptr::null_mut(); 5],
            ext_video_info: ptr::null_mut(),
        };
        let driver = self.ctx.driver();
        let mut parser: *mut c_void = ptr::null_mut();
        let rc = {
            let _current = self.ctx.push()?;
            // SAFETY: both pointers are valid for the call; the user data is
            // the state allocation, which outlives the parser (Drop destroys the
            // parser before freeing it).
            unsafe { (driver.create_parser)(&raw mut parser, &raw mut params) }
        };
        check("cuvidCreateVideoParser", rc, Some(self.codec))?;
        if parser.is_null() {
            return Err(HwError::Failed {
                detail: "cuvidCreateVideoParser returned a null parser".into(),
            });
        }
        Ok(parser)
    }

    /// The callbacks' state, for use between parse calls.
    fn state(&mut self) -> &mut ParserState {
        // SAFETY: the allocation lives until Drop, and callbacks only run inside
        // `feed`, which holds no reference from here across the parse call; the
        // `&mut self` borrow makes this the only reference.
        unsafe { self.state.as_mut() }
    }

    /// Feeds one packet to the parser, running whatever callbacks it triggers.
    fn feed(&mut self, flags: c_ulong, pts: i64) -> Result<(), HwError> {
        if let Some(err) = &self.state().error {
            return Err(err.clone());
        }
        let payload_size = c_ulong::try_from(self.packet.len()).map_err(|_| HwError::Failed {
            detail: format!("a {}-byte sample", self.packet.len()),
        })?;
        let mut packet = CuvidSourceDataPacket {
            flags,
            payload_size,
            payload: if self.packet.is_empty() {
                ptr::null()
            } else {
                self.packet.as_ptr()
            },
            timestamp: pts,
        };
        let driver = self.ctx.driver();
        let rc = {
            let _current = self.ctx.push()?;
            // SAFETY: the parser is live; the packet and its payload (`self.packet`,
            // not touched until the call returns) are valid for the call; no
            // reference to the callback state is held across it.
            unsafe { (driver.parse_data)(self.parser, &raw mut packet) }
        };
        // A callback's own error says more than the parse call's code.
        if let Some(err) = &self.state().error {
            return Err(err.clone());
        }
        check("cuvidParseVideoData", rc, Some(self.codec))
    }

    /// Converts the pictures the display callback copied out.
    fn drain(&mut self) -> Result<Vec<HwFrame>, HwError> {
        let ready = std::mem::take(&mut self.state().ready);
        ready
            .into_iter()
            .map(|surface| {
                let rgb = surface.to_rgb()?;
                HwFrame::new(
                    surface.geometry.width,
                    surface.geometry.height,
                    rgb,
                    HwTimestamp::new(surface.pts),
                )
            })
            .collect()
    }

    pub(crate) fn decode(
        &mut self,
        sample: &[u8],
        pts: HwTimestamp,
        keep: bool,
    ) -> Result<Vec<HwFrame>, HwError> {
        if sample.is_empty() {
            return Err(HwError::Failed {
                detail: "an empty sample".into(),
            });
        }
        let mut packet = std::mem::take(&mut self.packet);
        packet.clear();
        packet.extend_from_slice(&self.prefix);
        let built = if let Some(size) = self.nal_length_size {
            length_prefixed_to_annexb(sample, size, &mut packet)
        } else {
            packet.extend_from_slice(sample);
            Ok(())
        };
        self.packet = packet;
        built?;
        if keep {
            *self.state().keep.entry(pts.ticks()).or_insert(0) += 1;
        }
        self.feed(PKT_TIMESTAMP, pts.ticks())?;
        self.prefix.clear();
        self.drain()
    }

    pub(crate) fn flush(&mut self) -> Result<Vec<HwFrame>, HwError> {
        self.packet.clear();
        self.feed(PKT_ENDOFSTREAM, 0)?;
        let frames = self.drain()?;
        // Every kept sample should have come back through the display callback
        // by now, matched on its timestamp. A leftover means the parser
        // displayed pictures under timestamps it was not given - a stream whose
        // frames would silently vanish - so it is an error, not an empty answer.
        let missing: u32 = self.state().keep.drain().map(|(_, n)| n).sum();
        if missing > 0 {
            return Err(HwError::Failed {
                detail: format!(
                    "{missing} kept frame(s) never came out of the NVDEC parser under \
                     the timestamps they were fed with"
                ),
            });
        }
        Ok(frames)
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        let driver = self.ctx.driver();
        // The parser is destroyed first: it holds the state pointer.
        let mut parser_gone = true;
        if !self.parser.is_null() {
            // SAFETY: the parser came from cuvidCreateVideoParser and this is its
            // only destroy; no parse call is running (`&mut self`).
            let rc = unsafe { (driver.destroy_parser)(self.parser) };
            if rc != CuResult::SUCCESS {
                tracing::warn!(code = rc.0, "cuvidDestroyVideoParser failed");
                parser_gone = false;
            }
            self.parser = ptr::null_mut();
        }
        let state = self.state;
        match self.ctx.push() {
            Ok(_current) => {
                // SAFETY: the allocation is live until freed below, the parser
                // that could call into it is gone or idle, and `&mut self` makes
                // this the only reference.
                unsafe { (*state.as_ptr()).destroy_decoder() };
            }
            Err(err) => {
                tracing::warn!(%err, "cannot make the CUDA context current; the decoder is leaked");
            }
        }
        if parser_gone {
            // SAFETY: the pointer came from `Box::leak` in `open`, nothing else
            // holds it now that the parser is destroyed, and this is the only
            // place it is turned back into a Box.
            drop(unsafe { Box::from_raw(self.state.as_ptr()) });
        } else {
            // A parser that would not die may still call back into the state, so
            // the state is leaked rather than freed under it.
            tracing::warn!("the NVDEC parser was not destroyed; its state is leaked");
        }
        // `ctx` (the context retain) and `_permit` drop after this, in that order.
    }
}

pub(crate) fn probe() -> Result<HwProbe, HwError> {
    let driver = driver()?;
    let ctx = PrimaryContext::retain(driver)?;
    let current = ctx.push()?;
    let mut codecs = Vec::new();
    for codec in [HwCodec::H264, HwCodec::Hevc, HwCodec::Vp9, HwCodec::Av1] {
        let mut caps = caps_query(codec, StreamFormat::BASELINE);
        match driver.decoder_caps(codec, &mut caps) {
            Ok(()) if caps.is_supported != 0 => codecs.push(codec),
            Ok(()) | Err(HwError::Unsupported { .. }) => {}
            Err(err) => return Err(err),
        }
    }
    drop(current);
    Ok(HwProbe {
        decoder: "NVDEC",
        codecs,
    })
}
