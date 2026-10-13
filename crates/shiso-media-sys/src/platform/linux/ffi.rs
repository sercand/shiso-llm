//! Hand-written bindings for the part of the CUDA driver and NVDEC API this
//! crate calls.
//!
//! Transcribed from the NVDEC API headers (`cuviddec.h`, `nvcuvid.h`,
//! `NVDECAPI` 13.1) and the CUDA driver header, for 64-bit Linux only: `tcu_ulong`
//! is `unsigned long` (8 bytes on LP64), `CUdeviceptr` is `unsigned long long`
//! (the headers' 64-bit device-pointer branch, taken whenever `__LP64__` is
//! defined, which is why the 64-bit map and unmap entry points are the ones
//! bound), and bitfields are allocated from the least significant bit, as the
//! System V ABI lays them out on `x86_64` and `aarch64`.
//!
//! Nothing here was checked against a running driver. Each struct's size and
//! every offset this crate reads or writes are asserted at compile time below
//! (so `cargo check --target {x86_64,aarch64}-unknown-linux-gnu` proves the
//! transcription on both architectures without running anything), against
//! numbers derived by hand from the header: walk the fields in order, align each
//! to its own size (`int`/enums 4, `unsigned long`/pointers/`long long` 8,
//! `short` 2, `char` 1), and round the struct up to its largest alignment.
//!
//! The driver writes C enums (`cudaVideoCodec`, `cudaVideoChromaFormat`,
//! `CUresult`) back into these structs, so none of them is a Rust `enum`: a value
//! a newer driver adds would be undefined behaviour in a Rust enum and is just an
//! unknown integer here.
//!
//! The structs are plain data; constructing one is safe and zeroed is the
//! headers' documented "unset" for every reserved field.

use std::ffi::{c_int, c_long, c_short, c_uint, c_ulong, c_ulonglong, c_ushort, c_void};

// --- CUDA driver ---------------------------------------------------------

/// A `CUresult`: 0 is success, anything else is an error code.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub(super) struct CuResult(pub(super) c_int);

impl CuResult {
    /// `CUDA_SUCCESS`.
    pub(super) const SUCCESS: Self = Self(0);
}

/// `CUdevice`: a device ordinal handle.
pub(super) type CuDevice = c_int;
/// `CUcontext`: an opaque context handle.
pub(super) type CuContext = *mut c_void;
/// `CUdeviceptr` on a 64-bit build.
pub(super) type CuDevicePtr = c_ulonglong;

/// `CU_MEMORYTYPE_HOST`.
pub(super) const CU_MEMORYTYPE_HOST: c_uint = 1;
/// `CU_MEMORYTYPE_DEVICE`.
pub(super) const CU_MEMORYTYPE_DEVICE: c_uint = 2;

/// `CUDA_MEMCPY2D`: one pitched 2D copy.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct CudaMemcpy2d {
    pub(super) src_x_in_bytes: usize,
    pub(super) src_y: usize,
    pub(super) src_memory_type: c_uint,
    pub(super) src_host: *const c_void,
    pub(super) src_device: CuDevicePtr,
    pub(super) src_array: *mut c_void,
    pub(super) src_pitch: usize,
    pub(super) dst_x_in_bytes: usize,
    pub(super) dst_y: usize,
    pub(super) dst_memory_type: c_uint,
    pub(super) dst_host: *mut c_void,
    pub(super) dst_device: CuDevicePtr,
    pub(super) dst_array: *mut c_void,
    pub(super) dst_pitch: usize,
    pub(super) width_in_bytes: usize,
    pub(super) height: usize,
}

// --- NVDEC enums (as integers) --------------------------------------------

/// `cudaVideoCodec_H264`.
pub(super) const CODEC_H264: c_int = 4;
/// `cudaVideoCodec_HEVC`.
pub(super) const CODEC_HEVC: c_int = 8;
/// `cudaVideoCodec_VP9`.
pub(super) const CODEC_VP9: c_int = 10;
/// `cudaVideoCodec_AV1`.
pub(super) const CODEC_AV1: c_int = 11;

/// `cudaVideoChromaFormat_Monochrome`.
pub(super) const CHROMA_MONOCHROME: c_int = 0;
/// `cudaVideoChromaFormat_420`.
pub(super) const CHROMA_420: c_int = 1;
/// `cudaVideoChromaFormat_422`.
pub(super) const CHROMA_422: c_int = 2;
/// `cudaVideoChromaFormat_444`.
pub(super) const CHROMA_444: c_int = 3;

/// `cudaVideoSurfaceFormat_NV12`: 8-bit Y plane then interleaved UV.
pub(super) const SURFACE_NV12: c_int = 0;
/// `cudaVideoSurfaceFormat_P016`: the same with 16-bit samples, the significant
/// bits at the top.
pub(super) const SURFACE_P016: c_int = 1;

/// `cudaVideoDeinterlaceMode_Weave`.
pub(super) const DEINTERLACE_WEAVE: c_int = 0;
/// `cudaVideoDeinterlaceMode_Adaptive`.
pub(super) const DEINTERLACE_ADAPTIVE: c_int = 2;

/// `cudaVideoCreate_PreferCUVID`: the dedicated decode engine.
pub(super) const CREATE_PREFER_CUVID: c_ulong = 0x04;

/// `CUVID_PKT_ENDOFSTREAM`.
pub(super) const PKT_ENDOFSTREAM: c_ulong = 0x01;
/// `CUVID_PKT_TIMESTAMP`.
pub(super) const PKT_TIMESTAMP: c_ulong = 0x02;

// --- NVDEC structs --------------------------------------------------------

/// `CUVIDDECODECAPS`: in-out for `cuvidGetDecoderCaps`.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(super) struct CuvidDecodeCaps {
    pub(super) codec_type: c_int,
    pub(super) chroma_format: c_int,
    pub(super) bit_depth_minus8: c_uint,
    pub(super) reserved1: [c_uint; 3],
    pub(super) is_supported: u8,
    pub(super) num_nvdecs: u8,
    pub(super) output_format_mask: c_ushort,
    pub(super) max_width: c_uint,
    pub(super) max_height: c_uint,
    pub(super) max_mb_count: c_uint,
    pub(super) min_width: c_ushort,
    pub(super) min_height: c_ushort,
    pub(super) is_histogram_supported: u8,
    pub(super) counter_bit_depth: u8,
    pub(super) max_histogram_bins: c_ushort,
    pub(super) is_decode_stats_supported: u8,
    pub(super) reserved4: [u8; 3],
    pub(super) reserved3: [c_uint; 9],
}

/// A rectangle of `int`s (`CUVIDEOFORMAT::display_area`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub(super) struct RectInt {
    pub(super) left: c_int,
    pub(super) top: c_int,
    pub(super) right: c_int,
    pub(super) bottom: c_int,
}

/// A rectangle of `short`s (`CUVIDDECODECREATEINFO::display_area`, `target_rect`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[repr(C)]
pub(super) struct RectShort {
    pub(super) left: c_short,
    pub(super) top: c_short,
    pub(super) right: c_short,
    pub(super) bottom: c_short,
}

/// `CUVIDEOFORMAT::video_signal_description`. The first byte is the bitfields
/// `video_format : 3`, `video_full_range_flag : 1`, `reserved_zero_bits : 4`,
/// least significant first.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(super) struct VideoSignalDescription {
    pub(super) bits: u8,
    pub(super) color_primaries: u8,
    pub(super) transfer_characteristics: u8,
    pub(super) matrix_coefficients: u8,
}

impl VideoSignalDescription {
    /// `video_full_range_flag`, bit 3 of the bitfield byte.
    pub(super) fn full_range(self) -> bool {
        (self.bits >> 3) & 1 == 1
    }
}

/// `CUVIDEOFORMAT`: what the parser's sequence callback reports.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(super) struct CuVideoFormat {
    pub(super) codec: c_int,
    pub(super) frame_rate_numerator: c_uint,
    pub(super) frame_rate_denominator: c_uint,
    pub(super) progressive_sequence: u8,
    pub(super) bit_depth_luma_minus8: u8,
    pub(super) bit_depth_chroma_minus8: u8,
    pub(super) min_num_decode_surfaces: u8,
    pub(super) coded_width: c_uint,
    pub(super) coded_height: c_uint,
    pub(super) display_area: RectInt,
    pub(super) chroma_format: c_int,
    pub(super) bitrate: c_uint,
    pub(super) display_aspect_ratio_x: c_int,
    pub(super) display_aspect_ratio_y: c_int,
    pub(super) video_signal_description: VideoSignalDescription,
    pub(super) seqhdr_data_length: c_uint,
}

/// `CUVIDSOURCEDATAPACKET`: one buffer handed to `cuvidParseVideoData`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct CuvidSourceDataPacket {
    pub(super) flags: c_ulong,
    pub(super) payload_size: c_ulong,
    pub(super) payload: *const u8,
    /// `CUvideotimestamp`, a `long long`.
    pub(super) timestamp: i64,
}

/// `CUVIDPARSERDISPINFO`: one picture the display callback hands over.
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub(super) struct CuvidParserDispInfo {
    pub(super) picture_index: c_int,
    pub(super) progressive_frame: c_int,
    pub(super) top_field_first: c_int,
    pub(super) repeat_first_field: c_int,
    pub(super) timestamp: i64,
}

/// The leading fields of `CUVIDPICPARAMS`.
///
/// The decode callback receives a pointer to the parser's own, full structure
/// (a codec-specific union of several kilobytes follows `reserved`) and forwards
/// it untouched to `cuvidDecodePicture`. This type is never constructed and its
/// size is meaningless; only the fields here may be read through the pointer, and
/// their offsets are asserted below.
#[derive(Debug)]
#[repr(C)]
pub(super) struct CuvidPicParams {
    pub(super) pic_width_in_mbs: c_int,
    pub(super) frame_height_in_mbs: c_int,
    pub(super) curr_pic_idx: c_int,
    pub(super) field_pic_flag: c_int,
    pub(super) bottom_field_flag: c_int,
    pub(super) second_field: c_int,
    pub(super) bitstream_data_len: c_uint,
    pub(super) bitstream_data: *const u8,
    pub(super) num_slices: c_uint,
    pub(super) slice_data_offsets: *const c_uint,
    pub(super) ref_pic_flag: c_int,
    pub(super) intra_pic_flag: c_int,
    pub(super) timestamp: i64,
    pub(super) reserved: [c_uint; 28],
}

/// `CUVIDDECODECREATEINFO`: how `cuvidCreateDecoder` builds a decoder.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct CuvidDecodeCreateInfo {
    pub(super) width: c_ulong,
    pub(super) height: c_ulong,
    pub(super) num_decode_surfaces: c_ulong,
    pub(super) codec_type: c_int,
    pub(super) chroma_format: c_int,
    pub(super) creation_flags: c_ulong,
    pub(super) bit_depth_minus8: c_ulong,
    pub(super) intra_decode_only: c_ulong,
    pub(super) max_width: c_ulong,
    pub(super) max_height: c_ulong,
    pub(super) reserved1: c_ulong,
    pub(super) display_area: RectShort,
    pub(super) output_format: c_int,
    pub(super) deinterlace_mode: c_int,
    pub(super) target_width: c_ulong,
    pub(super) target_height: c_ulong,
    pub(super) num_output_surfaces: c_ulong,
    /// `CUvideoctxlock`; null, because every call this crate makes already has the
    /// context pushed on the calling thread.
    pub(super) vid_lock: *mut c_void,
    pub(super) target_rect: RectShort,
    pub(super) enable_histogram: c_ulong,
    pub(super) enable_decode_features: c_ulong,
    pub(super) reserved2: [c_ulong; 3],
}

/// `CUVIDPROCPARAMS`: how `cuvidMapVideoFrame64` post-processes a picture.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct CuvidProcParams {
    pub(super) progressive_frame: c_int,
    pub(super) second_field: c_int,
    pub(super) top_field_first: c_int,
    pub(super) unpaired_field: c_int,
    pub(super) reserved_flags: c_uint,
    pub(super) reserved_zero: c_uint,
    pub(super) raw_input_dptr: c_ulonglong,
    pub(super) raw_input_pitch: c_uint,
    pub(super) raw_input_format: c_uint,
    pub(super) raw_output_dptr: c_ulonglong,
    pub(super) raw_output_pitch: c_uint,
    pub(super) reserved1: c_uint,
    /// `CUstream`; null is the context's default stream.
    pub(super) output_stream: *mut c_void,
    pub(super) reserved: [c_uint; 46],
    pub(super) histogram_dptr: *mut c_ulonglong,
    pub(super) cuvid_proc_ext: *mut c_void,
}

/// `PFNVIDSEQUENCECALLBACK`.
pub(super) type SequenceCallback =
    unsafe extern "C" fn(user: *mut c_void, format: *mut CuVideoFormat) -> c_int;
/// `PFNVIDDECODECALLBACK`.
pub(super) type DecodeCallback =
    unsafe extern "C" fn(user: *mut c_void, params: *mut CuvidPicParams) -> c_int;
/// `PFNVIDDISPLAYCALLBACK`.
pub(super) type DisplayCallback =
    unsafe extern "C" fn(user: *mut c_void, info: *mut CuvidParserDispInfo) -> c_int;

/// `CUVIDPARSERPARAMS`: how `cuvidCreateVideoParser` builds a parser.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub(super) struct CuvidParserParams {
    pub(super) codec_type: c_int,
    pub(super) max_num_decode_surfaces: c_uint,
    pub(super) clock_rate: c_uint,
    pub(super) error_threshold: c_uint,
    pub(super) max_display_delay: c_uint,
    /// The bitfields `bAnnexb : 1`, `bMemoryOptimize : 1`, `uReserved : 30`.
    pub(super) flags: c_uint,
    pub(super) reserved1: [c_uint; 4],
    pub(super) user_data: *mut c_void,
    pub(super) sequence_callback: Option<SequenceCallback>,
    pub(super) decode_picture: Option<DecodeCallback>,
    pub(super) display_picture: Option<DisplayCallback>,
    /// `PFNVIDOPPOINTCALLBACK`; unset picks operating point 0.
    pub(super) get_operating_point: *mut c_void,
    /// `PFNVIDSEIMSGCALLBACK`; unset.
    pub(super) get_sei_msg: *mut c_void,
    pub(super) reserved2: [*mut c_void; 5],
    /// `CUVIDEOFORMATEX *`; unset, the parameter sets travel in the bitstream.
    pub(super) ext_video_info: *mut c_void,
}

// --- Entry points -----------------------------------------------------------

/// `cuInit`.
pub(super) type CuInit = unsafe extern "C" fn(flags: c_uint) -> CuResult;
/// `cuDeviceGet`.
pub(super) type CuDeviceGet =
    unsafe extern "C" fn(device: *mut CuDevice, ordinal: c_int) -> CuResult;
/// `cuDevicePrimaryCtxRetain`.
pub(super) type CuDevicePrimaryCtxRetain =
    unsafe extern "C" fn(ctx: *mut CuContext, device: CuDevice) -> CuResult;
/// `cuDevicePrimaryCtxRelease_v2`.
pub(super) type CuDevicePrimaryCtxRelease = unsafe extern "C" fn(device: CuDevice) -> CuResult;
/// `cuCtxPushCurrent_v2`.
pub(super) type CuCtxPushCurrent = unsafe extern "C" fn(ctx: CuContext) -> CuResult;
/// `cuCtxPopCurrent_v2`.
pub(super) type CuCtxPopCurrent = unsafe extern "C" fn(ctx: *mut CuContext) -> CuResult;
/// `cuMemcpy2D_v2`, synchronous with respect to the host for pageable memory.
pub(super) type CuMemcpy2d = unsafe extern "C" fn(copy: *const CudaMemcpy2d) -> CuResult;

/// `cuvidGetDecoderCaps`.
pub(super) type CuvidGetDecoderCaps = unsafe extern "C" fn(caps: *mut CuvidDecodeCaps) -> CuResult;
/// `cuvidCreateDecoder`.
pub(super) type CuvidCreateDecoder =
    unsafe extern "C" fn(decoder: *mut *mut c_void, info: *mut CuvidDecodeCreateInfo) -> CuResult;
/// `cuvidDestroyDecoder`.
pub(super) type CuvidDestroyDecoder = unsafe extern "C" fn(decoder: *mut c_void) -> CuResult;
/// `cuvidDecodePicture`.
pub(super) type CuvidDecodePicture =
    unsafe extern "C" fn(decoder: *mut c_void, params: *mut CuvidPicParams) -> CuResult;
/// `cuvidMapVideoFrame64`.
pub(super) type CuvidMapVideoFrame64 = unsafe extern "C" fn(
    decoder: *mut c_void,
    picture_index: c_int,
    device_ptr: *mut c_ulonglong,
    pitch: *mut c_uint,
    params: *mut CuvidProcParams,
) -> CuResult;
/// `cuvidUnmapVideoFrame64`.
pub(super) type CuvidUnmapVideoFrame64 =
    unsafe extern "C" fn(decoder: *mut c_void, device_ptr: c_ulonglong) -> CuResult;
/// `cuvidCreateVideoParser`.
pub(super) type CuvidCreateVideoParser =
    unsafe extern "C" fn(parser: *mut *mut c_void, params: *mut CuvidParserParams) -> CuResult;
/// `cuvidParseVideoData`.
pub(super) type CuvidParseVideoData =
    unsafe extern "C" fn(parser: *mut c_void, packet: *mut CuvidSourceDataPacket) -> CuResult;
/// `cuvidDestroyVideoParser`.
pub(super) type CuvidDestroyVideoParser = unsafe extern "C" fn(parser: *mut c_void) -> CuResult;

// --- Layout --------------------------------------------------------------

/// Asserts a field's offset at compile time.
macro_rules! at {
    ($ty:ty, $field:ident, $offset:expr) => {
        const _: () = assert!(std::mem::offset_of!($ty, $field) == $offset);
    };
}

// Every number below was derived by hand from the header (see the module doc),
// not read back from this file's own layout.
const _: () = assert!(size_of::<c_ulong>() == 8 && size_of::<c_long>() == 8);

const _: () = assert!(size_of::<CudaMemcpy2d>() == 128);
at!(CudaMemcpy2d, src_memory_type, 16);
at!(CudaMemcpy2d, src_host, 24);
at!(CudaMemcpy2d, src_device, 32);
at!(CudaMemcpy2d, src_pitch, 48);
at!(CudaMemcpy2d, dst_memory_type, 72);
at!(CudaMemcpy2d, dst_host, 80);
at!(CudaMemcpy2d, dst_pitch, 104);
at!(CudaMemcpy2d, width_in_bytes, 112);
at!(CudaMemcpy2d, height, 120);

const _: () = assert!(size_of::<CuvidDecodeCaps>() == 88);
at!(CuvidDecodeCaps, bit_depth_minus8, 8);
at!(CuvidDecodeCaps, is_supported, 24);
at!(CuvidDecodeCaps, output_format_mask, 26);
at!(CuvidDecodeCaps, max_width, 28);
at!(CuvidDecodeCaps, max_height, 32);
at!(CuvidDecodeCaps, max_mb_count, 36);
at!(CuvidDecodeCaps, min_width, 40);
at!(CuvidDecodeCaps, min_height, 42);
at!(CuvidDecodeCaps, reserved3, 52);

const _: () = assert!(size_of::<CuVideoFormat>() == 64);
at!(CuVideoFormat, progressive_sequence, 12);
at!(CuVideoFormat, bit_depth_luma_minus8, 13);
at!(CuVideoFormat, bit_depth_chroma_minus8, 14);
at!(CuVideoFormat, min_num_decode_surfaces, 15);
at!(CuVideoFormat, coded_width, 16);
at!(CuVideoFormat, coded_height, 20);
at!(CuVideoFormat, display_area, 24);
at!(CuVideoFormat, chroma_format, 40);
at!(CuVideoFormat, video_signal_description, 56);
at!(CuVideoFormat, seqhdr_data_length, 60);

const _: () = assert!(size_of::<CuvidSourceDataPacket>() == 32);
at!(CuvidSourceDataPacket, payload_size, 8);
at!(CuvidSourceDataPacket, payload, 16);
at!(CuvidSourceDataPacket, timestamp, 24);

const _: () = assert!(size_of::<CuvidParserDispInfo>() == 24);
at!(CuvidParserDispInfo, repeat_first_field, 12);
at!(CuvidParserDispInfo, timestamp, 16);

at!(CuvidPicParams, curr_pic_idx, 8);
at!(CuvidPicParams, bitstream_data_len, 24);
at!(CuvidPicParams, bitstream_data, 32);
at!(CuvidPicParams, num_slices, 40);
at!(CuvidPicParams, slice_data_offsets, 48);
at!(CuvidPicParams, timestamp, 64);
at!(CuvidPicParams, reserved, 72);

const _: () = assert!(size_of::<CuvidDecodeCreateInfo>() == 176);
at!(CuvidDecodeCreateInfo, codec_type, 24);
at!(CuvidDecodeCreateInfo, chroma_format, 28);
at!(CuvidDecodeCreateInfo, creation_flags, 32);
at!(CuvidDecodeCreateInfo, bit_depth_minus8, 40);
at!(CuvidDecodeCreateInfo, max_width, 56);
at!(CuvidDecodeCreateInfo, display_area, 80);
at!(CuvidDecodeCreateInfo, output_format, 88);
at!(CuvidDecodeCreateInfo, deinterlace_mode, 92);
at!(CuvidDecodeCreateInfo, target_width, 96);
at!(CuvidDecodeCreateInfo, target_height, 104);
at!(CuvidDecodeCreateInfo, num_output_surfaces, 112);
at!(CuvidDecodeCreateInfo, vid_lock, 120);
at!(CuvidDecodeCreateInfo, target_rect, 128);
at!(CuvidDecodeCreateInfo, enable_histogram, 136);
at!(CuvidDecodeCreateInfo, reserved2, 152);

const _: () = assert!(size_of::<CuvidProcParams>() == 264);
at!(CuvidProcParams, raw_input_dptr, 24);
at!(CuvidProcParams, raw_output_dptr, 40);
at!(CuvidProcParams, output_stream, 56);
at!(CuvidProcParams, reserved, 64);
at!(CuvidProcParams, histogram_dptr, 248);
at!(CuvidProcParams, cuvid_proc_ext, 256);

const _: () = assert!(size_of::<CuvidParserParams>() == 136);
at!(CuvidParserParams, max_display_delay, 16);
at!(CuvidParserParams, flags, 20);
at!(CuvidParserParams, user_data, 40);
at!(CuvidParserParams, sequence_callback, 48);
at!(CuvidParserParams, decode_picture, 56);
at!(CuvidParserParams, display_picture, 64);
at!(CuvidParserParams, get_operating_point, 72);
at!(CuvidParserParams, get_sei_msg, 80);
at!(CuvidParserParams, reserved2, 88);
at!(CuvidParserParams, ext_video_info, 128);

#[cfg(test)]
mod tests {
    use std::mem::offset_of;

    use super::*;

    // The const asserts above already fail the build; these repeat the headline
    // numbers as tests so a layout regression also shows up as a named failure.
    #[test]
    fn struct_sizes_match_the_header() {
        assert_eq!(size_of::<CudaMemcpy2d>(), 128);
        assert_eq!(size_of::<CuvidDecodeCaps>(), 88);
        assert_eq!(size_of::<CuVideoFormat>(), 64);
        assert_eq!(size_of::<CuvidSourceDataPacket>(), 32);
        assert_eq!(size_of::<CuvidParserDispInfo>(), 24);
        assert_eq!(size_of::<CuvidDecodeCreateInfo>(), 176);
        assert_eq!(size_of::<CuvidProcParams>(), 264);
        assert_eq!(size_of::<CuvidParserParams>(), 136);
    }

    #[test]
    fn read_offsets_match_the_header() {
        assert_eq!(offset_of!(CuvidPicParams, bitstream_data), 32);
        assert_eq!(offset_of!(CuvidPicParams, timestamp), 64);
        assert_eq!(offset_of!(CuVideoFormat, video_signal_description), 56);
        assert_eq!(offset_of!(CuvidParserParams, user_data), 40);
        assert_eq!(offset_of!(CuvidParserDispInfo, timestamp), 16);
    }

    #[test]
    fn full_range_is_bit_three_of_the_bitfield_byte() {
        let desc = |bits| VideoSignalDescription {
            bits,
            ..VideoSignalDescription::default()
        };
        // video_format = 5 (unspecified), full range clear, then set.
        assert!(!desc(0b0000_0101).full_range());
        assert!(desc(0b0000_1101).full_range());
        assert!(!desc(0b1111_0111).full_range());
    }
}
