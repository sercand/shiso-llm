//! Turns the committed elementary-stream fixtures into what a demuxer would
//! hand `HwDecoder`: the container's codec configuration record, and samples in
//! decode order with their presentation timestamps.
//!
//! There is no demuxer in this crate, so the fixtures are stored in the
//! simplest forms that still carry everything an MP4 track would:
//!
//! - **H.264, HEVC:** an Annex-B stream with an access-unit delimiter before
//!   every access unit, so splitting on the delimiter gives exactly one sample
//!   per frame with no slice-header parsing. The sample is rebuilt as the MP4
//!   would store it: parameter sets and delimiters dropped (they live in the
//!   configuration record), every other NAL unit behind a 4-byte length. The
//!   presentation timestamps sit in a `.pts` sidecar, one per sample in decode
//!   order, copied from the MP4 the stream was cut from, because Annex-B has
//!   none; they are what makes the display-order checks mean something.
//! - **VP9, AV1:** IVF, which carries a timestamp per frame. AV1 temporal
//!   delimiter OBUs are dropped, as an MP4 sample has none.
//! - **Every codec:** the configuration record (`.avcC`, `.hvcC`, `.vpcC`,
//!   `.av1C`) is the box payload exactly as ffmpeg's MP4 muxer wrote it, so it
//!   is what a demuxer would pass and comes from an implementation independent
//!   of this crate.
//!
//! The fixtures were generated once, by ffmpeg 9.0.2, as 10 frames of a
//! 160x96, 10 fps test pattern tagged BT.709 limited range. The pattern's
//! chroma is blurred first (luma keeps its detail): its saturated colour bars
//! end between chroma samples, and two correct decoders that upsample 4:2:0
//! chroma differently disagree there by more than the PSNR floor allows, which
//! would make the floor measure upsampling filters instead of planes, strides
//! and matrices.
//!
//! ```text
//! SRC=testsrc2=size=160x96:rate=10,format=yuv444p,boxblur=luma_radius=0:luma_power=0:chroma_radius=2:chroma_power=2
//! ffmpeg -f lavfi -i $SRC -frames:v 10 -pix_fmt yuv420p \
//!   -colorspace bt709 -color_primaries bt709 -color_trc bt709 -color_range tv <codec args> X.mp4
//!   h264: -c:v libx264 -profile:v high -bf 2 -g 10 -x264-params b-adapt=0:b-pyramid=none
//!   hevc: -c:v libx265 -tag:v hvc1 -g 10 -x265-params bframes=2
//!   vp9:  -c:v libvpx-vp9 -b:v 300k -g 10
//!   av1:  -c:v libsvtav1 -g 10 -svtav1-params preset=8
//! ffmpeg -i h264.mp4 -c copy -bsf:v h264_mp4toannexb,h264_metadata=aud=insert -f h264 h264-high-bframes.h264
//! ffmpeg -i hevc.mp4 -c copy -bsf:v hevc_mp4toannexb,hevc_metadata=aud=insert -f hevc hevc-main.hevc
//! ffmpeg -i vp9.mp4 -c copy -f ivf vp9-profile0.ivf      (and the same for av1)
//! ffprobe -v error -select_streams v:0 -show_entries packet=pts -of csv=p=0 X.mp4 > X.pts
//! ```
//!
//! and the configuration records were cut out of each MP4's `stsd` entry.
//!
//! The cropped fixtures (`h264-cropped`, `hevc-cropped`) are the same recipe at
//! `size=170x98`, which both encoders code as 176x112 with a crop.
//!
//! The reference frames under `test-vectors/references/media-sys/<name>.rgb24`
//! are ffmpeg's own decode of the committed fixture files, in display order,
//! converted with every choice spelled out so they do not depend on its
//! defaults:
//!
//! ```text
//! ffmpeg -i <fixture> -vf scale=in_color_matrix=bt709:in_range=tv:out_range=pc:in_h_chr_pos=0:in_v_chr_pos=0:flags=accurate_rnd+full_chroma_int+bilinear \
//!   -f rawvideo -pix_fmt rgb24 <name>.rgb24
//! ```
//!
//! The chroma position is the one convention matched to the device:
//! `VideoToolbox`'s BGRA conversion places 4:2:0 chroma at the top-left luma
//! sample for all four codecs. Measured against its output on an M4: about
//! 50 dB with that siting, 36 dB with H.264's default left siting, 32 dB
//! centred. Matrix and range are the stream's own (BT.709, limited); the same
//! output scored against a BT.601 or a full-range conversion of these fixtures
//! measures 29 and 26 dB, so the floor catches either mistake.

use std::path::{Path, PathBuf};

use shiso_media_sys::HwCodec;

/// Frames in every fixture.
pub(crate) const FRAMES: usize = 10;

/// One stream, ready for `HwDecoder`.
#[derive(Debug, Clone)]
pub(crate) struct Clip {
    /// The stream's codec.
    pub(crate) codec: HwCodec,
    /// The name its reference frames are stored under.
    pub(crate) name: &'static str,
    /// Picture width in pixels.
    pub(crate) width: u32,
    /// Picture height in pixels.
    pub(crate) height: u32,
    /// The container's configuration record.
    pub(crate) config: Vec<u8>,
    /// Samples in decode order with their presentation timestamps.
    pub(crate) samples: Vec<(Vec<u8>, i64)>,
}

/// The fixture directory.
pub(crate) fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn read(name: &str) -> Vec<u8> {
    let path = fixtures().join(name);
    std::fs::read(&path).unwrap_or_else(|e| panic!("fixture {}: {e}", path.display()))
}

/// Loads the 160x96 fixture for `codec`.
pub(crate) fn clip(codec: HwCodec) -> Clip {
    let clip = match codec {
        HwCodec::H264 => annexb_clip(codec, "h264-high-bframes", "avcC"),
        HwCodec::Hevc => annexb_clip(codec, "hevc-main", "hvcC"),
        HwCodec::Vp9 => ivf_clip(codec, "vp9-profile0", "vpcC"),
        HwCodec::Av1 => ivf_clip(codec, "av1-main", "av1C"),
        other => panic!("no fixture for {other}"),
    };
    Clip {
        name: match codec {
            HwCodec::H264 => "h264",
            HwCodec::Hevc => "hevc",
            HwCodec::Vp9 => "vp9",
            _ => "av1",
        },
        width: 160,
        height: 96,
        ..clip
    }
}

/// Loads the 170x98 H.264 or HEVC fixture: coded as 176x112, so its SPS
/// carries a crop that the decoded frames must honour.
pub(crate) fn cropped(codec: HwCodec) -> Clip {
    let (stem, config, name) = match codec {
        HwCodec::H264 => ("h264-cropped", "avcC", "h264-cropped"),
        HwCodec::Hevc => ("hevc-cropped", "hvcC", "hevc-cropped"),
        other => panic!("no cropped fixture for {other}"),
    };
    Clip {
        name,
        width: 170,
        height: 98,
        ..annexb_clip(codec, stem, config)
    }
}

fn annexb_clip(codec: HwCodec, stem: &str, config: &str) -> Clip {
    let units = access_units(codec, &read(&format!("{stem}.{}", ext(codec))));
    let pts: Vec<i64> = String::from_utf8(read(&format!("{stem}.pts")))
        .expect("utf-8 pts sidecar")
        .lines()
        .map(|l| l.trim().parse().expect("an integer pts per line"))
        .collect();
    assert_eq!(units.len(), pts.len(), "{stem}: one pts per access unit");
    Clip {
        codec,
        name: "",
        width: 0,
        height: 0,
        config: read(&format!("{stem}.{config}")),
        samples: units.into_iter().zip(pts).collect(),
    }
}

fn ext(codec: HwCodec) -> &'static str {
    match codec {
        HwCodec::H264 => "h264",
        _ => "hevc",
    }
}

fn ivf_clip(codec: HwCodec, stem: &str, config: &str) -> Clip {
    let frames = ivf_frames(&read(&format!("{stem}.ivf")));
    let samples = frames
        .into_iter()
        .map(|(pts, data)| {
            let data = if codec == HwCodec::Av1 {
                strip_temporal_delimiters(&data)
            } else {
                data
            };
            (data, pts)
        })
        .collect();
    Clip {
        codec,
        name: "",
        width: 0,
        height: 0,
        config: read(&format!("{stem}.{config}")),
        samples,
    }
}

/// Splits an Annex-B byte stream into NAL units (start codes removed).
pub(crate) fn nal_units(stream: &[u8]) -> Vec<&[u8]> {
    // Positions just past each `00 00 01` start code.
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut units = Vec::new();
    for (n, &start) in starts.iter().enumerate() {
        let mut end = starts.get(n + 1).map_or(stream.len(), |next| next - 3);
        // A four-byte start code's leading zero belongs to it, not to the unit.
        while end > start && stream[end - 1] == 0 {
            end -= 1;
        }
        units.push(&stream[start..end]);
    }
    units
}

/// Groups an Annex-B stream into MP4-style samples, one per access unit,
/// split on access-unit delimiters, with parameter sets and delimiters
/// dropped and every remaining NAL unit behind a 4-byte big-endian length.
pub(crate) fn access_units(codec: HwCodec, stream: &[u8]) -> Vec<Vec<u8>> {
    let (aud, dropped): (u8, &[u8]) = match codec {
        HwCodec::H264 => (9, &[7, 8, 9]),
        _ => (35, &[32, 33, 34, 35]),
    };
    let nal_type = |unit: &[u8]| match codec {
        HwCodec::H264 => unit[0] & 0x1f,
        _ => (unit[0] >> 1) & 0x3f,
    };
    let mut samples: Vec<Vec<u8>> = Vec::new();
    for unit in nal_units(stream) {
        let kind = nal_type(unit);
        if kind == aud {
            samples.push(Vec::new());
        }
        if dropped.contains(&kind) {
            continue;
        }
        let sample = samples
            .last_mut()
            .expect("the stream starts with an access-unit delimiter");
        sample.extend_from_slice(&u32::try_from(unit.len()).expect("small").to_be_bytes());
        sample.extend_from_slice(unit);
    }
    samples.retain(|s| !s.is_empty());
    samples
}

/// Reads an IVF file's frames as `(pts, payload)`.
pub(crate) fn ivf_frames(file: &[u8]) -> Vec<(i64, Vec<u8>)> {
    assert_eq!(&file[..4], b"DKIF", "IVF signature");
    let header = usize::from(u16::from_le_bytes([file[6], file[7]]));
    let mut frames = Vec::new();
    let mut at = header;
    while at + 12 <= file.len() {
        let size = u32::from_le_bytes(file[at..at + 4].try_into().expect("4 bytes")) as usize;
        let pts = i64::from_le_bytes(file[at + 4..at + 12].try_into().expect("8 bytes"));
        let data = file[at + 12..at + 12 + size].to_vec();
        frames.push((pts, data));
        at += 12 + size;
    }
    assert_eq!(at, file.len(), "IVF ends on a frame boundary");
    frames
}

/// Drops temporal-delimiter OBUs (type 2) from an AV1 temporal unit. Every
/// OBU must carry its size field, as IVF and MP4 AV1 always do.
pub(crate) fn strip_temporal_delimiters(unit: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(unit.len());
    let mut at = 0;
    while at < unit.len() {
        let header = unit[at];
        let kind = (header >> 3) & 0x0f;
        let extension = header & 0x04 != 0;
        assert!(header & 0x02 != 0, "an AV1 OBU without a size field");
        let mut pos = at + 1 + usize::from(extension);
        let mut size = 0usize;
        for shift in 0..8 {
            let byte = unit[pos];
            pos += 1;
            size |= usize::from(byte & 0x7f) << (7 * shift);
            if byte & 0x80 == 0 {
                break;
            }
        }
        let end = pos + size;
        if kind != 2 {
            out.extend_from_slice(&unit[at..end]);
        }
        at = end;
    }
    out
}

/// The reference frames for `clip`, or why they are not there.
pub(crate) fn reference(clip: &Clip) -> Result<Vec<u8>, String> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../test-vectors/references/media-sys")
        .join(format!("{}.rgb24", clip.name));
    std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))
}

/// Peak signal-to-noise ratio of two RGB8 images, in dB (infinite when equal).
pub(crate) fn psnr(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "images of different sizes");
    let sse: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum();
    if sse == 0.0 {
        return f64::INFINITY;
    }
    let mse = sse / f64::from(u32::try_from(a.len()).expect("an image under 4 GiB"));
    10.0 * (255.0 * 255.0 / mse).log10()
}
