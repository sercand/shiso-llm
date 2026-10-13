//! The `decode-native` step: symphonia, in process.
//!
//! The bytes are probed, the first audio track is decoded packet by packet,
//! each buffer is averaged down to mono at its own rate, and the result is
//! resampled once to [`MODEL_RATE`]. Two caps are enforced between packets,
//! because nothing here can interrupt a packet once it is being decoded:
//!
//! - the duration cap: decoding stops once `max_seconds + 1` seconds are held,
//!   so an hour-long upload costs one second more than the cap, not an hour of
//!   decode and resample. [`decode_audio`](crate::decode_audio)'s own check then
//!   reports it as too long;
//! - the wall-clock deadline of the [`AudioDecodePolicy`](crate::AudioDecodePolicy).
//!
//! Only "symphonia does not do this" moves the chain on to the next step
//! ([`Native::Unsupported`]): a codec this build has no decoder for (Opus), or a
//! container whose reader this build lacks. Bytes that claim a format symphonia
//! reads and then break it are refused as [`AudioError::Malformed`]; handing
//! them to ffmpeg would only spend a subprocess to learn the same thing.

use std::io::Cursor;
use std::time::Instant;

use shiso_vision::VisionError;
use symphonia::core::audio::GenericAudioBufferRef;
use symphonia::core::codecs::audio::well_known::{
    CODEC_ID_AAC, CODEC_ID_AC3, CODEC_ID_ALAC, CODEC_ID_ATRAC3, CODEC_ID_DCA, CODEC_ID_EAC3,
    CODEC_ID_FLAC, CODEC_ID_MP1, CODEC_ID_MP2, CODEC_ID_MP3, CODEC_ID_OPUS, CODEC_ID_SPEEX,
    CODEC_ID_TRUEHD, CODEC_ID_VORBIS, CODEC_ID_WAVPACK, CODEC_ID_WMA,
};
use symphonia::core::codecs::audio::{AudioCodecId, AudioDecoderOptions};
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::formats::probe::Hint;
use symphonia::core::formats::{FormatOptions, TrackType};
use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
use symphonia::core::meta::MetadataOptions;

use crate::decode::DecodeLimits;
use crate::error::AudioError;
use crate::pcm::{MODEL_RATE, Pcm};
use crate::resample::resample;

/// What the native step made of the bytes.
#[derive(Debug)]
pub(crate) enum Native {
    /// Decoded, capped and resampled to [`MODEL_RATE`].
    Decoded(Pcm),
    /// Symphonia in this build does not read this: the chain moves on.
    Unsupported {
        /// The codec, when the container was read far enough to name it.
        codec: Option<String>,
    },
}

/// The wall-clock point past which no further packet is decoded.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Deadline {
    /// `None` when the budget is past what `Instant` can represent: an
    /// operator who set an enormous timeout meant "no deadline", and clamping
    /// it to now would refuse every decode instead.
    at: Option<Instant>,
    secs: u64,
}

impl Deadline {
    /// `secs` from now.
    pub(crate) fn after(secs: u64) -> Self {
        let at = Instant::now().checked_add(std::time::Duration::from_secs(secs));
        Self { at, secs }
    }

    /// A deadline at an explicit instant, reported as a budget of `secs`.
    ///
    /// The public policy's budget is a whole, non-zero number of seconds,
    /// which a test cannot wait out cheaply; this lets one pass a deadline
    /// that has already passed.
    #[cfg(test)]
    pub(crate) fn at(at: Instant, secs: u64) -> Self {
        Self { at: Some(at), secs }
    }

    fn check(self) -> Result<(), AudioError> {
        if self.at.is_some_and(|at| Instant::now() >= at) {
            return Err(AudioError::Decoder(VisionError::DecoderTimeout {
                secs: self.secs,
            }));
        }
        Ok(())
    }
}

/// Decodes `bytes` with symphonia.
///
/// `container` is the sniffed name, used to tell a container symphonia cannot
/// find from one it should have found (see [`probe_failure`]).
///
/// # Errors
///
/// - [`AudioError::Malformed`] for a stream symphonia reads but that breaks
///   its own format.
/// - [`AudioError::Decoder`] carrying [`VisionError::DecoderTimeout`] when the
///   deadline passes between packets.
/// - [`AudioError::BadRate`] for a rate the resampler refuses.
pub(crate) fn decode(
    bytes: &[u8],
    container: &'static str,
    limits: DecodeLimits,
    deadline: Deadline,
) -> Result<Native, AudioError> {
    let malformed = |error: &SymphoniaError| AudioError::Malformed {
        container,
        reason: error.to_string(),
    };
    deadline.check()?;
    let bytes = skip_id3v2(bytes);
    let source = MediaSourceStream::new(
        Box::new(Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );
    let mut format = match symphonia::default::get_probe().probe(
        &Hint::new(),
        source,
        FormatOptions::default(),
        MetadataOptions::default(),
    ) {
        Ok(format) => format,
        Err(error @ SymphoniaError::Unsupported(_)) => {
            return probe_failure(bytes, container, &error);
        }
        Err(error) => return Err(malformed(&error)),
    };

    let Some(track) = format.first_track(TrackType::Audio) else {
        // A container with no audio track symphonia reads (a video-only mp4,
        // or a stream mapping the reader skipped): another decoder may know
        // more, so this is not ours to refuse.
        return Ok(Native::Unsupported { codec: None });
    };
    let track_id = track.id;
    let Some(params) = track.codec_params.as_ref().and_then(|p| p.audio()) else {
        return Ok(Native::Unsupported { codec: None });
    };
    let codec = params.codec;
    // Gapless is the default: encoder delay and padding are trimmed, so an
    // mp3 or vorbis clip comes back the length it was recorded at.
    let mut decoder = match symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
    {
        Ok(decoder) => decoder,
        Err(SymphoniaError::Unsupported(_)) => {
            return Ok(Native::Unsupported {
                codec: Some(codec_name(codec)),
            });
        }
        Err(error) => return Err(malformed(&error)),
    };

    let mut mono = Mono::new(container, limits);
    loop {
        deadline.check()?;
        let packet = match format.next_packet() {
            Ok(Some(packet)) => packet,
            Ok(None) => break,
            Err(SymphoniaError::Unsupported(_)) => {
                return Ok(Native::Unsupported {
                    codec: Some(codec_name(codec)),
                });
            }
            Err(error) => return Err(malformed(&error)),
        };
        if packet.track_id != track_id {
            continue;
        }
        let buffer = match decoder.decode(&packet) {
            Ok(buffer) => buffer,
            // A codec feature this decoder lacks (an AAC profile past LC, for
            // one): ffmpeg may have it.
            Err(SymphoniaError::Unsupported(_)) => {
                return Ok(Native::Unsupported {
                    codec: Some(codec_name(codec)),
                });
            }
            Err(error) => return Err(malformed(&error)),
        };
        if mono.push(&buffer)? {
            break;
        }
    }
    mono.finish().map(Native::Decoded)
}

/// Decoded buffers, averaged to mono at the stream's own rate and capped.
struct Mono {
    container: &'static str,
    max_seconds: u32,
    samples: Vec<f32>,
    /// Reused per buffer: one plane per channel.
    planes: Vec<Vec<f32>>,
    /// Rate and channel count of the first non-empty buffer. A stream that
    /// changes either part-way is refused rather than spliced: averaging and
    /// resampling assume one of each.
    layout: Option<(u32, usize)>,
    /// `max_seconds + 1` at the native rate, once the rate is known.
    cap: usize,
}

impl Mono {
    fn new(container: &'static str, limits: DecodeLimits) -> Self {
        Self {
            container,
            max_seconds: limits.max_seconds(),
            samples: Vec::new(),
            planes: Vec::new(),
            layout: None,
            cap: usize::MAX,
        }
    }

    /// Appends one buffer. `true` once the cap is reached: past it, so the
    /// caller's duration check reports it, without decoding the rest.
    fn push(&mut self, buffer: &GenericAudioBufferRef<'_>) -> Result<bool, AudioError> {
        if buffer.frames() == 0 {
            return Ok(false);
        }
        let rate = buffer.spec().rate();
        let channels = buffer.spec().channels().count();
        match self.layout {
            None => {
                if channels == 0 {
                    return Err(AudioError::Malformed {
                        container: self.container,
                        reason: "the stream declares zero channels".to_owned(),
                    });
                }
                self.layout = Some((rate, channels));
                let cap = (u64::from(self.max_seconds) + 1) * u64::from(rate);
                self.cap = usize::try_from(cap).unwrap_or(usize::MAX);
            }
            Some(first) if first != (rate, channels) => {
                return Err(AudioError::Malformed {
                    container: self.container,
                    reason: format!(
                        "the stream changes from {} Hz x {} to {rate} Hz x {channels} part-way",
                        first.0, first.1
                    ),
                });
            }
            Some(_) => {}
        }
        buffer.copy_to_vecs_planar::<f32>(&mut self.planes);
        downmix_into(&self.planes, &mut self.samples);
        if self.samples.len() >= self.cap {
            self.samples.truncate(self.cap);
            return Ok(true);
        }
        Ok(false)
    }

    /// Resamples what was collected to [`MODEL_RATE`].
    fn finish(self) -> Result<Pcm, AudioError> {
        let Some((rate, _)) = self.layout else {
            // Nothing decoded: the caller refuses an empty clip by name.
            return Pcm::new(Vec::new(), MODEL_RATE);
        };
        Pcm::new(resample(&self.samples, rate, MODEL_RATE)?, MODEL_RATE)
    }
}

/// Classifies a probe that found no reader.
///
/// For a bare mp3 or FLAC stream the reader is in this build, so a probe that
/// cannot place it means the bytes only start like one: that is malformed, and
/// ffmpeg would refuse it too. A container whose reader may decline a stream
/// it does not map (ogg, Matroska, mp4) or that was not recognised at all goes
/// on to the next step.
fn probe_failure(
    bytes: &[u8],
    container: &'static str,
    error: &SymphoniaError,
) -> Result<Native, AudioError> {
    if container == "flac" || (container == "mp3" && !is_mpeg_layer_1_or_2(bytes)) {
        // Symphonia words this as an unsupported feature; to the caller it is
        // a file that says mp3 and holds no mp3 frame.
        tracing::debug!(container, %error, "the probe found no stream");
        return Err(AudioError::Malformed {
            container,
            reason: format!("the bytes start like {container} but no {container} stream was found"),
        });
    }
    tracing::debug!(container, %error, "symphonia does not read this container");
    Ok(Native::Unsupported { codec: None })
}

/// `bytes` past any leading `ID3v2` tags.
///
/// The build carries no ID3 reader (the metadata is thrown away anyway), so
/// without this the probe scans a tag byte by byte for a frame sync and gives
/// up at its 1 MB window: an mp3 with embedded cover art would be refused. A
/// tag reader would instead refuse a file whose tag is slightly off, which
/// ffmpeg tolerates; skipping by the header's declared size does neither.
/// A header that does not parse, or claims more than the buffer holds, is left
/// for the probe to judge.
fn skip_id3v2(mut bytes: &[u8]) -> &[u8] {
    while let [b'I', b'D', b'3', _major, _minor, flags, s0, s1, s2, s3, ..] = *bytes {
        let size = [s0, s1, s2, s3];
        // Syncsafe: the top bit of every size byte is clear.
        if size.iter().any(|b| b & 0x80 != 0) {
            break;
        }
        let body = size
            .iter()
            .fold(0usize, |acc, &b| (acc << 7) | usize::from(b));
        // A footer (flag 0x10) repeats the 10-byte header after the body.
        let footer = if flags & 0x10 != 0 { 10 } else { 0 };
        let Some(rest) = bytes.get(10 + body + footer..) else {
            break;
        };
        bytes = rest;
    }
    bytes
}

/// Whether `bytes` start with an MPEG audio Layer I or II frame header.
///
/// Those sniff as `mp3` (the same sync word), but this build carries the
/// Layer III decoder only, so failing to probe one is "unsupported", not
/// "malformed".
fn is_mpeg_layer_1_or_2(bytes: &[u8]) -> bool {
    match bytes {
        // Layer bits 0b10 (II) or 0b11 (I); 0b01 is III and 0b00 is ADTS AAC.
        [0xFF, second, ..] if second & 0xE0 == 0xE0 => (second >> 1) & 0b11 >= 0b10,
        _ => false,
    }
}

/// Averages `planes` (one per channel, equal lengths) onto the end of `mono`.
fn downmix_into(planes: &[Vec<f32>], mono: &mut Vec<f32>) {
    let Some(first) = planes.first() else {
        return;
    };
    if planes.len() == 1 {
        mono.extend_from_slice(first);
        return;
    }
    // Channel counts are small (at most a few dozen), so this is exact.
    let scale = 1.0 / f32::from(u16::try_from(planes.len()).unwrap_or(u16::MAX));
    let start = mono.len();
    mono.extend_from_slice(first);
    for plane in &planes[1..] {
        for (out, &sample) in mono[start..].iter_mut().zip(plane) {
            *out += sample;
        }
    }
    for out in &mut mono[start..] {
        *out *= scale;
    }
}

/// A readable name for a codec symphonia identified but has no decoder for.
fn codec_name(id: AudioCodecId) -> String {
    let name = match id {
        CODEC_ID_OPUS => "opus",
        CODEC_ID_AAC => "aac",
        CODEC_ID_MP3 => "mp3",
        CODEC_ID_MP2 => "mp2",
        CODEC_ID_MP1 => "mp1",
        CODEC_ID_VORBIS => "vorbis",
        CODEC_ID_FLAC => "flac",
        CODEC_ID_ALAC => "alac",
        CODEC_ID_AC3 => "ac3",
        CODEC_ID_EAC3 => "eac3",
        CODEC_ID_DCA => "dts",
        CODEC_ID_TRUEHD => "truehd",
        CODEC_ID_WMA => "wma",
        CODEC_ID_SPEEX => "speex",
        CODEC_ID_WAVPACK => "wavpack",
        CODEC_ID_ATRAC3 => "atrac3",
        other => return format!("codec id {other}"),
    };
    name.to_owned()
}

#[cfg(test)]
#[path = "native_tests.rs"]
mod tests;
