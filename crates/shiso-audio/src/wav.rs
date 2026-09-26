//! RIFF/WAVE, read natively.
//!
//! WAV is the one container read without the subprocess decoder. It is what
//! most speech clients send, and it is simple enough to parse completely: a
//! chunk list, one format block, one data block. The reader trusts neither the
//! RIFF size nor the data size. Streaming recorders write `0xFFFFFFFF` or a
//! length that runs past the file, and the samples present are still the
//! samples.

use crate::error::AudioError;
use crate::num::f32_of;
use crate::pcm::Pcm;

/// Integer PCM.
const TAG_PCM: u16 = 1;
/// IEEE float.
const TAG_FLOAT: u16 = 3;
/// `WAVE_FORMAT_EXTENSIBLE`: the real tag is the head of the sub-format GUID.
const TAG_EXTENSIBLE: u16 = 0xFFFE;

/// Whether `bytes` start like a RIFF/WAVE file.
pub(crate) fn is_wav(bytes: &[u8]) -> bool {
    bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WAVE"
}

/// The parts of the `fmt ` chunk the decoder needs.
struct Format {
    tag: u16,
    channels: u16,
    rate: u32,
    block_align: u16,
    bits: u16,
}

fn u16_at(b: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(b.get(at..at + 2)?.try_into().ok()?))
}

fn u32_at(b: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(b.get(at..at + 4)?.try_into().ok()?))
}

fn parse_format(body: &[u8]) -> Result<Format, AudioError> {
    let truncated = || AudioError::MalformedWav {
        reason: "the fmt chunk is truncated",
    };
    let mut tag = u16_at(body, 0).ok_or_else(truncated)?;
    let channels = u16_at(body, 2).ok_or_else(truncated)?;
    let rate = u32_at(body, 4).ok_or_else(truncated)?;
    let block_align = u16_at(body, 12).ok_or_else(truncated)?;
    let bits = u16_at(body, 14).ok_or_else(truncated)?;
    if tag == TAG_EXTENSIBLE {
        // cbSize(2) validBits(2) channelMask(4), then the GUID whose first two
        // bytes are the real tag.
        tag = u16_at(body, 24).ok_or(AudioError::MalformedWav {
            reason: "an extensible fmt chunk has no sub-format",
        })?;
    }
    if channels == 0 {
        return Err(AudioError::MalformedWav {
            reason: "the file declares zero channels",
        });
    }
    if rate == 0 {
        return Err(AudioError::MalformedWav {
            reason: "the file declares a zero sample rate",
        });
    }
    Ok(Format {
        tag,
        channels,
        rate,
        block_align,
        bits,
    })
}

/// Decodes one sample from the start of `b`; the caller has checked that the
/// encoding is supported and that `b` holds a whole sample.
fn sample(tag: u16, bits: u16, b: &[u8]) -> f32 {
    match (tag, bits) {
        (TAG_PCM, 8) => (f32::from(b[0]) - 128.0) / 128.0,
        (TAG_PCM, 16) => f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0,
        (TAG_PCM, 24) => {
            // Place the three bytes high and shift back down to sign-extend.
            let v = i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8;
            f32_of(f64::from(v) / 8_388_608.0)
        }
        (TAG_PCM, 32) => {
            f32_of(f64::from(i32::from_le_bytes([b[0], b[1], b[2], b[3]])) / 2_147_483_648.0)
        }
        (TAG_FLOAT, 32) => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        (TAG_FLOAT, 64) => f32_of(f64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ])),
        _ => 0.0,
    }
}

/// Parses a WAV file to mono at its own rate.
///
/// # Errors
///
/// [`AudioError::MalformedWav`] for a broken chunk list or `fmt` block,
/// [`AudioError::UnsupportedWavFormat`] for an encoding outside the accepted
/// set, and [`AudioError::Empty`] for a file with no whole sample frame.
pub(crate) fn parse_wav(bytes: &[u8]) -> Result<Pcm, AudioError> {
    if !is_wav(bytes) {
        return Err(AudioError::MalformedWav {
            reason: "no RIFF/WAVE header",
        });
    }
    let mut at = 12usize;
    let mut format = None;
    let mut data = None;
    while at + 8 <= bytes.len() {
        let id = &bytes[at..at + 4];
        let declared =
            u32_at(bytes, at + 4).map_or(0, |n| usize::try_from(n).unwrap_or(usize::MAX));
        let body_start = at + 8;
        let body_end = body_start.saturating_add(declared).min(bytes.len());
        let body = &bytes[body_start..body_end];
        match id {
            b"fmt " => format = Some(parse_format(body)?),
            b"data" => {
                // Anything after the samples is metadata this reader does not
                // need, and a streamed length would run past it anyway.
                data = Some(body);
                break;
            }
            _ => {}
        }
        // Chunks are padded to an even length.
        at = body_start
            .saturating_add(declared)
            .saturating_add(declared & 1);
    }
    let format = format.ok_or(AudioError::MalformedWav {
        reason: "no fmt chunk",
    })?;
    let data = data.ok_or(AudioError::MalformedWav {
        reason: "no data chunk",
    })?;
    let supported = matches!(
        (format.tag, format.bits),
        (TAG_PCM, 8 | 16 | 24 | 32) | (TAG_FLOAT, 32 | 64)
    );
    if !supported {
        return Err(AudioError::UnsupportedWavFormat {
            tag: format.tag,
            bits: format.bits,
        });
    }
    let width = usize::from(format.bits / 8);
    let channels = usize::from(format.channels);
    let frame = width * channels;
    // A frame wider than its samples is a container this reader would walk at
    // the wrong stride - 24-bit samples padded to 32 without the extensible
    // header - and decode to noise. Refuse it rather than guess the padding.
    if usize::from(format.block_align) != frame {
        return Err(AudioError::MalformedWav {
            reason: "the block alignment disagrees with the channels and sample width",
        });
    }
    let frames = data.len() / frame;
    if frames == 0 {
        return Err(AudioError::Empty);
    }
    let scale = 1.0 / f32::from(format.channels);
    let mut out = Vec::with_capacity(frames);
    for f in 0..frames {
        let base = f * frame;
        let mut sum = 0.0f32;
        for c in 0..channels {
            sum += sample(format.tag, format.bits, &data[base + c * width..]);
        }
        out.push(sum * scale);
    }
    Pcm::new(out, format.rate)
}

#[cfg(test)]
#[path = "wav_tests.rs"]
mod tests;
