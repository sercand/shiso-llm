//! Moving an MP4's index ahead of its media, so a decoder reading from a pipe
//! can demux it.
//!
//! An MP4 is a list of boxes. The media is in `mdat` and the index that says
//! where each chunk of it is (`moov`) can come before or after it. Recorders
//! write `mdat` first because they do not know the index until they stop, so a
//! phone memo or a file straight out of an encoder ends in its `moov`. A
//! decoder reading a pipe cannot seek back to the media once it has read the
//! index, and past its probe buffer (under a minute of 64 kbit/s audio) it
//! gives up: the upload decodes to nothing.
//!
//! The fix is the one a "fast start" remux makes. `moov` moves ahead of the
//! first `mdat`, and every chunk offset in its `stco` and `co64` tables grows
//! by the size of `moov`, which is how far the media moved. Anything this does
//! not recognise, including a fragmented file (whose `moof` boxes carry their
//! own offsets) and any box that does not parse, is passed through untouched
//! for the decoder to accept or refuse as before.
//!
//! The box walker itself ([`boxes`], [`header`]) is shared with the in-house
//! demuxer in [`crate::demux`], which is always compiled; only the remux is
//! tied to the ffmpeg step, because only a decoder reading a pipe needs it.

#[cfg(feature = "decode-ffmpeg")]
use std::borrow::Cow;

/// One box: where it starts and how long it is, header included.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Span {
    pub(crate) kind: [u8; 4],
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl Span {
    /// Where the box ends, one past its last byte.
    ///
    /// Cannot overflow: [`boxes`] only builds spans whose end it has already
    /// checked against the buffer.
    pub(crate) fn end(self) -> usize {
        self.start.saturating_add(self.len)
    }
}

/// Containers on the path from `moov` to the chunk-offset tables.
#[cfg(feature = "decode-ffmpeg")]
const CONTAINERS: [&[u8; 4]; 5] = [b"moov", b"trak", b"mdia", b"minf", b"stbl"];

/// The boxes laid end to end in `bytes[from..to]`, or `None` when they do not
/// tile it exactly.
pub(crate) fn boxes(bytes: &[u8], from: usize, to: usize) -> Option<Vec<Span>> {
    let mut out = Vec::new();
    let mut at = from;
    while at < to {
        let head = bytes.get(at..at + 8)?;
        let small = u32::from_be_bytes(head[..4].try_into().ok()?);
        let kind: [u8; 4] = head[4..8].try_into().ok()?;
        let len = match small {
            0 => to - at,
            1 => usize::try_from(u64::from_be_bytes(
                bytes.get(at + 8..at + 16)?.try_into().ok()?,
            ))
            .ok()?,
            n => usize::try_from(n).ok()?,
        };
        if len < 8 || at.checked_add(len)? > to {
            return None;
        }
        out.push(Span {
            kind,
            start: at,
            len,
        });
        at += len;
    }
    Some(out)
}

/// Header bytes of a box: eight, or sixteen with a 64-bit size.
pub(crate) fn header(bytes: &[u8], span: Span) -> usize {
    if bytes.get(span.start..span.start.saturating_add(4)) == Some(&[0, 0, 0, 1]) {
        16
    } else {
        8
    }
}

/// Adds `shift` to every chunk offset under `span`, in place; `None` when a
/// table does not parse or an offset would leave the media.
#[cfg(feature = "decode-ffmpeg")]
fn patch(moov: &mut [u8], span: Span, shift: u64, media: (u64, u64)) -> Option<()> {
    let body = span.start + header(moov, span);
    if CONTAINERS.contains(&&span.kind) {
        for child in boxes(moov, body, span.start + span.len)? {
            patch(moov, child, shift, media)?;
        }
        return Some(());
    }
    let wide = match &span.kind {
        b"stco" => false,
        b"co64" => true,
        _ => return Some(()),
    };
    let count = usize::try_from(u32::from_be_bytes(
        moov.get(body + 4..body + 8)?.try_into().ok()?,
    ))
    .ok()?;
    let width = if wide { 8 } else { 4 };
    let table = body + 8;
    if table + count.checked_mul(width)? > span.start + span.len {
        return None;
    }
    for i in 0..count {
        let at = table + i * width;
        let old = if wide {
            u64::from_be_bytes(moov[at..at + 8].try_into().ok()?)
        } else {
            u64::from(u32::from_be_bytes(moov[at..at + 4].try_into().ok()?))
        };
        if old < media.0 || old >= media.1 {
            return None;
        }
        let new = old.checked_add(shift)?;
        if wide {
            moov[at..at + 8].copy_from_slice(&new.to_be_bytes());
        } else {
            moov[at..at + 4].copy_from_slice(&u32::try_from(new).ok()?.to_be_bytes());
        }
    }
    Some(())
}

/// `bytes` with a trailing `moov` moved ahead of the first `mdat` and its
/// chunk offsets shifted to match; `bytes` itself when the index already
/// leads, the file is fragmented, or anything fails to parse.
#[cfg(feature = "decode-ffmpeg")]
pub(crate) fn faststart(bytes: &[u8]) -> Cow<'_, [u8]> {
    match reorder(bytes) {
        Some(moved) => Cow::Owned(moved),
        None => Cow::Borrowed(bytes),
    }
}

#[cfg(feature = "decode-ffmpeg")]
fn reorder(bytes: &[u8]) -> Option<Vec<u8>> {
    if bytes.get(4..8)? != b"ftyp" {
        return None;
    }
    let top = boxes(bytes, 0, bytes.len())?;
    if top.iter().any(|b| &b.kind == b"moof") {
        return None;
    }
    let first_media = top.iter().position(|b| &b.kind == b"mdat")?;
    let index = top.iter().position(|b| &b.kind == b"moov")?;
    if index < first_media || top.iter().filter(|b| &b.kind == b"moov").count() != 1 {
        return None;
    }
    let moov = top[index];
    // The media moves back by the index's size; everything the offsets name
    // lies between the first `mdat` and the index.
    let media = (
        u64::try_from(top[first_media].start).ok()?,
        u64::try_from(moov.start).ok()?,
    );
    let shift = u64::try_from(moov.len).ok()?;
    let mut index_bytes = bytes[moov.start..moov.start + moov.len].to_vec();
    patch(
        &mut index_bytes,
        Span {
            kind: moov.kind,
            start: 0,
            len: moov.len,
        },
        shift,
        media,
    )?;
    let mut out = Vec::with_capacity(bytes.len());
    for (i, b) in top.iter().enumerate() {
        if i == first_media {
            out.extend_from_slice(&index_bytes);
        }
        if i != index {
            out.extend_from_slice(&bytes[b.start..b.start + b.len]);
        }
    }
    Some(out)
}

#[cfg(all(test, feature = "decode-ffmpeg"))]
#[path = "mp4_tests.rs"]
mod tests;
