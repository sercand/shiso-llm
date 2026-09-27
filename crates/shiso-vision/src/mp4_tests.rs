//! Moving an MP4's index ahead of its media, over synthetic box trees.

use std::borrow::Cow;

use super::faststart;

fn boxed(kind: [u8; 4], body: &[u8]) -> Vec<u8> {
    let mut out = u32::try_from(8 + body.len())
        .expect("small")
        .to_be_bytes()
        .to_vec();
    out.extend_from_slice(&kind);
    out.extend_from_slice(body);
    out
}

/// A chunk-offset table: version and flags, the count, then the offsets.
fn stco(offsets: &[u32]) -> Vec<u8> {
    let mut body = vec![0u8; 4];
    body.extend(u32::try_from(offsets.len()).expect("small").to_be_bytes());
    for o in offsets {
        body.extend(o.to_be_bytes());
    }
    boxed(*b"stco", &body)
}

fn co64(offsets: &[u64]) -> Vec<u8> {
    let mut body = vec![0u8; 4];
    body.extend(u32::try_from(offsets.len()).expect("small").to_be_bytes());
    for o in offsets {
        body.extend(o.to_be_bytes());
    }
    boxed(*b"co64", &body)
}

fn moov(table: &[u8]) -> Vec<u8> {
    let stbl = boxed(*b"stbl", table);
    let minf = boxed(*b"minf", &stbl);
    let mdia = boxed(*b"mdia", &minf);
    let trak = boxed(*b"trak", &mdia);
    boxed(*b"moov", &[boxed(*b"mvhd", &[0u8; 100]), trak].concat())
}

/// `ftyp`, `free`, `mdat` then `moov`: the layout a phone's recorder writes.
/// Returns the file and the absolute offsets of its two chunks.
fn trailing_index(table: impl Fn(&[u64]) -> Vec<u8>) -> (Vec<u8>, Vec<u64>) {
    let head = [boxed(*b"ftyp", b"M4A \0\0\0\0M4A "), boxed(*b"free", &[])].concat();
    let payload: Vec<u8> = (0..64u8).collect();
    let mdat = boxed(*b"mdat", &payload);
    let base = (head.len() + 8) as u64;
    let chunks = vec![base, base + 32];
    let file = [head, mdat, moov(&table(&chunks))].concat();
    (file, chunks)
}

/// The top-level box kinds, in order.
fn order(file: &[u8]) -> Vec<String> {
    let mut kinds = Vec::new();
    let mut at = 0;
    while at + 8 <= file.len() {
        let size = u32::from_be_bytes(file[at..at + 4].try_into().expect("4")) as usize;
        kinds.push(String::from_utf8_lossy(&file[at + 4..at + 8]).into_owned());
        at += size;
    }
    kinds
}

/// Every offset in the file's first chunk-offset table.
fn offsets(file: &[u8]) -> Vec<u64> {
    for (kind, wide) in [(b"stco", false), (b"co64", true)] {
        if let Some(at) = file.windows(4).position(|w| w == kind) {
            let n = u32::from_be_bytes(file[at + 8..at + 12].try_into().expect("4")) as usize;
            let width = if wide { 8 } else { 4 };
            return (0..n)
                .map(|i| {
                    let s = at + 12 + i * width;
                    if wide {
                        u64::from_be_bytes(file[s..s + 8].try_into().expect("8"))
                    } else {
                        u64::from(u32::from_be_bytes(file[s..s + 4].try_into().expect("4")))
                    }
                })
                .collect();
        }
    }
    Vec::new()
}

#[test]
fn a_trailing_index_moves_ahead_and_its_offsets_follow_the_media() {
    for wide in [false, true] {
        let (file, chunks) = trailing_index(|c| {
            if wide {
                co64(c)
            } else {
                stco(
                    &c.iter()
                        .map(|&o| u32::try_from(o).expect("small"))
                        .collect::<Vec<_>>(),
                )
            }
        });
        let moved = faststart(&file);
        assert!(matches!(moved, Cow::Owned(_)), "wide {wide}: reordered");
        assert_eq!(
            order(&moved),
            ["ftyp", "free", "moov", "mdat"],
            "wide {wide}"
        );
        assert_eq!(moved.len(), file.len(), "nothing added or lost");
        let now = offsets(&moved);
        for (old, new) in chunks.iter().zip(&now) {
            let (old, new) = (
                usize::try_from(*old).expect("fits"),
                usize::try_from(*new).expect("fits"),
            );
            assert_eq!(
                file[old..old + 32],
                moved[new..new + 32],
                "wide {wide}: each offset still names the same media bytes"
            );
        }
    }
}

#[test]
fn an_index_already_ahead_is_left_alone() {
    let head = boxed(*b"ftyp", b"isom\0\0\0\0isom");
    let file = [head, moov(&stco(&[200])), boxed(*b"mdat", &[1u8; 16])].concat();
    assert!(matches!(faststart(&file), Cow::Borrowed(_)));
}

#[test]
fn anything_that_does_not_parse_is_passed_through_untouched() {
    let (mut file, _) = trailing_index(|c| {
        stco(
            &c.iter()
                .map(|&o| u32::try_from(o).expect("small"))
                .collect::<Vec<_>>(),
        )
    });
    // A box that claims to run past the end of the file.
    let last = file.len() - 4;
    file[last - 4..last].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(matches!(faststart(&file), Cow::Borrowed(_)));
    assert!(matches!(faststart(b"not an mp4 at all"), Cow::Borrowed(_)));
    // Fragmented files carry their own offsets in `moof`; they are not moved.
    let (frag, _) = trailing_index(|_| stco(&[]));
    let frag = [frag, boxed(*b"moof", &[0u8; 8])].concat();
    assert!(matches!(faststart(&frag), Cow::Borrowed(_)));
}
