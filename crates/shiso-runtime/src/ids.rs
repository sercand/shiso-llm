//! Identifiers and clocks the wire formats need.
//!
//! # Why this is not a dependency
//!
//! Every surface has to mint an opaque identifier for a response and stamp it
//! with a wall-clock second. Both are four lines over the operating system's own
//! entropy source, and neither is load bearing for anything but readability: no
//! client parses these, no two of them are ever compared for ordering, and a
//! collision costs a duplicated log line. A crate for that would be a supply
//! chain for a coin flip.
//!
//! # Why the prefixes are not here
//!
//! `chatcmpl-`, `cmpl-` and `resp_` are facts about one wire format, and the
//! surface that owns the format mints them from [`uuid_v4`]. A helper here that
//! knew about prefixes would be this module knowing which surfaces exist.

use std::time::{SystemTime, UNIX_EPOCH};

/// A random version-4 identifier, hyphenated, lower case.
///
/// The bytes come from the operating system. When that read fails - a container
/// with no `/dev` entry, an exhausted descriptor table - the fallback mixes the
/// clock with the address of a stack local rather than refusing: a response
/// without an identifier is useless to a client, and an identifier that is
/// merely hard to predict is enough for a value nothing authenticates.
#[must_use]
pub fn uuid_v4() -> String {
    let mut bytes = random_bytes();
    // Version 4, variant 1. Written even on the fallback path so that a client
    // which validates the shape accepts the value either way.
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    let hex = bytes
        .iter()
        .fold(String::with_capacity(32), |mut out, byte| {
            use std::fmt::Write as _;
            // Writing into a String cannot fail; the result is discarded rather than
            // unwrapped so a formatting change can never introduce a panic here.
            let _ = write!(out, "{byte:02x}");
            out
        });
    let mut out = String::with_capacity(36);
    for (at, chunk) in [0..8, 8..12, 12..16, 16..20, 20..32]
        .into_iter()
        .enumerate()
    {
        if at > 0 {
            out.push('-');
        }
        out.push_str(hex.get(chunk).unwrap_or_default());
    }
    out
}

/// Seconds since the epoch, as every `created` field on every surface wants it.
///
/// A clock before the epoch answers zero rather than failing. The value is a
/// label on a response, and a server whose clock is wrong should still answer.
#[must_use]
pub fn unix_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// A fresh seed for a request that named none.
///
/// Sampling entropy and identifier entropy come from the same place on purpose:
/// there is one source of randomness in this process, and a second one would be
/// a second thing to check when a run that should have been reproducible was
/// not.
#[must_use]
pub fn entropy_u64() -> u64 {
    let bytes = random_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Sixteen bytes from the operating system, or from the clock when it refuses.
///
/// `read_exact` rather than a whole-file read: the entropy device never reports
/// end of file, so reading it to completion is a hang rather than a value.
fn random_bytes() -> [u8; 16] {
    use std::io::Read as _;
    let mut out = [0u8; 16];
    match std::fs::File::open("/dev/urandom").and_then(|mut file| file.read_exact(&mut out)) {
        Ok(()) => out,
        Err(_unavailable) => fallback_bytes(),
    }
}

/// The clock-derived fallback.
///
/// Split out so the entropy path above stays readable, and so a test can reach
/// it on a machine where `/dev/urandom` exists.
fn fallback_bytes() -> [u8; 16] {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0u128, |since| since.as_nanos());
    // The low 64 bits, taken with a mask rather than a cast, because a cast that
    // truncates is the shape of every silent numeric defect in this workspace.
    let low = u64::try_from(nanos & u128::from(u64::MAX)).unwrap_or(0);
    let counter = {
        use std::sync::atomic::{AtomicU64, Ordering};
        static NEXT: AtomicU64 = AtomicU64::new(0);
        NEXT.fetch_add(1, Ordering::Relaxed)
    };
    let mixed = low ^ counter.wrapping_mul(0x9e37_79b9_7f4a_7c15);
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&mixed.to_le_bytes());
    out[8..].copy_from_slice(&low.rotate_left(17).to_le_bytes());
    out
}

#[cfg(test)]
mod tests {
    use super::{fallback_bytes, unix_timestamp, uuid_v4};

    #[test]
    fn an_identifier_has_the_shape_a_client_that_validates_one_expects() {
        let id = uuid_v4();
        assert_eq!(id.len(), 36, "{id}");
        let groups: Vec<&str> = id.split('-').collect();
        assert_eq!(
            groups.iter().map(|group| group.len()).collect::<Vec<_>>(),
            vec![8, 4, 4, 4, 12],
            "{id}"
        );
        assert!(
            id.chars().all(|c| c.is_ascii_hexdigit() || c == '-'),
            "{id}"
        );
        assert!(groups[2].starts_with('4'), "version nibble: {id}");
        assert!(
            matches!(
                groups[3].as_bytes().first(),
                Some(b'8' | b'9' | b'a' | b'b')
            ),
            "variant nibble: {id}"
        );
    }

    /// Two identifiers minted back to back must differ, or every response in a
    /// log looks like the same one.
    #[test]
    fn two_identifiers_minted_together_are_not_the_same_identifier() {
        assert_ne!(uuid_v4(), uuid_v4());
    }

    /// The fallback is the path a container with no `/dev` takes, and it has to
    /// keep producing distinct values there too.
    #[test]
    fn the_clock_fallback_still_produces_distinct_bytes() {
        assert_ne!(fallback_bytes(), fallback_bytes());
    }

    #[test]
    fn the_clock_is_read_as_seconds_since_the_epoch() {
        // Anything after 2020. A zero here means the clock read failed, which is
        // the one outcome that would make every `created` field identical.
        assert!(unix_timestamp() > 1_577_836_800);
    }
}
