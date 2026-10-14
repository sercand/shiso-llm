//! `NVDEC`, NVIDIA's hardware video decoder, reached through the driver's
//! `libnvcuvid` at run time.
//!
//! The libraries are loaded on first use, so a machine without the NVIDIA driver
//! answers [`crate::HwError::Unavailable`] and the decode chain moves on. A
//! session shares device 0's primary context with the compute backend, makes it
//! current around every driver call (decoding runs on request threads), feeds
//! the container's samples to NVIDIA's bitstream parser as an elementary stream,
//! and copies each kept picture to host memory, where the shared colour module
//! converts it to RGB.
//!
//! The pieces that need no driver - the bindings' layouts, the Annex-B
//! conversion, configuration-record parsing, capability and error decisions,
//! surface conversion and the session cap - are compiled for tests on every
//! platform, so they are exercised on machines without a GPU. The driver
//! loading and the session itself are Linux-only.
#![cfg_attr(
    not(target_os = "linux"),
    expect(
        dead_code,
        reason = "off Linux only the driver-free pieces are compiled, for their unit tests; \
                  the session that uses them is Linux-only"
    )
)]

mod annexb;
mod config;
mod ffi;
mod semaphore;
mod surface;

#[cfg(target_os = "linux")]
mod driver;
#[cfg(target_os = "linux")]
mod session;

#[cfg(target_os = "linux")]
pub(crate) use session::{Session, probe};

#[cfg(all(test, target_os = "linux"))]
mod device {
    //! Device tests: they need the NVIDIA driver and a GPU with NVDEC, so they
    //! are ignored by default and say on stderr why they did nothing when the
    //! driver is missing. Run with `cargo test -p shiso-media-sys -- --ignored`
    //! on the GB10.

    use crate::{HwCodec, HwDecoder, HwDecoderDesc, HwError};

    /// Prints why a device test did nothing and reports whether it should stop.
    #[expect(clippy::print_stderr, reason = "a skipped case says why on stderr")]
    fn skipped(err: &HwError) -> bool {
        if let HwError::Unavailable { reason } = err {
            eprintln!("note: NVDEC is unavailable on this machine, skipping: {reason}");
            return true;
        }
        false
    }

    #[test]
    #[ignore = "needs the NVIDIA driver and an NVDEC-capable GPU"]
    fn probe_reports_h264() {
        match crate::probe() {
            Ok(probe) => {
                assert_eq!(probe.decoder, "NVDEC");
                assert!(probe.codecs.contains(&HwCodec::H264), "{probe:?}");
            }
            Err(err) if skipped(&err) => {}
            Err(err) => panic!("probe failed: {err}"),
        }
    }

    #[test]
    #[ignore = "needs the NVIDIA driver and an NVDEC-capable GPU"]
    fn opens_and_drops_more_sessions_than_the_cap() {
        // avcC, High profile, 4-byte lengths, one SPS and one PPS. `open` only
        // reads the record's structure; the parser sees these bytes on the
        // first sample, which this test never sends.
        let avcc = [
            1, 100, 0, 31, 0xff, 0xe1, 0, 4, 0x67, 100, 0, 31, 1, 0, 2, 0x68, 0xee,
        ];
        let desc = HwDecoderDesc::new(HwCodec::H264, &avcc, 1280, 720);
        // Opened and dropped one at a time, more than the session cap: each drop
        // has to return its permit and context retain, or this hangs.
        for _ in 0..(super::session::MAX_NVDEC_SESSIONS * 2) {
            match HwDecoder::open(&desc) {
                Ok(mut decoder) => {
                    let flushed = decoder.flush().expect("flushing an unfed decoder");
                    assert!(flushed.is_empty());
                }
                Err(err) if skipped(&err) => return,
                Err(err) => panic!("open failed: {err}"),
            }
        }
    }
}
