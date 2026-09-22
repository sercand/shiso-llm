//! The loud refusal, which is the only contract a backend knob can have.
//!
//! `SHISO_PLE_IO=uring` means "the ring or nothing". On a target with no
//! `io_uring` that has to be an error naming it, not a quiet fall back to
//! the thread pool - otherwise a measurement or a test that meant to exercise
//! the ring exercises the pool and says it exercised the ring.
//!
//! This is the whole of this binary on purpose: it sets a process-wide
//! variable, and a second test in the same process would race with it.

#![allow(
    unsafe_code,
    reason = "the contract under test is an environment variable, and writing one is unsafe in this edition; being the only test in this binary is what makes the write sound"
)]

use shiso_core::ids::TokenId;
use shiso_ple::{PleConfig, PleIo, PleParams, PleParamsDesc, PleRowType, PleStream};

fn params() -> PleParams {
    PleParams::new(PleParamsDesc {
        row_type: PleRowType::BlockNl4,
        ngram_size: 2,
        heads_per_ngram: 1,
        head_dim: 64,
        eos_token: TokenId::new(9),
        multipliers: vec![3, 5],
        head_offsets: vec![0],
        head_vocab: vec![64],
        head_bias: None,
    })
    .expect("valid")
}

#[test]
fn insisting_on_the_ring_where_there_is_none_fails_by_name() {
    // SAFETY: this is the only test in this binary, so no other thread of this
    // process is reading or writing the environment while it is set, and the
    // stream below is opened after the write returns.
    unsafe { std::env::set_var("SHISO_PLE_IO", "uring") };

    let file = tempfile::tempfile().expect("a temporary file");
    let opened = PleStream::open(
        file,
        0,
        64,
        params(),
        0,
        // Even a caller that asked for the pool is overridden: the variable is
        // the operator's, and its whole point is to force a backend the build
        // would not have chosen.
        &PleConfig {
            io: PleIo::Pool,
            ..PleConfig::default()
        },
    );

    #[cfg(target_os = "linux")]
    {
        // On a host with a working ring this is the success path, and the
        // backend must report the ring rather than the pool. `[INFERENCE]`
        // until it has run on such a host.
        match opened {
            Ok(stream) => assert_eq!(stream.backend(), shiso_ple::PleBackend::Uring),
            Err(err) => {
                // A kernel without io_uring is the other legitimate outcome,
                // and it too has to name the thing that is missing.
                assert!(format!("{err}").contains("io_uring"), "{err:?}");
            }
        }
    }
    #[cfg(not(target_os = "linux"))]
    {
        let err = opened.expect_err("this target has no io_uring");
        assert!(
            matches!(err, shiso_ple::PleError::BackendRefused(message) if message.contains("io_uring")),
            "the refusal has to name io_uring: {err:?}"
        );
    }
}
