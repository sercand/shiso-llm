//! Frame extraction for real-world containers, through a subprocess decoder.
//!
//! # Why a subprocess and not a linked decoder
//!
//! A linked pure-Rust H.264 decoder plus a demuxer builds in seconds and needs
//! no runtime dependency, but it covers H.264 ONLY - no H.265, VP9 or AV1,
//! which between them are a large share of what people actually send - and the
//! royalty-free patent grants in that space cover binaries the grantor
//! distributes, not a source build redistributed by someone else. That is a
//! licensing question for a project to answer deliberately, not one to settle
//! by adding a dependency.
//!
//! A subprocess needs no build or link dependency, decodes everything, and can
//! be swapped for an in-process decoder later without changing this module's
//! signatures. The cost is a runtime binary and a process spawn per request, so
//! it is OPT-IN - a caller that never constructs a [`FfmpegPolicy`] never
//! reaches a spawn - and an absent binary is reported by name rather than as a
//! decode failure.
//!
//! # What is bounded
//!
//! Everything the caller controls, because the input is an untrusted byte blob
//! from an HTTP request:
//!
//! - **No shell.** Arguments are passed as argv. Nothing is interpolated into a
//!   command string, so no input can become a flag or a second command.
//! - **No temp file.** The container goes in over stdin, so there is no path to
//!   traverse, collide on, or leave behind.
//! - **The decoder cannot reach for a terminal**, and stdin is the pipe.
//! - **Frame count** is capped by the decoder's own flag, so a long clip cannot
//!   decode forever.
//! - **Output size** is capped while reading, and the child is killed the
//!   moment the cap is passed rather than after it finishes.
//! - **Wall clock** is capped by a watchdog that kills the child; a decoder
//!   that hangs must not hold a request thread indefinitely.
//! - **No URL is ever opened.** Input is a pipe and output is a pipe, so the
//!   server-side request forgery surface that a remote-image fetch has simply
//!   does not exist here.

use std::io::{Read, Write};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

use crate::error::VisionError;
use crate::video::DecodedFrame;

/// The 8-byte PNG signature. Frames arrive concatenated on one stream, so this
/// is how they are separated.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// Frames the decoder may emit for one request, unless the operator says
/// otherwise. Matches the sampling ceiling, so the cap never truncates a clip
/// the sampler would have kept.
const DEFAULT_MAX_FRAMES: usize = 768;

/// Bytes of decoded output the reader will accept for one request. A clip of
/// the default frame count at ordinary resolutions sits comfortably under this;
/// the cap exists to bound a pathological stream, not to size a normal one.
const DEFAULT_MAX_OUTPUT_BYTES: usize = 512 * 1024 * 1024;

/// Wall-clock budget for one decode.
const DEFAULT_TIMEOUT_SECS: u64 = 120;

/// How often the watchdog checks whether the child is still running.
const WATCHDOG_TICK: Duration = Duration::from_millis(100);

/// Operator policy for subprocess decoding.
///
/// There is deliberately no `enabled` flag. The disabled state is the ABSENCE
/// of a policy, so there is no way to hold a value whose other fields are
/// meaningless, and no code path that has a policy in hand but must remember to
/// check a boolean before using it. Subprocess decoding is off by default
/// because the default is no policy at all.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub struct FfmpegPolicy {
    binary: PathBuf,
    max_frames: NonZeroUsize,
    max_output_bytes: NonZeroUsize,
    timeout_secs: NonZeroU64,
}

impl FfmpegPolicy {
    /// A policy running `binary` under the default caps.
    ///
    /// A bare name is resolved on `PATH`; an absolute path is used as given, so
    /// a deployment can pin a known build rather than inherit whatever the
    /// environment happens to offer.
    #[must_use]
    pub fn new(binary: PathBuf) -> Self {
        Self {
            binary,
            max_frames: NonZeroUsize::new(DEFAULT_MAX_FRAMES).unwrap_or(NonZeroUsize::MIN),
            max_output_bytes: NonZeroUsize::new(DEFAULT_MAX_OUTPUT_BYTES)
                .unwrap_or(NonZeroUsize::MIN),
            timeout_secs: NonZeroU64::new(DEFAULT_TIMEOUT_SECS).unwrap_or(NonZeroU64::MIN),
        }
    }

    /// The configured binary.
    #[must_use]
    pub fn binary(&self) -> &Path {
        &self.binary
    }

    /// Frames the decoder is allowed to emit.
    #[must_use]
    pub fn max_frames(&self) -> usize {
        self.max_frames.get()
    }

    /// Bytes of decoded output the reader will accept.
    #[must_use]
    pub fn max_output_bytes(&self) -> usize {
        self.max_output_bytes.get()
    }

    /// Wall-clock budget for one decode, in seconds.
    #[must_use]
    pub fn timeout_secs(&self) -> u64 {
        self.timeout_secs.get()
    }

    /// Caps the frames one decode may emit.
    #[must_use]
    pub fn with_max_frames(mut self, frames: NonZeroUsize) -> Self {
        self.max_frames = frames;
        self
    }

    /// Caps the decoded bytes one decode may produce.
    #[must_use]
    pub fn with_max_output_bytes(mut self, bytes: NonZeroUsize) -> Self {
        self.max_output_bytes = bytes;
        self
    }

    /// Sets the wall-clock budget for one decode.
    #[must_use]
    pub fn with_timeout_secs(mut self, secs: NonZeroU64) -> Self {
        self.timeout_secs = secs;
        self
    }
}

impl Default for FfmpegPolicy {
    /// The decoder resolved from `PATH`, under the default caps.
    ///
    /// This is the shape of an ENABLED policy; it is not the default posture of
    /// the pipeline, which is to hold no policy at all.
    fn default() -> Self {
        Self::new(PathBuf::from("ffmpeg"))
    }
}

/// What a startup probe found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// Usable, with the version line it reported.
    Ready {
        /// First line of the decoder's own version output.
        version: String,
    },
    /// Configured but not runnable.
    Missing {
        /// The binary as configured.
        binary: PathBuf,
        /// Why it could not be used.
        reason: String,
    },
    /// No policy was supplied, so nothing was asked for.
    Disabled,
}

/// Checks at BOOT whether the configured decoder can actually run.
///
/// Worth doing eagerly rather than discovering it on the first video request: a
/// deployment that enabled video decoding and does not have the binary is
/// misconfigured, and the operator should learn that while reading the startup
/// log, not from a user's failed request an hour later. The check is one
/// version invocation, so it costs nothing at boot.
#[must_use]
pub fn probe(policy: Option<&FfmpegPolicy>) -> Availability {
    let Some(policy) = policy else {
        return Availability::Disabled;
    };
    let output = Command::new(policy.binary())
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output();
    match output {
        Ok(out) if out.status.success() => {
            let version = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("unknown version")
                .trim()
                .to_owned();
            Availability::Ready { version }
        }
        Ok(out) => Availability::Missing {
            binary: policy.binary().to_path_buf(),
            reason: format!("it ran but exited {}", out.status),
        },
        Err(error) => Availability::Missing {
            binary: policy.binary().to_path_buf(),
            reason: format!("it could not be run: {error}"),
        },
    }
}

/// Decodes container bytes to RGB frames sampled at `target_fps`.
///
/// The decoder performs the temporal sampling itself, which is both faster and
/// more accurate than decoding everything and discarding most of it, and it
/// means the caller does not have to know the source frame rate. The frames
/// that come back are therefore ALREADY at `target_fps`.
///
/// # Errors
///
/// Every error [`run_decoder`] returns, plus [`VisionError::FramePng`] for a
/// frame that does not decode and [`VisionError::NoFrames`] for a container
/// with no video stream.
pub fn decode_frames(
    bytes: &[u8],
    target_fps: f64,
    policy: &FfmpegPolicy,
) -> Result<Vec<DecodedFrame>, VisionError> {
    let fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        crate::video::DEFAULT_FPS
    };
    let filter = format!("fps={fps}");
    let frame_cap = policy.max_frames().to_string();
    let out = run_decoder(
        policy,
        &[
            "-vf",
            &filter,
            "-frames:v",
            &frame_cap,
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
        ],
        bytes,
    )?;
    let frames = split_png_stream(&out)?;
    if frames.is_empty() {
        return Err(VisionError::NoFrames);
    }
    Ok(frames)
}

/// Runs the decoder over `bytes` with `output_args` and returns its raw output.
///
/// The command line is `-v error -nostdin -i pipe:0 <output_args…> pipe:1`:
/// the container goes in over stdin and whatever the output arguments ask for
/// comes back over stdout, under every cap the policy sets. The caller owns
/// the meaning of the bytes - frames, samples, anything the decoder can write
/// to a pipe.
///
/// # Errors
///
/// [`VisionError::DecoderSpawn`] when the binary cannot be started, which is a
/// deployment problem and reads as one; [`VisionError::DecoderTimeout`] and
/// [`VisionError::DecoderOutputCap`] when a cap is hit; and
/// [`VisionError::DecoderFailed`] carrying the decoder's own last complaint.
pub fn run_decoder(
    policy: &FfmpegPolicy,
    output_args: &[&str],
    bytes: &[u8],
) -> Result<Vec<u8>, VisionError> {
    tracing::debug!(binary = ?policy.binary(), "spawning the subprocess decoder");
    let mut args: Vec<&str> = vec![
        "-v", "error",
        // Without this the decoder can block on a prompt whenever it thinks
        // stdin is interactive.
        "-nostdin", "-i", "pipe:0",
    ];
    args.extend_from_slice(output_args);
    args.push("pipe:1");
    let mut child = Command::new(policy.binary())
        .args(&args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|source| VisionError::DecoderSpawn {
            binary: policy.binary().to_path_buf(),
            source,
        })?;

    let (Some(mut stdin), Some(mut stdout), Some(mut stderr)) =
        (child.stdin.take(), child.stdout.take(), child.stderr.take())
    else {
        return Err(VisionError::Io {
            what: "opening the decoder's pipes",
            source: std::io::Error::other("the decoder was spawned without its pipes"),
        });
    };

    // Feed the container on a thread. It MUST be concurrent with reading: the
    // decoder writes output while still consuming input, so a write-then-read
    // sequence deadlocks as soon as the output pipe buffer fills.
    let input = bytes.to_vec();
    let writer = std::thread::spawn(move || {
        // A broken pipe here is normal - the decoder stops reading once it has
        // the frames it was asked for - so the error is deliberately dropped.
        let _ = stdin.write_all(&input);
        drop(stdin);
    });

    let child = Arc::new(Mutex::new(child));
    let watchdog = spawn_watchdog(&Arc::clone(&child), policy.timeout_secs());

    // `take` bounds the read without trusting the child, and one byte over the
    // cap is enough to detect the overrun.
    let cap = policy.max_output_bytes();
    let mut out = Vec::new();
    let read_result = (&mut stdout)
        .take(u64::try_from(cap).unwrap_or(u64::MAX).saturating_add(1))
        .read_to_end(&mut out);

    // KILL FIRST, then drain stderr. The order matters and getting it wrong
    // deadlocks: once the cap is hit this stops reading stdout, so the child
    // blocks writing into a full pipe and never closes stderr, and a read on
    // stderr then waits for an end-of-file that only the watchdog will ever
    // cause. Killing here turns a two-minute hang into an immediate error.
    // Draining stdout instead would defeat the cap, which is the thing being
    // enforced.
    let over_cap = out.len() > cap;
    if over_cap {
        let _ = child.lock().kill();
    }

    let mut stderr_text = String::new();
    let _ = stderr.read_to_string(&mut stderr_text);
    let _ = writer.join();

    let status = child.lock().wait().map_err(|source| VisionError::Io {
        what: "waiting for the decoder",
        source,
    })?;
    let timed_out = watchdog.join().unwrap_or(false);

    read_result.map_err(|source| VisionError::Io {
        what: "reading decoded output",
        source,
    })?;
    if timed_out {
        return Err(VisionError::DecoderTimeout {
            secs: policy.timeout_secs(),
        });
    }
    if over_cap {
        return Err(VisionError::DecoderOutputCap { cap });
    }
    if !status.success() {
        let detail = stderr_text
            .lines()
            .next_back()
            .unwrap_or("no detail")
            .trim()
            .to_owned();
        return Err(VisionError::DecoderFailed { detail });
    }

    Ok(out)
}

/// Starts the watchdog that kills a decoder which overruns its budget.
///
/// Returns a handle whose value is whether the watchdog was the one that ended
/// the process, which is the only way to tell a killed decoder apart from one
/// that failed on its own.
fn spawn_watchdog(
    child: &Arc<Mutex<std::process::Child>>,
    timeout_secs: u64,
) -> std::thread::JoinHandle<bool> {
    let child = Arc::clone(child);
    std::thread::spawn(move || {
        let deadline = Instant::now() + Duration::from_secs(timeout_secs);
        loop {
            std::thread::sleep(WATCHDOG_TICK);
            let mut guard = child.lock();
            match guard.try_wait() {
                Ok(Some(_)) | Err(_) => return false,
                Ok(None) => {}
            }
            if Instant::now() >= deadline {
                let _ = guard.kill();
                tracing::warn!(
                    timeout_secs,
                    "killed the subprocess decoder at its deadline"
                );
                return true;
            }
        }
    })
}

/// Splits a concatenated PNG stream into frames.
///
/// Split on the SIGNATURE rather than by parsing end-of-image chunks: a PNG's
/// compressed payload can legitimately contain that chunk's byte pattern,
/// whereas the 8-byte signature only appears at a file start in a well-formed
/// stream.
///
/// # Errors
///
/// [`VisionError::FramePng`] naming the frame that did not decode.
fn split_png_stream(buf: &[u8]) -> Result<Vec<DecodedFrame>, VisionError> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + PNG_MAGIC.len() <= buf.len() {
        if buf.get(i..i + PNG_MAGIC.len()) == Some(&PNG_MAGIC[..]) {
            starts.push(i);
            i += PNG_MAGIC.len();
        } else {
            i += 1;
        }
    }
    let mut frames = Vec::with_capacity(starts.len());
    for (index, &start) in starts.iter().enumerate() {
        let end = starts.get(index + 1).copied().unwrap_or(buf.len());
        let decoded = image::load_from_memory_with_format(
            buf.get(start..end).unwrap_or_default(),
            image::ImageFormat::Png,
        )
        .map_err(|source| VisionError::FramePng { index, source })?;
        frames.push(DecodedFrame::from_image(decoded.to_rgb8()));
    }
    Ok(frames)
}

#[cfg(test)]
#[path = "ffmpeg_tests.rs"]
mod tests;
