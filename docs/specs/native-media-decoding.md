# Native media decoding — design

Date: 2026-10-14
Status: draft v2, verified against the repo and crates.io on 2026-10-14

## Purpose

Video, and every audio container other than WAV, is decoded today by spawning
an `ffmpeg` binary the user has to install. The goal is a server that decodes
the media people actually send without that binary:

- **OS decoders first, always.** VideoToolbox on macOS and NVDEC on Linux
  decode H.264, HEVC, VP9 and AV1 under the platform's own codec licences and
  add almost nothing to the build. H.264 and HEVC are decoded **only** here or
  by ffmpeg; no codec with a patent-licensing question is linked into the
  server.
- **Pure-Rust decoders as an opt-in build feature**, because they add build
  time: `symphonia` for audio, `rav1d` for AV1 (royalty-free codec).
- **ffmpeg stays available as an explicit fallback**, enabled by pointing the
  server at a binary. Without `--ffmpeg-path` ffmpeg is never loaded or
  spawned.

Success means:

- A `--no-default-features --features decode-native` build on macOS serves
  H.264/HEVC mp4 video (VideoToolbox) and mp3/m4a/flac/ogg audio (symphonia)
  with no ffmpeg on the machine.
- A build with `decode-ffmpeg` and `--ffmpeg-path` set decodes everything it
  does today.

Non-goals:
- Opus audio without ffmpeg. `opus-rs` is too young to depend on yet.
- A pure-Rust H.264 decoder. Considered (`rusty_h264-decoder`) and dropped:
  H.264 is OS or ffmpeg.
- An OS audio step on macOS (AudioToolbox). See "Later".
- VA-API for Intel/AMD GPUs on Linux.
- Containers other than MP4/MOV and MKV/WebM on the non-ffmpeg paths. AVI
  goes to ffmpeg or is refused.
- A GPU colour-conversion kernel.

## Decisions already made

| Decision | Choice |
|---|---|
| Order | OS → Rust → ffmpeg; each step exists only if the build has it and, for ffmpeg, only if `--ffmpeg-path` is set |
| OS decoders | Always compiled; used when available at runtime |
| Build features | `decode-ffmpeg` (default) and `decode-native` (additive) |
| Supported chains | os → ffmpeg (default), os → rust → ffmpeg, os → rust |
| Rust video | `rav1d` (AV1) only. H.264/HEVC/VP9 are OS or ffmpeg |
| Video opt-in | New `--video-allow-decode` gates the whole video chain, OS step included |
| ffmpeg opt-in | `--ffmpeg-path` (one flag, no default). Set → the ffmpeg step exists, for audio and video. Unset → it does not |
| Removed flags | `--video-allow-ffmpeg`, `--audio-allow-ffmpeg`, `--video-ffmpeg-path` (replaced by `--ffmpeg-path`; not aliased) |
| Audio opt-in | None beyond the above: any container the build can decode without ffmpeg is accepted; ffmpeg audio needs `--ffmpeg-path` |
| Unsafe | A new `shiso-media-sys` crate, the fifth allowed to relax `unsafe_code`. It holds the OS decoder FFI **and** the `rav1d` call-through |
| Fallthrough | Only on a typed *Unsupported*; malformed input is refused, never retried |

## Build features

The features live on `shiso-vision` and `shiso-audio`, and `shiso-server`
forwards them:

```toml
# shiso-server
default = ["decode-ffmpeg"]
decode-ffmpeg = ["shiso-vision/decode-ffmpeg", "shiso-audio/decode-ffmpeg"]
decode-native = ["shiso-vision/decode-native", "shiso-audio/decode-native"]
# shiso-audio
decode-ffmpeg = ["shiso-vision/decode-ffmpeg"]   # the ffmpeg runner lives in shiso-vision
```

`shiso-vision` and `shiso-audio` have **no** default features, and no library
crate enables a `decode-*` feature in its normal `[dependencies]`. The
workspace entries set `default-features = false`. Resolver 3 is already set;
the rule above is what keeps feature unification from switching ffmpeg back on.

`shiso-server` fails the build with `compile_error!` when neither feature is
on. The library crates build with neither, so `shiso-models`, `shiso-bench` and
their tests are unaffected by *features*; they are affected by the
`decode_audio` signature change (see Callers).

What each feature adds:

- **`decode-ffmpeg`:** `ffmpeg.rs` and the `--ffmpeg-path` flag. The flag is
  `cfg`-gated, so a native-only build refuses it instead of ignoring it.
- **`decode-native`:**
  - `symphonia` 0.6 with **explicit** features
    `["mp3", "aac", "isomp4", "flac", "ogg", "vorbis", "wav", "pcm", "mkv"]`
    (mp3, aac and isomp4 are off in its defaults), `default-features = false`;
  - `rav1d` 1.1 (BSD-2-Clause), consumed inside `shiso-media-sys`;
  - `yuvutils-rs` (YUV→RGB, BSD-3-Clause OR Apache-2.0).
  - `matroska-demuxer` (Zlib OR MIT OR Apache-2.0) is not in this list: the
    demuxer is always compiled.
  - Licences all appear in `deny.toml`'s allow list; confirm with
    `cargo deny check` when adding them.

## Architecture

```
            shiso-server (flags, policy, boot probe)
                 │
       ┌─────────┴──────────┐
  shiso-audio          shiso-vision
  decode chain         video decode chain ── demux (MP4, MKV)  [always]
   ├ symphonia [native]  ├ OS step ──────────► shiso-media-sys  [always; unsafe]
   └ ffmpeg [ffmpeg+path]│                       ├ macOS: VideoToolbox (objc2)
                         ├ rust step [native] ──►├ Linux: NVDEC (dlopen libnvcuvid)
                         │   (AV1 only)          └ av1: rav1d    [native]
                         └ ffmpeg [ffmpeg+path]
```

`shiso-media-sys` depends on nothing in the workspace except `thiserror` and
`tracing`. It never names `shiso-cuda` or `shiso-metal`, and `shiso-vision`
still names no backend. `shiso-vision` stays free of `unsafe`.

### The video chain (`shiso-vision`)

`decode_clip(bytes, fps, Option<&VideoDecodePolicy>)` keeps its shape. `None`
is still the disabled state, now meaning `--video-allow-decode` was not passed.

1. Empty bytes → `EmptyPayload`. A GIF decodes in process, as today.
2. `demux::open(bytes)` → `DemuxedTrack` or `DemuxError`.
   - **`Unsupported`** (AVI, a fragmented MP4, no video track the demuxer
     reads): the OS and Rust steps are skipped and the bytes go straight to
     ffmpeg, if that step exists.
   - **`Malformed`**: refused.
3. **OS step:** `shiso_media_sys::HwDecoder::open`.
   - `Unavailable` or `Unsupported` → the next step.
   - `Failed` → the request is refused.
4. **Rust step** (`decode-native`): AV1 only, through
   `shiso_media_sys::Av1Decoder`. Any other codec is `Unsupported`. 8- and
   10-bit 4:2:0 are handled; anything else (4:4:4, 12-bit, monochrome) is
   `Unsupported`.
5. **ffmpeg step** (`decode-ffmpeg`, `--ffmpeg-path` set): today's
   `decode_frames`, run on the original container bytes.
6. All steps `Unsupported` → `VisionError::NoVideoDecoder { container, codec,
   tried }`. The message lists what this build tried and how to get more
   (enable `decode-ffmpeg` and pass `--ffmpeg-path`, or build with
   `decode-native`).

Every step after the GIF case shares one internal seam, so the chain logic can
be tested with fakes:

```rust
enum Attempt { Done(Vec<DecodedFrame>), Unsupported(String) }
trait VideoStep { fn decode(&self, input: &StepInput<'_>, plan: &SamplePlan) -> Result<Attempt, VisionError>; }
```

`SamplePlan` turns `fps`, the track timestamps and `max_frames` into the set
of frame indices to keep, so the OS and Rust steps skip colour conversion for
frames they drop. It reuses the GIF path's sampler. The ffmpeg step ignores the
plan and keeps sampling with its own `fps=` filter.

**Caps without a child process.**
- In-process steps check frame count, output bytes and the
  `--video-decode-timeout-s` deadline between samples.
- They can't interrupt a single stuck sample. Accepted: `rav1d` is a Rust
  port with a small, fuzzed surface, and VideoToolbox decodes out of process
  (verify during the macOS phase).
- NVDEC's parser is closed-source C running inside the server, and `rav1d`
  is called through an `unsafe` C-ABI surface; that is why the video flag
  stays opt-in.

### Demux (`shiso-vision/src/demux/`, always compiled)

`DemuxedTrack { codec: VideoCodec, config: CodecConfig, width, height,
timescale, samples: Vec<Sample> }`. A `Sample` holds a byte range into the
request buffer (borrowed, not copied), its presentation timestamp, and whether
it is a keyframe. `VideoCodec` is `H264 | Hevc | Vp9 | Av1`.

- **MP4/MOV:** written in-house, extending the box walker in `mp4.rs`:
  - `moov/trak` with `hdlr = vide`;
  - `stsd` (`avc1`/`avc3` with `avcC`, `hvc1`/`hev1` with `hvcC`, `vp09` with
    `vpcC`, `av01` with `av1C`);
  - the sample tables `stsz`, `stco`/`co64`, `stsc`, `stts`, `ctts`, `stss`.
  - Fragmented MP4 is `Unsupported`.
  - Demuxing reads a seekable buffer, so `faststart` is only needed on the
    ffmpeg path.
- **MKV/WebM:** the `matroska-demuxer` crate. Codec IDs map as
  `V_MPEG4/ISO/AVC` → H264, `V_MPEGH/ISO/HEVC` → Hevc, `V_VP9` → Vp9,
  `V_AV1` → Av1, with `CodecPrivate` as the config.
- **Annex-B helper:** `avcc_to_annexb(sample, nal_len_size)` plus a
  parameter-set prefix from `avcC`/`hvcC`, in safe code. Used by NVDEC only
  (VideoToolbox takes length-prefixed samples; AV1 samples are already
  low-overhead OBUs, which is what `rav1d` takes).
- **Demux caps:** the sample count is capped at the existing input caps, and
  the table sizes are checked against the buffer length before any allocation.

### `shiso-media-sys` (new crate, always compiled)

```rust
#![allow(unsafe_code, reason = "OS video decoder FFI (VideoToolbox through objc2 \
on macOS, the NVDEC driver library through dlopen on Linux) and the C-ABI call \
into the rav1d AV1 decoder")]
```

Every block carries a `// SAFETY:` comment. The public API is safe:

```rust
pub enum HwCodec { H264, Hevc, Vp9, Av1 }
pub struct HwDecoderDesc<'a> { pub codec: HwCodec, pub config: &'a [u8], pub width: u32, pub height: u32 }
pub struct HwTimestamp(i64);            // track timescale ticks; newtype per rule 10
pub struct HwFrame { /* width, height, packed RGB8, HwTimestamp */ }
pub struct HwDecoder { /* platform session */ }
impl HwDecoder {
    pub fn open(desc: &HwDecoderDesc<'_>) -> Result<Self, HwError>;
    pub fn decode(&mut self, sample: &[u8], pts: HwTimestamp, keep: bool) -> Result<Vec<HwFrame>, HwError>;
    pub fn flush(&mut self) -> Result<Vec<HwFrame>, HwError>;
}
pub enum HwError { Unavailable { reason: String }, Unsupported { codec: HwCodec, reason: String }, Failed { detail: String } }
```

`Av1Decoder` (module `av1`, feature `decode-native` on this crate) has the
same `open`/`decode`/`flush` shape and the same `HwFrame`/`HwError`. The
module is a thin wrapper over `rav1d`'s `dav1d_*` C-ABI functions
(`dav1d_open`, `dav1d_send_data`, `dav1d_get_picture`, `dav1d_picture_unref`,
`dav1d_close`). Pictures are unref'd on every path, including errors
(rule 9). Planar YUV is converted to RGB8 with `yuvutils-rs` in safe code,
honouring the sequence's matrix and range.

- **Check at implementation:** that `rav1d`'s `asm` feature (default on)
  builds on the pinned toolchain for both targets. Start with
  `default-features = false, features = ["bitdepth_8", "bitdepth_16"]`,
  measure a clip, and turn asm on only if the speed matters and the build
  stays simple.

Samples go into `HwDecoder` as stored in the container (length-prefixed for
H.264/HEVC). Frames come out in display order. `keep = false` decodes the
sample, because later frames may reference it, but skips RGB conversion.
Dropping a decoder releases every OS or library handle.

**macOS — VideoToolbox**, through `objc2-video-toolbox`, `objc2-core-media`,
`objc2-core-video` and `objc2-core-foundation`, in a
`cfg(target_os = "macos")` target table:

1. Build the format description from the config record:
   `CMVideoFormatDescriptionCreateFromH264ParameterSets` / `...HEVC...`. For
   VP9 and AV1, attach the config record as a sample-description extension.
2. Call `VTRegisterSupplementalVideoDecoderIfAvailable` once for VP9 and AV1.
3. Open a `VTDecompressionSession` asking for `kCVPixelFormatType_32BGRA`
   output. VideoToolbox then handles colour matrix and range.
4. For each sample: build a `CMSampleBuffer` → `VTDecompressionSessionDecodeFrame`
   → `WaitForAsynchronousFrames`. The output callback reorders frames into
   display order, and kept frames are copied row by row (honouring
   `CVPixelBufferGetBytesPerRow`) into RGB8.
5. A session that refuses to open → `Unsupported` (for example AV1 before M3).
   On a Mac where VideoToolbox declines AV1, the Rust step picks it up.

**Linux — NVDEC.**
- **Loading:** `libnvcuvid.so.1` and `libcuda.so.1` are loaded at runtime
  through `libloading`, in a `cfg(target_os = "linux")` target table.
  Hand-written bindings cover the functions used. A missing library →
  `Unavailable`.
- **Context:** `cuInit` and `cuDevicePrimaryCtxRetain(0)`, so the session
  shares `shiso-cuda`'s primary context (`crates/shiso-cuda/src/device.rs`,
  `OwnedContext::retain`) without naming it. Every call pushes and pops that
  context, since decoding runs on request threads.
- **Capability check:** `cuvidGetDecoderCaps` on the codec, bit depth and
  size; a miss → `Unsupported`.
- **Parsing:** `cuvidCreateVideoParser` takes Annex-B input from the demux
  helper. The sequence callback creates the decoder, the decode callback calls
  `cuvidDecodePicture`, and the display callback runs `cuvidMapVideoFrame` for
  kept frames.
- **Colour:** the NV12 surface is copied to host memory and converted with the
  same YUV→RGB code the AV1 path uses, honouring the stream's matrix and range.
- **Concurrency:** at most `MAX_NVDEC_SESSIONS = 4` sessions per process,
  guarded by a counting semaphore; extra decodes wait. GB10 has one NVDEC
  engine, and sessions time-share it.

### The audio chain (`shiso-audio`)

`decode_audio(bytes, limits, &AudioDecodePolicy)`. The policy is no longer
optional.

1. WAV → `parse_wav`, as today.
2. `decode-native`: `symphonia`.
   - Probe → first audio track → decode packets → downmix to mono → the
     existing `resample` to 16 kHz.
   - Decoding stops once `max_seconds + 1` is reached, and the wall-clock
     deadline is checked per packet.
   - A codec symphonia lacks (Opus) → next step.
3. `decode-ffmpeg` **and** `--ffmpeg-path` set: today's `run_decoder` call.
4. Nothing left → `AudioError::UnsupportedContainer { container, codec }`,
   replacing `NeedsDecoder`. The message names `--ffmpeg-path` and
   `decode-native` as the ways to get more.

Audio is therefore still opt-in where it was: a default build with no
`--ffmpeg-path` decodes WAV only; the ffmpeg binary is neither loaded nor
spawned. `AudioDecodePolicy` carries the deadline and, with `decode-ffmpeg`,
an `Option<FfmpegPolicy>`.

### Server (`shiso-server`)

- **`cli.rs`:**
  - remove `video_allow_ffmpeg`, `audio_allow_ffmpeg` and `video_ffmpeg_path`;
  - add `ffmpeg_path: Option<PathBuf>` (`--ffmpeg-path`, env
    `SHISO_FFMPEG_PATH`, **no default**, only under `decode-ffmpeg`);
  - add `video_allow_decode` (`--video-allow-decode`, env
    `SHISO_VIDEO_ALLOW_DECODE`, default false);
  - `ffmpeg_policy()` / `audio_ffmpeg_policy()` become `video_decode_policy()
    -> Option<VideoDecodePolicy>` and `audio_decode_policy() ->
    AudioDecodePolicy`.
- **Callers move to the new types:** `lib.rs` (269, 408–410, 1006–1016),
  `embed/mod.rs`, `embed/prepare.rs`, `tokenizer/render_prompt.rs`,
  `tokenizer/message_preprocess.rs` and `audio/mod.rs`.
- **Boot probe:** when `--ffmpeg-path` is set, call the existing
  `shiso_vision::probe`.
  - `Ready` → an `info` line with the version.
  - `Missing` → a `warn` naming the path and the formats that will fail
    without it. Not fatal, because OS and Rust steps may cover the request.
  - The OS step's availability is logged the same way: VideoToolbox present,
    or NVDEC loaded and which codecs `cuvidGetDecoderCaps` reports.
- **README:** the flags table and the multimodal paragraphs change to match.

### Callers outside the server

Making the `decode_audio` policy non-optional touches more than the server:
`shiso-models` (`tests/nemotron_asr_metal.rs`, `qwen3_asr/offline_tests.rs`,
`embedding_gemma2/tests.rs`), `shiso-server/tests/{transcription_e2e,media_e2e}.rs`,
`shiso-vision/src/video_tests.rs`, and `shiso-audio/src/decode_tests.rs`. The
last builds its fixtures with ffmpeg and calls `probe`, so it is gated on
`decode-ffmpeg` until committed fixtures replace it.

## Errors

| Situation | Error | HTTP |
|---|---|---|
| Video sent, `--video-allow-decode` off | `VisionError::VideoDecodeDisabled { container }` (renamed from `SubprocessDecodeDisabled`; message names the new flag) | 400 |
| Malformed container or bitstream | `VisionError::Demux` / `VisionError::DecoderFailed` | 400 |
| No step in this build handles it | `VisionError::NoVideoDecoder` / `AudioError::UnsupportedContainer` | 400 |
| Cap hit (frames, bytes, deadline) | existing `DecoderOutputCap`, `DecoderTimeout`, `TooLong` | 400 |
| OS or AV1 decoder `Failed` mid-stream | `VisionError::DecoderFailed` carrying its detail | 400 |

Nothing falls back to a default. `Unsupported` is the only variant that moves
the chain forward.

## Gate and repo rules

- **`unsafe` allowlist:** `scripts/api_gate.sh` `UNSAFE_CRATES` gains
  `shiso-media-sys`; the workspace lints comment in `Cargo.toml` (line 181) and
  AGENTS.md's unsafe rule list it.
- **`PROVENANCE.md`:** a row for `shiso-media-sys`, and a row for the NVDEC
  bindings written from NVIDIA's `nvcuvid.h` and `cuviddec.h`. **To verify:**
  the licence text of those headers in the SDK, before the row says MIT.
- **Contracts:** `crates/shiso-bench/tests/contracts.rs` pins `HwDecoder`,
  `HwError`, `VideoDecodePolicy`, `AudioDecodePolicy` and `DemuxedTrack`.
- **Feature-set checks:** `scripts/check.sh` gains a `features` step that, for
  the two non-default sets (`decode-ffmpeg,decode-native` and
  `--no-default-features --features decode-native`, each with the platform
  backend feature):
  - runs `cargo check -p shiso-server --locked`;
  - for the native-only set, asserts with
    `cargo tree -p shiso-server -e features -i shiso-vision` that
    `decode-ffmpeg` is **not** enabled. A compile alone cannot catch
    unification switching ffmpeg back on.
- **libcuda check:** the macOS no-libcuda check still holds. NVDEC only loads
  at runtime and only on Linux.
- **`ffmpeg.rs` module doc** is rewritten. It currently argues against linking
  a decoder and for a subprocess; the new text says what the chain is, why
  H.264/HEVC are never linked, and that ffmpeg is opt-in through
  `--ffmpeg-path`.

## Testing

- **Unit, no device:**
  - Demux of committed MP4 and WebM fixtures: codec, config, sample count,
    timestamps, keyframes.
  - Malformed and truncated boxes → `Malformed`.
  - `avcc_to_annexb`.
  - `SamplePlan`.
  - The chain logic against fake steps: an `Unsupported` step falls through, a
    `Failed` step stops the chain, and every step `Unsupported` gives
    `NoVideoDecoder` listing what was tried.
  - Audio through symphonia on committed mp3, m4a, flac and ogg fixtures
    (`decode-native` only).
- **Fixtures:** small clips (each under 100 KB, a few frames) committed under
  `crates/shiso-vision/tests/fixtures/` and `crates/shiso-audio/tests/fixtures/`.
  Tests stop generating fixtures by running ffmpeg. Per-codec reference
  frames, decoded once by ffmpeg as an independent implementation, go under
  `test-vectors/references/`.
- **Device tests** (`#[ignore]`, gated by platform):
  - VideoToolbox on macOS for H.264, HEVC, VP9 and AV1 (AV1 skips, saying why,
    before M3).
  - NVDEC on Linux.
  - Each compares against the reference frames with a PSNR floor of 40 dB,
    which allows for colour-conversion rounding while still catching wrong
    planes or strides.
  - A device test that skips is reported as skipped, never as passing.
- **Rust path:** AV1 8-bit and 10-bit fixtures against the same references; a
  4:4:4 fixture returns `Unsupported`. H.264 given to the Rust step is
  `Unsupported`.
- **Server:**
  - `tests/cli.rs` covers the new and removed flags; passing a removed flag is
    a clap error; `--ffmpeg-path` is absent from a native-only build.
  - `media_audio.rs`: the mp3 refusal test is rewritten for the new
    semantics.
  - `media_render.rs`: the video refusal names `--video-allow-decode`.
  - `long_meeting.rs` and `keyword_bias.rs` drop `--audio-allow-ffmpeg`
    (and pass `--ffmpeg-path` where they need ffmpeg).

## Phases

The work is too large for one plan, so each phase gets its own plan and can be
merged on its own:

1. **Plumbing.**
   - Features, policy types and the chain seam with only the ffmpeg step.
   - The flag changes (`--ffmpeg-path`, `--video-allow-decode`), error
     renames, boot probe, the caller updates above, and the `ffmpeg.rs` doc.
   - The `features` gate step.
2. **Audio:** `symphonia` under `decode-native`, plus fixtures.
3. **Demux:** MP4 and MKV/WebM, `SamplePlan`, the Annex-B helper.
4. **`shiso-media-sys` on macOS:** VideoToolbox, the OS step, device tests on
   this machine.
5. **Rust AV1:** `rav1d` in `shiso-media-sys`, `yuvutils-rs` conversion.
6. **`shiso-media-sys` on Linux:** NVDEC, device tests on the GB10.

## Later (not in this design)

- **AudioToolbox as the OS audio step on macOS:** HE-AAC, and probably Opus,
  without ffmpeg.
- **An Opus decoder in Rust,** once `opus-rs` (or another crate) has been
  vetted.
- **A CUDA kernel for NV12→RGB on GB10,** if profiling shows the host
  conversion matters.
- **VA-API,** if the server is deployed on Intel or AMD GPUs.
