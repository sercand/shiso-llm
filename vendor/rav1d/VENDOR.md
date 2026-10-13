# Vendored rav1d

- Upstream: the `rav1d` crate, version 1.1.0 on crates.io (BSD-2-Clause, see
  `COPYING`), a Rust port of the dav1d AV1 decoder.
- Why vendored: 1.1.0 aborts the process when a frame's tile data is damaged.
  The error path calls `on_error` after the frame header has been cleared, and
  `on_error` unwraps it. The panic happens inside an `extern "C"` function, so it
  cannot unwind and the process aborts (release builds included). One flipped bit
  in a sample reaches it, so an uploaded clip could take the server down.
- The change (`src/decode.rs`, `on_error`): `f.frame_hdr.as_ref().unwrap()`
  became `f.frame_hdr.as_ref().is_some_and(..)`. Nothing else in the sources is
  edited. A mutation loop over damaged samples
  (`crates/shiso-media-sys/tests/av1.rs`, `damaged_samples_never_crash`) failed
  before the change and passes after it.
- Also removed: `src/arm` and `src/x86` (assembly, 8 MB) because the `asm`
  feature is off for this build (the published 1.1.0 package does not build with
  it on aarch64 macOS), and the library's `staticlib`/`cdylib` crate types, which
  nothing here links. Do not turn `asm` on without restoring them from the
  published crate.
- To drop this copy: when a rav1d release contains the fix, point
  `[workspace.dependencies]` back at crates.io and delete `vendor/rav1d`.
