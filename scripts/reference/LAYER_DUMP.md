# Per-layer dumps: what shiso writes today, and what the reference should write

Prep for C3 (bisecting a diverging position layer by layer). Read against
master at `01e6038`.

## Summary

- `SHISO_LAYER_DUMP` **does not fire for qwen4exp.** It instruments only the
  dense layer loop (`crates/shiso-engine/src/layers.rs`, used by the qwen35 and
  gemma4 families). Qwen4exp runs on the sparse stack (`shiso_engine::sparse`),
  whose equivalent switch is `SHISO_STEP_DUMP`.
- **Neither switch writes hidden-state values.** Both print summary statistics
  per tensor (norms, extremes, the first four floats), over every row of the
  chunk. Neither writes a per-position vector that could be diffed element by
  element against a reference.
- So there is no existing layout for the reference to copy. C3 needs a real
  per-position residual dump on the shiso side, which does not exist yet. The
  format proposed at the end is what that dump should write. The reference
  driver can write it now.

## `SHISO_LAYER_DUMP` (dense families only)

`crates/shiso-engine/src/trace.rs`, called from `layers.rs`.

- `=1` or `=2` turns it on. Each call syncs the queue and copies the block back
  to the host. Output goes to **stdout** through `println!`, not through
  `tracing`.
- Lines: `embed` (the stream after the embedding), then one per layer slot,
  `slot {i} gdn` or `slot {i} attn` (the residual after that slot). With `=2`,
  three more per slot: `  slot {i} block`, `  slot {i} ffn_in`,
  `  slot {i} ffn_out`.
- Covers: the first `hidden * n_tok` f32 of the buffer, meaning **every token
  of the current chunk** (a prefill chunk or a single decode row), pooled
  together. It does not pick out a position.
- Each line: `n= rms= absmax= zeros= non_finite= first=[4 floats]`, with rms
  accumulated in f64.

## `SHISO_STEP_DUMP` (sparse stack, qwen4exp)

`crates/shiso-engine/src/sparse/exec.rs` (`dump_destination`, `summarize`).

- `=1`: after every step (every kernel, trellis matmul, BLAS call, clear), a
  device barrier, then a readback of the step's **destination** operand.
  `=2` also reads every other operand. The routed-expert step always reports
  its inputs. `SHISO_STEP_SYNC=1` gives the barrier and timings without the
  readback.
- Output is `tracing::info!` events (event message `wrote` or `read`). The
  parity binary needs `SHISO_LOG=info` to show them.
- Fields: `at` (the step index **within the current `run_steps` call**, which
  restarts at every island and prologue, so it is not a global step number),
  `step` (the kernel entry name), `which` (the operand index), `buffer`, `count`,
  `nans`, `zeros`, `where_nan`, `last_nan`, `first_live`, `last_live`, `low`,
  `high`, `sum`, `l1`, `l2` (the square root of the sum of squares, in f64), and
  `head` (the first four floats).
- Covers: the operand's **whole allocation**, up to 64 MiB, read as f32. Scratch
  is sized for the widest chunk, so the statistics include rows past `n_tok`
  that the current pass did not write (left over from an earlier chunk). They
  do not select a position, and past 64 MiB they are truncated.
- The residual is `Buf::Residual`, f32, laid out `[token][stream][hidden]`.
  For this checkpoint that is 4 hyper-connection streams × 2560 = 10240 floats
  per token. The steps that write it:
  - `q4e_hc_init_kernel`: the embedding tiled across the 4 streams (the pass
    prologue);
  - `q4e_hc_combine_norm_kernel`: closes each island and adds the block's
    contribution into the streams. A layer has two islands, Mixer (GDN or
    attention) then Experts (MoE), so **the Experts combine is the "after layer
    L" residual** and the Mixer combine is the one in the middle of the layer.
    Its operand 1, `Buf::Normed`, holds the residual already normalized for the
    next block;
  - `q4e_add2_kernel` in the n-gram/PLE block (`sparse/ple.rs`): writes into the
    residual between a combine and the next mix, on the layers that carry it.
    Before relying on which layer's file it lands in, check the step order in
    `sparse/passes.rs`.
- A one-row pass (decode, or the single token `prefill_last` feeds) can be
  replayed from a captured device graph. A replayed island issues no steps
  through `run_steps`, so **it dumps nothing**. Prefill chunks of more than one
  row are never captured, so they always dump.

## Proposed layout for C3 (the reference can write this now)

Per case, at the case's frontier position (the same 0-based position as
`index.json` in the row dumps):

```
<dir>/index.json                        same schema as the row dumps
<dir>/<case-id>/embed.f32.npy           [4, 2560]  residual after the stream init
<dir>/<case-id>/L{00..47}.mixer.f32.npy [4, 2560]  residual after layer L's mixer combine
<dir>/<case-id>/L{00..47}.f32.npy       [4, 2560]  residual after layer L's Experts combine
```

- Use numpy `.npy` v1, dtype `<f4`, **raw (un-normalized) residual streams**,
  stream-major, matching `Buf::Residual` for one token. Write f32, not f16:
  C3 grades relative L2 error per layer, and f16 rounding (about 5e-4 relative)
  would hide the small early-layer errors the bisection is looking for.
- Each file is 40 KiB, so a case costs 97 × 40 KiB ≈ 3.8 MiB. Budget it before
  writing, as with the rows.
- If the reference keeps the streams in a different order (for example
  `[stream][token][hidden]` for the whole sequence), slice out the one token
  and write `[stream, hidden]`. If the order differs, record it under a `layout` key in
  `index.json`.
- The grading metric is `||shiso - ref||_2 / ||ref||_2` per file. The first
  layer where it jumps names the component: GDN or attention/indexer (mixer
  file), MoE (layer file), the hyper-connection mix and combine (both), or
  n-gram/PLE (on the layers that carry it; write a separate `L{nn}.ngram.f32.npy`
  after its add if the reference can isolate it).

On the shiso side, this needs a new diagnostic. A knob naming a position and a
directory would make the executor copy one token's `Buf::Residual` row
(`10240 * 4` bytes at offset `row * 40960`) after each `q4e_hc_init_kernel`,
Mixer combine and Experts combine, with graphs off. That is not built yet.
