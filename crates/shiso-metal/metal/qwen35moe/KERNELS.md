# qwen35moe Metal kernel contract

The dispatch contract for the routed feed-forward: entry-point names, argument
blocks, binding order, launch geometry and threadgroup memory. **Where this file
and a kernel body disagree, the body is right and this file is the bug.**

## Conventions

- **Two binding shapes live in this set, and which one an entry takes is decided
  by who wrote it.**
  - The **expert entries** -- the ones that read a weight slab -- take the
    argument block `constant q35moe_expert_args &` at `[[buffer(0)]]`, with the
    buffers following at `[[buffer(1)]]`, `[[buffer(2)]]`... in declaration
    order. That is the qwen35, vision and dense convention, and
    `expert_stage` in `crates/shiso-engine/src/moe.rs` pushes
    `KernelArg::Bytes(bytes_of(&stage.args))` first for exactly that reason.
  - The **seven general entries** -- the router, the three sort passes, the
    activation, the combine and the shared add -- have **no argument block**.
    They keep the binding order `crates/shiso-engine/src/moe.rs` already uses,
    buffer by buffer and scalar by scalar, because the *same host code* launches
    them and the CUDA bodies in `kernels/cuda/q4e/moe.cu` by the same entry
    names. Those bodies are fidelity-gated (`scripts/q4e_ptx_oracle.sh
    --mode=fidelity`) and the MSL here is a port of them, not a design: the
    arithmetic, the tie-break and the order of operations are theirs.
- Metal resolves an entry by name out of one flat library and ignores the module
  string a `KernelRef` carries, so `q4e_moe` and `qwen35moe` name the same
  library here. Every entry name below is therefore unique library-wide.
- `grid` in a launch is the threadgroup count, dispatched as given: no division,
  no rounding. `block` is the threads-per-threadgroup the kernel reads as
  `threads_per_threadgroup`.
- Every attribute-qualified parameter of one kernel must have the same rank:
  a `uint3 [[threadgroup_position_in_grid]]` beside a
  `uint [[thread_position_in_threadgroup]]` is a compile error, not a widening.
  The two-dimensional entries below therefore take `uint3` throughout and read
  `.x`, which is also the qwen35 house spelling.
- Reductions go through `q35moe_block_sum`, `q35moe_block_max` and
  `q35moe_block_argmax` in `q35moe_common.h`. Those are the CUDA reference's
  reduction *shape* -- simdgroup fold, publish to `scratch[simdgroup]`, barrier,
  simdgroup 0 re-folds, thread 0 broadcasts -- not a mathematically equal
  rewrite of it, for the reason `metal/qwen35/KERNELS.md` gives: a port is
  graded against the reference's summation order. The sum's simdgroup stage is
  the explicit `simd_shuffle_down` fold; the max's is `simd_max`, which is exact
  under any order; the argmax's is the explicit fold, because there is no
  builtin and because its tie rule is the point of it.
- No `#pragma METAL fp math_mode` in this set. Fast math is the library's build
  default and every file here is written against it. `compile_library`
  concatenates every source into ONE translation unit and MSL has no push/pop
  for that pragma, so an unclosed `safe` region would change the math mode of
  every file after it; `tests/metal_source_pragmas.rs` scans this directory for
  exactly that.
- Measured device limits: 32,768 B threadgroup memory, 1,024 threads per
  threadgroup. Every row below is inside both, and `run.rs` refuses (rather than
  silently runs) a launch that is not.

## Geometry

`grid` is the threadgroup count as `crates/shiso-engine/src/moe.rs` computes it,
dispatched directly. `columns` is `n_tok * n_used`; `BLOCK` is 256 there.

| Entry point | Buffers, in bind order | grid | block | threadgroup |
|---|---|---|---|---|
| `q4e_moe_route_kernel` | `ids, weights, logits, n_expert:u32, n_used:u32` | `n_tok` | 256 | `n_expert * 4` |
| `q4e_moe_sort_count_kernel` | `counts, ids, n_rows:u32, n_expert:u32` | `max(ceil(columns/256), 1)` | 256 | 0 |
| `q4e_moe_sort_scan_kernel` | `bounds, active, counts, n_expert:u32` | 1 | 32 | 0 |
| `q4e_moe_sort_scatter_kernel` | `sorted, cursor, bounds, ids, n_rows:u32, n_expert:u32` | `max(ceil(columns/256), 1)` | 256 | 0 |
| `q4e_swiglu_kernel` | `out, gate, up, n:u64` | `max(ceil(n/256), 1)` | 256 | 0 |
| `q4e_moe_combine_kernel` | `out, down, weights, n_embd:u32, n_used:u32` | `(max(ceil(n_embd/256), 1), n_tok)` | 256 | 0 |
| `q4e_shared_add_kernel` | `out, shared, logit, n_embd:u32` | `(max(ceil(n_embd/256), 1), n_tok)` | 256 | 0 |
| `kernel_q35moe_tile_probe` (grading probe, not dispatched by `moe.rs`) | `args, input, weights, scales, biases, out` | 1 | 256 | 0 |
| `kernel_q35moe_experts_gate_up_q4affine_f32` | `args, gate_w, gate_s, gate_b, up_w, up_s, up_b, input, sorted, bounds, active, out_gate, out_up` | `(ceil(out_dim/4), min(columns, n_expert))` | 128 | 0 |
| `kernel_q35moe_experts_down_q4affine_f32` | `args, down_w, down_s, down_b, mid, sorted, bounds, active, out` | `(ceil(out_dim/4), min(columns, n_expert))` | 128 | 0 |
| `kernel_q35moe_experts_gate_up_q4affine_bf16` | as the `f32` gate-and-up entry | `(ceil(out_dim/4), min(columns, n_expert))` | 128 | 0 |
| `kernel_q35moe_experts_down_q4affine_bf16` | as the `f32` down entry | `(ceil(out_dim/4), min(columns, n_expert))` | 128 | 0 |

The router's threadgroup request is dynamic: `run.rs` calls
`setThreadgroupMemoryLength_atIndex(n_expert * 4, 0)`, which is what the
`threadgroup float *smem [[threadgroup(0)]]` parameter binds to. The reduction
scratch, the chosen-weight array and the chosen-id array are static threadgroup
arrays inside the kernel and are *not* part of that request.

## Argument blocks

- `q35moe_expert_args {in_dim, out_dim, n_expert, n_used}`, four `int`s, sixteen
  bytes. Field order is the struct order in `q35moe_common.h` and the engine
  mirrors it field for field.
- The seven general entries have none. Their scalars are separate bindings, in
  the positions the table above gives.

## What the four expert entries pin

The four names are fixed by `expert_entry` in `crates/shiso-engine/src/moe.rs`
and Metal resolves out of one flat library, so each is unique library-wide. The
dtype suffix is the float the **activations** cross the launch boundary in --
the input rows and the outputs together. The weights stay `uchar` codes and the
scale and bias stay `bfloat` in both, because those are container widths and not
an activation choice.

- **The block shape.** `grid.y` indexes the **active** experts through the list
  the scan pass leaves in `active`, not all of them, and returns when it is past
  `active[n_expert]`: at one token eight experts of 256 receive a column, and the
  host bounds the extent by the column count because a chunk cannot activate more
  experts than it has columns. `grid.x` picks four output rows and each of the
  block's four simdgroups owns one, which is `SHISO_Q35MOE_EXPERT_WARPS` in
  `kernels/cuda/qwen35moe/experts.cuh`. A block then sweeps that expert's whole
  span of the sorted column map, `sorted[bounds[e] .. bounds[e+1]]`, so the
  weight row stays in cache across the columns routed to it. One path, no
  per-column variant: a threshold at which a model starts answering differently
  is the defect the grouping avoids.
- **The four seam conventions**, none of which is checkable by reading one
  kernel and all of which fail plausibly:
  1. a routed column is `token * n_used + slot`, so `col / n_used` is the token
     whose activation row gate and up read;
  2. the down projection reads its **own** column's row of the activated
     intermediate, not its token's;
  3. down writes `[column, out_dim]` row-major, which is column-major over
     (token, slot), because that is what `q4e_moe_combine_kernel` reads at
     `down[col * n_embd + e]`. The other way round is a correctly sized
     transpose;
  4. the per-expert parameters are indexed by expert id. The affine codec has no
     per-tensor scalar to get wrong the way NVFP4's `globals[e]` is, but the slab
     strides -- `expert * out_dim * in_dim / 2` bytes of codes and
     `expert * out_dim * in_dim / 64` parameters -- are the same mistake in a
     different place.
- **The slab reader is not `q35moe_affine_tile`.** The tile multiplies `Rows`
  *contiguous* activation rows by `TileN` outputs through an MPP `matmul2d`,
  which needs both operands addressable as tensors with a constant stride. A
  routed stage has neither property: its activation rows are the arbitrary,
  data-dependent gather `sorted` names, and the grid the host fixes spends one
  simdgroup on one output row rather than a threadgroup on a `Rows x TileN` tile.
  So the expert entries get their own reader, `q35moe_affine_row_dot`, over the
  same bytes and the same placement arithmetic. The two agree on the layout and
  differ only in the order they fold it: the tile factors the epilogue per group
  (`scale * sum a*code + bias * sum a`) because the matmul can only give it the
  code product, while the row dot dequantizes each code where it reads it, which
  is the shape of `shiso_q35moe_expert_dot` and of the host reference both are
  graded against.
- **`Q35MOE_STORAGE_N` is 256 and `out_dim % 256 == 0` is assumed.** The resident
  layout is one decision for the whole set -- the loader packs a projection once
  and every reader of that slab agrees with it -- so it is a constant rather than
  a field of the argument block. Hidden is 2,048 and the expert intermediate 512;
  both are whole multiples. A projection that would need a short last tile is a
  padding decision for the loader, not for the codec.
- **A group's 32 code bytes are contiguous**, which is what group-major buys, so
  lane `l` of a simdgroup takes one byte of every quantization group -- codes
  `2l` and `2l+1` -- and a simdgroup consumes a whole group in one coalesced
  load. That is a different partition of the reduction from the CUDA reference's
  `k = lane, lane + 32, ...`; the layout forces it, and nothing grades against
  it, because no CUDA entry reads this codec.

### What the dtype costs, measured

`crates/shiso-kerneltest/src/qwen35moe/q4affine.rs` grades each of the four
against one host reference, over a fixture whose activations are deliberately
**not** exactly representable -- so the number below is a measurement of two
summation orders and not a property of the fixture, which is the thing
`the_affine_tile_matches_the_host_product`'s zero cannot be.

| Entry | Depth | Worst divergence | Largest reference value | Budget |
|---|---|---|---|---|
| `..._gate_up_q4affine_f32` | 256 | 4.196e-5 | 60.51 | 2e-4 |
| `..._down_q4affine_f32` | 512 | 9.155e-5 | 122.46 | 4e-4 |
| `..._gate_up_q4affine_bf16` | 256 | 0.1233 | 60.34 | 0.5 |
| `..._down_q4affine_bf16` | 512 | 0.2461 | 122.25 | 1.0 |

The `f32` rows are about 7e-7 relative; the `bf16` rows are 2.0e-3, which is
2^-9 and not a coincidence -- bf16 carries eight mantissa bits, the activation
rows arrive already rounded to them and the outputs are rounded to them again on
the way out. Three thousand times the `f32` number for the same arithmetic. The
two `bf16` cases assert that ordering as well as their budget: **if `bf16` ever
measures tighter than `f32`, the `f32` path is the one to look at.**

## What the seven general entries pin

- **The routed column.** A routed column is `token * n_used + slot`, so
  `column / n_used` is the token. `q4e_moe_route_kernel` writes `ids` and
  `weights` in that order, `q4e_moe_combine_kernel` reads `down` and `weights`
  in it, and the expert matmul writes column-major over (token, slot) to match.
- **The routing law.** Softmax over *every* expert, then the best `n_used` by
  probability, then a renormalization over the chosen ones alone. Three details
  are load bearing:
  - the logit row is staged in `smem` **before** the selection rounds, because
    each round masks its winner off by writing `-INFINITY` into that copy;
  - the argmax keeps the **lower** expert id among equal values. `e` only grows
    across a thread's strided sweep, so keeping the first of equal values keeps
    the lower id, and `q35moe_block_argmax` carries the same rule across
    threads. A port that keeps the higher one differs only on a tie -- and then
    answers out of different experts, fluently, with every count still right;
  - the divisor is clamped **before** the division (`fmax(wsum, 1e-20f)`, then
    `w / wsum`), not after.
- **The activation.** `g / (1 + exp(-g)) * u`, the reference's division, not
  `g * sigmoid(g)`: the same function, and they do not round alike.
- **The counting sort.** `counts` and `cursor` are cleared by the host before
  each layer; both are accumulated atomically and outlive the layer. The order
  *inside* one expert's span is unspecified -- the atomic hands out slots in
  whatever order the hardware runs -- and nothing may depend on it, because
  every column writes its own output row. `active[n_expert]` holds the number of
  active experts and `bounds[n_expert]` the column total, which is why both
  buffers are `n_expert + 1` long.
- **The shared expert.** One scalar gate per token, not a per-channel one, and
  an add rather than a store, because the routed experts have already written
  that row.

## Grading

Every entry here has a device case in
`crates/shiso-kerneltest/src/qwen35moe/q4affine.rs`, wrapped in
`crates/shiso-metal/tests/q35moe_moe.rs`, graded against a plain-Rust reference
and printing the divergence it measured beside its budget. They need a device:

```
cargo test -p shiso-metal --features metal --test q35moe_moe -- --ignored --test-threads=1
```

`--test-threads=1` is not optional on these machines: Metal runs one workload at
a time.
