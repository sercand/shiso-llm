# sample entry-point names

The lookup table a loader resolves a kernel through. A module is loaded by its
registry key; a kernel inside it is fetched by the exact name recorded here.

Every entry carries C linkage, so each name is the name the driver answers to,
verbatim.

| Name | Grid | Block | Shared |
| --- | --- | --- | --- |
| `sample_suppress_rows_kernel` | one block per logits row | `256` | `0` |

Arguments, in order: `logits` (f32, rows x `n_vocab`, written in place),
`list` (u32: the count, then that many ids), `n_vocab` (u32).
