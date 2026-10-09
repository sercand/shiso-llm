/* The `generic` module: the q4_0 extended matvec at every lane split and row
 * blocking the engine can select, the plain f16 matvec a decode falls back
 * to, the plain q8_0 matvec that is the only decode path that codec has, the
 * gated SiLU a qwen35 feed-forward closes on, the two fused RMS norms a dense
 * slot passes through, the residual add, the q4_K extended matvec every
 * qwen35 linear layer is read through at one token, the q2_K and i-quant
 * extended matvecs the Saluki export's mixed sub-4-bit container is read
 * through, the iq1_m gather its vocabulary table needs, and the table copy
 * that grades this module's codebook transcriptions on the device.
 *
 * One translation unit per module, because a module name is the registry key
 * for a separately compiled image and the driver resolves entry names inside
 * one image only. The entries below and in the headers this unit includes -
 * thirty ext matvecs for the affine formats, one hundred thirty-five for the
 * mixed set, and the twelve the headers define - are the whole of this
 * module.
 *
 * Nothing in this directory is a byte copy of anything, so no file here carries
 * a provenance header and `scripts/q4e_ptx_oracle.sh --mode=fidelity` reports
 * them as generated. The arithmetic is derived from the Metal twins in
 * `crates/shiso-metal/metal/generic` and from the argument blocks in
 * `crates/shiso-engine/src/launch.rs`, both of which are in this tree.
 */

#include "mul_mv_ext.cuh"
#include "mul_mv_dense.cuh"
#include "mul_mv_q8_0.cuh"
#include "glu.cuh"
#include "norm.cuh"
#include "bin.cuh"
#include "get_rows_f16.cuh"
#include "get_rows_block.cuh"

/* Fifteen pairs: five row blockings by three lane splits. `ext_r1ptg` and
 * `ext_nxpsg` choose the two independently, so every pair is reachable and
 * every pair is here. */
SHISO_MV_EXT_Q4_0(1, 4)
SHISO_MV_EXT_Q4_0(1, 8)
SHISO_MV_EXT_Q4_0(1, 16)
SHISO_MV_EXT_Q4_0(2, 4)
SHISO_MV_EXT_Q4_0(2, 8)
SHISO_MV_EXT_Q4_0(2, 16)
SHISO_MV_EXT_Q4_0(3, 4)
SHISO_MV_EXT_Q4_0(3, 8)
SHISO_MV_EXT_Q4_0(3, 16)
SHISO_MV_EXT_Q4_0(4, 4)
SHISO_MV_EXT_Q4_0(4, 8)
SHISO_MV_EXT_Q4_0(4, 16)
SHISO_MV_EXT_Q4_0(5, 4)
SHISO_MV_EXT_Q4_0(5, 8)
SHISO_MV_EXT_Q4_0(5, 16)

/* Fifteen more pairs of the same template at the codec the bulk of a qwen35
 * `Q4_K_M` container is packed at. They are HERE rather than in `cuda/qwen35`
 * because `launch::ext_module` maps `Q4_K` to `launch::GENERIC` - the host
 * names the module and this backend answers where it is asked, whatever family
 * the weight belongs to. Without them `ext_name_for` returned `None` on a
 * backend that specializes by name, the projection fell through to
 * `plain_projection`, and a decode refused with "`Q4_K` has no plain matvec at
 * this width" - which named the family that could not serve it rather than the
 * one that should have. */
SHISO_MV_EXT_Q4_K(1, 4)
SHISO_MV_EXT_Q4_K(1, 8)
SHISO_MV_EXT_Q4_K(1, 16)
SHISO_MV_EXT_Q4_K(2, 4)
SHISO_MV_EXT_Q4_K(2, 8)
SHISO_MV_EXT_Q4_K(2, 16)
SHISO_MV_EXT_Q4_K(3, 4)
SHISO_MV_EXT_Q4_K(3, 8)
SHISO_MV_EXT_Q4_K(3, 16)
SHISO_MV_EXT_Q4_K(4, 4)
SHISO_MV_EXT_Q4_K(4, 8)
SHISO_MV_EXT_Q4_K(4, 16)
SHISO_MV_EXT_Q4_K(5, 4)
SHISO_MV_EXT_Q4_K(5, 8)
SHISO_MV_EXT_Q4_K(5, 16)

/* Fifteen pairs of the `kq` walk at each of the nine mixed formats the Saluki
 * export packs its linear layers in - `q2_K` and the eight i-quants, one
 * hundred thirty-five entries. They are HERE because `launch::ext_module`
 * sends every one of them to `launch::GENERIC` and `launch::ext_name_for`
 * names them with the lane split folded in, exactly as it does the two
 * formats above; a missing pair is a decode that resolves at one prompt
 * width and refuses at another. `crates/shiso-cuda/tests/dense_iq.rs` grades
 * every pair against the host reference decoder, and
 * `crates/shiso-engine/tests/ext_matvec_entries.rs` fails the build if a name
 * in the host's tables stops having an instantiation here. */
SHISO_MV_EXT_KQ(q2_K, 1, 4)
SHISO_MV_EXT_KQ(q2_K, 1, 8)
SHISO_MV_EXT_KQ(q2_K, 1, 16)
SHISO_MV_EXT_KQ(q2_K, 2, 4)
SHISO_MV_EXT_KQ(q2_K, 2, 8)
SHISO_MV_EXT_KQ(q2_K, 2, 16)
SHISO_MV_EXT_KQ(q2_K, 3, 4)
SHISO_MV_EXT_KQ(q2_K, 3, 8)
SHISO_MV_EXT_KQ(q2_K, 3, 16)
SHISO_MV_EXT_KQ(q2_K, 4, 4)
SHISO_MV_EXT_KQ(q2_K, 4, 8)
SHISO_MV_EXT_KQ(q2_K, 4, 16)
SHISO_MV_EXT_KQ(q2_K, 5, 4)
SHISO_MV_EXT_KQ(q2_K, 5, 8)
SHISO_MV_EXT_KQ(q2_K, 5, 16)
SHISO_MV_EXT_KQ(iq2_xxs, 1, 4)
SHISO_MV_EXT_KQ(iq2_xxs, 1, 8)
SHISO_MV_EXT_KQ(iq2_xxs, 1, 16)
SHISO_MV_EXT_KQ(iq2_xxs, 2, 4)
SHISO_MV_EXT_KQ(iq2_xxs, 2, 8)
SHISO_MV_EXT_KQ(iq2_xxs, 2, 16)
SHISO_MV_EXT_KQ(iq2_xxs, 3, 4)
SHISO_MV_EXT_KQ(iq2_xxs, 3, 8)
SHISO_MV_EXT_KQ(iq2_xxs, 3, 16)
SHISO_MV_EXT_KQ(iq2_xxs, 4, 4)
SHISO_MV_EXT_KQ(iq2_xxs, 4, 8)
SHISO_MV_EXT_KQ(iq2_xxs, 4, 16)
SHISO_MV_EXT_KQ(iq2_xxs, 5, 4)
SHISO_MV_EXT_KQ(iq2_xxs, 5, 8)
SHISO_MV_EXT_KQ(iq2_xxs, 5, 16)
SHISO_MV_EXT_KQ(iq2_xs, 1, 4)
SHISO_MV_EXT_KQ(iq2_xs, 1, 8)
SHISO_MV_EXT_KQ(iq2_xs, 1, 16)
SHISO_MV_EXT_KQ(iq2_xs, 2, 4)
SHISO_MV_EXT_KQ(iq2_xs, 2, 8)
SHISO_MV_EXT_KQ(iq2_xs, 2, 16)
SHISO_MV_EXT_KQ(iq2_xs, 3, 4)
SHISO_MV_EXT_KQ(iq2_xs, 3, 8)
SHISO_MV_EXT_KQ(iq2_xs, 3, 16)
SHISO_MV_EXT_KQ(iq2_xs, 4, 4)
SHISO_MV_EXT_KQ(iq2_xs, 4, 8)
SHISO_MV_EXT_KQ(iq2_xs, 4, 16)
SHISO_MV_EXT_KQ(iq2_xs, 5, 4)
SHISO_MV_EXT_KQ(iq2_xs, 5, 8)
SHISO_MV_EXT_KQ(iq2_xs, 5, 16)
SHISO_MV_EXT_KQ(iq2_s, 1, 4)
SHISO_MV_EXT_KQ(iq2_s, 1, 8)
SHISO_MV_EXT_KQ(iq2_s, 1, 16)
SHISO_MV_EXT_KQ(iq2_s, 2, 4)
SHISO_MV_EXT_KQ(iq2_s, 2, 8)
SHISO_MV_EXT_KQ(iq2_s, 2, 16)
SHISO_MV_EXT_KQ(iq2_s, 3, 4)
SHISO_MV_EXT_KQ(iq2_s, 3, 8)
SHISO_MV_EXT_KQ(iq2_s, 3, 16)
SHISO_MV_EXT_KQ(iq2_s, 4, 4)
SHISO_MV_EXT_KQ(iq2_s, 4, 8)
SHISO_MV_EXT_KQ(iq2_s, 4, 16)
SHISO_MV_EXT_KQ(iq2_s, 5, 4)
SHISO_MV_EXT_KQ(iq2_s, 5, 8)
SHISO_MV_EXT_KQ(iq2_s, 5, 16)
SHISO_MV_EXT_KQ(iq3_xxs, 1, 4)
SHISO_MV_EXT_KQ(iq3_xxs, 1, 8)
SHISO_MV_EXT_KQ(iq3_xxs, 1, 16)
SHISO_MV_EXT_KQ(iq3_xxs, 2, 4)
SHISO_MV_EXT_KQ(iq3_xxs, 2, 8)
SHISO_MV_EXT_KQ(iq3_xxs, 2, 16)
SHISO_MV_EXT_KQ(iq3_xxs, 3, 4)
SHISO_MV_EXT_KQ(iq3_xxs, 3, 8)
SHISO_MV_EXT_KQ(iq3_xxs, 3, 16)
SHISO_MV_EXT_KQ(iq3_xxs, 4, 4)
SHISO_MV_EXT_KQ(iq3_xxs, 4, 8)
SHISO_MV_EXT_KQ(iq3_xxs, 4, 16)
SHISO_MV_EXT_KQ(iq3_xxs, 5, 4)
SHISO_MV_EXT_KQ(iq3_xxs, 5, 8)
SHISO_MV_EXT_KQ(iq3_xxs, 5, 16)
SHISO_MV_EXT_KQ(iq3_s, 1, 4)
SHISO_MV_EXT_KQ(iq3_s, 1, 8)
SHISO_MV_EXT_KQ(iq3_s, 1, 16)
SHISO_MV_EXT_KQ(iq3_s, 2, 4)
SHISO_MV_EXT_KQ(iq3_s, 2, 8)
SHISO_MV_EXT_KQ(iq3_s, 2, 16)
SHISO_MV_EXT_KQ(iq3_s, 3, 4)
SHISO_MV_EXT_KQ(iq3_s, 3, 8)
SHISO_MV_EXT_KQ(iq3_s, 3, 16)
SHISO_MV_EXT_KQ(iq3_s, 4, 4)
SHISO_MV_EXT_KQ(iq3_s, 4, 8)
SHISO_MV_EXT_KQ(iq3_s, 4, 16)
SHISO_MV_EXT_KQ(iq3_s, 5, 4)
SHISO_MV_EXT_KQ(iq3_s, 5, 8)
SHISO_MV_EXT_KQ(iq3_s, 5, 16)
SHISO_MV_EXT_KQ(iq1_s, 1, 4)
SHISO_MV_EXT_KQ(iq1_s, 1, 8)
SHISO_MV_EXT_KQ(iq1_s, 1, 16)
SHISO_MV_EXT_KQ(iq1_s, 2, 4)
SHISO_MV_EXT_KQ(iq1_s, 2, 8)
SHISO_MV_EXT_KQ(iq1_s, 2, 16)
SHISO_MV_EXT_KQ(iq1_s, 3, 4)
SHISO_MV_EXT_KQ(iq1_s, 3, 8)
SHISO_MV_EXT_KQ(iq1_s, 3, 16)
SHISO_MV_EXT_KQ(iq1_s, 4, 4)
SHISO_MV_EXT_KQ(iq1_s, 4, 8)
SHISO_MV_EXT_KQ(iq1_s, 4, 16)
SHISO_MV_EXT_KQ(iq1_s, 5, 4)
SHISO_MV_EXT_KQ(iq1_s, 5, 8)
SHISO_MV_EXT_KQ(iq1_s, 5, 16)
SHISO_MV_EXT_KQ(iq1_m, 1, 4)
SHISO_MV_EXT_KQ(iq1_m, 1, 8)
SHISO_MV_EXT_KQ(iq1_m, 1, 16)
SHISO_MV_EXT_KQ(iq1_m, 2, 4)
SHISO_MV_EXT_KQ(iq1_m, 2, 8)
SHISO_MV_EXT_KQ(iq1_m, 2, 16)
SHISO_MV_EXT_KQ(iq1_m, 3, 4)
SHISO_MV_EXT_KQ(iq1_m, 3, 8)
SHISO_MV_EXT_KQ(iq1_m, 3, 16)
SHISO_MV_EXT_KQ(iq1_m, 4, 4)
SHISO_MV_EXT_KQ(iq1_m, 4, 8)
SHISO_MV_EXT_KQ(iq1_m, 4, 16)
SHISO_MV_EXT_KQ(iq1_m, 5, 4)
SHISO_MV_EXT_KQ(iq1_m, 5, 8)
SHISO_MV_EXT_KQ(iq1_m, 5, 16)
SHISO_MV_EXT_KQ(iq4_xs, 1, 4)
SHISO_MV_EXT_KQ(iq4_xs, 1, 8)
SHISO_MV_EXT_KQ(iq4_xs, 1, 16)
SHISO_MV_EXT_KQ(iq4_xs, 2, 4)
SHISO_MV_EXT_KQ(iq4_xs, 2, 8)
SHISO_MV_EXT_KQ(iq4_xs, 2, 16)
SHISO_MV_EXT_KQ(iq4_xs, 3, 4)
SHISO_MV_EXT_KQ(iq4_xs, 3, 8)
SHISO_MV_EXT_KQ(iq4_xs, 3, 16)
SHISO_MV_EXT_KQ(iq4_xs, 4, 4)
SHISO_MV_EXT_KQ(iq4_xs, 4, 8)
SHISO_MV_EXT_KQ(iq4_xs, 4, 16)
SHISO_MV_EXT_KQ(iq4_xs, 5, 4)
SHISO_MV_EXT_KQ(iq4_xs, 5, 8)
SHISO_MV_EXT_KQ(iq4_xs, 5, 16)

/* `shiso_kerneltest::dense::iq`'s table-copy argument block, the same four
 * bytes `shs_metal_args_iq_table_copy` carries in
 * `crates/shiso-metal/metal/generic/iq_tables.metal`. */
struct shiso_args_iq_table_copy {
    uint32_t table;
};
static_assert(sizeof(struct shiso_args_iq_table_copy) == 4,
              "IqTableCopyArgs is 4 bytes on the host and must be here too");

/* Copies one codebook table's raw bytes into a device buffer.
 *
 * The device twin of `kernel_iq_table_copy` in
 * `crates/shiso-metal/metal/generic/iq_tables.metal`, down to the table
 * order (0 `SHISO_IQ2XXS_GRID`, 1 `SHISO_IQ2XS_GRID`, 2 `SHISO_IQ2S_GRID`,
 * 3 `SHISO_IQ3XXS_GRID`, 4 `SHISO_IQ3S_GRID`, 5 `SHISO_IQ1S_GRID`,
 * 6 `SHISO_IQ4NL`, 7 `SHISO_KSIGNS_IQ2XS`, 8 `SHISO_KSIGNS64`) and the
 * index outside 0..8 writing nothing. The tables are format data transcribed
 * from `crates/shiso-modelfile/src/quant_tables.rs`, and this is the entry
 * that lets a test read them back off the device and byte-compare them
 * against the host constants: a changed value silently redefines every
 * checkpoint on disk, and the matvec cases would only catch it as a wrong
 * dot product that names no table.
 *
 * The element widths differ across the nine tables, so the copy moves
 * `uint64_t` words instead: every table's byte length is a multiple of
 * eight, every table is declared `__align__(8)` so the word view is aligned,
 * and the device is little-endian like the host, so the buffer ends up
 * holding the table's native-endian bytes with the elements in order. The
 * grid is sized for the largest table (2048 words); threads past the
 * selected table's word count exit. */
extern "C" __global__ void kernel_iq_table_copy(
        const struct shiso_args_iq_table_copy args, char *__restrict__ dst) {
    const uint64_t *src;
    uint32_t n_words;
    switch (args.table) {
        case 0: src = SHISO_IQ2XXS_GRID;    n_words = sizeof(SHISO_IQ2XXS_GRID) / 8; break;
        case 1: src = SHISO_IQ2XS_GRID;     n_words = sizeof(SHISO_IQ2XS_GRID) / 8; break;
        case 2: src = SHISO_IQ2S_GRID;      n_words = sizeof(SHISO_IQ2S_GRID) / 8; break;
        case 3: src = (const uint64_t *) SHISO_IQ3XXS_GRID;
                n_words = sizeof(SHISO_IQ3XXS_GRID) / 8; break;
        case 4: src = (const uint64_t *) SHISO_IQ3S_GRID;
                n_words = sizeof(SHISO_IQ3S_GRID) / 8; break;
        case 5: src = SHISO_IQ1S_GRID;      n_words = sizeof(SHISO_IQ1S_GRID) / 8; break;
        case 6: src = (const uint64_t *) SHISO_IQ4NL;
                n_words = sizeof(SHISO_IQ4NL) / 8; break;
        case 7: src = (const uint64_t *) SHISO_KSIGNS_IQ2XS;
                n_words = sizeof(SHISO_KSIGNS_IQ2XS) / 8; break;
        case 8: src = SHISO_KSIGNS64;       n_words = sizeof(SHISO_KSIGNS64) / 8; break;
        default: return;
    }
    const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n_words) {
        return;
    }
    ((uint64_t *) dst)[gid] = src[gid];
}
