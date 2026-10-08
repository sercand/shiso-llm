//! Anti-transcription-slip pins for the codebook tables in [`quant_tables`].
//!
//! A table transcribed with one wrong digit decodes every tensor of that
//! format slightly wrong and no test downstream notices: the logits drift and
//! nothing fails. So each table is pinned here by length, first element, last
//! element, and a wrapping-sum checksum over all elements.
//!
//! The expected values were computed directly from the table text in
//! `kernels/cuda/mmq/ggml-common.h` (the vendored, read-only format
//! definition) by a throwaway parse of its `GGML_TABLE_BEGIN` blocks, not from
//! the Rust transcription below, so the two sides of each assertion are
//! independent. The header line ranges are named next to each checksum.

use shiso_modelfile::quant_tables::{
    IQ1S_GRID, IQ2S_GRID, IQ2XS_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS64,
};

/// Wrapping sum over the whole table, in the table's own element width.
fn checksum_u64(table: &[u64]) -> u64 {
    table.iter().fold(0_u64, |acc, &v| acc.wrapping_add(v))
}

/// Wrapping sum over the whole table, in the table's own element width.
fn checksum_u32(table: &[u32]) -> u32 {
    table.iter().fold(0_u32, |acc, &v| acc.wrapping_add(v))
}

#[test]
fn ksigns64_matches_the_header() {
    // kernels/cuda/mmq/ggml-common.h:514-547
    assert_eq!(KSIGNS64.len(), 128);
    assert_eq!(KSIGNS64[0], 0x0000_0000_0000_0000);
    assert_eq!(KSIGNS64[127], 0xffff_ffff_ffff_ffff);
    assert_eq!(checksum_u64(&KSIGNS64), 0xffff_ffff_ffff_ffc0);
}

#[test]
fn iq2xs_grid_matches_the_header() {
    // kernels/cuda/mmq/ggml-common.h:617-746
    assert_eq!(IQ2XS_GRID.len(), 512);
    assert_eq!(IQ2XS_GRID[0], 0x0808_0808_0808_0808);
    assert_eq!(IQ2XS_GRID[511], 0x2b2b_2b2b_2b2b_2b2b);
    assert_eq!(checksum_u64(&IQ2XS_GRID), 0x2a14_e555_1003_96a1);
}

#[test]
fn iq2s_grid_matches_the_header() {
    // kernels/cuda/mmq/ggml-common.h:748-1005
    assert_eq!(IQ2S_GRID.len(), 1024);
    assert_eq!(IQ2S_GRID[0], 0x0808_0808_0808_0808);
    assert_eq!(IQ2S_GRID[1023], 0x2b2b_2b2b_2b2b_2b2b);
    assert_eq!(checksum_u64(&IQ2S_GRID), 0x75fe_5f6c_c209_e3f2);
}

#[test]
fn iq3xxs_grid_matches_the_header() {
    // kernels/cuda/mmq/ggml-common.h:1007-1040
    assert_eq!(IQ3XXS_GRID.len(), 256);
    assert_eq!(IQ3XXS_GRID[0], 0x0404_0404);
    assert_eq!(IQ3XXS_GRID[255], 0x3e34_1c04);
    assert_eq!(checksum_u32(&IQ3XXS_GRID), 0x3a8b_433a);
}

#[test]
fn iq3s_grid_matches_the_header() {
    // kernels/cuda/mmq/ggml-common.h:1042-1107
    assert_eq!(IQ3S_GRID.len(), 512);
    assert_eq!(IQ3S_GRID[0], 0x0101_0101);
    assert_eq!(IQ3S_GRID[511], 0x0f0f_0101);
    assert_eq!(checksum_u32(&IQ3S_GRID), 0x6894_9e56);
}

#[test]
fn iq1s_grid_matches_the_header() {
    // kernels/cuda/mmq/ggml-common.h:1124-1637 (NGRID_IQ1S = 2048)
    assert_eq!(IQ1S_GRID.len(), 2048);
    assert_eq!(IQ1S_GRID[0], 0xffff_ffff_ffff_ffff);
    assert_eq!(IQ1S_GRID[2047], 0x0101_0101_0101_0101);
    assert_eq!(checksum_u64(&IQ1S_GRID), 0x6956_4f61_5572_510c);
}
