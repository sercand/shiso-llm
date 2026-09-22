//! Page geometry for the paged key/value cache.
//!
//! These constants are shared by the host allocator and every kernel that
//! addresses a cache row, so the arithmetic is pinned here as data: a change to
//! a constant has to be a deliberate edit to one of these assertions.

use shiso_core::page::{
    BYTES_PER_PAGE, BYTES_PER_POSITION, MROPE_BACK, PAGE_MASK, PAGE_SHIFT, PAGE_TOKENS, kv_row,
    pages_for,
};

/// Indexer pooling ratio that the mrope prefix has to cover.
fn compression_ratio() -> u32 {
    4
}

#[test]
fn page_size_is_a_power_of_two_of_its_shift() {
    assert_eq!(PAGE_TOKENS, 1 << PAGE_SHIFT);
    assert_eq!(PAGE_MASK, PAGE_TOKENS - 1);
}

#[test]
fn page_size_divides_every_tiled_reader() {
    // The indexer pools four positions into one key, the attention kernels tile
    // 32 keys, and the score kernel tiles 64 blocks. A page that is a multiple
    // of all three means a tile never straddles a page boundary, so a tiled
    // reader translates one position per tile instead of one per element.
    assert_eq!(PAGE_TOKENS % 4, 0);
    assert_eq!(PAGE_TOKENS % 32, 0);
    assert_eq!(PAGE_TOKENS % 64, 0);
    assert_eq!(PAGE_TOKENS, 256);
    assert_eq!(PAGE_SHIFT, 8);
    assert_eq!(PAGE_MASK, 255);
}

#[test]
fn page_bytes_is_the_stated_seven_and_a_half_mebibytes() {
    assert_eq!(BYTES_PER_POSITION, 30 * 1024);
    assert_eq!(BYTES_PER_PAGE, 7_864_320);
    assert_eq!(BYTES_PER_PAGE, BYTES_PER_POSITION * u64::from(PAGE_TOKENS));
}

#[test]
fn mrope_back_covers_the_compression_ratio() {
    // The pooled-key kernel rounds a position down to the compression boundary,
    // which can name a position up to ratio-1 slots before the chunk starts, so
    // the prefix has to be at least that deep. The equality above is the pin;
    // the ratio is what makes the number mean something.
    assert_eq!(MROPE_BACK, 4);
    assert!(MROPE_BACK > compression_ratio() - 1);
}

#[test]
fn kv_row_walks_the_page_table() {
    // Position 300 sits in page index 1 at offset 44, and this table maps page
    // index 1 to physical page 0, so the row is 44. The 7 * 256 + 44 pairing is
    // position 556, whose page index is 2: the arithmetic below is the mirror of
    // the kernel's `pages[p >> SHIFT] * TOKENS + (p & MASK)`, so it is what the
    // kernels agree with rather than any single worked example.
    assert_eq!(kv_row(&[5, 0, 7], 300), Some(44));
    assert_eq!(kv_row(&[5, 0, 7], 556), Some(7 * 256 + 44));
    assert_eq!(kv_row(&[5, 7], 300), Some(7 * 256 + 44));
    assert_eq!(kv_row(&[5, 0, 7], 0), Some(5 * 256));
    assert_eq!(kv_row(&[5, 0, 7], 255), Some(5 * 256 + 255));
    assert_eq!(kv_row(&[5, 0, 7], 256), Some(0));
    assert_eq!(kv_row(&[5, 0, 7], 511), Some(255));
    assert_eq!(kv_row(&[5, 0, 7], 512), Some(7 * 256));
}

#[test]
fn kv_row_past_the_table_is_none() {
    assert_eq!(kv_row(&[5], 300), None);
    assert_eq!(kv_row(&[5], 256), None);
    assert_eq!(kv_row(&[], 0), None);
    assert_eq!(kv_row(&[5, 0], 512), None);
}

#[test]
fn every_position_of_a_two_page_table_maps_into_its_own_page() {
    let pages = [3, 9];
    for position in 0..2 * PAGE_TOKENS {
        let row = kv_row(&pages, position).expect("position is inside the table");
        let page_index = position / PAGE_TOKENS;
        let expected = pages[usize::try_from(page_index).expect("two pages")] * PAGE_TOKENS
            + position % PAGE_TOKENS;
        assert_eq!(row, expected);
        assert_eq!(
            row / PAGE_TOKENS,
            pages[usize::try_from(page_index).expect("two pages")]
        );
    }
}

#[test]
fn page_count_rounds_up() {
    assert_eq!(pages_for(0), 0);
    assert_eq!(pages_for(1), 1);
    assert_eq!(pages_for(PAGE_TOKENS), 1);
    assert_eq!(pages_for(PAGE_TOKENS + 1), 2);
    assert_eq!(pages_for(1_000), 4);
}
