use super::config::*;

#[test]
fn count_blocks_from_tensor_map() {
    let names = [
        "double_blocks.0.img_attn.qkv.weight",
        "double_blocks.4.img_attn.qkv.weight",
        "single_blocks.19.linear1.weight",
    ];

    let depth = count_blocks_internal(names.iter().copied(), "double_blocks.").expect("depth");
    assert_eq!(depth, 5);

    let single_depth =
        count_blocks_internal(names.iter().copied(), "single_blocks.").expect("depth");
    assert_eq!(single_depth, 20);
}
