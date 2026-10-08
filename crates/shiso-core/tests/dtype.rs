//! Tensor format table.
//!
//! The on-disk ids and the block dimensions are what a reader uses to work out
//! how many bytes a tensor occupies, so both are pinned as data. A wrong block
//! size does not fail loudly: it reads the next tensor's bytes and produces
//! plausible garbage.

use shiso_core::DType;

/// Every variant, so the round-trip test cannot silently skip one.
const ALL: [DType; 29] = [
    DType::F32,
    DType::F16,
    DType::Q4_0,
    DType::Q5_0,
    DType::Q5_1,
    DType::Q8_0,
    DType::Q2_K,
    DType::Q4_K,
    DType::Q5_K,
    DType::Q6_K,
    DType::Q8_K,
    DType::IQ2_XXS,
    DType::IQ2_XS,
    DType::IQ3_XXS,
    DType::IQ1_S,
    DType::IQ4_NL,
    DType::IQ3_S,
    DType::IQ2_S,
    DType::IQ4_XS,
    DType::I32,
    DType::IQ1_M,
    DType::BF16,
    DType::MXFP4,
    DType::Exl3K4,
    DType::Exl3K5,
    DType::Exl3K6,
    DType::Exl3Ngram6,
    DType::Pq2_0,
    DType::Ptq1_0,
];

#[test]
fn every_variant_round_trips_through_its_on_disk_id() {
    for dtype in ALL {
        let id = dtype.gguf_id();
        assert_eq!(DType::from_gguf_id(id), Some(dtype), "{id} round-trip");
    }
}

#[test]
fn on_disk_ids_are_unique() {
    for (index, a) in ALL.iter().enumerate() {
        for b in &ALL[index + 1..] {
            assert_ne!(a.gguf_id(), b.gguf_id(), "{a:?} and {b:?} share an id");
        }
    }
}

#[test]
fn discriminants_are_the_documented_ids() {
    let expected: [(DType, u32); 29] = [
        (DType::F32, 0),
        (DType::F16, 1),
        (DType::Q4_0, 2),
        (DType::Q5_0, 6),
        (DType::Q5_1, 7),
        (DType::Q8_0, 8),
        (DType::Q2_K, 10),
        (DType::Q4_K, 12),
        (DType::Q5_K, 13),
        (DType::Q6_K, 14),
        (DType::Q8_K, 15),
        (DType::IQ2_XXS, 16),
        (DType::IQ2_XS, 17),
        (DType::IQ3_XXS, 18),
        (DType::IQ1_S, 19),
        (DType::IQ4_NL, 20),
        (DType::IQ3_S, 21),
        (DType::IQ2_S, 22),
        (DType::IQ4_XS, 23),
        (DType::I32, 26),
        (DType::IQ1_M, 29),
        (DType::BF16, 30),
        (DType::MXFP4, 39),
        (DType::Exl3K4, 68),
        (DType::Exl3K5, 69),
        (DType::Exl3K6, 70),
        (DType::Exl3Ngram6, 72),
        (DType::Pq2_0, 142),
        (DType::Ptq1_0, 143),
    ];
    for (dtype, id) in expected {
        assert_eq!(dtype.gguf_id(), id);
        assert_eq!(DType::from_gguf_id(id), Some(dtype));
    }
}

#[test]
fn unknown_id_is_none_rather_than_a_default() {
    assert_eq!(DType::from_gguf_id(99), None);
    assert_eq!(DType::from_gguf_id(3), None);
    assert_eq!(DType::from_gguf_id(u32::MAX), None);
    // The ids in the gaps between the ones this build reads stay unknown.
    for id in 71..=100 {
        if id != 72 {
            assert_eq!(DType::from_gguf_id(id), None, "id {id} should be unknown");
        }
    }
}

#[test]
fn block_dimensions_of_the_formats_that_matter() {
    let pinned: [(DType, u32, u32); 15] = [
        (DType::Pq2_0, 128, 34),
        (DType::Ptq1_0, 128, 28),
        (DType::MlxAffine2Gs128, 128, 36),
        (DType::Q8_0, 32, 34),
        (DType::Q4_K, 256, 144),
        (DType::MXFP4, 32, 17),
        (DType::IQ2_XXS, 256, 66),
        (DType::Exl3Ngram6, 160, 122),
        (DType::IQ2_XS, 256, 74),
        (DType::IQ3_XXS, 256, 98),
        (DType::IQ1_S, 256, 50),
        (DType::IQ3_S, 256, 110),
        (DType::IQ2_S, 256, 82),
        (DType::IQ4_XS, 256, 136),
        (DType::IQ1_M, 256, 56),
    ];
    for (dtype, elems, bytes) in pinned {
        assert_eq!(dtype.block_elems().get(), elems, "{dtype:?} elems");
        assert_eq!(dtype.block_bytes().get(), bytes, "{dtype:?} bytes");
    }
}

#[test]
fn mxfp4_is_one_byte_narrower_than_a_same_width_block_quant() {
    // Both pack 32 elements and both carry one shared scale word, but the
    // microscaling exponent is a single byte where a block quant's scale is
    // two, so 17 rather than the 18 that IQ4_NL uses.
    assert_eq!(DType::MXFP4.block_elems(), DType::IQ4_NL.block_elems());
    assert_eq!(DType::MXFP4.block_bytes().get(), 17);
    assert_eq!(DType::IQ4_NL.block_bytes().get(), 18);
}

#[test]
fn unquantized_formats_are_one_element_per_block() {
    for dtype in [DType::F32, DType::F16, DType::BF16, DType::I32] {
        assert_eq!(dtype.block_elems().get(), 1, "{dtype:?}");
        assert!(!dtype.is_quantized(), "{dtype:?}");
    }
    for dtype in ALL {
        if dtype.block_elems().get() > 1 {
            assert!(dtype.is_quantized(), "{dtype:?}");
        }
    }
}

#[test]
fn block_bytes_never_zero_so_division_needs_no_guard() {
    for dtype in ALL {
        assert!(dtype.block_bytes().get() > 0);
        assert!(dtype.block_elems().get() > 0);
    }
}

#[test]
fn type_names_are_lowercase_and_distinct() {
    let mut seen: Vec<&str> = Vec::new();
    for dtype in ALL {
        let name = dtype.type_name();
        assert!(!name.is_empty());
        assert_eq!(name, name.to_lowercase(), "{dtype:?} name");
        assert!(!seen.contains(&name), "{name} repeated");
        seen.push(name);
    }
}

#[test]
fn ternary_and_two_bit_affine_are_told_apart() {
    // The two ternary blocks carry their scale inside the block; the MLX
    // 2-bit type keeps scales and biases in arrays of their own, which is why
    // it answers the affine question and they do not.
    for dtype in [DType::Pq2_0, DType::Ptq1_0] {
        assert!(dtype.is_ternary(), "{dtype:?}");
        assert!(!dtype.is_affine(), "{dtype:?}");
        assert_eq!(dtype.affine_bits(), None, "{dtype:?}");
    }
    assert!(DType::MlxAffine2Gs128.is_affine());
    assert!(!DType::MlxAffine2Gs128.is_ternary());
    assert_eq!(DType::MlxAffine2Gs128.affine_bits(), Some(2));
}
