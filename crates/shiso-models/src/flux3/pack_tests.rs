//! Ids, token order and the padded length against the reference's rules.

#![expect(
    clippy::float_cmp,
    reason = "the ids are small multiples of one half, exact in f32, and the reference's integers"
)]
#![expect(
    clippy::cast_precision_loss,
    reason = "reference indices are at most ten, exact in f32"
)]

use shiso_vision::generator::ImageGenError;

use super::pack::{Mode, Pack, image_ids, padded_len, reference_ids, text_ids};

#[test]
fn ids_scale_spatial_axes_half() {
    let ids = image_ids(2, 3);
    assert_eq!(ids.len(), 6);
    assert_eq!(ids[0], [0.0, 0.0, 0.0, 0.0]);
    assert_eq!(ids[1], [0.0, 0.0, 0.5, 0.0], "column steps the third axis");
    assert_eq!(ids[3], [0.0, 0.5, 0.0, 0.0], "row steps the second axis");
    assert_eq!(ids[5], [0.0, 0.5, 1.0, 0.0]);
}

#[test]
fn reference_ids_l_axis() {
    for index in 0..10 {
        let ids = reference_ids(1, 2, index);
        let l = -(1.0 + index as f32);
        assert_eq!(ids[0], [0.0, 0.0, 0.0, l]);
        assert_eq!(ids[1], [0.0, 0.0, 0.5, l]);
    }
}

#[test]
fn text_ids_are_positions_on_the_last_axis() {
    let ids = text_ids(4);
    assert_eq!(ids[0], [0.0; 4]);
    assert_eq!(ids[3], [0.0, 0.0, 0.0, 3.0]);
}

#[test]
fn padded_len_80_81_4096_4097_0() {
    assert_eq!(padded_len(1).expect("one token"), 80);
    assert_eq!(
        padded_len(80).expect("exact multiple gets no extra pad"),
        80
    );
    assert_eq!(padded_len(81).expect("one past"), 160);
    assert_eq!(padded_len(4000).expect("exact"), 4000);
    assert_eq!(padded_len(4001).expect("next multiple is 4080"), 4080);
    assert_eq!(
        padded_len(4081).expect("the cap beats the next multiple"),
        4096
    );
    assert_eq!(padded_len(4096).expect("the longest prompt"), 4096);
    assert!(matches!(
        padded_len(4097),
        Err(ImageGenError::PromptTooLong {
            tokens: 4097,
            max: 4096
        })
    ));
    assert!(matches!(padded_len(0), Err(ImageGenError::EmptyPrompt)));
}

#[test]
fn single_stream_order_matches_reference_modes() {
    let pack = Pack::single(1024, 256, 80);
    let order = pack.single_stream_order();
    assert_eq!(
        order,
        vec![
            (Mode::Image, 0..1024),
            (Mode::ImageCond, 0..256),
            (Mode::Text, 0..80),
        ]
    );
    assert_eq!(pack.stream_len(), 1360);
    let names_sorted = {
        let mut modes: Vec<Mode> = order.iter().map(|(m, _)| *m).collect();
        let before = modes.clone();
        modes.sort();
        modes == before
    };
    assert!(names_sorted, "the order is the sorted mode order");
}

#[test]
fn an_empty_mode_is_left_out_of_the_stream() {
    let pack = Pack::single(1024, 0, 160);
    assert_eq!(
        pack.single_stream_order(),
        vec![(Mode::Image, 0..1024), (Mode::Text, 0..160)]
    );
    assert_eq!(pack.stream_len(), 1184);
}
