//! The demux types and the container dispatch.

use super::{CodecConfig, ConfigFormat, DemuxError, Pts, open};

#[test]
fn nal_length_size_from_each_record() {
    let mut avcc = vec![1, 0x64, 0, 0x1f, 0xfc];
    for (bits, size) in [(0, 1), (1, 2), (3, 4)] {
        avcc[4] = 0xfc | bits;
        let config = CodecConfig::new(ConfigFormat::Avcc, avcc.clone());
        assert_eq!(config.nal_length_size(), Ok(size));
    }
    avcc[4] = 0xfe;
    assert!(matches!(
        CodecConfig::new(ConfigFormat::Avcc, avcc).nal_length_size(),
        Err(DemuxError::Malformed { .. })
    ));

    let mut hvcc = vec![0u8; 23];
    hvcc[21] = 0xf3;
    assert_eq!(
        CodecConfig::new(ConfigFormat::Hvcc, hvcc).nal_length_size(),
        Ok(4)
    );
    assert!(matches!(
        CodecConfig::new(ConfigFormat::Hvcc, vec![1; 21]).nal_length_size(),
        Err(DemuxError::Malformed { .. })
    ));
    for format in [
        ConfigFormat::Vpcc,
        ConfigFormat::Av1c,
        ConfigFormat::MatroskaVp9,
    ] {
        assert!(matches!(
            CodecConfig::new(format, vec![0; 32]).nal_length_size(),
            Err(DemuxError::Unsupported { .. })
        ));
    }
}

#[test]
fn timestamps_order_by_ticks() {
    assert!(Pts::new(-5) < Pts::new(0));
    assert_eq!(Pts::new(42).ticks(), 42);
}

#[test]
fn dispatch_refuses_what_it_does_not_read() {
    let riff = b"RIFF\0\0\0\0AVI LIST";
    let err = open(riff).expect_err("AVI");
    assert!(matches!(err, DemuxError::Unsupported { .. }));
    assert!(
        err.to_string().contains("avi") || err.to_string().contains("AVI"),
        "{err}"
    );
    // Matroska magic with nothing behind it claims the format and breaks it.
    assert!(matches!(
        open(&[0x1a, 0x45, 0xdf, 0xa3, 0x80]),
        Err(DemuxError::Malformed { .. })
    ));
    // As does an MP4 whose only box is the file type.
    let ftyp = [&20u32.to_be_bytes()[..], b"ftypisom\0\0\0\0isom"].concat();
    assert!(matches!(open(&ftyp), Err(DemuxError::Malformed { .. })));
}
