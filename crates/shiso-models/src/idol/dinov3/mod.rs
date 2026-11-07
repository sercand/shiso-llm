//! The `DINOv3` `ViT` tower: configuration, `RoPE` and the block graph.
//!
//! The tower is the picture's semantic half of the conditioning: it answers one
//! `CLS` token, four register tokens and one token per 16x16 patch, each 1280
//! wide. The denoiser reads that sequence beside the `VAE`'s, which is why the
//! two are assembled together in [`crate::idol::conditioning`].

mod config;
pub(crate) mod encoder;
mod rope;

pub use config::{DinoConfig, IMAGENET_MEAN, IMAGENET_STD, normalize_pixels};
pub use rope::rope_tables;

#[cfg(test)]
mod tests {
    use super::{DinoConfig, normalize_pixels, rope_tables};

    #[test]
    fn the_checkpoint_constants_are_the_ones_the_config_document_carries() {
        let config = DinoConfig::vit_h_plus();
        assert_eq!(config.hidden, 1280);
        assert_eq!(config.layers, 32);
        assert_eq!(config.heads, 20);
        assert_eq!(config.registers, 4);
        assert_eq!(config.patch, 16);
        assert_eq!(config.intermediate, 5120);
        assert!((config.rope_theta - 100.0).abs() < f64::EPSILON);
        assert!((config.layer_norm_eps - 1e-5).abs() < f64::EPSILON);
        // 1280 / 20 heads.
        assert_eq!(config.head_dim(), 64);
    }

    #[test]
    fn the_pixel_normalisation_is_the_imagenet_affine_in_unit_space() {
        // A mid-grey pixel maps to (0.5 - mean) / std on each channel.
        let rgb = [0.5, 0.5, 0.5];
        let out = normalize_pixels(&rgb);
        for (channel, value) in out.iter().enumerate() {
            let want = (0.5 - f64::from(super::IMAGENET_MEAN[channel]))
                / f64::from(super::IMAGENET_STD[channel]);
            assert!(
                (f64::from(*value) - want).abs() < 1e-6,
                "channel {channel}: got {value}, want {want}"
            );
        }
    }

    #[test]
    fn the_corner_patch_sits_at_minus_half() {
        // A 2x2 grid's first patch centre is (0.5/2)*2 - 1 = -0.5 on both
        // axes, and inv_freq[0] is 1, so its first angle is 2*pi*-0.5 = -pi.
        let (cos, sin) = rope_tables(&DinoConfig::vit_h_plus(), 2, 2);
        assert!((cos[0] + 1.0).abs() < 1e-5, "cos(-pi) = -1, got {}", cos[0]);
        assert!(sin[0].abs() < 1e-5, "sin(-pi) = 0, got {}", sin[0]);
    }

    #[test]
    fn the_angle_vector_is_tiled_twice() {
        // The reference builds head_dim/4 frequencies per axis, flattens the
        // two axes to head_dim/2, then tiles that to head_dim. `rotate_half`
        // reads the halves as one rotation, so the halves must be identical.
        let config = DinoConfig::vit_h_plus();
        let head_dim = config.head_dim();
        let (cos, _) = rope_tables(&config, 3, 3);
        for token in 0..9 {
            for i in 0..head_dim / 2 {
                assert!(
                    (cos[token * head_dim + i] - cos[token * head_dim + i + head_dim / 2]).abs()
                        < 1e-6,
                    "token {token} column {i}"
                );
            }
        }
    }

    #[test]
    fn every_angle_is_a_unit_pair() {
        let (cos, sin) = rope_tables(&DinoConfig::vit_h_plus(), 4, 4);
        assert_eq!(cos.len(), 16 * 64);
        for (c, s) in cos.iter().zip(&sin) {
            assert!((c * c + s * s - 1.0).abs() < 1e-5, "{c} {s}");
        }
    }

    #[test]
    fn a_wider_grid_moves_the_patch_centres() {
        // The same patch index is a different coordinate in a different grid:
        // the tower must compute the tables from the input, not from the
        // config's training size.
        let config = DinoConfig::vit_h_plus();
        let (square, _) = rope_tables(&config, 4, 4);
        let (wide, _) = rope_tables(&config, 4, 8);
        assert_ne!(square[..64], wide[..64]);
    }
}
