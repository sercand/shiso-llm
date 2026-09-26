//! The FFT against a direct DFT.

use super::Fft;
use crate::num::f64_of;

#[test]
fn matches_the_direct_dft() {
    let n = 512;
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    let frame: Vec<f64> = (0..n)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            f64_of(usize::try_from(state % 2001).expect("small")) / 1000.0 - 1.0
        })
        .collect();
    let fft = Fft::new(n).expect("power of two");
    assert_eq!(fft.len(), n);
    let mut work = fft.work();
    work.frame.copy_from_slice(&frame);
    fft.spectrum(&mut work);
    let (re, im): (Vec<f64>, Vec<f64>) = work.bins.iter().map(|c| (c.re, c.im)).unzip();
    assert_eq!(re.len(), n / 2 + 1);
    for k in 0..=n / 2 {
        let (mut dr, mut di) = (0.0, 0.0);
        for (t, x) in frame.iter().enumerate() {
            let w = -2.0 * std::f64::consts::PI * f64_of(k * t) / f64_of(n);
            dr += x * w.cos();
            di += x * w.sin();
        }
        assert!(
            (re[k] - dr).abs() < 1e-9 && (im[k] - di).abs() < 1e-9,
            "bin {k}"
        );
    }
}

#[test]
fn refuses_other_sizes() {
    assert!(Fft::new(400).is_err());
    assert!(Fft::new(1).is_err());
}
