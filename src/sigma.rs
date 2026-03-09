use crate::matrix::Scalar;

/// Non-linear σ function used in Sky98 PoW
///
/// Design goals:
/// - Deterministic across all hardware
/// - Non-linear (breaks algebraic shortcuts)
/// - Very cheap to compute
/// - Maps well to CPUs, GPUs, and TPUs
///
/// This function operates entirely in integer space.
/// Arithmetic is wrapping by construction (mod 2^32).
///
/// σ(x) is a reversible-looking 32-bit mixing function that preserves much
/// more entropy than the original MVP version.
#[inline]
pub fn sigma(x: Scalar) -> Scalar {
    let x1 = x ^ x.rotate_left(13) ^ 0x9E37_79B9;
    let x2 = x1.wrapping_mul(0x85EB_CA6B);
    let x3 = x2 ^ x2.rotate_right(11);
    x3.wrapping_mul(0xC2B2_AE35) ^ x.rotate_right(7)
}

/// Apply σ element-wise to a slice
///
/// This is useful for batch application outside of Matrix
/// Batch σ application (reserved for SIMD / GPU paths)
#[inline]
#[allow(dead_code)]
pub fn sigma_slice(data: &mut [Scalar]) {
    for v in data.iter_mut() {
        *v = sigma(*v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigma_basic() {
        let x: Scalar = 0x12345678;
        let y = sigma(x);

        // Determinism check
        assert_eq!(y, sigma(x));

        // σ should mutate the value
        assert_ne!(x, y);
    }

    #[test]
    fn test_sigma_zero() {
        assert_ne!(sigma(0), 0);
    }

    #[test]
    fn test_sigma_distribution() {
        // Ensure σ does not collapse many values into a tiny output space.
        let mut seen = std::collections::HashSet::new();
        for x in 0..10_000u32 {
            seen.insert(sigma(x));
        }

        assert!(seen.len() > 9_900);
    }

    #[test]
    fn test_sigma_slice() {
        let mut data = vec![1, 2, 3, 4, 5];
        let expected: Vec<Scalar> = data.iter().map(|&x| sigma(x)).collect();

        sigma_slice(&mut data);

        assert_eq!(data, expected);
    }
}
