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
/// σ(x) = ((x <<< 13) XOR x) & 0xFFFF
#[inline]
pub fn sigma(x: Scalar) -> Scalar {
    let rotated = x.rotate_left(13);
    (rotated ^ x) & 0xFFFF
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
        assert_ne!(x & 0xFFFF, y);
    }

    #[test]
    fn test_sigma_zero() {
        assert_eq!(sigma(0), 0);
    }

    #[test]
    fn test_sigma_distribution() {
        // Ensure σ does not collapse many values
        let mut seen = std::collections::HashSet::new();
        for x in 0..10_000u32 {
            seen.insert(sigma(x));
        }

        // Very weak sanity check: collisions should exist,
        // but output should not collapse to tiny space.
        assert!(seen.len() > 1_000);
    }

    #[test]
    fn test_sigma_slice() {
        let mut data = vec![1, 2, 3, 4, 5];
        let expected: Vec<Scalar> = data.iter().map(|&x| sigma(x)).collect();

        sigma_slice(&mut data);

        assert_eq!(data, expected);
    }
}
