use crate::matrix::Matrix;

/// Deterministic masking function for Sky98 PoW
///
/// The mask:
/// - Is derived from a public seed
/// - Is stable for a given (seed, round)
/// - Zeroes out selected elements to break structure
///
/// This mask does NOT need to be cryptographically secure.
/// It only needs to be unpredictable *before* the seed is known.
pub struct Mask {
    seed: u64,
}

impl Mask {
    /// Create a new mask generator from a seed
    pub fn new(seed: u64) -> Self {
        Mask { seed }
    }

    /// Apply the mask in-place to a matrix
    ///
    /// Rule:
    /// Each element is either kept or zeroed depending on a
    /// simple hash of (index XOR seed).
    pub fn apply(&self, matrix: &mut Matrix) {
        let n = matrix.n;
        let mut state = self.seed;

        for i in 0..n {
            for j in 0..n {
                // Deterministic pseudo-random bit
                state = mix(state ^ ((i * n + j) as u64));

                // Use lowest bit as mask decision
                if (state & 1) == 0 {
                    matrix.set(i, j, 0);
                }
            }
        }
    }
}

/// A tiny mixing function (xorshift-inspired)
///
/// Fast, deterministic, good diffusion.
/// NOT cryptographic — and does not need to be.
#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mask_determinism() {
        let mut m1 = Matrix::from_vec(3, vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]);

        let mut m2 = m1.clone();

        let mask = Mask::new(12345);
        mask.apply(&mut m1);
        mask.apply(&mut m2);

        assert_eq!(m1, m2);
    }

    #[test]
    fn test_mask_changes_matrix() {
        let mut m = Matrix::from_vec(4, (1..=16).collect());

        let original = m.clone();
        let mask = Mask::new(999);

        mask.apply(&mut m);

        // Ensure something changed
        assert_ne!(m, original);
    }

    #[test]
    fn test_mask_allows_nonzero_values() {
        let mut m = Matrix::from_vec(4, vec![42; 16]);
        let mask = Mask::new(1);

        mask.apply(&mut m);

        let non_zero_count = m.data.iter().filter(|&&x| x != 0).count();
        assert!(non_zero_count > 0);
    }
}
