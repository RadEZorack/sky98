use crate::matrix::{Matrix, Scalar};
use crate::sigma::sigma;
use crate::mask::Mask;

/// Parameters controlling the Sky98 PoW computation
#[derive(Debug, Clone)]
pub struct PowParams {
    pub matrix_size: usize, // n
    pub rounds: usize,      // r
}

/// A minimal seed representation for the PoW
///
/// In the MVP, we treat this as an opaque 64-bit value.
/// Later this can be expanded to hash(prev_block || epoch || identity).
#[derive(Debug, Clone, Copy)]
pub struct Seed {
    pub value: u64,
}

impl Seed {
    /// Derive a round-specific seed
    #[inline]
    pub fn round_seed(&self, round: usize) -> u64 {
        self.value ^ (round as u64).wrapping_mul(0x9E3779B97F4A7C15)
    }
}

/// Deterministically generate initial matrices from a seed and nonce
///
/// This is intentionally simple and auditable.
/// We only need:
/// - Determinism
/// - Uniform-ish distribution
/// - No miner control
pub fn seed_to_matrices(
    seed: Seed,
    nonce: u64,
    n: usize,
) -> (Matrix, Matrix) {
    let mut a = Matrix::zeros(n);
    let mut b = Matrix::zeros(n);

    let mut state = seed.value ^ nonce;

    for i in 0..n {
        for j in 0..n {
            state = mix(state);
            a.set(i, j, (state & 0xFFFF_FFFF) as Scalar);

            state = mix(state);
            b.set(i, j, (state & 0xFFFF_FFFF) as Scalar);
        }
    }

    (a, b)
}

/// The full Sky98 Proof-of-Work computation
///
/// This function:
/// - Is fully deterministic
/// - Performs O(r * n^3) work
/// - Maps cleanly to GPU/TPU pipelines
///
/// Returns the final matrix F_r
pub fn sky98_pow(
    seed: Seed,
    nonce: u64,
    params: &PowParams,
) -> Matrix {
    let n = params.matrix_size;
    let rounds = params.rounds;

    let (mut a, mut b) = seed_to_matrices(seed, nonce, n);

    for round in 0..rounds {
        // 1) Matrix multiplication
        let mut c = a.mul(&b);

        // 2) Apply σ element-wise
        c.map_inplace(sigma);

        // 3) Apply deterministic mask
        let mask = Mask::new(seed.round_seed(round));
        mask.apply(&mut c);

        // 4) Prepare next round
        a = c.clone();
        b = c.permute();
    }

    a
}

/// Tiny deterministic mixing function used for seed expansion
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
    fn test_pow_determinism() {
        let params = PowParams {
            matrix_size: 4,
            rounds: 3,
        };

        let seed = Seed { value: 123456 };
        let nonce = 42;

        let out1 = sky98_pow(seed, nonce, &params);
        let out2 = sky98_pow(seed, nonce, &params);

        assert_eq!(out1, out2);
    }

    #[test]
    fn test_pow_nonce_variation() {
        let params = PowParams {
            matrix_size: 4,
            rounds: 3,
        };

        let seed = Seed { value: 999 };

        let out1 = sky98_pow(seed, 1, &params);
        let out2 = sky98_pow(seed, 2, &params);

        assert_ne!(out1, out2);
    }

    #[test]
    fn test_pow_output_nonzero() {
        let params = PowParams {
            matrix_size: 4,
            rounds: 2,
        };

        let seed = Seed { value: 777 };
        let nonce = 0;

        let out = sky98_pow(seed, nonce, &params);

        let non_zero = out.data.iter().any(|&x| x != 0);
        assert!(non_zero);
    }
}
