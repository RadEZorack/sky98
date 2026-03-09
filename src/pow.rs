use crate::matrix::{Matrix, Scalar};
use crate::sigma::sigma;
use crate::mask::Mask;

/// Parameters controlling the Sky98 PoW computation
#[derive(Debug, Clone)]
pub struct PowParams {
    pub matrix_size: usize, // n
    pub rounds: usize,      // r
}

/// A deterministic summary of completed work.
///
/// The commitment is a lightweight stand-in for a future cryptographic digest.
/// The score is a toy ranking rule used by the demo harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkResult {
    pub commitment: u64,
    pub score: u32,
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

    /// Derive a per-attempt round seed so mask layouts vary across work units.
    #[inline]
    pub fn round_nonce_seed(&self, round: usize, nonce: u64) -> u64 {
        mix(
            self.round_seed(round)
                ^ nonce.rotate_left((round % 63) as u32)
                ^ 0xD6E8FEB86659FD93,
        )
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
        let mask = Mask::new(seed.round_nonce_seed(round, nonce));
        mask.apply(&mut c);

        // 4) Prepare next round
        a = c.clone();
        b = c.permute();
    }

    a
}

/// Compute a compact work summary from a completed matrix result.
///
/// This is intentionally simple for the MVP. A real network would replace the
/// commitment with a cryptographic digest and likely derive scoring from that.
pub fn summarize_work(matrix: &Matrix) -> WorkResult {
    let commitment = commit_matrix(matrix);
    let score = commitment.leading_zeros();

    WorkResult { commitment, score }
}

/// Execute a full work unit and return both the final matrix and its summary.
pub fn evaluate_work(
    seed: Seed,
    nonce: u64,
    params: &PowParams,
) -> (Matrix, WorkResult) {
    let matrix = sky98_pow(seed, nonce, params);
    let summary = summarize_work(&matrix);
    (matrix, summary)
}

/// Tiny deterministic mixing function used for seed expansion
#[inline]
fn mix(mut x: u64) -> u64 {
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

/// Deterministically fold a matrix into a 64-bit commitment.
#[inline]
fn commit_matrix(matrix: &Matrix) -> u64 {
    let mut state = 0x6A09E667F3BCC909u64 ^ (matrix.n as u64);

    for &value in &matrix.data {
        state ^= value as u64;
        state = mix(state.rotate_left(9).wrapping_mul(0x9E3779B97F4A7C15));
    }

    state
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

    #[test]
    fn test_summarize_work_is_deterministic() {
        let params = PowParams {
            matrix_size: 4,
            rounds: 2,
        };
        let seed = Seed { value: 1001 };
        let nonce = 9;

        let matrix = sky98_pow(seed, nonce, &params);
        let s1 = summarize_work(&matrix);
        let s2 = summarize_work(&matrix);

        assert_eq!(s1, s2);
    }

    #[test]
    fn test_evaluate_work_changes_with_nonce() {
        let params = PowParams {
            matrix_size: 4,
            rounds: 2,
        };
        let seed = Seed { value: 2024 };

        let (_, a) = evaluate_work(seed, 1, &params);
        let (_, b) = evaluate_work(seed, 2, &params);

        assert_ne!(a.commitment, b.commitment);
    }

    #[test]
    fn test_round_nonce_seed_changes_with_nonce() {
        let seed = Seed { value: 55 };

        assert_ne!(seed.round_nonce_seed(1, 10), seed.round_nonce_seed(1, 11));
    }
}
