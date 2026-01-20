use crate::matrix::{Matrix, Scalar};
use crate::sigma::sigma;

/// Verify a single matrix multiplication cell:
///
/// Checks:
///   C[i, j] == σ( sum_k A[i, k] * B[k, j] )
///
/// This is O(n) work and can be used as a probabilistic spot-check.
///
/// Returns true if the cell is valid.
pub fn verify_mul_cell(
    a_prev: &Matrix,
    b_prev: &Matrix,
    c_claimed: &Matrix,
    i: usize,
    j: usize,
) -> bool {
    assert!(a_prev.n == b_prev.n);
    assert!(a_prev.n == c_claimed.n);

    let n = a_prev.n;

    let mut acc: Scalar = 0;
    for k in 0..n {
        acc = acc.wrapping_add(
            a_prev.get(i, k).wrapping_mul(b_prev.get(k, j)),
        );
    }

    let expected = sigma(acc);
    let claimed = c_claimed.get(i, j);

    expected == claimed
}

/// Verify multiple randomly selected cells
///
/// This is the main verifier entry point.
///
/// - `checks` controls security vs cost
/// - Each check costs O(n)
///
/// If any check fails, the proof is invalid.
pub fn verify_random_cells(
    a_prev: &Matrix,
    b_prev: &Matrix,
    c_claimed: &Matrix,
    seed: u64,
    checks: usize,
) -> bool {
    let n = a_prev.n;
    let mut state = seed;

    for _ in 0..checks {
        state = mix(state);
        let i = (state as usize) % n;

        state = mix(state);
        let j = (state as usize) % n;

        if !verify_mul_cell(a_prev, b_prev, c_claimed, i, j) {
            return false;
        }
    }

    true
}

/// Deterministic mixing function for verifier randomness
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
    use crate::matrix::Matrix;

    #[test]
    fn test_verify_valid_cell() {
        let a = Matrix::from_vec(2, vec![
            1, 2,
            3, 4,
        ]);

        let b = Matrix::from_vec(2, vec![
            5, 6,
            7, 8,
        ]);

        let mut c = a.mul(&b);
        c.map_inplace(sigma);

        assert!(verify_mul_cell(&a, &b, &c, 0, 0));
        assert!(verify_mul_cell(&a, &b, &c, 1, 1));
    }

    #[test]
    fn test_verify_detects_cheat() {
        let a = Matrix::from_vec(2, vec![
            1, 2,
            3, 4,
        ]);

        let b = Matrix::from_vec(2, vec![
            5, 6,
            7, 8,
        ]);

        let mut c = a.mul(&b);
        c.map_inplace(sigma);

        // Corrupt one cell
        c.set(0, 1, c.get(0, 1).wrapping_add(1));

        assert!(!verify_mul_cell(&a, &b, &c, 0, 1));
    }

    #[test]
    fn test_random_verification() {
        let n = 8;

        let a = Matrix::zeros(n);
        let b = Matrix::zeros(n);

        let mut c = a.mul(&b);
        c.map_inplace(sigma);

        // Zero matrices are valid
        assert!(verify_random_cells(&a, &b, &c, 123, 10));
    }
}
