use crate::matrix::{Matrix, Scalar};
use crate::mask::Mask;
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

/// Verify a single full round output cell:
///
/// Checks:
///   C[i, j] == Mask(seed_round, i, j) * σ( sum_k A[i, k] * B[k, j] )
///
/// This matches the round transition in `sky98_pow`.
pub fn verify_round_cell(
    a_prev: &Matrix,
    b_prev: &Matrix,
    c_claimed: &Matrix,
    round_seed: u64,
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

    let masked_expected = if Mask::new(round_seed).keep_cell(n, i, j) {
        sigma(acc)
    } else {
        0
    };

    c_claimed.get(i, j) == masked_expected
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

        if !verify_round_cell(a_prev, b_prev, c_claimed, seed, i, j) {
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
    fn test_verify_round_cell_valid_with_mask() {
        let a = Matrix::from_vec(2, vec![
            1, 2,
            3, 4,
        ]);

        let b = Matrix::from_vec(2, vec![
            5, 6,
            7, 8,
        ]);

        let seed = 12345;
        let mask = Mask::new(seed);

        let mut c = a.mul(&b);
        c.map_inplace(sigma);
        mask.apply(&mut c);

        assert!(verify_round_cell(&a, &b, &c, seed, 0, 0));
        assert!(verify_round_cell(&a, &b, &c, seed, 1, 1));
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
        Mask::new(123).apply(&mut c);

        // Zero matrices remain valid after masking
        assert!(verify_random_cells(&a, &b, &c, 123, 10));
    }

    #[test]
    fn test_random_verification_detects_mask_cheat() {
        let a = Matrix::from_vec(3, vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]);
        let b = Matrix::from_vec(3, vec![
            9, 8, 7,
            6, 5, 4,
            3, 2, 1,
        ]);
        let seed = 77;

        let mut c = a.mul(&b);
        c.map_inplace(sigma);
        Mask::new(seed).apply(&mut c);

        for i in 0..c.n {
            for j in 0..c.n {
                if c.get(i, j) == 0 {
                    c.set(i, j, 1);
                    assert!(!verify_round_cell(&a, &b, &c, seed, i, j));
                    return;
                }
            }
        }

        panic!("expected at least one masked cell");
    }
}
