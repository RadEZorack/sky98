use std::fmt;

pub type Scalar = u32;

/// A dense square matrix stored in row-major order.
///
/// Indexing:
/// element (i, j) is stored at data[i * n + j]
#[derive(Clone, PartialEq, Eq)]
pub struct Matrix {
    pub n: usize,
    pub data: Vec<Scalar>,
}

impl Matrix {
    /// Create a new n x n zero matrix
    pub fn zeros(n: usize) -> Self {
        Matrix {
            n,
            data: vec![0; n * n],
        }
    }

    /// Create a matrix from raw data
    ///
    /// # Panics
    /// Panics if data.len() != n * n
    #[allow(dead_code)]
    pub fn from_vec(n: usize, data: Vec<Scalar>) -> Self {
        assert!(
            data.len() == n * n,
            "Matrix data length must be n * n"
        );
        Matrix { n, data }
    }

    /// Get element at (i, j)
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> Scalar {
        self.data[i * self.n + j]
    }

    /// Set element at (i, j)
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: Scalar) {
        self.data[i * self.n + j] = value;
    }

    /// Apply a function element-wise in place
    pub fn map_inplace<F>(&mut self, mut f: F)
    where
        F: FnMut(Scalar) -> Scalar,
    {
        for v in self.data.iter_mut() {
            *v = f(*v);
        }
    }

    /// Deterministic matrix multiplication (naive O(n^3))
    ///
    /// Uses wrapping arithmetic to enforce modulo 2^32 behavior.
    ///
    /// # Panics
    /// Panics if matrix dimensions do not match.
    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert!(self.n == other.n, "Matrix size mismatch");

        let n = self.n;
        let mut out = Matrix::zeros(n);

        for i in 0..n {
            for j in 0..n {
                let mut acc: Scalar = 0;
                for k in 0..n {
                    acc = acc.wrapping_add(
                        self.get(i, k).wrapping_mul(other.get(k, j)),
                    );
                }
                out.set(i, j, acc);
            }
        }

        out
    }

    /// Deterministic permutation used between rounds
    ///
    /// This performs a fixed cyclic shift of columns.
    /// (Simple, cheap, and breaks symmetry.)
    pub fn permute(&self) -> Matrix {
        let n = self.n;
        let mut out = Matrix::zeros(n);

        for i in 0..n {
            for j in 0..n {
                let src_j = (j + 1) % n;
                out.set(i, j, self.get(i, src_j));
            }
        }

        out
    }
}

/// Optional: Pretty-print small matrices for debugging
impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.n > 8 {
            return write!(f, "Matrix {{ n: {}, data: [...] }}", self.n);
        }

        writeln!(f, "Matrix {{")?;
        for i in 0..self.n {
            write!(f, "  [")?;
            for j in 0..self.n {
                if j > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{:08x}", self.get(i, j))?;
            }
            writeln!(f, "]")?;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_matrix() {
        let m = Matrix::zeros(4);
        assert_eq!(m.data.iter().all(|&x| x == 0), true);
    }

    #[test]
    fn test_get_set() {
        let mut m = Matrix::zeros(2);
        m.set(1, 0, 42);
        assert_eq!(m.get(1, 0), 42);
    }

    #[test]
    fn test_mul_identity_like() {
        let a = Matrix::from_vec(2, vec![
            1, 2,
            3, 4,
        ]);

        let b = Matrix::from_vec(2, vec![
            5, 6,
            7, 8,
        ]);

        let c = a.mul(&b);

        // Manual multiplication:
        // [1*5 + 2*7, 1*6 + 2*8]
        // [3*5 + 4*7, 3*6 + 4*8]
        assert_eq!(c.get(0, 0), 19);
        assert_eq!(c.get(0, 1), 22);
        assert_eq!(c.get(1, 0), 43);
        assert_eq!(c.get(1, 1), 50);
    }

    #[test]
    fn test_permute() {
        let m = Matrix::from_vec(3, vec![
            1, 2, 3,
            4, 5, 6,
            7, 8, 9,
        ]);

        let p = m.permute();

        assert_eq!(p.get(0, 0), 2);
        assert_eq!(p.get(0, 1), 3);
        assert_eq!(p.get(0, 2), 1);
    }
}
