//! Some very basic linear albegra applicable to Rust slices, and hence can be used standalone, or with other linear
//! algebrara libraries offering an as_slice type of conversion
//! - dot product
//! - norms, l2/euclidean, l-inf or max_norms, distance,
//! - component-wise addition and subtraction
//! - formatting / display
//!
use std::fmt::Display;

#[doc(hidden)]
pub trait Tolerance: Sized {
    fn tol_eq(self, other: Self, tol: f64) -> Option<bool>;
}

impl Tolerance for f64 {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Some((self - b).abs() < tol)
    }
}

impl Tolerance for f32 {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Some(f64::from((self - b).abs()) < tol)
    }
}

impl Tolerance for &[f64] {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        if self.len() != b.len() {
            return None;
        }
        Some(
            self.iter()
                .zip(b.iter())
                .all(|(a, b)| a.tol_eq(*b, tol) == Some(true)),
        )
    }
}

impl Tolerance for &[f32] {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        if self.len() != b.len() {
            return None;
        }
        Some(
            self.iter()
                .zip(b.iter())
                .all(|(a, b)| a.tol_eq(*b, tol) == Some(true)),
        )
    }
}

impl Tolerance for Vec<f64> {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Tolerance::tol_eq(self.as_slice(), b.as_slice(), tol)
    }
}

impl Tolerance for &Vec<f64> {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Tolerance::tol_eq(self.as_slice(), b.as_slice(), tol)
    }
}

impl Tolerance for Vec<f32> {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Tolerance::tol_eq(self.as_slice(), b.as_slice(), tol)
    }
}

impl Tolerance for &Vec<f32> {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Tolerance::tol_eq(self.as_slice(), b.as_slice(), tol)
    }
}

impl<const N: usize> Tolerance for [f64; N] {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Tolerance::tol_eq(self.as_slice(), b.as_slice(), tol)
    }
}

impl<const N: usize> Tolerance for [f32; N] {
    fn tol_eq(self, b: Self, tol: f64) -> Option<bool> {
        Tolerance::tol_eq(self.as_slice(), b.as_slice(), tol)
    }
}

/// Assert that two values are approximately equal.
///
/// Arguments can be f32, f64, `&[f32]`, `&[f64]`, `Vec<f32>`, or `Vec<f64>`.
/// The default tolerance is 1e-7, but can be overridden by providing a third argument.
///
/// Panics if the values are not approximately equal
///
///
/// ```
/// use stepwise::assert_approx_eq;
/// assert_approx_eq!(1.0, 1.0 + 1e-8);
/// assert_approx_eq!(1.0, 1.00001, 1e-4);
/// assert_approx_eq!(vec![1.0, 2.0], vec![1.0 + 1e-8, 2.0 - 1e-8]);
/// assert_approx_eq!([1.0, 2.0].as_slice(), [1.0, 2.0001].as_slice(), 1e-3);
/// ```
///
/// ```rust,should_panic
/// # use stepwise::assert_approx_eq;
/// // This code is expected to panic
/// assert_approx_eq!(1.0, 1.000001);
/// ```
///
#[macro_export]
macro_rules! assert_approx_eq {
    ($a:expr, $b:expr) => {
        assert_approx_eq!($a, $b, 1e-7);
    };
    ($a:expr, $b:expr, $tol:expr) => {
        let res = $crate::Tolerance::tol_eq($a, $b, $tol);
        assert!(!res.is_none(), "lengths differ, {:?} != {:?}", $a, $b,);
        assert!(
            res == Some(true),
            "{:?} != {:?} (approx, tol = {:?})",
            $a,
            $b,
            $tol
        );
    };
}

/// Assert that two values are comparable, but approximately not-equal.
///
/// Note that if you compare two slices of different lengths,
/// they will be neither approximately equal, nor approximately not-equal.
///
#[macro_export]
macro_rules! assert_approx_ne {
    ($a:expr, $b:expr) => {
        assert_approx_ne!($a, $b, 1e-7);
    };
    ($a:expr, $b:expr, $tol:expr) => {
        let res = $crate::Tolerance::tol_eq($a, $b, $tol);
        assert!(!res.is_none(), "lengths differ, {:?} != {:?}", $a, $b,);
        assert!(
            res == Some(false),
            "expected to differ, {:?} == {:?} (approx, tol = {:?})",
            $a,
            $b,
            $tol
        );
    };
}

/// Basic linear algebra functions.
#[allow(dead_code)]
pub trait VectorExt {
    fn zero(n: usize) -> Vec<f64>;
    fn nan(n: usize) -> Vec<f64>;
    fn is_finite(&self) -> bool;
    fn dot(&self, a: &Self) -> f64;
    fn norm_max(&self) -> f64;
    fn norm_l2(&self) -> f64;
    fn add_vec(&self, b: &Self) -> Vec<f64>;
    fn sub_vec(&self, b: &Self) -> Vec<f64>;
    fn dist_l2(&self, b: &Self) -> f64;
    fn dist_max(&self, b: &Self) -> f64;
    fn display(&self) -> impl Display;
}

#[allow(dead_code)]
impl VectorExt for [f64] {
    fn zero(n: usize) -> Vec<f64> {
        vec![0.0; n]
    }

    fn nan(n: usize) -> Vec<f64> {
        vec![f64::NAN; n]
    }

    fn is_finite(&self) -> bool {
        self.iter().copied().all(f64::is_finite)
    }

    fn dot(&self, b: &Self) -> f64 {
        self.iter().zip(b.iter()).map(|(a, b)| a * b).sum()
    }

    fn norm_max(&self) -> f64 {
        self.iter().copied().map(f64::abs).fold(f64::NAN, f64::max)
    }

    fn norm_l2(&self) -> f64 {
        self.dot(self).sqrt()
    }

    fn dist_l2(&self, b: &Self) -> f64 {
        self.iter()
            .zip(b.iter())
            .map(|(a, b)| a - b)
            .map(|c| c * c)
            .sum::<f64>()
            .sqrt()
    }

    fn dist_max(&self, b: &Self) -> f64 {
        self.iter()
            .zip(b.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(f64::NAN, f64::max)
    }

    fn add_vec(&self, b: &Self) -> Vec<f64> {
        self.iter().zip(b.iter()).map(|(a, b)| a + b).collect()
    }

    fn sub_vec(&self, b: &Self) -> Vec<f64> {
        self.iter().zip(b.iter()).map(|(a, b)| a - b).collect()
    }

    fn display(&self) -> impl Display {
        struct Displayable<'a>(&'a [f64]);

        impl Display for Displayable<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "[")?;
                let mut iter = self.0.iter();
                if let Some(xi) = iter.next() {
                    Display::fmt(xi, f)?;
                    for xi in iter {
                        write!(f, ", ")?;
                        Display::fmt(xi, f)?;
                    }
                }
                write!(f, "]")
            }
        }
        Displayable(self)
    }
}

/// Calculate the gradient of a function using central differences.
///
pub fn central_difference_gradient<F>(x: &[f64], f: F) -> Vec<f64>
where
    F: Fn(&[f64]) -> f64,
{
    let eps = 1e-7;
    let mut grad = vec![0.0; x.len()];
    #[allow(clippy::needless_range_loop)]
    for j in 0..x.len() {
        let f_p = f(&x
            .iter()
            .enumerate()
            .map(|(i, &xi)| if i == j { xi + eps } else { xi })
            .collect::<Vec<_>>());
        let f_m = f(&x
            .iter()
            .enumerate()
            .map(|(i, &xi)| if i == j { xi - eps } else { xi })
            .collect::<Vec<_>>());
        grad[j] = (f_p - f_m) / (2.0 * eps);
    }
    grad
}

// pub struct ExponentialMovingAvg {
//     window_size: usize,
//     avg: Mutex<Option<f64>>,
// }

// /// The value for the moving average.
// /// Formatting parameters appropriate for f64 will be used, see [std::fmt].
// impl Display for ExponentialMovingAvg {
//     fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
//         Display::fmt(&self.get(), f)
//     }
// }

// impl ExponentialMovingAvg {
//     pub fn with_window(window_size: usize) -> Self {
//         Self {
//             window_size,
//             avg: Mutex::new(None),
//         }
//     }

//     /// include a value in the average
//     pub fn collect(&self, value: f64) {
//         let mut avg = self.avg.lock().unwrap();
//         match *avg {
//             None => *avg = Some(value),

//             // Calculate new EMA value
//             // EMA = previous_value * (1 - alpha) + new_value * alpha
//             Some(current_value) => {
//                 *avg = Some(current_value * (1.0 - self.alpha()) + value * self.alpha())
//             }
//         }
//     }

//     /// include an optional value in the average, ignoring the value if it is [Option::None]
//     pub fn collect_optional(&self, value: Option<f64>) {
//         let Some(value) = value else { return };
//         self.collect(value);
//     }

//     pub fn alpha(&self) -> f64 {
//         2.0 / (1.0 + self.window_size as f64)
//     }

//     /// The current exponential moving average
//     /// returns [f64::NAN] if no values have been collected yet
//     pub fn get(&self) -> f64 {
//         let avg = self.avg.lock().unwrap();
//         avg.unwrap_or(f64::NAN)
//     }

//     /// Reset the EMA to uninitialized step
//     pub fn reset(&self) {
//         let mut avg = self.avg.lock().unwrap();
//         *avg = None;
//     }
// }

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    fn assert_approx_eq_core() {
        assert_approx_eq!(1.0, 1.0);
        assert_approx_eq!(1.0, 1.0001, 1e-2);
        assert_approx_eq!(1.0, 1.0 + 1e-8);
        assert_approx_eq!(1.0, 1.0 - 1e-8);
        assert_approx_eq!([1.0, 2.0].as_slice(), [1.0, 2.0001].as_slice(), 1e-3);
        assert_approx_eq!(vec![1.0, 2.0], vec![1.0 + 1e-8, 2.000000001]);
    }

    #[test]
    #[should_panic(expected = "1.0 != 1.000001 (approx, tol = 1e-7)")]
    fn assert_approx_eq_failure() {
        assert_approx_eq!(1.0, 1.000001);
    }

    #[test]
    #[should_panic(expected = "[1.0, 2.0] != [1.000001, 2.000001] (approx, tol = 1e-7)")]
    fn assert_approx_eq_slice_failure() {
        assert_approx_eq!([1.0, 2.0].as_slice(), [1.000001, 2.000001].as_slice());
    }

    #[test]
    #[should_panic(expected = "lengths differ, [1.0, 2.0, 3.0] != [1.0, 2.0]")]
    fn assert_len_eq_slice_failure() {
        assert_approx_eq!([1.0, 2.0, 3.0].as_slice(), [1.0, 2.0].as_slice());
    }

    #[test]
    fn vector_ext_core() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let zero = vec![0.0, 0.0, 0.0];

        assert_approx_eq!(a.dot(&b), 32.0);
        assert_approx_eq!(a.norm_max(), 3.0);
        assert_approx_eq!(a.norm_l2(), f64::sqrt(1.0 + 4.0 + 9.0));
        assert_approx_eq!(a.dist_l2(&b), f64::sqrt(9.0 + 9.0 + 9.0));
        assert_approx_eq!(a.dist_max(&b), 3.0);
        assert_approx_eq!(a.add_vec(&b), vec![5.0, 7.0, 9.0]);
        assert_approx_eq!(a.sub_vec(&b), vec![-3.0, -3.0, -3.0]);
        assert_approx_eq!(a.sub_vec(&a).dist_max(&zero), 0.0);
    }

    #[test]
    fn slice() {
        let zero = [0.0, 0.0, 0.0];
        let a = [2.0, 3.0, 4.0];
        let b = [-11.0, 5.0, 6.0];
        assert_approx_eq!(a.dot(&b), f64::from(-22 + 15 + 24));
        assert_approx_eq!(a.norm_max(), f64::from(4));
        assert_approx_eq!(b.norm_max(), f64::from(11));
        assert_approx_eq!(a.norm_l2(), f64::from(4 + 9 + 16).sqrt());
        assert_approx_eq!(b.norm_l2(), f64::from(121 + 25 + 36).sqrt());
        assert_approx_eq!(b.dist_l2(&a), a.dist_l2(&b));
        assert_approx_eq!(b.dist_l2(&a), f64::from(169 + 4 + 4).sqrt());
        assert_approx_eq!(b.dist_max(&a), a.dist_max(&b));
        assert_approx_eq!(b.dist_max(&a), f64::from(13));
        assert_approx_eq!(a.sub_vec(&a).dist_max(&zero), 0.0);
        assert_approx_eq!(zero.add_vec(&a).dist_max(&a), 0.0);
        assert_approx_eq!(a.add_vec(&b).dist_max(&[-9.0, 8.0, 10.0]), 0.0);
    }
}

// pub trait VectorExt {
//     fn dot(&self, a: &Self) -> f64;
//     fn norm(&self) -> f64;

//     #[must_use]
//     fn map(&self, f: impl FnMut(f64) -> f64) -> Self;

//     fn map_inplace(&mut self, f: impl FnMut(f64) -> f64);

//     /// <https://en.wikipedia.org/wiki/Cosine_similarity>
//     fn cosine_similarity(a: &Self, b: &Self) -> f64;
//     fn vec_div(a: f64, b: &Self) -> Self;
//     fn component_add(&self, b: &Self) -> Self;
//     fn component_sub(&self, b: &Self) -> Self;
//     fn component_mul(&self, b: &Self) -> Self;
//     fn add(&self, a: f64) -> Self;
//     fn add_assign(&mut self, a: f64);

//     fn sub(&self, a: f64) -> Self;
//     fn mul(&self, a: f64) -> Self;
//     fn div(&self, a: f64) -> Self;

//     /// random vector in of -1's and 1's
//     fn rademacher<R: Rng + ?Sized>(nrows: usize, rng: &mut R) -> Self;

//     /// gradient estimate
//     fn grad<R, F>(&self, f: F, rng: &mut R, delta: f64) -> Self
//     where
//         R: Rng + ?Sized,
//         F: FnMut(&Self) -> f64;

//     fn clamp(&self, l: f64, u: f64) -> Self;
//     fn powi(&self, i: i32) -> Self;
//     fn sqrt(&self) -> Self;
// }

// impl VectorExt for DVector<f64> {
//     fn dot(&self, a: &Self) -> f64 {
//         self.dot(a)
//     }

//     fn norm(&self) -> f64 {
//         self.norm()
//     }

//     fn map(&self, f: impl FnMut(f64) -> f64) -> Self {
//         self.map(f)
//     }

//     fn map_inplace(&mut self, mut f: impl FnMut(f64) -> f64) {
//         self.apply(|x| *x = f(*x));
//     }

//     fn cosine_similarity(a: &Self, b: &Self) -> f64 {
//         a.dot(b) / (a.norm() * b.norm())
//     }

//     fn vec_div(a: f64, b: &Self) -> Self {
//         a * b.map(|e| 1. / e)
//     }

//     fn component_add(&self, b: &Self) -> Self {
//         std::ops::Add::add(self, b)
//     }

//     fn component_sub(&self, b: &Self) -> Self {
//         std::ops::Sub::sub(self, b)
//     }

//     fn component_mul(&self, b: &Self) -> Self {
//         Self::component_mul(self, b)
//     }

//     fn add(&self, a: f64) -> Self {
//         self.map(|s| s + a)
//     }

//     fn add_assign(&mut self, a: f64) {
//         self.map_inplace(|s| s + a);
//     }

//     fn sub(&self, a: f64) -> Self {
//         self.map(|s| s - a)
//     }

//     fn mul(&self, a: f64) -> Self {
//         self.map(|s| a * s)
//     }

//     fn div(&self, a: f64) -> Self {
//         self.map(|s| s / a)
//     }

//     fn rademacher<R: Rng + ?Sized>(nrows: usize, rng: &mut R) -> Self {
//         Self::from_fn(nrows, |_, _| if rng.gen::<bool>() { 1.0 } else { -1.0 })
//     }

//     fn grad<R, F>(&self, mut f: F, rng: &mut R, delta: f64) -> Self
//     where
//         R: Rng + ?Sized,
//         F: FnMut(&Self) -> f64,
//     {
//         let dx = Self::rademacher(self.nrows(), rng) * delta;
//         let df = f(&(self + &dx)) - f(&(self - &dx));
//         0.5 * Self::vec_div(df, &dx)
//     }

//     fn clamp(&self, l: f64, u: f64) -> Self {
//         self.map(|elt| elt.clamp(l, u))
//     }

//     fn powi(&self, i: i32) -> Self {
//         self.map(|elt| elt.powi(i))
//     }

//     fn sqrt(&self) -> Self {
//         self.map(f64::sqrt)
//     }
// }

// impl VectorExt for Vec<f64> {
//     fn dot(&self, b: &Self) -> f64 {
//         self.iter().zip(b).map(|(a, b)| a * b).sum()
//     }

//     fn norm(&self) -> f64 {
//         self.iter().map(|a| a * a).sum::<f64>().sqrt()
//     }

//     fn map(&self, mut f: impl FnMut(f64) -> f64) -> Self {
//         self.iter().map(|x| f(*x)).collect::<Self>()
//     }

//     fn map_inplace(&mut self, mut f: impl FnMut(f64) -> f64) {
//         self.iter_mut().on_step(|x| *x = f(*x));
//     }

//     fn cosine_similarity(a: &Self, b: &Self) -> f64 {
//         a.dot(b) / (a.norm() * b.norm())
//     }

//     fn vec_div(a: f64, b: &Self) -> Self {
//         b.map(|e| 1. / e).mul(a)
//     }

//     fn component_add(&self, b: &Self) -> Self {
//         self.iter().zip(b).map(|x| x.0 + x.1).collect()
//     }

//     fn component_sub(&self, b: &Self) -> Self {
//         self.iter().zip(b).map(|x| x.0 - x.1).collect()
//     }

//     fn component_mul(&self, b: &Self) -> Self {
//         self.iter().zip(b).map(|x| x.0 * x.1).collect()
//     }

//     fn add(&self, a: f64) -> Self {
//         self.map(|s| s + a)
//     }

//     fn add_assign(&mut self, a: f64) {
//         self.map_inplace(|s| s + a);
//     }

//     fn sub(&self, a: f64) -> Self {
//         self.map(|s| s - a)
//     }

//     fn mul(&self, a: f64) -> Self {
//         self.map(|s| a * s)
//     }

//     fn div(&self, a: f64) -> Self {
//         self.map(|s| s / a)
//     }

//     fn rademacher<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Self {
//         (0..n).map(|_i| if rng.gen::<bool>() { 1.0 } else { -1.0 }).collect()
//     }

//     fn grad<R, F>(&self, mut f: F, rng: &mut R, delta: f64) -> Self
//     where
//         R: Rng + ?Sized,
//         F: FnMut(&Self) -> f64,
//     {
//         let dx = Self::rademacher(self.len(), rng).mul(delta);
//         let df = f(&(self.component_add(&dx))) - f(&(self.component_sub(&dx)));
//         Self::vec_div(df, &dx).div(2.0)
//     }

//     fn clamp(&self, l: f64, u: f64) -> Self {
//         self.map(|elt| elt.clamp(l, u))
//     }

//     fn powi(&self, i: i32) -> Self {
//         self.map(|elt| elt.powi(i))
//     }

//     fn sqrt(&self) -> Self {
//         self.map(f64::sqrt)
//     }
// }
