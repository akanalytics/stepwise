use std::{fmt::Display, sync::Mutex};

use rand::Rng;

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
    fn rademacher<R: Rng + ?Sized>(nrows: usize, rng: &mut R) -> Vec<f64>;
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

    fn rademacher<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Vec<f64> {
        (0..n)
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect()
    }

    fn display(&self) -> impl Display {
        struct Displayable<'a>(&'a [f64]);

        impl<'a> Display for Displayable<'a> {
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

pub struct ExponentialMovingAvg {
    window_size: usize,
    avg: Mutex<Option<f64>>,
}

/// The value for the moving average.
/// Formatting parameters appropriate for f64 will be used, see [std::fmt].
impl Display for ExponentialMovingAvg {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        Display::fmt(&self.get(), f)
    }
}

impl ExponentialMovingAvg {
    pub fn with_window(window_size: usize) -> Self {
        Self {
            window_size,
            avg: Mutex::new(None),
        }
    }

    /// include a value in the average
    pub fn collect(&self, value: f64) {
        let mut avg = self.avg.lock().unwrap();
        match *avg {
            None => *avg = Some(value),

            // Calculate new EMA value
            // EMA = previous_value * (1 - alpha) + new_value * alpha
            Some(current_value) => {
                *avg = Some(current_value * (1.0 - self.alpha()) + value * self.alpha())
            }
        }
    }

    /// include an optional value in the average, ignoring the value if it is [Option::None]
    pub fn collect_optional(&self, value: Option<f64>) {
        let Some(value) = value else { return };
        self.collect(value);
    }

    pub fn alpha(&self) -> f64 {
        2.0 / (1.0 + self.window_size as f64)
    }

    /// The current exponential moving average
    /// returns [f64::NAN] if no values have been collected yet
    pub fn get(&self) -> f64 {
        let avg = self.avg.lock().unwrap();
        avg.unwrap_or(f64::NAN)
    }

    /// Reset the EMA to uninitialized state
    pub fn reset(&self) {
        let mut avg = self.avg.lock().unwrap();
        *avg = None;
    }
}

#[cfg(test)]
pub mod tests {
    use rand::RngCore;
    use rand_core::impls;

    use super::*;


    /// Marsaglia multiply with carry random number generator
    pub struct MarsagliaRng(pub u32, pub u32);

    impl RngCore for MarsagliaRng {
        fn next_u32(&mut self) -> u32 {
            self.0 = 36969 * (self.0 & 0xffff) + (self.0 >> 16);
            self.1 = 18000 * (self.1 & 0xffff) + (self.1 >> 16);
            (self.0 << 16) + (self.1 & 0xffff)
        }

        fn next_u64(&mut self) -> u64 {
            impls::next_u64_via_u32(self)
        }

        fn fill_bytes(&mut self, dst: &mut [u8]) {
            impls::fill_bytes_via_next(self, dst);
        }
    }

    pub fn assert_approx_eq(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-7, "{a} != {b} (approximately)");
    }

    #[test]
    fn slice() {
        let zero = [0.0, 0.0, 0.0];
        let a = [2.0, 3.0, 4.0];
        let b = [-11.0, 5.0, 6.0];
        assert_approx_eq(a.dot(&b), f64::from(-22 + 15 + 24));
        assert_approx_eq(a.norm_max(), f64::from(4));
        assert_approx_eq(b.norm_max(), f64::from(11));
        assert_approx_eq(a.norm_l2(), f64::from(4 + 9 + 16).sqrt());
        assert_approx_eq(b.norm_l2(), f64::from(121 + 25 + 36).sqrt());
        assert_approx_eq(b.dist_l2(&a), a.dist_l2(&b));
        assert_approx_eq(b.dist_l2(&a), f64::from(169 + 4 + 4).sqrt());
        assert_approx_eq(b.dist_max(&a), a.dist_max(&b));
        assert_approx_eq(b.dist_max(&a), f64::from(13));
        assert_approx_eq(a.sub_vec(&a).dist_max(&zero), 0.0);
        assert_approx_eq(zero.add_vec(&a).dist_max(&a), 0.0);
        assert_approx_eq(a.add_vec(&b).dist_max(&[-9.0, 8.0, 10.0]), 0.0);
    }

    #[test]
    fn random() {
        let mut my_rand = MarsagliaRng(42, 43);
        let avg = (0..100_000)
            .map(|_| my_rand.random_range(0.0..1.0))
            // .for_each(|a| println!("{a}"))
            .sum::<f64>()
            / 100_000.0;
        assert!((avg - 0.5).abs() < 0.001, "average {avg}");

        let var = (0..100_000)
            .map(|_| my_rand.random_range(0.0..1_f64)) // .for_each(|x| println!("{x}"))
            .map(|r| (r - 0.5).powi(2))
            .sum::<f64>()
            / 100_000.0;
        assert!(
            (var - 1. / 12.0).abs() < 0.001,
            "variance {var} {}",
            1. / 12.
        );

        let mut my_rand = MarsagliaRng(42, 43);
        let vec = <[_]>::rademacher(5, &mut my_rand);
        println!("{vec:?}");
        assert_approx_eq(vec[0], 1.0);
        assert_approx_eq(vec[1], -1.0);
        assert_approx_eq(vec[2], -1.0);
        assert_approx_eq(vec[3], 1.0);
        assert_approx_eq(vec[4], -1.0);
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
//         self.iter_mut().for_each(|x| *x = f(*x));
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
