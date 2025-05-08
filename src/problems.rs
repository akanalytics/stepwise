//! Sample problems for testing optimization, along with their gradient functions
//!
//!
//! See <https://en.wikipedia.org/wiki/Test_functions_for_optimization>
//!
//! The sigmoid function can be used to bound the output range to \[0,1\] whilst
//! preserving minima.
//!
//!
use std::f64::consts::PI;

use crate::rng::TinyRng;

/// σ(x) = 1 / [1+e^(-x)]
///
/// Sigmoid function is useful in that it transforms [-inf,inf] into \[0, 1\]
///
/// Being monotonic, it preserves maximums and minimums
///
/// σ(-x) = 1 - σ(x)
pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

/// Sigmoid function gradient: ∇σ(x) = σ'(x) = σ(x) * (1 - σ(x))
///
pub fn sigmoid_grad(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

/// n-sphere centered at origin: f(x) = x₁² + x₂² + ... + xₙ²
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum()
}

/// n-sphere gradient: ∇f(x) = [2x₁, 2x₂, ..., 2xₙ]
pub fn sphere_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|x| 2.0 * x).collect()
}

/// n-dimensional absolute value: f(x) = |x₁| + |x₂| + ... + |xₙ|
pub fn abs(x: &[f64]) -> f64 {
    x.iter().map(|x| x.abs()).sum()
}

/// n-dim absolute value gradient: ∇f(x) = sgn(x) = [signum(x₁), signum(x₂), ..., signum(xₙ)]
///
/// <https://en.wikipedia.org/wiki/Sign_function>
pub fn abs_grad(x: &[f64]) -> Vec<f64> {
    x.iter().map(|x| x.signum()).collect()
}

/// rosenbrock: f(x) = sum(i=0,i<N/2) (x{2i} -a)^2 + b(x{2i+1} - [x{2i}^2)^2
///
/// Rosenbrock's banana function. Typically a=1, b=100
///
/// Defined only for even N.
///
/// Global minimum at origin [0,0, ..., 0]
///
/// <https://en.wikipedia.org/wiki/Rosenbrock_function>
///
/// Sum of uncoupled 2D Rosenbrock problem variant.
pub fn rosenbrock(x: &[f64], (a, b): (f64, f64)) -> f64 {
    assert!(x.len() % 2 == 0, "Rosenbrock fn requires even dimensions");
    let n = x.len() / 2;
    (0..n)
        .map(|i| {
            let xi = x[2 * i];
            let yi = x[2 * i + 1];
            (xi - a).powi(2) + b * (yi - xi.powi(2)).powi(2)
        })
        .sum()
}

/// rosenbrock gradient
///
/// For each i=0..i<N/2:
/// * ∇f(x){2i}   = 2 * (x{2i} - a) - 4 * b * x{2i} * (x{2i+1} - x{2i}^2)
/// * ∇f(x){2i+1} = 2 * b * (x{2i+1} - x{2i}^2)
pub fn rosenbrock_grad(x: &[f64], (a, b): (f64, f64)) -> Vec<f64> {
    assert!(x.len() % 2 == 0, "Rosenbrock grad requires even dimensions");
    // ∇f(x) = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]
    let n = x.len() / 2;
    (0..n)
        .flat_map(move |i| {
            let xi = x[2 * i];
            let yi = x[2 * i + 1];
            vec![
                2.0 * (xi - a) - 4.0 * b * xi * (yi - xi.powi(2)),
                2.0 * b * (yi - xi.powi(2)),
            ]
        })
        .collect()
}

/// Generates random noise in the f64 range [-1, +1]
///
/// For the same input, the same noise value is generated.
///
/// # Algorithm
/// * uses binary bits of `x` to seed a `PCG-XSH-RR 64`
/// * the top 53 bits of the generated random u64 is scaled and translated to [-1, +1]
///
pub fn uniform_noise(x: f64) -> f64 {
    let seed = x.to_bits();
    // https://www.reddit.com/r/programming/comments/jijhwa/generating_random_floatingpoint_numbers_by/
    let random_u64 = TinyRng::new(seed).next_u64();
    let random_f64 = (random_u64 >> 11) as f64 * (1.0 / (1u64 << 53) as f64); // range [0, 1)
    random_f64 * 2.0 - 1.0 // range [-1, 1)
}

/// Normal with mean = 0, and stand deviation 1
/// # Algorithm
/// * uses Box-Muller transform of `uniform_noise(x)` and `uniform_noise(x+1)`
pub fn gaussian_noise(x: f64) -> f64 {
    let u1: f64 = uniform_noise(x);
    let u2: f64 = uniform_noise(x + 1.0);

    // Box-Muller Transform
    (-(u1 * u1).ln() / 2.0).sqrt() * (2.0 * PI * u2).cos()
}

#[cfg(test)]
mod tests {
    use crate::{assert_approx_eq, math::central_difference_gradient};

    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_rosenbrock_grad() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.5, 6.5];
        let (a, b) = (1.0, 100.0);
        let grad = rosenbrock_grad(&x, (a, b));
        let grad_fd = central_difference_gradient(&x, |x| rosenbrock(x, (1.0, 100.0)));

        assert_approx_eq!(grad.as_slice(), grad_fd.as_slice(), 1e-4);
    }
    #[test]
    fn test_noise() {
        let x = 42.0;
        let noise1 = uniform_noise(x);
        let noise2 = uniform_noise(x);
        assert_approx_eq!(noise1, noise2, 1e-10); // should be the same for the same input

        const N: u32 = 100_000;
        let mean = (0..N).map(|x| uniform_noise(x as f64)).sum::<f64>() / N as f64;
        assert_approx_eq!(mean, 0.0, 0.001);
    }

    #[test]
    #[ignore]
    fn write_noise_file() {
        std::fs::write(
            "/tmp/stepwise-random-numbers.csv",
            (0..100_000)
                .map(|x| {
                    format!(
                        "{u} {g}\n",
                        u = uniform_noise(x as f64),
                        g = gaussian_noise(x as f64)
                    )
                })
                .collect::<Vec<String>>()
                .concat(),
        )
        .expect("Failed to write to file /tmp/stepwise-random-numbers.csv");
        // cargo nextest write_noise --run-ignored all
        // gnuplot -e "plot '/tmp/stepwise-random-numbers.csv' using 0:1 with points title 'uniform'"
        // gnuplot -e "plot '/tmp/stepwise-random-numbers.csv' using 0:2 with points title 'gaussian'"
        // gnuplot -e "bin_width=0.1; bin(x)=bin_width*floor(x/bin_width); plot '/tmp/stepwise-random-numbers.csv' using (bin(\$1)):(1.0) smooth freq with boxes title 'uniform'"
        // gnuplot -e "bin_width=0.1; bin(x)=bin_width*floor(x/bin_width); plot '/tmp/stepwise-random-numbers.csv' using (bin(\$2)):(1.0) smooth freq with boxes title 'gaussian'"
    }
}
