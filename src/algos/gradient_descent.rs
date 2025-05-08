use crate::{algo::StepControlFlow, Algo};
use std::{convert::Infallible, ops::ControlFlow};

///
/// Simple Gradient Descent method.
///
/// See <https://en.wikipedia.org/wiki/Gradient_descent> for details on the algorithm.
///
/// The [`x`](Self::x) returned is a vector.
///
/// # Hyper-parameters
///
/// The only hyperparameter is the learning rate. Typically a value of 0.01 works well. Too high
/// a learning rate can cause the algorithm to diverge or overshoot and oscillate around the solution,
/// while too low a learning rate can cause the algorithm to converge too slowly.
///
/// The learning rate can be adaptively changed during the optimization process. For example,
/// you can decrease the learning rate as the optimization progresses.
//
/// # Example
///
/// ```rust  
/// use stepwise::{assert_approx_eq, problems::sphere_grad};
/// use stepwise::{algos::GradientDescent, Driver, fixed_iters};
///
/// /// n-sphere gradient
/// let sphere_grad = |x: &[f64]| x.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
///
/// let learning_rate = 0.1;
/// let initial_estimate = vec![5.5, 6.5];
///
/// let gd = GradientDescent::new(learning_rate, initial_estimate, sphere_grad);
/// let (solved, _step) = fixed_iters(gd, 500)
///     .solve()
///     .expect("failed to solve");
///
/// assert_approx_eq!(solved.x(), [0.0, 0.0].as_slice());
/// ```
///
/// # Example 
/// The learning rate is adaptively changed during the optimization process.
///
/// ```rust
/// # use stepwise::{
/// #    assert_approx_eq, assert_approx_ne, problems::sphere_grad,
/// #    algos::GradientDescent, Driver, fixed_iters,
/// # };
/// # fn main() {
/// # let x0 = vec![5.55, 5.55];
/// let gd = GradientDescent::new(0.1, x0, sphere_grad);
///
/// let (solved, _step) = fixed_iters(gd, 200)
///     .on_step(|algo, _step| algo.learning_rate *= 0.99 )
///     .solve()
///     .expect("failed to solve");
///
/// # let x = solved.x();
/// # assert_approx_eq!([0.0, 0.0].as_slice(), &x, 0.01);
/// # }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct GradientDescent<G> {
    pub gradient_fn: G,
    pub gradient: Vec<f64>,
    pub x: Vec<f64>,
    pub learning_rate: f64,
}

impl<G> GradientDescent<G>
where
    G: FnMut(&[f64]) -> Vec<f64>,
{
    /// Create a new `GradientDescent` algo with gradient_fn and initial guess x0.
    /// The learning rate can be changed during the optimization process.
    pub fn new(learning_rate: f64, x0: Vec<f64>, gradient_fn: G) -> Self {
        Self {
            gradient_fn,
            gradient: vec![0.0; x0.len()],
            x: x0,
            learning_rate,
        }
    }

    pub fn x(&self) -> &[f64] {
        &self.x
    }

    pub fn update_gradient(&mut self) {
        self.gradient = (self.gradient_fn)(&self.x);
        #[allow(clippy::needless_range_loop)]
        for i in 0..self.x.len() {
            self.x[i] -= self.learning_rate * self.gradient[i];
        }
    }
}

/// Implement the `Algo` trait for `GradientDescent`.
impl<G> Algo for GradientDescent<G>
where
    G: FnMut(&[f64]) -> Vec<f64>,
{
    type Error = Infallible;

    fn step(&mut self) -> StepControlFlow<Self::Error> {
        self.update_gradient();
        // allow stepwise Driver to decide when to cease iteration
        ControlFlow::Continue(Ok(()))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;

    use super::*;
    use crate::{assert_approx_eq, fixed_iters, problems::sphere_grad, Driver};

    #[test]
    fn test_doc_gradient_descent() {
        let sphere_grad = |x: &[f64]| x.iter().map(|x| 2.0 * x).collect::<Vec<_>>();

        let learning_rate = 0.1;
        let initial_estimate = vec![5.55, 5.55];
        let gd = GradientDescent::new(learning_rate, initial_estimate, sphere_grad);
        let (solved, _step) = fixed_iters(gd, 500).solve().expect("failed to solve");
        assert_approx_eq!(solved.x(), &[0.0, 0.0]);
    }

    #[test]
    fn gradient_descent_core() -> Result<(), Box<dyn Error>> {
        let gd = GradientDescent::new(0.01, vec![5.0, 5.0], sphere_grad);
        let driver = fixed_iters(gd, 1000);
        let (solved, _step) = driver
            .on_step(|v, s| {
                if s.iteration() % 100 == 0 {
                    println!("{s:?} x: {:.9?}", v.x())
                }
            })
            .solve()?;

        let x = solved.x();
        assert_approx_eq!(x, &[0.0, 0.0], 1e-5);
        Ok(())
    }
}
