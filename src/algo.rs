use std::{error::Error, fmt::Debug, ops::ControlFlow};

/// Algorithms implement this trait to work with the library.
///
/// The algo has a `Step` method, similar to iterator's `next`.
/// It additionally has accessors to return the current solution or state (or the initial starting conditions
/// if called before the first step). By convention the current best solution would be called `x()`
/// for a scalar or vector solution.
///
/// The "problem" being solved and any initial starting conditions are part of the algo.
///
/// # Example 1:
/// ```
/// use std::{convert::Infallible, error::Error, ops::ControlFlow};
/// use stepwise::Algo;
///
/// // Note that the hyperparameter (learning rate) and initial/best
/// // solution are considered part of the algorithm
/// pub struct GradientDescent<G> {
///     pub gradient_fn:   G,
///     pub x:             Vec<f64>,
///     pub learning_rate: f64,
/// }
///
/// impl<G> Algo for GradientDescent<G>
/// where
///     G: Fn(&[f64]) -> Vec<f64>
/// {
///     type Error = Infallible;
///
///     fn step(&mut self) -> ControlFlow< Result<(),Infallible>, Result<(),Infallible> > {
///        let dfdx = (self.gradient_fn)(&self.x);
///        for i in 0..self.x.len() {
///            self.x[i] -=  self.learning_rate * dfdx[i];
///        }
///
///        // Allow the calling client to decide when to cease iteration.
///        // If our algorithm managed convergence/iteration counts internally,
///        // `ControlFlow::Break` would be returned to terminate iteration.
///        ControlFlow::Continue(Ok(()))
///     }
/// }
/// ```
/// # Observations
///
/// * The `step` method is similar to an iterator's `next` method, but can return an error. An
///   error can be returned regardless of whether the ControlFlow is Continue or Break.
///
/// * Unlike an Iterator which uses Some or None to indicate the end of the iteration,
///   Algo uses `ControlFlow::Continue` or `ControlFlow::Break`.
///
/// * Whereas iterators return their current value in the call to `next`, algorithms return `()` from `step`
///   but have accessors or public variables to access the outcome after each iterative step.
///
/// * By convention, for a 1-dimensional or n-dimensional vector result, the solution vector is called `x`.
///
/// * Returning `ControlFlow::Continue` allows the caller to decide when to stop iterating,
///   using either a fixed number of iterations, a tolerance, or a time limit.
///
/// * If the algo itself knows when it has converged, or if it tracks max iterations internally,
///   it can return `ControlFlow::Break` to indicate that no more steps are possible.
///
/// * The initial starting conditions are part of the algo, as is the problem being solved (in this case a gradient function)
///
/// * The learning rate is public, and can be changed (adaptive) during the optimization process.
///
/// * Most numeric algorithms, where the results of each iteration depend upon previous
///   iterations will return `ControlFlow::Break(Err(e))` on an error, to stop iteration
///
/// * An algorithm such as a file line-parser, might continue iteration, collecting all
///   errors. In this case `ControlFlow::Continue(Err(e))` might be used.
///
/// # Example 2:
/// This is the bisection algorithm, used to find the root (solution) to `f(x) = 0`
///
/// ```
/// # use stepwise::prelude::*;
/// # use std::ops::ControlFlow;
/// # use std::error::Error;
/// pub struct Bisection<F> {
///     f:     F,         // the objective function being solved: Fn(f64) -> Result<f64>
///     x:     [f64; 2],  // lower and upper bounds around the solution
///     f_lo:  f64,        
///     f_mid: f64,       // mid point of last function evaluations at lower and upper bounds 
/// }
/// 
/// // There are no restrictions on non-trait access (or setter) methods.
/// impl<F> Bisection<F> {
///     // The lower and upper bounds around the the "best solution", so are designated `x`
///     pub fn x(&self) -> [f64; 2] {
///         self.x
///     }
///     
///     // The midpoint of the current range.
///     pub fn x_mid(&self) -> f64 {
///         (self.x[0] + self.x[1]) / 2.0
///     }
///     
///     // The function, `f`, evaluated at the midpoint of the current range.
///     pub fn f_mid(&self) -> f64 {
///         self.f_mid
///     }
/// }
/// 
/// 
/// impl<E, F> Algo for Bisection<F>
/// where
///     F: FnMut(f64) -> Result<f64, E>,
///     E: 'static + Send + Sync + Error,
/// {
///     type Error = E;
/// 
///     fn step(&mut self) -> ControlFlow<Result<(), E>, Result<(), E>> {
///         let mid = (self.x[0] + self.x[1]) / 2.0;
///         match (self.f)(mid) {
///             Ok(y) => self.f_mid = y,
///             // our objective function is falliable, so we propagate any errors 
///             // we return `Break` if we encounter an error, as futher 
///             // iteration is not possible
///             Err(e) => return ControlFlow::Break(Err(e)),
///         };
/// 
///         if self.f_mid == 0.0 {
///             self.x = [mid, mid];
///             /// return `Break` if we know we have converged
///             return ControlFlow::Break(Ok(())); 
///         }
/// 
///         if self.f_mid * self.f_lo < 0.0 {
///             self.x[1] = mid;
///             ControlFlow::Continue(Ok(()))
///         } else {
///             self.x[0] = mid;
///             self.f_lo = self.f_mid;
///             ControlFlow::Continue(Ok(()))
///         }
///     }
/// }
/// ```
///  
pub trait Algo {
    /// The type of error that can occur during iteration. Use [`std::convert::Infallible`] if the algo cannot fail.
    type Error: Error + Send + Sync + 'static;

    /// Execute a single step of the algo. `ControlFlow::Continue(())` indicates that the algo should continue, and that
    /// the algo has not converged, or that the convergence is decided by the client.
    /// `ControlFlow::Break(())` indicates that the algo has converged, and no more iteration steps are possible
    /// Failure to converge should be indicated by returning an error.
    fn step(&mut self) -> ControlFlow<Result<(), Self::Error>, Result<(), Self::Error>>;
}


pub type StepControlFlow<E> = ControlFlow<Result<(), E>, Result<(), E>>;

impl<S: Algo + ?Sized> Algo for Box<S> {
    type Error = S::Error;

    fn step(&mut self) -> StepControlFlow<Self::Error> {
        (**self).step()
    }
}

pub fn result_to_control_flow<T, E>(res: Result<T, E>) -> ControlFlow<Result<(), E>, T> {
    match res {
        Ok(t) => ControlFlow::Continue(t),
        Err(e) => ControlFlow::Break(Err(e)),
    }
}

#[doc(hidden)]
pub trait TolX {
    type X: Debug + 'static;
    type Error: Error + Send + Sync + 'static;

    fn tol_x(&self, prior: Option<&Self::X>) -> Result<(Option<f64>, Self::X), Self::Error>;
}
