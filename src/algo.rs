use std::{error::Error, fmt::Debug, ops::ControlFlow};

/// Algorithms implement this trait to work with the library.
///
/// # Overview
/// The algo has a `Step` method, similar to iterator's `next`.
///
/// The "problem" being solved and any initial starting conditions are part of the algo, not
/// seperate entities.
///
/// Use [`std::convert::Infallible`] for `Error` if the algorithm cannot fail.
///
/// The return parameter indicates whether iteration *can* continue, and propagates any errors encountered.
///
/// `ControlFlow::Continue(())` indicates that the algo could continue
/// though the final decision is made by the calling code (perhaps by
/// logic within [`Driver::converge_when`](crate::Driver::converge_when) ).
///
/// `ControlFlow::Break(())` indicates that the algo has converged or errored, and no more iteration steps are possible.
///
/// Independently, any errors occuring are returned in the tuple.
///
/// For many algorithms, each step depends on the previous step. In such cases,
/// the algorithm will return `Break` whenever an error occurs, as iteration cannot continue.
/// An algorithm such as a file line-parser, might continue iteration, collecting up all
/// errors in the file. In this case `(ControlFlow::Continue, Err(e))` would be used.
///
/// Whereas iterators return their current value in the call to next,
/// algorithms return `()` from step.
///
/// But to be useful, they will need accessors or public variables to access
/// the solution after after, or during iteration.
///
/// If called before the first call to `step`, they
/// could return the initial starting conditions of the algorithm.
///
/// By convention, the current best solution would be called `x()`
/// for a scalar or vector solution.
///
///
/// # Example 1:
/// Note that the hyperparameter (learning rate) and initial/best
/// solution are considered part of the algorithm. As is the problem being solved (in this case a gradient function).
///
/// The learning rate is public, and can be changed (is _adaptive_) during the optimization process.
/// ```
/// use std::{convert::Infallible, error::Error, ops::ControlFlow};
/// use stepwise::Algo;
///
/// //
/// pub struct GradientDescent<G> {
///     pub gradient_fn:   G,
///     // current best solution, seeded with initial guess
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
///     fn step(&mut self) -> (ControlFlow<()>, Result<(),Infallible>) {
///        let dfdx = (self.gradient_fn)(&self.x);
///        for i in 0..self.x.len() {
///            self.x[i] -=  self.learning_rate * dfdx[i];
///        }
///
///        // Allow the calling client to decide when to cease iteration.
///        // If our algorithm managed convergence/iteration counts internally,
///        // `ControlFlow::Break` would be returned to terminate iteration.
///        (ControlFlow::Continue(()), Ok(()))
///     }
/// }
/// ```
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
/// // There are no restrictions on non-trait access (or setter) methods, as these are
/// // defined outside of the trait.
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
///     fn step(&mut self) -> (ControlFlow<()>, Result<(), E>) {
///         let mid = (self.x[0] + self.x[1]) / 2.0;
///         match (self.f)(mid) {
///             Ok(y) => self.f_mid = y,
///             // our objective function is falliable, so we propagate any errors
///             // we return `Break` if we encounter an error, as futher
///             // iteration is not possible
///             Err(e) => return (ControlFlow::Break(()), Err(e)),
///         };
///
///         if self.f_mid == 0.0 {
///             self.x = [mid, mid];
///             /// return `Break` if we know we have converged
///             return (ControlFlow::Break(()), Ok(()));
///         }
///
///         if self.f_mid * self.f_lo < 0.0 {
///             self.x[1] = mid;
///             (ControlFlow::Continue(()),Ok(()))
///         } else {
///             self.x[0] = mid;
///             self.f_lo = self.f_mid;
///             (ControlFlow::Continue(()),Ok(()))
///         }
///     }
/// }
/// ```
///  
pub trait Algo {
    type Error: Error + Send + Sync + 'static;

    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>);
}

/// The return type for the Algo trait's `step` method
///
pub(crate) type AlgoResult<E> = (ControlFlow<()>, Result<(), E>);

impl<S: Algo + ?Sized> Algo for Box<S> {
    type Error = S::Error;

    fn step(&mut self) -> AlgoResult<Self::Error> {
        (**self).step()
    }
}

pub fn result_to_control_flow<T, E>(res: Result<T, E>) -> ControlFlow<Result<(), E>, T> {
    match res {
        Ok(t) => ControlFlow::Continue(t),
        Err(e) => ControlFlow::Break(Err(e)),
    }
}

pub(crate) trait TolX {
    type X: Debug + 'static;
    type Error: Error + Send + Sync + 'static;

    fn tol_x(&self, prior: Option<&Self::X>) -> Result<(Option<f64>, Self::X), Self::Error>;
}
