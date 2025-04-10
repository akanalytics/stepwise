use std::{
    collections::VecDeque,
    error::Error,
    fmt::{Debug, Display},
    ops::ControlFlow,
    time::{Duration, Instant},
};

use crate::VectorExt;

#[derive()]
pub enum SolverError {
    FailPredicate,
    FailMaxElapsed,
    Overflow,
    UnexpectedError(Box<dyn Error>),
}

impl Debug for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FailPredicate => f.debug_tuple("FailPredicate").finish(),
            Self::FailMaxElapsed => f.debug_tuple("FailMaxElapsed").finish(),
            Self::Overflow => f.debug_tuple("Overflow").finish(),
            Self::UnexpectedError(e) => f.debug_tuple("UnexpectedError").field(e).finish(),
        }
    }
}

/// Formats a duration in the form 'hh:mm:ss.SSS'.
///
/// eg 5 hours 12 mins 17 secs and 345 millis would display as
///     `05:12:17.345`
///
pub fn format_duration(d: Duration) -> impl Display {
    struct DurationDisplay(Duration);
    impl Display for DurationDisplay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let total_seconds = self.0.as_secs();
            let millis = self.0.subsec_millis();
            let hours = total_seconds / 3600;
            let mins = (total_seconds % 3600) / 60;
            let secs = total_seconds % 60;
            write!(f, "{:02}:{:02}:{:02}.{:03}", hours, mins, secs, millis)
        }
    }
    DurationDisplay(d)
}

impl Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Error for SolverError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self {
            Self::UnexpectedError(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}

pub trait Solver {
    fn step(&mut self) -> Result<ControlFlow<()>, Box<dyn Error>>;
    fn x(&self) -> &[f64];
}

pub struct Step<S> {
    solver: S,
    iteration: usize,
    elapsed: Duration,
    trajectory_capacity: usize,
    trajectory: VecDeque<Vec<f64>>,
    fixed_iterations: Option<usize>,
    fail_after: Option<Duration>,
}

impl<S: Solver> Display for Step<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:>5}]", self.iteration())?;
        write!(f, " [{}]", format_duration(self.elapsed()))?;
        write!(f, " x = ")?;
        Display::fmt(&self.x().display(), f)?;
        if let Some(prior_x) = self.prior_x() {
            write!(f, " prior_x = ",)?;
            Display::fmt(&prior_x.display(), f)?;
        }
        Ok(())
    }
}

impl<S: Solver> Debug for Step<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed = format_duration(self.elapsed()).to_string();
        f.debug_struct("Step")
            .field("iteration", &self.iteration)
            .field("x", &self.x().display().to_string())
            .field("elapsed", &elapsed)
            .field("trajectory_capacity", &self.trajectory_capacity)
            .field("trajectory.len()", &self.trajectory.len())
            .finish()
    }
}

impl<S: Solver> Step<S> {
    /// The iteration or step count.
    ///
    /// Equivalently how many times [`Solver::step`] has been called.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Total elapsed time since the call to [`Driver::try_solve`].
    ///
    /// The elapsed time is only
    /// updated after each iteration step, and will not change between calls otherwise.
    ///
    /// See also
    /// - [`Driver::fail_after`]
    /// - [`format_duration`]
    ///
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// An estimate between `0` and `100` of how complete the solving process is.
    ///
    /// The most optimistic of
    /// - iterations out of fixed_iterations, or
    /// - elapsed time out of of "fail after" duration
    ///
    /// will be used for the estimate.
    pub fn progress_percentage(&self) -> Option<f64> {
        let prog_iters = if let Some(n) = self.fixed_iterations {
            Some(self.iteration() as f64 / n as f64).filter(|p| p.is_finite())
        } else {
            None
        };
        let prog_time = if let Some(tot) = self.fail_after {
            Some(self.elapsed().as_secs_f64() / tot.as_secs_f64()).filter(|p| p.is_finite())
        } else {
            None
        };

        match (prog_time, prog_iters) {
            (None, Some(i)) => Some(100.0 * i),
            (Some(t), None) => Some(100.0 * t),
            (Some(t), Some(i)) => Some(100.0 * f64::max(t, i)),
            (None, None) => None,
        }
    }

    /// An estimate of the time remaining.
    ///
    /// The estimate is based on [`Self::progress_percentage`], and how much time has elapsed so far.
    pub fn time_remaining(&self) -> Option<Duration> {
        let progress = self.progress_percentage()?;
        let secs = self.elapsed().as_secs_f64() * 100.0 / progress;
        if secs.is_finite() {
            Some(Duration::from_secs_f64(secs))
        } else {
            Some(Duration::ZERO)
        }
    }

    /// The current `x` as reported by the solver.
    ///
    /// Typically the best solution so far. Once the solver has converged via a
    /// successful call to [`Driver::try_solve`], this will represent the best solution.
    pub fn x(&self) -> &[f64] {
        self.solver.x()
    }

    /// The prior solution being iterated from the trajectory. Before the first call to [`Solver::step`] (iteration zero)
    /// this will be `None`. If trajectory capacity has been changed to zero
    /// via [`Driver::trajectory_capacity`], then `None` will always be returned.
    pub fn prior_x(&self) -> Option<&[f64]> {
        self.trajectory.iter().last().map(Vec::as_slice)
    }

    /// The prior solutions being iterated from the trajectory.
    /// There will be from zero (at iteration zero) to at most [`Driver::trajectory_capacity`] of x-points.
    pub fn trajectory_x(&self) -> impl Iterator<Item = &[f64]> {
        self.trajectory.iter().map(Vec::as_slice)
    }

    /// The underlying [`Solver`] being driven.
    ///
    /// Often the solver will report more information on each iteration, such as the total function evaluation calls,
    /// or the current `cost` of the objective function.
    ///  
    pub fn solver(&self) -> &S {
        &self.solver
    }

    /// The underlying [`Solver`] being driven.
    pub fn solver_mut(&mut self) -> &mut S {
        &mut self.solver
    }
}

/// an executer that offers methods to control iteration, convergence, logging and progress display
///
/// ## Implmenting the solver
///
/// ```
/// pub struct GradientDescent<G> {
///     pub gradient: G,
///     pub x: [f64; 2],
///     pub learning_rate: f64,
/// }
///
/// use stepwise::Solver;
/// use std::{error::Error, ops::ControlFlow};
///
/// impl<G> Solver for GradientDescent<G>
/// where
///     G: Fn([f64; 2]) -> [f64; 2],
/// {
///     fn step(&mut self) -> Result<ControlFlow<()>, Box<dyn Error>> {
///         let [x, y] = self.x;
///         let [dfdx, dfdy] = (self.gradient)([x, y]);
///         self.x = [x - self.learning_rate * dfdx, y - self.learning_rate * dfdy];
///
///         // allow stepwise Driver to decide when to cease iteration
///         Ok(ControlFlow::Continue(()))
///     }
///
///     fn x(&self) -> &[f64] {
///         self.x.as_slice()
///     }
/// }
/// ```
///
/// ## Using the solver (driving the solver)
///
/// ```
/// use stepwise::examples::GradientDescent;
/// use stepwise::Driver;
///
///
/// let my_solver = GradientDescent {
///     gradient: |[x, y]: [f64; 2]| [(x - 1.5), (y - 2.0)],
///     x: [5.0, 6.0],
///     learning_rate: 0.01,
/// };
///
/// let driver = Driver::new(my_solver)
///     .fixed_iterations(1000)
///     .for_each(|step| println!("{step}"));
///
///
/// let solution: Step = driver.try_solve().expect("solving failed!");
///
/// assert_eq!(solution.iteration(), 1000);
/// let x = solution.x()[0];
/// let y = solution.x()[1];
/// assert_eq!(format!("{x:1.3}", ), "1.500");
/// assert_eq!(format!("{y:1.3}", ), "2.000");
///
/// ```
///
///
/// The Driver for a solver is effectively a solution builder.
/// * contruct a driver from a [`Solver`]
/// * apply various convergence predicates (tolerances or iteration counts)
/// * apply any failure criteria (overflow, or max elapsed time or iteration count)
/// * calling [`Self::try_solve`] will repeated iterate to a solution which is returned
///
/// See [crate] for an example
#[allow(clippy::type_complexity)]
pub struct Driver<'a, S /* Solver */> {
    step: Step<S>,
    fail_if: Vec<Box<dyn FnMut(&mut Step<S>) -> bool + 'a>>,
    conv_when: Vec<Box<dyn FnMut(&mut Step<S>) -> bool + 'a>>,
    for_each: Vec<Box<dyn FnMut(&mut Step<S>) -> Result<(), Box<dyn Error>> + 'a>>,
    fail_on_overflow: bool,
    min_iters: usize,
}

impl<'a, S: Solver> Driver<'a, S> {
    pub fn new(solver: S) -> Self {
        Self {
            step: Step {
                solver,
                iteration: 0,
                elapsed: Duration::ZERO,
                trajectory_capacity: 1,
                trajectory: VecDeque::new(),
                fixed_iterations: None,
                fail_after: None,
            },
            fail_if: vec![],
            conv_when: vec![],
            for_each: vec![],
            fail_on_overflow: false,
            min_iters: 0,
        }
    }

    /// Runs the solver, until failure, iteration exhaustion, or convergence.
    ///
    /// In cases of success,
    /// the final (iteration step)[Step] is returned with (`x`)[Step::x] the
    /// best solution. The solver in its final state, along with total execution time, and iteration counts are available.
    /// * Example
    ///
    ///
    /// ```
    /// # use stepwise::examples::example_solver;
    /// # use stepwise::Driver;
    /// let my_solver = example_solver();
    ///
    /// let solution = Driver::new(my_solver)
    ///     .fixed_iterations(1000)
    ///     .for_each(|step| println!("{step}"))
    ///     .try_solve()?;
    ///
    ///
    /// // assert_eq!(solution.x(), /* approximately [1.5, 2.0] */ );
    /// # Ok::<(), stepwise::SolverError>(())
    /// ```
    ///
    pub fn try_solve(self) -> Result<Step<S>, SolverError> {
        match self.solve_with_status() {
            (s, Ok(_)) => Ok(s),
            (_, Err(e)) => Err(e),
        }
    }

    /// Runs the solver, until failure, iteration exhaustion, or convergence.
    /// In contrast to [`Self::try_solve`] the final (iteration step)[Step] is returned along with (`x``)[Step::x] the
    /// best solution (so far), even in cases when an error has occurred, or iteration did not result in convergence.
    /// Useful if tracking down the iteration step triggering an error, otherwise [`Self::try_solve`] offers a simpler
    /// approach.
    ///
    pub fn solve_with_status(mut self) -> (Step<S>, Result<(), SolverError>) {
        let start = Instant::now(); // restart the clock

        let res = self
            .for_each
            .iter_mut()
            .try_for_each(|f| (f)(&mut self.step));
        if let Err(e) = res {
            return (self.step, Err(SolverError::UnexpectedError(e)));
        };
        loop {
            // iters is at least 1, and prior_x always set
            let x = self.step.solver.x().to_vec();
            let res = self.step.solver.step();
            let flow = match res {
                Err(e) => return (self.step, Err(SolverError::UnexpectedError(e))),
                Ok(flow) => flow,
            };

            // record x in the window of trajectories (losing the front/last element if needed)
            if self.step.trajectory.len() >= self.step.trajectory_capacity {
                self.step.trajectory.pop_front();
            }
            self.step.trajectory.push_back(x);
            self.step.iteration += 1;

            self.step.elapsed = start.elapsed();

            let res = self
                .for_each
                .iter_mut()
                .try_for_each(|f| (f)(&mut self.step));

            if let Err(e) = res {
                return (self.step, Err(SolverError::UnexpectedError(e)));
            };

            // is finite => not inf, -inf or NaN
            if self.fail_on_overflow && self.step.x().iter().any(|x| !x.is_finite()) {
                return (self.step, Err(SolverError::Overflow));
            }

            if flow.is_break() {
                break;
            }

            if self.step.iteration() < self.min_iters
                || self
                    .step
                    .fixed_iterations
                    .is_some_and(|n| self.step.iteration() < n)
                || self.step.iteration() < self.step.trajectory_capacity
            {
                continue;
            }

            if self.step.fixed_iterations == Some(self.step.iteration())
                || self.conv_when.iter_mut().any(|f| (f)(&mut self.step))
            {
                break;
            }

            if self.fail_if.iter_mut().any(|f| (f)(&mut self.step)) {
                return (self.step, Err(SolverError::FailPredicate));
            }

            if self.step.fail_after.is_some_and(|d| d > self.step.elapsed) {
                return (self.step, Err(SolverError::FailMaxElapsed));
            }
        }
        (self.step, Ok(()))
    }

    /// Will stop iteration.
    /// Common convergence predicates are residuals being below a certain epsilon. Some specific convergence criteria
    /// are better handled by
    /// - [`Self::fixed_iterations`]
    ///
    pub fn converged_when<F>(mut self, pred: F) -> Self
    where
        F: FnMut(&mut Step<S>) -> bool + 'a,
    {
        self.conv_when.push(Box::new(pred));
        self
    }

    /// Always run at most this number of iterations, and *assume convergence* once this number of iterations has run, providing [`Self::fail_if`] is not true.
    /// Fewer iterations will be run if the [`Self::converged_when`] predicate succeeds, or if overflow occurs and is [designated a failure](Self::fail_on_overflow).
    pub fn fixed_iterations(mut self, iters: usize) -> Self {
        self.step.fixed_iterations = Some(iters);
        self
    }

    /// Always run at least this number of iterations. Useful if the convergence criteria
    /// are unreliable in the first few iterations. For example a poor initial guess may lead to
    /// misleading convergence indicators, as the solver might first need to navigate toward the
    /// region of interest before genuine convergence begins.
    ///
    pub fn min_iters(mut self, min_iters: usize) -> Self {
        self.min_iters = self.min_iters.max(min_iters);
        self
    }

    /// Set how many recent x coordinates to record.
    ///
    /// Defaults to 1, thus enabling [`Step::prior_x`]).
    ///
    /// Set to zero to avoid the overhead of allocating for a point x on every
    /// iteration, though [`Step::prior_x`] will always be None.
    ///
    /// Set to usize::MAX to record the full trajectory, perhaps using much memory.
    ///
    pub fn trajectory_capacity(mut self, capacity: usize) -> Self {
        self.step.trajectory_capacity = capacity;
        self
    }

    /// Abandon iteration with [`Self::try_solve`] returning [`SolverError::FailPredicate`].
    /// Convergence predicates are tested first.
    /// Common failure predicates are residuals growing in size after an initial set of iterations,
    /// or user cancelation - implemented perhaps by the closure predicate
    /// checking the value of an `AtomicBool`.
    ///
    /// Caertain specific failure predicates:
    /// - [`Self::fail_after`]
    /// - [`Self::fail_on_overflow`]
    ///
    pub fn fail_if<F>(mut self, pred: F) -> Self
    where
        F: FnMut(&mut Step<S>) -> bool + 'a,
    {
        self.fail_if.push(Box::new(pred));
        self
    }

    /// Sets a maximum elapsed time for iteration. Will cause [`Self::try_solve()`] to
    /// return a [`SolverError::FailMaxElapsed`] if the elapsed time has been exceeded.
    ///
    /// Maximum iteration time is only checked after each [`Solver::step`] and
    /// [convergence](Self::converged_when) or [conditional failure predicates](Self::fail_if) are
    /// checked first, so a successful solve can have an elapsed duration that exceeds the fail_after duration,
    /// as the final step will have started before the elapsed maximum duration but completed after.
    ///
    pub fn fail_after(mut self, d: Duration) -> Self {
        self.step.fail_after = Some(d);
        self
    }

    /// Will cause [`Self::try_solve()`] to return an error if any x-coordinate is non-finite during iteration.
    ///
    /// Defaults to false, which means the solution may contain [`f64::NAN`] or [`f64::INFINITY`].
    pub fn fail_on_overflow(mut self, fail_if_overflow: bool) -> Self {
        self.fail_on_overflow = fail_if_overflow;
        self
    }

    /// Invoked after each iteration, allowing printing or debugging of the iteration [step](Step).
    /// Can be used to capture details of the iteration. See [`Self::try_for_each(`].
    pub fn for_each<F>(self, mut f: F) -> Self
    where
        F: FnMut(&mut Step<S>) + 'a,
    {
        self.try_for_each(move |x| {
            (f)(x);
            Ok(())
        })
    }

    /// Invoked after each iteration, allowing printing or debugging of the iteration [Step]. Any errors will terminate the solving,
    /// and [`Self::try_solve`] will return an error.
    /// Can be used to capture details of the iteration. See [`Self::for_each(`].
    pub fn try_for_each<F>(mut self, f: F) -> Self
    where
        F: FnMut(&mut Step<S>) -> Result<(), Box<dyn Error>> + 'a,
    {
        self.for_each.push(Box::new(f));
        self
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use std::f64;
    use test_log::test;

    #[derive(Debug, Clone)]
    pub struct CountingSolver(f64);

    impl Solver for CountingSolver {
        fn step(&mut self) -> Result<ControlFlow<()>, Box<dyn Error>> {
            self.0 += 1.0;
            if self.0.round() == -10.0 {
                return Err(std::fmt::Error)?;
            }
            Ok(ControlFlow::Continue(()))
        }

        fn x(&self) -> &[f64] {
            std::slice::from_ref(&self.0)
        }
    }

    #[test]
    fn iterations() {
        let solver = CountingSolver(3.0);
        let solved = Driver::new(solver).fixed_iterations(5).try_solve().unwrap();
        assert_eq!(solved.iteration(), 5);
        assert_eq!(solved.x()[0].round(), 8.0);
        assert_eq!(solved.solver().0.round(), 8.0);

        let solver = CountingSolver(3.0);
        let solved = Driver::new(solver).fixed_iterations(1).try_solve().unwrap();
        assert_eq!(solved.iteration(), 1);
        assert_eq!(solved.prior_x().map(|x| x[0].round()), Some(3.0));
        assert_eq!(solved.x()[0].round(), 4.0);
    }

    #[test]
    fn try_solve() {
        // let solver = CountingSolver(0.0);
        // let solved = Driver::new(solver).fixed_iterations(0).try_solve().unwrap();
        // assert_eq!(solved.iteration(), 1);
        // assert_eq!(solved.prior_x().map(|x| x[0].round()), Some(0.0));
        // assert_eq!(solved.x()[0].round(), 1.0);
        let solver = CountingSolver(f64::INFINITY);
        let solved = Driver::new(solver)
            .fail_on_overflow(true)
            .fixed_iterations(1)
            .try_solve();
        assert!(solved.is_err());
        assert!(matches!(solved, Err(SolverError::Overflow)));

        let solver = CountingSolver(f64::INFINITY);
        let solved = Driver::new(solver).fixed_iterations(1).try_solve().unwrap();
        assert_eq!(solved.iteration(), 1);

        let solver = CountingSolver(f64::INFINITY);
        let solved = Driver::new(solver).fixed_iterations(1).try_solve().unwrap();
        assert_eq!(solved.iteration(), 1);

        let solver = CountingSolver(-20.0);
        let result = Driver::new(solver).fixed_iterations(20).try_solve();
        assert!(result.is_err());
        assert!(result
            .as_ref()
            .unwrap_err()
            .source()
            .expect("source was None")
            .is::<std::fmt::Error>());
        assert!(matches!(result, Err(SolverError::UnexpectedError(_))));
        if let Err(SolverError::UnexpectedError(e)) = result {
            assert!(e.is::<std::fmt::Error>());
        } else {
            panic!();
        };
    }

    #[test]
    fn solve_with_status() {
        let solver = CountingSolver(-20.0);
        let (solved, res) = Driver::new(solver).fixed_iterations(20).solve_with_status();
        assert!(res.is_err());
        assert!(res
            .as_ref()
            .unwrap_err()
            .source()
            .expect("source was None")
            .is::<std::fmt::Error>());
        assert!(matches!(res, Err(SolverError::UnexpectedError(_))));
        assert_eq!(solved.x()[0].round(), -10.0);
        if let Err(SolverError::UnexpectedError(e)) = res {
            assert!(e.is::<std::fmt::Error>());
        } else {
            panic!();
        };
    }

    // #[test]
    // fn solver2() {
    //     // stack(s.x()).dist_max() < 0.1

    //     // struct Prior<T: ToOwned>(Option<T::Owned>);
    //     // impl<T: ToOwned> Prior<T> {
    //     //     pub fn replace<'a>(&'a mut self, t: T) -> T::Owned {
    //     //         self.0.replace(t.to_owned()).unwrap()
    //     //     }
    //     // }

    //     // let mut prior = Prior(None);
    //     // let a1 = prior.replace(&5);
    //     // let a2 = prior.replace(&3);

    //     let path = Arc::new(Mutex::new(Vec::new()));
    //     let driver = Driver::new(MySolver {
    //         iters: 0,
    //         x: [1.5, 2.0],
    //     });
    //     // let mut p = Option::<Vec<f64>>::Some(vec![10.0, 11.0]);
    //     let driver = driver
    //         .fail_on_overflow(true)
    //         .fail_if(|s| s.x()[0] > 5.0)
    //         .converged_when(|s| s.prior_x().is_some_and(|p| p.dist_l2(s.x()) < 0.1))
    //         // .converged_when(|s| {
    //         //     <[_]>::dist_l2(p.replace(s.x().to_vec()).unwrap().as_slice(), s.x()) < 0.1
    //         // })
    //         .for_each(|s| println!("println#1 {:+0.5}", s))
    //         .for_each(|s| println!("println#2 {}", s.solver.x[0]))
    //         .for_each(|s| trace!("println#3 {:?}", s.solver))
    //         .for_each({
    //             let path = Arc::clone(&path);
    //             move |s| path.lock().unwrap().push(s.solver.x)
    //         });
    //     let solution = driver.try_solve().unwrap();
    //     assert_eq!(solution.iteration(), 8);
    //     println!("solution = {:?}", solution.x());
    //     println!(
    //         "path = {}",
    //         path.lock()
    //             .unwrap()
    //             .iter()
    //             .map(|x| x[0].to_string())
    //             .collect::<Vec<_>>()
    //             .join(", ")
    //     );
    // }
    // #[test]
    // fn solver3() {
    //     let mut path = Vec::new();
    //     let avg = ExponentialMovingAvg::with_window(10);
    //     let driver = Driver::new(MySolver {
    //         iters: 0,
    //         x: [1.5, 2.0],
    //     });
    //     let driver = driver
    //         .fail_on_overflow(true)
    //         .min_iters(10)
    //         .for_each(|s| avg.collect(s.x().norm_l2()))
    //         .converged_when(|_| avg.get() < 0.1)
    //         .for_each(|s| println!("println#1 {s:+0.5} {avg:0.4}"))
    //         .for_each(|s| path.push(s.solver.x))
    //         .fail_if(|s| s.x()[0] > 5.0);
    //     let solved = driver.try_solve().expect("failed to converge");
    //     println!("solution = {:?}", solved.x());
    //     println!(
    //         "path = {}",
    //         path.iter()
    //             .map(|x| x[0].to_string())
    //             .collect::<Vec<_>>()
    //             .join("\n")
    //     );
    // }
}
