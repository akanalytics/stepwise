use crate::{
    algo::result_to_control_flow, progress_bar::display_progress_bar, step::Step, Algo, Progress,
    DriverError, TolX,
};
use private::*;
use std::{
    error::Error,
    ops::ControlFlow,
    sync::Arc,
    time::{Duration, Instant},
};

/// Constructs a [`Driver`] with a fixed number of iterations
///
/// After this number of itersations the algo is deemed to have converged on a solution.
/// If [`Driver::converge_when`] is used, then iteration may terminate early.
pub fn fixed_iters<S: Algo>(algo: S, fixed_iters: usize) -> BasicDriver<S> {
    BasicDriver {
        iters: Some(fixed_iters),
        ..BasicDriver::from_solver(algo)
    }
}

/// Constructs a [`Driver`] that will fail once a certain number of iterations is reached.
///
/// Uless convergence has occurred, the algo will error after the designated iteration count.
/// Either the algo has internal logic to terminate iteration when a solution is found,
/// or the caller needs to use [`Driver::converge_when`]
/// to ensure iteration stops before a (max iterations exceeded error)[DriverError::MaxIterationsExceeded].
pub fn fail_after_iters<S: Algo>(algo: S, max_iters: usize) -> BasicDriver<S> {
    BasicDriver {
        iters: Some(max_iters), // TODO
        ..BasicDriver::from_solver(algo)
    }
}

/// Constructs a [`Driver`] that will fail after the timeout.
///
/// Either the algo
/// has internal logic to terminate iteration when a solution is found, or the caller needs to use [`Driver::converge_when`]
/// to ensure iteration stops before a (timeout error)[DriverError::Timeout].
pub fn with_timeout<S: Algo>(algo: S, timeout: Duration) -> BasicDriver<S> {
    BasicDriver {
        timeout: Some(timeout),
        ..BasicDriver::from_solver(algo)
    }
}

pub trait Driver
where
    Self: DriverExt,
{
    /// Runs the algorithm, until failure, iteration exhaustion, or convergence.
    ///
    /// In cases of success, the algo in its final state and
    /// the final [iteration step](Step) is returned.
    ///
    /// By convention, the algo variable in its final state is named `solved`, and will offer accessors
    /// to retrieve the solution or best approximation to the solution.
    ///
    /// # Example
    /// ```
    /// # use stepwise::algos::GradientDescent;
    /// # use stepwise::{Driver, fixed_iters};
    /// # use stepwise::assert_approx_eq;
    /// # fn dummy<G: Fn(&[f64]) -> Vec<f64>>(my_algo: GradientDescent<G>) -> Result<(), stepwise::DriverError> {
    ///
    /// let (solved, _step) = fixed_iters(my_algo, 1000)
    ///     .on_step(|_algo, step| println!("{step:?}"))
    ///     .solve()?;
    ///
    /// assert_approx_eq!(solved.x(),  &vec![1.5, 2.0] );
    /// # Ok(())
    /// # }
    /// ```
    fn solve(self) -> Result<(Self::Algo, Step), DriverError>
    where
        Self: Sized;

    fn iter_step(&mut self) -> Result<Option<(&mut Self::Algo, &Step)>, DriverError> {
        let cf = self.solve_once();
        match cf {
            ControlFlow::Continue(res) => {
                res?; // TODO! error should never happen in continue case
                let (algo, step) = self.solver_step_mut();
                Ok(Some((algo, step)))
            }

            ControlFlow::Break(res) => {
                res?;
                Ok(None)
            }
        }
    }


    /// Invoked after each iteration, allowing printing or debugging of the iteration [step](Step).
    /// Can be used to capture details of the iteration. See [`Self::try_on_step`].
    fn try_on_step<F, E>(self, f: F) -> TryForEachDriver<Self, F>
    where
        F: FnMut(&mut Self::Algo, &Step) -> Result<(), E>,
        E: Error + Sync + Send + 'static,

        Self: Sized,
    {
        TryForEachDriver { driver: self, f }
    }

    /// Invoked after each iteration, allowing printing or debugging of the iteration [Step]. Any errors 
    /// encountered by the underlying algorithm will terminate the solving, and [`Self::solve`] will return the error.
    /// 
    /// Can be used to capture details of the iteration, or update hyperparameters on the algorithm for adaptive learning.
    fn on_step<F>(self, f: F) -> ForEachDriver<Self, F>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step),
    {
        ForEachDriver { driver: self, f }
    }

    /// Used for early stopping.
    ///  
    /// Common convergence predicates are step size below a small epsilon, or residuals being below near zero. 
    /// Some specific convergence criteria are best handled by using [`metrics`](crate::metrics) within this callback.
    fn converge_when<F>(self, pred: F) -> ConvergedWhenDriver<Self, F>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> bool,
    {
        ConvergedWhenDriver { driver: self, pred }
    }

    /// Decide when to abandon iteration with [`Self::solve`] returning [`DriverError::FailIfPredicate`].
    /// 
    /// Convergence predicates are tested first.
    ///
    /// Common failure predicates are residuals growing in size after an initial set of iterations,
    /// or user cancelation - implemented perhaps by the closure predicate
    /// checking the value of an `AtomicBool`.
    ///
    fn fail_if<F>(self, pred: F) -> FailIfDriver<Self, F>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> bool,
    {
        FailIfDriver { driver: self, pred }
    }

    /// Display a progress bar to the terminal after the algo has been running for this duration.
    ///
    /// Use `Duration::MAX` to disable the progress bar, and `Duration::ZERO` to display it immediately.
    /// The progress bar will refresh at most every 50ms.
    ///
    /// The progress bar will not display if either:
    /// - stdout is not a terminal (perhaps redirected to a ulility or file)
    /// - the environment variable `NO_COLOR` is set
    /// - the elapsed time never reaches the duration set by this function
    ///
    /// # Example
    ///
    /// ```
    /// # use stepwise::prelude::*;
    /// # use std::time::Duration;
    /// # fn dummy<G: Fn(&[f64]) -> Vec<f64>>(my_algo: algos::GradientDescent<G>) -> Result<(), DriverError> {
    /// let solution = fixed_iters(my_algo, 1000)
    ///     // show a progress bar if the algo is slow to execute
    ///     .show_progress_bar_after(Duration::from_millis(250))  
    ///     .solve()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    fn show_progress_bar_after(self, after: Duration) -> ProgressBarDriver<Self>
    where
        Self: Sized,
    {
        ProgressBarDriver {
            driver: self,
            next_progress_display: after,
        }
    }

    /// Convenience method for when you want to dynamically type.
    ///
    /// # Examples
    /// The following wont compile as the if/then arms have different types...
    /// ```compile_fail
    /// # use stepwise::prelude::*;
    /// # fn dummy<G: Fn(&[f64]) -> Vec<f64> + 'static>(my_algo: GradientDescent<G>) -> Result<(), stepwise::DriverError> {
    /// # let log_enabled = true;
    /// let driver = if log_enabled {
    ///     fixed_iters(my_algo, 100).on_step(|_, step| println!("{step:?}"))
    /// } else {
    ///     fixed_iters(my_algo, 100)
    /// };
    /// # Ok(())}
    /// ```
    /// Boxing a dyn trait gets around this...
    /// ```
    /// # use stepwise::prelude::*;
    /// # use std::time::Duration;
    /// # fn dummy<G: Fn(&[f64]) -> Vec<f64>+ 'static>(my_algo: algos::GradientDescent<G>) -> Result<(), DriverError> {
    /// let log_enabled = true;
    /// let driver = if log_enabled {
    ///     fixed_iters(my_algo, 100).on_step(|_, step| println!("{step:?}")).into_boxed()
    /// } else {
    ///     fixed_iters(my_algo, 100).into_boxed()
    /// };
    /// // driver is type Box<dyn Driver>
    /// let (my_algo, _step) = driver.show_progress_bar_after(Duration::ZERO).solve()?;
    /// # Ok(())}
    /// ```
    fn into_boxed(self) -> Box<dyn Driver<Algo = Self::Algo>>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    // fn tol_x(self, epsilon: f64) -> TolXDriver<Self>
    // where
    //     Self: Sized,
    //     Self::Algo: TolX,
    // {
    //     TolXDriver {
    //         driver: self,
    //         epsilon,
    //         last_x: None,
    //     }
    // }
}

mod private {
    use super::*;
    pub trait DriverExt {
        type Algo;

        fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>>;
        fn take_solver(&mut self) -> Self::Algo;
        fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step);
        fn into_solver_step(self) -> (Self::Algo, Step);
    }
}
pub struct BasicDriver<S> {
    algo: Option<S>,
    step: Step,
    iters: Option<usize>,
    timeout: Option<Duration>,
    start: Instant,
}

pub struct ForEachDriver<D, F> {
    driver: D,
    f: F,
}

pub struct TryForEachDriver<D, F> {
    driver: D,
    f: F,
}

pub struct ConvergedWhenDriver<D, F> {
    driver: D,
    pred: F,
}

pub struct FailIfDriver<D, F> {
    driver: D,
    pred: F,
}

pub struct ProgressBarDriver<D> {
    driver: D,
    next_progress_display: Duration,
}

pub struct TolXDriver<D>
where
    D: DriverExt,
    D::Algo: TolX,
{
    driver: D,
    epsilon: f64,
    last_x: Option<<D::Algo as TolX>::X>,
}

impl<S> BasicDriver<S> {
    fn from_solver(algo: S) -> Self {
        Self {
            algo: Some(algo),
            step: Default::default(),
            iters: None,
            timeout: None,
            start: Instant::now(),
        }
    }
}

impl<D: Driver + ?Sized> DriverExt for Box<D> {
    type Algo = D::Algo;

    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        <D as DriverExt>::solve_once(self)
    }

    fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step) {
        <D as DriverExt>::solver_step_mut(self)
    }

    fn take_solver(&mut self) -> Self::Algo {
        <D as DriverExt>::take_solver(self)
    }

    // hacky
    fn into_solver_step(mut self) -> (Self::Algo, Step) {
        let (_, step) = self.solver_step_mut();
        let step = step.clone();
        let algo = self.take_solver();
        (algo, step)
    }
}

impl<S, T> Driver for T
where
    T: DriverExt<Algo = S>,
    S: Algo,
{
    // type S = S;

    fn solve(mut self) -> Result<(Self::Algo, Step), DriverError>
    where
        Self: Sized,
    {
        let (_, step) = self.solver_step_mut();
        if let Progress::Failed(e) = &step.progress {
            return Err(e.clone()); // TODO: yuck
        }
        if matches!(step.progress, Progress::NotStarted) {
            loop {
                match self.solve_once() {
                    ControlFlow::Continue(res) => res?,
                    ControlFlow::Break(res) => {
                        res?;
                        break;
                    }
                }
            }
        }
        Ok(self.into_solver_step())
    }
}

impl<S: Algo> DriverExt for BasicDriver<S> {
    type Algo = S;

    fn solver_step_mut(&mut self) -> (&mut <Self as DriverExt>::Algo, &mut Step) {
        (self.algo.as_mut().unwrap(), &mut self.step)
    }

    fn take_solver(&mut self) -> Self::Algo {
        self.algo.take().unwrap()
    }

    fn into_solver_step(self) -> (<Self as DriverExt>::Algo, Step) {
        (self.algo.unwrap(), self.step)
    }

    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        if self.timeout.is_some_and(|d| self.step.elapsed() >= d) {
            self.step.progress = Progress::Failed(DriverError::Timeout);
            return ControlFlow::Break(Err(DriverError::Timeout));
        }
        if self.iters.is_some_and(|n| self.step.iteration() >= n) {
            return ControlFlow::Break(Ok(()));
        }

        self.step.iteration += 1;
        self.step.elapsed = self.start.elapsed();

        let prog_iters = if let Some(n) = self.iters {
            Some(self.step.iteration() as f64 / n as f64).filter(|p| p.is_finite())
        } else {
            None
        };
        let prog_time = if let Some(tot) = self.timeout {
            Some(self.step.elapsed().as_secs_f64() / tot.as_secs_f64()).filter(|p| p.is_finite())
        } else {
            None
        };

        self.step.progress = match (prog_time, prog_iters) {
            (None, Some(i)) => Progress::InProgressPercentage(100.0 * i.clamp(0.0, 1.0)),
            (Some(t), None) => Progress::InProgressPercentage(100.0 * t.clamp(0.0, 1.0)),
            (Some(t), Some(i)) => {
                Progress::InProgressPercentage(100.0 * f64::max(t, i).clamp(0.0, 1.0))
            }
            (None, None) => Progress::InProgress,
        };

        let cf = self.algo.as_mut().unwrap().step();

        match cf {
            ControlFlow::Break(Ok(..)) => ControlFlow::Break(Ok(())),
            ControlFlow::Continue(Ok(..)) => ControlFlow::Continue(Ok(())),
            ControlFlow::Break(Err(e)) => {
                let e = DriverError::AlgoError(Arc::new(e));
                self.step.progress = Progress::Failed(e.clone());
                ControlFlow::Break(Err(e))
            }
            _ => unreachable!(),
        }
    }
}

impl<D, F> DriverExt for ForEachDriver<D, F>
where
    D: DriverExt,
    F: FnMut(&mut D::Algo, &Step),
{
    type Algo = D::Algo;

    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        // `?` means that errors prevent the on_step running
        let cf = self.driver.solve_once();
        match cf {
            ControlFlow::Continue(Ok(..)) => {
                let (algo, step) = self.driver.solver_step_mut();
                (self.f)(algo, step);
            }
            ControlFlow::Break(Ok(..)) => {}
            ControlFlow::Break(Err(..)) => {}
            ControlFlow::Continue(Err(..)) => unreachable!(),
        }
        cf // if res.is_some() {
    }

    fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step) {
        self.driver.solver_step_mut()
    }

    fn take_solver(&mut self) -> Self::Algo {
        self.driver.take_solver()
    }

    fn into_solver_step(self) -> (Self::Algo, Step) {
        self.driver.into_solver_step()
    }
}

impl<D, F, E> DriverExt for TryForEachDriver<D, F>
where
    D: DriverExt,
    F: FnMut(&mut D::Algo, &Step) -> Result<(), E>,
    E: Error + Sync + Send + 'static,
{
    type Algo = D::Algo;

    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        // `?` means that errors prevent the on_step running
        let _res = self.driver.solve_once()?;
        let (algo, step) = self.driver.solver_step_mut();
        let res =
            (self.f)(algo, step).map_err(|e| DriverError::AlgoError(Arc::from(e)));
        result_to_control_flow(res)?;
        ControlFlow::Continue(Ok(()))
    }

    fn take_solver(&mut self) -> Self::Algo {
        self.driver.take_solver()
    }

    fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step) {
        self.driver.solver_step_mut()
    }

    fn into_solver_step(self) -> (Self::Algo, Step) {
        self.driver.into_solver_step()
    }
}

impl<D, F> DriverExt for ConvergedWhenDriver<D, F>
where
    D: DriverExt,
    F: FnMut(&mut D::Algo, &Step) -> bool,
{
    type Algo = D::Algo;
    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        let _res = self.driver.solve_once()?;
        let (algo, step) = self.driver.solver_step_mut();
        match (self.pred)(algo, step) {
            true => {
                step.progress = Progress::Converged;
                ControlFlow::Break(Ok(()))
            }
            false => ControlFlow::Continue(Ok(())),
        }
    }

    fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step) {
        self.driver.solver_step_mut()
    }

    fn into_solver_step(self) -> (Self::Algo, Step) {
        self.driver.into_solver_step()
    }

    fn take_solver(&mut self) -> Self::Algo {
        self.driver.take_solver()
    }
}

impl<D, F> DriverExt for FailIfDriver<D, F>
where
    D: DriverExt,
    F: FnMut(&mut D::Algo, &Step) -> bool,
{
    type Algo = D::Algo;
    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        let _res = self.driver.solve_once()?;
        let (algo, step) = self.driver.solver_step_mut();
        match (self.pred)(algo, step) {
            true => {
                step.progress = Progress::Failed(DriverError::FailIfPredicate);
                ControlFlow::Break(Err(DriverError::FailIfPredicate))
            }
            false => ControlFlow::Continue(Ok(())),
        }
    }

    fn take_solver(&mut self) -> Self::Algo {
        self.driver.take_solver()
    }

    fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step) {
        self.driver.solver_step_mut()
    }

    fn into_solver_step(self) -> (Self::Algo, Step) {
        self.driver.into_solver_step()
    }
}

impl<D> DriverExt for ProgressBarDriver<D>
where
    D: DriverExt,
{
    type Algo = D::Algo;

    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        let res = self.driver.solve_once();
        let next_progress_display = self.next_progress_display;
        let (_, step) = self.solver_step_mut();
        self.next_progress_display = display_progress_bar(step, next_progress_display);
        res
    }

    fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step) {
        self.driver.solver_step_mut()
    }

    fn into_solver_step(self) -> (Self::Algo, Step) {
        self.driver.into_solver_step()
    }

    fn take_solver(&mut self) -> Self::Algo {
        self.driver.take_solver()
    }
}

impl<D> DriverExt for TolXDriver<D>
where
    D: DriverExt,
    D::Algo: TolX,
{
    type Algo = D::Algo;

    fn solve_once(&mut self) -> ControlFlow<Result<(), DriverError>, Result<(), DriverError>> {
        let _res = self.driver.solve_once()?;

        // let algo = {
        //     let (algo, _step) = self._solver_step_mut();
        //     &*algo
        // };

        let res = self
            .driver
            .solver_step_mut()
            .0
            .tol_x(self.last_x.as_ref())
            .map_err(|e| DriverError::AlgoError(Arc::new(e)));
        match res {
            Err(e) => {
                // TODO set progress = failed
                ControlFlow::Break(Err(e))
            }
            Ok((diff, last_x)) => {
                self.last_x = Some(last_x);
                if diff.is_some_and(|diff| diff < self.epsilon) {
                    self.driver.solver_step_mut().1.progress = Progress::Converged;
                    ControlFlow::Break(Ok(()))
                } else {
                    ControlFlow::Continue(Ok(()))
                }
            }
        }
    }

    fn solver_step_mut(&mut self) -> (&mut Self::Algo, &mut Step) {
        self.driver.solver_step_mut()
    }

    fn take_solver(&mut self) -> Self::Algo {
        self.driver.take_solver()
    }

    fn into_solver_step(self) -> (Self::Algo, Step) {
        self.driver.into_solver_step()
    }
}

#[cfg(test)]
mod tests {
    use crate::{fixed_iters, with_timeout, Algo, BoxedError, Driver, DriverError};
    use log::trace;
    use std::{error::Error, fmt, ops::ControlFlow, time::Duration};
    use test_log::test;

    #[derive(Debug, Clone)]
    pub struct CountingSolver(usize);

    impl CountingSolver {
        fn x(&self) -> usize {
            self.0
        }
    }

    impl Algo for CountingSolver {
        type Error = fmt::Error;

        fn step(&mut self) -> ControlFlow<Result<(), Self::Error>, Result<(), Self::Error>> {
            trace!("adding 1 to {} (MAX-{})", self.0, usize::MAX - self.0);

            let res = self.0.checked_add(1).ok_or(fmt::Error);
            match res {
                Ok(v) => self.0 = v,
                Err(e) => return ControlFlow::Break(Err(e)),
            };
            ControlFlow::Continue(Ok(()))
        }
    }

    // impl TolX for CountingSolver {
    //     type X = usize;
    //     type Error = std::fmt::Error;
    //     fn tol_x(&self, prior: Option<&usize>) -> Result<(Option<f64>, usize), std::fmt::Error> {
    //         let current_x = self.0;
    //         if let Some(prior) = prior {
    //             let diff = ((current_x as f64 - *prior as f64) / (1e-10 + current_x as f64)).abs();
    //             Ok((Some(diff), current_x))
    //         } else {
    //             Ok((None, current_x))
    //         }
    //     }
    // }

    #[test]
    fn functional_fixed_iters() -> Result<(), Box<dyn Error>> {
        let algo = CountingSolver(3);
        let (algo, step) = fixed_iters(algo, 5).solve()?;
        assert_eq!(algo.x(), 8);
        assert_eq!(step.iteration(), 5);
        Ok(())
    }

    #[test]
    fn functional_timeout() {
        let algo = CountingSolver(0);
        let result = with_timeout(algo, Duration::from_micros(5)).solve();
        assert!(result.is_err_and(|e| matches!(e, DriverError::Timeout)));
    }

    #[test]
    fn functional_values() -> Result<(), Box<dyn Error + Send + Sync>> {
        let algo = CountingSolver(10);
        let mut for_each_count = 0;
        // step.iteration() is 1..=3
        let _ = fixed_iters(algo, 3)
            .on_step(|v, s| {
                println!(
                    "[{:>2}] x: {} status: {:?}",
                    s.iteration(),
                    v.x(),
                    s.progress
                )
            })
            .on_step(|v, s| for_each_count += s.iteration() * v.x())
            .solve()?;

        #[allow(clippy::erasing_op, clippy::identity_op)]
        let expected = 1 * 11 + 2 * 12 + 3 * 13;
        assert_eq!(for_each_count, expected);
        Ok(())
    }

    #[test]
    fn functional_mutate() {
        let algo = CountingSolver(10);
        let (solved, _step) = fixed_iters(algo, 3)
            .on_step(|v, s| v.0 += s.iteration())
            .on_step(|v, s| println!("{} {}", s.iteration(), v.x()))
            .solve()
            .unwrap();

        #[allow(clippy::erasing_op, clippy::identity_op)]
        let expected = 10 + 1 + 1 + 1 + 2 + 1 + 3;
        assert_eq!(solved.0, expected);
    }

    #[test]
    fn functional_convergence() -> Result<(), Box<dyn Error>> {
        let algo = CountingSolver(0);
        let (algo, _step) = fixed_iters(algo, 10)
            .converge_when(|v, _s| v.x() == 5)
            .solve()?;
        assert_eq!(algo.x(), 5);
        Ok(())
    }

    // TODO: work out what to do with TolX
    // #[test]
    // fn functional_tolx() -> Result<(), Box<dyn Error>> {
    //     let algo = CountingSolver(0);
    //     let (algo, _step) = fixed_iters(algo, 20_000).tol_x(0.015).solve()?;
    //     assert_eq!(algo.x(), 67);
    //     Ok(())
    // }

    #[test]
    fn functional_solver_error() {
        let algo = CountingSolver(usize::MAX - 2);
        let mut for_each_count = 0;
        let step = fixed_iters(algo, 10)
            .on_step(|_, _| for_each_count += 1)
            .on_step(|v, s| println!("{} {}", s.iteration(), v.x()))
            .solve();
        assert!(step.is_err());
        assert_eq!(for_each_count, 2);
        assert!(step.is_err_and(|e| matches!(e, DriverError::AlgoError(_))));
    }

    #[test]
    fn functional_try_error() {
        let algo = CountingSolver(0);
        let mut for_each_count = 0;
        let result = fixed_iters(algo, 10)
            .try_on_step(|_, _| {
                if for_each_count < 5 {
                    for_each_count += 1;
                    Ok(())
                } else {
                    Err(std::fmt::Error)
                }
            })
            .solve();
        assert!(result.is_err());
        assert_eq!(for_each_count, 5);
        assert!(result.is_err_and(|e| matches!(e, DriverError::AlgoError(_))));
    }

    #[test]
    fn functional_fail_if() {
        let algo = CountingSolver(0);
        let result = fixed_iters(algo, 10).fail_if(|algo, _| algo.0 > 5).solve();
        assert!(result.is_err(), "{result:?}");
        assert!(result.is_err_and(|e| matches!(e, DriverError::FailIfPredicate)));
    }

    #[test]
    fn functional_progress_bar_fixed_iters() {
        println!("progress bar always fixed_iters...");
        let algo = CountingSolver(0);
        let (solved, _step) = fixed_iters(algo, 30)
            .show_progress_bar_after(Duration::ZERO)
            .on_step(|_, _| std::thread::sleep(Duration::from_millis(10)))
            .solve()
            .unwrap();
        assert_eq!(solved.x(), 30);
        println!("...done");
    }

    #[test]
    fn functional_progress_bar_timeout() {
        println!("progress bar always fixed_time...");
        let algo = CountingSolver(0);
        let failed = with_timeout(algo, Duration::from_millis(300))
            .show_progress_bar_after(Duration::ZERO)
            .on_step(|_, _| std::thread::sleep(Duration::from_millis(10)))
            .solve();
        assert!(matches!(failed.unwrap_err(), DriverError::Timeout));
        println!("...done");
    }

    #[test]
    fn while_loop_completes() {
        let solv = CountingSolver(3);
        let mut driver = fixed_iters(solv, 5);
        while let Ok(Some((algo, step))) = driver.iter_step() {
            println!("{step:?} x: {:?}", algo.x());
        }
        let (solved, step) = driver.solve().unwrap();
        assert_eq!(step.iteration(), 5);
        assert_eq!(solved.0, 8);
    }

    #[test]
    fn while_loop_error() {
        let algo = CountingSolver(usize::MAX - 2);
        let mut for_each_count = 0;
        let mut driver = fixed_iters(algo, 10);
        while let Ok(Some((_solver, _step))) = driver.iter_step() {
            for_each_count += 1;
        }
        let step = driver.solve();
        assert!(step.is_err());
        assert_eq!(for_each_count, 2);
        assert!(step.is_err_and(|e| matches!(e, DriverError::AlgoError(_))));
    }

    #[test]
    fn while_loop_break() -> Result<(), BoxedError> {
        let algo = CountingSolver(0);
        let mut driver = fixed_iters(algo, 100);
        while let Some((_v, s)) = driver.iter_step()? {
            if s.iteration() == 10 {
                break;
            }
        }
        let (solved, step) = driver.solve()?;
        assert_eq!(step.iteration(), 10);
        assert_eq!(solved.0, 10);
        Ok(())
    }

    #[test]
    fn dyn_compat_boxing() -> Result<(), BoxedError> {
        fn test_trait<D: Driver>(_d: &D) {}
        let driver1 = fixed_iters(CountingSolver(10), 100).on_step(|_, _| ());
        let driver2 = fixed_iters(CountingSolver(20), 130);
        let driver3 = fixed_iters(CountingSolver(10), 100).on_step(|_, _| ());
        let driver4 = fixed_iters(CountingSolver(20), 130);

        let boxed_driver1: Box<dyn Driver<Algo = CountingSolver>> = Box::new(driver1);
        let boxed_driver2: Box<dyn Driver<Algo = CountingSolver>> = Box::new(driver2);

        let boxed_driver3 = driver3.into_boxed();
        let boxed_driver4 = driver4.into_boxed();

        let vec = vec![boxed_driver1, boxed_driver2, boxed_driver3, boxed_driver4];
        for (i, d) in vec.into_iter().enumerate() {
            test_trait(&d);
            let (algo, step) = d.on_step(|_v, _s| ()).solve().unwrap();
            match i {
                0 | 2 => {
                    assert_eq!(step.iteration(), 100);
                    assert_eq!(algo.x(), 110);
                }
                1 | 3 => {
                    assert_eq!(step.iteration(), 130);
                    assert_eq!(algo.x(), 150);
                }
                _ => unreachable!(),
            }
            // assert_eq!(solved.0, 10);
        }
        Ok(())
    }

    #[test]
    fn dyn_if_then() -> Result<(), BoxedError> {
        let log_enabled = true;
        let algo = CountingSolver(10);
        let driver = if log_enabled {
            fixed_iters(algo, 100)
                .on_step(|_, step| println!("{step:?}"))
                .into_boxed()
        } else {
            fixed_iters(algo, 100).into_boxed()
        };

        let (algo, _step) = driver.solve()?;
        assert_eq!(algo.x(), 110);
        Ok(())
    }
}
