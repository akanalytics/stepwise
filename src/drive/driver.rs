use super::{
    checkpointing::{CheckpointData, Recovery},
    dynamic::DynAlgo,
};
use crate::{
    algo::{FileReader, FileWriter},
    log_trace,
    samplers::ProgressBar,
    Algo, DriverError, FileFormat, Step,
};
use std::{
    error::Error,
    fmt::{self, Debug},
    io,
    ops::ControlFlow,
    path::Path,
    sync::Arc,
    time::{Duration, Instant},
};

/// Constructs a [`Driver`](super::Driver) with a fixed number of iterations
///
/// After this number of itersations the algo is deemed to have converged on a solution.
/// If [`Driver::converge_when`](super::Driver::converge_when) is used, then iteration may terminate early.
///
/// Logically it  applies [`set_fixed_iters`](super::Driver::set_fixed_iters) to a new Driver
///
pub fn fixed_iters<A: Algo>(
    algo: A,
    fixed_iters: usize,
) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
    unlimited(algo).set_fixed_iters(fixed_iters)
}

fn unlimited<A: Algo>(algo: A) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
    let f = |_: &mut A, _: &Step| Ok(false);
    Drive::new(algo, State::default(), f)
}

/// Constructs a [`Driver`](super::Driver) that will fail once a certain number of iterations is reached.
///
/// Logically it  applies [`set_fail_after_iters`](super::Driver::set_fail_after_iters) to a new Driver
///
pub fn fail_after_iters<A: Algo>(
    algo: A,
    max_iters: usize,
) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
    unlimited(algo).set_fail_after_iters(max_iters)
}

/// Constructs a [`Driver`](super::Driver) that will fail after an elased duration.
///
/// Logically it applies [`set_timeout`](super::Driver::set_timeout) to a new Driver
///
pub fn with_timeout<A: Algo>(
    algo: A,
    timeout: Duration,
) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
    unlimited(algo).set_timeout(timeout)
}

/// Determines recovery behaviour for [`Driver::recovery`].
///
/// Typically you dont want to **always** recover from the last checkpoint file, only in cases where the iterative process was terminated or crashed. This
/// entails having a recovery file location different to the checkpoint location, and some manual process to move/rename files from
/// where they are written, to the recovery path location.
///
/// | Item | Description |
/// | :--- | :--- |
/// | [`RecoveryFile::Ignore`] | Never recover from a checkpoint file. |
/// | [`RecoveryFile::Require`]    | Checkpoint file must exist, error if missing. Probably useful only if set from a command line option or configuration. Used if the user manually wants to move a checkpoint file into the specified recovery file locatiton. |
/// | [`RecoveryFile::UseIfExists`] | Recover from checkpoint if the file exists, otherwise solve afresh. Perhaps a sensible default, but susceptible to typo's in recovery filename, if the checkpoint file was manually moved/renamed. |
///
///
///
#[derive(Debug, Clone, Hash, PartialEq)]
pub enum RecoveryFile {
    UseIfExists,
    Ignore,
    Require,
}

pub struct State {
    recovery: Option<Recovery>,
    pub(crate) step: Step,
    last_error: Option<DriverError>,
}

pub struct Drive<F, A> {
    state: State,
    algo: A,
    f: F,
}

pub type BoxedDriver<A> = Drive<Box<dyn FnMut(&mut A, &Step) -> Result<bool, DriverError>>, A>;
pub type DynDriver = BoxedDriver<DynAlgo>;

impl Default for State {
    fn default() -> Self {
        let step = Step::new(0, Instant::now());
        Self {
            recovery: None,
            step,
            last_error: None,
        }
    }
}

impl FileWriter for State {
    fn write_file(&self, fmt: &FileFormat, w: &mut dyn io::Write) -> Result<(), io::Error> {
        if !matches!(fmt, FileFormat::Control) {
            return Err(fmt.unsupported_error());
        }
        self.step.write_file(fmt, w)?;
        Ok(())
    }
}

impl FileReader for State {
    fn read_file(&mut self, fmt: &FileFormat, r: &mut dyn io::Read) -> Result<(), io::Error> {
        if !matches!(fmt, FileFormat::Control) {
            return Err(fmt.unsupported_error());
        }
        // preserve rest of State, only overwriting the step
        self.step.read_file(fmt, r)?;
        Ok(())
    }
}

impl<F, A> Drive<F, A> {
    fn new(algo: A, state: State, f: F) -> Self {
        log_trace!(
            "[     ] Driver created with algo {}",
            std::any::type_name_of_val(&algo)
        );

        Drive { algo, state, f }
    }
}

impl<F, A> Debug for Drive<F, A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Drive2")
            .field("algo", &"<opaque>")
            .field("step", &self.state.step)
            .field("iter_limit", &self.state.step.iter_limit)
            .field("time_limit", &self.state.step.time_limit)
            .finish()
    }
}

impl<F, A> Drive<F, A>
where
    F: FnMut(&mut A, &Step) -> Result<bool, DriverError>,
    A: Algo,
{
    pub fn solve(mut self) -> Result<(A, Step), DriverError> {
        if let Some(e) = self.state.last_error.take() {
            return Err(e);
        }
        if self.state.step.iteration > 0 {
            return Ok((self.algo, self.state.step));
        } else if let Some(recovery) = &self.state.recovery {
            // need to clone the recovery so b/c knows we no longer hold &state
            let recovery = recovery.clone();
            recovery.recover(&mut self.algo, &mut self.state)?;
            // execute the closure that never got executed due to checkpoint being written BEFORE closure exec
            log_trace!(
                "[{:>5}] executing (skipped) closure ...",
                self.state.step.iteration()
            );
            let done = (self.f)(&mut self.algo, &mut self.state.step)?;
            if done {
                return Ok((self.algo, self.state.step));
            }
        }
        loop {
            self.state.step.reset_elapsed();
            self.state.step.iteration += 1;
            log_trace!("[{:>5}] executing algo step", self.state.step.iteration());

            match self.algo.step() {
                (ControlFlow::Continue(_), Ok(_)) => {}
                (ControlFlow::Break(_), Ok(_)) => return Ok((self.algo, self.state.step)),
                (_, Err(e)) => {
                    let e = DriverError::AlgoError(Arc::new(e));
                    self.state.last_error = Some(e.clone());
                    return Err(e);
                }
            }
            log_trace!("[{:>5}] executing closure ...", self.state.step.iteration());
            let done = (self.f)(&mut self.algo, &mut self.state.step)?;
            if done {
                return Ok((self.algo, self.state.step));
            }
        }
    }

    pub fn iter_step(&mut self) -> Result<Option<(&mut A, &Step)>, DriverError> {
        if let Some(e) = self.state.last_error.take() {
            return Err(e);
        }
        self.state.step.reset_elapsed();
        self.state.step.iteration += 1;
        match self.algo.step() {
            (ControlFlow::Continue(_), Ok(_)) => {}
            (ControlFlow::Break(_), Ok(_)) => return Ok(None),
            (_, Err(e)) => {
                let e = DriverError::AlgoError(Arc::new(e));
                self.state.last_error = Some(e.clone());
                return Err(e);
            }
        }
        let done = (self.f)(&mut self.algo, &mut self.state.step)?;
        if done {
            Ok(None)
        } else {
            Ok(Some((&mut self.algo, &self.state.step)))
        }
    }

    #[allow(refining_impl_trait)]
    pub fn on_step<G>(
        mut self,
        mut g: G,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A>
    where
        G: FnMut(&mut A, &Step),
    {
        let f = move |a: &mut A, s: &Step| -> Result<bool, DriverError> {
            let b = (self.f)(a, s);
            (g)(a, s);
            b
        };
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn try_on_step<G, E>(
        mut self,
        mut g: G,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A>
    where
        Self: Sized,
        G: FnMut(&mut A, &Step) -> Result<(), E>,
        E: Error + Sync + Send + 'static,
    {
        let f = move |a: &mut A, s: &Step| -> Result<bool, DriverError> {
            let b = (self.f)(a, s);
            (g)(a, s).map_err(|e| DriverError::AlgoError(Arc::new(e)))?;
            b
        };
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn converge_when<G>(
        mut self,
        mut pred: G,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A>
    where
        G: FnMut(&mut A, &Step) -> bool,
    {
        let f = move |a: &mut A, s: &Step| -> Result<bool, DriverError> {
            Ok((self.f)(a, s)? || (pred)(a, s))
        };
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn fail_if<G>(
        mut self,
        mut pred: G,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A>
    where
        G: FnMut(&mut A, &Step) -> bool,
    {
        let f = move |a: &mut A, step: &Step| -> Result<bool, DriverError> {
            Ok((self.f)(a, step)?
                || if (pred)(a, step) {
                    Err(DriverError::FailIfPredicate)?
                } else {
                    false
                })
        };
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn show_progress_bar_after(
        mut self,
        after: Duration,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
        let mut pb = ProgressBar::after(after).on_stdout();
        let f = move |a: &mut A, s: &Step| {
            let b = (self.f)(a, s)?;
            pb.sample(s);
            Ok(b)
        };
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn set_fixed_iters(
        mut self,
        fixed_iters: usize,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
        self.state.step.iter_limit = fixed_iters;
        let f = move |a: &mut A, s: &Step| Ok((self.f)(a, s)? || s.iteration() >= fixed_iters);
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn set_fail_after_iters(
        mut self,
        max_iters: usize,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
        let f = move |a: &mut A, s: &Step| {
            Ok((self.f)(a, s)?
                || if s.iteration() >= max_iters {
                    Err(DriverError::MaxIterationsExceeded)?
                } else {
                    false
                })
        };
        self.state.step.iter_limit = max_iters;
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn set_timeout(
        mut self,
        timeout: Duration,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
        let f = move |a: &mut A, s: &Step| {
            Ok((self.f)(a, s)?
                || if s.elapsed() >= timeout {
                    Err(DriverError::Timeout)?
                } else {
                    false
                })
        };
        self.state.step.time_limit = Some(timeout);
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn checkpoint(
        mut self,
        path: impl AsRef<Path>,
        every: Duration,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
        let mut checkpoint = CheckpointData {
            path: path.as_ref().to_path_buf(),
            every,
            next_elapsed: every,
        };
        let f = move |a: &mut A, s: &Step| {
            checkpoint.checkpoint(a, s)?;
            let b = (self.f)(a, s)?;
            Ok(b)
        };
        Drive::new(self.algo, self.state, f)
    }

    #[allow(refining_impl_trait)]
    pub fn recovery(
        mut self,
        path: impl AsRef<Path>,
        action: RecoveryFile,
    ) -> Drive<impl FnMut(&mut A, &Step) -> Result<bool, DriverError>, A> {
        let recovery = Recovery {
            path: path.as_ref().to_path_buf(),
            action,
        };
        self.state.recovery = Some(recovery);
        self
    }

    pub fn into_boxed(self) -> BoxedDriver<A>
    where
        Self: Sized + 'static,
    {
        Drive::new(self.algo, self.state, Box::new(self.f))
    }

    pub fn into_dyn(mut self) -> BoxedDriver<DynAlgo>
    where
        Self: Sized + 'static,
    {
        let f = move |a: &mut DynAlgo, s: &Step| (self.f)(a.downcast_mut().unwrap(), s);
        Drive::new(DynAlgo::new(self.algo), self.state, Box::new(f))
    }
}

/// An executor for the algorithm controlling max iterations, execution time, convergence criteria, and callbacks.
pub trait Driver {
    /// The underlying algorithm being driven.
    type Algo;

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
    /// #
    /// let (solved, step) = fixed_iters(my_algo, 1000)
    ///     .solve()?;
    ///
    /// assert_approx_eq!(solved.x(),  &vec![1.5, 2.0] );
    /// assert_eq!(step.iteration(), 1000 );
    /// # Ok(())
    /// # }
    /// ```
    fn solve(self) -> Result<(Self::Algo, Step), DriverError>;

    fn iter_step(&mut self) -> Result<Option<(&mut Self::Algo, &Step)>, DriverError>;

    /// Invoked after each iteration, allowing printing or debugging of the iteration [Step]. Any errors
    /// encountered by the underlying algorithm will terminate the solving, and [`Self::solve`] will return the error.
    ///
    /// Can be used to capture details of the iteration, or update hyperparameters on the algorithm for adaptive learning.
    ///
    /// # Example
    /// ```
    /// # use stepwise::algos::GradientDescent;
    /// # use stepwise::{Driver, fixed_iters};
    /// # use stepwise::assert_approx_eq;
    /// # fn dummy<G: Fn(&[f64]) -> Vec<f64>>(my_algo: GradientDescent<G>) -> Result<(), stepwise::DriverError> {
    /// #
    /// let (solved, _step) = fixed_iters(my_algo, 1000)
    ///     .on_step(|_algo, step| println!("{step:?}"))
    ///     .solve()?;
    /// # Ok(())
    /// # }
    /// ```
    fn on_step<G>(self, g: G) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        G: FnMut(&mut Self::Algo, &Step);

    /// Invoked after each iteration, allowing printing or debugging of the iteration [step](Step).
    /// Can be used to capture details of the iteration. See [`Self::try_on_step`].
    fn try_on_step<F, E>(self, f: F) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> Result<(), E>,
        E: Error + Sync + Send + 'static;

    /// Used for early stopping.
    ///  
    /// Common convergence predicates are step size below a small epsilon, or residuals being below near zero.
    /// Some specific convergence criteria are best handled by using [`metrics`](crate::metrics) within this callback.
    fn converge_when<F>(self, pred: F) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> bool;

    /// Decide when to abandon iteration with [`Self::solve`] returning [`DriverError::FailIfPredicate`].
    ///
    /// Convergence predicates are tested first.
    ///
    /// Common failure predicates are residuals growing in size after an initial set of iterations,
    /// or user cancelation - implemented perhaps by the closure predicate
    /// checking the value of an `AtomicBool`.
    ///
    fn fail_if<F>(self, pred: F) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> bool;

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
    fn show_progress_bar_after(self, after: Duration) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized;

    /// After this number of iterations the algo is deemed to have *converged* on a solution.
    /// If [`Driver::converge_when`](super::Driver::converge_when) is used, then iteration may terminate early.
    ///
    /// Contrast with [`Driver::set_fail_after_iters`](super::Driver::set_fail_after_iters)
    ///
    fn set_fixed_iters(self, fixed_iters: usize) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized;

    /// Unless convergence has occurred, the driver will error after the designated iteration count.
    /// Either the algo has internal logic to terminate iteration when a solution is found,
    /// or the caller needs to use [`Driver::converge_when`](super::Driver::converge_when)
    /// to ensure iteration stops before a [`max iterations exceeded error`](DriverError::MaxIterationsExceeded).
    ///
    /// Contrast with [`Driver::set_fixed_iters`](super::Driver::set_fixed_iters)
    ///
    fn set_fail_after_iters(self, max_iters: usize) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized;

    /// Either the algo
    /// has internal logic to terminate iteration when a solution is found, or the caller
    /// needs to use [`Driver::converge_when`](super::Driver::converge_when)
    /// to ensure iteration stops before a (timeout error)[DriverError::Timeout].
    ///
    /// Any previous timeout will be overwritten
    fn set_timeout(self, timeout: Duration) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized;

    /// For writing checkpoint files, for possible later recovery.
    ///
    /// Checkpoints are useful for very long running iterations, where restarting from 'last checkpoint' in cases of network error or crash,
    /// is a timesaver.
    ///
    /// The path can contain replaceable tokens. Note that if the path is generated by a `format!` macro then the `{` will need to be
    /// escaped by doubling `{{`.
    /// - {iter} : current iteration number eg. 0000745
    /// - {elapsed} : current elapsed time in hours, mins, secs and millis eg 00h-34m-12s-034ms [`format_duration`](crate::format_duration)
    /// - {pid} : process ID of the running process
    ///
    /// The file format is determined by the file extension of the supplied path.      
    ///
    /// There will typically be a file written for the algorithm, and a `.control` file for the `Driver` state. Both are needed for recovery.
    /// The control file's location is the path and file name supplied, but with the extension set to '.control'.
    ///
    ///
    /// Which file formats (if any) are supported is algorithm dependent.
    ///
    /// # Example:  
    ///
    /// ```
    /// # use stepwise::prelude::*;
    /// # use std::time::Duration;
    /// # fn dummy<G: Fn(&[f64]) -> Vec<f64>>(my_algo: algos::GradientDescent<G>) -> Result<(), DriverError> {
    /// #
    /// let my_program_name = "algo-solver";
    /// let every = Duration::from_secs(60);
    /// let (solved, _) = fixed_iters(my_algo, 2)
    ///     .checkpoint(format!("/tmp/checkpoint-{my_program_name}-{{iter}}.json"), every)
    ///     .solve()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    fn checkpoint(self, path: impl AsRef<Path>, every: Duration) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized;

    /// For recovery from checkpoint files. Specifies the location of files used to recover algorithm and driver state.
    ///
    /// Typically you dont want to **always** recover from the last checkpoint file, only in cases where the iterative process was terminated. This
    /// entails having a recovery file location different to the checkpoint location, and some manual process to move/rename files from
    /// where they are written, to the recovery path location. See [`RecoveryFile`].
    ///
    /// There will typically be a file for the algorithm, and a ".control" file for the `Driver` state. Both are needed.
    ///
    /// Note that metrics outside the driver, such as moving averages will be re-initialized, even though the algorithm is `recovered`.
    ///
    /// Which file formats (if any) are supported is algorithm dependent.
    ///
    /// # Example:  
    ///
    /// ```
    /// # use stepwise::prelude::*;
    /// # use std::time::Duration;
    /// # fn dummy<G: Fn(&[f64]) -> Vec<f64>>(my_algo: algos::GradientDescent<G>) -> Result<(), DriverError> {
    /// # let command_line = "";
    /// let every = Duration::from_secs(60);
    ///
    /// let action = if command_line == "recover" {
    ///     RecoveryFile::Require
    /// } else {
    ///     RecoveryFile::Ignore
    /// };
    /// let (solved, _) = fixed_iters(my_algo, 2)
    ///     .recovery("/tmp/checkpoint-to-use.json", action)
    ///     .checkpoint(format!("/tmp/checkpoint-{{iter}}.json"), every)
    ///     .solve()?;
    /// # Ok(())
    /// # }
    /// ```
    fn recovery(
        self,
        path: impl AsRef<Path>,
        action: RecoveryFile,
    ) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized;

    /// Converts the `Driver` into a sized type ([`BoxedDriver<A>`]), but remains generic over the algorithm.
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
    /// // driver is type BoxedDriver<MyAlgo>
    /// let (my_algo, _step) = driver.show_progress_bar_after(Duration::ZERO).solve()?;
    /// # Ok(())}
    /// ```
    fn into_boxed(self) -> BoxedDriver<Self::Algo>
    where
        Self: Sized + 'static;

    /// Converts the `Driver` into a sized type without generics ([`DynDriver`])
    ///
    /// The prefered was of having common routines pass a Driver as a parameter
    /// is to pass Driver by `impl Driver`.
    /// But occasionally a concrete implementation is required.
    ///
    /// Whereas [`BoxedDriver<A>`] is generic over the algorithm, [`DynDriver`] is a concrete type
    /// and can be stored in non-generic structs.
    /// To call methods on `DynAlgo` you will need to downcast to the specific Algorithm used,
    /// via `DynAlgo::downcast_ref` or `DynAlgo::downcast_mut`
    ///
    /// Since closures cannot be named, if the algorithm is generic over closure/functions
    /// you will need to Box the closure. See examples folder.
    ///
    /// # Example:
    /// ```
    /// # use algos::GradientDescent;
    /// # use std::time::Duration;
    /// # use stepwise::{prelude::*, problems::sphere_grad};
    /// // note we need a concrete *function* to name and downcast to
    /// type GradFn = fn(&[f64]) -> Vec<f64>;
    ///
    /// let lr = 0.1;
    /// let my_closure = |x: &[f64]| vec![2.0 * sphere_grad(x)[0], sphere_grad(x)[1]];
    ///
    /// // coerce the closure into a function (must not capture from environment)
    /// let grad: GradFn = my_closure;
    ///
    /// let gd = GradientDescent::new(lr, vec![5.55, 5.55], grad);
    ///
    /// let (mut dyn_solved, _step) = fixed_iters(gd.clone(), 500)
    ///     .show_progress_bar_after(Duration::ZERO)
    ///     .into_dyn()
    ///     .solve()
    ///     .unwrap();
    ///
    ///
    /// let gd_algo: &GradientDescent<GradFn> =
    ///     dyn_solved.downcast_ref().expect("failed to downcast_ref");
    ///
    /// let solution = gd_algo.x();
    /// assert_approx_eq!(solution, &[0.0, 0.0]);
    ///```
    ///
    fn into_dyn(self) -> DynDriver
    where
        Self: Sized + 'static,
        Self::Algo: Algo;
}

impl<G, A> Driver for Drive<G, A>
where
    G: FnMut(&mut A, &Step) -> Result<bool, DriverError>,
    A: Algo,
{
    type Algo = A;

    fn solve(self) -> Result<(Self::Algo, Step), DriverError> {
        self.solve()
    }
    fn iter_step(&mut self) -> Result<Option<(&mut Self::Algo, &Step)>, DriverError> {
        self.iter_step()
    }

    fn on_step<F>(self, f: F) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step),
    {
        self.on_step(f)
    }

    fn try_on_step<F, E>(self, f: F) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> Result<(), E>,
        E: Error + Sync + Send + 'static,
    {
        self.try_on_step(f)
    }

    fn converge_when<F>(self, pred: F) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> bool,
    {
        self.converge_when(pred)
    }

    fn fail_if<F>(self, pred: F) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
        F: FnMut(&mut Self::Algo, &Step) -> bool,
    {
        self.fail_if(pred)
    }

    fn show_progress_bar_after(self, after: Duration) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
    {
        self.show_progress_bar_after(after)
    }

    fn set_fixed_iters(self, fixed_iters: usize) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
    {
        self.set_fixed_iters(fixed_iters)
    }

    fn set_fail_after_iters(self, max_iters: usize) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
    {
        self.set_fail_after_iters(max_iters)
    }

    fn set_timeout(self, timeout: Duration) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
    {
        self.set_timeout(timeout)
    }

    fn checkpoint(self, path: impl AsRef<Path>, every: Duration) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
    {
        self.checkpoint(path, every)
    }

    fn recovery(
        self,
        path: impl AsRef<Path>,
        action: RecoveryFile,
    ) -> impl Driver<Algo = Self::Algo>
    where
        Self: Sized,
    {
        self.recovery(path, action)
    }

    fn into_boxed(self) -> BoxedDriver<Self::Algo>
    where
        Self: Sized + 'static,
    {
        self.into_boxed()
    }

    fn into_dyn(self) -> BoxedDriver<DynAlgo>
    where
        Self: Sized + 'static,
    {
        self.into_dyn()
    }
}
