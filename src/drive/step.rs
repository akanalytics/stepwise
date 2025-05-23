use crate::{
    algo::{FileReader, FileWriter},
    FileFormat,
};
use std::{
    fmt::{Debug, Display},
    io,
    sync::OnceLock,
    time::{Duration, Instant},
};

#[derive(Debug, Clone)]
pub(crate) enum Progress {
    // FailedPredicate,
    // FailedTimeout,
    // MaxIterationsExceeded,
    // FailedAlgoError,
    NotStarted,
    // Converged,

    // InProgressPercentage(f64),
    // #[allow(clippy::enum_variant_names)]
    // InProgress,
}

impl Default for Progress {
    fn default() -> Self {
        Self::NotStarted
    }
}

// fn assert_send_sync<T: Send + Sync>() {
//     assert_send_sync::<Progress>();
// }

/// A small struct that holds iteration count, elapsed time etc for each algorithm step.
/// The elapsed time is calculated on first call or on [`Clone`].
///
/// The callbacks [`on_step`](crate::Driver::on_step) and [`converge_when`](crate::Driver::converge_when)
/// give access to a `Step` structure.
#[derive(Debug, PartialEq, Eq)]
pub struct Step {
    pub(crate) iteration: usize,
    solving_start: Instant,
    pub(crate) elapsed: OnceLock<Duration>,
    pub(crate) iter_limit: usize,
    pub(crate) time_limit: Option<Duration>,
}

impl Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:>6}]", self.iteration())?;
        write!(f, " [{}]", format_duration(self.elapsed()))?;
        Ok(())
    }
}

/// Clone but with the side-effect of crystalizing the [`elapsed`](Self::elapsed()) time
///
/// Elapsed is calculated at clone time, if it has not already been calculated. This
/// is typically want you want, but if the overhead of calculating the time is not desired,
/// and using the elapsed time is not needed, then [`Step::clone_without_elapsed`] can be called
///
impl Clone for Step {
    fn clone(&self) -> Self {
        // freeze the elapsed time
        let _elapsed = self.elapsed();
        Self {
            iteration: self.iteration,
            solving_start: self.solving_start,
            elapsed: self.elapsed.clone(),
            iter_limit: self.iter_limit,
            time_limit: self.time_limit,
        }
    }
}

impl FileWriter for Step {
    fn write_file(&self, fmt: &FileFormat, w: &mut dyn io::Write) -> Result<(), io::Error> {
        if !matches!(fmt, FileFormat::Control) {
            return Err(fmt.unsupported_error());
        }
        writeln!(w, "{}", self.iteration())
    }
}

impl FileReader for Step {
    fn read_file(&mut self, fmt: &FileFormat, r: &mut dyn io::Read) -> Result<(), io::Error> {
        if !matches!(fmt, FileFormat::Control) {
            return Err(fmt.unsupported_error());
        }
        let mut buf = String::new();
        r.read_to_string(&mut buf)?;
        // remove line end
        let buf = buf.trim_end();
        self.iteration = buf.parse().map_err(|_| {
            let kind = io::ErrorKind::InvalidData;
            let msg = format!("'{buf}' could not be parsed");
            io::Error::new(kind, msg)
        })?;
        Ok(())
    }
}

impl Step {
    pub(crate) fn new(iteration: usize, start: Instant) -> Self {
        Step {
            iteration,
            solving_start: start,
            elapsed: OnceLock::new(),
            iter_limit: usize::MAX,
            time_limit: None,
        }
    }

    #[inline]
    pub(crate) fn reset_elapsed(&mut self) {
        self.elapsed = OnceLock::new();
    }

    ///  The iteration count.
    ///
    /// Represents how many times 'step' has been invoked. Hence the first iteration is 1.
    #[inline]
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Total elapsed time since the call to [`crate::Driver::solve`].
    ///
    /// The elapsed time is only
    /// updated after each iteration step, and will not change between calls otherwise.
    ///
    /// # See also
    /// - [`crate::with_timeout`]
    /// - [`format_duration`]
    #[inline]
    pub fn elapsed(&self) -> Duration {
        *self.elapsed.get_or_init(|| self.solving_start.elapsed())
    }

    /// An estimate between `0` and `100` of how complete the solving process is.
    ///
    /// The most optimistic of
    /// - iterations out of iters, or
    /// - elapsed time out of of "fail after" duration
    ///
    /// will be used for the estimate.
    #[inline]
    pub fn progress_percentage(&self) -> Option<f64> {
        let prog_iters = if self.iter_limit != usize::MAX {
            Some(self.iteration() as f64 / self.iter_limit as f64).filter(|p| p.is_finite())
        } else {
            None
        };

        // only call elapsed (expensive) of timeout is set
        let prog_time = if let Some(tot) = self.time_limit {
            Some(self.elapsed().as_secs_f64() / tot.as_secs_f64()).filter(|p| p.is_finite())
        } else {
            None
        };

        // optimistically take the max of the estimates
        match (prog_iters, prog_time) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
        .map(|x| 100.0 * x.clamp(0.0, 1.0))
    }

    /// A version of [`Self::clone`] that does not crystalize the elapsed time at clone time,
    /// and hence is slightly more performant.
    ///
    pub fn clone_without_elapsed(&self) -> Self {
        Step {
            iteration: self.iteration,
            solving_start: self.solving_start,
            elapsed: self.elapsed.clone(),
            iter_limit: self.iter_limit,
            time_limit: self.time_limit,
        }
    }

    /// An estimate of the time remaining.
    ///
    /// The estimate is based on [`Self::progress_percentage`], and how much time has elapsed so far.
    pub fn time_remaining(&self) -> Option<Duration> {
        let est1 = self
            .time_limit
            .map(|time_limit| time_limit.saturating_sub(self.elapsed()));
        // try and refine using fixed_iters if possible
        let est2 = if self.iter_limit != usize::MAX {
            let time_remaining = (self.iter_limit as f64 / self.iteration() as f64 - 1.0)
                * self.elapsed().as_secs_f64();
            Duration::try_from_secs_f64(time_remaining).ok()
        } else {
            None
        };
        // optimistically take the min of the estimates
        match (est1, est2) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) => Some(a),
            (None, Some(b)) => Some(b),
            (None, None) => None,
        }
    }
}

/// Formats a duration in the form `05h-02m-07s-045ms`
///
/// eg 5 hours 12 mins 17 secs and 345 millis would display as
///     `05h-12m-17s-345ms`
pub fn format_duration(d: Duration) -> impl Display {
    struct DurationDisplay(Duration);
    impl Display for DurationDisplay {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let total_seconds = self.0.as_secs();
            let millis = self.0.subsec_millis();
            let hours = total_seconds / 3600;
            let mins = (total_seconds % 3600) / 60;
            let secs = total_seconds % 60;
            write!(f, "{:02}h-{:02}m-{:02}s-{:03}ms", hours, mins, secs, millis)
        }
    }
    DurationDisplay(d)
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    #[test]
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn step_sizes() {
        assert_eq!(size_of::<Vec<f64>>(), 24);
    }
}
