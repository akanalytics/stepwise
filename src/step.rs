use std::{
    fmt::{Debug, Display},
    time::Duration,
};

use crate::DriverError;

#[derive(Debug, Clone)]
pub(crate) enum Progress {
    Failed(DriverError),
    Converged,
    InProgressPercentage(f64),
    #[allow(clippy::enum_variant_names)]
    InProgress,
    NotStarted,
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
///
/// The callbacks [`on_step`](crate::Driver::on_step) and [`converge_when`](crate::Driver::converge_when)
/// give access to a `Step` structure.
#[derive(Debug, Default, Clone)]
pub struct Step {
    pub(crate) iteration: usize,
    pub(crate) elapsed: Duration,
    pub(crate) elapsed_iter: Duration,
    pub(crate) progress: Progress,
}

impl Display for Step {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:>6}]", self.iteration())?;
        write!(f, " [{}]", format_duration(self.elapsed()))?;
        Ok(())
    }
}

impl Step {
    ///  The iteration count.
    ///
    /// Represents how many times 'step' has been invoked. Hence the first iteration is 1.
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
    pub fn elapsed(&self) -> Duration {
        self.elapsed
    }

    /// Elapsed time for this iteration only.
    ///
    pub fn elapsed_iter(&self) -> Duration {
        self.elapsed_iter
    }

    /// An estimate between `0` and `100` of how complete the solving process is.
    ///
    /// The most optimistic of
    /// - iterations out of iters, or
    /// - elapsed time out of of "fail after" duration
    ///
    /// will be used for the estimate.
    pub fn progress_percentage(&self) -> Option<f64> {
        match self.progress {
            Progress::InProgressPercentage(p) => Some(p),
            _ => None,
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
}

/// Formats a duration in the form 'hh:mm:ss.SSS'.
///
/// eg 5 hours 12 mins 17 secs and 345 millis would display as
///     `05:12:17.345`
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

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    #[test]
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    fn step_sizes() {
        assert_eq!(size_of::<Vec<f64>>(), 24);
    }
}
