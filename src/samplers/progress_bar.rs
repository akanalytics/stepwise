use super::Destination;
use super::{Sampler, SamplingOutcome};
use crate::format_duration;
use crate::Step;
use std::time::Duration;

/// A progress bar which can display to Stderr or Stdout
///
/// It's a sampler of references to [`Step`] that will display if and only if it 'selects' its sampled `&Step`.
///
/// It will skip display/selection in either of the following cases:
/// - the relevant stdout/stderr is not a terminal (eg output redirected to a pipe)
/// - NO_COLOR env variable is set
/// - the last display was withing 50ms
/// - the elased is before the display_delay construction parameter
///
/// It will always return [`SamplingOutcome::Selected`]
///
/// Errors encounterd writing/flushing to stdout/stderr are ignored
#[derive(Debug)]
pub struct ProgressBar<'a> {
    next_progress_display: Duration,
    has_rendered: bool,
    destination: Destination<'a>,
}

impl Sampler<&Step> for ProgressBar<'_> {
    fn sample(&mut self, step: &Step) -> SamplingOutcome<&'static Step> {
        self.sample(step)
    }
}

impl<'a> ProgressBar<'a> {
    pub fn on_vec(self, vec: &'a mut Vec<String>) -> ProgressBar<'a> {
        let destination = Destination::StringVecRef(vec).check_visibility();
        ProgressBar {
            next_progress_display: self.next_progress_display,
            has_rendered: self.has_rendered,
            destination,
        }
    }
}

impl ProgressBar<'_> {
    /// \x1B[F  - moves the cursor up one line
    /// \x1B[2K - clears the entire line
    const MOVE_CURSOR: &'static str = "\x1B[F\x1B[2K";

    /// cpu saver - no need to redraw faster than this
    const REDRAW_EVERY_MILLIS: u64 = 50;

    pub fn after(display_delay: Duration) -> Self {
        let destination = Destination::Stdout.check_visibility();
        Self {
            next_progress_display: display_delay,
            has_rendered: false,
            destination,
        }
    }

    pub fn on_stdout(self) -> Self {
        let destination = Destination::Stdout.check_visibility();
        Self {
            destination,
            ..self
        }
    }

    pub fn on_stderr(self) -> Self {
        let destination = Destination::Stderr.check_visibility();
        Self {
            destination,
            ..self
        }
    }

    pub fn sample(&mut self, step: &Step) -> SamplingOutcome<&'static Step> {
        match self.render(step) {
            true => SamplingOutcome::Selected,
            // TODO: should be Rejected, but lifetimes!
            // fixable by making Sampler genric over lifetime of returned Outcome.
            false => SamplingOutcome::Selected,
        }
    }

    /// render the progress bar
    pub(crate) fn render(&mut self, step: &Step) -> bool {
        // early return for performance
        if matches!(self.destination, Destination::NoRender) {
            return false;
        }

        // FIXME better completion test
        if step.progress_percentage() == Some(100.0) && self.has_rendered {
            // will render as previously rendered < 100, and we need to overwrite
            // this with a completed progress bar
        } else if step.elapsed() < self.next_progress_display {
            return false;
        }

        let preamble = if self.has_rendered {
            Self::MOVE_CURSOR
        } else {
            ""
        };
        let percentage = step.progress_percentage();
        let width = 80;

        let bar = if let Some(percentage) = percentage {
            let filled = ((percentage / 100.0) * width as f64).round() as usize;
            let empty = width - filled;
            format!(
                "{}[{}{}] {:>6.2}%",
                preamble,
                "=".repeat(filled),
                " ".repeat(empty),
                percentage
            )
        } else {
            format!(
                "{}[{iter:>6}] elapsed: {elapsed}",
                preamble,
                iter = step.iteration(),
                elapsed = format_duration(step.elapsed())
            )
        };
        self.destination.render_on(&bar.to_string());
        self.has_rendered = true;
        self.next_progress_display =
            step.elapsed() + Duration::from_millis(Self::REDRAW_EVERY_MILLIS);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    #[test]
    fn test_progress_bar() {
        let mut step = Step::new(100, Instant::now());
        step.iter_limit = 1000;
        let mut vec = Vec::new();
        let mut pb = ProgressBar::after(Duration::from_millis(0)).on_vec(&mut vec);
        let rendered = pb.render(&step);
        assert!(rendered);
        assert!(pb.has_rendered);
        let expected = "[========                                                                        ]  10.00%";
        assert_eq!(&vec[0], expected);
    }
}
