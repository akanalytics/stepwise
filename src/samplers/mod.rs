//! Samplers are containers for sampling data or convergence trajectories; or recording epochs/iterations for later tabulation, plotting or analysis.
//!
//! | Container | Description |
//! |:---  |:---  |
//! | [`SampleVec`] | a reservoir sampler for periodic sampling  |
//! | [`SampleDeque`] | a reservoir sampler with guaranteed front and back minimum samples. Ideal for plots where the initial and final points are important to capture |
//! | [`ProgressBar`] | prints a progress bar |
//! | [`Tabulate`] | displays a simple table using `println` style formatting |
//!
//! ### Example
//! ```rust
//! # use stepwise::{assert_approx_eq, problems::sphere_grad, algos::GradientDescent};
//! # use std::time::Duration;
//! use stepwise::{VectorExt as _, Driver, fixed_iters, Step, samplers::SampleDeque};
//! # let sphere_grad = |x: &[f64]| x.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
//! # let learning_rate = 0.1;
//! # let initial_estimate = vec![5.5, 6.5];
//! # let gradient_descent = GradientDescent::new(learning_rate, initial_estimate, sphere_grad);
//!
//! // create a container with 10 initial, and 10 final samples,
//! // and 500 samples evenly taken from the 1,000,000 iterations
//! // this allows minimal memory use, and fast plotting
//! let mut sampler = SampleDeque::with_sizes(10, 500, 10);
//!
//! type Point = (usize, Vec<f64>);  // (iteration number, X-vector)
//!
//! let (_solved, _step) = fixed_iters(gradient_descent, 1_000_000)
//!     .on_step(|algo, step| {
//!         let point: Point = (step.iteration(), algo.x().to_vec());
//!         sampler.sample(point);
//!    })
//!    .solve()
//!    .unwrap();
//!
//!
//! // plot_data will have 500 evenly spaced samples, with the first and
//! // last 10 items guaranteed to be captured
//! let plot_data : Vec<Point> = sampler.into_unordered_iter().collect();
//! ```
mod progress_bar;
mod sample_deque;
mod sample_vec;
mod tabulate;

use std::io::{self, IsTerminal, Write};

pub use progress_bar::ProgressBar;
pub use sample_deque::SampleDeque;
pub use sample_vec::SampleVec;
pub use tabulate::Tabulate;

/// Indicates the action taken by sampling.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SamplingOutcome<T> {
    /// The item was selected for sampling, but no item was evicted.
    Selected,

    /// The candidate was selected, the evicted item is returned.
    /// An item can replace itself, if the item is processed, but not retained.
    Replaced(T),

    /// The candidate was rejected and not added to the sample, the candiate is returned
    Rejected(T),
}

/// Sampler trait
///
/// Samples an item and either...
/// - Selects the item
/// - Selects the item, but discards an existing item
/// - Rejects the item
///
pub trait Sampler<T> {
    fn sample(&mut self, item: T) -> SamplingOutcome<T>;
}

#[allow(dead_code)]
#[derive(Debug)]
pub(crate) enum Destination<'a> {
    Stdout,
    Stderr,
    StringVecRef(&'a mut Vec<String>),
    NoRender,
}

#[allow(dead_code)]
impl Destination<'_> {
    fn check_visibility(self) -> Self {
        match self {
            Self::StringVecRef(..) => self,
            Self::Stdout | Self::Stderr if std::env::var("NO_COLOR").is_ok() => Self::NoRender,
            Self::Stdout if !std::io::stdout().is_terminal() => Self::NoRender,
            Self::Stderr if !std::io::stderr().is_terminal() => Self::NoRender,
            _ => self,
        }
    }

    fn render_on(&mut self, text: &str) {
        match self {
            Destination::Stdout => {
                let _ = writeln!(io::stderr().lock(), "{text}");
                let _ = io::stdout().flush();
            }
            Destination::Stderr => {
                let _ = writeln!(io::stderr().lock(), "{text}");
                let _ = io::stderr().flush();
            }
            Destination::StringVecRef(vec) => {
                vec.push(text.to_string());
            }
            Destination::NoRender => {}
        }
    }
}
