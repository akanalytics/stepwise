#![doc=include_str!("../README.md")]
mod algo;
pub mod algos;
mod driver;
mod example_usage;
mod math;
pub mod metrics;
pub mod problems;
mod progress_bar;
mod rng;
pub mod samplers;
mod step;

use std::error::Error;
use std::fmt::{Debug, Display};
use std::sync::Arc;

pub use crate::math::VectorExt;
pub use crate::{
    algo::Algo,
    driver::Driver,
    driver::{fail_after_iters, fixed_iters, with_timeout},
    step::Step,
};

pub mod utilities {
    pub use crate::math::{central_difference_gradient, Tolerance};
    pub use crate::step::format_duration;
}

/// Use `stepwise::prelude::*` for a 'no fuss' include everything approach.
///
/// metrics, problems, algos are brought into scope at the module level, so
/// an individual metric would be refered to as [`metrics::MovingAvg`] and an algorithm
/// as [`algos::GradientDescent`]
pub mod prelude {
    pub use crate::math::VectorExt;
    pub use crate::{
        algos, assert_approx_eq, assert_approx_ne, fail_after_iters, fixed_iters, metrics,
        metrics::Metric,
        problems,
        utilities::{format_duration, Tolerance},
        with_timeout, Algo, Driver, DriverError,
    };
}

pub type BoxedError = Box<dyn Error + Send + Sync>;

#[derive(Clone, Debug)]
pub enum DriverError {
    FailIfPredicate,
    Timeout,
    MaxIterationsExceeded,
    Overflow,
    UnexpectedError(Arc<dyn Error + Send + Sync>),
    AlgoError(Arc<dyn Error + Send + Sync>),
    InitialCondition(String),
    Hyperparameter(String),
    General(String),
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<DriverError>()
};

impl Display for DriverError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        std::fmt::Debug::fmt(self, f)
    }
}

impl Error for DriverError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match &self {
            Self::UnexpectedError(e) => Some(e.as_ref()),
            Self::AlgoError(e) => Some(e.as_ref()),
            _ => None,
        }
    }
}
