#![doc=include_str!("../README.md")]
mod algo;
pub mod algos;
mod drive;
mod example_usage;
mod math;
pub mod metrics;
pub mod problems;
mod rng;
pub mod samplers;

use std::error::Error;
use std::fmt::{Debug, Display};
use std::sync::Arc;

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
    Checkpoint {
        filename: String,
        error: Arc<dyn Error + Send + Sync>,
    },
}

pub use crate::{
    algo::{Algo, FileFormat},
    drive::format_duration,
    drive::Step,
    drive::{fail_after_iters, fixed_iters, with_timeout},
    drive::{BoxedDriver, Driver, DynDriver, RecoveryFile},
    math::{central_difference_gradient, Tolerance, VectorExt},
};

/// Use `stepwise::prelude::*` for a 'no fuss' include everything approach.
///
/// metrics, problems, algos are brought into scope at the module level, so
/// an individual metric would be refered to as [`metrics::MovingAvg`] and an algorithm
/// as [`algos::GradientDescent`]
pub mod prelude {
    pub use crate::math::VectorExt;
    pub use crate::{
        algos, assert_approx_eq, assert_approx_ne,
        drive::format_duration,
        fail_after_iters, fixed_iters, metrics,
        metrics::{Metric, StatefulMetric},
        problems, samplers, with_timeout, Algo, BoxedDriver, BoxedError, Driver, DriverError,
        DynDriver, RecoveryFile,
    };
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

#[macro_export]
macro_rules! log_trace {
    ($($arg:tt)*) => {
        // println!("[{:<35}] {}", module_path!(), format!($($arg)*));
    };
}
