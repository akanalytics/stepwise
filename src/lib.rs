#![doc=include_str!("../README.md")]

mod driver;
mod math;

pub use crate::driver::{Driver, Solver, SolverError, Step};
pub use crate::math::VectorExt;

pub mod utilities {
    pub use crate::driver::format_duration;
    pub use crate::math::ExponentialMovingAvg;
}

#[doc(hidden)]
pub mod examples;
