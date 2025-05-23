use std::{error::Error, ffi::OsStr, io, ops::ControlFlow, path::Path};

/// Algorithms implement this trait to work with the library.
///
/// # Overview
/// The algo has a `Step` method, similar to iterator's `next`.
///
/// The "problem" being solved and any initial starting conditions are part of the algo, not
/// seperate entities.
///
/// The return parameter indicates whether iteration *can* continue, and propagates any errors encountered.
///
/// `ControlFlow::Continue(())` indicates that the algo could continue
/// though the final decision is made by the calling code (perhaps by
/// logic within [`Driver::converge_when`](crate::Driver::converge_when) ).
///
/// `ControlFlow::Break(())` indicates that the algo has converged or errored, and no more iteration steps are possible.
///
/// Independently, any errors occuring are returned in the tuple.
///
/// For many algorithms, each step depends on the previous step. In such cases,
/// the algorithm will return `Break` whenever an error occurs, as iteration cannot continue.
/// An algorithm such as a file line-parser, might continue iteration, collecting up all
/// errors in the file. In this case `(ControlFlow::Continue, Err(e))` would be used.
///
/// Whereas iterators return their current value in the call to next,
/// algorithms return `()` from step.
///
/// But to be useful, they will need accessors or public variables to access
/// the solution after after, or during iteration.
///
/// If called before the first call to `step`, they
/// could return the initial starting conditions of the algorithm.
///
/// By convention, the current best solution would be called `x()`
/// for a scalar or vector solution.
///
///
/// # Example 1:
/// Note that the hyperparameter (learning rate) and initial/best
/// solution are considered part of the algorithm. As is the problem being solved (in this case a gradient function).
///
/// The learning rate is public, and can be changed (is _adaptive_) during the optimization process.
/// ```
/// use std::{error::Error, ops::ControlFlow, convert::Infallible};
/// use stepwise::{Algo, BoxedError};
///
/// //
/// pub struct GradientDescent<G> {
///     pub gradient_fn:   G,
///     // current best solution, seeded with initial guess
///     pub x:             Vec<f64>,
///     pub learning_rate: f64,
/// }
///
/// impl<G> Algo for GradientDescent<G>
/// where
///     G: Fn(&[f64]) -> Vec<f64>
/// {
///     type Error = Infallible;
///
///     fn step(&mut self) -> (ControlFlow<()>, Result<(), Infallible>) {
///        let dfdx = (self.gradient_fn)(&self.x);
///        for i in 0..self.x.len() {
///            self.x[i] -=  self.learning_rate * dfdx[i];
///        }
///
///        // Allow the calling client to decide when to cease iteration.
///        // If our algorithm managed convergence/iteration counts internally,
///        // `ControlFlow::Break` would be returned to terminate iteration.
///        (ControlFlow::Continue(()), Ok(()))
///     }
/// }
/// ```
///
/// # Example 2:
/// This is the bisection algorithm, used to find the root (solution) to `f(x) = 0`
///
/// ```
/// # use stepwise::prelude::*;
/// # use std::ops::ControlFlow;
/// # use std::error::Error;
/// pub struct Bisection<F> {
///     f:     F,         // the objective function being solved: Fn(f64) -> Result<f64>
///     x:     [f64; 2],  // lower and upper bounds around the solution
///     f_lo:  f64,        
///     f_mid: f64,       // mid point of last function evaluations at lower and upper bounds
/// }
///
/// // There are no restrictions on non-trait access (or setter) methods, as these are
/// // defined outside of the trait.
/// impl<F> Bisection<F> {
///     // The lower and upper bounds around the the "best solution", so are designated `x`
///     pub fn x(&self) -> [f64; 2] {
///         self.x
///     }
///     
///     // The midpoint of the current range.
///     pub fn x_mid(&self) -> f64 {
///         (self.x[0] + self.x[1]) / 2.0
///     }
///     
///     // The function, `f`, evaluated at the midpoint of the current range.
///     pub fn f_mid(&self) -> f64 {
///         self.f_mid
///     }
/// }
///
///
/// impl<E, F> Algo for Bisection<F>
/// where
///     F: FnMut(f64) -> Result<f64, E>,
///     E: 'static + Send + Sync + Error,
/// {
///     type Error = E;
///
///     fn step(&mut self) -> (ControlFlow<()>, Result<(), E>) {
///         let mid = (self.x[0] + self.x[1]) / 2.0;
///         match (self.f)(mid) {
///             Ok(y) => self.f_mid = y,
///             // our objective function is falliable, so we propagate any errors
///             // we return `Break` if we encounter an error, as futher
///             // iteration is not possible
///             Err(e) => return (ControlFlow::Break(()), Err(e)),
///         };
///
///         if self.f_mid == 0.0 {
///             self.x = [mid, mid];
///             /// return `Break` if we know we have converged
///             return (ControlFlow::Break(()), Ok(()));
///         }
///
///         if self.f_mid * self.f_lo < 0.0 {
///             self.x[1] = mid;
///             (ControlFlow::Continue(()),Ok(()))
///         } else {
///             self.x[0] = mid;
///             self.f_lo = self.f_mid;
///             (ControlFlow::Continue(()),Ok(()))
///         }
///     }
/// }
/// ```
///  
pub trait Algo {
    type Error: Error + Send + Sync + 'static;

    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>);

    /// Implement this method to support recovery from a checkpoint file
    ///
    /// Typically the file format will be binary for checkpoint recovery (to avoid any loss of precision).
    ///
    /// # Example:
    /// ```ignore
    /// impl Algo for MyAlgo {
    ///
    ///     fn read(&mut self, format: &FileFormat, reader: &mut dyn io::Read) -> Result<(), io::Error> {
    ///         let mut new_algo: T = serde_json::from_reader(reader)?;
    ///
    ///         // typically some algo members will not be serialised: the objective function or closure perhaps
    ///         // these can cloned/copied/moved from the old self.
    ///         // `std::mem::swap` can be used to 'swap in' items as the old self is discarded (`*self = new_algo`)
    ///         move_partial_from(&mut new_algo, self);
    ///         *self = new_algo;
    ///         Ok(())
    ///     }
    /// }
    /// ```
    fn read_file(&mut self, fmt: &FileFormat, r: &mut dyn io::Read) -> Result<(), io::Error> {
        let _ = (r, fmt);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "file reading not supported by this algorithm".to_string(),
        ))
    }

    /// Implement this trait to support writing checkpoint files
    ///
    /// Typically the file format will be binary for checkpoint recovery (to avoid any loss of precision),
    /// but perhaps CSV also for logging and plotting.
    ///
    /// # Example:
    /// ```ignore
    /// impl Algo for MyAlgo {
    ///
    ///     fn write(&mut self, ff: FileFormat, w: &mut dyn io::Write) -> Result<(), io::Error> {
    ///         match ff {
    ///             FileFormat::Json => Ok(serde_json::to_writer_pretty(write, &self)?),
    ///             _ => return Err(ff.unsupported_error()),
    ///         }
    ///     }
    ///     // ..other methods
    /// }
    /// ```
    fn write_file(&self, fmt: &FileFormat, w: &mut dyn io::Write) -> Result<(), io::Error> {
        let _ = (w, fmt);
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "file writing not supported by this algorithm".to_string(),
        ))
    }
}

/// Used internally by algorithms to determine whether they can support writing a checkpoint file of the designated type.
///
/// When checkpoints are specified by [`Driver::checkpoint`](crate::Driver::checkpoint), the file format is determined from
/// the file extension of the supplied path.
///
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FileFormat {
    Csv { with_header: bool },
    Json,
    Control,
    Custom { extension: String },
}

impl FileFormat {
    pub fn from_path(path: &Path) -> Self {
        Self::from_os_str(path.extension().unwrap_or_default())
    }

    pub fn from_os_str(file_extension: &OsStr) -> Self {
        match file_extension
            .to_ascii_lowercase()
            .to_string_lossy()
            .as_ref()
        {
            "json" => Self::Json,
            "control" => Self::Control,
            ext => Self::Custom {
                extension: ext.to_string(),
            },
        }
    }

    pub fn to_extension(&self) -> String {
        match &self {
            FileFormat::Csv { with_header: _ } => "csv".to_string(),
            FileFormat::Json => "json".to_string(),
            FileFormat::Control => "control".to_string(),
            FileFormat::Custom { extension } => extension.to_string(),
        }
    }

    /// Will generate an [`io::Error`] indicating that the file format is unsupported
    pub fn unsupported_error(&self) -> io::Error {
        io::Error::new(
            io::ErrorKind::Unsupported,
            format!("{self:?} is not supported"),
        )
    }
}

pub(crate) trait FileReader {
    fn read_file(&mut self, format: &FileFormat, read: &mut dyn io::Read) -> Result<(), io::Error>;
}

pub(crate) trait FileWriter {
    fn write_file(&self, format: &FileFormat, write: &mut dyn io::Write) -> Result<(), io::Error>;
}

impl<A> FileReader for A
where
    A: Algo,
{
    fn read_file(&mut self, fmt: &FileFormat, reader: &mut dyn io::Read) -> Result<(), io::Error>
    where
        Self: Sized,
    {
        Algo::read_file(self, fmt, reader)
    }
}

impl<A> FileWriter for A
where
    A: Algo,
{
    fn write_file(&self, fmt: &FileFormat, writer: &mut dyn io::Write) -> Result<(), io::Error>
    where
        Self: Sized,
    {
        Algo::write_file(self, fmt, writer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algo_dyn_compatibility() {
        let _x: &dyn Algo<Error = io::Error>;
    }
}
