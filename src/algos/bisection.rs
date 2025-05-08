use crate::{Algo, DriverError};
use std::{convert::Infallible, error::Error, fmt::Debug, ops::ControlFlow, sync::Arc};

///
/// Simple bisection method.
///
/// See <https://en.wikipedia.org/wiki/Bisection_method> for details on the algorithm.
///
/// The [`x`](Self::x) returned is an array `[f64;2]` which bounds the root.
/// The midpoint of the range is the best estimate of the root, and x_mid and
/// f_mid (the function evalution at the midpoint) are also available on the algo
///
/// # Examples
///
/// from <https://en.wikipedia.org/wiki/Bisection_method#Example:_Finding_the_root_of_a_polynomial>
///
/// ```rust  
/// use stepwise::{Driver as _, fixed_iters, algos::Bisection, assert_approx_eq};
///
/// let f = |x: f64| x.powi(3) - x - 2.0;
///
/// let algo = Bisection::from_fn(f, [1.0, 2.0]).expect("range didn't bound the solution");
/// let mut driver = fixed_iters(algo, 15);
///
/// println!("iter       x0        x1         x-mid         f");
///
/// while let Ok(Some((algo, step))) = driver.iter_step() {
///     let [x0, x1] = algo.x();
///     println!(
///         "{i:>4}  [{x0:.7}, {x1:.7}]  {xmid:.7}  {f:+.7}",
///         i = step.iteration(),
///         xmid = (x0 + x1) / 2.0,
///         f = algo.f_mid()
///     );
/// }
///
/// let (solved, _step) = driver.solve().expect("solving failed");
/// assert_approx_eq!(solved.x().as_slice(), &[1.5213, 1.5213], 0.0001);
/// ```
///
#[derive(Clone, PartialEq)]
pub struct Bisection<F> {
    f: F,
    x: [f64; 2],
    f_lo: f64,
    f_mid: f64,
}

impl<F> Debug for Bisection<F> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bisection")
            .field("x", &self.x)
            .field("f_lo", &self.f_lo)
            .field("f_mid", &self.f_mid)
            .finish()
    }
}

impl Bisection<()> {
    /// Create a new bisection algo from an infalliable function `f: f64 -> f64`
    pub fn from_fn<G>(
        mut g: G,
        range: [f64; 2],
    ) -> Result<Bisection<impl FnMut(f64) -> Result<f64, Infallible>>, DriverError>
    where
        G: FnMut(f64) -> f64,
    {
        Bisection::from_falliable_fn::<_, Infallible>(move |x| Ok(g(x)), range)
    }
}

impl Bisection<()> {
    /// Create a new bisection algo from an falliable function `f: f64 -> Result<f64,E>`
    /// # Errors
    /// * [`DriverError::AlgoError`] if the calls to the function f fail.
    /// * [`DriverError::InitialCondition`] if \[lo,hi\] doesn't bound the root.
    pub fn from_falliable_fn<F, E>(
        mut f: F,
        [lo, hi]: [f64; 2],
    ) -> Result<Bisection<F>, DriverError>
    where
        F: FnMut(f64) -> Result<f64, E>,
        E: 'static + Send + Sync + Error,
    {
        let f_lo = f(lo).map_err(|e| DriverError::AlgoError(Arc::new(e)))?;
        let f_hi = f(hi).map_err(|e| DriverError::AlgoError(Arc::new(e)))?;
        if f_lo * f_hi > 0.0 {
            let msg = format!("f({lo}) = {f_lo} and f({hi}) = {f_hi} do not bound the root",);
            return Err(DriverError::InitialCondition(msg));
        }
        Ok(Bisection::<F> {
            f,
            x: [lo, hi],
            f_lo,
            f_mid: 0.,
        })
    }
}

impl<F> Bisection<F> {
    pub fn x(&self) -> [f64; 2] {
        self.x
    }

    /// The midpoint of the current range.
    pub fn x_mid(&self) -> f64 {
        // f64::midpoint(self.x[0], self.x[1])
        (self.x[0] + self.x[1]) / 2.0
    }

    /// The function, `f`, evaluated at the midpoint of the current range.
    pub fn f_mid(&self) -> f64 {
        self.f_mid
    }
}

impl<E, F> Algo for Bisection<F>
where
    F: FnMut(f64) -> Result<f64, E>,
    E: 'static + Send + Sync + Error,
{
    type Error = E;

    fn step(&mut self) -> ControlFlow<Result<(), E>, Result<(), E>> {
        // let mid = f64::midpoint(self.x[0], self.x[1]);
        let mid = (self.x[0] + self.x[1]) / 2.0;
        match (self.f)(mid) {
            Ok(y) => self.f_mid = y,
            Err(e) => return ControlFlow::Break(Err(e)),
        };

        if self.f_mid == 0.0 {
            self.x = [mid, mid];
            return ControlFlow::Break(Ok(())); // convergence
        }

        if self.f_mid * self.f_lo < 0.0 {
            self.x[1] = mid;
            ControlFlow::Continue(Ok(()))
        } else {
            self.x[0] = mid;
            self.f_lo = self.f_mid;
            ControlFlow::Continue(Ok(()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{assert_approx_eq, fixed_iters, with_timeout, Driver, Step};
    use std::{thread, time::Duration};
    use test_log::test;

    #[test]
    fn doc_example1() {
        let f = |x: f64| x.powi(3) - x - 2.0;

        let algo = Bisection::from_fn(f, [1.0, 2.0]).expect("range didn't bound the solution");
        let mut driver = fixed_iters(algo, 15);

        println!("iter       x0        x1         x-mid         f");

        while let Some((algo, step)) = driver.iter_step().unwrap() {
            let [x0, x1] = algo.x();
            println!(
                "{i:>4}  [{x0:.7}, {x1:.7}]  {xmid:.7}  {f:+.7}",
                i = step.iteration(),
                xmid = (x0 + x1) / 2.0,
                f = 0.0, // step.algo().f_mid() TODO!
            );
        }

        let (solved, _step) = driver.solve().expect("solving failed");
        assert_approx_eq!(solved.x().as_slice(), &[1.5213, 1.5213], 0.0001);
    }

    #[test]
    fn bisection_exact() {
        let f = |x: f64| x * x - 4.0;
        let algo = Bisection::from_fn(f, [0.0, 6.0]).unwrap();
        assert!(format!("{algo:?}").contains("x: [0.0, 6.0]"));

        let (solved, step) = fixed_iters(algo, 60)
            .on_step(|v, s| log::trace!("{s:?} {:?}", v.x()))
            .converge_when(|v, _s| v.x()[1] - v.x()[0] < 1e-10)
            .solve()
            .unwrap();
        assert_approx_eq!(solved.x_mid(), 2.);
        assert_approx_eq!(solved.f_mid(), 0.);
        assert!(step.iteration() < 60, "{step:?}"); // typically 53 iterations
    }

    #[test]
    fn bisection_falliable() {
        let f = |x: f64| -> Result<f64, Infallible> { Ok(x * x - 4.0) };
        let algo = Bisection::from_falliable_fn(f, [0.0, 6.0]).unwrap();

        let (solved, step) = fixed_iters(algo, 100)
            .converge_when(|v, _s| v.x()[1] - v.x()[0] < 1e-8)
            .solve()
            .unwrap();
        assert_approx_eq!(solved.x_mid(), 2.);
        assert_approx_eq!(solved.f_mid(), 0.);
        assert!(step.iteration() < 59); // typically 53 iterations
    }

    #[test]
    fn bisection_x_tol() {
        let f = |x: f64| x * x - 4.0;
        let algo = Bisection::from_fn(f, [0.0, 6.0]).unwrap();

        let mut driver = fixed_iters(algo, 100);
        while let Some((algo, step)) = driver.iter_step().unwrap() {
            if algo.x()[1] - algo.x()[0] < 1e-8 {
                break;
            }
            print_step(algo, step)
        }
        let (solved, step) = driver.solve().expect("solving failed!");
        assert_approx_eq!(solved.x_mid(), 2.);
        assert_approx_eq!(solved.f_mid(), 0.);
        assert!(step.iteration() < 40); // typically 30 iterations
    }

    fn print_step<F>(algo: &Bisection<F>, s: &Step) {
        println!(
            "#{i:>2} x: [{a:.9}, {b:.9}] f_mid: {f:+.9} {p:5.0}%   elapsed: {e:.0?} remaining: {d:.0?}",
            e = s.elapsed(),
            i = s.iteration(),
            a = algo.x[0],
            b = algo.x[1],
            f = algo.f_mid(),
            p = s.progress_percentage().unwrap_or_default(),
            d = s.time_remaining().unwrap_or_default()
        );
    }

    #[test]
    fn bisection_print() {
        let f = |x: f64| x * x - 4.0;
        let algo = Bisection::from_fn(f, [0.0, 6.0]).unwrap();

        let driver = with_timeout(algo, Duration::from_millis(35));
        let result = driver
            .on_step(|v, s| {
                print_step(v, s);
                thread::sleep(Duration::from_millis(1));
            })
            .solve();

        assert!(matches!(result.as_ref().unwrap_err(), DriverError::Timeout));
        // assert!(step_solved.unwrap().0.iteration() < 60); // < 35 iterations
    }
}
