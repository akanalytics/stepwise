//! Metrics collect maximums, averages etc, for use as convergence criteria.
//!
//! # Overview
//!
//! Often with stepwise algorithms, you will want to terminate early based on convergence criteria, but
//! batch gradients on noisy data, or stationary points will often mean a single spot check of gradient or
//! a spot check of step-size is unreliable for early termination.
//!
//! Instead use an exponential moving average of the l2 gradient norm,
//! or a [`MovingSpread`] of cost-function values (similar to PyTorch's `patience`)
//!
//! These metrics need to be mutably declared before the driver logic as they hold values between iterations, but will likely
//! be used within [`Driver::converge_when`](crate::Driver::converge_when).
//!
//! All the metrics implement the [`Metric`] trait.
//!
//! # Example 1:
//!
//! This first example uses the [`Delta`] metric, which tracks changes between current and prior iteration,
//! so see when the l2-distance between solution values is within a tolerance.
//!
//! Notice:
//! - The collected type needs to be an `Owned` type, hence `solv.x().to_vec()`
//! - The Delta metric takes a closure saying how to calculate distance between vectors at time t, and time t-1
//! - This closure's ref-parameter types must match the observe call (`Vec<f64>` matches `&Vec<f64>`)
//! - On the first iteration, when there is no prior t-1 vector, `observe` returns `f64::NAN` which will always compare false.
//!
//! ```rust
//! # use stepwise::prelude::*;
//! let gd = algos::GradientDescent::new(0.01, vec![1.0, 2.0], problems::sphere_grad);

//!
//! let dist_fn = |x_prev: &Vec<f64>, x: &Vec<f64>| x.sub_vec(x_prev).norm_l2();
//! let mut delta_x = metrics::Delta::new(dist_fn);
//!
//! let (solved, step) = fixed_iters(gd, 10_000)
//!     .converge_when(|algo, _step| delta_x.observe( algo.x().to_vec() ) < 1e-12)
//!     .solve()
//!     .unwrap();
//!
//! assert_approx_eq!(solved.x(), &[0.0, 0.0]);
//! assert_eq!(step.iteration(), 1215);
//! ```
//! # Example 2:
//!
//! This example tracks the exponential moving average (EMA) of the gradient norm (with a window of 10), using the [`Ema`] metric.
//! For the first 9 iterations [`Ema::observe`] will return `f64::NAN` as not enough samples will have been collected.
//!
//! Note: `f64::NAN < anything` is always false, so the convergence test is not trigger on the first 9 iterations.
//!
//! ```rust
//! # use stepwise::prelude::*;
//! # let gd = algos::GradientDescent::new(0.01, vec![1.0, 2.0], problems::sphere_grad);
//! // terminate when exp moving average of algo gradient has l2-norm near zero
//! let mut avg_of_norm = metrics::Ema::with_window(10);
//!
//! let (solved, step) = fixed_iters(gd, 10_000)
//!     .converge_when(|algo, _step| avg_of_norm.observe( algo.gradient.norm_l2() ) < 1e-12)
//!     .solve()
//!     .unwrap();
//!
//! assert_approx_eq!(solved.x(), &[0.0, 0.0]);
//! assert_eq!(step.iteration(), 1448);
//! ``````
//!
//! # Counter-example
//! Don't do this!
//!
//! Constructing the metric afresh each iteration means it will only ever observe one item, and always report `NAN`.
//! Metrics need to be declared up front, before any iteration.
//! ```should_panic
//! # use stepwise::prelude::*;
//! # let algo = algos::Bisection::from_fn(|x| x * x - 3.0, [0.0, 4.0]).unwrap();
//! let (_solved, step) = fixed_iters(algo, 10_000)
//!   .converge_when(|algo, _step| metrics::DeltaFloat::new().observe( algo.x()[0] ) < 1e-8)
//!   .solve()
//!   .unwrap();
//! assert!(step.iteration() < 10_000);
//!```
//!

use std::{
    collections::VecDeque,
    f64,
    fmt::{self, Debug, Display},
    sync::{
        atomic::{self, AtomicU64},
        Arc,
    },
};

/// A common interface accross metrics, with an invaluable [`Metric::observe`] method.
///
/// # Examples and Usage:
/// See [`crate::metrics`] for examples.
///
#[allow(clippy::len_without_is_empty)]
pub trait Metric<Input> {
    /// The type of elements recorded by the metric.
    /// For a delta of last vs current, this might be an n-dimentional vector. It typically needs to
    /// be an Owned type, rather than a reference.
    ///
    ///   For a moving average of gradient magnitudes, T would be `f64`, as only the norm value is being collected.    
    // type Input;
    type Output;

    /// Records the value, and return the calculated metric.
    /// If there are not enough samples to calculate the metric, [`f64::NAN`] is returned,
    /// which will always compare false. So a tolerance check `observe(x) < 0.0001` will
    /// fail until enough samples have been collected.
    /// If you don't like `NANs`, then [`Metric::observe_opt`] is your friend.
    fn observe(&mut self, x: Input) -> f64
    where
        Self::Output: Into<f64>,
    {
        self.observe_opt(x)
            .map(Into::<f64>::into)
            .unwrap_or(f64::NAN)
    }

    /// Similar to `observe` except None is returned instead of `f64::NAN` to indicate not enough data collected.
    /// Useful if you like
    /// ```ignore
    /// observe_opt(x).is_some_and( |m| m < 0.00001 )
    /// ```
    /// style coding.
    fn observe_opt(&mut self, x: Input) -> Option<Self::Output>;
}

pub trait StatefulMetric<Input>: Metric<Input> {
    fn observe_opt(&mut self, x: Input) -> Option<Self::Output> {
        self.store(x);
        self.value_opt()
    }

    /// The calculated value.
    ///
    /// Similar to observe_opt, but doesn't observe a new sample.
    /// # Examples:
    /// ```
    /// # use stepwise::prelude::*;
    /// # let mut metric = metrics::MovingMax::new(2);
    /// assert!(metric.value_opt().is_none());
    /// metric.observe(3.0);
    /// assert!(metric.value_opt().is_none());
    /// metric.observe(2.0);
    /// assert_eq!(metric.value_opt(), Some(3.0));
    /// ```
    fn value_opt(&self) -> Option<Self::Output>;

    /// The calculated value.
    ///
    /// Will return
    /// Similar to value_opt, but returns `NAN` rather than `None`.
    /// # Examples:
    /// ```
    /// # use stepwise::prelude::*;
    /// # let mut metric = metrics::MovingMax::new(1);
    /// assert!(metric.value().is_nan());
    /// metric.observe(3.0);
    /// assert_eq!(metric.value(), 3.0);
    /// ```
    fn value(&self) -> f64
    where
        Self::Output: Into<f64>,
    {
        self.value_opt().map(Into::<f64>::into).unwrap_or(f64::NAN)
    }

    /// Stores the value, and likely discards older values.
    fn store(&mut self, x: Input);
}

pub struct GradL2Norm; // TODO! GradL2Norm

/// Functions taking `Never` as an argument can never be called directly.
pub enum Never {}

/// Counts the number of invocations of a function.
///
/// The constructor [`with_fn`](InvocationCount::with_fn) constructs the metric
/// *and* a `counted function` which is used in place of the original function.
/// [`with_fn_mut`](InvocationCount::with_fn_mut) is for `FnMut` functions.
///
/// The `record` and `store` methods take a [`Never`] type and cannot be called directly,
/// instead the count is incremented by invoking the `counted function`.
///
/// ⚠️ Since many closures and all function pointers are `Copy`, be careful not to reuse
/// the uncounted original function after creating the metric.
///
/// ```
/// # use stepwise::metrics::InvocationCount;
/// let uncounted_fn = |x: i32| x * 2;
/// let (metric1, counted_fn) = InvocationCount::with_fn(uncounted_fn);
///
/// assert_eq!(uncounted_fn(1), 2); // not counted
/// assert_eq!(uncounted_fn(2), 4); // not counted
///
/// assert_eq!(counted_fn(1), 2);
/// assert_eq!(counted_fn(2), 4);
/// assert_eq!(counted_fn(3), 6);
///
/// assert_eq!(metric1.value(), 3);
/// ```
#[derive(Clone, Debug, Default)]
pub struct InvocationCount {
    count: Arc<AtomicU64>,
}

impl Metric<Never> for InvocationCount {
    type Output = u64;

    fn observe_opt(&mut self, _: Never) -> Option<Self::Output> {
        unreachable!()
    }
}

impl StatefulMetric<Never> for InvocationCount {
    fn store(&mut self, _: Never) {
        unreachable!()
    }

    fn value_opt(&self) -> Option<Self::Output> {
        Some(InvocationCount::value(self))
    }
}

impl InvocationCount {
    /// The number of times the function has been invoked.
    // impl specific as u64 isnt f64 theres no trait impl
    #[inline]
    pub fn value(&self) -> u64 {
        self.count.load(atomic::Ordering::Relaxed)
    }

    /// Creates the metric and a wrapped counted function, where target function is Fn
    pub fn with_fn<F, X, O>(f: F) -> (Self, impl Fn(X) -> O)
    where
        F: Fn(X) -> O,
    {
        let me = Self::default();
        let counted_f = {
            let local = Arc::clone(&me.count);
            #[inline]
            move |x: X| -> O {
                local.fetch_add(1, atomic::Ordering::Relaxed);
                f(x)
            }
        };
        (me, counted_f)
    }

    /// Creates the metric and a wrapped counted function, where target function is FnMut
    pub fn with_fn_mut<F, X, O>(mut f: F) -> (Self, impl FnMut(X) -> O)
    where
        F: FnMut(X) -> O,
    {
        let me = Self::default();
        let counted_f = {
            let local = Arc::clone(&me.count);
            #[inline]
            move |x: X| -> O {
                local.fetch_add(1, atomic::Ordering::Relaxed);
                f(x)
            }
        };
        (me, counted_f)
    }
}

impl<T, F> Display for Delta<T, F>
where
    Self: StatefulMetric<T, Output = f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value_opt() {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "None"),
        }
    }
}

impl Display for Ema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value_opt() {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "None"),
        }
    }
}

impl Display for Emv {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value_opt() {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "None"),
        }
    }
}

impl Display for MovingAvg {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value_opt() {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "None"),
        }
    }
}

impl Display for MovingMax {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value_opt() {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "None"),
        }
    }
}

impl Display for MovingSpread {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value_opt() {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "None"),
        }
    }
}

impl<T> Display for DeltaFloat<T>
where
    DeltaFloat<T>: StatefulMetric<T, Output = f64>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.value_opt() {
            Some(v) => write!(f, "{v}"),
            None => write!(f, "None"),
        }
    }
}

pub trait Float: Copy + Into<f64> + Debug + Display + Default {}
impl Float for f32 {}
impl Float for f64 {}

/// Simple metric for absolute difference of 1-dimentional f32 or f64's
///
/// # Example:
/// ```rust
/// # use stepwise::prelude::*;
/// let mut diff = metrics::DeltaFloat::new();
/// assert!(diff.observe(1.5_f32).is_nan());
/// assert_eq!(diff.observe(2.5_f32), 1.0);
/// ``````
#[derive(Clone, Debug, Default)]
pub struct DeltaFloat<T> {
    prev: Option<T>,
    prev2: Option<T>,
}

impl<T: Float> DeltaFloat<T> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<I: Float> Metric<I> for DeltaFloat<I> {
    type Output = f64;

    fn observe_opt(&mut self, x: I) -> Option<Self::Output> {
        StatefulMetric::observe_opt(self, x)
    }
}
impl<I: Float> StatefulMetric<I> for DeltaFloat<I> {
    fn value_opt(&self) -> Option<f64> {
        match (&self.prev2, &self.prev) {
            (&Some(p2), &Some(p1)) => Some((p2.into() - p1.into()).abs()),
            _ => None,
        }
    }

    fn store(&mut self, x: I) {
        self.prev2 = self.prev.take();
        self.prev = Some(x);
    }
}

/// Calculates a delta difference between the current and the last value seen.
///
#[derive(Clone, Debug)]
pub struct Delta<T, F> {
    f: F,
    prev: Option<T>,
    prev2: Option<T>,
}

impl<F: Fn(&T, &T) -> f64, T> Delta<T, F> {
    /// Creates a metric for measuring differences where the closure gives
    /// the formula for the difference between `last` and `current`
    /// Typical use would be
    /// ```ignore
    /// | prev: &Vec<f64>, current: &Vec<f64>| L2 distance prev to current
    /// ```
    /// or
    /// ```ignore
    /// | prev: &Vec<f64>, current: &Vec<f64>| l-infinity distance from prev to current
    /// ```
    pub fn new(f: F) -> Self {
        Delta {
            f,
            prev: None,
            prev2: None,
        }
    }
}

impl<I, F: Fn(&I, &I) -> f64> Metric<I> for Delta<I, F> {
    type Output = f64;

    fn observe_opt(&mut self, x: I) -> Option<Self::Output> {
        StatefulMetric::<I>::observe_opt(self, x)
    }
}

impl<I, F: Fn(&I, &I) -> f64> StatefulMetric<I> for Delta<I, F> {
    fn store(&mut self, x: I) {
        self.prev2 = self.prev.take();
        self.prev = Some(x);
    }

    fn value_opt(&self) -> Option<f64> {
        match (&self.prev2, &self.prev) {
            (Some(p2), Some(p1)) => Some((self.f)(p2, p1)),
            _ => None,
        }
    }

    // /// 0, 1 or 2
    // fn len(&self) -> usize {
    //     self.prev.iter().count() + self.prev2.iter().count()
    // }
}

#[derive(Clone, Debug)]
pub struct MovingStats {
    window_size: usize,
    min_samples: usize,
    items: VecDeque<f64>,
}

impl MovingStats {
    pub fn new(window_size: usize, min_samples: usize) -> Self {
        Self {
            window_size,
            min_samples,
            items: VecDeque::new(),
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub mean: f64,
    pub max: f64,
    pub min: f64,
    pub spread: f64,
    pub count: usize,
}

impl Metric<f64> for MovingStats {
    type Output = Stats;

    fn observe_opt(&mut self, x: f64) -> Option<Self::Output> {
        StatefulMetric::<f64>::observe_opt(self, x)
    }
}

impl StatefulMetric<f64> for MovingStats {
    fn store(&mut self, x: f64) {
        self.items.push_back(x);

        if self.items.len() > self.window_size {
            // If the deque exceeds the window size, remove the oldest value
            self.items.pop_front();
        }
    }

    fn value_opt(&self) -> Option<Self::Output> {
        if self.items.len() < self.min_samples {
            return None;
        }

        let max = self
            .items
            .iter()
            .cloned()
            .max_by(|a, b| a.total_cmp(b))
            .unwrap_or_default();
        let min = self
            .items
            .iter()
            .cloned()
            .min_by(|a, b| a.total_cmp(b))
            .unwrap_or_default();
        let spread = max - min;
        let count = self.items.len();
        let mean = self.items.iter().cloned().sum::<f64>() / count as f64;

        Some(Stats {
            count,
            mean,
            spread,
            max,
            min,
        })
    }
}

/// Records the maximum value in the last N values
///
/// # Example:
///
/// You can achieve MovingMin by negatiing `x`
/// ```rust
/// # use stepwise::prelude::*;
/// let mut max_metric = metrics::MovingMax::new(3);
/// let (x1, x2, x3) = (1.4, 1.2, 1.3);
/// max_metric.store(-x1);
/// max_metric.store(-x2);
/// max_metric.store(-x3);
/// let min = -max_metric.value_opt().unwrap_or(f64::NAN);
/// assert_eq!(min, 1.2);
/// ``````
#[derive(Clone, Debug)]
pub struct MovingMax {
    window_size: usize,
    items: VecDeque<f64>,
}

impl MovingMax {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            items: VecDeque::new(),
        }
    }
}

impl Metric<f64> for MovingMax {
    type Output = f64;

    fn observe_opt(&mut self, x: f64) -> Option<Self::Output> {
        StatefulMetric::<f64>::observe_opt(self, x)
    }
}

impl StatefulMetric<f64> for MovingMax {
    fn store(&mut self, x: f64) {
        self.items.push_back(x);

        if self.items.len() > self.window_size {
            // If the deque exceeds the window size, remove the oldest value
            self.items.pop_front();
        }
    }

    fn value_opt(&self) -> Option<f64> {
        if self.items.len() >= self.window_size {
            // max_by -> None if empty
            self.items.iter().cloned().max_by(|a, b| a.total_cmp(b))
        } else {
            None
        }
    }
}

/// Calculates the spread, the (max-min) in a moving window.
///
/// # Example:
///
/// ```rust
/// # use stepwise::prelude::*;
/// let mut spread = metrics::MovingSpread::new(3);
/// assert_eq!(spread.observe(1.4).is_nan(), true);
/// assert_eq!(spread.observe(1.2).is_nan(), true);
/// assert_approx_eq!(spread.observe(1.3), 0.2);
/// assert_approx_eq!(spread.observe(-0.5), 1.8);
/// ```
#[derive(Clone, Debug)]
pub struct MovingSpread {
    window_size: usize,
    items: VecDeque<f64>,
}

impl MovingSpread {
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            items: VecDeque::new(),
        }
    }
}

impl Metric<f64> for MovingSpread {
    type Output = f64;

    fn observe_opt(&mut self, x: f64) -> Option<Self::Output> {
        StatefulMetric::<f64>::observe_opt(self, x)
    }
}

impl StatefulMetric<f64> for MovingSpread {
    fn store(&mut self, x: f64) {
        self.items.push_back(x);
        if self.items.len() > self.window_size {
            self.items.pop_front();
        }
    }

    fn value_opt(&self) -> Option<f64> {
        if self.items.len() < self.window_size {
            return None;
        }

        let min = self.items.iter().copied().min_by(|a, b| a.total_cmp(b))?;
        let max = self.items.iter().copied().max_by(|a, b| a.total_cmp(b))?;
        Some(max - min)
    }
}

/// Records the average value in the last N values
#[derive(Clone, Debug)]
pub struct MovingAvg {
    window_size: usize,
    items: VecDeque<f64>,
}

impl MovingAvg {
    pub fn with_window_size(window_size: usize) -> Self {
        Self {
            window_size,
            items: VecDeque::new(),
        }
    }
}

impl Metric<f64> for MovingAvg {
    type Output = f64;

    fn observe_opt(&mut self, x: f64) -> Option<Self::Output> {
        StatefulMetric::<f64>::observe_opt(self, x)
    }
}

impl StatefulMetric<f64> for MovingAvg {
    fn store(&mut self, x: f64) {
        self.items.push_back(x);

        if self.items.len() > self.window_size {
            // If the deque exceeds the window size, remove the oldest value
            self.items.pop_front();
        }
    }

    fn value_opt(&self) -> Option<f64> {
        if self.items.len() >= self.window_size && !self.items.is_empty() {
            Some(self.items.iter().cloned().sum::<f64>() / self.items.len() as f64)
        } else {
            None
        }
    }
}

/// Exponential moving average.
///
/// See  <https://en.wikipedia.org/wiki/Exponential_smoothing>
#[derive(Clone, Debug)]
pub struct Ema {
    virtual_window_size: usize,
    count: usize,
    alpha: f64,
    avg: Option<f64>,
}

impl Ema {
    /// Creates a new exponential moving average metric
    ///
    /// The `smoothing factor`, alpha, is set to α = 2/(N + 1).
    ///
    /// Statistically, the EMA will be similar to a moving average of window size N,
    /// and to avoid bias, `None/NAN` will be returned until N items have been observed.
    /// If this is not desired, [`Ema::with_alpha`] can be used, where alpha can be specified,
    /// but the minimum samples can be set to 1.
    ///
    /// EMA = previous_value * (1 - alpha) + new_value * alpha
    ///
    pub fn with_window(virtual_window_size: usize) -> Self {
        Self {
            virtual_window_size,
            alpha: 2.0 / (1.0 + virtual_window_size as f64),
            avg: None,
            count: 0,
        }
    }

    /// None/NAN will be returned until `minimum samples` items have been observed.
    pub fn with_alpha(alpha: f64, minimum_samples: usize) -> Self {
        Self {
            virtual_window_size: minimum_samples,
            alpha,
            avg: None,
            count: 0,
        }
    }

    pub fn alpha(&self) -> f64 {
        self.alpha
    }

    /// Reset the EMA to uninitialized step
    pub fn reset(&mut self) {
        self.avg = None;
        self.count = 0;
    }
}

impl Metric<f64> for Ema {
    type Output = f64;

    fn observe_opt(&mut self, x: f64) -> Option<Self::Output> {
        StatefulMetric::observe_opt(self, x)
    }
}

impl StatefulMetric<f64> for Ema {
    fn store(&mut self, x: f64) {
        match self.avg {
            None => self.avg = Some(x),

            // Calculate new EMA value
            // EMA = previous_value * (1 - alpha) + new_value * alpha
            Some(current_value) => {
                self.avg = Some(current_value * (1.0 - self.alpha) + x * self.alpha)
            }
        };
        self.count += 1;
    }

    fn value_opt(&self) -> Option<f64> {
        if self.count < self.virtual_window_size {
            None
        } else {
            self.avg
        }
    }
}

/// Exponential moving variance. An estimate of the (max-min) spread for the last N items. (Work in progress!)
///
/// This effectively gives a spread (max-min) over a window size N without the need to store N values.
///
/// Welford's algorithm is used to calculate the running variance,
///
/// <https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm>
///
/// The spread_factor called the "the range rule of thumb", and is used to deduce a max-min spread from the
/// rolling standard deviation.
///
/// <https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule>
///
///
#[derive(Clone, Debug)]
pub struct Emv {
    ema: Ema,
    variance: f64,
    spread_factor: f64,
    count: usize,
}

impl Emv {
    /// The smoothing factor `alpha` is calculated from the virtual_window_size and
    /// the `spread_factor` represents a multiplier to estimate max-min from standard deviation.
    /// For Gaussian distributions, a factor of 6 assumes max-min = 6 standard deviations,
    /// a factor of 4 assumes spread = max-min = 4 standard deviations.
    ///
    pub fn with_factor(virtual_window_size: usize, spread_factor: f64) -> Self {
        Self {
            ema: Ema::with_window(virtual_window_size),
            variance: 0.0,
            spread_factor,
            count: 0,
        }
    }

    /// The smoothing factor `alpha` is calculated from the virtual_window_size and the
    /// spread factor is set to be 6 (using the rule of thumb for Gaussian distributions).
    /// For Gaussian distribution cdf(3) - cdf(-3) = 99.7%, so 3-(-3) is appropriate.
    pub fn new(virtual_window_size: usize) -> Self {
        Self {
            ema: Ema::with_window(virtual_window_size),
            variance: 0.0,
            spread_factor: 6.0,
            count: 0,
        }
    }

    pub fn std_deviation(&self) -> f64 {
        self.variance.sqrt()
    }

    /// Will return NaN before window size data items have been sampled
    pub fn mean(&self) -> f64 {
        self.ema.value()
    }
}

impl Metric<f64> for Emv {
    type Output = f64;

    fn observe_opt(&mut self, x: f64) -> Option<Self::Output> {
        StatefulMetric::observe_opt(self, x)
    }
}

impl StatefulMetric<f64> for Emv {
    fn store(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.ema.avg.unwrap_or(x);
        self.ema.store(x);

        let delta2 = self
            .ema
            .avg
            .expect("no average despite previous store call");

        // update the running variance using Welford's method
        self.variance += self.ema.alpha() * (delta * delta2 - self.variance);
    }

    fn value_opt(&self) -> Option<f64> {
        if self.count >= self.ema.virtual_window_size {
            Some(self.spread_factor * self.variance.sqrt())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        algos::{Bisection, GradientDescent},
        assert_approx_eq, fixed_iters,
        problems::sphere_grad,
        VectorExt,
    };

    use super::*;

    #[test]
    fn use_metrics() {
        let gd = GradientDescent::new(0.01, vec![1.0, 2.0], sphere_grad);

        // x[n] close to x[n-1] according to l2_norm
        let mut dist_l2 = Delta::new(|x: &Vec<f64>, y: &Vec<f64>| x.sub_vec(y).norm_l2());
        let (solved, step) = fixed_iters(gd.clone(), 10_000)
            .converge_when(|algo, _step| dist_l2.observe(algo.x().to_vec()) < 1e-12)
            .solve()
            .unwrap();
        assert_approx_eq!(solved.x(), &[0.0, 0.0]);
        assert_approx_eq!(step.iteration() as f64, 1215_f64, 5_f64);

        // spot check of algo gradient having near zero l2-norm
        let (solved, step) = fixed_iters(gd.clone(), 10_000)
            .converge_when(|algo, _step| algo.gradient.norm_l2() < 1e-12)
            .solve()
            .unwrap();
        assert_approx_eq!(solved.x(), &[0.0, 0.0]);
        assert_approx_eq!(step.iteration() as f64, 1443_f64, 5_f64);

        // exp moving average of algo gradient having near zero l2-norm
        let mut mov_avg_of_norm = Ema::with_window(10);
        let (solved, step) = fixed_iters(gd, 10_000)
            .converge_when(|algo, _step| mov_avg_of_norm.observe(algo.gradient.norm_l2()) < 1e-12)
            .solve()
            .unwrap();
        assert_approx_eq!(solved.x(), &[0.0, 0.0]);
        assert_approx_eq!(step.iteration() as f64, 1448_f64, 5_f64);

        let bi = Bisection::from_fn(|x| x * x - 3.0, [0.0, 4.0]).unwrap();
        let mut dist_max =
            Delta::new(|x: &[f64; 2], y: &[f64; 2]| (x[0] - y[0]).abs().max((x[1] - y[1]).abs()));
        let (solved, step) = fixed_iters(bi, 10_000)
            .converge_when(|algo, _step| dist_max.observe(algo.x()) < 1e-8)
            .solve()
            .unwrap();
        assert_approx_eq!(solved.x_mid(), f64::sqrt(3.0));
        assert_approx_eq!(step.iteration() as f64, 29_f64, 2_f64);

        let bi = Bisection::from_fn(|x| x * x - 3.0, [0.0, 4.0]).unwrap();
        let mut abs_diff = DeltaFloat::new();
        let (solved, step) = fixed_iters(bi, 10_000)
            .converge_when(|algo, _step| abs_diff.observe(algo.x_mid()) < 1e-8)
            .solve()
            .unwrap();
        assert_approx_eq!(solved.x_mid(), f64::sqrt(3.0));
        assert_approx_eq!(step.iteration() as f64, 29_f64, 2_f64);

        // DONT DO THIS
        let bi = Bisection::from_fn(|x| x * x - 3.0, [0.0, 4.0]).unwrap();
        let (_solved, step) = fixed_iters(bi, 10_000)
            .converge_when(|algo, _step| DeltaFloat::new().observe(algo.x()[0]) < 1e-8)
            .solve()
            .unwrap();
        assert_eq!(step.iteration(), 10_000);
    }

    #[test]
    fn test_max() {
        let mut max = MovingMax::new(3);

        max.store(1.0);
        assert_eq!(max.value_opt(), None);

        let val = max.observe(3.0);
        let cmp1 = val > 0.0;
        let cmp2 = val == 0.0;
        let cmp3 = val < 0.0;
        assert!(!(cmp1 || cmp2 || cmp3));

        max.store(2.0);
        // Maximum in the window [1.0, 3.0, 2.0]
        assert_eq!(max.value_opt(), Some(3.0));

        max.store(4.0);
        // Maximum in the window [3.0, 2.0, 4.0]
        assert_eq!(max.value_opt(), Some(4.0));

        max.store(0.0);
        // Maximum in the window [2.0, 4.0, 0.0]
        assert_eq!(max.value_opt(), Some(4.0));
    }

    #[test]
    fn test_moving_avg() {
        let mut moving_avg = MovingAvg::with_window_size(3);

        // Store values and check the average
        moving_avg.store(1.0);
        assert_eq!(moving_avg.value_opt(), None);

        moving_avg.store(3.0);
        assert_eq!(moving_avg.value_opt(), None);

        moving_avg.store(2.0);
        // Average of [1.0, 3.0, 2.0]
        assert_eq!(moving_avg.value_opt(), Some(2.0));

        moving_avg.store(4.0);
        // Average of [3.0, 2.0, 4.0]
        assert_eq!(moving_avg.value_opt(), Some(3.0));

        moving_avg.store(0.0);
        // Average of [2.0, 4.0, 0.0]
        assert_eq!(moving_avg.value_opt(), Some(2.0));
    }

    #[test]
    fn test_delta_f64() {
        let mut diff = DeltaFloat::new();
        assert!(diff.observe(1.5_f32).is_nan());
        assert_eq!(diff.observe(2.5_f32), 1.0);
        assert_eq!(diff.observe(2.0), 0.5);
        assert_eq!(diff.value_opt(), Some(0.5));
    }

    #[test]
    fn test_emv() {
        let mut emv = Emv::with_factor(10, 2.0);
        assert!(emv.observe(1.5).is_nan());
        assert!(emv.observe(2.5).is_nan());
        for d in [10.0, 12.5, 15.0, 17.5, 20.0].into_iter().cycle().take(100) {
            emv.store(d);
        }
        assert_approx_eq!(emv.mean(), 15.99, 0.01);
        assert_approx_eq!(emv.std_deviation(), 4.18, 0.01);
        assert_approx_eq!(emv.value(), 8.37, 0.01);
    }

    #[test]
    fn test_invocation_count_fn() {
        let (metric1, counted_fn) = InvocationCount::with_fn(|x: i32| x * 2);

        assert_eq!(counted_fn(1), 2);
        assert_eq!(counted_fn(2), 4);
        assert_eq!(counted_fn(3), 6);

        assert_eq!(metric1.value(), 3);

        // counted_again_fn calls the counted_fn too.
        let (metric2, counted_both_fn) = InvocationCount::with_fn(counted_fn);
        assert_eq!(counted_both_fn(1), 2);
        assert_eq!(counted_both_fn(2), 4);
        assert_eq!(counted_both_fn(3), 6);

        assert_eq!(metric2.value(), 3);
        assert_eq!(metric1.value(), 6);
    }

    #[test]
    fn test_invocation_count_mut() {
        let mut closure_data = 0_u32;
        let f = |x: i32| {
            closure_data += 1;
            x * 2
        };
        let (metric1, mut counted_fn_mut) = InvocationCount::with_fn_mut(f);

        assert_eq!(counted_fn_mut(1), 2);
        assert_eq!(counted_fn_mut(2), 4);
        assert_eq!(counted_fn_mut(3), 6);

        assert_eq!(metric1.value(), 3);
        drop(counted_fn_mut);
        assert_eq!(closure_data, 3);
    }
}
