[![Latest version](https://img.shields.io/crates/v/stepwise.svg)](https://crates.io/crates/stepwise)
[![Documentation](https://docs.rs/stepwise/badge.svg)](https://docs.rs/stepwise)
[![License](https://img.shields.io/crates/l/stepwise.svg)](https://choosealicense.com/licenses/)
[![MSRV](https://img.shields.io/crates/msrv/stepwise)](https://www.rust-lang.org)

# Overview

The `stepwise` crate is a zero-dependency helper library for writing iterative algorithms.  
It supports both numeric techniques—such as gradient descent, fixed-point iteration, Newton–Raphson—and non-numeric stepwise processes, like line-by-line file parsing or tree search.

With `stepwise`, you can:

- Use the [`Driver`](https://docs.rs/stepwise/latest/stepwise/trait.Driver.html) executor in a functional, iterator-like style to configure:
  - iteration limits, timeouts, convergence criteria,
  - logging, and
  - terminal progress bars.
- Control early stopping using [`metrics`](<https://docs.rs/stepwise/latest/stepwise/metrics/>) that compute moving averages, absolute/relative differences, invocation counts, or detect stalled solvers.
- Use included mini [`linear algebra functions`](<https://docs.rs/stepwise/latest/stepwise/trait.VectorExt.html>) for computing norms and distances.
- Collect structured data using [`samplers`](<https://docs.rs/stepwise/latest/stepwise/samplers/>) for ML training, convergence analysis, or plotting.
- Benchmark your own solvers on a suite of [`problems`](<https://docs.rs/stepwise/latest/stepwise/problems/>), and compare them with the built-in [`algorithms`](<https://docs.rs/stepwise/latest/stepwise/algos/>).

## Links

[`driver`](<https://docs.rs/stepwise/latest/stepwise/trait.Driver.html>) | 
[`algorithms`](<https://docs.rs/stepwise/latest/stepwise/algos/>) |
[`metrics`](<https://docs.rs/stepwise/latest/stepwise/metrics/>) |
[`linear algebra functions`](<https://docs.rs/stepwise/latest/stepwise/trait.VectorExt.html>) |
[`samplers`](<https://docs.rs/stepwise/latest/stepwise/samplers/>) |
[`problems`](<https://docs.rs/stepwise/latest/stepwise/problems/>) 


---

# Example 1

```rust
use stepwise::{Driver as _, fixed_iters, assert_approx_eq}; 
use stepwise::algos::GradientDescent;

let sphere_gradient = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
let learning_rate = 0.1;
let initial_guess = vec![5.5, 6.5];

let algo = GradientDescent::new(learning_rate, initial_guess, sphere_gradient);

let (solved, step) = fixed_iters(algo, 1000).solve().expect("solving failed!");

assert_approx_eq!(solved.x(), &[0.0, 0.0]);
```

# Example 2

```rust
use std::time::Duration;
use stepwise::prelude::*;

let gd = algos::GradientDescent::new(0.01, vec![1.0, 2.0], problems::sphere_grad);

let mut avg_of_norm = metrics::Ema::with_window(10);

let (solved, step) = fail_after_iters(gd, 10_000)
    .converge_when(|algo, _step| avg_of_norm.observe(algo.gradient.norm_l2()) < 1e-10)
    .show_progress_bar_after(Duration::from_millis(20))
    .on_step(|algo, _step| algo.learning_rate *= 0.9999)
    .solve()
    .expect("solving failed!");

assert_approx_eq!(solved.x(), &[0.0, 0.0]);
assert_eq!(step.iteration(), 1301);
```
In Example 2 (above):

- Early stopping is used to terminate the algorithm
- An exponential moving average of gradient's l2-norm is used to test it is near zero. 
- A progress bar is also printed in the terminal if algorithm runs for more than 20ms.
- The stepwise [`prelude`](<https://docs.rs/stepwise/latest/stepwise/prelude/>) is used to import both the [`metrics`](<https://docs.rs/stepwise/latest/stepwise/metrics/>) and the [`linear algebra trait`](<https://docs.rs/stepwise/latest/stepwise/trait.VectorExt.html>)
- A decaying learning rate is applied after each step.


# Creating an Algorithm
Callers will typically interact with your algorithm through the [`Driver`](<https://docs.rs/stepwise/latest/stepwise/trait.Driver.html>) trait.  
However, to integrate with `stepwise`, you must implement the core [`Algo`](https://docs.rs/stepwise/latest/stepwise/trait.Algo.html) trait for your algorithm.

This involves defining a single method:

- `step()`: returns a tuple of:
  - a [`ControlFlow`](https://doc.rust-lang.org/std/ops/enum.ControlFlow.html), indicating whether iteration should continue or stop
  - a `Result`, capturing any error that may occur during a step

This design separates control flow from failure, supporting fallible solvers with clean early-stopping logic.

Note: accessor methods like `x()` (for current state) are not part of the trait, but are expected in usable solvers, in order to retrieve the solution after or during iteration.


```rust
use std::{error::Error, ops::ControlFlow};

pub trait Algo {
    type Error: Send + Sync + Error + 'static;

    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>);

    // <default methods omitted>
}
```

# Links
- [`Changelog`](https://github.com/akanalytics/stepwise/blob/main/CHANGELOG.md)
- [`Motivation`](https://github.com/akanalytics/stepwise/blob/main/motivation.md)
- Related crate: [`stochy`](https://crates.io/crates/stochy/)


<!-- no blank line here -->
[`Driver`]: <https://docs.rs/stepwise/latest/stepwise/trait.Driver.html>
[`prelude`]: <https://docs.rs/stepwise/latest/stepwise/prelude>
[`metrics`]: <https://docs.rs/stepwise/latest/stepwise/metrics>
[`samplers`]: <https://docs.rs/stepwise/latest/stepwise/samplers>
[`problems`]: <https://docs.rs/stepwise/latest/stepwise/problems>
[`VectorExt`]: <https://docs.rs/stepwise/latest/stepwise/trait.VectorExt.html>
