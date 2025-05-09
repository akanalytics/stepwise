[![Latest version](https://img.shields.io/crates/v/stepwise.svg)](https://crates.io/crates/stepwise)
[![Documentation](https://docs.rs/stepwise/badge.svg)](https://docs.rs/stepwise)
[![License](https://img.shields.io/crates/l/stepwise.svg)](https://choosealicense.com/licenses/)
[![msrv](https://img.shields.io/crates/msrv/stepwise)](https://www.rust-lang.org)

# Overview

`stepwise` is zero-dependency helper library for iterative algorithms. It covers both numeric algorithms such as gradient descent, fixed point iteration, Newton–Raphson as well as non-numeric step wise processes such as line-by-line file parsing, or tree search.

With `stepwise` you can

- Use the [`Driver`] executor to specify max iterations, timeouts, convergence criteria, logging and terminal progress display
- Control early stopping through [`metrics`] that calculate moving averages, absolute/relative differences, stuck solvers
- Use the included mini [`linear algebra functions`](VectorExt) for calculating norms and distances with these metrics
- Collect data via [`samplers`] for ML, convergence analysis or plotting
- Test and benchmark your algorithms on a prepared set of [`problems`], and compare them with included algorithms and solvers



# Example 1

```rust
use stepwise::{Driver as _, fixed_iters, assert_approx_eq}; 
use stepwise::{algos::GradientDescent};

let sphere_gradient = |x: &[f64]| vec![2.0 * x[0], 2.0 * x[1]];
let learning_rate = 0.1;
let initial_guess = vec![5.5, 6.5];

let algo = GradientDescent::new(learning_rate, initial_guess, sphere_gradient);

let (solved, step) = fixed_iters(algo, 1000).solve().expect("solving failed!");

assert_approx_eq!(solved.x(), &[0.0, 0.0]);
```

# Example 2
In this example
- Early stopping is used to terminate the algorithm
- An exponential moving average of gradient's l2-norm is used to test it is near zero. 
- A progress bar is also printed in the terminal if solver suns for more than 20ms.
- The stepwise [`prelude`] is used to import both the [`metrics`] and the [`linear algebra trait`](VectorExt)
- Adaptive learning rate: on each step, the learning rate is decreased

```rust
# use std::time::Duration;
use stepwise::prelude::*;
let gd = algos::GradientDescent::new(0.01, vec![1.0, 2.0], problems::sphere_grad);

let mut avg_of_norm = metrics::Ema::with_window(10);

let (solved, step) = fixed_iters(gd, 10_000)
    .converge_when(|algo, _step| avg_of_norm.record( algo.gradient.norm_l2() ) < 1e-10)
    .show_progress_bar_after(Duration::from_millis(20))
    .on_step(|algo, _step| algo.learning_rate *= 0.9999 )
    .solve()
    .expect("solving failed!");

assert_approx_eq!(solved.x(), &[0.0, 0.0]);
assert_eq!(step.iteration(), 1301);
```

# Creating an Algorithm
To work with the library, algorithms must implement a single method on an iterator-like trait... [`Algo`].

The return parameter is just a combination of
- The std library's [`ControlFlow`](std::ops::ControlFlow), and indicates whether iteration should stop or continue
- A Result indicating whether any errors occured during the execution of the step

Accessors to retrieve the solution after or during iteration, are not part of the trait, but will
be required for an algorithm to be useful.

```rust
use std::{error::Error, ops::ControlFlow};

pub trait Algo {
    type Error: Error + Send + Sync + 'static;

    fn step(&mut self) -> (ControlFlow<()>, Result<(),Self::Error>);
}
```





<!-- no blank line here -->
[`Driver`]: <https://docs.rs/stepwise/latest/stepwise/trait.Driver.html>
[`prelude`]: <https://docs.rs/stepwise/latest/stepwise/prelude>
[`metrics`]: <https://docs.rs/stepwise/latest/stepwise/metrics>
[`samplers`]: <https://docs.rs/stepwise/latest/stepwise/samplers>
[`problems`]: <https://docs.rs/stepwise/latest/stepwise/problems>
[`VectorExt`]: <https://docs.rs/stepwise/latest/stepwise/trait.VectorExt.html>
