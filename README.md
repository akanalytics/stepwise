# Overview

Stepwise is zero-dependency helper library for iterative algorithms. It covers both numeric algorithms such as gradient descent, fixed point iteration, Newton–Raphson as well as non-numeric step wise processes such as line-by-line file parsing, or tree search.

Some sample algorithms are included, but others are external to the library, though it is strightforward to adapt existing published algorithms. As a library, rather than a framework, most modules can be used independently without the need to adopt other parts.

The emphasis is on simplicity, and being self-contained.

For a more extensive numeric solving library see [Argmin](https://crates.io/crates/argmin/)


The crate offers 

* [Driver](#driver) an executer that handles the iteration, conditions for convergence or divergence, logging, and progress display

* [Iteration step](#iteration_step) a standardized structure that has iteration count, elapsed time, estimated time to completion.

* [Algorithms](#algos): some simple algos, mainly to serve as examples, and links to external algos.

* [Linear algebra](#linear_algebra) basic functions applicable to Rust slices, and hence can be used standalone, or with other linear algebrara libraries offering an as_slice type of conversion

* [Metrics](#metrics) that can be used for early stopping, such as moving averages, or delta-differences between iterations.

* [Problems](#problems) useful for testing and benchmarking algorithms.

* [Samplers](#samplers) for sampling convergence trajectories, progress displays, or recording epochs/iterations for later tabulation, plotting or analysis


<br><br><br>

# <a name="driver">Driver</a>
### Defining an Algorithm
Algorithms must implement the [`Algo`] trait. Some algorithms [are included](crate::algos), others are available seperately. 

They require constructors that accept hyper-parameters and initial conditions. 

Additionally, a retrieval method for the current best solution is requied. Typically this is named `x()` and could be an `f64` for 1-dimensional problems, or a slice/`Vec<f64>` for n-dimensional problems. 

The hyper-parameters, initial starting conditions and `x()` accessor are not part of the trait, and their naming is by convention.

The [`step`](`Algo::step`) method runs one iteration, and returns [`ControlFlow::Break`](`std::ops::ControlFlow::Break`) or `Continue` from the standard library to indicate whether further iteration is possible. In either case, a Result is returned so the caller can check for errors.

See [`Algo`] for an example.

```rust
# use std::{error::Error, fmt::Debug, ops::ControlFlow};
pub trait Algo {
    type Error: Error + Send + Sync + 'static;

    fn step(&mut self) -> ControlFlow<Result<(), Self::Error>, Result<(), Self::Error>>;
}
```


### Using the Driver
This allows the algorithm to be controlled (driven) using the [`Driver`], which handles the iteration, conditions for convergence or divergence, logging, and progress display.
Here
- [`fixed_iters`] is the factory method that constructs the driver (taking temporary ownership of the algo)
- the 1000 iteration count specifies that after 1000 iterations, convergence is assumed
- use [`fail_after_iters`] if instead you wish to check for convergence explicitly, and want to *fail* if a maximum iteration count is reached
- the `solve()` method will return back the algo in a `solved` state along with details of the last step of iteration


```rust,no_run
# use stepwise::{assert_approx_eq, problems::sphere_grad, algos::GradientDescent};
# use std::time::Duration;
# let sphere_grad = |x: &[f64]| x.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
# let learning_rate = 0.1;
# let initial_estimate = vec![5.5, 6.5];
# let my_algo = GradientDescent::new(learning_rate, initial_estimate, sphere_grad);
# use stepwise::{Driver, fixed_iters, Step, VectorExt as _}; 
let (solved, step) = fixed_iters(my_algo, 1000).solve().expect("solving failed!");
```
Here is a more complete example

 ```rust
 use stepwise::{Driver, fixed_iters, Step, algos::GradientDescent, assert_approx_eq};

 let learning_rate = 0.01;
 let initial_x0 = vec![5.0, 6.0];
 let gradient_fn = |x: &[f64]| vec![x[0] - 1.5, x[1] - 2.0];

// Any algorithm that implements the Algo trait will work here
let my_algo = GradientDescent::new(learning_rate, initial_x0, gradient_fn);

let driver = fixed_iters(my_algo, 1000)
     .on_step(|_algo, step| println!("{step:?}"));

// by convention an algorithm that has been solved is renamed from `my_algo` to `solved`
// the `solve` method will run the algorithm until completion (either success or first failure)
let (solved, _step) = driver.solve().expect("solving failed!");

// again, by convention, solutions are accessed via an `x()` method
 let soln_vec = solved.x();
 assert_approx_eq!(soln_vec, &vec![1.5, 2.0], 0.001);
 ```



For more details see [`Driver`]
<br><br><br>
# <a name="iteration_step">Iteration step</a>
A standardized iteration [`Step`] that has iteration count, elapsed time, estimated time to completion.
```ignore
impl Step {
    pub fn iteration(&self) -> usize  // first iteration is 1
    pub fn progress_percentage(&self) -> Option<f64> 
    pub fn elapsed(&self) -> Duration 
    pub fn elapsed_iter(&self) -> Duration 
    pub fn time_remaining(&self) -> Option<Duration> 
}
```
Most of the callbacks/closures on [`Driver`] have a [`Step`] to access progress , and a mutable [`Algo`] that can be used to access `x`, the best solution so far, as well as other intermediate calculates such as gradients. Since the `Algo` is mutable,
hyperparameters can be changed during iteration - adaptive learning rates for example.

A more complete example...
```rust
# use stepwise::{assert_approx_eq, problems::sphere_grad, algos::GradientDescent};
# use std::time::Duration;
# let sphere_grad = |x: &[f64]| x.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
# let learning_rate = 0.1;
# let initial_estimate = vec![5.5, 6.5];
# let gradient_descent = GradientDescent::new(learning_rate, initial_estimate, sphere_grad);
// trait VectorExt adds norm_l2 to f64-slices
use stepwise::{Driver, fixed_iters, Step, VectorExt as _}; 
# use log::trace;

let (solved, _step) = fixed_iters(gradient_descent, 100_000)
    // this algo exposes it's gradient
    .converge_when(|algo, _step| algo.gradient.norm_l2() < 0.01 ) 
    .show_progress_bar_after(Duration::from_millis(250))
    .on_step(|algo, step| trace!("iter {} x = {:?}", step.iteration(), algo.x()) )
    .solve()
    .expect("failed to solve");

let best_x = solved.x();
```

<br><br><br>
# <a name="algos">Solvers</a>
Included algos which can serve as examples.
- [algos::Bisection]
- [algos::GradientDescent]

External algos
- SPSA [<https://crates.io/crates/stochy>]
- RSPSA [<https://crates.io/crates/stochy>]


<br><br><br>
# <a name="metrics">Metrics</a>
See [`metrics`] for details.





<br><br><br>
# <a name="samplers">Samplers</a>
See [`samplers`]

Samplers are containers for sampling data or convergence trajectories; or recording epochs/iterations for later tabulation, plotting or analysis. 

| Container | Description |
|:---  |:---  |
| [`samplers::SampleVec`] | a reservoir sampler for periodic sampling  |
| [`samplers::SampleDeque`] | a reservoir sampler with guaranteed front and back minimum samples. Ideal for plots where the initial and final points are important to capture |

### Example
```rust
# use stepwise::{assert_approx_eq, problems::sphere_grad, algos::GradientDescent};
# use std::time::Duration;
use stepwise::{VectorExt as _, Driver, fixed_iters, Step, samplers::SampleDeque}; 
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let sphere_grad = |x: &[f64]| x.iter().map(|x| 2.0 * x).collect::<Vec<_>>();
# let learning_rate = 0.1;
# let initial_estimate = vec![5.5, 6.5];
# let gradient_descent = GradientDescent::new(learning_rate, initial_estimate, sphere_grad);

// create a container with 10 initial, and 10 final samples, 
// and 500 samples evenly taken from the 1,000,000 iterations
// this allows minimal memory use, and fast plotting 
let mut sampler = SampleDeque::with_sizes(10, 500, 10); 

type Point = (usize, Vec<f64>);  // (iteration number, X-vector)

let mut driver = fixed_iters(gradient_descent, 1_000_000);
while let Some((algo, step)) = driver.iter_step()? {
    let point: Point = (step.iteration(), algo.x().to_vec());
    sampler.sample(point);
}

// plot_data will have 500 evenly spaced samples, with the first and 
// last 10 items guaranteed to be captured
let plot_data : Vec<Point> = sampler.into_unordered_iter().collect();
# Ok(()) }
```


<br><br><br>
# <a name="linear_algebra">Linear Algebra</a>
See [`VectorExt`].

Some very basic linear albegra applicable to Rust slices, and hence can be used standalone, or with other linear algebrara libraries offering an as_slice type of conversion
- dot product
- norms, l2/euclidean, l-inf or max_norms, distance, 
- component-wise addition and subtraction
- formatting / display

<br><br><br>
# <a name="problems">Problems</a>
See [`problems`]

Some classic objective functions and their gradients, along with 
uniform and gaussian noise functions. Handy for testing and benchmarking.






