use algos::GradientDescent;
use std::time::Duration;
use stepwise::{prelude::*, problems::sphere_grad};

fn main() {
    env_logger::init();
    example1();
    example2();
}

// This example show how you can downcast to an algorithm generic over a function
fn example1() {
    // note we need a concrete *function* to name and downcast to
    type GradFn = fn(&[f64]) -> Vec<f64>;

    let lr = 0.1;
    let my_closure = |x: &[f64]| vec![2.0 * sphere_grad(x)[0], sphere_grad(x)[1]];

    // coerce the closure into a function (must not capture from environment)
    let grad: GradFn = my_closure;

    let gd = GradientDescent::new(lr, vec![5.55, 5.55], grad);

    let (mut dyn_solved, _step) = fixed_iters(gd, 500)
        .show_progress_bar_after(Duration::ZERO)
        .into_dyn()
        .solve()
        .unwrap();

    let gd_algo: &GradientDescent<GradFn> =
        dyn_solved.downcast_ref().expect("failed to downcast_ref");

    let solution = gd_algo.x();
    println!("example 1 solution = {solution:?}");
    assert_approx_eq!(solution, &[0.0, 0.0]);

    let gradient = &gd_algo.gradient;
    assert_approx_eq!(gradient, &vec![0.0, 0.0]);

    let gd_mut: &mut GradientDescent<GradFn> =
        dyn_solved.downcast_mut().expect("failed to downcast_mut");
    gd_mut.learning_rate = 0.001;
}

// This example show how you can downcast to an algorithm generic over a closure,
// but not a function, by boxing the closure
fn example2() {
    // Step 1: Define a type alias for the closure type
    type BoxedGradFn = Box<dyn FnMut(&[f64]) -> Vec<f64>>;

    // Step 2: Create a closure which captures from the enviroment and box it
    let a = 2.0;
    let grad_fn: BoxedGradFn = Box::new(move |x| x.iter().map(|xi| a * xi).collect());

    // Step 3: Use the boxed closure in your algorithm
    let gd = GradientDescent::new(0.1, vec![5.55, 5.55], grad_fn);

    let (dyn_solved, _step) = fixed_iters(gd, 500).into_dyn().solve().unwrap();

    let gd_algo: &GradientDescent<BoxedGradFn> =
        dyn_solved.downcast_ref().expect("failed to downcast_ref");

    let solution = gd_algo.x();
    println!("example 2 solution = {solution:?}");
    assert_approx_eq!(solution, &[0.0, 0.0]);
}
