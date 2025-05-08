use log::debug;
use std::error::Error;
use stepwise::{
    algos::GradientDescent, assert_approx_eq, assert_approx_ne, fixed_iters, problems::abs_grad,
    Driver,
};

fn main() -> Result<(), Box<dyn Error>> {
    //
    // solve with constant learning rate
    //
    let lr = 0.1;
    let x0 = vec![5.55, 5.55];
    let gd = GradientDescent::new(lr, x0, abs_grad);
    let (solved, _step) = fixed_iters(gd, 200)
        .on_step(|v, s| debug!("{s:?} lr: {lr:.12}, x: {:+.12?}", v.x()))
        .solve()?;
    let x = solved.x();

    // with constant learning rate the solution oscillates, and only reaches +/- 0.1, but fails 0.01
    assert_approx_ne!(x, &[0.0, 0.0], 0.01);
    assert_approx_eq!(x, &[0.0, 0.0], 0.1);

    //
    // solve with adaptive learning rate
    //
    let lr = 0.1;
    let x0 = vec![5.55, 5.55];
    let gd = GradientDescent::new(lr, x0, abs_grad);

    let mut driver = fixed_iters(gd, 200);
    while let Some((algo, step)) = driver.iter_step()? {
        algo.learning_rate *= 0.99;

        let lr = algo.learning_rate;
        let x = algo.x();
        debug!("{step:?} lr: {lr:.12}, x: {x:+.12?}",)
    }
    let (solved, _step) = driver.solve()?;
    let x = solved.x();

    // with adaptive learning rate the solution is within +/- 0.01
    assert_approx_eq!(x, &[0.0, 0.0], 0.01);

    Ok(())
}
