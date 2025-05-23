use std::time::Duration;
use stepwise::{
    algos::GradientDescent, assert_approx_eq, fixed_iters, problems::sphere_grad, with_timeout,
};

fn main() {
    env_logger::init();

    let lr = 0.1;
    let gd = GradientDescent::new(lr, vec![5.55, 5.55], sphere_grad);

    println!("progress bar always...");
    let (solved, _step) = fixed_iters(gd.clone(), 500)
        .show_progress_bar_after(Duration::ZERO)
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(1)))
        .solve()
        .unwrap();
    assert_approx_eq!(solved.x(), &[0.0, 0.0]);
    println!("...done");

    println!("progress bar after 0.25 sec...");
    let _solved = fixed_iters(gd.clone(), 500)
        .show_progress_bar_after(Duration::from_secs_f64(0.25)) // 500 iters * 1ms = 0.5 sec total
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(1)))
        .solve()
        .unwrap();
    println!("...done");

    println!("progress bar after 0.25 sec time based...");
    let res = with_timeout(gd.clone(), Duration::from_secs_f64(0.5))
        .show_progress_bar_after(Duration::from_secs_f64(0.25)) // 500 iters * 1ms = 0.5 sec total
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(10)))
        .solve();
    assert!(res.is_err()); // Timeout
    println!("...done");

    let gd = GradientDescent::new(lr, vec![5.55, 5.55], sphere_grad);
    println!("progress bar after 3 sec... (wont display)");
    let _solved = fixed_iters(gd.clone(), 500)
        .show_progress_bar_after(Duration::from_secs(3))
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(1)))
        .solve()
        .unwrap();
    println!("...done");

    let gd = GradientDescent::new(lr, vec![5.55, 5.55], sphere_grad);
    println!("progress bar after MAX sec... (wont display)");
    let _solved = fixed_iters(gd.clone(), 500)
        .show_progress_bar_after(Duration::MAX)
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(1)))
        .solve()
        .unwrap();
    println!("...done");

    let gd = GradientDescent::new(lr, vec![5.55, 5.55], sphere_grad);
    println!("progress bar always but NO_COLOR set... (wont display)");
    std::env::set_var("NO_COLOR", "1");
    let _solved = fixed_iters(gd.clone(), 500)
        .show_progress_bar_after(Duration::ZERO)
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(1)))
        .solve()
        .unwrap();
    println!("...done");
}
