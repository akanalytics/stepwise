mod common;

use common::{CountingAlgo, FunctionAlgo};
use std::{error::Error, io, sync::Mutex, time::Duration};
use stepwise::{
    assert_approx_eq, fail_after_iters, fixed_iters, with_timeout, BoxedError, DriverError,
};
use test_log::test;

#[test]
fn test_scopes() {
    mod inner1 {
        // Needed: builder function
        // Not needed: Driver, DriverError, Algo
        use crate::common::CountingAlgo;
        use stepwise::fixed_iters;

        pub fn test() -> Result<(), Box<dyn std::error::Error>> {
            let algo = CountingAlgo(3);
            let (algo, step) = fixed_iters(algo, 5).solve()?;
            assert_eq!(algo.x(), 8);
            assert_eq!(step.iteration(), 5);
            Ok(())
        }
    }
    inner1::test().unwrap();

    mod inner2 {
        // Needed: builder function
        // Not needed: Driver, DriverError, Algo, Checkpointing/File traits
        use crate::common::CountingAlgo;
        use std::time::Duration;
        use stepwise::fixed_iters;

        pub fn test() -> Result<(), Box<dyn std::error::Error>> {
            let algo = CountingAlgo(3);
            let (algo, step) = fixed_iters(algo, 5)
                .checkpoint("/tmp/checkpoint.json", Duration::ZERO)
                .solve()?;
            assert_eq!(algo.x(), 8);
            assert_eq!(step.iteration(), 5);
            Ok(())
        }
    }
    inner2::test().unwrap();
}

#[test]
fn functional_fixed_iters() -> Result<(), Box<dyn Error>> {
    let algo = CountingAlgo(3);
    let (algo, step) = fixed_iters(algo, 5).solve()?;
    assert_eq!(step.iteration(), 5);
    assert_eq!(algo.x(), 8);
    Ok(())
}

#[test]
fn functional_max_iters() {
    let algo = CountingAlgo(3);
    let mut count = 0;
    let result = fail_after_iters(algo, 5).on_step(|_, _| count += 1).solve();
    assert!(result.is_err_and(|e| matches!(e, DriverError::MaxIterationsExceeded)));
    assert_eq!(count, 5);
}

#[test]
fn functional_timeout() {
    let algo = CountingAlgo(0);
    let result = with_timeout(algo, Duration::from_micros(5)).solve();
    assert!(result.is_err_and(|e| matches!(e, DriverError::Timeout)));
}

#[test]
fn functional_values() -> Result<(), Box<dyn Error + Send + Sync>> {
    let algo = CountingAlgo(10);
    let mut for_each_count = 0;
    // step.iteration() is 1..=3
    let _ = fixed_iters(algo, 3)
        .on_step(|a, s| {
            println!(
                "[{:>2}] x: {} {:>6.2} %",
                s.iteration(),
                a.x(),
                s.progress_percentage().unwrap_or_default()
            )
        })
        .on_step(|v, s| for_each_count += s.iteration() * v.x())
        .solve()?;

    #[allow(clippy::erasing_op, clippy::identity_op)]
    let expected = 1 * 11 + 2 * 12 + 3 * 13;
    assert_eq!(for_each_count, expected);
    Ok(())
}

#[test]
fn functional_mutate() {
    let algo = CountingAlgo(10);
    let (solved, _step) = fixed_iters(algo, 3)
        .on_step(|a, s| a.0 += s.iteration())
        .on_step(|a, s| println!("{} {}", s.iteration(), a.x()))
        .solve()
        .unwrap();

    #[allow(clippy::erasing_op, clippy::identity_op)]
    let expected = 10 + 1 + 1 + 1 + 2 + 1 + 3;
    assert_eq!(solved.0, expected);
}

#[test]
fn functional_order_of_execution() {
    let str = Mutex::new(String::from("*"));
    let algo = CountingAlgo(10);
    let (_solved, _step) = fixed_iters(algo, 4)
        .on_step(|_, s| {
            str.lock()
                .unwrap()
                .push_str(&format!("<{}>", s.iteration()))
        })
        .on_step(|_, s| {
            str.lock()
                .unwrap()
                .push_str(&format!("[{}]", s.iteration()))
        })
        .on_step(|_, s| {
            str.lock()
                .unwrap()
                .push_str(&format!("({})", s.iteration()))
        })
        .solve()
        .unwrap();

    assert_eq!(
        str.lock().unwrap().as_str(),
        "*<1>[1](1)<2>[2](2)<3>[3](3)<4>[4](4)"
    );
}

#[test]
fn functional_convergence() -> Result<(), Box<dyn Error>> {
    let algo = CountingAlgo(0);
    let (algo, _step) = fixed_iters(algo, 10)
        .converge_when(|a, _s| a.x() == 5)
        .solve()?;
    assert_eq!(algo.x(), 5);
    Ok(())
}

// TODO: work out what to do with TolX
// #[test]
// fn functional_tolx() -> Result<(), Box<dyn Error>> {
//     let algo = CountingSolver(0);
//     let (algo, _step) = fixed_iters(algo, 20_000).tol_x(0.015).solve()?;
//     assert_eq!(algo.x(), 67);
//     Ok(())
// }

#[test]
fn functional_algo_error() {
    let algo = CountingAlgo(usize::MAX - 2);
    let mut for_each_count = 0;
    let step = fixed_iters(algo, 10)
        .on_step(|_, _| for_each_count += 1)
        .on_step(|a, s| println!("{} {}", s.iteration(), a.x()))
        .solve();
    assert!(step.is_err());
    assert_eq!(for_each_count, 2);
    assert!(step.is_err_and(|e| matches!(e, DriverError::AlgoError(_))));
}

#[test]
fn functional_try_error() {
    let algo = CountingAlgo(0);
    let mut for_each_count = 0;
    let result = fixed_iters(algo, 10)
        .try_on_step(|_, _| {
            if for_each_count < 5 {
                for_each_count += 1;
                Ok(())
            } else {
                Err(std::fmt::Error)
            }
        })
        .solve();
    assert!(result.is_err());
    assert_eq!(for_each_count, 5);
    assert!(result.is_err_and(|e| matches!(e, DriverError::AlgoError(_))));
}

#[test]
fn functional_fail_if() {
    let algo = CountingAlgo(0);
    let result = fixed_iters(algo, 10).fail_if(|algo, _| algo.0 > 5).solve();
    assert!(result.is_err(), "{result:?}");
    assert!(result.is_err_and(|e| matches!(e, DriverError::FailIfPredicate)));
}

#[test]
fn tabulate() {
    // .tabulate("{iter}{elapsed}{time_remaining}", |a,s| (s.iteration(), s.elapsed(),s.time_remaining()))
    // println!("iter\telapsed\ttime_remaining\tprogress");
    // .on_step(|_, step| {
    //     println!(
    //         "{:>4}\t{:.0?}\t{:.2?}\t{:.2?}%",
    //         step.iteration(),
    //         step.elapsed(),
    //         step.time_remaining().unwrap(),
    //         step.progress_percentage().unwrap()
    //     )
    // })
}

#[test]
fn step_iters_and_progress() {
    let algo = FunctionAlgo::from_fn_and_value(
        |i: usize| -> Result<usize, io::Error> {
            std::thread::sleep(Duration::from_millis(30));
            Ok(i + 1)
        },
        0,
    );
    let mut vec = Vec::new();
    let (_solved, step) = fixed_iters(algo, 10)
        .on_step(|_, step| vec.push(step.clone()))
        .on_step(|_, step| {
            println!(
                "{:>4}\t{:.0?}\t{:.2?}\t{:.2?}%",
                step.iteration(),
                step.elapsed(),
                step.time_remaining().unwrap(),
                step.progress_percentage().unwrap()
            )
        })
        .solve()
        .unwrap();
    assert_eq!(vec.len(), 10);

    assert_eq!(vec[0].iteration(), 1);
    assert_eq!(vec[1].iteration(), 2);
    assert_eq!(vec[9].iteration(), 10);

    assert_approx_eq!(vec[0].progress_percentage().unwrap(), 10.0);
    assert_approx_eq!(vec[9].progress_percentage().unwrap(), 100.0);

    let half_way_estimate_ms = vec[4].time_remaining().unwrap().as_secs_f64() * 1000.0;
    assert_approx_eq!(half_way_estimate_ms, 150.0, 20.0);
    assert_eq!(vec[9], step);
}

// A bit time dependent
#[test]
fn step_time_and_progress() {
    let algo = FunctionAlgo::from_fn_and_value(
        |i: usize| -> Result<usize, io::Error> {
            std::thread::sleep(Duration::from_millis(30));
            Ok(i + 1)
        },
        0,
    );
    let mut vec = Vec::new();
    let _error = with_timeout(algo, Duration::from_millis(300))
        .on_step(|_, step| vec.push(step.clone()))
        .on_step(|_, step| {
            println!(
                "{:>4}\t{:.0?}\t{:.2?}\t{:.2?}%",
                step.iteration(),
                step.elapsed(),
                step.time_remaining().unwrap(),
                step.progress_percentage().unwrap()
            )
        })
        .solve()
        .unwrap_err();
    assert_eq!(vec.len(), 10);

    assert_eq!(vec[0].iteration(), 1);
    assert_eq!(vec[1].iteration(), 2);
    assert_eq!(vec[9].iteration(), 10);

    assert_approx_eq!(vec[0].progress_percentage().unwrap(), 10.0, 2.0);
    assert_approx_eq!(vec[9].progress_percentage().unwrap(), 100.0, 2.0);

    let half_way_estimate_ms = vec[4].time_remaining().unwrap().as_secs_f64() * 1000.0;
    assert_approx_eq!(half_way_estimate_ms, 150.0, 10.0);
}

#[test]
fn functional_progress_bar_fixed_iters() {
    println!("progress bar always fixed_iters...");
    let algo = CountingAlgo(0);
    let (solved, _step) = fixed_iters(algo, 30)
        .show_progress_bar_after(Duration::ZERO)
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(10)))
        .solve()
        .unwrap();
    assert_eq!(solved.x(), 30);
    println!("...done");
}

#[test]
fn functional_progress_bar_timeout() {
    println!("progress bar always fixed_time...");
    let algo = CountingAlgo(0);
    let failed = with_timeout(algo, Duration::from_millis(300))
        .show_progress_bar_after(Duration::ZERO)
        .on_step(|_, _| std::thread::sleep(Duration::from_millis(10)))
        .solve();
    assert!(matches!(failed.unwrap_err(), DriverError::Timeout));
    println!("...done");
}

#[test]
fn while_loop_completes() {
    let solv = CountingAlgo(3);
    let mut driver = fixed_iters(solv, 5);
    while let Ok(Some((algo, step))) = driver.iter_step() {
        println!("{step:?} x: {:?}", algo.x());
    }
    let (solved, step) = driver.solve().unwrap();
    assert_eq!(step.iteration(), 5);
    assert_eq!(solved.0, 8);
}

#[test]
fn while_loop_error() {
    let algo = CountingAlgo(usize::MAX - 2);
    let mut for_each_count = 0;
    let mut driver = fixed_iters(algo, 10);
    loop {
        let res = driver.iter_step();
        match (res, for_each_count) {
            (Ok(Some(..)), 0) => {}
            (Ok(Some(..)), 1) => {}
            (Err(e), 2) => {
                assert!(matches!(e, DriverError::AlgoError(_)));
                break;
            }
            _ => unreachable!(),
        }
        for_each_count += 1;
    }
    let res = driver.solve();
    assert_eq!(for_each_count, 2);
    assert!(res.is_err());
    assert!(res.is_err_and(|e| matches!(e, DriverError::AlgoError(_))));
}

#[test]
fn while_loop_break() -> Result<(), BoxedError> {
    let algo = CountingAlgo(0);
    let mut driver = fixed_iters(algo, 100);
    while let Some((_a, s)) = driver.iter_step()? {
        if s.iteration() == 10 {
            break;
        }
    }
    let (solved, step) = driver.solve()?;
    assert_eq!(step.iteration(), 10);
    assert_eq!(solved.0, 10);
    Ok(())
}
