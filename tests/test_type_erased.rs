mod common;

use common::{CountingAlgo, DoubleCount};
use std::sync::{Arc, Mutex};
use stepwise::{fixed_iters, BoxedDriver, BoxedError, Driver, DynDriver};
use test_log::test;

#[test]
fn box_compat_boxing() -> Result<(), BoxedError> {
    // fn test_trait<D: Driver>(_d: &D) {}
    let driver1 = fixed_iters(CountingAlgo(10), 100).on_step(|_, _| ());
    let driver2 = fixed_iters(CountingAlgo(20), 130);
    let driver3 = fixed_iters(CountingAlgo(10), 100).on_step(|_, _| ());
    let driver4 = fixed_iters(CountingAlgo(20), 130);

    let boxed_driver1: BoxedDriver<CountingAlgo> = driver1.into_boxed();
    let boxed_driver2: BoxedDriver<CountingAlgo> = driver2.into_boxed();

    let boxed_driver3 = driver3.into_boxed();
    let boxed_driver4 = driver4.into_boxed();

    let vec = vec![boxed_driver1, boxed_driver2, boxed_driver3, boxed_driver4];
    for (i, d) in vec.into_iter().enumerate() {
        // test_trait(&d);
        let d = d.on_step(|_a, _s| ());
        // test_trait(&d);
        let (algo, step) = d.solve().unwrap();
        match i {
            0 | 2 => {
                assert_eq!(step.iteration(), 100);
                assert_eq!(algo.x(), 110);
            }
            1 | 3 => {
                assert_eq!(step.iteration(), 130);
                assert_eq!(algo.x(), 150);
            }
            _ => unreachable!(),
        }
        // assert_eq!(solved.0, 10);
    }
    Ok(())
}

#[test]
fn box_if_then() -> Result<(), BoxedError> {
    let log_enabled = true;
    let algo = CountingAlgo(10);
    let driver = if log_enabled {
        fixed_iters(algo, 100)
            .on_step(|_, step| println!("{step:?}"))
            .into_boxed()
    } else {
        fixed_iters(algo, 100).into_boxed()
    };

    let (algo, _step) = driver.solve()?;
    assert_eq!(algo.x(), 110);
    Ok(())
}

#[test]
fn into_dyn() -> Result<(), BoxedError> {
    let log_enabled = true;
    let algo = CountingAlgo(10);
    let driver: DynDriver = if log_enabled {
        fixed_iters(algo, 100)
            .on_step(|_, step| println!("{step:?}"))
            .into_dyn()
    } else {
        fixed_iters(algo, 110).into_dyn()
    };

    // driver is Driver<DynHelper, DynAlgo>
    let (algo, step) = driver.solve()?;
    assert_eq!(step.iteration(), 100);
    let ca: &CountingAlgo = algo.downcast_ref().unwrap();
    assert_eq!(ca.x(), 110);
    assert_eq!(ca.twice_x(), 220);
    Ok(())
}

#[test]
fn dyn_boxed_impl_mix() {
    let algo = CountingAlgo(10);
    let vec1 = Mutex::new(vec![]);
    let (solved, _step) = fixed_iters(algo, 2)
        .on_step(|_, _| vec1.lock().unwrap().push("impl1"))
        .on_step(|_, _| vec1.lock().unwrap().push("impl2"))
        .on_step(|_, _| vec1.lock().unwrap().push("impl3"))
        .solve()
        .unwrap();
    assert_eq!(solved.x(), 12);

    let vec2 = Arc::new(Mutex::new(vec![]));
    let algo = CountingAlgo(10);
    let (solved, _step) = fixed_iters(algo, 2)
        .on_step({
            let vec2 = Arc::clone(&vec2);
            move |_, _| vec2.lock().unwrap().push("impl")
        })
        .into_dyn()
        .on_step({
            let vec2 = Arc::clone(&vec2);
            move |_, _| vec2.lock().unwrap().push("dyn")
        })
        .into_boxed()
        .on_step(|_, _| vec2.lock().unwrap().push("boxed"))
        .solve()
        .unwrap();
    assert_eq!(solved.downcast_ref::<CountingAlgo>().unwrap().x(), 12);
    assert_eq!(solved.downcast_ref::<CountingAlgo>().unwrap().twice_x(), 24);

    assert_eq!(
        vec1.lock().unwrap().as_slice(),
        ["impl1", "impl2", "impl3", "impl1", "impl2", "impl3"]
    );
    assert_eq!(
        vec2.lock().unwrap().as_slice(),
        ["impl", "dyn", "boxed", "impl", "dyn", "boxed"]
    );
}

#[test]
fn test_impl() {
    let algo = CountingAlgo(10);
    let vec1 = Mutex::new(vec![]);
    let driver = fixed_iters(algo, 2)
        .on_step(|_, _| vec1.lock().unwrap().push("impl1"))
        .on_step(|_, _| vec1.lock().unwrap().push("impl2"))
        .on_step(|_, _| vec1.lock().unwrap().push("impl3"));

    fn solver_it<D: Driver>(d: D) -> D::Algo {
        let (solved, _step) = d.solve().unwrap();
        solved
    }

    let solved = solver_it(driver);
    assert_eq!(solved.x(), 12);
    assert_eq!(solved.twice_x(), 24);
}
