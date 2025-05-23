use criterion::{criterion_group, criterion_main, Criterion};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use stepwise::{algos::Bisection, fail_after_iters, problems::rosenbrock};

pub fn bisection_raw_algorithm<F>(mut a: f64, mut b: f64, tol: f64, mut f: F) -> f64
where
    F: FnMut(f64) -> f64,
{
    while (b - a).abs() > tol {
        let c = (a + b) / 2.0;
        if f(c) == 0.0 {
            return c;
        } else if f(a) * f(c) < 0.0 {
            b = c;
        } else {
            a = c;
        }
    }
    (a + b) / 2.0
}

fn solve_sw(iters: u64) -> Duration {
    let mut f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let mut total = Duration::ZERO;

    // Custom iteration loop — you time `iters` executions
    for _ in 0..iters {
        let algo = black_box(Bisection::from_fn(&mut f, [0.8, 1.2]).unwrap());
        let driver =
            fail_after_iters(algo, 100000).converge_when(|algo, _| algo.x_spread() < 1e-12);

        let start = Instant::now();
        let (solved, _step) = driver.solve().unwrap();
        black_box(solved.x_mid());
        total += start.elapsed();
    }
    total
}

fn solve_bi(iters: u64) -> Duration {
    let mut f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let mut total = Duration::ZERO;

    // Custom iteration loop — you time `iters` executions
    for _ in 0..iters {
        let start = Instant::now();
        let result =
            bisection_raw_algorithm(black_box(0.8), black_box(1.2), black_box(1e-12), &mut f);
        black_box(result);
        total += start.elapsed();
        black_box(result);
    }
    total
}

fn benchmark_bisection(c: &mut Criterion) {
    c.bench_function("sw2 bisection 10000 iterations", |b| {
        b.iter_custom(solve_sw)
    });

    c.bench_function("raw bisection 10000 iterations", |b| {
        b.iter_custom(solve_bi)
    });
}

criterion_group!(benches, benchmark_bisection);
criterion_main!(benches);
