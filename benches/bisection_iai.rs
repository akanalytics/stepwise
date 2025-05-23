use iai_callgrind::{
    black_box, library_benchmark, library_benchmark_group, main, FlamegraphConfig,
    LibraryBenchmarkConfig,
};
use stepwise::{
    algos::Bisection, fail_after_iters, fixed_iters, problems::rosenbrock, Algo, BoxedDriver,
    Driver, DynDriver,
};

fn bisection_raw_algorithm<F>(mut a: f64, mut b: f64, tol: f64, f: F) -> f64
where
    F: Fn(f64) -> f64,
{
    let mut fa = f(a);

    while (b - a).abs() > tol {
        let c = (a + b) / 2.0;
        let fc = f(c);

        if fc == 0.0 {
            return c;
        } else if fa * fc < 0.0 {
            b = c;
        } else {
            a = c;
            fa = fc;
        }
    }

    (a + b) / 2.0
}

pub fn sw_setup_generic() -> impl Driver {
    let f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let algo = Bisection::from_fn(f, [0.8, 1.2]).unwrap();
    let driver =
        fail_after_iters(algo, 100000).converge_when(|algo, _step| algo.x_spread() < 1e-12);
    println!("{driver:#?}");
    driver
}

pub fn sw_setup_generic_fixed() -> impl Driver {
    let f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let algo = Bisection::from_fn(f, [0.8, 1.2]).unwrap();
    let driver = fixed_iters(algo, 100000).converge_when(|algo, _step| algo.x_spread() < 1e-12);
    println!("{driver:#?}");
    driver
}

pub fn sw_setup_boxed() -> BoxedDriver<impl Algo> {
    let f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let algo = Bisection::from_fn(f, [0.8, 1.2]).unwrap();
    let driver =
        fail_after_iters(algo, 100000).converge_when(|algo, _step| algo.x_spread() < 1e-12);
    println!("{driver:#?}");
    driver.into_boxed()
}

pub fn sw_setup_dyn() -> DynDriver {
    let f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let algo = Bisection::from_fn(f, [0.8, 1.2]).unwrap();
    let driver =
        fail_after_iters(algo, 100000).converge_when(|algo, _step| algo.x_spread() < 1e-12);
    println!("{driver:#?}");
    driver.into_dyn()
}

#[library_benchmark]
#[bench::generic(setup=sw_setup_generic)]
#[bench::generic_fixed(setup=sw_setup_generic_fixed)]
pub fn sw_bisection_generic(driver: impl Driver) {
    let (solved, _step) = driver.solve().unwrap();
    black_box(solved);
}

#[library_benchmark]
pub fn iai_sw_bisection_incl_setup() {
    let f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let algo = Bisection::from_fn(f, [0.8, 1.2]).unwrap();
    let driver = fail_after_iters(algo, 100000).converge_when(|algo, _| algo.x_spread() < 1e-12);
    let (solved, _step) = driver.solve().unwrap();
    black_box(solved.x_mid());
}

#[library_benchmark]
#[bench::generic(setup=sw_setup_boxed)]
pub fn sw_bisection_boxed(driver: BoxedDriver<impl Algo>) {
    let (solved, step) = driver.solve().unwrap();
    black_box(solved);
    assert_eq!(step.iteration(), 39);
}

#[library_benchmark]
#[bench::dynamic(setup=sw_setup_dyn)]
pub fn sw_bisection_dyn(driver: DynDriver) {
    let (solved, _step) = driver.solve().unwrap();
    black_box(solved);
}

#[library_benchmark]
pub fn iai_raw_bisection() {
    let f = |x| rosenbrock(&[x, 1.0], (1.0, 100.0)) - 15.0;
    let result = bisection_raw_algorithm(0.8, 1.2, 1e-12, f);
    black_box(result);
}

// Command line:
//
// cargo bench --bench bisection_iai
// google-chrome --new-tab target/iai/bisection_iai/bisection_group/iai_sw_bisection/callgrind.iai_sw_bisection.total.Ir.flamegraph.svg
// google-chrome --new-tab target/iai/stepwise/bisection_iai/bisection_group/iai_sw_bisection/callgrind.iai_sw_bisection.total.Ir.flamegraph.svg
library_benchmark_group!(name = bisection_group;
benchmarks =
    sw_bisection_generic,
    sw_bisection_boxed,
    sw_bisection_dyn,
    iai_sw_bisection_incl_setup,
    iai_raw_bisection,
);

main!(config = LibraryBenchmarkConfig::default()
        .flamegraph(FlamegraphConfig::default());
    library_benchmark_groups = bisection_group);

// results: 11/5/2025
//
// bisection_iai::bisection_group::iai_sw_bisection
//   Instructions:                        5518|5518                 (No change)
//   L1 Hits:                             7813|7813                 (No change)
//   L2 Hits:                                3|3                    (No change)
//   RAM Hits:                              46|46                   (No change)
//   Total read+write:                    7862|7862                 (No change)
//   Estimated Cycles:                    9438|9438                 (No change)
//
// bisection_iai::bisection_group::iai_raw_bisection
//   Instructions:                        6032|6032                 (No change)
//   L1 Hits:                             7575|7575                 (No change)
//   L2 Hits:                                0|0                    (No change)
//   RAM Hits:                              13|13                   (No change)
//   Total read+write:                    7588|7588                 (No change)
//   Estimated Cycles:                    8030|8030                 (No change)
//
// results 12/5/2025
//
//   Instructions:                        5430|5430                 (No change)
//   L1 Hits:                             7606|7606                 (No change)
//   L2 Hits:                                4|4                    (No change)
//   RAM Hits:                              44|44                   (No change)
//   Total read+write:                    7654|7654                 (No change)
//   Estimated Cycles:                    9166|9166                 (No change)
// bisection_iai::bisection_group::iai_raw_bisection
//   Instructions:                        6032|6032                 (No change)
//   L1 Hits:                             7575|7575                 (No change)
//   L2 Hits:                                0|0                    (No change)
//   RAM Hits:                              13|13                   (No change)
//   Total read+write:                    7588|7588                 (No change)
//   Estimated Cycles:                    8030|8030                 (No change)
//
// results 12/5/2025 PM
//
//   Instructions:                        4927|4927                 (No change)
//   L1 Hits:                             6765|6765                 (No change)
//   L2 Hits:                                3|3                    (No change)
//   RAM Hits:                              36|36                   (No change)
//   Total read+write:                    6804|6804                 (No change)
//   Estimated Cycles:                    8040|8040                 (No change)

// results 16/5/2025 PM
//   Instructions:                        4694|N/A                  (*********)
//   L1 Hits:                             6294|N/A                  (*********)
//   L2 Hits:                                3|N/A                  (*********)
//   RAM Hits:                              42|N/A                  (*********)
//   Total read+write:                    6339|N/A                  (*********)
//   Estimated Cycles:                    7779|N/A                  (*********)
//
// bisection_iai::bisection_group::iai_sw_bisection_only one:iai_sw_setup()
//   Instructions:                        4483|4483                 (No change)
//   L1 Hits:                             6013|6013                 (No change)
//   L2 Hits:                                2|2                    (No change)
//   RAM Hits:                              22|22                   (No change)
//   Total read+write:                    6037|6037                 (No change)
//   Estimated Cycles:                    6793|6793                 (No change)
// bisection_iai::bisection_group::iai_sw_bisection
//   Instructions:                        4694|4694                 (No change)
//   L1 Hits:                             6293|6293                 (No change)
//   L2 Hits:                                4|4                    (No change)
//   RAM Hits:                              42|42                   (No change)
//   Total read+write:                    6339|6339                 (No change)
//   Estimated Cycles:                    7783|7783                 (No change)

// bisection_iai::bisection_group::sw_bisection_generic generic:sw_setup_generic()
//   Instructions:                        2798|2798                 (No change)
//   L1 Hits:                             3846|3846                 (No change)
//   L2 Hits:                                0|0                    (No change)
//   RAM Hits:                               6|6                    (No change)
//   Total read+write:                    3852|3852                 (No change)
//   Estimated Cycles:                    4056|4056                 (No change)
// bisection_iai::bisection_group::sw_bisection_boxed generic:sw_setup_boxed()
//   Instructions:                        3490|6070                 (-42.5041%) [-1.73926x]
//   L1 Hits:                             4928|8591                 (-42.6376%) [-1.74330x]
//   L2 Hits:                                2|4                    (-50.0000%) [-2.00000x]
//   RAM Hits:                              14|29                   (-51.7241%) [-2.07143x]
//   Total read+write:                    4944|8624                 (-42.6716%) [-1.74434x]
//   Estimated Cycles:                    5428|9626                 (-43.6111%) [-1.77340x]
// bisection_iai::bisection_group::sw_bisection_dyn dynamic:sw_setup_dyn()
//   Instructions:                        6224|6968                 (-10.6774%) [-1.11954x]
//   L1 Hits:                             9049|9883                 (-8.43873%) [-1.09216x]
//   L2 Hits:                                3|6                    (-50.0000%) [-2.00000x]
//   RAM Hits:                              19|30                   (-36.6667%) [-1.57895x]
//   Total read+write:                    9071|9919                 (-8.54925%) [-1.09348x]
//   Estimated Cycles:                    9729|10963                (-11.2560%) [-1.12684x]
// bisection_iai::bisection_group::iai_sw_bisection
//   Instructions:                        2879|2879                 (No change)
//   L1 Hits:                             3636|3635                 (+0.02751%) [+1.00028x]
//   L2 Hits:                                1|2                    (-50.0000%) [-2.00000x]
//   RAM Hits:                              24|24                   (No change)
//   Total read+write:                    3661|3661                 (No change)
//   Estimated Cycles:                    4481|4485                 (-0.08919%) [-1.00089x]
// bisection_iai::bisection_group::iai_raw_bisection
//   Instructions:                        2652|2652                 (No change)
//   L1 Hits:                             3400|3399                 (+0.02942%) [+1.00029x]
//   L2 Hits:                                1|1                    (No change)
//   RAM Hits:                              13|14                   (-7.14286%) [-1.07692x]
//   Total read+write:                    3414|3414                 (No change)
//   Estimated Cycles:                    3860|3894                 (-0.87314%) [-1.00881x]

// bisection_iai::bisection_group::sw_bisection_generic generic:sw_setup_generic()
//   Instructions:                        2682|2682                 (No change)
//   L1 Hits:                             3726|3726                 (No change)
//   L2 Hits:                                0|0                    (No change)
//   RAM Hits:                              14|14                   (No change)
//   Total read+write:                    3740|3740                 (No change)
//   Estimated Cycles:                    4216|4216                 (No change)
// bisection_iai::bisection_group::sw_bisection_generic generic_fixed:sw_setup_generic_fixed()
//   Instructions:                        2564|2564                 (No change)
//   L1 Hits:                             3492|3492                 (No change)
//   L2 Hits:                                0|0                    (No change)
//   RAM Hits:                              13|13                   (No change)
//   Total read+write:                    3505|3505                 (No change)
//   Estimated Cycles:                    3947|3947                 (No change)
// bisection_iai::bisection_group::sw_bisection_boxed generic:sw_setup_boxed()
//   Instructions:                        3539|3539                 (No change)
//   L1 Hits:                             4943|4943                 (No change)
//   L2 Hits:                                1|1                    (No change)
//   RAM Hits:                              14|14                   (No change)
//   Total read+write:                    4958|4958                 (No change)
//   Estimated Cycles:                    5438|5438                 (No change)
// bisection_iai::bisection_group::sw_bisection_dyn dynamic:sw_setup_dyn()
//   Instructions:                        6351|6351                 (No change)
//   L1 Hits:                             9180|9180                 (No change)
//   L2 Hits:                                4|4                    (No change)
//   RAM Hits:                              20|20                   (No change)
//   Total read+write:                    9204|9204                 (No change)
//   Estimated Cycles:                    9900|9900                 (No change)
// bisection_iai::bisection_group::iai_sw_bisection
//   Instructions:                        2848|2848                 (No change)
//   L1 Hits:                             3940|3940                 (No change)
//   L2 Hits:                                2|2                    (No change)
//   RAM Hits:                              32|32                   (No change)
//   Total read+write:                    3974|3974                 (No change)
//   Estimated Cycles:                    5070|5070                 (No change)
// bisection_iai::bisection_group::iai_raw_bisection
//   Instructions:                        2652|2652                 (No change)
//   L1 Hits:                             3402|3402                 (No change)
//   L2 Hits:                                1|1                    (No change)
//   RAM Hits:                              11|11                   (No change)
//   Total read+write:                    3414|3414                 (No change)
//   Estimated Cycles:                    3792|3792                 (No change)
