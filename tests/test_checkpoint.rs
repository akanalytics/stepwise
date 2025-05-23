mod common;
use common::CountingAlgo;
use std::{error::Error, fs, ops::Not, time::Duration};
use stepwise::{fixed_iters, format_duration, log_trace, RecoveryFile, Step};

// ids as tests run in parallel
fn setup(id: &str) -> (CountingAlgo, Step) {
    let algo = CountingAlgo(3);
    let d = Duration::ZERO;
    fixed_iters(algo, 2)
        .checkpoint(format!("/tmp/stepwise/checkpoint-{id}") + "-{iter}.json", d)
        .solve()
        .unwrap()
}

#[test]
fn checkpoint_write() -> Result<(), Box<dyn Error>> {
    let (algo, step) = setup("write");
    assert_eq!(algo.x(), 5);
    assert_eq!(step.iteration(), 2);
    // the iterations are 1 and 2
    assert!(fs::exists("/tmp/stepwise/checkpoint-write-0000000.json")
        .unwrap()
        .not());
    assert!(fs::exists("/tmp/stepwise/checkpoint-write-0000001.json").unwrap());
    assert!(fs::exists("/tmp/stepwise/checkpoint-write-0000002.json").unwrap());
    assert!(fs::exists("/tmp/stepwise/checkpoint-write-0000003.json")
        .unwrap()
        .not());
    Ok(())
}

#[test]
fn checkpoint_replace_params() {
    let algo = CountingAlgo(3);
    let d = Duration::ZERO;
    let (solved, step) = fixed_iters(algo, 1)
        .checkpoint(
            "/tmp/stepwise/checkpoint-params-{iter}-{iter}-{pid}-{elapsed}.json",
            d,
        )
        .solve()
        .unwrap();
    assert_eq!(solved.x(), 4);
    assert_eq!(step.iteration(), 1);
    let pid = std::process::id().to_string();
    let elapsed = format_duration(step.elapsed());

    let file = format!("/tmp/stepwise/checkpoint-params-0000001-0000001-{pid}-{elapsed}.json");
    log_trace!("{file}");
    assert!(fs::exists(file).unwrap());
}

#[test]
fn checkpoint_twice() {
    let algo = CountingAlgo(3);
    let d = Duration::ZERO;
    let (solved, step) = fixed_iters(algo, 1)
        .checkpoint("/tmp/stepwise/checkpoint-c1-{pid}-{iter}.json", d)
        .checkpoint("/tmp/stepwise/checkpoint-c2-{pid}-{iter}.json", d)
        .solve()
        .unwrap();
    assert_eq!(solved.x(), 4);
    assert_eq!(step.iteration(), 1);
    let pid = std::process::id().to_string();
    assert!(fs::exists(format!("/tmp/stepwise/checkpoint-c1-{pid}-0000001.json")).unwrap());
    assert!(fs::exists(format!("/tmp/stepwise/checkpoint-c2-{pid}-0000001.json")).unwrap());
}
#[test]
fn checkpoint_ignore() -> Result<(), Box<dyn Error>> {
    // recovery ignored
    // so counter as above
    let (_, _) = setup("ignore");
    let algo = CountingAlgo(3);
    let (algo, step) = fixed_iters(algo, 2)
        .recovery(
            "/tmp/stepwise/checkpoint-ignore-0000000.json",
            RecoveryFile::Ignore,
        )
        .on_step(|a, s| println!("{a:?} {s:?}"))
        .solve()?;
    assert_eq!(algo.x(), 5);
    assert_eq!(step.iteration(), 2);
    Ok(())
}

#[test]
fn checkpoint_used2() -> Result<(), Box<dyn Error>> {
    // recovery used but 0th save,
    // so counter as above
    let (_, _) = setup("used2");
    let algo = CountingAlgo(3);
    let (algo, step) = fixed_iters(algo, 2)
        .recovery(
            "/tmp/stepwise/checkpoint-used2-0000002.json",
            RecoveryFile::Require,
        )
        .on_step(|a, s| println!("{a:?} {s:?}"))
        .solve()?;
    assert_eq!(algo.x(), 5);
    assert_eq!(step.iteration(), 2);
    Ok(())
}

#[test]
fn checkpoint_used1() -> Result<(), Box<dyn Error>> {
    // recovery used but before 1st save,
    // so counter = 3 + 2 = 5
    let (_, _) = setup("used1");
    let algo = CountingAlgo(3);
    let (algo, step) = fixed_iters(algo, 2)
        .recovery(
            "/tmp/stepwise/checkpoint-used1-0000001.json",
            RecoveryFile::Require,
        )
        .on_step(|a, s| println!("{a:?} {s:?}"))
        .solve()?;
    assert_eq!(algo.x(), 5);
    assert_eq!(step.iteration(), 2);
    Ok(())
}

#[test]
fn test_create_dirs() {
    let algo = CountingAlgo(3);
    let d = Duration::ZERO;

    // ignore errors if dir does not exist
    let _ = fs::remove_dir_all("/tmp/stepwise/stepwise-checkpoints/");
    // write checkpoints
    let (solved, _) = fixed_iters(algo.clone(), 2)
        .checkpoint(
            "/tmp/stepwise/stepwise-checkpoints/checkpoint-{iter}.json",
            d,
        )
        .solve()
        .unwrap();
    assert_eq!(solved.x(), 5);

    let mut vec = Vec::new();
    let (solved, _) = fixed_iters(algo, 2)
        .recovery(
            "/tmp/stepwise/stepwise-checkpoints/checkpoint-0000002.json",
            RecoveryFile::Require,
        )
        .on_step(|algo, step| {
            println!("{step:?}:  {algo:?} ");
            vec.push(step.iteration());
        })
        .solve()
        .unwrap();
    assert_eq!(solved.x(), 5);
    assert_eq!(vec.len(), 1);
}
