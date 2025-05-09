use log::trace;
use std::{error::Error, fmt, ops::ControlFlow};
use stepwise::{fixed_iters, Algo, Driver};

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct CountingSolver(usize);

impl CountingSolver {
    fn x(&self) -> usize {
        self.0
    }
}

impl Algo for CountingSolver {
    type Error = std::fmt::Error;

    fn step(&mut self) -> (ControlFlow<()>, Result<(), Self::Error>) {
        trace!("adding 1 to {} (MAX-{})", self.0, usize::MAX - self.0);

        let res = self.0.checked_add(1).ok_or(fmt::Error);
        match res {
            Ok(v) => self.0 = v,
            Err(e) => return (ControlFlow::Break(()), Err(e)),
        };
        (ControlFlow::Continue(()), Ok(()))
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let algo = CountingSolver(3);
    let (algo, step) = fixed_iters(algo, 5).solve()?;
    assert_eq!(algo.x(), 8);
    assert_eq!(step.iteration(), 5);
    Ok(())
}
