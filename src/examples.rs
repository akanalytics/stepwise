use crate::driver::Solver;
use std::{error::Error, ops::ControlFlow};

pub struct GradientDescent<G> {
    pub gradient: G,
    pub x: [f64; 2],
    pub learning_rate: f64,
}

impl<G> GradientDescent<G>
where
    G: Fn([f64; 2]) -> [f64; 2],
{
    pub fn new(learning_rate: f64, x0: [f64; 2], gradient: G) -> Self {
        Self {
            gradient,
            x: x0,
            learning_rate,
        }
    }
}

impl<G> Solver for GradientDescent<G>
where
    G: Fn([f64; 2]) -> [f64; 2],
{
    fn step(&mut self) -> Result<ControlFlow<()>, Box<dyn Error>> {
        let [x, y] = self.x;
        let [dfdx, dfdy] = (self.gradient)([x, y]);
        self.x = [x - self.learning_rate * dfdx, y - self.learning_rate * dfdy];

        // allow stepwise Driver to decide when to cease iteration
        Ok(ControlFlow::Continue(()))
    }

    fn x(&self) -> &[f64] {
        self.x.as_slice()
    }
}

pub fn example_solver() -> impl Solver {
    GradientDescent {
        gradient: |[x, y]: [f64; 2]| [(x - 1.5), (y - 2.0)],
        x: [5.0, 6.0],
        learning_rate: 0.01,
    }
}

/// 1 / [1+e^(-x)]
pub fn sigmoid(x: f64) -> f64 {
    1. / (1. + f64::exp(-x))
}

/// n-sphere centered at origin
pub fn sphere(x: &[f64]) -> f64 {
    x.iter().map(|x| x * x).sum()
}
