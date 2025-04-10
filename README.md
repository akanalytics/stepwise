# Overview

Stepwise is a helper library for iterative numeric solvers. 

The solvers themselves are external to the library, though it is strightforward to adapt existing published solvers. As a library, rather than a framework, most modules can be used independently without the need to adopt other parts.

Example of iterative numneric solvers are Gradient Descent, Conjugate Gradient, Fixed Point Iteration, Newton and Quasi-Newton Methods, Stochastic Perturbation etc.



The crate offers 

1. [Driver](#driver) an executer that handles the iteration, conditions for convergence or divergence, logging, and progress display

2. [Iteration Step](#iteration_step) a standardized structure that has iteration count, elapsed time, estimated time to completion, as well as access to the underlying solver

3. [Linear algebra](#linear_algebra) basic functions applicable to Rust slices, and hence typically can be used standalone, or with other linear algebrara libraries offering an as_slice type of conversion

4. [Metrics](#metrics) that can be used to monitor or assess convergence 

5. [Containers](#containers) 
for sampling convergence trajectories, or recording epochs/iterations for later tabulation, plotting or analysis

6. [Mini-Batcher](#mini_batcher): a mini-batch generator capable of holding, sorting, and randomizing slices of training data, and used in processes like SGD

7. [Examples](#examples): examples for common problems


# 1. <a name="driver">Driver</a>
[`Driver`], that handles the iteration, conditions for convergence or divergence, logging, and progress display
```
// my solver implements a step() method and a method 
// for retrieval of best solution so far
// typically constructed from 
//  - hyper parameters, 
//  - the problem or objective function, and 
//  - initial conditions

let my_solver = ...;

// the Driver has functional style iteration control methods,
// along with a `try_solve` which returns the final iteration step along with the solution

let driver = Driver::new(my_solver)
     .fixed_iterations(1000)
     .fail_if(|step| step.x().norm_max() > 1.5 ) 
     .converged_when(|step| step.x().norm_max() < 0.001 ) 
     .for_each(|step| println!("{step}"));


 let solution = driver.try_solve().expect("solving failed!");

 assert_approx_eq!(solution.x(), &[0.50, 0.75, 1.25]);
```
# 2. <a name="iteration_step">Iteration Step</a>
A standardized iteration [`Step`] that has iteration count, elapsed time, estimated time to completion, as well as access to the underlying solver

# 3. <a name="linear_algebra">Linear Algebra</a>
Some very basic linear albegra applicable to Rust slices, and hence typically can be used standalone, or with other linear algebrara libraries offering an as_slice type of conversion
- dot product
- norms, l2/euclidean, l-inf or max_norms, distance, 
- component-wise addition and subtraction
- formatting / display
- rademach

# 4. <a name="metrics">Metrics</a>
Metrics that can be used to monitor or assess convergence 

# 5. <a name="containers">Containers</a>
Containers for sampling convergence trajectories, or recording epochs/iterations for later tabulation, plotting or analysis


# 6. <a name="mini_batcher">Mini-Batcher</a>
A mini-batch generator capable of holding, sorting, and randomizing slices of training data, and used in processes like SGD
 
# 6. <a name="examples">Examples</a>
Examples for common issues and a FAQ

