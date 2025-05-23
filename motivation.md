# Background & Motivation
During lockdown, my hobby was writing a chess engine — a program that plays chess and competes in computer chess championships.

This involved:
- Optimizing a hand-crafted chess position evaluator (using logistic regression)  
- Training a neural network-based position evaluator (via gradient descent and backpropagation)  
- Tuning non-differentiable engine search parameters (using SPSA and RSPSA)  
- Solving runtime parameters adaptively (with bisection search or Brent's method)  
- A mix of Python (for machine learning pipelines and plotting) and Rust (for engine search and feature extraction)


My particular implementations required:
- Complex and stateful convergence criteria  
- Algorithms where the "problem", "solver", and batched "data" were deeply entangled  
- Long-running training processes requiring checkpointing and recovery across multiple standard checkpoint files  
- Problems with highly irregular solution spaces, indicated via typed error returns  
- Solvers that could handle these irregular spaces, returning and recovering from errors  
- Mixed-language programming, which required eliminating generics from structs to expose them to Python  
- Zero-cost abstraction for lightweight, runtime-efficient solvers  
- Adaptive solvers needing dynamic tuning of hyperparameters — not just passive state observation

Both **Bullet** and **Argmin** came close to being solid solutions, and I’ve used both to great effect. But ultimately, I ended up rolling my own.

Andy


# Existing crates
There are some very good numeric and ML crates for Rust. To highlight a few..
    

| Crate/Github                                                                | Description                                                                 |
| ------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| **Optimization** |  |
| [argmin](https://crates.io/crates/argmin)                           | General optimization toolbox: BFGS, Gauss-Newton, simulated annealing, etc. |
| [optimization-engine](https://crates.io/crates/optimization-engine) | Fast solver for nonlinear constrained optimization (e.g., MPC, robotics)    |
| [peroxide](https://crates.io/crates/peroxide)               | Scientific computing: linear algebra, stats, ODE, plotting |
| &nbsp; |  |
| **Machine Learning** |  |
| [burn](https://crates.io/crates/burn)                   | Deep learning framework similar to torch                             |
| [candle](https://github.com/huggingface/candle)         | Deep learning framework from Hugging Face (focus on inference)       |
| [dfdx](https://crates.io/crates/dfdx)                   | Fast pure-Rust autodiff + neural networks                            |
| [tch](https://crates.io/crates/tch)                     | PyTorch (libtorch) bindings via FFI                                  |
| [linfa](https://crates.io/crates/linfa)                 | Classical ML: KMeans, SVM, logistic regression                       |
| [rusty-machine](https://crates.io/crates/rusty-machine) | Early ML crate                                                       |
| &nbsp; |  |
| **General utility** |  |
| [ndarray](https://crates.io/crates/ndarray)                 | N-dimensional array (like NumPy)                           |
| [nalgebra](https://crates.io/crates/nalgebra)               | General-purpose linear algebra                             |
| [faer](https://crates.io/crates/faer)                       | Fast, low-level linear algebra with focus on performance and portability |
| [autodiff](https://crates.io/crates/autodiff)           | Lightweight forward-mode autodiff                                    |
| &nbsp; |  |
| **Chess specific** |  |
| [bullet](https://github.com/jw1912/bullet) | NN trainer for chess engines |
| [nnue pytorch](https://github.com/official-stockfish/nnue-pytorch) | Used by stockfish |
| [marlin flow](https://github.com/jnlt3/marlinflow) | Used by Black Marlin and other engines |


