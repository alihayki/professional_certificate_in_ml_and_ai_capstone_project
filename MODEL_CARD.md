# Model Card: Gaussian-Process Black-Box Optimizer

## Overview

| Item | Description |
|---|---|
| Approach | Gaussian-process Bayesian optimization |
| Model type | Probabilistic surrogate model for continuous objectives |
| Objective | Minimize the scalar output of each hidden function |
| Implementation | `BayesianOptimizer` in `utility/bayesian_optimizer.py` |
| Benchmark runners | `runner/function_1.py` through `runner/function_8.py` |
| Status | Capstone prototype; candidate suggestions require external evaluation |

## Intended use

The optimizer is intended to suggest input vectors for continuous, bounded black-box functions when evaluations are limited or costly and gradients are unavailable. The project examples use eight functions with 2 to 8 input dimensions, each bounded between 0 and 1.

The optimizer returns a proposed input only. It does not evaluate the hidden function, verify the result, or automatically add a newly measured output to the data.

### Out of scope

- Discrete, categorical, mixed-integer, or unconstrained problems.
- High-dimensional settings substantially larger than the benchmarks without further testing.
- Safety-critical or production decisions without domain-specific validation and human oversight.
- Treating the surrogate's predictions as a substitute for measured function outputs.

## Data

Each benchmark combines the course-provided starter arrays in `initial_data/function_N/` with 12 recorded additional input/output pairs in the corresponding runner class. The objective functions themselves are not included. Input dimensions range from 2 to 8 and are restricted to `[0, 1]`.

| Function | Dimensions | Starter observations | Additional observations |
|---|---:|---:|---:|
| 1 | 2 | 10 | 12 |
| 2 | 2 | 10 | 12 |
| 3 | 3 | 15 | 12 |
| 4 | 4 | 30 | 12 |
| 5 | 4 | 20 | 12 |
| 6 | 5 | 20 | 12 |
| 7 | 6 | 30 | 12 |
| 8 | 8 | 40 | 12 |

The additional observations are sequential optimization results, not an independent or randomly sampled test set. Comparisons across functions are not meaningful because their output scales differ.

## Model and optimization

The Gaussian process uses a constant kernel multiplied by an isotropic RBF kernel. Its noise regularization is set to `alpha=1e-6`; kernel hyperparameters are fit by scikit-learn. The configured initial length scale, number of GP optimizer restarts, and number of acquisition-search starts are set on each runner. The acquisition search uses SciPy's bounded L-BFGS-B optimizer.

The default acquisition is Expected Improvement (EI) for minimization, using `xi=7` and the lowest observed output as its incumbent. Lower Confidence Bound (LCB), computed as `mean - 2 * standard_deviation`, is also supported by the shared optimizer. Runner classes currently select EI.

Random starting points are generated without a fixed seed by default, so candidate proposals can vary between runs. Set `random_state` when creating a `BayesianOptimizer` instance if repeatable starting points are needed.

## Observed results

The following are minima in the current starter and additional observations. They are not estimates of the unknown global minima.

| Function | Lowest starter output | Lowest additional output | Lowest observed overall |
|---|---:|---:|---:|
| 1 | -0.003606 | -1.37e-61 | -0.003606 |
| 2 | -0.065624 | -0.166653 | -0.166653 |
| 3 | -0.398926 | -0.450368 | -0.450368 |
| 4 | -32.625660 | -32.683385 | -32.683385 |
| 5 | 0.112940 | 44.565423 | 0.112940 |
| 6 | -2.571170 | -2.323404 | -2.571170 |
| 7 | 0.002701 | 0.014529 | 0.002701 |
| 8 | 5.592193 | 7.051065 | 5.592193 |

See the [README results chart](images/additional_outputs.svg) for the additional outputs in evaluation order.

## Limitations and risks

- The stationary RBF kernel assumes a smooth response surface; abrupt changes or highly irregular functions may be poorly modeled.
- Sparse observations, especially in higher dimensions, leave much of the search space unexplored. Predictions far from observed points can be unreliable.
- EI and LCB depend on the surrogate's uncertainty estimates. Model misspecification can therefore lead to weak candidate suggestions.
- The default random initialization is not reproducible across runs unless a seed is supplied.
- The current runners append fixed observations and propose one new point. They do not provide an automated sequential loop, held-out evaluation, or performance guarantee.
- Real-world use should validate constraints and candidate behavior before any costly or consequential evaluation.

## Responsible use

This is an experimental decision-support tool, not an autonomous decision-maker. Review proposed inputs, ensure they satisfy application-specific constraints, and confirm any measured outputs before incorporating them into future training data. Avoid deployment where a poor suggestion could cause harm without additional validation and oversight.

## How to run

Install dependencies with `python -m pip install -r requirements.txt`, then run an individual benchmark, for example:

```powershell
python -m runner.function_1
```

See the [README](README.md) for PyCharm instructions and PDF plotting.
