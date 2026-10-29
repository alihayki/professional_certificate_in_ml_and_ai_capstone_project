"""Shared data-loading and execution logic for benchmark runners."""

from pathlib import Path

import numpy as np

from utility.bayesian_optimizer import BayesianOptimizer


class BenchmarkRunner:
    """Load one benchmark's observations and propose its next input."""

    FUNCTION_ID = None
    DIMENSIONS = None
    ADDITIONAL_INPUTS = None
    ADDITIONAL_OUTPUTS = None
    LENGTH_SCALE = 0.3
    N_RESTARTS_OPTIMIZER = 5
    N_STARTS = 20
    ACQUISITION = "ei"

    def __init__(self):
        if self.FUNCTION_ID is None or self.DIMENSIONS is None:
            raise TypeError("BenchmarkRunner subclasses must define a function ID and dimensions.")

        data_dir = Path(__file__).resolve().parents[1] / "initial_data" / f"function_{self.FUNCTION_ID}"
        initial_inputs = np.load(data_dir / "initial_inputs.npy")
        initial_outputs = np.load(data_dir / "initial_outputs.npy")
        self.X_train = np.vstack((initial_inputs, self.ADDITIONAL_INPUTS))
        self.y_train = np.concatenate((initial_outputs.reshape(-1), self.ADDITIONAL_OUTPUTS))

        if self.X_train.shape[1] != self.DIMENSIONS:
            raise ValueError(f"Function {self.FUNCTION_ID} expects {self.DIMENSIONS} input dimensions.")
        if self.X_train.shape[0] != self.y_train.size:
            raise ValueError("Every training input must have a corresponding output.")

    def run(self):
        """Propose one new input and return the augmented inputs and best value."""
        optimizer = BayesianOptimizer(
            X_train=self.X_train,
            y_train=self.y_train,
            bounds=[(0.0, 1.0)] * self.DIMENSIONS,
            acquisition=self.ACQUISITION,
            length_scale=self.LENGTH_SCALE,
            n_restarts_optimizer=self.N_RESTARTS_OPTIMIZER,
            n_starts=self.N_STARTS,
        )
        next_input = optimizer.suggest_next()
        X_samples = np.vstack((self.X_train, next_input))
        best_values = [self.y_train.min()]

        print(f"Function {self.FUNCTION_ID}: proposed next input")
        print(next_input)
        print("Best observed output:", best_values[0])
        return X_samples, best_values
