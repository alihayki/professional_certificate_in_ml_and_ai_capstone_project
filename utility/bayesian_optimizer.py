"""Gaussian-process optimizer for proposing one black-box evaluation."""

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF

from utility.acquisition_functions import expected_improvement, lower_confidence_bound


class BayesianOptimizer:
    """Fit a Gaussian process and select candidates for objective minimization."""

    def __init__(
        self,
        X_train,
        y_train,
        bounds,
        acquisition="ei",
        length_scale=0.3,
        n_restarts_optimizer=5,
        n_starts=20,
        random_state=None,
    ):
        self.X_train = np.asarray(X_train, dtype=float)
        self.y_train = np.asarray(y_train, dtype=float).reshape(-1)
        self.bounds = np.asarray(bounds, dtype=float)
        self.acquisition = acquisition.lower()
        self.length_scale = length_scale
        self.n_restarts_optimizer = n_restarts_optimizer
        self.n_starts = n_starts
        self.random_state = random_state

        if self.X_train.ndim != 2:
            raise ValueError("X_train must be a two-dimensional array.")
        if self.y_train.size != self.X_train.shape[0]:
            raise ValueError("X_train and y_train must contain the same number of observations.")
        if self.bounds.shape != (self.X_train.shape[1], 2):
            raise ValueError("Provide one lower and upper bound for each input dimension.")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]):
            raise ValueError("Each lower bound must be less than its upper bound.")
        if self.acquisition not in {"ei", "lcb"}:
            raise ValueError("acquisition must be either 'ei' or 'lcb'.")
        if length_scale <= 0 or n_restarts_optimizer < 0 or n_starts < 1:
            raise ValueError("Kernel length scale must be positive and optimization counts valid.")

    def suggest_next(self):
        """Return the candidate with the greatest improvement or lowest LCB."""
        kernel = ConstantKernel(1.0) * RBF(length_scale=self.length_scale)
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-6,
            n_restarts_optimizer=self.n_restarts_optimizer,
        )
        model.fit(self.X_train, self.y_train)

        def negative_acquisition(point):
            mean, std = model.predict(point.reshape(1, -1), return_std=True)
            if self.acquisition == "ei":
                score = expected_improvement(mean, std, self.y_train.min())
                return -float(np.asarray(score).item())
            else:
                score = lower_confidence_bound(mean, std, kappa=2.0)
                return float(np.asarray(score).item())

        rng = np.random.default_rng(self.random_state)
        lower_bounds, upper_bounds = self.bounds.T
        best_result = None

        for _ in range(self.n_starts):
            initial_point = rng.uniform(lower_bounds, upper_bounds)
            result = minimize(
                negative_acquisition,
                initial_point,
                bounds=self.bounds.tolist(),
                method="L-BFGS-B",
            )
            if not np.isfinite(result.fun) or not np.all(np.isfinite(result.x)):
                continue
            if best_result is None or result.fun < best_result.fun:
                best_result = result

        if best_result is None:
            raise RuntimeError("Acquisition optimization did not produce a valid candidate.")
        return best_result.x
