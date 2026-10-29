"""Acquisition scores that balance predicted performance and uncertainty."""

import numpy as np
from scipy.stats import norm


def upper_confidence_bound(mu, sigma, kappa=2.0):
    """
    Upper Confidence Bound (UCB) acquisition function.

    UCB = mean + kappa * std

    Parameters:
    -----------
    mu : predicted mean
    sigma : predicted standard deviation
    kappa : exploration parameter (higher = more exploration)
    """
    return mu + kappa * sigma


def lower_confidence_bound(mu, sigma, kappa=2.0):
    """Lower Confidence Bound (LCB), used to select low predicted values."""
    return mu - kappa * sigma


def expected_improvement(mu, sigma, y_best, xi=7):
    """
    Expected Improvement (EI) for minimizing an objective function.

    EI = E[max(y_best - f(x) - xi, 0)]

    Parameters:
    -----------
    mu : predicted mean
    sigma : predicted standard deviation
    y_best : best observed value so far
    xi : exploration parameter
    """
    with np.errstate(divide='warn'):
        # Positive improvement means a prediction lower than the best observation.
        improvement = y_best - mu - xi
        Z = improvement / sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        # Avoid an undefined division result when the model predicts no uncertainty.
        ei[sigma == 0.0] = 0.0
    return ei