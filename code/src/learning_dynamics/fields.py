"""Numerically stable scalar field functions."""

from __future__ import annotations

import numpy as np


def sigmoid(values):
    """Evaluate the logistic sigmoid without overflow."""

    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def thermal_probability(beta: float, fields):
    """Return ``sigmoid(beta * fields)``, including ``beta = infinity``."""

    fields = np.asarray(fields, dtype=float)
    if np.isinf(beta):
        return np.where(fields > 0.0, 1.0, np.where(fields < 0.0, 0.0, 0.5))
    return sigmoid(beta * fields)


def thermal_mean_action(beta: float, fields):
    """Return the conditional mean binary action."""

    fields = np.asarray(fields, dtype=float)
    if np.isinf(beta):
        return np.sign(fields)
    return np.tanh(0.5 * beta * fields)


def sign_nonzero(values):
    """Map zero to +1, consistently with the measure-zero sign convention."""

    return np.where(np.asarray(values) >= 0.0, 1.0, -1.0)


def correlation_root(matrix: np.ndarray) -> np.ndarray:
    """Return a symmetric positive-semidefinite square root."""

    matrix = 0.5 * (np.asarray(matrix) + np.asarray(matrix).T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
