"""Stable scalar functions used by analytical and empirical dynamics."""

from __future__ import annotations

import numpy as np


def sigmoid(values):
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def thermal_probability(beta: float, fields):
    """Return sigmoid(beta * field), including the exact beta=infinity limit."""

    fields = np.asarray(fields, dtype=float)
    if np.isinf(beta):
        return np.where(fields > 0.0, 1.0, np.where(fields < 0.0, 0.0, 0.5))
    return sigmoid(beta * fields)


def thermal_mean_action(beta: float, fields):
    """Return tanh(beta * field / 2), including the beta=infinity limit."""

    fields = np.asarray(fields, dtype=float)
    if np.isinf(beta):
        return np.sign(fields)
    return np.tanh(0.5 * beta * fields)


def sign_nonzero(values):
    return np.where(np.asarray(values) >= 0.0, 1.0, -1.0)


def correlation_root(matrix: np.ndarray) -> np.ndarray:
    matrix = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(matrix)
    values = np.maximum(values, 0.0)
    return (vectors * np.sqrt(values)) @ vectors.T
