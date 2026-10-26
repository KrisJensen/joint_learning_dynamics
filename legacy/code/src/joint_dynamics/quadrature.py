"""Split two-dimensional Gauss-Hermite quadrature utilities."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr, ndtri


@lru_cache(maxsize=None)
def normal_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for an expectation under N(0, 1)."""

    nodes, weights = hermgauss(order)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


@lru_cache(maxsize=None)
def normal_2d(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes, weights = normal_1d(order)
    first, second = np.meshgrid(nodes, nodes, indexing="ij")
    w_first, w_second = np.meshgrid(weights, weights, indexing="ij")
    return first, second, w_first * w_second


@lru_cache(maxsize=None)
def uniform_from_normal(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform-[0,1] quadrature obtained by applying Phi to GH nodes."""

    nodes, weights = normal_1d(order)
    return ndtr(nodes), weights


def normal_interval(
    mean,
    std: float,
    low_probability,
    high_probability,
    uniform_nodes,
):
    """Map uniform quadrature nodes into a normal CDF interval."""

    low = np.asarray(low_probability, dtype=float)
    high = np.asarray(high_probability, dtype=float)
    length = np.maximum(high - low, 0.0)
    probability = low[..., None] + length[..., None] * uniform_nodes
    probability = np.clip(probability, 1e-15, 1.0 - 1e-15)
    return np.asarray(mean)[..., None] + std * ndtri(probability), length


def conditional_u_parameters(state_matrix: np.ndarray, h, v) -> tuple[np.ndarray, float]:
    """Mean and standard deviation of U conditional on H and V."""

    cov_hv = state_matrix[1:, 1:]
    cov_u_hv = state_matrix[0, 1:]
    inverse = np.linalg.pinv(cov_hv)
    coefficients = cov_u_hv @ inverse
    mean = coefficients[0] * h + coefficients[1] * v
    variance = max(float(1.0 - cov_u_hv @ inverse @ cov_u_hv.T), 0.0)
    return mean, variance**0.5


def signed_normal_statistics(mean, std: float, sign: float):
    """Mass and first raw moment of a normal variable on one sign half-line."""

    mean = np.asarray(mean, dtype=float)
    if std <= 1e-12:
        keep = mean >= 0.0 if sign > 0.0 else mean < 0.0
        mass = keep.astype(float)
        return mass, mean * mass
    ratio = mean / std
    density = np.exp(-0.5 * ratio * ratio) / np.sqrt(2.0 * np.pi)
    if sign > 0.0:
        mass = ndtr(ratio)
        first = mean * mass + std * density
    else:
        mass = ndtr(-ratio)
        first = mean * mass - std * density
    return mass, first
