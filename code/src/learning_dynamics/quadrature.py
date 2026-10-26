"""Split Gauss-Hermite quadrature for discontinuous Gaussian integrands."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.special import ndtr, ndtri

from .config import State


@lru_cache(maxsize=None)
def normal_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and weights for an expectation under a standard normal."""

    nodes, weights = hermgauss(order)
    return np.sqrt(2.0) * nodes, weights / np.sqrt(np.pi)


@lru_cache(maxsize=None)
def uniform_rule(order: int) -> tuple[np.ndarray, np.ndarray]:
    """A transformed GH rule for integrals over a uniform probability axis."""

    nodes, weights = normal_1d(order)
    return ndtr(nodes), weights


def conditional_u(state: State, h, v) -> tuple[np.ndarray, float]:
    """Mean and standard deviation of ``U | H,V``."""

    covariance_hv = state.matrix[1:, 1:]
    covariance_u_hv = state.matrix[0, 1:]
    inverse = np.linalg.pinv(covariance_hv, rcond=1e-13)
    coefficients = covariance_u_hv @ inverse
    mean = coefficients[0] * h + coefficients[1] * v
    variance = max(float(1.0 - covariance_u_hv @ inverse @ covariance_u_hv), 0.0)
    return mean, np.sqrt(variance)


def signed_normal_moments(mean, std: float, sign: float):
    """Probability mass and first raw moment on one normal half-line."""

    mean = np.asarray(mean, dtype=float)
    if std <= 1e-12:
        selected = mean >= 0.0 if sign > 0.0 else mean < 0.0
        mass = selected.astype(float)
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


@dataclass(frozen=True)
class QuadratureBlock:
    """One fixed-sign region for ``(U,H,V)``."""

    sign_u: float
    sign_h: float
    sign_v: float
    h: np.ndarray
    v: np.ndarray
    weights: np.ndarray
    u_mass: np.ndarray
    u_first: np.ndarray


@dataclass(frozen=True)
class SplitQuadrature:
    blocks: tuple[QuadratureBlock, ...]


@dataclass(frozen=True)
class PlannerAcceptanceQuadrature:
    """One-dimensional split rule for planner-acceptance moments.

    The random field ``Y = sign(V) H`` has density

    ``2 phi(y) Phi(rho_sm y / sqrt(1 - rho_sm**2))``.

    Consequently, its distribution—and every planner-acceptance moment—only
    depends on ``rho_sm``.  The arrays below already include the skew-normal
    density factor, so expectations are simple weighted sums over ``y``.
    """

    rho_sm: float
    y: np.ndarray
    weights: np.ndarray


def prepare_planner_acceptance_quadrature(
    rho_sm: float,
    order: int,
) -> PlannerAcceptanceQuadrature:
    """Build a split Gauss-Hermite rule for the marginal planner field.

    Splitting the standard-normal probability interval at zero avoids placing
    a quadrature cell across the sign boundary inherited from ``sign(V)``.
    This one-dimensional rule is substantially cheaper than constructing the
    full ``(U,H,V)`` rule used by the learning dynamics.
    """

    rho_sm = float(rho_sm)
    if not np.isfinite(rho_sm) or abs(rho_sm) > 1.0 + 1e-12:
        raise ValueError("rho_sm must be finite and lie in [-1, 1]")
    rho_sm = float(np.clip(rho_sm, -1.0, 1.0))

    probability_nodes, base_weights = uniform_rule(order)
    probabilities = np.concatenate(
        (0.5 * probability_nodes, 0.5 + 0.5 * probability_nodes)
    )
    y = ndtri(np.clip(probabilities, 1e-15, 1.0 - 1e-15))

    residual = np.sqrt(max(1.0 - rho_sm**2, 0.0))
    if residual <= 1e-12:
        conditional_probability = (rho_sm * y > 0.0).astype(float)
    else:
        conditional_probability = ndtr(rho_sm * y / residual)

    # Each normal half-line has mass 1/2, while the density of Y contains
    # a factor of two.  These factors cancel, leaving one copy of the base
    # weights for each half-line.
    weights = np.tile(base_weights, 2) * conditional_probability
    return PlannerAcceptanceQuadrature(rho_sm, y, weights)


def prepare_split_quadrature(state: State, order: int) -> SplitQuadrature:
    """Build a rule whose cells never cross a sign discontinuity.

    ``V`` and ``H|V`` are integrated on truncated sign intervals using a
    transformed Gauss-Hermite rule. ``U|H,V`` is integrated analytically over
    each sign half-line.
    """

    state.validate()
    probability_nodes, base_weights = uniform_rule(order)
    blocks: list[QuadratureBlock] = []
    std_h = np.sqrt(max(1.0 - state.rho_sm**2, 0.0))

    for sign_v in (-1.0, 1.0):
        v_low, v_high = ((0.0, 0.5) if sign_v < 0.0 else (0.5, 1.0))
        v_probabilities = v_low + (v_high - v_low) * probability_nodes
        v = ndtri(np.clip(v_probabilities, 1e-15, 1.0 - 1e-15))
        v_weights = (v_high - v_low) * base_weights
        mean_h = state.rho_sm * v

        if std_h <= 1e-12:
            for sign_h in (-1.0, 1.0):
                selected = (mean_h < 0.0) if sign_h < 0.0 else (mean_h >= 0.0)
                if not np.any(selected):
                    continue
                h = mean_h[selected, None]
                v_grid = v[selected, None]
                weights = v_weights[selected, None]
                mean_u, std_u = conditional_u(state, h, v_grid)
                for sign_u in (-1.0, 1.0):
                    mass, first = signed_normal_moments(mean_u, std_u, sign_u)
                    blocks.append(
                        QuadratureBlock(
                            sign_u, sign_h, sign_v, h, v_grid, weights, mass, first
                        )
                    )
            continue

        boundary = np.clip(ndtr(-mean_h / std_h), 0.0, 1.0)
        for sign_h in (-1.0, 1.0):
            low = np.zeros_like(boundary) if sign_h < 0.0 else boundary
            high = boundary if sign_h < 0.0 else np.ones_like(boundary)
            interval = np.maximum(high - low, 0.0)
            probabilities = low[:, None] + interval[:, None] * probability_nodes[None, :]
            h = mean_h[:, None] + std_h * ndtri(
                np.clip(probabilities, 1e-15, 1.0 - 1e-15)
            )
            v_grid = np.broadcast_to(v[:, None], h.shape)
            weights = v_weights[:, None] * interval[:, None] * base_weights[None, :]
            if not np.any(weights):
                continue
            mean_u, std_u = conditional_u(state, h, v_grid)
            for sign_u in (-1.0, 1.0):
                mass, first = signed_normal_moments(mean_u, std_u, sign_u)
                blocks.append(
                    QuadratureBlock(
                        sign_u, sign_h, sign_v, h, v_grid, weights, mass, first
                    )
                )
    return SplitQuadrature(tuple(blocks))
