"""Bounded instantaneous control optimization."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb, fsum

import numpy as np
from scipy.optimize import minimize_scalar

from .config import Params, State
from .fields import thermal_probability
from .moments import (
    Aggregates,
    Moments,
    OnlineAggregates,
    aggregate_online,
    compute_moments,
    environment_reward_probability,
    prepare_moment_quadrature,
)
from .quadrature import normal_2d


@dataclass(frozen=True)
class Rates:
    eta_on1: float = 0.0
    eta_on2: float = 0.0
    eta_off: float = 0.0
    eta_wm: float = 0.0


@dataclass(frozen=True)
class KOptimization:
    K: int
    beta: float
    moments: Moments
    objective_value: float
    values: dict[int, float]


@dataclass(frozen=True)
class PrefixLengthOptimization:
    prefix_length: int
    beta: float
    moments: Moments
    objective_value: float
    values: dict[int, float]


@dataclass(frozen=True)
class PolicyResolution:
    params: Params
    beta: float
    moments: Moments


def greedy_policy_params(params: Params) -> Params:
    """Apply the exact K=1, beta=infinity greedy-policy limit."""

    return replace(
        params,
        K=1,
        K_mode="constant",
        beta=float("inf"),
        temperature_mode="constant",
    )


def expected_planner_acceptance(state: State, params: Params, beta: float) -> float:
    """Probability that at least one candidate is WM-correct."""

    theta, xi, weights = normal_2d(params.gh_order)
    residual = max(1.0 - state.rho_sm**2, 0.0) ** 0.5
    y = state.rho_sm * np.abs(theta) + residual * xi
    probability = thermal_probability(beta, y)
    power = np.ones_like(probability)
    terms = []
    for j in range(1, params.K + 1):
        power = power * probability
        a_j = float(np.sum(weights * power))
        terms.append(
            (-1) ** (j + 1)
            * comb(params.K, j)
            * a_j**params.effective_prefix_length
        )
    return float(np.clip(fsum(terms), 0.0, 1.0))


def optimal_temperature(state: State, params: Params) -> float:
    objective = lambda beta: -expected_planner_acceptance(state, params, float(beta))
    result = minimize_scalar(
        objective,
        bounds=(0.0, params.beta_max),
        method="bounded",
        options={"xatol": params.beta_tol},
    )
    candidates = [0.0, params.beta_max, float(result.x)]
    return float(max(candidates, key=lambda value: -objective(value)))


def effective_temperature(state: State, params: Params) -> float:
    return params.beta if params.temperature_mode == "constant" else optimal_temperature(state, params)


def _best_scalar(linear: float, rho: float, quadratic: float, T: int, cap: float) -> float:
    def value(eta):
        return eta * linear - rho * quadratic * eta * eta / (2.0 * T)

    candidates = [0.0, cap]
    denominator = rho * quadratic
    if abs(denominator) > 1e-14:
        candidates.append(float(np.clip(T * linear / denominator, 0.0, cap)))
    return float(max(candidates, key=value))


def _best_online_pair(
    state: State,
    params: Params,
    aggregates: OnlineAggregates,
) -> tuple[float, float]:
    rho = state.rho_se
    T = params.T
    L0 = aggregates.base.B - rho * aggregates.base.C
    LR = aggregates.observed_reward.B - rho * aggregates.observed_reward.C
    D0 = aggregates.base.D
    DR = aggregates.observed_reward.D
    cap1 = 0.0 if params.fix_eta_on1_zero else params.rate_cap("eta_on1")
    cap2 = 0.0 if params.fix_eta_on2_zero else params.rate_cap("eta_on2")

    # These are axis-constrained optima.  They must be optimized on the
    # remaining axis, rather than obtained by zeroing one component of the
    # unconstrained joint optimum.
    if params.fix_eta_on1_zero and params.fix_eta_on2_zero:
        return 0.0, 0.0
    if params.fix_eta_on1_zero:
        return 0.0, _best_scalar(LR, rho, DR, T, cap2)
    if params.fix_eta_on2_zero:
        return _best_scalar(L0, rho, D0, T, cap1), 0.0

    def value(pair):
        first, second = pair
        quadratic = first * first * D0 + (2.0 * first * second + second * second) * DR
        return first * L0 + second * LR - rho * quadratic / (2.0 * T)

    candidates = [(x, y) for x in (0.0, cap1) for y in (0.0, cap2)]
    matrix = np.array([[D0, DR], [DR, DR]], dtype=float)
    if abs(rho) > 1e-14:
        stationary = np.linalg.pinv(matrix) @ (T / rho * np.array([L0, LR]))
        if -1e-12 <= stationary[0] <= cap1 + 1e-12 and -1e-12 <= stationary[1] <= cap2 + 1e-12:
            candidates.append((float(stationary[0]), float(stationary[1])))
        for first in (0.0, cap1):
            if abs(rho * DR) > 1e-14:
                second = np.clip(T * LR / (rho * DR) - first, 0.0, cap2)
                candidates.append((first, float(second)))
        for second in (0.0, cap2):
            if abs(rho * D0) > 1e-14:
                first = np.clip((T * L0 / rho - DR * second) / D0, 0.0, cap1)
                candidates.append((float(first), second))
    return max(candidates, key=value)


def effective_online_student_rates(
    state: State,
    params: Params,
    online: OnlineAggregates,
) -> tuple[float, float]:
    """Return online student rates, including all hard zero constraints."""

    if params.learning_rate_mode == "optimal":
        return _best_online_pair(state, params, online)
    return (
        0.0 if params.fix_eta_on1_zero else params.eta_on1,
        0.0 if params.fix_eta_on2_zero else params.eta_on2,
    )


def online_student_drift_numerator(
    state: State,
    params: Params,
    online: OnlineAggregates,
    eta_on1: float,
    eta_on2: float,
) -> float:
    """Per-update numerator of the student-environment online drift."""

    rho = state.rho_se
    quadratic = (
        eta_on1**2 * online.base.D
        + (2.0 * eta_on1 * eta_on2 + eta_on2**2) * online.observed_reward.D
    )
    return float(
        eta_on1 * (online.base.B - rho * online.base.C)
        + eta_on2 * (online.observed_reward.B - rho * online.observed_reward.C)
        - rho * quadratic / (2.0 * params.T)
    )


def optimize_K(state: State, params: Params, objective: str | None = None) -> KOptimization:
    """Evaluate and retain an exact discrete K optimization over ``1..K_max``.

    If temperature or learning rates are adaptive, they are reoptimized for
    every candidate K. The environment-drift term is omitted from the drift
    objective because it is independent of K. State-dependent split-quadrature
    geometry is constructed once and reused by every candidate.
    """

    objective = params.K_mode if objective is None else objective
    if objective not in {"reward", "drift"}:
        raise ValueError("K objective must be 'reward' or 'drift'")
    state.validate()
    params.validate()
    values = {}
    candidates = {}
    quadrature = prepare_moment_quadrature(state, params)
    for K in range(1, params.K_max + 1):
        candidate = replace(params, K=K, K_mode="constant")
        beta = effective_temperature(state, candidate)
        moments = compute_moments(state, candidate, beta, quadrature)
        if objective == "reward":
            numerator = environment_reward_probability(moments, candidate)
        else:
            online = aggregate_online(moments, candidate)
            eta_on1, eta_on2 = effective_online_student_rates(state, candidate, online)
            numerator = online_student_drift_numerator(
                state, candidate, online, eta_on1, eta_on2
            )
        values[K] = float(numerator / candidate.c_on)
        candidates[K] = (beta, moments)
    selected = max(values, key=values.get)
    beta, moments = candidates[selected]
    return KOptimization(selected, beta, moments, values[selected], values)


def optimize_prefix_length(state: State, params: Params) -> PrefixLengthOptimization:
    """Choose prefix length maximizing the reward-independent eta_on1 signal.

    For each candidate prefix length M, beta is resolved according to
    ``temperature_mode`` for that candidate. The objective is the linear
    eta_on1 coefficient B0 - rho_se C0, before learning-rate optimization.
    """

    state.validate()
    params.validate()
    maximum = params.T if params.prefix_length_max is None else params.prefix_length_max
    values = {}
    candidates = {}
    quadrature = prepare_moment_quadrature(state, params)
    for prefix_length in range(1, maximum + 1):
        candidate = replace(
            params,
            prefix_length=prefix_length,
            prefix_length_mode="constant",
        )
        beta = effective_temperature(state, candidate)
        moments = compute_moments(state, candidate, beta, quadrature)
        online = aggregate_online(moments, candidate)
        values[prefix_length] = float(online.base.B - state.rho_se * online.base.C)
        candidates[prefix_length] = (beta, moments)
    selected = max(values, key=values.get)
    beta, moments = candidates[selected]
    return PrefixLengthOptimization(selected, beta, moments, values[selected], values)


def prefix_length_objective_values(state: State, params: Params) -> dict[int, float]:
    """Return eta_on1-signal objective values over candidate prefix lengths."""

    return optimize_prefix_length(state, params).values


def optimal_prefix_length(state: State, params: Params) -> int:
    """Return the smallest prefix length maximizing the eta_on1 signal."""

    if params.prefix_length_mode == "constant":
        return params.effective_prefix_length
    return optimize_prefix_length(state, params).prefix_length


def K_objective_values(state: State, params: Params, objective: str) -> dict[int, float]:
    """Return exact discrete online-K objective values over ``1..K_max``."""

    return optimize_K(state, params, objective).values


def optimal_K(state: State, params: Params, objective: str | None = None) -> int:
    """Return the smallest exact integer maximizer of the requested objective."""

    objective = params.K_mode if objective is None else objective
    if objective == "constant":
        return params.K
    return optimize_K(state, params, objective).K


def effective_K(state: State, params: Params) -> int:
    return optimal_K(state, params)


def resolve_online_policy(
    state: State,
    params: Params,
    *,
    greedy: bool = False,
) -> PolicyResolution:
    """Resolve K, beta, and moments for an online training/evaluation policy."""

    if greedy:
        active = greedy_policy_params(params)
        return PolicyResolution(
            active,
            active.beta,
            compute_moments(state, active, active.beta),
        )
    if params.K_mode != "constant":
        result = optimize_K(state, params)
        params = replace(params, K=result.K, K_mode="constant")
    active = replace(params, K_mode="constant")
    if active.prefix_length_mode != "constant":
        result = optimize_prefix_length(state, active)
        active = replace(
            active,
            prefix_length=result.prefix_length,
            prefix_length_mode="constant",
        )
        return PolicyResolution(active, result.beta, result.moments)
    active = replace(active, prefix_length_mode="constant")
    beta = effective_temperature(state, active)
    return PolicyResolution(active, beta, compute_moments(state, active, beta))


def effective_rates(
    state: State,
    params: Params,
    mode: str,
    online: OnlineAggregates,
    offline: Aggregates,
    *,
    L_m: float,
    d_m: float,
) -> Rates:
    if params.learning_rate_mode == "constant":
        if mode == "online":
            return Rates(
                eta_on1=0.0 if params.fix_eta_on1_zero else params.eta_on1,
                eta_on2=0.0 if params.fix_eta_on2_zero else params.eta_on2,
                eta_wm=0.0 if params.fix_eta_wm_zero else params.eta_wm,
            )
        return Rates(eta_off=params.eta_off)

    if mode == "online":
        eta_on1, eta_on2 = effective_online_student_rates(state, params, online)
        eta_wm = 0.0 if params.fix_eta_wm_zero else _best_scalar(
            L_m, state.rho_me, d_m, params.T, params.rate_cap("eta_wm")
        )
        return Rates(eta_on1=eta_on1, eta_on2=eta_on2, eta_wm=eta_wm)

    K_off = offline.E - state.rho_sm * offline.C
    eta_off = _best_scalar(
        K_off, state.rho_sm, offline.D, params.T, params.rate_cap("eta_off")
    )
    return Rates(eta_off=eta_off)
