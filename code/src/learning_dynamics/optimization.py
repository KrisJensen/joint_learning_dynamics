"""Instantaneous bounded optimization of policies and learning rates."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import comb, fsum

import numpy as np
from scipy.optimize import brentq

from .config import Params, State
from .fields import thermal_probability
from .moments import (
    Moments,
    OnlineAggregates,
    aggregate_online,
    compute_moments,
)
from .quadrature import (
    PlannerAcceptanceQuadrature,
    SplitQuadrature,
    prepare_planner_acceptance_quadrature,
    prepare_split_quadrature,
)


# The planner objective is only one-dimensional, so a high-order rule remains
# much cheaper than one evaluation of the full three-field moment table.  This
# minimum resolves the sharpest logistic layer allowed by the intended
# beta_max <= 30 regime while preserving a larger user-requested GH order.
_MIN_PLANNER_GH_ORDER = 256


@dataclass(frozen=True)
class Rates:
    eta_2f: float = 0.0
    eta_3f: float = 0.0
    eta_off: float = 0.0
    eta_wm: float = 0.0


@dataclass(frozen=True)
class PolicyResolution:
    """Resolved online controls and their already-computed moments."""

    params: Params
    beta: float
    moments: Moments
    online: OnlineAggregates
    rates: Rates
    objective: str
    objective_value: float


@dataclass(frozen=True)
class PlannerAcceptanceMoments:
    """The only moments required to optimize planning temperature."""

    A: np.ndarray
    dA: np.ndarray


def greedy_policy_params(params: Params) -> Params:
    """Return the exact ``K=1, beta=infinity`` training policy."""

    return replace(
        params,
        K=1,
        K_mode="constant",
        M_mode="constant",
        beta=float("inf"),
        temperature_mode="constant",
    )


def planner_acceptance(moments: Moments, K: int, M: int) -> float:
    """Return the probability that at least one candidate plan is accepted."""

    return _planner_acceptance_from_A(moments.A, K, M)


def _planner_acceptance_from_A(A: np.ndarray, K: int, M: int) -> float:
    terms = [
        (-1) ** (j + 1) * comb(K, j) * float(A[j]) ** M
        for j in range(1, K + 1)
    ]
    return float(np.clip(fsum(terms), 0.0, 1.0))


def compute_planner_acceptance_moments(
    beta: float,
    K: int,
    quadrature: PlannerAcceptanceQuadrature,
) -> PlannerAcceptanceMoments:
    """Compute ``A_j`` and ``dA_j/d beta`` without unrelated moments.

    Only the marginal planner field ``Y`` enters:

    ``A_j = E[sigma(beta Y)^j]`` and
    ``A'_j = j E[Y sigma(beta Y)^j sigma(-beta Y)]``.
    """

    beta = float(beta)
    if not np.isfinite(beta) or beta < 0.0:
        raise ValueError("bounded temperature optimization requires finite beta >= 0")
    if not isinstance(K, int) or K < 1:
        raise ValueError("K must be a positive integer")

    probability = thermal_probability(beta, quadrature.y)
    complement = 1.0 - probability
    power = np.ones_like(probability)
    A = np.zeros(K + 1, dtype=float)
    dA = np.zeros(K + 1, dtype=float)
    A[0] = 1.0
    for j in range(1, K + 1):
        power *= probability
        A[j] = np.dot(quadrature.weights, power)
        dA[j] = j * np.dot(
            quadrature.weights,
            quadrature.y * power * complement,
        )
    return PlannerAcceptanceMoments(A, dA)


def expected_planner_acceptance(
    state: State,
    params: Params,
    beta: float,
    *,
    quadrature: PlannerAcceptanceQuadrature | None = None,
) -> float:
    """Evaluate plan acceptance using only ``rho_sm`` and planner moments."""

    state.validate()
    params.validate()
    quadrature = quadrature or prepare_planner_acceptance_quadrature(
        state.rho_sm,
        max(params.gh_order, _MIN_PLANNER_GH_ORDER),
    )
    moments = compute_planner_acceptance_moments(beta, params.K, quadrature)
    return _planner_acceptance_from_A(moments.A, params.K, params.M)


def _planner_acceptance_derivative(
    moments: PlannerAcceptanceMoments,
    K: int,
    M: int,
) -> float:
    terms = [
        M
        * (-1) ** (j + 1)
        * comb(K, j)
        * float(moments.A[j]) ** (M - 1)
        * float(moments.dA[j])
        for j in range(1, K + 1)
    ]
    return float(fsum(terms))


def _single_step_planner_acceptance(
    beta: float,
    K: int,
    quadrature: PlannerAcceptanceQuadrature,
) -> tuple[float, float]:
    """Return acceptance and its derivative directly when ``M=1``.

    Evaluating ``1 - (1 - q)^K`` pointwise avoids the ill-conditioned
    alternating binomial sum that otherwise occurs when acceptance is very
    close to one. All operations remain float64.
    """

    probability = thermal_probability(beta, quadrature.y)
    failure = 1.0 - probability
    with np.errstate(divide="ignore", invalid="ignore"):
        acceptance = -np.expm1(K * np.log1p(-probability))
    derivative = (
        K
        * quadrature.y
        * probability
        * np.power(failure, K)
    )
    return (
        float(np.dot(quadrature.weights, acceptance)),
        float(np.dot(quadrature.weights, derivative)),
    )


def optimal_temperature(
    state: State,
    params: Params,
    *,
    quadrature: SplitQuadrature | None = None,
    acceptance_quadrature: PlannerAcceptanceQuadrature | None = None,
) -> tuple[float, Moments]:
    """Optimize a bounded unimodal planner-acceptance objective.

    A split one-dimensional quadrature supplies only the acceptance quantities
    required by the objective. Under the unimodality assumption, the
    derivative either identifies a boundary optimum or has one bracketed
    interior root. Full learning moments are computed exactly once, after
    selecting ``beta``.
    """

    state.validate()
    params.validate()
    acceptance_quadrature = (
        acceptance_quadrature
        or prepare_planner_acceptance_quadrature(
            state.rho_sm,
            max(params.gh_order, _MIN_PLANNER_GH_ORDER),
        )
    )
    cache: dict[float, tuple[float, float]] = {}

    def evaluate(beta: float) -> tuple[float, float]:
        key = float(beta)
        if key not in cache:
            if params.M == 1:
                cache[key] = _single_step_planner_acceptance(
                    key,
                    params.K,
                    acceptance_quadrature,
                )
            else:
                moments = compute_planner_acceptance_moments(
                    key,
                    params.K,
                    acceptance_quadrature,
                )
                cache[key] = (
                    _planner_acceptance_from_A(moments.A, params.K, params.M),
                    _planner_acceptance_derivative(moments, params.K, params.M),
                )
        return cache[key]

    lower = 0.0
    upper = float(params.beta_max)
    derivative_lower = evaluate(lower)[1]
    derivative_upper = evaluate(upper)[1]
    candidates = [lower, upper]
    if derivative_lower > 0.0 and derivative_upper < 0.0:
        root = brentq(
            lambda beta: evaluate(float(beta))[1],
            lower,
            upper,
            xtol=params.beta_tol,
            rtol=8.0 * np.finfo(float).eps,
        )
        candidates.append(float(root))

    # Explicit comparison protects boundary optima and numerical plateaus.
    best_beta = max(candidates, key=lambda beta: evaluate(beta)[0])
    quadrature = quadrature or prepare_split_quadrature(state, params.gh_order)
    best_moments = compute_moments(
        state,
        params,
        best_beta,
        K=params.K,
        quadrature=quadrature,
    )
    return float(best_beta), best_moments


def resolve_temperature(
    state: State,
    params: Params,
    quadrature: SplitQuadrature | None = None,
    acceptance_quadrature: PlannerAcceptanceQuadrature | None = None,
) -> tuple[float, Moments]:
    if params.temperature_mode == "optimal":
        return optimal_temperature(
            state,
            params,
            quadrature=quadrature,
            acceptance_quadrature=acceptance_quadrature,
        )
    return (
        params.beta,
        compute_moments(
            state, params, params.beta, K=params.K, quadrature=quadrature
        ),
    )


def best_scalar_rate(
    linear: float,
    rho: float,
    quadratic: float,
    T: int,
    cap: float,
) -> float:
    """Maximize a scalar linear-minus-quadratic drift on ``[0, cap]``."""

    def value(rate: float) -> float:
        return rate * linear - rho * quadratic * rate * rate / (2.0 * T)

    candidates = [0.0, float(cap)]
    denominator = rho * quadratic
    if abs(denominator) > 1e-14:
        stationary = T * linear / denominator
        if 0.0 <= stationary <= cap:
            candidates.append(float(stationary))
    return float(max(candidates, key=value))


def _best_scalar_rate_on_interval(
    linear: float,
    rho: float,
    quadratic: float,
    T: int,
    lower: float,
    upper: float,
) -> float:
    """Maximize a scalar linear-minus-quadratic objective on an interval."""

    lower = float(lower)
    upper = float(upper)
    if upper < lower:
        raise ValueError("rate interval has negative width")

    def value(rate: float) -> float:
        return rate * linear - rho * quadratic * rate * rate / (2.0 * T)

    candidates = [lower, upper]
    denominator = rho * quadratic
    if abs(denominator) > 1e-14:
        stationary = T * linear / denominator
        if lower <= stationary <= upper:
            candidates.append(float(stationary))
    return float(max(candidates, key=value))


def best_online_rates(
    state: State,
    params: Params,
    online: OnlineAggregates,
) -> tuple[float, float]:
    """Maximize online drift under a cap on the sum of student rates.

    In optimal-rate mode the feasible set is

    ``eta_2f >= 0, eta_3f >= 0, eta_2f + eta_3f <= eta_max``.

    Optional rate-specific caps further intersect this triangle.  The exact
    quadratic objective is evaluated at its feasible interior stationary
    point and at the optimum of every polygon edge.  This candidate
    construction also covers singular and non-concave cases without relying
    on divided closed forms.
    """

    if params.learning_rate_mode == "constant":
        return (
            0.0 if params.fix_eta_2f_zero else params.eta_2f,
            0.0 if params.fix_eta_3f_zero else params.eta_3f,
        )

    total_cap = float(params.eta_max)
    cap_2f = (
        0.0
        if params.fix_eta_2f_zero
        else min(total_cap, params.rate_cap("eta_2f"))
    )
    cap_3f = (
        0.0
        if params.fix_eta_3f_zero
        else min(total_cap, params.rate_cap("eta_3f"))
    )

    rho = state.rho_se
    L0 = online.base.B - rho * online.base.C
    LR = online.observed_reward.B - rho * online.observed_reward.C
    D0 = online.base.D
    DR = online.observed_reward.D

    def value(pair) -> float:
        eta_2f, eta_3f = pair
        quadratic = eta_2f**2 * D0 + (
            2.0 * eta_2f * eta_3f + eta_3f**2
        ) * DR
        return eta_2f * L0 + eta_3f * LR - rho * quadratic / (2.0 * params.T)

    candidates: list[tuple[float, float]] = [(0.0, 0.0)]

    def add(eta_2f: float, eta_3f: float) -> None:
        eta_2f = float(eta_2f)
        eta_3f = float(eta_3f)
        tolerance = 1e-12 * max(1.0, total_cap)
        if (
            eta_2f >= -tolerance
            and eta_3f >= -tolerance
            and eta_2f <= cap_2f + tolerance
            and eta_3f <= cap_3f + tolerance
            and eta_2f + eta_3f <= total_cap + tolerance
        ):
            candidates.append(
                (
                    float(np.clip(eta_2f, 0.0, cap_2f)),
                    float(np.clip(eta_3f, 0.0, cap_3f)),
                )
            )

    # The two single-rule edges.
    add(
        _best_scalar_rate_on_interval(
            L0,
            rho,
            D0,
            params.T,
            0.0,
            min(cap_2f, total_cap),
        ),
        0.0,
    )
    add(
        0.0,
        _best_scalar_rate_on_interval(
            LR,
            rho,
            DR,
            params.T,
            0.0,
            min(cap_3f, total_cap),
        ),
    )

    # Optional individual-cap edges.  These collapse harmlessly to vertices
    # when a rate is fixed to zero or its remaining interval is empty.
    if cap_2f <= total_cap:
        upper_3f = min(cap_3f, total_cap - cap_2f)
        if upper_3f >= 0.0:
            linear_3f = LR - rho * DR * cap_2f / params.T
            add(
                cap_2f,
                _best_scalar_rate_on_interval(
                    linear_3f,
                    rho,
                    DR,
                    params.T,
                    0.0,
                    upper_3f,
                ),
            )
    if cap_3f <= total_cap:
        upper_2f = min(cap_2f, total_cap - cap_3f)
        if upper_2f >= 0.0:
            linear_2f = L0 - rho * DR * cap_3f / params.T
            add(
                _best_scalar_rate_on_interval(
                    linear_2f,
                    rho,
                    D0,
                    params.T,
                    0.0,
                    upper_2f,
                ),
                cap_3f,
            )

    # Capped-sum edge: eta_3f = eta_max - eta_2f.
    sum_lower = max(0.0, total_cap - cap_3f)
    sum_upper = min(cap_2f, total_cap)
    if sum_lower <= sum_upper:
        eta_2f = _best_scalar_rate_on_interval(
            L0 - LR,
            rho,
            D0 - DR,
            params.T,
            sum_lower,
            sum_upper,
        )
        add(eta_2f, total_cap - eta_2f)

    matrix = np.array([[D0, DR], [DR, DR]], dtype=float)
    determinant = float(np.linalg.det(matrix))
    if abs(rho) > 1e-14 and abs(determinant) > 1e-14:
        stationary = np.linalg.solve(
            matrix,
            params.T / rho * np.array([L0, LR]),
        )
        add(float(stationary[0]), float(stationary[1]))

    return tuple(map(float, max(candidates, key=value)))


def online_drift_numerator(
    state: State,
    params: Params,
    online: OnlineAggregates,
    eta_2f: float,
    eta_3f: float,
) -> float:
    rho = state.rho_se
    quadratic = eta_2f**2 * online.base.D + (
        2.0 * eta_2f * eta_3f + eta_3f**2
    ) * online.observed_reward.D
    return float(
        eta_2f * (online.base.B - rho * online.base.C)
        + eta_3f
        * (online.observed_reward.B - rho * online.observed_reward.C)
        - rho * quadratic / (2.0 * params.T)
    )


def resolve_online_policy(state: State, params: Params) -> PolicyResolution:
    """Resolve adaptive ``K``, ``M``, temperature, and online student rates."""

    state.validate()
    params.validate()
    if params.greedy_training:
        active = greedy_policy_params(params)
        quadrature = prepare_split_quadrature(state, params.gh_order)
        beta, moments = resolve_temperature(state, active, quadrature)
        online = aggregate_online(moments, active)
        eta_2f, eta_3f = best_online_rates(state, active, online)
        return PolicyResolution(
            active,
            beta,
            moments,
            online,
            Rates(eta_2f=eta_2f, eta_3f=eta_3f),
            "constant",
            float("nan"),
        )

    objectives = {
        mode for mode in (params.K_mode, params.M_mode) if mode != "constant"
    }
    objective = next(iter(objectives), "constant")
    K_values = (
        range(1, params.K_max + 1)
        if params.K_mode != "constant"
        else (params.K,)
    )
    M_values = (
        range(1, params.T + 1) if params.M_mode != "constant" else (params.M,)
    )
    quadrature = prepare_split_quadrature(state, params.gh_order)
    acceptance_quadrature = (
        prepare_planner_acceptance_quadrature(
            state.rho_sm,
            max(params.gh_order, _MIN_PLANNER_GH_ORDER),
        )
        if params.temperature_mode == "optimal"
        else None
    )
    candidates: list[PolicyResolution] = []
    for K in K_values:
        for M in M_values:
            active = replace(
                params,
                K=int(K),
                M=int(M),
                K_mode="constant",
                M_mode="constant",
            )
            beta, moments = resolve_temperature(
                state,
                active,
                quadrature,
                acceptance_quadrature,
            )
            online = aggregate_online(moments, active)
            eta_2f, eta_3f = best_online_rates(state, active, online)
            if objective == "reward":
                value = online.gamma / active.c_on
            elif objective == "drift":
                value = online_drift_numerator(
                    state, active, online, eta_2f, eta_3f
                ) / active.c_on
            else:
                value = float("nan")
            candidates.append(
                PolicyResolution(
                    active,
                    beta,
                    moments,
                    online,
                    Rates(eta_2f=eta_2f, eta_3f=eta_3f),
                    objective,
                    float(value),
                )
            )
    if objective == "constant":
        return candidates[0]
    # Earlier iteration order provides deterministic smaller-K/smaller-M ties.
    return max(candidates, key=lambda candidate: candidate.objective_value)
