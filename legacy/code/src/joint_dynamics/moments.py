"""Split two-dimensional Gaussian moments from the joint-dynamics report."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, fsum

import numpy as np
from scipy.special import ndtr, ndtri

from .config import Params, State
from .fields import thermal_mean_action, thermal_probability
from .quadrature import conditional_u_parameters, signed_normal_statistics, uniform_from_normal


_NAMES = ("B", "C", "E", "D", "X")


@dataclass(frozen=True)
class Tagged:
    B: np.ndarray
    C: np.ndarray
    E: np.ndarray
    D: np.ndarray
    X: np.ndarray

    def get(self, name: str) -> np.ndarray:
        if name not in _NAMES:
            raise KeyError(name)
        return getattr(self, name)


@dataclass(frozen=True)
class Moments:
    beta: float
    A: np.ndarray
    P_me: np.ndarray
    P_sm: np.ndarray
    J: np.ndarray
    wm: Tagged
    sampled: Tagged
    wm_me: Tagged
    env_reward: Tagged
    greedy_sm: Tagged


@dataclass(frozen=True)
class Aggregates:
    B: float
    C: float
    E: float
    D: float
    X: float


@dataclass(frozen=True)
class OnlineAggregates:
    base: Aggregates
    true_reward: Aggregates
    observed_reward: Aggregates


@dataclass(frozen=True)
class QuadratureBlock:
    sign_v: float
    sign_h: float
    sign_u: float
    h: np.ndarray
    v: np.ndarray
    weights: np.ndarray
    u_mass: np.ndarray
    u_first: np.ndarray


@dataclass(frozen=True)
class MomentQuadrature:
    """State-dependent split-quadrature geometry reusable across controls."""

    blocks: tuple[QuadratureBlock, ...]


def _zeros(params: Params) -> dict[str, np.ndarray]:
    return {name: np.zeros(params.K + 1, dtype=float) for name in _NAMES}


def _tagged(values: dict[str, np.ndarray]) -> Tagged:
    return Tagged(**values)


def _bases(
    action, h, v, u_mass, u_first, sign_u: float, params: Params, *, stochastic=False
):
    score = 0.5 * action - params.lambda_s * h
    wm_score = 0.5 * params.mu_flip * sign_u - params.lambda_m * v
    squared_score = (
        0.25 - params.lambda_s * action * h + params.lambda_s**2 * h * h
        if stochastic
        else score * score
    )
    return {
        "B": u_first * score,
        "C": u_mass * h * score,
        "E": u_mass * v * score,
        "D": u_mass * squared_score,
        "X": u_mass * score * wm_score,
    }


def _add(accumulator, bases, powers, weights, multiplier=1.0):
    weighted = weights * multiplier
    power_matrix = powers.reshape((powers.shape[0], -1))
    for name in _NAMES:
        accumulator[name] += power_matrix @ (weighted * bases[name]).ravel()


def prepare_moment_quadrature(state: State, params: Params) -> MomentQuadrature:
    """Build the state-dependent geometry shared by all K and beta values."""

    state.validate()
    params.validate()
    return _prepare_moment_quadrature(state, params.gh_order)


@lru_cache(maxsize=64)
def _prepare_moment_quadrature(state: State, gh_order: int) -> MomentQuadrature:
    uniform, gh_weights = uniform_from_normal(gh_order)
    blocks = []
    std_h = max(1.0 - state.rho_sm**2, 0.0) ** 0.5
    for sign_v in (-1.0, 1.0):
        v_low, v_high = ((0.0, 0.5) if sign_v < 0.0 else (0.5, 1.0))
        v_prob = v_low + (v_high - v_low) * uniform
        v = ndtri(np.clip(v_prob, 1e-15, 1.0 - 1e-15))
        v_weights = (v_high - v_low) * gh_weights
        mean_h = state.rho_sm * v

        if std_h <= 1e-12:
            h_signs = [
                (
                    1.0 if np.all(mean_h >= 0.0) else -1.0,
                    mean_h[:, None],
                    np.ones((len(v), 1)),
                    np.ones(1),
                )
            ]
        else:
            boundary = np.clip(ndtr(-mean_h / std_h), 0.0, 1.0)
            h_signs = []
            for sign_h in (-1.0, 1.0):
                low = np.zeros_like(boundary) if sign_h < 0.0 else boundary
                high = boundary if sign_h < 0.0 else np.ones_like(boundary)
                interval = np.maximum(high - low, 0.0)
                probability = low[:, None] + interval[:, None] * uniform[None, :]
                h = mean_h[:, None] + std_h * ndtri(
                    np.clip(probability, 1e-15, 1.0 - 1e-15)
                )
                h_signs.append((sign_h, h, interval[:, None], gh_weights))

        for sign_h, h, h_interval, h_weights in h_signs:
            v_grid = np.broadcast_to(v[:, None], h.shape)
            weights = v_weights[:, None] * h_interval * np.asarray(h_weights)[None, :]
            if not np.any(weights):
                continue
            mean_u, std_u = conditional_u_parameters(state.matrix, h, v_grid)
            for sign_u in (-1.0, 1.0):
                u_mass, u_first = signed_normal_statistics(mean_u, std_u, sign_u)
                blocks.append(
                    QuadratureBlock(
                        sign_v=sign_v,
                        sign_h=sign_h,
                        sign_u=sign_u,
                        h=h,
                        v=v_grid,
                        weights=weights,
                        u_mass=u_mass,
                        u_first=u_first,
                    )
                )
    return MomentQuadrature(tuple(blocks))


def compute_moments(
    state: State,
    params: Params,
    beta: float,
    quadrature: MomentQuadrature | None = None,
) -> Moments:
    """Compute all one-step moments with split 2-D Gauss-Hermite quadrature.

    H and V are integrated explicitly on their four sign regions. U is
    conditioned on H,V and integrated analytically on its two sign regions.
    Consequently no quadrature cell crosses a sign discontinuity.
    """

    state.validate()
    params.validate()
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")

    quadrature = quadrature or prepare_moment_quadrature(state, params)
    size = params.K + 1
    A = np.zeros(size)
    P_me = np.zeros(size)
    P_sm = np.zeros(size)
    J = np.zeros(size)
    wm = _zeros(params)
    sampled = _zeros(params)
    wm_me = _zeros(params)
    env_reward = _zeros(params)
    greedy_sm = _zeros(params)

    for block in quadrature.blocks:
        sign_v = block.sign_v
        sign_h = block.sign_h
        sign_u = block.sign_u
        h = block.h
        v_grid = block.v
        weights = block.weights
        u_mass = block.u_mass
        u_first = block.u_first
        planner_probability = thermal_probability(beta, sign_v * h)
        powers = np.empty((params.K + 1,) + h.shape, dtype=float)
        powers[0] = 1.0
        powers[1:] = planner_probability
        np.cumprod(powers, axis=0, out=powers)
        sampled_action = thermal_mean_action(beta, h)
        is_sm = float(sign_h * sign_v > 0.0)
        is_me = float(sign_u * sign_v > 0.0)
        reward_probability = thermal_probability(beta, sign_u * h)

        power_matrix = powers.reshape((params.K + 1, -1))
        weighted_mass = weights * u_mass
        mass_by_power = power_matrix @ weighted_mass.ravel()
        A += mass_by_power
        P_me += is_me * mass_by_power
        P_sm += is_sm * mass_by_power
        J += power_matrix @ (weighted_mass * reward_probability).ravel()

        wm_bases = _bases(sign_v, h, v_grid, u_mass, u_first, sign_u, params)
        sampled_bases = _bases(
            sampled_action,
            h,
            v_grid,
            u_mass,
            u_first,
            sign_u,
            params,
            stochastic=True,
        )
        env_bases = _bases(sign_u, h, v_grid, u_mass, u_first, sign_u, params)
        greedy_bases = _bases(sign_h, h, v_grid, u_mass, u_first, sign_u, params)
        _add(wm, wm_bases, powers, weights)
        _add(sampled, sampled_bases, powers, weights)
        if is_me:
            _add(wm_me, wm_bases, powers, weights)
        _add(env_reward, env_bases, powers, weights, reward_probability)
        if is_sm:
            _add(greedy_sm, greedy_bases, powers, weights)

    return Moments(
        beta=float(beta),
        A=A,
        P_me=P_me,
        P_sm=P_sm,
        J=J,
        wm=_tagged(wm),
        sampled=_tagged(sampled),
        wm_me=_tagged(wm_me),
        env_reward=_tagged(env_reward),
        greedy_sm=_tagged(greedy_sm),
    )


def success_coefficient(j: int, K: int) -> int:
    return (-1) ** (j + 1) * comb(K, j)


def selected_zero_coefficient(j: int, K: int) -> int:
    return (-1) ** j * comb(K - 1, j)


def environment_reward_probability(moments: Moments, params: Params) -> float:
    """Expected true environment reward of one online execution."""

    prefix = params.effective_prefix_length
    suffix = params.T - prefix
    terms = [
        success_coefficient(j, params.K) * moments.P_me[j] ** prefix
        for j in range(1, params.K + 1)
    ]
    terms.extend(
        selected_zero_coefficient(j, params.K)
        * (moments.J[j] ** prefix - moments.P_me[j + 1] ** prefix)
        for j in range(params.K)
    )
    return float(np.clip(moments.J[0] ** suffix * fsum(terms), 0.0, 1.0))


def _aggregate_values(values: dict[str, float]) -> Aggregates:
    return Aggregates(**{name: float(values[name]) for name in _NAMES})


def aggregate_online(moments: Moments, params: Params) -> OnlineAggregates:
    """Build reward-independent, true-reward, and observed-reward moments."""

    prefix = params.effective_prefix_length
    suffix = params.T - prefix
    prefix_weight = prefix / params.T
    suffix_weight = suffix / params.T
    gamma_pre_terms = [
        success_coefficient(j, params.K) * moments.P_me[j] ** prefix
        for j in range(1, params.K + 1)
    ]
    gamma_pre_terms.extend(
        selected_zero_coefficient(j, params.K)
        * (moments.J[j] ** prefix - moments.P_me[j + 1] ** prefix)
        for j in range(params.K)
    )
    gamma_pre = fsum(gamma_pre_terms)
    base: dict[str, float] = {}
    true: dict[str, float] = {}
    for name in _NAMES:
        prefix_unweighted_terms = []
        prefix_reward_terms = []
        for j in range(1, params.K + 1):
            coefficient = success_coefficient(j, params.K)
            prefix_unweighted_terms.append(
                coefficient * moments.A[j] ** (prefix - 1) * moments.wm.get(name)[j]
            )
            prefix_reward_terms.append(
                coefficient
                * moments.P_me[j] ** (prefix - 1)
                * moments.wm_me.get(name)[j]
            )
        for j in range(params.K):
            coefficient = selected_zero_coefficient(j, params.K)
            prefix_unweighted_terms.append(
                coefficient
                * (
                    moments.A[j] ** (prefix - 1) * moments.sampled.get(name)[j]
                    - moments.A[j + 1] ** (prefix - 1) * moments.wm.get(name)[j + 1]
                )
            )
            prefix_reward_terms.append(
                coefficient
                * (
                    moments.J[j] ** (prefix - 1) * moments.env_reward.get(name)[j]
                    - moments.P_me[j + 1] ** (prefix - 1)
                    * moments.wm_me.get(name)[j + 1]
                )
            )
        prefix_base = fsum(prefix_unweighted_terms)
        prefix_true = moments.J[0] ** suffix * fsum(prefix_reward_terms)
        suffix_true = (
            gamma_pre
            * moments.J[0] ** (suffix - 1)
            * moments.env_reward.get(name)[0]
            if suffix > 0
            else 0.0
        )
        base[name] = prefix_weight * prefix_base + suffix_weight * moments.sampled.get(name)[0]
        true[name] = prefix_weight * prefix_true + suffix_weight * suffix_true
    observed = {
        name: params.epsilon * base[name] + params.delta_epsilon * true[name]
        for name in _NAMES
    }
    return OnlineAggregates(
        base=_aggregate_values(base),
        true_reward=_aggregate_values(true),
        observed_reward=_aggregate_values(observed),
    )


def aggregate_offline(moments: Moments, params: Params) -> Aggregates:
    factor = moments.P_sm[0] ** (params.T - 1)
    return Aggregates(
        **{name: float(factor * moments.greedy_sm.get(name)[0]) for name in _NAMES}
    )
