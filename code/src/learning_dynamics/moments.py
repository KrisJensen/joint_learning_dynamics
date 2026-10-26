"""One-step Gaussian moments and sequence-level aggregate operators."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, fsum

import numpy as np

from .config import Params, State
from .fields import thermal_mean_action, thermal_probability
from .quadrature import SplitQuadrature, prepare_split_quadrature


_NAMES = ("B", "C", "E", "D", "X")


@dataclass(frozen=True)
class TaggedMoments:
    """Moments of the five tagged functions in the report."""

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
    """Basic planner and tagged moments for powers ``j=0,...,K``."""

    beta: float
    A: np.ndarray
    P_me: np.ndarray
    P_sm: np.ndarray
    J: np.ndarray
    wm: TaggedMoments
    sampled: TaggedMoments
    wm_me: TaggedMoments
    environment_correct: TaggedMoments
    greedy_sm: TaggedMoments


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
    gamma_prefix: float
    gamma: float


def success_coefficient(j: int, K: int) -> int:
    return (-1) ** (j + 1) * comb(K, j)


def selected_zero_coefficient(j: int, K: int) -> int:
    return (-1) ** j * comb(K - 1, j)


def _empty(size: int) -> dict[str, np.ndarray]:
    return {name: np.zeros(size, dtype=float) for name in _NAMES}


def _as_tagged(values: dict[str, np.ndarray]) -> TaggedMoments:
    return TaggedMoments(**{key: np.asarray(value, dtype=float) for key, value in values.items()})


def _bases(action, block, mu_flip: float) -> dict[str, np.ndarray]:
    """Evaluate report fields after analytically integrating over signed U."""

    action = np.asarray(action)
    return {
        # |U| a sign(U) = U a.
        "B": block.u_first * action,
        # S Z = a H and S W = a V.
        "C": block.u_mass * block.h * action,
        "E": block.u_mass * block.v * action,
        "D": block.u_mass * np.ones_like(block.h),
        "X": block.u_mass * mu_flip * block.sign_u * action,
    }


def _accumulate(target, bases, powers, weights, multiplier=1.0) -> None:
    flattened_powers = powers.reshape((powers.shape[0], -1))
    weighted = weights * multiplier
    for name in _NAMES:
        values = (weighted * bases[name]).ravel()
        target[name] += flattened_powers @ values


def compute_moments(
    state: State,
    params: Params,
    beta: float,
    *,
    K: int | None = None,
    quadrature: SplitQuadrature | None = None,
) -> Moments:
    """Compute all report moments with split Gauss-Hermite quadrature."""

    state.validate()
    params.validate()
    if beta < 0.0:
        raise ValueError("beta must be nonnegative")
    active_K = params.K if K is None else int(K)
    if active_K < 1:
        raise ValueError("K must be positive")
    quadrature = quadrature or prepare_split_quadrature(state, params.gh_order)
    size = active_K + 1
    A = np.zeros(size, dtype=float)
    P_me = np.zeros(size, dtype=float)
    P_sm = np.zeros(size, dtype=float)
    J = np.zeros(size, dtype=float)
    wm = _empty(size)
    sampled = _empty(size)
    wm_me = _empty(size)
    environment_correct = _empty(size)
    greedy_sm = _empty(size)

    for block in quadrature.blocks:
        probability = thermal_probability(beta, block.sign_v * block.h)
        powers = np.empty((size,) + block.h.shape, dtype=float)
        powers[0] = 1.0
        powers[1:] = probability
        np.cumprod(powers, axis=0, out=powers)
        sampled_action = thermal_mean_action(beta, block.h)
        environment_probability = thermal_probability(beta, block.sign_u * block.h)
        is_me = float(block.sign_u * block.sign_v > 0.0)
        is_sm = float(block.sign_h * block.sign_v > 0.0)

        matrix = powers.reshape((size, -1))
        weighted_mass = block.weights * block.u_mass
        by_power = matrix @ weighted_mass.ravel()
        A += by_power
        P_me += is_me * by_power
        P_sm += is_sm * by_power
        J += matrix @ (weighted_mass * environment_probability).ravel()

        wm_bases = _bases(block.sign_v, block, params.mu_flip)
        sampled_bases = _bases(sampled_action, block, params.mu_flip)
        environment_bases = _bases(block.sign_u, block, params.mu_flip)
        greedy_bases = _bases(block.sign_h, block, params.mu_flip)
        _accumulate(wm, wm_bases, powers, block.weights)
        _accumulate(sampled, sampled_bases, powers, block.weights)
        if is_me:
            _accumulate(wm_me, wm_bases, powers, block.weights)
        _accumulate(
            environment_correct,
            environment_bases,
            powers,
            block.weights,
            environment_probability,
        )
        if is_sm:
            _accumulate(greedy_sm, greedy_bases, powers, block.weights)

    return Moments(
        beta=float(beta),
        A=A,
        P_me=P_me,
        P_sm=P_sm,
        J=J,
        wm=_as_tagged(wm),
        sampled=_as_tagged(sampled),
        wm_me=_as_tagged(wm_me),
        environment_correct=_as_tagged(environment_correct),
        greedy_sm=_as_tagged(greedy_sm),
    )


def _stable_sum(terms) -> float:
    return float(fsum(float(term) for term in terms))


def environment_reward_probability(moments: Moments, T: int, M: int, K: int) -> float:
    """Return the true full-sequence success probability ``Gamma_{K,M}``."""

    prefix_terms = [
        success_coefficient(j, K) * moments.P_me[j] ** M
        for j in range(1, K + 1)
    ]
    prefix_terms.extend(
        selected_zero_coefficient(j, K)
        * (moments.J[j] ** M - moments.P_me[j + 1] ** M)
        for j in range(K)
    )
    value = moments.J[0] ** (T - M) * _stable_sum(prefix_terms)
    if value < -1e-9 or value > 1.0 + 1e-9:
        raise FloatingPointError(f"computed environment reward probability {value}")
    return float(np.clip(value, 0.0, 1.0))


def aggregate_online(
    moments: Moments,
    params: Params,
    *,
    K: int | None = None,
    M: int | None = None,
) -> OnlineAggregates:
    """Build ``U_on``, ``H_on``, and noisy reward aggregate moments."""

    K = params.K if K is None else int(K)
    M = params.M if M is None else int(M)
    T = params.T
    suffix = T - M
    prefix_weight = M / T
    suffix_weight = suffix / T

    gamma_terms = [
        success_coefficient(j, K) * moments.P_me[j] ** M
        for j in range(1, K + 1)
    ]
    gamma_terms.extend(
        selected_zero_coefficient(j, K)
        * (moments.J[j] ** M - moments.P_me[j + 1] ** M)
        for j in range(K)
    )
    gamma_prefix = _stable_sum(gamma_terms)
    gamma = float(moments.J[0] ** suffix * gamma_prefix)

    base: dict[str, float] = {}
    true: dict[str, float] = {}
    for name in _NAMES:
        prefix_base_terms = []
        prefix_reward_terms = []
        for j in range(1, K + 1):
            coefficient = success_coefficient(j, K)
            prefix_base_terms.append(
                coefficient
                * moments.A[j] ** (M - 1)
                * moments.wm.get(name)[j]
            )
            prefix_reward_terms.append(
                coefficient
                * moments.P_me[j] ** (M - 1)
                * moments.wm_me.get(name)[j]
            )
        for j in range(K):
            coefficient = selected_zero_coefficient(j, K)
            prefix_base_terms.append(
                coefficient
                * (
                    moments.A[j] ** (M - 1) * moments.sampled.get(name)[j]
                    - moments.A[j + 1] ** (M - 1) * moments.wm.get(name)[j + 1]
                )
            )
            prefix_reward_terms.append(
                coefficient
                * (
                    moments.J[j] ** (M - 1)
                    * moments.environment_correct.get(name)[j]
                    - moments.P_me[j + 1] ** (M - 1)
                    * moments.wm_me.get(name)[j + 1]
                )
            )

        prefix_base = _stable_sum(prefix_base_terms)
        prefix_true = moments.J[0] ** suffix * _stable_sum(prefix_reward_terms)
        suffix_true = (
            gamma_prefix
            * moments.J[0] ** (suffix - 1)
            * moments.environment_correct.get(name)[0]
            if suffix > 0
            else 0.0
        )
        base[name] = (
            prefix_weight * prefix_base
            + suffix_weight * moments.sampled.get(name)[0]
        )
        true[name] = prefix_weight * prefix_true + suffix_weight * suffix_true

    # These report identities are exact and avoid propagating quadrature noise.
    base["D"] = 1.0
    true["D"] = gamma
    observed = {
        name: params.epsilon * base[name] + params.delta_epsilon * true[name]
        for name in _NAMES
    }
    return OnlineAggregates(
        base=Aggregates(**{name: float(base[name]) for name in _NAMES}),
        true_reward=Aggregates(**{name: float(true[name]) for name in _NAMES}),
        observed_reward=Aggregates(
            **{name: float(observed[name]) for name in _NAMES}
        ),
        gamma_prefix=float(np.clip(gamma_prefix, 0.0, 1.0)),
        gamma=float(np.clip(gamma, 0.0, 1.0)),
    )


def aggregate_offline(moments: Moments, T: int) -> Aggregates:
    """Build full-sequence greedy offline moments."""

    factor = moments.P_sm[0] ** (T - 1)
    return Aggregates(
        **{
            name: float(factor * moments.greedy_sm.get(name)[0])
            for name in _NAMES
        }
    )
