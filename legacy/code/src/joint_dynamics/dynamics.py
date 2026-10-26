"""Analytical joint spherical learning dynamics in elapsed time."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from math import pi, sqrt

import numpy as np

from .config import Params, State, validate_mode
from .moments import (
    Aggregates,
    OnlineAggregates,
    aggregate_offline,
    aggregate_online,
    compute_moments,
    environment_reward_probability,
)
from .modulation import environment_modulation
from .optimization import (
    Rates,
    effective_rates,
    effective_temperature,
    greedy_policy_params,
    resolve_online_policy,
)


@dataclass(frozen=True)
class WorldModelMoments:
    b_e: float
    b_s: float
    c_m: float
    d_m: float
    L_m: float
    K_m: float


@dataclass(frozen=True)
class Components:
    mode: str
    environment_modulation: float
    K: int
    prefix_length: int
    beta: float
    eta_on1: float
    eta_on2: float
    eta_off: float
    eta_wm: float
    B0: float
    C0: float
    E0: float
    D0: float
    X0: float
    BR: float
    CR: float
    ER: float
    DR: float
    XR: float
    B_off: float
    C_off: float
    E_off: float
    D_off: float
    d_m: float
    training_environment_reward_probability: float
    training_environment_reward: float
    dot_rho_se: float
    dot_rho_me: float
    dot_rho_sm: float

    def as_dict(self) -> dict:
        return asdict(self)


def world_model_moments(state: State, params: Params) -> WorldModelMoments:
    root = sqrt(2.0 / pi)
    mu = params.mu_flip
    b_e = 0.5 * mu * root - params.lambda_m * state.rho_me
    b_s = 0.5 * mu * state.rho_se * root - params.lambda_m * state.rho_sm
    c_m = 0.5 * mu * state.rho_me * root - params.lambda_m
    d_m = 0.25 + params.lambda_m**2 - mu * params.lambda_m * state.rho_me * root
    return WorldModelMoments(
        b_e=b_e,
        b_s=b_s,
        c_m=c_m,
        d_m=d_m,
        L_m=b_e - state.rho_me * c_m,
        K_m=b_s - state.rho_sm * c_m,
    )


def components(
    state: State,
    params: Params,
    mode: str = "online",
    time: float = 0.0,
) -> Components:
    state.validate()
    params.validate()
    validate_mode(mode)
    if mode == "online":
        policy = resolve_online_policy(state, params, greedy=params.greedy_training)
        active_params = policy.params
        K = active_params.K
        beta = policy.beta
        moment_values = policy.moments
    else:
        active_params = greedy_policy_params(params) if params.greedy_training else params
        active_params = replace(active_params, K_mode="constant")
        K = active_params.K
        beta = effective_temperature(state, active_params)
        moment_values = compute_moments(state, active_params, beta)
    online = aggregate_online(moment_values, active_params)
    offline = aggregate_offline(moment_values, active_params)
    wm = world_model_moments(state, active_params)
    rates = effective_rates(
        state, active_params, mode, online, offline, L_m=wm.L_m, d_m=wm.d_m
    )
    modulation = environment_modulation(time, active_params)

    if mode == "online":
        q_on = (
            rates.eta_on1**2 * online.base.D
            + (2.0 * rates.eta_on1 * rates.eta_on2 + rates.eta_on2**2)
            * online.observed_reward.D
        )
        dot_se = modulation * (
            rates.eta_on1 * (online.base.B - state.rho_se * online.base.C)
            + rates.eta_on2
            * (online.observed_reward.B - state.rho_se * online.observed_reward.C)
            - state.rho_se * q_on / (2.0 * active_params.T)
        ) / active_params.c_on - modulation * active_params.eta_e * state.rho_se
        dot_me = modulation * (
            rates.eta_wm * wm.L_m
            - rates.eta_wm**2 * state.rho_me * wm.d_m / (2.0 * active_params.T)
        ) / active_params.c_on - modulation * active_params.eta_e * state.rho_me
        dot_sm = modulation * (
            rates.eta_on1 * (online.base.E - state.rho_sm * online.base.C)
            + rates.eta_on2
            * (online.observed_reward.E - state.rho_sm * online.observed_reward.C)
            - state.rho_sm * q_on / (2.0 * active_params.T)
            + rates.eta_wm * wm.K_m
            - rates.eta_wm**2 * state.rho_sm * wm.d_m / (2.0 * active_params.T)
            + rates.eta_wm
            * (rates.eta_on1 * online.base.X + rates.eta_on2 * online.observed_reward.X)
            / active_params.T
        ) / active_params.c_on
    else:
        dot_se = modulation * (
            rates.eta_off * (offline.B - state.rho_se * offline.C)
            - rates.eta_off**2 * state.rho_se * offline.D / (2.0 * active_params.T)
        ) / active_params.c_off - modulation * active_params.eta_e * state.rho_se
        dot_me = -modulation * active_params.eta_e * state.rho_me
        dot_sm = (
            rates.eta_off * (offline.E - state.rho_sm * offline.C)
            - rates.eta_off**2 * state.rho_sm * offline.D / (2.0 * active_params.T)
        ) / active_params.c_off

    base, reward = online.base, online.observed_reward
    training_reward_probability = environment_reward_probability(
        moment_values, active_params
    )
    return Components(
        mode=mode,
        environment_modulation=modulation,
        K=K,
        prefix_length=active_params.effective_prefix_length,
        beta=beta,
        **asdict(rates),
        B0=base.B,
        C0=base.C,
        E0=base.E,
        D0=base.D,
        X0=base.X,
        BR=reward.B,
        CR=reward.C,
        ER=reward.E,
        DR=reward.D,
        XR=reward.X,
        B_off=offline.B,
        C_off=offline.C,
        E_off=offline.E,
        D_off=offline.D,
        d_m=wm.d_m,
        training_environment_reward_probability=training_reward_probability,
        training_environment_reward=modulation * training_reward_probability,
        dot_rho_se=float(dot_se),
        dot_rho_me=float(dot_me),
        dot_rho_sm=float(dot_sm),
    )


def rhs(time: float, values, params: Params, mode: str = "online") -> np.ndarray:
    state = State.from_array(values, project=True)
    result = components(state, params, mode, time)
    return np.array([result.dot_rho_se, result.dot_rho_me, result.dot_rho_sm])
