"""Analytical spherical learning dynamics in rescaled elapsed time."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import pi, sqrt

import numpy as np

from .config import Params, State
from .modulation import cycle_value
from .moments import Aggregates, aggregate_offline, compute_moments
from .optimization import (
    PolicyResolution,
    Rates,
    best_scalar_rate,
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
    """Resolved controls, observables, and vector-field components."""

    mode: str
    cycle_value: float
    K: int
    M: int
    beta: float
    eta_2f: float
    eta_3f: float
    eta_off: float
    eta_wm: float
    update_cost: float
    true_reward_probability: float
    observed_reward_probability: float
    true_reward_intensity: float
    reward_rate: float
    dot_rho_se: float
    dot_rho_me: float
    dot_rho_sm: float
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

    def as_dict(self) -> dict:
        return asdict(self)


def world_model_moments(state: State, params: Params) -> WorldModelMoments:
    """Closed-form supervised WM moments from the report."""

    root = sqrt(2.0 / pi)
    mu = params.mu_flip
    b_e = mu * root
    b_s = mu * state.rho_se * root
    c_m = mu * state.rho_me * root
    return WorldModelMoments(
        b_e=b_e,
        b_s=b_s,
        c_m=c_m,
        d_m=1.0,
        L_m=b_e - state.rho_me * c_m,
        K_m=b_s - state.rho_sm * c_m,
    )


def _zero_aggregates() -> Aggregates:
    return Aggregates(0.0, 0.0, 0.0, 0.0, 0.0)


def components(
    state: State,
    params: Params,
    mode: str = "online",
    time: float = 0.0,
    *,
    resolved: PolicyResolution | None = None,
) -> Components:
    """Evaluate the analytical vector field and expose its resolved controls.

    A previously resolved online policy may be supplied when the caller also
    needs observables at the same state. This avoids repeating temperature and
    discrete-control optimization.
    """

    state.validate(tolerance=1e-8)
    params.validate()
    if mode not in {"online", "offline"}:
        raise ValueError("mode must be online or offline")
    cycle = cycle_value(time, params)
    wm = world_model_moments(state, params)

    if mode == "online":
        resolution = (
            resolve_online_policy(state, params) if resolved is None else resolved
        )
        active = resolution.params
        online = resolution.online
        offline = _zero_aggregates()
        eta_2f = resolution.rates.eta_2f
        eta_3f = resolution.rates.eta_3f
        if params.fix_eta_wm_zero:
            eta_wm = 0.0
        elif params.learning_rate_mode == "optimal":
            eta_wm = best_scalar_rate(
                wm.L_m,
                state.rho_me,
                wm.d_m,
                active.T,
                params.rate_cap("eta_wm"),
            )
        else:
            eta_wm = params.eta_wm
        q_student = eta_2f**2 * online.base.D + (
            2.0 * eta_2f * eta_3f + eta_3f**2
        ) * online.observed_reward.D
        L0 = online.base.B - state.rho_se * online.base.C
        LR = online.observed_reward.B - state.rho_se * online.observed_reward.C
        K0 = online.base.E - state.rho_sm * online.base.C
        KR = online.observed_reward.E - state.rho_sm * online.observed_reward.C
        dot_se_base = (
            eta_2f * L0
            + eta_3f * LR
            - state.rho_se * q_student / (2.0 * active.T)
        ) / active.c_on - active.eta_e * state.rho_se
        dot_me_base = (
            eta_wm * wm.L_m
            - eta_wm**2 * state.rho_me / (2.0 * active.T)
        ) / active.c_on - active.eta_e * state.rho_me
        dot_sm_base = (
            eta_2f * K0
            + eta_3f * KR
            - state.rho_sm * q_student / (2.0 * active.T)
            + eta_wm * wm.K_m
            - eta_wm**2 * state.rho_sm / (2.0 * active.T)
            + eta_wm
            * (eta_2f * online.base.X + eta_3f * online.observed_reward.X)
            / active.T
        ) / active.c_on
        dot_se, dot_me, dot_sm = cycle * np.array(
            [dot_se_base, dot_me_base, dot_sm_base]
        )
        gamma = online.gamma
        observed_probability = active.epsilon + active.delta_epsilon * gamma
        update_cost = active.c_on
        beta = resolution.beta
        K, M = active.K, active.M
        eta_off = 0.0
    else:
        active = params
        # Offline aggregates do not depend on beta or planner powers.
        moments = compute_moments(state, active, 0.0, K=1)
        offline = aggregate_offline(moments, active.T)
        online = None
        if active.fix_eta_off_zero:
            eta_off = 0.0
        elif active.learning_rate_mode == "optimal":
            eta_off = best_scalar_rate(
                offline.E - state.rho_sm * offline.C,
                state.rho_sm,
                offline.D,
                active.T,
                active.rate_cap("eta_off"),
            )
        else:
            eta_off = active.eta_off
        dot_se_environment = (
            eta_off * (offline.B - state.rho_se * offline.C)
            - eta_off**2 * state.rho_se * offline.D / (2.0 * active.T)
        ) / active.delta_off - active.eta_e * state.rho_se
        dot_me_environment = -active.eta_e * state.rho_me
        dot_sm_mental = (
            eta_off * (offline.E - state.rho_sm * offline.C)
            - eta_off**2 * state.rho_sm * offline.D / (2.0 * active.T)
        ) / active.delta_off
        dot_se = cycle * dot_se_environment
        dot_me = cycle * dot_me_environment
        dot_sm = dot_sm_mental
        gamma = 0.0
        observed_probability = 0.0
        update_cost = active.delta_off
        beta = active.beta
        K, M = active.K, active.M
        eta_2f = eta_3f = eta_wm = 0.0

    base = online.base if online is not None else _zero_aggregates()
    reward = online.observed_reward if online is not None else _zero_aggregates()
    return Components(
        mode=mode,
        cycle_value=cycle,
        K=K,
        M=M,
        beta=beta,
        eta_2f=eta_2f,
        eta_3f=eta_3f,
        eta_off=eta_off,
        eta_wm=eta_wm,
        update_cost=update_cost,
        true_reward_probability=float(gamma),
        observed_reward_probability=float(observed_probability),
        true_reward_intensity=float(cycle * gamma),
        reward_rate=float(cycle * gamma / update_cost) if mode == "online" else 0.0,
        dot_rho_se=float(dot_se),
        dot_rho_me=float(dot_me),
        dot_rho_sm=float(dot_sm),
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
    )


def rhs(time: float, values, params: Params, mode: str = "online") -> np.ndarray:
    """ODE right-hand side in correlation coordinates."""

    state = State.from_array(values, project=True)
    result = components(state, params, mode, time)
    return np.array([result.dot_rho_se, result.dot_rho_me, result.dot_rho_sm])
