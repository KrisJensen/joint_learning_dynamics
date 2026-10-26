"""Observable agreement and sequence-reward probabilities."""

from __future__ import annotations

from dataclasses import replace
from math import asin, fsum, pi

import numpy as np

from .config import Params, State
from .moments import (
    Moments,
    compute_moments,
    environment_reward_probability,
    success_coefficient,
)
from .modulation import environment_modulation
from .optimization import PolicyResolution, resolve_online_policy


def agreement_probability(rho: float) -> float:
    return 0.5 + asin(float(np.clip(rho, -1.0, 1.0))) / pi


def observable_probabilities(
    state: State,
    params: Params,
    beta: float,
    moments: Moments | None = None,
) -> dict[str, float]:
    if moments is None:
        moments = compute_moments(state, params, beta)
    acceptance = fsum(
        [
            success_coefficient(j, params.K)
            * moments.A[j] ** params.effective_prefix_length
            for j in range(1, params.K + 1)
        ]
    )
    reward = environment_reward_probability(moments, params)
    p_se = agreement_probability(state.rho_se)
    p_me = agreement_probability(state.rho_me)
    p_sm = agreement_probability(state.rho_sm)
    return {
        "student_environment_agreement": p_se,
        "world_model_environment_agreement": p_me,
        "student_world_model_agreement": p_sm,
        "greedy_environment_reward_probability": p_se**params.T,
        "offline_world_model_reward_probability": p_sm**params.T,
        "planner_acceptance_probability": float(np.clip(acceptance, 0.0, 1.0)),
        "environment_reward_probability": float(np.clip(reward, 0.0, 1.0)),
        "observed_reward_probability": float(
            np.clip(params.epsilon + params.delta_epsilon * reward, 0.0, 1.0)
        ),
    }


def evaluation_probabilities(
    state: State,
    params: Params,
    *,
    planning_K: int | None = None,
    planning_beta: float | None = None,
    time: float = 0.0,
) -> dict[str, float | str]:
    """Evaluate planning and greedy policies without changing training dynamics."""

    if (planning_K is None) != (planning_beta is None):
        raise ValueError("planning_K and planning_beta must be supplied together")
    if planning_K is None:
        planning = resolve_online_policy(state, params, greedy=False)
    else:
        active = replace(params, K=planning_K, K_mode="constant")
        planning = PolicyResolution(
            active,
            planning_beta,
            compute_moments(state, active, planning_beta),
        )
    values = observable_probabilities(
        state, planning.params, planning.beta, planning.moments
    )
    planning_reward = values["environment_reward_probability"]
    greedy_reward = values["greedy_environment_reward_probability"]
    if params.greedy_evaluation:
        evaluation_policy = "greedy"
        evaluation_K = 1
        evaluation_beta = float("inf")
        evaluation_reward = greedy_reward
    else:
        evaluation_policy = "planning"
        evaluation_K = planning.params.K
        evaluation_beta = planning.beta
        evaluation_reward = planning_reward
    evaluation_cost = 1.0 + evaluation_K * params.tau_on
    modulation = environment_modulation(time, params)
    instantaneous_reward = modulation * evaluation_reward
    base_observed_reward = float(
        np.clip(
            params.epsilon + params.delta_epsilon * evaluation_reward,
            0.0,
            1.0,
        )
    )
    values.update(
        {
            "planning_K": planning.params.K,
            "planning_beta": planning.beta,
            "planning_environment_reward_probability": planning_reward,
            "evaluation_policy": evaluation_policy,
            "evaluation_K": evaluation_K,
            "evaluation_beta": evaluation_beta,
            "evaluation_cost": evaluation_cost,
            "evaluation_environment_reward_probability": evaluation_reward,
            "base_environment_reward_probability": evaluation_reward,
            "environment_modulation": modulation,
            "planning_environment_reward": modulation * planning_reward,
            "greedy_environment_reward": modulation * greedy_reward,
            "evaluation_environment_reward": instantaneous_reward,
            # This legacy field is consumed by existing result readers and plots.
            # With modulation enabled it is an instantaneous reward intensity,
            # rather than a conditional probability, and may therefore exceed 1.
            "environment_reward_probability": instantaneous_reward,
            "base_observed_reward_probability": base_observed_reward,
            "observed_reward_probability": modulation * base_observed_reward,
            "reward_rate": instantaneous_reward / evaluation_cost,
        }
    )
    return values
