"""Agreement, planning, and sequence-reward observables."""

from __future__ import annotations

from dataclasses import replace
from math import asin, fsum, pi

import numpy as np

from .config import Params, State
from .modulation import cycle_value
from .moments import (
    Moments,
    aggregate_online,
    compute_moments,
    selected_zero_coefficient,
    success_coefficient,
)
from .optimization import (
    PolicyResolution,
    planner_acceptance,
    resolve_online_policy,
)


def agreement_probability(rho: float) -> float:
    """Probability that two correlated centered Gaussian fields share a sign."""

    return 0.5 + asin(float(np.clip(rho, -1.0, 1.0))) / pi


def action_correctness_probabilities(
    moments: Moments,
    T: int,
    M: int,
    K: int,
) -> tuple[float, float, float]:
    """Return prefix, suffix, and sequence-averaged action correctness.

    Prefix actions are conditioned on the planner's selected candidate,
    including its fallback when no candidate is accepted. Suffix actions are
    sampled directly from the student. The individual-action observable
    averages these probabilities over all ``T`` action positions.
    """

    prefix_terms = [
        success_coefficient(j, K)
        * moments.A[j] ** (M - 1)
        * moments.P_me[j]
        for j in range(1, K + 1)
    ]
    prefix_terms.extend(
        selected_zero_coefficient(j, K)
        * (
            moments.A[j] ** (M - 1) * moments.J[j]
            - moments.A[j + 1] ** (M - 1) * moments.P_me[j + 1]
        )
        for j in range(K)
    )
    prefix = float(np.clip(fsum(prefix_terms), 0.0, 1.0))
    suffix = float(np.clip(moments.J[0], 0.0, 1.0))
    individual = float(
        np.clip((M * prefix + (T - M) * suffix) / T, 0.0, 1.0)
    )
    return prefix, suffix, individual


def observables(
    state: State,
    params: Params,
    *,
    time: float = 0.0,
    resolved: PolicyResolution | None = None,
) -> dict[str, float | str]:
    """Evaluate planning and optional greedy evaluation without changing training.

    ``resolved`` describes the training policy. It can also be reused for
    planning evaluation unless training is greedy. In that case the configured
    planning policy is resolved independently, with ``greedy_training``
    disabled, so evaluation does not accidentally inherit ``K=1, beta=inf``.
    """

    if params.greedy_training:
        planning_resolution = resolve_online_policy(
            state,
            replace(
                params,
                greedy_training=False,
                greedy_evaluation=False,
            ),
        )
    else:
        planning_resolution = (
            resolve_online_policy(state, params) if resolved is None else resolved
        )

    planning_params = planning_resolution.params
    planning_online = planning_resolution.online
    planning_acceptance = planner_acceptance(
        planning_resolution.moments,
        planning_params.K,
        planning_params.M,
    )
    (
        prefix_action_probability,
        suffix_action_probability,
        individual_action_probability,
    ) = action_correctness_probabilities(
        planning_resolution.moments,
        planning_params.T,
        planning_params.M,
        planning_params.K,
    )
    planning_reward = planning_online.gamma

    if params.greedy_evaluation:
        evaluation_params = replace(
            params,
            K=1,
            K_mode="constant",
            M_mode="constant",
            beta=float("inf"),
            temperature_mode="constant",
            greedy_training=False,
        )
        greedy_moments = compute_moments(
            state, evaluation_params, float("inf"), K=1
        )
        evaluation_reward = aggregate_online(
            greedy_moments, evaluation_params, K=1
        ).gamma
        evaluation_policy = "greedy"
        evaluation_K = 1
        evaluation_M = evaluation_params.M
        evaluation_beta = float("inf")
        evaluation_cost = evaluation_params.c_on
    else:
        evaluation_reward = planning_reward
        evaluation_policy = "planning"
        evaluation_K = planning_params.K
        evaluation_M = planning_params.M
        evaluation_beta = planning_resolution.beta
        evaluation_cost = planning_params.c_on

    cycle = cycle_value(time, params)
    observed_probability = float(
        np.clip(
            params.epsilon + params.delta_epsilon * evaluation_reward, 0.0, 1.0
        )
    )
    p_se = agreement_probability(state.rho_se)
    p_me = agreement_probability(state.rho_me)
    p_sm = agreement_probability(state.rho_sm)
    return {
        "student_environment_agreement": p_se,
        "world_model_environment_agreement": p_me,
        "student_world_model_agreement": p_sm,
        "greedy_environment_reward_probability": p_se**params.T,
        "offline_world_model_reward_probability": p_sm**params.T,
        "planner_acceptance_probability": planning_acceptance,
        "prefix_action_correct_probability": prefix_action_probability,
        "suffix_action_correct_probability": suffix_action_probability,
        "individual_action_correct_probability": individual_action_probability,
        "planning_environment_reward_probability": planning_reward,
        "evaluation_policy": evaluation_policy,
        "evaluation_K": evaluation_K,
        "evaluation_M": evaluation_M,
        "evaluation_beta": evaluation_beta,
        "evaluation_cost": evaluation_cost,
        "true_reward_probability": evaluation_reward,
        "observed_reward_probability": observed_probability,
        "cycle_value": cycle,
        "true_reward_intensity": cycle * evaluation_reward,
        "observed_reward_intensity": cycle * observed_probability,
        "reward_rate": cycle * evaluation_reward / evaluation_cost,
    }
