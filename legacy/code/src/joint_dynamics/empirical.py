"""Finite-dimensional empirical verification of the joint dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Params, Schedule, State
from .dynamics import components
from .fields import correlation_root, sign_nonzero, thermal_probability
from .modulation import (
    average_environment_modulation,
    integrated_environment_modulation,
)
from .observables import evaluation_probabilities
from .optimization import effective_temperature, greedy_policy_params


@dataclass(frozen=True)
class EpisodeControls:
    beta: float
    K: int
    eta_on1: float = 0.0
    eta_on2: float = 0.0
    eta_off: float = 0.0
    eta_wm: float = 0.0


def state_from_weights(environment, student, world_model) -> State:
    d = student.size
    return State(
        float(student @ environment / d),
        float(world_model @ environment / d),
        float(student @ world_model / d),
    ).projected()


def initial_weights(state: State, d: int, rng: np.random.Generator):
    if d < 3:
        raise ValueError("d must be at least three")
    raw = rng.normal(size=(d, 3))
    basis, _ = np.linalg.qr(raw)
    independent = basis.T * np.sqrt(d)
    weights = correlation_root(state.matrix) @ independent
    return tuple(row.copy() for row in weights)


def _renormalize(weights):
    return weights * (np.sqrt(weights.size) / np.linalg.norm(weights))


def _student_with_correlations(
    environment,
    student,
    world_model,
    rho_se: float,
    rho_sm: float,
):
    """Adjust student correlations while retaining its orthogonal direction."""

    d = student.size
    rho_me = float(environment @ world_model / d)
    pair = np.array([rho_se, rho_sm], dtype=float)
    denominator = pair[0] ** 2 + pair[1] ** 2 - 2.0 * rho_me * np.prod(pair)
    radius = max(1.0 - rho_me**2, 0.0)
    if denominator > radius and denominator > 0.0:
        pair *= np.sqrt(radius / denominator)
    gram = np.array([[1.0, rho_me], [rho_me, 1.0]])
    inverse = np.linalg.pinv(gram)
    coefficients = inverse @ pair
    current_pair = np.array(
        [student @ environment / d, student @ world_model / d]
    )
    current_coefficients = inverse @ current_pair
    residual = (
        student
        - current_coefficients[0] * environment
        - current_coefficients[1] * world_model
    )
    residual_variance = max(1.0 - pair @ coefficients, 0.0)
    residual_norm = np.linalg.norm(residual)
    if residual_norm > 1e-14 and residual_variance > 0.0:
        residual *= np.sqrt(d * residual_variance) / residual_norm
    else:
        residual = np.zeros_like(student)
    adjusted = (
        coefficients[0] * environment
        + coefficients[1] * world_model
        + residual
    )
    return _renormalize(adjusted)


def _apply_fixed_zero_overrides(controls: EpisodeControls, params: Params):
    """Apply hard control constraints after all adaptive calculations."""

    return EpisodeControls(
        beta=controls.beta,
        K=controls.K,
        eta_on1=0.0 if params.fix_eta_on1_zero else controls.eta_on1,
        eta_on2=0.0 if params.fix_eta_on2_zero else controls.eta_on2,
        eta_off=controls.eta_off,
        eta_wm=0.0 if params.fix_eta_wm_zero else controls.eta_wm,
    )


def _controls(environment, student, world_model, params, mode):
    state = state_from_weights(environment, student, world_model)
    if (mode == "online" and params.greedy_training) or params.learning_rate_mode == "optimal" or (
        mode == "online" and params.K_mode != "constant"
    ):
        comp = components(state, params, mode)
        return _apply_fixed_zero_overrides(
            EpisodeControls(
                comp.beta,
                comp.K,
                comp.eta_on1,
                comp.eta_on2,
                comp.eta_off,
                comp.eta_wm,
            ),
            params,
        )
    training_params = greedy_policy_params(params) if params.greedy_training else params
    beta = effective_temperature(state, training_params)
    if mode == "online":
        return _apply_fixed_zero_overrides(
            EpisodeControls(
                beta,
                training_params.K,
                params.eta_on1,
                params.eta_on2,
                0.0,
                params.eta_wm,
            ),
            params,
        )
    return EpisodeControls(beta, training_params.K, eta_off=params.eta_off)


def select_maximizing_plan(candidates, world_model_labels, rng, prefix_length=None):
    """Select uniformly from plans maximizing binary WM prefix reward."""

    prefix = world_model_labels.size if prefix_length is None else int(prefix_length)
    rewards = np.all(
        candidates[:, :prefix] == world_model_labels[None, :prefix],
        axis=1,
    ).astype(float)
    maximizers = np.flatnonzero(rewards == np.max(rewards))
    selected = int(rng.choice(maximizers))
    return candidates[selected], float(rewards[selected]), selected


def sample_episode_updates(
    environment, student, world_model, params, mode, rng, controls=None
):
    d = student.size
    controls = controls or _controls(environment, student, world_model, params, mode)
    inputs = rng.normal(size=(params.T, d))
    u = inputs @ environment / np.sqrt(d)
    h = inputs @ student / np.sqrt(d)
    v = inputs @ world_model / np.sqrt(d)
    y = sign_nonzero(u)
    yhat = sign_nonzero(v)

    if mode == "online":
        probability = thermal_probability(controls.beta, h)
        candidates = np.where(
            rng.random(size=(controls.K, params.T)) < probability[None, :], 1.0, -1.0
        )
        action, predicted_reward, _ = select_maximizing_plan(
            candidates, yhat, rng, params.effective_prefix_length
        )
        true_reward = float(np.all(action == y))
        observed_probability = 1.0 - params.epsilon if true_reward else params.epsilon
        observed_reward = float(rng.random() < observed_probability)
        eta_student = controls.eta_on1 + controls.eta_on2 * observed_reward
    else:
        action = sign_nonzero(h)
        predicted_reward = float(np.all(action == yhat))
        true_reward = float(np.all(action == y))
        observed_reward = np.nan
        eta_student = controls.eta_off * predicted_reward

    student_score = 0.5 * action - params.lambda_s * h
    delta_student = (
        eta_student
        * np.sum(student_score[:, None] * inputs, axis=0)
        / (params.T * np.sqrt(d))
    )

    if mode == "online" and controls.eta_wm != 0.0:
        flip = np.where(rng.random(params.T) < params.p_flip, -1.0, 1.0)
        target = flip * y
        wm_score = 0.5 * target - params.lambda_m * v
        delta_wm = (
            controls.eta_wm
            * np.sum(wm_score[:, None] * inputs, axis=0)
            / (params.T * np.sqrt(d))
        )
    else:
        delta_wm = np.zeros_like(world_model)

    return delta_student, delta_wm, controls, {
        "planner_accepted": float(mode == "online" and predicted_reward > 0.0),
        "environment_reward": true_reward,
        "predicted_reward": predicted_reward,
        "observed_reward": observed_reward,
    }


def _drift_environment(environment, rate, elapsed, rng):
    if rate == 0.0 or elapsed == 0.0:
        return environment
    retained = float(np.exp(-rate * elapsed))
    noise = rng.normal(size=environment.size)
    noise -= environment * (noise @ environment / environment.size)
    norm = np.linalg.norm(noise)
    if norm <= 1e-14:
        return environment
    noise *= np.sqrt(environment.size) / norm
    return retained * environment + np.sqrt(max(1.0 - retained**2, 0.0)) * noise


def run_empirical(
    initial: State,
    params: Params,
    *,
    schedule: Schedule | None = None,
    duration: float = 5.0,
    d: int = 2000,
    record_every: int = 100,
    seed: int | None = 0,
) -> list[dict]:
    initial.validate()
    params.validate()
    if d < 3 or record_every < 1:
        raise ValueError("d must be at least three and record_every positive")
    schedule = schedule or Schedule.single("online", duration)
    rng = np.random.default_rng(seed)
    environment, student, world_model = initial_weights(initial, d, rng)
    environment = _renormalize(environment)
    student = _renormalize(student)
    world_model = _renormalize(world_model)
    time = 0.0
    update = 0
    records = []
    window = {name: 0.0 for name in ("planner_accepted", "environment_reward", "predicted_reward", "observed_reward")}
    window_count = 0

    def record(phase_index, mode):
        nonlocal window, window_count
        state = state_from_weights(environment, student, world_model)
        control = _controls(environment, student, world_model, params, mode)
        reuse_training_policy = (
            not params.greedy_training
            and (mode == "online" or params.K_mode == "constant")
        )
        evaluation = evaluation_probabilities(
            state,
            params,
            planning_K=control.K if reuse_training_policy else None,
            planning_beta=control.beta if reuse_training_policy else None,
            time=time,
        )
        denominator = max(window_count, 1)
        records.append(
            {
                "time": time,
                "update": update,
                "phase": phase_index,
                "mode": mode,
                "rho_se": state.rho_se,
                "rho_me": state.rho_me,
                "rho_sm": state.rho_sm,
                "beta": control.beta,
                "K": control.K,
                "training_beta": control.beta,
                "training_K": control.K,
                "eta_on1": control.eta_on1,
                "eta_on2": control.eta_on2,
                "eta_off": control.eta_off,
                "eta_wm": control.eta_wm,
                **{f"recent_{key}": value / denominator for key, value in window.items()},
                **evaluation,
            }
        )
        window = {key: 0.0 for key in window}
        window_count = 0

    for phase_index, phase in enumerate(schedule.phases):
        if not records:
            record(phase_index, phase.mode)
        adaptive_K = (
            phase.mode == "online"
            and not params.greedy_training
            and params.K_mode != "constant"
        )
        fixed_cost = (
            1.0 + params.tau_on
            if phase.mode == "online" and params.greedy_training
            else params.c_on if phase.mode == "online" else params.c_off
        )
        maximum_modulation = 1.0 + params.M
        opportunity_factor = maximum_modulation if phase.mode == "online" else 1.0
        number = int(round(phase.duration * d * opportunity_factor / fixed_cost))
        phase_elapsed = 0.0
        phase_update = 0
        while True:
            control = _controls(environment, student, world_model, params, phase.mode)
            cost = (
                1.0 + control.K * params.tau_on
                if phase.mode == "online"
                else params.c_off
            )
            opportunity_factor = (
                maximum_modulation if phase.mode == "online" else 1.0
            )
            elapsed_per_update = cost / (d * opportunity_factor)
            if adaptive_K:
                if phase_elapsed + 0.5 * elapsed_per_update > phase.duration:
                    break
            elif phase_update >= number:
                break
            interval_start = time
            interval_end = time + elapsed_per_update
            mean_modulation = average_environment_modulation(
                interval_start, interval_end, params
            )
            active_environment = True
            if phase.mode == "online" and params.M != 0.0:
                active_environment = bool(
                    rng.random() < np.clip(mean_modulation / maximum_modulation, 0.0, 1.0)
                )
            if active_environment:
                state_before = (
                    state_from_weights(environment, student, world_model)
                    if phase.mode == "offline" and params.M != 0.0
                    else None
                )
                delta_student, delta_wm, _, info = sample_episode_updates(
                    environment,
                    student,
                    world_model,
                    params,
                    phase.mode,
                    rng,
                    controls=control,
                )
                student = _renormalize(student + delta_student)
                world_model = _renormalize(world_model + delta_wm)
                if state_before is not None:
                    state_after = state_from_weights(
                        environment, student, world_model
                    )
                    target_rho_se = state_before.rho_se + mean_modulation * (
                        state_after.rho_se - state_before.rho_se
                    )
                    student = _student_with_correlations(
                        environment,
                        student,
                        world_model,
                        target_rho_se,
                        state_after.rho_sm,
                    )
            else:
                info = {
                    "planner_accepted": 0.0,
                    "environment_reward": 0.0,
                    "predicted_reward": 0.0,
                    "observed_reward": 0.0,
                }
            drift_elapsed = integrated_environment_modulation(
                interval_start, interval_end, params
            )
            environment = _drift_environment(
                environment, params.eta_e, drift_elapsed, rng
            )
            update += 1
            phase_update += 1
            time += elapsed_per_update
            phase_elapsed += elapsed_per_update
            window_count += 1
            for key, value in info.items():
                if np.isfinite(value):
                    scale = maximum_modulation if phase.mode == "online" else 1.0
                    window[key] += scale * value
            if update % record_every == 0:
                record(phase_index, phase.mode)
        if not records or records[-1]["update"] != update:
            record(phase_index, phase.mode)
    return records
