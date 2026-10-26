"""Independent finite-dimensional empirical verification of the theory."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Params, Schedule, State
from .dynamics import components
from .fields import correlation_root, sign_nonzero, thermal_probability
from .modulation import average_cycle, integrated_cycle
from .observables import observables


@dataclass(frozen=True)
class EpisodeControls:
    beta: float
    K: int
    M: int
    eta_2f: float = 0.0
    eta_3f: float = 0.0
    eta_off: float = 0.0
    eta_wm: float = 0.0


def state_from_weights(environment, student, world_model) -> State:
    """Measure the three normalized overlaps from weight vectors."""

    d = student.size
    return State(
        float(student @ environment / d),
        float(world_model @ environment / d),
        float(student @ world_model / d),
    ).projected()


def initial_weights(state: State, d: int, rng: np.random.Generator):
    """Construct three norm-sqrt(d) weights with the requested Gram matrix."""

    if d < 3:
        raise ValueError("d must be at least three")
    basis, _ = np.linalg.qr(rng.normal(size=(d, 3)))
    independent = basis.T * np.sqrt(d)
    weights = correlation_root(state.matrix) @ independent
    return tuple(row.copy() for row in weights)


def _renormalize(weights):
    norm = np.linalg.norm(weights)
    if norm == 0.0:
        raise FloatingPointError("cannot normalize a zero weight vector")
    return weights * np.sqrt(weights.size) / norm


def _student_with_correlations(
    environment,
    student,
    world_model,
    rho_se: float,
    rho_sm: float,
):
    """Set student correlations while preserving an orthogonal direction."""

    d = student.size
    rho_me = float(environment @ world_model / d)
    pair = np.array([rho_se, rho_sm], dtype=float)
    gram = np.array([[1.0, rho_me], [rho_me, 1.0]])
    inverse = np.linalg.pinv(gram, rcond=1e-13)
    radial = float(pair @ inverse @ pair)
    if radial > 1.0 and radial > 0.0:
        pair /= np.sqrt(radial)
    coefficients = inverse @ pair
    current = np.array(
        [student @ environment / d, student @ world_model / d]
    )
    residual = student - (inverse @ current)[0] * environment - (
        inverse @ current
    )[1] * world_model
    residual_variance = max(1.0 - float(pair @ coefficients), 0.0)
    residual_norm = np.linalg.norm(residual)
    if residual_norm > 1e-14 and residual_variance > 0.0:
        residual *= np.sqrt(d * residual_variance) / residual_norm
    else:
        residual = np.zeros_like(student)
    return _renormalize(
        coefficients[0] * environment + coefficients[1] * world_model + residual
    )


def select_plan(
    candidates,
    world_model_labels,
    M: int,
    rng: np.random.Generator,
):
    """Uniformly choose among plans maximizing binary WM prefix reward."""

    rewards = np.all(
        candidates[:, :M] == world_model_labels[None, :M], axis=1
    ).astype(float)
    maximizers = np.flatnonzero(rewards == np.max(rewards))
    selected = int(rng.choice(maximizers))
    return candidates[selected], float(rewards[selected]), selected


def _controls(state: State, params: Params, mode: str, time: float) -> EpisodeControls:
    result = components(state, params, mode, time)
    return EpisodeControls(
        beta=result.beta,
        K=result.K,
        M=result.M,
        eta_2f=result.eta_2f,
        eta_3f=result.eta_3f,
        eta_off=result.eta_off,
        eta_wm=result.eta_wm,
    )


def sample_episode_updates(
    environment,
    student,
    world_model,
    params: Params,
    mode: str,
    rng: np.random.Generator,
    controls: EpisodeControls,
):
    """Sample one microscopic update from the report's learning rules."""

    d = student.size
    inputs = rng.normal(size=(params.T, d))
    scale = np.sqrt(d)
    u = inputs @ environment / scale
    h = inputs @ student / scale
    v = inputs @ world_model / scale
    y = sign_nonzero(u)
    yhat = sign_nonzero(v)

    if mode == "online":
        probability = thermal_probability(controls.beta, h)
        candidates = np.where(
            rng.random((controls.K, params.T)) < probability[None, :],
            1.0,
            -1.0,
        )
        action, predicted_reward, _ = select_plan(
            candidates, yhat, controls.M, rng
        )
        true_reward = float(np.all(action == y))
        reward_probability = (
            1.0 - params.epsilon if true_reward else params.epsilon
        )
        observed_reward = float(rng.random() < reward_probability)
        student_rate = controls.eta_2f + controls.eta_3f * observed_reward
    else:
        action = sign_nonzero(h)
        predicted_reward = float(np.all(action == yhat))
        true_reward = float(np.all(action == y))
        observed_reward = np.nan
        student_rate = controls.eta_off * predicted_reward

    # The common score factor 1/2 is already absorbed into every eta.
    delta_student = (
        student_rate * np.sum(action[:, None] * inputs, axis=0)
        / (params.T * scale)
    )
    if mode == "online" and controls.eta_wm != 0.0:
        flips = np.where(rng.random(params.T) < params.p_flip, -1.0, 1.0)
        target = flips * y
        delta_wm = (
            controls.eta_wm * np.sum(target[:, None] * inputs, axis=0)
            / (params.T * scale)
        )
    else:
        delta_wm = np.zeros_like(world_model)
    return delta_student, delta_wm, {
        "planner_accepted": float(mode == "online" and predicted_reward > 0.0),
        "true_reward": true_reward,
        "predicted_reward": predicted_reward,
        "observed_reward": observed_reward,
    }


def _drift_environment(environment, rate: float, exposure: float, rng):
    if rate == 0.0 or exposure == 0.0:
        return environment
    retained = float(np.exp(-rate * exposure))
    noise = rng.normal(size=environment.size)
    noise -= environment * (noise @ environment / environment.size)
    norm = np.linalg.norm(noise)
    if norm <= 1e-14:
        return environment
    noise *= np.sqrt(environment.size) / norm
    return _renormalize(
        retained * environment
        + np.sqrt(max(1.0 - retained**2, 0.0)) * noise
    )


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
    """Run a finite-dimensional trajectory in rescaled elapsed time."""

    initial.validate()
    params.validate()
    if d < 3 or record_every < 1:
        raise ValueError("d must be at least three and record_every positive")
    schedule = schedule or Schedule.single("online", duration)
    rng = np.random.default_rng(seed)
    environment, student, world_model = initial_weights(initial, d, rng)
    environment, student, world_model = map(
        _renormalize, (environment, student, world_model)
    )
    time = 0.0
    update = 0
    records: list[dict] = []
    window = {
        "planner_accepted": 0.0,
        "true_reward": 0.0,
        "predicted_reward": 0.0,
        "observed_reward": 0.0,
    }
    window_count = 0
    active_count = 0

    def record(phase_index: int, mode: str) -> None:
        nonlocal window, window_count, active_count
        state = state_from_weights(environment, student, world_model)
        control = _controls(state, params, mode, time)
        denominator = max(active_count, 1)
        records.append(
            {
                "time": time,
                "update": update,
                "phase": phase_index,
                "mode": mode,
                "rho_se": state.rho_se,
                "rho_me": state.rho_me,
                "rho_sm": state.rho_sm,
                "K": control.K,
                "M": control.M,
                "beta": control.beta,
                "eta_2f": control.eta_2f,
                "eta_3f": control.eta_3f,
                "eta_off": control.eta_off,
                "eta_wm": control.eta_wm,
                "recent_active_fraction": active_count / max(window_count, 1),
                **{
                    f"recent_{key}": value / denominator
                    for key, value in window.items()
                },
                **observables(state, params, time=time),
            }
        )
        window = {key: 0.0 for key in window}
        window_count = 0
        active_count = 0

    for phase_index, phase in enumerate(schedule.phases):
        if not records:
            record(phase_index, phase.mode)
        phase_start = time
        phase_updates = 0
        while True:
            state = state_from_weights(environment, student, world_model)
            control = _controls(state, params, phase.mode, time)
            cost = (
                params.online_cost(control.K, control.M)
                if phase.mode == "online"
                else params.delta_off
            )
            opportunity = 1.0 + params.C_A if phase.mode == "online" else 1.0
            elapsed = cost / (d * opportunity)
            if time - phase_start + 0.5 * elapsed > phase.duration:
                break
            interval_end = time + elapsed
            mean_cycle = average_cycle(time, interval_end, params)
            active = True
            if phase.mode == "online" and params.C_A != 0.0:
                active = bool(rng.random() < mean_cycle / opportunity)
            if active:
                before = (
                    state
                    if phase.mode == "offline" and params.C_A != 0.0
                    else None
                )
                delta_student, delta_wm, info = sample_episode_updates(
                    environment,
                    student,
                    world_model,
                    params,
                    phase.mode,
                    rng,
                    control,
                )
                student = _renormalize(student + delta_student)
                world_model = _renormalize(world_model + delta_wm)
                if before is not None:
                    after = state_from_weights(environment, student, world_model)
                    student = _student_with_correlations(
                        environment,
                        student,
                        world_model,
                        before.rho_se
                        + mean_cycle * (after.rho_se - before.rho_se),
                        after.rho_sm,
                    )
                active_count += 1
                for key, value in info.items():
                    if np.isfinite(value):
                        window[key] += value
            environment = _drift_environment(
                environment,
                params.eta_e,
                integrated_cycle(time, interval_end, params),
                rng,
            )
            time = interval_end
            update += 1
            phase_updates += 1
            window_count += 1
            if phase_updates % record_every == 0:
                record(phase_index, phase.mode)
        # Use the exact requested schedule boundary for the phase marker.
        time = phase_start + phase.duration
        record(phase_index, phase.mode)
    return records
