"""Elapsed-time integration of the analytical dynamics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .config import Params, Schedule, State
from .dynamics import components, rhs
from .observables import evaluation_probabilities


def _record(time, phase, mode, state, params):
    state = state.projected()
    comp = components(state, params, mode, time)
    reuse_training_policy = (
        not params.greedy_training and (mode == "online" or params.K_mode == "constant")
    )
    evaluation = evaluation_probabilities(
        state,
        params,
        planning_K=comp.K if reuse_training_policy else None,
        planning_beta=comp.beta if reuse_training_policy else None,
        time=time,
    )
    return {
        "time": float(time),
        "phase": phase,
        "mode": mode,
        "rho_se": state.rho_se,
        "rho_me": state.rho_me,
        "rho_sm": state.rho_sm,
        **comp.as_dict(),
        "training_K": comp.K,
        "training_beta": comp.beta,
        **evaluation,
    }


def run_analytical(
    initial: State,
    params: Params,
    *,
    schedule: Schedule | None = None,
    duration: float = 5.0,
    step: float = 0.05,
    method: str = "RK45",
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> list[dict]:
    initial.validate()
    params.validate()
    if step <= 0.0 or duration < 0.0:
        raise ValueError("step must be positive and duration nonnegative")
    schedule = schedule or Schedule.single("online", duration)
    state_values = initial.as_array()
    offset = 0.0
    records = []
    for phase_index, phase in enumerate(schedule.phases):
        if phase.duration == 0.0:
            if not records:
                records.append(_record(offset, phase_index, phase.mode, initial, params))
            continue
        count = max(1, int(np.ceil(phase.duration / step)))
        evaluation_times = np.linspace(0.0, phase.duration, count + 1)
        result = solve_ivp(
            lambda time, values: rhs(offset + time, values, params, phase.mode),
            (0.0, phase.duration),
            State.from_array(state_values, project=True).as_array(),
            t_eval=evaluation_times,
            method=method,
            rtol=rtol,
            atol=atol,
            max_step=step,
        )
        if not result.success:
            raise RuntimeError(result.message)
        for index, (local_time, values) in enumerate(zip(result.t, result.y.T)):
            if phase_index and index == 0:
                continue
            state = State.from_array(values, project=True)
            records.append(_record(offset + local_time, phase_index, phase.mode, state, params))
        state_values = State.from_array(result.y[:, -1], project=True).as_array()
        offset += phase.duration
    return records
