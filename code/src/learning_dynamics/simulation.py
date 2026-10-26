"""Elapsed-time integration of the analytical correlation dynamics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .config import Params, Schedule, State
from .dynamics import components, rhs
from .observables import observables
from .optimization import resolve_online_policy


def _record(
    time: float,
    phase_index: int,
    mode: str,
    state: State,
    params: Params,
) -> dict:
    state = state.projected()
    resolved = resolve_online_policy(state, params)
    result = components(
        state,
        params,
        mode,
        time,
        resolved=resolved if mode == "online" else None,
    )
    evaluation = observables(state, params, time=time, resolved=resolved)
    return {
        "time": float(time),
        "phase": phase_index,
        "mode": mode,
        "rho_se": state.rho_se,
        "rho_me": state.rho_me,
        "rho_sm": state.rho_sm,
        **result.as_dict(),
        **evaluation,
        # Keep the hypothetical evaluation rate used by existing analyses,
        # while exposing the rate actually earned by the active training mode.
        "evaluation_reward_rate": evaluation["reward_rate"],
        "actual_reward_rate": result.reward_rate,
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
    """Integrate a schedule of online/offline analytical dynamics."""

    initial.validate()
    params.validate()
    if duration < 0.0 or step <= 0.0:
        raise ValueError("duration must be nonnegative and step positive")
    schedule = schedule or Schedule.single("online", duration)
    values = initial.as_array()
    offset = 0.0
    records: list[dict] = []
    for phase_index, phase in enumerate(schedule.phases):
        if phase.duration == 0.0:
            if not records:
                records.append(
                    _record(offset, phase_index, phase.mode, initial, params)
                )
            continue
        count = max(1, int(np.ceil(phase.duration / step)))
        evaluation_times = np.linspace(0.0, phase.duration, count + 1)
        solution = solve_ivp(
            lambda local_time, state_values: rhs(
                offset + local_time, state_values, params, phase.mode
            ),
            (0.0, phase.duration),
            State.from_array(values, project=True).as_array(),
            t_eval=evaluation_times,
            max_step=step,
            method=method,
            rtol=rtol,
            atol=atol,
        )
        if not solution.success:
            raise RuntimeError(solution.message)
        for index, (local_time, state_values) in enumerate(
            zip(solution.t, solution.y.T)
        ):
            if phase_index and index == 0:
                continue
            state = State.from_array(state_values, project=True)
            records.append(
                _record(offset + local_time, phase_index, phase.mode, state, params)
            )
        values = State.from_array(solution.y[:, -1], project=True).as_array()
        offset += phase.duration
    return records
