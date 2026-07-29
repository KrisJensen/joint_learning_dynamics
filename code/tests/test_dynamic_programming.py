import importlib
from pathlib import Path

import numpy as np
import pytest

from learning_dynamics import (
    DPConfig,
    Params,
    State,
    load_dp_result,
    save_dp_result,
    solve_online_dp,
    solve_switching_dp,
    value_at,
)


dynamic_programming = importlib.import_module(
    "learning_dynamics.dynamic_programming"
)


def test_dp_commensurability_and_round_trip(tmp_path: Path) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        delta_switch=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    result = solve_online_dp(params, config)
    state = State(0.1, 0.5, 0.1)
    before = value_at(result, config.horizon, state)
    path = tmp_path / "table.npz"
    save_dp_result(path, result)
    loaded = load_dp_result(path)
    after = value_at(loaded, config.horizon, state)
    np.testing.assert_allclose(after, before)
    assert result.policy.shape[1] == round(
        config.horizon / config.horizon_step(params)
    ) + 1


def test_dp_rejects_noncommensurate_costs() -> None:
    params = Params(T=3, M=2, delta_on=0.7, delta_off=0.5)
    config = DPConfig(horizon=1.0)
    with pytest.raises(ValueError, match="commensurate"):
        config.validate(params)


def test_dp_can_optimize_K_at_fixed_M(tmp_path: Path) -> None:
    params = Params(
        T=2,
        M=2,
        delta_on=0.5,
        delta_off=0.5,
        delta_switch=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=3,
        M=1,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )

    result = solve_online_dp(params, config)
    assert [(action.K, action.M) for action in result.actions] == [
        (1, 1),
        (2, 1),
        (3, 1),
    ]
    path = tmp_path / "fixed_M.npz"
    save_dp_result(path, result)
    loaded = load_dp_result(path)
    assert loaded.config.M == 1
    assert [(action.K, action.M) for action in loaded.actions] == [
        (1, 1),
        (2, 1),
        (3, 1),
    ]

    with pytest.raises(ValueError, match="fixed DP M"):
        DPConfig(**{**config.__dict__, "M": params.T + 1}).validate(params)


def test_static_dp_caches_interpolations_and_matches_scalar_backups(
    monkeypatch,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        delta_switch=0.5,
        C_A=0.8,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    original_prepare = dynamic_programming.FeasibleInterpolator.prepare
    calls = 0

    def counted_prepare(self, queries):
        nonlocal calls
        calls += 1
        return original_prepare(self, queries)

    monkeypatch.setattr(
        dynamic_programming.FeasibleInterpolator,
        "prepare",
        counted_prepare,
    )
    result = solve_online_dp(params, config)

    assert result.params.C_A == 0.0
    assert calls == len(result.actions)

    reference_values = np.zeros_like(result.values)
    reference_policy = np.full_like(result.policy, -1)
    for layer in range(1, len(result.times)):
        best = np.zeros(len(result.points))
        chosen = np.full(len(result.points), -1, dtype=np.int16)
        for action_index, table in enumerate(result.tables):
            steps = table.spec.grid_steps
            if steps > layer:
                continue
            interpolation = original_prepare(
                result.interpolator,
                result.points + table.delta,
            )
            candidate = table.reward + interpolation.apply(
                reference_values[0, layer - steps]
            )
            improve = candidate > best + 1e-14
            best[improve] = candidate[improve]
            chosen[improve] = action_index
        reference_values[0, layer] = best
        reference_policy[0, layer] = chosen

    np.testing.assert_allclose(result.values, reference_values)
    np.testing.assert_array_equal(result.policy, reference_policy)


def test_switching_rollout_scores_actions_at_continuous_state() -> None:
    """Rollout decisions must not copy a contradictory nearest-grid policy."""

    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        delta_switch=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    result = solve_switching_dp(params, config)
    result.values.fill(0.0)
    result.policy.fill(-1)

    offline_index = next(
        index
        for index, action in enumerate(result.actions)
        if action.kind == "offline"
    )
    switch = next(action for action in result.actions if action.kind == "switch")
    full_layer = len(result.times) - 1
    continuation_layer = full_layer - switch.grid_steps
    result.values[0, continuation_layer] = 2.0
    result.values[1, continuation_layer] = 1.0
    result.policy[1, full_layer] = offline_index

    rows = dynamic_programming.rollout_policy(
        result,
        State(0.1, 0.5, 0.1),
        initial_mode="off",
    )

    assert rows[0]["action"] == "switch"
    assert rows[0]["value"] == pytest.approx(2.0)
