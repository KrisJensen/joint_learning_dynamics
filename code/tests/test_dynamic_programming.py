import importlib
from pathlib import Path

import numpy as np
import pytest

from learning_dynamics import (
    DPConfig,
    Params,
    State,
    additional_planning_time,
    build_dp_workspace,
    load_dp_result,
    minimum_online_block_time,
    optimal_action_at,
    rollout_policies,
    save_dp_result,
    solve_dp_workspace,
    solve_online_dp,
    solve_reward_only_planning_dp,
    solve_switching_dp,
    value_at,
)


dynamic_programming = importlib.import_module(
    "learning_dynamics.dynamic_programming"
)


def test_reward_only_planning_dp_matches_generic_shared_transition_solver() -> None:
    params = Params(
        T=2,
        M=1,
        beta=2.0,
        delta_on=0.5,
        delta_off=0.5,
        constant_prefix_time=True,
        gh_order=8,
    )
    config = DPConfig(
        horizon=4.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=2,
        M=None,
        t_switch=0.5,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    reference = solve_online_dp(params, config)
    counterfactual = solve_reward_only_planning_dp(reference)

    for table, original in zip(
        counterfactual.tables,
        reference.tables,
        strict=True,
    ):
        np.testing.assert_allclose(table.reward, original.reward)
        np.testing.assert_allclose(
            table.delta,
            counterfactual.tables[0].delta,
        )

    prepared = {
        table.spec: dynamic_programming._prepare_transition(
            table,
            counterfactual.points,
            counterfactual.params,
            counterfactual.config,
            0.0,
            counterfactual.interpolator,
        )
        for table in counterfactual.tables
    }
    workspace = dynamic_programming.DPWorkspace(
        counterfactual.config,
        counterfactual.params,
        counterfactual.points,
        counterfactual.interpolator,
        counterfactual.tables,
        prepared,
    )
    generic = solve_dp_workspace(workspace, variant="online")
    np.testing.assert_allclose(counterfactual.values, generic.values)
    np.testing.assert_array_equal(counterfactual.policy, generic.policy)

    initial = State(0.2, 0.5, 0.1)
    action = optimal_action_at(counterfactual, initial, config.horizon)
    rollout = dynamic_programming.rollout_policy(counterfactual, initial)
    assert action is not None
    assert action.name == rollout[0]["action"]


def test_additional_planning_time_uses_constant_prefix_baseline() -> None:
    params = Params(
        T=6,
        M=6,
        delta_on=0.1,
        constant_prefix_time=True,
    )
    one_short_burst = [
        {"action": "online(K=1,M=6)", "K": 1, "M": 6},
        {"action": "online(K=6,M=3)", "K": 6, "M": 3},
        {"action": "online(K=2,M=1)", "K": 2, "M": 1},
        {"action": "offline", "K": "", "M": ""},
    ]
    threshold = minimum_online_block_time(params, step_size=0.2)
    planning_time = additional_planning_time(
        one_short_burst,
        params,
        step_size=0.2,
    )

    assert threshold == pytest.approx(0.22)
    assert planning_time == pytest.approx(0.12)
    assert planning_time < threshold

    exactly_one_block = one_short_burst + [
        {"action": "online(K=6,M=3)", "K": 6, "M": 3},
    ]
    assert additional_planning_time(
        exactly_one_block,
        params,
        step_size=0.2,
    ) == pytest.approx(threshold)


def test_additional_planning_time_respects_proportional_M() -> None:
    params = Params(T=4, M=1, delta_on=0.4)
    rollout = [
        {"action": "online(K=2,M=3)", "K": 2, "M": 3},
        {"action": "online(K=3,M=1)", "K": 3, "M": 1},
    ]

    assert additional_planning_time(
        rollout,
        params,
        step_size=0.5,
    ) == pytest.approx(0.35)
    assert minimum_online_block_time(params, 0.5) == pytest.approx(0.55)


def test_constant_prefix_time_gives_all_M_equal_dp_duration() -> None:
    config = DPConfig(
        horizon=3.0,
        step_size=1.0,
        K_max=2,
        t_switch=0.5,
    )
    proportional = Params(T=4, M=1, delta_on=0.5)
    constant = Params(
        T=4,
        M=1,
        delta_on=0.5,
        delta_off=0.5,
        constant_prefix_time=True,
    )

    proportional_actions = dynamic_programming._action_specs(
        proportional,
        config,
        "online",
    )
    constant_actions = dynamic_programming._action_specs(
        constant,
        config,
        "online",
    )
    proportional_k2 = [
        action for action in proportional_actions if action.K == 2
    ]
    constant_k2 = [action for action in constant_actions if action.K == 2]

    assert [action.duration for action in proportional_k2] == pytest.approx(
        [1.25, 1.5, 1.75, 2.0]
    )
    assert [action.duration for action in constant_k2] == pytest.approx(
        [2.0, 2.0, 2.0, 2.0]
    )
    assert len({action.grid_steps for action in constant_k2}) == 1
    assert config.horizon_step(constant) == pytest.approx(0.5)
    assert constant_k2[0].grid_steps == 4


def test_constant_prefix_time_warns_and_uses_fine_grid_when_needed() -> None:
    params = Params(
        T=3,
        M=1,
        delta_on=0.3,
        delta_off=0.3,
        constant_prefix_time=True,
    )
    config = DPConfig(horizon=1.0, step_size=1.0, t_switch=0.3)

    with pytest.warns(RuntimeWarning, match="Falling back to q=delta_on/T"):
        config.validate(params)

    assert config.horizon_step(params) == pytest.approx(0.1)


def test_coarse_constant_prefix_grid_matches_fine_grid_values() -> None:
    common = dict(
        T=2,
        M=2,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    fine_params = Params(**common)
    coarse_params = Params(**common, constant_prefix_time=True)
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        M=2,
        t_switch=0.5,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )

    fine = solve_switching_dp(fine_params, config)
    coarse = solve_switching_dp(coarse_params, config)

    assert config.horizon_step(fine_params) == pytest.approx(0.25)
    assert config.horizon_step(coarse_params) == pytest.approx(0.5)
    np.testing.assert_allclose(fine.times[::2], coarse.times)
    np.testing.assert_allclose(
        fine.values[:, ::2],
        coarse.values,
        rtol=2e-6,
        atol=2e-7,
    )
    np.testing.assert_array_equal(fine.policy[:, ::2], coarse.policy)


def test_dp_commensurability_and_round_trip(
    tmp_path: Path,
    monkeypatch,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        t_switch=0.5,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    result = solve_online_dp(params, config)
    state = State(0.1, 0.5, 0.1)
    before = value_at(result, config.horizon, state)
    path = tmp_path / "table.npz"
    save_dp_result(path, result)
    assert result.values.dtype == np.float32
    for table in result.tables:
        assert table.delta.dtype == np.float64
        assert table.reward.dtype == np.float64
        assert table.vector_field.dtype == np.float64
        assert table.probability.dtype == np.float64

    def reject_action_table_rebuild(*args, **kwargs):
        raise AssertionError("saved action tables were rebuilt during loading")

    monkeypatch.setattr(
        dynamic_programming,
        "_build_action_table",
        reject_action_table_rebuild,
    )
    loaded = load_dp_result(path)
    assert loaded.values.dtype == np.float32
    after = value_at(loaded, config.horizon, state)
    np.testing.assert_allclose(after, before)
    for expected, actual in zip(result.tables, loaded.tables):
        assert actual.spec == expected.spec
        np.testing.assert_array_equal(actual.delta, expected.delta)
        np.testing.assert_array_equal(actual.reward, expected.reward)
        np.testing.assert_array_equal(actual.vector_field, expected.vector_field)
        np.testing.assert_array_equal(actual.probability, expected.probability)
    assert result.policy.shape[1] == round(
        config.horizon / config.horizon_step(params)
    ) + 1


def test_dp_rejects_noncommensurate_costs() -> None:
    params = Params(T=3, M=2, delta_on=0.7, delta_off=0.5)
    config = DPConfig(horizon=1.0)
    with pytest.raises(ValueError, match="commensurate"):
        config.validate(params)


def test_strategy_switch_cost_must_lie_on_horizon_grid() -> None:
    params = Params(T=2, M=1, delta_on=0.5, delta_off=0.5)
    with pytest.raises(ValueError, match="t_strategy_switch"):
        DPConfig(
            horizon=2.0,
            step_size=1.0,
            t_switch=0.5,
            t_strategy_switch=0.3,
        ).validate(params)


def test_zero_strategy_switch_cost_keeps_full_float32_history() -> None:
    params = Params(
        T=2,
        M=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=1.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        M=1,
        t_strategy_switch=0.0,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )

    result = solve_online_dp(params, config)

    assert result.config.t_strategy_switch == 0.0
    assert result.values is not None
    assert result.values.dtype == np.float32
    assert result.policy is not None
    assert all(action.kind != "strategy_switch" for action in result.actions)


def test_positive_strategy_switch_cost_matches_full_reference_and_round_trips(
    tmp_path: Path,
) -> None:
    params = Params(
        T=2,
        M=1,
        delta_on=0.5,
        delta_off=0.5,
        eta_e=0.1,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=2,
        M=1,
        t_switch=0.5,
        t_strategy_switch=0.5,
        record_compact_policy=True,
    )

    result = solve_switching_dp(params, config)
    assert result.values is None
    assert result.policy is None
    assert result.terminal_values is not None
    assert result.terminal_values.shape == (2, len(result.points))
    assert result.terminal_values.dtype == np.float64
    assert result.compact_policy is not None

    online_indices = [
        index
        for index, table in enumerate(result.tables)
        if table.spec.kind == "online"
    ]
    strategy_switch_index = next(
        index
        for index, table in enumerate(result.tables)
        if table.spec.kind == "strategy_switch"
    )
    strategy_switch_table = result.tables[strategy_switch_index]
    np.testing.assert_array_equal(
        strategy_switch_table.reward,
        np.zeros(len(result.points)),
    )
    np.testing.assert_array_equal(
        strategy_switch_table.probability,
        np.zeros(len(result.points)),
    )
    retained = np.exp(-params.eta_e * config.t_strategy_switch)
    next_points = result.points + strategy_switch_table.delta
    np.testing.assert_allclose(
        next_points[:, :2],
        retained * result.points[:, :2],
        atol=1e-14,
    )
    np.testing.assert_allclose(
        next_points[:, 2],
        result.points[:, 2],
        atol=1e-14,
    )
    offline_index = next(
        index
        for index, table in enumerate(result.tables)
        if table.spec.kind == "offline"
    )
    switch_index = next(
        index
        for index, table in enumerate(result.tables)
        if table.spec.kind == "switch"
    )
    prepared = [
        result.interpolator.prepare(result.points + table.delta)
        for table in result.tables
    ]
    layer_count = len(result.times)
    point_count = len(result.points)
    strategy_values = np.zeros(
        (len(online_indices), layer_count, point_count),
        dtype=np.float64,
    )
    envelope = np.zeros((layer_count, point_count), dtype=np.float64)
    offline_values = np.zeros((layer_count, point_count), dtype=np.float64)
    strategy_decisions = np.zeros_like(strategy_values, dtype=np.uint8)
    offline_decisions = np.zeros_like(offline_values, dtype=np.uint8)

    for layer in range(1, layer_count):
        for strategy_position, action_index in enumerate(online_indices):
            table = result.tables[action_index]
            best = np.zeros(point_count, dtype=np.float64)
            decisions = np.zeros(point_count, dtype=np.uint8)
            if table.spec.grid_steps <= layer:
                candidate = table.reward + prepared[action_index].apply(
                    strategy_values[
                        strategy_position,
                        layer - table.spec.grid_steps,
                    ]
                )
                improve = candidate > best
                best[improve] = candidate[improve]
                decisions[improve] = dynamic_programming._POLICY_CONTINUE
            strategy_switch = result.tables[strategy_switch_index]
            if strategy_switch.spec.grid_steps <= layer:
                candidate = prepared[strategy_switch_index].apply(
                    envelope[layer - strategy_switch.spec.grid_steps]
                )
                improve = candidate > best
                best[improve] = candidate[improve]
                decisions[improve] = (
                    dynamic_programming._POLICY_STRATEGY_SWITCH
                )
            mode_switch = result.tables[switch_index]
            if mode_switch.spec.grid_steps <= layer:
                candidate = prepared[switch_index].apply(
                    offline_values[layer - mode_switch.spec.grid_steps]
                )
                improve = candidate > best
                best[improve] = candidate[improve]
                decisions[improve] = dynamic_programming._POLICY_MODE_SWITCH
            strategy_values[strategy_position, layer] = best
            strategy_decisions[strategy_position, layer] = decisions
        envelope[layer] = np.max(strategy_values[:, layer], axis=0)

        offline = result.tables[offline_index]
        best_offline = np.zeros(point_count, dtype=np.float64)
        decisions = np.zeros(point_count, dtype=np.uint8)
        if offline.spec.grid_steps <= layer:
            candidate = offline.reward + prepared[offline_index].apply(
                offline_values[layer - offline.spec.grid_steps]
            )
            improve = candidate > best_offline
            best_offline[improve] = candidate[improve]
            decisions[improve] = dynamic_programming._POLICY_CONTINUE
        mode_switch = result.tables[switch_index]
        if mode_switch.spec.grid_steps <= layer:
            candidate = prepared[switch_index].apply(
                envelope[layer - mode_switch.spec.grid_steps]
            )
            improve = candidate > best_offline
            best_offline[improve] = candidate[improve]
            decisions[improve] = dynamic_programming._POLICY_MODE_SWITCH
        offline_values[layer] = best_offline
        offline_decisions[layer] = decisions

    np.testing.assert_allclose(
        result.terminal_values,
        np.stack((envelope[-1], offline_values[-1])),
        atol=1e-13,
        rtol=1e-13,
    )
    compact = result.compact_policy
    expected_width = (point_count + 3) // 4
    assert compact.strategy_decisions.shape == (
        len(online_indices),
        layer_count,
        expected_width,
    )
    assert compact.strategy_decisions.nbytes == (
        len(online_indices) * layer_count * expected_width
    )
    assert compact.nbytes == (
        compact.strategy_decisions.nbytes
        + compact.offline_decisions.nbytes
        + compact.envelope_strategy.nbytes
    )
    for strategy_position in range(len(online_indices)):
        for layer in range(layer_count):
            decoded = [
                dynamic_programming._unpack_policy_decision(
                    compact.strategy_decisions[strategy_position, layer],
                    point,
                )
                for point in range(point_count)
            ]
            np.testing.assert_array_equal(
                decoded,
                strategy_decisions[strategy_position, layer],
            )
    for layer in range(layer_count):
        decoded = [
            dynamic_programming._unpack_policy_decision(
                compact.offline_decisions[layer],
                point,
            )
            for point in range(point_count)
        ]
        np.testing.assert_array_equal(decoded, offline_decisions[layer])
    np.testing.assert_array_equal(
        compact.envelope_strategy,
        np.argmax(strategy_values, axis=0),
    )

    initial = State(0.1, 0.5, 0.1)
    assert np.isfinite(value_at(result, config.horizon, initial))
    with pytest.raises(ValueError, match="final-horizon"):
        value_at(result, config.horizon / 2.0, initial)
    rollout = rollout_policies(result, [initial])[0]
    assert rollout[-1]["action"] == "terminal"
    assert any(item["action"].startswith("online(") for item in rollout)

    path = tmp_path / "rolling_strategy_switch.npz"
    save_dp_result(path, result)
    with np.load(path, allow_pickle=False) as data:
        assert "terminal_values" in data.files
        assert "values" not in data.files
        assert "policy" not in data.files
        assert "compact_strategy_decisions" in data.files
        assert "compact_offline_decisions" in data.files
        assert "compact_envelope_strategy" in data.files
    loaded = load_dp_result(path)
    assert loaded.values is None
    assert loaded.policy is None
    assert loaded.compact_policy is not None
    assert loaded.config.t_strategy_switch == pytest.approx(0.5)
    np.testing.assert_array_equal(loaded.terminal_values, result.terminal_values)
    np.testing.assert_array_equal(
        loaded.compact_policy.strategy_decisions,
        compact.strategy_decisions,
    )
    loaded_rollout = rollout_policies(loaded, [initial])[0]
    assert [row["action"] for row in loaded_rollout] == [
        row["action"] for row in rollout
    ]


def test_single_strategy_still_has_shared_switch_action() -> None:
    params = Params(
        T=2,
        M=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    result = solve_online_dp(
        params,
        DPConfig(
            horizon=1.0,
            step_size=1.0,
            rho_points=3,
            rho_max=0.8,
            K_max=1,
            M=1,
            t_strategy_switch=0.5,
            fixed_wm=True,
            fixed_rho_me=0.5,
        ),
    )

    assert [action.kind for action in result.actions].count("online") == 1
    assert [action.kind for action in result.actions].count("strategy_switch") == 1
    assert result.compact_policy is None
    with pytest.raises(ValueError, match="compact policy recording"):
        rollout_policies(result, [State(0.1, 0.5, 0.1)])


def test_modulated_strategy_switch_uses_rolling_solver() -> None:
    params = Params(
        T=2,
        M=1,
        delta_on=0.5,
        delta_off=0.5,
        eta_e=0.1,
        C_A=0.4,
        C_T=2.0,
        gh_order=8,
    )
    result = solve_online_dp(
        params,
        DPConfig(
            horizon=1.0,
            step_size=1.0,
            rho_points=3,
            rho_max=0.8,
            K_max=1,
            M=1,
            t_strategy_switch=0.5,
            record_compact_policy=True,
            no_modulation=False,
        ),
    )

    assert result.values is None
    assert result.params.C_A == pytest.approx(params.C_A)
    assert np.all(np.isfinite(result.terminal_values))
    rows = rollout_policies(result, [State(0.0, 0.0, 0.0)])[0]
    assert rows[-1]["action"] == "terminal"


def test_switch_time_is_fixed_when_dp_resolution_changes() -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        eta_e=0.4,
        gh_order=8,
    )
    coarse = solve_switching_dp(
        params,
        DPConfig(
            horizon=2.0,
            step_size=1.0,
            t_switch=0.5,
            rho_points=3,
            rho_max=0.8,
            K_max=1,
            fixed_wm=True,
            fixed_rho_me=0.5,
        ),
    )
    fine = solve_switching_dp(
        params,
        DPConfig(
            horizon=2.0,
            step_size=0.5,
            t_switch=0.5,
            rho_points=3,
            rho_max=0.8,
            K_max=1,
            fixed_wm=True,
            fixed_rho_me=0.5,
        ),
    )
    coarse_switch = next(
        table for table in coarse.tables if table.spec.kind == "switch"
    )
    fine_switch = next(
        table for table in fine.tables if table.spec.kind == "switch"
    )

    assert coarse_switch.spec.duration == pytest.approx(0.5)
    assert fine_switch.spec.duration == pytest.approx(0.5)
    assert coarse_switch.spec.grid_steps == 2
    assert fine_switch.spec.grid_steps == 4
    np.testing.assert_allclose(coarse_switch.delta, fine_switch.delta)

    with pytest.raises(ValueError, match="t_switch"):
        DPConfig(
            horizon=2.0,
            step_size=0.5,
            t_switch=0.3,
        ).validate(params)


def test_dp_can_optimize_K_at_fixed_M(tmp_path: Path) -> None:
    params = Params(
        T=2,
        M=2,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=3,
        M=1,
        t_switch=0.5,
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


def test_workspace_reuses_actions_across_horizons_and_switch_times(
    tmp_path: Path,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=2,
        M=1,
        t_switch=0.5,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    workspace = build_dp_workspace(params, config)
    full = solve_dp_workspace(
        workspace,
        variant="switching",
        horizon=2.0,
        t_switch=0.5,
    )
    short = solve_dp_workspace(
        workspace,
        variant="switching",
        horizon=1.0,
        t_switch=0.5,
    )
    short_layers = len(short.times)
    np.testing.assert_allclose(full.values[:, :short_layers], short.values)
    np.testing.assert_array_equal(full.policy[:, :short_layers], short.policy)

    other_time = solve_dp_workspace(
        workspace,
        variant="switching",
        horizon=2.0,
        t_switch=1.0,
    )
    k1 = solve_dp_workspace(
        workspace,
        variant="switching",
        horizon=2.0,
        t_switch=1.0,
        allowed_K=(1,),
    )
    assert full.tables[0] is other_time.tables[0]
    assert other_time.tables[-1] is k1.tables[-1]
    assert {
        action.K
        for action in k1.actions
        if action.kind == "online"
    } == {1}
    assert set(workspace.switch_tables) == {0.5, 1.0}
    path = tmp_path / "restricted_workspace_result.npz"
    save_dp_result(path, k1)
    loaded = load_dp_result(path)
    assert loaded.actions == k1.actions
    assert len(loaded.tables) == len(k1.tables)
    assert loaded.config.t_switch == pytest.approx(1.0)

    rows = dynamic_programming.rollout_policy(
        full,
        State(0.1, 0.5, 0.1),
        horizon=1.0,
    )
    assert rows[0]["remaining_horizon"] == pytest.approx(1.0)


def test_workspace_rejects_a_requested_time_that_changes_its_quantum() -> None:
    params = Params(
        T=2,
        M=2,
        delta_on=0.5,
        delta_off=0.5,
        constant_prefix_time=True,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        M=2,
        t_switch=0.5,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    workspace = build_dp_workspace(params, config)

    with pytest.warns(RuntimeWarning, match="Falling back to q=delta_on/T"):
        with pytest.raises(ValueError, match="build a separate workspace"):
            solve_dp_workspace(
                workspace,
                variant="switching",
                horizon=1.75,
            )


def test_rollout_records_resolved_beta() -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        beta=2.75,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        t_switch=0.5,
    )
    result = solve_online_dp(params, config)
    rows = dynamic_programming.rollout_policy(result, State(0.0, 0.0, 0.0))
    online_rows = [
        row
        for row in rows
        if str(row.get("action", "")).startswith("online(")
    ]
    assert online_rows
    assert all(row["beta"] == pytest.approx(params.beta) for row in online_rows)


def test_optimal_temperature_three_factor_rollout_keeps_active_mask() -> None:
    """Resolving beta must not overwrite the rollout's boolean active mask."""

    params = Params(
        T=2,
        M=2,
        K=1,
        temperature_mode="optimal",
        eta_2f=0.0,
        eta_3f=1.0,
        fix_eta_2f_zero=True,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        M=2,
        t_switch=0.5,
    )
    result = solve_online_dp(params, config)
    rows = dynamic_programming.rollout_policy(result, State(0.0, 0.0, 0.0))
    online_rows = [
        row
        for row in rows
        if str(row.get("action", "")).startswith("online(")
    ]

    assert online_rows
    assert all(np.isfinite(row["beta"]) for row in online_rows)


def test_static_dp_caches_interpolations_and_matches_scalar_backups(
    monkeypatch,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        C_A=0.8,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        t_switch=0.5,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    original_prepare = dynamic_programming.FeasibleInterpolator.prepare
    original_backup = dynamic_programming._batched_backup
    calls = 0
    working_shapes = []

    def counted_prepare(self, queries):
        nonlocal calls
        calls += 1
        return original_prepare(self, queries)

    monkeypatch.setattr(
        dynamic_programming.FeasibleInterpolator,
        "prepare",
        counted_prepare,
    )

    def checked_backup(batch, layer, working_values, point_count):
        assert working_values.dtype == np.float64
        working_shapes.append(working_values.shape)
        return original_backup(batch, layer, working_values, point_count)

    monkeypatch.setattr(
        dynamic_programming,
        "_batched_backup",
        checked_backup,
    )
    result = solve_online_dp(params, config)

    assert result.params.C_A == 0.0
    assert result.values.dtype == np.float32
    assert calls == len(result.actions)
    expected_lookback = max(
        action.grid_steps
        for action in result.actions
        if action.grid_steps < len(result.times)
    )
    assert working_shapes
    assert {shape[1] for shape in working_shapes} == {expected_lookback + 1}

    reference_values = np.zeros(result.values.shape, dtype=np.float64)
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

    np.testing.assert_allclose(
        result.values,
        reference_values,
        rtol=6e-8,
        atol=6e-8,
    )
    np.testing.assert_array_equal(result.policy, reference_policy)

    applied_dtypes = []
    original_apply = dynamic_programming.Interpolation.apply

    def checked_apply(self, values):
        applied_dtypes.append(np.asarray(values).dtype)
        return original_apply(self, values)

    monkeypatch.setattr(
        dynamic_programming.Interpolation,
        "apply",
        checked_apply,
    )
    value_at(result, config.horizon, State(0.1, 0.5, 0.1))
    assert applied_dtypes[-1] == np.float64


def test_regular_grid_interpolation_masks_boundary_corners(monkeypatch) -> None:
    config = DPConfig(rho_points=5, rho_max=0.8)
    points = dynamic_programming._points(config)
    interpolator = dynamic_programming.FeasibleInterpolator(points, config)
    affine_values = (
        1.0
        + 2.0 * points[:, 0]
        - 3.0 * points[:, 1]
        + 0.5 * points[:, 2]
    )

    interior = np.array([[0.1, 0.12, 0.14]])
    regular = interpolator.prepare_grid_linear(interior)
    assert regular.indices.shape == (1, 8)
    assert np.count_nonzero(regular.weights[0]) == 8
    np.testing.assert_allclose(
        regular.apply(affine_values),
        1.0
        + 2.0 * interior[:, 0]
        - 3.0 * interior[:, 1]
        + 0.5 * interior[:, 2],
        atol=1e-14,
    )

    # This feasible point lies in a cell crossed by the determinant-zero
    # boundary. The structured method must mask the omitted corners without
    # calling the expensive Delaunay path.
    boundary = np.array([[0.02, 0.62, 0.62]])
    assert State(*boundary[0]).determinant > 0.0

    def fail_if_called(_queries):
        raise AssertionError("structured boundary interpolation used Delaunay")

    monkeypatch.setattr(interpolator, "prepare", fail_if_called)
    actual = interpolator.prepare_grid_linear(boundary)
    assert interpolator.triangulation is None
    assert actual.weights.dtype == np.float64
    axis = interpolator.grid_axis
    offsets = interpolator.grid_offsets
    spacing = axis[1] - axis[0]
    position = (boundary[0] - axis[0]) / spacing
    lower = np.clip(np.floor(position).astype(int), 0, len(axis) - 2)
    fractions = np.clip(position - lower, 0.0, 1.0)
    corners = lower[None, :] + offsets
    expected_indices = interpolator.grid_lookup[tuple(corners.T)]
    expected_weights = np.prod(
        np.where(offsets == 1, fractions[None, :], 1.0 - fractions[None, :]),
        axis=1,
    )
    feasible = expected_indices >= 0
    expected_weights *= feasible
    expected_weights /= np.sum(expected_weights)
    expected_indices[~feasible] = 0
    np.testing.assert_array_equal(actual.indices[0], expected_indices)
    np.testing.assert_allclose(actual.weights[0], expected_weights)
    np.testing.assert_allclose(np.sum(actual.weights, axis=1), 1.0)


@pytest.mark.parametrize(
    ("fixed_wm", "fixed_rho_me"),
    [
        (False, 1.0),
        (True, 0.4),
        (True, 1.0),
        (True, -1.0),
    ],
)
def test_vectorized_projection_matches_single_feasibility_projection(
    fixed_wm: bool,
    fixed_rho_me: float,
) -> None:
    config = DPConfig(
        fixed_wm=fixed_wm,
        fixed_rho_me=fixed_rho_me,
    )
    rng = np.random.default_rng(5)
    raw = rng.uniform(
        -1.4,
        1.4,
        size=(40, 2 if fixed_wm else 3),
    )
    clipped = np.clip(raw, -1.0, 1.0)
    if fixed_wm:
        a = clipped[:, 0]
        c = clipped[:, 1]
        denominator = a * a + c * c - 2.0 * fixed_rho_me * a * c
        radius = max(1.0 - fixed_rho_me**2, 0.0)
        outside = (denominator > radius) & (denominator > 0.0)
        scale = np.ones(len(raw))
        scale[outside] = np.sqrt(radius / denominator[outside])
        expected = clipped * scale[:, None]
    else:
        determinant = (
            1.0
            + 2.0 * np.prod(clipped, axis=1)
            - np.sum(clipped * clipped, axis=1)
        )
        expected = clipped.copy()
        outside = determinant < -1e-10
        expected[outside] = np.asarray(
            [
                State.from_array(point, project=True).as_array()
                for point in clipped[outside]
            ]
        )
    actual = dynamic_programming._project_points(raw, config)
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=0.0)


def test_projection_eigendecomposes_only_infeasible_rows(monkeypatch) -> None:
    config = DPConfig()
    points = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.9, 0.9, -0.9],
            [-0.2, 0.4, 0.1],
            [-0.95, 0.95, 0.95],
        ]
    )
    original = dynamic_programming._project_correlation_arrays
    projected_shapes = []

    def record(values):
        projected_shapes.append(values.shape)
        return original(values)

    monkeypatch.setattr(dynamic_programming, "_project_correlation_arrays", record)
    actual = dynamic_programming._project_points(points, config)

    assert projected_shapes == [(2, 3)]
    np.testing.assert_array_equal(actual[[0, 2]], points[[0, 2]])


def test_modulated_grid_interpolation_remains_close_to_delaunay_dp(
    monkeypatch,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        C_A=0.8,
        C_T=2.0,
        gh_order=8,
    )
    config = DPConfig(
        horizon=3.0,
        step_size=1.0,
        rho_points=7,
        rho_max=0.9,
        K_max=1,
        M=1,
        no_modulation=False,
    )
    with monkeypatch.context() as context:
        context.setattr(
            dynamic_programming.FeasibleInterpolator,
            "prepare_grid_linear",
            dynamic_programming.FeasibleInterpolator.prepare,
        )
        reference = solve_online_dp(params, config)
    regular = solve_online_dp(params, config)

    difference = np.abs(regular.values - reference.values)
    assert np.max(difference) < 0.05
    assert np.mean(difference) < 0.0025
    np.testing.assert_array_equal(regular.policy, reference.policy)


def test_modulated_transition_matches_at_grid_and_continuous_points() -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        C_A=0.5,
        C_T=2.0,
        fix_eta_2f_zero=True,
        fix_eta_3f_zero=True,
        fix_eta_wm_zero=True,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.5,
        step_size=1.0,
        rho_points=5,
        rho_max=0.8,
        K_max=1,
        M=1,
        no_modulation=False,
    )
    points = dynamic_programming._points(config)
    interpolator = dynamic_programming.FeasibleInterpolator(points, config)
    spec = dynamic_programming._action_specs(params, config, "online")[0]
    table = dynamic_programming._build_action_table(
        points,
        params,
        config,
        spec,
    )
    point_index = int(
        np.flatnonzero(
            np.all(np.isclose(points, np.array([0.4, 0.4, 0.4])), axis=1)
        )[0]
    )
    start_time = 0.3

    grid_next, grid_reward, grid_probability = (
        dynamic_programming._modulated_transition(
            table,
            points,
            params,
            config,
            start_time,
            interpolator,
            grid_aligned=True,
        )
    )
    point_next, point_reward, point_probability = (
        dynamic_programming._modulated_transition(
            table,
            points[point_index][None, :],
            params,
            config,
            start_time,
            interpolator,
        )
    )

    np.testing.assert_allclose(point_next[0], grid_next[point_index])
    np.testing.assert_allclose(point_reward[0], grid_reward[point_index])
    np.testing.assert_allclose(
        point_probability[0],
        grid_probability[point_index],
    )
    np.testing.assert_allclose(point_next[0], points[point_index])
    exposure = dynamic_programming.integrated_cycle(
        start_time,
        start_time + spec.duration,
        params,
    )
    action_cost = spec.duration / config.step_size
    assert point_reward[0] == pytest.approx(
        exposure * point_probability[0] / action_cost
    )


def test_modulated_auto_rollout_uses_bellman_transition(
    monkeypatch,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        C_A=0.5,
        C_T=2.0,
        fix_eta_2f_zero=True,
        fix_eta_3f_zero=True,
        fix_eta_wm_zero=True,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.5,
        step_size=1.0,
        rho_points=5,
        rho_max=0.8,
        K_max=1,
        M=1,
        no_modulation=False,
    )
    result = solve_online_dp(params, config)

    def reject_analytical_outcome(*args, **kwargs):
        raise AssertionError("modulated rollout used the analytical transition")

    monkeypatch.setattr(
        dynamic_programming,
        "_continuous_outcome",
        reject_analytical_outcome,
    )
    rows = dynamic_programming.rollout_policy(
        result,
        State(0.4, 0.4, 0.4),
    )

    assert rows[0]["action"] == "online(K=1,M=1)"
    assert rows[0]["reward_probability"] > 0.0


def test_nonmodulated_dp_does_not_use_modulated_transition(
    monkeypatch,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        fix_eta_2f_zero=True,
        fix_eta_3f_zero=True,
        fix_eta_wm_zero=True,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.5,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        M=1,
    )

    def reject_modulated_transition(*args, **kwargs):
        raise AssertionError("static DP called the modulated transition")

    monkeypatch.setattr(
        dynamic_programming,
        "_modulated_transition",
        reject_modulated_transition,
    )
    result = solve_online_dp(params, config)
    rows = dynamic_programming.rollout_policy(
        result,
        State(0.4, 0.4, 0.4),
    )

    assert rows[0]["action"] == "online(K=1,M=1)"


def test_switching_rollout_scores_actions_at_continuous_state() -> None:
    """Rollout decisions must not copy a contradictory nearest-grid policy."""

    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        t_switch=0.5,
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
    assert rows[0]["reward_probability"] == 0.0
    assert rows[0]["reward_rate"] == 0.0
    assert rows[1]["time"] - rows[0]["time"] == pytest.approx(
        result.config.t_switch
    )


def test_rollout_records_probability_and_rate_instead_of_scaled_reward() -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=1.25,
        step_size=0.5,
        rho_points=3,
        rho_max=0.8,
        K_max=1,
        M=1,
        fixed_wm=True,
        fixed_rho_me=0.4,
    )
    result = solve_online_dp(params, config)
    rows = dynamic_programming.rollout_policy(
        result,
        State(0.4, 0.4, 0.4),
    )
    first = rows[0]
    table = result.tables[0]
    point_index = int(
        np.flatnonzero(
            np.all(
                np.isclose(result.points, np.array([0.4, 0.4])),
                axis=1,
            )
        )[0]
    )
    expected_probability = (
        table.reward[point_index] / config.step_size
    )

    assert first["action"] == table.spec.name
    assert first["reward_probability"] == pytest.approx(expected_probability)
    assert first["reward_rate"] == pytest.approx(
        expected_probability
        / params.online_cost(table.spec.K, table.spec.M)
    )
    assert "immediate_reward" not in first


def test_batched_table_rollouts_match_analytical_reference(
    monkeypatch,
) -> None:
    params = Params(
        T=2,
        M=1,
        K=1,
        delta_on=0.5,
        delta_off=0.5,
        gh_order=8,
    )
    config = DPConfig(
        horizon=2.0,
        step_size=1.0,
        rho_points=5,
        rho_max=0.8,
        K_max=2,
        M=1,
        t_switch=0.5,
        fixed_wm=True,
        fixed_rho_me=0.5,
    )
    result = solve_switching_dp(params, config)
    states = [
        State(0.1, 0.5, 0.05),
        State(0.4, 0.5, 0.2),
        State(0.7, 0.5, 0.35),
    ]
    analytical = rollout_policies(
        result,
        states,
        outcome_method="analytical",
    )
    table = rollout_policies(result, states, outcome_method="table")

    assert [
        [row["action"] for row in rollout]
        for rollout in table
    ] == [
        [row["action"] for row in rollout]
        for rollout in analytical
    ]
    analytical_terminal = np.asarray(
        [
            [rollout[-1][name] for name in ("rho_se", "rho_me", "rho_sm")]
            for rollout in analytical
        ]
    )
    table_terminal = np.asarray(
        [
            [rollout[-1][name] for name in ("rho_se", "rho_me", "rho_sm")]
            for rollout in table
        ]
    )
    np.testing.assert_allclose(table_terminal, analytical_terminal, atol=0.02)

    def reject_analytical_outcome(*args, **kwargs):
        raise AssertionError("table rollout called the analytical dynamics")

    monkeypatch.setattr(
        dynamic_programming,
        "_continuous_outcome",
        reject_analytical_outcome,
    )
    automatic = rollout_policies(result, states)
    assert automatic == table
