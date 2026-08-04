import importlib
from dataclasses import replace

import numpy as np
import pytest

from learning_dynamics import Params, State, components
from learning_dynamics.modulation import integrated_cycle


dynamics = importlib.import_module("learning_dynamics.dynamics")
observables = importlib.import_module("learning_dynamics.observables")
simulation = importlib.import_module("learning_dynamics.simulation")


def test_parameter_names_costs_and_M_bound() -> None:
    params = Params(T=6, M=6, delta_on=0.25, delta_off=0.5)
    assert params.constant_prefix_time is False
    assert params.tau_on == 4.0
    assert params.tau_off == 2.0
    assert params.delta_element == pytest.approx(0.25 / 6)
    assert params.prefix_time(M=2) == pytest.approx(0.25 * 2 / 6)
    assert params.online_cost(K=3, M=2) == pytest.approx(1.25)
    constant_prefix = replace(params, constant_prefix_time=True)
    assert constant_prefix.prefix_time(M=2) == pytest.approx(0.25)
    assert constant_prefix.prefix_time(M=6) == pytest.approx(0.25)
    assert constant_prefix.online_cost(K=3, M=2) == pytest.approx(1.75)
    with pytest.raises(ValueError):
        Params(T=5, M=6).validate()


def test_zero_learning_reduces_to_environment_drift() -> None:
    params = Params(
        T=3,
        M=2,
        K=1,
        eta_2f=0.0,
        eta_3f=0.0,
        eta_wm=0.0,
        fix_eta_2f_zero=True,
        fix_eta_3f_zero=True,
        fix_eta_wm_zero=True,
        eta_e=0.2,
        gh_order=10,
    )
    state = State(0.3, 0.4, 0.1)
    result = components(state, params, "online")
    np.testing.assert_allclose(result.dot_rho_se, -0.2 * state.rho_se)
    np.testing.assert_allclose(result.dot_rho_me, -0.2 * state.rho_me)
    np.testing.assert_allclose(result.dot_rho_sm, 0.0, atol=1e-12)


def test_cycle_integral_over_a_period_is_one_period() -> None:
    params = Params(C_A=1.0, C_T=2.5, C_phi=0.37)
    np.testing.assert_allclose(integrated_cycle(1.2, 3.7, params), 2.5)


def test_record_reuses_one_online_policy_resolution(monkeypatch) -> None:
    params = Params(T=3, M=2, K=2, temperature_mode="optimal", gh_order=8)
    state = State(0.1, 0.3, 0.2)
    original = simulation.resolve_online_policy
    calls = 0

    def counted_resolution(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(simulation, "resolve_online_policy", counted_resolution)
    monkeypatch.setattr(dynamics, "resolve_online_policy", counted_resolution)
    monkeypatch.setattr(observables, "resolve_online_policy", counted_resolution)
    simulation._record(0.0, 0, "online", state, params)
    assert calls == 1


def test_greedy_training_does_not_force_greedy_planning_evaluation() -> None:
    state = State(0.3, 0.8, 0.2)
    params = Params(
        T=5,
        M=5,
        K=5,
        beta=2.0,
        temperature_mode="constant",
        greedy_training=True,
        greedy_evaluation=False,
        gh_order=12,
    )
    training = simulation.resolve_online_policy(state, params)
    assert training.params.K == 1
    assert np.isinf(training.beta)

    planning = observables.observables(
        state,
        params,
        resolved=training,
    )
    greedy = observables.observables(
        state,
        replace(params, greedy_evaluation=True),
        resolved=training,
    )

    assert planning["evaluation_policy"] == "planning"
    assert planning["evaluation_K"] == params.K
    assert planning["evaluation_beta"] == params.beta
    assert greedy["evaluation_policy"] == "greedy"
    assert greedy["evaluation_K"] == 1
    assert np.isinf(greedy["evaluation_beta"])
    assert planning["true_reward_probability"] != pytest.approx(
        greedy["true_reward_probability"]
    )
    assert planning["planning_environment_reward_probability"] == pytest.approx(
        greedy["planning_environment_reward_probability"]
    )


def test_individual_action_correctness_matches_greedy_agreement() -> None:
    state = State(0.4, 1.0, 0.4)
    params = Params(
        T=4,
        M=2,
        K=1,
        beta=float("inf"),
        temperature_mode="constant",
        gh_order=12,
    )

    values = observables.observables(state, params)
    expected = observables.agreement_probability(state.rho_se)

    assert values["prefix_action_correct_probability"] == pytest.approx(
        expected,
        abs=5e-7,
    )
    assert values["suffix_action_correct_probability"] == pytest.approx(
        expected,
        abs=5e-7,
    )
    assert values["individual_action_correct_probability"] == pytest.approx(
        expected,
        abs=5e-7,
    )


def test_recorded_actual_reward_rate_is_zero_offline() -> None:
    state = State(0.2, 0.4, 0.3)
    params = Params(T=3, M=3, K=2, beta=2.0, gh_order=8)

    online = simulation._record(0.0, 0, "online", state, params)
    offline = simulation._record(0.0, 1, "offline", state, params)

    assert online["actual_reward_rate"] == pytest.approx(online["reward_rate"])
    assert offline["actual_reward_rate"] == 0.0
    assert offline["evaluation_reward_rate"] == pytest.approx(
        offline["reward_rate"]
    )
    assert offline["evaluation_reward_rate"] > 0.0
