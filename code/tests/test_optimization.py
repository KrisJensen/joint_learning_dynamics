import numpy as np

from learning_dynamics import Params, State
from learning_dynamics import optimization
from learning_dynamics.optimization import (
    compute_planner_acceptance_moments,
    expected_planner_acceptance,
)
from learning_dynamics.quadrature import prepare_planner_acceptance_quadrature


def test_planner_acceptance_moments_only_depend_on_rho_sm() -> None:
    params = Params(T=5, M=3, K=7, gh_order=12)
    first = State(0.1, 0.2, 0.5)
    second = State(-0.3, 0.1, 0.5)

    first_value = expected_planner_acceptance(first, params, 2.4)
    second_value = expected_planner_acceptance(second, params, 2.4)
    np.testing.assert_allclose(first_value, second_value, atol=1e-14)

    rule = prepare_planner_acceptance_quadrature(first.rho_sm, 256)
    moments = compute_planner_acceptance_moments(0.0, params.K, rule)
    expected = 0.5 ** np.arange(params.K + 1)
    expected[0] = 1.0
    np.testing.assert_allclose(moments.A, expected, atol=2e-12)


def test_temperature_optimizer_finds_narrow_interior_maximum() -> None:
    state = State(0.0, 0.0, 0.5)
    params = Params(
        T=1,
        M=1,
        K=10,
        beta_max=20.0,
        beta_tol=1e-9,
        temperature_mode="optimal",
        gh_order=12,
    )

    beta, _ = optimization.optimal_temperature(state, params)
    np.testing.assert_allclose(beta, 0.1061263606, atol=2e-8)
    optimum = expected_planner_acceptance(state, params, beta)
    assert optimum > expected_planner_acceptance(state, params, 0.0)
    assert optimum > expected_planner_acceptance(state, params, params.beta_max)


def test_single_step_high_acceptance_avoids_binomial_cancellation() -> None:
    state = State(0.0, 0.0, 0.999)
    params = Params(
        T=1,
        M=1,
        K=30,
        beta_max=30.0,
        beta_tol=1e-9,
        temperature_mode="optimal",
        gh_order=12,
    )

    beta, _ = optimization.optimal_temperature(state, params)
    np.testing.assert_allclose(beta, 1.51544578, atol=3e-7)


def test_temperature_search_computes_full_moments_once(monkeypatch) -> None:
    state = State(0.1, 0.2, 0.5)
    params = Params(
        T=5,
        M=3,
        K=6,
        beta_max=20.0,
        temperature_mode="optimal",
        gh_order=12,
    )
    original = optimization.compute_moments
    calls = 0

    def counted_compute_moments(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(optimization, "compute_moments", counted_compute_moments)
    optimization.optimal_temperature(state, params)
    assert calls == 1
