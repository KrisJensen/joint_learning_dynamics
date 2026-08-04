import numpy as np
import pytest

from learning_dynamics import Params, State
from learning_dynamics import optimization
from learning_dynamics.optimization import (
    best_online_rates,
    compute_planner_acceptance_moments,
    expected_planner_acceptance,
)
from learning_dynamics.moments import Aggregates, OnlineAggregates
from learning_dynamics.quadrature import prepare_planner_acceptance_quadrature


def _online_aggregates(L0: float, LR: float, r: float) -> OnlineAggregates:
    base = Aggregates(B=L0, C=0.0, E=0.0, D=1.0, X=0.0)
    reward = Aggregates(B=LR, C=0.0, E=0.0, D=r, X=0.0)
    return OnlineAggregates(
        base=base,
        true_reward=reward,
        observed_reward=reward,
        gamma_prefix=r,
        gamma=r,
    )


def _rate_objective(
    eta_2f: float,
    eta_3f: float,
    *,
    rho: float,
    T: int,
    L0: float,
    LR: float,
    r: float,
) -> float:
    quadratic = eta_2f**2 + r * (
        2.0 * eta_2f * eta_3f + eta_3f**2
    )
    return eta_2f * L0 + eta_3f * LR - rho * quadratic / (2.0 * T)


def test_joint_online_rate_optimizer_uses_sum_cap() -> None:
    params = Params(T=1, learning_rate_mode="optimal", eta_max=1.0)
    state = State(1.0, 0.0, 0.0)
    online = _online_aggregates(L0=10.0, LR=10.0, r=0.5)

    eta_2f, eta_3f = best_online_rates(state, params, online)

    assert eta_2f + eta_3f == pytest.approx(params.eta_max)
    assert eta_2f == pytest.approx(0.0)
    assert eta_3f == pytest.approx(1.0)


def test_joint_online_rate_optimizer_recovers_interior_solution() -> None:
    rho = 0.5
    params = Params(T=2, learning_rate_mode="optimal", eta_max=2.0)
    state = State(rho, 0.0, 0.0)
    # These coefficients make s=0.8 and eta_2f=0.3 stationary.
    online = _online_aggregates(L0=0.125, LR=0.08, r=0.4)

    eta_2f, eta_3f = best_online_rates(state, params, online)

    np.testing.assert_allclose((eta_2f, eta_3f), (0.3, 0.5), atol=1e-13)


@pytest.mark.parametrize(
    ("fixed_name", "expected"),
    [
        ("fix_eta_2f_zero", (0.0, 1.0)),
        ("fix_eta_3f_zero", (1.0, 0.0)),
    ],
)
def test_fixed_zero_online_rate_reduces_to_scalar_cap(
    fixed_name: str,
    expected: tuple[float, float],
) -> None:
    params = Params(
        T=1,
        learning_rate_mode="optimal",
        eta_max=1.0,
        **{fixed_name: True},
    )
    state = State(0.5, 0.0, 0.0)
    online = _online_aggregates(L0=2.0, LR=2.0, r=0.5)

    np.testing.assert_allclose(
        best_online_rates(state, params, online),
        expected,
        atol=1e-14,
    )


def test_online_rate_candidates_cover_singular_and_nonconcave_cases() -> None:
    rng = np.random.default_rng(7)
    cases = [
        (0.0, 0.0),
        (0.0, 1.0),
        (-0.4, 0.0),
        (-0.4, 1.0),
        (-0.4, 0.37),
        (0.6, 0.0),
        (0.6, 1.0),
    ]
    cases.extend(
        (float(rng.uniform(-0.8, 0.9)), float(rng.uniform(0.0, 1.0)))
        for _ in range(8)
    )
    cap = 1.3
    axis = np.linspace(0.0, cap, 121)
    for rho, r in cases:
        L0, LR = rng.uniform(-1.5, 1.5, size=2)
        params = Params(T=3, learning_rate_mode="optimal", eta_max=cap)
        state = State(rho, 0.0, 0.0)
        online = _online_aggregates(L0=float(L0), LR=float(LR), r=r)
        optimum = best_online_rates(state, params, online)
        assert optimum[0] >= 0.0
        assert optimum[1] >= 0.0
        assert sum(optimum) <= cap + 1e-12
        optimum_value = _rate_objective(
            *optimum,
            rho=rho,
            T=params.T,
            L0=float(L0),
            LR=float(LR),
            r=r,
        )
        sampled_best = max(
            _rate_objective(
                eta_2f,
                eta_3f,
                rho=rho,
                T=params.T,
                L0=float(L0),
                LR=float(LR),
                r=r,
            )
            for eta_2f in axis
            for eta_3f in axis
            if eta_2f + eta_3f <= cap + 1e-12
        )
        assert optimum_value >= sampled_best - 1e-12


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
