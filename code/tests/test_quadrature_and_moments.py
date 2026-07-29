from math import asin, pi

import numpy as np

from learning_dynamics import Params, State
from learning_dynamics.moments import aggregate_online, compute_moments


def test_split_quadrature_recovers_sign_agreement() -> None:
    params = Params(T=3, M=2, K=3, gh_order=20)
    state = State(0.2, 0.55, -0.15)
    moments = compute_moments(state, params, params.beta)
    expected_me = 0.5 + asin(state.rho_me) / pi
    expected_sm = 0.5 + asin(state.rho_sm) / pi
    np.testing.assert_allclose(moments.A[0], 1.0, atol=2e-8)
    np.testing.assert_allclose(moments.P_me[0], expected_me, atol=3e-6)
    np.testing.assert_allclose(moments.P_sm[0], expected_sm, atol=3e-6)


def test_k1_selection_is_an_unmodified_sample() -> None:
    params = Params(T=4, M=2, K=1, gh_order=18)
    state = State(0.25, 0.4, 0.1)
    moments = compute_moments(state, params, params.beta)
    online = aggregate_online(moments, params)
    np.testing.assert_allclose(online.base.B, moments.sampled.B[0], atol=2e-7)
    np.testing.assert_allclose(online.base.C, moments.sampled.C[0], atol=2e-7)
    np.testing.assert_allclose(online.base.E, moments.sampled.E[0], atol=2e-7)
    np.testing.assert_allclose(online.gamma, moments.J[0] ** params.T, atol=2e-7)
    assert online.base.D == 1.0
    np.testing.assert_allclose(online.observed_reward.D, online.gamma, atol=1e-12)


def test_perfect_world_model_boundary_is_finite() -> None:
    params = Params(T=3, M=3, K=2, beta=float("inf"), gh_order=16)
    state = State(0.6, 1.0, 0.6)
    moments = compute_moments(state, params, params.beta)
    online = aggregate_online(moments, params)
    assert np.isfinite(online.gamma)
    assert 0.0 <= online.gamma <= 1.0


def test_moment_tables_use_float64() -> None:
    params = Params(T=4, M=2, K=5, gh_order=12)
    moments = compute_moments(State(0.2, 0.4, 0.1), params, params.beta)
    arrays = [moments.A, moments.P_me, moments.P_sm, moments.J]
    for tagged in (
        moments.wm,
        moments.sampled,
        moments.wm_me,
        moments.environment_correct,
        moments.greedy_sm,
    ):
        arrays.extend(
            (tagged.B, tagged.C, tagged.E, tagged.D, tagged.X)
        )
    assert all(array.dtype == np.float64 for array in arrays)
