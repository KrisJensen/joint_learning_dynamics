import numpy as np

from learning_dynamics.empirical import select_plan


def test_plan_tie_breaking_is_uniform() -> None:
    candidates = np.array(
        [
            [1.0, 1.0],
            [-1.0, 1.0],
            [1.0, -1.0],
        ]
    )
    labels = np.array([-1.0, -1.0])
    rng = np.random.default_rng(12)
    # With M=2 every candidate has zero reward, so all three are tied.
    counts = np.zeros(3, dtype=int)
    for _ in range(6000):
        _, _, selected = select_plan(candidates, labels, 2, rng)
        counts[selected] += 1
    np.testing.assert_allclose(counts / counts.sum(), np.full(3, 1 / 3), atol=0.025)
