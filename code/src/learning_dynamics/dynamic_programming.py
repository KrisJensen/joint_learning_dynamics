"""Finite-horizon dynamic programming on the feasible correlation domain."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Callable, Iterable
import warnings

import numpy as np
from scipy.spatial import Delaunay, cKDTree

from .config import Params, State
from .dynamics import components
from .modulation import integrated_cycle


@dataclass(frozen=True)
class DPConfig:
    """Discretization and objective for a report-style Bellman problem.

    ``M=None`` optimizes jointly over every prefix length in ``[1, T]``.
    Setting ``M`` fixes the prefix length while retaining optimization over
    ``K=1, ..., K_max``. A positive ``t_strategy_switch`` makes the active
    online ``(K, M)`` pair part of the Bellman state and selects the
    memory-efficient rolling solver. Its default value of zero retains the
    original free strategy selection and full float32 value history.
    """

    horizon: float = 10.0
    step_size: float = 1.0
    rho_points: int = 21
    rho_min: float = 0.0
    rho_max: float = 0.999
    K_max: int = 10
    M: int | None = None
    reward_kind: str = "true"
    t_switch: float = 0.1
    t_strategy_switch: float = 0.0
    record_compact_policy: bool = False
    no_modulation: bool = True
    fixed_wm: bool = False
    fixed_rho_me: float = 1.0

    def _time_quantum(self, params: Params) -> tuple[float, tuple[str, ...]]:
        """Return the largest supported exact DP cost quantum.

        A constant prefix duration removes the need to resolve the individual
        prefix elements.  In that case ``delta_on`` can replace
        ``delta_on / T`` provided every duration used by this Bellman problem
        still occupies an integer number of horizon layers.  Returning the
        incompatibilities separately lets :meth:`validate` issue one useful
        warning without making this frequently called helper noisy.
        """

        fine_quantum = params.delta_on / params.T
        if not params.constant_prefix_time:
            return fine_quantum, ()

        coarse_quantum = params.delta_on
        horizon_step = coarse_quantum * self.step_size
        layer_counts = {
            "unit online cost": 1.0 / coarse_quantum,
            "offline cost": params.delta_off / coarse_quantum,
            "mode-switch time": self.t_switch / horizon_step,
            "horizon": self.horizon / horizon_step,
        }
        if self.t_strategy_switch > 0.0:
            layer_counts["strategy-switch time"] = (
                self.t_strategy_switch / horizon_step
            )
        incompatible = tuple(
            f"{name} ({count:.12g} layers)"
            for name, count in layer_counts.items()
            if not np.isclose(count, round(count), atol=1e-9, rtol=0.0)
        )
        if incompatible:
            return fine_quantum, incompatible
        return coarse_quantum, ()

    def validate(self, params: Params, *, switching: bool = True) -> None:
        params.validate()
        if self.horizon <= 0.0 or self.step_size <= 0.0:
            raise ValueError("DP horizon and step_size must be positive")
        if self.rho_points < 3 or not -1.0 <= self.rho_min < self.rho_max <= 1.0:
            raise ValueError("invalid DP correlation grid")
        if self.K_max < 1:
            raise ValueError("DP K_max must be positive")
        if self.M is not None and (
            not isinstance(self.M, int) or not 1 <= self.M <= params.T
        ):
            raise ValueError("fixed DP M must be an integer in [1, T]")
        if self.reward_kind not in {"true", "observed"}:
            raise ValueError("reward_kind must be true or observed")
        if not np.isfinite(self.t_switch) or self.t_switch <= 0.0:
            raise ValueError("t_switch must be finite and positive")
        if (
            not np.isfinite(self.t_strategy_switch)
            or self.t_strategy_switch < 0.0
        ):
            raise ValueError("t_strategy_switch must be finite and nonnegative")
        if params.delta_on <= 0.0:
            raise ValueError("standard DP requires positive delta_on")
        q, coarse_grid_issues = self._time_quantum(params)
        if coarse_grid_issues:
            warnings.warn(
                "constant-prefix-time DP cannot use the coarse time quantum "
                "q=delta_on because these quantities have non-integer layer "
                "counts: "
                + ", ".join(coarse_grid_issues)
                + ". Falling back to q=delta_on/T; this remains exact but "
                "uses more Bellman layers.",
                RuntimeWarning,
                stacklevel=2,
            )
        ratios = {
            "unit action": 1.0 / q,
            "offline": params.delta_off / q,
        }
        for name, ratio in ratios.items():
            if not np.isclose(ratio, round(ratio), atol=1e-9, rtol=0.0):
                raise ValueError(f"{name} cost is not commensurate with delta_on/T")
        if switching:
            switch_steps = self.t_switch / self.horizon_step(params)
            if not np.isclose(
                switch_steps,
                round(switch_steps),
                atol=1e-9,
                rtol=0.0,
            ):
                raise ValueError(
                    "t_switch must be an integer multiple of the DP grid step"
                )
        if self.t_strategy_switch > 0.0:
            strategy_switch_steps = (
                self.t_strategy_switch / self.horizon_step(params)
            )
            if not np.isclose(
                strategy_switch_steps,
                round(strategy_switch_steps),
                atol=1e-9,
                rtol=0.0,
            ):
                raise ValueError(
                    "t_strategy_switch must be an integer multiple of the "
                    "DP grid step"
                )
        count = self.horizon / self.horizon_step(params)
        if not np.isclose(count, round(count), atol=1e-8, rtol=0.0):
            raise ValueError("horizon must be an integer multiple of the DP grid step")
        if self.fixed_wm:
            State(0.0, self.fixed_rho_me, 0.0).validate()

    def horizon_step(self, params: Params) -> float:
        """Return elapsed time represented by one Bellman layer."""

        quantum, _ = self._time_quantum(params)
        return quantum * self.step_size


@dataclass(frozen=True)
class Interpolation:
    indices: np.ndarray
    weights: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        return np.sum(values[self.indices] * self.weights, axis=1)


class FeasibleInterpolator:
    """Linear interpolation on the feasible correlation domain.

    General queries use a Delaunay triangulation. Modulated Bellman sweeps use
    :meth:`prepare_grid_linear`, which exploits the regular correlation grid.
    At the curved feasibility boundary, infeasible corners are discarded and
    the remaining multilinear weights are renormalized.
    """

    def __init__(self, points: np.ndarray, config: DPConfig | None = None):
        self.points = np.asarray(points, dtype=float)
        self.tree = cKDTree(self.points)
        centered_rank = np.linalg.matrix_rank(
            self.points - np.mean(self.points, axis=0, keepdims=True),
            tol=1e-12,
        )
        self._can_triangulate = (
            self.points.shape[1] > 1
            and centered_rank == self.points.shape[1]
            and len(self.points) > self.points.shape[1] + 1
        )
        self.triangulation = None
        self.grid_axis = None
        self.grid_lookup = None
        self.grid_offsets = None
        if config is not None:
            self._configure_regular_grid(config)

    def _configure_regular_grid(self, config: DPConfig) -> None:
        """Map regular-grid coordinates to compact feasible-point indices."""

        dimension = self.points.shape[1]
        expected_dimension = 2 if config.fixed_wm else 3
        if dimension != expected_dimension:
            raise ValueError("point dimension does not match the DP configuration")
        axis = np.linspace(
            config.rho_min,
            config.rho_max,
            config.rho_points,
            dtype=np.float64,
        )
        spacing = axis[1] - axis[0]
        grid_indices = np.rint((self.points - axis[0]) / spacing).astype(np.intp)
        if (
            np.any(grid_indices < 0)
            or np.any(grid_indices >= len(axis))
            or not np.allclose(
                axis[grid_indices],
                self.points,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise ValueError("points do not lie on the configured correlation grid")
        lookup = np.full(
            (len(axis),) * dimension,
            -1,
            dtype=np.intp,
        )
        lookup[tuple(grid_indices.T)] = np.arange(len(self.points), dtype=np.intp)
        self.grid_axis = axis
        self.grid_lookup = lookup
        vertex_count = 1 << dimension
        self.grid_offsets = (
            (
                np.arange(vertex_count, dtype=np.intp)[:, None]
                >> np.arange(dimension, dtype=np.intp)[None, :]
            )
            & 1
        )

    def prepare(self, queries: np.ndarray) -> Interpolation:
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        dimension = self.points.shape[1]
        indices = np.zeros((len(queries), dimension + 1), dtype=int)
        weights = np.zeros((len(queries), dimension + 1), dtype=float)
        if self.triangulation is None and self._can_triangulate:
            self.triangulation = Delaunay(self.points, qhull_options="QJ")
        if self.triangulation is not None:
            simplex = self.triangulation.find_simplex(queries)
            inside = simplex >= 0
            if np.any(inside):
                transform = self.triangulation.transform[simplex[inside]]
                delta = queries[inside] - transform[:, dimension]
                bary = np.einsum("nij,nj->ni", transform[:, :dimension], delta)
                bary = np.column_stack((bary, 1.0 - np.sum(bary, axis=1)))
                indices[inside] = self.triangulation.simplices[simplex[inside]]
                weights[inside] = bary
            outside = ~inside
        else:
            outside = np.ones(len(queries), dtype=bool)
        if np.any(outside):
            nearest = np.asarray(self.tree.query(queries[outside])[1], dtype=int)
            indices[outside, :] = nearest[:, None]
            weights[outside, 0] = 1.0
        return Interpolation(indices, weights)

    def prepare_grid_linear(self, queries: np.ndarray) -> Interpolation:
        """Prepare vectorized multilinear interpolation on the regular grid.

        A compact feasible grid omits lattice points outside the
        positive-semidefinite correlation domain. We give omitted corners zero
        weight and renormalize the remaining weights. This retains ordinary
        bilinear or trilinear interpolation in the interior without requiring
        an expensive spatial simplex search at the curved boundary.
        """

        queries = np.atleast_2d(np.asarray(queries, dtype=np.float64))
        if (
            self.grid_axis is None
            or self.grid_lookup is None
            or self.grid_offsets is None
        ):
            return self.prepare(queries)

        axis = self.grid_axis
        dimension = self.points.shape[1]
        vertex_count = 1 << dimension
        indices = np.zeros((len(queries), vertex_count), dtype=np.intp)
        weights = np.zeros((len(queries), vertex_count), dtype=np.float64)
        if len(queries) == 0:
            return Interpolation(indices, weights)

        clipped = np.clip(queries, axis[0], axis[-1])
        spacing = axis[1] - axis[0]
        positions = (clipped - axis[0]) / spacing
        lower = np.floor(positions).astype(np.intp)
        lower = np.clip(lower, 0, len(axis) - 2)
        fractions = np.clip(positions - lower, 0.0, 1.0)
        offsets = self.grid_offsets
        corner_grid = lower[:, None, :] + offsets[None, :, :]
        corner_indices = self.grid_lookup[
            tuple(corner_grid[:, :, coordinate] for coordinate in range(dimension))
        ]
        factors = np.where(
            offsets[None, :, :] == 1,
            fractions[:, None, :],
            1.0 - fractions[:, None, :],
        )
        weights = np.prod(factors, axis=2)
        feasible = corner_indices >= 0
        weights *= feasible
        totals = np.sum(weights, axis=1)
        usable = totals > 1e-14
        weights[usable] /= totals[usable, None]
        corner_indices[~feasible] = 0
        indices[:] = corner_indices

        if np.any(~usable):
            nearest = np.asarray(self.tree.query(clipped[~usable])[1], dtype=np.intp)
            indices[~usable] = np.repeat(
                nearest[:, None],
                vertex_count,
                axis=1,
            )
            weights[~usable] = 0.0
            weights[~usable, 0] = 1.0
        return Interpolation(indices, weights)


@dataclass(frozen=True)
class ActionSpec:
    name: str
    kind: str
    grid_steps: int
    duration: float
    K: int | None = None
    M: int | None = None


@dataclass(frozen=True)
class ActionTable:
    spec: ActionSpec
    delta: np.ndarray
    reward: np.ndarray
    vector_field: np.ndarray
    probability: np.ndarray


@dataclass(frozen=True)
class PreparedTransition:
    """Reward and state-interpolation data for one action."""

    reward: np.ndarray
    interpolation: Interpolation


@dataclass(frozen=True)
class ActionBatch:
    """Stacked action data for one vectorized Bellman backup."""

    action_indices: np.ndarray
    next_modes: np.ndarray
    grid_steps: np.ndarray
    rewards: np.ndarray
    interpolation_indices: np.ndarray
    interpolation_weights: np.ndarray


@dataclass(frozen=True)
class CompactPolicy:
    """Bit-packed policy history for a rolling strategy-switching solve.

    Each online strategy and the offline state use two bits per grid point and
    horizon layer. ``envelope_strategy`` records which online strategy attains
    the shared best-online envelope at each grid point.
    """

    strategy_decisions: np.ndarray
    offline_decisions: np.ndarray | None
    envelope_strategy: np.ndarray
    point_count: int

    @property
    def nbytes(self) -> int:
        """Return the in-memory size of the compact policy arrays."""

        return int(
            self.strategy_decisions.nbytes
            + self.envelope_strategy.nbytes
            + (
                0
                if self.offline_decisions is None
                else self.offline_decisions.nbytes
            )
        )


@dataclass
class DPResult:
    variant: str
    config: DPConfig
    params: Params
    times: np.ndarray
    points: np.ndarray
    values: np.ndarray | None
    policy: np.ndarray | None
    actions: tuple[ActionSpec, ...]
    mode_names: tuple[str, ...]
    interpolator: FeasibleInterpolator
    tables: tuple[ActionTable, ...]
    terminal_values: np.ndarray | None = None
    compact_policy: CompactPolicy | None = None


@dataclass
class DPWorkspace:
    """Reusable state grid and action transitions for related DP solves.

    Horizon and online/offline switching time are deliberately excluded from
    the reusable online/offline tables. Switch tables are cached separately by
    time. The online-strategy switching time is part of the workspace because
    its action is shared by every solve that uses these strategy tables.
    """

    config: DPConfig
    params: Params
    points: np.ndarray
    interpolator: FeasibleInterpolator
    tables: tuple[ActionTable, ...]
    prepared: dict[ActionSpec, PreparedTransition] | None
    switch_tables: dict[float, ActionTable] = field(default_factory=dict)
    switch_prepared: dict[float, PreparedTransition] = field(default_factory=dict)


def _project_state(state: State, config: DPConfig) -> State:
    projected = state.projected()
    if not config.fixed_wm:
        return projected
    a, c = projected.rho_se, projected.rho_sm
    b = config.fixed_rho_me
    if abs(1.0 - abs(b)) <= 1e-12:
        if b > 0.0:
            a = c = 0.5 * (a + c)
        else:
            a = 0.5 * (a - c)
            c = -a
        result = State(float(a), float(b), float(c))
        result.validate(tolerance=1e-8)
        return result
    denominator = a * a + c * c - 2.0 * b * a * c
    radius = max(1.0 - b * b, 0.0)
    if denominator > radius and denominator > 0.0:
        scale = np.sqrt(radius / denominator)
        a, c = a * scale, c * scale
    result = State(float(a), float(b), float(c))
    result.validate(tolerance=1e-8)
    return result


def _project_correlation_arrays(values: np.ndarray) -> np.ndarray:
    """Project many correlation triples using ``State.projected`` semantics."""

    values = np.atleast_2d(np.asarray(values, dtype=np.float64))
    if values.shape[1] != 3:
        raise ValueError("correlation arrays must contain three columns")
    matrices = np.empty((len(values), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0
    matrices[:, 1, 1] = 1.0
    matrices[:, 2, 2] = 1.0
    matrices[:, 0, 1] = matrices[:, 1, 0] = values[:, 0]
    matrices[:, 0, 2] = matrices[:, 2, 0] = values[:, 1]
    matrices[:, 1, 2] = matrices[:, 2, 1] = values[:, 2]
    eigenvalues, eigenvectors = np.linalg.eigh(matrices)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    positive = (eigenvectors * eigenvalues[:, None, :]) @ np.swapaxes(
        eigenvectors,
        1,
        2,
    )
    scales = np.sqrt(
        np.maximum(np.diagonal(positive, axis1=1, axis2=2), 1e-12)
    )
    correlation = positive / (scales[:, :, None] * scales[:, None, :])
    return np.clip(
        np.column_stack(
            (
                correlation[:, 0, 1],
                correlation[:, 0, 2],
                correlation[:, 1, 2],
            )
        ),
        -1.0,
        1.0,
    )


def _project_points(points: np.ndarray, config: DPConfig) -> np.ndarray:
    """Project timed-transition destinations onto the feasible state domain.

    Most Euler destinations are already feasible. Full three-correlation
    states therefore use the relatively expensive eigendecomposition only for
    rows whose determinant is negative. Fixed-world-model states have a
    two-dimensional elliptical boundary and can be projected directly.
    """

    points = np.clip(
        np.atleast_2d(np.asarray(points, dtype=np.float64)),
        -1.0,
        1.0,
    )
    if config.fixed_wm:
        if points.shape[1] != 2:
            raise ValueError("fixed-WM points must contain two columns")
        a = points[:, 0]
        c = points[:, 1]
        b = config.fixed_rho_me
        denominator = a * a + c * c - 2.0 * b * a * c
        radius = max(1.0 - b * b, 0.0)
        outside = (denominator > radius) & (denominator > 0.0)
        scale = np.ones(len(points), dtype=np.float64)
        scale[outside] = np.sqrt(radius / denominator[outside])
        return points * scale[:, None]

    if points.shape[1] != 3:
        raise ValueError("correlation points must contain three columns")
    determinant = (
        1.0
        + 2.0 * np.prod(points, axis=1)
        - np.sum(points * points, axis=1)
    )
    outside = determinant < -1e-10
    if not np.any(outside):
        return points
    projected = points.copy()
    projected[outside] = _project_correlation_arrays(points[outside])
    return projected


def _points(config: DPConfig) -> np.ndarray:
    axis = np.linspace(config.rho_min, config.rho_max, config.rho_points)
    if config.fixed_wm:
        values = []
        for rho_se in axis:
            for rho_sm in axis:
                state = State(float(rho_se), config.fixed_rho_me, float(rho_sm))
                if state.determinant >= -1e-10:
                    values.append((rho_se, rho_sm))
        return np.asarray(values, dtype=float)
    values = []
    for rho_se in axis:
        for rho_me in axis:
            for rho_sm in axis:
                state = State(float(rho_se), float(rho_me), float(rho_sm))
                if state.determinant >= -1e-10:
                    values.append((rho_se, rho_me, rho_sm))
    return np.asarray(values, dtype=float)


def _state_from_point(
    point,
    config: DPConfig,
    *,
    project: bool = False,
) -> State:
    if config.fixed_wm:
        state = State(float(point[0]), config.fixed_rho_me, float(point[1]))
        return state.projected() if project else state
    return State.from_array(point, project=project)


def _point_from_state(state: State, config: DPConfig) -> np.ndarray:
    if config.fixed_wm:
        return np.array([state.rho_se, state.rho_sm])
    return state.as_array()


def _active_params(params: Params, config: DPConfig) -> Params:
    if not config.fixed_wm:
        return params
    # A reduced fixed-WM problem represents an externally clamped rho_me.
    return replace(
        params,
        eta_wm=0.0,
        fix_eta_wm_zero=True,
        eta_e=0.0,
    )


def _action_specs(params: Params, config: DPConfig, variant: str) -> tuple[ActionSpec, ...]:
    q, _ = config._time_quantum(params)
    horizon_step = config.horizon_step(params)
    specs: list[ActionSpec] = []
    M_values = range(1, params.T + 1) if config.M is None else (config.M,)
    for K in range(1, config.K_max + 1):
        for M in M_values:
            steps = int(round(params.online_cost(K, M) / q))
            specs.append(
                ActionSpec(
                    f"online(K={K},M={M})",
                    "online",
                    steps,
                    steps * horizon_step,
                    K,
                    M,
                )
            )
    if config.t_strategy_switch > 0.0:
        specs.append(
            ActionSpec(
                "strategy_switch",
                "strategy_switch",
                int(
                    round(
                        config.t_strategy_switch
                        / horizon_step
                    )
                ),
                config.t_strategy_switch,
            )
        )
    if variant == "switching":
        specs.extend(
            [
                ActionSpec(
                    "offline",
                    "offline",
                    int(round(params.delta_off / q)),
                    params.delta_off * config.step_size,
                ),
                ActionSpec(
                    "switch",
                    "switch",
                    int(round(config.t_switch / config.horizon_step(params))),
                    config.t_switch,
                ),
            ]
        )
    return tuple(specs)


def _build_tables(
    points: np.ndarray,
    params: Params,
    config: DPConfig,
    variant: str,
    progress: Callable[[str], None] | None = None,
) -> tuple[ActionTable, ...]:
    tables = []
    specs = _action_specs(params, config, variant)
    for index, spec in enumerate(specs):
        tables.append(_build_action_table(points, params, config, spec))
        if progress is not None:
            progress(f"action table {index + 1}/{len(specs)}: {spec.name}")
    return tuple(tables)


def _build_action_table(
    points: np.ndarray,
    params: Params,
    config: DPConfig,
    spec: ActionSpec,
    *,
    training_without_planning: bool = False,
) -> ActionTable:
    """Build state increments and rewards for one action specification.

    ``training_without_planning`` is used only to construct the shared greedy
    learning transition for the reward-only planning counterfactual.  The
    ordinary action tables always use plan-conditioned learning.
    """

    active_base = replace(_active_params(params, config), C_A=0.0)
    deltas = []
    rewards = []
    vector_fields = []
    probabilities = []
    for point in points:
        state = _state_from_point(point, config)
        if spec.kind == "online":
            active = replace(
                active_base,
                K=spec.K,
                M=spec.M,
                K_mode="constant",
                M_mode="constant",
                greedy_training=training_without_planning,
            )
            result = components(state, active, "online", 0.0)
            raw = state.as_array() + spec.duration * np.array(
                [result.dot_rho_se, result.dot_rho_me, result.dot_rho_sm]
            )
            probability = (
                result.true_reward_probability
                if config.reward_kind == "true"
                else result.observed_reward_probability
            )
            reward = config.step_size * probability
        elif spec.kind == "offline":
            result = components(state, active_base, "offline", 0.0)
            raw = state.as_array() + spec.duration * np.array(
                [result.dot_rho_se, result.dot_rho_me, result.dot_rho_sm]
            )
            probability = 0.0
            reward = 0.0
        else:
            result = None
            retained = np.exp(-active_base.eta_e * spec.duration)
            raw = state.as_array()
            raw[:2] *= retained
            probability = 0.0
            reward = 0.0
        if result is None:
            vector_field = np.zeros(points.shape[1], dtype=np.float64)
        else:
            full_field = np.array(
                [result.dot_rho_se, result.dot_rho_me, result.dot_rho_sm],
                dtype=np.float64,
            )
            vector_field = (
                full_field[[0, 2]]
                if config.fixed_wm
                else full_field
            )
        next_state = _project_state(State.from_array(raw, project=True), config)
        deltas.append(_point_from_state(next_state, config) - point)
        rewards.append(reward)
        vector_fields.append(vector_field)
        probabilities.append(probability)
    return ActionTable(
        spec,
        np.asarray(deltas, dtype=np.float64),
        np.asarray(rewards, dtype=float),
        np.asarray(vector_fields, dtype=np.float64),
        np.asarray(probabilities, dtype=np.float64),
    )


def _interpolate_action_fields(
    table: ActionTable,
    points: np.ndarray,
    interpolator: FeasibleInterpolator,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate one action's vector field and reward probability."""

    interpolation = interpolator.prepare_grid_linear(points)
    fields = np.einsum(
        "nv,nvd->nd",
        interpolation.weights,
        table.vector_field[interpolation.indices],
        optimize=False,
    )
    probabilities = interpolation.apply(table.probability)
    return fields, probabilities


def _modulated_transition(
    table: ActionTable,
    points: np.ndarray,
    params: Params,
    config: DPConfig,
    start_time: float,
    interpolator: FeasibleInterpolator,
    *,
    grid_aligned: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Advance one modulated action with a shared midpoint approximation.

    Online dynamics have the form ``d rho / dt = C(t) f(rho)``. We therefore
    integrate them in exact cycle exposure, using a midpoint state to update
    the state-dependent optimal controls and reward probability. Offline
    environment components use cycle exposure while the student-WM component
    advances in ordinary elapsed time. Switch transitions retain their exact
    exponential solution.
    """

    points = np.atleast_2d(np.asarray(points, dtype=np.float64))
    spec = table.spec
    end_time = start_time + spec.duration
    if spec.kind in {"switch", "strategy_switch"}:
        exposure = integrated_cycle(start_time, end_time, params)
        retained = np.exp(-params.eta_e * exposure)
        next_points = points.copy()
        if config.fixed_wm:
            next_points[:, 0] *= retained
        else:
            next_points[:, :2] *= retained
        zeros = np.zeros(len(points), dtype=np.float64)
        return _project_points(next_points, config), zeros, zeros

    if grid_aligned:
        start_fields = table.vector_field
        start_probabilities = table.probability
        if len(start_fields) != len(points):
            raise ValueError("grid-aligned transition has the wrong point count")
    else:
        start_fields, start_probabilities = _interpolate_action_fields(
            table,
            points,
            interpolator,
        )

    midpoint_time = start_time + 0.5 * spec.duration
    half_exposure = integrated_cycle(start_time, midpoint_time, params)
    total_exposure = integrated_cycle(start_time, end_time, params)
    if spec.kind == "online":
        midpoint = points + half_exposure * start_fields
    else:
        half_scale = np.full(points.shape[1], half_exposure, dtype=np.float64)
        half_scale[-1] = 0.5 * spec.duration
        midpoint = points + start_fields * half_scale
    midpoint = _project_points(midpoint, config)
    midpoint_fields, midpoint_probabilities = _interpolate_action_fields(
        table,
        midpoint,
        interpolator,
    )

    if spec.kind == "online":
        next_points = points + total_exposure * midpoint_fields
        action_cost = spec.duration / config.step_size
        rewards = total_exposure * midpoint_probabilities / action_cost
    else:
        full_scale = np.full(points.shape[1], total_exposure, dtype=np.float64)
        full_scale[-1] = spec.duration
        next_points = points + midpoint_fields * full_scale
        rewards = np.zeros(len(points), dtype=np.float64)
    return (
        _project_points(next_points, config),
        np.asarray(rewards, dtype=np.float64),
        np.asarray(start_probabilities, dtype=np.float64),
    )


def _prepare_transition(
    table: ActionTable,
    points: np.ndarray,
    params: Params,
    config: DPConfig,
    start_time: float,
    interpolator: FeasibleInterpolator,
) -> PreparedTransition:
    """Prepare one action's reward and destination interpolation mapping."""

    if params.C_A != 0.0:
        next_points, reward, _ = _modulated_transition(
            table,
            points,
            params,
            config,
            start_time,
            interpolator,
            grid_aligned=True,
        )
    else:
        next_points = points + table.delta
        reward = table.reward
    prepare = (
        interpolator.prepare_grid_linear
        if params.C_A != 0.0
        else interpolator.prepare
    )
    return PreparedTransition(
        np.asarray(reward, dtype=float),
        prepare(next_points),
    )


def _action_batch(
    candidates: list[tuple[int, int]],
    tables: tuple[ActionTable, ...],
    prepared: dict[int, PreparedTransition],
) -> ActionBatch | None:
    """Stack prepared actions that lead to potentially different modes."""

    if not candidates:
        return None
    action_indices = np.asarray(
        [action_index for action_index, _ in candidates],
        dtype=np.intp,
    )
    return ActionBatch(
        action_indices=action_indices,
        next_modes=np.asarray(
            [next_mode for _, next_mode in candidates],
            dtype=np.intp,
        ),
        grid_steps=np.asarray(
            [tables[action_index].spec.grid_steps for action_index in action_indices],
            dtype=np.intp,
        ),
        rewards=np.stack(
            [prepared[action_index].reward for action_index in action_indices]
        ),
        interpolation_indices=np.stack(
            [
                prepared[action_index].interpolation.indices
                for action_index in action_indices
            ]
        ),
        interpolation_weights=np.stack(
            [
                prepared[action_index].interpolation.weights
                for action_index in action_indices
            ]
        ),
    )


def _batched_backup(
    batch: ActionBatch | None,
    layer: int,
    working_values: np.ndarray,
    point_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate actions from a circular float64 Bellman dependency buffer."""

    best = np.zeros(point_count, dtype=float)
    chosen = np.full(point_count, -1, dtype=np.int16)
    if batch is None:
        return best, chosen

    eligible = batch.grid_steps <= layer
    if not np.any(eligible):
        return best, chosen
    action_indices = batch.action_indices[eligible]
    next_modes = batch.next_modes[eligible]
    previous_layers = layer - batch.grid_steps[eligible]
    buffer_layers = np.remainder(previous_layers, working_values.shape[1])
    source_values = working_values[next_modes, buffer_layers]
    interpolation_indices = batch.interpolation_indices[eligible]
    gathered = source_values[
        np.arange(len(action_indices))[:, None, None],
        interpolation_indices,
    ]
    continuation = np.einsum(
        "anj,anj->an",
        gathered,
        batch.interpolation_weights[eligible],
        optimize=False,
    )
    candidates = batch.rewards[eligible] + continuation
    positions = np.argmax(candidates, axis=0)
    columns = np.arange(point_count)
    maxima = candidates[positions, columns]
    improve = maxima > 1e-14
    best[improve] = maxima[improve]
    chosen[improve] = action_indices[positions[improve]]
    return best, chosen


def build_dp_workspace(
    params: Params,
    config: DPConfig,
    *,
    include_offline: bool = True,
    progress: Callable[[str], None] | None = None,
) -> DPWorkspace:
    """Build reusable action transitions for a family of Bellman problems."""

    if config.no_modulation and params.C_A != 0.0:
        params = replace(params, C_A=0.0)
    config.validate(params, switching=include_offline)
    points = _points(config)
    interpolator = FeasibleInterpolator(points, config)
    variant = "switching" if include_offline else "online"
    all_tables = _build_tables(points, params, config, variant, progress)
    switch_tables = {
        float(config.t_switch): table
        for table in all_tables
        if table.spec.kind == "switch"
    }
    tables = tuple(table for table in all_tables if table.spec.kind != "switch")
    prepared = None
    switch_prepared = {}
    if params.C_A == 0.0:
        prepared = {
            table.spec: _prepare_transition(
                table,
                points,
                params,
                config,
                0.0,
                interpolator,
            )
            for table in tables
        }
        switch_prepared = {
            switch_time: _prepare_transition(
                table,
                points,
                params,
                config,
                0.0,
                interpolator,
            )
            for switch_time, table in switch_tables.items()
        }
    return DPWorkspace(
        config,
        params,
        points,
        interpolator,
        tables,
        prepared,
        switch_tables,
        switch_prepared,
    )


def _workspace_switch_table(
    workspace: DPWorkspace,
    t_switch: float,
) -> ActionTable:
    """Return a cached switch table for one elapsed switching time."""

    switch_time = float(t_switch)
    config = replace(workspace.config, t_switch=switch_time)
    config.validate(workspace.params)
    if switch_time not in workspace.switch_tables:
        spec = ActionSpec(
            "switch",
            "switch",
            int(round(switch_time / config.horizon_step(workspace.params))),
            switch_time,
        )
        table = _build_action_table(
            workspace.points,
            workspace.params,
            config,
            spec,
        )
        workspace.switch_tables[switch_time] = table
        if workspace.params.C_A == 0.0:
            workspace.switch_prepared[switch_time] = _prepare_transition(
                table,
                workspace.points,
                workspace.params,
                config,
                0.0,
                workspace.interpolator,
            )
    return workspace.switch_tables[switch_time]


def _rolling_length(layers: int, *lookbacks: int) -> int:
    """Return the smallest circular-buffer length for eligible lookbacks."""

    maximum = max(
        (lookback for lookback in lookbacks if lookback <= layers),
        default=0,
    )
    return int(maximum + 1)


_POLICY_STOP = np.uint8(0)
_POLICY_CONTINUE = np.uint8(1)
_POLICY_STRATEGY_SWITCH = np.uint8(2)
_POLICY_MODE_SWITCH = np.uint8(3)


def _packed_policy_width(point_count: int) -> int:
    """Return the bytes needed for one two-bit decision per grid point."""

    return (int(point_count) + 3) // 4


def _pack_policy_slice(decisions: np.ndarray, destination: np.ndarray) -> None:
    """Pack one uint8 decision vector into a preallocated two-bit slice."""

    decisions = np.asarray(decisions, dtype=np.uint8)
    expected = _packed_policy_width(len(decisions))
    if destination.shape != (expected,):
        raise ValueError("packed policy destination has the wrong shape")
    if np.any(decisions > 3):
        raise ValueError("two-bit policy decisions must lie in [0, 3]")
    destination.fill(0)
    for offset in range(4):
        values = decisions[offset::4]
        destination[: len(values)] |= values << np.uint8(2 * offset)


def _unpack_policy_decision(packed: np.ndarray, point_index: int) -> int:
    """Read one two-bit decision from a packed horizon slice."""

    point_index = int(point_index)
    byte = int(np.asarray(packed, dtype=np.uint8)[point_index // 4])
    return (byte >> (2 * (point_index % 4))) & 0b11


def _solve_strategy_switching_workspace(
    workspace: DPWorkspace,
    config: DPConfig,
    params: Params,
    online_tables: tuple[ActionTable, ...],
    *,
    variant: str,
    switch_table: ActionTable | None,
    progress: Callable[[str], None] | None,
) -> DPResult:
    """Solve positive-cost online-strategy switching with rolling values.

    Every online ``(K, M)`` strategy has its own continuation value. Entering
    the shared strategy-switch state consumes ``t_strategy_switch`` and then
    uses the best-online envelope, including the strategy that was active
    before the switch. Complete horizon histories are deliberately neither
    allocated nor retained in this branch.
    """

    strategy_switch_table = next(
        (
            table
            for table in workspace.tables
            if table.spec.kind == "strategy_switch"
        ),
        None,
    )
    if strategy_switch_table is None:
        raise RuntimeError(
            "positive t_strategy_switch requires a strategy-switch action"
        )

    offline_table = None
    if variant == "switching":
        offline_table = next(
            (
                table
                for table in workspace.tables
                if table.spec.kind == "offline"
            ),
            None,
        )
        if offline_table is None or switch_table is None:
            raise ValueError("switching solve requires offline and switch actions")

    tables_list = list(online_tables)
    strategy_switch_index = len(tables_list)
    tables_list.append(strategy_switch_table)
    offline_index = None
    switch_index = None
    if offline_table is not None and switch_table is not None:
        offline_index = len(tables_list)
        tables_list.append(offline_table)
        switch_index = len(tables_list)
        tables_list.append(switch_table)
    tables = tuple(tables_list)

    step = config.horizon_step(params)
    layers = int(round(config.horizon / step))
    times = np.arange(layers + 1, dtype=np.float64) * step
    point_count = len(workspace.points)
    strategy_buffers = [
        np.zeros(
            (
                _rolling_length(layers, table.spec.grid_steps),
                point_count,
            ),
            dtype=np.float64,
        )
        for table in online_tables
    ]
    envelope_buffer = np.zeros(
        (
            _rolling_length(
                layers,
                strategy_switch_table.spec.grid_steps,
                *(
                    (switch_table.spec.grid_steps,)
                    if switch_table is not None
                    else ()
                ),
            ),
            point_count,
        ),
        dtype=np.float64,
    )
    offline_buffer = None
    if offline_table is not None and switch_table is not None:
        offline_buffer = np.zeros(
            (
                _rolling_length(
                    layers,
                    offline_table.spec.grid_steps,
                    switch_table.spec.grid_steps,
                ),
                point_count,
            ),
            dtype=np.float64,
        )

    strategy_policy = None
    offline_policy = None
    envelope_strategy = None
    if config.record_compact_policy:
        packed_width = _packed_policy_width(point_count)
        strategy_policy = np.zeros(
            (len(online_tables), layers + 1, packed_width),
            dtype=np.uint8,
        )
        if variant == "switching":
            offline_policy = np.zeros(
                (layers + 1, packed_width),
                dtype=np.uint8,
            )
        if len(online_tables) <= 256:
            strategy_dtype = np.uint8
        elif len(online_tables) <= 65_536:
            strategy_dtype = np.uint16
        else:
            strategy_dtype = np.uint32
        envelope_strategy = np.zeros(
            (layers + 1, point_count),
            dtype=strategy_dtype,
        )
        if progress is not None:
            policy_bytes = (
                strategy_policy.nbytes
                + envelope_strategy.nbytes
                + (0 if offline_policy is None else offline_policy.nbytes)
            )
            progress(
                "Compact policy storage: "
                f"{policy_bytes / (1024.0**3):.3f} GiB"
            )

    static = params.C_A == 0.0
    static_prepared: dict[int, PreparedTransition] | None = None
    if static:
        if workspace.prepared is None:
            raise RuntimeError("static workspace is missing prepared transitions")
        static_prepared = {}
        for action_index, table in enumerate(tables):
            if table.spec.kind == "switch":
                static_prepared[action_index] = workspace.switch_prepared[
                    float(config.t_switch)
                ]
            else:
                static_prepared[action_index] = workspace.prepared[table.spec]

    terminal_envelope = np.zeros(point_count, dtype=np.float64)
    terminal_offline = np.zeros(point_count, dtype=np.float64)
    for layer in range(1, layers + 1):
        if static_prepared is not None:
            prepared = static_prepared
        else:
            start_time = config.horizon - layer * step
            prepared = {
                action_index: _prepare_transition(
                    table,
                    workspace.points,
                    params,
                    config,
                    start_time,
                    workspace.interpolator,
                )
                for action_index, table in enumerate(tables)
                if table.spec.grid_steps <= layer
            }

        strategy_switch_value = None
        strategy_steps = strategy_switch_table.spec.grid_steps
        if strategy_steps <= layer:
            source = envelope_buffer[
                (layer - strategy_steps) % len(envelope_buffer)
            ]
            strategy_switch_value = prepared[
                strategy_switch_index
            ].interpolation.apply(source)

        offline_switch_value = None
        if (
            switch_table is not None
            and switch_index is not None
            and offline_buffer is not None
            and switch_table.spec.grid_steps <= layer
        ):
            source = offline_buffer[
                (layer - switch_table.spec.grid_steps) % len(offline_buffer)
            ]
            offline_switch_value = prepared[switch_index].interpolation.apply(
                source
            )

        current_envelope = np.zeros(point_count, dtype=np.float64)
        current_strategy = (
            np.zeros(point_count, dtype=envelope_strategy.dtype)
            if envelope_strategy is not None
            else None
        )
        for strategy_index, (table, buffer) in enumerate(
            zip(online_tables, strategy_buffers)
        ):
            best = np.zeros(point_count, dtype=np.float64)
            decisions = (
                np.zeros(point_count, dtype=np.uint8)
                if strategy_policy is not None
                else None
            )
            action_steps = table.spec.grid_steps
            if action_steps <= layer:
                source = buffer[(layer - action_steps) % len(buffer)]
                transition = prepared[strategy_index]
                candidate = transition.reward + transition.interpolation.apply(
                    source
                )
                if decisions is None:
                    np.maximum(best, candidate, out=best)
                else:
                    improve = candidate > best
                    best[improve] = candidate[improve]
                    decisions[improve] = _POLICY_CONTINUE
            if strategy_switch_value is not None:
                if decisions is None:
                    np.maximum(best, strategy_switch_value, out=best)
                else:
                    improve = strategy_switch_value > best
                    best[improve] = strategy_switch_value[improve]
                    decisions[improve] = _POLICY_STRATEGY_SWITCH
            if offline_switch_value is not None:
                if decisions is None:
                    np.maximum(best, offline_switch_value, out=best)
                else:
                    improve = offline_switch_value > best
                    best[improve] = offline_switch_value[improve]
                    decisions[improve] = _POLICY_MODE_SWITCH
            buffer[layer % len(buffer)] = best
            if decisions is not None and strategy_policy is not None:
                _pack_policy_slice(
                    decisions,
                    strategy_policy[strategy_index, layer],
                )
            if current_strategy is None:
                np.maximum(current_envelope, best, out=current_envelope)
            else:
                improve = best > current_envelope
                current_envelope[improve] = best[improve]
                current_strategy[improve] = strategy_index

        envelope_buffer[layer % len(envelope_buffer)] = current_envelope
        if envelope_strategy is not None and current_strategy is not None:
            envelope_strategy[layer] = current_strategy
        terminal_envelope = current_envelope

        if (
            offline_table is not None
            and offline_index is not None
            and switch_table is not None
            and switch_index is not None
            and offline_buffer is not None
        ):
            best_offline = np.zeros(point_count, dtype=np.float64)
            offline_decisions = (
                np.zeros(point_count, dtype=np.uint8)
                if offline_policy is not None
                else None
            )
            if offline_table.spec.grid_steps <= layer:
                source = offline_buffer[
                    (layer - offline_table.spec.grid_steps) % len(offline_buffer)
                ]
                transition = prepared[offline_index]
                candidate = transition.reward + transition.interpolation.apply(
                    source
                )
                if offline_decisions is None:
                    np.maximum(best_offline, candidate, out=best_offline)
                else:
                    improve = candidate > best_offline
                    best_offline[improve] = candidate[improve]
                    offline_decisions[improve] = _POLICY_CONTINUE
            if switch_table.spec.grid_steps <= layer:
                source = envelope_buffer[
                    (layer - switch_table.spec.grid_steps)
                    % len(envelope_buffer)
                ]
                candidate = prepared[switch_index].interpolation.apply(source)
                if offline_decisions is None:
                    np.maximum(best_offline, candidate, out=best_offline)
                else:
                    improve = candidate > best_offline
                    best_offline[improve] = candidate[improve]
                    offline_decisions[improve] = _POLICY_MODE_SWITCH
            offline_buffer[layer % len(offline_buffer)] = best_offline
            if offline_decisions is not None and offline_policy is not None:
                _pack_policy_slice(
                    offline_decisions,
                    offline_policy[layer],
                )
            terminal_offline = best_offline

        if progress is not None and (
            layer == layers or layer % max(1, layers // 10) == 0
        ):
            progress(f"Bellman layers {layer}/{layers}")

    mode_names = ("on",) if variant == "online" else ("on", "off")
    terminal_values = (
        terminal_envelope[None, :]
        if variant == "online"
        else np.stack((terminal_envelope, terminal_offline))
    )
    compact_policy = None
    if strategy_policy is not None and envelope_strategy is not None:
        compact_policy = CompactPolicy(
            strategy_decisions=strategy_policy,
            offline_decisions=offline_policy,
            envelope_strategy=envelope_strategy,
            point_count=point_count,
        )
    return DPResult(
        variant=variant,
        config=config,
        params=params,
        times=times,
        points=workspace.points,
        values=None,
        policy=None,
        actions=tuple(table.spec for table in tables),
        mode_names=mode_names,
        interpolator=workspace.interpolator,
        tables=tables,
        terminal_values=np.asarray(terminal_values, dtype=np.float64),
        compact_policy=compact_policy,
    )


def solve_dp_workspace(
    workspace: DPWorkspace,
    *,
    variant: str = "online",
    horizon: float | None = None,
    t_switch: float | None = None,
    allowed_K: tuple[int, ...] | None = None,
    progress: Callable[[str], None] | None = None,
) -> DPResult:
    """Solve a Bellman problem using precomputed action transitions.

    ``allowed_K`` restricts online actions without rebuilding their tables.
    A separate Bellman sweep is still required for each switching time, but
    all online/offline dynamics and static interpolation data are shared.
    """

    if variant not in {"online", "switching"}:
        raise ValueError("variant must be online or switching")
    active_switch_time = (
        workspace.config.t_switch if t_switch is None else float(t_switch)
    )
    config = replace(
        workspace.config,
        horizon=workspace.config.horizon if horizon is None else float(horizon),
        t_switch=active_switch_time,
    )
    params = workspace.params
    config.validate(params, switching=variant == "switching")
    workspace_step = workspace.config.horizon_step(params)
    requested_step = config.horizon_step(params)
    if not np.isclose(
        requested_step,
        workspace_step,
        atol=1e-15,
        rtol=1e-12,
    ):
        raise ValueError(
            "the requested horizon or switch time changes the DP time "
            "quantum; build a separate workspace for this configuration"
        )
    switch_table = None
    if variant == "switching":
        switch_table = _workspace_switch_table(
            workspace,
            config.t_switch,
        )
    allowed = None if allowed_K is None else {int(K) for K in allowed_K}
    online_tables = tuple(
        table
        for table in workspace.tables
        if table.spec.kind == "online"
        and (allowed is None or table.spec.K in allowed)
    )
    if not online_tables:
        raise ValueError("the workspace contains no eligible online actions")
    if config.t_strategy_switch > 0.0:
        return _solve_strategy_switching_workspace(
            workspace,
            config,
            params,
            online_tables,
            variant=variant,
            switch_table=switch_table,
            progress=progress,
        )
    if variant == "online":
        tables = online_tables
    else:
        offline_table = next(
            (
                table
                for table in workspace.tables
                if table.spec.kind == "offline"
            ),
            None,
        )
        if offline_table is None or switch_table is None:
            raise ValueError("switching solve requires offline and switch actions")
        tables = online_tables + (offline_table, switch_table)

    step = config.horizon_step(params)
    layers = int(round(config.horizon / step))
    times = np.arange(layers + 1) * step
    modes = ("on",) if variant == "online" else ("on", "off")
    # Keep the complete history compact for later value queries and rollouts.
    # Bellman backups never read this archive; they use the float64 circular
    # dependency buffer below, so archival quantization cannot accumulate.
    values = np.zeros(
        (len(modes), layers + 1, len(workspace.points)),
        dtype=np.float32,
    )
    policy = np.full(
        (len(modes), layers + 1, len(workspace.points)),
        -1,
        dtype=np.int16,
    )

    online_indices = [
        index for index, table in enumerate(tables) if table.spec.kind == "online"
    ]
    offline_index = next(
        (index for index, table in enumerate(tables) if table.spec.kind == "offline"),
        None,
    )
    switch_index = next(
        (index for index, table in enumerate(tables) if table.spec.kind == "switch"),
        None,
    )
    on_candidates = [(action_index, 0) for action_index in online_indices]
    if variant == "switching" and switch_index is not None:
        on_candidates.append((switch_index, 1))
    off_candidates = []
    if variant == "switching":
        if offline_index is not None:
            off_candidates.append((offline_index, 1))
        if switch_index is not None:
            off_candidates.append((switch_index, 0))

    # With no cycle modulation, reuse the workspace's prepared interpolation
    # data rather than reconstructing it for each related Bellman problem.
    static = params.C_A == 0.0
    static_on_batch = None
    static_off_batch = None
    if static:
        if workspace.prepared is None:
            raise RuntimeError("static workspace is missing prepared transitions")
        prepared = {}
        for action_index, table in enumerate(tables):
            if table.spec.kind == "switch":
                prepared[action_index] = workspace.switch_prepared[
                    float(config.t_switch)
                ]
            else:
                prepared[action_index] = workspace.prepared[table.spec]
        static_on_batch = _action_batch(on_candidates, tables, prepared)
        static_off_batch = _action_batch(off_candidates, tables, prepared)

    maximum_lookback = max(
        (
            table.spec.grid_steps
            for table in tables
            if table.spec.grid_steps <= layers
        ),
        default=0,
    )
    working_values = np.zeros(
        (len(modes), maximum_lookback + 1, len(workspace.points)),
        dtype=np.float64,
    )

    for layer in range(1, layers + 1):
        if static:
            on_batch = static_on_batch
            off_batch = static_off_batch
        else:
            start_time = config.horizon - layer * step
            eligible_candidates = [
                candidate
                for candidate in on_candidates + off_candidates
                if tables[candidate[0]].spec.grid_steps <= layer
            ]
            needed = {action_index for action_index, _ in eligible_candidates}
            prepared = {
                action_index: _prepare_transition(
                    tables[action_index],
                    workspace.points,
                    params,
                    config,
                    start_time,
                    workspace.interpolator,
                )
                for action_index in needed
            }
            on_batch = _action_batch(
                [
                    candidate
                    for candidate in on_candidates
                    if candidate[0] in prepared
                ],
                tables,
                prepared,
            )
            off_batch = _action_batch(
                [
                    candidate
                    for candidate in off_candidates
                    if candidate[0] in prepared
                ],
                tables,
                prepared,
            )

        best_on, chosen_on = _batched_backup(
            on_batch,
            layer,
            working_values,
            len(workspace.points),
        )

        best_off = None
        chosen_off = None
        if variant == "switching":
            best_off, chosen_off = _batched_backup(
                off_batch,
                layer,
                working_values,
                len(workspace.points),
            )

        buffer_layer = layer % working_values.shape[1]
        working_values[0, buffer_layer] = best_on
        values[0, layer] = best_on
        policy[0, layer] = chosen_on
        if best_off is not None and chosen_off is not None:
            working_values[1, buffer_layer] = best_off
            values[1, layer] = best_off
            policy[1, layer] = chosen_off
        if progress is not None and (
            layer == layers or layer % max(1, layers // 10) == 0
        ):
            progress(f"Bellman layers {layer}/{layers}")
    return DPResult(
        variant,
        config,
        params,
        times,
        workspace.points,
        values,
        policy,
        tuple(table.spec for table in tables),
        modes,
        workspace.interpolator,
        tables,
        np.asarray(values[:, -1], dtype=np.float32),
    )


def solve_dp(
    params: Params,
    config: DPConfig,
    *,
    variant: str = "online",
    progress: Callable[[str], None] | None = None,
) -> DPResult:
    """Build a workspace and solve one Bellman problem."""

    if variant not in {"online", "switching"}:
        raise ValueError("variant must be online or switching")
    workspace = build_dp_workspace(
        params,
        config,
        include_offline=variant == "switching",
        progress=progress,
    )
    return solve_dp_workspace(
        workspace,
        variant=variant,
        horizon=config.horizon,
        t_switch=config.t_switch,
        progress=progress,
    )


def solve_online_dp(params: Params, config: DPConfig, progress=None) -> DPResult:
    return solve_dp(params, config, variant="online", progress=progress)


def solve_reward_only_planning_dp(
    reference: DPResult,
    progress: Callable[[str], None] | None = None,
) -> DPResult:
    """Solve an online DP in which planning cannot affect learning.

    Planning retains its action-specific immediate reward and elapsed cost.
    Every online action instead uses the same state transition produced by one
    greedy-training update (``K=1``, ``M=1``, and infinite inverse
    temperature).  Consequently future value can influence the choice of K,
    but only through remaining time and subsequent rewards—not through a
    plan-conditioned improvement in the learner or world model.

    The saved action rewards and state grid in ``reference`` are reused.  Since
    all counterfactual actions share one transition, actions with the same time
    cost are reduced to their pointwise best immediate reward before each
    Bellman sweep.  This grouping is exact and substantially reduces work.
    """

    if reference.params.C_A != 0.0:
        raise ValueError("reward-only planning DP currently requires no modulation")
    if reference.config.t_strategy_switch != 0.0:
        raise ValueError(
            "reward-only planning DP currently requires zero strategy-switch cost"
        )
    params = reference.params
    config = reference.config
    config.validate(params, switching=False)
    online_tables = tuple(
        table for table in reference.tables if table.spec.kind == "online"
    )
    if not online_tables:
        raise ValueError("reference result contains no online action tables")
    baseline_spec = next(
        (
            table.spec
            for table in online_tables
            if table.spec.K == 1 and table.spec.M == 1
        ),
        None,
    )
    if baseline_spec is None:
        raise ValueError("reference result is missing the K=1, M=1 action")

    if progress is not None:
        progress("Building shared greedy-training transition")
    greedy_table = _build_action_table(
        reference.points,
        params,
        config,
        baseline_spec,
        training_without_planning=True,
    )
    shared_transition = _prepare_transition(
        greedy_table,
        reference.points,
        params,
        config,
        0.0,
        reference.interpolator,
    ).interpolation
    tables = tuple(
        ActionTable(
            table.spec,
            greedy_table.delta,
            table.reward,
            greedy_table.vector_field,
            table.probability,
        )
        for table in online_tables
    )

    # Actions with equal duration also have equal counterfactual continuation
    # values. Keep only their pointwise best immediate reward while retaining
    # the winning original action index for policy reconstruction.
    unique_steps = np.unique(
        [table.spec.grid_steps for table in tables]
    ).astype(np.intp)
    point_count = len(reference.points)
    grouped_rewards = np.empty((len(unique_steps), point_count), dtype=np.float64)
    grouped_actions = np.empty((len(unique_steps), point_count), dtype=np.int16)
    for group_index, grid_steps in enumerate(unique_steps):
        action_indices = np.asarray(
            [
                index
                for index, table in enumerate(tables)
                if table.spec.grid_steps == grid_steps
            ],
            dtype=np.intp,
        )
        rewards = np.stack([tables[index].reward for index in action_indices])
        winners = np.argmax(rewards, axis=0)
        grouped_rewards[group_index] = rewards[winners, np.arange(point_count)]
        grouped_actions[group_index] = action_indices[winners]

    step = config.horizon_step(params)
    layers = int(round(config.horizon / step))
    times = np.arange(layers + 1, dtype=np.float64) * step
    values = np.zeros((1, layers + 1, point_count), dtype=np.float32)
    policy = np.full((1, layers + 1, point_count), -1, dtype=np.int16)
    working_values = np.zeros(
        (int(np.max(unique_steps)) + 1, point_count),
        dtype=np.float64,
    )
    point_indices = np.arange(point_count)

    for layer in range(1, layers + 1):
        eligible = unique_steps <= layer
        if np.any(eligible):
            active_steps = unique_steps[eligible]
            buffer_layers = np.remainder(
                layer - active_steps,
                len(working_values),
            )
            sources = working_values[buffer_layers]
            gathered = sources[:, shared_transition.indices]
            continuations = np.einsum(
                "gnv,nv->gn",
                gathered,
                shared_transition.weights,
                optimize=False,
            )
            scores = grouped_rewards[eligible] + continuations
            winning_groups = np.argmax(scores, axis=0)
            best = scores[winning_groups, point_indices]
            eligible_actions = grouped_actions[eligible]
            chosen = eligible_actions[winning_groups, point_indices]
        else:
            best = np.zeros(point_count, dtype=np.float64)
            chosen = np.full(point_count, -1, dtype=np.int16)

        buffer_layer = layer % len(working_values)
        working_values[buffer_layer] = best
        values[0, layer] = best
        policy[0, layer] = chosen
        if progress is not None and (
            layer == layers or layer % max(1, layers // 10) == 0
        ):
            progress(f"Reward-only Bellman layers {layer}/{layers}")

    return DPResult(
        variant="online",
        config=config,
        params=params,
        times=times,
        points=reference.points,
        values=values,
        policy=policy,
        actions=tuple(table.spec for table in tables),
        mode_names=("on",),
        interpolator=reference.interpolator,
        tables=tables,
        terminal_values=np.asarray(values[:, -1], dtype=np.float32),
    )


def solve_switching_dp(params: Params, config: DPConfig, progress=None) -> DPResult:
    return solve_dp(params, config, variant="switching", progress=progress)


def value_at(
    result: DPResult,
    horizon: float,
    state: State,
    mode: str = "on",
) -> float:
    """Interpolate a value table at a continuous correlation state."""

    layer = int(round(np.clip(horizon, 0.0, result.config.horizon) / result.times[1]))
    mode_index = result.mode_names.index(mode)
    point = _point_from_state(_project_state(state, result.config), result.config)
    prepare = (
        result.interpolator.prepare_grid_linear
        if result.params.C_A != 0.0
        else result.interpolator.prepare
    )
    weights = prepare(point[None, :])
    if result.values is None:
        final_layer = len(result.times) - 1
        if layer != final_layer or result.terminal_values is None:
            raise ValueError(
                "this rolling DP result stores only final-horizon values"
            )
        value_slice = np.asarray(
            result.terminal_values[mode_index],
            dtype=np.float64,
        )
    else:
        value_slice = np.asarray(
            result.values[mode_index, layer],
            dtype=np.float64,
        )
    return float(weights.apply(value_slice)[0])


def _continuous_outcome(
    state: State,
    spec: ActionSpec,
    params: Params,
    config: DPConfig,
    time: float,
) -> tuple[State, float, float]:
    """Return the next state, objective reward, and raw reward probability."""

    if spec.kind in {"switch", "strategy_switch"}:
        active = _active_params(params, config)
        exposure = integrated_cycle(time, time + spec.duration, active)
        retained = np.exp(-active.eta_e * exposure)
        raw = state.as_array()
        raw[:2] *= retained
        return (
            _project_state(State.from_array(raw, project=True), config),
            0.0,
            0.0,
        )
    active = _active_params(params, config)
    if spec.kind == "online":
        active = replace(
            active,
            K=spec.K,
            M=spec.M,
            K_mode="constant",
            M_mode="constant",
            greedy_training=False,
        )
        result = components(state, active, "online", time)
        probability = (
            result.true_reward_probability
            if config.reward_kind == "true"
            else result.observed_reward_probability
        )
        reward = config.step_size * probability
    else:
        result = components(state, active, "offline", time)
        reward = 0.0
        probability = 0.0
    raw = state.as_array() + spec.duration * np.array(
        [result.dot_rho_se, result.dot_rho_me, result.dot_rho_sm]
    )
    return (
        _project_state(State.from_array(raw, project=True), config),
        reward,
        float(probability),
    )


def _rollout_layer(result: DPResult, horizon: float | None) -> int:
    """Validate a rollout horizon and return its Bellman-layer index."""

    active_horizon = result.config.horizon if horizon is None else float(horizon)
    if not 0.0 < active_horizon <= result.config.horizon:
        raise ValueError("rollout horizon must lie in (0, result horizon]")
    layer_position = active_horizon / result.times[1]
    if not np.isclose(layer_position, round(layer_position), atol=1e-8, rtol=0.0):
        raise ValueError("rollout horizon must lie on the Bellman time grid")
    return int(round(layer_position))


def _rollout_outcome_method(result: DPResult, method: str) -> str:
    """Resolve the requested continuous-action approximation."""

    if method not in {"auto", "table", "analytical"}:
        raise ValueError(
            "outcome_method must be auto, table, or analytical"
        )
    if method == "auto":
        return "table"
    return method


def _table_candidate_outcomes(
    result: DPResult,
    states: list[State],
    trajectory_indices: np.ndarray,
    action_indices: np.ndarray,
) -> tuple[list[State], np.ndarray, np.ndarray]:
    """Interpolate cached state increments, rewards, and probabilities."""

    current_points = np.asarray(
        [_point_from_state(state, result.config) for state in states]
    )
    current = result.interpolator.prepare(current_points)
    next_points = np.empty(
        (len(action_indices), current_points.shape[1]),
        dtype=float,
    )
    rewards = np.empty(len(action_indices), dtype=float)
    probabilities = np.empty(len(action_indices), dtype=float)
    for action_index in np.unique(action_indices):
        positions = np.flatnonzero(action_indices == action_index)
        selected_trajectories = trajectory_indices[positions]
        vertices = current.indices[selected_trajectories]
        weights = current.weights[selected_trajectories]
        table = result.tables[int(action_index)]
        deltas = np.einsum(
            "nv,nvd->nd",
            weights,
            table.delta[vertices],
            optimize=False,
        )
        next_points[positions] = (
            current_points[selected_trajectories] + deltas
        )
        rewards[positions] = np.einsum(
            "nv,nv->n",
            weights,
            table.reward[vertices],
            optimize=False,
        )
        probabilities[positions] = (
            rewards[positions] / result.config.step_size
            if table.spec.kind == "online"
            else 0.0
        )
    next_states = [
        _project_state(
            _state_from_point(point, result.config),
            result.config,
        )
        for point in next_points
    ]
    return next_states, rewards, probabilities


def _modulated_candidate_outcomes(
    result: DPResult,
    states: list[State],
    trajectory_indices: np.ndarray,
    action_indices: np.ndarray,
    times: np.ndarray,
) -> tuple[list[State], np.ndarray, np.ndarray]:
    """Use the Bellman transition kernel for continuous modulated rollouts."""

    current_points = np.asarray(
        [_point_from_state(state, result.config) for state in states],
        dtype=np.float64,
    )
    next_points = np.empty(
        (len(action_indices), current_points.shape[1]),
        dtype=np.float64,
    )
    rewards = np.empty(len(action_indices), dtype=np.float64)
    probabilities = np.empty(len(action_indices), dtype=np.float64)
    for position, (trajectory_index, action_index) in enumerate(
        zip(trajectory_indices, action_indices)
    ):
        next_point, reward, probability = _modulated_transition(
            result.tables[int(action_index)],
            current_points[int(trajectory_index)][None, :],
            result.params,
            result.config,
            float(times[int(trajectory_index)]),
            result.interpolator,
        )
        next_points[position] = next_point[0]
        rewards[position] = reward[0]
        probabilities[position] = probability[0]
    next_states = [
        _state_from_point(point, result.config)
        for point in next_points
    ]
    return next_states, rewards, probabilities


def _candidate_continuations(
    result: DPResult,
    states: list[State],
    modes: list[str],
    layers: np.ndarray,
) -> np.ndarray:
    """Interpolate continuation values for a batch of candidate outcomes."""

    points = np.asarray(
        [_point_from_state(state, result.config) for state in states]
    )
    prepare = (
        result.interpolator.prepare_grid_linear
        if result.params.C_A != 0.0
        else result.interpolator.prepare
    )
    interpolation = prepare(points)
    mode_indices = np.asarray(
        [result.mode_names.index(mode) for mode in modes],
        dtype=np.intp,
    )
    source = np.asarray(
        result.values[mode_indices, layers],
        dtype=np.float64,
    )
    gathered = source[
        np.arange(len(states))[:, None],
        interpolation.indices,
    ]
    return np.einsum(
        "nv,nv->n",
        gathered,
        interpolation.weights,
        optimize=False,
    )


def optimal_action_at(
    result: DPResult,
    state: State,
    horizon: float,
    *,
    mode: str = "on",
    time: float = 0.0,
    outcome_method: str = "auto",
) -> ActionSpec | None:
    """Return the continuous-state Bellman action for one initial condition.

    The returned first action accounts for optimal continuation through the
    complete remaining ``horizon``. This is therefore equivalent to starting a
    rollout from the supplied state and horizon, but avoids simulating the
    already-solved continuation merely to inspect its first decision.
    """

    if result.values is None:
        raise ValueError("continuous action lookup requires full value history")
    if mode not in result.mode_names:
        raise ValueError("mode is not available for this DP result")
    remaining_layer = _rollout_layer(result, horizon)
    action_indices = np.asarray(
        [
            index
            for index, spec in enumerate(result.actions)
            if spec.grid_steps <= remaining_layer
            and not (mode == "on" and spec.kind == "offline")
            and not (mode == "off" and spec.kind == "online")
        ],
        dtype=np.intp,
    )
    if len(action_indices) == 0:
        return None

    method = _rollout_outcome_method(result, outcome_method)
    projected = _project_state(state, result.config)
    trajectory_indices = np.zeros(len(action_indices), dtype=np.intp)
    if method == "table":
        if result.params.C_A == 0.0:
            next_states, rewards, _ = _table_candidate_outcomes(
                result,
                [projected],
                trajectory_indices,
                action_indices,
            )
        else:
            next_states, rewards, _ = _modulated_candidate_outcomes(
                result,
                [projected],
                trajectory_indices,
                action_indices,
                np.full(len(action_indices), float(time), dtype=np.float64),
            )
    else:
        outcomes = [
            _continuous_outcome(
                projected,
                result.actions[int(action_index)],
                result.params,
                result.config,
                float(time),
            )
            for action_index in action_indices
        ]
        next_states = [outcome[0] for outcome in outcomes]
        rewards = np.asarray([outcome[1] for outcome in outcomes], dtype=float)

    next_modes = [
        ("off" if mode == "on" else "on")
        if result.actions[int(action_index)].kind == "switch"
        else mode
        for action_index in action_indices
    ]
    continuation_layers = np.asarray(
        [
            remaining_layer - result.actions[int(action_index)].grid_steps
            for action_index in action_indices
        ],
        dtype=np.intp,
    )
    continuations = _candidate_continuations(
        result,
        next_states,
        next_modes,
        continuation_layers,
    )
    scores = rewards + continuations
    winner = int(np.argmax(scores))
    if float(scores[winner]) <= 1e-14:
        return None
    return result.actions[int(action_indices[winner])]


def _nearest_policy_point(result: DPResult, state: State) -> int:
    """Return the nearest feasible grid point used by a compact policy."""

    point = _point_from_state(_project_state(state, result.config), result.config)
    return int(result.interpolator.tree.query(point)[1])


def _compact_envelope_strategy(
    result: DPResult,
    layer: int,
    state: State,
) -> int:
    """Read the active strategy selected by the compact online envelope."""

    compact = result.compact_policy
    if compact is None:
        raise ValueError("DP result does not contain a compact policy")
    point_index = _nearest_policy_point(result, state)
    return int(compact.envelope_strategy[layer, point_index])


def _compact_action_outcome(
    result: DPResult,
    state: State,
    action_index: int,
    time: float,
    method: str,
) -> tuple[State, float, float]:
    """Advance one compact-policy action with the requested outcome kernel."""

    if method == "analytical":
        return _continuous_outcome(
            state,
            result.actions[action_index],
            result.params,
            result.config,
            time,
        )
    if result.params.C_A == 0.0:
        states, rewards, probabilities = _table_candidate_outcomes(
            result,
            [state],
            np.array([0], dtype=np.intp),
            np.array([action_index], dtype=np.intp),
        )
    else:
        states, rewards, probabilities = _modulated_candidate_outcomes(
            result,
            [state],
            np.array([0], dtype=np.intp),
            np.array([action_index], dtype=np.intp),
            np.array([time], dtype=np.float64),
        )
    return states[0], float(rewards[0]), float(probabilities[0])


def _rollout_compact_policy(
    result: DPResult,
    initial: State,
    *,
    initial_mode: str,
    horizon: float | None,
    outcome_method: str,
) -> list[dict]:
    """Roll out a bit-packed grid policy with continuous action transitions.

    Decisions are read at the nearest feasible correlation-grid point. State
    transitions retain the same cached interpolation or analytical kernel as
    ordinary rollouts, so only the decision boundary is grid-quantized.
    """

    compact = result.compact_policy
    if compact is None:
        raise ValueError("DP result does not contain a compact policy")
    if initial_mode not in result.mode_names:
        raise ValueError("initial_mode is not available for this DP variant")
    method = _rollout_outcome_method(result, outcome_method)
    remaining_layer = _rollout_layer(result, horizon)
    state = _project_state(initial, result.config)
    mode = initial_mode
    time = 0.0
    rows: list[dict] = []
    online_indices = [
        index
        for index, action in enumerate(result.actions)
        if action.kind == "online"
    ]
    strategy_switch_index = next(
        (
            index
            for index, action in enumerate(result.actions)
            if action.kind == "strategy_switch"
        ),
        None,
    )
    offline_index = next(
        (
            index
            for index, action in enumerate(result.actions)
            if action.kind == "offline"
        ),
        None,
    )
    switch_index = next(
        (
            index
            for index, action in enumerate(result.actions)
            if action.kind == "switch"
        ),
        None,
    )
    active_strategy = (
        _compact_envelope_strategy(result, remaining_layer, state)
        if mode == "on"
        else None
    )

    while remaining_layer > 0:
        point_index = _nearest_policy_point(result, state)
        if mode == "on":
            if active_strategy is None or active_strategy >= len(online_indices):
                raise ValueError("compact policy selected an invalid strategy")
            decision = _unpack_policy_decision(
                compact.strategy_decisions[
                    active_strategy,
                    remaining_layer,
                ],
                point_index,
            )
            if decision == _POLICY_STOP:
                action_index = None
            elif decision == _POLICY_CONTINUE:
                action_index = online_indices[active_strategy]
            elif decision == _POLICY_STRATEGY_SWITCH:
                action_index = strategy_switch_index
            elif decision == _POLICY_MODE_SWITCH:
                action_index = switch_index
            else:  # pragma: no cover - decisions are two-bit values
                raise ValueError("invalid compact online-policy decision")
        else:
            if compact.offline_decisions is None:
                raise ValueError("compact policy is missing offline decisions")
            decision = _unpack_policy_decision(
                compact.offline_decisions[remaining_layer],
                point_index,
            )
            if decision == _POLICY_STOP:
                action_index = None
            elif decision == _POLICY_CONTINUE:
                action_index = offline_index
            elif decision == _POLICY_MODE_SWITCH:
                action_index = switch_index
            else:
                raise ValueError("invalid compact offline-policy decision")

        if action_index is None:
            rows.append(
                {
                    "time": time,
                    "remaining_horizon": remaining_layer * result.times[1],
                    "mode": mode,
                    "action": "stop",
                    "rho_se": state.rho_se,
                    "rho_me": state.rho_me,
                    "rho_sm": state.rho_sm,
                }
            )
            break
        spec = result.actions[action_index]
        if spec.grid_steps > remaining_layer:
            raise ValueError("compact policy selected an infeasible action")
        next_state, reward, probability = _compact_action_outcome(
            result,
            state,
            action_index,
            time,
            method,
        )
        reward_rate = (
            probability / result.params.online_cost(spec.K, spec.M)
            if spec.kind == "online"
            else 0.0
        )
        beta: float | str = ""
        if spec.kind == "online":
            if result.params.temperature_mode == "constant":
                beta = result.params.beta
            else:
                action_params = replace(
                    _active_params(result.params, result.config),
                    K=spec.K,
                    M=spec.M,
                    K_mode="constant",
                    M_mode="constant",
                    greedy_training=False,
                )
                beta = components(state, action_params, "online", time).beta
        rows.append(
            {
                "time": time,
                "remaining_horizon": remaining_layer * result.times[1],
                "mode": mode,
                "action": spec.name,
                "K": "" if spec.K is None else spec.K,
                "M": "" if spec.M is None else spec.M,
                "beta": beta,
                "rho_se": state.rho_se,
                "rho_me": state.rho_me,
                "rho_sm": state.rho_sm,
                "value": "",
                "reward_probability": probability,
                "reward_rate": reward_rate,
            }
        )
        previous_mode = mode
        state = next_state
        time += spec.duration
        remaining_layer -= spec.grid_steps
        if spec.kind == "switch":
            mode = "off" if previous_mode == "on" else "on"
            active_strategy = None
        if spec.kind == "strategy_switch" or (
            spec.kind == "switch" and previous_mode == "off"
        ):
            active_strategy = _compact_envelope_strategy(
                result,
                remaining_layer,
                state,
            )

    rows.append(
        {
            "time": time,
            "remaining_horizon": max(remaining_layer, 0) * result.times[1],
            "mode": mode,
            "action": "terminal",
            "rho_se": state.rho_se,
            "rho_me": state.rho_me,
            "rho_sm": state.rho_sm,
        }
    )
    return rows


def rollout_policies(
    result: DPResult,
    initials: Iterable[State],
    *,
    initial_mode: str = "on",
    horizon: float | None = None,
    outcome_method: str = "auto",
) -> list[list[dict]]:
    """Roll out many continuous initial states in one batched calculation.

    ``outcome_method="table"`` uses the same cached action data and transition
    kernel as the Bellman solver. Static runs interpolate completed action
    increments; modulated runs integrate cached vector fields with the shared
    midpoint kernel. ``"analytical"`` retains the previous direct one-step
    calculation as a diagnostic reference. ``"auto"`` selects the compatible
    table/kernel path.

    A shorter ``horizon`` may reuse the leading Bellman layers of a result
    solved at a larger horizon when the dynamics are time-homogeneous.
    """

    if result.values is None and result.compact_policy is None:
        raise ValueError(
            "trajectory generation requires full value history; positive "
            "t_strategy_switch results require compact policy recording"
        )

    initial_states = list(initials)
    if not initial_states:
        return []
    if result.values is None:
        return [
            _rollout_compact_policy(
                result,
                initial,
                initial_mode=initial_mode,
                horizon=horizon,
                outcome_method=outcome_method,
            )
            for initial in initial_states
        ]
    if initial_mode not in result.mode_names:
        raise ValueError("initial_mode is not available for this DP variant")
    method = _rollout_outcome_method(result, outcome_method)
    initial_layer = _rollout_layer(result, horizon)
    count = len(initial_states)
    states = [
        _project_state(initial, result.config)
        for initial in initial_states
    ]
    modes = [initial_mode] * count
    remaining_layers = np.full(count, initial_layer, dtype=np.intp)
    times = np.zeros(count, dtype=float)
    active = np.ones(count, dtype=bool)
    rows: list[list[dict]] = [[] for _ in range(count)]

    while np.any(active & (remaining_layers > 0)):
        active_indices = np.flatnonzero(active & (remaining_layers > 0))
        trajectory_indices: list[int] = []
        action_indices: list[int] = []
        positions_by_trajectory: list[list[int]] = [
            [] for _ in range(count)
        ]
        for trajectory_index in active_indices:
            mode = modes[trajectory_index]
            remaining_layer = remaining_layers[trajectory_index]
            for action_index, spec in enumerate(result.actions):
                if spec.grid_steps > remaining_layer:
                    continue
                if mode == "on" and spec.kind == "offline":
                    continue
                if mode == "off" and spec.kind == "online":
                    continue
                position = len(action_indices)
                trajectory_indices.append(int(trajectory_index))
                action_indices.append(action_index)
                positions_by_trajectory[trajectory_index].append(position)

        for trajectory_index in active_indices:
            if positions_by_trajectory[trajectory_index]:
                continue
            state = states[trajectory_index]
            rows[trajectory_index].append(
                {
                    "time": times[trajectory_index],
                    "remaining_horizon": (
                        remaining_layers[trajectory_index] * result.times[1]
                    ),
                    "mode": modes[trajectory_index],
                    "action": "stop",
                    "rho_se": state.rho_se,
                    "rho_me": state.rho_me,
                    "rho_sm": state.rho_sm,
                }
            )
            active[trajectory_index] = False

        if not action_indices:
            continue
        trajectory_array = np.asarray(trajectory_indices, dtype=np.intp)
        action_array = np.asarray(action_indices, dtype=np.intp)
        if method == "table":
            if result.params.C_A == 0.0:
                next_states, rewards, probabilities = _table_candidate_outcomes(
                    result,
                    states,
                    trajectory_array,
                    action_array,
                )
            else:
                next_states, rewards, probabilities = (
                    _modulated_candidate_outcomes(
                        result,
                        states,
                        trajectory_array,
                        action_array,
                        times,
                    )
                )
        else:
            outcomes = [
                _continuous_outcome(
                    states[trajectory_index],
                    result.actions[action_index],
                    result.params,
                    result.config,
                    times[trajectory_index],
                )
                for trajectory_index, action_index in zip(
                    trajectory_indices,
                    action_indices,
                )
            ]
            next_states = [outcome[0] for outcome in outcomes]
            rewards = np.asarray(
                [outcome[1] for outcome in outcomes],
                dtype=float,
            )
            probabilities = np.asarray(
                [outcome[2] for outcome in outcomes],
                dtype=float,
            )
        next_modes = [
            (
                "off" if modes[trajectory_index] == "on" else "on"
            )
            if result.actions[action_index].kind == "switch"
            else modes[trajectory_index]
            for trajectory_index, action_index in zip(
                trajectory_indices,
                action_indices,
            )
        ]
        continuation_layers = np.asarray(
            [
                remaining_layers[trajectory_index]
                - result.actions[action_index].grid_steps
                for trajectory_index, action_index in zip(
                    trajectory_indices,
                    action_indices,
                )
            ],
            dtype=np.intp,
        )
        continuations = _candidate_continuations(
            result,
            next_states,
            next_modes,
            continuation_layers,
        )
        scores = rewards + continuations

        for trajectory_index in active_indices:
            positions = positions_by_trajectory[trajectory_index]
            if not positions:
                continue
            local_scores = scores[positions]
            selected_position = positions[int(np.argmax(local_scores))]
            best_score = float(scores[selected_position])
            state = states[trajectory_index]
            if best_score <= 1e-14:
                rows[trajectory_index].append(
                    {
                        "time": times[trajectory_index],
                        "remaining_horizon": (
                            remaining_layers[trajectory_index]
                            * result.times[1]
                        ),
                        "mode": modes[trajectory_index],
                        "action": "stop",
                        "rho_se": state.rho_se,
                        "rho_me": state.rho_me,
                        "rho_sm": state.rho_sm,
                    }
                )
                active[trajectory_index] = False
                continue
            action_index = int(action_array[selected_position])
            spec = result.actions[action_index]
            probability = float(probabilities[selected_position])
            reward_rate = (
                probability / result.params.online_cost(spec.K, spec.M)
                if spec.kind == "online"
                else 0.0
            )
            beta: float | str = ""
            if spec.kind == "online":
                if result.params.temperature_mode == "constant":
                    beta = result.params.beta
                else:
                    action_params = replace(
                        _active_params(result.params, result.config),
                        K=spec.K,
                        M=spec.M,
                        K_mode="constant",
                        M_mode="constant",
                        greedy_training=False,
                    )
                    beta = components(
                        state,
                        action_params,
                        "online",
                        times[trajectory_index],
                    ).beta
            rows[trajectory_index].append(
                {
                    "time": times[trajectory_index],
                    "remaining_horizon": (
                        remaining_layers[trajectory_index] * result.times[1]
                    ),
                    "mode": modes[trajectory_index],
                    "action": spec.name,
                    "K": "" if spec.K is None else spec.K,
                    "M": "" if spec.M is None else spec.M,
                    "beta": beta,
                    "rho_se": state.rho_se,
                    "rho_me": state.rho_me,
                    "rho_sm": state.rho_sm,
                    "value": best_score,
                    "reward_probability": probability,
                    "reward_rate": reward_rate,
                }
            )
            states[trajectory_index] = next_states[selected_position]
            times[trajectory_index] += spec.duration
            remaining_layers[trajectory_index] -= spec.grid_steps
            modes[trajectory_index] = next_modes[selected_position]

    for trajectory_index in range(count):
        state = states[trajectory_index]
        rows[trajectory_index].append(
            {
                "time": times[trajectory_index],
                "remaining_horizon": (
                    max(remaining_layers[trajectory_index], 0)
                    * result.times[1]
                ),
                "mode": modes[trajectory_index],
                "action": "terminal",
                "rho_se": state.rho_se,
                "rho_me": state.rho_me,
                "rho_sm": state.rho_sm,
            }
        )
    return rows


def rollout_policy(
    result: DPResult,
    initial: State,
    *,
    initial_mode: str = "on",
    horizon: float | None = None,
    outcome_method: str = "auto",
) -> list[dict]:
    """Roll out one state using the batched continuous-policy machinery."""

    return rollout_policies(
        result,
        [initial],
        initial_mode=initial_mode,
        horizon=horizon,
        outcome_method=outcome_method,
    )[0]


def additional_planning_time(
    rollout: Iterable[dict],
    params: Params,
    step_size: float,
) -> float:
    """Return time spent planning above the ``K=1, M=1`` minimum.

    Only completed online actions with ``K>1`` contribute. For each such
    action, the minimum prefix-simulation time associated with one
    ``K=1, M=1`` action is subtracted before multiplying by the DP block size.
    Execution time is excluded because it is incurred for every online block.
    """

    step_size = float(step_size)
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    minimum_prefix_time = params.prefix_time(1)
    total = 0.0
    for row in rollout:
        K_value = row.get("K", "")
        if K_value in {"", None}:
            continue
        K_float = float(K_value)
        K = int(round(K_float))
        if not np.isclose(K_float, K, atol=1e-12, rtol=0.0) or K < 1:
            raise ValueError("rollout K values must be positive integers")
        if K == 1:
            continue
        M_value = row.get("M", "")
        if M_value in {"", None}:
            raise ValueError("rollout actions with K>1 must specify M")
        M_float = float(M_value)
        M = int(round(M_float))
        if not np.isclose(M_float, M, atol=1e-12, rtol=0.0) or not 1 <= M <= params.T:
            raise ValueError("rollout M values must be integers in [1, T]")
        extra_prefix_time = K * params.prefix_time(M) - minimum_prefix_time
        if extra_prefix_time < -1e-12:
            raise ValueError("additional planning time cannot be negative")
        total += step_size * max(extra_prefix_time, 0.0)
    return float(total)


def minimum_online_block_time(params: Params, step_size: float) -> float:
    """Return the duration of one ``K=1, M=1`` online action block."""

    step_size = float(step_size)
    if not np.isfinite(step_size) or step_size <= 0.0:
        raise ValueError("step_size must be finite and positive")
    return float(step_size * params.online_cost(1, 1))


def save_dp_result(path: str | Path, result: DPResult) -> None:
    """Save a versioned DP result, including reusable action-table arrays."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value_storage = "full" if result.values is not None else "terminal"
    metadata = {
        "schema_version": 5,
        "variant": result.variant,
        "config": asdict(result.config),
        "params": asdict(result.params),
        "actions": [asdict(action) for action in result.actions],
        "mode_names": result.mode_names,
        "value_storage": value_storage,
        "value_dtype": "float32" if value_storage == "full" else "float64",
        "policy_storage": (
            "compact"
            if result.compact_policy is not None
            else ("full" if result.policy is not None else "none")
        ),
    }
    arrays = {
        "metadata": np.array(json.dumps(metadata)),
        "times": result.times,
        "points": result.points,
        "action_delta": np.stack([table.delta for table in result.tables]),
        "action_reward": np.stack([table.reward for table in result.tables]),
        "action_vector_field": np.stack(
            [table.vector_field for table in result.tables]
        ),
        "action_probability": np.stack(
            [table.probability for table in result.tables]
        ),
    }
    if result.values is None:
        if result.terminal_values is None:
            raise ValueError("rolling DP result is missing terminal values")
        arrays["terminal_values"] = np.asarray(
            result.terminal_values,
            dtype=np.float64,
        )
    else:
        if result.policy is None:
            raise ValueError("full-history DP result is missing its policy")
        arrays["values"] = np.asarray(result.values, dtype=np.float32)
        arrays["policy"] = result.policy
        arrays["terminal_values"] = np.asarray(
            result.values[:, -1],
            dtype=np.float32,
        )
    if result.compact_policy is not None:
        arrays["compact_strategy_decisions"] = np.asarray(
            result.compact_policy.strategy_decisions,
            dtype=np.uint8,
        )
        arrays["compact_envelope_strategy"] = np.asarray(
            result.compact_policy.envelope_strategy,
        )
        if result.compact_policy.offline_decisions is not None:
            arrays["compact_offline_decisions"] = np.asarray(
                result.compact_policy.offline_decisions,
                dtype=np.uint8,
            )
    np.savez_compressed(path, **arrays)


def load_dp_result(path: str | Path) -> DPResult:
    """Load a DP result, reusing serialized action tables when available."""

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        params = Params(**metadata["params"])
        config = DPConfig(**metadata["config"])
        if config.no_modulation and params.C_A != 0.0:
            params = replace(params, C_A=0.0)
        points = np.asarray(data["points"])
        actions = tuple(ActionSpec(**item) for item in metadata["actions"])
        table_keys = (
            "action_delta",
            "action_reward",
            "action_vector_field",
            "action_probability",
        )
        schema_version = int(metadata.get("schema_version", 1))
        if schema_version >= 3:
            missing = [key for key in table_keys if key not in data.files]
            if missing:
                raise ValueError(
                    "DP result is missing serialized action arrays: "
                    + ", ".join(missing)
                )
            action_delta = np.asarray(data["action_delta"])
            action_reward = np.asarray(data["action_reward"])
            action_vector_field = np.asarray(data["action_vector_field"])
            action_probability = np.asarray(data["action_probability"])
            action_count = len(actions)
            point_count, dimension = points.shape
            if (
                action_delta.shape != (action_count, point_count, dimension)
                or action_reward.shape != (action_count, point_count)
                or action_vector_field.shape
                != (action_count, point_count, dimension)
                or action_probability.shape != (action_count, point_count)
            ):
                raise ValueError("serialized action-table arrays have invalid shapes")
            tables = tuple(
                ActionTable(
                    action,
                    action_delta[index],
                    action_reward[index],
                    action_vector_field[index],
                    action_probability[index],
                )
                for index, action in enumerate(actions)
            )
        else:
            # Older result files did not persist these expensive analytical
            # calculations, so retain a compatibility path for existing data.
            tables = tuple(
                _build_action_table(points, params, config, action)
                for action in actions
            )
        value_storage = metadata.get("value_storage", "full")
        if value_storage == "full":
            if "values" not in data.files or "policy" not in data.files:
                raise ValueError("full-history DP result is missing values or policy")
            values = np.asarray(data["values"])
            policy = np.asarray(data["policy"])
            terminal_values = (
                np.asarray(data["terminal_values"])
                if "terminal_values" in data.files
                else np.asarray(values[:, -1])
            )
        elif value_storage == "terminal":
            if "terminal_values" not in data.files:
                raise ValueError("rolling DP result is missing terminal values")
            values = None
            policy = None
            terminal_values = np.asarray(data["terminal_values"])
        else:
            raise ValueError(f"unknown DP value storage mode {value_storage!r}")
        policy_storage = metadata.get(
            "policy_storage",
            "full" if value_storage == "full" else "none",
        )
        compact_policy = None
        if policy_storage == "compact":
            required = {
                "compact_strategy_decisions",
                "compact_envelope_strategy",
            }
            missing = sorted(required.difference(data.files))
            if missing:
                raise ValueError(
                    "compact DP policy is missing arrays: " + ", ".join(missing)
                )
            strategy_decisions = np.asarray(
                data["compact_strategy_decisions"]
            )
            offline_decisions = (
                np.asarray(data["compact_offline_decisions"])
                if "compact_offline_decisions" in data.files
                else None
            )
            envelope_strategy = np.asarray(
                data["compact_envelope_strategy"]
            )
            strategy_count = sum(
                action.kind == "online" for action in actions
            )
            layer_count = len(data["times"])
            packed_width = _packed_policy_width(len(points))
            if strategy_decisions.shape != (
                strategy_count,
                layer_count,
                packed_width,
            ):
                raise ValueError("compact strategy policy has an invalid shape")
            if envelope_strategy.shape != (layer_count, len(points)):
                raise ValueError("compact envelope policy has an invalid shape")
            expected_offline_shape = (layer_count, packed_width)
            if metadata["variant"] == "switching":
                if (
                    offline_decisions is None
                    or offline_decisions.shape != expected_offline_shape
                ):
                    raise ValueError("compact offline policy has an invalid shape")
            elif offline_decisions is not None:
                raise ValueError("online-only compact policy contains offline data")
            compact_policy = CompactPolicy(
                strategy_decisions=strategy_decisions,
                offline_decisions=offline_decisions,
                envelope_strategy=envelope_strategy,
                point_count=len(points),
            )
        elif policy_storage not in {"full", "none"}:
            raise ValueError(f"unknown DP policy storage mode {policy_storage!r}")
        return DPResult(
            variant=metadata["variant"],
            config=config,
            params=params,
            times=np.asarray(data["times"]),
            points=points,
            values=values,
            policy=policy,
            actions=actions,
            mode_names=tuple(metadata["mode_names"]),
            interpolator=FeasibleInterpolator(points, config),
            tables=tables,
            terminal_values=terminal_values,
            compact_policy=compact_policy,
        )
