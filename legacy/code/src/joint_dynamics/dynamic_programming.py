"""Finite-horizon dynamic programming for online and offline control."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from itertools import product
import json
from pathlib import Path

import numpy as np
from scipy.spatial import Delaunay, cKDTree
import sys

from .config import Params, State
from .dynamics import components
from .modulation import (
    average_environment_modulation,
    environment_modulation,
    integrated_environment_modulation,
)


_VALUE_TIE_TOL = 1e-12


@dataclass(frozen=True)
class DPConfig:
    horizon: float = 10.0
    time_step: float = 0.5
    rho_points: int = 9
    rho_min: float = -0.95
    rho_max: float = 0.95
    K_max: int | None = None
    reward_kind: str = "observed"
    fixed_rho_me: float = 1.0
    fixed_wm: bool = False
    switching_cost: float = 1.0
    interpolation: str = "structured"

    def validate(self, params: Params, variant: str) -> None:
        if variant not in {"online", "switching"}:
            raise ValueError("unknown dynamic-programming variant")
        if self.horizon <= 0.0 or self.time_step <= 0.0:
            raise ValueError("horizon and time_step must be positive")
        ratio = self.horizon / self.time_step
        if abs(ratio - round(ratio)) > 1e-9:
            raise ValueError("horizon must be an integer multiple of time_step")
        if self.rho_points < 3 or not -1.0 < self.rho_min < self.rho_max < 1.0:
            raise ValueError("correlation grid must have at least three interior points")
        if self.K_max is not None and self.K_max < 1:
            raise ValueError("K_max must be positive or None")
        if self.reward_kind not in {"observed", "true"}:
            raise ValueError("reward_kind must be 'observed' or 'true'")
        if self.interpolation not in {"delaunay", "structured"}:
            raise ValueError("interpolation must be 'delaunay' or 'structured'")
        if not -1.0 <= self.fixed_rho_me <= 1.0:
            raise ValueError("fixed_rho_me must lie in [-1, 1]")
        if variant == "switching" and self.switching_cost <= 0.0:
            raise ValueError("switching_cost must be positive")
        timing_unit = _planning_time_unit(params)
        if timing_unit <= 0.0:
            if params.partial_evaluation:
                raise ValueError("tau_element must be positive for partial dynamic programming")
            raise ValueError("tau_on must be positive for dynamic programming")
        reciprocal = 1.0 / timing_unit
        if abs(reciprocal - round(reciprocal)) > 1e-9:
            raise ValueError("1 / elementary thinking time must be an integer")
        costs = [spec.cost for spec in _online_action_specs(params, self)]
        if variant == "switching":
            offline_ratio = params.tau_off / timing_unit
            if abs(offline_ratio - round(offline_ratio)) > 1e-9:
                raise ValueError(
                    "tau_off must be an integer multiple of the elementary thinking time"
                )
            duration = _abstract_action_duration(params, self)
            costs.extend([duration * params.tau_off, self.switching_cost])
        for cost in costs:
            ratio = cost / self.time_step
            if abs(ratio - round(ratio)) > 1e-9:
                raise ValueError(
                    "every action cost must be an integer multiple of time_step"
                )


@dataclass(frozen=True)
class InterpolationWeights:
    vertices: np.ndarray
    weights: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        selected = values[self.vertices]
        weights = self.weights
        while weights.ndim < selected.ndim:
            weights = weights[..., None]
        return np.sum(selected * weights, axis=1)


class FeasibleInterpolator:
    """Reusable barycentric interpolation on feasible state-grid points."""

    def __init__(self, points: np.ndarray):
        self.points = np.asarray(points, dtype=float)
        self.origin = np.mean(self.points, axis=0)
        _, singular, vectors = np.linalg.svd(self.points - self.origin, full_matrices=False)
        tolerance = max(self.points.shape) * np.finfo(float).eps * singular[0]
        self.dimension = int(np.sum(singular > tolerance))
        if self.dimension < 1:
            raise ValueError("state grid must contain more than one distinct point")
        self.basis = vectors[: self.dimension].T
        self.coordinates = (self.points - self.origin) @ self.basis
        self.tree = cKDTree(self.coordinates)
        self.triangulation = (
            Delaunay(self.coordinates) if self.dimension >= 2 else None
        )
        if self.dimension == 1:
            self.order = np.argsort(self.coordinates[:, 0])
            self.sorted_coordinates = self.coordinates[self.order, 0]

    def prepare(self, queries: np.ndarray) -> InterpolationWeights:
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        coordinates = (queries - self.origin) @ self.basis
        if self.dimension == 1:
            positions = np.searchsorted(self.sorted_coordinates, coordinates[:, 0])
            upper_position = np.clip(positions, 1, len(self.order) - 1)
            lower_position = upper_position - 1
            lower = self.order[lower_position]
            upper = self.order[upper_position]
            low_coordinate = self.sorted_coordinates[lower_position]
            high_coordinate = self.sorted_coordinates[upper_position]
            fraction = np.divide(
                coordinates[:, 0] - low_coordinate,
                high_coordinate - low_coordinate,
                out=np.zeros(len(queries)),
                where=np.abs(high_coordinate - low_coordinate) > 1e-14,
            )
            fraction = np.clip(fraction, 0.0, 1.0)
            return InterpolationWeights(
                np.column_stack([lower, upper]),
                np.column_stack([1.0 - fraction, fraction]),
            )
        simplex = self.triangulation.find_simplex(coordinates)
        vertices = np.empty((len(queries), self.dimension + 1), dtype=int)
        weights = np.zeros((len(queries), self.dimension + 1), dtype=float)
        inside = simplex >= 0
        if np.any(inside):
            selected = simplex[inside]
            transform = self.triangulation.transform[selected]
            delta = coordinates[inside] - transform[:, self.dimension, :]
            first = np.einsum("nij,nj->ni", transform[:, : self.dimension, :], delta)
            vertices[inside] = self.triangulation.simplices[selected]
            weights[inside, : self.dimension] = first
            weights[inside, self.dimension] = 1.0 - np.sum(first, axis=1)
        if np.any(~inside):
            nearest = self.tree.query(coordinates[~inside])[1]
            vertices[~inside] = np.repeat(nearest[:, None], self.dimension + 1, axis=1)
            weights[~inside, 0] = 1.0
        return InterpolationWeights(vertices, weights)

    def nearest(self, queries: np.ndarray) -> np.ndarray:
        coordinates = (np.atleast_2d(queries) - self.origin) @ self.basis
        return np.asarray(self.tree.query(coordinates)[1])


class StructuredInterpolator:
    """Fast multilinear interpolation on the underlying Cartesian rho grid.

    The physical correlation domain omits infeasible grid vertices. Their
    multilinear weights are discarded and the remaining feasible weights are
    renormalized. Queries with no weighted feasible corner fall back to the
    nearest feasible grid point.
    """

    def __init__(
        self,
        points: np.ndarray,
        rho_min: float,
        rho_max: float,
        rho_points: int,
    ):
        self.points = np.asarray(points, dtype=float)
        self.dimension = self.points.shape[1]
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.rho_points = int(rho_points)
        self.spacing = (self.rho_max - self.rho_min) / (self.rho_points - 1)
        shape = (self.rho_points,) * self.dimension
        self.lookup = np.full(shape, -1, dtype=np.int64)
        indices = np.rint(
            (self.points - self.rho_min) / self.spacing
        ).astype(int)
        self.lookup[tuple(indices.T)] = np.arange(len(self.points))
        self.offsets = np.asarray(
            tuple(product((0, 1), repeat=self.dimension)), dtype=int
        )
        self.tree = cKDTree(self.points)

    def prepare(self, queries: np.ndarray) -> InterpolationWeights:
        queries = np.atleast_2d(np.asarray(queries, dtype=float))
        clipped = np.clip(queries, self.rho_min, self.rho_max)
        positions = (clipped - self.rho_min) / self.spacing
        lower = np.floor(positions).astype(int)
        lower = np.clip(lower, 0, self.rho_points - 2)
        fractions = np.clip(positions - lower, 0.0, 1.0)

        corners = lower[:, None, :] + self.offsets[None, :, :]
        vertices = self.lookup[tuple(np.moveaxis(corners, -1, 0))]
        factors = np.where(
            self.offsets[None, :, :] == 1,
            fractions[:, None, :],
            1.0 - fractions[:, None, :],
        )
        weights = np.prod(factors, axis=2)
        feasible = vertices >= 0
        weights *= feasible
        totals = np.sum(weights, axis=1)
        usable = totals > 1e-14
        weights[usable] /= totals[usable, None]
        vertices[~feasible] = 0

        if np.any(~usable):
            nearest = self.tree.query(clipped[~usable])[1]
            vertices[~usable] = np.repeat(
                nearest[:, None], len(self.offsets), axis=1
            )
            weights[~usable] = 0.0
            weights[~usable, 0] = 1.0
        return InterpolationWeights(vertices, weights)

    def nearest(self, queries: np.ndarray) -> np.ndarray:
        return np.asarray(self.tree.query(np.atleast_2d(queries))[1])


def _make_interpolator(points: np.ndarray, config: DPConfig):
    if config.interpolation == "structured":
        points = np.asarray(points, dtype=float)
        intrinsic_dimension = np.linalg.matrix_rank(points - np.mean(points, axis=0))
        if intrinsic_dimension == points.shape[1]:
            return StructuredInterpolator(
                points, config.rho_min, config.rho_max, config.rho_points
            )
        # For example, fixed rho_me=1 collapses the nominal two-dimensional
        # grid to a line. The existing intrinsic one-dimensional interpolator
        # is both exact and already avoids Delaunay simplex searches.
    return FeasibleInterpolator(points)


@dataclass(frozen=True)
class ActionTable:
    name: str
    cost: float
    reward: np.ndarray
    next_points: np.ndarray
    interpolation: InterpolationWeights
    delta: np.ndarray
    kind: str
    base_reward_per_action: np.ndarray | None = None
    beta: np.ndarray | None = None
    eta_on1: np.ndarray | None = None
    eta_on2: np.ndarray | None = None
    eta_off: np.ndarray | None = None
    eta_wm: np.ndarray | None = None
    prefix_length: np.ndarray | None = None
    K: int | None = None
    prefix_length_value: int | None = None


@dataclass
class DPResult:
    variant: str
    config: DPConfig
    params: Params
    times: np.ndarray
    points: np.ndarray
    values: np.ndarray
    policy: np.ndarray
    action_names: tuple[str, ...]
    mode_names: tuple[str, ...]
    interpolator: FeasibleInterpolator | StructuredInterpolator
    tables: tuple[ActionTable, ...] | None = None


@dataclass
class DPActionWorkspace:
    """Precomputed action tables for repeated Bellman solves."""

    config: DPConfig
    params: Params
    points: np.ndarray
    tables: tuple[ActionTable, ...]
    action_names: tuple[str, ...]
    mode_names: tuple[str, ...]
    interpolator: FeasibleInterpolator | StructuredInterpolator


@dataclass(frozen=True)
class ActionOutcome:
    name: str
    cost: float
    reward: float
    next_state: State
    base_reward_per_action: float = 0.0
    K: int | None = None
    beta: float | None = None
    eta_on1: float | None = None
    eta_on2: float | None = None
    eta_off: float | None = None
    eta_wm: float | None = None
    prefix_length: float | None = None
    next_mode: str = "on"


@dataclass(frozen=True)
class OnlineActionSpec:
    K: int
    prefix_length: int | None
    duration: float
    cost: float

    @property
    def name(self) -> str:
        if self.prefix_length is None:
            return f"online({self.K})"
        return f"online({self.K},M={self.prefix_length})"


def _K_values(params: Params, config: DPConfig) -> tuple[int, ...]:
    maximum = params.K_max if config.K_max is None else config.K_max
    return tuple(range(1, maximum + 1))


def _planning_time_unit(params: Params) -> float:
    if params.partial_evaluation:
        return float(params.effective_tau_element)
    return float(params.tau_on)


def _abstract_action_duration(params: Params, config: DPConfig) -> float:
    """Number of execution-time units represented by one abstract action."""

    return float(config.time_step / _planning_time_unit(params))


def _online_action_spec(
    K: int,
    params: Params,
    config: DPConfig,
    *,
    prefix_length: int | None = None,
) -> OnlineActionSpec:
    duration = _abstract_action_duration(params, config)
    if params.partial_evaluation:
        active_prefix = params.T if prefix_length is None else prefix_length
        cost = duration * (1.0 + K * active_prefix * _planning_time_unit(params))
        return OnlineActionSpec(K, active_prefix, duration, float(cost))
    cost = duration * (1.0 + K * params.tau_on)
    return OnlineActionSpec(K, None, duration, float(cost))


def _online_action_cost(K: int, params: Params, config: DPConfig) -> float:
    """Elapsed cost while holding an online action fixed for one abstract step."""

    return _online_action_spec(K, params, config).cost


def _online_action_specs(params: Params, config: DPConfig) -> tuple[OnlineActionSpec, ...]:
    if not params.partial_evaluation:
        return tuple(_online_action_spec(K, params, config) for K in _K_values(params, config))
    maximum = params.T if params.prefix_length_max is None else params.prefix_length_max
    return tuple(
        _online_action_spec(K, params, config, prefix_length=prefix_length)
        for K in _K_values(params, config)
        for prefix_length in range(1, maximum + 1)
    )


def _reward(gamma: float, params: Params, config: DPConfig) -> float:
    if config.reward_kind == "true":
        return gamma
    return float(np.clip(params.epsilon + params.delta_epsilon * gamma, 0.0, 1.0))


def _project_fixed(point, rho_me: float) -> np.ndarray:
    a, c = np.clip(np.asarray(point, dtype=float), -1.0, 1.0)
    denominator = a * a + c * c - 2.0 * rho_me * a * c
    radius = 1.0 - rho_me * rho_me
    if denominator > radius and denominator > 0.0:
        scale = np.sqrt(max(radius, 0.0) / denominator)
        a *= scale
        c *= scale
    return np.array([a, c], dtype=float)


def _project_state(
    values,
    config: DPConfig,
) -> State:
    """Project a raw Euler destination according to the DP state constraints."""

    if config.fixed_wm:
        array = np.asarray(values, dtype=float)
        reduced = _project_fixed(
            (array[0], array[1]) if array.shape == (2,) else (array[0], array[2]),
            config.fixed_rho_me,
        )
        return State(float(reduced[0]), config.fixed_rho_me, float(reduced[1]))
    return State.from_array(values, project=True)


def _fixed_grid(config: DPConfig) -> np.ndarray:
    axis = np.linspace(config.rho_min, config.rho_max, config.rho_points)
    first, second = np.meshgrid(axis, axis, indexing="ij")
    points = np.column_stack([first.ravel(), second.ravel()])
    b = config.fixed_rho_me
    determinant = 1.0 + 2.0 * points[:, 0] * b * points[:, 1]
    determinant -= points[:, 0] ** 2 + b * b + points[:, 1] ** 2
    return points[determinant >= -1e-12]


def _full_grid(config: DPConfig) -> np.ndarray:
    axis = np.linspace(config.rho_min, config.rho_max, config.rho_points)
    a, b, c = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack([a.ravel(), b.ravel(), c.ravel()])
    determinant = 1.0 + 2.0 * points[:, 0] * points[:, 1] * points[:, 2]
    determinant -= np.sum(points * points, axis=1)
    return points[determinant >= -1e-12]


def _state_from_point(point, config: DPConfig) -> State:
    if config.fixed_wm:
        return State(float(point[0]), config.fixed_rho_me, float(point[1]))
    return State.from_array(point, project=True)


def _point_from_state(state: State, config: DPConfig) -> np.ndarray:
    if config.fixed_wm:
        return np.array([state.rho_se, state.rho_sm])
    return state.as_array()


def _online_outcome(
    state: State,
    action: int | OnlineActionSpec,
    params: Params,
    config: DPConfig,
    start_time: float = 0.0,
) -> ActionOutcome:
    spec = (
        action
        if isinstance(action, OnlineActionSpec)
        else _online_action_spec(int(action), params, config)
    )
    active = replace(
        params,
        K=spec.K,
        K_mode="constant",
        greedy_training=False,
        greedy_evaluation=False,
    )
    if spec.prefix_length is not None:
        active = replace(
            active,
            prefix_length=spec.prefix_length,
            prefix_length_mode="constant",
        )
    if config.fixed_wm:
        active = replace(active, eta_wm=0.0, fix_eta_wm_zero=True)
        state = State(state.rho_se, config.fixed_rho_me, state.rho_sm)
    base_active = replace(active, M=0.0)
    comp = components(state, base_active, "online")
    modulation = average_environment_modulation(
        start_time, start_time + spec.cost, active
    )
    raw = state.as_array() + modulation * spec.cost * np.array(
        [comp.dot_rho_se, comp.dot_rho_me, comp.dot_rho_sm]
    )
    next_state = _project_state(raw, config)
    gamma = comp.training_environment_reward_probability
    base_reward_per_action = _reward(gamma, active, config)
    return ActionOutcome(
        name=spec.name,
        cost=spec.cost,
        reward=modulation * spec.duration * base_reward_per_action,
        next_state=next_state,
        base_reward_per_action=base_reward_per_action,
        K=spec.K,
        beta=comp.beta,
        eta_on1=comp.eta_on1,
        eta_on2=comp.eta_on2,
        eta_wm=comp.eta_wm,
        prefix_length=comp.prefix_length,
        next_mode="on",
    )


def _offline_outcome(
    state: State,
    params: Params,
    config: DPConfig,
    start_time: float = 0.0,
) -> ActionOutcome:
    active = replace(
        params,
        K_mode="constant",
        prefix_length=params.T,
        prefix_length_mode="constant",
        greedy_training=False,
        greedy_evaluation=False,
    )
    comp = components(state, replace(active, M=0.0), "offline")
    cost = _abstract_action_duration(active, config) * active.c_off
    modulation = average_environment_modulation(
        start_time, start_time + cost, active
    )
    delta = cost * np.array(
        [comp.dot_rho_se, comp.dot_rho_me, comp.dot_rho_sm]
    )
    delta[:2] *= modulation
    raw = state.as_array() + delta
    return ActionOutcome(
        "offline",
        cost,
        0.0,
        _project_state(raw, config),
        eta_off=comp.eta_off,
        prefix_length=active.T,
        next_mode="off",
    )


def _switch_outcome(
    state: State,
    mode: str,
    params: Params,
    config: DPConfig,
    start_time: float = 0.0,
) -> ActionOutcome:
    cost = config.switching_cost
    drift_exposure = integrated_environment_modulation(
        start_time, start_time + cost, params
    )
    retained = float(np.exp(-params.eta_e * drift_exposure))
    raw = (
        np.array([retained * state.rho_se, state.rho_sm])
        if config.fixed_wm
        else np.array([retained * state.rho_se, retained * state.rho_me, state.rho_sm])
    )
    next_state = _project_state(raw, config)
    try:
        next_state.validate()
    except ValueError:
        next_state = next_state.projected()
    return ActionOutcome(
        "switch",
        cost,
        0.0,
        next_state,
        next_mode="off" if mode == "on" else "on",
    )


def _action_table(
    name: str,
    cost: float,
    rewards,
    next_points,
    points,
    interpolator: FeasibleInterpolator | StructuredInterpolator,
    kind: str,
    base_reward_per_action=None,
    beta=None,
    eta_on1=None,
    eta_on2=None,
    eta_off=None,
    eta_wm=None,
    prefix_length=None,
    K: int | None = None,
    prefix_length_value: int | None = None,
) -> ActionTable:
    next_points = np.asarray(next_points, dtype=float)
    points = np.asarray(points, dtype=float)
    count = len(points)

    def optional_array(values, fill=np.nan):
        if values is None:
            return np.full(count, fill, dtype=float)
        return np.asarray(values, dtype=float)

    return ActionTable(
        name,
        float(cost),
        np.asarray(rewards, dtype=float),
        next_points,
        interpolator.prepare(next_points),
        next_points - points,
        kind,
        optional_array(base_reward_per_action, 0.0),
        optional_array(beta),
        optional_array(eta_on1),
        optional_array(eta_on2),
        optional_array(eta_off),
        optional_array(eta_wm),
        optional_array(prefix_length),
        K,
        prefix_length_value,
    )


def _online_tables(
    points: np.ndarray,
    interpolator: FeasibleInterpolator | StructuredInterpolator,
    params: Params,
    config: DPConfig,
    progress: Callable[[str], None] | None = None,
) -> list[ActionTable]:
    tables = []
    base_params = replace(params, M=0.0)
    specs = _online_action_specs(params, config)
    total = len(points) * len(specs)
    interval = max(1, total // 10)
    completed = 0
    for spec in specs:
        outcomes = []
        for point in points:
            outcomes.append(
                _online_outcome(
                    _state_from_point(point, config),
                    spec,
                    base_params,
                    config,
                )
            )
            completed += 1
            if progress is not None and (
                completed == total or completed % interval == 0
            ):
                progress(
                    f"Action tables: {completed}/{total} "
                    f"({100.0 * completed / total:.0f}%)"
                )
                sys.stdout.flush()
        tables.append(
            _action_table(
                spec.name,
                outcomes[0].cost,
                [outcome.reward for outcome in outcomes],
                [
                    _point_from_state(outcome.next_state, config)
                    for outcome in outcomes
                ],
                points,
                interpolator,
                "online",
                base_reward_per_action=[
                    outcome.base_reward_per_action for outcome in outcomes
                ],
                beta=[outcome.beta for outcome in outcomes],
                eta_on1=[outcome.eta_on1 for outcome in outcomes],
                eta_on2=[outcome.eta_on2 for outcome in outcomes],
                eta_wm=[outcome.eta_wm for outcome in outcomes],
                prefix_length=[outcome.prefix_length for outcome in outcomes],
                K=spec.K,
                prefix_length_value=spec.prefix_length,
            )
        )
    return tables


def _project_query_points(
    points: np.ndarray,
    config: DPConfig,
) -> np.ndarray:
    """Project time-scaled Euler destinations back to the feasible domain."""

    points = np.clip(np.asarray(points, dtype=float), -1.0, 1.0)
    if config.fixed_wm:
        a = points[:, 0]
        c = points[:, 1]
        denominator = a * a + c * c - 2.0 * config.fixed_rho_me * a * c
        radius = 1.0 - config.fixed_rho_me**2
        outside = (denominator > radius) & (denominator > 0.0)
        scale = np.ones(len(points))
        scale[outside] = np.sqrt(np.maximum(radius, 0.0) / denominator[outside])
        return points * scale[:, None]
    determinant = 1.0 + 2.0 * np.prod(points, axis=1) - np.sum(points * points, axis=1)
    outside = np.flatnonzero(determinant < -1e-10)
    if len(outside):
        points = points.copy()
        points[outside] = np.asarray(
            [State.from_array(points[index], project=True).as_array() for index in outside]
        )
    return points


def _timed_action_table(
    table: ActionTable,
    points: np.ndarray,
    interpolator: FeasibleInterpolator | StructuredInterpolator,
    params: Params,
    config: DPConfig,
    start_time: float,
) -> ActionTable:
    """Apply time modulation to a precomputed base action table."""

    if params.M == 0.0:
        return table
    end_time = start_time + table.cost
    if table.kind == "switch":
        exposure = integrated_environment_modulation(start_time, end_time, params)
        retained = np.exp(-params.eta_e * exposure)
        next_points = np.asarray(points, dtype=float).copy()
        if config.fixed_wm:
            next_points[:, 0] *= retained
        else:
            next_points[:, :2] *= retained
        next_points = _project_query_points(next_points, config)
        reward = table.reward
    else:
        modulation = average_environment_modulation(start_time, end_time, params)
        scale = np.full(table.delta.shape[1], modulation)
        if table.kind == "offline":
            scale[-1] = 1.0
        next_points = points + table.delta * scale
        next_points = _project_query_points(next_points, config)
        reward = table.reward * modulation if table.kind == "online" else table.reward
    return replace(
        table,
        reward=reward,
        next_points=next_points,
        interpolation=interpolator.prepare(next_points),
    )


def _future_values(
    values: np.ndarray,
    table: ActionTable,
    remaining: float,
    time_step: float,
) -> np.ndarray:
    position = max(remaining, 0.0) / time_step
    lower = int(np.floor(position + 1e-12))
    upper = int(np.ceil(position - 1e-12))
    fraction = position - lower
    low_values = table.interpolation.apply(values[lower])
    if upper == lower:
        return low_values
    high_values = table.interpolation.apply(values[upper])
    return (1.0 - fraction) * low_values + fraction * high_values


def solve_online_dp(
    params: Params,
    config: DPConfig,
    progress: Callable[[str], None] | None = None,
) -> DPResult:
    """Solve online-only control."""

    params.validate()
    config.validate(params, "online")
    points = _fixed_grid(config) if config.fixed_wm else _full_grid(config)
    interpolator = _make_interpolator(points, config)
    tables = _online_tables(points, interpolator, params, config, progress)
    times = np.arange(round(config.horizon / config.time_step) + 1) * config.time_step
    values = np.zeros((len(times), len(points)), dtype=float)
    policy = np.full((len(times), len(points)), -1, dtype=np.int16)
    total_layers = len(times) - 1
    interval = max(1, total_layers // 10)
    for index, horizon in enumerate(times[1:], start=1):
        start_time = config.horizon - horizon
        best = np.zeros(len(points), dtype=float)
        selected = np.full(len(points), -1, dtype=np.int16)
        for action_index, base_table in enumerate(tables):
            table = _timed_action_table(
                base_table,
                points,
                interpolator,
                params,
                config,
                start_time,
            )
            if table.cost > horizon + 1e-12:
                continue
            future = _future_values(
                values, table, horizon - table.cost, config.time_step
            )
            candidate = table.reward + future
            improve = candidate > best + 1e-14
            best[improve] = candidate[improve]
            selected[improve] = action_index
        values[index] = best
        policy[index] = selected
        if progress is not None and (
            index == total_layers or index % interval == 0
        ):
            progress(
                f"Bellman layers: {index}/{total_layers} "
                f"({100.0 * index / total_layers:.0f}%)"
            )
            sys.stdout.flush()
    return DPResult(
        "online",
        config,
        params,
        times,
        points,
        values,
        policy,
        tuple(table.name for table in tables),
        ("on",),
        interpolator,
        tuple(tables),
    )


def build_switching_action_workspace(
    params: Params,
    config: DPConfig,
    progress: Callable[[str], None] | None = None,
) -> DPActionWorkspace:
    """Build reusable action tables for the switching dynamic program."""

    params.validate()
    config.validate(params, "switching")
    points = _fixed_grid(config) if config.fixed_wm else _full_grid(config)
    interpolator = _make_interpolator(points, config)
    online = _online_tables(points, interpolator, params, config, progress)
    if progress is not None:
        progress("Building offline and switching action tables...")
    states = [_state_from_point(point, config) for point in points]
    base_params = replace(params, M=0.0)
    offline_outcomes = [
        _offline_outcome(state, base_params, config) for state in states
    ]
    offline = _action_table(
        "offline",
        offline_outcomes[0].cost,
        np.zeros(len(points)),
        [
            _point_from_state(outcome.next_state, config)
            for outcome in offline_outcomes
        ],
        points,
        interpolator,
        "offline",
        eta_off=[outcome.eta_off for outcome in offline_outcomes],
        prefix_length=[outcome.prefix_length for outcome in offline_outcomes],
        prefix_length_value=params.T,
    )
    switch_outcomes = [
        _switch_outcome(state, "on", base_params, config) for state in states
    ]
    switch = _action_table(
        "switch",
        switch_outcomes[0].cost,
        np.zeros(len(points)),
        [
            _point_from_state(outcome.next_state, config)
            for outcome in switch_outcomes
        ],
        points,
        interpolator,
        "switch",
    )
    tables = tuple(online) + (offline, switch)
    return DPActionWorkspace(
        config,
        params,
        points,
        tables,
        tuple(table.name for table in tables),
        ("on", "off"),
        interpolator,
    )


def solve_switching_workspace(
    workspace: DPActionWorkspace,
    *,
    allowed_online_K: tuple[int, ...] | None = None,
    allow_offline: bool = True,
    allow_switch: bool = True,
    progress: Callable[[str], None] | None = None,
) -> DPResult:
    """Solve a switching workspace with optional action restrictions."""

    params = workspace.params
    config = workspace.config
    points = workspace.points
    interpolator = workspace.interpolator
    online_indices = [
        index
        for index, table in enumerate(workspace.tables)
        if table.kind == "online"
        and (allowed_online_K is None or table.K in allowed_online_K)
    ]
    offline_indices = [
        index for index, table in enumerate(workspace.tables) if table.kind == "offline"
    ]
    switch_indices = [
        index for index, table in enumerate(workspace.tables) if table.kind == "switch"
    ]
    if allow_offline and len(offline_indices) != 1:
        raise ValueError("switching workspace must contain exactly one offline table")
    if allow_switch and len(switch_indices) != 1:
        raise ValueError("switching workspace must contain exactly one switch table")
    offline_index = offline_indices[0] if offline_indices else -1
    switch_index = switch_indices[0] if switch_indices else -1
    times = np.arange(round(config.horizon / config.time_step) + 1) * config.time_step
    values = np.zeros((2, len(times), len(points)), dtype=float)
    policy = np.full((2, len(times), len(points)), -1, dtype=np.int16)
    total_layers = len(times) - 1
    interval = max(1, total_layers // 10)
    for index, horizon in enumerate(times[1:], start=1):
        start_time = config.horizon - horizon
        best_on = np.zeros(len(points))
        selected_on = np.full(len(points), -1, dtype=np.int16)
        for action_index in online_indices:
            base_table = workspace.tables[action_index]
            table = _timed_action_table(
                base_table,
                points,
                interpolator,
                params,
                config,
                start_time,
            )
            if table.cost > horizon + 1e-12:
                continue
            future = _future_values(
                values[0], table, horizon - table.cost, config.time_step
            )
            candidate = table.reward + future
            improve = candidate > best_on + 1e-14
            best_on[improve] = candidate[improve]
            selected_on[improve] = action_index
        timed_switch = None
        if allow_switch:
            timed_switch = _timed_action_table(
                workspace.tables[switch_index],
                points,
                interpolator,
                params,
                config,
                start_time,
            )
        if timed_switch is not None and timed_switch.cost <= horizon + 1e-12:
            candidate = _future_values(
                values[1], timed_switch, horizon - timed_switch.cost, config.time_step
            )
            improve = candidate > best_on + 1e-14
            best_on[improve] = candidate[improve]
            selected_on[improve] = switch_index
        values[0, index] = best_on
        policy[0, index] = selected_on

        best_off = np.zeros(len(points))
        selected_off = np.full(len(points), -1, dtype=np.int16)
        if timed_switch is not None and timed_switch.cost <= horizon + 1e-12:
            candidate = _future_values(
                values[0],
                timed_switch,
                horizon - timed_switch.cost,
                config.time_step,
            )
            improve = candidate > best_off + _VALUE_TIE_TOL
            best_off[improve] = candidate[improve]
            selected_off[improve] = switch_index
        timed_offline = None
        if allow_offline:
            timed_offline = _timed_action_table(
                workspace.tables[offline_index],
                points,
                interpolator,
                params,
                config,
                start_time,
            )
        if timed_offline is not None and timed_offline.cost <= horizon + 1e-12:
            candidate = _future_values(
                values[1],
                timed_offline,
                horizon - timed_offline.cost,
                config.time_step,
            )
            improve = candidate > best_off + _VALUE_TIE_TOL
            best_off[improve] = candidate[improve]
            selected_off[improve] = offline_index
        values[1, index] = best_off
        policy[1, index] = selected_off
        if progress is not None and (
            index == total_layers or index % interval == 0
        ):
            progress(
                f"Bellman layers: {index}/{total_layers} "
                f"({100.0 * index / total_layers:.0f}%)"
            )
            sys.stdout.flush()
            
    return DPResult(
        "switching",
        config,
        params,
        times,
        points,
        values,
        policy,
        workspace.action_names,
        ("on", "off"),
        interpolator,
        workspace.tables,
    )


def solve_switching_dp(
    params: Params,
    config: DPConfig,
    progress: Callable[[str], None] | None = None,
) -> DPResult:
    """Solve joint online/offline control with switching costs."""

    workspace = build_switching_action_workspace(params, config, progress)
    return solve_switching_workspace(workspace, progress=progress)


def _value_at(result: DPResult, horizon: float, point, mode: str = "on") -> float:
    horizon = float(np.clip(horizon, 0.0, result.config.horizon))
    position = horizon / result.config.time_step
    lower = int(np.floor(position + 1e-12))
    upper = int(np.ceil(position - 1e-12))
    fraction = position - lower
    weights = result.interpolator.prepare(np.atleast_2d(point))
    table = result.values if result.values.ndim == 2 else result.values[result.mode_names.index(mode)]
    low = weights.apply(table[lower])[0]
    if upper == lower:
        return float(low)
    high = weights.apply(table[upper])[0]
    return float((1.0 - fraction) * low + fraction * high)


def value_at(
    result: DPResult,
    horizon: float,
    state: State | np.ndarray,
    mode: str = "on",
) -> float:
    """Interpolate a solved value table at a continuous state."""

    point = (
        _point_from_state(state, result.config)
        if isinstance(state, State)
        else state
    )
    return _value_at(result, horizon, point, mode)


def _tables_for_result(result: DPResult) -> tuple[ActionTable, ...]:
    if result.tables is not None:
        return result.tables
    if result.variant == "switching":
        return build_switching_action_workspace(
            result.params, result.config
        ).tables
    points = (
        _fixed_grid(result.config)
        if result.config.fixed_wm
        else _full_grid(result.config)
    )
    interpolator = _make_interpolator(points, result.config)
    return tuple(
        _online_tables(
            points,
            interpolator,
            result.params,
            result.config,
        )
    )


def _optional_float(value: float | np.ndarray | None) -> float | None:
    if value is None:
        return None
    scalar = float(np.asarray(value))
    return None if not np.isfinite(scalar) else scalar


def _interpolated_action_outcome(
    result: DPResult,
    table: ActionTable,
    state: State,
    mode: str,
    elapsed: float,
) -> ActionOutcome:
    point = _point_from_state(state, result.config)
    timed = _timed_action_table(
        table,
        result.points,
        result.interpolator,
        result.params,
        result.config,
        elapsed,
    )
    weights = result.interpolator.prepare(np.atleast_2d(point))
    next_point = weights.apply(timed.next_points)[0]
    next_state = _project_state(next_point, result.config)
    next_mode = "on"
    if table.kind == "offline":
        next_mode = "off"
    elif table.kind == "switch":
        next_mode = "off" if mode == "on" else "on"
    return ActionOutcome(
        table.name,
        timed.cost,
        float(weights.apply(timed.reward)[0]),
        next_state,
        base_reward_per_action=float(weights.apply(timed.base_reward_per_action)[0]),
        K=table.K,
        beta=_optional_float(weights.apply(timed.beta)[0]),
        eta_on1=_optional_float(weights.apply(timed.eta_on1)[0]),
        eta_on2=_optional_float(weights.apply(timed.eta_on2)[0]),
        eta_off=_optional_float(weights.apply(timed.eta_off)[0]),
        eta_wm=_optional_float(weights.apply(timed.eta_wm)[0]),
        prefix_length=_optional_float(weights.apply(timed.prefix_length)[0]),
        next_mode=next_mode,
    )


def rollout_policy(
    result: DPResult,
    initial: State,
    *,
    initial_mode: str = "on",
    rollout_step: float | None = None,
) -> list[dict]:
    """Roll out the Bellman policy from a continuous initial state."""

    rollout_step = result.config.time_step if rollout_step is None else rollout_step
    if rollout_step <= 0.0:
        raise ValueError("rollout_step must be positive")
    ratio = rollout_step / result.config.time_step
    if ratio < 1.0 - 1e-12 or abs(ratio - round(ratio)) > 1e-9:
        raise ValueError("rollout_step must be an integer multiple of DP time_step")
    if initial_mode not in result.mode_names:
        raise ValueError("initial_mode is not available for this DP variant")
    state = initial
    if result.config.fixed_wm:
        reduced = _project_fixed((state.rho_se, state.rho_sm), result.config.fixed_rho_me)
        state = State(reduced[0], result.config.fixed_rho_me, reduced[1])
        state.validate()
    else:
        state.validate()
    mode = initial_mode
    remaining = result.config.horizon
    elapsed = 0.0
    cumulative = 0.0
    records = []
    tables = _tables_for_result(result)
    while True:
        outcomes = []
        for table in tables:
            if table.cost > remaining + 1e-12:
                continue
            if result.variant == "switching":
                if mode == "on" and table.kind == "offline":
                    continue
                if mode == "off" and table.kind == "online":
                    continue
            outcomes.append(
                (
                    table,
                    _interpolated_action_outcome(
                        result, table, state, mode, elapsed
                    ),
                )
            )
        if not outcomes:
            records.append(
                {
                    "time": elapsed,
                    "remaining_time": remaining,
                    "rho_se": state.rho_se,
                    "rho_me": state.rho_me,
                    "rho_sm": state.rho_sm,
                    "mode": mode,
                    "action": "stop",
                    "K": "",
                    "beta": "",
                    "eta_on1": "",
                    "eta_on2": "",
                    "eta_off": "",
                    "eta_wm": "",
                    "prefix_length": "",
                    "immediate_reward": 0.0,
                    "action_integrated_reward": 0.0,
                    "cumulative_reward": cumulative,
                    "environment_modulation": average_environment_modulation(
                        elapsed, elapsed, result.params
                    ),
                    "decision_boundary": True,
                }
            )
            break
        scores = []
        for _, outcome in outcomes:
            next_point = _point_from_state(outcome.next_state, result.config)
            scores.append(
                outcome.reward
                + _value_at(
                    result,
                    remaining - outcome.cost,
                    next_point,
                    outcome.next_mode,
                )
            )
        best_score = float(np.max(scores))
        if best_score <= 1e-14:
            records.append(
                {
                    "time": elapsed,
                    "remaining_time": remaining,
                    "rho_se": state.rho_se,
                    "rho_me": state.rho_me,
                    "rho_sm": state.rho_sm,
                    "mode": mode,
                    "action": "stop",
                    "K": "",
                    "beta": "",
                    "eta_on1": "",
                    "eta_on2": "",
                    "eta_off": "",
                    "eta_wm": "",
                    "prefix_length": "",
                    "immediate_reward": 0.0,
                    "action_integrated_reward": 0.0,
                    "cumulative_reward": cumulative,
                    "environment_modulation": average_environment_modulation(
                        elapsed, elapsed, result.params
                    ),
                    "decision_boundary": True,
                }
            )
            break
        best_index = int(np.argmax(scores))
        best_score = scores[best_index]
        for index, (table, _) in enumerate(outcomes):
            if scores[index] < best_score - _VALUE_TIE_TOL:
                continue
            if (
                result.variant == "switching"
                and mode == "off"
                and table.kind == "switch"
            ):
                best_index = index
                best_score = scores[index]
                break
        selected_table, outcome = outcomes[best_index]
        records.append(
            {
                "time": elapsed,
                "remaining_time": remaining,
                "rho_se": state.rho_se,
                "rho_me": state.rho_me,
                "rho_sm": state.rho_sm,
                "mode": mode,
                "action": outcome.name,
                "K": "" if outcome.K is None else outcome.K,
                "beta": "" if outcome.beta is None else outcome.beta,
                "eta_on1": "" if outcome.eta_on1 is None else outcome.eta_on1,
                "eta_on2": "" if outcome.eta_on2 is None else outcome.eta_on2,
                "eta_off": "" if outcome.eta_off is None else outcome.eta_off,
                "eta_wm": "" if outcome.eta_wm is None else outcome.eta_wm,
                "prefix_length": "" if outcome.prefix_length is None else outcome.prefix_length,
                "immediate_reward": environment_modulation(
                    elapsed, result.params
                ) * outcome.base_reward_per_action,
                "action_integrated_reward": outcome.reward,
                "cumulative_reward": cumulative,
                "environment_modulation": average_environment_modulation(
                    elapsed, elapsed, result.params
                ),
                "decision_boundary": True,
            }
        )
        substep = rollout_step
        while substep < outcome.cost - 1e-12:
            if outcome.name == "switch":
                exposure = integrated_environment_modulation(
                    elapsed, elapsed + substep, result.params
                )
                retained = float(np.exp(-result.params.eta_e * exposure))
                intermediate = _project_state(
                    np.array(
                        [
                            retained * state.rho_se,
                            retained * state.rho_me,
                            state.rho_sm,
                        ]
                    ),
                    result.config,
                )
            else:
                fraction = substep / outcome.cost
                partial_values = state.as_array() + fraction * (
                    outcome.next_state.as_array() - state.as_array()
                )
                intermediate = _project_state(
                    partial_values,
                    result.config,
                )
            display = _interpolated_action_outcome(
                result,
                selected_table,
                intermediate,
                mode,
                elapsed + substep,
            )
            records.append(
                {
                    "time": elapsed + substep,
                    "remaining_time": remaining - substep,
                    "rho_se": intermediate.rho_se,
                    "rho_me": intermediate.rho_me,
                    "rho_sm": intermediate.rho_sm,
                    "mode": mode,
                    "action": outcome.name,
                    "K": "" if display.K is None else display.K,
                    "beta": "" if display.beta is None else display.beta,
                    "eta_on1": "" if display.eta_on1 is None else display.eta_on1,
                    "eta_on2": "" if display.eta_on2 is None else display.eta_on2,
                    "eta_off": "" if display.eta_off is None else display.eta_off,
                    "eta_wm": "" if display.eta_wm is None else display.eta_wm,
                    "prefix_length": "" if display.prefix_length is None else display.prefix_length,
                    "immediate_reward": environment_modulation(
                        elapsed + substep, result.params
                    ) * display.base_reward_per_action,
                    "action_integrated_reward": display.reward,
                    "cumulative_reward": cumulative,
                    "environment_modulation": average_environment_modulation(
                        elapsed + substep,
                        elapsed + substep,
                        result.params,
                    ),
                    "decision_boundary": False,
                }
            )
            substep += rollout_step
        cumulative += outcome.reward
        elapsed += outcome.cost
        remaining = max(remaining - outcome.cost, 0.0)
        state = outcome.next_state
        mode = outcome.next_mode
    return records


def save_dp_result(path: str | Path, result: DPResult) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = json.dumps(
        {
            "variant": result.variant,
            "config": asdict(result.config),
            "params": asdict(result.params),
            "action_names": result.action_names,
            "mode_names": result.mode_names,
        }
    )
    np.savez_compressed(
        path,
        metadata=metadata,
        times=result.times,
        points=result.points,
        values=result.values,
        policy=result.policy,
    )


def save_switching_value_map_cache(
    path: str | Path,
    optimal: DPResult,
    online_only: DPResult,
    k1_online: DPResult,
    k1_switching: DPResult | None = None,
) -> None:
    """Save switching value maps together with the action tables needed for rollout."""

    if optimal.variant != "switching":
        raise ValueError("switching cache requires switching DP results")
    if online_only.variant != "switching" or k1_online.variant != "switching":
        raise ValueError("all cached DP results must use the switching variant")
    if k1_switching is not None and k1_switching.variant != "switching":
        raise ValueError("all cached DP results must use the switching variant")
    tables = _tables_for_result(optimal)
    table_specs = []
    arrays = {
        "times": optimal.times,
        "points": optimal.points,
        "optimal_values": optimal.values,
        "optimal_policy": optimal.policy,
        "online_only_values": online_only.values,
        "online_only_policy": online_only.policy,
        "k1_online_values": k1_online.values,
        "k1_online_policy": k1_online.policy,
    }
    if k1_switching is not None:
        arrays["k1_switching_values"] = k1_switching.values
        arrays["k1_switching_policy"] = k1_switching.policy
    for index, table in enumerate(tables):
        table_specs.append(
            {
                "name": table.name,
                "cost": table.cost,
                "kind": table.kind,
                "K": table.K,
                "prefix_length_value": table.prefix_length_value,
            }
        )
        prefix = f"table_{index}"
        arrays[f"{prefix}_reward"] = table.reward
        arrays[f"{prefix}_next_points"] = table.next_points
        arrays[f"{prefix}_interpolation_vertices"] = table.interpolation.vertices
        arrays[f"{prefix}_interpolation_weights"] = table.interpolation.weights
        arrays[f"{prefix}_delta"] = table.delta
        arrays[f"{prefix}_base_reward_per_action"] = table.base_reward_per_action
        arrays[f"{prefix}_beta"] = table.beta
        arrays[f"{prefix}_eta_on1"] = table.eta_on1
        arrays[f"{prefix}_eta_on2"] = table.eta_on2
        arrays[f"{prefix}_eta_off"] = table.eta_off
        arrays[f"{prefix}_eta_wm"] = table.eta_wm
        arrays[f"{prefix}_prefix_length"] = table.prefix_length
    metadata = json.dumps(
        {
            "variant": optimal.variant,
            "config": asdict(optimal.config),
            "params": asdict(optimal.params),
            "action_names": optimal.action_names,
            "mode_names": optimal.mode_names,
            "has_k1_switching": k1_switching is not None,
            "table_specs": table_specs,
        }
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, metadata=metadata, **arrays)


def load_switching_value_map_cache(
    path: str | Path,
    *,
    rebuild_missing_tables: bool = False,
    solve_missing_k1_switching: bool = False,
    include_k1_switching: bool = False,
    progress: Callable[[str], None] | None = None,
) -> (
    tuple[DPResult, DPResult, DPResult]
    | tuple[DPResult, DPResult, DPResult, DPResult | None]
):
    """Load switching value maps and cached action tables from disk."""

    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        config = DPConfig(**metadata["config"])
        params = Params(**metadata["params"])
        points = np.array(data["points"])
        times = np.array(data["times"])
        optimal_values = np.array(data["optimal_values"])
        optimal_policy = np.array(data["optimal_policy"])
        online_only_values = np.array(data["online_only_values"])
        online_only_policy = np.array(data["online_only_policy"])
        k1_online_values = np.array(data["k1_online_values"])
        k1_online_policy = np.array(data["k1_online_policy"])
        has_k1_switching = (
            "k1_switching_values" in data and "k1_switching_policy" in data
        )
        if has_k1_switching:
            k1_switching_values = np.array(data["k1_switching_values"])
            k1_switching_policy = np.array(data["k1_switching_policy"])
        else:
            k1_switching_values = None
            k1_switching_policy = None
        action_names = tuple(metadata["action_names"])
        mode_names = tuple(metadata["mode_names"])
        table_specs = metadata.get("table_specs")
        if table_specs is None:
            if not rebuild_missing_tables:
                raise KeyError("cached value map does not include action tables")
            if progress is not None:
                progress("Cached value map has no action tables; rebuilding them once...")
            workspace = build_switching_action_workspace(params, config, progress)
            interpolator = workspace.interpolator
            tables = workspace.tables
        else:
            interpolator = _make_interpolator(points, config)
            tables = []
            for index, spec in enumerate(table_specs):
                prefix = f"table_{index}"
                prefix_lengths = (
                    np.array(data[f"{prefix}_prefix_length"])
                    if f"{prefix}_prefix_length" in data
                    else np.full(len(points), np.nan)
                )
                tables.append(
                    ActionTable(
                        spec["name"],
                        spec["cost"],
                        np.array(data[f"{prefix}_reward"]),
                        np.array(data[f"{prefix}_next_points"]),
                        InterpolationWeights(
                            np.array(data[f"{prefix}_interpolation_vertices"]),
                            np.array(data[f"{prefix}_interpolation_weights"]),
                        ),
                        np.array(data[f"{prefix}_delta"]),
                        spec["kind"],
                        np.array(data[f"{prefix}_base_reward_per_action"]),
                        np.array(data[f"{prefix}_beta"]),
                        np.array(data[f"{prefix}_eta_on1"]),
                        np.array(data[f"{prefix}_eta_on2"]),
                        np.array(data[f"{prefix}_eta_off"]),
                        np.array(data[f"{prefix}_eta_wm"]),
                        prefix_lengths,
                        spec["K"],
                        spec.get("prefix_length_value"),
                    )
                )
            tables = tuple(tables)
    optimal = DPResult(
        "switching",
        config,
        params,
        times,
        points,
        optimal_values,
        optimal_policy,
        action_names,
        mode_names,
        interpolator,
        tables,
    )
    online_only = DPResult(
        "switching",
        config,
        params,
        times,
        points,
        online_only_values,
        online_only_policy,
        action_names,
        mode_names,
        interpolator,
        tables,
    )
    k1_online = DPResult(
        "switching",
        config,
        params,
        times,
        points,
        k1_online_values,
        k1_online_policy,
        action_names,
        mode_names,
        interpolator,
        tables,
    )
    if k1_switching_values is None or k1_switching_policy is None:
        k1_switching = None
    else:
        k1_switching = DPResult(
            "switching",
            config,
            params,
            times,
            points,
            k1_switching_values,
            k1_switching_policy,
            action_names,
            mode_names,
            interpolator,
            tables,
        )
    if k1_switching is None and solve_missing_k1_switching:
        if progress is not None:
            progress("Cached value map has no K=1 switching solution; solving it...")
        workspace = DPActionWorkspace(
            config,
            params,
            points,
            tables,
            action_names,
            mode_names,
            interpolator,
        )
        k1_switching = solve_switching_workspace(
            workspace,
            allowed_online_K=(1,),
            progress=progress,
        )
    if include_k1_switching:
        return optimal, online_only, k1_online, k1_switching
    return optimal, online_only, k1_online


def load_dp_result(path: str | Path) -> DPResult:
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        points = np.array(data["points"])
        config = DPConfig(**metadata["config"])
        return DPResult(
            metadata["variant"],
            config,
            Params(**metadata["params"]),
            np.array(data["times"]),
            points,
            np.array(data["values"]),
            np.array(data["policy"]),
            tuple(metadata["action_names"]),
            tuple(metadata["mode_names"]),
            _make_interpolator(points, config),
        )
