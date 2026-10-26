"""Validated configuration objects for the learning-dynamics model."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable, Literal

import numpy as np


ControlMode = Literal["constant", "reward", "drift"]
LearningMode = Literal["constant", "optimal"]
TemperatureMode = Literal["constant", "optimal"]
PhaseMode = Literal["online", "offline"]
_STATE_EPS = 1e-12


@dataclass(frozen=True)
class State:
    """The three pairwise correlations ``(rho_se, rho_me, rho_sm)``."""

    rho_se: float
    rho_me: float
    rho_sm: float

    @property
    def matrix(self) -> np.ndarray:
        """Return the correlation matrix of ``(U, H, V)``."""

        return np.array(
            [
                [1.0, self.rho_se, self.rho_me],
                [self.rho_se, 1.0, self.rho_sm],
                [self.rho_me, self.rho_sm, 1.0],
            ],
            dtype=float,
        )

    @property
    def determinant(self) -> float:
        a, b, c = self.rho_se, self.rho_me, self.rho_sm
        return 1.0 + 2.0 * a * b * c - a * a - b * b - c * c

    def as_array(self) -> np.ndarray:
        return np.array([self.rho_se, self.rho_me, self.rho_sm], dtype=float)

    def validate(self, *, tolerance: float = 1e-9) -> None:
        values = (self.rho_se, self.rho_me, self.rho_sm)
        if not all(isfinite(value) for value in values):
            raise ValueError("state correlations must be finite")
        if any(abs(value) > 1.0 + tolerance for value in values):
            raise ValueError("state correlations must lie in [-1, 1]")
        if self.determinant < -tolerance:
            raise ValueError("state correlation matrix must be positive semidefinite")

    def projected(self) -> "State":
        """Project to a positive-semidefinite unit-diagonal correlation matrix."""

        matrix = 0.5 * (self.matrix + self.matrix.T)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        eigenvalues = np.maximum(eigenvalues, _STATE_EPS)
        positive = (eigenvectors * eigenvalues) @ eigenvectors.T
        scales = np.sqrt(np.maximum(np.diag(positive), _STATE_EPS))
        correlation = positive / np.outer(scales, scales)
        np.fill_diagonal(correlation, 1.0)
        state = State(
            float(np.clip(correlation[0, 1], -1.0, 1.0)),
            float(np.clip(correlation[0, 2], -1.0, 1.0)),
            float(np.clip(correlation[1, 2], -1.0, 1.0)),
        )
        state.validate(tolerance=1e-8)
        return state

    @classmethod
    def from_array(cls, values, *, project: bool = False) -> "State":
        array = np.asarray(values, dtype=float)
        if array.shape != (3,):
            raise ValueError("a state array must contain exactly three values")
        state = cls(*map(float, array))
        if project:
            state = state.projected()
        state.validate(tolerance=1e-8 if project else 1e-9)
        return state


@dataclass(frozen=True)
class Params:
    """Model, control, elapsed-cost, and numerical parameters.

    ``delta_on`` and ``delta_off`` are elapsed costs. Their inverses are
    exposed as derived properties. By default, a simulated prefix element
    costs ``delta_on / T``. With ``constant_prefix_time=True``, every complete
    prefix costs ``delta_on`` instead, independently of its length.
    """

    T: int = 5
    M: int = 5
    M_mode: ControlMode = "constant"
    K: int = 5
    K_mode: ControlMode = "constant"
    K_max: int = 30
    beta: float = 3.0
    temperature_mode: TemperatureMode = "constant"
    beta_max: float = 30.0
    beta_tol: float = 1e-6
    eta_2f: float = 1.0
    eta_3f: float = 1.0
    eta_off: float = 1.0
    eta_wm: float = 1.0
    learning_rate_mode: LearningMode = "constant"
    fix_eta_2f_zero: bool = False
    fix_eta_3f_zero: bool = False
    fix_eta_off_zero: bool = False
    fix_eta_wm_zero: bool = False
    eta_max: float = 20.0
    eta_2f_max: float | None = None
    eta_3f_max: float | None = None
    eta_off_max: float | None = None
    eta_wm_max: float | None = None
    epsilon: float = 0.0
    p_flip: float = 0.0
    eta_e: float = 0.0
    delta_on: float = 0.1
    delta_off: float = 0.1
    C_A: float = 0.0
    C_T: float = 1.0
    C_phi: float = 0.0
    greedy_training: bool = False
    greedy_evaluation: bool = False
    gh_order: int = 24
    constant_prefix_time: bool = False

    @property
    def delta_epsilon(self) -> float:
        return 1.0 - 2.0 * self.epsilon

    @property
    def mu_flip(self) -> float:
        return 1.0 - 2.0 * self.p_flip

    @staticmethod
    def _inverse_cost(value: float) -> float:
        return float("inf") if value == 0.0 else 1.0 / value

    @property
    def tau_on(self) -> float:
        return self._inverse_cost(self.delta_on)

    @property
    def tau_off(self) -> float:
        return self._inverse_cost(self.delta_off)

    @property
    def delta_element(self) -> float:
        """Return the elementary ``delta_on / T`` reference-grid cost."""

        return self.delta_on / self.T

    def prefix_time(self, M: int | None = None) -> float:
        """Return elapsed simulation time for one candidate prefix."""

        active_M = self.M if M is None else int(M)
        if self.constant_prefix_time:
            return self.delta_on
        return active_M * self.delta_element

    def online_cost(self, K: int | None = None, M: int | None = None) -> float:
        """Elapsed cost of one online microscopic update."""

        active_K = self.K if K is None else int(K)
        return 1.0 + active_K * self.prefix_time(M)

    @property
    def c_on(self) -> float:
        return self.online_cost()

    @property
    def c_off(self) -> float:
        return self.delta_off

    def rate_cap(self, name: str) -> float:
        specific = getattr(self, f"{name}_max")
        return float(self.eta_max if specific is None else specific)

    def validate(self) -> None:
        if not isinstance(self.T, int) or self.T < 1:
            raise ValueError("T must be a positive integer")
        if not isinstance(self.M, int) or not 1 <= self.M <= self.T:
            raise ValueError("M must be an integer in [1, T]")
        if not isinstance(self.K, int) or self.K < 1:
            raise ValueError("K must be a positive integer")
        if not isinstance(self.K_max, int) or self.K_max < 1:
            raise ValueError("K_max must be a positive integer")
        modes = {"constant", "reward", "drift"}
        if self.K_mode not in modes or self.M_mode not in modes:
            raise ValueError("K_mode and M_mode must be constant, reward, or drift")
        adaptive = {mode for mode in (self.K_mode, self.M_mode) if mode != "constant"}
        if len(adaptive) > 1:
            raise ValueError("adaptive K and M must optimize the same objective")
        if self.temperature_mode not in {"constant", "optimal"}:
            raise ValueError("temperature_mode must be constant or optimal")
        if self.learning_rate_mode not in {"constant", "optimal"}:
            raise ValueError("learning_rate_mode must be constant or optimal")
        if self.beta < 0.0 or self.beta_max <= 0.0 or self.beta_tol <= 0.0:
            raise ValueError("beta values and tolerances must be nonnegative/positive")
        rates = (self.eta_2f, self.eta_3f, self.eta_off, self.eta_wm)
        if any(not isfinite(rate) or rate < 0.0 for rate in rates):
            raise ValueError("learning rates must be finite and nonnegative")
        if not isfinite(self.eta_max) or self.eta_max < 0.0:
            raise ValueError("eta_max must be finite and nonnegative")
        for name in ("eta_2f_max", "eta_3f_max", "eta_off_max", "eta_wm_max"):
            value = getattr(self, name)
            if value is not None and (not isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must lie in [0, 1]")
        if not 0.0 <= self.p_flip <= 1.0:
            raise ValueError("p_flip must lie in [0, 1]")
        if self.eta_e < 0.0 or not isfinite(self.eta_e):
            raise ValueError("eta_e must be finite and nonnegative")
        if self.delta_on < 0.0:
            raise ValueError("delta_on must be nonnegative")
        if self.delta_off <= 0.0:
            raise ValueError("delta_off must be positive")
        if not 0.0 <= self.C_A <= 1.0:
            raise ValueError("C_A must lie in [0, 1]")
        if self.C_T <= 0.0 or not isfinite(self.C_T) or not isfinite(self.C_phi):
            raise ValueError("C_T must be positive and C_phi finite")
        if self.gh_order < 4:
            raise ValueError("gh_order must be at least four")


@dataclass(frozen=True)
class Phase:
    """One online or offline schedule phase."""

    mode: PhaseMode
    duration: float

    def validate(self) -> None:
        if self.mode not in {"online", "offline"}:
            raise ValueError("phase mode must be online or offline")
        if not isfinite(self.duration) or self.duration < 0.0:
            raise ValueError("phase duration must be finite and nonnegative")


@dataclass(frozen=True)
class Schedule:
    """An ordered tuple of learning phases."""

    phases: tuple[Phase, ...]

    def __init__(self, phases: Iterable[Phase]):
        values = tuple(phases)
        if not values:
            raise ValueError("a schedule requires at least one phase")
        for phase in values:
            phase.validate()
        object.__setattr__(self, "phases", values)

    @property
    def total_duration(self) -> float:
        return float(sum(phase.duration for phase in self.phases))

    @classmethod
    def single(cls, mode: PhaseMode, duration: float) -> "Schedule":
        return cls((Phase(mode, duration),))
