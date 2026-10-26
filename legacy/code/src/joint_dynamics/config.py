"""Validated configuration objects for the joint learning dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

import numpy as np


_STATE_EPS = 1e-10
_MODES = {"online", "offline"}


@dataclass(frozen=True)
class State:
    """Pairwise correlations of student, world model, and environment."""

    rho_se: float
    rho_me: float
    rho_sm: float

    @property
    def matrix(self) -> np.ndarray:
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

    def validate(self) -> None:
        values = (self.rho_se, self.rho_me, self.rho_sm)
        if not all(isfinite(value) for value in values):
            raise ValueError("state correlations must be finite")
        if any(abs(value) > 1.0 + 1e-9 for value in values):
            raise ValueError("state correlations must lie in [-1, 1]")
        if self.determinant < -1e-8:
            raise ValueError("state correlation matrix must be positive semidefinite")

    def projected(self) -> "State":
        matrix = 0.5 * (self.matrix + self.matrix.T)
        values, vectors = np.linalg.eigh(matrix)
        values = np.maximum(values, _STATE_EPS)
        psd = (vectors * values) @ vectors.T
        scale = np.sqrt(np.maximum(np.diag(psd), _STATE_EPS))
        corr = psd / np.outer(scale, scale)
        np.fill_diagonal(corr, 1.0)
        return State(
            float(np.clip(corr[0, 1], -1.0 + _STATE_EPS, 1.0 - _STATE_EPS)),
            float(np.clip(corr[0, 2], -1.0 + _STATE_EPS, 1.0 - _STATE_EPS)),
            float(np.clip(corr[1, 2], -1.0 + _STATE_EPS, 1.0 - _STATE_EPS)),
        )

    @classmethod
    def from_array(cls, values, *, project: bool = False) -> "State":
        array = np.asarray(values, dtype=float)
        if array.shape != (3,):
            raise ValueError("state vector must have length three")
        state = cls(*map(float, array))
        if project:
            state = state.projected()
        state.validate()
        return state


@dataclass(frozen=True)
class Params:
    """Model, numerical, and control parameters."""

    T: int = 5
    prefix_length: int | None = None
    prefix_length_mode: str = "constant"
    prefix_length_max: int | None = None
    K: int = 10
    K_mode: str = "constant"
    K_max: int = 30
    beta: float = 3.0
    temperature_mode: str = "constant"
    greedy_training: bool = False
    greedy_evaluation: bool = False
    lambda_s: float = 0.25
    lambda_m: float = 0.25
    eta_on1: float = 1.0
    eta_on2: float = 1.0
    eta_off: float = 1.0
    eta_wm: float = 1.0
    learning_rate_mode: str = "constant"
    fix_eta_on1_zero: bool = False
    fix_eta_on2_zero: bool = False
    fix_eta_wm_zero: bool = False
    eta_max: float = 20.0
    eta_on1_max: float | None = None
    eta_on2_max: float | None = None
    eta_off_max: float | None = None
    eta_wm_max: float | None = None
    epsilon: float = 0.0
    p_flip: float = 0.0
    eta_e: float = 0.0
    M: float = 0.0
    T_M: float = 1.0
    phi_M: float = 0.0
    partial_evaluation: bool = False
    tau_element: float | None = None
    tau_on: float = 0.0
    tau_off: float = 1.0
    gh_order: int = 20
    beta_max: float = 20.0
    beta_tol: float = 1e-5

    @property
    def delta_epsilon(self) -> float:
        return 1.0 - 2.0 * self.epsilon

    @property
    def mu_flip(self) -> float:
        return 1.0 - 2.0 * self.p_flip

    @property
    def effective_prefix_length(self) -> int:
        return self.T if self.prefix_length is None else self.prefix_length

    @property
    def effective_tau_element(self) -> float:
        return self.tau_on / self.T if self.tau_element is None else self.tau_element

    @property
    def c_on(self) -> float:
        if self.partial_evaluation:
            return 1.0 + self.K * self.effective_prefix_length * self.effective_tau_element
        return 1.0 + self.K * self.tau_on

    @property
    def c_off(self) -> float:
        return self.tau_off

    def rate_cap(self, name: str) -> float:
        specific = getattr(self, f"{name}_max")
        return float(self.eta_max if specific is None else specific)

    def validate(self) -> None:
        if self.T < 1 or self.K < 1 or self.K_max < 1:
            raise ValueError("T, K, and K_max must be at least one")
        if self.prefix_length is not None and not (
            isinstance(self.prefix_length, int) and 1 <= self.prefix_length <= self.T
        ):
            raise ValueError("prefix_length must be an integer in [1, T]")
        if self.prefix_length_mode not in {"constant", "optimal"}:
            raise ValueError("prefix_length_mode must be 'constant' or 'optimal'")
        if self.prefix_length_max is not None and not (
            isinstance(self.prefix_length_max, int)
            and 1 <= self.prefix_length_max <= self.T
        ):
            raise ValueError("prefix_length_max must be an integer in [1, T]")
        if self.K_mode not in {"constant", "reward", "drift"}:
            raise ValueError("K_mode must be 'constant', 'reward', or 'drift'")
        if not self.beta >= 0.0 or self.beta_max <= 0.0 or self.beta_tol <= 0.0:
            raise ValueError("temperature values and tolerances must be valid")
        if self.temperature_mode not in {"constant", "optimal"}:
            raise ValueError("temperature_mode must be 'constant' or 'optimal'")
        if self.learning_rate_mode not in {"constant", "optimal"}:
            raise ValueError("learning_rate_mode must be 'constant' or 'optimal'")
        rates = (self.eta_on1, self.eta_on2, self.eta_off, self.eta_wm)
        if any(rate < 0.0 or not isfinite(rate) for rate in rates):
            raise ValueError("constant learning rates must be finite and nonnegative")
        if self.eta_max < 0.0:
            raise ValueError("eta_max must be nonnegative")
        for name in ("eta_on1_max", "eta_on2_max", "eta_off_max", "eta_wm_max"):
            value = getattr(self, name)
            if value is not None and value < 0.0:
                raise ValueError(f"{name} must be nonnegative or None")
        if self.lambda_s < 0.0 or self.lambda_m < 0.0:
            raise ValueError("score decay parameters must be nonnegative")
        if not 0.0 <= self.epsilon <= 1.0 or not 0.0 <= self.p_flip <= 1.0:
            raise ValueError("epsilon and p_flip must lie in [0, 1]")
        if self.eta_e < 0.0 or self.tau_on < 0.0:
            raise ValueError("eta_e and tau_on must be nonnegative")
        if not isinstance(self.partial_evaluation, bool):
            raise ValueError("partial_evaluation must be boolean")
        if self.tau_element is not None and (
            self.tau_element <= 0.0 or not isfinite(self.tau_element)
        ):
            raise ValueError("tau_element must be positive or None")
        if self.partial_evaluation:
            if self.effective_tau_element <= 0.0:
                raise ValueError("partial evaluation requires positive effective tau_element")
        if not 0.0 <= self.M <= 1.0 or not isfinite(self.M):
            raise ValueError("M must lie in [0, 1]")
        if self.T_M <= 0.0 or not isfinite(self.T_M) or not isfinite(self.phi_M):
            raise ValueError("T_M must be finite and positive and phi_M finite")
        if self.tau_off <= 0.0:
            raise ValueError("tau_off must be positive; zero is a singular limit")
        if self.gh_order < 4:
            raise ValueError("gh_order must be at least four")


@dataclass(frozen=True)
class Phase:
    mode: str
    duration: float

    def validate(self) -> None:
        validate_mode(self.mode)
        if not isfinite(self.duration) or self.duration < 0.0:
            raise ValueError("phase duration must be finite and nonnegative")


@dataclass(frozen=True)
class Schedule:
    phases: tuple[Phase, ...]

    def __init__(self, phases: Iterable[Phase]):
        object.__setattr__(self, "phases", tuple(phases))
        if not self.phases:
            raise ValueError("schedule must have at least one phase")
        for phase in self.phases:
            phase.validate()

    @property
    def total_duration(self) -> float:
        return float(sum(phase.duration for phase in self.phases))

    @classmethod
    def single(cls, mode: str, duration: float) -> "Schedule":
        return cls([Phase(mode, duration)])


def validate_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError("mode must be 'online' or 'offline'")
