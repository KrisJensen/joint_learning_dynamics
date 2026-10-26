"""Cyclic modulation of environment-dependent rates and reward intensity."""

from __future__ import annotations

from math import pi, sin

from .config import Params


def cycle_value(time: float, params: Params) -> float:
    """Return ``1 + C_A sin(2 pi (t/C_T + C_phi))``."""

    phase = 2.0 * pi * (float(time) / params.C_T + params.C_phi)
    return float(1.0 + params.C_A * sin(phase))


def integrated_cycle(start: float, end: float, params: Params) -> float:
    """Integrate the cycle exactly over an elapsed-time interval."""

    start = float(start)
    end = float(end)
    if end < start:
        raise ValueError("cycle interval end precedes its start")
    duration = end - start
    if duration == 0.0 or params.C_A == 0.0:
        return duration
    omega = 2.0 * pi / params.C_T
    phase = 2.0 * pi * params.C_phi
    half_width = 0.5 * omega * duration
    sinc = 1.0 if half_width == 0.0 else sin(half_width) / half_width
    midpoint_phase = omega * (start + end) / 2.0 + phase
    mean = 1.0 + params.C_A * sin(midpoint_phase) * sinc
    return float(duration * max(mean, 0.0))


def average_cycle(start: float, end: float, params: Params) -> float:
    """Return the exact interval-average cycle value."""

    duration = float(end) - float(start)
    if duration == 0.0:
        return cycle_value(start, params)
    value = integrated_cycle(start, end, params) / duration
    return float(np_clip(value, 1.0 - params.C_A, 1.0 + params.C_A))


def np_clip(value: float, low: float, high: float) -> float:
    """Small scalar clip helper that avoids importing NumPy in this module."""

    return min(max(value, low), high)
