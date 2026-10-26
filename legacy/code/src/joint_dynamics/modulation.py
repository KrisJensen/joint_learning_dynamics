"""Global time modulation of environment-dependent dynamics."""

from __future__ import annotations

from math import pi, sin

from .config import Params


def environment_modulation(time: float, params: Params) -> float:
    """Return ``1 + M sin(2 pi (t / T_M + phi_M))``."""

    phase = 2.0 * pi * (float(time) / params.T_M + params.phi_M)
    return float(1.0 + params.M * sin(phase))


def integrated_environment_modulation(
    start: float,
    end: float,
    params: Params,
) -> float:
    """Return the exact integral of the modulation over an elapsed interval."""

    start = float(start)
    end = float(end)
    if end < start:
        raise ValueError("modulation interval end must not precede its start")
    duration = end - start
    if duration == 0.0 or params.M == 0.0:
        return duration
    omega = 2.0 * pi / params.T_M
    phase = 2.0 * pi * params.phi_M
    half_width = 0.5 * omega * duration
    sinc = 1.0 if half_width == 0.0 else sin(half_width) / half_width
    midpoint_phase = omega * (0.5 * (start + end)) + phase
    mean = 1.0 + params.M * sin(midpoint_phase) * sinc
    # The analytical integral is nonnegative for 0 <= M <= 1. Guard against
    # roundoff at a zero-modulation minimum before it is used as a drift time.
    return float(duration * max(mean, 0.0))


def average_environment_modulation(
    start: float,
    end: float,
    params: Params,
) -> float:
    """Return the interval-average modulation, or its point value at zero width."""

    duration = float(end) - float(start)
    if duration == 0.0:
        return environment_modulation(start, params)
    mean = integrated_environment_modulation(start, end, params) / duration
    return float(min(max(mean, 1.0 - params.M), 1.0 + params.M))
