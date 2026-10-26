"""Generate progressive reveals of a joint-learning DP trajectory.

The three figures successively show the world-model correlation, the student
correlation, and a smoothed version of the optimal planning horizon ``K``.
Correlations use the left axis and ``K`` uses the right axis.  All figures reuse
the limits, ticks, and plotting area of the complete figure so that the curves
can be revealed without the axes moving between presentation slides.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "presentations/figures"

# Use writable caches when running on a compute node with a read-only home.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# RHO_ME_COLOR = "#8c8c8c"
# RHO_SE_COLOR = "black"
# K_COLOR = "#1f77b4"
# OFFLINE_SHADE = "#dbeafe"
# SWITCH_SHADE = "#60a5fa"
RHO_ME_COLOR = "#F2AA84"
RHO_SE_COLOR = "#1f77b4"
K_COLOR = "black"
OFFLINE_SHADE = "#cccccc"
SWITCH_SHADE = "#939393"
LINE_WIDTH = 2.4


def reveal_specs(variant: str) -> tuple:
    """Return progressive output specifications for a DP variant."""

    prefix = f"dynamic_programming_{variant}_joint_rho_se"
    return (
        (f"{prefix}_1_rho_me.pdf", False, False),
        (f"{prefix}_2_rho_me_rho_se.pdf", True, False),
        (f"{prefix}_3_rho_me_rho_se_K.pdf", True, True),
    )


def default_input(variant: str) -> Path:
    """Return the report rollout used for ``variant``."""

    return (
        REPOSITORY_ROOT
        / f"results/report/dynamic_programming_{variant}_joint.csv"
    )


def read_records(path: Path) -> list[dict[str, str]]:
    """Read the dynamic-programming rollout records from ``path``."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def finite_series(
    records: list[dict[str, str]], metric: str
) -> tuple[np.ndarray, np.ndarray]:
    """Return finite times and values for a rollout metric."""

    points = []
    for row in records:
        if row.get("time", "") in {"", None} or row.get(metric, "") in {"", None}:
            continue
        time = float(row["time"])
        value = float(row[metric])
        if math.isfinite(time) and math.isfinite(value):
            points.append((time, value))
    if not points:
        raise ValueError(f"no finite {metric} values found in the rollout")
    points.sort(key=lambda point: point[0])
    return (
        np.asarray([point[0] for point in points], dtype=np.float64),
        np.asarray([point[1] for point in points], dtype=np.float64),
    )


def smoothed_step_series(
    records: list[dict[str, str]],
    metric: str,
    sigma: float,
    sample_count: int = 2001,
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth a piecewise-constant action metric in elapsed time.

    The action selected at a rollout row applies until the next row.  It is
    therefore first reconstructed with zero-order interpolation on a uniform
    time grid.  A Gaussian kernel is then applied to that grid.  Edge padding
    prevents the smoother from inventing drops at the episode boundaries.
    """

    if sigma <= 0.0:
        raise ValueError("smoothing sigma must be positive")
    row_points = []
    for row in records:
        if row.get("time", "") in {"", None}:
            continue
        time = float(row["time"])
        if not math.isfinite(time):
            continue
        raw_value = row.get(metric, "")
        value = np.nan if raw_value in {"", None} else float(raw_value)
        row_points.append((time, value if math.isfinite(value) else np.nan))
    if not row_points or not any(math.isfinite(point[1]) for point in row_points):
        raise ValueError(f"no finite {metric} values found in the rollout")
    row_points.sort(key=lambda point: point[0])
    row_times = np.asarray([point[0] for point in row_points], dtype=np.float64)
    row_values = np.asarray([point[1] for point in row_points], dtype=np.float64)
    start_time = float(row_times[0])
    end_time = float(row_times[-1])
    if end_time <= start_time:
        raise ValueError("the rollout must span a positive elapsed time")

    grid_times = np.linspace(start_time, end_time, sample_count, dtype=np.float64)
    indices = np.searchsorted(row_times, grid_times, side="right") - 1
    indices = np.clip(indices, 0, len(row_values) - 1)
    step_values = row_values[indices]

    grid_spacing = float(grid_times[1] - grid_times[0])
    radius = max(1, int(math.ceil(4.0 * sigma / grid_spacing)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64) * grid_spacing
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= np.sum(kernel)
    smoothed_values = np.full_like(step_values, np.nan)
    finite = np.isfinite(step_values)
    change_indices = np.flatnonzero(np.diff(np.r_[False, finite, False]))
    for start, stop in change_indices.reshape(-1, 2):
        segment = step_values[start:stop]
        padded = np.pad(segment, radius, mode="edge")
        smoothed_values[start:stop] = np.convolve(padded, kernel, mode="valid")
    return grid_times, smoothed_values


def rollout_spans(
    records: list[dict[str, str]],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return offline and strategy-switch time spans from a rollout."""

    offline_spans = []
    switch_spans = []
    for row, next_row in zip(records, records[1:]):
        start = float(row["time"])
        end = float(next_row["time"])
        if end <= start:
            continue
        if row.get("mode") == "off" and row.get("action") != "switch":
            offline_spans.append((start, end))
        if row.get("action") == "switch":
            switch_spans.append((start, end))
    return offline_spans, switch_spans


def shade_spans(correlation_axis, offline_spans, switch_spans) -> None:
    """Draw the report's offline and switch interval shading."""

    for start, end in offline_spans:
        correlation_axis.axvspan(start, end, color=OFFLINE_SHADE, zorder=0)
    for start, end in switch_spans:
        correlation_axis.axvspan(start, end, color=SWITCH_SHADE, zorder=0)


def append_span_legend(handles, labels, offline_spans, switch_spans) -> None:
    """Append visible interval types to a combined legend."""

    if offline_spans:
        handles.append(Patch(facecolor=OFFLINE_SHADE, edgecolor="none"))
        labels.append("offline")
    if switch_spans:
        handles.append(Patch(facecolor=SWITCH_SHADE, edgecolor="none"))
        labels.append("switch")


def plot_curves(
    correlation_axis,
    k_axis,
    times: np.ndarray,
    rho_me: np.ndarray,
    rho_se: np.ndarray,
    k_times: np.ndarray,
    k_values: np.ndarray,
    *,
    show_rho_se: bool,
    show_k: bool,
) -> tuple[list, list[str]]:
    """Plot one reveal and return its visible legend entries."""

    rho_me_line = correlation_axis.plot(
        times,
        rho_me,
        color=RHO_ME_COLOR,
        linewidth=LINE_WIDTH,
        label=r"$\rho_{me}$",
        zorder=10,
    )[0]
    handles = [rho_me_line]
    labels = [r"$\rho_{me}$"]

    if show_rho_se:
        rho_se_line = correlation_axis.plot(
            times,
            rho_se,
            color=RHO_SE_COLOR,
            linewidth=LINE_WIDTH,
            label=r"$\rho_{se}$",
            zorder=11,
        )[0]
        handles.append(rho_se_line)
        labels.append(r"$\rho_{se}$")

    if show_k:
        # Add K to the correlation axis so Matplotlib can respect line-level
        # z-ordering across the two scales.  The right-axis transform still
        # maps its values onto the K scale.
        k_line = correlation_axis.plot(
            k_times,
            k_values,
            color=K_COLOR,
            linewidth=LINE_WIDTH,
            transform=k_axis.transData,
            label=r"$K^*$ (smoothed)",
            zorder=2,
        )[0]
        handles.append(k_line)
        labels.append(r"$K^*$ (smoothed)")

    return handles, labels


def format_axes(correlation_axis, k_axis, *, show_k_axis: bool) -> None:
    """Apply the report style to the correlation and K axes."""

    correlation_axis.set_xlabel("elapsed time")
    correlation_axis.set_ylabel("correlation")
    correlation_axis.grid(axis="y", color="#dddddd", linewidth=0.7, zorder=1)
    correlation_axis.spines["top"].set_visible(False)
    correlation_axis.spines["right"].set_visible(False)

    k_axis.set_ylabel(r"optimal plans-per-action", color=K_COLOR, rotation = 270, labelpad = 14)
    k_axis.tick_params(axis="y", labelcolor=K_COLOR)
    k_axis.spines["top"].set_visible(False)
    k_axis.set_visible(show_k_axis)


def shared_plot_settings(
    times: np.ndarray,
    rho_me: np.ndarray,
    rho_se: np.ndarray,
    k_times: np.ndarray,
    k_values: np.ndarray,
    k_max: float,
    offline_spans: list[tuple[float, float]],
    switch_spans: list[tuple[float, float]],
) -> tuple:
    """Measure limits, ticks, and layout from the complete reveal."""

    figure, correlation_axis = plt.subplots(figsize=(5.6, 3.8))
    k_axis = correlation_axis.twinx()
    shade_spans(correlation_axis, offline_spans, switch_spans)
    handles, labels = plot_curves(
        correlation_axis,
        k_axis,
        times,
        rho_me,
        rho_se,
        k_times,
        k_values,
        show_rho_se=True,
        show_k=True,
    )
    format_axes(correlation_axis, k_axis, show_k_axis=True)
    k_axis.set_ylim(0.0, 1.0 + k_max)
    append_span_legend(handles, labels, offline_spans, switch_spans)
    #correlation_axis.legend(handles, labels, frameon=False, loc="lower right")
    figure.tight_layout()
    figure.canvas.draw()
    settings = (
        correlation_axis.get_xlim(),
        correlation_axis.get_ylim(),
        correlation_axis.get_xticks(),
        correlation_axis.get_yticks(),
        k_axis.get_ylim(),
        k_axis.get_yticks(),
        (
            figure.subplotpars.left,
            figure.subplotpars.right,
            figure.subplotpars.bottom,
            figure.subplotpars.top,
        ),
    )
    plt.close(figure)
    return settings


def main(default_variant: str = "online") -> None:
    """Write three progressive joint-learning DP figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("online", "switching"),
        default=default_variant,
    )
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=0.5,
        help="Gaussian K-smoothing standard deviation in elapsed-time units",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    source = args.input or default_input(args.variant)
    records = read_records(source)
    if not records:
        raise ValueError(f"no records found in {source}")

    rho_me_times, rho_me = finite_series(records, "rho_me")
    rho_se_times, rho_se = finite_series(records, "rho_se")
    if not np.array_equal(rho_me_times, rho_se_times):
        raise ValueError("rho_me and rho_se must be recorded at the same times")
    k_times, k_values = smoothed_step_series(
        records,
        "K",
        args.smoothing_sigma,
    )
    _, raw_k_values = finite_series(records, "K")
    offline_spans, switch_spans = rollout_spans(records)

    (
        xlim,
        correlation_ylim,
        xticks,
        correlation_yticks,
        k_ylim,
        k_yticks,
        subplot_bounds,
    ) = shared_plot_settings(
        rho_me_times,
        rho_me,
        rho_se,
        k_times,
        k_values,
        float(np.max(raw_k_values)),
        offline_spans,
        switch_spans,
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for filename, show_rho_se, show_k in reveal_specs(args.variant):
        figure, correlation_axis = plt.subplots(figsize=(5.6, 3.8))
        k_axis = correlation_axis.twinx()
        shade_spans(correlation_axis, offline_spans, switch_spans)
        handles, labels = plot_curves(
            correlation_axis,
            k_axis,
            rho_me_times,
            rho_me,
            rho_se,
            k_times,
            k_values,
            show_rho_se=show_rho_se,
            show_k=show_k,
        )
        format_axes(correlation_axis, k_axis, show_k_axis=show_k)
        append_span_legend(handles, labels, offline_spans, switch_spans)
        #correlation_axis.legend(handles, labels, frameon=False, loc="lower right")

        # Setting ticks can expand view limits; restore the common limits last.
        correlation_axis.set_xticks([])
        #correlation_axis.set_yticks(correlation_yticks)
        #k_axis.set_yticks(k_yticks)
        k_axis.set_yticks(range(0,21,4))
        correlation_axis.set_yticks(np.arange(0,11,2)/10)
        correlation_axis.set_xlim(*xlim)
        #correlation_axis.set_ylim(*correlation_ylim)
        #k_axis.set_ylim(*k_ylim)
        correlation_axis.set_ylim(-0.02, 1.02)
        k_axis.set_ylim(-0.4, 20.4)
        figure.subplots_adjust(
            left=subplot_bounds[0],
            right=subplot_bounds[1],
            bottom=subplot_bounds[2],
            top=subplot_bounds[3],
        )

        output = args.output_directory / filename
        figure.savefig(output, dpi=args.dpi, transparent=True)
        plt.close(figure)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
