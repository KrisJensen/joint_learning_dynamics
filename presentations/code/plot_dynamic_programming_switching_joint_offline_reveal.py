"""Reveal the switching joint-DP trajectory around its offline period.

The four figures successively show the world-model and policy trajectories
before offline learning, through offline learning, through the rest of the
episode, and finally with optimal plans-per-action overlaid.  The y-axis has no
labels or ticks; correlations are mapped onto the plotting height without a
separate visible scale.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_dynamic_programming_online_joint_k_first_reveal import (
    K_YLIM,
    correlation_plot_values,
    elapsed_time_limits,
)
from plot_dynamic_programming_online_joint_reveal import (
    DEFAULT_OUTPUT_DIRECTORY,
    K_COLOR,
    LINE_WIDTH,
    OFFLINE_SHADE,
    RHO_ME_COLOR,
    RHO_SE_COLOR,
    REPOSITORY_ROOT,
    SWITCH_SHADE,
    default_input,
    finite_series,
    read_records,
    rollout_spans,
    smoothed_step_series,
)

import matplotlib.pyplot as plt


OUTPUT_SPECS = (
    "dynamic_programming_switching_joint_offline_1_pre_offline.pdf",
    "dynamic_programming_switching_joint_offline_2_through_offline.pdf",
    "dynamic_programming_switching_joint_offline_3_full_correlations.pdf",
    "dynamic_programming_switching_joint_offline_4_with_K.pdf",
)


def merge_touching_spans(
    spans: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Merge adjacent rollout intervals into presentation-level spans."""

    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        previous_start, previous_end = merged[-1]
        tolerance = 1e-9 * max(1.0, abs(previous_end), abs(start))
        if start <= previous_end + tolerance:
            merged[-1] = (previous_start, max(previous_end, end))
        else:
            merged.append((start, end))
    return merged


def truncated_series(
    times: np.ndarray,
    values: np.ndarray,
    end_time: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a trajectory through an inclusive elapsed-time boundary."""

    tolerance = 1e-9 * max(1.0, abs(end_time))
    keep = times <= end_time + tolerance
    return times[keep], values[keep]


def shade_through(
    axis,
    offline_spans: list[tuple[float, float]],
    switch_spans: list[tuple[float, float]],
    end_time: float,
) -> None:
    """Draw only portions of offline and switch spans revealed so far."""

    for spans, color in (
        (offline_spans, OFFLINE_SHADE),
        (switch_spans, SWITCH_SHADE),
    ):
        for start, end in spans:
            if start >= end_time:
                continue
            axis.axvspan(start, min(end, end_time), color=color, zorder=0)


def plot_correlations(
    axis,
    times: np.ndarray,
    rho_me: np.ndarray,
    rho_se: np.ndarray,
    end_time: float,
) -> None:
    """Plot both correlation trajectories through ``end_time``."""

    visible_times, visible_rho_me = truncated_series(times, rho_me, end_time)
    _, visible_rho_se = truncated_series(times, rho_se, end_time)
    axis.plot(
        visible_times,
        correlation_plot_values(visible_rho_me),
        color=RHO_ME_COLOR,
        linewidth=LINE_WIDTH,
        zorder=10,
    )
    axis.plot(
        visible_times,
        correlation_plot_values(visible_rho_se),
        color=RHO_SE_COLOR,
        linewidth=LINE_WIDTH,
        zorder=11,
    )


def format_axis(axis, xlim: tuple[float, float]) -> None:
    """Apply the shared axis-free presentation formatting."""

    axis.set_xlabel("elapsed time")
    axis.set_ylabel("")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlim(*xlim)
    axis.set_ylim(*K_YLIM)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


def main() -> None:
    """Write the four offline-period switching reveals."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--counterfactual-input",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results/report/dynamic_programming_switching_joint_reward_only_K.csv"
        ),
        help=(
            "reward-only K table written by "
            "compute_reward_only_counterfactual_k.py"
        ),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=DEFAULT_OUTPUT_DIRECTORY,
    )
    parser.add_argument(
        "--smoothing-sigma",
        type=float,
        default=1.0,
        help="Gaussian K-smoothing standard deviation in elapsed-time units",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    source = args.input or default_input("switching")
    records = read_records(source)
    if not records:
        raise ValueError(f"no records found in {source}")

    times, rho_me = finite_series(records, "rho_me")
    rho_se_times, rho_se = finite_series(records, "rho_se")
    if not np.array_equal(times, rho_se_times):
        raise ValueError("rho_me and rho_se must be recorded at the same times")
    k_times, k_values = smoothed_step_series(
        records,
        "K",
        args.smoothing_sigma,
    )
    counterfactual_records = read_records(args.counterfactual_input)
    counterfactual_times, counterfactual_values = smoothed_step_series(
        counterfactual_records,
        "K",
        args.smoothing_sigma,
    )
    offline_spans, switch_spans = rollout_spans(records)
    offline_spans = merge_touching_spans(offline_spans)
    switch_spans = merge_touching_spans(switch_spans)
    if len(offline_spans) != 1:
        raise ValueError(
            "this reveal requires exactly one contiguous offline-learning period"
        )

    if len(switch_spans) != 2:
        raise ValueError(
            "this reveal requires one switch into and one switch out of offline mode"
        )
    # Treat both switching costs as part of the presentation's offline period.
    # Thus the pre-offline reveal ends where the first switch starts, and the
    # through-offline reveal ends only after the second switch completes.
    offline_start = switch_spans[0][0]
    offline_end = switch_spans[-1][1]
    episode_end = float(times[-1])
    reveal_ends = (offline_start, offline_end, episode_end, episode_end)
    xlim = elapsed_time_limits(times)

    # Use the completed figure to fix one plotting area for every reveal.
    reference_figure, reference_axis = plt.subplots(figsize=(3.6, 2.8))
    shade_through(
        reference_axis,
        offline_spans,
        switch_spans,
        episode_end,
    )
    reference_axis.plot(
        counterfactual_times,
        counterfactual_values,
        color=K_COLOR,
        linestyle="--",
        linewidth=LINE_WIDTH,
        zorder=12,
    )
    reference_axis.plot(
        k_times,
        k_values,
        color=K_COLOR,
        linewidth=LINE_WIDTH,
        zorder=13,
    )
    plot_correlations(reference_axis, times, rho_me, rho_se, episode_end)
    format_axis(reference_axis, xlim)
    reference_figure.tight_layout()
    subplot_bounds = (
        reference_figure.subplotpars.left,
        reference_figure.subplotpars.right,
        reference_figure.subplotpars.bottom,
        reference_figure.subplotpars.top,
    )
    plt.close(reference_figure)

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for index, (filename, end_time) in enumerate(
        zip(OUTPUT_SPECS, reveal_ends, strict=True)
    ):
        figure, axis = plt.subplots(figsize=(3.6, 2.8))
        shade_through(axis, offline_spans, switch_spans, end_time)
        if index == len(OUTPUT_SPECS) - 1:
            # axis.plot(
            #     counterfactual_times,
            #     counterfactual_values,
            #     color=K_COLOR,
            #     linestyle="--",
            #     linewidth=LINE_WIDTH,
            #     zorder=12,
            # )
            axis.plot(
                k_times,
                k_values,
                color=K_COLOR,
                linewidth=LINE_WIDTH,
                zorder=13,
            )
        plot_correlations(axis, times, rho_me, rho_se, end_time)
        format_axis(axis, xlim)
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
