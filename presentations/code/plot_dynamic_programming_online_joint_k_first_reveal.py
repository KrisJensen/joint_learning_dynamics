"""Generate K-first progressive reveals of the online joint DP rollout.

The figures successively reveal optimal plans-per-action, world-model quality,
policy quality, and the reward-only counterfactual planning policy.  Optimal K
is the only quantity with a visible numeric axis.  The two correlations are
mapped onto the plotting height without adding a second axis, because they are
included as qualitative trajectories for this presentation sequence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from plot_dynamic_programming_online_joint_reveal import (
    DEFAULT_OUTPUT_DIRECTORY,
    K_COLOR,
    LINE_WIDTH,
    REPOSITORY_ROOT,
    RHO_ME_COLOR,
    RHO_SE_COLOR,
    default_input,
    finite_series,
    read_records,
    smoothed_step_series,
)

import matplotlib.pyplot as plt


K_YLIM = (-0.4, 20.4)
K_YTICKS = np.arange(0.0, 21.0, 4.0)
CORRELATION_YLIM = (-0.02, 1.02)

REVEALS = (
    ("dynamic_programming_online_joint_K_first_1_K.pdf", False, False, False),
    ("dynamic_programming_online_joint_K_first_2_K_rho_me.pdf", True, False, False),
    (
        "dynamic_programming_online_joint_K_first_3_K_rho_me_rho_se.pdf",
        True,
        True,
        False,
    ),
    (
        "dynamic_programming_online_joint_K_first_4_with_reward_only.pdf",
        True,
        True,
        True,
    ),
)


def correlation_plot_values(values: np.ndarray) -> np.ndarray:
    """Map correlations onto the K plotting height without adding an axis."""

    correlation_min, correlation_max = CORRELATION_YLIM
    k_min, k_max = K_YLIM
    fractions = (values - correlation_min) / (correlation_max - correlation_min)
    return k_min + fractions * (k_max - k_min)


def plot_reveal(
    axis,
    times: np.ndarray,
    rho_me: np.ndarray,
    rho_se: np.ndarray,
    k_times: np.ndarray,
    k_values: np.ndarray,
    counterfactual_times: np.ndarray,
    counterfactual_values: np.ndarray,
    *,
    show_rho_me: bool,
    show_rho_se: bool,
    show_counterfactual: bool,
) -> None:
    """Draw one progressive reveal on a single axes object."""

    axis.plot(
        k_times,
        k_values,
        color=K_COLOR,
        linewidth=LINE_WIDTH,
        zorder=13,
    )
    if show_rho_me:
        axis.plot(
            times,
            correlation_plot_values(rho_me),
            color=RHO_ME_COLOR,
            linewidth=LINE_WIDTH,
            zorder=10,
        )
    if show_rho_se:
        axis.plot(
            times,
            correlation_plot_values(rho_se),
            color=RHO_SE_COLOR,
            linewidth=LINE_WIDTH,
            zorder=11,
        )
    if show_counterfactual:
        axis.plot(
            counterfactual_times,
            counterfactual_values,
            color=K_COLOR,
            linestyle="--",
            linewidth=LINE_WIDTH,
            zorder=14,
        )


def format_axis(axis, xlim: tuple[float, float]) -> None:
    """Apply the presentation formatting shared by all four reveals."""

    axis.set_xlabel("elapsed time")
    #axis.set_ylabel("optimal plans-per-action", color=K_COLOR, labelpad=10)
    axis.set_ylabel("optimal planning duration", color=K_COLOR, labelpad=3)
    axis.tick_params(axis="y", labelcolor=K_COLOR)
    axis.set_xticks([])
    axis.set_yticks([])#K_YTICKS)
    axis.set_xlim(*xlim)
    axis.set_ylim(*K_YLIM)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)


def elapsed_time_limits(times: np.ndarray) -> tuple[float, float]:
    """Return Matplotlib's standard five-percent time margin."""

    start = float(np.min(times))
    end = float(np.max(times))
    span = end - start
    if span <= 0.0:
        raise ValueError("the rollout must span a positive elapsed time")
    margin = 0.05 * span
    return start - margin, end + margin


def main() -> None:
    """Write the four K-first online-joint reveal figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--counterfactual-input",
        type=Path,
        default=(
            REPOSITORY_ROOT
            / "results/report/dynamic_programming_online_joint_reward_only_K.csv"
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

    source = args.input or default_input("online")
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
    counterfactual_records = read_records(args.counterfactual_input)
    counterfactual_times, counterfactual_values = smoothed_step_series(
        counterfactual_records,
        "K",
        args.smoothing_sigma,
    )
    xlim = elapsed_time_limits(rho_me_times)

    # Measure the complete figure once and reuse its subplot bounds so only the
    # visible trajectories change between presentation slides.
    reference_figure, reference_axis = plt.subplots(figsize=(3.6, 2.4))
    plot_reveal(
        reference_axis,
        rho_me_times,
        rho_me,
        rho_se,
        k_times,
        k_values,
        counterfactual_times,
        counterfactual_values,
        show_rho_me=True,
        show_rho_se=True,
        show_counterfactual=True,
    )
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
    for filename, show_rho_me, show_rho_se, show_counterfactual in REVEALS:
        figure, axis = plt.subplots(figsize=(3.6, 2.8))
        plot_reveal(
            axis,
            rho_me_times,
            rho_me,
            rho_se,
            k_times,
            k_values,
            counterfactual_times,
            counterfactual_values,
            show_rho_me=show_rho_me,
            show_rho_se=show_rho_se,
            show_counterfactual=show_counterfactual,
        )
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
