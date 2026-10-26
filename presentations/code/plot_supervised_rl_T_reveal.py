"""Plot progressive reveals of supervised and RL learners at T=6 and T=7."""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "results/report/supervised_rl_T_comparison.csv"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "presentations/figures"

# Use writable caches on compute nodes with read-only home directories.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


CONDITIONS = (
    ("reinforcement", 6),
    ("supervised", 6),
    ("reinforcement", 7),
    ("supervised", 7),
)

REVEALS = (
    (
        "supervised_rl_T_expected_reward_rate_1_RL_T6.pdf",
        frozenset(CONDITIONS[:1]),
    ),
    (
        "supervised_rl_T_expected_reward_rate_2_supervised_T6.pdf",
        frozenset(CONDITIONS[:2]),
    ),
    (
        "supervised_rl_T_expected_reward_rate_3_RL_T7.pdf",
        frozenset(CONDITIONS[:3]),
    ),
    (
        "supervised_rl_T_expected_reward_rate_4_supervised_T7.pdf",
        frozenset(CONDITIONS),
    ),
)


def read_records(path: Path) -> list[dict[str, str]]:
    """Read analytical trajectory records from ``path``."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def condition_rows(
    records: list[dict[str, str]], learner: str, T: int
) -> list[dict[str, str]]:
    """Return finite normalized reward-rate rows for one condition."""

    rows = [
        row
        for row in records
        if row.get("learner") == learner
        and int(float(row.get("T_value", "nan"))) == T
        and row.get("learner_reward_rate", "") not in {"", None}
        and math.isfinite(float(row["learner_reward_rate"]))
    ]
    return sorted(rows, key=lambda row: float(row["time"]))


def line_style(learner: str, T: int) -> dict:
    """Encode learner by color and sequence length by line style."""

    return {
        "color": "black" if learner == "supervised" else "#1f77b4",
        #"color": "#1f77b4" if learner == "supervised" else "black",
        "linestyle": "-" if T == 6 else "--",
        "linewidth": 3.0,
    }


def condition_label(learner: str, T: int) -> str:
    """Return a concise presentation legend label."""

    name = "RL" if learner == "reinforcement" else "supervised"
    return rf"{name}, $T={T}$"


def plot_curves(axis, records, selected) -> None:
    """Plot selected conditions in the fixed full-comparison order."""

    for learner, T in CONDITIONS:
        if (learner, T) not in selected:
            continue
        rows = condition_rows(records, learner, T)
        if not rows:
            raise ValueError(f"no finite reward-rate records for {learner}, T={T}")
        axis.plot(
            [float(row["time"]) for row in rows],
            [float(row["learner_reward_rate"]) for row in rows],
            label=condition_label(learner, T),
            **line_style(learner, T),
        )


def add_fixed_legend(axis, selected) -> None:
    """Keep every condition's legend slot fixed throughout the reveal."""

    handles = []
    labels = []
    visible = []
    for learner, T in CONDITIONS:
        handle = Line2D([], [], **line_style(learner, T))
        is_visible = (learner, T) in selected
        if not is_visible:
            handle.set_alpha(0.0)
        handles.append(handle)
        labels.append(condition_label(learner, T))
        visible.append(is_visible)
    # legend = axis.legend(handles, labels, frameon=False, loc="lower right")
    # for text, is_visible in zip(legend.get_texts(), visible, strict=True):
    #     if not is_visible:
    #         text.set_alpha(0.0)


def format_axis(axis) -> None:
    """Apply the policy-regime presentation formatting."""

    axis.set_xlabel("elapsed time")
    axis.set_ylabel("reward rate")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlim(0, 70)


def shared_axis_settings(records) -> tuple:
    """Measure plot settings once from the complete comparison."""

    complete = frozenset(CONDITIONS)
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    plot_curves(axis, records, complete)
    format_axis(axis)
    add_fixed_legend(axis, complete)
    figure.tight_layout()
    settings = (
        axis.get_xlim(),
        axis.get_ylim(),
        axis.get_xticks(),
        axis.get_yticks(),
        (
            figure.subplotpars.left,
            figure.subplotpars.right,
            figure.subplotpars.bottom,
            figure.subplotpars.top,
        ),
    )
    plt.close(figure)
    return settings


def main() -> None:
    """Write the four progressive learner-comparison figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    records = read_records(args.input)
    if not records:
        raise ValueError(f"no records found in {args.input}")

    xlim, ylim, xticks, yticks, bounds = shared_axis_settings(records)
    args.output_directory.mkdir(parents=True, exist_ok=True)
    for filename, selected in REVEALS:
        figure, axis = plt.subplots(figsize=(5.2, 3.6))
        plot_curves(axis, records, selected)
        format_axis(axis)
        add_fixed_legend(axis, selected)
        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        figure.subplots_adjust(
            left=bounds[0], right=bounds[1], bottom=bounds[2], top=bounds[3]
        )
        output = args.output_directory / filename
        figure.savefig(output, dpi=args.dpi, transparent=True)
        plt.close(figure)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
