"""Generate progressive reveals of the policy-regime reward-rate figure.

The four output figures successively show no-planning, evaluation-planning,
planning-for-learning, and full planning policies.  Axis limits and tick
locations are calculated from all four curves and then reused so that only the
number of visible curves changes between figures.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPOSITORY_ROOT / "results/report/policy_regime_comparison.csv"
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "presentations/figures"

# Use writable caches when running on a compute node with a read-only home.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt


# This order matches code/plot/plot_policy_regimes.py, including its layering.
POLICY_COMBINATIONS = (
    ("planning", "planning"),
    ("planning", "greedy"),
    ("greedy", "planning"),
    ("greedy", "greedy"),
)

REVEALS = (
    (
        "policy_regime_expected_reward_rate_1_no_planning.pdf",
        frozenset({("greedy", "greedy")}),
    ),
    (
        "policy_regime_expected_reward_rate_2_evaluation_planning.pdf",
        frozenset({("greedy", "greedy"), ("greedy", "planning")}),
    ),
    (
        "policy_regime_expected_reward_rate_3_learning_planning.pdf",
        frozenset(
            {
                ("greedy", "greedy"),
                ("greedy", "planning"),
                ("planning", "greedy"),
            }
        ),
    ),
    (
        "policy_regime_expected_reward_rate_4_all.pdf",
        frozenset(POLICY_COMBINATIONS),
    ),
)


def read_records(path: Path) -> list[dict[str, str]]:
    """Read policy-regime trajectory records from ``path``."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def policy_rows(
    records: list[dict[str, str]],
    training_policy: str,
    evaluation_policy: str,
) -> list[dict[str, str]]:
    """Return finite reward-rate records for one policy combination."""

    rows = [
        row
        for row in records
        if row.get("training_policy") == training_policy
        and row.get("evaluation_policy") == evaluation_policy
        and row.get("reward_rate", "") not in {"", None}
        and math.isfinite(float(row["reward_rate"]))
    ]
    return sorted(rows, key=lambda row: float(row["time"]))


def plot_curves(axis, records, combinations) -> None:
    """Plot selected policy combinations using the report figure's styles."""

    for training_policy, evaluation_policy in POLICY_COMBINATIONS:
        if (training_policy, evaluation_policy) not in combinations:
            continue
        rows = policy_rows(records, training_policy, evaluation_policy)
        if not rows:
            raise ValueError(
                "no finite reward-rate records for "
                f"{training_policy} training / {evaluation_policy} evaluation"
            )
        axis.plot(
            [float(row["time"]) for row in rows],
            [float(row["reward_rate"]) for row in rows],
            color= "black" if training_policy == "planning" else "#1f77b4",
            #color="#1f77b4" if training_policy == "planning" else "black",
            linestyle="--" if evaluation_policy == "planning" else "-",
            linewidth=3.5,
            label=(
                f"learn {'with' if training_policy == 'planning' else 'without'} "
                "plans, evaluate "
                f"{'with' if evaluation_policy == 'planning' else 'without'} plans"
            ),
        )


def format_axis(axis) -> None:
    """Apply the formatting used by the report reward-rate figure."""

    axis.set_xlabel("elapsed time")
    axis.set_ylabel("reward rate")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.set_xlim(0, 70)


def shared_axis_settings(records) -> tuple:
    """Measure the limits and tick locations of the complete report plot."""

    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    plot_curves(axis, records, frozenset(POLICY_COMBINATIONS))
    format_axis(axis)
    figure.canvas.draw()
    settings = (
        axis.get_xlim(),
        axis.get_ylim(),
        axis.get_xticks(),
        axis.get_yticks(),
    )
    plt.close(figure)
    return settings


def main() -> None:
    """Write the four progressive policy-regime figures."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    records = read_records(args.input)
    if not records:
        raise ValueError(f"no records found in {args.input}")

    xlim, ylim, xticks, yticks = shared_axis_settings(records)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    for filename, combinations in REVEALS:
        figure, axis = plt.subplots(figsize=(5.2, 3.6))
        plot_curves(axis, records, combinations)
        format_axis(axis)
        # ``set_ticks`` may expand an axis to include ticks just beyond the
        # visible range, so restore the shared limits after setting the ticks.
        #axis.set_xticks(xticks)
        axis.set_xticks([])
        axis.set_yticks(yticks)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        figure.tight_layout()
        output = args.output_directory / filename
        figure.savefig(output, dpi=args.dpi, transparent=True)
        plt.close(figure)
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
