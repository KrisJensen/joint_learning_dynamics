"""Plot planning and greedy policy regimes."""

import argparse
import math
from pathlib import Path

from _common import add_plot_args, paths, plot_dashboard, read_records

import matplotlib.pyplot as plt


def policy_rows(
    records: list[dict],
    training_policy: str,
    evaluation_policy: str,
) -> list[dict]:
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


def metric_output(summary_output: Path, metric: str) -> Path:
    """Replace the summary stem while retaining its suffix and directory."""

    prefix = "policy_regime_comparison"
    if summary_output.stem.startswith(prefix):
        suffix = summary_output.stem[len(prefix) :]
        stem = f"policy_regime_{metric}{suffix}"
    else:
        stem = f"{summary_output.stem}_{metric}"
    return summary_output.with_name(f"{stem}{summary_output.suffix}")


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "policy_regime_comparison")
args = parser.parse_args()
source, summary_output = paths(args)
records = read_records(source)
if not records:
    raise ValueError(f"no records found in {source}")

plot_dashboard(
    records,
    summary_output,
    "Training and evaluation policy regimes",
    args.dpi,
)
print(f"Wrote {summary_output}")

policy_combinations = (
    ("planning", "planning"),
    ("planning", "greedy"),
    ("greedy", "planning"),
    ("greedy", "greedy"),
)

expected_output = metric_output(summary_output, "expected_reward_rate")
figure, axis = plt.subplots(figsize=(5.2, 3.6))
for training_policy, evaluation_policy in policy_combinations:
    rows = policy_rows(records, training_policy, evaluation_policy)
    if not rows:
        continue
    axis.plot(
        [float(row["time"]) for row in rows],
        [float(row["reward_rate"]) for row in rows],
        color="#1f77b4" if training_policy == "planning" else "black",
        linestyle="--" if evaluation_policy == "planning" else "-",
        linewidth=3.5,
        label=(
            f"learn {'with' if training_policy == 'planning' else 'without'} plans, "
            f"evaluate {'with' if evaluation_policy == 'planning' else 'without'} plans"
        ),
    )
axis.set_xlabel("elapsed time")
axis.set_ylabel("expected reward rate")
axis.grid(axis="y", color="#dddddd", linewidth=0.7)
axis.spines[["top", "right"]].set_visible(False)
axis.set_xlim(0, 70)
figure.tight_layout()
expected_output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(expected_output, dpi=args.dpi, transparent=True)
plt.close(figure)
print(f"Wrote {expected_output}")

average_output = metric_output(summary_output, "average_reward_rate")
figure, axis = plt.subplots(figsize=(5.2, 3.6))
for training_policy, evaluation_policy in policy_combinations:
    rows = policy_rows(records, training_policy, evaluation_policy)
    if not rows:
        continue
    times = [float(row["time"]) for row in rows]
    rates = [float(row["reward_rate"]) for row in rows]
    integrated_reward = [0.0]
    average_rates = [rates[0]]
    for index in range(1, len(times)):
        dt = times[index] - times[index - 1]
        integrated_reward.append(
            integrated_reward[-1]
            + 0.5 * (rates[index - 1] + rates[index]) * dt
        )
        elapsed = times[index] - times[0]
        average_rates.append(
            integrated_reward[-1] / elapsed if elapsed > 0.0 else rates[index]
        )
    axis.plot(
        times,
        average_rates,
        color="#1f77b4" if training_policy == "planning" else "black",
        linestyle="--" if evaluation_policy == "planning" else "-",
        linewidth=3.5,
        label=(
            f"learn {'with' if training_policy == 'planning' else 'without'} plans, "
            f"evaluate {'with' if evaluation_policy == 'planning' else 'without'} plans"
        ),
    )
axis.set_xlabel("elapsed time")
axis.set_ylabel("average reward rate")
axis.grid(axis="y", color="#dddddd", linewidth=0.7)
axis.spines[["top", "right"]].set_visible(False)
axis.set_xlim(0, 250)
figure.tight_layout()
average_output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(average_output, dpi=args.dpi, transparent=True)
plt.close(figure)
print(f"Wrote {average_output}")
