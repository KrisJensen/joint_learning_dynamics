"""Plot static reward observables across M."""

import argparse

import numpy as np

from _common import add_plot_args, grouped, paths, read_records

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "prefix_length_observables")
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
metrics = (
    ("true_reward_probability", "true sequence reward"),
    ("reward_rate", "reward rate"),
    ("planner_acceptance_probability", "WM prefix acceptance"),
)
figure, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
for axis, (metric, label) in zip(axes, metrics):
    for condition, rows in grouped(records).items():
        rows = sorted(rows, key=lambda row: int(float(row["M"])))
        axis.plot(
            [int(float(row["M"])) for row in rows],
            [float(row[metric]) for row in rows],
            marker="o",
            label=condition,
        )
    axis.set_xlabel(r"$M$")
    axis.set_ylabel(label)
    axis.set_xticks(range(1, max(int(float(row["M"])) for row in records) + 1))
    axis.grid(axis="y", color="#dddddd")
    axis.spines[["top", "right"]].set_visible(False)
axes[-1].legend(frameon=False)
figure.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=args.dpi, transparent=True)
plt.close(figure)
print(f"Wrote {output}")
