"""Plot reward and action-correctness observables across prefix length."""

import argparse

import numpy as np

from _common import add_plot_args, paths, read_records

import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "prefix_length_observables")
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
if not records:
    raise ValueError(f"no records found in {source}")

metrics = (
    ("sequence_correct_probability", "correct sequence probability"),
    ("reward_rate", "reward rate"),
    ("individual_action_correct_probability", "correct action probability"),
    (
        "planner_prefix_correct_probability",
        "planner-correct prefix probability",
    ),
)
missing = [
    metric
    for metric, _ in metrics
    if any(row.get(metric, "") in {"", None} for row in records)
]
if missing:
    names = ", ".join(missing)
    raise ValueError(
        f"{source} is missing {names}; rerun compare_prefix_length_observables.py"
    )

rho_values = list(dict.fromkeys(float(row["rho"]) for row in records))
K_values = {int(float(row["K"])) for row in records}
T_values = {
    int(float(row.get("T", row.get("config_T", ""))))
    for row in records
}
if len(K_values) != 1 or len(T_values) != 1:
    raise ValueError("prefix-length plots require one fixed K and T")
K = next(iter(K_values))
T = next(iter(T_values))
colors = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(rho_values)))

for metric, ylabel in metrics:
    metric_output = output.with_name(
        f"{output.stem}_{metric}{output.suffix}"
    )
    figure, axis = plt.subplots(figsize=(5.2, 3.6))
    for color, rho in zip(colors, rho_values):
        rows = [
            row
            for row in records
            if np.isclose(float(row["rho"]), rho)
        ]
        rows.sort(key=lambda row: int(float(row["M"])))
        x_values = [int(float(row["M"])) for row in rows]
        y_values = [float(row[metric]) for row in rows]
        axis.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.5,
            marker="o",
            label=rf"$\rho={rho:g}$",
        )
        best_index = int(np.argmax(y_values))
        axis.plot(
            [x_values[best_index]],
            [y_values[best_index]],
            color=color,
            marker="*",
            markersize=13,
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="None",
        )
    axis.set_xlabel("prefix length")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    if metric == "sequence_correct_probability":
        axis.legend(frameon=False)
    #figure.suptitle(rf"{ylabel} ($K={K}$, $T={T}$)")
    figure.tight_layout()
    metric_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        metric_output,
        dpi=args.dpi,
        transparent=True,
    )
    plt.close(figure)
    print(f"Wrote {metric_output}")
