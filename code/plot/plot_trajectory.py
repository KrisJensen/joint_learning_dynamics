"""Plot a single analytical trajectory and optional stand-alone quantities."""

import argparse

import numpy as np

from _common import (
    CORRELATION_STYLES,
    add_plot_args,
    offline_spans,
    paths,
    plot_dashboard,
    read_records,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "analytical_trajectory")
parser.add_argument(
    "--plot-correlations",
    action="store_true",
    help="also write a shared-axis plot of rho_se, rho_me, and rho_sm",
)
parser.add_argument(
    "--plot-planning-success-probability",
    "--plot-planning-success",
    dest="plot_planning_success_probability",
    action="store_true",
    help="also plot the probability that planning produces a WM-successful prefix",
)
parser.add_argument(
    "--plot-environment-reward-probability",
    "--plot-environment-reward",
    dest="plot_environment_reward_probability",
    action="store_true",
    help="also plot the true environment reward probability of the planned action",
)
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
if not records:
    raise ValueError(f"no records found in {source}")

plot_dashboard(records, output, "Analytical learning dynamics", args.dpi)
print(f"Wrote {output}")

standalone_specs = []
if args.plot_correlations:
    standalone_specs.append(
        (
            (
                *CORRELATION_STYLES,
            ),
            r"correlation",
            "correlations",
        )
    )
if args.plot_planning_success_probability:
    standalone_specs.append(
        (
            (
                (
                    "planner_acceptance_probability",
                    "planning success probability",
                    "#000000",
                ),
            ),
            "planning success probability",
            "planning_success_probability",
        )
    )
if args.plot_environment_reward_probability:
    standalone_specs.append(
        (
            (
                (
                    "planning_environment_reward_probability",
                    "environment reward probability",
                    "#000000",
                ),
            ),
            "environment reward probability",
            "environment_reward_probability",
        )
    )

spans = offline_spans(records)
for series, ylabel, suffix_name in standalone_specs:
    standalone_output = output.with_name(
        f"{output.stem}_{suffix_name}{output.suffix}"
    )
    figure, axis = plt.subplots(figsize=(4.5, 3.8))
    for start, end in spans:
        axis.axvspan(start, end, color="#e9ecef", zorder=0)
    maxval = 0.0
    for metric, label, color in series:
        usable = [
            row
            for row in records
            if row.get(metric, "") not in {"", None}
            and row.get("time", "") not in {"", None}
            and np.isfinite(float(row[metric]))
        ]
        if not usable:
            raise ValueError(f"{source} has no finite values for {metric}")
        xs, ys = [float(row["time"]) for row in usable], [float(row[metric]) for row in usable]
        maxval = max(maxval, 0.1*int(np.ceil(max(ys)*10)))
        axis.plot(
            xs, ys,
            color=color,
            linewidth=2.2,
            label=label,
        )
    axis.set_xlabel("elapsed time")
    axis.set_ylabel(ylabel)
    #axis.set_ylim(0.0, 1.0)
    axis.set_ylim(0.0, maxval)
    axis.set_yticks(np.arange(int(maxval*10+1))/10)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    if spans:
        handles.append(Patch(facecolor="#e9ecef", edgecolor="none"))
        labels.append("offline")
    if len(series) > 1 or spans:
        axis.legend(handles, labels, frameon=False)
    figure.tight_layout()
    standalone_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(standalone_output, dpi=args.dpi, transparent=True)
    plt.close(figure)
    print(f"Wrote {standalone_output}")
