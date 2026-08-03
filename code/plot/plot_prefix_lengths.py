"""Plot selected aspects of the prefix-length comparison."""

import argparse

import numpy as np

from _common import (
    add_plot_args,
    grouped,
    offline_spans,
    paths,
    plot_dashboard,
    read_records,
)

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def label_for(rows: list[dict]) -> str:
    """Return the legacy-style label for one comparison group."""

    first = rows[0]
    group = first.get("comparison_group", "")
    M = first.get("compared_M", "")
    if group == "eta_2f_fixed_M":
        return rf"$2f$ ($M={M}$)"
    if group == "eta_2f_optimal_M":
        return r"$2f$ (optimal)"
    if group == "eta_3f_only":
        return r"$3f$"
    return first.get("condition", "trajectory")


def condition_style(rows: list[dict], fixed_colors: dict[float, tuple]) -> dict:
    """Return the legacy line style for one comparison group."""

    first = rows[0]
    group = first.get("comparison_group", "")
    if group == "eta_2f_fixed_M":
        M = float(first.get("compared_M", "nan"))
        return {
            "color": fixed_colors[M],
            "linewidth": 1.7,
            "alpha": 0.9,
        }
    if group == "eta_2f_optimal_M":
        return {"color": "black", "linewidth": 2.5}
    if group == "eta_3f_only":
        return {"color": "#d5a336", "linestyle": "-", "linewidth": 2.5}
    return {}


def sort_key(item: tuple[str, list[dict]]) -> tuple[int, float, str]:
    """Sort fixed M values before optimal-M and eta_3f trajectories."""

    condition, rows = item
    first = rows[0]
    group = first.get("comparison_group", "")
    if group == "eta_2f_fixed_M":
        return (0, float(first.get("compared_M", "inf")), condition)
    if group == "eta_2f_optimal_M":
        return (1, 0.0, condition)
    if group == "eta_3f_only":
        return (2, 0.0, condition)
    return (3, 0.0, condition)


def shade_offline(axis, spans: list[tuple[float, float]]) -> None:
    """Shade offline periods using the legacy prefix-plot color."""

    for start, end in spans:
        axis.axvspan(start, end, color="#e9ecef", zorder=0)


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "prefix_length_comparison")
parser.add_argument(
    "--include-fixed-M",
    action="store_true",
    help="plot eta_2f-only trajectories for fixed M values",
)
parser.add_argument(
    "--include-optimal-M",
    action="store_true",
    help="plot eta_2f-only trajectories with dynamically optimized M",
)
parser.add_argument(
    "--include-eta-3f",
    action="store_true",
    help="plot the eta_3f-only baseline",
)
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
if not records:
    raise ValueError(f"no records found in {source}")

group_flags = {
    "eta_2f_fixed_M": args.include_fixed_M,
    "eta_2f_optimal_M": args.include_optimal_M,
    "eta_3f_only": args.include_eta_3f,
}
if any(group_flags.values()):
    selected_groups = {
        group for group, selected in group_flags.items() if selected
    }
    present_groups = {row.get("comparison_group", "") for row in records}
    missing_groups = selected_groups - present_groups
    if missing_groups:
        missing = ", ".join(sorted(missing_groups))
        raise ValueError(f"{source} is missing requested groups: {missing}")
    records = [
        row
        for row in records
        if row.get("comparison_group", "") in selected_groups
    ]

plot_dashboard(
    records,
    output,
    "Evaluated prefix-length comparison",
    args.dpi,
)
print(f"Wrote {output}")

groups = dict(sorted(grouped(records).items(), key=sort_key))
fixed_M_values = sorted(
    {
        float(rows[0]["compared_M"])
        for rows in groups.values()
        if rows[0].get("comparison_group") == "eta_2f_fixed_M"
    }
)
colors = (
    np.asarray([[0.2, 0.2, 0.2, 1.0]])
    if not fixed_M_values
    else plt.get_cmap("viridis")(
        np.linspace(0.12, 0.9, len(fixed_M_values))
    )
)
fixed_colors = dict(zip(fixed_M_values, colors))
spans = offline_spans(records)
rate_mode = records[0].get("nonzero_rate_mode", "")
title = "Prefix-length comparison"
if rate_mode:
    title += f" ({rate_mode} non-zero rates)"

single_metrics = (
    ("rho_se", r"$\rho_{se}$", "student_environment_correlation"),
    ("M", r"selected $M$", "prefix_length"),
    ("reward_rate", "reward rate", "reward_rate"),
)
for metric, ylabel, stem_suffix in single_metrics:
    single_output = output.with_name(
        f"{output.stem}_{stem_suffix}{output.suffix}"
    )
    figure, axis = plt.subplots(figsize=(5.4, 3.8))
    shade_offline(axis, spans)
    for rows in groups.values():
        usable = [
            row
            for row in rows
            if row.get(metric, "") not in {"", None}
            and np.isfinite(float(row[metric]))
        ]
        if not usable:
            continue
        axis.plot(
            [float(row["time"]) for row in usable],
            [float(row[metric]) for row in usable],
            label=label_for(rows),
            **condition_style(rows, fixed_colors),
        )
    axis.set_xlabel("elapsed time")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    if spans:
        handles.append(Patch(facecolor="#e9ecef", edgecolor="none"))
        labels.append("offline")
    if handles and (metric == "rho_se"):
        axis.legend(handles, labels, frameon=False)
        axis.set_ylim(0, 1)
    #figure.suptitle(title)
    figure.tight_layout()
    single_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(single_output, dpi=args.dpi, transparent=True)
    plt.close(figure)
    print(f"Wrote {single_output}")
