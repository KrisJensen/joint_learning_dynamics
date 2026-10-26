"""Plot the prefix-length comparison."""

from __future__ import annotations

import argparse

import numpy as np

from _common import add_plot_args, grouped, offline_spans, paths, read_records, shade_offline

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def label_for(rows: list[dict]) -> str:
    first = rows[0]
    group = first.get("comparison_group", "")
    prefix = first.get("compared_prefix_length", "")
    if group == "eta_on1_fixed_prefix":
        return rf"$\eta_{{on,1}}$, $M={prefix}$"
    if group == "eta_on1_optimal_prefix":
        return rf"$\eta_{{on,1}}$, optimal $M$"
    if group == "eta_on2_only":
        return rf"$\eta_{{on,2}}$ only"
    return first.get("condition", "trajectory")


def condition_style(rows: list[dict], fixed_colors: dict[float, tuple]) -> dict:
    first = rows[0]
    group = first.get("comparison_group", "")
    if group == "eta_on1_fixed_prefix":
        prefix = float(first.get("compared_prefix_length", "nan"))
        return {
            "color": fixed_colors[prefix],
            "linewidth": 1.7,
            "alpha": 0.9,
        }
    if group == "eta_on1_optimal_prefix":
        return {"color": "black", "linewidth": 2.6}
    if group == "eta_on2_only":
        return {"color": "#d62728", "linestyle": "--", "linewidth": 2.3}
    return {}


def sort_key(item: tuple[str, list[dict]]) -> tuple[int, float, str]:
    condition, rows = item
    first = rows[0]
    group = first.get("comparison_group", "")
    if group == "eta_on1_fixed_prefix":
        return (0, float(first.get("compared_prefix_length", "inf")), condition)
    if group == "eta_on1_optimal_prefix":
        return (1, 0.0, condition)
    if group == "eta_on2_only":
        return (2, 0.0, condition)
    return (3, 0.0, condition)


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "prefix_length_comparison")
args = parser.parse_args()

source, destination = paths(args)
records = read_records(source)
groups = dict(sorted(grouped(records).items(), key=sort_key))
fixed_prefixes = sorted(
    {
        float(rows[0]["compared_prefix_length"])
        for rows in groups.values()
        if rows[0].get("comparison_group") == "eta_on1_fixed_prefix"
    }
)
colors = (
    np.asarray([[0.2, 0.2, 0.2, 1.0]])
    if not fixed_prefixes
    else plt.get_cmap("viridis")(np.linspace(0.12, 0.9, len(fixed_prefixes)))
)
fixed_colors = dict(zip(fixed_prefixes, colors))

metrics = (
    ("rho_se", r"$\rho_{se}$"),
    ("environment_reward_probability", "environment reward"),
    ("reward_rate", "reward rate"),
    ("prefix_length", r"selected $M$"),
    ("eta_on1", r"$\eta_{on,1}$"),
    ("eta_on2", r"$\eta_{on,2}$"),
)
fig, axes = plt.subplots(3, 2, figsize=(11.5, 9.0), sharex=True)
spans = offline_spans(records)
for axis, (metric, ylabel) in zip(axes.ravel(), metrics):
    shade_offline(axis, spans)
    for _, rows in groups.items():
        usable = [
            row
            for row in rows
            if row.get(metric, "") not in ("", None)
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
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
for axis in axes[-1]:
    axis.set_xlabel("elapsed time")

handles, labels = axes[0, 0].get_legend_handles_labels()
if spans:
    handles.append(Patch(facecolor="#e9ecef", edgecolor="none"))
    labels.append("offline")
if handles:
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=min(len(labels), 5),
    )
rate_mode = records[0].get("nonzero_rate_mode", "")
title = "Prefix-length comparison"
if rate_mode:
    title += f" ({rate_mode} non-zero rates)"
fig.suptitle(title, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.88))
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=args.dpi, transparent=True)
plt.close(fig)
print(f"Wrote {destination}")

single_metrics = (
    ("rho_se", r"$\rho_{se}$", "student_environment_correlation"),
    ("prefix_length", r"selected $M$", "prefix_length"),
    ("reward_rate", "reward rate", "reward_rate"),
)
for metric, ylabel, stem_suffix in single_metrics:
    single_destination = destination.with_name(
        f"{destination.stem}_{stem_suffix}{destination.suffix}"
    )
    fig, axis = plt.subplots(figsize=(5.4, 3.8))
    shade_offline(axis, spans)
    for _, rows in groups.items():
        usable = [
            row
            for row in rows
            if row.get(metric, "") not in ("", None)
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
    if handles:
        axis.legend(handles, labels, frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    single_destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(single_destination, dpi=args.dpi, transparent=True)
    plt.close(fig)
    print(f"Wrote {single_destination}")
