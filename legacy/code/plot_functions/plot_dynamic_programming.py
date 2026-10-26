"""Plot a rollout produced by run_dynamic_programming.py."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import sys

import numpy as np

from _common import REPO_ROOT, figures_dir, results_dir

SRC = REPO_ROOT / "code" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from joint_dynamics import load_dp_result

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--variant",
    choices=["online", "switching"],
    default="online",
)
parser.add_argument("--fixed-wm", action="store_true")
parser.add_argument("--partial-evaluation", action="store_true")
parser.add_argument("--suffix", default="")
parser.add_argument("--input", type=Path, default=None)
parser.add_argument("--table-input", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--output-subdir",
    default="",
    help="relative subdirectory under results/ and figures/ for default paths",
)
parser.add_argument(
    "--plot-rho-se",
    action="store_true",
    help="also write a single-panel figure of student-environment correlation over time",
)
parser.add_argument(
    "--plot-k",
    "--plot-K",
    dest="plot_k",
    action="store_true",
    help="also write a single-panel figure of chosen K over time",
)
parser.add_argument(
    "--plot-prefix-length",
    action="store_true",
    help="also write a single-panel figure of chosen prefix length over time",
)
parser.add_argument(
    "--plot-beta",
    "--plot-temperature",
    dest="plot_beta",
    action="store_true",
    help="also write a single-panel figure of planning inverse temperature beta over time",
)
parser.add_argument(
    "--plot-individuals",
    action="store_true",
    help="also write all single-panel trajectory figures",
)
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()

suffix = args.suffix.strip().strip("_")
stem = f"dynamic_programming_{args.variant}"
if args.fixed_wm:
    stem += "_fixed_wm"
if args.partial_evaluation:
    stem += "_partial"
if suffix:
    stem += f"_{suffix}"
source = args.input or results_dir(args.output_subdir) / f"{stem}.csv"
table_source = args.table_input or results_dir(args.output_subdir) / f"{stem}.npz"
destination = args.output or figures_dir(args.output_subdir) / f"{stem}.pdf"
with source.open(newline="") as handle:
    rows = list(csv.DictReader(handle))

times = [float(row["time"]) for row in rows]
offline_spans = [
    (float(row["time"]), float(next_row["time"]))
    for row, next_row in zip(rows, rows[1:])
    if row["mode"] == "off"
    and row["action"] != "switch"
    and float(next_row["time"]) > float(row["time"])
]
switching_spans = [
    (float(row["time"]), float(next_row["time"]))
    for row, next_row in zip(rows, rows[1:])
    if row["action"] == "switch"
    and float(next_row["time"]) > float(row["time"])
]
result = load_dp_result(table_source)
has_modulation = result.params.M != 0.0
has_partial_evaluation = result.params.partial_evaluation
trajectory_panel_count = 4 + int(has_partial_evaluation) + int(has_modulation)
policy_panel_count = 2 if result.variant == "switching" else 1
panel_count = trajectory_panel_count + policy_panel_count
fig, axes = plt.subplots(panel_count, 1, figsize=(9, 3 * panel_count))
for axis in axes[:trajectory_panel_count]:
    for start, end in offline_spans:
        axis.axvspan(start, end, color="#e9ecef", zorder=0)
    for start, end in switching_spans:
        axis.axvspan(start, end, color="#dbeafe", zorder=0)
for name, label in (
    ("rho_se", r"$\rho_{se}$"),
    ("rho_me", r"$\rho_{me}$"),
    ("rho_sm", r"$\rho_{sm}$"),
):
    axes[0].plot(times, [float(row[name]) for row in rows], label=label)
axes[0].set_ylabel("correlation")
handles, legend_labels = axes[0].get_legend_handles_labels()
if offline_spans:
    handles.append(Patch(facecolor="#e9ecef", edgecolor="none"))
    legend_labels.append("offline")
if switching_spans:
    handles.append(Patch(facecolor="#dbeafe", edgecolor="none"))
    legend_labels.append("switching")
axes[0].legend(handles, legend_labels, frameon=False, ncol=4)

K_values = [float(row["K"]) if row["K"] else float("nan") for row in rows]
axes[1].plot(times, K_values, label="K")
axes[1].set_ylabel("action")
axes[1].legend(frameon=False)

axis_index = 2
if has_partial_evaluation:
    prefix_values = [
        float(row["prefix_length"]) if row.get("prefix_length") else float("nan")
        for row in rows
    ]
    axes[axis_index].plot(times, prefix_values, label="prefix length")
    axes[axis_index].set_ylabel("prefix length")
    axes[axis_index].legend(frameon=False)
    axis_index += 1

axes[axis_index].plot(
    times,
    [float(row["immediate_reward"]) for row in rows],
    label="reward per physical action",
)
axes[axis_index].set_ylabel("reward")
axes[axis_index].legend(frameon=False)
axis_index += 1

for name, label in (
    ("eta_on1", r"$\eta_{on,1}$"),
    ("eta_on2", r"$\eta_{on,2}$"),
    ("eta_off", r"$\eta_{off}$"),
    ("eta_wm", r"$\eta_{wm}$"),
):
    values = np.array(
        [
            float(row[name])
            if row.get(name) not in (None, "")
            else np.nan
            for row in rows
        ],
        dtype=float,
    )
    if np.any(np.isfinite(values)):
        axes[axis_index].plot(times, values, label=label)
axes[axis_index].set_ylabel("learning rate")
if axes[axis_index].get_legend_handles_labels()[0]:
    axes[axis_index].legend(frameon=False, ncol=4)
axis_index += 1

if has_modulation:
    axes[axis_index].plot(
        times,
        [float(row["environment_modulation"]) for row in rows],
        color="#7b2cbf",
        label=r"$m(t)$",
    )
    axes[axis_index].axhline(1.0, color="#999999", linewidth=0.8, linestyle=":")
    axes[axis_index].set_ylabel(r"$m(t)$")
    axes[axis_index].legend(frameon=False)

if result.config.fixed_wm:
    rho_values = np.unique(result.points[:, 0])
    rho_me_slice = result.config.fixed_rho_me
    rho_sm_values = rho_values * rho_me_slice
else:
    rho_values = np.linspace(
        result.config.rho_min,
        result.config.rho_max,
        result.config.rho_points,
    )
    rho_me_slice = float(np.max(result.points[:, 1]))
    rho_sm_values = rho_values * rho_me_slice
if result.config.fixed_wm:
    queries = np.column_stack([rho_values, rho_sm_values])
else:
    queries = np.column_stack(
        [rho_values, np.full_like(rho_values, rho_me_slice), rho_sm_values]
    )
nearest = result.interpolator.nearest(queries)
stop_index = len(result.action_names)
labels = list(result.action_names) + ["stop"]
online_indices = [
    index for index, name in enumerate(result.action_names) if name.startswith("online(")
]
online_colors = plt.get_cmap("viridis")(
    np.linspace(0.12, 0.9, max(len(online_indices), 1))
)
colors = []
online_position = 0
for name in result.action_names:
    if name.startswith("online("):
        colors.append(online_colors[online_position])
        online_position += 1
    elif name == "switch":
        colors.append("#d627a3")
    elif name == "offline":
        colors.append("#f28e2b")
    else:
        colors.append("#7f7f7f")
colors.append("#d9d9d9")
cmap = ListedColormap(colors)
norm = BoundaryNorm(np.arange(-0.5, len(labels) + 0.5), cmap.N)
if result.policy.ndim == 2:
    policy_tables = [("online state", result.policy)]
else:
    policy_tables = [
        ("online state", result.policy[0]),
        ("offline state", result.policy[1]),
    ]
for axis, (mode_label, policy_table) in zip(
    axes[trajectory_panel_count:], policy_tables
):
    policy_slice = policy_table[:, nearest]
    display_policy = np.where(policy_slice < 0, stop_index, policy_slice)
    image = axis.pcolormesh(
        rho_values,
        result.times,
        display_policy,
        shading="nearest",
        cmap=cmap,
        norm=norm,
    )
    colorbar = fig.colorbar(image, ax=axis, ticks=np.arange(len(labels)))
    colorbar.ax.set_yticklabels(
        ["change state" if label == "switch" else label for label in labels]
    )
    axis.set_ylabel("time to go")
    axis.set_xlabel(r"$\rho_{se}$")
    axis.set_title(
        rf"optimal policy ({mode_label}) at $\rho_{{me}}={rho_me_slice:g}$"
    )

for axis in axes[:trajectory_panel_count]:
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
axes[trajectory_panel_count - 1].set_xlabel("elapsed time")
fig.suptitle(f"Dynamic programming: {result.variant}, fixed_wm={result.config.fixed_wm}")
fig.tight_layout()
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=args.dpi, transparent=True)
plt.close(fig)
print(f"Wrote {destination}")

individual_specs = []
if args.plot_individuals or args.plot_rho_se:
    individual_specs.append(("rho_se", r"$\rho_{se}$", "rho_se"))
if args.plot_individuals or args.plot_k:
    individual_specs.append(("K", r"$K$", "K"))
if args.plot_individuals or args.plot_prefix_length:
    individual_specs.append(("prefix_length", "prefix length", "prefix_length"))
if args.plot_individuals or args.plot_beta:
    individual_specs.append(("beta", r"$\beta$", "beta"))

for metric, ylabel, suffix_name in individual_specs:
    individual_destination = destination.with_name(
        f"{destination.stem}_{suffix_name}{destination.suffix}"
    )
    fig, axis = plt.subplots(figsize=(5.6, 3.8))
    for start, end in offline_spans:
        axis.axvspan(start, end, color="#e9ecef", zorder=0)
    for start, end in switching_spans:
        axis.axvspan(start, end, color="#dbeafe", zorder=0)
    usable_times = []
    usable_values = []
    for row in rows:
        value = row.get(metric, "")
        if value in ("", None):
            continue
        value = float(value)
        if not np.isfinite(value):
            continue
        usable_times.append(float(row["time"]))
        usable_values.append(value)
    if usable_values:
        axis.plot(usable_times, usable_values, color="#1f77b4", linewidth=2.4)
    axis.set_xlabel("elapsed time")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    handles = []
    labels = []
    if offline_spans:
        handles.append(Patch(facecolor="#e9ecef", edgecolor="none"))
        labels.append("offline")
    if switching_spans:
        handles.append(Patch(facecolor="#dbeafe", edgecolor="none"))
        labels.append("switching")
    if handles:
        axis.legend(handles, labels, frameon=False)
    fig.suptitle(
        f"Dynamic programming: {result.variant}, fixed_wm={result.config.fixed_wm}"
    )
    fig.tight_layout()
    individual_destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(individual_destination, dpi=args.dpi, transparent=True)
    plt.close(fig)
    print(f"Wrote {individual_destination}")
