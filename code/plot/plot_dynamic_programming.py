"""Plot a dynamic-programming rollout."""

import argparse
from pathlib import Path

import numpy as np

from _common import FIGURES_ROOT, RESULTS_ROOT, _safe_subdir, read_records

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OFFLINE_SHADE = "#dbeafe"
SWITCH_SHADE = "#60a5fa"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=["online", "switching"], default="online")
parser.add_argument("--fixed-wm", action="store_true")
parser.add_argument("--suffix", default="")
parser.add_argument("--output-subdir", default="")
parser.add_argument("--input", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()
stem = f"dynamic_programming_{args.variant}"
if args.fixed_wm:
    stem += "_fixed_wm"
if args.suffix.strip("_"):
    stem += f"_{args.suffix.strip('_')}"
subdir = _safe_subdir(args.output_subdir)
source = args.input or RESULTS_ROOT / subdir / f"{stem}.csv"
output = args.output or FIGURES_ROOT / subdir / f"{stem}.pdf"
rows = read_records(source)
times = np.asarray([float(row["time"]) for row in rows])
offline_spans = [
    (float(row["time"]), float(next_row["time"]))
    for row, next_row in zip(rows, rows[1:])
    if row["mode"] == "off"
    and row["action"] != "switch"
    and float(next_row["time"]) > float(row["time"])
]
switch_spans = [
    (float(row["time"]), float(next_row["time"]))
    for row, next_row in zip(rows, rows[1:])
    if row["action"] == "switch"
    and float(next_row["time"]) > float(row["time"])
]

figure, axes = plt.subplots(4, 1, figsize=(9, 10), sharex=True)
for axis in axes:
    for start, end in offline_spans:
        axis.axvspan(start, end, color=OFFLINE_SHADE, zorder=0)
    for start, end in switch_spans:
        axis.axvspan(start, end, color=SWITCH_SHADE, zorder=0)

for name, label in (
    ("rho_se", r"$\rho_{se}$"),
    ("rho_me", r"$\rho_{me}$"),
    ("rho_sm", r"$\rho_{sm}$"),
):
    axes[0].plot(times, [float(row[name]) for row in rows], label=label)
handles, labels = axes[0].get_legend_handles_labels()
if offline_spans:
    handles.append(Patch(facecolor=OFFLINE_SHADE, edgecolor="none"))
    labels.append("offline")
if switch_spans:
    handles.append(Patch(facecolor=SWITCH_SHADE, edgecolor="none"))
    labels.append("switch")
axes[0].legend(handles, labels, frameon=False)
for axis, key, label in zip(
    axes[1:],
    ("K", "M", "immediate_reward"),
    (r"$K$", r"$M$", "immediate reward"),
):
    values = [
        float(row[key]) if row.get(key, "") not in {"", None} else np.nan
        for row in rows
    ]
    axis.step(times, values, where="post")
    axis.set_ylabel(label)
for axis in axes:
    axis.grid(axis="y", color="#dddddd")
    axis.spines[["top", "right"]].set_visible(False)
axes[-1].set_xlabel("elapsed time")
figure.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=args.dpi, transparent=True)
plt.close(figure)
print(f"Wrote {output}")
