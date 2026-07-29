"""Plot selected DP rollouts referenced by phase-diagram result rows."""

import argparse
from pathlib import Path
import sys

import numpy as np

from _common import CODE_ROOT, FIGURES_ROOT, RESULTS_ROOT, _safe_subdir, read_records

SRC = CODE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from learning_dynamics import State, load_dp_result, rollout_policy  # noqa: E402

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OFFLINE_SHADE = "#dbeafe"
SWITCH_SHADE = "#60a5fa"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--examples", required=True, help="semicolon-separated rho_se,rho_me pairs")
parser.add_argument("--input", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--suffix", default="")
parser.add_argument("--output-subdir", default="")
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()
suffix = f"_{args.suffix.strip('_')}" if args.suffix.strip("_") else ""
subdir = _safe_subdir(args.output_subdir)
source = args.input or RESULTS_ROOT / subdir / f"phase_diagram_initial{suffix}.csv"
output = args.output or FIGURES_ROOT / subdir / f"phase_diagram_initial_examples{suffix}.pdf"
rows = read_records(source)
targets = [
    tuple(map(float, item.split(",", 1)))
    for item in args.examples.split(";")
    if item.strip()
]
selected = [
    min(
        rows,
        key=lambda row: (float(row["rho_se0"]) - target[0]) ** 2
        + (float(row["rho_me0"]) - target[1]) ** 2,
    )
    for target in targets
]
figure, axes = plt.subplots(len(selected), 2, figsize=(10, 3.5 * len(selected)), squeeze=False)
cache = {}
for row_index, row in enumerate(selected):
    path = row["table_path"]
    if path not in cache:
        cache[path] = load_dp_result(path)
    result = cache[path]
    state = State(float(row["rho_se0"]), float(row["rho_me0"]), float(row["rho_sm0"]))
    rollout = rollout_policy(result, state)
    times = [float(item["time"]) for item in rollout]
    offline_spans = [
        (float(item["time"]), float(next_item["time"]))
        for item, next_item in zip(rollout, rollout[1:])
        if item["mode"] == "off"
        and item["action"] != "switch"
        and float(next_item["time"]) > float(item["time"])
    ]
    switch_spans = [
        (float(item["time"]), float(next_item["time"]))
        for item, next_item in zip(rollout, rollout[1:])
        if item["action"] == "switch"
        and float(next_item["time"]) > float(item["time"])
    ]
    for axis in axes[row_index]:
        for start, end in offline_spans:
            axis.axvspan(start, end, color=OFFLINE_SHADE, zorder=0)
        for start, end in switch_spans:
            axis.axvspan(start, end, color=SWITCH_SHADE, zorder=0)

    for name in ("rho_se", "rho_me", "rho_sm"):
        axes[row_index, 0].plot(times, [float(item[name]) for item in rollout], label=name)
    axes[row_index, 1].step(
        times,
        [float(item["K"]) if item.get("K", "") not in {"", None} else np.nan for item in rollout],
        where="post",
    )
    handles, labels = axes[row_index, 0].get_legend_handles_labels()
    if offline_spans:
        handles.append(Patch(facecolor=OFFLINE_SHADE, edgecolor="none"))
        labels.append("offline")
    if switch_spans:
        handles.append(Patch(facecolor=SWITCH_SHADE, edgecolor="none"))
        labels.append("switch")
    axes[row_index, 0].legend(handles, labels, frameon=False)
    axes[row_index, 0].set_ylabel("correlation")
    axes[row_index, 1].set_ylabel("K")
    for axis in axes[row_index]:
        axis.set_xlabel("elapsed time")
        axis.grid(axis="y", color="#dddddd")
figure.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=args.dpi, transparent=True)
plt.close(figure)
print(f"Wrote {output}")
