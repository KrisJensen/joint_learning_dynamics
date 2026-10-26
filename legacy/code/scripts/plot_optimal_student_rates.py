"""Compute and plot optimal student rates when the world model is perfect."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
sys.path.insert(0, str(CODE_ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

from joint_dynamics import Params, State, components  # noqa: E402


def safe_subdir(subdir: str) -> Path:
    subdir = subdir.strip().strip("/")
    if not subdir:
        return Path()
    path = Path(subdir)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("output subdirectory must be a relative path without '.' or '..'")
    return path


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rho-min", type=float, default=0.00)
parser.add_argument("--rho-max", type=float, default=0.999)
parser.add_argument("--points", type=int, default=100)
parser.add_argument("--T", type=int, default=9)
parser.add_argument(
    "--prefix-lengths",
    default=None,
    help="comma-separated evaluated prefix lengths; defaults to 1..T",
)
parser.add_argument(
    "--K-values",
    default="1,5,25",
    help="comma-separated K values to plot as rows",
)
parser.add_argument("--beta", type=float, default=3.0)
parser.add_argument("--temperature-mode", choices=["constant", "optimal"], default="optimal")
parser.add_argument("--lambda-s", type=float, default=0.25)
parser.add_argument("--epsilon", type=float, default=0.0)
parser.add_argument("--eta-max", type=float, default=100.0)
parser.add_argument("--gh-order", type=int, default=20)
parser.add_argument(
    "--data-output",
    type=Path,
    default=None,
)
parser.add_argument(
    "--output",
    type=Path,
    default=None,
)
parser.add_argument("--suffix", default="")
parser.add_argument("--output-subdir", default="")
parser.add_argument("--dpi", type=int, default=180)
parser.add_argument(
    "--log-y",
    action="store_true",
    help="plot learning rates on a logarithmic y-axis",
)
parser.add_argument(
    "--y-min",
    type=float,
    default=None,
    help="minimum y-axis value for all panels",
)
parser.add_argument(
    "--optimum",
    choices=["joint", "axis-aligned"],
    default="joint",
    help="which online student-rate optimum to plot",
)
args = parser.parse_args()

suffix = args.suffix.strip().strip("_")
optimum_slug = args.optimum.replace("-", "_")
stem_base = f"optimal_student_learning_rates_{optimum_slug}"
stem = f"{stem_base}_{suffix}" if suffix else stem_base
subdir = safe_subdir(args.output_subdir)
data_output = args.data_output or REPO_ROOT / "results" / subdir / f"{stem}.csv"
figure_output = args.output or REPO_ROOT / "figures" / subdir / f"{stem}.pdf"
optimum_label = (
    "joint optimum" if args.optimum == "joint" else "axis-aligned optimum"
)

if not -1.0 < args.rho_min < args.rho_max < 1.0:
    parser.error("require -1 < rho-min < rho-max < 1")
if args.points < 2:
    parser.error("points must be at least two")
if args.y_min is not None and not np.isfinite(args.y_min):
    parser.error("y-min must be finite")
if args.log_y and args.y_min is not None and args.y_min <= 0.0:
    parser.error("y-min must be positive when using --log-y")

prefix_lengths = list(range(1, args.T + 1, 2))

k_values = [
    int(value.strip())
    for value in args.K_values.split(",")
    if value.strip()
]
if not k_values:
    parser.error("K-values must contain at least one integer")
if any(value < 1 for value in k_values):
    parser.error("K-values must be positive integers")

records = []
for K in k_values:
    for prefix_length in prefix_lengths:
        base_params = dict(
            T=args.T,
            prefix_length=prefix_length,
            K=K,
            beta=args.beta,
            temperature_mode=args.temperature_mode,
            lambda_s=args.lambda_s,
            epsilon=args.epsilon,
            learning_rate_mode="optimal",
            fix_eta_wm_zero=True,
            eta_max=args.eta_max,
            gh_order=args.gh_order,
        )
        joint_params = Params(**base_params)
        if args.optimum == "axis-aligned":
            only_eta_on1_params = Params(**base_params, fix_eta_on2_zero=True)
            only_eta_on2_params = Params(**base_params, fix_eta_on1_zero=True)
        for rho in np.linspace(args.rho_min, args.rho_max, args.points):
            state = State(rho_se=float(rho), rho_me=1.0, rho_sm=float(rho))
            joint = components(state, joint_params, "online")
            offline = components(state, joint_params, "offline")
            if args.optimum == "joint":
                eta_on1 = joint.eta_on1
                eta_on2 = joint.eta_on2
                eta_on1_source = joint
                eta_on2_source = joint
            else:
                eta_on1_source = components(state, only_eta_on1_params, "online")
                eta_on2_source = components(state, only_eta_on2_params, "online")
                eta_on1 = eta_on1_source.eta_on1
                eta_on2 = eta_on2_source.eta_on2

            linear_eta_on1 = eta_on1_source.B0 - rho * eta_on1_source.C0
            linear_eta_on2 = eta_on2_source.BR - rho * eta_on2_source.CR
            eta_on1_dot_rho_se_component = (
                eta_on1_source.environment_modulation
                * (
                    eta_on1 * linear_eta_on1
                    - rho * eta_on1**2 * eta_on1_source.D0 / (2.0 * joint_params.T)
                )
                / joint_params.c_on
            )
            eta_on2_dot_rho_se_component = (
                eta_on2_source.environment_modulation
                * (
                    eta_on2 * linear_eta_on2
                    - rho * eta_on2**2 * eta_on2_source.DR / (2.0 * joint_params.T)
                )
                / joint_params.c_on
            )
            records.append(
                {
                    "K": K,
                    "prefix_length": prefix_length,
                    "rho_se": float(rho),
                    "optimum": args.optimum,
                    "eta_on1": eta_on1,
                    "eta_on2": eta_on2,
                    "eta_on_rewarded": eta_on1 + eta_on2,
                    "eta_off": offline.eta_off,
                    "eta_on1_dot_rho_se_component": eta_on1_dot_rho_se_component,
                    "eta_on2_dot_rho_se_component": eta_on2_dot_rho_se_component,
                    "beta_on1": eta_on1_source.beta,
                    "beta_on2": eta_on2_source.beta,
                }
            )

data_output.parent.mkdir(parents=True, exist_ok=True)
with data_output.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


figure, axes = plt.subplots(
    len(k_values),
    2,
    figsize=(12.8, 3.6 * len(k_values)),
    sharex=True,
    squeeze=False,
)
colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.9, len(prefix_lengths)))
for row_index, K in enumerate(k_values):
    joint_axis = axes[row_index, 0]
    drift_axis = axes[row_index, 1]
    if row_index == 0:
        joint_axis.set_title(f"{optimum_label} learning rates")
        drift_axis.set_title(
            rf"{optimum_label} $\dot{{\rho}}_{{se}}$ components"
        )
    for color, prefix_length in zip(colors, prefix_lengths):
        rows = [
            row
            for row in records
            if row["K"] == K and row["prefix_length"] == prefix_length
        ]
        x_values = [row["rho_se"] for row in rows]
        eta_on2 = [row["eta_on2"] for row in rows]
        eta_on1 = [row["eta_on1"] for row in rows]
        eta_on2_drift = [row["eta_on2_dot_rho_se_component"] for row in rows]
        eta_on1_drift = [row["eta_on1_dot_rho_se_component"] for row in rows]
        if args.log_y:
            eta_on2 = [value if value > 0 else np.nan for value in eta_on2]
            eta_on1 = [value if value > 0 else np.nan for value in eta_on1]
            eta_on2_drift = [
                value if value > 0 else np.nan for value in eta_on2_drift
            ]
            eta_on1_drift = [
                value if value > 0 else np.nan for value in eta_on1_drift
            ]
        joint_axis.plot(
            x_values,
            eta_on2,
            color=color,
            linestyle="-",
            linewidth=2.0,
        )
        joint_axis.plot(
            x_values,
            eta_on1,
            color=color,
            linestyle="--",
            linewidth=2.0,
        )
        drift_axis.plot(
            x_values,
            eta_on2_drift,
            color=color,
            linestyle="-",
            linewidth=2.0,
        )
        drift_axis.plot(
            x_values,
            eta_on1_drift,
            color=color,
            linestyle="--",
            linewidth=2.0,
        )
    for axis in (joint_axis, drift_axis):
        if args.log_y:
            axis.set_yscale("log")
        if args.y_min is not None:
            axis.set_ylim(bottom=args.y_min)
        axis.set_xlim(args.rho_min, args.rho_max)
        axis.spines[["top", "right"]].set_visible(False)
        if not args.log_y:
            axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    joint_axis.set_ylabel(f"K={K}\noptimal learning rate")
    drift_axis.set_ylabel(r"$d\rho_{se}/d\alpha$")
for axis in axes[-1, :]:
    axis.set_xlabel(r"student-environment correlation $\rho_{se}$")

prefix_handles = [
    Line2D([0], [0], color=color, linewidth=2.0, label=rf"$M={prefix_length}$")
    for color, prefix_length in zip(colors, prefix_lengths)
]
rate_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="-",
        linewidth=2.0,
        label=rf"$\eta_{{\mathrm{{on}},2}}^*$",
    ),
    Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=rf"$\eta_{{\mathrm{{on}},1}}^*$",
    ),
]
figure.legend(
    handles=prefix_handles,
    title="prefix length",
    frameon=False,
    ncol=min(len(prefix_handles), 6),
    loc="lower center",
    bbox_to_anchor=(0.46, 0.0),
)
figure.legend(
    handles=rate_handles,
    title="rate",
    frameon=False,
    loc="lower right",
    bbox_to_anchor=(0.99, 0.0),
)
figure.tight_layout(rect=(0.0, 0.08, 1.0, 1.0))
figure_output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(figure_output, dpi=args.dpi, transparent=True)
plt.close(figure)

print(f"Wrote {data_output}")
print(f"Wrote {figure_output}")
