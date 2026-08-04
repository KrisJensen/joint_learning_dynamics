"""Plot a dynamic-programming rollout."""

import argparse
import json
from pathlib import Path
import shlex

import numpy as np

from _common import FIGURES_ROOT, RESULTS_ROOT, _safe_subdir, read_records

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


OFFLINE_SHADE = "#dbeafe"
SWITCH_SHADE = "#60a5fa"


def command_float(tokens: list[str], flag: str, default: float) -> float:
    """Read a numeric option from a recorded command line."""

    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return float(tokens[index + 1])
        prefix = f"{flag}="
        if token.startswith(prefix):
            return float(token[len(prefix):])
    return default


def cycle_parameters(source: Path) -> tuple[float, float, float] | None:
    """Load cycle parameters from the lightweight result manifest."""

    manifest_path = source.with_suffix(".json")
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        metadata = manifest.get("metadata", {})
        if all(name in metadata for name in ("C_A", "C_T", "C_phi")):
            return (
                float(metadata["C_A"]),
                float(metadata["C_T"]),
                float(metadata["C_phi"]),
            )
        # Schema-v1 manifests only recorded the command. This compatibility
        # path lets existing modulated results produce the new figure.
        tokens = shlex.split(manifest.get("command", ""))
        return (
            command_float(tokens, "--C-A", 0.0),
            command_float(tokens, "--C-T", 1.0),
            command_float(tokens, "--C-phi", 0.0),
        )
    except (json.JSONDecodeError, OSError, ValueError):
        return None


def model_horizon(source: Path, rows: list[dict]) -> int:
    """Read ``T`` from the result manifest, with legacy-result fallbacks."""

    manifest_path = source.with_suffix(".json")
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            metadata = manifest.get("metadata", {})
            if "T" in metadata:
                return int(metadata["T"])
            tokens = shlex.split(manifest.get("command", ""))
            # Six is the run-script default and therefore also covers legacy
            # manifests whose command did not explicitly include --T.
            return int(command_float(tokens, "--T", 6.0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            pass
    observed = [
        int(float(row["M"]))
        for row in rows
        if row.get("M", "") not in {"", None}
    ]
    if not observed:
        raise ValueError("cannot determine T from the result manifest or rollout")
    return max(observed)


def uses_constant_prefix_time(source: Path) -> bool:
    """Return whether the rollout used a prefix-length-independent cost."""

    manifest_path = source.with_suffix(".json")
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
        metadata = manifest.get("metadata", {})
        if "constant_prefix_time" in metadata:
            return bool(metadata["constant_prefix_time"])
        tokens = shlex.split(manifest.get("command", ""))
        return "--constant-prefix-time" in tokens
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return False


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=["online", "switching"], default="online")
parser.add_argument("--fixed-wm", action="store_true")
parser.add_argument("--suffix", default="")
parser.add_argument("--output-subdir", default="")
parser.add_argument("--input", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--plot-rho-se",
    "--plot-correlations",
    dest="plot_rho_se",
    action="store_true",
    help="also write a single-panel figure containing all three correlations",
)
parser.add_argument(
    "--plot-k",
    "--plot-K",
    dest="plot_k",
    action="store_true",
    help="also write a single-panel combined K/M figure",
)
parser.add_argument(
    "--plot-prefix-length",
    action="store_true",
    help="also write a single-panel combined K/M figure",
)
parser.add_argument(
    "--plot-beta",
    "--plot-temperature",
    dest="plot_beta",
    action="store_true",
    help="also write a single-panel inverse-temperature figure",
)
parser.add_argument(
    "--plot-modulation",
    action="store_true",
    help="write the cycle-modulation figure when cycle parameters are available",
)
parser.add_argument(
    "--plot-individuals",
    action="store_true",
    help="write all legacy single-panel trajectory figures",
)
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
if not rows:
    raise ValueError(f"no records found in {source}")
if all(row.get("action") == "value_only" for row in rows):
    raise ValueError(
        "this result contains only a final-horizon value; positive "
        "t_strategy_switch runs do not store the history required for "
        "trajectory plots"
    )
times = np.asarray([float(row["time"]) for row in rows])
T = model_horizon(source, rows)
constant_prefix_time = uses_constant_prefix_time(source)
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

K_COLOR = "#1f77b4"
M_COLOR = "#f28e2b"
K_values = [
    float(row["K"]) if row.get("K", "") not in {"", None} else np.nan
    for row in rows
]
M_values = []
for row, K in zip(rows, K_values):
    M = float(row["M"]) if row.get("M", "") not in {"", None} else np.nan
    # When only one plan is sampled, M has no effect under a constant prefix
    # duration. Hide the arbitrary optimizer tie-break in that regime.
    if constant_prefix_time and np.isclose(K, 1.0):
        M = np.nan
    M_values.append(M)
show_M = bool(np.any(np.isfinite(M_values)))
planning_axis = axes[1]
K_line = planning_axis.step(
    times,
    K_values,
    where="post",
    color=K_COLOR,
    label=r"$K$",
)[0]
planning_axis.set_ylabel(r"$K$", color=K_COLOR)
planning_axis.tick_params(axis="y", labelcolor=K_COLOR)
planning_handles = [K_line]
if show_M:
    prefix_axis = planning_axis.twinx()
    M_line = prefix_axis.step(
        times,
        M_values,
        where="post",
        color=M_COLOR,
        label=r"$M$",
    )[0]
    prefix_axis.set_ylabel(r"$M$", color=M_COLOR)
    prefix_axis.tick_params(axis="y", labelcolor=M_COLOR)
    prefix_axis.spines["top"].set_visible(False)
    prefix_axis.set_ylim(0, T + 1)
    planning_handles.append(M_line)
planning_axis.legend(handles=planning_handles, frameon=False)

for axis, key, label in zip(
    axes[2:],
    ("reward_probability", "reward_rate"),
    ("reward probability", "reward rate"),
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

individual_specs = []
if args.plot_individuals or args.plot_rho_se:
    individual_specs.append(
        (
            (
                ("rho_se", r"$\rho_{se}$"),
                ("rho_me", r"$\rho_{me}$"),
                ("rho_sm", r"$\rho_{sm}$"),
            ),
            "correlation",
            "rho_se",
        )
    )
if args.plot_individuals or args.plot_beta:
    individual_specs.append(((("beta", None),), r"$\beta$", "beta"))

for series, ylabel, suffix_name in individual_specs:
    individual_output = output.with_name(
        f"{output.stem}_{suffix_name}{output.suffix}"
    )
    figure, axis = plt.subplots(figsize=(5.6, 3.8))
    for start, end in offline_spans:
        axis.axvspan(start, end, color=OFFLINE_SHADE, zorder=0)
    for start, end in switch_spans:
        axis.axvspan(start, end, color=SWITCH_SHADE, zorder=0)
    for metric, label in series:
        usable_times = []
        usable_values = []
        for row in rows:
            value = row.get(metric, "")
            if value in {"", None}:
                continue
            value = float(value)
            if not np.isfinite(value):
                continue
            usable_times.append(float(row["time"]))
            usable_values.append(value)
        if usable_values:
            axis.plot(
                usable_times,
                usable_values,
                color="#1f77b4" if len(series) == 1 else None,
                linewidth=2.4,
                label=label,
            )
    axis.set_xlabel("elapsed time")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    if offline_spans:
        handles.append(Patch(facecolor=OFFLINE_SHADE, edgecolor="none"))
        labels.append("offline")
    if switch_spans:
        handles.append(Patch(facecolor=SWITCH_SHADE, edgecolor="none"))
        labels.append("switch")
    if handles:
        axis.legend(handles, labels, frameon=False)
    # figure.suptitle(
    #     f"Dynamic programming: {args.variant}, fixed_wm={args.fixed_wm}"
    # )
    figure.tight_layout()
    individual_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(individual_output, dpi=args.dpi, transparent=True)
    plt.close(figure)
    print(f"Wrote {individual_output}")

if args.plot_individuals or args.plot_k or args.plot_prefix_length:
    planning_output = output.with_name(
        f"{output.stem}_K_M{output.suffix}"
    )
    figure, planning_axis = plt.subplots(figsize=(5.6, 3.8))
    for start, end in offline_spans:
        planning_axis.axvspan(start, end, color=OFFLINE_SHADE, zorder=0)
    for start, end in switch_spans:
        planning_axis.axvspan(start, end, color=SWITCH_SHADE, zorder=0)
    K_line = planning_axis.step(
        times,
        K_values,
        where="post",
        color=K_COLOR,
        linewidth=2.4,
        label=r"$K$",
    )[0]
    planning_axis.set_xlabel("elapsed time")
    planning_axis.set_ylabel(r"$K$", color=K_COLOR)
    planning_axis.tick_params(axis="y", labelcolor=K_COLOR)
    planning_axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    planning_axis.spines["top"].set_visible(False)
    planning_axis.set_ylim(0, max(K_values) + 1)
    handles = [K_line]
    labels = [r"$K$"]
    if show_M:
        prefix_axis = planning_axis.twinx()
        M_line = prefix_axis.step(
            times,
            M_values,
            where="post",
            color=M_COLOR,
            linewidth=2.4,
            label=r"$M$",
        )[0]
        prefix_axis.set_ylabel(r"$M$", color=M_COLOR)
        prefix_axis.tick_params(axis="y", labelcolor=M_COLOR)
        prefix_axis.spines["top"].set_visible(False)
        prefix_axis.set_ylim(0, T + 1)
        handles.append(M_line)
        labels.append(r"$M$")
    if offline_spans:
        handles.append(Patch(facecolor=OFFLINE_SHADE, edgecolor="none"))
        labels.append("offline")
    if switch_spans:
        handles.append(Patch(facecolor=SWITCH_SHADE, edgecolor="none"))
        labels.append("switch")
    planning_axis.legend(handles, labels, frameon=False)
    figure.tight_layout()
    planning_output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(planning_output, dpi=args.dpi, transparent=True)
    plt.close(figure)
    print(f"Wrote {planning_output}")

cycle = cycle_parameters(source)
if (args.plot_individuals or args.plot_modulation) and cycle is not None:
    C_A, C_T, C_phi = cycle
    if C_A != 0.0:
        modulation_output = output.with_name(
            f"{output.stem}_modulation{output.suffix}"
        )
        start_time = float(np.min(times))
        end_time = float(np.max(times))
        period_count = max((end_time - start_time) / C_T, 1.0)
        sample_count = max(400, int(np.ceil(200.0 * period_count)) + 1)
        modulation_times = np.linspace(start_time, end_time, sample_count)
        modulation_values = 1.0 + C_A * np.sin(
            2.0 * np.pi * (modulation_times / C_T + C_phi)
        )
        figure, axis = plt.subplots(figsize=(5.6, 3.8))
        axis.plot(
            modulation_times,
            modulation_values,
            color="#59a14f",
            linewidth=2.4,
        )
        axis.set_xlabel("elapsed time")
        axis.set_ylabel(r"$C(t)$")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        figure.tight_layout()
        modulation_output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(modulation_output, dpi=args.dpi, transparent=True)
        plt.close(figure)
        print(f"Wrote {modulation_output}")
