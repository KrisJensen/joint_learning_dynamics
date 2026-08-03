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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


OFFLINE_SHADE = "#dbeafe"
SWITCH_SHADE = "#60a5fa"


def setting_key(row: dict) -> tuple[float | None, float | None]:
    horizon = row.get("initial_horizon", "")
    switch_time = row.get("t_switch", "")
    return (
        None if horizon in {"", None} else float(horizon),
        None if switch_time in {"", None} else float(switch_time),
    )


def rollout_spans(rollout):
    """Return offline and switching spans for one rollout."""

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
    return offline_spans, switch_spans


def example_figure(rows, targets, cache):
    selected = [
        min(
            rows,
            key=lambda row: (float(row["rho_se0"]) - target[0]) ** 2
            + (float(row["rho_me0"]) - target[1]) ** 2,
        )
        for target in targets
    ]
    figure, axes = plt.subplots(
        len(selected),
        2,
        figsize=(10, 3.5 * len(selected)),
        squeeze=False,
    )
    rollouts = []
    for row_index, row in enumerate(selected):
        path = row["table_path"]
        if path not in cache:
            cache[path] = load_dp_result(path)
        result = cache[path]
        state = State(
            float(row["rho_se0"]),
            float(row["rho_me0"]),
            float(row["rho_sm0"]),
        )
        horizon = setting_key(row)[0]
        rollout = rollout_policy(result, state, horizon=horizon)
        rollouts.append(rollout)
        times = [float(item["time"]) for item in rollout]
        offline_spans, switch_spans = rollout_spans(rollout)
        for axis in axes[row_index]:
            for start, end in offline_spans:
                axis.axvspan(start, end, color=OFFLINE_SHADE, zorder=0)
            for start, end in switch_spans:
                axis.axvspan(start, end, color=SWITCH_SHADE, zorder=0)

        for name in ("rho_se", "rho_me", "rho_sm"):
            axes[row_index, 0].plot(
                times,
                [float(item[name]) for item in rollout],
                label=name,
            )
        axes[row_index, 0].set_ylim(0.0, 1.0)
        planning_durations = np.asarray(
            [
                float(item["K"])
                if item.get("K", "") not in {"", None}
                else np.nan
                for item in rollout
            ],
            dtype=float,
        )
        axes[row_index, 1].step(
            times,
            planning_durations,
            where="post",
        )
        finite_durations = planning_durations[np.isfinite(planning_durations)]
        highest_duration = (
            float(np.max(finite_durations)) if finite_durations.size else 0.0
        )
        axes[row_index, 1].set_ylim(0.0, highest_duration+1)
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
    return figure, selected, rollouts


def correlation_figure(
    row: dict,
    rollout,
    setting: tuple[float | None, float | None],
):
    """Plot only the three correlations for one selected example rollout."""

    times = [float(item["time"]) for item in rollout]
    offline_spans, switch_spans = rollout_spans(rollout)
    figure, axis = plt.subplots(figsize=(3.8, 3.8))
    for start, end in offline_spans:
        axis.axvspan(start, end, color=OFFLINE_SHADE, zorder=0)
    for start, end in switch_spans:
        axis.axvspan(start, end, color=SWITCH_SHADE, zorder=0)
    for name, label in (
        ("rho_se", r"$\rho_{se}$"),
        ("rho_me", r"$\rho_{me}$"),
        ("rho_sm", r"$\rho_{sm}$"),
    ):
        axis.plot(
            times,
            [float(item[name]) for item in rollout],
            linewidth=2.0,
            label=label,
        )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("elapsed time")
    axis.set_ylabel("correlation")
    # initial_title = (
    #     rf"$\rho_{{se}}(0)={float(row['rho_se0']):g}$, "
    #     rf"$\rho_{{me}}(0)={float(row['rho_me0']):g}$"
    # )
    # title = setting_title(setting)
    # axis.set_title(
    #     initial_title if title is None else f"{title}\n{initial_title}"
    # )
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    handles, labels = axis.get_legend_handles_labels()
    if offline_spans:
        handles.append(Patch(facecolor=OFFLINE_SHADE, edgecolor="none"))
        labels.append("offline")
    if switch_spans:
        handles.append(Patch(facecolor=SWITCH_SHADE, edgecolor="none"))
        labels.append("switch")
    axis.legend(handles, labels, frameon=False)
    figure.tight_layout()
    return figure


def number_slug(value: float) -> str:
    """Format a numeric setting for use in an output filename."""

    return f"{value:g}".replace("-", "m").replace(".", "p")


def setting_filename(setting: tuple[float | None, float | None]) -> str:
    """Return a unique filename stem for one horizon/switch-time setting."""

    horizon, switch_time = setting
    parts = []
    if horizon is not None:
        parts.append(f"initial_horizon_{number_slug(horizon)}")
    if switch_time is not None:
        parts.append(f"t_switch_{number_slug(switch_time)}")
    return "_".join(parts) if parts else "default"


def setting_title(setting: tuple[float | None, float | None]) -> str | None:
    """Return the descriptive title for one optional parameter setting."""

    horizon, switch_time = setting
    if setting == (None, None):
        return None
    labels = []
    if horizon is not None:
        labels.append(f"initial horizon={horizon:g}")
    if switch_time is not None:
        labels.append(rf"$t_{{switch}}$={switch_time:g}")
    return ", ".join(labels)


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--examples", required=True, help="semicolon-separated rho_se,rho_me pairs")
parser.add_argument("--input", type=Path, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--suffix", default="")
parser.add_argument("--output-subdir", default="")
parser.add_argument("--initial-horizon", type=float, default=None)
parser.add_argument("--t-switch", type=float, default=None)
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
if args.initial_horizon is not None:
    rows = [
        row
        for row in rows
        if setting_key(row)[0] is not None
        and np.isclose(setting_key(row)[0], args.initial_horizon)
    ]
if args.t_switch is not None:
    rows = [
        row
        for row in rows
        if setting_key(row)[1] is not None
        and np.isclose(setting_key(row)[1], args.t_switch)
    ]
if not rows:
    raise ValueError("no phase-example rows match the requested settings")
settings = sorted(
    {setting_key(row) for row in rows},
    key=lambda item: (
        float("-inf") if item[0] is None else item[0],
        float("-inf") if item[1] is None else item[1],
    ),
)
cache = {}
output.parent.mkdir(parents=True, exist_ok=True)
individual_directory = output.parent / f"{output.stem}_individuals"
individual_directory.mkdir(parents=True, exist_ok=True)
correlation_directory = output.parent / f"{output.stem}_correlations"
correlation_directory.mkdir(parents=True, exist_ok=True)
pdf = PdfPages(output) if len(settings) > 1 else None
try:
    for setting in settings:
        setting_rows = [row for row in rows if setting_key(row) == setting]
        figure, selected, rollouts = example_figure(setting_rows, targets, cache)
        title = setting_title(setting)
        if title is not None:
            figure.suptitle(title)
            figure.tight_layout(rect=(0, 0, 1, 0.96))
        else:
            figure.tight_layout()

        if pdf is None:
            figure.savefig(output, dpi=args.dpi, transparent=True)
        else:
            pdf.savefig(figure, dpi=args.dpi, transparent=True)
        individual_output = individual_directory / (
            f"{setting_filename(setting)}{output.suffix}"
        )
        figure.savefig(individual_output, dpi=args.dpi, transparent=True)
        plt.close(figure)
        print(f"Wrote {individual_output}")

        for example_index, (row, rollout) in enumerate(
            zip(selected, rollouts),
            start=1,
        ):
            correlation_output = correlation_directory / (
                f"{setting_filename(setting)}_example_{example_index}"
                f"_rho_se_{number_slug(float(row['rho_se0']))}"
                f"_rho_me_{number_slug(float(row['rho_me0']))}"
                f"{output.suffix}"
            )
            correlation = correlation_figure(row, rollout, setting)
            correlation.savefig(
                correlation_output,
                dpi=args.dpi,
                transparent=True,
            )
            plt.close(correlation)
            print(f"Wrote {correlation_output}")
finally:
    if pdf is not None:
        pdf.close()
print(f"Wrote {output}")
