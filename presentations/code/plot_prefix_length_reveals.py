"""Generate progressive reveals of the report's T6 prefix-length figures."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_T6 = (
    REPOSITORY_ROOT / "results/report/prefix_length_comparison_T6.csv"
)
DEFAULT_INPUT_T9 = (
    REPOSITORY_ROOT / "results/report/prefix_length_comparison_T9.csv"
)
DEFAULT_OUTPUT_DIRECTORY = REPOSITORY_ROOT / "presentations/figures"

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


METRICS = (
    (
        "rho_se",
        r"$\rho_{se}$",
        "prefix_length_comparison_2factor_T6_student_environment_correlation",
    ),
    (
        "M",
        r"selected $M$",
        "prefix_length_comparison_2factor_T6_prefix_length",
    ),
)

FIXED_REVEALS = (
    ("1_M6", frozenset({("eta_2f_fixed_M", 6)})),
    (
        "2_M6_M1",
        frozenset({("eta_2f_fixed_M", 6), ("eta_2f_fixed_M", 1)}),
    ),
    (
        "3_all_constant_M",
        frozenset(("eta_2f_fixed_M", M) for M in range(1, 7)),
    ),
    (
        "4_with_adaptive_M",
        frozenset(("eta_2f_fixed_M", M) for M in range(1, 7))
        | {("eta_2f_optimal_M", None)},
    ),
)

OPTIMAL_REVEALS = (
    ("1_2factor", frozenset({("eta_2f_optimal_M", None)})),
    ("2_3factor", frozenset({("eta_3f_only", None)})),
    (
        "3_2factor_3factor",
        frozenset({("eta_2f_optimal_M", None), ("eta_3f_only", None)}),
    ),
    (
        "4_with_joint_optimal",
        frozenset(
            {
                ("eta_3f_only", None),
                ("eta_2f_optimal_M", None),
                ("joint_optimal_M", None),
            }
        ),
    ),
)


def read_records(path: Path) -> list[dict[str, str]]:
    """Read the prefix-length comparison table."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def group_records(records: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """Group trajectories and return them in report plotting order."""

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in records:
        grouped.setdefault(row["condition"], []).append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: float(row["time"]))

    order = {
        "eta_2f_fixed_M": 0,
        "eta_2f_optimal_M": 1,
        "eta_3f_only": 2,
        "joint_optimal_M": 3,
    }
    return sorted(
        grouped.values(),
        key=lambda rows: (
            order.get(rows[0].get("comparison_group", ""), 4),
            (
                float(rows[0]["compared_M"])
                if rows[0].get("comparison_group") == "eta_2f_fixed_M"
                else 0.0
            ),
            rows[0]["condition"],
        ),
    )


def group_key(rows: list[dict[str, str]]) -> tuple[str, int | None]:
    """Return the selection key for one trajectory."""

    group = rows[0].get("comparison_group", "")
    M = int(float(rows[0]["compared_M"])) if group == "eta_2f_fixed_M" else None
    return group, M


def label_for(rows: list[dict[str, str]]) -> str:
    """Return the label used by the report plot."""

    group, M = group_key(rows)
    if group == "eta_2f_fixed_M":
        return rf"$2f$ ($M={M}$)"
    if group == "eta_2f_optimal_M":
        return r"$2f$ (optimal)"
    if group == "eta_3f_only":
        return r"$3f$"
    if group == "joint_optimal_M":
        return r"$2f+3f$ (optimal)"
    return rows[0]["condition"]


def line_style(
    rows: list[dict[str, str]], fixed_colors: dict[int, np.ndarray]
) -> dict:
    """Return the line style used by the report plot."""

    group, M = group_key(rows)
    if group == "eta_2f_fixed_M":
        return {"color": fixed_colors[M], "linewidth": 1.7, "alpha": 0.9}
    if group == "eta_2f_optimal_M":
        return {"color": "#d55e00", "linewidth": 2.5}
    if group == "eta_3f_only":
        return {"color": "#0072b2", "linestyle": "-", "linewidth": 2.5}
    if group == "joint_optimal_M":
        return {"color": "#000000", "linestyle": "--", "linewidth": 2.5}
    return {}


def offline_spans(records: list[dict[str, str]]) -> list[tuple[float, float]]:
    """Return offline intervals from the first trajectory."""

    condition = records[0]["condition"]
    rows = [row for row in records if row["condition"] == condition]
    rows.sort(key=lambda row: float(row["time"]))
    return [
        (float(first["time"]), float(second["time"]))
        for first, second in zip(rows, rows[1:])
        if first.get("mode") == "offline"
    ]


def draw_plot(
    axis,
    groups: list[list[dict[str, str]]],
    selected: frozenset[tuple[str, int | None]],
    complete: frozenset[tuple[str, int | None]],
    metric: str,
    ylabel: str,
    fixed_colors: dict[int, np.ndarray],
    spans: list[tuple[float, float]],
) -> None:
    """Draw selected trajectories using the report figure's formatting."""

    for start, end in spans:
        axis.axvspan(start, end, color="#e9ecef", zorder=0)
    for rows in groups:
        if group_key(rows) not in selected:
            continue
        usable = [
            row
            for row in rows
            if row.get(metric, "") not in {"", None}
            and np.isfinite(float(row[metric]))
        ]
        axis.plot(
            [float(row["time"]) for row in usable],
            [float(row[metric]) for row in usable],
            label=label_for(rows),
            **line_style(rows, fixed_colors),
        )
    axis.set_xlabel("elapsed time")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    if metric == "rho_se":
        legend_rows = [rows for rows in groups if group_key(rows) in complete]
        handles = []
        labels = []
        visible = []
        for rows in legend_rows:
            handle = Line2D([], [], **line_style(rows, fixed_colors))
            is_visible = group_key(rows) in selected
            if not is_visible:
                handle.set_alpha(0.0)
            handles.append(handle)
            labels.append(label_for(rows))
            visible.append(is_visible)
        if spans:
            handles.append(Patch(facecolor="#e9ecef", edgecolor="none"))
            labels.append("offline")
            visible.append(True)
        legend = axis.legend(
            handles,
            labels,
            frameon=False,
            loc="lower right",
        )
        for text, is_visible in zip(legend.get_texts(), visible, strict=True):
            if not is_visible:
                text.set_alpha(0.0)
        axis.set_ylim(0, 1)


def write_reveals(
    groups: list[list[dict[str, str]]],
    reveals,
    metric: str,
    ylabel: str,
    output_stem: str,
    fixed_colors: dict[int, np.ndarray],
    spans: list[tuple[float, float]],
    output_directory: Path,
    dpi: int,
) -> None:
    """Write a reveal sequence with fixed limits, ticks, and plot bounds."""

    complete = reveals[-1][1]
    reference, reference_axis = plt.subplots(figsize=(5.4, 3.8))
    draw_plot(
        reference_axis,
        groups,
        complete,
        complete,
        metric,
        ylabel,
        fixed_colors,
        spans,
    )
    reference.tight_layout()
    settings = (
        reference_axis.get_xlim(),
        reference_axis.get_ylim(),
        reference_axis.get_xticks(),
        reference_axis.get_yticks(),
        (
            reference.subplotpars.left,
            reference.subplotpars.right,
            reference.subplotpars.bottom,
            reference.subplotpars.top,
        ),
    )
    plt.close(reference)

    xlim, ylim, xticks, yticks, bounds = settings
    for suffix, selected in reveals:
        figure, axis = plt.subplots(figsize=(5.4, 3.8))
        draw_plot(
            axis,
            groups,
            selected,
            complete,
            metric,
            ylabel,
            fixed_colors,
            spans,
        )
        axis.set_xticks(xticks)
        axis.set_yticks(yticks)
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        figure.subplots_adjust(
            left=bounds[0], right=bounds[1], bottom=bounds[2], top=bounds[3]
        )
        output = output_directory / f"{output_stem}_{suffix}.pdf"
        figure.savefig(output, dpi=dpi, transparent=True)
        plt.close(figure)
        print(f"Wrote {output}")


def main() -> None:
    """Generate the T6 prefix-length and T6/T9 learner reveal sequences."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-T6", type=Path, default=DEFAULT_INPUT_T6)
    parser.add_argument("--input-T9", type=Path, default=DEFAULT_INPUT_T9)
    parser.add_argument(
        "--output-directory", type=Path, default=DEFAULT_OUTPUT_DIRECTORY
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    records_T6 = read_records(args.input_T6)
    if not records_T6:
        raise ValueError(f"no records found in {args.input_T6}")
    groups_T6 = group_records(records_T6)
    fixed_M = sorted(
        M
        for rows in groups_T6
        if (group := group_key(rows))[0] == "eta_2f_fixed_M"
        for M in [group[1]]
        if M is not None
    )
    fixed_colors = dict(
        zip(
            fixed_M,
            plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(fixed_M))),
        )
    )
    spans_T6 = offline_spans(records_T6)
    args.output_directory.mkdir(parents=True, exist_ok=True)

    for metric, ylabel, output_stem in METRICS:
        write_reveals(
            groups_T6,
            FIXED_REVEALS,
            metric,
            ylabel,
            output_stem,
            fixed_colors,
            spans_T6,
            args.output_directory,
            args.dpi,
        )
    write_reveals(
        groups_T6,
        OPTIMAL_REVEALS,
        "rho_se",
        r"$\rho_{se}$",
        "prefix_length_comparison_optimal_T6_student_environment_correlation",
        fixed_colors,
        spans_T6,
        args.output_directory,
        args.dpi,
    )

    records_T9 = read_records(args.input_T9)
    if not records_T9:
        raise ValueError(f"no records found in {args.input_T9}")
    write_reveals(
        group_records(records_T9),
        OPTIMAL_REVEALS,
        "rho_se",
        r"$\rho_{se}$",
        "prefix_length_comparison_optimal_T9_student_environment_correlation",
        {},
        offline_spans(records_T9),
        args.output_directory,
        args.dpi,
    )


if __name__ == "__main__":
    main()
