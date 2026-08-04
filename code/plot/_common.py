"""Shared result loading and plotting utilities."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
RESULTS_ROOT = REPO_ROOT / "results"
FIGURES_ROOT = REPO_ROOT / "figures"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

# Shared palette for analytical correlation trajectories.  Keeping this in one
# place ensures analytical-only and empirical-comparison figures use exactly
# the same visual encoding.
CORRELATION_STYLES = (
    ("rho_se", r"$\rho_{se}$", "#0072b2"),
    ("rho_me", r"$\rho_{me}$", "#d55e00"),
    ("rho_sm", r"$\rho_{sm}$", "#009e73"),
)


def _safe_subdir(text: str) -> Path:
    text = text.strip().strip("/")
    if not text:
        return Path()
    path = Path(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("subdirectory must be a safe relative path")
    return path


def add_plot_args(parser: argparse.ArgumentParser, stem: str) -> None:
    parser.set_defaults(stem=stem)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output-subdir", default="")
    parser.add_argument("--dpi", type=int, default=180)


def paths(args) -> tuple[Path, Path]:
    suffix = args.suffix.strip().strip("_")
    stem = f"{args.stem}_{suffix}" if suffix else args.stem
    subdir = _safe_subdir(args.output_subdir)
    source = args.input or RESULTS_ROOT / subdir / f"{stem}.csv"
    output = args.output or FIGURES_ROOT / subdir / f"{stem}.pdf"
    output = output.resolve()
    if FIGURES_ROOT.resolve() not in output.parents:
        raise ValueError("plot outputs must remain under figures/")
    return source, output


def read_records(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def grouped(records: list[dict], key: str = "condition") -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in records:
        groups.setdefault(row.get(key, "trajectory"), []).append(row)
    for rows in groups.values():
        rows.sort(key=lambda row: float(row.get("time", 0.0)))
    return groups


def offline_spans(records: list[dict]) -> list[tuple[float, float]]:
    source = [row for row in records if row.get("source") == "analytical"] or records
    if not source:
        return []
    condition = source[0].get("condition", "trajectory")
    rows = [
        row for row in source if row.get("condition", "trajectory") == condition
    ]
    spans = []
    for first, second in zip(rows, rows[1:]):
        if first.get("mode") == "offline":
            spans.append((float(first["time"]), float(second["time"])))
    return spans


def swept_value_styles(
    records: list[dict], value_key: str
) -> dict[str, dict[str, object]]:
    """Map numeric sweep values to viridis and an ``optimal`` run to black.

    Styles are keyed by condition rather than row so that the same mapping can be
    reused by dashboards and stand-alone figures.
    """
    import matplotlib as mpl

    values_by_condition: dict[str, float | None] = {}
    numeric_values: set[float] = set()
    for condition, rows in grouped(records).items():
        raw_value = next(
            (row.get(value_key) for row in rows if row.get(value_key, "") != ""),
            "",
        )
        if str(raw_value).strip().lower() == "optimal":
            values_by_condition[condition] = None
            continue
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            # Non-numeric comparison conditions are conventionally reference or
            # optimized runs and should remain visually distinct from the sweep.
            values_by_condition[condition] = None
        else:
            values_by_condition[condition] = value
            numeric_values.add(value)

    ordered_values = sorted(numeric_values)
    viridis = mpl.colormaps["viridis"]
    if len(ordered_values) == 1:
        colors = {ordered_values[0]: viridis(0.5)}
    elif ordered_values:
        colors = {
            value: viridis(0.1 + 0.8*(index / (len(ordered_values) - 1)))
            for index, value in enumerate(ordered_values)
        }
    else:
        colors = {}
    return {
        condition: (
            {"color": "black", "linewidth": 2.4, "zorder": 3}
            if value is None
            else {"color": colors[value], "linewidth": 1.8, "zorder": 2}
        )
        for condition, value in values_by_condition.items()
    }


def plot_dashboard(
    records: list[dict],
    output: Path,
    title: str,
    dpi: int = 180,
    condition_styles: dict[str, dict[str, object]] | None = None,
) -> None:
    import matplotlib.pyplot as plt

    metrics = (
        ("rho_se", r"$\rho_{se}$"),
        ("rho_me", r"$\rho_{me}$"),
        ("rho_sm", r"$\rho_{sm}$"),
        ("true_reward_probability", "true reward probability"),
        ("reward_rate", "reward rate"),
        ("beta", r"$\beta$"),
        ("K", r"$K$"),
        ("M", r"$M$"),
        ("eta_3f", r"$\eta_{3f}$"),
    )
    groups = grouped(records)
    condition_styles = condition_styles or {}
    spans = offline_spans(records)
    figure, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        for start, end in spans:
            axis.axvspan(start, end, color="#e9ecef", zorder=0)
        for condition, rows in groups.items():
            usable = [
                row
                for row in rows
                if row.get(metric, "") not in {"", None}
                and row.get("time", "") not in {"", None}
            ]
            if usable:
                axis.plot(
                    [float(row["time"]) for row in usable],
                    [float(row[metric]) for row in usable],
                    label=condition,
                    **condition_styles.get(condition, {}),
                )
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("elapsed time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(handles, labels, frameon=False, loc="upper center", ncol=4)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, transparent=True)
    plt.close(figure)


def plot_metric_panels(
    records: list[dict],
    output: Path,
    metrics: tuple[tuple[str, str], ...],
    condition_styles: dict[str, dict[str, object]],
    *,
    dpi: int = 180,
    title: str | None = None,
    ylims: dict[str, tuple[float, float]] | None = None,
    width = 4.5,
) -> None:
    """Plot selected trajectory quantities in a separate comparison figure."""
    import matplotlib.pyplot as plt

    groups = grouped(records)
    spans = offline_spans(records)
    height = 3.8 if len(metrics) == 1 else 2.8 * len(metrics)
    figure, axes = plt.subplots(
        len(metrics), 1, figsize=(width, height), sharex=True, squeeze=False
    )
    flat_axes = axes[:, 0]
    for axis, (metric, ylabel) in zip(flat_axes, metrics, strict=True):
        for start, end in spans:
            axis.axvspan(start, end, color="#e9ecef", zorder=0)
        for condition, rows in groups.items():
            usable = [
                row
                for row in rows
                if row.get(metric, "") not in {"", None}
                and row.get("time", "") not in {"", None}
            ]
            if usable:
                axis.plot(
                    [float(row["time"]) for row in usable],
                    [float(row[metric]) for row in usable],
                    label=condition,
                    **condition_styles.get(condition, {}),
                )
        axis.set_ylabel(ylabel)
        if ylims and metric in ylims:
            axis.set_ylim(*ylims[metric])
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    flat_axes[-1].set_xlabel("elapsed time")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            ncol=min(3, len(handles)),
        )
    if title:
        figure.suptitle(title)
    top = 0.84 if title else 0.88
    figure.tight_layout(rect=(0, 0, 1, top))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, transparent=True)
    plt.close(figure)


def run_dashboard(args, title: str) -> None:
    source, output = paths(args)
    plot_dashboard(read_records(source), output, title, args.dpi)
    print(f"Wrote {output}")
