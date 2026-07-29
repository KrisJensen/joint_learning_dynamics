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


def plot_dashboard(records: list[dict], output: Path, title: str, dpi: int = 180) -> None:
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


def run_dashboard(args, title: str) -> None:
    source, output = paths(args)
    plot_dashboard(read_records(source), output, title, args.dpi)
    print(f"Wrote {output}")
