"""CSV loading and plotting helpers for joint-dynamics results."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
OFFLINE_COLOR = "#e9ecef"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")


def add_plot_args(parser: argparse.ArgumentParser, stem: str) -> None:
    parser.set_defaults(stem=stem)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--suffix",
        default="",
        help="suffix appended to default result and figure filenames",
    )
    parser.add_argument(
        "--output-subdir",
        default="",
        help="relative subdirectory under results/ and figures/ for default paths",
    )
    parser.add_argument("--dpi", type=int, default=180)


def _safe_subdir(subdir: str) -> Path:
    subdir = subdir.strip().strip("/")
    if not subdir:
        return Path()
    path = Path(subdir)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("output subdirectory must be a relative path without '.' or '..'")
    return path


def results_dir(output_subdir: str = "") -> Path:
    return REPO_ROOT / "results" / _safe_subdir(output_subdir)


def figures_dir(output_subdir: str = "") -> Path:
    return REPO_ROOT / "figures" / _safe_subdir(output_subdir)


def paths(args):
    suffix = args.suffix.strip().strip("_")
    stem = f"{args.stem}_{suffix}" if suffix else args.stem
    output_subdir = getattr(args, "output_subdir", "")
    source = args.input or results_dir(output_subdir) / f"{stem}.csv"
    destination = args.output or figures_dir(output_subdir) / f"{stem}.pdf"
    return source, destination


def read_records(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    for record in records:
        if record.get("source") == "empirical":
            if record.get("mode") == "offline":
                # A NaN keeps the row's time coordinate while breaking the plotted
                # empirical reward curve throughout the no-execution phase.
                record["environment_reward_probability"] = "nan"
                record["reward_rate"] = "nan"
            elif not record.get("environment_reward_probability") and record.get(
                "recent_environment_reward"
            ):
                record["environment_reward_probability"] = record[
                    "recent_environment_reward"
                ]

        reward = record.get("environment_reward_probability", "")
        K = record.get("K", "") or record.get("config_K", "")
        if K != "" and not record.get("K"):
            record["K"] = K
        tau_on = record.get("config_tau_on", "")
        if (
            not record.get("reward_rate")
            and reward != ""
            and K != ""
            and tau_on != ""
        ):
            # One sequence reward is produced per online update, whose elapsed
            # cost is c_on = 1 + K tau_on in executed-sequence time units.
            record["reward_rate"] = str(
                float(reward) / (1.0 + float(K) * float(tau_on))
            )
    return records


def grouped(records, key="condition"):
    result = {}
    for row in records:
        result.setdefault(row.get(key, "trajectory"), []).append(row)
    for rows in result.values():
        rows.sort(key=lambda row: float(row["time"]))
    return result


def offline_spans(records: list[dict]) -> list[tuple[float, float]]:
    """Return elapsed-time intervals belonging to offline phases."""

    preferred = [row for row in records if row.get("source") == "analytical"]
    source = preferred or records
    if not source:
        return []

    # One condition is sufficient and avoids mixing replicated trajectories.
    condition = source[0].get("condition", "trajectory")
    source = [row for row in source if row.get("condition", "trajectory") == condition]
    phases: dict[int, list[dict]] = {}
    for row in source:
        if row.get("phase", "") == "" or row.get("mode", "") == "":
            continue
        phases.setdefault(int(float(row["phase"])), []).append(row)

    spans = []
    previous_end = None
    for phase in sorted(phases):
        rows = sorted(phases[phase], key=lambda row: float(row["time"]))
        start = float(rows[0]["time"]) if previous_end is None else previous_end
        end = max(float(row["time"]) for row in rows)
        previous_end = end
        if rows[-1]["mode"] == "offline" and end > start:
            spans.append((start, end))
    return spans


def shade_offline(axis, spans: list[tuple[float, float]]) -> None:
    for start, end in spans:
        axis.axvspan(start, end, color=OFFLINE_COLOR, zorder=0)


def plot_dashboard(
    records,
    destination: Path,
    title: str,
    dpi: int = 180,
    condition_styles: dict[str, dict] | None = None,
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    groups = grouped(records)
    spans = offline_spans(records)
    metrics = (
        ("rho_se", r"$\rho_{se}$"),
        ("rho_me", r"$\rho_{me}$"),
        ("rho_sm", r"$\rho_{sm}$"),
        ("environment_reward_probability", "environment reward"),
        ("beta", r"$\beta$"),
        ("K", r"$K$"),
        ("reward_rate", "reward rate"),
        ("eta_on2", r"$\eta_{on,2}$"),
        ("eta_wm", r"$\eta_{wm}$"),
    )
    fig, axes = plt.subplots(3, 3, figsize=(12, 9.0), sharex=True)
    for axis, (metric, label) in zip(axes.ravel(), metrics):
        shade_offline(axis, spans)
        for condition, rows in groups.items():
            usable = [row for row in rows if row.get(metric, "") != ""]
            if usable:
                style = (
                    {"linestyle": "--", "color": "black"}
                    if condition == "analytical"
                    else {}
                )
                if condition_styles is not None:
                    style.update(condition_styles.get(condition, {}))
                axis.plot(
                    [float(row["time"]) for row in usable],
                    [float(row[metric]) for row in usable],
                    label=condition,
                    **style,
                )
        axis.set_ylabel(label)
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("elapsed time")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if spans:
        handles.append(Patch(facecolor=OFFLINE_COLOR, edgecolor="none"))
        labels.append("offline")
    if handles:
        fig.legend(
            handles,
            labels,
            frameon=False,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.95),
            ncol=max(1, len(labels)),
        )
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=dpi, transparent=True)
    plt.close(fig)


def run_plot(args, title):
    source, destination = paths(args)
    plot_dashboard(read_records(source), destination, title, args.dpi)
    print(f"Wrote {destination}")
