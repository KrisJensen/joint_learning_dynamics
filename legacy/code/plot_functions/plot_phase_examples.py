"""Plot optimal rollout examples selected from a phase-diagram grid."""

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

from joint_dynamics import State, load_switching_value_map_cache, rollout_policy  # noqa: E402

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402


def suffixed_stem(base: str, suffix: str) -> str:
    suffix = suffix.strip().strip("_")
    return f"{base}_{suffix}" if suffix else base


def parameter_suffix(
    switching_cost: float | None,
    T: int | None,
    fixed_wm: bool,
    extra: str,
) -> str:
    parts = []
    if switching_cost is not None:
        switching_slug = (
            f"{switching_cost:.6g}"
            .replace("-", "m")
            .replace("+", "")
            .replace(".", "p")
        )
        parts.append(f"switch_{switching_slug}")
    if T is not None:
        parts.append(f"T_{T}")
    if fixed_wm:
        parts.append("fixed_wm")
    extra = extra.strip().strip("_")
    if extra:
        parts.append(extra)
    return "_".join(parts)


def read_records(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_examples(text: str) -> list[tuple[float, float]]:
    examples = []
    for item in text.split(";"):
        item = item.strip()
        if not item:
            continue
        first, second = item.split(",", 1)
        examples.append((float(first), float(second)))
    if not examples:
        raise ValueError("examples must contain at least one x,y pair")
    return examples


def nearest_record(records: list[dict], x_key: str, y_key: str, target) -> dict:
    target_x, target_y = target
    distances = [
        (float(row[x_key]) - target_x) ** 2 + (float(row[y_key]) - target_y) ** 2
        for row in records
    ]
    return records[int(np.argmin(distances))]


def finite_values(rows: list[dict], key: str) -> np.ndarray:
    return np.asarray(
        [
            float(row[key]) if row.get(key, "") not in ("", None) else np.nan
            for row in rows
        ],
        dtype=float,
    )


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--analysis",
    choices=["initial", "learning-rates"],
    default="initial",
)
parser.add_argument(
    "--examples",
    required=True,
    help=(
        "semicolon-separated x,y pairs. For initial: rho_se0,rho_me0. "
        "For learning-rates: eta_on2_off,eta_wm"
    ),
)
parser.add_argument("--input", type=Path, default=None)
parser.add_argument("--suffix", default="")
parser.add_argument(
    "--learning-glob",
    default=None,
    help="glob for per-pair learning-rate CSVs; defaults to the standard stem",
)
parser.add_argument(
    "--switching-cost",
    type=float,
    default=None,
    help="add switch_<cost> to the default result and figure filename suffix",
)
parser.add_argument(
    "--T",
    type=int,
    default=None,
    help="add T_<T> to the default result and figure filename suffix",
)
parser.add_argument(
    "--fixed-wm",
    action="store_true",
    help="add fixed_wm to the default result and figure filename suffix",
)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--output-subdir",
    default="",
    help="relative subdirectory under results/ and figures/ for default paths",
)
parser.add_argument("--rollout-step", type=float, default=None)
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()

suffix = parameter_suffix(args.switching_cost, args.T, args.fixed_wm, args.suffix)
result_root = results_dir(args.output_subdir)
figure_root = figures_dir(args.output_subdir)
stem = (
    "phase_diagrams_initial"
    if args.analysis == "initial"
    else "phase_diagrams_learning_rates"
)
destination = args.output or figure_root / (
    f"{suffixed_stem(stem + '_examples', suffix)}.pdf"
)
records: list[dict] = []
if args.input is not None:
    records.extend(read_records(args.input))
else:
    source = result_root / f"{suffixed_stem(stem, suffix)}.csv"
    if args.analysis == "learning-rates":
        learning_stem = suffixed_stem(stem, suffix)
        learning_dir = result_root / learning_stem
        source = learning_dir / f"{learning_stem}.csv"
        pattern = args.learning_glob or (
            f"{learning_stem}_eta_*_wm_*.csv"
        )
        search_dir = learning_dir if learning_dir.exists() else result_root
        for path in sorted(search_dir.glob(pattern)):
            records.extend(read_records(path))
        if not records and source.exists():
            records.extend(read_records(source))
        legacy_source = result_root / f"{learning_stem}.csv"
        if not records and legacy_source.exists():
            records.extend(read_records(legacy_source))
    elif source.exists():
        records.extend(read_records(source))
    else:
        raise FileNotFoundError(f"missing phase-diagram input: {source}")
analysis_name = "initial_correlations" if args.analysis == "initial" else "learning_rates"
records = [row for row in records if row.get("analysis") == analysis_name]
if not records:
    raise ValueError(f"no {analysis_name} records found")

if args.analysis == "initial":
    x_key, y_key = "rho_se0", "rho_me0"
    x_label, y_label = r"$\rho_{se}(0)$", r"$\rho_{me}(0)$"
else:
    x_key, y_key = "eta_on2_off", "eta_wm"
    x_label, y_label = r"$\eta_{on,2}=\eta_{off}$", r"$\eta_{wm}$"

examples = parse_examples(args.examples)
selected = [nearest_record(records, x_key, y_key, example) for example in examples]
cache: dict[str, object] = {}
rollouts = []
for row in selected:
    table_path = row.get("table_path", "")
    if not table_path:
        raise ValueError("phase-diagram records must include table_path")
    if table_path not in cache:
        cache[table_path] = load_switching_value_map_cache(table_path)[0]
    result = cache[table_path]
    state = State(
        float(row["rho_se0"]),
        float(row["rho_me0"]),
        float(row["rho_sm0"]),
    )
    rollout_step = args.rollout_step or result.config.time_step
    rollouts.append(rollout_policy(result, state, rollout_step=rollout_step))

fig, axes = plt.subplots(
    len(selected),
    3,
    figsize=(12.5, 2.8 * len(selected)),
    sharex=False,
    squeeze=False,
)
for row_index, (record, rollout) in enumerate(zip(selected, rollouts)):
    times = [float(row["time"]) for row in rollout]
    offline_spans = [
        (float(row["time"]), float(next_row["time"]))
        for row, next_row in zip(rollout, rollout[1:])
        if row["mode"] == "off"
        and row["action"] != "switch"
        and float(next_row["time"]) > float(row["time"])
    ]
    switching_spans = [
        (float(row["time"]), float(next_row["time"]))
        for row, next_row in zip(rollout, rollout[1:])
        if row["action"] == "switch"
        and float(next_row["time"]) > float(row["time"])
    ]
    for axis in axes[row_index]:
        for start, end in offline_spans:
            axis.axvspan(start, end, color="#e9ecef", zorder=0)
        for start, end in switching_spans:
            axis.axvspan(start, end, color="#dbeafe", zorder=0)

    axes[row_index, 0].plot(times, finite_values(rollout, "rho_se"), label=r"$\rho_{se}$")
    axes[row_index, 0].plot(times, finite_values(rollout, "rho_me"), label=r"$\rho_{me}$")
    axes[row_index, 0].plot(times, finite_values(rollout, "rho_sm"), label=r"$\rho_{sm}$")
    axes[row_index, 0].set_ylabel("correlation")

    axes[row_index, 1].plot(times, finite_values(rollout, "K"), color="#4c78a8")
    axes[row_index, 1].set_ylabel(r"$K$")

    reward = finite_values(rollout, "immediate_reward")
    costs = np.asarray(
        [
            1.0 + value * result.params.tau_on if np.isfinite(value) else np.nan
            for value in finite_values(rollout, "K")
        ],
        dtype=float,
    )
    reward_rate = reward / costs
    axes[row_index, 2].plot(times, reward_rate, color="#59a14f")
    axes[row_index, 2].set_ylabel("reward rate")

    matched = (
        f"{x_label}={float(record[x_key]):g}, "
        f"{y_label}={float(record[y_key]):g}"
    )
    requested = f"requested ({examples[row_index][0]:g}, {examples[row_index][1]:g})"
    axes[row_index, 0].set_title(f"example {row_index + 1}: {matched}\n{requested}")
    if row_index == 0:
        axes[row_index, 1].set_title(r"$K(t)$")
        axes[row_index, 2].set_title("reward rate")
    for axis in axes[row_index]:
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.set_xlabel("elapsed time")

handles, labels = axes[0, 0].get_legend_handles_labels()
handles.extend(
    [
        Patch(facecolor="#e9ecef", edgecolor="none", label="offline"),
        Patch(facecolor="#dbeafe", edgecolor="none", label="switching"),
    ]
)
labels.extend(["offline", "switching"])
fig.legend(
    handles,
    labels,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=min(len(labels), 5),
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=args.dpi, transparent=True)
plt.close(fig)
print(f"Wrote {destination}")
