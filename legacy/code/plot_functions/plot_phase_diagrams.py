"""Plot dynamic-programming phase diagrams."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from _common import figures_dir, results_dir

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


CATEGORY_LABELS = {
    "online_K1": "online, K=1",
    "online_planning": "online, K>1",
    "offline_K1": "offline, K=1",
    "offline_planning": "offline, K>1",
}
CATEGORY_COLORS = {
    "online_K1": "#4c78a8",
    "online_planning": "#72b7b2",
    "offline_K1": "#f58518",
    "offline_planning": "#e45756",
}
VALUE_CATEGORY_THRESHOLD = 1e-3
RELATIVE_EPS = 1e-14


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


def gridded_values(records, x_key, y_key, values) -> tuple[np.ndarray, list[float], list[float]]:
    x_values = sorted({float(row[x_key]) for row in records})
    y_values = sorted({float(row[y_key]) for row in records})
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan, dtype=float)
    for row, value in zip(records, values):
        grid[y_index[float(row[y_key])], x_index[float(row[x_key])]] = value
    return grid, x_values, y_values


def imshow_extent(values: list[float]) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if high > low:
        return low, high
    return low - 0.5, high + 0.5


def plot_categories(
    axis,
    records,
    x_key,
    y_key,
    title,
    xlabel,
    ylabel,
    *,
    categories=None,
) -> None:
    category_index = {category: index for index, category in enumerate(CATEGORY_LABELS)}
    categories = [row["category"] for row in records] if categories is None else categories
    values = [category_index.get(category, np.nan) for category in categories]
    grid, x_values, y_values = gridded_values(records, x_key, y_key, values)
    x_low, x_high = imshow_extent(x_values)
    y_low, y_high = imshow_extent(y_values)
    cmap = ListedColormap([CATEGORY_COLORS[key] for key in CATEGORY_LABELS])
    cmap.set_bad("#ffffff")
    norm = BoundaryNorm(
        np.arange(-0.5, len(CATEGORY_LABELS) + 0.5, 1.0),
        cmap.N,
    )
    axis.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=(x_low, x_high, y_low, y_high),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=norm,
    )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def set_grid_limits(axis, records, x_key, y_key) -> None:
    x_values = [float(row[x_key]) for row in records]
    y_values = [float(row[y_key]) for row in records]
    x_low, x_high = min(x_values), max(x_values)
    y_low, y_high = min(y_values), max(y_values)
    if x_high == x_low:
        x_low -= 0.5
        x_high += 0.5
    if y_high == y_low:
        y_low -= 0.5
        y_high += 0.5
    axis.set_xlim(x_low, x_high)
    axis.set_ylim(y_low, y_high)


def plot_advantage(
    axis,
    records,
    x_key,
    y_key,
    title,
    xlabel,
    ylabel,
    fig,
    *,
    values,
    colorbar_label,
) -> None:
    grid, x_values, y_values = gridded_values(records, x_key, y_key, values)
    x_low, x_high = imshow_extent(x_values)
    y_low, y_high = imshow_extent(y_values)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("#ffffff")
    finite = grid[np.isfinite(grid)]
    image_kwargs = {}
    if finite.size:
        vmin = np.nanquantile(finite, 0.01)
        vmax = np.nanquantile(finite, 0.99)
        if vmax > vmin:
            image_kwargs["vmin"] = vmin
            image_kwargs["vmax"] = vmax
    image = axis.imshow(
        np.ma.masked_invalid(grid),
        origin="lower",
        extent=(x_low, x_high, y_low, y_high),
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        **image_kwargs,
    )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(colorbar_label)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)


def relative_advantage(intact: float, lesion: float) -> float:
    """Relative lesion advantage: (intact - lesion) / lesion."""

    if not np.isfinite(intact) or not np.isfinite(lesion) or abs(lesion) <= RELATIVE_EPS:
        return np.nan
    return (intact - lesion) / lesion


def relative_advantage_from_row(row: dict, lesion_key: str) -> float:
    return relative_advantage(
        float(row["optimal_reward_rate"]),
        float(row[lesion_key]),
    )


def offline_online_advantage(row: dict) -> float:
    """Relative full-model advantage over online-only any-K."""

    return relative_advantage_from_row(row, "online_only_reward_rate")


def offline_k1_advantage(row: dict) -> float:
    """Relative full-model advantage over switching with K=1."""

    return relative_advantage_from_row(row, "k1_switching_reward_rate")


def value_difference_category(row: dict) -> str:
    """Categorize the policy class by value differences instead of rollout actions."""

    offline_advantage = offline_online_advantage(row)
    planning_advantage = offline_k1_advantage(row)
    mode = "offline" if offline_advantage > VALUE_CATEGORY_THRESHOLD else "online"
    K_class = "planning" if planning_advantage > VALUE_CATEGORY_THRESHOLD else "K1"
    return f"{mode}_{K_class}"


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--suffix", default="")
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
parser.add_argument(
    "--analysis",
    choices=["auto", "initial", "learning-rates"],
    default="auto",
    help=(
        "which phase-diagram version to plot; auto chooses one version and never "
        "combines initial and learning-rate panels"
    ),
)
parser.add_argument("--input", type=Path, default=None, help="combined CSV")
parser.add_argument("--initial-input", type=Path, default=None)
parser.add_argument("--learning-input", type=Path, default=None)
parser.add_argument(
    "--learning-glob",
    default=None,
    help="glob for per-pair learning-rate CSVs; defaults to the standard stem",
)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument(
    "--output-subdir",
    default="",
    help="relative subdirectory under results/ and figures/ for default paths",
)
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()

suffix = parameter_suffix(args.switching_cost, args.T, args.fixed_wm, args.suffix)
result_root = results_dir(args.output_subdir)
figure_root = figures_dir(args.output_subdir)
analysis = args.analysis
records: list[dict] = []
if args.input is not None:
    records.extend(read_records(args.input))
else:
    initial_source = args.initial_input or result_root / (
        f"{suffixed_stem('phase_diagrams_initial', suffix)}.csv"
    )
    if analysis in {"auto", "initial"} and initial_source.exists():
        records.extend(read_records(initial_source))
        analysis = "initial"
    elif analysis == "initial":
        raise FileNotFoundError(f"missing initial phase-diagram input: {initial_source}")

    learning_stem = suffixed_stem("phase_diagrams_learning_rates", suffix)
    learning_dir = result_root / learning_stem
    learning_source = args.learning_input or learning_dir / f"{learning_stem}.csv"
    if not records and analysis in {"auto", "learning-rates"}:
        pattern = args.learning_glob or (
            f"{learning_stem}_eta_*_wm_*.csv"
        )
        search_dir = learning_dir if learning_dir.exists() else result_root
        for path in sorted(search_dir.glob(pattern)):
            records.extend(read_records(path))
        if not records and learning_source.exists():
            records.extend(read_records(learning_source))
        legacy_source = result_root / f"{learning_stem}.csv"
        if not records and legacy_source.exists():
            records.extend(read_records(legacy_source))
        if records:
            analysis = "learning-rates"

if not records:
    raise ValueError("no phase-diagram records found")

if analysis == "auto":
    available_analyses = {row.get("analysis") for row in records}
    if "initial_correlations" in available_analyses:
        analysis = "initial"
    elif "learning_rates" in available_analyses:
        analysis = "learning-rates"
    else:
        raise ValueError("no recognised phase-diagram records found")

if analysis == "initial":
    selected_records = [
        row for row in records if row.get("analysis") == "initial_correlations"
    ]
    if not selected_records:
        raise ValueError("no initial-correlation phase-diagram records found")
    panels = [
        (
            "category",
            selected_records,
            "rho_se0",
            "rho_me0",
            "rollout categories",
            r"$\rho_{se}(0)$",
            r"$\rho_{me}(0)$",
        ),
        (
            "category",
            selected_records,
            "rho_se0",
            "rho_me0",
            "value categories",
            r"$\rho_{se}(0)$",
            r"$\rho_{me}(0)$",
            [value_difference_category(row) for row in selected_records],
        ),
        (
            "advantage",
            selected_records,
            "rho_se0",
            "rho_me0",
            "optimal reward rate",
            r"$\rho_{se}(0)$",
            r"$\rho_{me}(0)$",
            [float(row["optimal_reward_rate"]) for row in selected_records],
            "average reward rate",
        ),
        (
            "advantage",
            selected_records,
            "rho_se0",
            "rho_me0",
            "offline-vs-online advantage (%)",
            r"$\rho_{se}(0)$",
            r"$\rho_{me}(0)$",
            [offline_online_advantage(row) for row in selected_records],
            "(full - online-only) / online-only",
        ),
        (
            "advantage",
            selected_records,
            "rho_se0",
            "rho_me0",
            "K > 1 advantage (%)",
            r"$\rho_{se}(0)$",
            r"$\rho_{me}(0)$",
            [offline_k1_advantage(row) for row in selected_records],
            "(full - K=1 switching) / K=1 switching",
        ),
    ]
    output_stem = "phase_diagrams_initial"
else:
    selected_records = [
        row for row in records if row.get("analysis") == "learning_rates"
    ]
    if not selected_records:
        raise ValueError("no learning-rate phase-diagram records found")
    panels = [
        (
            "category",
            selected_records,
            "eta_on2_off",
            "eta_wm",
            "rollout categories",
            r"$\eta_{on,2}=\eta_{off}$",
            r"$\eta_{wm}$",
        ),
        (
            "category",
            selected_records,
            "eta_on2_off",
            "eta_wm",
            "value categories (>1% relative)",
            r"$\eta_{on,2}=\eta_{off}$",
            r"$\eta_{wm}$",
            [value_difference_category(row) for row in selected_records],
        ),
        (
            "advantage",
            selected_records,
            "eta_on2_off",
            "eta_wm",
            "full-model reward rate",
            r"$\eta_{on,2}=\eta_{off}$",
            r"$\eta_{wm}$",
            [float(row["optimal_reward_rate"]) for row in selected_records],
            "average reward rate",
        ),
        (
            "advantage",
            selected_records,
            "eta_on2_off",
            "eta_wm",
            "offline-vs-online relative advantage",
            r"$\eta_{on,2}=\eta_{off}$",
            r"$\eta_{wm}$",
            [offline_online_advantage(row) for row in selected_records],
            "(full - online-only) / online-only",
        ),
        (
            "advantage",
            selected_records,
            "eta_on2_off",
            "eta_wm",
            "K > 1 relative advantage",
            r"$\eta_{on,2}=\eta_{off}$",
            r"$\eta_{wm}$",
            [offline_k1_advantage(row) for row in selected_records],
            "(full - K=1 switching) / K=1 switching",
        ),
    ]
    output_stem = "phase_diagrams_learning_rates"

fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.2))
axes = np.atleast_1d(axes)
for axis, panel in zip(
    axes, panels
):
    kind, panel_records, x_key, y_key, title, xlabel, ylabel, *extra = panel
    if kind == "category":
        categories = extra[0] if extra else None
        plot_categories(
            axis,
            panel_records,
            x_key,
            y_key,
            title,
            xlabel,
            ylabel,
            categories=categories,
        )
    else:
        values, colorbar_label = extra
        plot_advantage(
            axis,
            panel_records,
            x_key,
            y_key,
            title,
            xlabel,
            ylabel,
            fig,
            values=values,
            colorbar_label=colorbar_label,
        )
    set_grid_limits(axis, panel_records, x_key, y_key)
    axis.grid(False)
    axis.spines[["top", "right"]].set_visible(False)

handles = [
    Patch(facecolor=CATEGORY_COLORS[key], edgecolor="none", label=label)
    for key, label in CATEGORY_LABELS.items()
]
fig.legend(
    handles=handles,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(handles),
)
fig.tight_layout()
destination = args.output or figure_root / (
    f"{suffixed_stem(output_stem, suffix)}.pdf"
)
destination.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(destination, dpi=args.dpi, bbox_inches="tight", transparent=True)
plt.close(fig)
print(f"Wrote {destination}")
