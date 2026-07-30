"""Plot the initial-correlation dynamic-programming phase diagram."""

import argparse

import numpy as np

from _common import add_plot_args, paths, read_records

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap, TwoSlopeNorm


CATEGORY_ORDER = (
    "online_K1",
    "online_planning",
    "offline_K1",
    "offline_planning",
)
CATEGORY_COLORS = (
    "#4c78a8",
    "#72b7b2",
    "#f58518",
    "#e45756",
)
RELATIVE_EPS = 1e-14


def relative_advantage(full: float, restricted: float) -> float:
    """Return the relative value gained by removing a model restriction."""

    if abs(restricted) <= RELATIVE_EPS:
        return np.nan
    return (full - restricted) / restricted


def setting_key(row: dict) -> tuple[float | None, float | None]:
    """Return the optional horizon/switch-cost coordinates of one result row."""

    horizon = row.get("initial_horizon", "")
    switch_cost = row.get("delta_switch", "")
    return (
        None if horizon in {"", None} else float(horizon),
        None if switch_cost in {"", None} else float(switch_cost),
    )


def phase_figure(rows: list[dict], threshold: float):
    """Build one four-panel dashboard for a single phase-diagram setting."""

    x_values = sorted({float(row["rho_se0"]) for row in rows})
    y_values = sorted({float(row["rho_me0"]) for row in rows})
    category_index = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    category_grid = np.full((len(y_values), len(x_values)), np.nan)
    value_category_grid = np.full_like(category_grid, np.nan)
    switching_advantage_grid = np.full_like(category_grid, np.nan)
    K_difference_grid = np.full_like(category_grid, np.nan)
    x_index = {value: index for index, value in enumerate(x_values)}
    y_index = {value: index for index, value in enumerate(y_values)}
    for row in rows:
        x = x_index[float(row["rho_se0"])]
        y = y_index[float(row["rho_me0"])]
        category_grid[y, x] = category_index[row["category"]]
        full = float(row["optimal_reward_rate"])
        online = float(row["online_only_reward_rate"])
        K1 = float(row["k1_switching_reward_rate"])
        switching_advantage = relative_advantage(full, online)
        K_advantage = relative_advantage(full, K1)
        switching_advantage_grid[y, x] = switching_advantage
        K_difference_grid[y, x] = full - K1
        mode = "offline" if switching_advantage > threshold else "online"
        K_class = "planning" if K_advantage > threshold else "K1"
        value_category_grid[y, x] = category_index[f"{mode}_{K_class}"]

    extent = (min(x_values), max(x_values), min(y_values), max(y_values))
    category_cmap = ListedColormap(CATEGORY_COLORS)
    category_norm = BoundaryNorm(
        np.arange(-0.5, len(CATEGORY_ORDER) + 0.5),
        category_cmap.N,
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, grid, title in (
        (axes[0, 0], category_grid, "optimal rollout category"),
        (
            axes[0, 1],
            value_category_grid,
            f"value-difference category (>{100.0 * threshold:g}% relative)",
        ),
    ):
        image = axis.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap=category_cmap,
            norm=category_norm,
            extent=extent,
        )
        axis.set_title(title)
        colorbar = figure.colorbar(
            image,
            ax=axis,
            ticks=range(len(CATEGORY_ORDER)),
        )
        colorbar.ax.set_yticklabels(CATEGORY_ORDER)

    image = axes[1, 0].imshow(
        switching_advantage_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=extent,
    )
    axes[1, 0].set_title("switching advantage over online")
    colorbar = figure.colorbar(image, ax=axes[1, 0])
    colorbar.set_label(r"$(V_{\rm full}-V_{\rm online})/V_{\rm online}$")

    finite_K_differences = K_difference_grid[np.isfinite(K_difference_grid)]
    K_limit = (
        max(float(np.max(np.abs(finite_K_differences))), 1e-12)
        if finite_K_differences.size
        else 1.0
    )
    image = axes[1, 1].imshow(
        K_difference_grid,
        origin="lower",
        aspect="auto",
        cmap="coolwarm",
        norm=TwoSlopeNorm(vmin=-K_limit, vcenter=0.0, vmax=K_limit),
        extent=extent,
    )
    axes[1, 1].set_title(r"value gained by allowing $K>1$")
    colorbar = figure.colorbar(image, ax=axes[1, 1])
    colorbar.set_label(r"$V_{\rm full}-V_{K=1}$ reward rate")

    for axis in axes.ravel():
        axis.set_xlabel(r"$\rho_{se}(0)$")
        axis.set_ylabel(r"$\rho_{me}(0)$")
    return figure


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "phase_diagram_initial")
parser.add_argument(
    "--value-category-threshold",
    type=float,
    default=1e-3,
    help="minimum relative value gain used by the value-based categories",
)
parser.add_argument("--initial-horizon", type=float, default=None)
parser.add_argument("--delta-switch", type=float, default=None)
args = parser.parse_args()
if args.value_category_threshold < 0.0:
    parser.error("--value-category-threshold must be nonnegative")
source, output = paths(args)
rows = read_records(source)
if args.initial_horizon is not None:
    rows = [
        row
        for row in rows
        if setting_key(row)[0] is not None
        and np.isclose(setting_key(row)[0], args.initial_horizon)
    ]
if args.delta_switch is not None:
    rows = [
        row
        for row in rows
        if setting_key(row)[1] is not None
        and np.isclose(setting_key(row)[1], args.delta_switch)
    ]
if not rows:
    raise ValueError("no phase-diagram rows match the requested settings")
settings = sorted(
    {setting_key(row) for row in rows},
    key=lambda item: (
        float("-inf") if item[0] is None else item[0],
        float("-inf") if item[1] is None else item[1],
    ),
)
output.parent.mkdir(parents=True, exist_ok=True)
if len(settings) == 1:
    setting = settings[0]
    figure = phase_figure(rows, args.value_category_threshold)
    if setting != (None, None):
        figure.suptitle(
            f"initial horizon={setting[0]:g}, "
            rf"$\delta_{{switch}}$={setting[1]:g}"
        )
        figure.tight_layout(rect=(0, 0, 1, 0.96))
    else:
        figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, transparent=True)
    plt.close(figure)
else:
    with PdfPages(output) as pdf:
        for setting in settings:
            setting_rows = [row for row in rows if setting_key(row) == setting]
            figure = phase_figure(setting_rows, args.value_category_threshold)
            figure.suptitle(
                f"initial horizon={setting[0]:g}, "
                rf"$\delta_{{switch}}$={setting[1]:g}"
            )
            figure.tight_layout(rect=(0, 0, 1, 0.96))
            pdf.savefig(figure, dpi=args.dpi, transparent=True)
            plt.close(figure)
print(f"Wrote {output}")
