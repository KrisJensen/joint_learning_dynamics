"""Plot the initial-correlation dynamic-programming phase diagram."""

import argparse

import numpy as np

from _common import add_plot_args, paths, read_records

import matplotlib.pyplot as plt
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


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "phase_diagram_initial")
parser.add_argument(
    "--value-category-threshold",
    type=float,
    default=1e-3,
    help="minimum relative value gain used by the value-based categories",
)
args = parser.parse_args()
if args.value_category_threshold < 0.0:
    parser.error("--value-category-threshold must be nonnegative")
source, output = paths(args)
rows = read_records(source)
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
    mode = (
        "offline"
        if switching_advantage > args.value_category_threshold
        else "online"
    )
    K_class = (
        "planning"
        if K_advantage > args.value_category_threshold
        else "K1"
    )
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
        (
            "value-difference category "
            f"(>{100.0 * args.value_category_threshold:g}% relative)"
        ),
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
figure.tight_layout()
output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(output, dpi=args.dpi, transparent=True)
plt.close(figure)
print(f"Wrote {output}")
