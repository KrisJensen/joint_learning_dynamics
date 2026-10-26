"""Plot the initial-correlation dynamic-programming phase diagram."""

import argparse

import numpy as np

from _common import add_plot_args, paths, read_records

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import BoundaryNorm, ListedColormap


CATEGORY_ORDER = (
    "online_K1",
    "online_planning",
    "offline_K1",
    "offline_planning",
)
CATEGORY_COLORS = (
    "#4c78a8",
    "#72b7b2",
    #"#f58518",
    "#e45756",
    "#e45756",
)
RELATIVE_EPS = 1e-14


def relative_advantage(full: float, restricted: float) -> float:
    """Return the relative value gained by removing a model restriction."""

    if abs(restricted) <= RELATIVE_EPS:
        return np.nan
    return (full - restricted) / restricted


def setting_key(row: dict) -> tuple[float | None, float | None]:
    """Return the optional horizon/switch-time coordinates of one result row."""

    horizon = row.get("initial_horizon", "")
    switch_time = row.get("t_switch", "")
    return (
        None if horizon in {"", None} else float(horizon),
        None if switch_time in {"", None} else float(switch_time),
    )


def parse_float_list(text: str) -> tuple[float, ...]:
    """Parse a non-empty comma-separated list of numeric plot settings."""

    items = [item.strip() for item in text.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("selection must contain at least one value")
    try:
        return tuple(dict.fromkeys(float(item) for item in items))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "selection must be a comma-separated list of numbers"
        ) from error


def select_setting_values(
    rows: list[dict],
    coordinate: int,
    requested: tuple[float, ...] | None,
    label: str,
) -> list[dict]:
    """Keep rows matching requested horizon or switch-time values."""

    if requested is None:
        return rows
    available = sorted(
        {
            setting_key(row)[coordinate]
            for row in rows
            if setting_key(row)[coordinate] is not None
        }
    )
    missing = [
        value
        for value in requested
        if not any(np.isclose(value, candidate) for candidate in available)
    ]
    if missing:
        missing_text = ", ".join(f"{value:g}" for value in missing)
        available_text = ", ".join(f"{value:g}" for value in available)
        raise ValueError(
            f"requested {label} not found: {missing_text}; "
            f"available values are: {available_text}"
        )
    return [
        row
        for row in rows
        if setting_key(row)[coordinate] is not None
        and any(
            np.isclose(setting_key(row)[coordinate], value)
            for value in requested
        )
    ]


def phase_grids(rows: list[dict], threshold: float):
    """Build the plotted grids for one phase-diagram parameter setting."""

    x_values = sorted({float(row["rho_se0"]) for row in rows})
    y_values = sorted({float(row["rho_me0"]) for row in rows})
    category_index = {name: index for index, name in enumerate(CATEGORY_ORDER)}
    category_grid = np.full((len(y_values), len(x_values)), np.nan)
    value_category_grid = np.full_like(category_grid, np.nan)
    switching_advantage_grid = np.full_like(category_grid, np.nan)
    K_advantage_grid = np.full_like(category_grid, np.nan)
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
        K_advantage_grid[y, x] = K_advantage
        mode = "offline" if switching_advantage > threshold else "online"
        K_class = "planning" if K_advantage > threshold else "K1"
        value_category_grid[y, x] = category_index[f"{mode}_{K_class}"]

    extent = (min(x_values), max(x_values), min(y_values), max(y_values))
    return (
        extent,
        category_grid,
        value_category_grid,
        switching_advantage_grid,
        K_advantage_grid,
    )


def category_scale():
    """Return the shared categorical colour map and normalization."""

    category_cmap = ListedColormap(CATEGORY_COLORS)
    category_norm = BoundaryNorm(
        np.arange(-0.5, len(CATEGORY_ORDER) + 0.5),
        category_cmap.N,
    )
    return category_cmap, category_norm


def phase_figure(rows: list[dict], threshold: float):
    """Build one four-panel dashboard for a single phase-diagram setting."""

    (
        extent,
        category_grid,
        value_category_grid,
        switching_advantage_grid,
        K_advantage_grid,
    ) = phase_grids(rows, threshold)
    category_cmap, category_norm = category_scale()
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
    #axes[1, 0].set_title("switching advantage over online")
    colorbar = figure.colorbar(image, ax=axes[1, 0])
    colorbar.set_label(r"$(V_{\rm full}-V_{\rm online})/V_{\rm online}$")

    image = axes[1, 1].imshow(
        K_advantage_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        extent=extent,
    )
    #axes[1, 1].set_title(r"value gained by allowing $K>1$")
    colorbar = figure.colorbar(image, ax=axes[1, 1])
    colorbar.set_label(r"$(V_{\rm full}-V_{K=1})/V_{K=1}$")

    for axis in axes.ravel():
        axis.set_xlabel(r"$\rho_{se}(0)$")
        axis.set_ylabel(r"$\rho_{me}(0)$")
    return figure


def value_category_figure(rows: list[dict], threshold: float):
    """Build the individual value-difference categorisation figure."""

    extent, _, value_category_grid, _, _ = phase_grids(rows, threshold)
    category_cmap, category_norm = category_scale()
    figure, axis = plt.subplots(figsize=(4.8, 4.8))
    image = axis.imshow(
        value_category_grid,
        origin="lower",
        aspect="auto",
        cmap=category_cmap,
        norm=category_norm,
        extent=extent,
    )
    # axis.set_title(
    #     f"value-difference category (>{100.0 * threshold:g}% relative)"
    # )
    axis.set_xlabel(r"$\rho_{se}(0)$")
    axis.set_ylabel(r"$\rho_{me}(0)$")
    # colorbar = figure.colorbar(
    #     image,
    #     ax=axis,
    #     ticks=range(len(CATEGORY_ORDER)),
    # )
    # colorbar.ax.set_yticklabels(CATEGORY_ORDER)
    return figure


def rollout_category_figure(rows: list[dict], threshold: float):
    """Build the individual realised-rollout categorisation figure."""

    extent, category_grid, _, _, _ = phase_grids(rows, threshold)
    category_cmap, category_norm = category_scale()
    figure, axis = plt.subplots(figsize=(4.8, 4.8))
    axis.imshow(
        category_grid,
        origin="lower",
        aspect="auto",
        cmap=category_cmap,
        norm=category_norm,
        extent=extent,
    )
    axis.set_xlabel(r"$\rho_{se}(0)$")
    axis.set_ylabel(r"$\rho_{me}(0)$")
    return figure


def planning_time_histogram(rows: list[dict]):
    """Plot episode-level additional planning time across initial states."""

    values = np.asarray(
        [float(row["additional_planning_time"]) for row in rows],
        dtype=np.float64,
    )
    thresholds = np.asarray(
        [float(row["planning_time_threshold"]) for row in rows],
        dtype=np.float64,
    )
    if (
        not np.all(np.isfinite(values))
        or np.any(values < -1e-12)
        or not np.all(np.isfinite(thresholds))
        or np.any(thresholds <= 0.0)
    ):
        raise ValueError("planning-time values and thresholds must be valid")
    threshold = float(thresholds[0])
    if not np.allclose(thresholds, threshold, atol=1e-12, rtol=1e-12):
        raise ValueError("planning-time threshold varies within one setting")

    upper = max(float(np.max(values)), threshold)
    full_edges = np.linspace(0.0, 1.05 * upper, 31)
    zoom_edges = np.linspace(0.0, 0.52, 27)-0.01
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))
    clipped_values = np.maximum(values, 0.0)
    for axis, bin_edges, title in (
        (axes[0], full_edges, "full range"),
        (axes[1], zoom_edges, "zoom"),
    ):
        axis.hist(
            clipped_values,
            bins=bin_edges,
            color="#4c78a8",
            edgecolor="white",
            linewidth=0.5,
        )
        if bin_edges[0] <= threshold <= bin_edges[-1]:
            axis.axvline(
                threshold,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=r"$K=1, M=1$ block",
            )
            axis.legend(frameon=False)
        axis.set_title(title)
        axis.set_xlabel("episode additional planning time")
        axis.set_ylabel("rollouts")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].set_xlim(zoom_edges[0], zoom_edges[-1])
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
add_plot_args(parser, "phase_diagram_initial")
parser.add_argument(
    "--value-category-threshold",
    type=float,
    default=1e-3,
    help="minimum relative value gain used by the value-based categories",
)
horizon_selection = parser.add_mutually_exclusive_group()
horizon_selection.add_argument("--initial-horizon", type=float, default=None)
horizon_selection.add_argument(
    "--initial-horizons",
    type=parse_float_list,
    default=None,
    help="comma-separated initial horizons to include",
)
switch_selection = parser.add_mutually_exclusive_group()
switch_selection.add_argument("--t-switch", type=float, default=None)
switch_selection.add_argument(
    "--switch-times",
    type=parse_float_list,
    default=None,
    help="comma-separated t_switch values to include",
)
args = parser.parse_args()
if args.value_category_threshold < 0.0:
    parser.error("--value-category-threshold must be nonnegative")
source, output = paths(args)
rows = read_records(source)
requested_horizons = args.initial_horizons
if args.initial_horizon is not None:
    requested_horizons = (args.initial_horizon,)
requested_switch_times = args.switch_times
if args.t_switch is not None:
    requested_switch_times = (args.t_switch,)
rows = select_setting_values(rows, 0, requested_horizons, "initial horizons")
rows = select_setting_values(rows, 1, requested_switch_times, "switch times")
if not rows:
    raise ValueError("no phase-diagram rows match the requested settings")
planning_columns = {"additional_planning_time", "planning_time_threshold"}
if any(not planning_columns.issubset(row) for row in rows):
    raise ValueError(
        "phase results do not contain rollout planning times; rerun "
        "phase_diagram_initial.py with the current implementation"
    )
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
    title = setting_title(setting)
    # if title is not None:
    #     #figure.suptitle(title)
    #     figure.tight_layout(rect=(0, 0, 1, 0.96))
    # else:
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, transparent=True)
    plt.close(figure)
else:
    with PdfPages(output) as pdf:
        for setting in settings:
            setting_rows = [row for row in rows if setting_key(row) == setting]
            figure = phase_figure(setting_rows, args.value_category_threshold)
            #figure.suptitle(setting_title(setting))
            figure.tight_layout()#rect=(0, 0, 1, 0.96))
            pdf.savefig(figure, dpi=args.dpi, transparent=True)
            plt.close(figure)
print(f"Wrote {output}")

histogram_output = output.with_name(
    f"{output.stem}_planning_time_histogram{output.suffix}"
)
if len(settings) == 1:
    figure = planning_time_histogram(rows)
    figure.tight_layout()
    figure.savefig(histogram_output, dpi=args.dpi, transparent=True)
    plt.close(figure)
else:
    with PdfPages(histogram_output) as pdf:
        for setting in settings:
            setting_rows = [row for row in rows if setting_key(row) == setting]
            figure = planning_time_histogram(setting_rows)
            title = setting_title(setting)
            if title is not None:
                figure.suptitle(title)
                figure.tight_layout(rect=(0, 0, 1, 0.94))
            else:
                figure.tight_layout()
            pdf.savefig(figure, dpi=args.dpi, transparent=True)
            plt.close(figure)
print(f"Wrote {histogram_output}")

individual_directory = output.parent / (
    f"{output.stem}_value_difference_categories"
)
individual_directory.mkdir(parents=True, exist_ok=True)
rollout_directory = output.parent / f"{output.stem}_rollout_categories"
rollout_directory.mkdir(parents=True, exist_ok=True)
for setting in settings:
    setting_rows = [row for row in rows if setting_key(row) == setting]
    filename = f"{setting_filename(setting)}{output.suffix}"
    for directory, figure_function in (
        (individual_directory, value_category_figure),
        (rollout_directory, rollout_category_figure),
    ):
        figure = figure_function(
            setting_rows,
            args.value_category_threshold,
        )
        figure.tight_layout()
        individual_output = directory / filename
        figure.savefig(individual_output, dpi=args.dpi, transparent=True)
        plt.close(figure)
        print(f"Wrote {individual_output}")
