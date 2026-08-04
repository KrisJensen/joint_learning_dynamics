"""Plot empirical and analytical trajectories."""

import argparse

import numpy as np

from _common import (
    CORRELATION_STYLES,
    add_plot_args,
    offline_spans,
    paths,
    plot_dashboard,
    read_records,
)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "empirical_vs_analytical")
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
if not records:
    raise ValueError(f"no records found in {source}")

plot_dashboard(records, output, "Empirical versus analytical dynamics", args.dpi)
print(f"Wrote {output}")

# Keep replicates as independent paths: concatenating equal-time replicates
# would otherwise add spurious line segments between their endpoints.
analytical_rows = sorted(
    (row for row in records if row.get("source") == "analytical"),
    key=lambda row: float(row["time"]),
)
empirical_groups: dict[tuple[int, str], list[dict]] = {}
for row in records:
    if row.get("source") != "empirical":
        continue
    dimension = int(float(row["d"]))
    key = (dimension, row.get("replicate", "0"))
    empirical_groups.setdefault(key, []).append(row)
for rows in empirical_groups.values():
    rows.sort(key=lambda row: float(row["time"]))

dimensions = sorted({dimension for dimension, _ in empirical_groups})
viridis = plt.get_cmap("viridis")
if len(dimensions) == 1:
    dimension_colors = {dimensions[0]: viridis(0.5)}
else:
    dimension_colors = {
        dimension: viridis(position)
        for dimension, position in zip(
            dimensions,
            np.linspace(0.2, 0.8, len(dimensions)),
            strict=True,
        )
    }

correlation_output = output.with_name(
    f"{output.stem}_correlations{output.suffix}"
)
figure, axis = plt.subplots(figsize=(4.5, 3.8))
spans = offline_spans(records)
for start, end in spans:
    axis.axvspan(start, end, color="#e9ecef", zorder=0)

for (dimension, _), rows in empirical_groups.items():
    for metric, _, _ in CORRELATION_STYLES:
        usable = [
            row
            for row in rows
            if row.get(metric, "") not in {"", None}
            and np.isfinite(float(row[metric]))
        ]
        if usable:
            axis.plot(
                [float(row["time"]) for row in usable],
                [float(row[metric]) for row in usable],
                color=dimension_colors[dimension],
                linewidth=1.5,
                alpha=0.75,
                zorder=2,
                linestyle = "-",
            )

analytical_handles = []
for metric, label, color in CORRELATION_STYLES:
    usable = [
        row
        for row in analytical_rows
        if row.get(metric, "") not in {"", None}
        and np.isfinite(float(row[metric]))
    ]
    if not usable:
        raise ValueError(f"{source} has no finite analytical values for {metric}")
    analytical_handles.append(
        axis.plot(
            [float(row["time"]) for row in usable],
            [float(row[metric]) for row in usable],
            color=color,
            linewidth=2.,
            label=f"analytical {label}",
            linestyle = "--",
            zorder=3,
        )[0]
    )

dimension_handles = [
    Line2D(
        [],
        [],
        color=dimension_colors[dimension],
        linewidth=1.6,
        label=rf"empirical $d={dimension}$",
    )
    for dimension in dimensions
]
handles = [*analytical_handles, *dimension_handles]
if spans:
    handles.append(
        Patch(facecolor="#e9ecef", edgecolor="none", label="offline")
    )
axis.set_xlabel("elapsed time")
axis.set_ylabel("correlation")
axis.set_ylim(0.0, 1.0)
axis.set_axisbelow(True)
axis.grid(axis="y", color="#dddddd", linewidth=0.7)
axis.spines[["top", "right"]].set_visible(False)
axis.legend(handles=handles, frameon=False, fontsize=8, ncol=2)
figure.tight_layout()
correlation_output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(correlation_output, dpi=args.dpi, transparent=True)
plt.close(figure)
print(f"Wrote {correlation_output}")
