"""Visualize student and world-model learning trajectories on a unit sphere."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from _common import figures_dir, grouped, read_records, results_dir


ENVIRONMENT = np.array([0.0, 0.0, 1.0])
STUDENT_COLOR = "#2563eb"
WORLD_MODEL_COLOR = "#f97316"
SPHERE_OFFLINE_COLOR = "#9ca3af"
REFERENCE_COLOR = "#111827"
VIEW_ELEVATION = 24.0
VIEW_AZIMUTH = 38.0


def vectors_from_row(row: dict) -> tuple[np.ndarray, np.ndarray]:
    """Embed a correlation state as three unit vectors in R^3."""

    rho_se = float(row["rho_se"])
    rho_me = float(row["rho_me"])
    rho_sm = float(row["rho_sm"])

    world_model_y = max(1.0 - rho_me**2, 0.0) ** 0.5
    world_model = np.array([0.0, world_model_y, rho_me])
    student_z = float(np.clip(rho_se, -1.0, 1.0))
    if world_model_y <= 1e-12:
        student_y = 0.0
    else:
        student_y = (rho_sm - student_z * rho_me) / world_model_y
    maximum_y = max(1.0 - student_z**2, 0.0) ** 0.5
    student_y = float(np.clip(student_y, -maximum_y, maximum_y))
    student_x = max(1.0 - student_z**2 - student_y**2, 0.0) ** 0.5
    student = np.array([student_x, student_y, student_z])
    return student, world_model


def _thin(rows: list[dict], maximum: int) -> list[dict]:
    if maximum <= 0 or len(rows) <= maximum:
        return rows
    indices = np.linspace(0, len(rows) - 1, maximum).round().astype(int)
    return [rows[index] for index in np.unique(indices)]


def _trajectory(rows: list[dict], vector: str) -> np.ndarray:
    pairs = [vectors_from_row(row) for row in rows]
    index = 0 if vector == "student" else 1
    return np.asarray([pair[index] for pair in pairs])


def _plot_sphere_surface(axis) -> None:
    longitude = np.linspace(0.0, 2.0 * np.pi, 48)
    polar = np.linspace(0.0, np.pi, 24)
    x = np.outer(np.cos(longitude), np.sin(polar))
    y = np.outer(np.sin(longitude), np.sin(polar))
    z = np.outer(np.ones_like(longitude), np.cos(polar))
    axis.plot_surface(
        x, y, z, color="#d9d9d9", alpha=0.12, linewidth=0.0, shade=False, zorder=0
    )

    angle = np.linspace(0.0, 2.0 * np.pi, 180)
    for height in np.linspace(-0.75, 0.75, 7):
        radius = np.sqrt(1.0 - height**2)
        axis.plot(
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.full_like(angle, height),
            color="#d0d0d0",
            linewidth=0.4,
            alpha=0.45,
            zorder=1,
        )
    for longitude_value in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
        latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 180)
        axis.plot(
            np.cos(latitude) * np.cos(longitude_value),
            np.cos(latitude) * np.sin(longitude_value),
            np.sin(latitude),
            color="#d0d0d0",
            linewidth=0.4,
            alpha=0.35,
            zorder=1,
        )

    for vector in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), ENVIRONMENT):
        axis.plot(
            [0.0, vector[0]],
            [0.0, vector[1]],
            [0.0, vector[2]],
            color=REFERENCE_COLOR,
            alpha=0.45,
            linewidth=1.5,
            zorder=3,
        )
    axis.scatter(
        [0.0], [0.0], [1.0], color=REFERENCE_COLOR, s=34,
        depthshade=False, label="environment", zorder=4,
    )


def _plot_segments(axis, rows, points, color, linestyle) -> None:
    if len(points) == 0:
        return
    for start, end, end_row in zip(points[:-1], points[1:], rows[1:]):
        offline = end_row.get("mode") == "offline"
        axis.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            [start[2], end[2]],
            color=SPHERE_OFFLINE_COLOR if offline else color,
            linestyle=linestyle,
            linewidth=3.0 if offline else 2.0,
            zorder=12,
        )
    axis.scatter(
        points[0:1, 0], points[0:1, 1], points[0:1, 2],
        color=color, s=22, marker="o", depthshade=False, zorder=13,
    )


def plot_sphere(
    records: list[dict],
    output: Path,
    *,
    source: str = "all",
    label_key: str = "condition",
    max_points: int = 600,
    dpi: int = 180,
) -> None:
    import matplotlib.pyplot as plt

    if source != "all":
        records = [row for row in records if row.get("source") == source]
    if not records:
        raise ValueError("no records remain after applying the source filter")
    for key in ("rho_se", "rho_me", "rho_sm"):
        if not any(row.get(key, "") != "" for row in records):
            raise ValueError(f"missing required CSV column: {key}")

    figure = plt.figure(figsize=(7.2, 6.6))
    axis = figure.add_subplot(111, projection="3d")
    _plot_sphere_surface(axis)
    colors = plt.get_cmap("tab10")

    for index, (label, rows) in enumerate(grouped(records, label_key).items()):
        rows = _thin(rows, max_points)
        if label.lower() == "analytical" or len(grouped(records, label_key)) == 1:
            student_color, model_color = STUDENT_COLOR, WORLD_MODEL_COLOR
        else:
            student_color = model_color = colors(index % 10)
        student = _trajectory(rows, "student")
        world_model = _trajectory(rows, "world_model")
        _plot_segments(axis, rows, student, student_color, "-")
        _plot_segments(axis, rows, world_model, model_color, "--")
        axis.plot([], [], [], color=student_color, linewidth=2.0, label=f"{label} student")
        axis.plot(
            [], [], [], color=model_color, linestyle="--", linewidth=2.0,
            label=f"{label} world model",
        )

    if any(row.get("mode") == "offline" for row in records):
        axis.plot([], [], [], color=SPHERE_OFFLINE_COLOR, linewidth=3.0, label="offline")
    axis.set(xlim=(-1.02, 1.02), ylim=(-1.02, 1.02), zlim=(-1.02, 1.02))
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    axis.set_axis_off()
    axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.72, 0.82))
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=None,
    )
    parser.add_argument(
        "--output", type=Path, default=None,
    )
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output-subdir", default="")
    parser.add_argument("--source", choices=["all", "analytical", "empirical"], default="all")
    parser.add_argument("--label-key", default="condition")
    parser.add_argument("--max-points", type=int, default=600)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    suffix = args.suffix.strip().strip("_")
    input_stem = f"analytical_trajectory_{suffix}" if suffix else "analytical_trajectory"
    output_stem = (
        f"analytical_trajectory_sphere_{suffix}"
        if suffix
        else "analytical_trajectory_sphere"
    )
    input_path = args.input or results_dir(args.output_subdir) / f"{input_stem}.csv"
    output_path = args.output or figures_dir(args.output_subdir) / f"{output_stem}.pdf"
    plot_sphere(
        read_records(input_path), output_path, source=args.source,
        label_key=args.label_key, max_points=args.max_points, dpi=args.dpi,
    )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
