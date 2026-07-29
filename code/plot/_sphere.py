"""Shared unit-sphere geometry for static and animated trajectory plots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np


ENVIRONMENT = np.array([0.0, 0.0, 1.0], dtype=np.float64)
STUDENT_COLOR = "#2563eb"
WORLD_MODEL_COLOR = "#f97316"
REFERENCE_COLOR = "#111827"
OFFLINE_ALPHA = 0.4
VIEW_ELEVATION = 24.0
VIEW_AZIMUTH = 38.0


def vectors(row: Mapping[str, object]) -> tuple[np.ndarray, np.ndarray]:
    """Embed one correlation state as student and world-model unit vectors.

    The environment is fixed at the north pole. The world model is placed in
    the yz-plane, and the positive x solution is used for the student. Their
    pairwise dot products therefore reproduce ``rho_se``, ``rho_me``, and
    ``rho_sm`` up to floating-point roundoff.
    """

    rho_se = float(np.clip(float(row["rho_se"]), -1.0, 1.0))
    rho_me = float(np.clip(float(row["rho_me"]), -1.0, 1.0))
    rho_sm = float(np.clip(float(row["rho_sm"]), -1.0, 1.0))
    wm_y = np.sqrt(max(1.0 - rho_me**2, 0.0))
    world_model = np.array([0.0, wm_y, rho_me], dtype=np.float64)

    if wm_y <= 1e-12:
        student_y = 0.0
    else:
        student_y = (rho_sm - rho_se * rho_me) / wm_y
    maximum_y = np.sqrt(max(1.0 - rho_se**2, 0.0))
    student_y = float(np.clip(student_y, -maximum_y, maximum_y))
    student_x = np.sqrt(max(1.0 - rho_se**2 - student_y**2, 0.0))
    student = np.array([student_x, student_y, rho_se], dtype=np.float64)
    return student, world_model


def draw_sphere(axis) -> None:
    """Draw the translucent unit sphere and fixed environment direction."""

    longitude = np.linspace(0.0, 2.0 * np.pi, 36)
    polar = np.linspace(0.0, np.pi, 18)
    axis.plot_surface(
        np.outer(np.cos(longitude), np.sin(polar)),
        np.outer(np.sin(longitude), np.sin(polar)),
        np.outer(np.ones_like(longitude), np.cos(polar)),
        color="#dddddd",
        alpha=0.12,
        linewidth=0.0,
        shade=False,
    )

    # A light latitude/longitude mesh keeps the sphere legible after GIF
    # palette quantization, where the translucent surface alone is too faint.
    angle = np.linspace(0.0, 2.0 * np.pi, 120)
    for height in np.linspace(-0.75, 0.75, 7):
        radius = np.sqrt(1.0 - height**2)
        axis.plot(
            radius * np.cos(angle),
            radius * np.sin(angle),
            np.full_like(angle, height),
            color="#c8c8c8",
            linewidth=0.4,
            alpha=0.55,
        )
    latitude = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 120)
    for longitude_value in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
        axis.plot(
            np.cos(latitude) * np.cos(longitude_value),
            np.cos(latitude) * np.sin(longitude_value),
            np.sin(latitude),
            color="#c8c8c8",
            linewidth=0.4,
            alpha=0.45,
        )

    axis.plot(
        [0.0, ENVIRONMENT[0]],
        [0.0, ENVIRONMENT[1]],
        [0.0, ENVIRONMENT[2]],
        color=REFERENCE_COLOR,
        linewidth=1.2,
        alpha=0.5,
    )
    axis.scatter(
        [ENVIRONMENT[0]],
        [ENVIRONMENT[1]],
        [ENVIRONMENT[2]],
        color=REFERENCE_COLOR,
        s=30,
        depthshade=False,
        label="environment",
    )


def configure_axis(axis) -> None:
    """Apply a common equal-aspect, axis-free view to a sphere plot."""

    axis.set(
        xlim=(-1.02, 1.02),
        ylim=(-1.02, 1.02),
        zlim=(-1.02, 1.02),
    )
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.view_init(elev=VIEW_ELEVATION, azim=VIEW_AZIMUTH)
    axis.set_axis_off()


def _segments(points: np.ndarray) -> np.ndarray:
    """Convert an ordered point path to pairs of 3D segment endpoints."""

    if len(points) < 2:
        return np.empty((0, 2, 3), dtype=np.float64)
    return np.stack((points[:-1], points[1:]), axis=1)


def _segment_colors(
    records: Sequence[Mapping[str, object]],
    color: str,
) -> list[tuple[float, float, float, float]]:
    """Color each segment by the phase in which its endpoint was recorded."""

    from matplotlib.colors import to_rgba

    return [
        to_rgba(
            color,
            OFFLINE_ALPHA if record.get("mode") == "offline" else 1.0,
        )
        for record in records[1:]
    ]


def draw_trajectory(
    axis,
    records: Sequence[Mapping[str, object]],
    points: np.ndarray,
    *,
    color: str,
    label: str,
    linewidth: float = 2.2,
):
    """Draw a solid trajectory with translucent offline segments."""

    from matplotlib.colors import to_rgba
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    segments = _segments(points)
    colors = _segment_colors(records, color)
    if not len(segments):
        # Keep single-point trajectories compatible with Matplotlib versions
        # that cannot attach an empty collection to a 3D axis.
        segments = np.repeat(points[0][None, None, :], 2, axis=1)
        colors = [to_rgba(color)]
    collection = Line3DCollection(
        segments,
        colors=colors,
        linewidths=linewidth,
    )
    axis.add_collection3d(collection)
    # A proxy line gives the 3D collection a reliable legend entry.
    axis.plot([], [], [], color=color, linewidth=linewidth, label=label)
    return collection


def _actual_reward_rate(row: Mapping[str, object]) -> float:
    """Return realized reward per elapsed time, including offline zeros."""

    value = row.get("actual_reward_rate", "")
    if value not in {"", None}:
        return float(value)
    # Older result files predate the explicit field. Their online
    # ``reward_rate`` is usable, but their offline value is hypothetical.
    if row.get("mode") == "offline":
        return 0.0
    fallback = row.get("reward_rate", "")
    if fallback in {"", None}:
        raise ValueError("reward-rate bar requires an actual_reward_rate value")
    return float(fallback)


def _evaluation_reward_rate(row: Mapping[str, object]) -> float:
    """Return the rate the online evaluation policy would currently achieve."""

    value = row.get("evaluation_reward_rate", "")
    if value not in {"", None}:
        return float(value)
    fallback = row.get("reward_rate", "")
    if fallback in {"", None}:
        raise ValueError("reward-rate bar requires an evaluation_reward_rate value")
    return float(fallback)


def animate_sphere(
    records: Sequence[Mapping[str, object]],
    output: Path,
    *,
    frame_count: int = 36,
    fps: int = 10,
    dpi: int = 80,
    show_reward_rate: bool = False,
) -> None:
    """Animate sphere trajectories and optionally their instantaneous reward rate."""

    if not records:
        raise ValueError("cannot animate an empty trajectory")
    if frame_count < 2:
        raise ValueError("frame_count must be at least two")
    if fps <= 0 or dpi <= 0:
        raise ValueError("fps and dpi must be positive")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    sample_indices = np.linspace(
        0,
        len(records) - 1,
        min(frame_count, len(records)),
    ).round().astype(int)
    sample_indices = np.unique(sample_indices)
    sampled = [records[index] for index in sample_indices]
    pairs = [vectors(row) for row in sampled]
    student = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    world_model = np.asarray([pair[1] for pair in pairs], dtype=np.float64)

    figure = plt.figure(
        figsize=(6.0, 4.6) if show_reward_rate else (5.0, 4.6),
        facecolor="white",
    )
    reward_axis = None
    reward_bar = None
    reward_rates = None
    hypothetical_bar = None
    evaluation_rates = None
    if show_reward_rate:
        reward_rates = np.asarray(
            [_actual_reward_rate(row) for row in sampled],
            dtype=np.float64,
        )
        evaluation_rates = np.asarray(
            [_evaluation_reward_rate(row) for row in sampled],
            dtype=np.float64,
        )
        if (
            not np.all(np.isfinite(reward_rates))
            or np.any(reward_rates < 0.0)
            or not np.all(np.isfinite(evaluation_rates))
            or np.any(evaluation_rates < 0.0)
        ):
            raise ValueError("reward_rate values must be finite and nonnegative")
        layout = figure.add_gridspec(1, 2, width_ratios=(5.0, 1.0), wspace=0.02)
        axis = figure.add_subplot(layout[0, 0], projection="3d")
        reward_axis = figure.add_subplot(layout[0, 1])
    else:
        axis = figure.add_subplot(111, projection="3d")

    draw_sphere(axis)
    configure_axis(axis)
    # Newer Matplotlib versions try to concatenate a collection's segments
    # when it is attached to a 3D axis and reject a completely empty
    # collection. Seed each path with an invisible zero-length segment; the
    # first animation update replaces it before rendering.
    student_seed = np.repeat(student[0][None, None, :], 2, axis=1)
    model_seed = np.repeat(world_model[0][None, None, :], 2, axis=1)
    student_path = Line3DCollection(
        student_seed, colors=[STUDENT_COLOR], linewidths=2.2
    )
    model_path = Line3DCollection(
        model_seed, colors=[WORLD_MODEL_COLOR], linewidths=2.2
    )
    axis.add_collection3d(student_path)
    axis.add_collection3d(model_path)
    axis.plot([], [], [], color=STUDENT_COLOR, linewidth=2.2, label="student")
    axis.plot([], [], [], color=WORLD_MODEL_COLOR, linewidth=2.2, label="world model")
    student_point, = axis.plot(
        [], [], [], color=STUDENT_COLOR, marker="o", markersize=6.0,
        linestyle="none",
    )
    model_point, = axis.plot(
        [], [], [], color=WORLD_MODEL_COLOR, marker="o", markersize=6.0,
        linestyle="none",
    )
    time_label = axis.text2D(0.04, 0.94, "", transform=axis.transAxes)
    axis.legend(frameon=False, loc="upper right")

    if (
        reward_axis is not None
        and reward_rates is not None
        and evaluation_rates is not None
    ):
        reward_ceiling = 1.12 * float(
            max(np.max(reward_rates), np.max(evaluation_rates))
        )
        if reward_ceiling <= 0.0:
            reward_ceiling = 1.0
        reward_axis.set(
            xlim=(-0.65, 0.65),
            ylim=(0.0, reward_ceiling),
            xticks=[],
            title="instantaneous\nreward rate",
        )
        hypothetical_bar = reward_axis.bar(
            [0.0],
            [0.0],
            width=0.72,
            color="#dcfce7",
            edgecolor="#bbf7d0",
            linewidth=0.6,
            zorder=2,
        )[0]
        reward_bar = reward_axis.bar(
            [0.0], [0.0], width=0.72, color="#16a34a", zorder=3
        )[0]
        reward_axis.grid(axis="y", color="#dddddd", linewidth=0.6)
        reward_axis.spines[["top", "right", "bottom"]].set_visible(False)
        reward_axis.tick_params(axis="y", labelsize=8)

    if show_reward_rate:
        # ``tight_layout`` does not support a mixed 3D/2D figure reliably.
        figure.subplots_adjust(
            left=0.01,
            right=0.97,
            bottom=0.02,
            top=0.88,
            wspace=0.04,
        )
    else:
        figure.tight_layout(pad=0.2)

    def update(frame: int):
        """Extend both paths through the selected trajectory sample."""

        end = frame + 1
        visible_records = sampled[:end]
        if end == 1:
            student_path.set_segments(student_seed)
            student_path.set_color([STUDENT_COLOR])
            model_path.set_segments(model_seed)
            model_path.set_color([WORLD_MODEL_COLOR])
        else:
            student_path.set_segments(_segments(student[:end]))
            student_path.set_color(_segment_colors(visible_records, STUDENT_COLOR))
            model_path.set_segments(_segments(world_model[:end]))
            model_path.set_color(
                _segment_colors(visible_records, WORLD_MODEL_COLOR)
            )
        student_point.set_data_3d(*student[frame].reshape(3, 1))
        model_point.set_data_3d(*world_model[frame].reshape(3, 1))
        time_label.set_text(f"time = {float(sampled[frame]['time']):.1f}")
        artists = [
            student_path,
            model_path,
            student_point,
            model_point,
            time_label,
        ]
        if (
            reward_bar is not None
            and reward_rates is not None
            and hypothetical_bar is not None
            and evaluation_rates is not None
        ):
            value = float(reward_rates[frame])
            offline = sampled[frame].get("mode") == "offline"
            hypothetical_value = (
                float(evaluation_rates[frame]) if offline else 0.0
            )
            hypothetical_bar.set_height(hypothetical_value)
            reward_bar.set_height(value)
            artists.extend((hypothetical_bar, reward_bar))
        return tuple(artists)

    animation = FuncAnimation(
        figure,
        update,
        frames=len(sampled),
        interval=1000.0 / fps,
        blit=False,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(figure)
