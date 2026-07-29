"""Embed a correlation trajectory as student and WM paths on a unit sphere."""

import argparse

import numpy as np

from _common import add_plot_args, grouped, paths, read_records
from _sphere import (
    STUDENT_COLOR,
    WORLD_MODEL_COLOR,
    configure_axis,
    draw_sphere,
    draw_trajectory,
    vectors,
)


def main() -> None:
    """Read a trajectory CSV and render its static sphere projection."""

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser, "analytical_trajectory")
    args = parser.parse_args()
    source, base_output = paths(args)
    output = base_output.with_name(base_output.stem + "_sphere" + base_output.suffix)
    rows = read_records(source)
    figure = plt.figure(figsize=(7, 6.5))
    axis = figure.add_subplot(111, projection="3d")
    draw_sphere(axis)
    for condition, condition_rows in grouped(rows).items():
        pairs = [vectors(row) for row in condition_rows]
        student = np.asarray([pair[0] for pair in pairs])
        world_model = np.asarray([pair[1] for pair in pairs])
        draw_trajectory(
            axis,
            condition_rows,
            student,
            color=STUDENT_COLOR,
            label=f"{condition} student",
        )
        draw_trajectory(
            axis,
            condition_rows,
            world_model,
            color=WORLD_MODEL_COLOR,
            label=f"{condition} WM",
        )
    configure_axis(axis)
    axis.legend(frameon=False)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, transparent=True)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
