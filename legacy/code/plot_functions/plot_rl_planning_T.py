"""Plot reinforcement learning and planning across sequence lengths T."""

import argparse

from _common import add_plot_args, paths, plot_dashboard, read_records


def comparison_styles(records):
    T_values = sorted({int(float(row["T_value"])) for row in records})
    colors = {T: f"C{index % 10}" for index, T in enumerate(T_values)}
    styles = {}
    for row in records:
        T = int(float(row["T_value"]))
        styles[row["condition"]] = {
            "color": colors[T],
            "linestyle": "--"
            if row["method"] == "reinforcement learning"
            else "-",
        }
    return styles


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser, "rl_planning_T_comparison")
    args = parser.parse_args()
    source, destination = paths(args)
    records = read_records(source)
    plot_dashboard(
        records,
        destination,
        "Reinforcement Learning and Planning Across T",
        args.dpi,
        condition_styles=comparison_styles(records),
    )
    print(f"Wrote {destination}")
