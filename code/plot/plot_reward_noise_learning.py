"""Plot 2-factor and 3-factor learning across reward observation noise."""

from __future__ import annotations

import argparse

from _common import add_plot_args, offline_spans, paths, read_records


def main() -> None:
    """Plot both learning rules in one panel, colored by reward noise."""

    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser, "unused")
    parser.add_argument(
        "--fixed-wm",
        action="store_true",
        help="read and plot the fixed-world-model comparison",
    )
    args = parser.parse_args()
    world_model = "fixed" if args.fixed_wm else "learned"
    args.stem = f"reward_noise_learning_comparison_{world_model}_wm"
    source, output = paths(args)
    records = read_records(source)
    if not records:
        raise ValueError("comparison result is empty")

    required = ("time", "rho_se", "reward_noise", "learning_rule", "world_model_type")
    for key in required:
        if any(row.get(key, "") == "" for row in records):
            raise ValueError(f"missing required result column: {key}")
    recorded_world_models = {row["world_model_type"] for row in records}
    if recorded_world_models != {world_model}:
        raise ValueError(
            f"expected {world_model} WM results, found "
            f"{sorted(recorded_world_models)}"
        )

    noise_values = sorted({float(row["reward_noise"]) for row in records})
    lower = noise_values[0]
    upper = noise_values[-1]
    normalization = Normalize(
        vmin=lower,
        vmax=upper if upper > lower else lower + 1.0,
    )
    color_map = plt.get_cmap("viridis")
    learning_rules = {
        "2-factor": "-",
        "3-factor": "--",
    }
    figure, axis = plt.subplots(figsize=(8.5, 5.4))
    spans = offline_spans(records)
    for start, end in spans:
        axis.axvspan(start, end, color="#e9ecef", zorder=0)
    for epsilon in noise_values:
        color = color_map(normalization(epsilon))
        for learning_rule, line_style in learning_rules.items():
            trajectory = [
                row
                for row in records
                if row["learning_rule"] == learning_rule
                and float(row["reward_noise"]) == epsilon
            ]
            trajectory.sort(key=lambda row: float(row["time"]))
            if trajectory:
                axis.plot(
                    [float(row["time"]) for row in trajectory],
                    [float(row["rho_se"]) for row in trajectory],
                    color=color,
                    linestyle=line_style,
                    linewidth=2.0,
                )

    axis.set(
        xlabel="elapsed time",
        ylabel=r"student–environment correlation $\rho_{se}$",
    )
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    rule_legend = axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="black",
                linestyle=line_style,
                linewidth=2.0,
                label=f"{learning_rule} learning",
            )
            for learning_rule, line_style in learning_rules.items()
        ],
        frameon=False,
        loc="upper left",
    )
    axis.add_artist(rule_legend)

    color_bar = figure.colorbar(
        ScalarMappable(norm=normalization, cmap=color_map),
        ax=axis,
        pad=0.025,
        fraction=0.055,
    )
    color_bar.set_label(r"reward observation noise $\epsilon$")
    if args.fixed_wm:
        fixed_value = records[0].get("fixed_rho_me", "")
        world_model_label = (
            f"fixed WM ($\\rho_{{me}}={float(fixed_value):g}$)"
            if fixed_value not in {"", None}
            else "fixed WM"
        )
    else:
        world_model_label = "learned WM"
    figure.suptitle(
        f"Reward-noise sensitivity of 2-factor and 3-factor learning · "
        f"{world_model_label}"
    )
    figure.subplots_adjust(
        left=0.11,
        right=0.88,
        bottom=0.11,
        top=0.89,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=args.dpi, transparent=True)
    plt.close(figure)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
