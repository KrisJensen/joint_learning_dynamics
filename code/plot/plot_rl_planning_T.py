"""Plot reward-modulated learning and planning across T."""

import argparse

from _common import add_plot_args, plot_dashboard, paths, read_records


parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "rl_planning_T_comparison")
args = parser.parse_args()
source, output = paths(args)
plot_dashboard(
    read_records(source),
    output,
    "Reinforcement learning and planning across T",
    args.dpi,
)
print(f"Wrote {output}")
