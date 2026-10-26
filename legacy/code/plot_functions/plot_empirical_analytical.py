"""Plot empirical and analytical trajectories."""

import argparse
from _common import add_plot_args, run_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser, "empirical_vs_analytical")
    run_plot(parser.parse_args(), "Empirical versus analytical dynamics")

