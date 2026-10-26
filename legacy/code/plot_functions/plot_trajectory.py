"""Plot a single analytical trajectory."""

import argparse
from _common import add_plot_args, run_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser, "analytical_trajectory")
    run_plot(parser.parse_args(), "Analytical joint dynamics")

