"""Plot the planning-temperature comparison."""

import argparse
from _common import add_plot_args, run_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser, "temperature_comparison")
    run_plot(parser.parse_args(), "Constant versus optimal planning temperature")

