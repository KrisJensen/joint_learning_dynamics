"""Plot online learning dynamics for fixed and instantaneously optimal K."""

import argparse

from _common import add_plot_args, run_plot


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_args(parser, "K_comparison")
    run_plot(parser.parse_args(), "Online Learning Across Fixed and Optimal K")
