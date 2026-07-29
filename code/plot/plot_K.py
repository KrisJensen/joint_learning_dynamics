"""Plot fixed and optimized K comparisons."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "K_comparison")
run_dashboard(parser.parse_args(), "Planner sample-count comparison")
