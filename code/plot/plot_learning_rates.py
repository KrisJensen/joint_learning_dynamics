"""Plot fixed and optimized learning rates."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "learning_rate_comparison")
run_dashboard(parser.parse_args(), "Learning-rate comparison")
