"""Plot a single analytical trajectory."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "analytical_trajectory")
run_dashboard(parser.parse_args(), "Analytical learning dynamics")
