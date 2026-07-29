"""Plot empirical and analytical trajectories."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "empirical_vs_analytical")
run_dashboard(parser.parse_args(), "Empirical versus analytical dynamics")
