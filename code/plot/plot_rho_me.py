"""Plot initial environment-WM correlation comparisons."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "rho_me_comparison")
run_dashboard(parser.parse_args(), "Initial environment-WM correlation")
