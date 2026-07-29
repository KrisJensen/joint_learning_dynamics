"""Plot a generic parameter sweep result."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("parameter")
args, remaining = parser.parse_known_args()
add_plot_args(parser, f"{args.parameter}_sweep")
run_dashboard(parser.parse_args(), f"{args.parameter} sweep")
