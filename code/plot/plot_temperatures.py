"""Plot fixed and optimized planning temperatures."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "temperature_comparison")
run_dashboard(parser.parse_args(), "Planning inverse-temperature comparison")
