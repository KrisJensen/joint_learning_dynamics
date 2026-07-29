"""Plot cyclic modulation comparisons."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "modulation_comparison")
run_dashboard(parser.parse_args(), "Cyclic environment modulation")
