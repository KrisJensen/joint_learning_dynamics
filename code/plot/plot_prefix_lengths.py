"""Plot fixed and optimized prefix lengths."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "prefix_length_comparison")
run_dashboard(parser.parse_args(), "Evaluated prefix-length comparison")
