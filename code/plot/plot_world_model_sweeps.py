"""Plot world-model parameter sweeps."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "world_model_sweeps")
run_dashboard(parser.parse_args(), "World-model parameter sweeps")
