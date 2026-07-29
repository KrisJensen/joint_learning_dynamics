"""Plot planning and greedy policy regimes."""
import argparse
from _common import add_plot_args, run_dashboard

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "policy_regime_comparison")
run_dashboard(parser.parse_args(), "Training and evaluation policy regimes")
