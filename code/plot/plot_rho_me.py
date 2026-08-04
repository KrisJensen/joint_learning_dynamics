"""Plot initial environment-WM correlation comparisons."""
import argparse
from _common import (
    add_plot_args,
    paths,
    plot_dashboard,
    plot_metric_panels,
    read_records,
    swept_value_styles,
)

parser = argparse.ArgumentParser(description=__doc__)
add_plot_args(parser, "rho_me_comparison")
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
styles = swept_value_styles(records, "initial_rho_me")
baseline_conditions = {
    row.get("condition", "trajectory")
    for row in records
    if row.get("comparison_group") == "greedy_baseline"
}
for condition in baseline_conditions:
    styles[condition] = {
        "color": "black",
        "linestyle": "--",
        "linewidth": 1.8,
        "zorder": 1,
    }

plot_dashboard(
    records, output, "Initial environment-WM correlation", args.dpi, styles
)
correlation_output = output.with_name(
    f"{output.stem}_student_environment_correlation{output.suffix}"
)
plot_metric_panels(
    records,
    correlation_output,
    (("rho_se", r"$\rho_{se}$"),),
    styles,
    dpi=args.dpi,
    ylims={"rho_se": (0.0, 1.0)},
)
print(f"Wrote {output}")
print(f"Wrote {correlation_output}")
