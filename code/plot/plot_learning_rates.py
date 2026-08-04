"""Plot fixed and optimized learning rates."""
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
add_plot_args(parser, "learning_rate_comparison")
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
styles = swept_value_styles(records, "rate_value")

plot_dashboard(records, output, "Learning-rate comparison", args.dpi, styles)
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
learning_rate_output = output.with_name(
    f"{output.stem}_learning_rate{output.suffix}"
)
candidate_rate_metrics = (
    ("eta_2f", r"$\eta_{2f}$"),
    ("eta_3f", r"$\eta_{3f}$"),
)
rate_metrics = tuple(
    metric
    for metric in candidate_rate_metrics
    if any(
        row.get(metric[0], "") not in {"", None}
        and abs(float(row[metric[0]])) > 1e-14
        for row in records
    )
) or candidate_rate_metrics
plot_metric_panels(
    records,
    learning_rate_output,
    rate_metrics,
    styles,
    dpi=args.dpi,
)
print(f"Wrote {output}")
print(f"Wrote {correlation_output}")
print(f"Wrote {learning_rate_output}")
