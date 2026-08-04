"""Plot fixed and optimized planning temperatures."""
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
add_plot_args(parser, "temperature_comparison")
args = parser.parse_args()
source, output = paths(args)
records = read_records(source)
styles = swept_value_styles(records, "temperature_value")

plot_dashboard(
    records, output, "Planning inverse-temperature comparison", args.dpi, styles
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
temperature_output = output.with_name(
    f"{output.stem}_temperature{output.suffix}"
)
plot_metric_panels(
    records,
    temperature_output,
    (("beta", r"$\beta$"),),
    styles,
    dpi=args.dpi,
)
print(f"Wrote {output}")
print(f"Wrote {correlation_output}")
print(f"Wrote {temperature_output}")
