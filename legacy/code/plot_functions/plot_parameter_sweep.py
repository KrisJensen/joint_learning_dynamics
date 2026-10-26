"""Plot a parameter-sweep CSV."""

import argparse
from pathlib import Path
from _common import figures_dir, plot_dashboard, read_records, results_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("parameter")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output-subdir", default="")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()
    suffix = args.suffix.strip().strip("_")
    stem = f"{args.parameter}_sweep_{suffix}" if suffix else f"{args.parameter}_sweep"
    source = args.input or results_dir(args.output_subdir) / f"{stem}.csv"
    output = args.output or figures_dir(args.output_subdir) / f"{stem}.pdf"
    plot_dashboard(read_records(source), output, f"{args.parameter} sweep", args.dpi)
    print(f"Wrote {output}")
