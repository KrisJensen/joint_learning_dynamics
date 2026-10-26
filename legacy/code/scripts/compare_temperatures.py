"""Compare a constant-temperature sweep with an optimal schedule."""

import argparse

from _common import add_analytical_args, add_common_args, analytical_kwargs, initial_state, make_params, output_path, with_metadata, write_records
from joint_dynamics import run_analytical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(schedule="online:20", duration=20.0)
    add_analytical_args(parser)
    parser.add_argument(
        "--temperature-values",
        "--beta-values",
        dest="temperature_values",
        default="0,0.5,1,2,3,5,10",
        help="comma-separated constant planning inverse temperatures beta",
    )
    args = parser.parse_args()
    temperature_values = [
        float(value.strip())
        for value in args.temperature_values.split(",")
        if value.strip()
    ]
    if not temperature_values:
        parser.error("--temperature-values must contain at least one value")
    if any(value < 0.0 for value in temperature_values):
        parser.error("temperature values must be nonnegative")

    rows = []
    for temperature in temperature_values:
        print(f"Running analytical simulation for constant beta={temperature:g}...")
        params = make_params(
            args,
            beta=temperature,
            temperature_mode="constant",
        )
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=rf"constant $\beta={temperature:g}$",
            temperature_value=temperature,
        )

    optimal = make_params(args, temperature_mode="optimal")
    rows += with_metadata(
        run_analytical(initial_state(args), optimal, **analytical_kwargs(args)),
        optimal,
        source="analytical",
        condition=r"optimal $\beta(t)$",
        temperature_value="optimal",
    )
    destination = args.output or output_path("temperature_comparison", args.suffix, args.output_subdir)
    write_records(destination, rows)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
