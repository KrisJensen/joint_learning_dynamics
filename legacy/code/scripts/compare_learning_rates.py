"""Compare five shared rate levels with an optimal constrained schedule."""

import argparse

from _common import add_analytical_args, add_common_args, analytical_kwargs, initial_state, make_params, output_path, with_metadata, write_records
from joint_dynamics import run_analytical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(schedule="online:20", duration=20.0, eta_e=0.0)
    add_analytical_args(parser)
    parser.add_argument(
        "--rate-values",
        default="0.5,1,2.5,5,10",
        help=(
            "five comma-separated values assigned jointly to "
            "eta_on2, eta_off, and eta_wm"
        ),
    )
    args = parser.parse_args()
    rate_values = [
        float(value.strip()) for value in args.rate_values.split(",") if value.strip()
    ]
    if any(value < 0.0 for value in rate_values):
        parser.error("--rate-values must be nonnegative")

    rows = []
    for rate in rate_values:
        print("Running analytical simulation for rate =", rate)
        params = make_params(
            args,
            learning_rate_mode="constant",
            eta_on1=0.0,
            eta_on2=rate,
            eta_off=rate,
            eta_wm=rate,
            fix_eta_on1_zero=True,
            fix_eta_on2_zero=False,
            fix_eta_wm_zero=False,
        )
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"constant rate = {rate:g}",
            rate_value=rate,
        )

    optimal = make_params(
        args,
        learning_rate_mode="optimal",
        eta_on1=0.0,
        fix_eta_on1_zero=True,
        fix_eta_on2_zero=False,
        fix_eta_wm_zero=False,
    )
    rows += with_metadata(
        run_analytical(initial_state(args), optimal, **analytical_kwargs(args)),
        optimal,
        source="analytical",
        condition="optimal rates",
        rate_value="optimal",
    )
    destination = args.output or output_path("learning_rate_comparison", args.suffix, args.output_subdir)
    write_records(destination, rows)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
