"""Generate online dynamics for different initial environment-WM correlations."""

from __future__ import annotations

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    make_params,
    output_path,
    with_metadata,
    write_records,
)
from joint_dynamics import State, run_analytical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(schedule="online:20", duration=20.0)
    add_analytical_args(parser)
    parser.add_argument(
        "--rho-me-values",
        default="0,0.3,0.6,0.8,0.9,1",
        help="comma-separated initial environment-WM correlations",
    )
    args = parser.parse_args()

    rho_me_values = [
        float(value.strip())
        for value in args.rho_me_values.split(",")
        if value.strip()
    ]
    if not rho_me_values:
        parser.error("--rho-me-values must contain at least one value")
    if any(not -1.0 <= value <= 1.0 for value in rho_me_values):
        parser.error("rho_me values must lie in [-1, 1]")

    params = make_params(args, eta_wm=0.0, fix_eta_wm_zero=True)
    records = []
    for rho_me in rho_me_values:
        print(f"Running analytical simulation for rho_me={rho_me:g}...")
        initial = State(
            rho_se=args.rho_se0,
            rho_me=rho_me
            ,
            rho_sm=args.rho_sm0,
        )
        try:
            initial.validate()
        except ValueError as error:
            parser.error(f"invalid initial state for rho_me={rho_me:g}: {error}")
        records += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=rf"$\rho_{{me}}(0)={rho_me:g}$",
            initial_rho_me=rho_me,
        )

    destination = args.output or output_path("rho_me_comparison", args.suffix, args.output_subdir)
    write_records(destination, records)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
