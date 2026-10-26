"""Compare fixed and instantaneously optimal online planner sample counts K."""

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
    parser.set_defaults(
        schedule="online:10",
        duration=10.0,
        eta_e=0.0,
        temperature_mode="optimal",
    )
    add_analytical_args(parser)
    parser.add_argument(
        "--K-values",
        default="1,4,15,50",
        help="comma-separated positive planner sample counts",
    )
    args = parser.parse_args()

    K_values = [
        int(value.strip()) for value in args.K_values.split(",") if value.strip()
    ]
    if not K_values or any(value < 1 for value in K_values):
        parser.error("--K-values must contain positive integers")

    initial = State(
        rho_se=args.rho_se0,
        rho_me=args.rho_me0,
        rho_sm=args.rho_sm0,
    )
    initial.validate()

    records = []
    for K in K_values:
        print(f"Running analytical simulation for K={K}...")
        params = make_params(
            args,
            K=K,
            K_mode="constant",
            eta_e=0.0,
        )
        records += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=rf"$K={K}$",
            K_value=K,
        )

    for criterion in ("reward",):#, "drift"):
        print(f"Running analytical simulation for {criterion}-rate-optimal K...")
        params = make_params(
            args,
            K_mode=criterion,
            eta_e=0.0,
        )
        records += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=rf"optimal $K$ ({criterion} rate)",
            K_value=f"optimal_{criterion}",
        )

    destination = args.output or output_path("K_comparison", args.suffix, args.output_subdir)
    write_records(destination, records)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
