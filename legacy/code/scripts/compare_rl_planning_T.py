"""Compare reinforcement learning and planning across sequence lengths T."""

from __future__ import annotations

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    initial_state,
    make_params,
    output_path,
    with_metadata,
    write_records,
)
from joint_dynamics import run_analytical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(schedule="online:20", duration=20.0)
    add_analytical_args(parser)
    parser.add_argument(
        "--T-values",
        default="1,3,5,7,9",
        help="comma-separated positive sequence lengths",
    )
    args = parser.parse_args()

    T_values = [
        int(value.strip()) for value in args.T_values.split(",") if value.strip()
    ]
    if not T_values or any(T < 1 for T in T_values):
        parser.error("--T-values must contain positive integers")

    initial = initial_state(args)
    records = []
    methods = (
        ("reinforcement learning", 1, float("inf"), "constant"),
        ("planning", 5, args.beta, "optimal"),
    )
    for T in T_values:
        for method, K, beta, temperature_mode in methods:
            print(f"Running {method} for T={T}...")
            params = make_params(
                args,
                T=T,
                K=K,
                K_mode="constant",
                beta=beta,
                temperature_mode=temperature_mode,
                eta_on1=0.0,
                fix_eta_on1_zero=True,
            )
            records += with_metadata(
                run_analytical(initial, params, **analytical_kwargs(args)),
                params,
                source="analytical",
                condition=rf"{method}, $T={T}$",
                method=method,
                T_value=T,
            )

    destination = args.output or output_path("rl_planning_T_comparison", args.suffix, args.output_subdir)
    write_records(destination, records)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
