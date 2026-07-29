"""Compare reward-modulated learning and planning across sequence lengths T."""

from __future__ import annotations

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    destination,
    initial_state,
    make_params,
    parse_list,
    with_metadata,
    write_records,
)
from learning_dynamics import run_analytical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(schedule="online:20")
    add_analytical_args(parser)
    parser.add_argument("--T-values", default="1,3,5,7,9")
    args = parser.parse_args()

    rows = []
    methods = (
        ("reinforcement learning", 1, float("inf"), "constant"),
        ("planning", 5, args.beta, "optimal"),
    )
    for T in parse_list(args.T_values, int):
        print("running T", T)
        if T < 1:
            parser.error("T values must be positive")
        for method, K, beta, temperature_mode in methods:
            params = make_params(
                args,
                T=T,
                M=T,
                K=K,
                K_mode="constant",
                M_mode="constant",
                beta=beta,
                temperature_mode=temperature_mode,
                eta_2f=0.0,
                fix_eta_2f_zero=True,
            )
            rows += with_metadata(
                run_analytical(
                    initial_state(args), params, **analytical_kwargs(args)
                ),
                params,
                source="analytical",
                condition=f"{method}, T={T}",
                method=method,
                T_value=T,
            )
    path = destination(args, "rl_planning_T_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
