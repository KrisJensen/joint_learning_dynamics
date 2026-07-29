"""Evaluate reward and action observables over all M in [1,T]."""

import argparse

from _common import (
    add_common_args,
    destination,
    make_params,
    parse_list,
    with_metadata,
    write_records,
)
from learning_dynamics import State, observables, resolve_online_policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--rho-values",
        default="0.1,0.3,0.5,0.7,0.9",
        help="uses rho_se=rho_sm=rho and rho_me=1",
    )
    args = parser.parse_args()
    rows = []
    for rho in parse_list(args.rho_values):
        state = State(rho, 1.0, rho)
        state.validate()
        for M in range(1, args.T + 1):
            params = make_params(args, M=M, M_mode="constant")
            resolved = resolve_online_policy(state, params)
            values = observables(state, params, resolved=resolved)
            rows += with_metadata(
                [
                    {
                        "rho": rho,
                        "rho_se": state.rho_se,
                        "rho_me": state.rho_me,
                        "rho_sm": state.rho_sm,
                        "K": resolved.params.K,
                        "M": M,
                        "beta": resolved.beta,
                        "online_cost": resolved.params.c_on,
                        **values,
                    }
                ],
                params,
                source="analytical",
                condition=f"rho={rho:g}",
            )
    path = destination(args, "prefix_length_observables")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
