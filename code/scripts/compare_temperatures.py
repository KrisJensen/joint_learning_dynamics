"""Compare fixed inverse temperatures beta with instantaneous optimization."""

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
    add_analytical_args(parser)
    parser.add_argument("--beta-values", default="0,0.5,1,2,3,5,10")
    args = parser.parse_args()
    rows = []
    for beta in parse_list(args.beta_values):
        print("running temperature", beta)
        params = make_params(args, beta=beta, temperature_mode="constant")
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"beta={beta:g}",
            temperature_value=beta,
        )
    params = make_params(args, temperature_mode="optimal")
    rows += with_metadata(
        run_analytical(initial_state(args), params, **analytical_kwargs(args)),
        params,
        source="analytical",
        condition="optimal beta",
        temperature_value="optimal",
    )
    path = destination(args, "temperature_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
