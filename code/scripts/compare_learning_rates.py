"""Compare fixed shared learning rates with bounded instantaneous optima."""

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
    parser.add_argument("--rate-values", default="0.2,0.5,1.0,2.0")
    args = parser.parse_args()
    rows = []
    for rate in parse_list(args.rate_values):
        print("running rate", rate)
        params = make_params(
            args,
            learning_rate_mode="constant",
            eta_2f=rate,
            eta_3f=rate,
            eta_off=rate,
            eta_wm=rate,
        )
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"rate={rate:g}",
            rate_value=rate,
        )
    params = make_params(args, learning_rate_mode="optimal")
    rows += with_metadata(
        run_analytical(initial_state(args), params, **analytical_kwargs(args)),
        params,
        source="analytical",
        condition="optimal rates",
        rate_value="optimal",
    )
    path = destination(args, "learning_rate_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
