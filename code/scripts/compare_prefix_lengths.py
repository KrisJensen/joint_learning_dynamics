"""Compare eta_2f-only prefix lengths with an optional eta_3f-only baseline."""

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
    print("running!")
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    parser.add_argument("--M-values", default=None, help="defaults to 1..T")
    parser.add_argument("--objectives", default="drift")
    parser.add_argument(
        "--include-eta-3f",
        action="store_true",
        help="also run an eta_3f-only baseline with eta_2f=0 and M=T",
    )
    args = parser.parse_args()
    values = (
        list(range(1, args.T + 1))
        if args.M_values is None
        else parse_list(args.M_values, int)
    )
    rows = []
    for M in values:
        print(f"running eta_2f-only M = {M}")
        params = make_params(
            args,
            M=M,
            M_mode="constant",
            eta_3f=0.0,
            fix_eta_2f_zero=False,
            fix_eta_3f_zero=True,
        )
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"eta_2f only, M={M}",
            comparison_group="eta_2f_fixed_M",
            compared_M=M,
            nonzero_rate="eta_2f",
            nonzero_rate_mode=args.learning_rate_mode,
        )
    for objective in parse_list(args.objectives, str):
        if objective not in {"reward", "drift"}:
            parser.error("objectives must be reward and/or drift")
        print(f"running eta_2f-only optimal M ({objective})")
        params = make_params(
            args,
            M_mode=objective,
            eta_3f=0.0,
            fix_eta_2f_zero=False,
            fix_eta_3f_zero=True,
        )
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"eta_2f only, optimal M ({objective})",
            comparison_group="eta_2f_optimal_M",
            compared_M=f"optimal_{objective}",
            nonzero_rate="eta_2f",
            nonzero_rate_mode=args.learning_rate_mode,
        )
    if args.include_eta_3f:
        print(f"running eta_3f-only baseline with M = {args.T}")
        params = make_params(
            args,
            M=args.T,
            M_mode="constant",
            eta_2f=0.0,
            fix_eta_2f_zero=True,
            fix_eta_3f_zero=False,
        )
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"eta_3f only, M={args.T}",
            comparison_group="eta_3f_only",
            compared_M=args.T,
            nonzero_rate="eta_3f",
            nonzero_rate_mode=args.learning_rate_mode,
        )
    path = destination(args, "prefix_length_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
