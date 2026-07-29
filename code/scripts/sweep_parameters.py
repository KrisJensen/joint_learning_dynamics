"""Run analytical trajectories over values of one model parameter."""

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


INTEGER_PARAMETERS = {"T", "M", "K", "K_max", "gh_order"}
PARAMETERS = (
    "T",
    "M",
    "K",
    "beta",
    "eta_2f",
    "eta_3f",
    "eta_off",
    "eta_wm",
    "epsilon",
    "p_flip",
    "eta_e",
    "delta_on",
    "delta_off",
    "delta_switch",
    "C_A",
    "C_T",
    "C_phi",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    parser.add_argument("--parameter", required=True, choices=PARAMETERS)
    parser.add_argument("--values", required=True)
    args = parser.parse_args()
    converter = int if args.parameter in INTEGER_PARAMETERS else float
    rows = []
    for value in parse_list(args.values, converter):
        overrides = {args.parameter: value}
        if args.parameter == "T" and args.M is None:
            overrides["M"] = value
        params = make_params(args, **overrides)
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"{args.parameter}={value}",
            sweep_parameter=args.parameter,
            sweep_value=value,
        )
    path = destination(args, f"{args.parameter}_sweep")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
