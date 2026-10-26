"""Compare analytical trajectories across values of one model parameter."""

import argparse

from _common import add_analytical_args, add_common_args, analytical_kwargs, initial_state, make_params, output_path, with_metadata, write_records
from joint_dynamics import run_analytical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    parser.add_argument("--parameter", required=True, choices=["T", "K", "beta", "epsilon", "p_flip", "eta_e", "tau_on", "tau_off"])
    parser.add_argument("--values", required=True, help="comma-separated numeric values")
    args = parser.parse_args()
    rows = []
    for text in args.values.split(","):
        value = int(text) if args.parameter in {"T", "K"} else float(text)
        params = make_params(args, **{args.parameter: value})
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params, source="analytical", condition=f"{args.parameter}={value}",
        )
    destination = args.output or output_path(f"{args.parameter}_sweep", args.suffix, args.output_subdir)
    write_records(destination, rows)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
