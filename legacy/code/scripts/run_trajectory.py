"""Run one analytical joint-dynamics trajectory."""

import argparse

from _common import add_analytical_args, add_common_args, analytical_kwargs, initial_state, make_params, output_path, with_metadata, write_records
from joint_dynamics import run_analytical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    args = parser.parse_args()
    params = make_params(args)
    rows = with_metadata(
        run_analytical(initial_state(args), params, **analytical_kwargs(args)),
        params,
        source="analytical",
        condition="trajectory",
    )
    destination = args.output or output_path("analytical_trajectory", args.suffix, args.output_subdir)
    write_records(destination, rows)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
