"""Run one analytical learning-dynamics trajectory."""

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    destination,
    initial_state,
    make_params,
    with_metadata,
    write_records,
)
from learning_dynamics import run_analytical


def main() -> None:
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
    path = destination(args, "analytical_trajectory")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
