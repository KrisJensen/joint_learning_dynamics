"""Compare analytical trajectories with independent finite-dimensional runs."""

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    add_empirical_args,
    analytical_kwargs,
    destination,
    initial_state,
    make_params,
    parse_list,
    schedule,
    with_metadata,
    write_records,
)
from learning_dynamics import run_analytical, run_empirical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    add_empirical_args(parser)
    args = parser.parse_args()
    params = make_params(args)
    initial = initial_state(args)
    rows = with_metadata(
        run_analytical(initial, params, **analytical_kwargs(args)),
        params,
        source="analytical",
        condition="analytical",
        d="",
        replicate="",
    )
    dimensions = parse_list(args.dimensions, int)
    for d in dimensions:
        for replicate in range(args.replicates):
            seed = args.seed + replicate
            record_every = args.record_every or max(
                1, int(round(args.step * d / params.c_on))
            )
            empirical = run_empirical(
                initial,
                params,
                schedule=schedule(args),
                d=d,
                record_every=record_every,
                seed=seed,
            )
            rows += with_metadata(
                empirical,
                params,
                source="empirical",
                condition=f"empirical d={d}",
                d=d,
                replicate=replicate,
                seed=seed,
            )
    path = destination(args, "empirical_vs_analytical")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
