"""Compare analytical and finite-dimensional empirical trajectories."""

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    add_empirical_args,
    analytical_kwargs,
    empirical_dimensions,
    initial_state,
    make_params,
    output_path,
    run_schedule,
    with_metadata,
    write_records,
)
from joint_dynamics import run_analytical, run_empirical


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    add_empirical_args(parser)
    args = parser.parse_args()
    params = make_params(args)
    initial = initial_state(args)
    records = with_metadata(
        run_analytical(initial, params, **analytical_kwargs(args)),
        params,
        source="analytical",
        condition="analytical",
        d="",
    )
    schedule = run_schedule(args)
    for dimension in empirical_dimensions(args):
        print(f"Running empirical simulation for d={dimension}...")
        record_every = args.record_every or max(
            1, int(round(args.step * dimension / params.c_on))
        )
        records += with_metadata(
            run_empirical(
                initial,
                params,
                schedule=schedule,
                d=dimension,
                record_every=record_every,
                seed=args.seed,
            ),
            params,
            source="empirical",
            condition=f"empirical (d={dimension})",
            d=dimension,
        )
    destination = args.output or output_path("empirical_vs_analytical", args.suffix, args.output_subdir)
    write_records(destination, records)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
