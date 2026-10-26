"""Compare online-offline-online learning for two modulation amplitudes."""

from __future__ import annotations

import argparse

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    initial_state,
    make_params,
    output_path,
    with_metadata,
    write_records,
)
from joint_dynamics import run_analytical


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(
        schedule="online:4,offline:4,online:4",
        T_M=4.0,
        phi_M=0.0,
    )
    add_analytical_args(parser)
    args = parser.parse_args()

    records = []
    for amplitude in (1.0, 0.0):
        print(f"Running analytical simulation for M={amplitude:g}...")
        params = make_params(args, M=amplitude)
        records += with_metadata(
            run_analytical(
                initial_state(args), params, **analytical_kwargs(args)
            ),
            params,
            source="analytical",
            condition=rf"$M={amplitude:g}$",
            modulation_amplitude=amplitude,
        )

    destination = args.output or output_path("modulation_comparison", args.suffix, args.output_subdir)
    write_records(destination, records)
    print(f"Wrote {destination}")


if __name__ == "__main__":
    main()
