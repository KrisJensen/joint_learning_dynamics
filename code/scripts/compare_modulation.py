"""Compare cyclic environment modulation amplitudes."""

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
    parser.set_defaults(schedule="online:4,offline:4,online:4", C_T=4.0)
    add_analytical_args(parser)
    parser.add_argument("--amplitudes", default="0,1")
    args = parser.parse_args()
    rows = []
    for amplitude in parse_list(args.amplitudes):
        params = make_params(args, C_A=amplitude)
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"C_A={amplitude:g}",
            modulation_amplitude=amplitude,
        )
    path = destination(args, "modulation_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
