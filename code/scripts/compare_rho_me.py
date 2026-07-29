"""Compare trajectories across initial environment-WM correlations."""

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
    parser.add_argument("--rho-me-values", default="0,0.3,0.6,0.8,1")
    args = parser.parse_args()
    params = make_params(args, eta_wm=0.0, fix_eta_wm_zero=True)
    rows = []
    for rho_me in parse_list(args.rho_me_values):
        initial = initial_state(args, rho_me=rho_me)
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"rho_me(0)={rho_me:g}",
            initial_rho_me=rho_me,
        )
    path = destination(args, "rho_me_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
