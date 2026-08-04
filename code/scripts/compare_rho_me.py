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
    parser.add_argument("--rho-me-values", default="0.0,0.5,0.7,0.9,1")
    parser.add_argument(
        "--include-greedy-baseline",
        "--include-greedy",
        dest="include_greedy_baseline",
        action="store_true",
        help=(
            "also calculate a greedy-RL reference using greedy training and "
            "evaluation with no world-model learning"
        ),
    )
    args = parser.parse_args()
    params = make_params(args, eta_wm=0.0, fix_eta_wm_zero=True)
    rows = []
    for rho_me in parse_list(args.rho_me_values):
        print("running with rho_me(0) =", rho_me)
        initial = initial_state(args, rho_me=rho_me)
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=r'$\rho_{me}$'+f"={rho_me:g}",
            initial_rho_me=rho_me,
        )
    if args.include_greedy_baseline:
        print("running greedy RL baseline")
        greedy_params = make_params(
            args,
            eta_wm=0.0,
            fix_eta_wm_zero=True,
            greedy_training=True,
            greedy_evaluation=True,
        )
        greedy_initial = initial_state(args, rho_me=0.0, rho_sm=0.0)
        rows += with_metadata(
            run_analytical(
                greedy_initial,
                greedy_params,
                **analytical_kwargs(args),
            ),
            greedy_params,
            source="analytical",
            condition="greedy RL",
            initial_rho_me="greedy",
            comparison_group="greedy_baseline",
            training_policy="greedy",
        )
    path = destination(args, "rho_me_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
