"""Compare world-model rates, label noise, and greedy student rates."""

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
    parser.add_argument("--eta-wm-values", default="0.1,0.3,0.5,1")
    parser.add_argument("--p-flip-values", default="0,0.1,0.2,0.4")
    parser.add_argument("--greedy-rate-values", default="0.5,1,2.5,5")
    args = parser.parse_args()
    initial = initial_state(args)
    rows = []
    for value in parse_list(args.eta_wm_values):
        params = make_params(
            args,
            eta_2f=0.0,
            eta_3f=0.0,
            fix_eta_2f_zero=True,
            fix_eta_3f_zero=True,
            eta_wm=value,
        )
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"eta_wm={value:g}",
            sweep="eta_wm",
            sweep_value=value,
        )
    for value in parse_list(args.p_flip_values):
        params = make_params(
            args,
            eta_2f=0.0,
            eta_3f=0.0,
            fix_eta_2f_zero=True,
            fix_eta_3f_zero=True,
            p_flip=value,
        )
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"p_flip={value:g}",
            sweep="p_flip",
            sweep_value=value,
        )
    for value in parse_list(args.greedy_rate_values):
        params = make_params(
            args,
            greedy_training=True,
            greedy_evaluation=True,
            eta_2f=0.0,
            eta_3f=value,
            fix_eta_2f_zero=True,
        )
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"greedy eta_3f={value:g}",
            sweep="greedy_eta_3f",
            sweep_value=value,
        )
    path = destination(args, "world_model_sweeps")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
