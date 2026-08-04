"""Compare planning/greedy training and evaluation combinations."""

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
    # The report absorbs the legacy zero-field score factor of 1/2 into every
    # learning rate. Thus each rate below is half the corresponding legacy
    # value of 2.5, including the optimization cap (legacy value 20).
    parser.set_defaults(
        schedule="online:150",
        T=6,
        K=5,
        K_mode="constant",
        beta=2.0,
        beta_max=20.0,
        temperature_mode="constant",
        eta_2f=0.0,
        eta_3f=1.0,
        eta_off=1.0,
        eta_wm=1.0,
        eta_max=1.0,
        delta_on=0.05,
        gh_order=25,
    )
    args = parser.parse_args()
    initial = initial_state(
        args,
        rho_me=1.0,
        rho_sm=args.rho_se0,
    )
    rows = []
    for greedy_training, greedy_evaluation in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        training = "greedy" if greedy_training else "planning"
        evaluation = "greedy" if greedy_evaluation else "planning"
        print(f"training: {training}, evaluation: {evaluation}")
        params = make_params(
            args,
            K_mode="constant",
            eta_e=0.0,
            eta_wm=0.0,
            fix_eta_wm_zero=True,
            greedy_training=greedy_training,
            greedy_evaluation=greedy_evaluation,
        )
        rows += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=f"{training} training / {evaluation} evaluation",
            training_policy=training,
            evaluation_policy=evaluation,
        )
    path = destination(args, "policy_regime_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
