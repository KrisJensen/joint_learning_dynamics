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
    args = parser.parse_args()
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
            greedy_training=greedy_training,
            greedy_evaluation=greedy_evaluation,
        )
        rows += with_metadata(
            run_analytical(initial_state(args), params, **analytical_kwargs(args)),
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
