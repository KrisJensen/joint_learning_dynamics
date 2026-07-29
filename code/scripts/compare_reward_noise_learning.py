"""Compare 2-factor and 3-factor learning across reward observation noise."""

from __future__ import annotations

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
    """Run the learning-rule, world-model, and reward-noise comparison."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    parser.add_argument(
        "--epsilon-values",
        default="0,0.1,0.2,0.3,0.4,0.5",
        help="comma-separated reward observation noise probabilities",
    )
    parser.add_argument(
        "--fixed-wm",
        action="store_true",
        help="hold the world-model/environment correlation fixed",
    )
    parser.add_argument(
        "--fixed-rho-me",
        type=float,
        default=1.0,
        help="fixed world-model/environment correlation (default: 1)",
    )
    args = parser.parse_args()
    epsilon_values = parse_list(args.epsilon_values)
    if any(not 0.0 <= epsilon <= 0.5 for epsilon in epsilon_values):
        parser.error("epsilon values must lie in [0, 0.5]")
    if args.fixed_wm and not -1.0 <= args.fixed_rho_me <= 1.0:
        parser.error("fixed-rho-me must lie in [-1, 1]")
    if args.fixed_wm and args.eta_e != 0.0:
        parser.error(
            "fixed-wm requires eta-e=0 so rho_me remains fixed"
        )

    learning_rules = (
        (
            "2-factor",
            {
                "eta_3f": 0.0,
                "fix_eta_2f_zero": False,
                "fix_eta_3f_zero": True,
                "eta_2f": 1.0
            },
        ),
        (
            "3-factor",
            {
                "eta_2f": 0.0,
                "fix_eta_2f_zero": True,
                "fix_eta_3f_zero": False,
                "eta_3f": 1.0
            },
        ),
    )
    world_model = "fixed" if args.fixed_wm else "learned"
    if args.fixed_wm:
        initial = initial_state(args, rho_me=args.fixed_rho_me)
        world_model_overrides = {
            "eta_wm": 0.0,
            "fix_eta_wm_zero": True,
        }
    else:
        initial = initial_state(args)
        world_model_overrides = {"fix_eta_wm_zero": False,
                                 "eta_wm": 1.0}

    rows = []
    for learning_rule, rule_overrides in learning_rules:
        for epsilon in epsilon_values:
            print(
                f"running {learning_rule}, {world_model} world model, "
                f"epsilon={epsilon:g}"
            )
            params = make_params(
                args,
                epsilon=epsilon,
                **world_model_overrides,
                **rule_overrides,
            )
            condition = (
                f"{learning_rule} / {world_model} WM / epsilon={epsilon:g}"
            )
            rows += with_metadata(
                run_analytical(
                    initial,
                    params,
                    **analytical_kwargs(args),
                ),
                params,
                source="analytical",
                condition=condition,
                learning_rule=learning_rule,
                world_model_type=world_model,
                fixed_rho_me=args.fixed_rho_me if args.fixed_wm else "",
                reward_noise=epsilon,
            )

    stem = f"reward_noise_learning_comparison_{world_model}_wm"
    path = destination(args, stem)
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
