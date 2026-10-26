"""Compare per-action supervised learning and reinforcement learning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPOSITORY_ROOT / "code" / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _common import (  # noqa: E402
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    destination,
    initial_state,
    make_params,
    with_metadata,
    write_records,
)
from learning_dynamics import run_analytical  # noqa: E402


CONDITIONS = (
    ("reinforcement", 6),
    ("supervised", 6),
    ("reinforcement", 7),
    ("supervised", 7),
)


def add_learner_metrics(records: list[dict], learner: str, T: int) -> list[dict]:
    """Expose comparable learner performance for either state vector."""

    enriched = []
    for record in records:
        if learner == "reinforcement":
            correlation = record["rho_se"]
            action_accuracy = record["student_environment_agreement"]
            reward_probability = record["greedy_environment_reward_probability"]
            reward_rate = record["reward_rate"]
        else:
            correlation = record["rho_me"]
            action_accuracy = record["world_model_environment_agreement"]
            reward_probability = action_accuracy**T
            reward_rate = (
                record["cycle_value"]
                * reward_probability
                / record["update_cost"]
            )
        enriched.append(
            {
                **record,
                "learner_environment_correlation": correlation,
                "learner_action_accuracy": action_accuracy,
                "learner_reward_probability": reward_probability,
                "learner_reward_rate": reward_rate,
            }
        )
    return enriched


def main() -> None:
    """Run the four learner/problem-length combinations."""

    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    add_analytical_args(parser)
    # These defaults reproduce the report invocation of
    # code/scripts/compare_policy_regimes.py. In particular, the T=6
    # reinforcement condition below is its greedy/greedy baseline.
    parser.set_defaults(
        schedule="online:250",
        T=6,
        K=5,
        K_mode="constant",
        beta=3.0,
        beta_max=20.0,
        temperature_mode="constant",
        eta_2f=0.0,
        eta_3f=1.0,
        eta_max=1.0,
        delta_on=0.05,
        gh_order=25,
        constant_prefix_time=True,
        output_subdir="report",
    )
    args = parser.parse_args()

    rows = []
    for learner, T in CONDITIONS:
        print(f"running {learner} learner with T={T}")
        if learner == "reinforcement":
            # This is exactly the greedy-training/greedy-evaluation baseline
            # from compare_policy_regimes.py when T=6.
            initial = initial_state(args, rho_me=1.0, rho_sm=args.rho_se0)
            params = make_params(
                args,
                T=T,
                M=T,
                K_mode="constant",
                eta_e=0.0,
                eta_wm=0.0,
                fix_eta_wm_zero=True,
                greedy_training=True,
                greedy_evaluation=True,
            )
            training_signal = "sequence reward"
        else:
            # The world-model vector is the supervised learner: each action
            # receives its environment label, exactly as in WM training.
            initial = initial_state(args, rho_me=0.0, rho_sm=0.0)
            params = make_params(
                args,
                T=T,
                M=T,
                K=1,
                K_mode="constant",
                beta=float("inf"),
                temperature_mode="constant",
                eta_2f=0.0,
                eta_3f=0.0,
                fix_eta_2f_zero=True,
                fix_eta_3f_zero=True,
                eta_wm=args.eta_wm,
                fix_eta_wm_zero=False,
                eta_e=0.0,
                greedy_training=True,
                greedy_evaluation=True,
            )
            training_signal = "per-action labels"

        trajectory = add_learner_metrics(
            run_analytical(initial, params, **analytical_kwargs(args)),
            learner,
            T,
        )
        rows += with_metadata(
            trajectory,
            params,
            source="analytical",
            condition=f"{learner}, T={T}",
            learner=learner,
            T_value=T,
            training_signal=training_signal,
            training_policy=(
                "greedy" if learner == "reinforcement" else "supervised"
            ),
            evaluation_policy="greedy",
        )

    path = destination(args, "supervised_rl_T_comparison")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
