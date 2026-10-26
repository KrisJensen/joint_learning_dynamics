"""Compute reward-only counterfactual K along joint-DP report rollouts.

One online Bellman problem is solved in which planning changes immediate reward
and elapsed cost but every action has the same greedy-training transition. Its
policy is queried at every state and remaining horizon in both the online and
switching rollouts. The Bellman result and small trajectory tables are cached
under the selected results subdirectory.
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
from pathlib import Path

from _common import output_path, write_records
from learning_dynamics import (
    State,
    load_dp_result,
    optimal_action_at,
    save_dp_result,
    solve_reward_only_planning_dp,
)


def read_records(path: Path) -> list[dict[str, str]]:
    """Read one saved rollout CSV."""

    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_counterfactual(
    path: Path,
    rows: list[dict[str, str]],
    result,
) -> None:
    """Evaluate and save the reward-only first action at every rollout row."""

    output_rows = []
    total = len(rows)
    for index, row in enumerate(rows):
        horizon = float(row["remaining_horizon"])
        if horizon > result.config.horizon:
            if not math.isclose(
                horizon,
                result.config.horizon,
                abs_tol=1e-8,
                rel_tol=0.0,
            ):
                raise ValueError(
                    f"rollout horizon {horizon:g} exceeds the counterfactual "
                    f"DP horizon {result.config.horizon:g}"
                )
            horizon = result.config.horizon
        action = None
        if horizon > 0.0:
            state = State(
                float(row["rho_se"]),
                float(row["rho_me"]),
                float(row["rho_sm"]),
            )
            action = optimal_action_at(result, state, horizon, mode="on")
        output_rows.append(
            {
                "time": row["time"],
                "remaining_horizon": row["remaining_horizon"],
                "rho_se": row["rho_se"],
                "rho_me": row["rho_me"],
                "rho_sm": row["rho_sm"],
                "K": "" if action is None else action.K,
                "M": "" if action is None else action.M,
                "action": "stop" if action is None else action.name,
            }
        )
        if (index + 1) % max(1, total // 10) == 0 or index + 1 == total:
            print(f"Counterfactual states {index + 1}/{total}: {path.stem}")

    write_records(
        path,
        output_rows,
        metadata={
            "counterfactual": "planning_affects_reward_not_learning",
            "initial_mode": "on",
        },
    )
    print(f"Wrote {path}")


def main() -> None:
    """Solve or load the counterfactual DP and evaluate report trajectories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-subdir", default="report")
    parser.add_argument("--suffix", default="joint")
    parser.add_argument(
        "--variants",
        default="online,switching",
        help="comma-separated report rollouts to evaluate",
    )
    parser.add_argument(
        "--force-solve",
        action="store_true",
        help="recompute the reward-only Bellman table even if its cache is current",
    )
    args = parser.parse_args()

    variants = tuple(part.strip() for part in args.variants.split(",") if part.strip())
    invalid = sorted(set(variants).difference({"online", "switching"}))
    if not variants or invalid:
        parser.error(f"invalid --variants value: {','.join(invalid) or 'empty'}")

    reference_path = output_path(
        "dynamic_programming_online",
        args.suffix,
        args.output_subdir,
        ".npz",
    )
    counterfactual_path = reference_path.with_name(
        f"{reference_path.stem}_reward_only.npz"
    )
    cache_is_current = (
        counterfactual_path.exists()
        and counterfactual_path.stat().st_mtime >= reference_path.stat().st_mtime
    )
    if args.force_solve or not cache_is_current:
        print(f"Loading reference DP {reference_path}")
        reference = load_dp_result(reference_path)
        result = solve_reward_only_planning_dp(reference, progress=print)
        del reference
        gc.collect()
        save_dp_result(counterfactual_path, result)
        print(f"Wrote {counterfactual_path}")
    else:
        print(f"Loading cached reward-only DP {counterfactual_path}")
        result = load_dp_result(counterfactual_path)

    for variant in variants:
        rollout_path = output_path(
            f"dynamic_programming_{variant}",
            args.suffix,
            args.output_subdir,
        )
        result_path = rollout_path.with_name(
            f"{rollout_path.stem}_reward_only_K.csv"
        )
        write_counterfactual(result_path, read_records(rollout_path), result)


if __name__ == "__main__":
    main()
