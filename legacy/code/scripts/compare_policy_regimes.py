"""Compute and plot all planning/greedy training and evaluation combinations."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

from _common import (
    CODE_ROOT,
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    figure_path as default_figure_path,
    make_params,
    output_path,
    with_metadata,
    write_records,
)
from joint_dynamics import State, run_analytical

if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from plot_functions._common import plot_dashboard  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
add_common_args(parser)
parser.set_defaults(
    schedule="online:150",
    duration=150.0,
    temperature_mode="constant",
    eta_on2=2.5,
    eta_on1=0.0,
    eta_wm=2.5,
    beta=3.0,
    T=6,
    K=5,
    tau_on = 0.05,
)

add_analytical_args(parser)
parser.add_argument("--figure-output", type=Path, default=None)
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()

records = []
scenario_records = {}
scenario_styles = {}
scenarios = (
    (
        "fixed_wm",
        "fixed WM, fixed K",
        State(args.rho_se0, 1.0, args.rho_se0),
        {
            "K_mode": "constant",
            "eta_e": 0.0,
            "eta_wm": 0.0,
            "fix_eta_wm_zero": True,
        },
    ),
    # (
    #     "learned_wm",
    #     "learned WM, reward-optimal K",
    #     State(args.rho_se0, 0.0, args.rho_sm0),
    #     {"K_mode": "reward"},
    # ),
)

for scenario_index, (scenario_slug, scenario, initial, scenario_overrides) in enumerate(
    scenarios
):
    initial.validate()
    scenario_records[scenario_slug] = []
    scenario_styles[scenario_slug] = {}
    for greedy_training, greedy_evaluation in (
        (False, False),
        (False, True),
        (True, False),
        (True, True),
    ):
        training_policy = "greedy" if greedy_training else "planning"
        evaluation_policy = "greedy" if greedy_evaluation else "planning"
        condition = (
            f"{scenario}: {training_policy} training / {evaluation_policy} evaluation"
        )
        print(f"Running {condition}...")
        params = make_params(
            args,
            **scenario_overrides,
            greedy_training=greedy_training,
            greedy_evaluation=greedy_evaluation,
        )
        trajectory = with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=condition,
            scenario=scenario,
            training_policy=training_policy,
            evaluation_policy=evaluation_policy,
        )
        records += trajectory
        scenario_records[scenario_slug] += trajectory
        style = {
            "color": f"C{2 * scenario_index + int(greedy_training)}",
            "linestyle": "-" if greedy_evaluation else "--",
        }
        scenario_styles[scenario_slug][condition] = style

result_path = args.output or output_path("policy_regime_comparison", args.suffix, args.output_subdir)
suffix = args.suffix.strip().strip("_")

write_records(result_path, records)
print(f"Wrote {result_path}")
for scenario_slug, scenario, _, _ in scenarios:
    if args.figure_output is None:
        figure_stem = f"policy_regime_comparison_{scenario_slug}"
        figure_path = default_figure_path(
            figure_stem,
            suffix,
            args.output_subdir,
        )
    else:
        figure_path = args.figure_output.with_name(
            f"{args.figure_output.stem}_{scenario_slug}{args.figure_output.suffix}"
        )
    plot_dashboard(
        scenario_records[scenario_slug],
        figure_path,
        f"Policy Regimes: {scenario}",
        args.dpi,
        condition_styles=scenario_styles[scenario_slug],
    )
    print(f"Wrote {figure_path}")

    if args.figure_output is None:
        reward_figure_path = default_figure_path(
            f"policy_regime_expected_reward_rate_{scenario_slug}",
            suffix,
            args.output_subdir,
        )
    else:
        reward_figure_path = args.figure_output.with_name(
            f"{args.figure_output.stem}_expected_reward_rate_{scenario_slug}"
            f"{args.figure_output.suffix}"
        )
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    for training_policy, evaluation_policy in (
        ("planning", "planning"),
        ("planning", "greedy"),
        ("greedy", "planning"),
        ("greedy", "greedy"),
    ):
        rows = [
            row
            for row in scenario_records[scenario_slug]
            if row.get("training_policy") == training_policy
            and row.get("evaluation_policy") == evaluation_policy
            and row.get("reward_rate", "") != ""
            and math.isfinite(float(row["reward_rate"]))
        ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: float(row["time"]))
        axis.plot(
            [float(row["time"]) for row in rows],
            [float(row["reward_rate"]) for row in rows],
            color="#1f77b4" if training_policy == "planning" else "black",
            linestyle="--" if evaluation_policy == "planning" else "-",
            linewidth=3.5,
            label=(
                f"learn {'with' if training_policy == 'planning' else 'without'} plans, "
                f"evaluate {'with' if evaluation_policy == 'planning' else 'without'} plans"
            ),
        )
    axis.set_xlabel("elapsed time")
    axis.set_ylabel("expected reward rate")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    #axis.legend(frameon=False)
    axis.set_xlim(0,50)
    fig.tight_layout()
    reward_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(reward_figure_path, dpi=args.dpi, transparent=True)
    plt.close(fig)
    print(f"Wrote {reward_figure_path}")

    if args.figure_output is None:
        average_figure_path = default_figure_path(
            f"policy_regime_average_reward_rate_{scenario_slug}",
            suffix,
            args.output_subdir,
        )
    else:
        average_figure_path = args.figure_output.with_name(
            f"{args.figure_output.stem}_average_reward_rate_{scenario_slug}"
            f"{args.figure_output.suffix}"
        )
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    for training_policy, evaluation_policy in (
        ("planning", "planning"),
        ("planning", "greedy"),
        ("greedy", "planning"),
        ("greedy", "greedy"),
    ):
        rows = [
            row
            for row in scenario_records[scenario_slug]
            if row.get("training_policy") == training_policy
            and row.get("evaluation_policy") == evaluation_policy
            and row.get("reward_rate", "") != ""
            and math.isfinite(float(row["reward_rate"]))
        ]
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: float(row["time"]))
        times = [float(row["time"]) for row in rows]
        rates = [float(row["reward_rate"]) for row in rows]
        integrated_reward = [0.0]
        average_rates = [rates[0]]
        for index in range(1, len(times)):
            dt = times[index] - times[index - 1]
            integrated_reward.append(
                integrated_reward[-1] + 0.5 * (rates[index - 1] + rates[index]) * dt
            )
            elapsed = times[index] - times[0]
            average_rates.append(
                integrated_reward[-1] / elapsed if elapsed > 0.0 else rates[index]
            )
        axis.plot(
            times,
            average_rates,
            color="#1f77b4" if training_policy == "planning" else "black",
            linestyle="--" if evaluation_policy == "planning" else "-",
            linewidth=3.5,
            label=(
                f"learn {'with' if training_policy == 'planning' else 'without'} plans, "
                f"evaluate {'with' if evaluation_policy == 'planning' else 'without'} plans"
            ),
        )
    axis.set_xlabel("elapsed time")
    axis.set_ylabel("average reward rate")
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    #axis.legend(frameon=False)
    fig.tight_layout()
    average_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(average_figure_path, dpi=args.dpi, transparent=True)
    plt.close(fig)
    print(f"Wrote {average_figure_path}")
