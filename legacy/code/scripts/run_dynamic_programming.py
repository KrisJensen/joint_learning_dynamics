"""Solve and roll out one finite-horizon dynamic-programming variant."""

from __future__ import annotations

import argparse
from pathlib import Path

from _common import (
    add_common_args,
    make_params,
    output_path,
    results_dir,
    write_records,
)
from joint_dynamics import (
    DPConfig,
    State,
    rollout_policy,
    save_dp_result,
    solve_online_dp,
    solve_switching_dp,
)


parser = argparse.ArgumentParser(description=__doc__)
add_common_args(parser)
parser.set_defaults(schedule="", temperature_mode="constant", tau_on = 0.1, tau_off = 0.2, eta_on1 = 0.0, eta_on2 = 2.5, eta_off = 2.5, eta_wm = 1.5, T = 5,
                    beta=3.0, p_flip = 0.1)
parser.add_argument(
    "--variant",
    choices=["online", "switching"],
    default="online",
)
parser.add_argument("--horizon", type=float, default=20.0)
parser.add_argument(
    "--dp-step",
    type=float,
    default=0.01,
    help=(
        "Bellman elapsed-time grid step. With partial evaluation this is the "
        "elapsed time of one evaluated prefix element; otherwise it is the "
        "elapsed time of one full-sequence mental simulation."
    ),
)
parser.add_argument(
    "--rollout-step",
    type=float,
    default=None,
    help="output spacing; defaults to dp-step and must be its integer multiple",
)
parser.add_argument("--rho-points", type=int, default=31)
parser.add_argument("--rho-min", type=float, default=0.0)
parser.add_argument("--rho-max", type=float, default=0.999)
parser.add_argument("--dp-K-max", type=int, default=15)
parser.add_argument("--reward-kind", choices=["observed", "true"], default="true")
parser.add_argument("--fixed-rho-me", type=float, default=1.0)
parser.add_argument(
    "--fixed-wm",
    action="store_true",
    help="use a fixed world model: keep rho_me fixed and solve on (rho_se, rho_sm)",
)
parser.add_argument("--switching-cost", type=float, default=0.5)
parser.add_argument(
    "--interpolation",
    choices=["delaunay", "structured"],
    default="structured",
    help="state-value interpolation used by the Bellman solver",
)
parser.add_argument("--initial-mode", choices=["on", "off"], default="on")
parser.add_argument("--table-output", type=Path, default=None)
args = parser.parse_args()

params = make_params(
    args,
    greedy_training=False,
    greedy_evaluation=False,
)
config = DPConfig(
    horizon=args.horizon,
    time_step=args.dp_step,
    rho_points=args.rho_points,
    rho_min=args.rho_min,
    rho_max=args.rho_max,
    K_max=args.dp_K_max,
    reward_kind=args.reward_kind,
    fixed_rho_me=args.fixed_rho_me,
    fixed_wm=args.fixed_wm,
    switching_cost=args.switching_cost,
    interpolation=args.interpolation,
)

description = (
    f"{args.variant}, fixed_wm={config.fixed_wm}, "
    f"partial_evaluation={params.partial_evaluation}"
)
print(f"Solving {description} dynamic program...")
if args.variant == "online":
    result = solve_online_dp(params, config, progress=print)
else:
    result = solve_switching_dp(params, config, progress=print)

if config.fixed_wm:
    initial = State(args.rho_se0, args.fixed_rho_me, args.rho_sm0)
else:
    initial = State(args.rho_se0, args.rho_me0, args.rho_sm0)

rollout = rollout_policy(
    result,
    initial,
    initial_mode=args.initial_mode,
    rollout_step=args.rollout_step,
)
stem = f"dynamic_programming_{args.variant}"
if config.fixed_wm:
    stem += "_fixed_wm"
if params.partial_evaluation:
    stem += "_partial"
rollout_path = args.output or output_path(stem, args.suffix, args.output_subdir)
suffix = args.suffix.strip().strip("_")
table_stem = f"{stem}_{suffix}" if suffix else stem
table_path = args.table_output or results_dir(args.output_subdir) / f"{table_stem}.npz"
write_records(rollout_path, rollout)
save_dp_result(table_path, result)
print(f"Wrote {rollout_path}")
print(f"Wrote {table_path}")
