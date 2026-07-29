"""Solve and roll out one finite-horizon dynamic program."""

import argparse
from pathlib import Path

from _common import (
    add_common_args,
    destination,
    initial_state,
    make_params,
    output_path,
    results_dir,
    write_records,
)
from learning_dynamics import (
    DPConfig,
    rollout_policy,
    save_dp_result,
    solve_online_dp,
    solve_switching_dp,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--variant", choices=["online", "switching"], default="online")
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--dp-step-size", type=float, default=1.0)
    parser.add_argument("--rho-points", type=int, default=30)
    parser.add_argument("--rho-min", type=float, default=0.0)
    parser.add_argument("--rho-max", type=float, default=0.999)
    parser.add_argument("--dp-K-max", type=int, default=5)
    parser.add_argument(
        "--dp-M",
        type=int,
        default=None,
        help="fix M while optimizing K; by default the DP optimizes both K and M",
    )
    parser.add_argument("--reward-kind", choices=["true", "observed"], default="true")
    modulation = parser.add_mutually_exclusive_group()
    modulation.add_argument(
        "--no-modulation",
        dest="no_modulation",
        action="store_true",
        help="fix C_A=0 and cache time-independent action interpolations",
    )
    modulation.add_argument(
        "--with-modulation",
        dest="no_modulation",
        action="store_false",
        help="use the configured cycle modulation",
    )
    parser.set_defaults(no_modulation=True)
    parser.add_argument("--fixed-wm", action="store_true")
    parser.add_argument("--fixed-rho-me", type=float, default=1.0)
    parser.add_argument("--initial-mode", choices=["on", "off"], default="on")
    parser.add_argument("--table-output", type=Path, default=None)
    args = parser.parse_args()
    params = make_params(args, C_A=0.0 if args.no_modulation else args.C_A)
    config = DPConfig(
        horizon=args.horizon,
        step_size=args.dp_step_size,
        rho_points=args.rho_points,
        rho_min=args.rho_min,
        rho_max=args.rho_max,
        K_max=args.dp_K_max,
        M=args.dp_M,
        reward_kind=args.reward_kind,
        no_modulation=args.no_modulation,
        fixed_wm=args.fixed_wm,
        fixed_rho_me=args.fixed_rho_me,
    )
    solver = solve_online_dp if args.variant == "online" else solve_switching_dp
    result = solver(params, config, progress=print)
    initial = (
        initial_state(args, rho_me=args.fixed_rho_me)
        if args.fixed_wm
        else initial_state(args)
    )
    rows = rollout_policy(result, initial, initial_mode=args.initial_mode)
    stem = f"dynamic_programming_{args.variant}"
    if args.fixed_wm:
        stem += "_fixed_wm"
    csv_path = destination(args, stem)
    table_path = output_path(stem, args.suffix, args.output_subdir, ".npz")
    if args.table_output is not None:
        table_path = args.table_output.resolve()
        if results_dir().resolve() not in table_path.parents:
            parser.error("--table-output must remain under results/")
    write_records(csv_path, rows)
    save_dp_result(table_path, result)
    print(f"Wrote {csv_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()
