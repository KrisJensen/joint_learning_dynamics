"""Solve and roll out one finite-horizon dynamic program."""

import argparse
import gc
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
    build_dp_workspace,
    rollout_policy,
    save_dp_result,
    solve_dp_workspace,
    solve_online_dp,
    solve_switching_dp,
)


def write_result(
    args,
    result,
    initial,
    variant: str,
    suffix: str,
    *,
    allow_explicit_paths: bool,
) -> None:
    """Write one rollout and Bellman table using the standard DP filenames."""

    stem = f"dynamic_programming_{variant}"
    if args.fixed_wm:
        stem += "_fixed_wm"
    if allow_explicit_paths:
        csv_path = destination(args, stem)
        table_path = output_path(stem, suffix, args.output_subdir, ".npz")
        if args.table_output is not None:
            table_path = args.table_output.resolve()
            if results_dir().resolve() not in table_path.parents:
                raise ValueError("--table-output must remain under results/")
    else:
        csv_path = output_path(stem, suffix, args.output_subdir)
        table_path = output_path(stem, suffix, args.output_subdir, ".npz")

    # Preserve the expensive Bellman solution before doing the comparatively
    # cheap rollout, so a rollout or CSV failure does not discard the table.
    save_dp_result(table_path, result)
    print(f"Wrote {table_path}")
    rows = rollout_policy(result, initial, initial_mode=args.initial_mode)
    write_records(
        csv_path,
        rows,
        metadata={
            "C_A": result.params.C_A,
            "C_T": result.params.C_T,
            "C_phi": result.params.C_phi,
        },
    )
    print(f"Wrote {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument(
        "--variant",
        choices=["online", "switching", "both"],
        default="online",
    )
    parser.add_argument(
        "--online-suffix",
        default=None,
        help="output suffix for the online result when --variant=both",
    )
    parser.add_argument(
        "--switching-suffix",
        default=None,
        help="output suffix for the switching result when --variant=both",
    )
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument("--dp-step-size", type=float, default=1.0)
    parser.add_argument(
        "--t-switch",
        type=float,
        default=0.1,
        help="fixed macroscopic elapsed time required to switch modes",
    )
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
    if args.variant == "both" and (
        args.output is not None or args.table_output is not None
    ):
        parser.error("--output and --table-output cannot be used with --variant=both")
    if args.variant == "both" and args.initial_mode != "on":
        parser.error("--variant=both requires --initial-mode=on")
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
        t_switch=args.t_switch,
        no_modulation=args.no_modulation,
        fixed_wm=args.fixed_wm,
        fixed_rho_me=args.fixed_rho_me,
    )
    initial = (
        initial_state(args, rho_me=args.fixed_rho_me)
        if args.fixed_wm
        else initial_state(args)
    )
    if args.variant == "both":
        print("Building shared online/offline action workspace")
        workspace = build_dp_workspace(
            params,
            config,
            include_offline=True,
            progress=print,
        )
        online_suffix = (
            args.suffix if args.online_suffix is None else args.online_suffix
        )
        switching_suffix = (
            args.suffix
            if args.switching_suffix is None
            else args.switching_suffix
        )
        print("Solving online-only DP from shared workspace")
        online = solve_dp_workspace(
            workspace,
            variant="online",
            progress=print,
        )
        write_result(
            args,
            online,
            initial,
            "online",
            online_suffix,
            allow_explicit_paths=False,
        )
        del online
        gc.collect()

        print("Solving switching DP from shared workspace")
        switching = solve_dp_workspace(
            workspace,
            variant="switching",
            progress=print,
        )
        write_result(
            args,
            switching,
            initial,
            "switching",
            switching_suffix,
            allow_explicit_paths=False,
        )
        return

    solver = solve_online_dp if args.variant == "online" else solve_switching_dp
    result = solver(params, config, progress=print)
    write_result(
        args,
        result,
        initial,
        args.variant,
        args.suffix,
        allow_explicit_paths=True,
    )


if __name__ == "__main__":
    main()
