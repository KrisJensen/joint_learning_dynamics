"""Compute a DP phase diagram over initial rho_se and rho_me."""

import argparse
from dataclasses import replace

from _common import (
    add_common_args,
    destination,
    make_params,
    output_path,
    parse_list,
    write_records,
)
from learning_dynamics import (
    DPConfig,
    State,
    build_dp_workspace,
    rollout_policies,
    save_dp_result,
    solve_dp_workspace,
    value_at,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--rho-se-values", default="0,0.25,0.5,0.75")
    parser.add_argument("--rho-me-values", default="0.25,0.5,0.75,0.99")
    parser.add_argument("--horizon", type=float, default=5.0)
    parser.add_argument(
        "--initial-horizons",
        default="",
        help="comma-separated horizons; defaults to --horizon",
    )
    parser.add_argument(
        "--switch-costs",
        default="",
        help="comma-separated delta-switch values; defaults to --delta-switch",
    )
    parser.add_argument("--dp-step-size", type=float, default=0.1)
    parser.add_argument("--rho-points", type=int, default=31)
    parser.add_argument("--dp-K-max", type=int, default=10)
    parser.add_argument(
        "--dp-M",
        type=int,
        default=None,
        help="fix M while optimizing K; by default the DP optimizes both K and M",
    )
    parser.add_argument("--reward-kind", choices=["true", "observed"], default="true")
    parser.add_argument(
        "--rollout-outcome-method",
        choices=["auto", "table", "analytical"],
        default="auto",
        help=(
            "continuous rollout dynamics; auto uses cached tables without "
            "modulation and analytical moments otherwise"
        ),
    )
    args = parser.parse_args()
    params = make_params(
        args, eta_wm=0.0, fix_eta_wm_zero=True, eta_e=0.0
    )
    horizons = sorted(
        set(
            parse_list(args.initial_horizons)
            if args.initial_horizons.strip()
            else [args.horizon]
        )
    )
    switch_costs = sorted(
        set(
            parse_list(args.switch_costs)
            if args.switch_costs.strip()
            else [params.delta_switch]
        )
    )
    if any(horizon <= 0.0 for horizon in horizons):
        parser.error("initial horizons must be positive")
    if any(cost <= 0.0 for cost in switch_costs):
        parser.error("switch costs must be positive")
    params = replace(params, delta_switch=switch_costs[0])
    maximum_horizon = max(horizons)
    rho_se_values = parse_list(args.rho_se_values)
    rho_me_values = parse_list(args.rho_me_values)
    path = destination(args, "phase_diagram_initial")
    rows = []
    for rho_me in rho_me_values:
        config = DPConfig(
            horizon=maximum_horizon,
            step_size=args.dp_step_size,
            rho_points=args.rho_points,
            rho_min=0.0,
            rho_max=0.999,
            K_max=args.dp_K_max,
            M=args.dp_M,
            reward_kind=args.reward_kind,
            fixed_wm=True,
            fixed_rho_me=rho_me,
        )
        print(f"Building action workspace for rho_me={rho_me:.6g}")
        workspace = build_dp_workspace(
            params,
            config,
            include_offline=True,
            progress=print,
        )
        print(f"Solving online-only DP through horizon {maximum_horizon:g}")
        online = solve_dp_workspace(
            workspace,
            variant="online",
            horizon=maximum_horizon,
            progress=print,
        )
        for switch_cost in switch_costs:
            print(
                "Solving full switching DP for "
                f"rho_me={rho_me:.6g}, delta_switch={switch_cost:g}"
            )
            full = solve_dp_workspace(
                workspace,
                variant="switching",
                horizon=maximum_horizon,
                delta_switch=switch_cost,
                progress=print,
            )
            print(
                "Solving K=1 switching DP for "
                f"rho_me={rho_me:.6g}, delta_switch={switch_cost:g}"
            )
            k1 = solve_dp_workspace(
                workspace,
                variant="switching",
                horizon=maximum_horizon,
                delta_switch=switch_cost,
                allowed_K=(1,),
                progress=print,
            )
            rho_slug = f"{rho_me:.12g}".replace("-", "m").replace(".", "p")
            switch_slug = (
                f"{switch_cost:.12g}".replace("-", "m").replace(".", "p")
            )
            table_path = output_path(
                (
                    f"phase_diagram_initial_rho_me_{rho_slug}"
                    f"_switch_{switch_slug}"
                ),
                args.suffix,
                args.output_subdir,
                ".npz",
            )
            save_dp_result(table_path, full)
            effective_rollout_method = args.rollout_outcome_method
            if effective_rollout_method == "auto":
                effective_rollout_method = (
                    "table" if full.params.C_A == 0.0 else "analytical"
                )
            for horizon in horizons:
                states = [
                    State(rho_se, rho_me, rho_se * rho_me)
                    for rho_se in rho_se_values
                ]
                rollouts = rollout_policies(
                    full,
                    states,
                    horizon=horizon,
                    outcome_method=args.rollout_outcome_method,
                )
                for rho_se, state, rollout in zip(
                    rho_se_values,
                    states,
                    rollouts,
                ):
                    full_value = value_at(full, horizon, state, "on")
                    online_value = value_at(online, horizon, state, "on")
                    k1_value = value_at(k1, horizon, state, "on")
                    uses_offline = any(
                        item.get("mode") == "off"
                        or item.get("action") == "switch"
                        for item in rollout
                    )
                    max_K = max(
                        [
                            int(item["K"])
                            for item in rollout
                            if item.get("K", "") not in {"", None}
                        ],
                        default=0,
                    )
                    rows.append(
                        {
                            "analysis": "initial_correlations",
                            "rollout_outcome_method": effective_rollout_method,
                            "initial_horizon": horizon,
                            "delta_switch": switch_cost,
                            "rho_se0": rho_se,
                            "rho_me0": rho_me,
                            "rho_sm0": rho_se * rho_me,
                            "optimal_reward_rate": full_value / horizon,
                            "online_only_reward_rate": online_value / horizon,
                            "k1_switching_reward_rate": k1_value / horizon,
                            "uses_offline": uses_offline,
                            "max_K": max_K,
                            "category": (
                                f"{'offline' if uses_offline else 'online'}_"
                                f"{'planning' if max_K > 1 else 'K1'}"
                            ),
                            "table_path": str(table_path),
                        }
                    )
            # Long multi-setting runs retain all completed settings if stopped.
            write_records(path, rows)
            print(f"Checkpointed {path}")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
