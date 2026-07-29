"""Compute a DP phase diagram over initial rho_se and rho_me."""

import argparse

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
    rollout_policy,
    save_dp_result,
    solve_online_dp,
    solve_switching_dp,
    value_at,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.add_argument("--rho-se-values", default="0,0.25,0.5,0.75")
    parser.add_argument("--rho-me-values", default="0.25,0.5,0.75,0.99")
    parser.add_argument("--horizon", type=float, default=5.0)
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
    args = parser.parse_args()
    params = make_params(
        args, eta_wm=0.0, fix_eta_wm_zero=True, eta_e=0.0
    )
    rows = []
    for rho_me in parse_list(args.rho_me_values):
        config = DPConfig(
            horizon=args.horizon,
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
        full = solve_switching_dp(params, config, progress=print)
        online = solve_online_dp(params, config, progress=print)
        k1_config = DPConfig(**{**config.__dict__, "K_max": 1})
        k1 = solve_switching_dp(params, k1_config, progress=print)
        slug = str(rho_me).replace(".", "p")
        table_path = output_path(
            f"phase_diagram_initial_rho_me_{slug}",
            args.suffix,
            args.output_subdir,
            ".npz",
        )
        save_dp_result(table_path, full)
        for rho_se in parse_list(args.rho_se_values):
            state = State(rho_se, rho_me, rho_se * rho_me)
            full_value = value_at(full, config.horizon, state, "on")
            online_value = value_at(online, config.horizon, state, "on")
            k1_value = value_at(k1, config.horizon, state, "on")
            rollout = rollout_policy(full, state)
            uses_offline = any(
                row.get("mode") == "off" or row.get("action") == "switch"
                for row in rollout
            )
            max_K = max(
                [
                    int(row["K"])
                    for row in rollout
                    if row.get("K", "") not in {"", None}
                ],
                default=0,
            )
            rows.append(
                {
                    "analysis": "initial_correlations",
                    "rho_se0": rho_se,
                    "rho_me0": rho_me,
                    "rho_sm0": rho_se * rho_me,
                    "optimal_reward_rate": full_value / config.horizon,
                    "online_only_reward_rate": online_value / config.horizon,
                    "k1_switching_reward_rate": k1_value / config.horizon,
                    "uses_offline": uses_offline,
                    "max_K": max_K,
                    "category": f"{'offline' if uses_offline else 'online'}_{'planning' if max_K > 1 else 'K1'}",
                    "table_path": str(table_path),
                }
            )
    path = destination(args, "phase_diagram_initial")
    write_records(path, rows)
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
