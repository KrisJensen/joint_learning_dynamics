"""Initial-correlation dynamic-programming phase diagram."""

from __future__ import annotations

import argparse
from dataclasses import replace
from itertools import product

import numpy as np

from _common import results_dir, write_records
from joint_dynamics import (
    DPConfig,
    Params,
    State,
    build_switching_action_workspace,
    load_switching_value_map_cache,
    rollout_policy,
    save_switching_value_map_cache,
    solve_switching_workspace,
    value_at,
)


DEFAULT_SWITCHING_COST = 0.1
DEFAULT_T = 6
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--switching-cost",
    type=float,
    default=DEFAULT_SWITCHING_COST,
    help="elapsed-time cost of switching between online and offline states",
)
parser.add_argument("--T", type=int, default=DEFAULT_T, help="executed sequence length")
parser.add_argument("--beta", type=float, default=3.0, help="temperature parameter")
parser.add_argument("--horizon", type=float, default=10.0, help="horizon")
parser.add_argument("--temperature_mode", type=str, default="optimal", help="temperature mode")
parser.add_argument(
    "--suffix",
    default=None,
    help="extra suffix appended after the automatic switch_<cost>_T_<T> suffix",
)
parser.add_argument("--recompute-value-map", action="store_true")
parser.add_argument(
    "--output-subdir",
    default="",
    help="relative subdirectory under results/ for default outputs",
)
args = parser.parse_args()
if args.switching_cost <= 0.0:
    raise ValueError("switching cost must be positive")
if args.T < 1:
    raise ValueError("T must be positive")

switching_slug = (
    f"{args.switching_cost:.6g}"
    .replace("-", "m")
    .replace("+", "")
    .replace(".", "p")
)
suffix_parts = [f"switch_{switching_slug}", f"T_{args.T}"]
if args.suffix is not None and args.suffix.strip().strip("_"):
    suffix_parts.append(args.suffix.strip().strip("_"))
suffix = "_".join(suffix_parts)
output_stem = f"phase_diagrams_initial_{suffix}"
OUTPUT_DIR = results_dir(args.output_subdir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = DPConfig(
    horizon=args.horizon,
    time_step=0.01,
    rho_points=26,
    rho_min=0.0,
    # The DP interpolation grid must be strictly interior; fixed_rho_me itself
    # may still be exactly one in other scripts/configurations.
    rho_max=0.9999,
    K_max=15,
    reward_kind="true",
    fixed_wm=True,
    switching_cost=args.switching_cost,
    interpolation="structured",
)
PARAMS = Params(
    T=args.T,
    K=5,
    K_max=15,
    beta=args.beta,
    temperature_mode=args.temperature_mode,
    learning_rate_mode="optimal",
    eta_wm=0.0,
    fix_eta_wm_zero=True,
    eta_max=20.0,
    tau_on=0.05,
    tau_off=0.05,
)
CSV_PATH = OUTPUT_DIR / f"{output_stem}.csv"
RECOMPUTE_VALUE_MAP = args.recompute_value_map

INITIAL_RHO_SE_POINTS = 50
INITIAL_RHO_SE_MIN = 0.0
INITIAL_RHO_SE_MAX = 0.99
RHO_SE_GRID = np.linspace(
    INITIAL_RHO_SE_MIN,
    INITIAL_RHO_SE_MAX,
    INITIAL_RHO_SE_POINTS,
)
RHO_ME_TABLE_GRID = np.linspace(CONFIG.rho_min, CONFIG.rho_max, CONFIG.rho_points)

records = []
for rho_me0 in RHO_ME_TABLE_GRID:
    rho_me = float(rho_me0)
    rho_me_slug = (
        f"{rho_me:.6g}"
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )
    table_path = OUTPUT_DIR / f"{output_stem}_rho_me_{rho_me_slug}.npz"
    row_config = replace(CONFIG, fixed_rho_me=rho_me)

    if table_path.exists() and not RECOMPUTE_VALUE_MAP:
        print(
            "initial: loading cached 2D value map "
            f"rho_me={rho_me:g} from {table_path}"
        )
        optimal, online_only, k1_online, k1_switching = (
            load_switching_value_map_cache(
                table_path,
                include_k1_switching=True,
            )
        )
        if not np.isclose(optimal.config.fixed_rho_me, rho_me):
            raise ValueError(
                "cached fixed-WM value table has fixed_rho_me="
                f"{optimal.config.fixed_rho_me:g}, expected {rho_me:g}; "
                "rerun with --recompute-value-map"
            )
    else:
        print(f"initial: building 2D action tables for rho_me={rho_me:g}")
        workspace = build_switching_action_workspace(
            PARAMS,
            row_config,
            progress=print,
        )
        print(f"initial: solving full switching policy for rho_me={rho_me:g}")
        optimal = solve_switching_workspace(workspace, progress=print)
        print(f"initial: solving online-only policy for rho_me={rho_me:g}")
        online_only = solve_switching_workspace(
            workspace,
            allow_offline=False,
            allow_switch=False,
            progress=print,
        )
        print(f"initial: solving K=1 online-only policy for rho_me={rho_me:g}")
        k1_online = solve_switching_workspace(
            workspace,
            allowed_online_K=(1,),
            allow_offline=False,
            allow_switch=False,
            progress=print,
        )
        print(f"initial: solving K=1 switching policy for rho_me={rho_me:g}")
        k1_switching = solve_switching_workspace(
            workspace,
            allowed_online_K=(1,),
            progress=print,
        )
        save_switching_value_map_cache(
            table_path,
            optimal,
            online_only,
            k1_online,
            k1_switching,
        )
        print(f"Wrote {table_path}")

    for rho_se0 in RHO_SE_GRID:
        state = State(float(rho_se0), rho_me, float(rho_se0 * rho_me))
        horizon = optimal.config.horizon
        optimal_rate = value_at(optimal, horizon, state, "on") / horizon
        online_rate = value_at(online_only, horizon, state, "on") / horizon
        k1_rate = value_at(k1_online, horizon, state, "on") / horizon
        k1_switching_rate = value_at(k1_switching, horizon, state, "on") / horizon
        relative_online = (
            np.nan
            if abs(online_rate) <= 1e-14
            else (optimal_rate - online_rate) / online_rate
        )
        relative_k1_online = (
            np.nan
            if abs(k1_rate) <= 1e-14
            else (optimal_rate - k1_rate) / k1_rate
        )
        relative_k1_switching = (
            np.nan
            if abs(k1_switching_rate) <= 1e-14
            else (optimal_rate - k1_switching_rate) / k1_switching_rate
        )
        rollout = rollout_policy(
            optimal,
            state,
            rollout_step=optimal.config.time_step,
        )
        decision_rows = [row for row in rollout if row["decision_boundary"]]
        early_decision_rows = [
            row
            for row in decision_rows
            if float(row["time"]) <= 0.95 * horizon + 1e-12
        ]
        uses_offline = any(
            row["mode"] == "off" or row["action"] in {"offline", "switch"}
            for row in early_decision_rows
        )
        all_K_values = [
            int(float(row["K"]))
            for row in decision_rows
            if row.get("K", "") not in ("", None)
        ]
        K_values = [
            int(float(row["K"]))
            for row in early_decision_rows
            if row.get("K", "") not in ("", None)
        ]
        max_K = max(K_values, default=0)
        max_K_all = max(all_K_values, default=0)
        records.append(
            {
                "analysis": "initial_correlations",
                "rho_se0": state.rho_se,
                "rho_me0": state.rho_me,
                "rho_sm0": state.rho_sm,
                "eta_on2_off": "",
                "eta_wm": "",
                "switching_cost": row_config.switching_cost,
                "T": PARAMS.T,
                "fixed_wm": row_config.fixed_wm,
                "optimal_reward_rate": optimal_rate,
                "online_only_reward_rate": online_rate,
                "k1_online_reward_rate": k1_rate,
                "k1_switching_reward_rate": k1_switching_rate,
                "optimal_relative_online_only": relative_online,
                "optimal_relative_k1_online": relative_k1_online,
                "optimal_relative_k1_switching": relative_k1_switching,
                "optimal_minus_online_only": relative_online,
                "optimal_minus_k1_online": relative_k1_online,
                "optimal_minus_k1_switching": relative_k1_switching,
                "category": (
                    f"{'offline' if uses_offline else 'online'}_"
                    f"{'planning' if max_K > 1 else 'K1'}"
                ),
                "uses_offline": uses_offline,
                "max_K": max_K,
                "max_K_all": max_K_all,
                "max_K_window_fraction": 0.95,
                "table_path": str(table_path),
            }
        )

write_records(CSV_PATH, records)
print(f"Wrote {CSV_PATH}")
