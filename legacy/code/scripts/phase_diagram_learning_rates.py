"""Fixed-learning-rate dynamic-programming phase diagram."""

from __future__ import annotations

import argparse
from itertools import product
import shlex

import numpy as np

from _common import REPO_ROOT, results_dir, write_records
from submit_slurm import submit_slurm
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


DEFAULT_SWITCHING_COST = 0.5
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
parser.add_argument(
    "--temperature-mode",
    choices=["constant", "optimal"],
    default="optimal",
    help="temperature schedule used for every value map",
)
parser.add_argument(
    "--suffix",
    default=None,
    help="extra suffix appended after the automatic switch_<cost>_T_<T> suffix",
)
parser.add_argument("--recompute-value-map", action="store_true")
parser.add_argument(
    "--run-single",
    action="store_true",
    help="compute exactly one eta_on2/eta_off and eta_wm pair; used by Slurm jobs",
)
parser.add_argument("--eta-on2-off", type=float, default=None)
parser.add_argument("--eta-wm", type=float, default=None)
parser.add_argument("--submit-time", default="2:00:00")
parser.add_argument("--submit-mem", default="16G")
parser.add_argument("--submit-cpus", type=int, default=1)
parser.add_argument("--submit-partition", default="cpu")
parser.add_argument(
    "--dry-run",
    action="store_true",
    help="print Slurm commands without submitting them",
)
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
if args.run_single and (args.eta_on2_off is None or args.eta_wm is None):
    raise ValueError("--run-single requires --eta-on2-off and --eta-wm")

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
output_stem = f"phase_diagrams_learning_rates_{suffix}"
OUTPUT_DIR = results_dir(args.output_subdir)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RUN_DIR = OUTPUT_DIR / output_stem
RUN_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = DPConfig(
    horizon=30.0,
    time_step=0.01,
    rho_points=30,
    rho_min=0.0,
    rho_max=0.999,
    K_max=15,
    reward_kind="true",
    switching_cost=args.switching_cost,
    interpolation="structured",
)
RATE_VALUES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
ETA_WM_VALUES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
RATE_VALUES = [1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0,7.0]
ETA_WM_VALUES = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.8,1.0]
               
# RATE_VALUES = (0.6, 1.2)
# ETA_WM_VALUES = (0.6, 1.2)
INITIAL_STATE = State(0.0, 0.0, 0.0)
CSV_PATH = RUN_DIR / f"{output_stem}.csv"
RECOMPUTE_VALUE_MAP = args.recompute_value_map

if not args.run_single:
    script_path = REPO_ROOT / "code" / "scripts" / "phase_diagram_learning_rates.py"
    submitted = 0
    for eta_on2_off, eta_wm in product(RATE_VALUES, ETA_WM_VALUES):
        eta_slug = (
            f"{eta_on2_off:.6g}"
            .replace("-", "m")
            .replace("+", "")
            .replace(".", "p")
        )
        wm_slug = (
            f"{eta_wm:.6g}"
            .replace("-", "m")
            .replace("+", "")
            .replace(".", "p")
        )
        command_parts = [
            "python",
            str(script_path),
            "--run-single",
            "--switching-cost",
            str(args.switching_cost),
            "--T",
            str(args.T),
            "--beta",
            str(args.beta),
            "--temperature-mode",
            args.temperature_mode,
            "--eta-on2-off",
            str(eta_on2_off),
            "--eta-wm",
            str(eta_wm),
        ]
        if args.suffix is not None and args.suffix.strip().strip("_"):
            command_parts.extend(["--suffix", args.suffix.strip().strip("_")])
        if args.output_subdir.strip().strip("/"):
            command_parts.extend(["--output-subdir", args.output_subdir])
        if args.recompute_value_map:
            command_parts.append("--recompute-value-map")
        command = " ".join(shlex.quote(part) for part in command_parts)
        jobname = f"phase_lr_{suffix}_eta_{eta_slug}_wm_{wm_slug}"
        if args.dry_run:
            print(f"{jobname}: {command}")
        else:
            submit_slurm(
                command,
                jobname,
                time=args.submit_time,
                mem=args.submit_mem,
                cpus_per_task=args.submit_cpus,
                partition=args.submit_partition,
            )
            print(f"Submitted {jobname}")
        submitted += 1
    action = "Prepared" if args.dry_run else "Submitted"
    print(f"{action} {submitted} learning-rate phase-diagram jobs.")
    raise SystemExit

records = []
for eta_on2_off, eta_wm in [(args.eta_on2_off, args.eta_wm)]:
    eta_slug = (
        f"{eta_on2_off:.6g}"
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )
    wm_slug = (
        f"{eta_wm:.6g}"
        .replace("-", "m")
        .replace("+", "")
        .replace(".", "p")
    )
    pair_stem = f"{output_stem}_eta_{eta_slug}_wm_{wm_slug}"
    pair_csv = RUN_DIR / f"{pair_stem}.csv"
    table_path = RUN_DIR / f"{pair_stem}.npz"

    params = Params(
        T=args.T,
        K=5,
        K_max=15,
        beta=args.beta,
        temperature_mode=args.temperature_mode,
        learning_rate_mode="constant",
        eta_on1=0.0,
        fix_eta_on1_zero=True,
        eta_on2=eta_on2_off,
        eta_off=eta_on2_off,
        eta_wm=eta_wm,
        eta_max=20.0,
        tau_on=0.05,
        tau_off=0.05,
    )
    label = f"learning eta={eta_on2_off:g}, wm={eta_wm:g}"

    if table_path.exists() and not RECOMPUTE_VALUE_MAP:
        print(f"{label}: loading cached value map from {table_path}")
        optimal, online_only, k1_online, k1_switching = load_switching_value_map_cache(
            table_path,
            rebuild_missing_tables=True,
            include_k1_switching=True,
        )
        if k1_switching is None:
            print(f"{label}: cached value map lacks K=1 switching policy; solving it")
            optimal, online_only, k1_online, k1_switching = (
                load_switching_value_map_cache(
                    table_path,
                    rebuild_missing_tables=True,
                    solve_missing_k1_switching=True,
                    include_k1_switching=True,
                    progress=print,
                )
            )
            save_switching_value_map_cache(
                table_path,
                optimal,
                online_only,
                k1_online,
                k1_switching,
            )
            print(f"Wrote {table_path}")
    else:
        print(f"{label}: building action tables")
        workspace = build_switching_action_workspace(params, CONFIG, progress=print)
        print(f"{label}: solving full switching policy")
        optimal = solve_switching_workspace(workspace, progress=print)
        print(f"{label}: solving online-only policy")
        online_only = solve_switching_workspace(
            workspace,
            allow_offline=False,
            allow_switch=False,
            progress=print,
        )
        print(f"{label}: solving K=1 online-only policy")
        k1_online = solve_switching_workspace(
            workspace,
            allowed_online_K=(1,),
            allow_offline=False,
            allow_switch=False,
            progress=print,
        )
        print(f"{label}: solving K=1 switching policy")
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

    horizon = optimal.config.horizon
    optimal_rate = value_at(optimal, horizon, INITIAL_STATE, "on") / horizon
    online_rate = value_at(online_only, horizon, INITIAL_STATE, "on") / horizon
    k1_rate = value_at(k1_online, horizon, INITIAL_STATE, "on") / horizon
    k1_switching_rate = value_at(k1_switching, horizon, INITIAL_STATE, "on") / horizon
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
        INITIAL_STATE,
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
    record = {
        "analysis": "learning_rates",
        "rho_se0": INITIAL_STATE.rho_se,
        "rho_me0": INITIAL_STATE.rho_me,
        "rho_sm0": INITIAL_STATE.rho_sm,
        "eta_on2_off": eta_on2_off,
        "eta_wm": eta_wm,
        "switching_cost": optimal.config.switching_cost,
        "T": optimal.params.T,
        "fixed_wm": optimal.config.fixed_wm,
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
    write_records(pair_csv, [record])
    records.append(record)
    print(f"Wrote {pair_csv}")

if records and not args.run_single:
    write_records(CSV_PATH, records)
    print(f"Wrote {CSV_PATH}")
