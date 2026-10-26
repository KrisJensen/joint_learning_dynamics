"""Shared command-line and CSV helpers."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
SRC = CODE_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from joint_dynamics import Params, Phase, Schedule, State  # noqa: E402


def parse_schedule(text: str) -> Schedule:
    phases = []
    for item in text.split(","):
        mode, duration = item.strip().split(":", 1)
        phases.append(Phase(mode.strip(), float(duration)))
    return Schedule(phases)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schedule", default="online:20", help="e.g. online:2,offline:2")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rho-se0", type=float, default=0.0)
    parser.add_argument("--rho-me0", type=float, default=0.0)
    parser.add_argument("--rho-sm0", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=6)
    parser.add_argument(
        "--prefix-length",
        type=int,
        default=None,
        help="number of executed actions evaluated by the WM; defaults to T",
    )
    parser.add_argument(
        "--prefix-length-mode",
        choices=["constant", "optimal"],
        default="constant",
        help="use fixed prefix length, or optimize it for the eta_on1 signal",
    )
    parser.add_argument(
        "--prefix-length-max",
        type=int,
        default=None,
        help="largest prefix length considered when optimizing prefix length",
    )
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument(
        "--K-mode",
        choices=["constant", "reward", "drift"],
        default="constant",
        help="use fixed K, or optimize K for instantaneous reward/drift rate",
    )
    parser.add_argument("--K-max", type=int, default=20, help="largest K considered when optimizing K")
    parser.add_argument("--beta", type=float, default=3.0)
    parser.add_argument("--temperature-mode", choices=["constant", "optimal"], default="constant")
    parser.add_argument(
        "--greedy-training",
        action="store_true",
        help="train with K=1 and beta=infinity while retaining planning parameters for evaluation",
    )
    parser.add_argument(
        "--greedy-evaluation",
        action="store_true",
        help="evaluate rewards with K=1 and beta=infinity without changing training",
    )
    #parser.add_argument("--lambda-s", type=float, default=0.25)
    #parser.add_argument("--lambda-m", type=float, default=0.25)
    parser.add_argument("--lambda-s", type=float, default=0.0)
    parser.add_argument("--lambda-m", type=float, default=0.0)
    parser.add_argument("--eta-on1", type=float, default=2.5)
    parser.add_argument("--eta-on2", type=float, default=0.0)
    parser.add_argument("--eta-off", type=float, default=2.5)
    parser.add_argument("--eta-wm", type=float, default=2.5)
    parser.add_argument("--learning-rate-mode", choices=["constant", "optimal"], default="constant")
    parser.add_argument("--fix-eta-on1-zero", action="store_true")
    parser.add_argument("--fix-eta-on2-zero", action="store_true")
    parser.add_argument("--fix-eta-wm-zero", action="store_true")
    parser.add_argument("--eta-max", type=float, default=20.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--p-flip", type=float, default=0.0)
    parser.add_argument("--eta-e", type=float, default=0.0) # environment drift rate
    parser.add_argument("--M", type=float, default=0.0, help="environment-modulation amplitude")
    parser.add_argument("--T-M", type=float, default=1.0, help="environment-modulation period")
    parser.add_argument("--phi-M", type=float, default=0.0, help="environment-modulation phase in cycles")
    parser.add_argument(
        "--partial-evaluation",
        action="store_true",
        help="allow online DP to choose K repetitions of length-M partial evaluations",
    )
    parser.add_argument(
        "--tau-element",
        type=float,
        default=None,
        help="thinking time per simulated sequence element for partial evaluation; defaults to tau_on/T",
    )
    parser.add_argument("--tau-on", type=float, default=0.1) # online thinking time
    parser.add_argument("--tau-off", type=float, default=0.1) # offline replay time
    parser.add_argument("--gh-order", type=int, default=25)
    parser.add_argument("--beta-max", type=float, default=20.0)
    parser.add_argument(
        "--suffix",
        default="",
        help="suffix appended to the default result filename",
    )
    parser.add_argument(
        "--output-subdir",
        default="",
        help="relative subdirectory under results/ and figures/ for default outputs",
    )
    parser.add_argument("--output", type=Path, default=None)


def add_analytical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--method", default="RK45")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)


def add_empirical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dimensions",
        default="300,1200,5000",
        help="comma-separated empirical dimensions",
    )
    parser.add_argument(
        "--d",
        type=int,
        default=None,
        help="run one dimension instead of --dimensions",
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=None,
        help="updates between records; defaults to approximately one analytical step",
    )
    parser.add_argument("--seed", type=int, default=0)


def empirical_dimensions(args) -> list[int]:
    dimensions = [args.d] if args.d is not None else [
        int(value.strip()) for value in args.dimensions.split(",") if value.strip()
    ]
    if not dimensions or any(dimension < 3 for dimension in dimensions):
        raise ValueError("empirical dimensions must contain integers of at least three")
    return dimensions


def initial_state(args) -> State:
    return State(args.rho_se0, args.rho_me0, args.rho_sm0)


def make_params(args, **overrides) -> Params:
    values = {
        "T": args.T,
        "prefix_length": args.prefix_length,
        "prefix_length_mode": args.prefix_length_mode,
        "prefix_length_max": args.prefix_length_max,
        "K": args.K,
        "K_mode": args.K_mode,
        "K_max": args.K_max,
        "beta": args.beta,
        "temperature_mode": args.temperature_mode,
        "greedy_training": args.greedy_training,
        "greedy_evaluation": args.greedy_evaluation,
        "lambda_s": args.lambda_s,
        "lambda_m": args.lambda_m,
        "eta_on1": args.eta_on1,
        "eta_on2": args.eta_on2,
        "eta_off": args.eta_off,
        "eta_wm": args.eta_wm,
        "learning_rate_mode": args.learning_rate_mode,
        "fix_eta_on1_zero": args.fix_eta_on1_zero,
        "fix_eta_on2_zero": args.fix_eta_on2_zero,
        "fix_eta_wm_zero": args.fix_eta_wm_zero,
        "eta_max": args.eta_max,
        "epsilon": args.epsilon,
        "p_flip": args.p_flip,
        "eta_e": args.eta_e,
        "M": args.M,
        "T_M": args.T_M,
        "phi_M": args.phi_M,
        "partial_evaluation": args.partial_evaluation,
        "tau_element": args.tau_element,
        "tau_on": args.tau_on,
        "tau_off": args.tau_off,
        "gh_order": args.gh_order,
        "beta_max": args.beta_max,
    }
    values.update(overrides)
    return Params(**values)


def run_schedule(args) -> Schedule:
    return parse_schedule(args.schedule) if args.schedule else Schedule.single("online", args.duration)


def analytical_kwargs(args) -> dict:
    return {
        "schedule": run_schedule(args),
        "step": args.step,
        "method": args.method,
        "rtol": args.rtol,
        "atol": args.atol,
    }


def _safe_subdir(subdir: str) -> Path:
    subdir = subdir.strip().strip("/")
    if not subdir:
        return Path()
    path = Path(subdir)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("output subdirectory must be a relative path without '.' or '..'")
    return path


def results_dir(output_subdir: str = "") -> Path:
    return REPO_ROOT / "results" / _safe_subdir(output_subdir)


def figures_dir(output_subdir: str = "") -> Path:
    return REPO_ROOT / "figures" / _safe_subdir(output_subdir)


def pdf_path(path: Path) -> Path:
    return path.with_suffix(".pdf")


def output_path(name: str, suffix: str = "", output_subdir: str = "") -> Path:
    suffix = suffix.strip().strip("_")
    stem = f"{name}_{suffix}" if suffix else name
    return results_dir(output_subdir) / f"{stem}.csv"


def figure_path(name: str, suffix: str = "", output_subdir: str = "") -> Path:
    suffix = suffix.strip().strip("_")
    stem = f"{name}_{suffix}" if suffix else name
    return figures_dir(output_subdir) / f"{stem}.pdf"


def write_records(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError("no records to write")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def with_metadata(records, params: Params, **labels):
    metadata = {f"config_{key}": value for key, value in asdict(params).items()}
    return [{**labels, **metadata, **record} for record in records]
