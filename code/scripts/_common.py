"""Shared command-line, metadata, and result-writing helpers."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import json
from pathlib import Path
import shlex
import sys


CODE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_ROOT.parent
SRC_ROOT = CODE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from learning_dynamics import Params, Phase, Schedule, State  # noqa: E402


def parse_list(text: str, converter=float) -> list:
    values = [converter(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError("a comma-separated list must not be empty")
    return values


def parse_schedule(text: str) -> Schedule:
    phases = []
    for item in text.split(","):
        mode, duration = item.strip().split(":", 1)
        phases.append(Phase(mode.strip(), float(duration)))
    return Schedule(phases)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--schedule", default="online:20")
    parser.add_argument("--rho-se0", type=float, default=0.0)
    parser.add_argument("--rho-me0", type=float, default=0.0)
    parser.add_argument("--rho-sm0", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=5)
    parser.add_argument("--M", type=int, default=None, help="evaluated prefix length; defaults to T")
    parser.add_argument("--M-mode", choices=["constant", "reward", "drift"], default="constant")
    parser.add_argument("--K", type=int, default=5)
    parser.add_argument("--K-mode", choices=["constant", "reward", "drift"], default="constant")
    parser.add_argument("--K-max", type=int, default=20)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--temperature-mode", choices=["constant", "optimal"], default="constant")
    parser.add_argument("--beta-max", type=float, default=30.0)
    parser.add_argument("--eta-2f", type=float, default=1.0)
    parser.add_argument("--eta-3f", type=float, default=1.0)
    parser.add_argument("--eta-off", type=float, default=1.0)
    parser.add_argument("--eta-wm", type=float, default=1.0)
    parser.add_argument("--learning-rate-mode", choices=["constant", "optimal"], default="constant")
    parser.add_argument("--fix-eta-2f-zero", action="store_true")
    parser.add_argument("--fix-eta-3f-zero", action="store_true")
    parser.add_argument("--fix-eta-off-zero", action="store_true")
    parser.add_argument("--fix-eta-wm-zero", action="store_true")
    parser.add_argument("--eta-max", type=float, default=20.0)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--p-flip", type=float, default=0.0)
    parser.add_argument("--eta-e", type=float, default=0.0)
    parser.add_argument("--delta-on", type=float, default=0.1)
    parser.add_argument("--delta-off", type=float, default=0.1)
    parser.add_argument("--tau-on", type=float, default=None, help="inverse of delta-on")
    parser.add_argument("--tau-off", type=float, default=None, help="inverse of delta-off")
    parser.add_argument("--C-A", dest="C_A", type=float, default=0.0)
    parser.add_argument("--C-T", dest="C_T", type=float, default=1.0)
    parser.add_argument("--C-phi", dest="C_phi", type=float, default=0.0)
    parser.add_argument("--greedy-training", action="store_true")
    parser.add_argument("--greedy-evaluation", action="store_true")
    parser.add_argument("--gh-order", type=int, default=24)
    parser.add_argument("--suffix", default="")
    parser.add_argument("--output-subdir", default="")
    parser.add_argument("--output", type=Path, default=None)


def add_analytical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--method", default="RK45")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)


def add_empirical_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dimensions", default="250,1000,4000")
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--record-every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)


def _delta(delta: float, tau: float | None, name: str) -> float:
    if tau is None:
        return delta
    if tau <= 0.0:
        raise ValueError(f"{name} must be positive")
    return 1.0 / tau


def make_params(args, **overrides) -> Params:
    values = {
        "T": args.T,
        "M": args.T if args.M is None else args.M,
        "M_mode": args.M_mode,
        "K": args.K,
        "K_mode": args.K_mode,
        "K_max": args.K_max,
        "beta": args.beta,
        "temperature_mode": args.temperature_mode,
        "beta_max": args.beta_max,
        "eta_2f": args.eta_2f,
        "eta_3f": args.eta_3f,
        "eta_off": args.eta_off,
        "eta_wm": args.eta_wm,
        "learning_rate_mode": args.learning_rate_mode,
        "fix_eta_2f_zero": args.fix_eta_2f_zero,
        "fix_eta_3f_zero": args.fix_eta_3f_zero,
        "fix_eta_off_zero": args.fix_eta_off_zero,
        "fix_eta_wm_zero": args.fix_eta_wm_zero,
        "eta_max": args.eta_max,
        "epsilon": args.epsilon,
        "p_flip": args.p_flip,
        "eta_e": args.eta_e,
        "delta_on": _delta(args.delta_on, args.tau_on, "tau_on"),
        "delta_off": _delta(args.delta_off, args.tau_off, "tau_off"),
        "C_A": args.C_A,
        "C_T": args.C_T,
        "C_phi": args.C_phi,
        "greedy_training": args.greedy_training,
        "greedy_evaluation": args.greedy_evaluation,
        "gh_order": args.gh_order,
    }
    values.update(overrides)
    params = Params(**values)
    params.validate()
    return params


def initial_state(args, **overrides) -> State:
    values = {
        "rho_se": args.rho_se0,
        "rho_me": args.rho_me0,
        "rho_sm": args.rho_sm0,
    }
    values.update(overrides)
    state = State(**values)
    state.validate()
    return state


def schedule(args) -> Schedule:
    return parse_schedule(args.schedule)


def analytical_kwargs(args) -> dict:
    return {
        "schedule": schedule(args),
        "step": args.step,
        "method": args.method,
        "rtol": args.rtol,
        "atol": args.atol,
    }


def _safe_subdir(text: str) -> Path:
    text = text.strip().strip("/")
    if not text:
        return Path()
    path = Path(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("output subdirectory must be a safe relative path")
    return path


def results_dir(subdir: str = "") -> Path:
    return REPO_ROOT / "results" / _safe_subdir(subdir)


def output_path(stem: str, suffix: str = "", subdir: str = "", extension: str = ".csv") -> Path:
    suffix = suffix.strip().strip("_")
    name = f"{stem}_{suffix}" if suffix else stem
    return results_dir(subdir) / f"{name}{extension}"


def with_metadata(records: list[dict], params: Params, **labels) -> list[dict]:
    metadata = {f"config_{key}": value for key, value in asdict(params).items()}
    metadata.update(
        {
            "config_tau_on": params.tau_on,
            "config_tau_off": params.tau_off,
            "schema_version": 1,
        }
    )
    return [{**labels, **metadata, **record} for record in records]


def write_records(
    path: Path,
    records: list[dict],
    *,
    argv=None,
    metadata: dict | None = None,
) -> None:
    if not records:
        raise ValueError("no result records were produced")
    path = path.resolve()
    root = results_dir().resolve()
    if root not in path.parents:
        raise ValueError(f"result outputs must remain under {root}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for record in records:
        for key in record:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    manifest = {
        "schema_version": 1,
        "result": str(path.relative_to(REPO_ROOT)),
        "command": shlex.join(argv or sys.argv),
        "rows": len(records),
        "columns": fields,
    }
    if metadata is not None:
        manifest["metadata"] = metadata
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")


def destination(args, stem: str, extension: str = ".csv") -> Path:
    return args.output or output_path(stem, args.suffix, args.output_subdir, extension)
