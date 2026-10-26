"""Compute and plot learning curves for world-model parameter sweeps."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from _common import (
    add_analytical_args,
    add_common_args,
    analytical_kwargs,
    figure_path as default_figure_path,
    initial_state,
    make_params,
    output_path,
    with_metadata,
    write_records,
)
from joint_dynamics import run_analytical

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt  # noqa: E402


def parse_float_list(text: str, name: str) -> list[float]:
    values = [float(value.strip()) for value in text.split(",") if value.strip()]
    if not values:
        raise ValueError(f"{name} must contain at least one value")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    parser.set_defaults(schedule="online:30", duration=30.0)
    add_analytical_args(parser)
    parser.add_argument(
        "--eta-wm-values",
        default="0.1,0.2,0.3,0.4,0.5,0.7,1.0",
        help="comma-separated world-model learning rates for the eta_wm sweep",
    )
    parser.add_argument(
        "--p-flip-values",
        default="0,0.05,0.1,0.2,0.3,0.4",
        help="comma-separated label-flip probabilities for the p_flip sweep",
    )
    parser.add_argument(
        "--greedy-learning-rate-values",
        default="0.5,1,1.5,2.5,3,4,5",
        help=(
            "comma-separated student learning rates for the greedy K=1 sweep; "
            "applied to --greedy-rate-name"
        ),
    )
    parser.add_argument(
        "--greedy-rate-name",
        choices=["eta_on1", "eta_on2"],
        default="eta_on2",
        help="student learning-rate component varied in the greedy K=1 sweep",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=None,
        help="figure path; defaults to figures/world_model_sweeps[_suffix].pdf",
    )
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    eta_wm_values = parse_float_list(args.eta_wm_values, "--eta-wm-values")
    p_flip_values = parse_float_list(args.p_flip_values, "--p-flip-values")
    greedy_learning_rate_values = parse_float_list(
        args.greedy_learning_rate_values,
        "--greedy-learning-rate-values",
    )
    if any(value < 0.0 for value in eta_wm_values):
        parser.error("eta_wm values must be nonnegative")
    if any(not 0.0 <= value <= 1.0 for value in p_flip_values):
        parser.error("p_flip values must lie in [0, 1]")
    if any(value < 0.0 for value in greedy_learning_rate_values):
        parser.error("greedy learning-rate values must be nonnegative")

    initial = initial_state(args)
    initial.validate()

    records = []
    world_model_policy_overrides = {
        "K": 1,
        "K_mode": "constant",
        "K_max": 1,
        "beta": float("inf"),
        "temperature_mode": "constant",
        "greedy_training": True,
        "greedy_evaluation": True,
        "learning_rate_mode": "constant",
        "eta_on1": 0.0,
        "eta_on2": 0.0,
        "fix_eta_on1_zero": True,
        "fix_eta_on2_zero": True,
    }
    for eta_wm in eta_wm_values:
        print(f"Running eta_wm sweep value eta_wm={eta_wm:g}...")
        params = make_params(
            args,
            eta_wm=eta_wm,
            **world_model_policy_overrides,
        )
        records += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=rf"$\eta_{{wm}}={eta_wm:g}$",
            sweep="eta_wm",
            sweep_value=eta_wm,
        )

    for p_flip in p_flip_values:
        print(f"Running p_flip sweep value p_flip={p_flip:g}...")
        params = make_params(
            args,
            eta_wm=1.0,
            p_flip=p_flip,
            **world_model_policy_overrides,
        )
        records += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=rf"$p_{{flip}}={p_flip:g}$",
            sweep="p_flip",
            sweep_value=p_flip,
        )

    for learning_rate in greedy_learning_rate_values:
        print(
            "Running greedy learner sweep value "
            f"{args.greedy_rate_name}={learning_rate:g}..."
        )
        rate_overrides = (
            {
                "eta_on1": learning_rate,
                "eta_on2": 0.0,
                "fix_eta_on1_zero": False,
                "fix_eta_on2_zero": True,
            }
            if args.greedy_rate_name == "eta_on1"
            else {
                "eta_on1": 0.0,
                "eta_on2": learning_rate,
                "fix_eta_on1_zero": True,
                "fix_eta_on2_zero": False,
            }
        )
        params = make_params(
            args,
            K=1,
            K_mode="constant",
            K_max=1,
            beta=float("inf"),
            temperature_mode="constant",
            greedy_training=True,
            greedy_evaluation=True,
            learning_rate_mode="constant",
            **rate_overrides,
        )
        records += with_metadata(
            run_analytical(initial, params, **analytical_kwargs(args)),
            params,
            source="analytical",
            condition=rf"greedy ${args.greedy_rate_name}={learning_rate:g}$",
            sweep="greedy_learning_rate",
            sweep_value=learning_rate,
            greedy_rate_name=args.greedy_rate_name,
        )

    data_path = args.output or output_path("world_model_sweeps", args.suffix, args.output_subdir)
    write_records(data_path, records)
    print(f"Wrote {data_path}")

    figure_path = args.figure_output or default_figure_path(
        "world_model_sweeps",
        args.suffix,
        args.output_subdir,
    )
    figure_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.0), sharex=True)
    sweep_specs = (
        ("eta_wm", r"$\eta_{wm}$ sweep"),
        ("p_flip", r"$p_{flip}$ sweep"),
    )
    for axis, (sweep_name, title) in zip(axes[:2], sweep_specs):
        sweep_rows = [row for row in records if row["sweep"] == sweep_name]
        values = sorted({float(row["sweep_value"]) for row in sweep_rows})
        colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(values)))
        for color, value in zip(colors, values):
            rows = [
                row
                for row in sweep_rows
                if float(row["sweep_value"]) == value
            ]
            rows.sort(key=lambda row: float(row["time"]))
            label = f"{value:g}"
            times = [float(row["time"]) for row in rows]
            axis.plot(
                times,
                [float(row["rho_me"]) for row in rows],
                color=color,
                label=label,
            )
        axis.set_title(title)
        axis.set_ylabel(r"$\rho_{me}$")
        axis.grid(axis="y", color="#dddddd", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(frameon=False, title="value", ncol=3)

    greedy_rows = [row for row in records if row["sweep"] == "greedy_learning_rate"]
    greedy_values = sorted({float(row["sweep_value"]) for row in greedy_rows})
    greedy_colors = plt.get_cmap("plasma")(
        np.linspace(0.08, 0.92, len(greedy_values))
    )
    for color, value in zip(greedy_colors, greedy_values):
        rows = [
            row
            for row in greedy_rows
            if float(row["sweep_value"]) == value
        ]
        rows.sort(key=lambda row: float(row["time"]))
        axes[2].plot(
            [float(row["time"]) for row in rows],
            [float(row["rho_se"]) for row in rows],
            color=color,
            label=f"{value:g}",
        )
    axes[2].set_title(
        rf"greedy learner ($K=1$, greedy actions), varying ${args.greedy_rate_name}$"
    )
    axes[2].set_ylabel(r"$\rho_{se}$")
    axes[2].set_xlabel("elapsed time")
    axes[2].grid(axis="y", color="#dddddd", linewidth=0.7)
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].legend(frameon=False, title=args.greedy_rate_name, ncol=3)

    fig.tight_layout()
    fig.savefig(figure_path, dpi=args.dpi, transparent=True)
    plt.close(fig)
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
