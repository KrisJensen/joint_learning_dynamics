"""Sweep prefix length and plot reward/action correctness observables."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

from _common import figure_path as default_figure_path
from _common import output_path, write_records
from joint_dynamics import Params, State
from joint_dynamics.moments import (
    compute_moments,
    environment_reward_probability,
    prepare_moment_quadrature,
    selected_zero_coefficient,
    success_coefficient,
)
from joint_dynamics.observables import observable_probabilities
from joint_dynamics.optimization import effective_temperature

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib.pyplot as plt  # noqa: E402


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--T", type=int, default=6)
parser.add_argument("--beta", type=float, default=3.0)
parser.add_argument("--tau-on", type=float, default=0.05)
parser.add_argument(
    "--rho-values",
    default="0.1,0.3,0.5,0.7,0.9",
    help="comma-separated rho values; uses rho_se=rho_sm=rho and rho_me=1",
)
parser.add_argument(
    "--temperature-mode",
    choices=["constant", "optimal"],
    default="constant",
)
parser.add_argument(
    "--tau-element",
    type=float,
    default=None,
    help="thinking time per simulated sequence element; defaults to tau_on/T",
)
parser.add_argument("--gh-order", type=int, default=25)
parser.add_argument("--suffix", default="")
parser.add_argument("--output-subdir", default="")
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--figure-output", type=Path, default=None)
parser.add_argument("--dpi", type=int, default=180)
args = parser.parse_args()

if args.T < 1:
    parser.error("T must be at least one")
if args.gh_order < 4:
    parser.error("gh-order must be at least four")
if args.tau_on < 0.0:
    parser.error("tau-on must be nonnegative")
if args.tau_element is not None and args.tau_element <= 0.0:
    parser.error("tau-element must be positive when explicitly set")
if args.tau_element is None and args.tau_on <= 0.0:
    parser.error("tau-on must be positive when tau-element is not explicitly set")
rho_values = [
    float(value.strip())
    for value in args.rho_values.split(",")
    if value.strip()
]
if not rho_values:
    parser.error("rho-values must contain at least one value")
if any(not -1.0 <= rho <= 1.0 for rho in rho_values):
    parser.error("rho-values must lie in [-1, 1]")

K = 10
records = []
for rho in rho_values:
    state = State(rho_se=rho, rho_me=1.0, rho_sm=rho)
    state.validate()
    quadrature = prepare_moment_quadrature(
        state,
        Params(
            T=args.T,
            K=K,
            tau_on=args.tau_on,
            tau_element=args.tau_element,
            partial_evaluation=True,
            gh_order=args.gh_order,
        ),
    )
    moments_by_beta = {}
    for prefix_length in range(1, args.T + 1):
        params = Params(
            T=args.T,
            prefix_length=prefix_length,
            K=K,
            beta=args.beta,
            temperature_mode=args.temperature_mode,
            partial_evaluation=True,
            tau_on=args.tau_on,
            tau_element=args.tau_element,
            gh_order=args.gh_order,
        )
        params.validate()
        beta = effective_temperature(state, params)
        if beta not in moments_by_beta:
            moments_by_beta[beta] = compute_moments(state, params, beta, quadrature)
        moments = moments_by_beta[beta]
        observables = observable_probabilities(state, params, beta, moments)
        sequence_probability = environment_reward_probability(moments, params)
        reward_rate = sequence_probability / params.c_on

        prefix_action_probability = 0.0
        for j in range(1, K + 1):
            prefix_action_probability += (
                success_coefficient(j, K)
                * moments.A[j] ** (prefix_length - 1)
                * moments.P_me[j]
            )
        for j in range(K):
            prefix_action_probability += selected_zero_coefficient(j, K) * (
                moments.A[j] ** (prefix_length - 1) * moments.J[j]
                - moments.A[j + 1] ** (prefix_length - 1) * moments.P_me[j + 1]
            )
        prefix_action_probability = float(
            np.clip(prefix_action_probability, 0.0, 1.0)
        )
        suffix_action_probability = float(moments.J[0])
        individual_action_probability = (
            prefix_length * prefix_action_probability
            + (args.T - prefix_length) * suffix_action_probability
        ) / args.T
        records.append(
            {
                "rho": rho,
                "rho_se": state.rho_se,
                "rho_me": state.rho_me,
                "rho_sm": state.rho_sm,
                "K": K,
                "T": args.T,
                "prefix_length": prefix_length,
                "beta": beta,
                "temperature_mode": args.temperature_mode,
                "tau_element": params.effective_tau_element,
                "tau_on": params.tau_on,
                "think_act_time": params.c_on,
                "sequence_correct_probability": sequence_probability,
                "reward_rate": reward_rate,
                "planner_prefix_correct_probability": observables[
                    "planner_acceptance_probability"
                ],
                "individual_action_correct_probability": individual_action_probability,
                "prefix_action_correct_probability": prefix_action_probability,
                "suffix_action_correct_probability": suffix_action_probability,
            }
        )

result_path = args.output or output_path(
    "prefix_length_observables",
    args.suffix,
    args.output_subdir,
)
figure_path = args.figure_output or default_figure_path(
    "prefix_length_observables",
    args.suffix,
    args.output_subdir,
)
write_records(result_path, records)
print(f"Wrote {result_path}")

metrics = (
    ("sequence_correct_probability", "correct sequence probability"),
    ("reward_rate", "reward rate"),
    ("individual_action_correct_probability", "correct action probability"),
    ("planner_prefix_correct_probability", "planner-correct prefix probability"),
)
colors = plt.get_cmap("viridis")(np.linspace(0.12, 0.9, len(rho_values)))
for metric, ylabel in metrics:
    metric_figure_path = figure_path.with_name(
        f"{figure_path.stem}_{metric}{figure_path.suffix}"
    )
    fig, axis = plt.subplots(figsize=(5.2, 3.6))
    for color, rho in zip(colors, rho_values):
        rows = [
            row
            for row in records
            if np.isclose(row["rho"], rho)
        ]
        rows.sort(key=lambda row: row["prefix_length"])
        x_values = [row["prefix_length"] for row in rows]
        y_values = [row[metric] for row in rows]
        axis.plot(
            x_values,
            y_values,
            color=color,
            linewidth=2.5,
            marker="o",
            label=rf"$\rho={rho:g}$",
        )
        best_index = int(np.argmax(y_values))
        axis.plot(
            [x_values[best_index]],
            [y_values[best_index]],
            color=color,
            marker="*",
            markersize=13,
            markeredgecolor="black",
            markeredgewidth=0.7,
            linestyle="None",
        )
    axis.set_xlabel("prefix length")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", color="#dddddd", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False)
    fig.suptitle(rf"{ylabel} ($K={K}$, $T={args.T}$)")
    fig.tight_layout()
    metric_figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(metric_figure_path, dpi=args.dpi, transparent=True)
    plt.close(fig)
    print(f"Wrote {metric_figure_path}")
