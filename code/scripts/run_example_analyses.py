
"""Run the complete example analysis and plotting suite.

This is the flat, new-API counterpart of
``legacy/code/scripts/run_all_analyses.py``. All outputs are placed below
``results/examples`` and ``figures/examples``.
"""

from pathlib import Path
import subprocess as sp
import sys

import numpy as np


ROOT = "/ceph/behrens/kris/research/joint_learning_dynamics/code/"
REPO_ROOT = "/ceph/behrens/kris/research/joint_learning_dynamics/"
PYTHON = "python"
OUTPUT = ["--output-subdir", "examples"]


# %% compare empirical and analytical
sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_empirical_analytical.py",
        "--eta-e",
        "0.1",
        "--eta-2f",
        "1.5",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [PYTHON, ROOT+"plot/plot_empirical_analytical.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)


# %% compare empirical and analytical with partial WM evaluation, M=3
sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_empirical_analytical.py",
        "--eta-e",
        "0.1",
        "--eta-2f",
        "1.5",
        "--M",
        "3",
        "--suffix",
        "prefix_M3",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_empirical_analytical.py",
        "--suffix",
        "prefix_M3",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)


# %% compare 3-factor and 2-factor learning across planner quality
sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_rho_me.py",
        "--fix-eta-2f-zero",
        "--eta-3f",
        "2.5",
        "--suffix",
        "eta_2f_zero",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_rho_me.py",
        "--suffix",
        "eta_2f_zero",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)

sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_rho_me.py",
        "--fix-eta-3f-zero",
        "--eta-2f",
        "2.5",
        "--suffix",
        "eta_3f_zero",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_rho_me.py",
        "--suffix",
        "eta_3f_zero",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)

#%% compare across reward noise

sp.run([PYTHON, ROOT+"scripts/compare_reward_noise_learning.py", "--schedule", "online:40"])
sp.run([PYTHON, ROOT+"plot/plot_reward_noise_learning.py", "--schedule", "online:40"])

sp.run([PYTHON, ROOT+"scripts/compare_reward_noise_learning.py", "--schedule", "online:40", "--fixed-wm"])
sp.run([PYTHON, ROOT+"plot/plot_reward_noise_learning.py", "--schedule", "online:40", "--fixed-wm"])

# %% compare different prefix lengths
sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_prefix_lengths.py",
        "--temperature-mode",
        "optimal",
        "--T",
        "9",
        "--eta-max",
        "20",
        "--K",
        "10",
        "--learning-rate-mode",
        "optimal",
        "--suffix",
        "optimal",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_prefix_lengths.py",
        "--suffix",
        "optimal",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)

sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_prefix_lengths.py",
        "--eta-2f",
        "2.5",
        "--eta-3f",
        "2.5",
        "--temperature-mode",
        "optimal",
        "--T",
        "9",
        "--K",
        "10",
        "--schedule",
        "online:30",
        "--suffix",
        "constant",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_prefix_lengths.py",
        "--suffix",
        "constant",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)


# %% learning-rate comparisons
sp.run(
    [PYTHON, ROOT+"scripts/compare_learning_rates.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [PYTHON, ROOT+"plot/plot_learning_rates.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)


# %% temperature sweep
sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_temperatures.py",
        "--rho-me0",
        "1",
        "--eta-wm",
        "0",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [PYTHON, ROOT+"plot/plot_temperatures.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)


# %% different values of K
sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_K.py",
        "--rho-me0",
        "1",
        "--eta-wm",
        "0",
        "--suffix",
        "student",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_K.py",
        "--suffix",
        "student",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)

sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_K.py",
        "--suffix",
        "joint",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_K.py",
        "--suffix",
        "joint",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)


# %% compare reward-modulated learning and planning across T
sp.run(
    [PYTHON, ROOT+"scripts/compare_rl_planning_T.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [PYTHON, ROOT+"plot/plot_rl_planning_T.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)


# %% compare planning/greedy training and evaluation regimes
sp.run(
    [PYTHON, ROOT+"scripts/compare_policy_regimes.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [PYTHON, ROOT+"plot/plot_policy_regimes.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)


# %% compare cyclic environment-modulation amplitudes
sp.run(
    [
        PYTHON,
        ROOT+"scripts/compare_modulation.py",
        "--eta-e",
        "0.1",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [PYTHON, ROOT+"plot/plot_modulation.py", *OUTPUT],
    check=True,
    cwd=REPO_ROOT,
)


# %% dynamic programming: online optimization over K and M
# delta_on=0.0625 gives the legacy element cost 0.0625/8.
# dp_step_size=0.128 gives the corresponding horizon spacing Delta h=0.001.
sp.run(
    [
        PYTHON,
        ROOT+"scripts/run_dynamic_programming.py",
        "--variant",
        "online",
        "--fixed-wm",
        "--fixed-rho-me",
        "0.999",
        "--temperature-mode",
        "optimal",
        "--suffix",
        "optimal_KM_2f",
        "--dp-step-size",
        "0.128",
        "--delta-on",
        "0.0625",
        "--delta-off",
        "0.0625",
        "--T",
        "8",
        "--horizon",
        "40",
        "--rho-points",
        "100",
        "--dp-K-max",
        "15",
        "--eta-2f",
        "1.0",
        "--eta-3f",
        "0",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)

sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_dynamic_programming.py",
        "--variant",
        "online",
        "--fixed-wm",
        "--suffix",
        "optimal_KM_2f",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)


# %% dynamic programming: online/offline switching
T = "5"
for condition in ["2f", "3f"][1:]:
    flags = ["--fix-eta-3f-zero"] if condition == "2f" else ["--fix-eta-2f-zero", "--dp-M", T]

    sp.run(
        [
            PYTHON,
            ROOT+"scripts/run_dynamic_programming.py",
            "--variant",
            "switching",
            "--T", T,
            "--horizon", "10",
            "--dp-step-size",
            "0.1",
            "--rho-points",
            "30",
            "--delta-on",
            "0.0625",
            "--delta-off",
            "0.0625",
            "--dp-K-max",
            "15",
            "--t-switch",
            "0.625",
            "--learning-rate-mode", "optimal",
            "--eta-max", "1.0",
            "--temperature-mode",
            "optimal",
            "--no-modulation",
            "--suffix",
            condition,
            *flags,
            *OUTPUT,
        ],
        check=True,
        cwd=REPO_ROOT,
    )
    sp.run(
        [
            PYTHON,
            ROOT+"plot/plot_dynamic_programming.py",
            "--variant",
            "switching",
            "--suffix",
            condition,
            *OUTPUT,
        ],
        check=True,
        cwd=REPO_ROOT,
    )


# %% dynamic programming with environmental drift and cycle modulation
# delta_on/T=0.01, so dp_step_size=0.1 gives Delta h=0.001.
sp.run(
    [
        PYTHON,
        ROOT+"scripts/run_dynamic_programming.py",
        "--variant",
        "switching",
        "--t-switch",
        "0.1",
        "--learning-rate-mode",
        "optimal",
        "--p-flip",
        "0",
        "--fix-eta-2f-zero",
        "--T",
        "5",
        "--dp-M",
        "5",
        "--eta-e",
        "0.3",
        "--rho-points",
        "31",
        "--suffix",
        "drift",
        "--horizon",
        "9",
        "--dp-step-size",
        "0.1",
        "--eta-max",
        "3",
        "--delta-on",
        "0.05",
        "--delta-off",
        "0.05",
        "--C-T",
        "2",
        "--C-phi",
        "0",
        "--C-A",
        "0.95",
        "--with-modulation",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_dynamic_programming.py",
        "--variant",
        "switching",
        "--suffix",
        "drift",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)


# %% initial-correlation phase diagram
rho_se_values = ",".join(
    f"{value:.12g}" for value in np.linspace(0.0, 0.99, 50)
)
rho_me_values = ",".join(
    f"{value:.12g}" for value in np.linspace(0.0, 0.999, 26)
)

sp.run(
    [
        PYTHON,
        ROOT+"scripts/phase_diagram_initial.py",
        "--switch-times",
        "0.5,1.0,1.5",
        "--T",
        "6",
        "--dp-M",
        "6",
        "--temperature-mode",
        "constant",
        "--beta",
        "3",
        "--suffix",
        "constant_beta",
        "--initial-horizons",
        "10,15,20",
        "--dp-step-size",
        "0.2",
        "--rho-points",
        "26",
        "--dp-K-max",
        "10",
        "--eta-2f", "0.0",
        "--eta-3f", "1.0",
        "--delta-on",
        "0.05",
        "--delta-off",
        "0.05",
        "--eta-wm",
        "1.0",
        "--rho-se-values",
        rho_se_values,
        "--rho-me-values",
        rho_me_values,
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_phase_diagrams.py",
        "--suffix",
        "constant_beta",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
sp.run(
    [
        PYTHON,
        ROOT+"plot/plot_phase_examples.py",
        "--suffix",
        "constant_beta",
        "--examples",
        "0.0,0.0;0.0,0.8;0.6,0.9",
        *OUTPUT,
    ],
    check=True,
    cwd=REPO_ROOT,
)
