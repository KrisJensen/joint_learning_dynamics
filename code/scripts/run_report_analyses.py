#%%
import numpy as np
import subprocess as sp
ROOT = "/ceph/behrens/kris/research/joint_learning_dynamics/code/"

subdir = "report"
prefix_time = "--constant-prefix-time"

#%% generate empirical vs analytical comparison

sp.call(["python", ROOT + "scripts/compare_empirical_analytical.py", "--output-subdir", subdir, "--dimensions", "200,4000", "--fix-eta-2f-zero", prefix_time] )
sp.call(["python", ROOT + "plot/plot_empirical_analytical.py", "--output-subdir", subdir] )

# also just simple trajectory (and plot on sphere)
sp.call(["python", ROOT + "scripts/run_trajectory.py", "--output-subdir", subdir, "--fix-eta-2f-zero", prefix_time] )
sp.call(["python", ROOT + "plot/plot_trajectory.py", "--output-subdir", subdir, "--plot-correlations", "--plot-planning-success-probability", "--plot-environment-reward-probability"] )
sp.call(["python", ROOT + "plot/plot_sphere.py", "--output-subdir", subdir] )

#%% also different levels of world model goodness

sp.call(["python", ROOT + "scripts/compare_rho_me.py", "--output-subdir", subdir, "--fix-eta-2f-zero", "--include-greedy-baseline", prefix_time] )
sp.call(["python", ROOT + "plot/plot_rho_me.py", "--output-subdir", subdir] )

#%% now generate optimal learning rate and temperature

sp.call(["python", ROOT + "scripts/compare_learning_rates.py", "--output-subdir", subdir, "--fix-eta-2f-zero", "--eta-max", "2.0", prefix_time] )
sp.call(["python", ROOT + "plot/plot_learning_rates.py", "--output-subdir", subdir] )

sp.call(["python", ROOT + "scripts/compare_temperatures.py", "--output-subdir", subdir, "--fix-eta-2f-zero", prefix_time] )
sp.call(["python", ROOT + "plot/plot_temperatures.py", "--output-subdir", subdir] )


#%% generate figure that qualitatively compares different types of learning
sp.call(["python", ROOT + "scripts/compare_policy_regimes.py", "--output-subdir", subdir, prefix_time] )
sp.call(["python", ROOT + "plot/plot_policy_regimes.py", "--output-subdir", subdir] )

#%% compare action correctness probabilities for different prefix lengths
sp.call(["python", ROOT + "scripts/compare_prefix_length_observables.py", "--output-subdir", subdir] )
sp.call(["python", ROOT + "plot/plot_prefix_length_observables.py", "--output-subdir", subdir] )


#%% compare learning for different prefix lengths
common_args = ["--K", "10", "--learning-rate-mode", "optimal", "--eta-max", "1.0", prefix_time]
sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T6", "--T", "6", *common_args] )

sp.call(["python", ROOT + "plot/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T6", "--output", ROOT + "../figures/report/prefix_length_comparison_2factor_T6.pdf", "--include-fixed-M", "--include-optimal-M"] )
sp.call(["python", ROOT + "plot/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T6", "--output", ROOT + "../figures/report/prefix_length_comparison_optimal_T6.pdf", "--include-optimal-M", "--include-eta-3f", "--include-joint"] )


sp.call(["python", ROOT + "scripts/compare_prefix_lengths.py", "--output-subdir", subdir, "--T", "9", "--K", "10", "--suffix", "T9", *common_args] )

sp.call(["python", ROOT + "plot/plot_prefix_lengths.py", "--output-subdir", subdir, "--suffix", "T9", "--output", ROOT + "../figures/report/prefix_length_comparison_optimal_T9.pdf", "--include-optimal-M", "--include-eta-3f", "--include-joint"] )

#%% now run dynamic programming

T = "6"
shared_args = ["python", ROOT + "scripts/run_dynamic_programming.py", "--temperature-mode", "optimal", "--dp-step-size", "0.10", "--T", T, "--horizon", "20", "--learning-rate-mode", "optimal", "--eta-max", "1.0", "--output-subdir", subdir, "--delta-on", "0.1", "--delta-off", "0.1", "--dp-K-max", "10", "--rho-points", "30", "--t-switch", "1.0", prefix_time]

# solve online-only and switching combined DPs from one action workspace
sp.call(shared_args+["--variant", "both", "--online-suffix", f"joint", "--switching-suffix", f"joint"])

sp.call(["python", ROOT + "plot/plot_dynamic_programming.py", "--variant", "online", "--suffix", f"joint", "--output-subdir", subdir, "--plot-individuals"])
sp.call(["python", ROOT + "plot/plot_dynamic_programming.py", "--variant", "switching", "--suffix", f"joint", "--output-subdir", subdir, "--plot-individuals"])


#%% add phase modulation

sp.run(["python", ROOT+"scripts/run_dynamic_programming.py", "--variant", "switching", "--t-switch", "0.1", "--learning-rate-mode",
        "optimal","--p-flip","0", "--T","5","--dp-M","5","--eta-e","0.3","--rho-points", "31",
        "--suffix","drift","--horizon","9","--dp-step-size","0.1","--eta-max", "3", "--delta-on","0.05", "--delta-off",
        "0.05", "--C-T", "2", "--C-phi", "0", "--C-A", "0.95", "--with-modulation", "--output-subdir", subdir, prefix_time])
sp.run(["python", ROOT+"plot/plot_dynamic_programming.py", "--variant","switching","--suffix","drift", "--output-subdir", subdir, "--plot-individuals"])



#%% generate phase diagram

rho_se_values = ",".join(
    f"{value:.12g}" for value in np.linspace(0.0, 0.97, 50)
)
rho_me_values = ",".join(
    f"{value:.12g}" for value in np.linspace(0.0, 0.97, 30)
)

T = "6"
suffix = "joint"

sp.run(["python", ROOT+"scripts/phase_diagram_initial.py", "--switch-times","1.0, 2.0, 3.0","--T",T,"--temperature-mode", "constant",
        "--beta","3","--initial-horizons","10,15,20","--dp-step-size","0.2","--rho-points","30","--dp-K-max","10", "--suffix", suffix, "--learning-rate-mode", "optimal", "--eta-max", "1.0", "--delta-on","0.1","--delta-off","0.1", prefix_time,
        "--rho-se-values", rho_se_values, "--rho-me-values", rho_me_values, "--output-subdir", subdir])

sp.run(["python", ROOT+"plot/plot_phase_diagrams.py", "--output-subdir", subdir, "--switch-times","1.0,2.0,3.0", "--suffix", suffix, "--initial-horizons","10,15,20"])
sp.run(["python", ROOT+"plot/plot_phase_examples.py", "--examples", "0.05,0.05;0.05,0.9;0.5,0.9", "--suffix", suffix, "--output-subdir", subdir])
